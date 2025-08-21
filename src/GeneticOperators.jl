module GeneticOperators

using Random
using Statistics
using ..GATypes
using ..ParameterEncoding

export tournament_selection, uniform_crossover!, gaussian_mutation!,
       elite_selection, create_offspring, evolve_population!

# =============================================================================
# SELECTION OPERATORS
# =============================================================================

function tournament_selection(
    population::M,
    fitness::V,
    tournament_size::Int32;
    rng::AbstractRNG = Random.default_rng()
)::Int32 where {M<:AbstractMatrix{Float32}, V<:AbstractVector{Float32}}

    pop_size = size(population, 1)
    tourn_size = min(tournament_size, Int32(pop_size))
    
    # Select random individuals for tournament
    best_idx = rand(rng, 1:pop_size)
    best_fitness = fitness[best_idx]

    for _ in 2:tourn_size
        idx = rand(rng, 1:pop_size)
        if fitness[idx] > best_fitness
            best_fitness = fitness[idx]
            best_idx = idx
        end
    end
    
    return Int32(best_idx)
end

function elite_selection(fitness::V, elite_size::Int32)::Vector{Int32} where {V<:AbstractVector{Float32}}
    sorted_indices = sortperm(fitness, rev=true)
    n_elite = min(elite_size, length(fitness))
    return Int32.(sorted_indices[1:n_elite])
end

# =============================================================================
# CROSSOVER OPERATORS
# =============================================================================

function uniform_crossover!(
    offspring1::V, offspring2::V,
    parent1::V, parent2::V,
    crossover_rate::Float32;
    rng::AbstractRNG = Random.default_rng()
) where {V<:AbstractVector{Float32}}
    
    if rand(rng, Float32) > crossover_rate
        offspring1 .= parent1
        offspring2 .= parent2
        return
    end
    
    for i in eachindex(parent1)
        if rand(rng) < 0.5
            offspring1[i] = parent1[i]
            offspring2[i] = parent2[i]
        else
            offspring1[i] = parent2[i]
            offspring2[i] = parent1[i]
        end
    end
end

# =============================================================================
# MUTATION OPERATORS
# =============================================================================

function gaussian_mutation!(
    chromosome::V,
    mutation_rate::Float32,
    ranges::ParameterRanges;
    rng::AbstractRNG = Random.default_rng(),
    mutation_strength::Float32 = 0.1f0
) where {V<:AbstractVector{Float32}}
    
    for i in eachindex(chromosome)
        if rand(rng, Float32) < mutation_rate
            bounds = get_parameter_bounds(Int32(i), ranges)
            
            if i == 7  # Binary parameter
                chromosome[i] = chromosome[i] > 0.5f0 ? 0.0f0 : 1.0f0
            elseif i == 11  # Discrete parameter
                n_options = Int32(bounds[2])
                current = round(Int32, chromosome[i])
                step = rand(rng, (-1, 1))
                chromosome[i] = Float32(clamp(current + step, 1, n_options))
            else  # Continuous parameters
                range_width = bounds[2] - bounds[1]
                perturbation = randn(rng, Float32) * mutation_strength * range_width
                chromosome[i] = clamp(chromosome[i] + perturbation, bounds...)
            end
        end
    end
end

# =============================================================================
# POPULATION EVOLUTION
# =============================================================================

function evolve_population!(
    population::M,
    fitness::V,
    ga_params::GAParameters,
    ranges::ParameterRanges;
    rng::AbstractRNG = Random.default_rng()
) where {M<:AbstractMatrix{Float32}, V<:AbstractVector{Float32}}
    
    pop_size = size(population, 1)
    new_population = similar(population)
    
    # Preserve elite
    elite_indices = elite_selection(fitness, ga_params.elite_size)
    for (i, idx) in enumerate(elite_indices)
        new_population[i, :] = @view population[idx, :]
    end
    
    # Generate offspring to fill rest of population
    offspring_start = length(elite_indices) + 1
    
    # Preallocate parent and offspring vectors of the correct type
    parent1 = similar(fitness, 13)
    parent2 = similar(fitness, 13)
    offspring1 = similar(fitness, 13)
    offspring2 = similar(fitness, 13)

    for i in offspring_start:2:pop_size
        parent1_idx = tournament_selection(population, fitness, ga_params.tournament_size, rng=rng)
        parent2_idx = tournament_selection(population, fitness, ga_params.tournament_size, rng=rng)

        parent1 .= @view population[parent1_idx, :]
        parent2 .= @view population[parent2_idx, :]
        
        uniform_crossover!(offspring1, offspring2, parent1, parent2, ga_params.crossover_rate, rng=rng)
        
        gaussian_mutation!(offspring1, ga_params.mutation_rate, ranges, rng=rng)
        gaussian_mutation!(offspring2, ga_params.mutation_rate, ranges, rng=rng)
        
        apply_bounds!(offspring1, ranges)
        apply_bounds!(offspring2, ranges)
        
        new_population[i, :] = offspring1
        if i + 1 <= pop_size
            new_population[i + 1, :] = offspring2
        end
    end
    
    # Replace old population
    population .= new_population
    return population
end

end # module GeneticOperators