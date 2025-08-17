# src/GeneticOperators.jl - Genetic Algorithm Operators for 13-Parameter Chromosomes
# Tournament selection, uniform crossover, and Gaussian mutation

module GeneticOperators

using Random
using Statistics

export tournament_selection, uniform_crossover!, gaussian_mutation!,
       elite_selection, create_offspring, evolve_population!,
       population_diversity, fitness_variance

# =============================================================================
# SELECTION OPERATORS
# =============================================================================

"""
Tournament selection - select one parent via tournament
Higher fitness wins the tournament
"""
function tournament_selection(population::Matrix{Float32}, 
                             fitness::Vector{Float32},
                             tournament_size::Int32 = Int32(5);
                             rng::AbstractRNG = Random.default_rng())::Int32
    
    pop_size = size(population, 1)
    
    # Ensure tournament size is valid
    tournament_size = min(tournament_size, Int32(pop_size))
    
    # Select random individuals for tournament
    contestants = rand(rng, 1:pop_size, tournament_size)
    
    # Find winner (highest fitness)
    best_idx = contestants[1]
    best_fitness = fitness[contestants[1]]
    
    for i in 2:tournament_size
        idx = contestants[i]
        if fitness[idx] > best_fitness
            best_fitness = fitness[idx]
            best_idx = idx
        end
    end
    
    return Int32(best_idx)
end

"""
Elite selection - get indices of top n individuals
"""
function elite_selection(fitness::Vector{Float32}, elite_size::Int32)::Vector{Int32}
    # Get indices sorted by fitness (descending)
    sorted_indices = sortperm(fitness, rev=true)
    
    # Return top elite_size individuals
    n_elite = min(elite_size, length(fitness))
    return Int32.(sorted_indices[1:n_elite])
end

# =============================================================================
# CROSSOVER OPERATORS
# =============================================================================

"""
Uniform crossover - each gene has 50% chance from each parent
Modifies offspring1 and offspring2 in place
"""
function uniform_crossover!(offspring1::Vector{Float32},
                           offspring2::Vector{Float32},
                           parent1::Vector{Float32},
                           parent2::Vector{Float32},
                           crossover_rate::Float32 = 0.7f0;
                           rng::AbstractRNG = Random.default_rng())
    
    # Apply crossover with probability
    if rand(rng, Float32) > crossover_rate
        # No crossover - offspring are copies of parents
        offspring1 .= parent1
        offspring2 .= parent2
        return
    end
    
    # Uniform crossover - each gene independently
    for i in 1:13
        if rand(rng) < 0.5
            offspring1[i] = parent1[i]
            offspring2[i] = parent2[i]
        else
            offspring1[i] = parent2[i]
            offspring2[i] = parent1[i]
        end
    end
end

"""
Single-point crossover (alternative)
"""
function single_point_crossover!(offspring1::Vector{Float32},
                                offspring2::Vector{Float32},
                                parent1::Vector{Float32},
                                parent2::Vector{Float32},
                                crossover_rate::Float32 = 0.7f0;
                                rng::AbstractRNG = Random.default_rng())
    
    if rand(rng, Float32) > crossover_rate
        offspring1 .= parent1
        offspring2 .= parent2
        return
    end
    
    # Select crossover point
    point = rand(rng, 1:12)  # Don't split complex weight
    
    # Create offspring
    offspring1[1:point] .= parent1[1:point]
    offspring1[(point+1):13] .= parent2[(point+1):13]
    
    offspring2[1:point] .= parent2[1:point]
    offspring2[(point+1):13] .= parent1[(point+1):13]
end

# =============================================================================
# MUTATION OPERATORS
# =============================================================================

"""
Gaussian mutation with parameter-specific scaling
Modifies chromosome in place
"""
function gaussian_mutation!(chromosome::Vector{Float32},
                           mutation_rate::Float32,
                           ranges;
                           rng::AbstractRNG = Random.default_rng(),
                           mutation_strength::Float32 = 0.1f0)
    
    for i in 1:13
        # Apply mutation with probability
        if rand(rng, Float32) < mutation_rate
            bounds = Main.ParameterEncoding.get_parameter_bounds(Int32(i), ranges)
            
            if i == 7  # Binary parameter (enable_clamping)
                # Flip bit
                chromosome[i] = chromosome[i] > 0.5f0 ? 0.0f0 : 1.0f0
                
            elseif i == 11  # Discrete parameter (phase_error_history_length)
                # Random walk in discrete space
                n_options = Int32(bounds[2])
                current = round(Int32, chromosome[i])
                
                # Move up or down by 1
                if rand(rng) < 0.5
                    new_val = max(1, current - 1)
                else
                    new_val = min(n_options, current + 1)
                end
                chromosome[i] = Float32(new_val)
                
            else  # Continuous parameters
                # Gaussian perturbation scaled by range
                range_width = bounds[2] - bounds[1]
                perturbation = randn(rng, Float32) * mutation_strength * range_width
                
                # Apply mutation and clamp
                chromosome[i] = clamp(chromosome[i] + perturbation, bounds...)
            end
        end
    end
    
    return chromosome
end

"""
Adaptive mutation - mutation strength decreases with fitness
"""
function adaptive_mutation!(chromosome::Vector{Float32},
                           base_mutation_rate::Float32,
                           fitness::Float32,
                           max_fitness::Float32,
                           ranges;
                           rng::AbstractRNG = Random.default_rng())
    
    # Scale mutation rate based on fitness
    # Higher fitness = lower mutation rate
    fitness_ratio = max(0.0f0, fitness / max(max_fitness, 1.0f-6))
    mutation_rate = base_mutation_rate * (1.0f0 - 0.5f0 * fitness_ratio)
    
    # Scale mutation strength
    mutation_strength = 0.2f0 * (1.0f0 - 0.7f0 * fitness_ratio)
    
    gaussian_mutation!(chromosome, mutation_rate, ranges, 
                      rng=rng, mutation_strength=mutation_strength)
end

# =============================================================================
# POPULATION EVOLUTION
# =============================================================================

"""
Create offspring from parents using crossover and mutation
"""
function create_offspring(parent1_idx::Int32,
                        parent2_idx::Int32,
                        population::Matrix{Float32},
                        crossover_rate::Float32,
                        mutation_rate::Float32,
                        ranges;
                        rng::AbstractRNG = Random.default_rng())::Tuple{Vector{Float32}, Vector{Float32}}
    
    # Get parent chromosomes
    parent1 = population[parent1_idx, :]
    parent2 = population[parent2_idx, :]
    
    # Allocate offspring
    offspring1 = Vector{Float32}(undef, 13)
    offspring2 = Vector{Float32}(undef, 13)
    
    # Apply crossover
    uniform_crossover!(offspring1, offspring2, parent1, parent2, 
                      crossover_rate, rng=rng)
    
    # Apply mutation
    gaussian_mutation!(offspring1, mutation_rate, ranges, rng=rng)
    gaussian_mutation!(offspring2, mutation_rate, ranges, rng=rng)
    
    # Ensure bounds
    Main.ParameterEncoding.apply_bounds!(offspring1, ranges)
    Main.ParameterEncoding.apply_bounds!(offspring2, ranges)
    
    return (offspring1, offspring2)
end

"""
Evolve entire population for one generation
"""
function evolve_population!(population::Matrix{Float32},
                           fitness::Vector{Float32},
                           ga_params,
                           ranges;
                           rng::AbstractRNG = Random.default_rng())
    
    pop_size = size(population, 1)
    new_population = Matrix{Float32}(undef, pop_size, 13)
    
    # Preserve elite
    elite_indices = elite_selection(fitness, ga_params.elite_size)
    for (i, idx) in enumerate(elite_indices)
        new_population[i, :] = population[idx, :]
    end
    
    # Generate offspring to fill rest of population
    offspring_start = length(elite_indices) + 1
    
    for i in offspring_start:2:pop_size
        # Select parents via tournament
        parent1_idx = tournament_selection(population, fitness, 
                                          ga_params.tournament_size, rng=rng)
        parent2_idx = tournament_selection(population, fitness, 
                                          ga_params.tournament_size, rng=rng)
        
        # Create offspring
        offspring1, offspring2 = create_offspring(
            Int32(parent1_idx), Int32(parent2_idx),
            population,
            ga_params.crossover_rate,
            ga_params.mutation_rate,
            ranges,
            rng=rng
        )
        
        # Add to new population
        new_population[i, :] = offspring1
        if i + 1 <= pop_size
            new_population[i + 1, :] = offspring2
        end
    end
    
    # Replace old population
    population .= new_population
    
    return population
end

# =============================================================================
# DIVERSITY METRICS
# =============================================================================

"""
Calculate population diversity (average pairwise distance)
"""
function population_diversity(population::Matrix{Float32})::Float32
    pop_size = size(population, 1)
    
    if pop_size < 2
        return 0.0f0
    end
    
    total_distance = 0.0f0
    n_pairs = 0
    
    for i in 1:(pop_size-1)
        for j in (i+1):pop_size
            # Euclidean distance between chromosomes
            distance = sqrt(sum((population[i, :] .- population[j, :]).^2))
            total_distance += distance
            n_pairs += 1
        end
    end
    
    return Float32(total_distance / n_pairs)
end

"""
Calculate fitness variance (measure of selection pressure)
"""
function fitness_variance(fitness::Vector{Float32})::Float32
    return var(fitness)
end

# =============================================================================
# SPECIALIZED OPERATORS
# =============================================================================

"""
Blend crossover (BLX-α) for continuous parameters
"""
function blend_crossover!(offspring::Vector{Float32},
                         parent1::Vector{Float32},
                         parent2::Vector{Float32},
                         ranges;
                         alpha::Float32 = 0.5f0,
                         rng::AbstractRNG = Random.default_rng())
    
    for i in 1:13
        if i in [7, 11]  # Skip binary and discrete parameters
            # Use uniform crossover for these
            offspring[i] = rand(rng) < 0.5 ? parent1[i] : parent2[i]
        else
            # Blend for continuous parameters
            min_val = min(parent1[i], parent2[i])
            max_val = max(parent1[i], parent2[i])
            range = max_val - min_val
            
            # Extend range by alpha
            min_val -= alpha * range
            max_val += alpha * range
            
            # Generate offspring value
            offspring[i] = min_val + rand(rng, Float32) * (max_val - min_val)
            
            # Apply bounds
            bounds = Main.ParameterEncoding.get_parameter_bounds(Int32(i), ranges)
            offspring[i] = clamp(offspring[i], bounds...)
        end
    end
    
    return offspring
end

"""
Polynomial mutation for fine-tuning
"""
function polynomial_mutation!(chromosome::Vector{Float32},
                             mutation_rate::Float32,
                             ranges;
                             eta::Float32 = 20.0f0,  # Distribution index
                             rng::AbstractRNG = Random.default_rng())
    
    for i in 1:13
        if rand(rng, Float32) < mutation_rate
            if i in [7, 11]  # Binary or discrete
                # Use standard mutation for these
                continue
            end
            
            bounds = Main.ParameterEncoding.get_parameter_bounds(Int32(i), ranges)
            
            # Polynomial mutation
            y = chromosome[i]
            yl = bounds[1]
            yu = bounds[2]
            
            delta1 = (y - yl) / (yu - yl)
            delta2 = (yu - y) / (yu - yl)
            
            mut_pow = 1.0f0 / (eta + 1.0f0)
            
            u = rand(rng, Float32)
            if u <= 0.5f0
                xy = 1.0f0 - delta1
                val = 2.0f0 * u + (1.0f0 - 2.0f0 * u) * xy^(eta + 1.0f0)
                deltaq = val^mut_pow - 1.0f0
            else
                xy = 1.0f0 - delta2
                val = 2.0f0 * (1.0f0 - u) + 2.0f0 * (u - 0.5f0) * xy^(eta + 1.0f0)
                deltaq = 1.0f0 - val^mut_pow
            end
            
            y = y + deltaq * (yu - yl)
            chromosome[i] = clamp(y, bounds...)
        end
    end
    
    return chromosome
end

end # module GeneticOperators