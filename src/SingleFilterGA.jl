module SingleFilterGA

using Random
using Statistics
using Printf
using ..GATypes
using ..ParameterEncoding
using ..GeneticOperators
using ..PopulationInit

export SingleFilterGAComplete, evolve!, get_best_solution, get_statistics

# --- Moved struct definition here from GATypes ---
mutable struct SingleFilterGAComplete{M<:AbstractMatrix{Float32}, V<:AbstractVector{Float32}}
    period::Int32
    filter_index::Int32
    population::M
    fitness::V
    best_chromosome::V
    best_fitness::Float32
    best_generation::Int32
    generations_since_improvement::Int32
    generation::Int32
    total_evaluations::Int64
    converged::Bool
    param_ranges::ParameterRanges
    ga_params::GAParameters
    fitness_history::Vector{Float32}
    diversity_history::Vector{Float32}
    mean_fitness_history::Vector{Float32}
    rng::AbstractRNG
end

# --- Struct to replace Dictionary for statistics ---
struct GAStatistics
    period::Int32
    filter_index::Int32
    generation::Int32
    converged::Bool
    best_fitness::Float32
    best_generation::Int32
    generations_since_improvement::Int32
    total_evaluations::Int64
    current_mean_fitness::Float32
    current_max_fitness::Float32
    current_min_fitness::Float32
    current_fitness_std::Float32
    current_diversity::Float32
    mean_diversity::Float32
end

# --- Constructor ---
function SingleFilterGAComplete(
    period::Int32, filter_index::Int32, pop_size::Int32,
    param_ranges::ParameterRanges, ga_params::GAParameters;
    ArrayType::Type=Array{Float32}, seed::Union{Int, Nothing}=nothing,
    init_strategy::Symbol=:random, init_chromosome::Union{AbstractVector{Float32}, Nothing}=nothing
)
    rng = (seed === nothing) ? Random.default_rng() : MersenneTwister(seed)
    
    if init_strategy == :seeded && init_chromosome !== nothing
        population = initialize_from_seed(init_chromosome, pop_size, param_ranges, rng=rng)
    else
        population = initialize_population(pop_size, param_ranges, ArrayType=ArrayType, rng=rng)
    end
    
    fitness = similar(population, pop_size)
    fill!(fitness, 0.0f0)
    best_chromosome = similar(population, 13)
    fill!(best_chromosome, 0.0f0)
    
    return SingleFilterGAComplete{typeof(population), typeof(fitness)}(
        period, filter_index, population, fitness, best_chromosome,
        -Inf32, 0, 0, 0, 0, false, param_ranges, ga_params,
        Float32[], Float32[], Float32[], rng
    )
end

# --- Functions ---

function evaluate_fitness!(ga::SingleFilterGAComplete, fitness_function::Function)
    pop_size = size(ga.population, 1)
    cpu_population = Array(ga.population)
    cpu_fitness = zeros(Float32, pop_size)

    for i in 1:pop_size
        chromosome = @view cpu_population[i, :]
        cpu_fitness[i] = fitness_function(chromosome, ga.period)
        ga.total_evaluations += 1
    end
    
    copyto!(ga.fitness, cpu_fitness)
end

function update_best!(ga::SingleFilterGAComplete)
    cpu_fitness = Array(ga.fitness)
    if isempty(cpu_fitness) return end
    
    best_idx = argmax(cpu_fitness)
    current_best_fitness = cpu_fitness[best_idx]
    
    if current_best_fitness > ga.best_fitness
        ga.best_fitness = current_best_fitness
        ga.best_chromosome .= @view ga.population[best_idx, :]
        ga.best_generation = ga.generation
        ga.generations_since_improvement = 0
    else
        ga.generations_since_improvement += 1
    end
end

function check_convergence(ga::SingleFilterGAComplete)::Bool
    if ga.converged return true end
    if ga.generation < 10 return false end
    if ga.generations_since_improvement >= ga.ga_params.early_stopping_patience return true end
    if ga.generation >= ga.ga_params.max_generations return true end

    if length(ga.fitness_history) >= 10
        recent_fitness = @view ga.fitness_history[end-9:end]
        if var(recent_fitness) < ga.ga_params.convergence_threshold
            return true
        end
    end
    
    return false
end

function evolve!(ga::SingleFilterGAComplete; fitness_function::Function, verbose::Bool=false)
    if ga.generation == 0
        evaluate_fitness!(ga, fitness_function)
    end
    
    update_best!(ga)
    
    push!(ga.fitness_history, ga.best_fitness)
    # push!(ga.diversity_history, population_diversity(ga.population)) # population_diversity would need to be made generic
    push!(ga.mean_fitness_history, mean(Array(ga.fitness)))
    
    if check_convergence(ga)
        ga.converged = true
        if verbose
             println("Filter $(ga.period): Converged at generation $(ga.generation)")
        end
        return
    end
    
    evolve_population!(ga.population, ga.fitness, ga.ga_params, ga.param_ranges, rng=ga.rng)
    
    ga.generation += 1
    
    evaluate_fitness!(ga, fitness_function)
    
    if verbose && ga.generation % 10 == 0
        @printf("Filter %d - Gen %d: Best=%.4f, Mean=%.4f\n",
                ga.period, ga.generation, ga.best_fitness, 
                mean(ga.fitness))
    end
end

function get_best_solution(ga::SingleFilterGAComplete)
    return decode_chromosome(Array(ga.best_chromosome), ga.param_ranges)
end

function reset_ga!(ga::SingleFilterGAComplete; keep_best::Bool = true, init_strategy::Symbol = :random)
    pop_size = Int32(size(ga.population, 1))
    
    if keep_best && ga.best_fitness > -Inf32
        ga.population = initialize_from_seed(ga.best_chromosome, pop_size,
                                             ga.param_ranges, 
                                             diversity=0.2f0, rng=ga.rng)
    else
        ga.population = initialize_population(pop_size, ga.param_ranges, 
                                                ArrayType=typeof(ga.population), rng=ga.rng)
        ga.best_chromosome .= 0
        ga.best_fitness = -Inf32
    end
    
    ga.fitness .= 0
    ga.generation = 0
    ga.generations_since_improvement = 0
    ga.converged = false
    ga.best_generation = 0
    
    empty!(ga.fitness_history)
    empty!(ga.diversity_history)
    empty!(ga.mean_fitness_history)
end

function get_statistics(ga::SingleFilterGAComplete)::GAStatistics
    cpu_fitness = Array(ga.fitness)
    
    return GAStatistics(
        ga.period,
        ga.filter_index,
        ga.generation,
        ga.converged,
        ga.best_fitness,
        ga.best_generation,
        ga.generations_since_improvement,
        ga.total_evaluations,
        isempty(cpu_fitness) ? 0.0f0 : mean(cpu_fitness),
        isempty(cpu_fitness) ? 0.0f0 : maximum(cpu_fitness),
        isempty(cpu_fitness) ? 0.0f0 : minimum(cpu_fitness),
        isempty(cpu_fitness) || length(cpu_fitness) < 2 ? 0.0f0 : std(cpu_fitness),
        isempty(ga.diversity_history) ? 0.0f0 : ga.diversity_history[end],
        isempty(ga.diversity_history) ? 0.0f0 : mean(ga.diversity_history)
    )
end

end # module SingleFilterGA