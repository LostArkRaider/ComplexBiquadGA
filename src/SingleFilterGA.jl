# src/SingleFilterGA.jl - Complete GA Implementation for Single Filter
# Replaces the stub with full genetic algorithm functionality

module SingleFilterGA

using Random
using Statistics
using Printf

export SingleFilterGAComplete, evolve!, evaluate_fitness!, update_best!,
       check_convergence, get_best_solution, reset_ga!, get_statistics

# Include dependencies
if !isdefined(Main, :ParameterEncoding)
    include("ParameterEncoding.jl")
end
if !isdefined(Main, :GeneticOperators)
    include("GeneticOperators.jl")
end
if !isdefined(Main, :PopulationInit)
    include("PopulationInit.jl")
end

using Main.ParameterEncoding
using Main.GeneticOperators
using Main.PopulationInit

# =============================================================================
# SINGLE FILTER GA STRUCTURE
# =============================================================================

"""
Complete GA implementation for a single filter
"""
mutable struct SingleFilterGAComplete
    # Filter identity
    period::Int32                            # Fibonacci period
    filter_index::Int32                      # Position in bank
    
    # GA population (13 parameters per individual)
    population::Matrix{Float32}              # population_size × 13
    fitness::Vector{Float32}                 # population_size
    
    # Best solution tracking
    best_chromosome::Vector{Float32}         # 13 parameters
    best_fitness::Float32
    best_generation::Int32
    generations_since_improvement::Int32
    
    # Evolution state
    generation::Int32
    total_evaluations::Int64
    converged::Bool
    
    # Configuration
    param_ranges                             # ParameterRanges struct
    ga_params                                # GAParameters struct
    
    # Statistics
    fitness_history::Vector{Float32}        # Best fitness per generation
    diversity_history::Vector{Float32}      # Population diversity per generation
    mean_fitness_history::Vector{Float32}   # Mean fitness per generation
    
    # Random number generator
    rng::AbstractRNG
end

"""
Constructor for SingleFilterGAComplete
"""
function SingleFilterGAComplete(period::Int32, 
                               filter_index::Int32,
                               pop_size::Int32,
                               param_ranges,
                               ga_params;
                               seed::Union{Int, Nothing} = nothing,
                               init_strategy::Symbol = :random,
                               init_chromosome::Union{Vector{Float32}, Nothing} = nothing)
    
    # Set up RNG
    rng = seed === nothing ? Random.default_rng() : MersenneTwister(seed)
    
    # Initialize population based on strategy
    if init_strategy == :seeded && init_chromosome !== nothing
        population = initialize_from_seed(init_chromosome, pop_size, 
                                         param_ranges, rng=rng)
    elseif init_strategy == :lhs
        population = Main.PopulationInit.initialize_lhs(pop_size, param_ranges, rng=rng)
    elseif init_strategy == :diverse
        population = Main.PopulationInit.initialize_diverse(pop_size, param_ranges, 
                                                           nothing, rng=rng)
    else  # :random
        population = initialize_population(pop_size, param_ranges, rng=rng)
    end
    
    # Initialize fitness
    fitness = zeros(Float32, pop_size)
    
    # Initialize best solution
    best_chromosome = zeros(Float32, 13)
    
    # Initialize history
    fitness_history = Float32[]
    diversity_history = Float32[]
    mean_fitness_history = Float32[]
    
    return SingleFilterGAComplete(
        period, filter_index,
        population, fitness,
        best_chromosome, -Inf32, 0, 0,
        0, 0, false,
        param_ranges, ga_params,
        fitness_history, diversity_history, mean_fitness_history,
        rng
    )
end

# =============================================================================
# FITNESS EVALUATION
# =============================================================================

"""
Evaluate fitness for entire population
This is a stub that returns random fitness - will be replaced in Chunk 3
"""
function evaluate_fitness!(ga::SingleFilterGAComplete, 
                          fitness_function::Union{Function, Nothing} = nothing)
    
    pop_size = size(ga.population, 1)
    
    for i in 1:pop_size
        chromosome = ga.population[i, :]
        
        if fitness_function !== nothing
            # Use provided fitness function
            ga.fitness[i] = fitness_function(chromosome, ga.period)
        else
            # Stub: return random fitness for testing
            # In Chunk 3, this will evaluate actual filter quality
            ga.fitness[i] = rand(ga.rng, Float32)
            
            # Add small penalty for extreme parameters (helps testing)
            for j in 1:13
                bounds = get_parameter_bounds(Int32(j), ga.param_ranges)
                range = bounds[2] - bounds[1]
                normalized = (chromosome[j] - bounds[1]) / range
                
                # Penalize extreme values
                if normalized < 0.1f0 || normalized > 0.9f0
                    ga.fitness[i] *= 0.95f0
                end
            end
        end
        
        ga.total_evaluations += 1
    end
    
    return ga.fitness
end

# =============================================================================
# EVOLUTION
# =============================================================================

"""
Evolve population for one generation
"""
function evolve!(ga::SingleFilterGAComplete;
                fitness_function::Union{Function, Nothing} = nothing,
                verbose::Bool = false)
    
    # For generation 0, we need initial fitness evaluation
    # For subsequent generations, fitness was already evaluated at end of previous generation
    if ga.generation == 0
        evaluate_fitness!(ga, fitness_function)
    end
    
    # Update best solution
    update_best!(ga)
    
    # Record statistics
    push!(ga.fitness_history, ga.best_fitness)
    push!(ga.diversity_history, population_diversity(ga.population))
    push!(ga.mean_fitness_history, mean(ga.fitness))
    
    # Check convergence
    if check_convergence(ga)
        ga.converged = true
        if verbose
            println("Filter $(ga.period): Converged at generation $(ga.generation)")
        end
        return
    end
    
    # Evolve population
    evolve_population!(ga.population, ga.fitness, ga.ga_params, 
                      ga.param_ranges, rng=ga.rng)
    
    # Increment generation
    ga.generation += 1
    
    # Evaluate new population (needed for next generation)
    evaluate_fitness!(ga, fitness_function)
    
    # Verbose output
    if verbose && ga.generation % 10 == 0
        @printf("Filter %d - Gen %d: Best=%.4f, Mean=%.4f, Diversity=%.4f\n",
                ga.period, ga.generation, ga.best_fitness, 
                mean(ga.fitness), ga.diversity_history[end])
    end
end

"""
Update best solution tracking
"""
function update_best!(ga::SingleFilterGAComplete)
    # Find best individual in current population
    best_idx = argmax(ga.fitness)
    current_best_fitness = ga.fitness[best_idx]
    
    # Update if improved
    if current_best_fitness > ga.best_fitness
        ga.best_fitness = current_best_fitness
        ga.best_chromosome .= ga.population[best_idx, :]
        ga.best_generation = ga.generation
        ga.generations_since_improvement = 0
    else
        ga.generations_since_improvement += 1
    end
end

# =============================================================================
# CONVERGENCE DETECTION
# =============================================================================

"""
Check if GA has converged
"""
function check_convergence(ga::SingleFilterGAComplete)::Bool
    # Already converged
    if ga.converged
        return true
    end
    
    # Not enough generations
    if ga.generation < 10
        return false
    end
    
    # Check fitness improvement stagnation
    if ga.generations_since_improvement >= ga.ga_params.early_stopping_patience
        return true
    end
    
    # Check fitness variance
    if length(ga.fitness_history) >= 10
        recent_fitness = ga.fitness_history[end-9:end]
        fitness_variance = var(recent_fitness)
        
        if fitness_variance < ga.ga_params.convergence_threshold
            return true
        end
    end
    
    # Check diversity
    if length(ga.diversity_history) >= 5
        recent_diversity = mean(ga.diversity_history[end-4:end])
        if recent_diversity < 0.01f0  # Very low diversity
            return true
        end
    end
    
    # Check max generations
    if ga.generation >= ga.ga_params.max_generations
        return true
    end
    
    return false
end

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

"""
Get best solution as decoded parameters
"""
function get_best_solution(ga::SingleFilterGAComplete)
    return decode_chromosome(ga.best_chromosome, ga.param_ranges)
end

"""
Reset GA with new population
"""
function reset_ga!(ga::SingleFilterGAComplete;
                  keep_best::Bool = true,
                  init_strategy::Symbol = :random)
    
    pop_size = Int32(size(ga.population, 1))  # Convert to Int32
    
    if keep_best && ga.best_fitness > -Inf32
        # Reinitialize around best solution
        ga.population = initialize_from_seed(ga.best_chromosome, pop_size,
                                            ga.param_ranges, 
                                            diversity=0.2f0, rng=ga.rng)
    else
        # Complete reinitialization
        if init_strategy == :lhs
            ga.population = Main.PopulationInit.initialize_lhs(pop_size, 
                                                              ga.param_ranges, 
                                                              rng=ga.rng)
        else
            ga.population = initialize_population(pop_size, ga.param_ranges, 
                                                 rng=ga.rng)
        end
        
        ga.best_chromosome .= 0
        ga.best_fitness = -Inf32
    end
    
    # Reset state
    ga.fitness .= 0
    ga.generation = 0
    ga.generations_since_improvement = 0
    ga.converged = false
    ga.best_generation = 0
    
    # Clear history
    empty!(ga.fitness_history)
    empty!(ga.diversity_history)
    empty!(ga.mean_fitness_history)
end

"""
Get GA statistics
"""
function get_statistics(ga::SingleFilterGAComplete)::Dict{String, Any}
    stats = Dict{String, Any}()
    
    stats["period"] = ga.period
    stats["filter_index"] = ga.filter_index
    stats["generation"] = ga.generation
    stats["converged"] = ga.converged
    stats["best_fitness"] = ga.best_fitness
    stats["best_generation"] = ga.best_generation
    stats["generations_since_improvement"] = ga.generations_since_improvement
    stats["total_evaluations"] = ga.total_evaluations
    
    if !isempty(ga.fitness)
        stats["current_mean_fitness"] = mean(ga.fitness)
        stats["current_max_fitness"] = maximum(ga.fitness)
        stats["current_min_fitness"] = minimum(ga.fitness)
        stats["current_fitness_std"] = std(ga.fitness)
    end
    
    if !isempty(ga.diversity_history)
        stats["current_diversity"] = ga.diversity_history[end]
        stats["mean_diversity"] = mean(ga.diversity_history)
    end
    
    return stats
end

# =============================================================================
# ADVANCED FEATURES
# =============================================================================

"""
Inject new genetic material to increase diversity
"""
function inject_diversity!(ga::SingleFilterGAComplete;
                         injection_rate::Float32 = 0.1f0)
    
    pop_size = size(ga.population, 1)
    n_inject = round(Int, pop_size * injection_rate)
    
    if n_inject > 0
        # Generate new random individuals
        new_individuals = initialize_population(Int32(n_inject), 
                                               ga.param_ranges, rng=ga.rng)
        
        # Replace worst individuals
        worst_indices = sortperm(ga.fitness)[1:n_inject]
        
        for (i, idx) in enumerate(worst_indices)
            ga.population[idx, :] = new_individuals[i, :]
            ga.fitness[idx] = 0.0f0  # Will be evaluated next generation
        end
    end
end

"""
Apply adaptive parameter control
"""
function adapt_parameters!(ga::SingleFilterGAComplete)
    # Adapt mutation rate based on diversity
    if !isempty(ga.diversity_history)
        current_diversity = ga.diversity_history[end]
        target_diversity = 0.1f0
        
        if current_diversity < target_diversity * 0.5f0
            # Low diversity - increase mutation
            ga.ga_params = Main.GATypes.GAParameters(
                mutation_rate = min(0.3f0, ga.ga_params.mutation_rate * 1.1f0),
                ga.ga_params.crossover_rate,
                ga.ga_params.elite_size,
                ga.ga_params.tournament_size,
                ga.ga_params.max_generations,
                ga.ga_params.convergence_threshold,
                ga.ga_params.early_stopping_patience
            )
        elseif current_diversity > target_diversity * 2.0f0
            # High diversity - decrease mutation
            ga.ga_params = Main.GATypes.GAParameters(
                mutation_rate = max(0.01f0, ga.ga_params.mutation_rate * 0.9f0),
                ga.ga_params.crossover_rate,
                ga.ga_params.elite_size,
                ga.ga_params.tournament_size,
                ga.ga_params.max_generations,
                ga.ga_params.convergence_threshold,
                ga.ga_params.early_stopping_patience
            )
        end
    end
end

"""
Run complete evolution with all features
"""
function run_evolution!(ga::SingleFilterGAComplete;
                       generations::Int = 100,
                       fitness_function::Union{Function, Nothing} = nothing,
                       verbose::Bool = false,
                       adaptive::Bool = true,
                       diversity_injection::Bool = true)
    
    for gen in 1:generations
        # Check if already converged
        if ga.converged
            break
        end
        
        # Evolve one generation
        evolve!(ga, fitness_function=fitness_function, verbose=verbose)
        
        # Adaptive parameter control
        if adaptive && gen % 10 == 0
            adapt_parameters!(ga)
        end
        
        # Diversity injection
        if diversity_injection && ga.generations_since_improvement > 10
            inject_diversity!(ga, injection_rate=0.05f0)
            if verbose
                println("Filter $(ga.period): Injected diversity at generation $(ga.generation)")
            end
        end
    end
    
    if verbose
        println("\nFilter $(ga.period) Evolution Complete:")
        println("  Final generation: $(ga.generation)")
        println("  Best fitness: $(ga.best_fitness)")
        println("  Converged: $(ga.converged)")
        println("  Total evaluations: $(ga.total_evaluations)")
    end
end

end # module SingleFilterGA