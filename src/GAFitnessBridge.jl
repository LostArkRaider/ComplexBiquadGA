# src/GAFitnessBridge.jl - Bridge between GATypes and FitnessEvaluation

"""
GA Fitness Bridge Module - Chunk 3

Provides integration between the existing GA infrastructure (GATypes) 
and the new fitness evaluation system (Chunk 3).

This module ensures that Chunk 3 works seamlessly with the existing
GA system while maintaining the project's architectural design.
"""

module GAFitnessBridge

using ..GATypes
using ..FitnessEvaluation
using ..FilterIntegration

export evaluate_chromosome_fitness,
       evaluate_population_fitness_ga,
       update_filter_ga_fitness!,
       update_filter_bank_fitness!

# =============================================================================
# FITNESS EVALUATION FOR GA TYPES
# =============================================================================

"""
Evaluate fitness for a chromosome from the GA system
This is the main integration point between GATypes and FitnessEvaluation
"""
function evaluate_chromosome_fitness(
    chromosome::Vector{Float32},
    fibonacci_number::Int32;
    weights::Union{FitnessEvaluation.FitnessWeights, Nothing} = nothing,
    use_pll::Bool = true
)::Float32
    
    # Validate chromosome
    @assert length(chromosome) == 13 "Chromosome must have exactly 13 genes"
    
    # Use FitnessEvaluation's function directly
    # It already accepts Vector{Float32}
    return FitnessEvaluation.evaluate_filter_fitness(
        chromosome,
        fibonacci_number,
        weights = weights
    )
end

"""
Evaluate fitness for an entire population from SingleFilterGA
"""
function evaluate_population_fitness_ga(
    filter_ga::GATypes.SingleFilterGA;
    weights::Union{FitnessEvaluation.FitnessWeights, Nothing} = nothing,
    use_pll::Bool = true
)::Vector{Float32}
    
    pop_size = size(filter_ga.population, 1)
    fitness_values = Vector{Float32}(undef, pop_size)
    
    # Get fibonacci number from the filter
    fibonacci_number = filter_ga.period
    
    # Evaluate each chromosome
    for i in 1:pop_size
        chromosome = filter_ga.population[i, :]
        fitness_values[i] = evaluate_chromosome_fitness(
            chromosome,
            fibonacci_number,
            weights = weights,
            use_pll = use_pll
        )
    end
    
    return fitness_values
end

"""
Update fitness values in a SingleFilterGA
Evaluates the entire population and updates the fitness array
"""
function update_filter_ga_fitness!(
    filter_ga::GATypes.SingleFilterGA;
    weights::Union{FitnessEvaluation.FitnessWeights, Nothing} = nothing,
    use_pll::Bool = true
)
    # Evaluate all chromosomes
    fitness_values = evaluate_population_fitness_ga(
        filter_ga,
        weights = weights,
        use_pll = use_pll
    )
    
    # Update fitness array
    filter_ga.fitness .= fitness_values
    
    # Update best fitness tracking
    best_idx = argmax(fitness_values)
    if fitness_values[best_idx] > filter_ga.best_fitness
        filter_ga.best_fitness = fitness_values[best_idx]
        filter_ga.best_chromosome .= filter_ga.population[best_idx, :]
        filter_ga.generations_since_improvement = Int32(0)
    else
        filter_ga.generations_since_improvement += Int32(1)
    end
    
    return nothing
end

"""
Update fitness for an entire FilterBankGA
Processes all filters in the bank
"""
function update_filter_bank_fitness!(
    filter_bank::GATypes.FilterBankGA;
    weights::Union{FitnessEvaluation.FitnessWeights, Nothing} = nothing,
    use_pll::Bool = true
)
    # Update each filter's fitness independently
    for filter_ga in filter_bank.filter_gas
        if !filter_ga.converged
            update_filter_ga_fitness!(
                filter_ga,
                weights = weights,
                use_pll = use_pll
            )
        end
    end
    
    # Update bank-level statistics
    filter_bank.generation += Int32(1)
    
    # Track best fitness across all filters
    best_fitnesses = [ga.best_fitness for ga in filter_bank.filter_gas]
    push!(filter_bank.best_fitness_history, maximum(best_fitnesses))
    
    return nothing
end

# =============================================================================
# STUB FITNESS REPLACEMENT
# =============================================================================

"""
Drop-in replacement for the stub fitness function used in Chunks 1-2
This function signature matches what SingleFilterGA expects
"""
function evaluate_fitness_stub_replacement(
    chromosome::Vector{Float32},
    period::Int32
)::Float32
    # Use default weights or load from config
    weights = FitnessEvaluation.create_default_weights()
    FitnessEvaluation.normalize_weights!(weights)
    
    return evaluate_chromosome_fitness(
        chromosome,
        period,
        weights = weights,
        use_pll = true
    )
end

# =============================================================================
# CONFIGURATION INTEGRATION
# =============================================================================

"""
Load fitness weights from instrument configuration
Integrates with the existing TOML configuration system
"""
function load_weights_from_config(
    instrument_config::GATypes.InstrumentConfig
)::FitnessEvaluation.FitnessWeights
    
    # Try to load from instrument's config file
    config_path = instrument_config.config_path
    
    if isfile(config_path)
        return FitnessEvaluation.load_fitness_weights(config_path)
    else
        # Use defaults if no config
        weights = FitnessEvaluation.create_default_weights()
        FitnessEvaluation.normalize_weights!(weights)
        return weights
    end
end

"""
Create fitness configuration from GA parameters
"""
function create_fitness_config_from_ga(
    ga_params::GATypes.GAParameters;
    weights::Union{FitnessEvaluation.FitnessWeights, Nothing} = nothing
)::FitnessEvaluation.FitnessConfig
    
    if weights === nothing
        weights = FitnessEvaluation.create_default_weights()
        FitnessEvaluation.normalize_weights!(weights)
    end
    
    # Map GA parameters to fitness config
    return FitnessEvaluation.create_fitness_config(
        weights,
        use_pll = true,  # Could be made configurable
        signal_length = Int32(1000),
        warmup_samples = Int32(100),
        enable_caching = true,
        parallel_evaluation = false,  # Could use ga_params if it has this
        max_cache_size = Int32(1000)
    )
end

end # module GAFitnessBridge