module GAFitnessBridge

using ..GATypes
using ..FitnessEvaluation
using ..FilterIntegration

export evaluate_chromosome_fitness,
       evaluate_population_fitness_ga,
       update_filter_ga_fitness!,
       update_filter_bank_fitness!

# =============================================================================
# FITNESS EVALUATION FOR GENERIC GA TYPES
# =============================================================================

function evaluate_chromosome_fitness(
    chromosome::V,
    fibonacci_number::Int32;
    weights::Union{FitnessWeights, Nothing} = nothing,
    use_pll::Bool = true
)::Float32 where {V<:AbstractVector{Float32}}
    
    # Create a minimal fitness config for this evaluation
    config = FitnessEvaluation.create_fitness_config(
        weights === nothing ? FitnessEvaluation.create_default_weights() : weights,
        use_pll = use_pll,
        signal_length = Int32(500),
        warmup_samples = Int32(50),
        enable_caching = false # No caching for single ad-hoc evals
    )
    
    # FitnessEvaluation.evaluate_fitness is already hardware-agnostic
    result = evaluate_fitness(chromosome, fibonacci_number, config)
    return result.total_fitness
end

function evaluate_population_fitness_ga(
    filter_ga::SingleFilterGAComplete;
    weights::Union{FitnessWeights, Nothing} = nothing,
    use_pll::Bool = true
)::Vector{Float32}
    
    config = FitnessEvaluation.create_fitness_config(
        weights === nothing ? FitnessEvaluation.create_default_weights() : weights,
        use_pll = use_pll
    )
    
    # evaluate_population_fitness expects a Matrix, which is correct
    return evaluate_population_fitness(
        filter_ga.population,
        filter_ga.period,
        config
    )
end

function update_filter_ga_fitness!(
    filter_ga::SingleFilterGAComplete;
    weights::Union{FitnessWeights, Nothing} = nothing,
    use_pll::Bool = true
)
    # Evaluate all chromosomes and get results as a CPU vector
    fitness_values_cpu = evaluate_population_fitness_ga(
        filter_ga,
        weights = weights,
        use_pll = use_pll
    )
    
    # Copy the CPU results to the fitness vector (which could be on CPU or GPU)
    copyto!(filter_ga.fitness, fitness_values_cpu)
    
    # Update best fitness tracking
    # argmax on a GPU vector is efficient
    best_idx = argmax(filter_ga.fitness)
    current_best_fitness_on_device = filter_ga.fitness[best_idx]
    
    # To compare with a scalar, bring the single value to the CPU
    if Array(current_best_fitness_on_device)[1] > filter_ga.best_fitness
        filter_ga.best_fitness = Array(current_best_fitness_on_device)[1]
        filter_ga.best_chromosome .= @view filter_ga.population[best_idx, :]
        filter_ga.generations_since_improvement = Int32(0)
    else
        filter_ga.generations_since_improvement += Int32(1)
    end
end

function update_filter_bank_fitness!(
    filter_bank::FilterBankGAComplete;
    weights::Union{FitnessWeights, Nothing} = nothing,
    use_pll::Bool = true
)
    # This can be parallelized
    for filter_ga in filter_bank.filter_gas
        if !filter_ga.converged
            update_filter_ga_fitness!(filter_ga, weights=weights, use_pll=use_pll)
        end
    end
    
    filter_bank.generation += Int32(1)
    
    best_fitnesses = [ga.best_fitness for ga in filter_bank.filter_gas]
    push!(filter_bank.best_fitness_history, maximum(best_fitnesses))
end

end # module GAFitnessBridge