module FitnessEvaluation

using ..FilterIntegration
using ..SignalMetrics
using Statistics
using TOML

export FitnessWeights, FitnessConfig, evaluate_fitness, evaluate_population_fitness,
       load_fitness_weights, FitnessCache, FitnessResult, PopulationStats

# =============================================================================
# FITNESS CONFIGURATION & DATA STRUCTURES (NO-DICT REFACTOR)
# =============================================================================

@with_kw mutable struct FitnessWeights
    snr_weight::Float32 = 0.35f0
    lock_quality_weight::Float32 = 0.35f0
    ringing_penalty_weight::Float32 = 0.20f0
    frequency_selectivity_weight::Float32 = 0.10f0
    normalized::Bool = false
end

struct FitnessConfig
    weights::FitnessWeights
    use_pll::Bool
    target_period::Float32
    signal_length::Int32
    warmup_samples::Int32
    enable_caching::Bool
end

struct FitnessResult
    total_fitness::Float32
    metrics::FilterMetrics
    evaluation_time_ms::Float32
    cache_hit::Bool
end

struct PopulationStats
    mean::Float32
    std::Float32
    min::Float32
    max::Float32
    median::Float32
end

# --- Fitness Cache implemented with Vectors instead of Dict ---
mutable struct FitnessCache
    keys::Vector{UInt64}
    values::Vector{FitnessResult}
    max_size::Int32
    hits::Int64
    misses::Int64
    
    function FitnessCache(max_size::Int32 = Int32(1000))
        new(Vector{UInt64}(), Vector{FitnessResult}(), max_size, 0, 0)
    end
end

# =============================================================================
# CACHE & WEIGHT MANAGEMENT
# =============================================================================

function get_cache_key(chromosome::AbstractVector{Float32})::UInt64
    return hash(chromosome)
end

function check_cache(cache::FitnessCache, chromosome::AbstractVector{Float32})::Union{FitnessResult, Nothing}
    key = get_cache_key(chromosome)
    idx = findfirst(==(key), cache.keys)
    
    if idx !== nothing
        cache.hits += 1
        # Return a new result indicating it was a cache hit
        cached_val = cache.values[idx]
        return FitnessResult(cached_val.total_fitness, cached_val.metrics, cached_val.evaluation_time_ms, true)
    else
        cache.misses += 1
        return nothing
    end
end

function store_in_cache!(cache::FitnessCache, chromosome::AbstractVector{Float32}, result::FitnessResult)
    if length(cache.keys) >= cache.max_size
        # Simple FIFO eviction
        popfirst!(cache.keys)
        popfirst!(cache.values)
    end
    push!(cache.keys, get_cache_key(chromosome))
    push!(cache.values, result)
end

# ... (Weight management functions like normalize_weights! and load_fitness_weights remain largely the same)

# =============================================================================
# FITNESS EVALUATION (Hardware-Agnostic)
# =============================================================================

function evaluate_fitness(
    chromosome::V,
    fibonacci_number::Int32,
    config::FitnessConfig;
    test_signal::Union{AbstractVector{ComplexF32}, Nothing} = nothing,
    cache::Union{FitnessCache, Nothing} = nothing
)::FitnessResult where {V<:AbstractVector{Float32}}
    
    start_time = time_ns()
    
    if cache !== nothing && config.enable_caching
        cached_result = check_cache(cache, chromosome)
        if cached_result !== nothing
            return cached_result
        end
    end
    
    # Test signal generation remains on CPU
    if test_signal === nothing
        # ... logic to generate test_signal ...
    end
    
    filter = create_filter_from_chromosome(chromosome, fibonacci_number, Int32(1), config.use_pll)
    output_signal = evaluate_filter_with_signal(filter, test_signal)
    
    # Metrics calculations are now hardware-agnostic
    metrics = calculate_all_metrics(output_signal, test_signal, filter, target_period=config.target_period)
    
    weights = config.weights
    total_fitness = (
        weights.snr_weight * metrics.snr.normalized_value +
        weights.lock_quality_weight * metrics.lock_quality.normalized_value +
        weights.ringing_penalty_weight * metrics.ringing_penalty.normalized_value +
        weights.frequency_selectivity_weight * metrics.frequency_selectivity.normalized_value
    )
    total_fitness = clamp(total_fitness, 0.0f0, 1.0f0)
    
    eval_time_ms = Float32((time_ns() - start_time) / 1_000_000.0)
    result = FitnessResult(total_fitness, metrics, eval_time_ms, false)
    
    if cache !== nothing && config.enable_caching
        store_in_cache!(cache, chromosome, result)
    end
    
    return result
end

function evaluate_population_fitness(
    population::M,
    fibonacci_number::Int32,
    config::FitnessConfig;
    cache::Union{FitnessCache, Nothing} = nothing
)::Vector{Float32} where {M<:AbstractMatrix{Float32}}
    
    pop_size = size(population, 1)
    fitness_values = Vector{Float32}(undef, pop_size)
    
    # Test signal generated once on CPU
    test_signal = Vector{ComplexF32}() # Placeholder for signal generation
    
    # This loop can be parallelized
    for i in 1:pop_size
        result = evaluate_fitness(
            @view(population[i, :]),
            fibonacci_number,
            config,
            test_signal=test_signal,
            cache=cache
        )
        fitness_values[i] = result.total_fitness
    end
    
    return fitness_values
end

function get_population_stats(fitness_values::Vector{Float32})::PopulationStats
    return PopulationStats(
        mean(fitness_values),
        std(fitness_values),
        minimum(fitness_values),
        maximum(fitness_values),
        median(fitness_values)
    )
end

end # module FitnessEvaluation