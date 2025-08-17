# src/FitnessEvaluation.jl - Main fitness evaluation module

"""
Fitness Evaluation Module - Chunk 3

Main orchestrator for filter fitness evaluation with configurable metric weights.
Bridges GA chromosomes to filter instances and calculates weighted fitness scores.

Features:
- Configurable metric weights via TOML or runtime
- Batch evaluation for populations
- Caching of test signals
- Support for both synthetic and real tick data
- Automatic weight normalization
"""

module FitnessEvaluation

using ..FilterIntegration
using ..SignalMetrics
using Statistics
using TOML

export FitnessWeights,
       FitnessConfig,
       evaluate_fitness,
       evaluate_population_fitness,
       load_fitness_weights,
       create_default_weights,
       normalize_weights!,
       FitnessCache,
       FitnessResult

# =============================================================================
# FITNESS CONFIGURATION
# =============================================================================

"""
Configurable weights for fitness metrics
All weights should be non-negative except penalties
"""
mutable struct FitnessWeights
    snr_weight::Float32
    lock_quality_weight::Float32
    ringing_penalty_weight::Float32  # Usually positive (penalty reduces fitness)
    frequency_selectivity_weight::Float32
    
    # Auto-normalization flag
    normalized::Bool
end

"""
Complete fitness evaluation configuration
"""
struct FitnessConfig
    weights::FitnessWeights
    use_pll::Bool
    target_period::Float32
    signal_length::Int32
    warmup_samples::Int32
    
    # Performance settings
    enable_caching::Bool
    parallel_evaluation::Bool
    max_cache_size::Int32
end

"""
Fitness evaluation result with breakdown
"""
struct FitnessResult
    total_fitness::Float32
    snr_contribution::Float32
    lock_quality_contribution::Float32
    ringing_penalty_contribution::Float32
    frequency_selectivity_contribution::Float32
    
    # Raw metrics
    metrics::FilterMetrics
    
    # Evaluation metadata
    evaluation_time_ms::Float32
    cache_hit::Bool
end

# =============================================================================
# WEIGHT MANAGEMENT
# =============================================================================

"""
Create default fitness weights
"""
function create_default_weights()::FitnessWeights
    return FitnessWeights(
        0.35f0,  # SNR
        0.35f0,  # Lock quality
        0.20f0,  # Ringing penalty
        0.10f0,  # Frequency selectivity
        false    # Not normalized yet
    )
end

"""
Normalize weights to sum to 1.0
"""
function normalize_weights!(weights::FitnessWeights)
    total = weights.snr_weight + 
            weights.lock_quality_weight + 
            weights.ringing_penalty_weight + 
            weights.frequency_selectivity_weight
    
    if total <= 0.0f0
        # Reset to defaults if invalid
        weights.snr_weight = 0.35f0
        weights.lock_quality_weight = 0.35f0
        weights.ringing_penalty_weight = 0.20f0
        weights.frequency_selectivity_weight = 0.10f0
        total = 1.0f0
    end
    
    # Normalize
    weights.snr_weight /= total
    weights.lock_quality_weight /= total
    weights.ringing_penalty_weight /= total
    weights.frequency_selectivity_weight /= total
    
    weights.normalized = true
end

"""
Load fitness weights from TOML configuration
"""
function load_fitness_weights(config_path::String)::FitnessWeights
    if !isfile(config_path)
        @warn "Config file not found: $config_path, using defaults"
        weights = create_default_weights()
        normalize_weights!(weights)
        return weights
    end
    
    try
        config = TOML.parsefile(config_path)
        
        # Look for fitness weights section
        if haskey(config, "fitness") && haskey(config["fitness"], "weights")
            weight_config = config["fitness"]["weights"]
            
            weights = FitnessWeights(
                Float32(get(weight_config, "snr", 0.35)),
                Float32(get(weight_config, "lock_quality", 0.35)),
                Float32(get(weight_config, "ringing_penalty", 0.20)),
                Float32(get(weight_config, "frequency_selectivity", 0.10)),
                false
            )
            
            normalize_weights!(weights)
            return weights
        else
            @info "No fitness.weights section found, using defaults"
            weights = create_default_weights()
            normalize_weights!(weights)
            return weights
        end
    catch e
        @error "Error loading fitness weights: $e"
        weights = create_default_weights()
        normalize_weights!(weights)
        return weights
    end
end

"""
Create fitness configuration with weights
"""
function create_fitness_config(
    weights::FitnessWeights = create_default_weights();
    use_pll::Bool = true,
    target_period::Float32 = 26.0f0,
    signal_length::Int32 = Int32(1000),
    warmup_samples::Int32 = Int32(100),
    enable_caching::Bool = true,
    parallel_evaluation::Bool = false,
    max_cache_size::Int32 = Int32(1000)
)::FitnessConfig
    
    # Ensure weights are normalized
    if !weights.normalized
        normalize_weights!(weights)
    end
    
    return FitnessConfig(
        weights,
        use_pll,
        target_period,
        signal_length,
        warmup_samples,
        enable_caching,
        parallel_evaluation,
        max_cache_size
    )
end

# =============================================================================
# FITNESS CACHING
# =============================================================================

"""
Cache for fitness evaluations to avoid redundant calculations
"""
mutable struct FitnessCache
    cache::Dict{UInt64, FitnessResult}
    max_size::Int32
    hits::Int64
    misses::Int64
    
    function FitnessCache(max_size::Int32 = Int32(1000))
        new(Dict{UInt64, FitnessResult}(), max_size, 0, 0)
    end
end

"""
Get cache key for chromosome (works with Vector{Float32})
"""
function get_cache_key(chromosome::Vector{Float32})::UInt64
    return hash(chromosome)
end

"""
Check cache for fitness result
"""
function check_cache(cache::FitnessCache, chromosome::Vector{Float32})::Union{FitnessResult, Nothing}
    key = get_cache_key(chromosome)
    
    if haskey(cache.cache, key)
        cache.hits += 1
        result = cache.cache[key]
        # Mark as cache hit
        return FitnessResult(
            result.total_fitness,
            result.snr_contribution,
            result.lock_quality_contribution,
            result.ringing_penalty_contribution,
            result.frequency_selectivity_contribution,
            result.metrics,
            result.evaluation_time_ms,
            true  # cache_hit
        )
    else
        cache.misses += 1
        return nothing
    end
end

"""
Store result in cache
"""
function store_in_cache!(cache::FitnessCache, chromosome::Vector{Float32}, result::FitnessResult)
    # Check cache size
    if length(cache.cache) >= cache.max_size
        # Remove oldest entry (simple FIFO for now)
        # In production, use LRU
        first_key = first(keys(cache.cache))
        delete!(cache.cache, first_key)
    end
    
    key = get_cache_key(chromosome)
    cache.cache[key] = result
end

# =============================================================================
# FITNESS EVALUATION
# =============================================================================

"""
Generate or retrieve test signal for evaluation
"""
function get_test_signal(
    fibonacci_number::Int32,
    signal_length::Int32;
    signal_type::Symbol = :pure_sine
)::Vector{ComplexF32}
    
    # Calculate target period with doubling
    target_period = fibonacci_number == 1 ? 2.01f0 : Float32(2 * fibonacci_number)
    
    # Generate simple test signal
    t = 0:signal_length-1
    frequency = 2π / target_period
    
    if signal_type == :pure_sine
        # Pure sine wave with unit amplitude
        real_signal = sin.(frequency .* t)
        # Add small imaginary component (volume proxy)
        signal = ComplexF32.(real_signal, ones(signal_length) * 0.1f0)
    elseif signal_type == :noisy_sine
        # Sine with noise
        real_signal = sin.(frequency .* t) .+ 0.1f0 .* randn(signal_length)
        signal = ComplexF32.(real_signal, ones(signal_length) * 0.1f0)
    else
        # Chirp signal for frequency response testing
        freq_sweep = frequency .* (1 .+ 0.2 .* sin.(0.01 .* t))
        real_signal = sin.(freq_sweep .* t)
        signal = ComplexF32.(real_signal, ones(signal_length) * 0.1f0)
    end
    
    return signal
end

"""
Evaluate fitness for a single chromosome
Accepts Vector{Float32} directly to work with GA system
"""
function evaluate_fitness(
    chromosome::Vector{Float32},
    fibonacci_number::Int32,
    config::FitnessConfig;
    test_signal::Union{Vector{ComplexF32}, Nothing} = nothing,
    cache::Union{FitnessCache, Nothing} = nothing
)::FitnessResult
    
    # Use nanosecond precision for timing
    start_time = time_ns()
    
    # Validate chromosome
    @assert length(chromosome) == 13 "Chromosome must have 13 genes"
    
    # Check cache if enabled
    if cache !== nothing && config.enable_caching
        cached_result = check_cache(cache, chromosome)
        if cached_result !== nothing
            return cached_result
        end
    end
    
    # Generate test signal if not provided
    if test_signal === nothing
        test_signal = get_test_signal(
            fibonacci_number,
            config.signal_length,
            signal_type = :pure_sine
        )
    end
    
    # Create filter from chromosome
    filter = FilterIntegration.create_filter_from_chromosome(
        chromosome,
        fibonacci_number,
        Int32(1),  # filter_index as Int32
        config.use_pll
    )
    
    # Process signal through filter
    output_signal = FilterIntegration.evaluate_filter_with_signal(
        filter,
        test_signal
    )
    
    # Skip warmup samples for metrics
    if config.warmup_samples > 0 && length(output_signal) > config.warmup_samples
        output_for_metrics = output_signal[config.warmup_samples+1:end]
        input_for_metrics = test_signal[config.warmup_samples+1:end]
    else
        output_for_metrics = output_signal
        input_for_metrics = test_signal
    end
    
    # Calculate all metrics
    metrics = SignalMetrics.calculate_all_metrics(
        output_for_metrics,
        input_for_metrics,
        filter,
        target_period = config.target_period
    )
    
    # Apply weights to get final fitness
    weights = config.weights
    
    # Ensure weights are normalized
    if !weights.normalized
        normalize_weights!(weights)
    end
    
    # Calculate weighted contributions
    snr_contribution = weights.snr_weight * metrics.snr.normalized_value
    lock_contribution = weights.lock_quality_weight * metrics.lock_quality.normalized_value
    
    # Ringing penalty reduces fitness (but metric is already inverted)
    ringing_contribution = weights.ringing_penalty_weight * metrics.ringing_penalty.normalized_value
    
    selectivity_contribution = weights.frequency_selectivity_weight * metrics.frequency_selectivity.normalized_value
    
    # Total fitness
    total_fitness = snr_contribution + lock_contribution + 
                   ringing_contribution + selectivity_contribution
    
    # Clamp to [0, 1]
    total_fitness = clamp(total_fitness, 0.0f0, 1.0f0)
    
    # Calculate evaluation time in milliseconds with nanosecond precision
    end_time = time_ns()
    elapsed_ns = end_time - start_time
    evaluation_time_ms = Float32(elapsed_ns / 1_000_000.0)
    
    # Ensure minimum non-zero time (even very fast operations should register)
    if evaluation_time_ms < 0.001f0
        evaluation_time_ms = 0.001f0  # Minimum 1 microsecond
    end
    
    result = FitnessResult(
        total_fitness,
        snr_contribution,
        lock_contribution,
        ringing_contribution,
        selectivity_contribution,
        metrics,
        evaluation_time_ms,
        false  # not from cache
    )
    
    # Store in cache if enabled
    if cache !== nothing && config.enable_caching
        store_in_cache!(cache, chromosome, result)
    end
    
    return result
end

# =============================================================================
# BATCH EVALUATION
# =============================================================================

"""
Evaluate fitness for entire population
"""
function evaluate_population_fitness(
    population::Matrix{Float32},  # population_size × 13
    fibonacci_number::Int32,
    config::FitnessConfig;
    test_signal::Union{Vector{ComplexF32}, Nothing} = nothing,
    cache::Union{FitnessCache, Nothing} = nothing
)::Vector{Float32}
    
    pop_size = size(population, 1)
    fitness_values = Vector{Float32}(undef, pop_size)
    
    # Generate test signal once for all evaluations
    if test_signal === nothing
        test_signal = get_test_signal(
            fibonacci_number,
            config.signal_length,
            signal_type = :pure_sine
        )
    end
    
    # Evaluate each individual
    # TODO: Add parallel evaluation if config.parallel_evaluation is true
    for i in 1:pop_size
        chromosome = vec(population[i, :])  # Get row as vector
        result = evaluate_fitness(
            chromosome,
            fibonacci_number,
            config,
            test_signal = test_signal,
            cache = cache
        )
        fitness_values[i] = result.total_fitness
    end
    
    return fitness_values
end

# =============================================================================
# STUB FITNESS REPLACEMENT
# =============================================================================

"""
Drop-in replacement for stub fitness function
Returns fitness in range [0, 1] where higher is better
Accepts Vector{Float32} directly
"""
function evaluate_filter_fitness(
    chromosome::Vector{Float32},
    fibonacci_number::Int32 = 13;
    weights::Union{FitnessWeights, Nothing} = nothing
)::Float32
    
    # Validate
    @assert length(chromosome) == 13 "Chromosome must have 13 genes"
    
    # Use provided weights or defaults
    if weights === nothing
        weights = create_default_weights()
        normalize_weights!(weights)
    end
    
    # Create minimal config
    config = create_fitness_config(
        weights,
        use_pll = true,
        signal_length = Int32(500),  # Shorter for speed
        warmup_samples = Int32(50),
        enable_caching = false  # No caching for single eval
    )
    
    # Evaluate
    result = evaluate_fitness(
        chromosome,
        fibonacci_number,
        config
    )
    
    return result.total_fitness
end

# =============================================================================
# UTILITIES
# =============================================================================

"""
Print fitness breakdown for debugging
"""
function print_fitness_breakdown(result::FitnessResult)
    println("="^50)
    println("Fitness Breakdown:")
    println("-"^50)
    println("Total Fitness:        $(round(result.total_fitness, digits=4))")
    println("SNR Contribution:     $(round(result.snr_contribution, digits=4))")
    println("Lock Quality:         $(round(result.lock_quality_contribution, digits=4))")
    println("Ringing Penalty:      $(round(result.ringing_penalty_contribution, digits=4))")
    println("Frequency Select:     $(round(result.frequency_selectivity_contribution, digits=4))")
    println("-"^50)
    println("Raw Metrics:")
    println("  SNR (dB):           $(round(result.metrics.snr.raw_value, digits=2))")
    println("  Lock Quality:       $(round(result.metrics.lock_quality.raw_value, digits=3))")
    println("  Ringing:            $(round(result.metrics.ringing_penalty.raw_value, digits=3))")
    println("  Selectivity:        $(round(result.metrics.frequency_selectivity.raw_value, digits=2))")
    println("-"^50)
    println("Evaluation Time:      $(round(result.evaluation_time_ms, digits=2)) ms")
    println("Cache Hit:            $(result.cache_hit)")
    println("="^50)
end

"""
Get fitness statistics for a population
"""
function get_population_stats(fitness_values::Vector{Float32})::Dict{String, Float32}
    return Dict{String, Float32}(
        "mean" => mean(fitness_values),
        "std" => std(fitness_values),
        "min" => minimum(fitness_values),
        "max" => maximum(fitness_values),
        "median" => median(fitness_values)
    )
end

end # module FitnessEvaluation