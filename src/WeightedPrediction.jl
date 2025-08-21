module WeightedPrediction

using Statistics
using LinearAlgebra
using Random
using CircularArrays
using Dates
using ..GATypes
using ..PredictionMetrics

export # All public functions and types
       WeightSet, PredictionSystem, StreamingPredictor, HorizonPrediction,
       initialize_weights_rms, create_weight_population,
       evaluate_weight_fitness, evolve_weights,
       predict_price_change_extrapolated,
       create_prediction_system, update_prediction!, get_prediction_at_horizon,
       create_streaming_predictor, process_tick!, get_current_predictions

# --- Struct to replace Dictionary ---
struct HorizonPrediction
    horizon::Int32
    prediction::Float32
end

# =============================================================================
# WEIGHT & PREDICTION STRUCTURES (Hardware-Agnostic)
# =============================================================================

struct WeightSet{V<:AbstractVector{Float32}}
    horizon::Int32
    weights::V
    fitness::Float32
    mse::Float32
    mae::Float32
    directional_accuracy::Float32
end

mutable struct PredictionSystem{V<:AbstractVector, M<:AbstractMatrix, CV<:CircularVector}
    n_filters::Int32
    weights::V
    filter_periods::V
    filter_frequencies::V # Pre-calculated for performance
    horizon_range::Tuple{Int32, Int32}
    input_buffer::CV
    current_tick::Int64
end

mutable struct StreamingPredictor{P<:PredictionSystem}
    system::P
    warmup_period::Int32
    is_warmed_up::Bool
    predictions::Vector{HorizonPrediction} # Replaces Dict
end

# =============================================================================
# INITIALIZATION
# =============================================================================

function initialize_weights_rms(filter_outputs::M; target_rms::Union{Float32, Nothing} = nothing) where {M<:AbstractMatrix{ComplexF32}}
    rms_values = sqrt.(mean(abs2.(filter_outputs), dims=1))
    rms_values = max.(rms_values, 1.f-10)
    
    target = target_rms === nothing ? mean(rms_values) : target_rms
    
    weights = target ./ rms_values
    weights = clamp.(weights, 0.0f0, 1.0f0)
    
    return weights ./ sum(weights)
end

function create_weight_population(n_filters::Int, pop_size::Int; ArrayType::Type=Array{Float32})
    population = ArrayType(undef, pop_size, n_filters)
    # Generate on CPU and copy to device
    cpu_pop = rand(Float32, pop_size, n_filters)
    for i in 1:pop_size
        cpu_pop[i, :] ./= sum(@view cpu_pop[i, :])
    end
    copyto!(population, cpu_pop)
    return population
end

# =============================================================================
# CORE PREDICTION (Hardware-Agnostic)
# =============================================================================

function predict_price_change_extrapolated(
    current_outputs::V,
    filter_frequencies::V,
    weights::V,
    n_ticks::Int32
)::Float32 where {V<:AbstractVector}
    
    magnitudes = abs.(current_outputs)
    phases = angle.(current_outputs)
    
    phase_advances = filter_frequencies .* Float32(n_ticks)
    projected_phases = phases .+ phase_advances
    
    # Reconstruct complex vector from projected phases and original magnitudes
    # This is a key vectorized operation
    projected_outputs = magnitudes .* exp.(im .* projected_phases)
    
    # Weighted sum and return real part
    return real(sum(weights .* projected_outputs))
end

# =============================================================================
# WEIGHT OPTIMIZATION (Hardware-Agnostic)
# =============================================================================

function evaluate_weight_fitness(
    weights::V,
    filter_outputs::M,
    actual_future::V,
    horizon::Int32,
    filter_frequencies::V
)::Tuple{Float32, Float32, Float32, Float32} where {V<:AbstractVector, M<:AbstractMatrix}
    
    n_samples = size(filter_outputs, 1)
    n_predictions = n_samples - horizon
    
    if n_predictions <= 0
        return (0.0f0, Inf32, Inf32, 0.0f0)
    end
    
    predictions = similar(weights, n_predictions)
    
    for t in 1:n_predictions
        predictions[t] = predict_price_change_extrapolated(
            @view(filter_outputs[t, :]),
            filter_frequencies,
            weights,
            horizon
        )
    end
    
    actuals_view = @view actual_future[horizon+1:end]
    
    # Metrics are calculated on the CPU
    mse, mae, dir_acc = calculate_mse(predictions, actuals_view),
                        calculate_mae(predictions, actuals_view),
                        calculate_directional_accuracy(predictions, actuals_view)

    fitness = 1.0f0 / (1.0f0 + mse) * (0.7f0 + 0.3f0 * dir_acc)
    
    return (fitness, mse, mae, dir_acc)
end

# =============================================================================
# SYSTEM MANAGEMENT
# =============================================================================

function create_prediction_system(
    n_filters::Int32,
    initial_weights::V,
    filter_periods::V,
    horizon_range::Tuple{Int32, Int32}
) where {V<:AbstractVector{Float32}}
    
    max_horizon = horizon_range[2]
    ArrayType = typeof(initial_weights).name.wrapper
    
    return PredictionSystem(
        n_filters,
        initial_weights,
        filter_periods,
        Float32(2π) ./ filter_periods, # Pre-calculate frequencies
        horizon_range,
        CircularVector(ArrayType{ComplexF32}(undef, max_horizon)), # Input buffer
        Int64(0) # current_tick
    )
end

function create_streaming_predictor(
    system::PredictionSystem;
    warmup_period::Int32 = Int32(100)
)::StreamingPredictor
    return StreamingPredictor(system, warmup_period, false, HorizonPrediction[])
end

function process_tick!(
    predictor::StreamingPredictor,
    filter_outputs::V,
    input_signal::ComplexF32
) where {V<:AbstractVector{ComplexF32}}
    
    sys = predictor.system
    sys.current_tick += 1
    push!(sys.input_buffer, input_signal)

    if !predictor.is_warmed_up && sys.current_tick >= predictor.warmup_period
        predictor.is_warmed_up = true
    end
    
    if predictor.is_warmed_up
        empty!(predictor.predictions)
        key_horizons = Int32[100, 250, 500, 1000] # Example horizons
        
        for h in key_horizons
            if sys.horizon_range[1] <= h <= sys.horizon_range[2]
                pred = predict_price_change_extrapolated(
                    filter_outputs,
                    sys.filter_frequencies,
                    sys.weights,
                    h
                )
                push!(predictor.predictions, HorizonPrediction(h, pred))
            end
        end
    end
end

function get_current_predictions(predictor::StreamingPredictor)::Vector{HorizonPrediction}
    return predictor.predictions
end

end # module WeightedPrediction