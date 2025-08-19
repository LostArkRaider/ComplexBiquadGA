# src/PricePrediction.jl - Price Change Prediction via Phase-Extrapolated Vector Sum
# FIXED: Removed CircularArrays dependency, using simple circular buffer implementation

"""
Price Prediction Module - Chunk 4 (REVISED)

Implements price change prediction using phase-based extrapolation of filter outputs.
Each filter's rotating phasor is projected forward n ticks based on its design frequency,
then weighted and summed for prediction.

Key Features:
- Phase-based extrapolation using filter design frequencies
- 4-phase rotation tracking via Q component
- Constant magnitude projection (no decay)
- Scalar weight multiplication preserving phase
- I-component comparison for normalized price changes
"""

module PricePrediction

using Statistics
using LinearAlgebra

export PredictionSystem, PredictionBuffer, 
       predict_price_change_extrapolated, predict_batch_extrapolated,
       create_prediction_system, update_prediction!,
       get_prediction_at_horizon, evaluate_predictions,
       PredictionResult, StreamingPredictor,
       calculate_filter_frequencies, project_filter_forward

# =============================================================================
# SIMPLE CIRCULAR BUFFER IMPLEMENTATION
# =============================================================================

"""
Simple circular buffer implementation to replace CircularArrays dependency
"""
mutable struct CircularBuffer{T}
    buffer::Vector{T}
    capacity::Int
    size::Int
    head::Int  # Next write position
    
    function CircularBuffer{T}(capacity::Int) where T
        new{T}(Vector{T}(undef, capacity), capacity, 0, 1)
    end
end

function Base.push!(cb::CircularBuffer{T}, item::T) where T
    cb.buffer[cb.head] = item
    cb.head = mod1(cb.head + 1, cb.capacity)
    cb.size = min(cb.size + 1, cb.capacity)
end

function Base.getindex(cb::CircularBuffer{T}, i::Int) where T
    if i < 1 || i > cb.size
        throw(BoundsError(cb, i))
    end
    # Calculate actual index in buffer
    if cb.size < cb.capacity
        return cb.buffer[i]
    else
        # Full buffer, need to account for circular wrap
        actual_idx = mod1(cb.head + i - cb.size - 1, cb.capacity)
        return cb.buffer[actual_idx]
    end
end

Base.length(cb::CircularBuffer) = cb.size
Base.isempty(cb::CircularBuffer) = cb.size == 0
Base.lastindex(cb::CircularBuffer) = cb.size

function Base.last(cb::CircularBuffer{T}) where T
    if cb.size == 0
        throw(ArgumentError("CircularBuffer is empty"))
    end
    # Last written item is at head-1
    last_idx = mod1(cb.head - 1, cb.capacity)
    return cb.buffer[last_idx]
end

# =============================================================================
# CONSTANTS
# =============================================================================

# 4-phase rotation: each tick advances phase by π/2
const PHASE_PER_TICK = π / 2  # 90 degrees per tick

# =============================================================================
# PREDICTION STRUCTURES
# =============================================================================

"""
Result of a single prediction with extrapolation details
"""
struct PredictionResult
    horizon::Int32                   # Prediction horizon in ticks
    predicted_value::Float32         # Predicted I-component (price change)
    actual_value::Float32           # Actual I-component (if available)
    confidence::Float32             # Confidence score [0,1]
    timestamp::Int64                # When prediction was made
    filter_contributions::Vector{Float32}  # Contribution from each filter
    projected_phases::Vector{Float32}      # Projected phase for each filter
end

"""
Filter state for phase tracking
"""
mutable struct FilterPhaseState
    current_output::ComplexF32      # Current filter output
    design_frequency::Float32       # Design frequency (2π/period)
    period::Float32                # Filter period (after doubling)
    tick_count::Int64              # Current tick position
    phase_offset::Float32          # Initial phase offset
end

"""
Circular buffer for efficient streaming predictions
"""
mutable struct PredictionBuffer
    capacity::Int32                      # Buffer size (max horizon)
    current_position::Int32              # Current write position
    n_filters::Int32                     # Number of filters
    
    # Separate buffers per filter for efficiency
    filter_buffers::Vector{CircularBuffer{ComplexF32}}
    
    # Phase states for each filter
    filter_states::Vector{FilterPhaseState}
end

function PredictionBuffer(capacity::Int32, n_filters::Int32, filter_periods::Vector{Float32})
    filter_buffers = [CircularBuffer{ComplexF32}(capacity) for _ in 1:n_filters]
    
    # Initialize phase states with design frequencies
    filter_states = Vector{FilterPhaseState}(undef, n_filters)
    for i in 1:n_filters
        period = filter_periods[i]
        design_freq = 2π / period  # Radians per bar
        filter_states[i] = FilterPhaseState(
            ComplexF32(0, 0),      # current_output
            design_freq,           # design_frequency
            period,               # period
            0,                    # tick_count
            0.0f0                 # phase_offset
        )
    end
    
    return PredictionBuffer(
        capacity,
        Int32(0),
        n_filters,
        filter_buffers,
        filter_states
    )
end

"""
Main prediction system managing weights and phase extrapolation
"""
mutable struct PredictionSystem
    n_filters::Int32                    # Number of filters in bank
    weights::Vector{Float32}            # Current scalar weights
    filter_periods::Vector{Float32}     # Filter periods (after doubling)
    horizon_range::Tuple{Int32, Int32}  # Min/max prediction horizon
    
    # Buffers for streaming
    prediction_buffer::PredictionBuffer  # Filter output and phase tracking
    input_buffer::CircularBuffer{ComplexF32}  # Input signal history
    
    # Performance tracking
    predictions_made::Int64
    total_error::Float64
    directional_accuracy::Float64
    
    # Weight adaptation (optional)
    adaptive_weights::Bool
    learning_rate::Float32
    
    # Current tick position
    current_tick::Int64
end

"""
Streaming predictor for real-time operation
"""
mutable struct StreamingPredictor
    system::PredictionSystem
    warmup_period::Int32
    is_warmed_up::Bool
    
    # Prediction cache for multiple horizons
    horizon_predictions::Dict{Int32, Float32}
end

# =============================================================================
# FREQUENCY AND PHASE CALCULATIONS
# =============================================================================

"""
Calculate design frequencies for filters based on periods
"""
function calculate_filter_frequencies(filter_periods::Vector{Float32})::Vector{Float32}
    # Design frequency = 2π / period (radians per bar)
    # Note: periods are already doubled (e.g., Fib 13 → period 26)
    return 2π ./ filter_periods
end

"""
Extract instantaneous phase from complex output considering 4-phase rotation
The Q component encodes the phase position in the rotation cycle
"""
function extract_instantaneous_phase(z::ComplexF32, tick_count::Int64)::Float32
    # Base phase from complex argument
    base_phase = angle(z)
    
    # 4-phase rotation position (0, π/2, π, 3π/2)
    rotation_phase = PHASE_PER_TICK * (tick_count - 1)
    
    # Combined phase
    return base_phase + rotation_phase
end

"""
Project filter output forward n ticks using phase extrapolation
Magnitude remains constant (no decay assumption)
"""
function project_filter_forward(
    current_output::ComplexF32,
    design_frequency::Float32,
    n_ticks::Int32,
    current_tick::Int64
)::ComplexF32
    
    # Extract magnitude (remains constant)
    magnitude = abs(current_output)
    
    # Current phase considering 4-phase rotation
    current_phase = angle(current_output)
    
    # Phase advancement due to filter's natural frequency
    # Design frequency is in radians per bar
    # Need to convert to radians per tick
    # Assuming ticks come at a rate that maintains the design frequency
    phase_advance = design_frequency * n_ticks
    
    # 4-phase rotation advancement (π/2 per tick)
    rotation_advance = PHASE_PER_TICK * n_ticks
    
    # Total projected phase
    projected_phase = current_phase + phase_advance + rotation_advance
    
    # Reconstruct complex prediction
    return magnitude * exp(im * projected_phase)
end

"""
Alternative: Project using only design frequency (simpler approach)
"""
function project_filter_simple(
    current_output::ComplexF32,
    design_frequency::Float32,
    n_ticks::Int32
)::ComplexF32
    
    magnitude = abs(current_output)
    current_phase = angle(current_output)
    
    # Project phase forward using design frequency
    # Frequency is in radians per bar, convert based on tick rate
    projected_phase = current_phase + design_frequency * n_ticks
    
    return magnitude * exp(im * projected_phase)
end

# =============================================================================
# CORE PREDICTION FUNCTIONS WITH EXTRAPOLATION
# =============================================================================

"""
Predict price change using phase-extrapolated weighted vector sum
Returns I-component (real part) of the prediction
"""
function predict_price_change_extrapolated(
    filter_outputs::Vector{ComplexF32},
    filter_frequencies::Vector{Float32},
    weights::Vector{Float32},
    n_ticks::Int32,
    current_tick::Int64 = Int64(0)
)::Float32
    
    @assert length(filter_outputs) == length(weights) == length(filter_frequencies) "Dimension mismatch"
    
    # Project each filter forward
    projected_outputs = Vector{ComplexF32}(undef, length(filter_outputs))
    
    for i in 1:length(filter_outputs)
        # Use simpler projection (can switch to full version if needed)
        projected_outputs[i] = project_filter_simple(
            filter_outputs[i],
            filter_frequencies[i],
            n_ticks
        )
    end
    
    # Weighted vector sum of projected outputs
    prediction_vector = ComplexF32(0, 0)
    for i in 1:length(weights)
        prediction_vector += weights[i] * projected_outputs[i]
    end
    
    # Extract I-component (normalized price change)
    return real(prediction_vector)
end

"""
Predict with detailed contribution tracking
"""
function predict_price_change_detailed(
    filter_outputs::Vector{ComplexF32},
    filter_frequencies::Vector{Float32},
    weights::Vector{Float32},
    n_ticks::Int32
)::Tuple{Float32, Vector{Float32}, Vector{Float32}}
    
    n_filters = length(filter_outputs)
    contributions = Vector{Float32}(undef, n_filters)
    projected_phases = Vector{Float32}(undef, n_filters)
    
    # Project and track each filter
    prediction = Float32(0)
    
    for i in 1:n_filters
        # Project forward
        projected = project_filter_simple(
            filter_outputs[i],
            filter_frequencies[i],
            n_ticks
        )
        
        # Track projected phase
        projected_phases[i] = angle(projected)
        
        # Apply weight
        weighted_output = weights[i] * projected
        
        # Extract I-component contribution
        contribution = real(weighted_output)
        contributions[i] = contribution
        prediction += contribution
    end
    
    return (prediction, contributions, projected_phases)
end

"""
Batch prediction with phase extrapolation
"""
function predict_batch_extrapolated(
    filter_outputs::Matrix{ComplexF32},
    filter_periods::Vector{Float32},
    weights::Vector{Float32},
    horizon::Int32
)::Vector{Float32}
    
    n_samples, n_filters = size(filter_outputs)
    n_predictions = n_samples - horizon
    
    if n_predictions <= 0
        return Float32[]
    end
    
    # Calculate design frequencies
    filter_frequencies = calculate_filter_frequencies(filter_periods)
    
    predictions = Vector{Float32}(undef, n_predictions)
    
    for t in 1:n_predictions
        # Get current filter outputs
        current_outputs = filter_outputs[t, :]
        
        # Predict future value with extrapolation
        predictions[t] = predict_price_change_extrapolated(
            current_outputs,
            filter_frequencies,
            weights,
            Int32(horizon),
            Int64(t)
        )
    end
    
    return predictions
end

# =============================================================================
# PREDICTION SYSTEM MANAGEMENT
# =============================================================================

"""
Create prediction system with phase extrapolation support
"""
function create_prediction_system(
    n_filters::Int32,
    initial_weights::Vector{Float32},
    filter_periods::Vector{Float32},
    horizon_range::Tuple{Int32, Int32};
    adaptive::Bool = false,
    learning_rate::Float32 = 0.01f0
)::PredictionSystem
    
    @assert length(initial_weights) == n_filters "Weight dimension mismatch"
    @assert length(filter_periods) == n_filters "Period dimension mismatch"
    
    # Create buffers with phase tracking
    max_horizon = horizon_range[2]
    prediction_buffer = PredictionBuffer(max_horizon, n_filters, filter_periods)
    input_buffer = CircularBuffer{ComplexF32}(max_horizon)
    
    return PredictionSystem(
        n_filters,
        copy(initial_weights),
        filter_periods,
        horizon_range,
        prediction_buffer,
        input_buffer,
        0,     # predictions_made
        0.0,   # total_error
        0.0,   # directional_accuracy
        adaptive,
        learning_rate,
        0      # current_tick
    )
end

"""
Update prediction system with new filter outputs
"""
function update_prediction!(
    system::PredictionSystem,
    filter_outputs::Vector{ComplexF32},
    input_signal::ComplexF32
)
    
    system.current_tick += 1
    
    # Update filter states
    for i in 1:system.n_filters
        state = system.prediction_buffer.filter_states[i]
        state.current_output = filter_outputs[i]
        state.tick_count = system.current_tick
        
        # Store in buffer
        push!(system.prediction_buffer.filter_buffers[i], filter_outputs[i])
    end
    
    # Store input
    push!(system.input_buffer, input_signal)
    
    system.prediction_buffer.current_position += 1
    
    # Adaptive weight update (if enabled)
    if system.adaptive_weights && system.prediction_buffer.current_position > system.horizon_range[1]
        adapt_weights!(system)
    end
end

"""
Get prediction for specific horizon using phase extrapolation
"""
function get_prediction_at_horizon(system::PredictionSystem, horizon::Int32)::Float32
    
    # Validate horizon
    if horizon < system.horizon_range[1] || horizon > system.horizon_range[2]
        error("Horizon $horizon outside range $(system.horizon_range)")
    end
    
    # Check if we have enough history
    if system.prediction_buffer.current_position < 1
        return Float32(0)
    end
    
    # Get current filter outputs and frequencies
    current_outputs = Vector{ComplexF32}(undef, system.n_filters)
    filter_frequencies = calculate_filter_frequencies(system.filter_periods)
    
    for i in 1:system.n_filters
        state = system.prediction_buffer.filter_states[i]
        current_outputs[i] = state.current_output
    end
    
    # Make prediction with extrapolation
    return predict_price_change_extrapolated(
        current_outputs,
        filter_frequencies,
        system.weights,
        horizon,
        system.current_tick
    )
end

"""
Evaluate predictions against actual values
"""
function evaluate_predictions(
    system::PredictionSystem,
    horizon::Int32
)::Tuple{Float32, Float32, Float32}
    
    # Check if we have enough history for evaluation
    buffer_size = length(system.input_buffer)
    if buffer_size < horizon + 1
        return (Float32(0), Float32(0), Float32(0))
    end
    
    n_evaluations = buffer_size - horizon
    errors = Vector{Float32}(undef, n_evaluations)
    correct_directions = 0
    
    # Get filter frequencies
    filter_frequencies = calculate_filter_frequencies(system.filter_periods)
    
    for i in 1:n_evaluations
        # Get historical filter outputs
        historical_outputs = Vector{ComplexF32}(undef, system.n_filters)
        for j in 1:system.n_filters
            historical_outputs[j] = system.prediction_buffer.filter_buffers[j][i]
        end
        
        # Make prediction with extrapolation
        predicted = predict_price_change_extrapolated(
            historical_outputs,
            filter_frequencies,
            system.weights,
            horizon,
            Int64(i)
        )
        
        # Get actual value
        actual_idx = i + horizon
        actual = real(system.input_buffer[actual_idx])
        
        # Calculate error
        errors[i] = predicted - actual
        
        # Check direction
        if sign(predicted) == sign(actual)
            correct_directions += 1
        end
    end
    
    # Calculate metrics
    mse = mean(errors .^ 2)
    mae = mean(abs.(errors))
    dir_accuracy = Float32(correct_directions / n_evaluations)
    
    # Update system statistics
    system.total_error += sum(abs.(errors))
    system.predictions_made += n_evaluations
    system.directional_accuracy = (system.directional_accuracy * (system.predictions_made - n_evaluations) +
                                   correct_directions) / system.predictions_made
    
    return (mse, mae, dir_accuracy)
end

# =============================================================================
# ADAPTIVE WEIGHT ADJUSTMENT
# =============================================================================

"""
Adapt weights based on recent prediction errors (online learning)
"""
function adapt_weights!(system::PredictionSystem)
    
    # Use shortest horizon for fastest feedback
    horizon = system.horizon_range[1]
    
    # Check if we have enough data
    if length(system.input_buffer) < horizon + 1
        return
    end
    
    # Get recent prediction point
    current_idx = length(system.input_buffer) - horizon
    
    # Get filter outputs from that time
    historical_outputs = Vector{ComplexF32}(undef, system.n_filters)
    for i in 1:system.n_filters
        historical_outputs[i] = system.prediction_buffer.filter_buffers[i][current_idx]
    end
    
    # Get filter frequencies
    filter_frequencies = calculate_filter_frequencies(system.filter_periods)
    
    # Prediction from that time with extrapolation
    predicted = predict_price_change_extrapolated(
        historical_outputs,
        filter_frequencies,
        system.weights,
        horizon,
        Int64(current_idx)
    )
    
    # Actual value (now available)
    actual = real(last(system.input_buffer))
    
    # Prediction error
    error = predicted - actual
    
    # Project filters forward to get derivatives
    projected_outputs = Vector{ComplexF32}(undef, system.n_filters)
    for i in 1:system.n_filters
        projected_outputs[i] = project_filter_simple(
            historical_outputs[i],
            filter_frequencies[i],
            horizon
        )
    end
    
    # Gradient descent update
    for i in 1:system.n_filters
        # Gradient: derivative of MSE w.r.t. weight_i
        # ∂MSE/∂w_i = 2 * error * real(projected_output_i)
        gradient = 2.0f0 * error * real(projected_outputs[i])
        
        # Update weight (gradient descent)
        system.weights[i] -= system.learning_rate * gradient
        
        # Clamp to valid range
        system.weights[i] = clamp(system.weights[i], 0.0f0, 1.0f0)
    end
    
    # Renormalize weights
    weight_sum = sum(system.weights)
    if weight_sum > 0
        system.weights ./= weight_sum
    end
end

# =============================================================================
# STREAMING PREDICTOR
# =============================================================================

"""
Create streaming predictor for real-time operation
"""
function create_streaming_predictor(
    n_filters::Int32,
    initial_weights::Vector{Float32},
    filter_periods::Vector{Float32},
    horizon_range::Tuple{Int32, Int32};
    warmup_period::Int32 = Int32(100)
)::StreamingPredictor
    
    system = create_prediction_system(
        n_filters, 
        initial_weights, 
        filter_periods,
        horizon_range
    )
    
    return StreamingPredictor(
        system,
        warmup_period,
        false,               # is_warmed_up
        Dict{Int32, Float32}()  # horizon_predictions
    )
end

"""
Process new tick in streaming mode
"""
function process_tick!(
    predictor::StreamingPredictor,
    filter_outputs::Vector{ComplexF32},
    input_signal::ComplexF32
)
    
    # Update system
    update_prediction!(predictor.system, filter_outputs, input_signal)
    
    # Check warmup
    if !predictor.is_warmed_up && predictor.system.current_tick >= predictor.warmup_period
        predictor.is_warmed_up = true
        println("Predictor warmed up at tick $(predictor.system.current_tick)")
    end
    
    # Generate predictions for multiple horizons if warmed up
    if predictor.is_warmed_up
        empty!(predictor.horizon_predictions)
        
        # Generate predictions for key horizons
        key_horizons = [100, 250, 500, 1000, 1500, 2000]
        
        for horizon in key_horizons
            if horizon >= predictor.system.horizon_range[1] && 
               horizon <= predictor.system.horizon_range[2]
                
                prediction = get_prediction_at_horizon(predictor.system, Int32(horizon))
                predictor.horizon_predictions[Int32(horizon)] = prediction
            end
        end
    end
end

"""
Get current predictions for all horizons
"""
function get_current_predictions(predictor::StreamingPredictor)::Dict{Int32, Float32}
    if !predictor.is_warmed_up
        return Dict{Int32, Float32}()
    end
    
    return copy(predictor.horizon_predictions)
end

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

"""
Calculate prediction confidence based on filter phase coherence
"""
function calculate_prediction_confidence(
    filter_outputs::Vector{ComplexF32},
    projected_outputs::Vector{ComplexF32},
    weights::Vector{Float32}
)::Float32
    
    # Check phase coherence of projected outputs
    phases = angle.(projected_outputs)
    
    # Calculate circular variance
    mean_vector = sum(exp.(im .* phases)) / length(phases)
    coherence = abs(mean_vector)  # 0 = random phases, 1 = aligned phases
    
    # Weight by contribution strength
    weighted_coherence = sum(weights .* abs.(projected_outputs)) / sum(weights)
    
    # Combine metrics
    confidence = 0.6f0 * coherence + 0.4f0 * min(1.0f0, weighted_coherence)
    
    return clamp(confidence, 0.0f0, 1.0f0)
end

"""
Generate prediction report
"""
function generate_prediction_report(predictor::StreamingPredictor)::String
    if !predictor.is_warmed_up
        return "Predictor still warming up ($(predictor.system.current_tick)/$(predictor.warmup_period) ticks)"
    end
    
    report = "Prediction Report at tick $(predictor.system.current_tick)\n"
    report *= "="^50 * "\n"
    
    # Current predictions
    report *= "Current Predictions (with phase extrapolation):\n"
    for (horizon, prediction) in sort(collect(predictor.horizon_predictions))
        report *= "  Horizon $horizon ticks: $(round(prediction, digits=4))\n"
    end
    
    # System statistics
    sys = predictor.system
    if sys.predictions_made > 0
        avg_error = sys.total_error / sys.predictions_made
        report *= "\nPerformance Statistics:\n"
        report *= "  Predictions made: $(sys.predictions_made)\n"
        report *= "  Average error: $(round(avg_error, digits=4))\n"
        report *= "  Directional accuracy: $(round(sys.directional_accuracy * 100, digits=1))%\n"
    end
    
    # Weight distribution
    report *= "\nWeight Distribution:\n"
    for i in 1:sys.n_filters
        report *= "  Filter $i (period $(sys.filter_periods[i])): $(round(sys.weights[i], digits=3))\n"
    end
    
    # Filter frequencies
    freqs = calculate_filter_frequencies(sys.filter_periods)
    report *= "\nFilter Design Frequencies (rad/bar):\n"
    for i in 1:sys.n_filters
        report *= "  Filter $i: $(round(freqs[i], digits=4))\n"
    end
    
    return report
end

end # module PricePrediction