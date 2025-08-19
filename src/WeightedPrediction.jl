# src/WeightedPrediction.jl - Unified Weight Optimization and Phase-Based Prediction
# Chunk 4 - Merged from WeightOptimization.jl and PricePrediction.jl

"""
Weighted Prediction Module - Chunk 4 (Unified)

Combines weight optimization and phase-based prediction into a single module.
Optimizes scalar weights [0,1] for filter outputs to minimize prediction error
using phase extrapolation for accurate future price prediction.

Key Features:
- Scalar weights with RMS-based initialization
- Phase-based extrapolation using design frequencies
- Multi-horizon prediction support
- GA-based weight optimization
- Streaming prediction with CircularArrays
- Vectorized operations for GPU compatibility
"""

module WeightedPrediction

using Statistics
using LinearAlgebra
using Random
using CircularArrays  # Using package instead of custom implementation
using Dates

export # Weight Structures
       WeightSet, PredictionWeights, WeightOptimizer, WeightPopulation,
       # Prediction Structures
       PredictionSystem, PredictionResult, StreamingPredictor,
       FilterPhaseState, PredictionBuffer,
       # Initialization
       initialize_weights_rms, create_weight_population,
       # Optimization
       optimize_weights, evaluate_weight_fitness, evolve_weights,
       mutate_weights, crossover_weights,
       # Prediction
       predict_price_change_extrapolated, predict_batch_extrapolated,
       project_filter_forward, calculate_filter_frequencies,
       # System Management
       create_prediction_system, update_prediction!,
       get_prediction_at_horizon, evaluate_predictions,
       # Streaming
       create_streaming_predictor, process_tick!,
       get_current_predictions, generate_prediction_report,
       # Multi-horizon
       get_weights_for_horizon, optimize_weights_range,
       # Utilities
       apply_weights, calculate_prediction_confidence

# =============================================================================
# CONSTANTS
# =============================================================================

const PHASE_PER_TICK = Float32(π / 2)  # 4-phase rotation: 90° per tick

# =============================================================================
# WEIGHT STRUCTURES
# =============================================================================

"""
Weight set for a specific prediction horizon
"""
struct WeightSet
    horizon::Int32                    # Prediction horizon in ticks
    weights::Vector{Float32}          # Scalar weights [0,1] per filter
    fitness::Float32                  # Fitness score for this weight set
    mse::Float32                     # Mean squared error
    mae::Float32                     # Mean absolute error
    directional_accuracy::Float32    # Percentage of correct direction predictions
end

"""
Collection of weights for multiple horizons
"""
mutable struct PredictionWeights
    filter_index::Int32               # Which filter these weights belong to
    fibonacci_number::Int32           # Fibonacci period of the filter
    horizon_range::Tuple{Int32, Int32}  # Min and max horizon
    weight_sets::Vector{WeightSet}   # Weight sets for different horizons
    rms_normalization::Float32       # RMS value used for initialization
    last_update::Int64               # Timestamp of last update
end

"""
Weight optimizer for a single filter
"""
mutable struct WeightOptimizer
    filter_index::Int32
    fibonacci_number::Int32
    population_size::Int32
    mutation_rate::Float32
    crossover_rate::Float32
    elite_size::Int32
    
    # Current population of weight vectors
    population::Matrix{Float32}      # population_size × n_filters
    fitness::Vector{Float32}         # Fitness for each individual
    
    # Best weights found
    best_weights::Vector{Float32}
    best_fitness::Float32
    
    # Evolution tracking
    generation::Int32
    generations_since_improvement::Int32
    fitness_history::Vector{Float32}
end

"""
Population of weight sets for GA optimization
"""
mutable struct WeightPopulation
    individuals::Matrix{Float32}     # population_size × n_filters
    fitness::Vector{Float32}
    horizon::Int32
    generation::Int32
    
    # Statistics
    best_fitness::Float32
    mean_fitness::Float32
    worst_fitness::Float32
end

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
    
    # Circular buffers for each filter
    filter_buffers::Vector{CircularVector{ComplexF32}}
    
    # Phase states for each filter
    filter_states::Vector{FilterPhaseState}
end

function PredictionBuffer(capacity::Int32, n_filters::Int32, filter_periods::Vector{Float32})
    # Create circular buffers using CircularArrays package
    filter_buffers = [CircularVector{ComplexF32}(capacity) for _ in 1:n_filters]
    
    # Initialize phase states with design frequencies
    filter_states = Vector{FilterPhaseState}(undef, n_filters)
    for i in 1:n_filters
        period = filter_periods[i]
        design_freq = Float32(2π / period)  # Radians per tick
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
    input_buffer::CircularVector{ComplexF32}  # Input signal history
    
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
# RMS-BASED WEIGHT INITIALIZATION
# =============================================================================

"""
Calculate RMS values for filter outputs over a calibration period
Vectorized for efficiency
"""
function calculate_filter_rms(filter_outputs::Vector{Vector{ComplexF32}})::Vector{Float32}
    n_filters = length(filter_outputs)
    rms_values = Vector{Float32}(undef, n_filters)
    
    @inbounds for i in 1:n_filters
        if isempty(filter_outputs[i])
            rms_values[i] = 1.0f0  # Default if no data
        else
            # Vectorized RMS calculation
            rms_values[i] = sqrt(mean(abs2.(filter_outputs[i])))
            # Prevent division by zero
            rms_values[i] = max(rms_values[i], 1.0f-10)
        end
    end
    
    return rms_values
end

"""
Initialize weights using RMS normalization for equal contributions
"""
function initialize_weights_rms(filter_outputs::Vector{Vector{ComplexF32}};
                               target_rms::Union{Float32, Nothing} = nothing)::Vector{Float32}
    
    # Calculate RMS for each filter
    rms_values = calculate_filter_rms(filter_outputs)
    
    # Determine target RMS (default: mean of all RMS values)
    if target_rms === nothing
        target_rms = mean(rms_values)
    end
    
    # Vectorized weight initialization
    n_filters = length(rms_values)
    weights = Vector{Float32}(undef, n_filters)
    
    @inbounds for i in 1:n_filters
        # Weight inversely proportional to RMS
        weights[i] = target_rms / rms_values[i]
        # Clamp to [0, 1] range
        weights[i] = clamp(weights[i], 0.0f0, 1.0f0)
    end
    
    # Normalize so sum of weights = 1
    weight_sum = sum(weights)
    if weight_sum > 0
        weights ./= weight_sum
    else
        weights .= 1.0f0 / n_filters  # Equal weights fallback
    end
    
    return weights
end

# =============================================================================
# FREQUENCY AND PHASE CALCULATIONS
# =============================================================================

"""
Calculate design frequencies for filters based on periods
Vectorized operation
"""
function calculate_filter_frequencies(filter_periods::Vector{Float32})::Vector{Float32}
    # Design frequency = 2π / period (radians per tick)
    return Float32(2π) ./ filter_periods
end

"""
Project filter output forward n ticks using phase extrapolation
Magnitude remains constant (no decay assumption)
"""
function project_filter_forward(
    current_output::ComplexF32,
    design_frequency::Float32,
    n_ticks::Int32
)::ComplexF32
    
    # Extract magnitude (remains constant)
    magnitude = abs(current_output)
    
    # Current phase
    current_phase = angle(current_output)
    
    # Phase advancement based on design frequency
    phase_advance = design_frequency * Float32(n_ticks)
    
    # Total projected phase
    projected_phase = current_phase + phase_advance
    
    # Reconstruct complex prediction
    return magnitude * exp(im * projected_phase)
end

"""
Vectorized projection for multiple filters
"""
function project_filters_forward(
    filter_outputs::Vector{ComplexF32},
    frequencies::Vector{Float32},
    n_ticks::Int32
)::Vector{ComplexF32}
    
    n_filters = length(filter_outputs)
    projected = Vector{ComplexF32}(undef, n_filters)
    
    @inbounds for i in 1:n_filters
        projected[i] = project_filter_forward(
            filter_outputs[i],
            frequencies[i],
            n_ticks
        )
    end
    
    return projected
end

# =============================================================================
# WEIGHT APPLICATION
# =============================================================================

"""
Apply scalar weights to filter outputs (preserves phase)
Vectorized dot product
"""
function apply_weights(filter_outputs::Vector{ComplexF32}, 
                       weights::Vector{Float32})::ComplexF32
    
    @assert length(filter_outputs) == length(weights) "Dimension mismatch"
    
    # Vectorized weighted sum
    return sum(weights .* filter_outputs)
end

"""
Apply weights to batch of filter outputs
Fully vectorized
"""
function apply_weights(filter_outputs::Matrix{ComplexF32},
                       weights::Vector{Float32})::Vector{ComplexF32}
    
    n_samples, n_filters = size(filter_outputs)
    @assert n_filters == length(weights) "Dimension mismatch"
    
    # Matrix-vector multiplication
    return filter_outputs * weights
end

# =============================================================================
# CORE PREDICTION FUNCTIONS
# =============================================================================

"""
Predict price change using phase-extrapolated weighted vector sum
Returns I-component (real part) of the prediction
"""
function predict_price_change_extrapolated(
    filter_outputs::Vector{ComplexF32},
    filter_frequencies::Vector{Float32},
    weights::Vector{Float32},
    n_ticks::Int32
)::Float32
    
    @assert length(filter_outputs) == length(weights) == length(filter_frequencies) "Dimension mismatch"
    
    # Project all filters forward (vectorized)
    projected_outputs = project_filters_forward(
        filter_outputs,
        filter_frequencies,
        n_ticks
    )
    
    # Apply weights and sum
    prediction_vector = apply_weights(projected_outputs, weights)
    
    # Extract I-component (normalized price change)
    return real(prediction_vector)
end

"""
Batch prediction with phase extrapolation
Optimized for performance
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
    
    # Pre-calculate design frequencies
    filter_frequencies = calculate_filter_frequencies(filter_periods)
    
    # Pre-allocate output
    predictions = Vector{Float32}(undef, n_predictions)
    
    # Vectorized prediction loop
    @inbounds for t in 1:n_predictions
        # Get current filter outputs
        current_outputs = @view filter_outputs[t, :]
        
        # Predict future value with extrapolation
        predictions[t] = predict_price_change_extrapolated(
            current_outputs,
            filter_frequencies,
            weights,
            Int32(horizon)
        )
    end
    
    return predictions
end

# =============================================================================
# WEIGHT OPTIMIZATION
# =============================================================================

"""
Evaluate fitness of weight set for prediction with phase extrapolation
Optimized for performance with vectorized operations
"""
function evaluate_weight_fitness(weights::Vector{Float32},
                                filter_outputs::Matrix{ComplexF32},
                                actual_future::Vector{ComplexF32},
                                horizon::Int32;
                                filter_periods::Vector{Float32} = Float32[])::Tuple{Float32, Float32, Float32, Float32}
    
    n_samples, n_filters = size(filter_outputs)
    n_predictions = n_samples - horizon
    
    if n_predictions <= 0
        return (0.0f0, Inf32, Inf32, 0.0f0)
    end
    
    # Calculate filter frequencies
    if !isempty(filter_periods)
        @assert length(filter_periods) == n_filters "Period count mismatch"
        filter_frequencies = calculate_filter_frequencies(filter_periods)
    else
        # Default: assume standard Fibonacci periods
        default_periods = Float32[2.01, 4, 6, 10, 16, 26, 42, 68, 110]
        filter_frequencies = Float32(2π) ./ default_periods[1:min(n_filters, length(default_periods))]
    end
    
    # Pre-allocate arrays
    predictions = Vector{Float32}(undef, n_predictions)
    actuals = Vector{Float32}(undef, n_predictions)
    
    # Vectorized prediction loop
    @inbounds for t in 1:n_predictions
        # Get current filter outputs
        current_outputs = @view filter_outputs[t, :]
        
        # Project filters forward (vectorized)
        projected = project_filters_forward(
            current_outputs,
            filter_frequencies,
            horizon
        )
        
        # Apply weights to projected outputs
        weighted_sum = apply_weights(projected, weights)
        
        # Extract I-component for comparison
        predictions[t] = real(weighted_sum)
        actuals[t] = real(actual_future[t + horizon])
    end
    
    # Vectorized metric calculations
    errors = predictions .- actuals
    mse = mean(abs2.(errors))
    mae = mean(abs.(errors))
    
    # Directional accuracy (vectorized sign comparison)
    correct_direction = sum(sign.(predictions) .== sign.(actuals))
    dir_accuracy = Float32(correct_direction / n_predictions)
    
    # Fitness: Weighted combination (higher is better)
    fitness = 1.0f0 / (1.0f0 + mse) * (0.7f0 + 0.3f0 * dir_accuracy)
    
    return (fitness, mse, mae, dir_accuracy)
end

"""
Create initial population of weight vectors
"""
function create_weight_population(n_filters::Int, population_size::Int;
                                 initial_weights::Union{Vector{Float32}, Nothing} = nothing)::Matrix{Float32}
    
    population = Matrix{Float32}(undef, population_size, n_filters)
    
    if initial_weights !== nothing
        # Seed population with variations of initial weights
        @inbounds for i in 1:population_size
            if i == 1
                # Keep one copy of initial weights
                population[i, :] = initial_weights
            else
                # Add noise to initial weights
                noise_scale = 0.1f0 * (1.0f0 + Float32(i - 2) / population_size)
                noise = randn(Float32, n_filters) * noise_scale
                population[i, :] = clamp.(initial_weights .+ noise, 0.0f0, 1.0f0)
            end
            
            # Normalize
            row_sum = sum(@view population[i, :])
            if row_sum > 0
                population[i, :] ./= row_sum
            end
        end
    else
        # Random initialization
        @inbounds for i in 1:population_size
            population[i, :] = rand(Float32, n_filters)
            # Normalize
            population[i, :] ./= sum(@view population[i, :])
        end
    end
    
    return population
end

"""
Mutate weight vector
"""
function mutate_weights!(weights::Vector{Float32}, mutation_rate::Float32;
                        mutation_strength::Float32 = 0.1f0)
    
    n_filters = length(weights)
    
    @inbounds for i in 1:n_filters
        if rand() < mutation_rate
            # Gaussian mutation
            mutation = randn(Float32) * mutation_strength
            weights[i] = clamp(weights[i] + mutation, 0.0f0, 1.0f0)
        end
    end
    
    # Renormalize
    weight_sum = sum(weights)
    if weight_sum > 0
        weights ./= weight_sum
    end
    
    return weights
end

"""
Crossover two weight vectors
"""
function crossover_weights(parent1::Vector{Float32}, parent2::Vector{Float32},
                          crossover_rate::Float32)::Tuple{Vector{Float32}, Vector{Float32}}
    
    n_filters = length(parent1)
    child1 = copy(parent1)
    child2 = copy(parent2)
    
    if rand() < crossover_rate
        # Uniform crossover
        @inbounds for i in 1:n_filters
            if rand() < 0.5
                child1[i], child2[i] = child2[i], child1[i]
            end
        end
        
        # Renormalize
        child1 ./= sum(child1)
        child2 ./= sum(child2)
    end
    
    return (child1, child2)
end

"""
Evolve population of weights using GA with phase extrapolation
Optimized with vectorized operations
"""
function evolve_weights(population::Matrix{Float32},
                       filter_outputs::Matrix{ComplexF32},
                       actual_future::Vector{ComplexF32},
                       horizon::Int32;
                       filter_periods::Vector{Float32} = Float32[],
                       mutation_rate::Float32 = 0.1f0,
                       crossover_rate::Float32 = 0.7f0,
                       elite_size::Int = 2)::Tuple{Matrix{Float32}, Vector{Float32}}
    
    population_size, n_filters = size(population)
    
    # Evaluate fitness for all individuals (can be parallelized)
    fitness = Vector{Float32}(undef, population_size)
    @inbounds for i in 1:population_size
        fitness[i], _, _, _ = evaluate_weight_fitness(
            @view(population[i, :]), 
            filter_outputs, 
            actual_future, 
            horizon,
            filter_periods=filter_periods
        )
    end
    
    # Sort by fitness (descending)
    sorted_indices = sortperm(fitness, rev=true)
    
    # New population
    new_population = Matrix{Float32}(undef, population_size, n_filters)
    
    # Elitism: Keep best individuals
    @inbounds for i in 1:elite_size
        new_population[i, :] = population[sorted_indices[i], :]
    end
    
    # Tournament selection and reproduction
    @inbounds for i in (elite_size + 1):2:population_size
        # Tournament selection
        tournament_size = 3
        
        parent1_idx = sorted_indices[rand(1:min(tournament_size, population_size))]
        parent2_idx = sorted_indices[rand(1:min(tournament_size, population_size))]
        
        parent1 = @view population[parent1_idx, :]
        parent2 = @view population[parent2_idx, :]
        
        # Crossover
        child1, child2 = crossover_weights(parent1, parent2, crossover_rate)
        
        # Mutation
        mutate_weights!(child1, mutation_rate)
        if i + 1 <= population_size
            mutate_weights!(child2, mutation_rate)
        end
        
        # Add to new population
        new_population[i, :] = child1
        if i + 1 <= population_size
            new_population[i + 1, :] = child2
        end
    end
    
    return (new_population, fitness)
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
    input_buffer = CircularVector{ComplexF32}(max_horizon)
    
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
    @inbounds for i in 1:system.n_filters
        state = system.prediction_buffer.filter_states[i]
        state.current_output = filter_outputs[i]
        state.tick_count = system.current_tick
        
        # Store in circular buffer
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
    
    @inbounds for i in 1:system.n_filters
        state = system.prediction_buffer.filter_states[i]
        current_outputs[i] = state.current_output
    end
    
    # Make prediction with extrapolation
    return predict_price_change_extrapolated(
        current_outputs,
        filter_frequencies,
        system.weights,
        horizon
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
    
    @inbounds for i in 1:n_evaluations
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
            horizon
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
    
    # Calculate metrics (vectorized)
    mse = mean(abs2.(errors))
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
        key_horizons = Int32[100, 250, 500, 1000, 1500, 2000]
        
        @inbounds for horizon in key_horizons
            if horizon >= predictor.system.horizon_range[1] && 
               horizon <= predictor.system.horizon_range[2]
                
                prediction = get_prediction_at_horizon(predictor.system, horizon)
                predictor.horizon_predictions[horizon] = prediction
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
# MULTI-HORIZON SUPPORT
# =============================================================================

"""
Get optimal weights for a specific horizon (with interpolation)
"""
function get_weights_for_horizon(pred_weights::PredictionWeights, 
                                horizon::Int32)::Vector{Float32}
    
    # Check if horizon is in range
    if horizon < pred_weights.horizon_range[1] || horizon > pred_weights.horizon_range[2]
        error("Horizon $horizon outside range $(pred_weights.horizon_range)")
    end
    
    # Find closest weight sets
    if isempty(pred_weights.weight_sets)
        # Return equal weights as fallback
        n_weights = pred_weights.fibonacci_number  # Approximate
        return ones(Float32, n_weights) ./ n_weights
    end
    
    # Find bracketing weight sets
    lower_set = nothing
    upper_set = nothing
    exact_match = nothing
    
    @inbounds for ws in pred_weights.weight_sets
        if ws.horizon == horizon
            exact_match = ws
            break
        elseif ws.horizon < horizon
            if lower_set === nothing || ws.horizon > lower_set.horizon
                lower_set = ws
            end
        else  # ws.horizon > horizon
            if upper_set === nothing || ws.horizon < upper_set.horizon
                upper_set = ws
            end
        end
    end
    
    # Return exact match if found
    if exact_match !== nothing
        return exact_match.weights
    end
    
    # Interpolate between closest sets
    if lower_set !== nothing && upper_set !== nothing
        # Linear interpolation
        alpha = Float32(horizon - lower_set.horizon) / 
                Float32(upper_set.horizon - lower_set.horizon)
        
        interpolated = (1 - alpha) * lower_set.weights + alpha * upper_set.weights
        
        # Renormalize
        return interpolated ./ sum(interpolated)
        
    elseif lower_set !== nothing
        return lower_set.weights
    elseif upper_set !== nothing
        return upper_set.weights
    else
        # No weight sets available
        n_weights = pred_weights.fibonacci_number
        return ones(Float32, n_weights) ./ n_weights
    end
end

"""
Optimize weights for multiple horizons in range
"""
function optimize_weights_range(filter_outputs::Matrix{ComplexF32},
                               actual_future::Vector{ComplexF32},
                               filter_periods::Vector{Float32},
                               horizon_range::Tuple{Int32, Int32};
                               n_horizons::Int = 10,
                               population_size::Int = 50,
                               n_generations::Int = 100)::Vector{WeightSet}
    
    # Select horizons to optimize
    min_h, max_h = horizon_range
    horizons = Int32.(round.(range(min_h, max_h, length=n_horizons)))
    
    weight_sets = Vector{WeightSet}()
    
    # Initialize with RMS-based weights
    n_filters = size(filter_outputs, 2)
    filter_outputs_vec = [filter_outputs[:, i] for i in 1:n_filters]
    initial_weights = initialize_weights_rms(filter_outputs_vec)
    
    for horizon in horizons
        println("Optimizing weights for horizon $horizon...")
        
        # Create population
        population = create_weight_population(n_filters, 
                                             population_size, 
                                             initial_weights=initial_weights)
        
        # Evolve
        best_fitness = -Inf32
        best_weights = initial_weights
        
        for gen in 1:n_generations
            population, fitness = evolve_weights(population, filter_outputs, 
                                                actual_future, horizon,
                                                filter_periods=filter_periods)
            
            # Track best
            max_fitness = maximum(fitness)
            if max_fitness > best_fitness
                best_fitness = max_fitness
                best_idx = argmax(fitness)
                best_weights = population[best_idx, :]
            end
            
            if gen % 10 == 0
                println("  Generation $gen: Best fitness = $best_fitness")
            end
        end
        
        # Calculate final metrics
        _, mse, mae, dir_acc = evaluate_weight_fitness(best_weights, filter_outputs,
                                                       actual_future, horizon,
                                                       filter_periods=filter_periods)
        
        push!(weight_sets, WeightSet(horizon, best_weights, best_fitness, 
                                     mse, mae, dir_acc))
    end
    
    return weight_sets
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
    @inbounds for i in 1:system.n_filters
        historical_outputs[i] = system.prediction_buffer.filter_buffers[i][current_idx]
    end
    
    # Get filter frequencies
    filter_frequencies = calculate_filter_frequencies(system.filter_periods)
    
    # Prediction from that time with extrapolation
    predicted = predict_price_change_extrapolated(
        historical_outputs,
        filter_frequencies,
        system.weights,
        horizon
    )
    
    # Actual value (now available)
    actual = real(last(system.input_buffer))
    
    # Prediction error
    error = predicted - actual
    
    # Project filters forward to get derivatives
    projected_outputs = project_filters_forward(
        historical_outputs,
        filter_frequencies,
        horizon
    )
    
    # Gradient descent update (vectorized)
    gradients = 2.0f0 * error * real.(projected_outputs)
    
    # Update weights
    system.weights .-= system.learning_rate * gradients
    
    # Clamp to valid range
    system.weights .= clamp.(system.weights, 0.0f0, 1.0f0)
    
    # Renormalize weights
    weight_sum = sum(system.weights)
    if weight_sum > 0
        system.weights ./= weight_sum
    end
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
    
    # Calculate circular variance (vectorized)
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
    
    report = "PREDICTION REPORT - Tick $(predictor.system.current_tick)\n"
    report *= "="^50 * "\n"
    
    # Current predictions
    report *= "Current Predictions (phase-extrapolated):\n"
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
    report *= "\nFilter Design Frequencies (rad/tick):\n"
    for i in 1:sys.n_filters
        report *= "  Filter $i: $(round(freqs[i], digits=4))\n"
    end
    
    return report
end

end # module WeightedPrediction