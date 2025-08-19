# src/WeightOptimization.jl - Scalar Weight Optimization for Filter Bank

"""
Weight Optimization Module - Chunk 4

Optimizes scalar weights [0,1] for filter outputs to minimize prediction error.
Uses RMS-based initialization to ensure equal initial contributions from all filters.

Key Features:
- Scalar weights only (preserves phase relationships)
- RMS-based weight initialization
- Support for range of prediction horizons (100-2000 ticks)
- Per-filter independent optimization
- Multi-horizon weight sets
"""

module WeightOptimization

using Statistics
using LinearAlgebra
using Random

export WeightSet, WeightOptimizer, PredictionWeights,
       initialize_weights_rms, optimize_weights,
       apply_weights, get_weights_for_horizon,
       evaluate_weight_fitness, mutate_weights,
       crossover_weights, WeightPopulation,
       create_weight_population, evolve_weights

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
    horizon_range::Tuple{Int32, Int32}  # Min and max horizon (e.g., (100, 2000))
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
# RMS-BASED WEIGHT INITIALIZATION
# =============================================================================

"""
Calculate RMS values for filter outputs over a calibration period
"""
function calculate_filter_rms(filter_outputs::Vector{Vector{ComplexF32}})::Vector{Float32}
    n_filters = length(filter_outputs)
    rms_values = Vector{Float32}(undef, n_filters)
    
    for i in 1:n_filters
        if isempty(filter_outputs[i])
            rms_values[i] = 1.0f0  # Default if no data
        else
            # RMS of complex signal magnitude
            rms_values[i] = sqrt(mean(abs2.(filter_outputs[i])))
            
            # Prevent division by zero
            if rms_values[i] < 1e-10
                rms_values[i] = 1e-10
            end
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
    
    # Initialize weights to normalize RMS
    n_filters = length(rms_values)
    weights = Vector{Float32}(undef, n_filters)
    
    for i in 1:n_filters
        # Weight inversely proportional to RMS
        weights[i] = target_rms / rms_values[i]
        
        # Clamp to [0, 1] range
        weights[i] = clamp(weights[i], 0.0f0, 1.0f0)
    end
    
    # Normalize so sum of weights = 1 (optional, for stability)
    weight_sum = sum(weights)
    if weight_sum > 0
        weights ./= weight_sum
    else
        weights .= 1.0f0 / n_filters  # Equal weights fallback
    end
    
    return weights
end

"""
Initialize weights with calibration data
"""
function initialize_weights_rms(filter_outputs::Matrix{ComplexF32};
                               calibration_samples::Int = 1000)::Vector{Float32}
    
    n_samples, n_filters = size(filter_outputs)
    cal_samples = min(calibration_samples, n_samples)
    
    # Extract calibration period
    cal_outputs = [filter_outputs[1:cal_samples, i] for i in 1:n_filters]
    
    return initialize_weights_rms(cal_outputs)
end

# =============================================================================
# WEIGHT APPLICATION
# =============================================================================

"""
Apply scalar weights to filter outputs (preserves phase)
"""
function apply_weights(filter_outputs::Vector{ComplexF32}, 
                       weights::Vector{Float32})::ComplexF32
    
    @assert length(filter_outputs) == length(weights) "Dimension mismatch"
    
    # Scalar multiplication and vector sum
    weighted_sum = ComplexF32(0, 0)
    for i in 1:length(weights)
        weighted_sum += weights[i] * filter_outputs[i]
    end
    
    return weighted_sum
end

"""
Apply weights to batch of filter outputs
"""
function apply_weights(filter_outputs::Matrix{ComplexF32},
                       weights::Vector{Float32})::Vector{ComplexF32}
    
    n_samples, n_filters = size(filter_outputs)
    @assert n_filters == length(weights) "Dimension mismatch"
    
    predictions = Vector{ComplexF32}(undef, n_samples)
    
    for t in 1:n_samples
        predictions[t] = apply_weights(filter_outputs[t, :], weights)
    end
    
    return predictions
end

# =============================================================================
# WEIGHT OPTIMIZATION
# =============================================================================

"""
Evaluate fitness of weight set for prediction with phase extrapolation
Compares I-component (real part) only
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
    
    # Calculate filter frequencies if periods provided
    if !isempty(filter_periods)
        @assert length(filter_periods) == n_filters "Period count mismatch"
        filter_frequencies = 2π ./ filter_periods
    else
        # Default: assume standard Fibonacci periods
        default_periods = Float32[2.01, 4, 6, 10, 16, 26, 42, 68, 110]
        filter_frequencies = 2π ./ default_periods[1:min(n_filters, length(default_periods))]
    end
    
    # Generate predictions with phase extrapolation
    predictions = Vector{Float32}(undef, n_predictions)
    actuals = Vector{Float32}(undef, n_predictions)
    
    for t in 1:n_predictions
        # Get current filter outputs
        current_outputs = filter_outputs[t, :]
        
        # Project each filter forward by horizon ticks
        projected = Vector{ComplexF32}(undef, n_filters)
        for k in 1:n_filters
            magnitude = abs(current_outputs[k])
            phase = angle(current_outputs[k])
            # Project phase forward
            projected_phase = phase + filter_frequencies[k] * horizon
            projected[k] = magnitude * exp(im * projected_phase)
        end
        
        # Apply weights to projected outputs
        weighted_sum = apply_weights(projected, weights)
        
        # Extract I-component for comparison
        predictions[t] = real(weighted_sum)
        actuals[t] = real(actual_future[t + horizon])
    end
    
    # Calculate metrics
    errors = predictions .- actuals
    mse = mean(errors .^ 2)
    mae = mean(abs.(errors))
    
    # Directional accuracy (sign agreement)
    correct_direction = sum(sign.(predictions) .== sign.(actuals))
    dir_accuracy = correct_direction / n_predictions
    
    # Fitness: Weighted combination (higher is better)
    # Prioritize MSE but include directional accuracy
    fitness = 1.0f0 / (1.0f0 + mse) * (0.7f0 + 0.3f0 * dir_accuracy)
    
    return (fitness, mse, mae, dir_accuracy)
end

"""
Gradient-based weight optimization (for comparison with GA)
"""
function optimize_weights_gradient(filter_outputs::Matrix{ComplexF32},
                                  actual_future::Vector{ComplexF32},
                                  horizon::Int32;
                                  learning_rate::Float32 = 0.01f0,
                                  max_iterations::Int = 1000,
                                  initial_weights::Union{Vector{Float32}, Nothing} = nothing)::Vector{Float32}
    
    n_samples, n_filters = size(filter_outputs)
    
    # Initialize weights
    if initial_weights === nothing
        weights = initialize_weights_rms(filter_outputs)
    else
        weights = copy(initial_weights)
    end
    
    # Gradient descent optimization
    for iter in 1:max_iterations
        # Calculate gradient numerically (finite differences)
        gradient = Vector{Float32}(undef, n_filters)
        epsilon = 1e-4
        
        base_fitness, _, _, _ = evaluate_weight_fitness(weights, filter_outputs, 
                                                        actual_future, horizon)
        
        for i in 1:n_filters
            weights_plus = copy(weights)
            weights_plus[i] += epsilon
            weights_plus[i] = clamp(weights_plus[i], 0.0f0, 1.0f0)
            
            fitness_plus, _, _, _ = evaluate_weight_fitness(weights_plus, filter_outputs,
                                                            actual_future, horizon)
            
            gradient[i] = (fitness_plus - base_fitness) / epsilon
        end
        
        # Update weights (gradient ascent for fitness maximization)
        weights .+= learning_rate * gradient
        
        # Clamp to valid range
        weights .= clamp.(weights, 0.0f0, 1.0f0)
        
        # Normalize (optional)
        weight_sum = sum(weights)
        if weight_sum > 0
            weights ./= weight_sum
        end
        
        # Early stopping if converged
        if maximum(abs.(gradient)) < 1e-6
            break
        end
    end
    
    return weights
end

# =============================================================================
# GENETIC ALGORITHM OPERATIONS
# =============================================================================

"""
Create initial population of weight vectors
"""
function create_weight_population(n_filters::Int, population_size::Int;
                                 initial_weights::Union{Vector{Float32}, Nothing} = nothing)::Matrix{Float32}
    
    population = Matrix{Float32}(undef, population_size, n_filters)
    
    if initial_weights !== nothing
        # Seed population with variations of initial weights
        for i in 1:population_size
            if i == 1
                # Keep one copy of initial weights
                population[i, :] = initial_weights
            else
                # Add noise to initial weights
                noise_scale = 0.1f0 * (1.0f0 + (i - 2) / population_size)
                noise = randn(Float32, n_filters) * noise_scale
                population[i, :] = clamp.(initial_weights .+ noise, 0.0f0, 1.0f0)
            end
            
            # Normalize
            row_sum = sum(population[i, :])
            if row_sum > 0
                population[i, :] ./= row_sum
            end
        end
    else
        # Random initialization
        for i in 1:population_size
            population[i, :] = rand(Float32, n_filters)
            # Normalize
            population[i, :] ./= sum(population[i, :])
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
    
    for i in 1:n_filters
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
        for i in 1:n_filters
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
    
    # Evaluate fitness for all individuals with phase extrapolation
    fitness = Vector{Float32}(undef, population_size)
    for i in 1:population_size
        fitness[i], _, _, _ = evaluate_weight_fitness(population[i, :], 
                                                      filter_outputs, 
                                                      actual_future, 
                                                      horizon,
                                                      filter_periods=filter_periods)
    end
    
    # Sort by fitness (descending)
    sorted_indices = sortperm(fitness, rev=true)
    
    # New population
    new_population = Matrix{Float32}(undef, population_size, n_filters)
    
    # Elitism: Keep best individuals
    for i in 1:elite_size
        new_population[i, :] = population[sorted_indices[i], :]
    end
    
    # Tournament selection and reproduction
    for i in (elite_size + 1):2:population_size
        # Tournament selection
        tournament_size = 3
        
        parent1_idx = sorted_indices[rand(1:min(tournament_size, population_size))]
        parent2_idx = sorted_indices[rand(1:min(tournament_size, population_size))]
        
        parent1 = population[parent1_idx, :]
        parent2 = population[parent2_idx, :]
        
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
        return ones(Float32, length(pred_weights.weight_sets[1].weights)) ./ 
               length(pred_weights.weight_sets[1].weights)
    end
    
    # Find bracketing weight sets
    lower_set = nothing
    upper_set = nothing
    exact_match = nothing
    
    for ws in pred_weights.weight_sets
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
        n_weights = pred_weights.fibonacci_number  # Approximate
        return ones(Float32, n_weights) ./ n_weights
    end
end

"""
Optimize weights for multiple horizons in range
"""
function optimize_weights_range(filter_outputs::Matrix{ComplexF32},
                               actual_future::Vector{ComplexF32},
                               horizon_range::Tuple{Int32, Int32};
                               n_horizons::Int = 10,
                               population_size::Int = 50,
                               n_generations::Int = 100)::Vector{WeightSet}
    
    # Select horizons to optimize
    min_h, max_h = horizon_range
    horizons = Int32.(round.(range(min_h, max_h, length=n_horizons)))
    
    weight_sets = Vector{WeightSet}()
    
    # Initialize with RMS-based weights
    initial_weights = initialize_weights_rms(filter_outputs)
    
    for horizon in horizons
        println("Optimizing weights for horizon $horizon...")
        
        # Create population
        population = create_weight_population(size(filter_outputs, 2), 
                                             population_size, 
                                             initial_weights=initial_weights)
        
        # Evolve
        best_fitness = -Inf32
        best_weights = initial_weights
        
        for gen in 1:n_generations
            population, fitness = evolve_weights(population, filter_outputs, 
                                                actual_future, horizon)
            
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
                                                       actual_future, horizon)
        
        push!(weight_sets, WeightSet(horizon, best_weights, best_fitness, 
                                     mse, mae, dir_acc))
    end
    
    return weight_sets
end

end # module WeightOptimization