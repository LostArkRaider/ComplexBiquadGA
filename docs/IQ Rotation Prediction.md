# I/Q Rotation Prediction - Geometric Approach Implementation Guide

## Preface: Why Avoid Trigonometry in Signal Processing

### Computational Cost
Trigonometric functions (sin, cos, tan) are among the most computationally expensive operations in signal processing:
- **CPU Cycles**: A single sin/cos call can take 50-200 CPU cycles, compared to 1-5 cycles for multiplication
- **No Parallelization**: Most trig implementations are sequential, preventing SIMD vectorization
- **Pipeline Stalls**: Complex trig operations cause instruction pipeline stalls in modern processors

### Numerical Precision Issues
- **Accumulating Errors**: Repeated trig calculations accumulate floating-point errors
- **Range Reduction Problems**: Large arguments require range reduction, introducing additional errors
- **Platform Inconsistencies**: Different hardware/libraries may produce slightly different results

### Real-Time Constraints
Signal processing often operates under strict timing requirements:
- **Deterministic Execution**: Trig functions may have variable execution time based on input
- **Interrupt Latency**: Long trig operations can delay interrupt handling
- **Power Consumption**: Complex operations increase power draw in embedded systems

### Alternative Approaches Are Superior
- **Lookup Tables**: O(1) access time with predictable memory patterns
- **Linear Algebra**: Matrix operations utilize optimized BLAS libraries and GPU acceleration
- **Algebraic Methods**: Use only addition, multiplication, and bit shifts
- **CORDIC Algorithms**: Hardware-friendly iterative methods using only shifts and adds

For these reasons, production signal processing systems minimize or eliminate runtime trigonometric operations, computing them only during initialization or offline preprocessing.

## Executive Summary

This document provides efficient geometric methods for predicting I/Q (In-phase/Quadrature) components of a rotating vector from a filter output. The approaches avoid expensive trigonometric function calls at runtime by leveraging geometric properties of circular rotation. Special attention is given to optimization techniques suitable for Genetic Algorithm (GA) systems.

## Problem Statement

Given:
- Current I and Q components of a signal at tick count `t₀`
- Filter period `T` (in ticks)
- Desired prediction horizon `n` (ticks into the future)

Find: Predicted I and Q values at tick count `t₀ + n`

## Mathematical Background

The I/Q components represent a 2D vector rotating on a circle. Each tick advances the rotation by angle `θ = 2π/T`. The geometric constraint is that rotation preserves vector magnitude:

```
I² + Q² = constant (radius squared)
```

## Implementation Approaches

### Approach 1: Precomputed Rotation Constants (Recommended)

This is the most practical approach for production systems. Compute rotation constants once during initialization, then use simple arithmetic at runtime.

```julia
# Structure to hold precomputed rotation constants
struct RotationPredictor
    cos_theta::Float64  # cos(2π/T)
    sin_theta::Float64  # sin(2π/T)
    period::Int64       # Filter period T
end

# Initialize predictor with filter period
function init_predictor(T::Int64)
    theta = 2π / T
    return RotationPredictor(cos(theta), sin(theta), T)
end

# Predict I/Q values n ticks ahead
function predict_iq(predictor::RotationPredictor, I0::Float64, Q0::Float64, n::Int64)
    # Handle period wraparound
    n_effective = n % predictor.period
    
    # Apply rotation n times iteratively
    I, Q = I0, Q0
    for _ in 1:n_effective
        I_new = predictor.cos_theta * I - predictor.sin_theta * Q
        Q_new = predictor.sin_theta * I + predictor.cos_theta * Q
        I, Q = I_new, Q_new
    end
    
    return I, Q
end

# Example usage
predictor = init_predictor(100)  # Filter period = 100 ticks
I_current, Q_current = 10.0, 5.0
I_future, Q_future = predict_iq(predictor, I_current, Q_current, 25)
```

### Approach 2: Lookup Table for Fixed Period

When memory is available and speed is critical, precompute all possible rotations.

```julia
struct LookupPredictor
    cos_table::Vector{Float64}
    sin_table::Vector{Float64}
    period::Int64
end

function init_lookup_predictor(T::Int64)
    cos_table = zeros(Float64, T)
    sin_table = zeros(Float64, T)
    
    for i in 0:(T-1)
        theta = 2π * i / T
        cos_table[i+1] = cos(theta)
        sin_table[i+1] = sin(theta)
    end
    
    return LookupPredictor(cos_table, sin_table, T)
end

function predict_iq_lookup(predictor::LookupPredictor, I0::Float64, Q0::Float64, n::Int64)
    # Wrap n to valid table index
    idx = (n % predictor.period) + 1
    
    c = predictor.cos_table[idx]
    s = predictor.sin_table[idx]
    
    I_new = c * I0 - s * Q0
    Q_new = s * I0 + c * Q0
    
    return I_new, Q_new
end
```

### Approach 3: Special Case Optimizations

For common filter periods, use exact geometric relationships without any trig functions.

```julia
# Optimized prediction for special periods
function predict_iq_special(I0::Float64, Q0::Float64, n::Int64, T::Int64)
    n_mod = n % T
    
    # Square (90-degree rotations)
    if T == 4
        if n_mod == 0
            return I0, Q0
        elseif n_mod == 1
            return -Q0, I0
        elseif n_mod == 2
            return -I0, -Q0
        else  # n_mod == 3
            return Q0, -I0
        end
    
    # Half rotation (180 degrees)
    elseif T == 2
        if n_mod == 0
            return I0, Q0
        else
            return -I0, -Q0
        end
    
    # Hexagon (60-degree rotations)
    elseif T == 6
        if n_mod == 0
            return I0, Q0
        elseif n_mod == 1
            # cos(60°) = 1/2, sin(60°) = √3/2
            return 0.5 * I0 - 0.8660254037844387 * Q0,
                   0.8660254037844387 * I0 + 0.5 * Q0
        elseif n_mod == 2
            # cos(120°) = -1/2, sin(120°) = √3/2
            return -0.5 * I0 - 0.8660254037844387 * Q0,
                   0.8660254037844387 * I0 - 0.5 * Q0
        elseif n_mod == 3
            return -I0, -Q0
        elseif n_mod == 4
            return -0.5 * I0 + 0.8660254037844387 * Q0,
                   -0.8660254037844387 * I0 - 0.5 * Q0
        else  # n_mod == 5
            return 0.5 * I0 + 0.8660254037844387 * Q0,
                   -0.8660254037844387 * I0 + 0.5 * Q0
        end
    
    # Octagon (45-degree rotations)  
    elseif T == 8
        sqrt2_2 = 0.7071067811865476  # √2/2
        
        if n_mod == 0
            return I0, Q0
        elseif n_mod == 1
            return sqrt2_2 * (I0 - Q0), sqrt2_2 * (I0 + Q0)
        elseif n_mod == 2
            return -Q0, I0
        elseif n_mod == 3
            return sqrt2_2 * (-I0 - Q0), sqrt2_2 * (I0 - Q0)
        elseif n_mod == 4
            return -I0, -Q0
        elseif n_mod == 5
            return sqrt2_2 * (-I0 + Q0), sqrt2_2 * (-I0 - Q0)
        elseif n_mod == 6
            return Q0, -I0
        else  # n_mod == 7
            return sqrt2_2 * (I0 + Q0), sqrt2_2 * (-I0 + Q0)
        end
    
    else
        # Fall back to general method
        error("Period $T not optimized. Use general predictor.")
    end
end
```

### Approach 4: Complex Number Method

Julia has excellent complex number support, making this approach very clean.

```julia
struct ComplexPredictor
    rotation_factor::ComplexF64
    period::Int64
end

function init_complex_predictor(T::Int64)
    theta = 2π / T
    rotation_factor = exp(im * theta)
    return ComplexPredictor(rotation_factor, T)
end

function predict_iq_complex(predictor::ComplexPredictor, I0::Float64, Q0::Float64, n::Int64)
    # Create complex number from I/Q
    z = complex(I0, Q0)
    
    # Apply rotation
    z_rotated = z * (predictor.rotation_factor ^ (n % predictor.period))
    
    # Extract I/Q components
    return real(z_rotated), imag(z_rotated)
end
```

### Approach 5: Fast Power-of-Two Periods

For filter periods that are powers of 2, use bit manipulation for ultra-fast prediction.

```julia
# Optimized for T = 2^k periods
function predict_iq_power2(I0::Float64, Q0::Float64, n::Int64, log2_T::Int64)
    T = 1 << log2_T  # T = 2^log2_T
    n_mod = n & (T - 1)  # Fast modulo for power of 2
    
    # Precomputed constants for common power-of-2 periods
    if log2_T == 2  # T = 4
        return predict_iq_special(I0, Q0, n_mod, 4)
    elseif log2_T == 3  # T = 8
        return predict_iq_special(I0, Q0, n_mod, 8)
    else
        # Use general method with fast modulo
        theta = 2π * n_mod / T
        c, s = cos(theta), sin(theta)
        return c * I0 - s * Q0, s * I0 + c * Q0
    end
end
```

## Performance Optimization Tips

### 1. Method Selection Guide

| Filter Period | Best Method | Reasoning |
|--------------|-------------|-----------|
| T = 2, 4, 6, 8 | Special Case | Exact values, no trig |
| T < 360 | Lookup Table | Fast indexing, memory trade-off |
| T = 2^k | Power-of-Two | Bit operations for modulo |
| General/Large T | Precomputed Constants | Memory efficient, good speed |
| Variable T | Complex Numbers | Clean code, Julia optimized |

### 2. Numerical Stability

For long prediction horizons or iterative applications:

```julia
# Periodically renormalize to prevent drift
function predict_iq_stable(predictor::RotationPredictor, I0::Float64, Q0::Float64, n::Int64)
    I, Q = predict_iq(predictor, I0, Q0, n)
    
    # Renormalize to maintain radius
    radius = sqrt(I0^2 + Q0^2)
    current_radius = sqrt(I^2 + Q^2)
    
    if abs(current_radius) > 1e-10
        scale = radius / current_radius
        I *= scale
        Q *= scale
    end
    
    return I, Q
end
```

### 3. SIMD Optimization

For batch predictions, leverage Julia's SIMD capabilities:

```julia
using SIMD

function predict_iq_batch(predictor::RotationPredictor, 
                         I_array::Vector{Float64}, 
                         Q_array::Vector{Float64}, 
                         n::Int64)
    N = length(I_array)
    I_out = similar(I_array)
    Q_out = similar(Q_array)
    
    # Precompute rotation for n steps
    c, s = predictor.cos_theta, predictor.sin_theta
    for _ in 2:n
        c_new = c * predictor.cos_theta - s * predictor.sin_theta
        s_new = s * predictor.cos_theta + c * predictor.sin_theta
        c, s = c_new, s_new
    end
    
    # Vectorized rotation
    @inbounds @simd for i in 1:N
        I_out[i] = c * I_array[i] - s * Q_array[i]
        Q_out[i] = s * I_array[i] + c * Q_array[i]
    end
    
    return I_out, Q_out
end
```

## Testing and Validation

```julia
# Test harness for verification
function test_predictor(T::Int64, test_points::Int64=100)
    predictor = init_predictor(T)
    
    # Start with known point
    I0, Q0 = 10.0, 0.0
    radius = sqrt(I0^2 + Q0^2)
    
    max_error = 0.0
    
    for n in 1:test_points
        I_pred, Q_pred = predict_iq(predictor, I0, Q0, n)
        
        # Check radius preservation
        pred_radius = sqrt(I_pred^2 + Q_pred^2)
        error = abs(pred_radius - radius)
        max_error = max(max_error, error)
        
        # Verify against ground truth
        theta = 2π * n / T
        I_true = radius * cos(theta)
        Q_true = radius * sin(theta)
        
        position_error = sqrt((I_pred - I_true)^2 + (Q_pred - Q_true)^2)
        
        if position_error > 1e-10
            println("Warning: Error at n=$n: $position_error")
        end
    end
    
    println("Max radius error for T=$T: $max_error")
    return max_error < 1e-10
end

# Run tests
@assert test_predictor(4)
@assert test_predictor(100)
@assert test_predictor(256)
```

## Usage Example

```julia
# Production usage pattern
function process_iq_stream(filter_period::Int64)
    # One-time initialization
    predictor = filter_period in [2, 4, 6, 8] ? 
                nothing : init_predictor(filter_period)
    
    # Process stream
    I_current, Q_current = 10.0, 5.0
    
    for lookahead in [1, 5, 10, 25, 50]
        if filter_period in [2, 4, 6, 8]
            I_future, Q_future = predict_iq_special(I_current, Q_current, 
                                                    lookahead, filter_period)
        else
            I_future, Q_future = predict_iq(predictor, I_current, Q_current, 
                                           lookahead)
        end
        
        println("t+$lookahead: I=$I_future, Q=$Q_future")
    end
end

# Run example
process_iq_stream(100)
```

## GA-Specific Optimization Strategies

Since this is part of a Genetic Algorithm system, we can leverage evolutionary optimization to improve prediction accuracy beyond the basic geometric methods.

### Approach 6: GA-Optimized Correction Factors

The key insight is that real filters may not produce perfect circular rotation due to:
- Phase distortion
- Amplitude variations
- Nonlinearities
- Quantization effects

We can use GA to learn correction factors that account for these imperfections.

```julia
using Random
using Statistics

# Chromosome structure for GA optimization
struct CorrectionChromosome
    amplitude_corrections::Vector{Float64}  # Per-angle amplitude adjustments
    phase_corrections::Vector{Float64}      # Per-angle phase adjustments
    harmonic_weights::Vector{Float64}       # Weights for harmonic distortions
    dc_offset_i::Float64                    # DC bias correction
    dc_offset_q::Float64                    # DC bias correction
end

# Enhanced predictor with GA-learned corrections
struct GAOptimizedPredictor
    base_predictor::RotationPredictor
    corrections::CorrectionChromosome
    period::Int64
end

function predict_iq_ga_optimized(predictor::GAOptimizedPredictor, 
                                 I0::Float64, Q0::Float64, n::Int64)
    # Base geometric prediction
    I_base, Q_base = predict_iq(predictor.base_predictor, I0, Q0, n)
    
    # Apply learned corrections
    angle_index = (n % predictor.period) + 1
    
    # Amplitude correction
    amplitude = sqrt(I_base^2 + Q_base^2)
    amplitude_corrected = amplitude * predictor.corrections.amplitude_corrections[angle_index]
    
    # Phase correction
    phase = atan(Q_base, I_base)
    phase_corrected = phase + predictor.corrections.phase_corrections[angle_index]
    
    # Apply harmonic corrections (for nonlinear distortions)
    for k in 1:length(predictor.corrections.harmonic_weights)
        phase_corrected += predictor.corrections.harmonic_weights[k] * sin(k * phase)
    end
    
    # Reconstruct I/Q with corrections
    I_corrected = amplitude_corrected * cos(phase_corrected) + 
                  predictor.corrections.dc_offset_i
    Q_corrected = amplitude_corrected * sin(phase_corrected) + 
                  predictor.corrections.dc_offset_q
    
    return I_corrected, Q_corrected
end
```

### GA Training Framework

```julia
# Fitness function for GA optimization
function evaluate_fitness(chromosome::CorrectionChromosome, 
                         training_data::Vector{Tuple{Float64, Float64, Int64, Float64, Float64}},
                         base_predictor::RotationPredictor)
    total_error = 0.0
    
    for (I_start, Q_start, n_ahead, I_true, Q_true) in training_data
        # Create temporary predictor with this chromosome
        ga_predictor = GAOptimizedPredictor(base_predictor, chromosome, base_predictor.period)
        
        # Make prediction
        I_pred, Q_pred = predict_iq_ga_optimized(ga_predictor, I_start, Q_start, n_ahead)
        
        # Calculate error (MSE)
        error = (I_pred - I_true)^2 + (Q_pred - Q_true)^2
        total_error += error
    end
    
    # Return fitness (inverse of error for maximization)
    return 1.0 / (1.0 + total_error)
end

# GA evolution step
function evolve_population(population::Vector{CorrectionChromosome}, 
                          training_data, base_predictor;
                          mutation_rate=0.01, crossover_rate=0.7, elite_ratio=0.1)
    
    pop_size = length(population)
    elite_count = floor(Int, pop_size * elite_ratio)
    
    # Evaluate fitness for all chromosomes
    fitness_scores = [evaluate_fitness(chrom, training_data, base_predictor) 
                     for chrom in population]
    
    # Sort by fitness
    sorted_indices = sortperm(fitness_scores, rev=true)
    
    # Keep elite chromosomes
    new_population = population[sorted_indices[1:elite_count]]
    
    # Generate rest through crossover and mutation
    while length(new_population) < pop_size
        # Selection (tournament)
        parent1 = tournament_select(population, fitness_scores)
        parent2 = tournament_select(population, fitness_scores)
        
        # Crossover
        if rand() < crossover_rate
            child = crossover(parent1, parent2)
        else
            child = rand() < 0.5 ? parent1 : parent2
        end
        
        # Mutation
        if rand() < mutation_rate
            child = mutate(child)
        end
        
        push!(new_population, child)
    end
    
    return new_population
end

function tournament_select(population::Vector{CorrectionChromosome}, 
                          fitness_scores::Vector{Float64}, 
                          tournament_size=3)
    indices = rand(1:length(population), tournament_size)
    best_idx = indices[argmax([fitness_scores[i] for i in indices])]
    return population[best_idx]
end

function crossover(parent1::CorrectionChromosome, parent2::CorrectionChromosome)
    # Uniform crossover
    n = length(parent1.amplitude_corrections)
    
    amplitude_child = [rand() < 0.5 ? parent1.amplitude_corrections[i] : 
                      parent2.amplitude_corrections[i] for i in 1:n]
    phase_child = [rand() < 0.5 ? parent1.phase_corrections[i] : 
                  parent2.phase_corrections[i] for i in 1:n]
    
    # Blend crossover for continuous values
    α = rand()
    harmonic_child = α .* parent1.harmonic_weights .+ (1-α) .* parent2.harmonic_weights
    dc_i_child = α * parent1.dc_offset_i + (1-α) * parent2.dc_offset_i
    dc_q_child = α * parent1.dc_offset_q + (1-α) * parent2.dc_offset_q
    
    return CorrectionChromosome(amplitude_child, phase_child, harmonic_child, 
                                dc_i_child, dc_q_child)
end

function mutate(chromosome::CorrectionChromosome, σ=0.01)
    # Gaussian mutation
    n = length(chromosome.amplitude_corrections)
    
    amplitude_mutated = chromosome.amplitude_corrections .+ σ * randn(n)
    phase_mutated = chromosome.phase_corrections .+ σ * randn(n)
    harmonic_mutated = chromosome.harmonic_weights .+ σ * randn(length(chromosome.harmonic_weights))
    dc_i_mutated = chromosome.dc_offset_i + σ * randn()
    dc_q_mutated = chromosome.dc_offset_q + σ * randn()
    
    return CorrectionChromosome(amplitude_mutated, phase_mutated, harmonic_mutated,
                                dc_i_mutated, dc_q_mutated)
end
```

### Adaptive Learning Strategies

```julia
# Online adaptation using streaming data
mutable struct AdaptiveGAPredictor
    base_predictor::RotationPredictor
    corrections::CorrectionChromosome
    error_history::Vector{Float64}
    adaptation_rate::Float64
end

function adapt_online!(predictor::AdaptiveGAPredictor, 
                       I_predicted::Float64, Q_predicted::Float64,
                       I_actual::Float64, Q_actual::Float64, 
                       angle_index::Int64)
    # Calculate prediction error
    error = sqrt((I_predicted - I_actual)^2 + (Q_predicted - Q_actual)^2)
    push!(predictor.error_history, error)
    
    # Keep only recent history
    if length(predictor.error_history) > 1000
        popfirst!(predictor.error_history)
    end
    
    # Adapt if error exceeds threshold
    if error > mean(predictor.error_history) + 2*std(predictor.error_history)
        # Gradient-based local adjustment
        amplitude_error = sqrt(I_actual^2 + Q_actual^2) - sqrt(I_predicted^2 + Q_predicted^2)
        phase_error = atan(Q_actual, I_actual) - atan(Q_predicted, I_predicted)
        
        # Update corrections
        predictor.corrections.amplitude_corrections[angle_index] += 
            predictor.adaptation_rate * amplitude_error
        predictor.corrections.phase_corrections[angle_index] += 
            predictor.adaptation_rate * phase_error
    end
end
```

### Multi-Objective Optimization

For GA systems, we often want to optimize multiple objectives simultaneously:

```julia
struct MultiObjectiveFitness
    prediction_accuracy::Float64
    computation_speed::Float64
    numerical_stability::Float64
    memory_usage::Float64
end

function evaluate_pareto_fitness(chromosome::CorrectionChromosome, 
                                 test_data, base_predictor)
    # Accuracy metric
    accuracy = evaluate_fitness(chromosome, test_data, base_predictor)
    
    # Speed metric (complexity of corrections)
    nonzero_corrections = count(x -> abs(x) > 1e-6, chromosome.amplitude_corrections)
    speed = 1.0 / (1.0 + nonzero_corrections)
    
    # Stability metric (variation in corrections)
    stability = 1.0 / (1.0 + std(chromosome.amplitude_corrections) + 
                      std(chromosome.phase_corrections))
    
    # Memory metric
    memory = 1.0 / (1.0 + length(chromosome.amplitude_corrections))
    
    return MultiObjectiveFitness(accuracy, speed, stability, memory)
end

# NSGA-II style selection for multi-objective optimization
function pareto_rank(population::Vector{CorrectionChromosome}, test_data, base_predictor)
    fitness_values = [evaluate_pareto_fitness(chrom, test_data, base_predictor) 
                     for chrom in population]
    
    # Implement non-dominated sorting
    # ... (NSGA-II implementation)
    
    return fitness_values
end
```

## Recommendations

1. **For production systems**: Use Approach 1 (Precomputed Constants) as the default, with Approach 3 (Special Cases) for common filter periods.

2. **For embedded/real-time systems**: Use Approach 2 (Lookup Table) if memory permits, otherwise Approach 1.

3. **For research/prototyping**: Use Approach 4 (Complex Numbers) for clean, maintainable code.

4. **For GA-enhanced systems**: 
   - Start with base geometric predictor
   - Collect training data from actual filter outputs
   - Use GA to learn correction factors for systematic errors
   - Implement online adaptation for non-stationary systems
   - Consider multi-objective optimization for balanced performance

5. **Always validate** numerical stability for your specific filter period and prediction horizons.

6. **Profile your specific use case** - Julia's `@benchmark` macro from BenchmarkTools.jl is excellent for this.

## GA Optimization Best Practices

1. **Training Data Collection**: Gather diverse examples covering the full rotation cycle
2. **Chromosome Encoding**: Use real-valued encoding for smooth optimization landscape
3. **Population Diversity**: Maintain diversity to avoid premature convergence
4. **Hybrid Approaches**: Combine GA global search with local gradient descent
5. **Transfer Learning**: Use corrections learned from one filter as initialization for similar filters

## Contact

For questions about implementation or optimizations for specific filter periods, please reach out to the signal processing team.