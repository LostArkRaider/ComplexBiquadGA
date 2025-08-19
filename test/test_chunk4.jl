# tests/test_chunk4_fixed.jl - Updated tests for phase extrapolation
# Uses the merged WeightedPrediction module instead of separate modules

"""
Test Suite for Chunk 4 (REVISED) - Weight Optimization with Phase Extrapolation

Tests all components of the phase-based prediction system:
- Phase extrapolation of filter outputs
- Frequency calculation from filter periods
- RMS-based weight initialization
- Scalar weight application to projected outputs
- Vector summation for prediction
- I-component comparison
- Multi-horizon support with projection
"""

using Test
using Statistics
using Random
using LinearAlgebra

# Include the merged module
include("../src/WeightedPrediction.jl")
include("../src/PredictionMetrics.jl")
include("../src/SyntheticSignalGenerator.jl")

using .WeightedPrediction
using .PredictionMetrics
using .SyntheticSignalGenerator

# Set random seed for reproducibility
Random.seed!(42)

# =============================================================================
# TEST UTILITIES
# =============================================================================

"""
Generate synthetic filter outputs with known frequency content
"""
function generate_test_filter_outputs_with_frequency(
    n_samples::Int, 
    n_filters::Int,
    filter_periods::Vector{Float32}
)::Matrix{ComplexF32}
    
    @assert length(filter_periods) == n_filters "Period count mismatch"
    
    outputs = Matrix{ComplexF32}(undef, n_samples, n_filters)
    
    for i in 1:n_filters
        # Design frequency for this filter
        freq = 2π / filter_periods[i]
        
        # Generate complex sinusoid at design frequency
        for t in 1:n_samples
            # Consistent magnitude for testing
            magnitude = 1.0f0
            phase = freq * t
            outputs[t, i] = magnitude * exp(im * phase)
        end
    end
    
    return outputs
end

"""
Generate predictable future signal for validation
"""
function generate_predictable_future(
    n_samples::Int,
    base_frequency::Float32
)::Vector{ComplexF32}
    
    future = Vector{ComplexF32}(undef, n_samples)
    
    for t in 1:n_samples
        # Predictable sinusoid
        phase = base_frequency * t
        price_change = sin(phase)
        future[t] = ComplexF32(price_change, 1.0f0)
    end
    
    return future
end

"""
Calculate theoretical prediction using phase extrapolation
"""
function theoretical_prediction(
    filter_outputs::Vector{ComplexF32},
    filter_periods::Vector{Float32},
    weights::Vector{Float32},
    n_ticks::Int32
)::Float32
    
    projected = Vector{ComplexF32}(undef, length(filter_outputs))
    
    for i in 1:length(filter_outputs)
        magnitude = abs(filter_outputs[i])
        phase = angle(filter_outputs[i])
        frequency = 2π / filter_periods[i]
        
        # Project phase forward
        projected_phase = phase + frequency * n_ticks
        projected[i] = magnitude * exp(im * projected_phase)
    end
    
    # Weighted sum
    weighted_sum = sum(weights .* projected)
    return real(weighted_sum)
end

# =============================================================================
# PHASE EXTRAPOLATION TESTS
# =============================================================================

@testset "Phase Extrapolation Tests" begin
    
    @testset "Filter Frequency Calculation" begin
        # Test frequency calculation from periods
        periods = Float32[2.01, 4.0, 26.0, 52.0]
        frequencies = WeightedPrediction.calculate_filter_frequencies(periods)
        
        @test length(frequencies) == length(periods)
        
        # Check specific frequencies
        @test isapprox(frequencies[1], 2π / 2.01, rtol=1e-5)
        @test isapprox(frequencies[2], 2π / 4.0, rtol=1e-5)
        @test isapprox(frequencies[3], 2π / 26.0, rtol=1e-5)
        @test isapprox(frequencies[4], 2π / 52.0, rtol=1e-5)
        
        # Verify frequency ordering (higher period = lower frequency)
        @test issorted(frequencies, rev=true)
    end
    
    @testset "Phase Projection Forward" begin
        # Test single filter projection
        current_output = ComplexF32(1.0, 0.0)  # Magnitude 1, phase 0
        frequency = Float32(π/4)  # 45 degrees per tick
        n_ticks = Int32(4)
        
        projected = WeightedPrediction.project_filter_forward(
            current_output, frequency, n_ticks
        )
        
        # After 4 ticks at π/4 per tick, phase should be π (180°)
        expected_phase = π
        expected = ComplexF32(cos(expected_phase), sin(expected_phase))
        
        @test isapprox(abs(projected), 1.0, rtol=1e-5)  # Magnitude preserved
        @test isapprox(angle(projected), expected_phase, rtol=1e-3)
        @test isapprox(projected, expected, rtol=1e-3)
    end
    
    @testset "Multi-Filter Phase Extrapolation" begin
        # Create filters with different periods
        n_filters = 3
        filter_periods = Float32[4.0, 8.0, 16.0]
        filter_frequencies = WeightedPrediction.calculate_filter_frequencies(filter_periods)
        
        # Current outputs (all starting at phase 0)
        filter_outputs = [ComplexF32(1.0, 0.0) for _ in 1:n_filters]
        
        # Project forward 8 ticks
        n_ticks = Int32(8)
        weights = Float32[0.3, 0.4, 0.3]
        
        # Use the actual function
        prediction = WeightedPrediction.predict_price_change_extrapolated(
            filter_outputs, filter_frequencies, weights, n_ticks
        )
        
        # Calculate expected
        expected = theoretical_prediction(
            filter_outputs, filter_periods, weights, n_ticks
        )
        
        @test isapprox(prediction, expected, rtol=1e-4)
    end
    
    @testset "Phase Extrapolation with Complex Initial Phase" begin
        # Test with non-zero initial phases
        filter_outputs = [
            ComplexF32(cos(π/6), sin(π/6)),   # 30° initial phase
            ComplexF32(cos(π/3), sin(π/3)),   # 60° initial phase
            ComplexF32(cos(π/2), sin(π/2))    # 90° initial phase
        ]
        
        filter_periods = Float32[6.0, 12.0, 24.0]
        frequencies = WeightedPrediction.calculate_filter_frequencies(filter_periods)
        weights = Float32[0.4, 0.3, 0.3]
        n_ticks = Int32(6)
        
        prediction = WeightedPrediction.predict_price_change_extrapolated(
            filter_outputs, frequencies, weights, n_ticks
        )
        
        # Each filter should advance by different amounts
        # Filter 1: 2π/6 * 6 = 2π (full rotation)
        # Filter 2: 2π/12 * 6 = π (half rotation)
        # Filter 3: 2π/24 * 6 = π/2 (quarter rotation)
        
        @test isfinite(prediction)
        @test abs(prediction) <= sum(weights)  # Bounded by weight sum
    end
end

# =============================================================================
# WEIGHT OPTIMIZATION WITH PHASE EXTRAPOLATION TESTS
# =============================================================================

@testset "Weight Optimization with Phase Extrapolation" begin
    
    @testset "Fitness Evaluation with Projection" begin
        # Generate test data
        n_samples = 500
        n_filters = 3
        filter_periods = Float32[4.0, 8.0, 16.0]
        horizon = Int32(10)
        
        # Generate filter outputs with known frequencies
        filter_outputs = generate_test_filter_outputs_with_frequency(
            n_samples, n_filters, filter_periods
        )
        
        # Generate predictable future
        actual_future = generate_predictable_future(n_samples, 0.1f0)
        
        # Test fitness evaluation with phase extrapolation
        weights = Float32[0.3, 0.4, 0.3]
        
        fitness, mse, mae, dir_acc = WeightedPrediction.evaluate_weight_fitness(
            weights, filter_outputs, actual_future, horizon,
            filter_periods=filter_periods
        )
        
        @test 0 <= fitness <= 1
        @test mse >= 0
        @test mae >= 0
        @test 0 <= dir_acc <= 1
        
        # Fitness should be reasonable for matched frequencies
        @test fitness > 0.1  # Not terrible
    end
    
    @testset "Weight Evolution with Phase Projection" begin
        # Setup
        n_samples = 300
        n_filters = 3
        filter_periods = Float32[4.0, 8.0, 16.0]
        horizon = Int32(20)
        
        filter_outputs = generate_test_filter_outputs_with_frequency(
            n_samples, n_filters, filter_periods
        )
        actual_future = generate_predictable_future(n_samples, 0.15f0)
        
        # Create population
        population_size = 10
        population = WeightedPrediction.create_weight_population(
            n_filters, population_size
        )
        
        # Evolve with phase extrapolation
        initial_fitness = -Inf32
        
        for gen in 1:5
            population, fitness = WeightedPrediction.evolve_weights(
                population, filter_outputs, actual_future, horizon,
                filter_periods=filter_periods
            )
            
            max_fitness = maximum(fitness)
            if gen == 1
                initial_fitness = max_fitness
            end
        end
        
        final_fitness = maximum(fitness)
        
        # Should improve or maintain
        @test final_fitness >= initial_fitness
    end
end

# =============================================================================
# PREDICTION SYSTEM WITH PHASE EXTRAPOLATION TESTS
# =============================================================================

@testset "Prediction System with Phase Extrapolation" begin
    
    @testset "System Creation with Periods" begin
        n_filters = Int32(4)
        filter_periods = Float32[4.0, 8.0, 16.0, 32.0]
        initial_weights = Float32[0.25, 0.25, 0.25, 0.25]
        horizon_range = (Int32(10), Int32(100))
        
        system = WeightedPrediction.create_prediction_system(
            n_filters, initial_weights, filter_periods, horizon_range
        )
        
        @test system.n_filters == n_filters
        @test system.filter_periods == filter_periods
        @test system.weights == initial_weights
        @test system.horizon_range == horizon_range
    end
    
    @testset "System Update and Phase-Based Prediction" begin
        # Create system with known periods
        n_filters = Int32(3)
        filter_periods = Float32[4.0, 8.0, 16.0]
        weights = Float32[0.3, 0.4, 0.3]
        
        system = WeightedPrediction.create_prediction_system(
            n_filters, weights, filter_periods, (Int32(5), Int32(50))
        )
        
        # Feed data
        for t in 1:100
            # Generate filter outputs at design frequencies
            filter_outputs = Vector{ComplexF32}(undef, n_filters)
            for i in 1:n_filters
                freq = 2π / filter_periods[i]
                filter_outputs[i] = ComplexF32(cos(freq * t), sin(freq * t))
            end
            
            input_signal = ComplexF32(sin(0.1 * t), 1.0)
            
            WeightedPrediction.update_prediction!(system, filter_outputs, input_signal)
        end
        
        # Get prediction at specific horizon
        horizon = Int32(20)
        prediction = WeightedPrediction.get_prediction_at_horizon(system, horizon)
        
        @test isfinite(prediction)
        @test abs(prediction) <= 3.0  # Reasonable range
    end
    
    @testset "Streaming Predictor with Phase Extrapolation" begin
        n_filters = Int32(3)
        filter_periods = Float32[4.0, 8.0, 16.0]
        weights = Float32[0.3, 0.4, 0.3]
        
        predictor = WeightedPrediction.create_streaming_predictor(
            n_filters, weights, filter_periods,
            (Int32(10), Int32(100)),
            warmup_period = Int32(20)
        )
        
        @test !predictor.is_warmed_up
        
        # Process ticks with known pattern
        for t in 1:50
            filter_outputs = Vector{ComplexF32}(undef, n_filters)
            for i in 1:n_filters
                freq = 2π / filter_periods[i]
                filter_outputs[i] = ComplexF32(cos(freq * t), sin(freq * t))
            end
            
            input_signal = ComplexF32(sin(0.1 * t), 1.0)
            
            WeightedPrediction.process_tick!(predictor, filter_outputs, input_signal)
        end
        
        @test predictor.is_warmed_up
        @test predictor.system.current_tick == 50
        
        # Check predictions exist
        predictions = WeightedPrediction.get_current_predictions(predictor)
        @test !isempty(predictions)
        
        # All predictions should be finite
        for (horizon, pred) in predictions
            @test isfinite(pred)
        end
    end
end

# =============================================================================
# INTEGRATION TESTS WITH PHASE EXTRAPOLATION
# =============================================================================

@testset "Integration Tests with Phase Extrapolation" begin
    
    @testset "End-to-End with Synthetic Data" begin
        # Generate synthetic signal with known frequency
        signal = SyntheticSignalGenerator.generate_synthetic_signal(
            n_bars = 500,
            ticks_per_bar = 89,
            signal_type = :pure_sine,
            signal_params = Dict(:period => 13.0, :amplitude => 1.0)
        )
        
        # Simulate filter bank with matched periods
        n_filters = 3
        filter_periods = Float32[10.0, 13.0, 20.0]  # Middle matches signal
        n_samples = length(signal.signal_complex)
        
        # Generate filter outputs (simplified - would come from actual filters)
        filter_outputs = Matrix{ComplexF32}(undef, n_samples, n_filters)
        for i in 1:n_filters
            freq = 2π / filter_periods[i]
            for t in 1:n_samples
                # Filters extract their design frequency
                filter_outputs[t, i] = signal.signal_complex[t] * 
                                       exp(im * freq * t)
            end
        end
        
        # Initialize weights with RMS
        filter_outputs_vec = [filter_outputs[:, i] for i in 1:n_filters]
        initial_weights = WeightedPrediction.initialize_weights_rms(filter_outputs_vec)
        
        # Create prediction system
        system = WeightedPrediction.create_prediction_system(
            Int32(n_filters), initial_weights, filter_periods,
            (Int32(10), Int32(50))
        )
        
        # Make predictions with phase extrapolation
        horizon = Int32(26)  # Two periods of the signal
        predictions = Float32[]
        
        for t in 1:(n_samples - horizon)
            current_outputs = filter_outputs[t, :]
            frequencies = WeightedPrediction.calculate_filter_frequencies(filter_periods)
            
            pred = WeightedPrediction.predict_price_change_extrapolated(
                current_outputs, frequencies, initial_weights, horizon
            )
            push!(predictions, pred)
        end
        
        # Compare with actual future values
        actual_future = real.(signal.signal_complex[(1+horizon):end])
        predictions_aligned = predictions[1:length(actual_future)]
        
        # Calculate metrics
        mse = mean((predictions_aligned .- actual_future) .^ 2)
        correlation = cor(predictions_aligned, actual_future)
        
        @test mse < 1.0  # Should have reasonable error
        @test abs(correlation) > 0.1  # Should have some correlation
    end
    
    @testset "Phase Coherence Check" begin
        # Test that phase extrapolation maintains coherence
        n_filters = 4
        filter_periods = Float32[4.0, 8.0, 12.0, 16.0]
        
        # All filters start in phase
        filter_outputs = [ComplexF32(1.0, 0.0) for _ in 1:n_filters]
        weights = ones(Float32, n_filters) / n_filters
        
        # After period of shortest filter, it should complete full rotation
        n_ticks = Int32(4)
        frequencies = WeightedPrediction.calculate_filter_frequencies(filter_periods)
        
        # Project all filters
        projected = Vector{ComplexF32}(undef, n_filters)
        for i in 1:n_filters
            projected[i] = WeightedPrediction.project_filter_forward(
                filter_outputs[i], frequencies[i], n_ticks
            )
        end
        
        # Check phases
        phases = angle.(projected)
        
        # First filter (period 4) should return to 0 (or 2π)
        @test isapprox(abs(projected[1]), 1.0, rtol=1e-5)
        @test isapprox(cos(phases[1]), 1.0, rtol=1e-3)  # Back to real axis
        
        # Second filter (period 8) should be at π
        @test isapprox(cos(phases[2]), -1.0, rtol=1e-3)
        
        # Others should be at intermediate phases
        @test !isapprox(phases[3], 0.0, rtol=0.1)
        @test !isapprox(phases[4], 0.0, rtol=0.1)
    end
end

# =============================================================================
# PERFORMANCE TESTS WITH PHASE EXTRAPOLATION
# =============================================================================

@testset "Performance Tests with Phase Extrapolation" begin
    
    @testset "Prediction Speed with Projection" begin
        n_samples = 5000
        n_filters = 10
        filter_periods = Float32[2^i for i in 1:n_filters]
        
        filter_outputs = generate_test_filter_outputs_with_frequency(
            n_samples, n_filters, filter_periods
        )
        weights = ones(Float32, n_filters) / n_filters
        
        # Time batch prediction with phase extrapolation
        start_time = time()
        predictions = WeightedPrediction.predict_batch_extrapolated(
            filter_outputs, filter_periods, weights, Int32(100)
        )
        elapsed = time() - start_time
        
        predictions_per_second = length(predictions) / elapsed
        ms_per_prediction = elapsed * 1000 / length(predictions)
        
        println("\nPhase-based prediction performance:")
        println("  Time per prediction: $(round(ms_per_prediction, digits=3)) ms")
        println("  Throughput: $(round(Int, predictions_per_second)) predictions/second")
        
        @test ms_per_prediction < 1.0  # Should still meet <1ms target
    end
    
    @testset "Weight Optimization Speed with Projection" begin
        n_samples = 500
        n_filters = 5
        filter_periods = Float32[4.0, 8.0, 16.0, 32.0, 64.0]
        
        filter_outputs = generate_test_filter_outputs_with_frequency(
            n_samples, n_filters, filter_periods
        )
        actual_future = generate_predictable_future(n_samples, 0.1f0)
        
        weights = ones(Float32, n_filters) / n_filters
        
        # Time fitness evaluation with phase extrapolation
        start_time = time()
        fitness, mse, mae, dir_acc = WeightedPrediction.evaluate_weight_fitness(
            weights, filter_outputs, actual_future, Int32(50),
            filter_periods=filter_periods
        )
        elapsed = time() - start_time
        
        println("\nFitness evaluation with phase extrapolation:")
        println("  Evaluation time: $(round(elapsed * 1000, digits=2)) ms")
        println("  Overhead vs direct: ~$(round(elapsed * 1000 / n_filters, digits=2)) ms/filter")
        
        @test elapsed < 0.02  # Should be under 20ms even with projection
    end
end

# =============================================================================
# VALIDATION TESTS
# =============================================================================

@testset "Mathematical Validation" begin
    
    @testset "Magnitude Preservation" begin
        # Verify magnitude stays constant during projection
        for magnitude in [0.5f0, 1.0f0, 2.0f0]
            for phase in [0f0, π/4, π/2, π]
                output = magnitude * exp(im * phase)
                
                for n_ticks in [1, 10, 100]
                    projected = WeightedPrediction.project_filter_forward(
                        output, 0.1f0, Int32(n_ticks)
                    )
                    
                    @test isapprox(abs(projected), magnitude, rtol=1e-5)
                end
            end
        end
    end
    
    @testset "Linear Phase Advancement" begin
        # Verify phase advances linearly with tick count
        output = ComplexF32(1.0, 0.0)
        frequency = Float32(π/10)  # π/10 radians per tick
        
        phases = Float32[]
        for n in 0:20
            projected = WeightedPrediction.project_filter_forward(
                output, frequency, Int32(n)
            )
            push!(phases, angle(projected))
        end
        
        # Check linearity
        phase_diffs = diff(phases)
        @test all(isapprox.(phase_diffs, frequency, rtol=1e-3))
    end
    
    @testset "Frequency Independence" begin
        # Verify filters with different frequencies project independently
        n_filters = 3
        outputs = [ComplexF32(1.0, 0.0) for _ in 1:n_filters]
        frequencies = Float32[0.1, 0.2, 0.3]
        n_ticks = Int32(10)
        
        projected = Vector{ComplexF32}(undef, n_filters)
        for i in 1:n_filters
            projected[i] = WeightedPrediction.project_filter_forward(
                outputs[i], frequencies[i], n_ticks
            )
        end
        
        # Each should have advanced by frequency * n_ticks
        for i in 1:n_filters
            expected_phase = frequencies[i] * n_ticks
            @test isapprox(angle(projected[i]), expected_phase, rtol=1e-3)
        end
    end
end

# Run all tests
println("\n" * "="^60)
println("CHUNK 4 PHASE EXTRAPOLATION TEST SUMMARY")
println("="^60)
println("All phase-based prediction tests completed successfully! ✅")
println("="^60)