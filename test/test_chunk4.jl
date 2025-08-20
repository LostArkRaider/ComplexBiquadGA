# test/test_chunk4.jl - Updated to follow Julia best practices

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

# =============================================================================
# SETUP AND IMPORTS
#
# Assumes that load_all.jl has been run and all modules are available in Main.
# =============================================================================

using Test # Required for the test suite
using Random
using Main.WeightedPrediction
using Main.PredictionMetrics
using Main.SyntheticSignalGenerator
using Main.GATypes # For PredictionSystem and other types

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
        periods = Float32[2.01, 4.0, 26.0, 52.0]
        frequencies = WeightedPrediction.calculate_filter_frequencies(periods)

        @test length(frequencies) == length(periods)

        @test isapprox(frequencies[1], 2π / 2.01, rtol=1e-5)
        @test isapprox(frequencies[2], 2π / 4.0, rtol=1e-5)
        @test isapprox(frequencies[3], 2π / 26.0, rtol=1e-5)
        @test isapprox(frequencies[4], 2π / 52.0, rtol=1e-5)

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

        # After 4 ticks at π/4 per tick, phase should be 180°
        # We test the real and imaginary parts directly to avoid floating point issues
        # with the angle() function's branch cut at -π.
        expected_real = Float32(-1.0)
        expected_imag = Float32(0.0)

        @test isapprox(real(projected), expected_real, atol=1e-5)
        @test isapprox(imag(projected), expected_imag, atol=1e-5)

        # Also confirm magnitude is preserved
        @test isapprox(abs(projected), 1.0f0, atol=1e-5)
    end

    @testset "Multi-Filter Phase Extrapolation" begin
        n_filters = 3
        filter_periods = Float32[4.0, 8.0, 16.0]
        filter_frequencies = WeightedPrediction.calculate_filter_frequencies(filter_periods)

        filter_outputs = [ComplexF32(1.0, 0.0) for _ in 1:n_filters]

        n_ticks = Int32(8)
        weights = Float32[0.3, 0.4, 0.3]

        prediction = WeightedPrediction.predict_price_change_extrapolated(
            filter_outputs, filter_frequencies, weights, n_ticks
        )

        expected = theoretical_prediction(
            filter_outputs, filter_periods, weights, n_ticks
        )

        @test isapprox(prediction, expected, rtol=1e-4)
    end

    @testset "Phase Extrapolation with Complex Initial Phase" begin
        filter_outputs = [
            ComplexF32(cos(π/6), sin(π/6)),
            ComplexF32(cos(π/3), sin(π/3)),
            ComplexF32(cos(π/2), sin(π/2))
        ]

        filter_periods = Float32[6.0, 12.0, 24.0]
        frequencies = WeightedPrediction.calculate_filter_frequencies(filter_periods)
        weights = Float32[0.4, 0.3, 0.3]
        n_ticks = Int32(6)

        prediction = WeightedPrediction.predict_price_change_extrapolated(
            filter_outputs, frequencies, weights, n_ticks
        )

        @test isfinite(prediction)
        @test abs(prediction) <= sum(weights)
    end
end

# =============================================================================
# WEIGHT OPTIMIZATION WITH PHASE EXTRAPOLATION TESTS
# =============================================================================

@testset "Weight Optimization with Phase Extrapolation" begin

    @testset "Fitness Evaluation with Projection" begin
        n_samples = 500
        n_filters = 3
        filter_periods = Float32[4.0, 8.0, 16.0]
        horizon = Int32(10)

        filter_outputs = generate_test_filter_outputs_with_frequency(
            n_samples, n_filters, filter_periods
        )

        actual_future = generate_predictable_future(n_samples, 0.1f0)

        weights = Float32[0.3, 0.4, 0.3]

        fitness, mse, mae, dir_acc = WeightedPrediction.evaluate_weight_fitness(
            weights, filter_outputs, actual_future, horizon, filter_periods=filter_periods
        )

        @test 0 <= fitness <= 1
        @test mse >= 0
        @test mae >= 0
        @test 0 <= dir_acc <= 1
        @test fitness > 0.1
    end

    @testset "Weight Evolution with Phase Projection" begin
        n_samples = 300
        n_filters = 3
        filter_periods = Float32[4.0, 8.0, 16.0]
        horizon = Int32(20)
        filter_outputs = generate_test_filter_outputs_with_frequency(
            n_samples, n_filters, filter_periods
        )
        actual_future = generate_predictable_future(n_samples, 0.15f0)

        population_size = 10
        population = WeightedPrediction.create_weight_population(
            n_filters, population_size
        )

        initial_fitness = -Inf32
        for gen in 1:5
            population, fitness = WeightedPrediction.evolve_weights(
                population, filter_outputs, actual_future, horizon, filter_periods=filter_periods
            )
            max_fitness = maximum(fitness)
            if gen == 1
                initial_fitness = max_fitness
            end
        end
        final_fitness = maximum(fitness)

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
        n_filters = Int32(3)
        filter_periods = Float32[4.0, 8.0, 16.0]
        weights = Float32[0.3, 0.4, 0.3]
        system = WeightedPrediction.create_prediction_system(
            n_filters, weights, filter_periods, (Int32(5), Int32(50))
        )

        for t in 1:100
            filter_outputs = Vector{ComplexF32}(undef, n_filters)
            for i in 1:n_filters
                freq = 2π / filter_periods[i]
                phase = freq * t
                filter_outputs[i] = 1.0f0 * exp(im * phase)
            end

            pred = WeightedPrediction.update_and_predict(system, filter_outputs)

            @test isfinite(pred.current_prediction)
        end
    end
end

# =============================================================================
# ADDITIONAL TESTS
# =============================================================================

@testset "RMS-based Weight Initialization" begin
    n_filters = 4
    n_samples = 1000
    filter_outputs = rand(ComplexF32, n_samples, n_filters)

    weights = WeightedPrediction.initialize_weights_rms(filter_outputs)

    @test length(weights) == n_filters
    @test sum(weights) ≈ 1.0f0 atol=1e-5
    @test all(w -> w >= 0, weights)
end

@testset "Scalar Weight Application" begin
    outputs = [1.0+0.5im, 2.0-1.0im]
    weights = [0.4, 0.6]

    result = WeightedPrediction.apply_scalar_weights(outputs, weights)

    @test result ≈ ComplexF32(1.6, -0.4)
end

@testset "Weight Mutation and Crossover" begin
    parent = Float32[0.2, 0.3, 0.5]
    mutant = WeightedPrediction.mutate_weights(parent)
    @test length(mutant) == 3
    @test sum(mutant) ≈ 1.0f0 atol=1e-5
    @test all(w -> w >= 0, mutant)

    parent1 = Float32[0.1, 0.2, 0.7]
    parent2 = Float32[0.6, 0.3, 0.1]
    child1, child2 = WeightedPrediction.crossover_weights(parent1, parent2)
    @test length(child1) == 3
    @test length(child2) == 3
    @test sum(child1) ≈ 1.0f0 atol=1e-5
    @test sum(child2) ≈ 1.0f0 atol=1e-5
    @test all(w -> w >= 0, child1)
    @test all(w -> w >= 0, child2)
end