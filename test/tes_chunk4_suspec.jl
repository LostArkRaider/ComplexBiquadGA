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
using Main.PredictionMetrics # Your original file imports this, but it's not used now
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

    @assert length(filter_periods) == n_filters "Periods must match filters"

    # Generate synthetic data
    t = 1:n_samples
    outputs = Matrix{ComplexF32}(undef, n_samples, n_filters)

    for i in 1:n_filters
        period = filter_periods[i]
        frequency = 2f0 * Float32(π) / period
        phase_offset = rand(Float32) * 2f0 * Float32(π)
        magnitude = 10f0 + rand(Float32) * 5f0
        
        # Simple sine wave with magnitude and phase
        outputs[:, i] = magnitude .* exp.(im .* (frequency .* t .+ phase_offset))
    end
    
    return outputs
end

"""
Generate synthetic actual future values for testing
"""
function generate_test_actual_values(n_samples::Int)::Vector{ComplexF32}
    # Generate random walk as a simple test case
    return cumsum(randn(ComplexF32, n_samples))
end

# =============================================================================
# CORE TEST SUITE
# =============================================================================

@testset "Phase Extrapolation Tests" begin
    # Test a single filter
    current_output = ComplexF32(1.0, 0.0)
    design_frequency = Float32(π / 4) # 45 degrees per tick
    n_ticks = Int32(2)

    projected = WeightedPrediction.project_filter_forward(current_output, design_frequency, n_ticks)

    @test projected ≈ ComplexF32(0.7071, 0.7071) rtol=1e-3
    @test abs(projected) ≈ 1.0f0 rtol=1e-6
    @test angle(projected) ≈ design_frequency * n_ticks rtol=1e-6

    # Test vectorized projection
    outputs = [1.0f0+0.0f0im, 0.0f0+1.0f0im]
    frequencies = [Float32(π / 2), Float32(π / 2)]
    n_ticks = Int32(1)
    
    projected_batch = WeightedPrediction.project_filters_forward(outputs, frequencies, n_ticks)
    
    @test projected_batch[1] ≈ ComplexF32(0, 1) rtol=1e-3
    @test projected_batch[2] ≈ ComplexF32(-1, 0) rtol=1e-3
    @test all(abs.(projected_batch) .≈ 1.0f0) rtol=1e-6

    # Test frequency calculation
    periods = Float32[2, 4, 8, 16]
    frequencies = WeightedPrediction.calculate_filter_frequencies(periods)
    
    @test length(frequencies) == 4
    @test frequencies[1] ≈ Float32(π) rtol=1e-6
    @test frequencies[2] ≈ Float32(π / 2) rtol=1e-6
end

# =============================================================================
# WEIGHT OPTIMIZATION WITH PHASE EXTRAPOLATION
# =============================================================================

@testset "Weight Optimization with Phase Extrapolation" begin

    n_samples = 2000
    n_filters = 10
    horizon = Int32(20)
    filter_periods = Float32[2, 4, 6, 10, 16, 26, 42, 68, 110, 180]

    # Generate test data
    filter_outputs = generate_test_filter_outputs_with_frequency(n_samples, n_filters, filter_periods)
    actual_future = generate_test_actual_values(n_samples)

    @testset "Fitness Evaluation with Projection" begin
        initial_weights = WeightedPrediction.initialize_weights_rms(filter_outputs)

        # Corrected line to capture all four return values
        fitness, mse, mae, dir_acc = WeightedPrediction.evaluate_weight_fitness(
            initial_weights,
            filter_outputs,
            actual_future,
            horizon,
            filter_periods=filter_periods
        )
        
        @test isfinite(fitness)
        @test 0.0f0 <= fitness <= 1.0f0
        @test isfinite(mse)
        @test isfinite(mae)
        @test 0.0f0 <= dir_acc <= 1.0f0
    end

    @testset "Weight Evolution with Phase Projection" begin
        population_size = 50
        n_generations = 10

        initial_weights = WeightedPrediction.initialize_weights_rms(filter_outputs)
        population = WeightedPrediction.create_weight_population(n_filters, population_size, initial_weights=initial_weights)

        # Evolve the population
        evolved_population, fitness = WeightedPrediction.evolve_weights(
            population,
            filter_outputs,
            actual_future,
            horizon,
            filter_periods=filter_periods,
            n_generations=n_generations
        )
        
        @test size(evolved_population) == (population_size, n_filters)
        @test all(sum(evolved_population, dims=2) .≈ 1.0f0) rtol=1e-5
        @test all(evolved_population .>= 0.0f0)
    end
end

# =============================================================================
# STREAMING PREDICTOR TESTS
# =============================================================================

@testset "Streaming Predictor with Phase Extrapolation" begin
    n_filters = 5
    filter_periods = Float32[2, 4, 8, 16, 32]
    horizon_range = (Int32(10), Int32(50))
    initial_weights = WeightedPrediction.initialize_weights_rms(
        [rand(ComplexF32, 100) for _ in 1:n_filters]
    )

    predictor = WeightedPrediction.create_streaming_predictor(
        n_filters,
        initial_weights,
        filter_periods,
        horizon_range,
        warmup_period = Int32(10)
    )

    @test !predictor.is_warmed_up
    @test length(WeightedPrediction.get_current_predictions(predictor)) == 0

    # Process some ticks
    for _ in 1:20
        # Create a single tick of filter outputs
        filter_outputs = [rand(ComplexF32) for _ in 1:n_filters]
        WeightedPrediction.process_tick!(predictor, filter_outputs, rand(ComplexF32))
    end
    
    @test predictor.is_warmed_up
    @test !isempty(WeightedPrediction.get_current_predictions(predictor))

    # Test the prediction report
    report = WeightedPrediction.generate_prediction_report(predictor)
    @test occursin("PREDICTION REPORT", report)
    @test occursin("Current Predictions", report)
    @test occursin("Weight Distribution", report)
end