# run_chunk4_tests.jl - Test runner for Chunk 4 with proper module loading
# Run from project root directory

println("\n" * "="^70)
println("GA OPTIMIZATION SYSTEM - CHUNK 4 TEST RUNNER")
println("="^70)
println("Testing Weight Optimization with Phase Extrapolation")
println("="^70 * "\n")

# ============================================================================
# STEP 1: LOAD ALL MODULES
# ============================================================================

println("Step 1: Loading all modules...")
include("load_all.jl")  # Use the fixed load_all.jl

# Verify critical modules loaded
required_modules = [
    :WeightedPrediction,
    :PredictionMetrics,
    :SyntheticSignalGenerator,
    :ProductionFilterBank,
    :ModernConfigSystem
]

missing_modules = String[]
for mod in required_modules
    if !isdefined(Main, mod)
        push!(missing_modules, string(mod))
    end
end

if !isempty(missing_modules)
    error("Missing required modules: $(join(missing_modules, ", "))")
end

println("✅ All required modules loaded successfully\n")

# ============================================================================
# STEP 2: ADD SIGNAL GENERATOR PATCHES
# ============================================================================

println("Step 2: Patching SyntheticSignalGenerator with Complex I/Q support...")

# Add the Complex I/Q generation functions
include("SyntheticSignalGenerator_patch.jl")

println("✅ Signal generator patched\n")

# ============================================================================
# STEP 3: RUN BASIC FUNCTIONALITY TESTS
# ============================================================================

println("Step 3: Running basic functionality tests...")
println("-"^50)

using Test
using Statistics
using Random
using LinearAlgebra

# Set random seed for reproducibility
Random.seed!(42)

@testset "Basic Functionality Tests" begin
    
    @testset "Module Loading" begin
        @test isdefined(Main, :WeightedPrediction)
        @test isdefined(Main.WeightedPrediction, :WeightSet)
        @test isdefined(Main.WeightedPrediction, :PredictionSystem)
        @test isdefined(Main.WeightedPrediction, :predict_price_change_extrapolated)
    end
    
    @testset "Frequency Calculation" begin
        periods = Float32[2.01, 4.0, 26.0, 52.0]
        frequencies = Main.WeightedPrediction.calculate_filter_frequencies(periods)
        
        @test length(frequencies) == length(periods)
        @test isapprox(frequencies[1], 2π / 2.01, rtol=1e-5)
        @test isapprox(frequencies[2], 2π / 4.0, rtol=1e-5)
        @test issorted(frequencies, rev=true)  # Higher period = lower frequency
    end
    
    @testset "Phase Projection" begin
        # Test single filter projection
        current_output = ComplexF32(1.0, 0.0)  # Magnitude 1, phase 0
        frequency = Float32(π/4)  # 45 degrees per tick
        n_ticks = Int32(4)
        
        projected = Main.WeightedPrediction.project_filter_forward(
            current_output, frequency, n_ticks
        )
        
        # After 4 ticks at π/4 per tick, phase should be π (180°)
        @test isapprox(abs(projected), 1.0, rtol=1e-5)  # Magnitude preserved
        @test isapprox(angle(projected), π, rtol=1e-3)
    end
    
    @testset "Complex I/Q Signal Generation" begin
        # Test signal generation with 4-phase rotation
        n_ticks = 100
        signal = generate_complex_iq_signal(
            n_ticks = n_ticks,
            signal_type = :pure_sine,
            period = 26.0f0,
            amplitude = 50.0f0,
            normalization_scale = 50.0f0
        )
        
        @test length(signal) == n_ticks
        @test all(abs.(real.(signal)) .<= 1.0)  # Normalized to [-1, 1]
        
        # Verify 4-phase rotation pattern
        for i in 1:min(10, n_ticks)
            pos = phase_pos_global(Int64(i))
            if abs(signal[i]) > 1e-6
                phase = angle(signal[i])
                expected_phases = [0, π/2, π, -π/2]
                # Check if phase aligns with expected quadrant
                @test any(abs(phase - exp_phase) < 0.2 || 
                         abs(phase - exp_phase - 2π) < 0.2 || 
                         abs(phase - exp_phase + 2π) < 0.2 
                         for exp_phase in expected_phases)
            end
        end
    end
end

println("\n✅ Basic functionality tests passed\n")

# ============================================================================
# STEP 4: RUN PERFORMANCE TESTS
# ============================================================================

println("Step 4: Running performance tests...")
println("-"^50)

@testset "Performance Tests" begin
    
    @testset "Prediction Speed" begin
        n_samples = 5000
        n_filters = 10
        filter_periods = Float32[2^i for i in 1:n_filters]
        
        # Generate test data
        filter_outputs = Matrix{ComplexF32}(undef, n_samples, n_filters)
        for i in 1:n_samples, j in 1:n_filters
            filter_outputs[i, j] = ComplexF32(randn(), randn()) * 0.1f0
        end
        
        weights = ones(Float32, n_filters) / n_filters
        
        # Time batch prediction
        start_time = time()
        predictions = Main.WeightedPrediction.predict_batch_extrapolated(
            filter_outputs, filter_periods, weights, Int32(100)
        )
        elapsed = time() - start_time
        
        predictions_per_second = length(predictions) / elapsed
        ms_per_prediction = elapsed * 1000 / length(predictions)
        
        println("\n  Phase-based prediction performance:")
        println("    Time per prediction: $(round(ms_per_prediction, digits=3)) ms")
        println("    Throughput: $(round(Int, predictions_per_second)) predictions/second")
        
        @test ms_per_prediction < 1.0  # Should meet <1ms target
    end
    
    @testset "Weight Optimization Speed" begin
        n_samples = 500
        n_filters = 5
        filter_periods = Float32[4.0, 8.0, 16.0, 32.0, 64.0]
        
        # Generate test data
        filter_outputs = Matrix{ComplexF32}(undef, n_samples, n_filters)
        for i in 1:n_samples, j in 1:n_filters
            freq = 2π / filter_periods[j]
            filter_outputs[i, j] = ComplexF32(sin(freq * i), cos(freq * i)) * 0.1f0
        end
        
        actual_future = [ComplexF32(sin(0.1 * i), 0.1) for i in 1:n_samples]
        weights = ones(Float32, n_filters) / n_filters
        
        # Time fitness evaluation
        start_time = time()
        fitness, mse, mae, dir_acc = Main.WeightedPrediction.evaluate_weight_fitness(
            weights, filter_outputs, actual_future, Int32(50),
            filter_periods=filter_periods
        )
        elapsed = time() - start_time
        
        println("\n  Fitness evaluation with phase extrapolation:")
        println("    Evaluation time: $(round(elapsed * 1000, digits=2)) ms")
        println("    Fitness: $(round(fitness, digits=4))")
        println("    MSE: $(round(mse, digits=6))")
        println("    Directional accuracy: $(round(dir_acc * 100, digits=1))%")
        
        @test elapsed < 0.02  # Should be under 20ms
    end
end

println("\n✅ Performance tests passed\n")

# ============================================================================
# STEP 5: RUN INTEGRATION TESTS
# ============================================================================

println("Step 5: Running integration tests...")
println("-"^50)

@testset "Integration Tests" begin
    
    @testset "End-to-End Prediction System" begin
        # Create a simple prediction system
        n_filters = Int32(3)
        filter_periods = Float32[4.0, 8.0, 16.0]
        initial_weights = Float32[0.3, 0.4, 0.3]
        
        system = Main.WeightedPrediction.create_prediction_system(
            n_filters, initial_weights, filter_periods,
            (Int32(10), Int32(100))
        )
        
        @test system.n_filters == n_filters
        @test system.filter_periods == filter_periods
        @test system.weights == initial_weights
        
        # Feed some test data
        for t in 1:200
            # Generate filter outputs with known frequencies
            filter_outputs = Vector{ComplexF32}(undef, n_filters)
            for i in 1:n_filters
                freq = 2π / filter_periods[i]
                filter_outputs[i] = ComplexF32(sin(freq * t), cos(freq * t)) * 0.1f0
            end
            
            # Complex I/Q input signal
            input_signal = apply_quad_phase(Float32(sin(0.1 * t)), phase_pos_global(Int64(t)))
            
            Main.WeightedPrediction.update_prediction!(system, filter_outputs, input_signal)
        end
        
        # Get prediction at specific horizon
        horizon = Int32(20)
        prediction = Main.WeightedPrediction.get_prediction_at_horizon(system, horizon)
        
        @test isfinite(prediction)
        @test abs(prediction) <= 3.0  # Reasonable range
        
        # Evaluate performance
        mse, mae, dir_acc = Main.WeightedPrediction.evaluate_predictions(system, horizon)
        
        println("\n  Prediction system evaluation:")
        println("    MSE: $(round(mse, digits=6))")
        println("    MAE: $(round(mae, digits=6))")
        println("    Directional accuracy: $(round(dir_acc * 100, digits=1))%")
        
        @test mse >= 0
        @test mae >= 0
        @test 0 <= dir_acc <= 1
    end
    
    @testset "Streaming Predictor" begin
        n_filters = Int32(3)
        filter_periods = Float32[4.0, 8.0, 16.0]
        weights = Float32[0.3, 0.4, 0.3]
        
        predictor = Main.WeightedPrediction.create_streaming_predictor(
            n_filters, weights, filter_periods,
            (Int32(10), Int32(100)),
            warmup_period = Int32(20)
        )
        
        @test !predictor.is_warmed_up
        
        # Process ticks
        for t in 1:50
            filter_outputs = Vector{ComplexF32}(undef, n_filters)
            for i in 1:n_filters
                freq = 2π / filter_periods[i]
                filter_outputs[i] = ComplexF32(sin(freq * t), cos(freq * t)) * 0.1f0
            end
            
            input_signal = apply_quad_phase(Float32(sin(0.1 * t)), phase_pos_global(Int64(t)))
            
            Main.WeightedPrediction.process_tick!(predictor, filter_outputs, input_signal)
        end
        
        @test predictor.is_warmed_up
        @test predictor.system.current_tick == 50
        
        # Check predictions exist
        predictions = Main.WeightedPrediction.get_current_predictions(predictor)
        @test !isempty(predictions)
        
        # All predictions should be finite
        for (horizon, pred) in predictions
            @test isfinite(pred)
        end
        
        # Generate report
        report = Main.WeightedPrediction.generate_prediction_report(predictor)
        println("\n  Streaming predictor report:")
        println("  " * replace(report, "\n" => "\n  "))
    end
end

println("\n✅ Integration tests passed\n")

# ============================================================================
# SUMMARY
# ============================================================================

println("="^70)
println("CHUNK 4 TEST SUMMARY")
println("="^70)
println("✅ All tests passed successfully!")
println("")
println("Key achievements:")
println("  • Module loading and dependencies resolved")
println("  • Phase extrapolation working correctly")
println("  • Complex I/Q signal generation with 4-phase rotation")
println("  • Performance targets met (<1ms per prediction)")
println("  • Integration with prediction system successful")
println("")
println("Ready for Chunk 5: System Integration")
println("="^70)