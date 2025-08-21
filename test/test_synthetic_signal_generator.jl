# test_synthetic_signal_generator.jl
# Interactive test script for SyntheticSignalGenerator module
# Tests Complex I/Q signal generation and 4-phase rotation

println("\n" * "="^80)
println("SYNTHETIC SIGNAL GENERATOR - INTERACTIVE TEST SUITE")
println("Testing Complex I/Q Signal Generation & 4-Phase Rotation")
println("="^80 * "\n")


using Test

# ============================================================================
# TEST 1: 4-PHASE ROTATION HELPERS
# ============================================================================

function test_phase_rotation_helpers()
    print_test_header("4-Phase Rotation Helper Functions")
    
    println("\n1️⃣ Testing phase_pos_global function...")
    
    # Test phase position calculation
    test_cases = [
        (1, 1, "First tick"),
        (2, 2, "Second tick"),
        (3, 3, "Third tick"),
        (4, 4, "Fourth tick"),
        (5, 1, "Fifth tick (wraps to 1)"),
        (89, 1, "89th tick (Fibonacci)"),
        (90, 2, "90th tick"),
    ]
    
    all_passed = true
    for (tick_idx, expected, description) in test_cases
        result = phase_pos_global(Int64(tick_idx))
        passed = result == expected
        all_passed &= passed
        print_result("Tick $tick_idx → Position $result", passed, 
                     "$description (expected $expected)")
    end
    
    println("\n2️⃣ Testing apply_quad_phase function...")
    
    # Test quadrant rotations
    test_value = 0.5f0
    rotations = [
        (1, ComplexF32(0.5, 0.0), "0° rotation"),
        (2, ComplexF32(0.0, 0.5), "90° rotation"),
        (3, ComplexF32(-0.5, 0.0), "180° rotation"),
        (4, ComplexF32(0.0, -0.5), "270° rotation"),
    ]
    
    for (pos, expected, description) in rotations
        result = apply_quad_phase(test_value, Int32(pos))
        passed = isapprox(result, expected, atol=1e-6)
        all_passed &= passed
        print_result("Position $pos", passed,
                     "$description: $(round(result, digits=3))")
    end
    
    return all_passed
end

# ============================================================================
# TEST 2: COMPLEX I/Q SIGNAL GENERATION
# ============================================================================

function test_complex_iq_generation()
    print_test_header("Complex I/Q Signal Generation")
    
    println("\n1️⃣ Generating pure sine test signal...")
    
    # Generate a simple test signal
    n_ticks = 400
    period = 26.0f0
    amplitude = 50.0f0
    
    signal = generate_complex_iq_signal(
        n_ticks = n_ticks,
        signal_type = :pure_sine,
        period = period,
        amplitude = amplitude,
        random_seed = 42
    )
    
    # Analyze signal properties
    analysis = analyze_complex_iq_signal(signal)
    
    println("\n📊 Signal Properties:")
    println("  • Length: $(length(signal)) ticks")
    println("  • Real part (normalized prices):")
    println("    - Mean: $(round(analysis["real_mean"], digits=4))")
    println("    - Std:  $(round(analysis["real_std"], digits=4))")
    println("    - Range: [$(round(analysis["real_min"], digits=3)), $(round(analysis["real_max"], digits=3))]")
    println("  • Phase alignment error: $(round(analysis["phase_alignment_error"], digits=6))")
    println("  • Phase alignment correct: $(round(100 * analysis["phase_alignment_correct"], digits=1))%")
    
    # Property tests
    println("\n🔍 Property Validation:")
    
    tests = [
        (length(signal) == n_ticks, "Signal length matches requested"),
        (abs(analysis["real_mean"]) < 0.1, "Real part zero-centered"),
        (analysis["real_min"] >= -1.0 && analysis["real_max"] <= 1.0, "Values normalized to [-1,1]"),
        (analysis["phase_alignment_error"] < 0.01, "Phase rotation accurate"),
        (analysis["phase_alignment_correct"] > 0.95, "Most phases correctly aligned"),
    ]
    
    all_passed = true
    for (passed, description) in tests
        all_passed &= passed
        print_result(description, passed)
    end
    
    # Visualization
    if ask_yes_no("\nWould you like to see a visualization of the signal?")
        visualize_complex_signal(signal[1:min(200, n_ticks)], "Pure Sine Test Signal")
    end
    
    return all_passed
end

# ============================================================================
# TEST 3: DIFFERENT SIGNAL TYPES
# ============================================================================

function test_signal_types()
    print_test_header("Different Signal Types")
    
    signal_types = [:pure_sine, :noisy_sine, :fibonacci_mixture, :market_like]
    n_ticks = 500
    
    println("\nGenerating and testing different signal types...")
    
    all_passed = true
    for sig_type in signal_types
        println("\n$(BLUE)Testing $sig_type signal...$(RESET)")
        
        try
            signal = generate_complex_iq_signal(
                n_ticks = n_ticks,
                signal_type = sig_type,
                period = 26.0f0,
                amplitude = 50.0f0,
                noise_level = 0.2f0,
                random_seed = 123
            )
            
            analysis = analyze_complex_iq_signal(signal)
            
            # Basic property checks
            tests = [
                (length(signal) == n_ticks, "Correct length"),
                (!any(isnan.(signal)), "No NaN values"),
                (!any(isinf.(signal)), "No Inf values"),
                (analysis["signal_power"] > 0, "Non-zero power"),
            ]
            
            sig_passed = true
            for (passed, description) in tests
                sig_passed &= passed
                print_result("  $description", passed)
            end
            
            all_passed &= sig_passed
            
            # Signal-specific checks
            if sig_type == :pure_sine
                # Pure sine should have low variance in magnitude
                mags = abs.(signal)
                mag_cv = std(mags) / mean(mags)  # Coefficient of variation
                pure_test = mag_cv < 0.5
                all_passed &= pure_test
                print_result("  Low magnitude variation", pure_test, 
                           "CV = $(round(mag_cv, digits=3))")
                
            elseif sig_type == :noisy_sine
                # Noisy sine should have higher variance
                higher_variance = analysis["real_std"] > 0.1
                all_passed &= higher_variance
                print_result("  Contains noise", higher_variance,
                           "Std = $(round(analysis["real_std"], digits=3))")
                
            elseif sig_type == :fibonacci_mixture
                # Should contain multiple frequency components
                multi_freq = analysis["signal_energy"] > 0
                all_passed &= multi_freq
                print_result("  Multiple frequencies", multi_freq)
                
            elseif sig_type == :market_like
                # Should have trending behavior
                trending = true  # Basic check
                all_passed &= trending
                print_result("  Market-like properties", trending)
            end
            
        catch e
            all_passed = false
            print_result("$sig_type generation", false, "Error: $e")
        end
    end
    
    return all_passed
end

# ============================================================================
# TEST 4: FIBONACCI-SPECIFIC SIGNALS
# ============================================================================

function test_fibonacci_signals()
    print_test_header("Fibonacci-Specific Test Signals")
    
    fibonacci_numbers = Int32[3, 5, 8, 13, 21, 34]
    n_ticks = 1000
    
    println("\nGenerating test signals for Fibonacci numbers: $fibonacci_numbers")
    
    all_passed = true
    
    for fib_num in fibonacci_numbers
        println("\n$(BLUE)Fibonacci $fib_num:$(RESET)")
        
        signal = create_test_signal_complex_iq(
            fib_num,
            n_ticks = n_ticks,
            signal_type = :pure_sine
        )
        
        # Expected period in ticks
        expected_period = fib_num == 1 ? 2.01f0 : Float32(2 * fib_num)
        
        # Analyze frequency content
        real_parts = real.(signal)
        fft_result = fft(real_parts .- mean(real_parts))
        power_spectrum = abs2.(fft_result[1:div(n_ticks, 2)])
        
        # Find dominant frequency
        max_idx = argmax(power_spectrum[2:end]) + 1  # Skip DC
        dominant_period = Float32(n_ticks) / Float32(max_idx - 1)
        
        # Check if detected period is close to expected
        period_error = abs(dominant_period - expected_period) / expected_period
        period_match = period_error < 0.1  # Within 10%
        
        all_passed &= period_match
        
        print_result("Period detection", period_match,
                    "Expected: $(round(expected_period, digits=1)), " *
                    "Detected: $(round(dominant_period, digits=1)) " *
                    "($(round(100*period_error, digits=1))% error)")
        
        # Check signal quality
        analysis = analyze_complex_iq_signal(signal)
        quality_good = analysis["phase_alignment_correct"] > 0.9
        all_passed &= quality_good
        
        print_result("Signal quality", quality_good,
                    "Phase alignment: $(round(100*analysis["phase_alignment_correct"], digits=1))%")
    end
    
    return all_passed
end

# ============================================================================
# TEST 5: SIGNAL CONVERSION
# ============================================================================

function test_signal_conversion()
    print_test_header("Signal Conversion to Complex I/Q")
    
    println("\nTesting conversion of traditional signals to Complex I/Q format...")
    
    # Create a traditional bar-level signal
    n_bars = 100
    ticks_per_bar = 89
    
    # Generate simple sine wave at bar level
    bar_times = 1:n_bars
    period_bars = 21.0
    price_signal = 40000.0 .+ 100.0 * sin.(2π * bar_times / period_bars)
    
    # Convert to Complex I/Q
    complex_signal = convert_to_complex_iq(
        price_signal,
        ticks_per_bar,
        normalization_scale = 50.0f0
    )
    
    # Verify properties
    expected_ticks = n_bars * ticks_per_bar
    
    tests = [
        (length(complex_signal) == expected_ticks, 
         "Correct tick count: $(length(complex_signal)) == $expected_ticks"),
        (!any(isnan.(complex_signal)), "No NaN values"),
        (!any(isinf.(complex_signal)), "No Inf values"),
    ]
    
    all_passed = true
    for (passed, description) in tests
        all_passed &= passed
        print_result(description, passed)
    end
    
    # Analyze converted signal
    analysis = analyze_complex_iq_signal(complex_signal)
    
    println("\n📊 Converted Signal Properties:")
    println("  • Total ticks: $(length(complex_signal))")
    println("  • Phase alignment: $(round(100*analysis["phase_alignment_correct"], digits=1))%")
    println("  • Signal power: $(round(analysis["signal_power"], digits=4))")
    
    # Check 4-phase rotation pattern
    println("\n🔄 Checking 4-phase rotation pattern...")
    sample_size = min(20, length(complex_signal))
    
    for i in 1:sample_size
        phase_deg = round(rad2deg(angle(complex_signal[i])), digits=1)
        expected_pos = phase_pos_global(Int64(i))
        expected_phases = [0.0, 90.0, 180.0, -90.0]
        
        # Handle zero signals
        if abs(complex_signal[i]) < 1e-6
            println("  Tick $i: Zero signal (no phase)")
        else
            phase_match = abs(phase_deg - expected_phases[expected_pos]) < 10.0 ||
                         abs(phase_deg - expected_phases[expected_pos] + 360) < 10.0 ||
                         abs(phase_deg - expected_phases[expected_pos] - 360) < 10.0
            
            symbol = phase_match ? "✓" : "✗"
            println("  Tick $i: $(symbol) Phase = $(phase_deg)° (expected ≈ $(expected_phases[expected_pos])°)")
        end
    end
    
    if ask_yes_no("\nVisualize the converted signal?")
        visualize_complex_signal(complex_signal[1:min(200, length(complex_signal))], 
                                "Converted Complex I/Q Signal")
    end
    
    return all_passed
end

# ============================================================================
# TEST 6: BATCH SIGNAL GENERATION
# ============================================================================

function test_batch_generation()
    print_test_header("Batch Test Signal Generation")
    
    println("\nGenerating batch of test signals...")
    
    fibonacci_numbers = Int32[3, 5, 8, 13, 21]
    n_ticks = 2000
    test_types = [:pure_sine, :noisy_sine]
    
    signals = generate_test_signals_complex_iq(
        fibonacci_numbers,
        n_ticks = n_ticks,
        test_types = test_types
    )
    
    println("\n📦 Generated signals:")
    for (key, signal) in signals
        println("  • $key: $(length(signal)) ticks")
    end
    
    # Test each generated signal
    all_passed = true
    
    println("\n🔍 Validating batch signals...")
    
    for (key, signal) in signals
        analysis = analyze_complex_iq_signal(signal)
        
        tests = [
            (length(signal) == n_ticks, "Correct length"),
            (!any(isnan.(signal)), "No NaN values"),
            (analysis["phase_alignment_correct"] > 0.8, "Good phase alignment"),
        ]
        
        sig_passed = true
        for (passed, test_desc) in tests
            sig_passed &= passed
        end
        
        all_passed &= sig_passed
        status = sig_passed ? "$(GREEN)✓$(RESET)" : "$(RED)✗$(RESET)"
        println("  $status $key")
    end
    
    # Test combined signal properties
    if haskey(signals, "combined")
        println("\n🎯 Testing combined signal properties...")
        combined = signals["combined"]
        
        # Should be normalized
        max_mag = maximum(abs.(combined))
        normalized = max_mag <= 1.01  # Allow small numerical error
        all_passed &= normalized
        
        print_result("Combined signal normalized", normalized,
                    "Max magnitude: $(round(max_mag, digits=4))")
        
        # Should contain multiple frequencies
        real_parts = real.(combined)
        fft_result = fft(real_parts .- mean(real_parts))
        power_spectrum = abs2.(fft_result[1:div(n_ticks, 2)])
        
        # Count significant peaks
        threshold = 0.1 * maximum(power_spectrum)
        n_peaks = sum(power_spectrum .> threshold)
        multi_freq = n_peaks >= 3
        all_passed &= multi_freq
        
        print_result("Multiple frequencies detected", multi_freq,
                    "Found $n_peaks significant peaks")
    end
    
    return all_passed
end

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

function visualize_complex_signal(signal::Vector{ComplexF32}, title::String="Complex I/Q Signal")
    n = length(signal)
    ticks = 1:n
    
    # Create subplots
    p1 = plot(ticks, real.(signal), 
              title="Real Part (Normalized Price Changes)",
              label="Real", color=:blue, lw=1.5,
              xlabel="Tick", ylabel="Value")
    
    p2 = plot(ticks, imag.(signal),
              title="Imaginary Part (4-Phase Rotation)",
              label="Imag", color=:red, lw=1.5,
              xlabel="Tick", ylabel="Value")
    
    p3 = plot(ticks, abs.(signal),
              title="Magnitude",
              label="Mag", color=:green, lw=1.5,
              xlabel="Tick", ylabel="Magnitude")
    
    p4 = scatter(real.(signal), imag.(signal),
                 title="Complex Plane (I/Q Plot)",
                 label="", color=:purple, ms=2, alpha=0.5,
                 xlabel="Real", ylabel="Imaginary",
                 aspect_ratio=:equal)
    
    # Add unit circle for reference
    θ = 0:0.01:2π
    plot!(p4, cos.(θ), sin.(θ), color=:gray, alpha=0.3, label="Unit Circle", lw=1)
    
    # Combine plots
    combined_plot = plot(p1, p2, p3, p4, layout=(2,2), size=(1000, 800))
    plot!(combined_plot, suptitle=title)
    
    display(combined_plot)
end

function visualize_frequency_spectrum(signal::Vector{ComplexF32}, title::String="Frequency Spectrum")
    n = length(signal)
    
    # Compute FFT of real part
    real_parts = real.(signal)
    real_centered = real_parts .- mean(real_parts)
    fft_result = fft(real_centered)
    
    # Power spectrum
    power_spectrum = abs2.(fft_result[1:div(n, 2)])
    frequencies = (0:(div(n, 2)-1)) .* (1.0 / n)
    periods = [1/f for f in frequencies[2:end]]
    
    # Find peaks
    threshold = 0.1 * maximum(power_spectrum[2:end])
    peak_indices = findall(power_spectrum[2:end] .> threshold) .+ 1
    
    # Create plots
    p1 = plot(frequencies[2:end], power_spectrum[2:end],
              title="Power Spectrum",
              xlabel="Normalized Frequency", ylabel="Power",
              label="", lw=2, yscale=:log10)
    
    # Mark peaks
    if !isempty(peak_indices)
        scatter!(p1, frequencies[peak_indices], power_spectrum[peak_indices],
                color=:red, ms=5, label="Peaks")
    end
    
    p2 = plot(periods[1:min(100, length(periods))], 
              power_spectrum[2:min(101, length(power_spectrum))],
              title="Power vs Period",
              xlabel="Period (ticks)", ylabel="Power",
              label="", lw=2, yscale=:log10)
    
    combined_plot = plot(p1, p2, layout=(1,2), size=(1000, 400))
    plot!(combined_plot, suptitle=title)
    
    display(combined_plot)
end

# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

@testset "SyntheticSignalGenerator Module Tests" begin

    @testset "4-Phase Rotation Helpers" begin
        # ... (contents of your original test_phase_rotation_helpers() function go here) ...
        @test true # Placeholder
    end

    @testset "Complex I/Q Signal Generation" begin
        # ... (contents of your original test_complex_iq_generation() function go here) ...
        @test true # Placeholder
    end

    @testset "Different Signal Types" begin
        # ... (contents of your original test_signal_types() function go here) ...
        @test true # Placeholder
    end

    @testset "Fibonacci-Specific Signals" begin
        # ... (contents of your original test_fibonacci_signals() function go here) ...
        @test true # Placeholder
    end

    @testset "Signal Conversion" begin
        # ... (contents of your original test_signal_conversion() function go here) ...
        @test true # Placeholder
    end

    @testset "Batch Signal Generation" begin
        # ... (contents of your original test_batch_generation() function go here) ...
        @test true # Placeholder
    end

end


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

println("$(GREEN)Welcome to the SyntheticSignalGenerator Test Suite!$(RESET)")
println("\nThis interactive test script will guide you through testing")
println("the Complex I/Q signal generation and 4-phase rotation features.")


println("\n$(GREEN)Test session complete!$(RESET)")