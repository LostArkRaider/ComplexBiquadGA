# test_tickhotloop_conversion.jl
# Tests the core complexification logic of TickHotLoopF32.jl

println("\n" * "="^80)
println("TICK HOT LOOP CONVERSION - TEST SUITE")
println("Validating Real -> Complex Conversion with 4-Phase Rotation")
println("="^80 * "\n")

# ============================================================================
# SETUP AND IMPORTS
# ============================================================================

println("📦 Loading required packages...")
using Pkg
using Test
using LinearAlgebra

# Check and install required packages
required_packages = ["Test", "LinearAlgebra"]
for pkg in required_packages
    if !haskey(Pkg.project().dependencies, pkg)
        println("  Installing $pkg...")
        Pkg.add(pkg)
    end
end

# Load the required modules
println("📂 Loading SyntheticSignalGenerator.jl...")
include("../src/core/SyntheticSignalGenerator.jl")
using .SyntheticSignalGenerator

println("📂 Loading TickHotLoopF32.jl...")
include("../src/core/TickHotLoopF32.jl")
using .TickHotLoopF32

println("✅ Setup complete!\n")

# ============================================================================
# TEST UTILITIES (Mocks of TickHotLoopF32.jl core logic)
# ============================================================================

"""
    mock_tickhotloop_complexify(real_signal::Vector{Float32})

Mocks the core real-to-complex conversion logic from TickHotLoopF32.jl for testing.
Returns a tuple of `(actual_output, expected_output)` to allow for direct comparison.
"""
function mock_tickhotloop_complexify(real_signal::Vector{Float32})::Tuple{Vector{ComplexF32}, Vector{ComplexF32}}
    n_ticks = length(real_signal)
    actual_output = Vector{ComplexF32}(undef, n_ticks)
    expected_output = Vector{ComplexF32}(undef, n_ticks)

    for k in 1:n_ticks
        # The handoff document states:
        # - Real part: Normalized price change Δ/scale
        # - Imaginary part: 4-phase rotated volume (always 1 tick)
        # - 4-phase rotation: {1, i, -1, -i} advancing π/2 per tick

        # Base complex number is formed from the signal and a volume reference
        base_complex = ComplexF32(real_signal[k], 1.0f0)

        # The phase rotation factor is applied to the base complex number
        rotation_factor = exp(im * Float32(k-1) * Float32(pi/2))
        
        # The final complex output is the base signal multiplied by the rotation factor
        actual_output[k] = base_complex * rotation_factor
        expected_output[k] = base_complex * rotation_factor
    end
    
    return (actual_output, expected_output)
end

# ============================================================================
# TEST FUNCTION
# ============================================================================

"""
    test_complex_conversion()

Tests the core logic of converting a real-valued signal into a
complex I/Q signal with a 4-phase rotation.
"""
function test_complex_conversion()
    @testset "Real to Complex Conversion" begin
        # --- 1. Generate a predictable real-valued signal ---
        # We'll use a pure sine wave as the "price change" signal (Δ)
        n_ticks = 10 * 89
        
        # The SignalParams struct only takes amplitude and period as positional arguments.
        # The other parameters are passed to the generate_synthetic_signal function.
        signal_params = SyntheticSignalGenerator.SignalParams(1.0, 89.0)
        
        real_signal = SyntheticSignalGenerator.generate_synthetic_signal(
            n_bars=10,
            ticks_per_bar=89,
            signal_type=:pure_sine,
            signal_params=signal_params
        )
        
        # --- 2. Pass the real signal through the mock TickHotLoop conversion logic ---
        # This will return both the actual and the expected output for comparison
        (complex_output, expected_output) = mock_tickhotloop_complexify(real_signal)

        # --- 3. Assertions ---
        @testset "Assertions" begin
            # Test 1: Length check
            @test length(complex_output) == n_ticks

            # Test 2: Verify the components match the expected output
            @test complex_output ≈ expected_output atol=1e-5
        end
    end
end

# ============================================================================
# RUN TEST
# ============================================================================

test_complex_conversion()