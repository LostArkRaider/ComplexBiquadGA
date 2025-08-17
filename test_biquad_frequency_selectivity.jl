# test_biquad_frequency_selectivity.jl
# Test frequency selectivity with realistic biquad filter responses

include("src/SignalMetrics.jl")
using .SignalMetrics

println("\n" * "="^60)
println("Testing Frequency Selectivity with Biquad Filter Characteristics")
println("="^60)

# Generate test signals
n_samples = 200
target_period = 26.0f0  # Fibonacci 13 doubled
target_freq = 2π / target_period
t = 0:n_samples-1

# Create multi-frequency input signal (simulating market data with harmonics)
input_signal = ComplexF32.(
    sin.(target_freq .* t) .+ 
    0.3 * sin.(3*target_freq .* t) .+ 
    0.2 * sin.(5*target_freq .* t) .+
    0.1 * randn(n_samples),  # Add some noise
    ones(n_samples)  # Imaginary part (volume)
)

# Simulate GOOD biquad filter response
# A good biquad bandpass filter with Q=2 would:
# - Pass the target frequency with ~0.9-1.0 gain
# - Attenuate 3x frequency by ~10-20 dB (factor of 0.1-0.3)
# - Attenuate 5x frequency by ~20-30 dB (factor of 0.03-0.1)
println("\n1. Testing GOOD BIQUAD FILTER (Q≈2):")
good_biquad_output = ComplexF32.(
    0.95 * sin.(target_freq .* t) .+      # Strong passband
    0.08 * sin.(3*target_freq .* t) .+    # -22 dB attenuation
    0.02 * sin.(5*target_freq .* t) .+    # -34 dB attenuation
    0.01 * randn(n_samples),               # Reduced noise
    0.95 * ones(n_samples)                 # Slightly attenuated volume
)

good_result = SignalMetrics.calculate_frequency_selectivity(
    good_biquad_output,
    input_signal,
    target_period=target_period,
    debug=true
)
println("GOOD biquad normalized score: $(good_result.normalized_value)")

# Simulate MEDIOCRE biquad filter response (low Q, wide bandwidth)
println("\n2. Testing MEDIOCRE BIQUAD FILTER (Q≈0.7):")
mediocre_biquad_output = ComplexF32.(
    0.85 * sin.(target_freq .* t) .+      # Lower passband gain
    0.25 * sin.(3*target_freq .* t) .+    # -12 dB attenuation (poor)
    0.12 * sin.(5*target_freq .* t) .+    # -18 dB attenuation (poor)
    0.05 * randn(n_samples),               # More noise passes through
    0.85 * ones(n_samples)
)

mediocre_result = SignalMetrics.calculate_frequency_selectivity(
    mediocre_biquad_output,
    input_signal,
    target_period=target_period,
    debug=true
)
println("MEDIOCRE biquad normalized score: $(mediocre_result.normalized_value)")

# Simulate EXCELLENT biquad filter response (high Q, narrow bandwidth)
println("\n3. Testing EXCELLENT BIQUAD FILTER (Q≈5):")
excellent_biquad_output = ComplexF32.(
    0.98 * sin.(target_freq .* t) .+      # Near-unity passband gain
    0.02 * sin.(3*target_freq .* t) .+    # -34 dB attenuation
    0.005 * sin.(5*target_freq .* t) .+   # -46 dB attenuation
    0.002 * randn(n_samples),              # Very low noise
    0.98 * ones(n_samples)
)

excellent_result = SignalMetrics.calculate_frequency_selectivity(
    excellent_biquad_output,
    input_signal,
    target_period=target_period,
    debug=true
)
println("EXCELLENT biquad normalized score: $(excellent_result.normalized_value)")

# Simulate OFF-FREQUENCY biquad (tuned to wrong frequency)
println("\n4. Testing OFF-FREQUENCY BIQUAD (wrong center freq):")
wrong_freq = 1.5 * target_freq  # Filter centered at wrong frequency
off_freq_output = ComplexF32.(
    0.3 * sin.(target_freq .* t) .+       # Target heavily attenuated
    0.7 * sin.(3*target_freq .* t) .+     # Wrong freq passes more
    0.15 * sin.(5*target_freq .* t) .+    # Some high freq
    0.08 * randn(n_samples),
    0.6 * ones(n_samples)
)

off_freq_result = SignalMetrics.calculate_frequency_selectivity(
    off_freq_output,
    input_signal,
    target_period=target_period,
    debug=false  # Less debug output
)
println("OFF-FREQUENCY biquad normalized score: $(off_freq_result.normalized_value)")

# Summary
println("\n" * "="^60)
println("SUMMARY - BIQUAD FILTER SELECTIVITY SCORES:")
println("="^60)
println("Excellent (Q≈5):    $(excellent_result.normalized_value) (should be > 0.7)")
println("Good (Q≈2):         $(good_result.normalized_value) (should be > 0.5)")
println("Mediocre (Q≈0.7):   $(mediocre_result.normalized_value) (should be > 0.3)")
println("Off-frequency:      $(off_freq_result.normalized_value) (should be < 0.3)")

# Test expectations
println("\n" * "="^60)
println("TEST RESULTS:")
println("="^60)

test_results = [
    ("Excellent > 0.7", excellent_result.normalized_value > 0.7f0),
    ("Good > 0.5", good_result.normalized_value > 0.5f0),
    ("Mediocre > 0.3", mediocre_result.normalized_value > 0.3f0),
    ("Off-freq < 0.3", off_freq_result.normalized_value < 0.3f0),
    ("Excellent > Good", excellent_result.normalized_value > good_result.normalized_value),
    ("Good > Mediocre", good_result.normalized_value > mediocre_result.normalized_value),
    ("Mediocre > Off-freq", mediocre_result.normalized_value > off_freq_result.normalized_value)
]

for (test_name, passed) in test_results
    status = passed ? "✅ PASS" : "❌ FAIL"
    println("$status: $test_name")
end

all_passed = all(last.(test_results))
if all_passed
    println("\n🎉 ALL BIQUAD TESTS PASSED! Frequency selectivity working correctly.")
else
    println("\n⚠️ Some tests failed. Review the scores and debug output.")
end

println("="^60)