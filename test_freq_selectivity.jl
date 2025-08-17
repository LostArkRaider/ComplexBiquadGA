# Test script to verify the frequency selectivity fix
# Run this after updating SignalMetrics.jl

# Reload the module
include("src/SignalMetrics.jl")
using .SignalMetrics

println("\n" * "="^60)
println("Testing FIXED Frequency Selectivity Implementation")
println("="^60)

# Generate test signals
n_samples = 200
freq = 2π / 26.0
t = 0:n_samples-1

# Multi-frequency INPUT (has target + harmonics)
multi_freq_input = ComplexF32.(
    sin.(freq .* t) .+ 0.3 * sin.(3*freq .* t) .+ 0.2 * sin.(5*freq .* t),
    0.1 * ones(n_samples)
)

# GOOD filter output (passes mostly target, attenuates harmonics)
good_filter_output = ComplexF32.(
    0.95 * sin.(freq .* t) .+ 0.03 * sin.(3*freq .* t) .+ 0.02 * sin.(5*freq .* t),
    0.1 * ones(n_samples)
)

# POOR filter output (barely attenuates harmonics)
poor_filter_output = ComplexF32.(
    0.9 * sin.(freq .* t) .+ 0.27 * sin.(3*freq .* t) .+ 0.18 * sin.(5*freq .* t),
    0.1 * ones(n_samples)
)

# PERFECT filter output (only target frequency)
perfect_filter_output = ComplexF32.(sin.(freq .* t), 0.1 * ones(n_samples))

# Test with debug output
println("\n1. GOOD FILTER (should have selectivity > 0.3):")
good_result = SignalMetrics.calculate_frequency_selectivity(
    good_filter_output,
    multi_freq_input,
    target_period=26.0f0,
    debug=true
)

println("\n2. POOR FILTER (should have lower selectivity):")
poor_result = SignalMetrics.calculate_frequency_selectivity(
    poor_filter_output,
    multi_freq_input,
    target_period=26.0f0,
    debug=true
)

println("\n3. PERFECT FILTER (should have highest selectivity > 0.5):")
perfect_result = SignalMetrics.calculate_frequency_selectivity(
    perfect_filter_output,
    multi_freq_input,
    target_period=26.0f0,
    debug=true
)

# Summary
println("\n" * "="^60)
println("SUMMARY OF FIXED RESULTS:")
println("="^60)
println("Good filter:    $(good_result.normalized_value) (target > 0.3)")
println("Poor filter:    $(poor_result.normalized_value) (should be < good)")
println("Perfect filter: $(perfect_result.normalized_value) (target > 0.5)")

# Check if tests would pass
test_results = [
    ("Good > 0.3", good_result.normalized_value > 0.3f0),
    ("Poor < Good", poor_result.normalized_value < good_result.normalized_value),
    ("Perfect > 0.5", perfect_result.normalized_value > 0.5f0),
    ("Perfect > Good", perfect_result.normalized_value > good_result.normalized_value)
]

println("\n" * "="^60)
println("TEST RESULTS:")
println("="^60)
for (test_name, passed) in test_results
    status = passed ? "✅ PASS" : "❌ FAIL"
    println("$status: $test_name")
end

all_passed = all(last.(test_results))
if all_passed
    println("\n🎉 ALL TESTS PASSED! The fix is working correctly.")
else
    println("\n⚠️ Some tests still failing. Further investigation needed.")
end

println("="^60)