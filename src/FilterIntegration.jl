# src/FilterIntegration.jl - Bridge between GA parameters and filter instances

"""
Filter Integration Module - Chunk 3

Bridges GA chromosome parameters to ProductionFilterBank filter instances.
Handles parameter conversion, scaling, and filter instantiation for fitness evaluation.

Key responsibilities:
- Convert 13-parameter GA chromosome to filter configuration
- Create filter instances from parameters
- Apply proper scaling based on parameter types
- Support both ComplexBiquad and PLLFilterState

The 13 parameters are:
1. q_factor [0.5, 10.0] - LINEAR
2. batch_size [100, 5000] - LOGARITHMIC  
3. phase_detector_gain [0.001, 1.0] - LOGARITHMIC
4. loop_bandwidth [0.0001, 0.1] - LOGARITHMIC
5. lock_threshold [0.0, 1.0] - LINEAR
6. ring_decay [0.9, 1.0] - LINEAR
7. enable_clamping {false, true} - BINARY
8. clamping_threshold [1e-8, 1e-3] - LOGARITHMIC
9. volume_scaling [0.1, 10.0] - LOGARITHMIC
10. max_frequency_deviation [0.01, 0.5] - LINEAR
11. phase_error_history_length {5,10,15,20,30,40,50} - DISCRETE
12-13. complex_weight - COMPLEX (magnitude [0,2], phase [0,2π])
"""

module FilterIntegration

using ..ParameterEncoding
using Random
using Statistics

export create_filter_from_chromosome, 
       create_filter_bank_from_population,
       apply_parameters_to_filter!,
       FilterParameters,
       create_test_filter,
       evaluate_filter_with_signal

# =============================================================================
# PARAMETER STRUCTURES
# =============================================================================

"""
Decoded filter parameters from GA chromosome
"""
struct FilterParameters
    # Filter parameters
    q_factor::Float32
    batch_size::Int32
    
    # PLL parameters
    phase_detector_gain::Float32
    loop_bandwidth::Float32
    lock_threshold::Float32
    ring_decay::Float32
    enable_clamping::Bool
    clamping_threshold::Float32
    
    # Signal processing
    volume_scaling::Float32
    max_frequency_deviation::Float32
    phase_error_history_length::Int32
    
    # Complex weight (for Stage 2)
    complex_weight::ComplexF32
    
    # Metadata
    fibonacci_number::Int32
    filter_index::Int32
end

# =============================================================================
# MOCK FILTER STRUCTURES (until ProductionFilterBank integration)
# =============================================================================

"""
Simplified ComplexBiquad for testing
Mimics the structure from ProductionFilterBank
"""
mutable struct MockComplexBiquad
    # Filter coefficients
    b0::ComplexF64
    b1::ComplexF64
    b2::ComplexF64
    a1::ComplexF64
    a2::ComplexF64
    
    # State variables
    x1::ComplexF64
    x2::ComplexF64
    y1::ComplexF64
    y2::ComplexF64
    
    # Metadata
    fibonacci_number::Int32
    actual_period::Float64
    q_factor::Float64
    center_frequency::Float64
end

"""
Simplified PLL filter state for testing
Mimics PLLFilterState from ProductionFilterBank
"""
mutable struct MockPLLFilterState
    base_filter::MockComplexBiquad
    
    # PLL control state
    vco_phase::Float64
    vco_frequency::Float64
    center_frequency::Float64
    
    # PLL parameters
    phase_detector_gain::Float64
    loop_bandwidth::Float64
    lock_threshold::Float64
    max_frequency_deviation::Float64
    
    # Adaptive state tracking
    loop_integrator::Float64
    phase_error_history::Vector{Float64}
    phase_error_index::Int32
    phase_error_count::Int32
    lock_quality::Float64
    
    # Ringing state
    is_ringing::Bool
    ring_amplitude::ComplexF64
    ring_decay::Float64
    
    # Additional GA parameters
    enable_clamping::Bool
    clamping_threshold::Float32
    volume_scaling::Float32
    complex_weight::ComplexF32
end

# =============================================================================
# PARAMETER CONVERSION
# =============================================================================

"""
Convert GA chromosome to filter parameters
Accepts Vector{Float32} directly
"""
function chromosome_to_parameters(
    chromosome::Vector{Float32},
    fibonacci_number::Int32 = Int32(13),
    filter_index::Int32 = Int32(1)
)::FilterParameters
    
    # Validate
    @assert length(chromosome) == 13 "Expected 13 genes, got $(length(chromosome))"
    
    # Get genes vector
    genes = chromosome
    
    # For now, we'll do the decoding directly here to avoid dependency issues
    # This matches what ParameterEncoding.decode_chromosome would do
    decoded = decode_genes_directly(genes)
    
    # Extract individual parameters
    q_factor = decoded[1]
    batch_size = Int32(round(decoded[2]))
    phase_detector_gain = decoded[3]
    loop_bandwidth = decoded[4]
    lock_threshold = decoded[5]
    ring_decay = decoded[6]
    enable_clamping = decoded[7] > 0.5f0
    clamping_threshold = decoded[8]
    volume_scaling = decoded[9]
    max_frequency_deviation = decoded[10]
    phase_error_history_length = Int32(round(decoded[11]))
    
    # Complex weight from last two parameters
    weight_mag = decoded[12]
    weight_phase = decoded[13]
    complex_weight = ComplexF32(
        weight_mag * cos(weight_phase),
        weight_mag * sin(weight_phase)
    )
    
    return FilterParameters(
        q_factor,
        batch_size,
        phase_detector_gain,
        loop_bandwidth,
        lock_threshold,
        ring_decay,
        enable_clamping,
        clamping_threshold,
        volume_scaling,
        max_frequency_deviation,
        phase_error_history_length,
        complex_weight,
        fibonacci_number,
        filter_index
    )
end

"""
Decode genes directly without using ParameterEncoding module
This avoids dependency issues with GATypes
"""
function decode_genes_directly(genes::Vector{Float32})::Vector{Float32}
    @assert length(genes) == 13 "Expected 13 genes"
    
    decoded = Vector{Float32}(undef, 13)
    
    # Parameter 1: q_factor [0.5, 10.0] - LINEAR
    decoded[1] = 0.5f0 + genes[1] * 9.5f0
    
    # Parameter 2: batch_size [100, 5000] - LOGARITHMIC
    decoded[2] = exp(log(100f0) + genes[2] * (log(5000f0) - log(100f0)))
    
    # Parameter 3: phase_detector_gain [0.001, 1.0] - LOGARITHMIC
    decoded[3] = exp(log(0.001f0) + genes[3] * (log(1.0f0) - log(0.001f0)))
    
    # Parameter 4: loop_bandwidth [0.0001, 0.1] - LOGARITHMIC  
    decoded[4] = exp(log(0.0001f0) + genes[4] * (log(0.1f0) - log(0.0001f0)))
    
    # Parameter 5: lock_threshold [0.0, 1.0] - LINEAR
    decoded[5] = genes[5]
    
    # Parameter 6: ring_decay [0.9, 1.0] - LINEAR
    decoded[6] = 0.9f0 + genes[6] * 0.1f0
    
    # Parameter 7: enable_clamping - BINARY (will be converted to Bool later)
    decoded[7] = genes[7]
    
    # Parameter 8: clamping_threshold [1e-8, 1e-3] - LOGARITHMIC
    decoded[8] = exp(log(1e-8) + genes[8] * (log(1e-3) - log(1e-8)))
    
    # Parameter 9: volume_scaling [0.1, 10.0] - LOGARITHMIC
    decoded[9] = exp(log(0.1f0) + genes[9] * (log(10.0f0) - log(0.1f0)))
    
    # Parameter 10: max_frequency_deviation [0.01, 0.5] - LINEAR
    decoded[10] = 0.01f0 + genes[10] * 0.49f0
    
    # Parameter 11: phase_error_history_length - DISCRETE {5,10,15,20,30,40,50}
    discrete_values = Float32[5, 10, 15, 20, 30, 40, 50]
    idx = Int(round(genes[11] * 6) + 1)
    idx = clamp(idx, 1, 7)
    decoded[11] = discrete_values[idx]
    
    # Parameter 12: complex_weight magnitude [0, 2]
    decoded[12] = genes[12] * 2.0f0
    
    # Parameter 13: complex_weight phase [0, 2π]
    decoded[13] = genes[13] * 2.0f0 * Float32(π)
    
    return decoded
end

# =============================================================================
# FILTER CREATION
# =============================================================================

"""
Apply period doubling (matches ProductionFilterBank logic)
"""
function apply_period_doubling(fibonacci_number::Int32)::Float64
    if fibonacci_number == 1
        return 2.01  # Avoid Nyquist
    else
        return Float64(2 * fibonacci_number)
    end
end

"""
Design bandpass filter coefficients
Simplified version matching ProductionFilterBank
"""
function design_bandpass_coefficients(period_bars::Float64, Q::Float64)
    # Normalized center frequency
    fc = 1.0 / period_bars
    
    # Ensure we don't exceed Nyquist
    if fc >= 0.5
        fc = 1.0 / (period_bars + 0.01)
    end
    
    # Design parameters
    ωc = 2π * fc
    bandwidth = fc / Q
    α = sin(π * bandwidth) / cos(π * bandwidth)
    
    # Bandpass coefficients
    b0 = α
    b1 = 0.0
    b2 = -α
    a0 = 1.0 + α
    a1 = -2.0 * cos(ωc)
    a2 = 1.0 - α
    
    # Normalize
    b0 /= a0
    b1 /= a0
    b2 /= a0
    a1 /= a0
    a2 /= a0
    
    return (
        b0 = ComplexF64(b0),
        b1 = ComplexF64(b1),
        b2 = ComplexF64(b2),
        a1 = ComplexF64(a1),
        a2 = ComplexF64(a2)
    )
end

"""
Create a mock ComplexBiquad filter from parameters
"""
function create_biquad_from_parameters(params::FilterParameters)::MockComplexBiquad
    actual_period = apply_period_doubling(params.fibonacci_number)
    coeffs = design_bandpass_coefficients(actual_period, Float64(params.q_factor))
    
    return MockComplexBiquad(
        coeffs.b0, coeffs.b1, coeffs.b2, coeffs.a1, coeffs.a2,
        ComplexF64(0), ComplexF64(0), ComplexF64(0), ComplexF64(0),
        params.fibonacci_number,
        actual_period,
        Float64(params.q_factor),
        2π / actual_period
    )
end

"""
Create a mock PLL filter from parameters
"""
function create_pll_filter_from_parameters(params::FilterParameters)::MockPLLFilterState
    # Create base filter
    base_filter = create_biquad_from_parameters(params)
    
    # Calculate center frequency
    actual_period = apply_period_doubling(params.fibonacci_number)
    center_freq = 2π / actual_period
    
    # Initialize phase error history
    history_length = params.phase_error_history_length
    phase_error_history = zeros(Float64, history_length)
    
    return MockPLLFilterState(
        base_filter,
        0.0,  # vco_phase
        center_freq,  # vco_frequency
        center_freq,  # center_frequency
        Float64(params.phase_detector_gain),
        Float64(params.loop_bandwidth),
        Float64(params.lock_threshold),
        Float64(params.max_frequency_deviation),
        0.0,  # loop_integrator
        phase_error_history,
        Int32(1),  # phase_error_index
        Int32(0),  # phase_error_count
        0.0,  # lock_quality
        false,  # is_ringing
        ComplexF64(0),  # ring_amplitude
        Float64(params.ring_decay),
        params.enable_clamping,
        params.clamping_threshold,
        params.volume_scaling,
        params.complex_weight
    )
end

"""
Create filter from GA chromosome
Main entry point for fitness evaluation
Accepts Vector{Float32} directly
"""
function create_filter_from_chromosome(
    chromosome::Vector{Float32},
    fibonacci_number::Int32 = Int32(13),
    filter_index::Int32 = Int32(1),
    use_pll::Bool = true
)::Union{MockComplexBiquad, MockPLLFilterState}
    
    # Ensure we have a valid chromosome
    @assert length(chromosome) == 13 "Chromosome must have 13 genes"
    
    # Convert chromosome to parameters
    params = chromosome_to_parameters(chromosome, fibonacci_number, filter_index)
    
    # Create appropriate filter type
    if use_pll
        return create_pll_filter_from_parameters(params)
    else
        return create_biquad_from_parameters(params)
    end
end

# =============================================================================
# BATCH FILTER CREATION
# =============================================================================

"""
Create multiple filters from a population
Used for batch fitness evaluation
"""
function create_filter_bank_from_population(
    population::Matrix{Float32},  # population_size × 13
    fibonacci_number::Int32 = Int32(13),
    use_pll::Bool = true
)::Vector{Union{MockComplexBiquad, MockPLLFilterState}}
    
    pop_size = size(population, 1)
    filters = Vector{Union{MockComplexBiquad, MockPLLFilterState}}(undef, pop_size)
    
    for i in 1:pop_size
        # Pass the row directly as a Vector{Float32}
        filters[i] = create_filter_from_chromosome(
            vec(population[i, :]), 
            fibonacci_number, 
            Int32(i),  # filter_index as Int32
            use_pll
        )
    end
    
    return filters
end

# =============================================================================
# FILTER PROCESSING (SIMPLIFIED)
# =============================================================================

"""
Process sample through biquad filter
Direct Form II implementation
"""
function process_sample!(filter::MockComplexBiquad, input::ComplexF64)::ComplexF64
    # Direct Form II
    output = filter.b0 * input + filter.b1 * filter.x1 + filter.b2 * filter.x2 -
             filter.a1 * filter.y1 - filter.a2 * filter.y2
    
    # Update state
    filter.x2 = filter.x1
    filter.x1 = input
    filter.y2 = filter.y1
    filter.y1 = output
    
    return output
end

"""
Simplified PLL processing for fitness evaluation
"""
function process_sample_pll!(filter::MockPLLFilterState, input::ComplexF32)::ComplexF32
    # Convert to ComplexF64 for processing
    z = ComplexF64(input)
    
    # Process through base filter
    filter_output = process_sample!(filter.base_filter, z)
    
    # Simplified PLL logic for fitness evaluation
    if abs(z) > 1e-6
        filter.is_ringing = false
        
        # Phase detection (with optional clamping)
        if filter.enable_clamping && abs(real(z)) > filter.clamping_threshold
            pd_input = ComplexF64(sign(real(z)), imag(z))
            reference_phase = angle(pd_input)
        else
            reference_phase = angle(z)
        end
        
        # Phase error
        if abs(filter_output) > 1e-6
            output_phase = angle(filter_output)
            phase_error = output_phase - filter.vco_phase
        else
            phase_error = reference_phase - filter.vco_phase
        end
        
        # Wrap to [-π, π]
        while phase_error > π; phase_error -= 2π; end
        while phase_error < -π; phase_error += 2π; end
        
        # Update phase error history
        if filter.phase_error_count < length(filter.phase_error_history)
            filter.phase_error_count += 1
            filter.phase_error_history[filter.phase_error_count] = abs(phase_error)
        else
            filter.phase_error_history[filter.phase_error_index] = abs(phase_error)
            filter.phase_error_index = (filter.phase_error_index % length(filter.phase_error_history)) + 1
        end
        
        # Update lock quality
        if filter.phase_error_count >= 5
            avg_error = mean(@view filter.phase_error_history[1:filter.phase_error_count])
            filter.lock_quality = exp(-2.0 * avg_error)
        end
        
        # PLL loop filter
        filter.loop_integrator += filter.loop_bandwidth * phase_error * 2.0
        frequency_correction = filter.phase_detector_gain * phase_error * 1.5 + filter.loop_integrator
        filter.vco_frequency = filter.center_frequency + frequency_correction
        
        # Limit frequency deviation
        max_dev = filter.center_frequency * filter.max_frequency_deviation
        filter.vco_frequency = clamp(
            filter.vco_frequency,
            filter.center_frequency - max_dev,
            filter.center_frequency + max_dev
        )
        
        # Store ring amplitude
        if filter.lock_quality > filter.lock_threshold * 0.5
            filter.ring_amplitude = filter_output * (1.0 + filter.lock_quality)
        end
    else
        # Ringing mode
        if filter.lock_quality > filter.lock_threshold * 0.3
            filter.is_ringing = true
            filter_output = filter.ring_amplitude * exp(im * filter.vco_phase)
            filter.ring_amplitude *= filter.ring_decay
            filter.lock_quality *= 0.995
        end
    end
    
    # Advance VCO phase
    filter.vco_phase += filter.vco_frequency
    filter.vco_phase = mod(filter.vco_phase, 2π)
    
    # Apply volume scaling and complex weight
    output = filter_output * filter.volume_scaling
    
    # Apply complex weight to real part only
    weighted_real = real(output) * real(filter.complex_weight) - 
                   real(output) * imag(filter.complex_weight)
    weighted_output = ComplexF64(weighted_real, imag(output))
    
    return ComplexF32(weighted_output)
end

# =============================================================================
# SIGNAL EVALUATION
# =============================================================================

"""
Evaluate filter with a signal
Returns filter outputs for metric calculation
"""
function evaluate_filter_with_signal(
    filter::Union{MockComplexBiquad, MockPLLFilterState},
    signal::Vector{ComplexF32}
)::Vector{ComplexF32}
    
    n_samples = length(signal)
    outputs = Vector{ComplexF32}(undef, n_samples)
    
    for i in 1:n_samples
        if isa(filter, MockPLLFilterState)
            outputs[i] = process_sample_pll!(filter, signal[i])
        else
            outputs[i] = ComplexF32(process_sample!(filter, ComplexF64(signal[i])))
        end
    end
    
    return outputs
end

# =============================================================================
# TEST HELPERS
# =============================================================================

"""
Create a test filter with known good parameters
"""
function create_test_filter(
    fibonacci_number::Int32 = Int32(13);
    use_pll::Bool = true
)::Union{MockComplexBiquad, MockPLLFilterState}
    
    # Create a reasonable test chromosome as raw vector
    test_genes = Float32[
        0.5,   # q_factor (normalized)
        0.3,   # batch_size (normalized)
        0.5,   # phase_detector_gain (normalized)
        0.5,   # loop_bandwidth (normalized)
        0.7,   # lock_threshold
        0.95,  # ring_decay
        1.0,   # enable_clamping (true)
        0.5,   # clamping_threshold (normalized)
        0.5,   # volume_scaling (normalized)
        0.3,   # max_frequency_deviation
        0.4,   # phase_error_history_length (normalized)
        1.0,   # complex_weight magnitude
        0.0    # complex_weight phase
    ]
    
    return create_filter_from_chromosome(
        test_genes,
        fibonacci_number,
        Int32(1),  # filter_index as Int32
        use_pll
    )
end

"""
Apply parameters to existing filter (for updates)
"""
function apply_parameters_to_filter!(
    filter::MockPLLFilterState,
    params::FilterParameters
)
    # Update PLL parameters
    filter.phase_detector_gain = Float64(params.phase_detector_gain)
    filter.loop_bandwidth = Float64(params.loop_bandwidth)
    filter.lock_threshold = Float64(params.lock_threshold)
    filter.max_frequency_deviation = Float64(params.max_frequency_deviation)
    filter.ring_decay = Float64(params.ring_decay)
    filter.enable_clamping = params.enable_clamping
    filter.clamping_threshold = params.clamping_threshold
    filter.volume_scaling = params.volume_scaling
    filter.complex_weight = params.complex_weight
    
    # Update base filter Q factor if changed
    if filter.base_filter.q_factor != Float64(params.q_factor)
        coeffs = design_bandpass_coefficients(
            filter.base_filter.actual_period,
            Float64(params.q_factor)
        )
        filter.base_filter.b0 = coeffs.b0
        filter.base_filter.b1 = coeffs.b1
        filter.base_filter.b2 = coeffs.b2
        filter.base_filter.a1 = coeffs.a1
        filter.base_filter.a2 = coeffs.a2
        filter.base_filter.q_factor = Float64(params.q_factor)
    end
    
    # Resize history if needed
    if length(filter.phase_error_history) != params.phase_error_history_length
        filter.phase_error_history = zeros(Float64, params.phase_error_history_length)
        filter.phase_error_index = Int32(1)
        filter.phase_error_count = Int32(0)
    end
end

end # module FilterIntegration