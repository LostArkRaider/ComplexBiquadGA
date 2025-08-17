# src/ParameterEncoding.jl - Parameter Encoding/Decoding for 13-Parameter Chromosomes
# Handles scaling (linear, logarithmic, discrete) and complex weight encoding

module ParameterEncoding

using Random

export encode_chromosome, decode_chromosome, encode_parameter, decode_parameter,
       encode_complex_weight, decode_complex_weight, apply_bounds!,
       get_parameter_bounds, validate_chromosome

# =============================================================================
# PARAMETER SCALING FUNCTIONS
# =============================================================================

"""
Encode a single parameter based on its type and scaling
Returns a Float32 value suitable for GA operations
"""
function encode_parameter(value::Real, param_index::Int32, ranges)::Float32
    if param_index == 1  # q_factor - LINEAR
        return Float32(value)
        
    elseif param_index == 2  # batch_size - LOGARITHMIC
        return Float32(log(max(value, ranges.batch_size_range[1])))
        
    elseif param_index == 3  # phase_detector_gain - LOGARITHMIC
        return Float32(log(max(value, ranges.phase_detector_gain_range[1])))
        
    elseif param_index == 4  # loop_bandwidth - LOGARITHMIC
        return Float32(log(max(value, ranges.loop_bandwidth_range[1])))
        
    elseif param_index == 5  # lock_threshold - LINEAR
        return Float32(value)
        
    elseif param_index == 6  # ring_decay - LINEAR
        return Float32(value)
        
    elseif param_index == 7  # enable_clamping - BINARY
        # Handle both boolean and numeric inputs
        if isa(value, Bool)
            return value ? 1.0f0 : 0.0f0
        else
            return Float32(value) > 0.5f0 ? 1.0f0 : 0.0f0
        end
        
    elseif param_index == 8  # clamping_threshold - LOGARITHMIC
        return Float32(log(max(value, ranges.clamping_threshold_range[1])))
        
    elseif param_index == 9  # volume_scaling - LOGARITHMIC
        return Float32(log(max(value, ranges.volume_scaling_range[1])))
        
    elseif param_index == 10  # max_frequency_deviation - LINEAR
        return Float32(value)
        
    elseif param_index == 11  # phase_error_history_length - DISCRETE
        # Map to index in options array
        idx = findfirst(==(Int32(value)), ranges.phase_error_history_length_options)
        return Float32(idx !== nothing ? idx : 1)
        
    elseif param_index in [12, 13]  # complex_weight components - LINEAR
        return Float32(value)
        
    else
        error("Invalid parameter index: $param_index")
    end
end

"""
Decode a single parameter from its encoded form
"""
function decode_parameter(encoded_value::Float32, param_index::Int32, ranges)
    if param_index == 1  # q_factor - LINEAR
        return clamp(encoded_value, ranges.q_factor_range...)
        
    elseif param_index == 2  # batch_size - LOGARITHMIC
        val = exp(encoded_value)
        return clamp(round(Int32, val), ranges.batch_size_range...)
        
    elseif param_index == 3  # phase_detector_gain - LOGARITHMIC
        val = exp(encoded_value)
        return clamp(Float32(val), ranges.phase_detector_gain_range...)
        
    elseif param_index == 4  # loop_bandwidth - LOGARITHMIC
        val = exp(encoded_value)
        return clamp(Float32(val), ranges.loop_bandwidth_range...)
        
    elseif param_index == 5  # lock_threshold - LINEAR
        return clamp(encoded_value, ranges.lock_threshold_range...)
        
    elseif param_index == 6  # ring_decay - LINEAR
        return clamp(encoded_value, ranges.ring_decay_range...)
        
    elseif param_index == 7  # enable_clamping - BINARY
        return encoded_value > 0.5f0
        
    elseif param_index == 8  # clamping_threshold - LOGARITHMIC
        val = exp(encoded_value)
        return clamp(Float32(val), ranges.clamping_threshold_range...)
        
    elseif param_index == 9  # volume_scaling - LOGARITHMIC
        val = exp(encoded_value)
        return clamp(Float32(val), ranges.volume_scaling_range...)
        
    elseif param_index == 10  # max_frequency_deviation - LINEAR
        return clamp(encoded_value, ranges.max_frequency_deviation_range...)
        
    elseif param_index == 11  # phase_error_history_length - DISCRETE
        idx = clamp(round(Int32, encoded_value), 1, length(ranges.phase_error_history_length_options))
        return ranges.phase_error_history_length_options[idx]
        
    elseif param_index in [12, 13]  # complex_weight components
        if param_index == 12  # magnitude component
            return clamp(encoded_value, ranges.complex_weight_mag_range...)
        else  # phase component
            return clamp(encoded_value, ranges.complex_weight_phase_range...)
        end
        
    else
        error("Invalid parameter index: $param_index")
    end
end

# =============================================================================
# COMPLEX WEIGHT HANDLING
# =============================================================================

"""
Encode complex weight as magnitude and phase components
"""
function encode_complex_weight(real::Float32, imag::Float32)::Tuple{Float32, Float32}
    magnitude = sqrt(real^2 + imag^2)
    phase = atan(imag, real)
    
    # Normalize phase to [0, 2π]
    if phase < 0
        phase += 2π
    end
    
    return (Float32(magnitude), Float32(phase))
end

"""
Decode complex weight from magnitude and phase components
"""
function decode_complex_weight(magnitude::Float32, phase::Float32)::Tuple{Float32, Float32}
    real = magnitude * cos(phase)
    imag = magnitude * sin(phase)
    return (Float32(real), Float32(imag))
end

# =============================================================================
# CHROMOSOME OPERATIONS
# =============================================================================

"""
Encode a full parameter set into a chromosome
Input: Raw parameter values (mixed types)
Output: Float32 chromosome vector suitable for GA operations
"""
function encode_chromosome(params::Vector, ranges)::Vector{Float32}
    if length(params) != 13
        error("Expected 13 parameters, got $(length(params))")
    end
    
    chromosome = Vector{Float32}(undef, 13)
    
    # Encode first 11 parameters
    for i in 1:11
        chromosome[i] = encode_parameter(params[i], Int32(i), ranges)
    end
    
    # Handle complex weight (params 12-13 are real and imag) - ensure Float32
    mag, phase = encode_complex_weight(Float32(params[12]), Float32(params[13]))
    chromosome[12] = mag
    chromosome[13] = phase
    
    return chromosome
end

"""
Decode a chromosome back to parameter values
"""
function decode_chromosome(chromosome::Vector{Float32}, ranges)
    if length(chromosome) != 13
        error("Expected 13-element chromosome, got $(length(chromosome))")
    end
    
    params = Vector{Any}(undef, 13)
    
    # Decode first 11 parameters
    for i in 1:11
        params[i] = decode_parameter(chromosome[i], Int32(i), ranges)
    end
    
    # Decode complex weight from magnitude and phase
    real, imag = decode_complex_weight(chromosome[12], chromosome[13])
    params[12] = real
    params[13] = imag
    
    return params
end

# =============================================================================
# BOUNDS AND VALIDATION
# =============================================================================

"""
Get bounds for encoded parameters (used in GA operations)
"""
function get_parameter_bounds(param_index::Int32, ranges)::Tuple{Float32, Float32}
    if param_index == 1  # q_factor
        return ranges.q_factor_range
        
    elseif param_index == 2  # batch_size (log scale)
        return (Float32(log(ranges.batch_size_range[1])), 
                Float32(log(ranges.batch_size_range[2])))
        
    elseif param_index == 3  # phase_detector_gain (log scale)
        return (Float32(log(ranges.phase_detector_gain_range[1])), 
                Float32(log(ranges.phase_detector_gain_range[2])))
        
    elseif param_index == 4  # loop_bandwidth (log scale)
        return (Float32(log(ranges.loop_bandwidth_range[1])), 
                Float32(log(ranges.loop_bandwidth_range[2])))
        
    elseif param_index == 5  # lock_threshold
        return ranges.lock_threshold_range
        
    elseif param_index == 6  # ring_decay
        return ranges.ring_decay_range
        
    elseif param_index == 7  # enable_clamping
        return (0.0f0, 1.0f0)
        
    elseif param_index == 8  # clamping_threshold (log scale)
        return (Float32(log(ranges.clamping_threshold_range[1])), 
                Float32(log(ranges.clamping_threshold_range[2])))
        
    elseif param_index == 9  # volume_scaling (log scale)
        return (Float32(log(ranges.volume_scaling_range[1])), 
                Float32(log(ranges.volume_scaling_range[2])))
        
    elseif param_index == 10  # max_frequency_deviation
        return ranges.max_frequency_deviation_range
        
    elseif param_index == 11  # phase_error_history_length (discrete)
        return (1.0f0, Float32(length(ranges.phase_error_history_length_options)))
        
    elseif param_index == 12  # complex weight magnitude
        return ranges.complex_weight_mag_range
        
    elseif param_index == 13  # complex weight phase
        return ranges.complex_weight_phase_range
        
    else
        error("Invalid parameter index: $param_index")
    end
end

"""
Apply bounds to a chromosome in-place
"""
function apply_bounds!(chromosome::Vector{Float32}, ranges)
    for i in 1:13
        bounds = get_parameter_bounds(Int32(i), ranges)
        chromosome[i] = clamp(chromosome[i], bounds...)
    end
    return chromosome
end

"""
Validate that a chromosome is within bounds
"""
function validate_chromosome(chromosome::Vector{Float32}, ranges)::Bool
    if length(chromosome) != 13
        return false
    end
    
    for i in 1:13
        bounds = get_parameter_bounds(Int32(i), ranges)
        if chromosome[i] < bounds[1] || chromosome[i] > bounds[2]
            return false
        end
    end
    
    return true
end

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

"""
Generate a random chromosome within bounds
"""
function random_chromosome(ranges, rng::AbstractRNG = Random.default_rng())::Vector{Float32}
    chromosome = Vector{Float32}(undef, 13)
    
    for i in 1:13
        bounds = get_parameter_bounds(Int32(i), ranges)
        
        if i == 7  # Binary parameter
            chromosome[i] = rand(rng) > 0.5 ? 1.0f0 : 0.0f0
        elseif i == 11  # Discrete parameter
            chromosome[i] = Float32(rand(rng, 1:Int32(bounds[2])))
        else  # Continuous parameters
            chromosome[i] = bounds[1] + (bounds[2] - bounds[1]) * rand(rng, Float32)
        end
    end
    
    return chromosome
end

"""
Create a chromosome from default values
"""
function default_chromosome(defaults, ranges)::Vector{Float32}
    params = [
        defaults.default_q_factor,
        defaults.default_batch_size,
        defaults.default_pll_gain,
        defaults.default_loop_bandwidth,
        defaults.default_lock_threshold,
        defaults.default_ring_decay,
        defaults.default_enable_clamping,
        defaults.default_clamping_threshold,
        defaults.default_volume_scaling,
        defaults.default_max_frequency_deviation,
        defaults.default_phase_error_history_length,
        defaults.default_complex_weight_real,
        defaults.default_complex_weight_imag
    ]
    
    return encode_chromosome(params, ranges)
end

end # module ParameterEncoding