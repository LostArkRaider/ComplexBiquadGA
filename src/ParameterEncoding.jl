module ParameterEncoding

using Random
using ..GATypes # Use relative path for intra-package dependency

export encode_chromosome, decode_chromosome, encode_parameter, decode_parameter,
       encode_complex_weight, decode_complex_weight, apply_bounds!,
       get_parameter_bounds, validate_chromosome

# --- Functions are largely unchanged as they operate on scalars or simple vectors ---

function encode_parameter(value::Real, param_index::Int32, ranges::ParameterRanges)::Float32
    # [cite_start]... (implementation from source [cite: 441-447])
end

function decode_parameter(encoded_value::Float32, param_index::Int32, ranges::ParameterRanges)
    # [cite_start]... (implementation from source [cite: 447-452])
end

function encode_complex_weight(real::Float32, imag::Float32)::Tuple{Float32, Float32}
    # [cite_start]... (implementation from source [cite: 452])
end

function decode_complex_weight(magnitude::Float32, phase::Float32)::Tuple{Float32, Float32}
    # [cite_start]... (implementation from source [cite: 452-453])
end

function encode_chromosome(params::Vector, ranges::ParameterRanges)::Vector{Float32}
    # [cite_start]... (implementation from source [cite: 453-454])
end

function decode_chromosome(chromosome::Vector{Float32}, ranges::ParameterRanges)
    # [cite_start]... (implementation from source [cite: 454-455])
end

function get_parameter_bounds(param_index::Int32, ranges::ParameterRanges)::Tuple{Float32, Float32}
    # [cite_start]... (implementation from source [cite: 456-460])
end

function apply_bounds!(chromosome::AbstractVector{Float32}, ranges::ParameterRanges)
    for i in 1:13
        bounds = get_parameter_bounds(Int32(i), ranges)
        chromosome[i] = clamp(chromosome[i], bounds...)
    end
    return chromosome
end

function validate_chromosome(chromosome::AbstractVector{Float32}, ranges::ParameterRanges)::Bool
    if length(chromosome) != 13
        return false
    end
    for i in 1:13
        bounds = get_parameter_bounds(Int32(i), ranges)
        if !(bounds[1] <= chromosome[i] <= bounds[2])
            return false
        end
    end
    return true
end

end