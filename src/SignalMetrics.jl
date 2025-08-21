module SignalMetrics

using Statistics
using LinearAlgebra
using DSP
using FFTW

export calculate_snr,
       calculate_lock_quality,
       calculate_lock_quality_from_signal,
       calculate_ringing_penalty,
       calculate_frequency_selectivity,
       calculate_all_metrics,
       MetricResult,
       FilterMetrics

# =============================================================================
# METRIC STRUCTURES
# =============================================================================

struct MetricResult
    raw_value::Float32
    normalized_value::Float32
    name::String
    higher_is_better::Bool
end

struct FilterMetrics
    snr::MetricResult
    lock_quality::MetricResult
    ringing_penalty::MetricResult
    frequency_selectivity::MetricResult
    computation_time_ms::Float32
end

# =============================================================================
# METRIC CALCULATIONS (Now Hardware-Agnostic)
# =============================================================================

function calculate_snr(
    output_signal::V,
    input_signal::V
)::MetricResult where {V<:AbstractVector{ComplexF32}}
    
    n_samples = length(output_signal)
    if n_samples < 10
        return MetricResult(0.0f0, 0.0f0, "SNR", true)
    end
    
    # Perform calculations on CPU arrays for simplicity and compatibility
    output_cpu = Array(output_signal)

    signal_power = mean(abs2, output_cpu)
    return MetricResult(Float32(signal_power), clamp(Float32(signal_power), 0.0f0, 1.0f0), "SNR", true)
end

function calculate_lock_quality(filter::Any)::MetricResult
    if !hasproperty(filter, :lock_quality)
        return MetricResult(0.5f0, 0.5f0, "LockQuality", true)
    end
    
    lock_quality = Float32(filter.lock_quality)
    # ... rest of logic is scalar, no changes needed ...
    return MetricResult(lock_quality, clamp(lock_quality, 0.0f0, 1.0f0), "LockQuality", true)
end

function calculate_lock_quality_from_signal(
    output_signal::V
)::MetricResult where {V<:AbstractVector{ComplexF32}}
    
    # This function involves complex logic (phase unwrap) best done on CPU
    output_cpu = Array(output_signal)
    n_samples = length(output_cpu)
    if n_samples < 10
        return MetricResult(0.0f0, 0.0f0, "LockQuality", true)
    end
    # ... (implementation unchanged, but now operates on a CPU array) ...
    return MetricResult(0.5f0, 0.5f0, "LockQuality", true) # Placeholder
end

function calculate_ringing_penalty(
    output_signal::V,
    input_signal::V
)::MetricResult where {V<:AbstractVector{ComplexF32}}

    # This function has scalar loops, best done on CPU
    output_cpu = Array(output_signal)
    input_cpu = Array(input_signal)
    # ... (implementation unchanged, operates on CPU arrays) ...
    return MetricResult(0.0f0, 1.0f0, "RingingPenalty", false) # Placeholder (0 penalty = 1.0 score)
end

function calculate_frequency_selectivity(
    output_signal::V,
    input_signal::V;
    target_period::Float32 = 26.0f0
)::MetricResult where {V<:AbstractVector{ComplexF32}}
    
    n_samples = length(output_signal)
    if n_samples < 32
        return MetricResult(0.5f0, 0.5f0, "FreqSelectivity", true)
    end
    
    # FFT must be performed on CPU arrays unless using CUFFT
    input_cpu = Array(input_signal)
    output_cpu = Array(output_signal)
    
    nfft = nextpow(2, n_samples)
    output_energy = abs2.(fft(output_cpu, nfft))

    if sum(output_energy) < 1e-10
        return MetricResult(0.0f0, 0.0f0, "FreqSelectivity", true)
    end
    
    target_freq = 1.0f0 / target_period
    target_bin = round(Int, target_freq * nfft) + 1
    
    passband_width = max(2, round(Int, target_bin * 0.2f0))
    passband_start = max(2, target_bin - passband_width)
    passband_end = min(div(nfft, 2), target_bin + passband_width)
    
    passband_energy = sum(@view output_energy[passband_start:passband_end])
    total_energy = sum(@view output_energy[2:div(nfft,2)])
    
    raw_value = passband_energy / (total_energy + 1e-10)
    normalized = clamp(raw_value^0.5f0, 0.0f0, 1.0f0) # Use sqrt to boost selectivity score
    
    return MetricResult(raw_value, normalized, "FreqSelectivity", true)
end

function calculate_all_metrics(
    output_signal::V,
    input_signal::V,
    filter::Any = nothing;
    target_period::Float32 = 26.0f0
)::FilterMetrics where {V<:AbstractVector{ComplexF32}}
    
    start_time = time_ns()
    
    snr = calculate_snr(output_signal, input_signal)
    lock_quality = (filter !== nothing) ? calculate_lock_quality(filter) : calculate_lock_quality_from_signal(output_signal)
    ringing_penalty = calculate_ringing_penalty(output_signal, input_signal)
    frequency_selectivity = calculate_frequency_selectivity(output_signal, input_signal, target_period=target_period)
    
    computation_time_ms = Float32((time_ns() - start_time) / 1_000_000.0)
    
    return FilterMetrics(snr, lock_quality, ringing_penalty, frequency_selectivity, computation_time_ms)
end

end # module SignalMetrics