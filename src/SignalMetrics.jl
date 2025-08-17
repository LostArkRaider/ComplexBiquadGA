# src/SignalMetrics.jl - Signal quality metrics for fitness evaluation

"""
Signal Metrics Module - Chunk 3

Calculates signal quality metrics for filter fitness evaluation.
Each metric returns a normalized score in [0, 1] range for combination.

Metrics implemented:
1. Signal-to-Noise Ratio (SNR) - Measures signal clarity
2. PLL Lock Quality - Measures phase tracking accuracy
3. Ringing Detection - Penalizes excessive oscillation
4. Frequency Selectivity - Measures bandpass effectiveness

All metrics are designed to be fast and suitable for batch evaluation.
"""

module SignalMetrics

using Statistics
using LinearAlgebra
using DSP
using FFTW  # Added for FFT functionality

export calculate_snr,
       calculate_lock_quality,
       calculate_lock_quality_from_signal,
       calculate_ringing_penalty,
       calculate_frequency_selectivity,
       calculate_all_metrics,
       normalize_metric,
       MetricResult,
       FilterMetrics

# =============================================================================
# METRIC STRUCTURES
# =============================================================================

"""
Individual metric result with raw and normalized values
"""
struct MetricResult
    raw_value::Float32
    normalized_value::Float32
    name::String
    higher_is_better::Bool
end

"""
Complete set of filter metrics
"""
struct FilterMetrics
    snr::MetricResult
    lock_quality::MetricResult
    ringing_penalty::MetricResult
    frequency_selectivity::MetricResult
    
    # Aggregated info
    computation_time_ms::Float32
end

# =============================================================================
# SIGNAL-TO-NOISE RATIO (SNR)
# =============================================================================

"""
Calculate Signal-to-Noise Ratio
Higher SNR indicates cleaner signal extraction
"""
function calculate_snr(
    output_signal::Vector{ComplexF32},
    input_signal::Vector{ComplexF32};
    target_frequency::Union{Float64, Nothing} = nothing
)::MetricResult
    
    n_samples = length(output_signal)
    
    if n_samples < 10
        return MetricResult(0.0f0, 0.0f0, "SNR", true)
    end
    
    # Calculate signal power (magnitude squared)
    signal_power = mean(abs2.(output_signal))
    
    if signal_power < 1e-10
        return MetricResult(0.0f0, 0.0f0, "SNR", true)
    end
    
    # Estimate noise as deviation from smooth signal
    # Use moving average as "ideal" signal
    window_size = min(10, n_samples ÷ 4)
    
    if window_size < 3
        # Too few samples for meaningful SNR
        return MetricResult(1.0f0, 0.5f0, "SNR", true)
    end
    
    # Calculate moving average (smooth signal)
    smooth_signal = Vector{ComplexF32}(undef, n_samples)
    
    for i in 1:n_samples
        start_idx = max(1, i - window_size ÷ 2)
        end_idx = min(n_samples, i + window_size ÷ 2)
        smooth_signal[i] = mean(@view output_signal[start_idx:end_idx])
    end
    
    # Noise is deviation from smooth signal
    noise = output_signal .- smooth_signal
    noise_power = mean(abs2.(noise))
    
    # Avoid division by zero
    if noise_power < 1e-10
        # Very low noise - excellent SNR
        snr_db = 40.0f0  # Cap at 40 dB
    else
        # SNR in dB
        snr_db = 10.0f0 * log10(signal_power / noise_power)
        snr_db = clamp(snr_db, -20.0f0, 40.0f0)
    end
    
    # Normalize to [0, 1]
    # Map -20dB to 0, 40dB to 1
    normalized = (snr_db + 20.0f0) / 60.0f0
    
    return MetricResult(snr_db, normalized, "SNR", true)
end

# =============================================================================
# PLL LOCK QUALITY
# =============================================================================

"""
Calculate PLL lock quality from filter state
Requires access to filter's internal state
"""
function calculate_lock_quality(
    filter::Any;  # MockPLLFilterState or similar
    output_signal::Union{Vector{ComplexF32}, Nothing} = nothing
)::MetricResult
    
    # Check if filter has PLL state
    if !hasproperty(filter, :lock_quality)
        # Not a PLL filter - return neutral score
        return MetricResult(0.5f0, 0.5f0, "LockQuality", true)
    end
    
    # Direct lock quality from filter
    lock_quality = Float32(filter.lock_quality)
    
    # Additional quality factors if available
    quality_score = lock_quality
    
    if hasproperty(filter, :phase_error_history) && filter.phase_error_count > 0
        # Calculate phase error consistency
        phase_errors = @view filter.phase_error_history[1:filter.phase_error_count]
        
        if length(phase_errors) > 5
            # Lower variance is better
            error_variance = var(phase_errors)
            error_mean = mean(phase_errors)
            
            # Coefficient of variation (normalized variance)
            if error_mean > 1e-6
                cv = sqrt(error_variance) / error_mean
                consistency = exp(-cv)  # High consistency for low CV
            else
                consistency = 1.0f0  # Very low errors - excellent
            end
            
            # Combine lock quality with consistency
            quality_score = 0.7f0 * lock_quality + 0.3f0 * consistency
        end
    end
    
    # Check if currently in lock
    if hasproperty(filter, :is_ringing) && filter.is_ringing
        # Penalize if in ringing mode (lost lock)
        quality_score *= 0.5f0
    end
    
    # Already normalized [0, 1]
    normalized = clamp(quality_score, 0.0f0, 1.0f0)
    
    return MetricResult(quality_score, normalized, "LockQuality", true)
end

"""
Alternative lock quality calculation from output signal only
Used when filter state is not available
"""
function calculate_lock_quality_from_signal(
    output_signal::Vector{ComplexF32}
)::MetricResult
    
    n_samples = length(output_signal)
    
    if n_samples < 10
        return MetricResult(0.0f0, 0.0f0, "LockQuality", true)
    end
    
    # Estimate lock quality from phase stability
    phases = angle.(output_signal)
    
    # Unwrap phases
    unwrapped = copy(phases)
    for i in 2:n_samples
        diff = unwrapped[i] - unwrapped[i-1]
        if diff > π
            unwrapped[i:end] .-= 2π
        elseif diff < -π
            unwrapped[i:end] .+= 2π
        end
    end
    
    # Calculate phase derivative (instantaneous frequency)
    phase_derivative = diff(unwrapped)
    
    # Lock quality from frequency stability
    if length(phase_derivative) > 5
        freq_variance = var(phase_derivative)
        freq_mean = abs(mean(phase_derivative))
        
        # Lower variance relative to mean = better lock
        if freq_mean > 1e-6
            stability = exp(-10.0 * freq_variance / freq_mean^2)
        else
            stability = 0.5f0  # No clear frequency
        end
        
        # Also check amplitude stability as lock indicator
        amplitudes = abs.(output_signal)
        amp_cv = std(amplitudes) / (mean(amplitudes) + 1e-6)
        amp_stability = exp(-amp_cv)
        
        # Combine phase and amplitude stability
        quality_score = 0.6f0 * stability + 0.4f0 * amp_stability
    else
        quality_score = 0.0f0
    end
    
    normalized = clamp(quality_score, 0.0f0, 1.0f0)
    
    return MetricResult(quality_score, normalized, "LockQuality", true)
end

# =============================================================================
# RINGING DETECTION
# =============================================================================

"""
Calculate ringing penalty
Detects and penalizes excessive oscillation after input stops
"""
function calculate_ringing_penalty(
    output_signal::Vector{ComplexF32},
    input_signal::Vector{ComplexF32};
    decay_threshold::Float32 = 0.01f0
)::MetricResult
    
    n_samples = length(output_signal)
    
    if n_samples < 20
        return MetricResult(0.0f0, 0.0f0, "RingingPenalty", false)
    end
    
    # Find periods where input is near zero
    input_magnitudes = abs.(input_signal)
    threshold = maximum(input_magnitudes) * 0.01f0
    
    # Identify quiet periods (where input is minimal)
    quiet_mask = input_magnitudes .< threshold
    
    # Look for ringing: output continues when input stops
    ringing_score = 0.0f0
    ringing_events = 0
    
    # Scan for transitions to quiet periods
    for i in 10:n_samples-10
        if !quiet_mask[i-1] && quiet_mask[i]
            # Transition to quiet period
            # Measure output decay over next samples
            quiet_start = i
            quiet_end = min(i + 20, n_samples)
            
            # Get output magnitudes during quiet period
            quiet_outputs = abs.(@view output_signal[quiet_start:quiet_end])
            
            if length(quiet_outputs) > 5
                # Check for ringing (slow decay)
                initial_mag = quiet_outputs[1]
                
                if initial_mag > decay_threshold
                    # Measure decay rate
                    decay_samples = 0
                    for j in 2:length(quiet_outputs)
                        if quiet_outputs[j] < initial_mag * 0.1f0
                            decay_samples = j
                            break
                        end
                    end
                    
                    if decay_samples == 0
                        # Still ringing at end
                        decay_samples = length(quiet_outputs)
                    end
                    
                    # Penalty based on decay time
                    # Normalize by expected decay (3-5 samples is good)
                    ring_penalty = max(0.0f0, (decay_samples - 5.0f0) / 15.0f0)
                    ringing_score += ring_penalty
                    ringing_events += 1
                end
            end
        end
    end
    
    # Average ringing penalty
    if ringing_events > 0
        avg_ringing = ringing_score / ringing_events
    else
        # No quiet periods found - check overall decay characteristics
        # Look at autocorrelation decay
        output_mags = abs.(output_signal)
        if std(output_mags) > 0
            # High variance might indicate ringing
            cv = std(output_mags) / mean(output_mags)
            avg_ringing = min(1.0f0, cv / 2.0f0)
        else
            avg_ringing = 0.0f0
        end
    end
    
    # Penalty score (0 = no ringing, 1 = severe ringing)
    penalty = clamp(avg_ringing, 0.0f0, 1.0f0)
    
    # For normalized value, invert (1 = good, 0 = bad)
    normalized = 1.0f0 - penalty
    
    return MetricResult(penalty, normalized, "RingingPenalty", false)
end

# =============================================================================
# FREQUENCY SELECTIVITY - FINAL VERSION 2 (ENERGY-BASED)
# =============================================================================

"""
Calculate frequency selectivity using energy distribution analysis
Works with both ideal test signals and real filter outputs
"""
function calculate_frequency_selectivity(
    output_signal::Vector{ComplexF32},
    input_signal::Vector{ComplexF32};
    target_period::Float32 = 26.0f0,  # Default for Fib 13 (2*13)
    debug::Bool = false
)::MetricResult
    
    n_samples = length(output_signal)
    
    if debug
        println("DEBUG: n_samples = $n_samples")
        println("DEBUG: target_period = $target_period")
    end
    
    if n_samples < 32
        return MetricResult(0.5f0, 0.5f0, "FreqSelectivity", true)
    end
    
    # Calculate frequency response using FFT
    nfft = nextpow(2, n_samples)
    
    # Pad signals
    padded_input = vcat(input_signal, zeros(ComplexF32, nfft - n_samples))
    padded_output = vcat(output_signal, zeros(ComplexF32, nfft - n_samples))
    
    # FFT
    input_fft = fft(padded_input)
    output_fft = fft(padded_output)
    
    # Calculate energy spectral density (more robust than coherence for ideal signals)
    input_energy = abs2.(input_fft)
    output_energy = abs2.(output_fft)
    
    # Normalize energies to get probability distributions
    input_energy_norm = input_energy ./ (sum(input_energy) + 1e-10)
    output_energy_norm = output_energy ./ (sum(output_energy) + 1e-10)
    
    # Target frequency bin
    target_freq = Float32(1.0) / target_period
    target_bin = Int(round(target_freq * Float32(nfft))) + 1
    target_bin = clamp(target_bin, 2, div(nfft, 2))  # Skip DC
    
    # Define frequency bands for biquad filters
    # Use tighter bands for sharp biquad response
    passband_width = max(2, Int(round(Float32(target_bin) * 0.2f0)))  # 20% bandwidth
    transition_width = max(1, passband_width ÷ 2)
    
    # Passband
    passband_start = max(2, target_bin - passband_width)
    passband_end = min(div(nfft, 2), target_bin + passband_width)
    
    # Stopband (skip transition regions)
    stopband1_end = max(2, passband_start - transition_width)
    stopband2_start = min(div(nfft, 2), passband_end + transition_width)
    
    if debug
        println("DEBUG: target_bin = $target_bin")
        println("DEBUG: passband = $passband_start:$passband_end")
        println("DEBUG: stopband1 = 2:$stopband1_end, stopband2 = $stopband2_start:$(div(nfft,2))")
    end
    
    # Calculate energy concentration metrics
    if passband_end > passband_start
        # Passband energy (should be high for good filter)
        passband_energy_in = sum(input_energy_norm[passband_start:passband_end])
        passband_energy_out = sum(output_energy_norm[passband_start:passband_end])
        
        # Stopband energy (should be low for good filter)
        stopband_energy_out = 0.0f0
        stopband_samples = 0
        
        if stopband1_end >= 2
            stopband_energy_out += sum(output_energy_norm[2:stopband1_end])
            stopband_samples += stopband1_end - 1
        end
        
        if stopband2_start <= div(nfft, 2)
            stopband_energy_out += sum(output_energy_norm[stopband2_start:div(nfft,2)])
            stopband_samples += div(nfft, 2) - stopband2_start + 1
        end
        
        # Calculate input stopband energy for reference
        stopband_energy_in = 0.0f0
        if stopband1_end >= 2
            stopband_energy_in += sum(input_energy_norm[2:stopband1_end])
        end
        if stopband2_start <= div(nfft, 2)
            stopband_energy_in += sum(input_energy_norm[stopband2_start:div(nfft,2)])
        end
        
        if debug
            println("DEBUG: passband_energy_in = $passband_energy_in")
            println("DEBUG: passband_energy_out = $passband_energy_out")
            println("DEBUG: stopband_energy_in = $stopband_energy_in")
            println("DEBUG: stopband_energy_out = $stopband_energy_out")
        end
        
        # Calculate selectivity metrics
        
        # 1. Energy concentration: How much of output energy is in passband?
        energy_concentration = passband_energy_out / (passband_energy_out + stopband_energy_out + 1e-10)
        
        # 2. Passband fidelity: How well does filter preserve passband energy?
        if passband_energy_in > 1e-10
            passband_fidelity = min(1.0, passband_energy_out / passband_energy_in)
        else
            passband_fidelity = 0.5f0
        end
        
        # 3. Stopband rejection: How much does filter attenuate stopband?
        if stopband_energy_in > 1e-10
            stopband_rejection = 1.0f0 - min(1.0, stopband_energy_out / stopband_energy_in)
        else
            stopband_rejection = 0.5f0
        end
        
        # 4. Peak preservation: Check if peak is at target frequency
        output_peak_bin = argmax(output_energy[2:div(nfft,2)]) + 1
        peak_accuracy = exp(-abs(output_peak_bin - target_bin) / Float32(passband_width))
        
        if debug
            println("DEBUG: energy_concentration = $energy_concentration")
            println("DEBUG: passband_fidelity = $passband_fidelity")
            println("DEBUG: stopband_rejection = $stopband_rejection")
            println("DEBUG: peak_accuracy = $peak_accuracy")
            println("DEBUG: output_peak_bin = $output_peak_bin vs target = $target_bin")
        end
        
        # Combine metrics with weights
        # Energy concentration is most important for bandpass filters
        normalized = (
            0.35f0 * energy_concentration +
            0.25f0 * passband_fidelity +
            0.25f0 * stopband_rejection +
            0.15f0 * peak_accuracy
        )
        
        # Apply non-linear scaling to spread scores
        # This helps differentiate between good and excellent filters
        if normalized > 0.5f0
            # Good filters get boosted
            normalized = 0.5f0 + 0.5f0 * ((normalized - 0.5f0) * 2.0f0)^0.7f0
        else
            # Poor filters get penalized
            normalized = 0.5f0 * (normalized * 2.0f0)^1.3f0
        end
        
        normalized = clamp(normalized, 0.0f0, 1.0f0)
        
        # Raw value is energy concentration for reference
        raw_value = energy_concentration
        
        if debug
            println("DEBUG: final normalized = $normalized")
        end
    else
        # Cannot measure selectivity
        raw_value = 0.5f0
        normalized = 0.5f0
        
        if debug
            println("DEBUG: Cannot measure - passband too small")
        end
    end
    
    if debug
        println("="^50)
    end
    
    return MetricResult(raw_value, normalized, "FreqSelectivity", true)
end

# =============================================================================
# METRIC COMBINATION
# =============================================================================

"""
Calculate all metrics for a filter
"""
function calculate_all_metrics(
    output_signal::Vector{ComplexF32},
    input_signal::Vector{ComplexF32},
    filter::Any = nothing;
    target_period::Float32 = 26.0f0,
    debug::Bool = false
)::FilterMetrics
    
    start_time = time_ns()
    
    # Calculate individual metrics
    snr = calculate_snr(output_signal, input_signal)
    
    # Try to get lock quality from filter state, fall back to signal-based
    if filter !== nothing && hasproperty(filter, :lock_quality)
        lock_quality = calculate_lock_quality(filter, output_signal=output_signal)
    else
        lock_quality = calculate_lock_quality_from_signal(output_signal)
    end
    
    ringing_penalty = calculate_ringing_penalty(output_signal, input_signal)
    
    frequency_selectivity = calculate_frequency_selectivity(
        output_signal, 
        input_signal,
        target_period=target_period,
        debug=debug
    )
    
    end_time = time_ns()
    elapsed_ns = end_time - start_time
    
    # Convert nanoseconds to milliseconds
    computation_time_ms = Float32(elapsed_ns / 1_000_000.0)
    
    # Ensure minimum non-zero time
    if computation_time_ms < 0.001f0
        computation_time_ms = 0.001f0
    end
    
    if debug
        println("DEBUG: Computation time = $(computation_time_ms) ms")
    end
    
    return FilterMetrics(
        snr,
        lock_quality,
        ringing_penalty,
        frequency_selectivity,
        computation_time_ms
    )
end

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

"""
Normalize a metric value to [0, 1] range
"""
function normalize_metric(
    value::Float32,
    min_val::Float32,
    max_val::Float32;
    higher_is_better::Bool = true
)::Float32
    
    if max_val <= min_val
        return 0.5f0
    end
    
    normalized = (value - min_val) / (max_val - min_val)
    normalized = clamp(normalized, 0.0f0, 1.0f0)
    
    if !higher_is_better
        normalized = 1.0f0 - normalized
    end
    
    return normalized
end

end # module SignalMetrics