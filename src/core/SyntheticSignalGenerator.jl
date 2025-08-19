# src/SyntheticSignalGenerator.jl - Synthetic Signal Generation for Testing

"""
Synthetic Signal Generator for Fibonacci Filter Bank Testing

Generates controlled synthetic signals for comprehensive testing of the filter bank system.
Supports pure sine waves, Fibonacci frequency mixtures, and market-like signals with
configurable AM/PM noise. All signals are sampled at Fibonacci tick intervals to form bars.

Key Features:
- Tick-level signal generation with bar-level sampling
- Configurable ticks per bar (must be Fibonacci numbers)
- Pure sine, Fibonacci mixture, and market simulation modes
- Amplitude and phase modulation noise injection
- Complex signal output matching filter bank input format
- Realistic tick timing (0.8056 seconds/tick default, matching market average)

Usage:
    using .SyntheticSignalGenerator
    
    # Generate pure sine wave
    signal = generate_synthetic_signal(
        n_bars = 1000,
        ticks_per_bar = 89,
        signal_type = :pure_sine,
        signal_params = Dict(:period => 21.0, :amplitude => 100.0)
    )
    
    # Generate Fibonacci mixture
    signal = generate_synthetic_signal(
        n_bars = 1000,
        ticks_per_bar = 89,
        signal_type = :fibonacci_mixture,
        signal_params = Dict(:fib_numbers => [3, 5, 8, 13])
    )
"""

module SyntheticSignalGenerator

using DataFrames
using Statistics
using Random
using Dates
using FFTW  # Added for FFT functionality

export generate_synthetic_signal, create_test_dataframe,
       add_amplitude_noise!, add_phase_noise!,
       validate_fibonacci_number, get_fibonacci_numbers,
       generate_test_signal_for_filter, generate_test_signal_set,
       analyze_frequency_content, validate_signal, print_signal_summary,
       SignalParams, SyntheticSignal, generate_complex_iq_signal, create_test_signal_complex_iq,
       phase_pos_global, apply_quad_phase,
       convert_to_complex_iq, analyze_complex_iq_signal,
       generate_test_signals_complex_iq

# =============================================================================
# CONSTANTS AND TYPES
# =============================================================================

# Pre-computed Fibonacci numbers up to reasonable testing limits
const FIBONACCI_NUMBERS = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987]

# Default tick interval in seconds (based on market average: 71.7 seconds / 89 ticks)
const DEFAULT_TICK_INTERVAL = 71.7 / 89.0  # 0.8056 seconds per tick

"""
Parameters for synthetic signal generation
"""
struct SignalParams
    # Common parameters
    n_bars::Int
    ticks_per_bar::Int
    tick_interval::Float64  # Added: seconds per tick for realistic timing
    signal_type::Symbol
    random_seed::Union{Int, Nothing}
    
    # Signal-specific parameters
    period::Union{Float64, Nothing}
    amplitude::Union{Float64, Nothing}
    phase::Union{Float64, Nothing}
    fib_numbers::Union{Vector{Int}, Nothing}
    amplitudes::Union{Vector{Float64}, Nothing}
    phases::Union{Vector{Float64}, Nothing}
    
    # Market simulation parameters
    base_trend::Union{Float64, Nothing}
    volatility::Union{Float64, Nothing}
    mean_reversion_rate::Union{Float64, Nothing}
    fib_components::Union{Vector{Int}, Nothing}
    
    # Noise parameters
    am_depth::Union{Float64, Nothing}
    am_frequency::Union{Float64, Nothing}
    pm_deviation::Union{Float64, Nothing}
    pm_frequency::Union{Float64, Nothing}
end

"""
Container for synthetic signal data
"""
struct SyntheticSignal
    # Raw tick-level signal
    tick_signal::Vector{Float64}
    tick_times::Vector{Float64}
    
    # Bar-sampled data
    bar_signal::Vector{Float64}
    bar_times::Vector{Float64}
    bar_changes::Vector{Float64}
    
    # Complex signal for filter input
    signal_complex::Vector{ComplexF64}
    
    # Metadata
    params::SignalParams
    generation_timestamp::DateTime
end

# =============================================================================
# FIBONACCI UTILITIES
# =============================================================================

"""
Validate if a number is a Fibonacci number
"""
function validate_fibonacci_number(n::Int)::Bool
    if n <= 0
        return false
    end
    
    # Check if n is in pre-computed list
    if n in FIBONACCI_NUMBERS
        return true
    end
    
    # For larger numbers, check using the mathematical property:
    # n is Fibonacci if one of (5*n^2 + 4) or (5*n^2 - 4) is a perfect square
    test1 = 5 * n^2 + 4
    test2 = 5 * n^2 - 4
    
    return is_perfect_square(test1) || is_perfect_square(test2)
end

"""
Check if a number is a perfect square
"""
function is_perfect_square(n::Int)::Bool
    if n < 0
        return false
    end
    root = isqrt(n)
    return root * root == n
end

"""
Get n-th Fibonacci number
"""
function get_fibonacci_number(n::Int)::Int
    if n <= 0
        error("Fibonacci index must be positive")
    end
    
    if n <= length(FIBONACCI_NUMBERS)
        return FIBONACCI_NUMBERS[n]
    end
    
    # Compute for larger indices
    a, b = FIBONACCI_NUMBERS[end-1], FIBONACCI_NUMBERS[end]
    for i in (length(FIBONACCI_NUMBERS)+1):n
        a, b = b, a + b
    end
    return b
end

"""
Get list of Fibonacci numbers up to a maximum value
"""
function get_fibonacci_numbers(max_value::Int)::Vector{Int}
    fibs = Int[]
    a, b = 1, 1
    while a <= max_value
        push!(fibs, a)
        a, b = b, a + b
    end
    return fibs
end

# =============================================================================
# SIGNAL GENERATION FUNCTIONS
# =============================================================================

"""
Generate tick-level pure sine wave signal
NOTE: period_bars is the period in bars, but we generate in real time (seconds)
"""
function generate_pure_sine_ticks(n_ticks::Int, period_bars::Float64, 
                                 amplitude::Float64, phase::Float64,
                                 ticks_per_bar::Int, tick_interval::Float64)::Vector{Float64}
    # Convert period from bars to seconds
    period_seconds = period_bars * ticks_per_bar * tick_interval
    
    # Generate sine wave in real time
    signal = zeros(Float64, n_ticks)
    for i in 1:n_ticks
        t_seconds = (i - 1) * tick_interval  # Time in seconds
        signal[i] = amplitude * sin(2π * t_seconds / period_seconds + phase)
    end
    
    return signal
end

"""
Generate tick-level Fibonacci mixture signal
"""
function generate_fibonacci_mixture_ticks(n_ticks::Int, fib_numbers::Vector{Int},
                                        amplitudes::Vector{Float64}, phases::Vector{Float64},
                                        ticks_per_bar::Int, tick_interval::Float64)::Vector{Float64}
    if length(fib_numbers) != length(amplitudes) || length(fib_numbers) != length(phases)
        error("Fibonacci numbers, amplitudes, and phases must have same length")
    end
    
    # Initialize signal
    signal = zeros(Float64, n_ticks)
    
    # Add each Fibonacci component
    for (idx, fib_num) in enumerate(fib_numbers)
        # Apply period doubling (with special case for Fib 1)
        period_bars = fib_num == 1 ? 2.01 : 2.0 * fib_num
        
        # Convert to seconds
        period_seconds = period_bars * ticks_per_bar * tick_interval
        
        # Add sine component in real time
        for i in 1:n_ticks
            t_seconds = (i - 1) * tick_interval  # Time in seconds
            signal[i] += amplitudes[idx] * sin(2π * t_seconds / period_seconds + phases[idx])
        end
    end
    
    return signal
end

"""
Generate market-like signal with trending and mean reversion
"""
function generate_market_simulation_ticks(n_ticks::Int, base_trend::Float64,
                                        volatility::Float64, mean_reversion_rate::Float64,
                                        fib_components::Vector{Int}, ticks_per_bar::Int,
                                        tick_interval::Float64, rng::AbstractRNG)::Vector{Float64}
    signal = zeros(Float64, n_ticks)
    
    # Initialize with base level
    current_level = 40000.0  # Typical YM futures level
    mean_level = current_level
    
    # Generate Fibonacci component amplitudes based on period
    fib_amplitudes = [volatility * sqrt(fib) for fib in fib_components]
    fib_phases = [2π * rand(rng) for _ in fib_components]
    
    # Add trend, mean reversion, and Fibonacci components
    for i in 1:n_ticks
        t_seconds = (i - 1) * tick_interval  # Time in seconds
        t_bars = t_seconds / (ticks_per_bar * tick_interval)  # Time in bars for compatibility
        
        # Trend component (per second)
        trend = base_trend * t_seconds
        
        # Mean reversion
        mean_level += base_trend * tick_interval
        reversion = -mean_reversion_rate * (current_level - mean_level)
        
        # Random walk component (scaled by tick interval)
        random_walk = volatility * randn(rng) * sqrt(tick_interval)
        
        # Fibonacci oscillations in real time
        fib_sum = 0.0
        for (idx, fib_num) in enumerate(fib_components)
            period_bars = fib_num == 1 ? 2.01 : 2.0 * fib_num
            period_seconds = period_bars * ticks_per_bar * tick_interval
            fib_sum += fib_amplitudes[idx] * sin(2π * t_seconds / period_seconds + fib_phases[idx])
        end
        
        # Update level
        current_level += trend * tick_interval + reversion * tick_interval + random_walk + fib_sum * tick_interval
        signal[i] = current_level
    end
    
    return signal
end

"""
Add amplitude modulation noise to signal
"""
function add_amplitude_noise!(signal::Vector{Float64}, modulation_depth::Float64,
                             modulation_frequency::Float64, ticks_per_bar::Int,
                             tick_interval::Float64)
    n_ticks = length(signal)
    
    for i in 1:n_ticks
        t_seconds = (i - 1) * tick_interval  # Time in seconds
        # Modulation frequency is in Hz (cycles per second)
        modulation = 1.0 + modulation_depth * sin(2π * t_seconds * modulation_frequency)
        signal[i] *= modulation
    end
    
    return signal
end

"""
Add phase modulation noise to signal
"""
function add_phase_noise!(signal::Vector{Float64}, phase_deviation::Float64,
                         modulation_frequency::Float64, ticks_per_bar::Int,
                         tick_interval::Float64)
    n_ticks = length(signal)
    
    # Apply phase modulation via Hilbert transform approximation
    # For simplicity, we'll use time-domain phase shifting
    modulated = zeros(Float64, n_ticks)
    
    for i in 1:n_ticks
        t_seconds = (i - 1) * tick_interval  # Time in seconds
        phase_shift = phase_deviation * sin(2π * t_seconds * modulation_frequency)
        
        # Approximate phase shift by interpolating signal
        # Convert phase shift to sample shift
        shift_seconds = phase_shift / (2π * modulation_frequency)
        shift_samples = shift_seconds / tick_interval
        
        # Linear interpolation for fractional sample shifts
        idx = i + shift_samples
        if idx < 1 || idx > n_ticks
            modulated[i] = signal[i]
        else
            idx_floor = floor(Int, idx)
            idx_ceil = ceil(Int, idx)
            
            if idx_floor >= 1 && idx_ceil <= n_ticks
                weight = idx - idx_floor
                modulated[i] = (1 - weight) * signal[idx_floor] + weight * signal[idx_ceil]
            else
                modulated[i] = signal[i]
            end
        end
    end
    
    # Copy back to original signal
    signal .= modulated
    return signal
end

"""
Sample tick-level signal at bar boundaries
"""
function sample_to_bars(tick_signal::Vector{Float64}, ticks_per_bar::Int)::Tuple{Vector{Float64}, Vector{Float64}}
    n_ticks = length(tick_signal)
    n_bars = div(n_ticks, ticks_per_bar)
    
    bar_signal = zeros(Float64, n_bars)
    bar_changes = zeros(Float64, n_bars)
    
    # Sample at end of each bar
    for i in 1:n_bars
        bar_end_idx = i * ticks_per_bar
        bar_signal[i] = tick_signal[bar_end_idx]
        
        # Calculate change from previous bar
        if i == 1
            # First bar change is from initial value (assumed 0 or first tick)
            bar_changes[i] = bar_signal[i] - tick_signal[1]
        else
            bar_changes[i] = bar_signal[i] - bar_signal[i-1]
        end
    end
    
    return bar_signal, bar_changes
end

"""
Create complex signal representation for filter bank input
"""
function create_complex_signal(bar_changes::Vector{Float64}, ticks_per_bar::Int)::Vector{ComplexF64}
    n_bars = length(bar_changes)
    signal_complex = Vector{ComplexF64}(undef, n_bars)
    
    for i in 1:n_bars
        # Real part: change from previous bar
        # Imaginary part: number of ticks (volume proxy)
        signal_complex[i] = ComplexF64(bar_changes[i], Float64(ticks_per_bar))
    end
    
    return signal_complex
end

# =============================================================================
# MAIN GENERATION FUNCTION
# =============================================================================

"""
Generate synthetic signal with specified parameters

# Arguments
- `n_bars::Int`: Number of bars to generate
- `ticks_per_bar::Int`: Ticks per bar (must be Fibonacci number)
- `signal_type::Symbol`: Type of signal (:pure_sine, :fibonacci_mixture, :market_simulation)
- `signal_params::Dict`: Parameters specific to signal type
- `noise_params::Dict`: Optional noise parameters
- `random_seed::Union{Int, Nothing}`: Random seed for reproducibility
- `tick_interval::Float64`: Seconds per tick (default 0.8056 based on market average)

# Returns
- `SyntheticSignal`: Complete synthetic signal data structure
"""
function generate_synthetic_signal(;
    n_bars::Int,
    ticks_per_bar::Int,
    signal_type::Symbol,
    signal_params::Dict = Dict(),
    noise_params::Dict = Dict(),
    random_seed::Union{Int, Nothing} = nothing,
    tick_interval::Float64 = DEFAULT_TICK_INTERVAL
)::SyntheticSignal
    
    # Validate ticks per bar
    if !validate_fibonacci_number(ticks_per_bar)
        error("ticks_per_bar must be a Fibonacci number, got $ticks_per_bar")
    end
    
    # Set random seed if provided
    if random_seed !== nothing
        Random.seed!(random_seed)
    end
    rng = random_seed === nothing ? Random.default_rng() : MersenneTwister(random_seed)
    
    # Calculate total ticks
    n_ticks = n_bars * ticks_per_bar
    
    # Generate tick-level signal based on type
    if signal_type == :pure_sine
        period = get(signal_params, :period, 21.0)
        amplitude = get(signal_params, :amplitude, 100.0)
        phase = get(signal_params, :phase, 0.0)
        
        tick_signal = generate_pure_sine_ticks(n_ticks, period, amplitude, phase, ticks_per_bar, tick_interval)
        
    elseif signal_type == :fibonacci_mixture
        fib_numbers = get(signal_params, :fib_numbers, [3, 5, 8, 13, 21])
        
        # Default amplitudes inversely proportional to period
        default_amplitudes = [100.0 / sqrt(fib) for fib in fib_numbers]
        amplitudes = get(signal_params, :amplitudes, default_amplitudes)
        
        # Random phases if not specified
        default_phases = [2π * rand(rng) for _ in fib_numbers]
        phases = get(signal_params, :phases, default_phases)
        
        tick_signal = generate_fibonacci_mixture_ticks(n_ticks, fib_numbers, amplitudes, phases, 
                                                      ticks_per_bar, tick_interval)
        
    elseif signal_type == :market_simulation
        base_trend = get(signal_params, :base_trend, 0.1)
        volatility = get(signal_params, :volatility, 10.0)
        mean_reversion_rate = get(signal_params, :mean_reversion_rate, 0.01)
        fib_components = get(signal_params, :fib_components, [3, 5, 8, 13, 21, 34])
        
        tick_signal = generate_market_simulation_ticks(n_ticks, base_trend, volatility,
                                                     mean_reversion_rate, fib_components,
                                                     ticks_per_bar, tick_interval, rng)
    else
        error("Unknown signal type: $signal_type")
    end
    
    # Add noise if requested
    if haskey(noise_params, :am_depth) && noise_params[:am_depth] > 0
        am_depth = noise_params[:am_depth]
        am_frequency = get(noise_params, :am_frequency, 0.1)  # Hz
        add_amplitude_noise!(tick_signal, am_depth, am_frequency, ticks_per_bar, tick_interval)
    end
    
    if haskey(noise_params, :pm_deviation) && noise_params[:pm_deviation] > 0
        pm_deviation = noise_params[:pm_deviation]
        pm_frequency = get(noise_params, :pm_frequency, 0.1)  # Hz
        add_phase_noise!(tick_signal, pm_deviation, pm_frequency, ticks_per_bar, tick_interval)
    end
    
    # Sample to bars
    bar_signal, bar_changes = sample_to_bars(tick_signal, ticks_per_bar)
    
    # Create complex signal
    signal_complex = create_complex_signal(bar_changes, ticks_per_bar)
    
    # Create time vectors
    tick_times = collect(0:(n_ticks-1)) .* tick_interval  # Real time in seconds
    bar_times = collect(1:n_bars) .* (ticks_per_bar * tick_interval)  # Bar end times in seconds
    
    # Create parameter struct with tick_interval added
    params = SignalParams(
        n_bars, ticks_per_bar, tick_interval, signal_type, random_seed,
        get(signal_params, :period, nothing),
        get(signal_params, :amplitude, nothing),
        get(signal_params, :phase, nothing),
        get(signal_params, :fib_numbers, nothing),
        get(signal_params, :amplitudes, nothing),
        get(signal_params, :phases, nothing),
        get(signal_params, :base_trend, nothing),
        get(signal_params, :volatility, nothing),
        get(signal_params, :mean_reversion_rate, nothing),
        get(signal_params, :fib_components, nothing),
        get(noise_params, :am_depth, nothing),
        get(noise_params, :am_frequency, nothing),
        get(noise_params, :pm_deviation, nothing),
        get(noise_params, :pm_frequency, nothing)
    )
    
    return SyntheticSignal(
        tick_signal, tick_times,
        bar_signal, bar_times, bar_changes,
        signal_complex,
        params, now()
    )
end

# =============================================================================
# DATAFRAME CREATION
# =============================================================================

"""
Create a DataFrame compatible with the filter bank system from synthetic signal
"""
function create_test_dataframe(signal::SyntheticSignal; 
                              base_price::Float64 = 40000.0,
                              base_volume::Float64 = 89.0)::DataFrame
    
    n_bars = signal.params.n_bars
    
    # Create timestamps with synthetic market time
    start_time = DateTime(2025, 1, 1, 9, 30, 0)  # Market open
    # Use realistic bar duration based on tick interval and ticks per bar
    bar_duration_seconds = signal.params.ticks_per_bar * signal.params.tick_interval
    bar_duration_ms = round(Int, bar_duration_seconds * 1000)
    timestamps = [start_time + Millisecond(i * bar_duration_ms) for i in 0:(n_bars-1)]
    
    # Create synthetic OHLCV data based on signal
    # Use signal changes to create realistic OHLC relationships
    df = DataFrame()
    
    df.reference_timestamp = timestamps
    
    # Generate prices from cumulative signal
    prices = base_price .+ cumsum(signal.bar_changes)
    
    # Create OHLC from signal with realistic relationships
    df.open = prices
    df.close = prices .+ signal.bar_changes
    
    # High/Low based on signal volatility
    bar_volatility = abs.(signal.bar_changes) .+ 1.0
    df.high = max.(df.open, df.close) .+ 0.5 * bar_volatility
    df.low = min.(df.open, df.close) .- 0.5 * bar_volatility
    
    # Volume is ticks per bar (can vary for different tests)
    df.volume = fill(Float64(signal.params.ticks_per_bar), n_bars)
    
    # Market time - cumulative bar count (like the real system)
    df.market_time = Float64.(1:n_bars)
    
    # Additional columns from the real system
    df.bar_duration = fill(bar_duration_seconds, n_bars)  # Realistic duration
    df.tick_count = fill(signal.params.ticks_per_bar, n_bars)
    
    # Complex inputs for direct testing
    df.signal_complex = signal.signal_complex
    
    return df
end

# =============================================================================
# ANALYSIS UTILITIES
# =============================================================================

"""
Analyze frequency content of synthetic signal using DFT
"""
function analyze_frequency_content(signal::SyntheticSignal)::Dict{String, Any}
    bar_signal = signal.bar_signal
    n_bars = length(bar_signal)
    
    # Remove DC component
    signal_centered = bar_signal .- mean(bar_signal)
    
    # Compute DFT using FFTW
    fft_result = fft(signal_centered)
    
    # Compute power spectrum
    power_spectrum = abs2.(fft_result[1:div(n_bars, 2)])
    
    # Frequency calculation needs to account for real sampling rate
    bar_duration_seconds = signal.params.ticks_per_bar * signal.params.tick_interval
    sampling_frequency = 1.0 / bar_duration_seconds  # Hz
    frequencies = (0:(div(n_bars, 2)-1)) .* (sampling_frequency / n_bars)
    
    # Find dominant frequencies
    sorted_indices = sortperm(power_spectrum, rev=true)
    dominant_indices = sorted_indices[1:min(10, length(sorted_indices))]
    
    dominant_freqs = frequencies[dominant_indices]
    dominant_powers = power_spectrum[dominant_indices]
    dominant_periods = [1/f for f in dominant_freqs if f > 0]
    
    return Dict(
        "frequencies" => frequencies,
        "power_spectrum" => power_spectrum,
        "dominant_frequencies" => dominant_freqs,
        "dominant_powers" => dominant_powers,
        "dominant_periods" => dominant_periods,
        "total_power" => sum(power_spectrum),
        "sampling_frequency" => sampling_frequency
    )
end

"""
Validate synthetic signal properties
"""
function validate_signal(signal::SyntheticSignal)::Dict{String, Any}
    validation_results = Dict{String, Any}()
    
    # Check signal dimensions
    validation_results["n_bars_correct"] = length(signal.bar_signal) == signal.params.n_bars
    validation_results["n_ticks_correct"] = length(signal.tick_signal) == signal.params.n_bars * signal.params.ticks_per_bar
    
    # Check for NaN or Inf values
    validation_results["no_nan_values"] = !any(isnan.(signal.bar_signal))
    validation_results["no_inf_values"] = !any(isinf.(signal.bar_signal))
    
    # Check signal statistics
    validation_results["mean"] = mean(signal.bar_signal)
    validation_results["std"] = std(signal.bar_signal)
    validation_results["min"] = minimum(signal.bar_signal)
    validation_results["max"] = maximum(signal.bar_signal)
    
    # Check complex signal formation
    validation_results["complex_signal_valid"] = length(signal.signal_complex) == signal.params.n_bars
    
    # Check timing
    validation_results["tick_interval"] = signal.params.tick_interval
    validation_results["bar_duration_seconds"] = signal.params.ticks_per_bar * signal.params.tick_interval
    
    # Frequency analysis for pure sine validation
    if signal.params.signal_type == :pure_sine && signal.params.period !== nothing
        freq_analysis = analyze_frequency_content(signal)
        # Expected frequency in Hz, accounting for real time
        bar_duration_seconds = signal.params.ticks_per_bar * signal.params.tick_interval
        expected_freq = 1.0 / (signal.params.period * bar_duration_seconds)
        
        # Find closest detected frequency
        if length(freq_analysis["dominant_frequencies"]) > 0
            detected_freq = freq_analysis["dominant_frequencies"][1]
            freq_error = abs(detected_freq - expected_freq)
            validation_results["frequency_error"] = freq_error
            validation_results["frequency_accurate"] = freq_error < 0.01
            validation_results["expected_frequency_hz"] = expected_freq
            validation_results["detected_frequency_hz"] = detected_freq
        end
    end
    
    return validation_results
end

# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

"""
Generate standard test signal for filter validation
"""
function generate_test_signal_for_filter(fib_number::Int; 
                                       n_bars::Int = 1000,
                                       ticks_per_bar::Int = 89,
                                       amplitude::Float64 = 100.0,
                                       tick_interval::Float64 = DEFAULT_TICK_INTERVAL)::SyntheticSignal
    # Calculate filter period (with doubling)
    period = fib_number == 1 ? 2.01 : 2.0 * fib_number
    
    # Generate pure sine at filter frequency
    return generate_synthetic_signal(
        n_bars = n_bars,
        ticks_per_bar = ticks_per_bar,
        signal_type = :pure_sine,
        signal_params = Dict(
            :period => period,
            :amplitude => amplitude,
            :phase => 0.0
        ),
        tick_interval = tick_interval
    )
end

"""
Generate comprehensive test signal set for multiple filters
"""
function generate_test_signal_set(fib_numbers::Vector{Int};
                                 n_bars::Int = 2000,
                                 ticks_per_bar::Int = 89,
                                 tick_interval::Float64 = DEFAULT_TICK_INTERVAL)::Dict{String, SyntheticSignal}
    
    test_signals = Dict{String, SyntheticSignal}()
    
    # Individual pure sines for each Fibonacci number
    for fib_num in fib_numbers
        signal = generate_test_signal_for_filter(fib_num, n_bars=n_bars, ticks_per_bar=ticks_per_bar, 
                                                tick_interval=tick_interval)
        test_signals["pure_sine_fib$fib_num"] = signal
    end
    
    # Mixture of all Fibonacci frequencies
    mixture_signal = generate_synthetic_signal(
        n_bars = n_bars,
        ticks_per_bar = ticks_per_bar,
        signal_type = :fibonacci_mixture,
        signal_params = Dict(:fib_numbers => fib_numbers),
        tick_interval = tick_interval
    )
    test_signals["fibonacci_mixture"] = mixture_signal
    
    # Market simulation
    market_signal = generate_synthetic_signal(
        n_bars = n_bars,
        ticks_per_bar = ticks_per_bar,
        signal_type = :market_simulation,
        signal_params = Dict(:fib_components => fib_numbers),
        tick_interval = tick_interval
    )
    test_signals["market_simulation"] = market_signal
    
    # Noisy versions
    for noise_type in ["am", "pm", "both"]
        noise_params = Dict{Symbol, Float64}()
        
        if noise_type in ["am", "both"]
            noise_params[:am_depth] = 0.2
            noise_params[:am_frequency] = 0.05  # Hz
        end
        
        if noise_type in ["pm", "both"]
            noise_params[:pm_deviation] = 0.3
            noise_params[:pm_frequency] = 0.05  # Hz
        end
        
        noisy_signal = generate_synthetic_signal(
            n_bars = n_bars,
            ticks_per_bar = ticks_per_bar,
            signal_type = :fibonacci_mixture,
            signal_params = Dict(:fib_numbers => fib_numbers),
            noise_params = noise_params,
            tick_interval = tick_interval
        )
        test_signals["mixture_$(noise_type)_noise"] = noisy_signal
    end
    
    return test_signals
end

"""
Print signal summary for debugging
"""
function print_signal_summary(signal::SyntheticSignal)
    println("="^60)
    println("SYNTHETIC SIGNAL SUMMARY")
    println("="^60)
    println("Type: $(signal.params.signal_type)")
    println("Bars: $(signal.params.n_bars)")
    println("Ticks per bar: $(signal.params.ticks_per_bar)")
    println("Tick interval: $(round(signal.params.tick_interval, digits=4)) seconds")
    println("Bar duration: $(round(signal.params.ticks_per_bar * signal.params.tick_interval, digits=2)) seconds")
    
    if signal.params.signal_type == :pure_sine
        println("Period: $(signal.params.period) bars")
        println("Amplitude: $(signal.params.amplitude)")
    elseif signal.params.signal_type == :fibonacci_mixture
        println("Fibonacci components: $(signal.params.fib_numbers)")
    end
    
    validation = validate_signal(signal)
    println("\nSignal Statistics:")
    println("  Mean: $(round(validation["mean"], digits=2))")
    println("  Std: $(round(validation["std"], digits=2))")
    println("  Range: [$(round(validation["min"], digits=2)), $(round(validation["max"], digits=2))]")
    
    if haskey(validation, "frequency_accurate")
        println("\nFrequency Analysis:")
        println("  Expected: $(round(validation["expected_frequency_hz"], digits=6)) Hz")
        println("  Detected: $(round(validation["detected_frequency_hz"], digits=6)) Hz")
        println("  Accuracy: $(validation["frequency_accurate"] ? "✓" : "✗")")
    end
    
    println("="^60)
end

# SyntheticSignalGenerator_patch.jl
# Add these functions to SyntheticSignalGenerator.jl for proper Complex I/Q signal generation
# with 4-phase rotation matching TickHotLoopF32 format

# Add to exports:
# export generate_complex_iq_signal, create_test_signal_complex_iq,
#        apply_4phase_rotation, normalize_price_change

# =============================================================================
# 4-PHASE ROTATION HELPERS (matching TickHotLoopF32)
# =============================================================================

"""
Get 4-phase position for tick index (1-based)
Position ∈ {1,2,3,4} → {0°, 90°, 180°, 270°}
"""
function phase_pos_global(tick_idx::Int64)::Int32
    return Int32(((tick_idx - 1) & 0x3) + 1)
end

# Unit complex multipliers for the four quadrants
const QUAD4 = (ComplexF32(1,0), ComplexF32(0,1), ComplexF32(-1,0), ComplexF32(0,-1))

"""
Apply 4-phase rotation to normalized value
"""
function apply_quad_phase(normalized_value::Float32, pos::Int32)::ComplexF32
    q = QUAD4[pos]  # q ∈ {1, i, -1, -i}
    return ComplexF32(normalized_value * real(q), normalized_value * imag(q))
end

# =============================================================================
# COMPLEX I/Q SIGNAL GENERATION
# =============================================================================

"""
Generate Complex I/Q signal matching TickHotLoopF32 format
Real part: Normalized price change [-1, +1]
Imaginary part: 4-phase rotated volume (always 1 tick)
"""
function generate_complex_iq_signal(;
    n_ticks::Int,
    signal_type::Symbol = :pure_sine,
    period::Float32 = 26.0f0,  # Filter period in ticks
    amplitude::Float32 = 50.0f0,  # Price change amplitude in ticks
    noise_level::Float32 = 0.0f0,
    normalization_scale::Float32 = 50.0f0,  # Typical price change scale
    random_seed::Union{Int, Nothing} = nothing
)::Vector{ComplexF32}
    
    # Set random seed if provided
    if random_seed !== nothing
        Random.seed!(random_seed)
    end
    
    # Generate price changes
    t = Float32.(0:n_ticks-1)
    frequency = Float32(2π) / period
    
    if signal_type == :pure_sine
        # Pure sine wave price changes
        price_changes = amplitude * sin.(frequency .* t)
        
    elseif signal_type == :noisy_sine
        # Sine with noise
        price_changes = amplitude * sin.(frequency .* t)
        price_changes .+= noise_level * amplitude * randn(Float32, n_ticks)
        
    elseif signal_type == :fibonacci_mixture
        # Multiple frequency components
        price_changes = zeros(Float32, n_ticks)
        fib_periods = Float32[4, 6, 10, 16, 26, 42, 68]
        for (i, fib_period) in enumerate(fib_periods)
            freq = Float32(2π) / fib_period
            # Decreasing amplitude for higher frequencies
            amp = amplitude / Float32(i)
            price_changes .+= amp * sin.(freq .* t .+ randn() * 2π)
        end
        
    elseif signal_type == :market_like
        # Market-like with trends and mean reversion
        price_changes = zeros(Float32, n_ticks)
        current_level = 0.0f0
        mean_level = 0.0f0
        mean_reversion_rate = 0.01f0
        trend = 0.1f0
        
        for i in 1:n_ticks
            # Trend component
            mean_level += trend
            
            # Mean reversion
            reversion = -mean_reversion_rate * (current_level - mean_level)
            
            # Random walk
            random_walk = amplitude * 0.1f0 * randn()
            
            # Fibonacci oscillation
            fib_component = amplitude * 0.5f0 * sin(frequency * t[i])
            
            # Update level
            current_level += trend + reversion + random_walk + fib_component
            price_changes[i] = current_level
        end
        
    else
        error("Unknown signal type: $signal_type")
    end
    
    # Generate Complex I/Q signal with 4-phase rotation
    signal = Vector{ComplexF32}(undef, n_ticks)
    
    for tick_idx in 1:n_ticks
        # Normalize price change to [-1, +1] range
        normalized_price = clamp(price_changes[tick_idx] / normalization_scale, -1.0f0, 1.0f0)
        
        # Get 4-phase position
        pos = phase_pos_global(Int64(tick_idx))
        
        # Apply 4-phase rotation to normalized value
        # This rotates the normalized price through the complex plane
        signal[tick_idx] = apply_quad_phase(normalized_price, pos)
    end
    
    return signal
end

"""
Create test signal for filter evaluation with proper Complex I/Q format
"""
function create_test_signal_complex_iq(
    fibonacci_number::Int32;
    n_ticks::Int = 1000,
    signal_type::Symbol = :pure_sine,
    amplitude_factor::Float32 = 1.0f0
)::Vector{ComplexF32}
    
    # Calculate filter period (with doubling)
    period = fibonacci_number == 1 ? 2.01f0 : Float32(2 * fibonacci_number)
    
    # Typical amplitude scales with period
    amplitude = Float32(10 + fibonacci_number * 2) * amplitude_factor
    
    # Generate signal
    return generate_complex_iq_signal(
        n_ticks = n_ticks,
        signal_type = signal_type,
        period = period,
        amplitude = amplitude,
        normalization_scale = amplitude  # Self-normalizing
    )
end

"""
Convert traditional signal to Complex I/Q format with 4-phase rotation
Useful for adapting existing test signals
"""
function convert_to_complex_iq(
    price_signal::Vector{Float64},
    ticks_per_bar::Int;
    normalization_scale::Float32 = 50.0f0
)::Vector{ComplexF32}
    
    n_bars = length(price_signal)
    n_ticks = n_bars * ticks_per_bar
    
    # Interpolate price signal to tick level
    tick_prices = zeros(Float32, n_ticks)
    for bar in 1:n_bars
        start_idx = (bar - 1) * ticks_per_bar + 1
        end_idx = bar * ticks_per_bar
        
        if bar == 1
            # First bar: assume no change
            tick_prices[start_idx:end_idx] .= Float32(price_signal[1])
        else
            # Linear interpolation between bars
            start_price = Float32(price_signal[bar-1])
            end_price = Float32(price_signal[bar])
            for i in 0:(ticks_per_bar-1)
                alpha = Float32(i) / Float32(ticks_per_bar)
                tick_prices[start_idx + i] = start_price + alpha * (end_price - start_price)
            end
        end
    end
    
    # Calculate price changes
    price_changes = zeros(Float32, n_ticks)
    price_changes[2:end] = diff(tick_prices)
    
    # Convert to Complex I/Q with 4-phase rotation
    signal = Vector{ComplexF32}(undef, n_ticks)
    
    for tick_idx in 1:n_ticks
        # Normalize price change
        normalized_price = clamp(price_changes[tick_idx] / normalization_scale, -1.0f0, 1.0f0)
        
        # Get 4-phase position
        pos = phase_pos_global(Int64(tick_idx))
        
        # Apply 4-phase rotation
        signal[tick_idx] = apply_quad_phase(normalized_price, pos)
    end
    
    return signal
end

"""
Analyze Complex I/Q signal properties
"""
function analyze_complex_iq_signal(signal::Vector{ComplexF32})::Dict{String, Any}
    n_ticks = length(signal)
    
    # Extract components
    real_parts = real.(signal)
    imag_parts = imag.(signal)
    magnitudes = abs.(signal)
    phases = angle.(signal)
    
    # Statistics
    stats = Dict{String, Any}()
    
    # Real part (normalized price changes)
    stats["real_mean"] = mean(real_parts)
    stats["real_std"] = std(real_parts)
    stats["real_min"] = minimum(real_parts)
    stats["real_max"] = maximum(real_parts)
    stats["real_rms"] = sqrt(mean(real_parts.^2))
    
    # Imaginary part (4-phase rotation)
    stats["imag_mean"] = mean(imag_parts)
    stats["imag_std"] = std(imag_parts)
    stats["imag_unique"] = length(unique(round.(imag_parts, digits=3)))
    
    # Magnitude statistics
    stats["mag_mean"] = mean(magnitudes)
    stats["mag_std"] = std(magnitudes)
    stats["mag_max"] = maximum(magnitudes)
    
    # Phase distribution
    stats["phase_mean"] = mean(phases)
    stats["phase_std"] = std(phases)
    
    # 4-phase rotation verification
    # Check if phases follow expected pattern
    expected_phases = [0, π/2, π, -π/2]
    phase_errors = Float32[]
    for i in 1:min(100, n_ticks)
        pos = phase_pos_global(Int64(i))
        expected = expected_phases[pos]
        if magnitudes[i] > 1e-6  # Only check non-zero signals
            actual = phases[i]
            # Wrap phase difference to [-π, π]
            diff = actual - expected
            while diff > π; diff -= 2π; end
            while diff < -π; diff += 2π; end
            push!(phase_errors, abs(diff))
        end
    end
    
    if !isempty(phase_errors)
        stats["phase_alignment_error"] = mean(phase_errors)
        stats["phase_alignment_correct"] = mean(phase_errors .< 0.1)
    else
        stats["phase_alignment_error"] = 0.0
        stats["phase_alignment_correct"] = 1.0
    end
    
    # Signal quality metrics
    stats["signal_power"] = mean(magnitudes.^2)
    stats["signal_energy"] = sum(magnitudes.^2)
    stats["n_ticks"] = n_ticks
    
    return stats
end

"""
Generate test signal set for Chunk 4 testing
"""
function generate_test_signals_complex_iq(
    fibonacci_numbers::Vector{Int32};
    n_ticks::Int = 5000,
    test_types::Vector{Symbol} = [:pure_sine, :noisy_sine, :fibonacci_mixture]
)::Dict{String, Vector{ComplexF32}}
    
    signals = Dict{String, Vector{ComplexF32}}()
    
    # Generate individual filter test signals
    for fib_num in fibonacci_numbers
        for test_type in test_types
            key = "fib$(fib_num)_$(test_type)"
            signals[key] = create_test_signal_complex_iq(
                fib_num,
                n_ticks = n_ticks,
                signal_type = test_type
            )
            println("  Generated: $key ($(n_ticks) ticks)")
        end
    end
    
    # Generate combined signal
    combined = zeros(ComplexF32, n_ticks)
    for fib_num in fibonacci_numbers
        weight = 1.0f0 / sqrt(Float32(fib_num))  # Higher frequencies get less weight
        signal = create_test_signal_complex_iq(
            fib_num,
            n_ticks = n_ticks,
            signal_type = :pure_sine,
            amplitude_factor = weight
        )
        combined .+= signal
    end
    
    # Normalize combined signal
    max_mag = maximum(abs.(combined))
    if max_mag > 0
        combined ./= max_mag
    end
    
    signals["combined"] = combined
    println("  Generated: combined signal")
    
    return signals
end

end # module SyntheticSignalGenerator