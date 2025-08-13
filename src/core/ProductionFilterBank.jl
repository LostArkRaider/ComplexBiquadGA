# src/ProductionFilterBank.jl - Modern Filter Bank Implementation (Single Stream)

"""
Production Fibonacci Filter Bank with Direct Struct Configuration

This module provides a single-stream filter bank system that:
- Processes only tick stream I/Q complex signals
- Implements proper split signal path for PLL:
  * Clamped signal → Phase Detector
  * Unclamped signal → Filter input (preserves AM)
- Uses direct struct configuration (no dictionary conversions)
- Maintains type safety with compile-time validation

Key Features:
- Single stream processing (removed dX/dY)
- Proper PLL split signal path
- Direct field access: config.processing.fibonacci_periods
- Period doubling: [1,2,3,5,8...] → [2.01,4,6,10,16...]

Usage:
    config = load_filter_config("pll")
    bank = PLLFibonacciFilterBank(config)
    update!(bank, z_complex)
    outputs = get_band_outputs(bank)
"""

module ProductionFilterBank

using DataFrames
using JLD2
using DSP
using LinearAlgebra
using Statistics
using Dates
using Printf
using ProgressMeter

# Import common statistical functions for convenience
import Statistics: mean, std, var

# Use the ModernConfigSystem module
if !isdefined(Main, :ModernConfigSystem)
    include("ModernConfigSystem.jl")
end
using Main.ModernConfigSystem

export FibonacciFilterBank, PLLFibonacciFilterBank,
       update!, get_band_outputs, num_bands,
       process_ym_data, create_filter_bank,
       ProcessingMetadata

# =============================================================================
# UTILITY STRUCTURES
# =============================================================================

"""
Apply consistent period doubling for all filter implementations
Fibonacci numbers are doubled to get actual filter periods
Special handling for period 1 to avoid Nyquist issues
"""
function apply_period_doubling(fibonacci_number::Int)::Float64
    if fibonacci_number == 1
        # For Fibonacci number 1, add small fraction to avoid Nyquist
        return 2.01
    else
        return Float64(2 * fibonacci_number)
    end
end

"""
Apply period doubling to an array of Fibonacci numbers
"""
function apply_period_doubling(fibonacci_numbers::Vector{Int})::Vector{Float64}
    return [apply_period_doubling(n) for n in fibonacci_numbers]
end

"""
Processing metadata struct for type-safe metadata storage
"""
struct ProcessingMetadata
    # Processing info
    timestamp::DateTime
    julia_version::String
    configuration_type::String
    filter_bank_type::String
    processing_duration::Millisecond
    
    # Configuration info
    config_name::String
    config_description::String
    config_version::String
    fibonacci_periods::Vector{Int}
    actual_filter_periods::Vector{Float64}
    period_doubling_applied::Bool
    q_factor::Float64
    batch_size::Int
    include_diagnostics::Bool
    
    # Data info
    total_ticks::Int
    
    # File info
    input_file::String
    output_file::String
    compression_enabled::Bool
    
    # PLL performance (optional)
    pll_enabled::Bool
    pll_phase_detector_gain::Float64
    pll_loop_bandwidth::Float64
    pll_lock_threshold::Float64
    pll_clamping_enabled::Bool
    pll_average_lock_quality::Float64
    pll_high_quality_lock_percentage::Float64
    pll_total_ringing_events::Int
    pll_clamping_threshold::Float64
end

# =============================================================================
# MODERN FILTER BANK STRUCTURES (SINGLE STREAM ONLY)
# =============================================================================

"""
Complex biquad filter with direct parameter access
"""
mutable struct ComplexBiquad
    # Filter coefficients (designed once)
    b0::ComplexF64
    b1::ComplexF64
    b2::ComplexF64
    a1::ComplexF64
    a2::ComplexF64
    
    # State variables (updated each sample)
    x1::ComplexF64
    x2::ComplexF64
    y1::ComplexF64
    y2::ComplexF64
    
    # Metadata
    fibonacci_number::Int
    actual_period::Float64
    q_factor::Float64
    center_frequency::Float64
    name::String
    
    function ComplexBiquad(fibonacci_number::Int, q_factor::Float64, name::String = "Fib$fibonacci_number")
        # Apply period doubling
        actual_period = apply_period_doubling(fibonacci_number)
        
        # Design bandpass filter coefficients with doubled period
        coeffs = design_bandpass_coefficients(actual_period, q_factor)
        
        new(coeffs.b0, coeffs.b1, coeffs.b2, coeffs.a1, coeffs.a2,
            ComplexF64(0), ComplexF64(0), ComplexF64(0), ComplexF64(0),
            fibonacci_number, actual_period, q_factor, 2π / actual_period, name)
    end
end

"""
Standard filter bank with single stream processing
"""
mutable struct FibonacciFilterBank
    filters::Vector{ComplexBiquad}
    filter_names::Vector{String}
    fibonacci_numbers::Vector{Int}
    actual_periods::Vector{Float64}
    config::FilterConfig
    creation_time::DateTime
    
    function FibonacciFilterBank(config::FilterConfig)
        println("🔧 Creating standard Fibonacci filter bank (single stream)...")
        
        fibonacci_numbers = config.processing.fibonacci_periods
        actual_periods = apply_period_doubling(fibonacci_numbers)
        q_factor = config.processing.q_factor
        
        println("   Fibonacci numbers: $fibonacci_numbers")
        println("   Actual periods (2×): $actual_periods")
        
        # Create single set of filters
        filters = [ComplexBiquad(fib_num, q_factor, "Fib$(fib_num)") 
                  for fib_num in fibonacci_numbers]
        
        # Generate filter names
        filter_names = ["Fib$fib_num" for fib_num in fibonacci_numbers]
        
        println("✅ Created $(length(fibonacci_numbers)) filters")
        
        new(filters, filter_names, fibonacci_numbers, actual_periods, config, now())
    end
end

"""
PLL-enhanced filter state with adaptive tracking and split signal path
"""
mutable struct PLLFilterState
    base_filter::ComplexBiquad
    
    # PLL control state
    vco_phase::Float64
    vco_frequency::Float64
    center_frequency::Float64
    
    # PLL parameters (from config)
    phase_detector_gain::Float64
    loop_bandwidth::Float64
    lock_threshold::Float64
    max_frequency_deviation::Float64
    
    # Adaptive state tracking
    loop_integrator::Float64
    phase_error_history::Vector{Float64}
    phase_error_index::Int
    phase_error_count::Int
    phase_error_capacity::Int
    lock_quality::Float64
    
    # Ringing state
    is_ringing::Bool
    ring_amplitude::ComplexF64
    ring_decay::Float64
    
    function PLLFilterState(fibonacci_number::Int, pll_config::PLLConfig, name::String = "PLLFib$fibonacci_number")
        base = ComplexBiquad(fibonacci_number, 2.0, name)  # Fixed Q for PLL base filter
        actual_period = apply_period_doubling(fibonacci_number)
        center_freq = 2π / actual_period
        capacity = pll_config.phase_error_history_length
        
        new(base, 0.0, center_freq, center_freq,
            pll_config.phase_detector_gain, pll_config.loop_bandwidth, 
            pll_config.lock_threshold, pll_config.max_frequency_deviation,
            0.0, zeros(Float64, capacity), 1, 0, capacity, 0.0,
            false, ComplexF64(0), pll_config.ring_decay)
    end
end

"""
PLL-enhanced filter bank with adaptive capabilities (single stream)
"""
mutable struct PLLFibonacciFilterBank
    tick_filters::Vector{PLLFilterState}
    filter_names::Vector{String}
    fibonacci_numbers::Vector{Int}
    actual_periods::Vector{Float64}
    config::ExtendedFilterConfig
    creation_time::DateTime
    
    # Output buffer for efficient band output retrieval
    output_buffer::Vector{ComplexF32}
    
    function PLLFibonacciFilterBank(config::ExtendedFilterConfig)
        println("🔒 Creating PLL-enhanced Fibonacci filter bank (single stream)...")
        
        fibonacci_numbers = config.processing.fibonacci_periods
        actual_periods = apply_period_doubling(fibonacci_numbers)
        pll_config = config.pll
        
        println("   Fibonacci numbers: $fibonacci_numbers")
        println("   Actual periods (2×): $actual_periods")
        
        # Create single set of PLL filters
        tick_filters = [PLLFilterState(fib_num, pll_config, "PLLFib$(fib_num)") 
                       for fib_num in fibonacci_numbers]
        
        # Generate filter names
        filter_names = ["PLLFib$fib_num" for fib_num in fibonacci_numbers]
        
        # Pre-allocate output buffer
        output_buffer = Vector{ComplexF32}(undef, length(fibonacci_numbers))
        
        println("✅ Created $(length(fibonacci_numbers)) PLL filters")
        println("🎛️  PLL gain: $(pll_config.phase_detector_gain), bandwidth: $(pll_config.loop_bandwidth)")
        
        new(tick_filters, filter_names, fibonacci_numbers, actual_periods, 
            config, now(), output_buffer)
    end
end

# =============================================================================
# TICK-BASED PROCESSING FUNCTIONS WITH PROPER SPLIT SIGNAL PATH
# =============================================================================

"""
Clamp complex signal for phase detector input
Clamps ONLY the real part to {-1, 0, +1}
Preserves imaginary part for PD sync
"""
function clamp_for_pd(z::ComplexF32)::ComplexF32
    # Clamp real part to 3-level: {-1, 0, +1}
    real_clamped = Float32(sign(real(z)))
    
    # Keep imaginary part unchanged for PD sync
    imag_unchanged = imag(z)
    
    # Return clamped signal for phase detector
    return ComplexF32(real_clamped, imag_unchanged)
end

"""
Process tick through PLL filter with PROPER split signal path
- Unclamped signal goes to filter input (preserves AM)
- Clamped signal goes to phase detector
"""
function process_tick_pll!(filter::PLLFilterState, z::ComplexF32, enable_clamping::Bool = true)::ComplexF32
    # Initialize state if needed
    if filter.loop_integrator == 0.0 && filter.vco_frequency == 0.0
        filter.loop_integrator = 1e-6
        filter.vco_frequency = filter.center_frequency
    end
    
    # CRITICAL FIX: Process UNCLAMPED signal through filter to preserve AM
    filter_output = process_sample!(filter.base_filter, ComplexF64(z))
    
    # Phase detector processing with optional clamping
    if abs(z) > 1e-6  # Significant input
        filter.is_ringing = false
        
        # Split signal path:
        # - Clamped signal for phase detection (if enabled)
        # - Original signal already processed through filter
        if enable_clamping
            pd_input = clamp_for_pd(z)
            reference_phase = angle(ComplexF64(pd_input))
        else
            reference_phase = angle(ComplexF64(z))
        end
        
        # Phase error calculation using filter output
        if abs(filter_output) > 1e-6
            output_phase = angle(filter_output)
            phase_error = output_phase - filter.vco_phase
        else
            phase_error = reference_phase - filter.vco_phase
        end
        
        # Wrap phase error to [-π, π]
        while phase_error > π; phase_error -= 2π; end
        while phase_error < -π; phase_error += 2π; end
        
        # Update phase error history (store absolute value)
        if filter.phase_error_count < filter.phase_error_capacity
            filter.phase_error_count += 1
            filter.phase_error_history[filter.phase_error_count] = abs(phase_error)
        else
            filter.phase_error_history[filter.phase_error_index] = abs(phase_error)
            filter.phase_error_index = (filter.phase_error_index % filter.phase_error_capacity) + 1
        end
        
        # Update lock quality
        if filter.phase_error_count >= 5
            avg_error = mean(filter.phase_error_history[1:filter.phase_error_count])
            filter.lock_quality = exp(-2.0 * avg_error)
        end
        
        # PLL loop filter
        filter.loop_integrator += filter.loop_bandwidth * phase_error * 2.0
        
        # Frequency correction
        frequency_correction = filter.phase_detector_gain * phase_error * 1.5 + filter.loop_integrator
        filter.vco_frequency = filter.center_frequency + frequency_correction
        
        # Limit frequency deviation
        max_deviation = filter.center_frequency * filter.max_frequency_deviation
        filter.vco_frequency = clamp(filter.vco_frequency,
                                    filter.center_frequency - max_deviation,
                                    filter.center_frequency + max_deviation)
        
        # Store ring amplitude for potential ringing
        if filter.lock_quality > filter.lock_threshold * 0.5
            filter.ring_amplitude = filter_output * (1.0 + filter.lock_quality)
        end
    else
        # Ringing mode
        if filter.lock_quality > filter.lock_threshold * 0.3
            filter.is_ringing = true
            
            # Generate VCO output
            filter_output = filter.ring_amplitude * exp(im * filter.vco_phase)
            
            # Apply decay
            filter.ring_amplitude *= filter.ring_decay
            filter.lock_quality *= 0.995
        else
            # Minimal processing to maintain state
            filter_output = process_sample!(filter.base_filter, ComplexF64(1e-6))
        end
    end
    
    # Advance VCO phase
    filter.vco_phase += filter.vco_frequency
    filter.vco_phase = mod(filter.vco_phase, 2π)
    
    # Return enhanced output
    if filter.lock_quality > 0.1
        vco_enhancement = exp(im * filter.vco_phase) * filter.lock_quality
        return ComplexF32(filter_output * (1.0 + vco_enhancement * 0.5))
    else
        return ComplexF32(filter_output)
    end
end

"""
Update standard filter bank with tick data
"""
function update!(bank::FibonacciFilterBank, z::ComplexF32)
    for filter in bank.filters
        process_sample!(filter, ComplexF64(z))
    end
end

"""
Update PLL filter bank with tick data using proper split signal path
"""
function update!(bank::PLLFibonacciFilterBank, z::ComplexF32)
    enable_clamping = bank.config.pll.enable_clamping
    
    for (i, filter) in enumerate(bank.tick_filters)
        output = process_tick_pll!(filter, z, enable_clamping)
        # Store in output buffer for efficient retrieval
        bank.output_buffer[i] = output
    end
end

"""
Get band outputs from standard filter bank
Returns Vector{ComplexF32} with one output per band
"""
function get_band_outputs(bank::FibonacciFilterBank)::Vector{ComplexF32}
    outputs = Vector{ComplexF32}(undef, length(bank.filters))
    for (i, filter) in enumerate(bank.filters)
        # Return the most recent output (y1 state variable)
        outputs[i] = ComplexF32(filter.y1)
    end
    return outputs
end

"""
Get band outputs from PLL filter bank
Returns Vector{ComplexF32} with one output per band
"""
function get_band_outputs(bank::PLLFibonacciFilterBank)::Vector{ComplexF32}
    # Return copy of the output buffer (already updated in update!)
    return copy(bank.output_buffer)
end

"""
Get number of bands in standard filter bank
"""
function num_bands(bank::FibonacciFilterBank)::Int
    return length(bank.fibonacci_numbers)
end

"""
Get number of bands in PLL filter bank
"""
function num_bands(bank::PLLFibonacciFilterBank)::Int
    return length(bank.fibonacci_numbers)
end

# =============================================================================
# FILTER DESIGN AND PROCESSING
# =============================================================================

"""
Design stable bandpass filter coefficients with validation
NOTE: This function expects the already-doubled period value
"""
function design_bandpass_coefficients(period_bars::Float64, Q::Float64)
    # Normalized center frequency
    fc = 1.0 / period_bars
    
    # Ensure we don't exceed Nyquist
    if fc >= 0.5
        # Adjust period slightly to avoid aliasing
        adjusted_period = period_bars + 0.01
        fc = 1.0 / adjusted_period
        @info "Period $period_bars adjusted to $adjusted_period to avoid Nyquist frequency"
    end
    
    # Design parameters
    ωc = 2π * fc
    bandwidth = fc / Q
    α = sin(π * bandwidth) / cos(π * bandwidth)
    
    # Bandpass coefficients (normalized)
    b0 = α
    b1 = 0.0
    b2 = -α
    a0 = 1.0 + α
    a1 = -2.0 * cos(ωc)
    a2 = 1.0 - α
    
    # Normalize by a0
    b0 /= a0
    b1 /= a0
    b2 /= a0
    a1 /= a0
    a2 /= a0
    
    # Stability validation
    if abs(a2) >= 1.0 || abs(a1) >= (1.0 + a2)
        @warn "Filter may be unstable for period $period_bars, Q=$Q"
    end
    
    return (b0=ComplexF64(b0), b1=ComplexF64(b1), b2=ComplexF64(b2), 
            a1=ComplexF64(a1), a2=ComplexF64(a2))
end

"""
Process sample through standard biquad filter
"""
function process_sample!(filter::ComplexBiquad, input::ComplexF64)::ComplexF64
    # Direct Form II implementation
    output = filter.b0 * input + filter.b1 * filter.x1 + filter.b2 * filter.x2 -
             filter.a1 * filter.y1 - filter.a2 * filter.y2
    
    # Update state variables
    filter.x2 = filter.x1
    filter.x1 = input
    filter.y2 = filter.y1
    filter.y1 = output
    
    return output
end

# =============================================================================
# SIMPLIFIED BAR-BASED PROCESSING (Legacy Support)
# =============================================================================

"""
Process YM data with standard filter bank (bar-based legacy)
"""
function process_ym_data(config::FilterConfig)::Tuple{DataFrame, FibonacciFilterBank, Dict}
    println("🚀 Processing YM data with standard configuration: $(config.name)")
    println("="^70)
    
    try
        # Step 1: Load and validate data
        bars_df, raw_data = load_ym_data_robust(config.io.input_file)
        
        # Step 2: Create filter bank
        filter_bank = FibonacciFilterBank(config)
        
        # Step 3: Process bars
        bars_df = process_bars!(bars_df, filter_bank, config)
        
        # Step 4: Save results
        save_results_robust(bars_df, filter_bank, config, raw_data)
        
        println("✅ Standard processing completed successfully!")
        return bars_df, filter_bank, raw_data
        
    catch e
        println("❌ Standard processing failed: $e")
        rethrow(e)
    end
end

"""
Process YM data with PLL-enhanced filter bank (bar-based legacy)
"""
function process_ym_data(config::ExtendedFilterConfig)::Tuple{DataFrame, PLLFibonacciFilterBank, Dict}
    println("🚀 Processing YM data with PLL-enhanced configuration: $(config.name)")
    println("="^70)
    
    try
        # Step 1: Load and validate data
        bars_df, raw_data = load_ym_data_robust(config.io.input_file)
        
        # Step 2: Create PLL filter bank
        filter_bank = PLLFibonacciFilterBank(config)
        
        # Step 3: Process bars
        bars_df = process_bars!(bars_df, filter_bank, config)
        
        # Step 4: Save results
        save_results_robust(bars_df, filter_bank, config, raw_data)
        
        println("✅ PLL processing completed successfully!")
        return bars_df, filter_bank, raw_data
        
    catch e
        println("❌ PLL processing failed: $e")
        rethrow(e)
    end
end

# =============================================================================
# DATA LOADING (Simplified)
# =============================================================================

"""
Robust YM data loading with comprehensive error handling
"""
function load_ym_data_robust(filepath::String)::Tuple{DataFrame, Dict}
    println("🔄 Loading YM data: $(basename(filepath))")
    
    if !isfile(filepath)
        error("Input file not found: $filepath")
    end
    
    try
        # Ensure CodecZlib is loaded for compressed JLD2 files
        @eval using CodecZlib
    catch
        # Continue without compression support
    end
    
    try
        raw_data = JLD2.load(filepath)
        
        # Find main DataFrame
        bars_df = find_main_dataframe(raw_data)
        
        println("✅ Data loaded successfully: $(nrow(bars_df)) bars")
        return bars_df, raw_data
        
    catch e
        error("Failed to load YM data: $filepath\nError: $e")
    end
end

"""
Find the main DataFrame in JLD2 data
"""
function find_main_dataframe(data::Dict)::DataFrame
    # Priority search for DataFrame
    candidate_keys = ["bars", "filtered_bars", "data", "ym_bars", "market_data"]
    
    for key in candidate_keys
        if haskey(data, key) && isa(data[key], DataFrame)
            return data[key]
        end
    end
    
    # Search for any DataFrame
    for (key, value) in data
        if isa(value, DataFrame) && nrow(value) > 100
            return value
        end
    end
    
    error("No suitable DataFrame found")
end

# =============================================================================
# SIMPLIFIED BAR PROCESSING
# =============================================================================

"""
Process bars through filter bank (simplified single-stream)
"""
function process_bars!(df::DataFrame, filter_bank::Union{FibonacciFilterBank, PLLFibonacciFilterBank}, 
                      config::Union{FilterConfig, ExtendedFilterConfig})
    println("🎯 Processing $(nrow(df)) bars through filter bank...")
    
    n_bars = nrow(df)
    n_bands = num_bands(filter_bank)
    
    # Create simple complex input from price changes
    if !("price_change" in names(df))
        df.price_change = df.close - df.open
    end
    
    # Initialize output columns
    for b in 1:n_bands
        fib_num = filter_bank.fibonacci_numbers[b]
        col_name = Symbol("Fib$(fib_num)_output")
        df[!, col_name] = Vector{ComplexF64}(undef, n_bars)
    end
    
    # Process each bar
    progress = Progress(n_bars, desc="Processing bars...")
    
    for i in 1:n_bars
        # Create complex input (price change + volume)
        z = ComplexF32(df.price_change[i], df.volume[i] / 100.0)
        
        # Update filter bank
        update!(filter_bank, z)
        
        # Get outputs
        outputs = get_band_outputs(filter_bank)
        
        # Store results
        for b in 1:n_bands
            fib_num = filter_bank.fibonacci_numbers[b]
            col_name = Symbol("Fib$(fib_num)_output")
            df[i, col_name] = ComplexF64(outputs[b])
        end
        
        next!(progress)
    end
    
    finish!(progress)
    println("✅ Bar processing completed")
    return df
end

# =============================================================================
# RESULTS SAVING (Simplified)
# =============================================================================

"""
Save results with metadata
"""
function save_results_robust(df::DataFrame, 
                            filter_bank::Union{FibonacciFilterBank, PLLFibonacciFilterBank}, 
                            config::Union{FilterConfig, ExtendedFilterConfig}, 
                            raw_data::Dict)
    output_file = config.io.output_file
    
    # Create output directory if needed
    output_dir = dirname(output_file)
    if !isempty(output_dir) && !isdir(output_dir)
        mkpath(output_dir)
    end
    
    # Create metadata
    metadata = create_processing_metadata(config, filter_bank, nrow(df))
    
    try
        # Save with appropriate compression
        if config.io.compress_output
            JLD2.jldsave(output_file; filtered_bars=df, filter_bank=filter_bank, metadata=metadata)
        else
            JLD2.jldopen(output_file, "w"; compress=false) do file
                file["filtered_bars"] = df
                file["filter_bank"] = filter_bank
                file["metadata"] = metadata
            end
        end
        
        file_size_mb = round(stat(output_file).size / 1024^2, digits=1)
        println("💾 Results saved: $output_file ($(file_size_mb) MB)")
        
    catch e
        @warn "Failed to save to $output_file: $e"
        error("Save failed: $e")
    end
end

"""
Create processing metadata
"""
function create_processing_metadata(config::Union{FilterConfig, ExtendedFilterConfig}, 
                                   filter_bank::Union{FibonacciFilterBank, PLLFibonacciFilterBank},
                                   n_rows::Int)::ProcessingMetadata
    
    # PLL metrics (if applicable)
    if isa(config, ExtendedFilterConfig) && config.pll.enabled
        pll_enabled = true
        pll_phase_detector_gain = config.pll.phase_detector_gain
        pll_loop_bandwidth = config.pll.loop_bandwidth
        pll_lock_threshold = config.pll.lock_threshold
        pll_clamping_enabled = config.pll.enable_clamping
        pll_clamping_threshold = config.pll.clamping_threshold
        # These would be calculated from actual processing
        pll_average_lock_quality = 0.0
        pll_high_quality_lock_percentage = 0.0
        pll_total_ringing_events = 0
    else
        pll_enabled = false
        pll_phase_detector_gain = 0.0
        pll_loop_bandwidth = 0.0
        pll_lock_threshold = 0.0
        pll_clamping_enabled = false
        pll_clamping_threshold = 0.0
        pll_average_lock_quality = 0.0
        pll_high_quality_lock_percentage = 0.0
        pll_total_ringing_events = 0
    end
    
    return ProcessingMetadata(
        now(),
        string(VERSION),
        isa(config, ExtendedFilterConfig) ? "PLL-Enhanced" : "Standard",
        string(typeof(filter_bank)),
        now() - filter_bank.creation_time,
        config.name,
        config.description,
        config.version,
        config.processing.fibonacci_periods,
        filter_bank.actual_periods,
        true,  # period_doubling_applied
        config.processing.q_factor,
        config.processing.batch_size,
        config.processing.include_diagnostics,
        n_rows,
        config.io.input_file,
        config.io.output_file,
        config.io.compress_output,
        pll_enabled,
        pll_phase_detector_gain,
        pll_loop_bandwidth,
        pll_lock_threshold,
        pll_clamping_enabled,
        pll_average_lock_quality,
        pll_high_quality_lock_percentage,
        pll_total_ringing_events,
        pll_clamping_threshold
    )
end

# =============================================================================
# HELPER FUNCTION FOR CREATING FILTER BANKS
# =============================================================================

"""
Create appropriate filter bank based on configuration type
"""
function create_filter_bank(config::Union{FilterConfig, ExtendedFilterConfig})
    if isa(config, ExtendedFilterConfig) && config.pll.enabled
        return PLLFibonacciFilterBank(config)
    else
        return FibonacciFilterBank(config)
    end
end

end # module ProductionFilterBank