# src/ModernConfigSystem.jl - Type-Safe Configuration System

"""
Modern Type-Safe Configuration System for Fibonacci Filter Bank

Eliminates dictionary-based configurations and provides compile-time
type safety with comprehensive validation. Uses Parameters.jl for
constrained struct definitions and direct TOML integration.

Key Features:
- Pure struct-based parameter hierarchy
- Compile-time type validation
- Direct field access (no dictionary lookups)
- Comprehensive error reporting
- TOML persistence with automatic conversion

Usage:
    config = load_filter_config("default")
    result = process_ym_data(config)  # Direct struct passing
"""

module ModernConfigSystem

using Parameters
using TOML
using Dates
using Statistics

export FilterConfig, ExtendedFilterConfig, ProcessingConfig, PLLConfig, IOConfig,
       load_filter_config, save_filter_config, validate_config,
       create_default_configs, list_available_configs,
       show_config_summary

# =============================================================================
# CORE CONFIGURATION STRUCTURES
# =============================================================================

"""
Processing parameters with compile-time validation
"""
@with_kw struct ProcessingConfig
    fibonacci_periods::Vector{Int} = [1, 2, 3, 5, 8, 13, 21, 34, 55]  # Will be doubled to [2.01, 4, 6, 10, 16, 26, 42, 68, 110]
    @assert !isempty(fibonacci_periods) "fibonacci_periods cannot be empty"
    @assert all(p > 0 for p in fibonacci_periods) "All periods must be positive"
    @assert allunique(fibonacci_periods) "fibonacci_periods must be unique"
    
    q_factor::Float64 = 2.0
    @assert q_factor > 0.0 "q_factor must be positive"
    @assert q_factor ≤ 10.0 "q_factor > 10 may cause instability"
    
    sma_window::Int = 20
    @assert sma_window > 0 "sma_window must be positive"
    @assert sma_window ≤ 200 "sma_window > 200 may over-smooth data"
    
    batch_size::Int = 1000
    @assert batch_size > 0 "batch_size must be positive"
    
    # Feature flags
    include_diagnostics::Bool = true
    include_price_levels::Bool = true
    include_momentum::Bool = true
    include_volatility::Bool = true
end

"""
PLL-specific parameters with strict validation
"""
@with_kw struct PLLConfig
    enabled::Bool = false
    
    phase_detector_gain::Float64 = 0.1
    @assert 0.001 ≤ phase_detector_gain ≤ 1.0 "phase_detector_gain must be in [0.001, 1.0]"
    
    loop_bandwidth::Float64 = 0.01
    @assert 0.0001 ≤ loop_bandwidth ≤ 0.1 "loop_bandwidth must be in [0.0001, 0.1]"
    
    lock_threshold::Float64 = 0.7
    @assert 0.0 ≤ lock_threshold ≤ 1.0 "lock_threshold must be in [0.0, 1.0]"
    
    ring_decay::Float64 = 0.995
    @assert 0.9 < ring_decay < 1.0 "ring_decay must be in (0.9, 1.0)"
    
    enable_clamping::Bool = false
    clamping_threshold::Float64 = 1e-6
    @assert clamping_threshold > 0.0 "clamping_threshold must be positive"
    
    volume_scaling::Float64 = 1.0
    @assert volume_scaling > 0.0 "volume_scaling must be positive"
    
    # Advanced PLL parameters
    max_frequency_deviation::Float64 = 0.2
    @assert 0.0 < max_frequency_deviation ≤ 0.5 "max_frequency_deviation must be in (0.0, 0.5]"
    
    phase_error_history_length::Int = 20
    @assert phase_error_history_length > 0 "phase_error_history_length must be positive"
end

"""
I/O and performance configuration
"""
@with_kw struct IOConfig
    input_file::String = "data/YM_06-25_bars_market_time.jld2"
    output_file::String = "data/YM_06-25_fibonacci_filtered.jld2"
    
    save_intermediate::Bool = false
    compress_output::Bool = true
    parallel_processing::Bool = false
    
    max_memory_gb::Float64 = 12.0
    @assert max_memory_gb > 0.0 "max_memory_gb must be positive"
    
    log_progress::Bool = true
    progress_interval::Int = 1000
    @assert progress_interval > 0 "progress_interval must be positive"
    
    # Validation settings
    validate_inputs::Bool = true
    error_on_missing_file::Bool = true
    backup_on_overwrite::Bool = false
end

"""
Standard filter bank configuration
"""
@with_kw struct FilterConfig
    name::String = "default"
    description::String = "Default filter bank configuration"
    
    processing::ProcessingConfig = ProcessingConfig()
    pll::PLLConfig = PLLConfig()  # PLL disabled by default
    io::IOConfig = IOConfig()
    
    created::DateTime = now()
    version::String = "2.0"
end

"""
Extended filter bank configuration with PLL enabled
"""
@with_kw struct ExtendedFilterConfig
    name::String = "pll_enhanced"
    description::String = "PLL-enhanced filter bank configuration"
    
    processing::ProcessingConfig = ProcessingConfig()
    pll::PLLConfig = PLLConfig(enabled=true)  # PLL enabled by default
    io::IOConfig = IOConfig()
    
    created::DateTime = now()
    version::String = "2.0"
end

# =============================================================================
# CONFIGURATION LOADING AND SAVING
# =============================================================================

"""
Load configuration with automatic type detection
"""
function load_filter_config(config_name::String)::Union{FilterConfig, ExtendedFilterConfig}
    config_file = resolve_config_path(config_name)
    
    if !isfile(config_file)
        println("⚠️  Configuration file not found: $(basename(config_file))")
        println("🔧 Creating default configurations...")
        create_default_configs()
        config_file = resolve_config_path(config_name)
        
        if !isfile(config_file)
            error("Failed to create or find configuration: $config_name")
        end
    end
    
    println("📋 Loading configuration: $(basename(config_file))")
    
    try
        toml_data = TOML.parsefile(config_file)
        config = parse_toml_to_config(toml_data, config_name)
        validate_config(config)
        
        println("✅ Configuration loaded: $(config.name)")
        return config
        
    catch e
        error("Failed to load configuration '$config_name': $e")
    end
end

"""
Parse TOML data into appropriate configuration struct
"""
function parse_toml_to_config(toml_data::Dict, name::String)::Union{FilterConfig, ExtendedFilterConfig}
    # Extract sections with safe defaults
    metadata = get(toml_data, "metadata", Dict())
    processing_dict = get(toml_data, "processing", Dict())
    pll_dict = get(toml_data, "pll", Dict())
    clamping_dict = get(toml_data, "clamping", Dict())
    io_dict = get(toml_data, "io", Dict())
    features_dict = get(toml_data, "features", Dict())
    performance_dict = get(toml_data, "performance", Dict())
    validation_dict = get(toml_data, "validation", Dict())
    
    # Build processing configuration
    processing = ProcessingConfig(
        fibonacci_periods = Vector{Int}(get(processing_dict, "fibonacci_periods", [3, 5, 8, 13, 21, 34, 55])),
        q_factor = Float64(get(processing_dict, "Q_factor", 2.0)),
        sma_window = Int(get(processing_dict, "sma_window", 20)),
        batch_size = Int(get(performance_dict, "batch_size", 1000)),
        include_diagnostics = Bool(get(features_dict, "include_diagnostics", true)),
        include_price_levels = Bool(get(features_dict, "include_price_levels", true)),
        include_momentum = Bool(get(features_dict, "include_momentum", true)),
        include_volatility = Bool(get(features_dict, "include_volatility", true))
    )
    
    # Build PLL configuration
    pll_enabled = Bool(get(pll_dict, "enable_pll", false))
    pll = PLLConfig(
        enabled = pll_enabled,
        phase_detector_gain = Float64(get(pll_dict, "phase_detector_gain", 0.1)),
        loop_bandwidth = Float64(get(pll_dict, "loop_bandwidth", 0.01)),
        lock_threshold = Float64(get(pll_dict, "lock_threshold", 0.7)),
        ring_decay = Float64(get(pll_dict, "ring_decay", 0.995)),
        enable_clamping = Bool(get(clamping_dict, "enable_clamping", false)),
        clamping_threshold = Float64(get(clamping_dict, "clamping_threshold", 1e-6)),
        volume_scaling = Float64(get(clamping_dict, "volume_scaling", 1.0)),
        max_frequency_deviation = Float64(get(pll_dict, "max_frequency_deviation", 0.2)),
        phase_error_history_length = Int(get(pll_dict, "phase_error_history_length", 20))
    )
    
    # Build I/O configuration
    io = IOConfig(
        input_file = String(get(io_dict, "input_file", "data/YM_06-25_bars_market_time.jld2")),
        output_file = String(get(io_dict, "output_file", "data/YM_06-25_fibonacci_filtered.jld2")),
        save_intermediate = Bool(get(io_dict, "save_intermediate", false)),
        compress_output = Bool(get(io_dict, "compress_output", true)),
        parallel_processing = Bool(get(performance_dict, "parallel_processing", false)),
        max_memory_gb = Float64(get(validation_dict, "max_memory_gb", 8.0)),
        log_progress = Bool(get(performance_dict, "log_progress", true)),
        progress_interval = Int(get(performance_dict, "progress_interval", 1000)),
        validate_inputs = Bool(get(validation_dict, "validate_inputs", true)),
        error_on_missing_file = Bool(get(validation_dict, "error_on_missing_file", true)),
        backup_on_overwrite = Bool(get(io_dict, "backup_on_overwrite", false))
    )
    
    # Determine configuration type and create appropriate struct
    config_name = String(get(metadata, "name", name))
    description = String(get(metadata, "description", "Configuration loaded from TOML"))
    
    if pll_enabled
        return ExtendedFilterConfig(
            name = config_name,
            description = description,
            processing = processing,
            pll = pll,
            io = io
        )
    else
        return FilterConfig(
            name = config_name,
            description = description,
            processing = processing,
            pll = pll,
            io = io
        )
    end
end

"""
Save configuration to TOML file with proper type handling
"""
function save_filter_config(config::Union{FilterConfig, ExtendedFilterConfig}, filename::String)
    # Ensure directory exists
    config_dir = dirname(filename)
    if !isempty(config_dir) && !isdir(config_dir)
        mkpath(config_dir)
    end
    
    # Convert struct to TOML-compatible dictionary
    toml_dict = struct_to_toml_dict(config)
    
    # Write to file
    open(filename, "w") do f
        TOML.print(f, toml_dict)
    end
    
    println("💾 Configuration saved: $filename")
end

"""
Convert configuration struct to TOML dictionary
"""
function struct_to_toml_dict(config::Union{FilterConfig, ExtendedFilterConfig})::Dict{String, Any}
    return Dict{String, Any}(
        "metadata" => Dict(
            "name" => config.name,
            "description" => config.description,
            "version" => config.version,
            "created" => string(config.created),
            "config_type" => isa(config, ExtendedFilterConfig) ? "extended" : "standard"
        ),
        "processing" => Dict(
            "fibonacci_periods" => config.processing.fibonacci_periods,
            "Q_factor" => config.processing.q_factor,
            "sma_window" => config.processing.sma_window
        ),
        "features" => Dict(
            "include_diagnostics" => config.processing.include_diagnostics,
            "include_price_levels" => config.processing.include_price_levels,
            "include_momentum" => config.processing.include_momentum,
            "include_volatility" => config.processing.include_volatility
        ),
        "io" => Dict(
            "input_file" => config.io.input_file,
            "output_file" => config.io.output_file,
            "save_intermediate" => config.io.save_intermediate,
            "compress_output" => config.io.compress_output,
            "backup_on_overwrite" => config.io.backup_on_overwrite
        ),
        "performance" => Dict(
            "batch_size" => config.processing.batch_size,
            "parallel_processing" => config.io.parallel_processing,
            "log_progress" => config.io.log_progress,
            "progress_interval" => config.io.progress_interval
        ),
        "validation" => Dict(
            "validate_inputs" => config.io.validate_inputs,
            "max_memory_gb" => config.io.max_memory_gb,
            "error_on_missing_file" => config.io.error_on_missing_file
        ),
        "pll" => Dict(
            "enable_pll" => config.pll.enabled,
            "phase_detector_gain" => config.pll.phase_detector_gain,
            "loop_bandwidth" => config.pll.loop_bandwidth,
            "lock_threshold" => config.pll.lock_threshold,
            "ring_decay" => config.pll.ring_decay,
            "max_frequency_deviation" => config.pll.max_frequency_deviation,
            "phase_error_history_length" => config.pll.phase_error_history_length
        ),
        "clamping" => Dict(
            "enable_clamping" => config.pll.enable_clamping,
            "clamping_threshold" => config.pll.clamping_threshold,
            "volume_scaling" => config.pll.volume_scaling
        )
    )
end

# =============================================================================
# CONFIGURATION VALIDATION
# =============================================================================

"""
Comprehensive configuration validation with detailed error reporting
"""
function validate_config(config::Union{FilterConfig, ExtendedFilterConfig})::Bool
    errors = String[]
    warnings = String[]
    
    # File system validation
    validate_file_system(config, errors, warnings)
    
    # Processing validation
    validate_processing_parameters(config, errors, warnings)
    
    # PLL validation (if enabled)
    if config.pll.enabled
        validate_pll_parameters(config, errors, warnings)
    end
    
    # Memory estimation
    validate_memory_requirements(config, errors, warnings)
    
    # Filter stability analysis
    validate_filter_stability(config, errors, warnings)
    
    # Report results
    if !isempty(errors)
        error("Configuration validation failed:\n" * join(errors, "\n"))
    end
    
    if !isempty(warnings)
        println("⚠️  Configuration warnings:")
        for warning in warnings
            println("   - $warning")
        end
    end
    
    println("✅ Configuration validation passed")
    return true
end

function validate_file_system(config::Union{FilterConfig, ExtendedFilterConfig}, 
                             errors::Vector{String}, warnings::Vector{String})
    # Input file validation
    if config.io.error_on_missing_file && !isfile(config.io.input_file)
        push!(errors, "Input file not found: $(config.io.input_file)")
    elseif !isfile(config.io.input_file)
        push!(warnings, "Input file not found: $(config.io.input_file)")
    end
    
    # Output directory validation
    output_dir = dirname(config.io.output_file)
    if !isempty(output_dir) && !isdir(output_dir)
        push!(warnings, "Output directory does not exist: $output_dir (will be created)")
    end
    
    # File overwrite protection
    if isfile(config.io.output_file) && !config.io.backup_on_overwrite
        push!(warnings, "Output file exists and will be overwritten: $(config.io.output_file)")
    end
end

function validate_processing_parameters(config::Union{FilterConfig, ExtendedFilterConfig}, 
                                      errors::Vector{String}, warnings::Vector{String})
    # Period validation
    max_period = maximum(config.processing.fibonacci_periods)
    if max_period > 200
        push!(warnings, "Very large period ($max_period) may require long data sequences")
    end
    
    # Q factor stability warnings
    if config.processing.q_factor > 5.0
        push!(warnings, "High Q factor ($(config.processing.q_factor)) may cause filter instability")
    end
    
    # Fibonacci sequence validation
    fib_sequence = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377]
    non_fib_periods = filter(p -> !(p in fib_sequence), config.processing.fibonacci_periods)
    if !isempty(non_fib_periods)
        push!(warnings, "Non-Fibonacci periods detected: $non_fib_periods")
    end
end

function validate_pll_parameters(config::Union{FilterConfig, ExtendedFilterConfig}, 
                                errors::Vector{String}, warnings::Vector{String})
    pll = config.pll
    
    # Stability analysis
    gain_bandwidth_product = pll.phase_detector_gain * pll.loop_bandwidth
    if gain_bandwidth_product > 0.05
        push!(warnings, "High gain×bandwidth product ($gain_bandwidth_product) may cause PLL instability")
    end
    
    # Lock threshold analysis
    if pll.lock_threshold < 0.5
        push!(warnings, "Low lock threshold ($(pll.lock_threshold)) may cause poor tracking")
    elseif pll.lock_threshold > 0.9
        push!(warnings, "Very high lock threshold ($(pll.lock_threshold)) may prevent locking")
    end
    
    # Clamping validation
    if pll.enable_clamping && pll.clamping_threshold > 1e-3
        push!(warnings, "High clamping threshold ($(pll.clamping_threshold)) may affect signal fidelity")
    end
end

function validate_memory_requirements(config::Union{FilterConfig, ExtendedFilterConfig}, 
                                    errors::Vector{String}, warnings::Vector{String})
    estimated_memory = estimate_memory_usage(config)
    
    if estimated_memory > config.io.max_memory_gb
        push!(errors, "Estimated memory usage ($(round(estimated_memory, digits=1))GB) exceeds limit ($(config.io.max_memory_gb)GB)")
    elseif estimated_memory > config.io.max_memory_gb * 0.8
        push!(warnings, "Estimated memory usage ($(round(estimated_memory, digits=1))GB) near limit ($(config.io.max_memory_gb)GB)")
    end
end

function validate_filter_stability(config::Union{FilterConfig, ExtendedFilterConfig}, 
                                  errors::Vector{String}, warnings::Vector{String})
    # Check each filter period for potential stability issues
    # NOTE: Periods are doubled internally, so we check the actual filter periods
    for fibonacci_num in config.processing.fibonacci_periods
        actual_period = if fibonacci_num == 1
            2.01  # Special case for Fibonacci 1
        else
            2.0 * fibonacci_num
        end
        
        fc = 1.0 / actual_period
        
        # Nyquist frequency check with doubled periods
        if fc >= 0.5
            # This should be very rare with doubled periods and the 2.01 adjustment
            push!(errors, "Fibonacci number $fibonacci_num (period $actual_period) causes aliasing (fc > Nyquist frequency)")
        elseif fc >= 0.45
            push!(warnings, "Fibonacci number $fibonacci_num (period $actual_period) near Nyquist frequency (fc = $(round(fc, digits=3)))")
        end
        
        # Q factor stability for doubled period
        bandwidth = fc / config.processing.q_factor
        if bandwidth < 0.001
            push!(warnings, "Very narrow bandwidth for Fibonacci $fibonacci_num (period $actual_period, BW=$(round(bandwidth, digits=4))) may cause instability")
        end
    end
end

"""
Estimate memory usage based on configuration parameters
"""
function estimate_memory_usage(config::Union{FilterConfig, ExtendedFilterConfig})::Float64
    # Assumptions for estimation based on actual data size
    estimated_bars = 62000  # Based on your actual data
    n_filters = length(config.processing.fibonacci_periods)
    
    # Base memory requirements (original data)
    # 36 columns * 8 bytes * 62000 rows
    base_memory_mb = estimated_bars * 36 * 8 / 1024^2
    
    # Complex input columns (3 complex columns)
    complex_input_mb = estimated_bars * 3 * 16 / 1024^2
    
    # Filter output memory (2 complex columns per filter)
    filter_memory_mb = estimated_bars * n_filters * 2 * 16 / 1024^2
    
    # PLL-specific memory (additional columns if enabled)
    pll_memory_mb = 0.0
    if config.pll.enabled
        # lock_quality, is_ringing per filter + system columns
        pll_memory_mb = estimated_bars * n_filters * 2 * 8 / 1024^2  # 2 Float64 columns per filter
        pll_memory_mb += estimated_bars * 3 * 8 / 1024^2  # system_lock_quality, ringing_count, clamped_active_count
    end
    
    # Diagnostic columns if enabled
    diagnostic_memory_mb = 0.0
    if config.processing.include_diagnostics
        diagnostic_memory_mb = estimated_bars * 4 * 8 / 1024^2  # 4 diagnostic columns
    end
    
    # Intermediate calculation memory (roughly 50% overhead)
    overhead_memory_mb = (base_memory_mb + complex_input_mb + filter_memory_mb + pll_memory_mb + diagnostic_memory_mb) * 0.5
    
    total_memory_gb = (base_memory_mb + complex_input_mb + filter_memory_mb + pll_memory_mb + diagnostic_memory_mb + overhead_memory_mb) / 1024
    
    return total_memory_gb
end

# =============================================================================
# CONFIGURATION MANAGEMENT
# =============================================================================

"""
Resolve configuration file path with fallback logic
"""
function resolve_config_path(config_name::String)::String
    if endswith(config_name, ".toml")
        return config_name
    end
    
    # Search paths in order of preference
    search_paths = [
        "config/$config_name.toml",
        joinpath(@__DIR__, "..", "config", "$config_name.toml"),
        "$config_name.toml"
    ]
    
    for path in search_paths
        if isfile(path)
            return path
        end
    end
    
    # Return preferred path for creation
    return "config/$config_name.toml"
end

"""
List available configuration files
"""
function list_available_configs()::Vector{String}
    config_dir = "config"
    if !isdir(config_dir)
        return String[]
    end
    
    config_files = filter(f -> endswith(f, ".toml"), readdir(config_dir))
    return [replace(f, ".toml" => "") for f in config_files]
end

"""
Create default configuration presets
"""
function create_default_configs()
    config_dir = "config"
    if !isdir(config_dir)
        mkpath(config_dir)
        println("📁 Created config directory: $config_dir")
    end
    
    # Define configuration presets (adjusted for period doubling)
    presets = [
        ("default", "Balanced configuration for general use", false),
        ("fast", "Fast processing with fewer periods", false),
        ("comprehensive", "Extended analysis with more periods", false),
        ("high_q", "High Q factor for sharp frequency response", false),
        ("pll", "PLL-enhanced configuration", true),
        ("pll_fast", "Fast PLL processing", true),
        ("pll_precision", "High-precision PLL processing", true)
    ]
    
    configs_created = 0
    for (name, description, use_pll) in presets
        if create_config_preset(name, description, use_pll)
            configs_created += 1
        end
    end
    
    println("✅ Created $configs_created configuration presets")
end

"""
Create a specific configuration preset
"""
function create_config_preset(name::String, description::String, use_pll::Bool)::Bool
    config_file = "config/$name.toml"
    
    if isfile(config_file)
        return false  # Don't overwrite existing configs
    end
    
    # Define preset-specific parameters
    # NOTE: These Fibonacci numbers will be doubled internally to get actual filter periods
    # Period 1 becomes 2.01 to avoid Nyquist issues
    if name == "fast"
        periods = [1, 2, 3, 5, 8, 13, 21]  # Actual periods: [2.01, 4, 6, 10, 16, 26, 42]
        q_factor = 1.5
        sma_window = 15
        batch_size = 2000
    elseif name == "comprehensive"
        periods = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]  # Actual periods: [2.01, 4, 6, 10, 16, 26, 42, 68, 110, 178, 288]
        q_factor = 2.5
        sma_window = 25
        batch_size = 500
    elseif name == "high_q"
        periods = [1, 2, 3, 5, 8, 13, 21, 34]  # Actual periods: [2.01, 4, 6, 10, 16, 26, 42, 68]
        q_factor = 4.0
        sma_window = 20
        batch_size = 1000
    elseif contains(name, "precision")
        periods = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]  # Actual periods: [2.01, 4, 6, 10, 16, 26, 42, 68, 110, 178]
        q_factor = 4.0
        sma_window = 50
        batch_size = 500
    else
        periods = [1, 2, 3, 5, 8, 13, 21, 34, 55]  # Actual periods: [2.01, 4, 6, 10, 16, 26, 42, 68, 110]
        q_factor = 2.0
        sma_window = 20
        batch_size = 1000
    end
    
    # Create configuration
    if use_pll
        # PLL-specific parameters
        if contains(name, "fast")
            pll_gain, pll_bandwidth, lock_threshold = 0.15, 0.02, 0.6
        elseif contains(name, "precision")
            pll_gain, pll_bandwidth, lock_threshold = 0.05, 0.005, 0.8
        else
            pll_gain, pll_bandwidth, lock_threshold = 0.1, 0.01, 0.7
        end
        
        pll_config = PLLConfig(
            enabled = true,
            phase_detector_gain = pll_gain,
            loop_bandwidth = pll_bandwidth,
            lock_threshold = lock_threshold,
            enable_clamping = true
        )
        
        config = ExtendedFilterConfig(
            name = name,
            description = description,
            processing = ProcessingConfig(
                fibonacci_periods = periods,
                q_factor = q_factor,
                sma_window = sma_window,
                batch_size = batch_size
            ),
            pll = pll_config,
            io = IOConfig(
                output_file = "data/YM_06-25_fibonacci_filtered_pll.jld2"
            )
        )
    else
        config = FilterConfig(
            name = name,
            description = description,
            processing = ProcessingConfig(
                fibonacci_periods = periods,
                q_factor = q_factor,
                sma_window = sma_window,
                batch_size = batch_size
            )
        )
    end
    
    save_filter_config(config, config_file)
    return true
end

"""
Display configuration summary
"""
function show_config_summary(config::Union{FilterConfig, ExtendedFilterConfig})
    println("📋 CONFIGURATION SUMMARY: $(config.name)")
    println("="^60)
    println("Description: $(config.description)")
    println("Type: $(isa(config, ExtendedFilterConfig) ? "Extended (PLL)" : "Standard")")
    println()
    
    println("🎛️  Processing Parameters:")
    println("   Fibonacci periods: $(config.processing.fibonacci_periods)")
    println("   Q factor: $(config.processing.q_factor)")
    println("   SMA window: $(config.processing.sma_window)")
    println("   Batch size: $(config.processing.batch_size)")
    println()
    
    println("🎯 Feature Flags:")
    println("   Diagnostics: $(config.processing.include_diagnostics)")
    println("   Price levels: $(config.processing.include_price_levels)")
    println("   Momentum: $(config.processing.include_momentum)")
    println("   Volatility: $(config.processing.include_volatility)")
    println()
    
    if config.pll.enabled
        println("🔒 PLL Configuration:")
        println("   Enabled: $(config.pll.enabled)")
        println("   Phase detector gain: $(config.pll.phase_detector_gain)")
        println("   Loop bandwidth: $(config.pll.loop_bandwidth)")
        println("   Lock threshold: $(config.pll.lock_threshold)")
        println("   Ring decay: $(config.pll.ring_decay)")
        println("   Clamping: $(config.pll.enable_clamping)")
        if config.pll.enable_clamping
            println("   Clamping threshold: $(config.pll.clamping_threshold)")
            println("   Volume scaling: $(config.pll.volume_scaling)")
        end
        println()
    end
    
    println("📁 File Configuration:")
    println("   Input: $(basename(config.io.input_file))")
    println("   Output: $(basename(config.io.output_file))")
    println("   Compress output: $(config.io.compress_output)")
    println()
    
    println("⚡ Performance Settings:")
    println("   Parallel processing: $(config.io.parallel_processing)")
    println("   Max memory: $(config.io.max_memory_gb) GB")
    println("   Log progress: $(config.io.log_progress)")
    println()
    
    # Memory estimation
    estimated_memory = estimate_memory_usage(config)
    println("💾 Estimated memory usage: $(round(estimated_memory, digits=1)) GB")
end

end # module ModernConfigSystem