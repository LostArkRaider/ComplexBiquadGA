# src/ModernConfigSystem.jl - Type-Safe Configuration System with Per-Filter Parameters
# Modified for v3.0 to support independent parameters for each filter

"""
Modern Type-Safe Configuration System for Fibonacci Filter Bank

Version 3.0 Changes:
- Added per-filter parameter support via FilterBank structure
- Each filter maintains independent parameters (12 params per filter)
- NO DICTIONARIES - all struct-based for performance
- Direct integration with FilterParameterGA module

Key Features:
- Pure struct-based parameter hierarchy (NO DICTIONARIES)
- Compile-time type validation
- Direct field access for performance
- Per-filter parameter independence
- TOML persistence with automatic conversion
"""

module ModernConfigSystem

using Parameters
using TOML
using Dates
using Statistics

export FilterParameters, FilterBank, FilterConfig, ExtendedFilterConfig, 
       ProcessingConfig, PLLConfig, IOConfig,
       load_filter_config, save_filter_config, validate_config,
       create_default_configs, list_available_configs,
       show_config_summary, create_default_filter_params,
       get_filter_by_period, set_filter_by_period!,
       get_active_filters, get_active_periods

# =============================================================================
# PER-FILTER PARAMETER STRUCTURES (NEW IN v3.0)
# =============================================================================

"""
Complete parameter set for a single filter (12 parameters)
All filters have independent copies of these parameters
"""
@with_kw struct FilterParameters
    period::Int                          # Fibonacci period this filter is for
    q_factor::Float64 = 2.0             # Filter Q factor/bandwidth
    sma_window::Int = 20                # Simple moving average window
    batch_size::Int = 1000              # Processing batch size
    phase_detector_gain::Float64 = 0.1  # PLL phase detector sensitivity
    loop_bandwidth::Float64 = 0.01      # PLL loop filter bandwidth
    lock_threshold::Float64 = 0.7       # PLL lock quality threshold
    ring_decay::Float64 = 0.995         # Ringing decay factor
    enable_clamping::Bool = false       # Enable signal clamping
    clamping_threshold::Float64 = 1e-6  # Clamping activation threshold
    volume_scaling::Float64 = 1.0       # Volume component scaling
    max_frequency_deviation::Float64 = 0.2  # Maximum frequency deviation
    phase_error_history_length::Int = 20    # Phase error buffer size
end

"""
Create default filter parameters for a given period
"""
function create_default_filter_params(period::Int)::FilterParameters
    return FilterParameters(
        period = period,
        q_factor = 2.0,
        sma_window = 20,
        batch_size = 1000,
        phase_detector_gain = 0.1,
        loop_bandwidth = 0.01,
        lock_threshold = 0.7,
        ring_decay = 0.995,
        enable_clamping = false,
        clamping_threshold = 1e-6,
        volume_scaling = 1.0,
        max_frequency_deviation = 0.2,
        phase_error_history_length = 20
    )
end

"""
FilterBank stores all filter configurations using direct struct access
NO DICTIONARIES - uses pre-allocated arrays for performance
"""
mutable struct FilterBank
    filters::Vector{FilterParameters}   # All filter configurations
    active_mask::Vector{Bool}          # Which filters are active
    periods::Vector{Int}               # Period for each filter slot
    num_slots::Int                     # Total allocated slots
    num_active::Int                    # Number of active filters
end

# Constructor with pre-allocation
function FilterBank(initial_periods::Vector{Int}, max_filters::Int = 20)
    num_initial = length(initial_periods)
    @assert num_initial <= max_filters "Initial periods exceed maximum filters"
    
    # Pre-allocate arrays
    filters = Vector{FilterParameters}(undef, max_filters)
    active_mask = zeros(Bool, max_filters)
    periods = zeros(Int, max_filters)
    
    # Initialize provided filters
    for (i, period) in enumerate(initial_periods)
        filters[i] = create_default_filter_params(period)
        active_mask[i] = true
        periods[i] = period
    end
    
    # Initialize remaining slots with defaults
    for i in (num_initial + 1):max_filters
        filters[i] = create_default_filter_params(0)  # Period 0 indicates unused
        active_mask[i] = false
        periods[i] = 0
    end
    
    return FilterBank(filters, active_mask, periods, max_filters, num_initial)
end

# Get filter by period (linear search - acceptable for small n)
function get_filter_by_period(bank::FilterBank, period::Int)::Union{FilterParameters, Nothing}
    for i in 1:bank.num_slots
        if bank.active_mask[i] && bank.periods[i] == period
            return bank.filters[i]
        end
    end
    return nothing
end

# Set filter by period
function set_filter_by_period!(bank::FilterBank, params::FilterParameters)
    period = params.period
    
    # Check if period already exists
    for i in 1:bank.num_slots
        if bank.periods[i] == period
            bank.filters[i] = params
            if !bank.active_mask[i]
                bank.active_mask[i] = true
                bank.num_active += 1
            end
            return
        end
    end
    
    # Find first empty slot
    for i in 1:bank.num_slots
        if !bank.active_mask[i]
            bank.filters[i] = params
            bank.active_mask[i] = true
            bank.periods[i] = period
            bank.num_active += 1
            return
        end
    end
    
    error("FilterBank is full (max $(bank.num_slots) filters)")
end

# Get active filters
function get_active_filters(bank::FilterBank)::Vector{FilterParameters}
    active = FilterParameters[]
    for i in 1:bank.num_slots
        if bank.active_mask[i]
            push!(active, bank.filters[i])
        end
    end
    return active
end

# Get active periods
function get_active_periods(bank::FilterBank)::Vector{Int}
    periods = Int[]
    for i in 1:bank.num_slots
        if bank.active_mask[i]
            push!(periods, bank.periods[i])
        end
    end
    return sort(periods)
end

# =============================================================================
# CORE CONFIGURATION STRUCTURES (MODIFIED FOR v3.0)
# =============================================================================

"""
Processing parameters with compile-time validation
Modified in v3.0: Removed per-filter params (now in FilterBank)
"""
@with_kw struct ProcessingConfig
    # Feature flags only (parameters moved to FilterBank)
    include_diagnostics::Bool = true
    include_price_levels::Bool = true
    include_momentum::Bool = true
    include_volatility::Bool = true
end

"""
PLL-specific global configuration
Modified in v3.0: Removed per-filter params (now in FilterBank)
"""
@with_kw struct PLLConfig
    enabled::Bool = false  # Global PLL enable/disable
end

"""
I/O and performance configuration (unchanged)
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
Modified in v3.0: Now uses FilterBank for parameters
"""
@with_kw struct FilterConfig
    name::String = "default"
    description::String = "Default filter bank configuration"
    
    filter_bank::FilterBank = FilterBank([3, 5, 8, 13, 21, 34, 55])  # Default Fibonacci periods
    processing::ProcessingConfig = ProcessingConfig()
    pll::PLLConfig = PLLConfig()  # PLL disabled by default
    io::IOConfig = IOConfig()
    
    created::DateTime = now()
    version::String = "3.0"  # Updated version for per-filter support
end

"""
Extended filter bank configuration with PLL enabled
Modified in v3.0: Now uses FilterBank for per-filter parameters
"""
@with_kw struct ExtendedFilterConfig
    name::String = "pll_enhanced"
    description::String = "PLL-enhanced filter bank configuration"
    
    filter_bank::FilterBank = FilterBank([1, 2, 3, 5, 8, 13, 21, 34, 55])  # More periods
    processing::ProcessingConfig = ProcessingConfig()
    pll::PLLConfig = PLLConfig(enabled=true)  # PLL enabled by default
    io::IOConfig = IOConfig()
    
    created::DateTime = now()
    version::String = "3.0"  # Updated version for per-filter support
    ga_fitness::Float64 = 0.0  # Track GA optimization fitness if applicable
end

# =============================================================================
# CONFIGURATION LOADING AND SAVING (MODIFIED FOR v3.0)
# =============================================================================

"""
Load configuration with automatic type detection
Modified in v3.0 to handle FilterBank
"""
function load_filter_config(config_name::String; validate::Bool = true)::Union{FilterConfig, ExtendedFilterConfig}
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
        
        if validate
            validate_config(config)
        else
            println("⚠️  Skipping validation (validate=false)")
        end
        
        println("✅ Configuration loaded: $(config.name)")
        return config
        
    catch e
        error("Failed to load configuration '$config_name': $e")
    end
end

"""
Parse TOML data into appropriate configuration struct
Modified in v3.0 to construct FilterBank from TOML
"""
function parse_toml_to_config(toml_data::Dict, name::String)::Union{FilterConfig, ExtendedFilterConfig}
    # Extract sections with safe defaults
    metadata = get(toml_data, "metadata", Dict())
    processing_dict = get(toml_data, "processing", Dict())
    pll_dict = get(toml_data, "pll", Dict())
    io_dict = get(toml_data, "io", Dict())
    filters_dict = get(toml_data, "filters", Dict())
    
    # Determine active periods
    active_periods = Vector{Int}(get(metadata, "active_periods", [3, 5, 8, 13, 21, 34, 55]))
    
    # Build FilterBank from TOML data
    max_filters = get(metadata, "max_filters", 20)
    filter_bank = FilterBank(Int[], max_filters)  # Start empty
    
    # Load filter configurations
    if !isempty(filters_dict)
        # Per-filter parameters provided
        for (period_str, filter_data) in filters_dict
            period = parse(Int, period_str)
            params = FilterParameters(
                period = period,
                q_factor = Float64(get(filter_data, "q_factor", 2.0)),
                sma_window = Int(get(filter_data, "sma_window", 20)),
                batch_size = Int(get(filter_data, "batch_size", 1000)),
                phase_detector_gain = Float64(get(filter_data, "phase_detector_gain", 0.1)),
                loop_bandwidth = Float64(get(filter_data, "loop_bandwidth", 0.01)),
                lock_threshold = Float64(get(filter_data, "lock_threshold", 0.7)),
                ring_decay = Float64(get(filter_data, "ring_decay", 0.995)),
                enable_clamping = Bool(get(filter_data, "enable_clamping", false)),
                clamping_threshold = Float64(get(filter_data, "clamping_threshold", 1e-6)),
                volume_scaling = Float64(get(filter_data, "volume_scaling", 1.0)),
                max_frequency_deviation = Float64(get(filter_data, "max_frequency_deviation", 0.2)),
                phase_error_history_length = Int(get(filter_data, "phase_error_history_length", 20))
            )
            set_filter_by_period!(filter_bank, params)
        end
    else
        # Legacy format or default initialization
        for period in active_periods
            set_filter_by_period!(filter_bank, create_default_filter_params(period))
        end
    end
    
    # Build processing configuration
    processing = ProcessingConfig(
        include_diagnostics = Bool(get(processing_dict, "include_diagnostics", true)),
        include_price_levels = Bool(get(processing_dict, "include_price_levels", true)),
        include_momentum = Bool(get(processing_dict, "include_momentum", true)),
        include_volatility = Bool(get(processing_dict, "include_volatility", true))
    )
    
    # Build PLL configuration
    pll_enabled = Bool(get(pll_dict, "enabled", false))
    pll = PLLConfig(enabled = pll_enabled)
    
    # Build I/O configuration
    io = IOConfig(
        input_file = String(get(io_dict, "input_file", "data/YM_06-25_bars_market_time.jld2")),
        output_file = String(get(io_dict, "output_file", "data/YM_06-25_fibonacci_filtered.jld2")),
        save_intermediate = Bool(get(io_dict, "save_intermediate", false)),
        compress_output = Bool(get(io_dict, "compress_output", true)),
        parallel_processing = Bool(get(io_dict, "parallel_processing", false)),
        max_memory_gb = Float64(get(io_dict, "max_memory_gb", 12.0)),
        log_progress = Bool(get(io_dict, "log_progress", true)),
        progress_interval = Int(get(io_dict, "progress_interval", 1000)),
        validate_inputs = Bool(get(io_dict, "validate_inputs", true)),
        error_on_missing_file = Bool(get(io_dict, "error_on_missing_file", true)),
        backup_on_overwrite = Bool(get(io_dict, "backup_on_overwrite", false))
    )
    
    # Determine configuration type and create appropriate struct
    config_name = String(get(metadata, "name", name))
    description = String(get(metadata, "description", "Configuration loaded from TOML"))
    ga_fitness = Float64(get(metadata, "ga_fitness", 0.0))
    
    if pll_enabled
        return ExtendedFilterConfig(
            name = config_name,
            description = description,
            filter_bank = filter_bank,
            processing = processing,
            pll = pll,
            io = io,
            ga_fitness = ga_fitness
        )
    else
        return FilterConfig(
            name = config_name,
            description = description,
            filter_bank = filter_bank,
            processing = processing,
            pll = pll,
            io = io
        )
    end
end

"""
Save configuration to TOML file with proper type handling
Modified in v3.0 to save FilterBank structure
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
Modified in v3.0 to handle FilterBank
"""
function struct_to_toml_dict(config::Union{FilterConfig, ExtendedFilterConfig})::Dict{String, Any}
    # Build metadata
    metadata = Dict(
        "name" => config.name,
        "description" => config.description,
        "version" => config.version,
        "created" => string(config.created),
        "config_type" => isa(config, ExtendedFilterConfig) ? "extended" : "standard",
        "active_periods" => get_active_periods(config.filter_bank),
        "max_filters" => config.filter_bank.num_slots
    )
    
    # Add GA fitness if ExtendedFilterConfig
    if isa(config, ExtendedFilterConfig)
        metadata["ga_fitness"] = config.ga_fitness
    end
    
    # Build filters section
    filters = Dict{String, Any}()
    for i in 1:config.filter_bank.num_slots
        if config.filter_bank.active_mask[i]
            fp = config.filter_bank.filters[i]
            filters[string(fp.period)] = Dict(
                "q_factor" => fp.q_factor,
                "sma_window" => fp.sma_window,
                "batch_size" => fp.batch_size,
                "phase_detector_gain" => fp.phase_detector_gain,
                "loop_bandwidth" => fp.loop_bandwidth,
                "lock_threshold" => fp.lock_threshold,
                "ring_decay" => fp.ring_decay,
                "enable_clamping" => fp.enable_clamping,
                "clamping_threshold" => fp.clamping_threshold,
                "volume_scaling" => fp.volume_scaling,
                "max_frequency_deviation" => fp.max_frequency_deviation,
                "phase_error_history_length" => fp.phase_error_history_length
            )
        end
    end
    
    return Dict{String, Any}(
        "metadata" => metadata,
        "processing" => Dict(
            "include_diagnostics" => config.processing.include_diagnostics,
            "include_price_levels" => config.processing.include_price_levels,
            "include_momentum" => config.processing.include_momentum,
            "include_volatility" => config.processing.include_volatility
        ),
        "pll" => Dict(
            "enabled" => config.pll.enabled
        ),
        "io" => Dict(
            "input_file" => config.io.input_file,
            "output_file" => config.io.output_file,
            "save_intermediate" => config.io.save_intermediate,
            "compress_output" => config.io.compress_output,
            "backup_on_overwrite" => config.io.backup_on_overwrite,
            "parallel_processing" => config.io.parallel_processing,
            "max_memory_gb" => config.io.max_memory_gb,
            "log_progress" => config.io.log_progress,
            "progress_interval" => config.io.progress_interval,
            "validate_inputs" => config.io.validate_inputs,
            "error_on_missing_file" => config.io.error_on_missing_file
        ),
        "filters" => filters
    )
end

# =============================================================================
# CONFIGURATION VALIDATION (MODIFIED FOR v3.0)
# =============================================================================

"""
Comprehensive configuration validation with detailed error reporting
Modified in v3.0 to validate FilterBank
"""
function validate_config(config::Union{FilterConfig, ExtendedFilterConfig})::Bool
    errors = String[]
    warnings = String[]
    
    # File system validation
    validate_file_system(config, errors, warnings)
    
    # Filter bank validation
    validate_filter_bank(config, errors, warnings)
    
    # Memory estimation
    validate_memory_requirements(config, errors, warnings)
    
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

function validate_filter_bank(config::Union{FilterConfig, ExtendedFilterConfig}, 
                              errors::Vector{String}, warnings::Vector{String})
    bank = config.filter_bank
    
    if bank.num_active == 0
        push!(errors, "No active filters in FilterBank")
    end
    
    # Validate each active filter
    for i in 1:bank.num_slots
        if bank.active_mask[i]
            fp = bank.filters[i]
            period = fp.period
            
            # Q factor validation
            if fp.q_factor > 5.0
                push!(warnings, "Filter $period: High Q factor ($(fp.q_factor)) may cause instability")
            end
            
            # PLL stability analysis (if enabled)
            if config.pll.enabled
                gain_bandwidth_product = fp.phase_detector_gain * fp.loop_bandwidth
                if gain_bandwidth_product > 0.05
                    push!(warnings, "Filter $period: High gain×bandwidth product ($gain_bandwidth_product)")
                end
                
                if fp.lock_threshold < 0.5
                    push!(warnings, "Filter $period: Low lock threshold ($(fp.lock_threshold))")
                elseif fp.lock_threshold > 0.9
                    push!(warnings, "Filter $period: Very high lock threshold ($(fp.lock_threshold))")
                end
            end
            
            # Filter stability check
            actual_period = period == 1 ? 2.01 : 2.0 * period
            fc = 1.0 / actual_period
            
            if fc >= 0.5
                push!(errors, "Filter $period: Frequency exceeds Nyquist limit")
            elseif fc >= 0.45
                push!(warnings, "Filter $period: Near Nyquist frequency (fc = $(round(fc, digits=3)))")
            end
            
            # Bandwidth check
            bandwidth = fc / fp.q_factor
            if bandwidth < 0.001
                push!(warnings, "Filter $period: Very narrow bandwidth (BW=$(round(bandwidth, digits=4)))")
            end
        end
    end
    
    # Check for duplicate periods
    active_periods = get_active_periods(bank)
    if length(active_periods) != length(unique(active_periods))
        push!(errors, "Duplicate filter periods detected")
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

"""
Estimate memory usage based on configuration parameters
Modified in v3.0 to account for per-filter parameters
"""
function estimate_memory_usage(config::Union{FilterConfig, ExtendedFilterConfig})::Float64
    # Assumptions for estimation based on actual data size
    estimated_bars = 62000  # Based on your actual data
    n_filters = config.filter_bank.num_active
    
    # Base memory requirements (original data)
    base_memory_mb = estimated_bars * 36 * 8 / 1024^2
    
    # Complex input columns (3 complex columns)
    complex_input_mb = estimated_bars * 3 * 16 / 1024^2
    
    # Filter output memory (2 complex columns per filter)
    filter_memory_mb = estimated_bars * n_filters * 2 * 16 / 1024^2
    
    # PLL-specific memory (additional columns if enabled)
    pll_memory_mb = 0.0
    if config.pll.enabled
        pll_memory_mb = estimated_bars * n_filters * 2 * 8 / 1024^2  # 2 Float64 columns per filter
        pll_memory_mb += estimated_bars * 3 * 8 / 1024^2  # system columns
    end
    
    # Diagnostic columns if enabled
    diagnostic_memory_mb = 0.0
    if config.processing.include_diagnostics
        diagnostic_memory_mb = estimated_bars * 4 * 8 / 1024^2  # 4 diagnostic columns
    end
    
    # Intermediate calculation memory (roughly 50% overhead)
    overhead_memory_mb = (base_memory_mb + complex_input_mb + filter_memory_mb + 
                         pll_memory_mb + diagnostic_memory_mb) * 0.5
    
    total_memory_gb = (base_memory_mb + complex_input_mb + filter_memory_mb + 
                      pll_memory_mb + diagnostic_memory_mb + overhead_memory_mb) / 1024
    
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
        "config/filters/$config_name.toml",    # Filter configs
        "config/ga/$config_name.toml",         # GA configs
        "config/$config_name.toml",            # Root config dir
        joinpath("config", "filters", "$config_name.toml"),  # Alternative path format
        joinpath("config", "ga", "$config_name.toml"),       # Alternative path format
        "$config_name.toml"                    # Current directory fallback
    ]
    
    for path in search_paths
        if isfile(path)
            return path
        end
    end
    
    # Return preferred path for creation (filter configs by default)
    return "config/filters/$config_name.toml"
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
Modified in v3.0 to use FilterBank
"""
function create_default_configs()
    config_dir = "config"
    if !isdir(config_dir)
        mkpath(config_dir)
        println("📁 Created config directory: $config_dir")
    end
    
    # Define configuration presets
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
Modified in v3.0 to create per-filter parameters
"""
function create_config_preset(name::String, description::String, use_pll::Bool)::Bool
    config_file = "config/$name.toml"
    
    if isfile(config_file)
        return false  # Don't overwrite existing configs
    end
    
    # Define preset-specific parameters
    if name == "fast"
        periods = [1, 2, 3, 5, 8, 13, 21]
        bank = FilterBank(periods)
        # Customize parameters for fast processing
        for i in 1:bank.num_active
            fp = bank.filters[i]
            bank.filters[i] = FilterParameters(
                period = fp.period,
                q_factor = 1.5 + 0.1 * i,
                sma_window = 15,
                batch_size = 2000,
                phase_detector_gain = 0.15,
                loop_bandwidth = 0.02,
                lock_threshold = 0.6,
                ring_decay = 0.994,
                enable_clamping = false,
                clamping_threshold = 1e-6,
                volume_scaling = 1.0,
                max_frequency_deviation = 0.3,
                phase_error_history_length = 10
            )
        end
        
    elseif name == "comprehensive"
        periods = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
        bank = FilterBank(periods, 15)  # More slots for comprehensive
        
    elseif name == "high_q"
        periods = [1, 2, 3, 5, 8, 13, 21, 34]
        bank = FilterBank(periods)
        # Set high Q factors
        for i in 1:bank.num_active
            fp = bank.filters[i]
            bank.filters[i] = FilterParameters(
                period = fp.period,
                q_factor = 4.0,
                sma_window = 20,
                batch_size = 1000,
                phase_detector_gain = 0.05,
                loop_bandwidth = 0.005,
                lock_threshold = 0.8,
                ring_decay = 0.997,
                enable_clamping = true,
                clamping_threshold = 1e-7,
                volume_scaling = 1.0,
                max_frequency_deviation = 0.1,
                phase_error_history_length = 30
            )
        end
        
    else
        periods = [1, 2, 3, 5, 8, 13, 21, 34, 55]
        bank = FilterBank(periods)
    end
    
    # Create configuration
    if use_pll
        pll_config = PLLConfig(enabled = true)
        
        config = ExtendedFilterConfig(
            name = name,
            description = description,
            filter_bank = bank,
            processing = ProcessingConfig(),
            pll = pll_config,
            io = IOConfig(
                output_file = "data/YM_06-25_fibonacci_filtered_pll.jld2"
            )
        )
    else
        config = FilterConfig(
            name = name,
            description = description,
            filter_bank = bank,
            processing = ProcessingConfig(),
            pll = PLLConfig(enabled = false),
            io = IOConfig()
        )
    end
    
    save_filter_config(config, config_file)
    return true
end

"""
Display configuration summary
Modified in v3.0 to show per-filter parameters
"""
function show_config_summary(config::Union{FilterConfig, ExtendedFilterConfig})
    println("📋 CONFIGURATION SUMMARY: $(config.name)")
    println("="^60)
    println("Description: $(config.description)")
    println("Type: $(isa(config, ExtendedFilterConfig) ? "Extended (PLL)" : "Standard")")
    println("Version: $(config.version)")
    println()
    
    println("🎛️  Filter Bank:")
    println("   Active filters: $(config.filter_bank.num_active)")
    println("   Active periods: $(get_active_periods(config.filter_bank))")
    println("   Total slots: $(config.filter_bank.num_slots)")
    println()
    
    # Show per-filter parameters
    println("📊 Per-Filter Parameters:")
    active_filters = get_active_filters(config.filter_bank)
    for fp in active_filters
        println("   Filter $(fp.period):")
        println("      Q factor: $(round(fp.q_factor, digits=2))")
        println("      SMA window: $(fp.sma_window)")
        println("      Batch size: $(fp.batch_size)")
        if config.pll.enabled
            println("      PLL gain: $(round(fp.phase_detector_gain, digits=4))")
            println("      PLL bandwidth: $(round(fp.loop_bandwidth, digits=5))")
            println("      Lock threshold: $(round(fp.lock_threshold, digits=2))")
        end
    end
    println()
    
    println("🎯 Feature Flags:")
    println("   Diagnostics: $(config.processing.include_diagnostics)")
    println("   Price levels: $(config.processing.include_price_levels)")
    println("   Momentum: $(config.processing.include_momentum)")
    println("   Volatility: $(config.processing.include_volatility)")
    println()
    
    if config.pll.enabled
        println("🔒 PLL Configuration:")
        println("   Global PLL enabled: $(config.pll.enabled)")
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
    
    # GA fitness if available
    if isa(config, ExtendedFilterConfig) && config.ga_fitness > 0
        println("🎯 GA Optimization Fitness: $(round(config.ga_fitness, digits=4))")
    end
end

end # module ModernConfigSystem