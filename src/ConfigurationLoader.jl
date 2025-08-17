# src/ConfigurationLoader.jl - Configuration Loading and Management
# Handles master config, per-instrument configs, and integration with existing modules

module ConfigurationLoader

using TOML
using Dates

export initialize_ga_system, create_default_configs,
       load_or_create_instrument, validate_system_setup,
       migrate_from_legacy_config

# =============================================================================
# SYSTEM INITIALIZATION
# =============================================================================

"""
Initialize complete GA system from master configuration
"""
function initialize_ga_system(master_config_path::String = "data/master_config.toml")
    println("\n🚀 Initializing GA Optimization System")
    println("=" ^ 60)
    
    # Create system
    system = Main.GATypes.InstrumentGASystem(master_config_path)
    
    # Load master configuration
    Main.InstrumentManager.load_master_config!(system)
    
    # Validate system setup
    if !validate_system_setup(system)
        @error "System validation failed"
        return system
    end
    
    # Check memory requirements
    if !Main.InstrumentManager.check_memory_requirements(system)
        @warn "Memory requirements may exceed limits"
    end
    
    # List available instruments
    Main.InstrumentManager.list_instruments(system)
    
    println("\n✅ GA system initialized successfully")
    return system
end

"""
Validate system setup and directory structure
"""
function validate_system_setup(system)::Bool
    println("\n🔍 Validating system setup...")
    
    all_valid = true
    
    # Check master config exists
    if !isfile(system.master_config_path)
        @error "Master configuration not found: $(system.master_config_path)"
        all_valid = false
    end
    
    # Validate each instrument
    for (symbol, config) in system.instruments
        println("  Checking $symbol...")
        
        # Check directory structure
        base_path = "data/$symbol"
        required_dirs = [
            base_path,
            "$base_path/parameters",
            "$base_path/ga_workspace"
        ]
        
        for dir in required_dirs
            if !isdir(dir)
                @warn "    Missing directory: $dir"
                all_valid = false
            end
        end
        
        # Validate configuration
        if !Main.GATypes.validate_instrument_config(config)
            @error "    Invalid configuration for $symbol"
            all_valid = false
        end
    end
    
    if all_valid
        println("✅ System validation passed")
    else
        println("⚠️ System validation found issues")
    end
    
    return all_valid
end

# =============================================================================
# CONFIGURATION CREATION
# =============================================================================

"""
Create default configurations for common instruments
"""
function create_default_configs()
    println("\n📝 Creating default configurations...")
    
    # Default instrument specifications
    instruments = [
        ("YM", 50, [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]),
        ("ES", 75, [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]),
        ("NQ", 100, [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233])
    ]
    
    # Create master system
    system = Main.GATypes.InstrumentGASystem()
    
    for (symbol, num_filters, periods) in instruments
        println("  Creating config for $symbol...")
        
        # Create instrument configuration
        config = Main.GATypes.InstrumentConfig(
            symbol = symbol,
            num_filters = Int32(num_filters),
            population_size = Int32(100),
            parameter_path = "data/$symbol/parameters/active.jld2",
            ga_workspace_path = "data/$symbol/ga_workspace/",
            config_path = "data/$symbol/config.toml",
            fibonacci_periods = Int32[p for p in periods],
            max_generations = Int32(500),
            convergence_threshold = Float32(0.001)
        )
        
        # Add to system
        Main.InstrumentManager.add_instrument!(system, config)
        
        # Create directories
        Main.InstrumentManager.create_instrument_directories(config)
        
        # Save instrument config
        Main.InstrumentManager.save_instrument_config(config)
        
        # Create default parameters file
        create_default_parameters(config)
    end
    
    # Save master configuration
    Main.InstrumentManager.save_master_config(system)
    
    println("✅ Created default configurations for $(length(instruments)) instruments")
end

"""
Create default parameters file for an instrument
"""
function create_default_parameters(config)
    # Initialize storage
    storage = Main.StorageSystem.initialize_storage(config)
    
    # Create defaults file if it doesn't exist
    defaults_path = "data/$(config.symbol)/defaults.toml"
    if !isfile(defaults_path)
        defaults = Main.GATypes.FilterDefaults()
        Main.StorageSystem.save_filter_defaults(defaults, defaults_path)
    end
    
    # Apply defaults
    Main.StorageSystem.apply_defaults!(storage, config.fibonacci_periods)
    
    println("  ✅ Created default parameters for $(config.symbol)")
end

# =============================================================================
# INSTRUMENT LOADING
# =============================================================================

"""
Load or create an instrument configuration
"""
function load_or_create_instrument(symbol::String, system)
    # Check if already loaded
    if haskey(system.instruments, symbol)
        return system.instruments[symbol]
    end
    
    # Try to load from config file
    config_path = "data/$symbol/config.toml"
    if isfile(config_path)
        config = Main.InstrumentManager.load_instrument_config(config_path)
        if config !== nothing
            Main.InstrumentManager.add_instrument!(system, config)
            return config
        end
    end
    
    # Create default configuration
    println("Creating default configuration for $symbol...")
    
    # Default settings based on symbol
    defaults = Dict(
        "YM" => (50, [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]),
        "ES" => (75, [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]),
        "NQ" => (100, [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233]),
        "RTY" => (40, [1, 2, 3, 5, 8, 13, 21, 34, 55])
    )
    
    if haskey(defaults, symbol)
        num_filters, periods = defaults[symbol]
    else
        # Generic defaults for unknown symbols
        num_filters = 50
        periods = [1, 2, 3, 5, 8, 13, 21, 34, 55]
    end
    
    config = Main.GATypes.InstrumentConfig(
        symbol = symbol,
        num_filters = Int32(num_filters),
        population_size = Int32(100),
        parameter_path = "data/$symbol/parameters/active.jld2",
        ga_workspace_path = "data/$symbol/ga_workspace/",
        config_path = "data/$symbol/config.toml",
        fibonacci_periods = Int32[p for p in periods],
        max_generations = Int32(500),
        convergence_threshold = Float32(0.001)
    )
    
    # Add to system and create structure
    Main.InstrumentManager.add_instrument!(system, config)
    Main.InstrumentManager.create_instrument_directories(config)
    Main.InstrumentManager.save_instrument_config(config)
    create_default_parameters(config)
    
    return config
end

# =============================================================================
# MIGRATION FROM LEGACY SYSTEMS
# =============================================================================

"""
Migrate from legacy TOML configuration to new GA system
"""
function migrate_from_legacy_config(legacy_toml_path::String, symbol::String, 
                                   system)
    println("\n🔄 Migrating from legacy configuration...")
    
    if !isfile(legacy_toml_path)
        @error "Legacy configuration not found: $legacy_toml_path"
        return false
    end
    
    try
        # Load legacy configuration
        legacy_config = TOML.parsefile(legacy_toml_path)
        
        # Extract filter information
        filters_dict = get(legacy_config, "filters", Dict())
        periods = Int32[]
        
        # Get all filter periods
        for (period_str, _) in filters_dict
            push!(periods, parse(Int32, period_str))
        end
        
        sort!(periods)
        
        println("  Found $(length(periods)) filters in legacy config")
        
        # Create new instrument configuration
        config = Main.GATypes.InstrumentConfig(
            symbol = symbol,
            num_filters = Int32(length(periods)),
            population_size = Int32(100),
            parameter_path = "data/$symbol/parameters/active.jld2",
            ga_workspace_path = "data/$symbol/ga_workspace/",
            config_path = "data/$symbol/config.toml",
            fibonacci_periods = periods,
            max_generations = Int32(500),
            convergence_threshold = Float32(0.001)
        )
        
        # Add to system
        Main.InstrumentManager.add_instrument!(system, config)
        Main.InstrumentManager.create_instrument_directories(config)
        
        # Initialize storage
        storage = Main.StorageSystem.initialize_storage(config)
        
        # Migrate parameters
        println("  Migrating parameters...")
        for (i, period) in enumerate(periods)
            period_str = string(period)
            if haskey(filters_dict, period_str)
                filter_params = filters_dict[period_str]
                
                # Convert to chromosome format (13 parameters)
                chromosome = Float32[
                    Float32(get(filter_params, "q_factor", 2.0)),
                    Float32(log(get(filter_params, "batch_size", 1000))),
                    Float32(log(get(filter_params, "phase_detector_gain", 0.1))),
                    Float32(log(get(filter_params, "loop_bandwidth", 0.01))),
                    Float32(get(filter_params, "lock_threshold", 0.7)),
                    Float32(get(filter_params, "ring_decay", 0.995)),
                    get(filter_params, "enable_clamping", false) ? 1.0f0 : 0.0f0,
                    Float32(log(get(filter_params, "clamping_threshold", 1e-6))),
                    Float32(log(get(filter_params, "volume_scaling", 1.0))),
                    Float32(get(filter_params, "max_frequency_deviation", 0.2)),
                    Float32(get(filter_params, "phase_error_history_length", 20)),
                    1.0f0,  # Default complex weight real
                    0.0f0   # Default complex weight imag
                ]
                
                Main.StorageSystem.set_active_parameters!(storage, Int32(i), chromosome)
                println("    ✓ Migrated filter $period")
            end
        end
        
        # Save migrated configuration
        Main.StorageSystem.sync_to_storage!(storage)
        Main.InstrumentManager.save_instrument_config(config)
        
        println("✅ Successfully migrated $symbol from legacy configuration")
        return true
        
    catch e
        @error "Failed to migrate from legacy configuration: $e"
        return false
    end
end

# =============================================================================
# SYSTEM STATUS AND REPORTING
# =============================================================================

"""
Print comprehensive system status
"""
function print_system_status(system)
    println("\n" * "="^70)
    println("📊 GA OPTIMIZATION SYSTEM STATUS")
    println("="^70)
    
    # Global settings
    println("\n🌐 Global Settings:")
    println("  Master Config: $(system.master_config_path)")
    println("  GPU Enabled: $(system.gpu_enabled)")
    println("  Memory Limit: $(system.max_memory_gb) GB")
    println("  Checkpoint Interval: $(system.checkpoint_interval) generations")
    
    # Current instrument
    if system.current_instrument !== nothing
        println("\n🎯 Current Instrument: $(system.current_instrument)")
    else
        println("\n⚠️ No instrument currently selected")
    end
    
    # Instrument summary
    println("\n📈 Configured Instruments:")
    println("  " * "-"^60)
    
    for symbol in system.active_instruments
        if haskey(system.instruments, symbol)
            config = system.instruments[symbol]
            status = symbol == system.current_instrument ? "ACTIVE" : "idle"
            
            println("  $symbol:")
            println("    Status: $status")
            println("    Filters: $(config.num_filters)")
            println("    Population: $(config.population_size)")
            println("    Periods: $(config.fibonacci_periods[1:min(5, end)])$(length(config.fibonacci_periods) > 5 ? "..." : "")")
            
            # Check if parameters exist
            if isfile(config.parameter_path)
                file_size = round(stat(config.parameter_path).size / 1024, digits=1)
                println("    Parameters: ✓ ($(file_size) KB)")
            else
                println("    Parameters: ✗ (not initialized)")
            end
        end
    end
    
    # Memory usage
    println("\n💾 Memory Usage:")
    total_memory = 0.0f0
    for (symbol, config) in system.instruments
        memory_mb = Main.InstrumentManager.estimate_memory_usage(config)
        total_memory += memory_mb
        println("  $symbol: $(round(memory_mb, digits=1)) MB")
    end
    println("  " * "-"^30)
    println("  Total: $(round(total_memory, digits=1)) MB / $(system.max_memory_gb * 1024) MB")
    
    println("\n" * "="^70)
end

"""
Generate system report for logging
"""
function generate_system_report(system)::Dict{String, Any}
    report = Dict{String, Any}()
    
    # System info
    report["timestamp"] = now()
    report["master_config"] = system.master_config_path
    report["gpu_enabled"] = system.gpu_enabled
    report["memory_limit_gb"] = system.max_memory_gb
    report["current_instrument"] = something(system.current_instrument, "none")
    
    # Instrument details
    report["instruments"] = Dict{String, Any}()
    
    for (symbol, config) in system.instruments
        inst_report = Dict{String, Any}()
        inst_report["num_filters"] = config.num_filters
        inst_report["population_size"] = config.population_size
        inst_report["fibonacci_periods"] = config.fibonacci_periods
        inst_report["max_generations"] = config.max_generations
        inst_report["parameters_exist"] = isfile(config.parameter_path)
        
        if isfile(config.parameter_path)
            inst_report["parameter_file_size_kb"] = round(stat(config.parameter_path).size / 1024, digits=1)
        end
        
        # Storage info
        storage_path = config.parameter_path
        workspace_path = config.ga_workspace_path
        
        inst_report["storage_exists"] = isfile(storage_path)
        inst_report["workspace_exists"] = isdir(workspace_path)
        
        # Count checkpoints
        if isdir(dirname(storage_path))
            checkpoints = filter(f -> startswith(f, "checkpoint_"), readdir(dirname(storage_path)))
            inst_report["num_checkpoints"] = length(checkpoints)
        else
            inst_report["num_checkpoints"] = 0
        end
        
        report["instruments"][symbol] = inst_report
    end
    
    # Memory estimation
    total_memory = 0.0f0
    for (_, config) in system.instruments
        total_memory += Main.InstrumentManager.estimate_memory_usage(config)
    end
    report["total_memory_mb"] = round(total_memory, digits=1)
    report["memory_usage_percent"] = round(100 * total_memory / (system.max_memory_gb * 1024), digits=1)
    
    return report
end

end # module ConfigurationLoader