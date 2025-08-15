# src/InstrumentManager.jl - Multi-Instrument Management System
# Handles instrument switching, configuration, and cross-instrument initialization

module InstrumentManager

using TOML
using Dates
using Printf
using JLD2

# Import GATypes
if !isdefined(Main, :GATypes)
    include("GATypes.jl")
end
using Main.GATypes

export load_master_config!, add_instrument!, remove_instrument!,
       switch_instrument!, get_current_instrument, list_instruments,
       initialize_from_instrument!, create_instrument_directories,
       save_master_config, load_instrument_config, save_instrument_config

# =============================================================================
# MASTER CONFIGURATION MANAGEMENT
# =============================================================================

"""
Load master configuration file and populate InstrumentGASystem
"""
function load_master_config!(system::InstrumentGASystem)
    if !isfile(system.master_config_path)
        @warn "Master config not found at $(system.master_config_path), creating default"
        create_default_master_config(system.master_config_path)
    end
    
    config = TOML.parsefile(system.master_config_path)
    
    # Load global settings
    if haskey(config, "global")
        global_cfg = config["global"]
        system.gpu_enabled = get(global_cfg, "gpu_enabled", false)
        system.max_memory_gb = Float32(get(global_cfg, "max_memory_gb", 12.0))
        system.checkpoint_interval = get(global_cfg, "checkpoint_interval", 50)
    end
    
    # Load instruments section
    if haskey(config, "instruments")
        inst_cfg = config["instruments"]
        active = get(inst_cfg, "active", String[])
        system.active_instruments = active
        
        # Load each active instrument's configuration
        for symbol in active
            if haskey(config, symbol)
                instrument_config = parse_instrument_config(symbol, config[symbol])
                if validate_instrument_config(instrument_config)
                    system.instruments[symbol] = instrument_config
                    println("✅ Loaded instrument: $symbol with $(instrument_config.num_filters) filters")
                else
                    @error "Failed to validate configuration for $symbol"
                end
            else
                @warn "Configuration missing for active instrument: $symbol"
            end
        end
    end
    
    println("📋 Loaded $(length(system.instruments)) instruments from master config")
    return system
end

"""
Parse individual instrument configuration from TOML
"""
function parse_instrument_config(symbol::String, config::Dict)::InstrumentConfig
    # Extract configuration values with defaults
    num_filters = Int32(get(config, "num_filters", 50))
    population_size = Int32(get(config, "population_size", 100))
    
    # Parse Fibonacci periods
    periods_raw = get(config, "fibonacci_periods", [1, 2, 3, 5, 8, 13, 21, 34, 55])
    fibonacci_periods = Int32[p for p in periods_raw]
    
    # Build paths
    base_path = "data/$symbol"
    parameter_path = "$base_path/parameters/active.jld2"
    ga_workspace_path = "$base_path/ga_workspace/"
    config_path = "$base_path/config.toml"
    
    # Parse GA parameters
    ga_params = GAParameters()
    if haskey(config, "ga_params")
        ga_cfg = config["ga_params"]
        ga_params = GAParameters(
            mutation_rate = Float32(get(ga_cfg, "mutation_rate", 0.1)),
            crossover_rate = Float32(get(ga_cfg, "crossover_rate", 0.7)),
            elite_size = Int32(get(ga_cfg, "elite_size", 10)),
            tournament_size = Int32(get(ga_cfg, "tournament_size", 5)),
            max_generations = Int32(get(ga_cfg, "max_generations", 500)),
            convergence_threshold = Float32(get(ga_cfg, "convergence_threshold", 0.001)),
            early_stopping_patience = Int32(get(ga_cfg, "early_stopping_patience", 20))
        )
    end
    
    # Other settings
    max_generations = Int32(get(config, "max_generations", 500))
    convergence_threshold = Float32(get(config, "convergence_threshold", 0.001))
    initialization_source = get(config, "initialization_source", nothing)
    
    return InstrumentConfig(
        symbol = symbol,
        num_filters = num_filters,
        population_size = population_size,
        parameter_path = parameter_path,
        ga_workspace_path = ga_workspace_path,
        config_path = config_path,
        fibonacci_periods = fibonacci_periods,
        max_generations = max_generations,
        convergence_threshold = convergence_threshold,
        initialization_source = initialization_source,
        ga_params = ga_params
    )
end

"""
Save master configuration to TOML file
"""
function save_master_config(system::InstrumentGASystem)
    config = Dict{String, Any}()
    
    # Global settings
    config["global"] = Dict(
        "gpu_enabled" => system.gpu_enabled,
        "max_memory_gb" => system.max_memory_gb,
        "checkpoint_interval" => system.checkpoint_interval
    )
    
    # Instruments section
    config["instruments"] = Dict(
        "active" => system.active_instruments,
        "default_population_size" => 100,
        "default_generations" => 500
    )
    
    # Individual instrument configurations
    for (symbol, inst_config) in system.instruments
        config[symbol] = Dict(
            "num_filters" => inst_config.num_filters,
            "population_size" => inst_config.population_size,
            "fibonacci_periods" => inst_config.fibonacci_periods,
            "max_generations" => inst_config.max_generations,
            "convergence_threshold" => inst_config.convergence_threshold,
            "initialization_source" => something(inst_config.initialization_source, "")
        )
        
        # Add GA parameters
        config[symbol]["ga_params"] = Dict(
            "mutation_rate" => inst_config.ga_params.mutation_rate,
            "crossover_rate" => inst_config.ga_params.crossover_rate,
            "elite_size" => inst_config.ga_params.elite_size,
            "tournament_size" => inst_config.ga_params.tournament_size,
            "max_generations" => inst_config.ga_params.max_generations,
            "convergence_threshold" => inst_config.ga_params.convergence_threshold,
            "early_stopping_patience" => inst_config.ga_params.early_stopping_patience
        )
    end
    
    # Write to file
    open(system.master_config_path, "w") do io
        TOML.print(io, config)
    end
    
    println("💾 Saved master configuration to $(system.master_config_path)")
end

"""
Create default master configuration file
"""
function create_default_master_config(path::String)
    default_config = """
    [global]
    gpu_enabled = false
    max_memory_gb = 12.0
    checkpoint_interval = 50

    [instruments]
    active = ["YM"]
    default_population_size = 100
    default_generations = 500

    [YM]
    num_filters = 50
    fibonacci_periods = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
    population_size = 100
    max_generations = 500
    convergence_threshold = 0.001
    initialization_source = ""

    [YM.ga_params]
    mutation_rate = 0.1
    crossover_rate = 0.7
    elite_size = 10
    tournament_size = 5
    max_generations = 500
    convergence_threshold = 0.001
    early_stopping_patience = 20
    """
    
    # Ensure directory exists
    dir = dirname(path)
    if !isdir(dir)
        mkpath(dir)
    end
    
    open(path, "w") do io
        write(io, default_config)
    end
    
    println("✅ Created default master configuration at $path")
end

# =============================================================================
# INSTRUMENT MANAGEMENT
# =============================================================================

"""
Add a new instrument to the system
"""
function add_instrument!(system::InstrumentGASystem, config::InstrumentConfig)
    symbol = config.symbol
    
    # Validate configuration
    if !validate_instrument_config(config)
        @error "Invalid configuration for instrument $symbol"
        return false
    end
    
    # Create directory structure
    create_instrument_directories(config)
    
    # Add to system
    system.instruments[symbol] = config
    if !(symbol in system.active_instruments)
        push!(system.active_instruments, symbol)
    end
    
    # Save updated master config
    save_master_config(system)
    
    println("✅ Added instrument: $symbol")
    return true
end

"""
Remove an instrument from the system
"""
function remove_instrument!(system::InstrumentGASystem, symbol::String)
    if !haskey(system.instruments, symbol)
        @warn "Instrument $symbol not found"
        return false
    end
    
    # Remove from active list
    filter!(x -> x != symbol, system.active_instruments)
    
    # Remove from instruments
    delete!(system.instruments, symbol)
    
    # Clear current if it was selected
    if system.current_instrument == symbol
        system.current_instrument = nothing
    end
    
    # Save updated master config
    save_master_config(system)
    
    println("✅ Removed instrument: $symbol")
    return true
end

"""
Switch to a different instrument for optimization
"""
function switch_instrument!(system::InstrumentGASystem, symbol::String)
    if !haskey(system.instruments, symbol)
        @error "Instrument $symbol not found in system"
        return false
    end
    
    # Save current instrument's state if exists
    if system.current_instrument !== nothing
        println("💾 Saving state for $(system.current_instrument)")
        # This would trigger storage sync in actual implementation
    end
    
    # Switch to new instrument
    system.current_instrument = symbol
    println("🔄 Switched to instrument: $symbol")
    
    return true
end

"""
Get the currently active instrument configuration
"""
function get_current_instrument(system::InstrumentGASystem)::Union{InstrumentConfig, Nothing}
    if system.current_instrument === nothing
        return nothing
    end
    
    return get(system.instruments, system.current_instrument, nothing)
end

"""
List all available instruments
"""
function list_instruments(system::InstrumentGASystem)
    println("\n📊 Available Instruments:")
    println("=" ^ 60)
    
    for symbol in system.active_instruments
        if haskey(system.instruments, symbol)
            config = system.instruments[symbol]
            status = symbol == system.current_instrument ? "✓ ACTIVE" : "  idle"
            println(@sprintf("  %s %-6s: %3d filters, population %3d", 
                           status, symbol, config.num_filters, config.population_size))
        end
    end
    
    if isempty(system.active_instruments)
        println("  No instruments configured")
    end
    
    println("=" ^ 60)
end

# =============================================================================
# CROSS-INSTRUMENT INITIALIZATION
# =============================================================================

"""
Initialize new instrument from successful one
"""
function initialize_from_instrument!(system::InstrumentGASystem, 
                                    target_symbol::String, 
                                    source_symbol::String)
    # Validate both instruments exist
    if !haskey(system.instruments, target_symbol)
        @error "Target instrument $target_symbol not found"
        return false
    end
    
    if !haskey(system.instruments, source_symbol)
        @error "Source instrument $source_symbol not found"
        return false
    end
    
    target_config = system.instruments[target_symbol]
    source_config = system.instruments[source_symbol]
    
    println("🔄 Initializing $target_symbol from $source_symbol...")
    
    # Load source parameters if they exist
    if isfile(source_config.parameter_path)
        source_params = JLD2.load(source_config.parameter_path, "parameters")
        println("  Loaded $(size(source_params, 1)) filters from $source_symbol")
        
        # Create initial population for target based on source
        target_params = zeros(Float32, target_config.num_filters, 13)
        
        for (i, period) in enumerate(target_config.fibonacci_periods)
            # Find matching period in source
            source_idx = findfirst(==(period), source_config.fibonacci_periods)
            
            if source_idx !== nothing && source_idx <= size(source_params, 1)
                # Copy parameters with small random perturbation
                base_params = source_params[source_idx, :]
                perturbation = randn(Float32, 13) * 0.1f0
                target_params[i, :] = base_params + perturbation
                println("  ✓ Initialized filter $period from source")
            else
                # Random initialization for unmatched periods
                target_params[i, :] = randn(Float32, 13)
                println("  ⚡ Random initialization for filter $period (no match)")
            end
        end
        
        # Save initialized parameters
        ensure_directory_exists(dirname(target_config.parameter_path))
        JLD2.save(target_config.parameter_path, "parameters", target_params)
        println("✅ Saved initialized parameters for $target_symbol")
        
        return true
    else
        @warn "Source parameters not found at $(source_config.parameter_path)"
        return false
    end
end

# =============================================================================
# DIRECTORY MANAGEMENT
# =============================================================================

"""
Create directory structure for an instrument
"""
function create_instrument_directories(config::InstrumentConfig)
    base_path = "data/$(config.symbol)"
    
    directories = [
        base_path,
        "$base_path/parameters",
        "$base_path/ga_workspace",
        dirname(config.config_path)
    ]
    
    for dir in directories
        ensure_directory_exists(dir)
    end
    
    println("📁 Created directory structure for $(config.symbol)")
end

"""
Ensure a directory exists, create if not
"""
function ensure_directory_exists(path::String)
    if !isdir(path)
        mkpath(path)
        println("  Created: $path")
    end
end

# =============================================================================
# INSTRUMENT CONFIGURATION I/O
# =============================================================================

"""
Load instrument-specific configuration from TOML
"""
function load_instrument_config(path::String)::Union{InstrumentConfig, Nothing}
    if !isfile(path)
        @error "Configuration file not found: $path"
        return nothing
    end
    
    try
        config = TOML.parsefile(path)
        symbol = get(config["instrument"], "symbol", "UNKNOWN")
        return parse_instrument_config(symbol, config)
    catch e
        @error "Failed to load configuration from $path: $e"
        return nothing
    end
end

"""
Save instrument configuration to TOML
"""
function save_instrument_config(config::InstrumentConfig)
    toml_dict = Dict{String, Any}()
    
    # Instrument section
    toml_dict["instrument"] = Dict(
        "symbol" => config.symbol,
        "description" => "Configuration for $(config.symbol)",
        "num_filters" => config.num_filters,
        "population_size" => config.population_size
    )
    
    # Filters section
    toml_dict["filters"] = Dict(
        "count" => config.num_filters,
        "periods" => config.fibonacci_periods
    )
    
    # GA section
    toml_dict["ga"] = Dict(
        "population_size" => config.population_size,
        "mutation_rate" => config.ga_params.mutation_rate,
        "crossover_rate" => config.ga_params.crossover_rate,
        "elite_size" => config.ga_params.elite_size,
        "tournament_size" => config.ga_params.tournament_size
    )
    
    # Optimization section
    toml_dict["optimization"] = Dict(
        "max_generations" => config.max_generations,
        "convergence_threshold" => config.convergence_threshold,
        "early_stopping_patience" => config.ga_params.early_stopping_patience
    )
    
    # Storage section
    toml_dict["storage"] = Dict(
        "parameter_path" => config.parameter_path,
        "ga_workspace_path" => config.ga_workspace_path
    )
    
    # Write to file
    ensure_directory_exists(dirname(config.config_path))
    open(config.config_path, "w") do io
        TOML.print(io, toml_dict)
    end
    
    println("💾 Saved configuration for $(config.symbol) to $(config.config_path)")
end

# =============================================================================
# MEMORY ESTIMATION
# =============================================================================

"""
Estimate memory usage for an instrument
"""
function estimate_memory_usage(config::InstrumentConfig)::Float32
    # Per filter memory (in bytes)
    population_memory = config.population_size * 13 * 4  # Float32
    fitness_memory = config.population_size * 4           # Float32
    working_memory = population_memory * 3                # Buffers
    
    per_filter_bytes = population_memory + fitness_memory + working_memory
    total_bytes = config.num_filters * per_filter_bytes
    
    # Convert to MB
    return Float32(total_bytes / (1024 * 1024))
end

"""
Check if system has sufficient memory for all instruments
"""
function check_memory_requirements(system::InstrumentGASystem)::Bool
    total_memory_mb = 0.0f0
    
    println("\n💾 Memory Usage Estimation:")
    println("=" ^ 40)
    
    for (symbol, config) in system.instruments
        memory_mb = estimate_memory_usage(config)
        total_memory_mb += memory_mb
        println(@sprintf("  %-6s: %6.1f MB", symbol, memory_mb))
    end
    
    println("-" ^ 40)
    println(@sprintf("  Total:  %6.1f MB", total_memory_mb))
    println(@sprintf("  Limit:  %6.1f MB", system.max_memory_gb * 1024))
    
    if total_memory_mb > system.max_memory_gb * 1024
        @error "Memory requirements exceed limit!"
        return false
    end
    
    println("✅ Memory requirements within limits")
    return true
end

end # module InstrumentManager