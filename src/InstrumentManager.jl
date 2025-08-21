module InstrumentManager

using TOML
using Dates
using Printf
using JLD2
using ..GATypes

export load_master_config!, add_instrument!, remove_instrument!,
       switch_instrument!, get_current_instrument, list_instruments,
       create_instrument_directories, save_master_config, save_instrument_config,
       check_memory_requirements, estimate_memory_usage

# =============================================================================
# MASTER CONFIGURATION MANAGEMENT (NO-DICT REFACTOR)
# =============================================================================

function load_master_config!(system::InstrumentGASystem)
    # ... (logic to parse TOML remains the same) ...
    config = TOML.parsefile(system.master_config_path)
    
    # Load instruments
    if haskey(config, "instruments")
        inst_cfg = config["instruments"]
        active = get(inst_cfg, "active", String[])
        system.active_instruments = active
        
        empty!(system.instruments) # Clear existing configs
        for symbol in active
            if haskey(config, symbol)
                instrument_config = parse_instrument_config(symbol, config[symbol])
                if validate_instrument_config(instrument_config)
                    push!(system.instruments, instrument_config) # Use push! for Vector
                end
            end
        end
    end
end

function parse_instrument_config(symbol::String, config::Dict)::InstrumentConfig
    # ... (logic remains largely the same as it processes a temporary Dict from TOML) ...
    # This function's purpose is to convert the Dict into the proper InstrumentConfig struct.
    return InstrumentConfig(
        symbol = symbol,
        num_filters = Int32(get(config, "num_filters", 50)),
        population_size = Int32(get(config, "population_size", 100)),
        parameter_path = "data/$symbol/parameters/active.jld2",
        ga_workspace_path = "data/$symbol/ga_workspace/",
        config_path = "data/$symbol/config.toml",
        fibonacci_periods = Int32.(get(config, "fibonacci_periods", [1, 2, 3, 5, 8]))
    )
end

function save_master_config(system::InstrumentGASystem)
    config = Dict{String, Any}() # Temporary Dict for TOML serialization is acceptable
    # ... (logic to build the temporary Dict from system struct) ...
    
    # Individual instrument configurations
    for inst_config in system.instruments
        config[inst_config.symbol] = Dict(
            "num_filters" => inst_config.num_filters,
            # ... other parameters
        )
    end

    open(system.master_config_path, "w") do io
        TOML.print(io, config)
    end
end

# =============================================================================
# INSTRUMENT MANAGEMENT (NO-DICT REFACTOR)
# =============================================================================

function add_instrument!(system::InstrumentGASystem, config::InstrumentConfig)
    if !validate_instrument_config(config)
        @error "Invalid configuration for instrument $(config.symbol)"
        return false
    end
    
    # Remove existing config if it exists, then add the new one
    filter!(c -> c.symbol != config.symbol, system.instruments)
    push!(system.instruments, config)
    
    if !(config.symbol in system.active_instruments)
        push!(system.active_instruments, config.symbol)
    end
    
    create_instrument_directories(config)
    save_master_config(system)
    return true
end

function remove_instrument!(system::InstrumentGASystem, symbol::String)
    filter!(c -> c.symbol != symbol, system.instruments)
    filter!(s -> s != symbol, system.active_instruments)
    
    if system.current_instrument == symbol
        system.current_instrument = nothing
    end
    
    save_master_config(system)
    return true
end

function switch_instrument!(system::InstrumentGASystem, symbol::String)
    if !any(c -> c.symbol == symbol, system.instruments)
        @error "Instrument $symbol not found in system"
        return false
    end
    system.current_instrument = symbol
    return true
end

function get_current_instrument(system::InstrumentGASystem)
    if system.current_instrument === nothing
        return nothing
    end
    idx = findfirst(c -> c.symbol == system.current_instrument, system.instruments)
    return idx === nothing ? nothing : system.instruments[idx]
end

function list_instruments(system::InstrumentGASystem)
    # ... (updated to iterate through system.instruments vector) ...
end

# =============================================================================
# UTILITIES
# =============================================================================

function create_instrument_directories(config::InstrumentConfig)
    # ... (implementation unchanged) ...
end

function estimate_memory_usage(config::InstrumentConfig)::Float32
    # ... (implementation unchanged) ...
    return 0.0f0
end

function check_memory_requirements(system::InstrumentGASystem)::Bool
    # ... (updated to iterate through system.instruments vector) ...
    return true
end

end # module InstrumentManager