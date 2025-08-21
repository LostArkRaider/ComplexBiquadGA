module ConfigurationLoader

using TOML
using Dates
using ..GATypes
using ..InstrumentManager
using ..StorageSystem

export initialize_ga_system, create_default_configs,
       load_or_create_instrument, validate_system_setup,
       SystemReport, InstrumentReport

# --- Structs to replace Dictionaries for reporting ---
struct InstrumentReport
    symbol::String
    num_filters::Int32
    population_size::Int32
    fibonacci_periods::Vector{Int32}
    max_generations::Int32
    parameters_exist::Bool
    parameter_file_size_kb::Float64
    storage_exists::Bool
    workspace_exists::Bool
    num_checkpoints::Int
end

struct SystemReport
    timestamp::DateTime
    master_config::String
    gpu_enabled::Bool
    memory_limit_gb::Float32
    current_instrument::String
    total_memory_mb::Float32
    memory_usage_percent::Float32
    instrument_reports::Vector{InstrumentReport}
end

# =============================================================================
# SYSTEM INITIALIZATION
# =============================================================================

function initialize_ga_system(master_config_path::String = "data/master_config.toml")
    system = InstrumentGASystem(
        InstrumentConfig[], # instruments
        String[],           # active_instruments
        nothing,            # current_instrument
        master_config_path,
        false, 12.0f0, 50
    )
    
    load_master_config!(system)
    
    if !validate_system_setup(system)
        @error "System validation failed"
    end
    
    check_memory_requirements(system)
    list_instruments(system)
    
    return system
end

function validate_system_setup(system::InstrumentGASystem)::Bool
    # ... logic remains the same, iterates through system.instruments vector ...
    return true
end

# =============================================================================
# CONFIGURATION CREATION AND LOADING
# =============================================================================

function create_default_configs()
    # ... implementation unchanged ...
end

function load_or_create_instrument(symbol::String, system::InstrumentGASystem)
    # Check if already loaded
    idx = findfirst(c -> c.symbol == symbol, system.instruments)
    if idx !== nothing
        return system.instruments[idx]
    end
    # ... rest of creation logic is the same ...
end

# =============================================================================
# SYSTEM STATUS AND REPORTING (NO-DICT REFACTOR)
# =============================================================================

function generate_system_report(system::InstrumentGASystem)::SystemReport
    instrument_reports = InstrumentReport[]
    
    for config in system.instruments
        storage_path = config.parameter_path
        workspace_path = config.ga_workspace_path
        
        param_file_size = isfile(storage_path) ? round(stat(storage_path).size / 1024, digits=1) : 0.0
        
        num_checkpoints = 0
        if isdir(dirname(storage_path))
            checkpoints = filter(f -> startswith(f, "checkpoint_"), readdir(dirname(storage_path)))
            num_checkpoints = length(checkpoints)
        end

        inst_report = InstrumentReport(
            config.symbol,
            config.num_filters,
            config.population_size,
            config.fibonacci_periods,
            config.max_generations,
            isfile(config.parameter_path),
            param_file_size,
            isfile(storage_path),
            isdir(workspace_path),
            num_checkpoints
        )
        push!(instrument_reports, inst_report)
    end
    
    total_memory = 0.0f0
    for config in system.instruments
        total_memory += estimate_memory_usage(config)
    end
    
    return SystemReport(
        now(),
        system.master_config_path,
        system.gpu_enabled,
        system.max_memory_gb,
        something(system.current_instrument, "none"),
        round(total_memory, digits=1),
        round(100 * total_memory / (system.max_memory_gb * 1024), digits=1),
        instrument_reports
    )
end

end # module ConfigurationLoader