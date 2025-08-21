module StorageSystem

using JLD2
using TOML
using Dates
using Printf
using ..GATypes

export sync_to_storage!, load_from_storage!, mark_filter_dirty!,
       create_checkpoint, restore_from_checkpoint, list_checkpoints,
       load_filter_defaults, save_filter_defaults, apply_defaults!,
       get_active_parameters, set_active_parameters!,
       initialize_storage, cleanup_old_checkpoints, StorageStats

# --- Struct to replace Dictionary for statistics ---
struct StorageStats
    num_filters::Int
    parameters_per_filter::Int
    total_parameters::Int
    pending_updates::Int32
    last_sync::DateTime
    time_since_sync::Period
    storage_path::String
    storage_exists::Bool
    storage_size_kb::Float64
    num_checkpoints::Int
end

# =============================================================================
# STORAGE INITIALIZATION & I/O
# =============================================================================

function initialize_storage(config::InstrumentConfig; ArrayType::Type=Array{Float32})
    storage = WriteThruStorage(
        ArrayType(undef, config.num_filters, 13),
        config.parameter_path,
        now(),
        10, # Default sync interval
        ArrayType{Bool}(undef, config.num_filters),
        Int32(0),
        FilterDefaults()
    )
    fill!(storage.dirty_filters, false)

    if isfile(config.parameter_path)
        load_from_storage!(storage)
    else
        # apply_defaults logic might need to be here or called after
    end
    return storage
end

function sync_to_storage!(storage::WriteThruStorage)
    if storage.pending_updates == 0
        return
    end
    
    try
        dir = dirname(storage.jld2_path)
        if !isdir(dir)
            mkpath(dir)
        end
        
        # Data must be on CPU to save to JLD2
        cpu_params = Array(storage.active_params)
        
        JLD2.save(storage.jld2_path, "parameters", cpu_params)
        
        storage.last_sync = now()
        storage.pending_updates = 0
        fill!(storage.dirty_filters, false)
        
    catch e
        @error "Failed to sync to storage: $e"
    end
end

function load_from_storage!(storage::WriteThruStorage)
    if !isfile(storage.jld2_path)
        @warn "Storage file not found: $(storage.jld2_path)"
        return false
    end
    
    try
        data = JLD2.load(storage.jld2_path)
        if haskey(data, "parameters")
            cpu_params = data["parameters"]
            if size(cpu_params) == size(storage.active_params)
                copyto!(storage.active_params, cpu_params) # Handles CPU->GPU copy
            else
                @error "Parameter dimension mismatch in storage file."
                return false
            end
        end
        return true
    catch e
        @error "Failed to load from storage: $e"
        return false
    end
end

# =============================================================================
# PARAMETER AND CHECKPOINT MANAGEMENT
# =============================================================================

function set_active_parameters!(storage::WriteThruStorage, filter_index::Int32, params::AbstractVector{Float32})
    if 1 <= filter_index <= size(storage.active_params, 1) && length(params) == 13
        storage.active_params[filter_index, :] .= params
        if !storage.dirty_filters[filter_index]
            storage.dirty_filters[filter_index] = true
            storage.pending_updates += 1
        end
    else
        @error "Invalid index or parameter length in set_active_parameters!"
    end
end

function get_active_parameters(storage::WriteThruStorage, filter_index::Int32)
    return @view storage.active_params[filter_index, :]
end

function create_checkpoint(storage::WriteThruStorage, generation::Int32, fitness::Float32 = 0.0f0)::String
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    checkpoint_dir = dirname(storage.jld2_path)
    checkpoint_file = joinpath(checkpoint_dir, "checkpoint_gen$(generation)_$(timestamp).jld2")
    
    try
        JLD2.save(checkpoint_file,
                 "parameters", Array(storage.active_params), # Save CPU copy
                 "generation", generation,
                 "fitness", fitness)
        return checkpoint_file
    catch e
        @error "Failed to create checkpoint: $e"
        return ""
    end
end

function restore_from_checkpoint(storage::WriteThruStorage, checkpoint_file::String)::Bool
    # ... logic to load from checkpoint and sync ...
    return true
end

function list_checkpoints(storage::WriteThruStorage)::Vector{String}
    # ... logic to find and sort checkpoints ...
    return String[]
end

# =============================================================================
# DEFAULT PARAMETER MANAGEMENT (NO-DICT REFACTOR)
# =============================================================================

function load_filter_defaults(path::String)::FilterDefaults
    # ... logic to parse TOML and build FilterDefaults with Vector{PeriodOverride} ...
    return FilterDefaults()
end

function save_filter_defaults(defaults::FilterDefaults, path::String)
    # ... logic to create a temporary Dict and write to TOML ...
end

function apply_defaults!(storage::WriteThruStorage, fibonacci_periods::Vector{Int32})
    # ... logic to apply defaults to storage.active_params ...
end

function get_storage_stats(storage::WriteThruStorage)::StorageStats
    storage_size = isfile(storage.jld2_path) ? round(stat(storage.jld2_path).size / 1024, digits=1) : 0.0
    
    return StorageStats(
        size(storage.active_params, 1),
        size(storage.active_params, 2),
        length(storage.active_params),
        storage.pending_updates,
        storage.last_sync,
        now() - storage.last_sync,
        storage.jld2_path,
        isfile(storage.jld2_path),
        storage_size,
        length(list_checkpoints(storage))
    )
end

end # module StorageSystem