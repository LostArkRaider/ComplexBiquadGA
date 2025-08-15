# src/StorageSystem.jl - Write-Through Storage System with JLD2 Persistence
# Automatic persistence, change tracking, and checkpoint/recovery

module StorageSystem

using JLD2
using TOML
using Dates
using Printf

# Import GATypes
if !isdefined(Main, :GATypes)
    include("GATypes.jl")
end
using Main.GATypes

export sync_to_storage!, load_from_storage!, mark_filter_dirty!,
       create_checkpoint, restore_from_checkpoint, list_checkpoints,
       load_filter_defaults, save_filter_defaults, apply_defaults!,
       get_active_parameters, set_active_parameters!,
       initialize_storage, cleanup_old_checkpoints

# =============================================================================
# STORAGE INITIALIZATION
# =============================================================================

"""
Initialize storage system for an instrument
"""
function initialize_storage(config::InstrumentConfig)::WriteThruStorage
    # Create storage with proper dimensions
    storage = WriteThruStorage(
        config.num_filters,
        config.parameter_path,
        10  # Default sync interval
    )
    
    # Load defaults if available
    defaults_path = "data/$(config.symbol)/defaults.toml"
    if isfile(defaults_path)
        storage.default_config = load_filter_defaults(defaults_path)
        println("📋 Loaded defaults from $defaults_path")
    end
    
    # Try to load existing parameters
    if isfile(config.parameter_path)
        load_from_storage!(storage)
        println("✅ Loaded existing parameters from $(config.parameter_path)")
    else
        # Initialize with defaults
        apply_defaults!(storage, config.fibonacci_periods)
        println("🆕 Initialized parameters with defaults")
    end
    
    return storage
end

# =============================================================================
# CORE STORAGE OPERATIONS
# =============================================================================

"""
Sync memory-resident parameters to JLD2 backing store
"""
function sync_to_storage!(storage::WriteThruStorage)
    # Check if any filters are dirty
    if storage.pending_updates == 0
        return  # Nothing to sync
    end
    
    try
        # Create directory if needed
        dir = dirname(storage.jld2_path)
        if !isdir(dir)
            mkpath(dir)
        end
        
        # Save to JLD2
        JLD2.save(storage.jld2_path, 
                 "parameters", storage.active_params,
                 "last_sync", storage.last_sync,
                 "dirty_filters", storage.dirty_filters,
                 "timestamp", now())
        
        # Update sync metadata
        storage.last_sync = now()
        storage.pending_updates = 0
        fill!(storage.dirty_filters, false)
        
        println("💾 Synced $(sum(storage.dirty_filters)) filters to $(basename(storage.jld2_path))")
        
    catch e
        @error "Failed to sync to storage: $e"
        rethrow(e)
    end
end

"""
Load parameters from JLD2 backing store
"""
function load_from_storage!(storage::WriteThruStorage)
    if !isfile(storage.jld2_path)
        @warn "Storage file not found: $(storage.jld2_path)"
        return false
    end
    
    try
        data = JLD2.load(storage.jld2_path)
        
        # Load parameters
        if haskey(data, "parameters")
            params = data["parameters"]
            
            # Validate dimensions
            if size(params) == size(storage.active_params)
                storage.active_params .= params
                println("✅ Loaded $(size(params, 1)) filters from storage")
            else
                @error "Parameter dimension mismatch: expected $(size(storage.active_params)), got $(size(params))"
                return false
            end
        end
        
        # Load metadata
        if haskey(data, "last_sync")
            storage.last_sync = data["last_sync"]
        end
        
        if haskey(data, "dirty_filters")
            storage.dirty_filters .= data["dirty_filters"]
            storage.pending_updates = sum(storage.dirty_filters)
        end
        
        return true
        
    catch e
        @error "Failed to load from storage: $e"
        return false
    end
end

"""
Mark a filter as dirty (needs sync)
"""
function mark_filter_dirty!(storage::WriteThruStorage, filter_index::Int32)
    if filter_index < 1 || filter_index > length(storage.dirty_filters)
        @error "Invalid filter index: $filter_index"
        return
    end
    
    if !storage.dirty_filters[filter_index]
        storage.dirty_filters[filter_index] = true
        storage.pending_updates += 1
    end
end

"""
Get active parameters for a specific filter
"""
function get_active_parameters(storage::WriteThruStorage, filter_index::Int32)::Vector{Float32}
    if filter_index < 1 || filter_index > size(storage.active_params, 1)
        @error "Invalid filter index: $filter_index"
        return Float32[]
    end
    
    return storage.active_params[filter_index, :]
end

"""
Set active parameters for a specific filter
"""
function set_active_parameters!(storage::WriteThruStorage, filter_index::Int32, 
                               params::Vector{Float32})
    if filter_index < 1 || filter_index > size(storage.active_params, 1)
        @error "Invalid filter index: $filter_index"
        return
    end
    
    if length(params) != 13
        @error "Invalid parameter vector length: $(length(params)) (expected 13)"
        return
    end
    
    storage.active_params[filter_index, :] = params
    mark_filter_dirty!(storage, filter_index)
end

# =============================================================================
# CHECKPOINT MANAGEMENT
# =============================================================================

"""
Create a checkpoint of current parameters
"""
function create_checkpoint(storage::WriteThruStorage, generation::Int32, 
                          fitness::Float32 = 0.0f0)::String
    # Generate checkpoint filename
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    checkpoint_dir = dirname(storage.jld2_path)
    checkpoint_file = joinpath(checkpoint_dir, 
                              "checkpoint_gen$(generation)_$(timestamp).jld2")
    
    try
        # Save checkpoint
        JLD2.save(checkpoint_file,
                 "parameters", storage.active_params,
                 "generation", generation,
                 "fitness", fitness,
                 "timestamp", now(),
                 "dirty_filters", storage.dirty_filters)
        
        println("✅ Created checkpoint: $(basename(checkpoint_file))")
        return checkpoint_file
        
    catch e
        @error "Failed to create checkpoint: $e"
        return ""
    end
end

"""
Restore parameters from a checkpoint
"""
function restore_from_checkpoint(storage::WriteThruStorage, checkpoint_file::String)::Bool
    if !isfile(checkpoint_file)
        @error "Checkpoint file not found: $checkpoint_file"
        return false
    end
    
    try
        data = JLD2.load(checkpoint_file)
        
        if haskey(data, "parameters")
            params = data["parameters"]
            
            # Validate dimensions
            if size(params) == size(storage.active_params)
                storage.active_params .= params
                
                # Mark all filters as dirty after restore
                fill!(storage.dirty_filters, true)
                storage.pending_updates = length(storage.dirty_filters)
                
                # Get checkpoint info
                gen = get(data, "generation", 0)
                fitness = get(data, "fitness", 0.0f0)
                
                println("✅ Restored from checkpoint: generation $gen, fitness $fitness")
                
                # Immediately sync to main storage
                sync_to_storage!(storage)
                
                return true
            else
                @error "Checkpoint dimension mismatch"
                return false
            end
        else
            @error "No parameters found in checkpoint"
            return false
        end
        
    catch e
        @error "Failed to restore from checkpoint: $e"
        return false
    end
end

"""
List available checkpoints
"""
function list_checkpoints(storage::WriteThruStorage)::Vector{String}
    checkpoint_dir = dirname(storage.jld2_path)
    
    if !isdir(checkpoint_dir)
        return String[]
    end
    
    # Find all checkpoint files
    files = readdir(checkpoint_dir)
    checkpoints = filter(f -> startswith(f, "checkpoint_") && endswith(f, ".jld2"), files)
    
    # Sort by modification time (newest first)
    checkpoint_paths = [joinpath(checkpoint_dir, f) for f in checkpoints]
    sorted_indices = sortperm([stat(f).mtime for f in checkpoint_paths], rev=true)
    
    return checkpoint_paths[sorted_indices]
end

"""
Clean up old checkpoints, keeping only the most recent n
"""
function cleanup_old_checkpoints(storage::WriteThruStorage, keep_n::Int = 5)
    checkpoints = list_checkpoints(storage)
    
    if length(checkpoints) <= keep_n
        return  # Nothing to clean up
    end
    
    # Delete old checkpoints
    to_delete = checkpoints[(keep_n+1):end]
    
    for checkpoint in to_delete
        try
            rm(checkpoint)
            println("🗑️  Deleted old checkpoint: $(basename(checkpoint))")
        catch e
            @warn "Failed to delete checkpoint $checkpoint: $e"
        end
    end
    
    println("✅ Cleaned up $(length(to_delete)) old checkpoints")
end

# =============================================================================
# DEFAULT PARAMETER MANAGEMENT
# =============================================================================

"""
Load filter defaults from TOML file
"""
function load_filter_defaults(path::String)::FilterDefaults
    if !isfile(path)
        @warn "Defaults file not found: $path"
        return FilterDefaults()
    end
    
    try
        config = TOML.parsefile(path)
        
        # Parse default parameters
        defaults = get(config, "default_parameters", Dict())
        
        filter_defaults = FilterDefaults(
            default_q_factor = Float32(get(defaults, "q_factor", 2.0)),
            default_batch_size = Int32(get(defaults, "batch_size", 1000)),
            default_pll_gain = Float32(get(defaults, "phase_detector_gain", 0.1)),
            default_loop_bandwidth = Float32(get(defaults, "loop_bandwidth", 0.01)),
            default_lock_threshold = Float32(get(defaults, "lock_threshold", 0.7)),
            default_ring_decay = Float32(get(defaults, "ring_decay", 0.995)),
            default_enable_clamping = Bool(get(defaults, "enable_clamping", false)),
            default_clamping_threshold = Float32(get(defaults, "clamping_threshold", 1e-6)),
            default_volume_scaling = Float32(get(defaults, "volume_scaling", 1.0)),
            default_max_frequency_deviation = Float32(get(defaults, "max_frequency_deviation", 0.2)),
            default_phase_error_history_length = Int32(get(defaults, "phase_error_history_length", 20)),
            default_complex_weight_real = Float32(get(defaults, "complex_weight_real", 1.0)),
            default_complex_weight_imag = Float32(get(defaults, "complex_weight_imag", 0.0))
        )
        
        # Parse period-specific overrides
        if haskey(config, "period_overrides")
            overrides = Dict{Int32, Vector{Float32}}()
            
            for (period_str, override_dict) in config["period_overrides"]
                period = parse(Int32, period_str)
                
                # Build override chromosome
                override_params = Float32[
                    get(override_dict, "q_factor", filter_defaults.default_q_factor),
                    log(get(override_dict, "batch_size", filter_defaults.default_batch_size)),
                    log(get(override_dict, "phase_detector_gain", filter_defaults.default_pll_gain)),
                    log(get(override_dict, "loop_bandwidth", filter_defaults.default_loop_bandwidth)),
                    get(override_dict, "lock_threshold", filter_defaults.default_lock_threshold),
                    get(override_dict, "ring_decay", filter_defaults.default_ring_decay),
                    get(override_dict, "enable_clamping", filter_defaults.default_enable_clamping) ? 1.0f0 : 0.0f0,
                    log(get(override_dict, "clamping_threshold", filter_defaults.default_clamping_threshold)),
                    log(get(override_dict, "volume_scaling", filter_defaults.default_volume_scaling)),
                    get(override_dict, "max_frequency_deviation", filter_defaults.default_max_frequency_deviation),
                    Float32(get(override_dict, "phase_error_history_length", filter_defaults.default_phase_error_history_length)),
                    get(override_dict, "complex_weight_real", filter_defaults.default_complex_weight_real),
                    get(override_dict, "complex_weight_imag", filter_defaults.default_complex_weight_imag)
                ]
                
                overrides[period] = override_params
            end
            
            filter_defaults = FilterDefaults(
                filter_defaults.default_q_factor,
                filter_defaults.default_batch_size,
                filter_defaults.default_pll_gain,
                filter_defaults.default_loop_bandwidth,
                filter_defaults.default_lock_threshold,
                filter_defaults.default_ring_decay,
                filter_defaults.default_enable_clamping,
                filter_defaults.default_clamping_threshold,
                filter_defaults.default_volume_scaling,
                filter_defaults.default_max_frequency_deviation,
                filter_defaults.default_phase_error_history_length,
                filter_defaults.default_complex_weight_real,
                filter_defaults.default_complex_weight_imag,
                overrides
            )
        end
        
        return filter_defaults
        
    catch e
        @error "Failed to load defaults from $path: $e"
        return FilterDefaults()
    end
end

"""
Save filter defaults to TOML file
"""
function save_filter_defaults(defaults::FilterDefaults, path::String)
    config = Dict{String, Any}()
    
    # Default parameters section
    config["default_parameters"] = Dict(
        "q_factor" => defaults.default_q_factor,
        "batch_size" => defaults.default_batch_size,
        "phase_detector_gain" => defaults.default_pll_gain,
        "loop_bandwidth" => defaults.default_loop_bandwidth,
        "lock_threshold" => defaults.default_lock_threshold,
        "ring_decay" => defaults.default_ring_decay,
        "enable_clamping" => defaults.default_enable_clamping,
        "clamping_threshold" => defaults.default_clamping_threshold,
        "volume_scaling" => defaults.default_volume_scaling,
        "max_frequency_deviation" => defaults.default_max_frequency_deviation,
        "phase_error_history_length" => defaults.default_phase_error_history_length,
        "complex_weight_real" => defaults.default_complex_weight_real,
        "complex_weight_imag" => defaults.default_complex_weight_imag
    )
    
    # Period-specific overrides
    if !isempty(defaults.period_overrides)
        config["period_overrides"] = Dict{String, Any}()
        
        for (period, params) in defaults.period_overrides
            # Decode from chromosome format
            config["period_overrides"][string(period)] = Dict(
                "q_factor" => params[1],
                "batch_size" => Int32(exp(params[2])),
                "phase_detector_gain" => exp(params[3]),
                "loop_bandwidth" => exp(params[4]),
                "lock_threshold" => params[5],
                "ring_decay" => params[6],
                "enable_clamping" => params[7] > 0.5,
                "clamping_threshold" => exp(params[8]),
                "volume_scaling" => exp(params[9]),
                "max_frequency_deviation" => params[10],
                "phase_error_history_length" => Int32(params[11]),
                "complex_weight_real" => params[12],
                "complex_weight_imag" => params[13]
            )
        end
    end
    
    # Ensure directory exists
    dir = dirname(path)
    if !isdir(dir)
        mkpath(dir)
    end
    
    # Write to file
    open(path, "w") do io
        TOML.print(io, config)
    end
    
    println("💾 Saved filter defaults to $path")
end

"""
Apply defaults to storage for initialization
"""
function apply_defaults!(storage::WriteThruStorage, fibonacci_periods::Vector{Int32})
    for (i, period) in enumerate(fibonacci_periods)
        if i > size(storage.active_params, 1)
            break  # Don't exceed storage dimensions
        end
        
        # Get default chromosome for this period
        default_params = get_default_chromosome(period, storage.default_config)
        
        # Set parameters
        storage.active_params[i, :] = default_params
        mark_filter_dirty!(storage, Int32(i))
    end
    
    println("✅ Applied defaults to $(length(fibonacci_periods)) filters")
    
    # Sync to storage
    sync_to_storage!(storage)
end

# =============================================================================
# STORAGE STATISTICS
# =============================================================================

"""
Get storage statistics
"""
function get_storage_stats(storage::WriteThruStorage)::Dict{String, Any}
    stats = Dict{String, Any}()
    
    stats["num_filters"] = size(storage.active_params, 1)
    stats["parameters_per_filter"] = size(storage.active_params, 2)
    stats["total_parameters"] = prod(size(storage.active_params))
    stats["pending_updates"] = storage.pending_updates
    stats["last_sync"] = storage.last_sync
    stats["time_since_sync"] = now() - storage.last_sync
    stats["storage_path"] = storage.jld2_path
    stats["storage_exists"] = isfile(storage.jld2_path)
    
    if isfile(storage.jld2_path)
        stats["storage_size_kb"] = round(stat(storage.jld2_path).size / 1024, digits=1)
    else
        stats["storage_size_kb"] = 0.0
    end
    
    # Count checkpoints
    checkpoints = list_checkpoints(storage)
    stats["num_checkpoints"] = length(checkpoints)
    
    return stats
end

"""
Print storage status
"""
function print_storage_status(storage::WriteThruStorage)
    stats = get_storage_stats(storage)
    
    println("\n📊 Storage Status:")
    println("=" ^ 50)
    println("  Filters: $(stats["num_filters"])")
    println("  Parameters per filter: $(stats["parameters_per_filter"])")
    println("  Total parameters: $(stats["total_parameters"])")
    println("  Pending updates: $(stats["pending_updates"])")
    println("  Last sync: $(stats["last_sync"])")
    println("  Time since sync: $(stats["time_since_sync"])")
    println("  Storage file: $(basename(stats["storage_path"]))")
    println("  Storage size: $(stats["storage_size_kb"]) KB")
    println("  Checkpoints: $(stats["num_checkpoints"])")
    println("=" ^ 50)
end

end # module StorageSystem