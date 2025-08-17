# src/GATypes.jl - Core Type Definitions for GA Optimization System
# Multi-Instrument Support and Storage Architecture (Chunk 2)

module GATypes

using Dates
using Parameters

export InstrumentConfig, InstrumentGASystem, WriteThruStorage, FilterDefaults,
       SingleFilterGA, FilterBankGA, GAParameters, ParameterRanges,
       validate_instrument_config, get_default_chromosome

# =============================================================================
# CORE GA PARAMETER STRUCTURES (MUST BE DEFINED FIRST)
# =============================================================================

# GA algorithm parameters
@with_kw struct GAParameters
    mutation_rate::Float32 = 0.1f0
    crossover_rate::Float32 = 0.7f0
    elite_size::Int32 = 10
    tournament_size::Int32 = 5
    max_generations::Int32 = 500
    convergence_threshold::Float32 = 0.001f0
    early_stopping_patience::Int32 = 20
end

# Parameter ranges for each of the 13 parameters
@with_kw struct ParameterRanges
    # Linear scaling parameters
    q_factor_range::Tuple{Float32, Float32} = (0.5f0, 10.0f0)
    lock_threshold_range::Tuple{Float32, Float32} = (0.0f0, 1.0f0)
    ring_decay_range::Tuple{Float32, Float32} = (0.9f0, 1.0f0)
    max_frequency_deviation_range::Tuple{Float32, Float32} = (0.01f0, 0.5f0)
    
    # Logarithmic scaling parameters
    batch_size_range::Tuple{Int32, Int32} = (100, 5000)
    phase_detector_gain_range::Tuple{Float32, Float32} = (0.001f0, 1.0f0)
    loop_bandwidth_range::Tuple{Float32, Float32} = (0.0001f0, 0.1f0)
    clamping_threshold_range::Tuple{Float32, Float32} = (1.0f-8, 1.0f-3)
    volume_scaling_range::Tuple{Float32, Float32} = (0.1f0, 10.0f0)
    
    # Binary parameter
    enable_clamping_options::Tuple{Bool, Bool} = (false, true)
    
    # Discrete parameter
    phase_error_history_length_options::Vector{Int32} = Int32[5, 10, 15, 20, 30, 40, 50]
    
    # Complex weight (magnitude and phase)
    complex_weight_mag_range::Tuple{Float32, Float32} = (0.0f0, 2.0f0)
    complex_weight_phase_range::Tuple{Float32, Float32} = (0.0f0, Float32(2π))
end

# =============================================================================
# STORAGE STRUCTURES (DEFINED BEFORE GA TYPES THAT USE THEM)
# =============================================================================

# Default configuration for new filters
@with_kw struct FilterDefaults
    # Default parameter values (Float32 for GPU efficiency)
    default_q_factor::Float32 = 2.0f0
    default_batch_size::Int32 = 1000
    default_pll_gain::Float32 = 0.1f0
    default_loop_bandwidth::Float32 = 0.01f0
    default_lock_threshold::Float32 = 0.7f0
    default_ring_decay::Float32 = 0.995f0
    default_enable_clamping::Bool = false
    default_clamping_threshold::Float32 = 1.0f-6
    default_volume_scaling::Float32 = 1.0f0
    default_max_frequency_deviation::Float32 = 0.2f0
    default_phase_error_history_length::Int32 = 20
    
    # Default complex weight
    default_complex_weight_real::Float32 = 1.0f0
    default_complex_weight_imag::Float32 = 0.0f0
    
    # Period-specific overrides (optional)
    period_overrides::Dict{Int32, Vector{Float32}} = Dict{Int32, Vector{Float32}}()
end

# Automatic persistence to JLD2
mutable struct WriteThruStorage
    # Memory-resident parameters
    active_params::Matrix{Float32}           # num_filters × 13
    
    # JLD2 backing store
    jld2_path::String
    last_sync::DateTime
    sync_interval::Int32                     # Generations between syncs
    
    # Change tracking
    dirty_filters::BitVector                 # Which filters changed
    pending_updates::Int32
    
    # TOML defaults for new filters
    default_config::FilterDefaults
    
    # Constructor
    function WriteThruStorage(num_filters::Int32, jld2_path::String, 
                            sync_interval::Int32 = 10)
        active_params = zeros(Float32, num_filters, 13)
        dirty_filters = falses(num_filters)
        default_config = FilterDefaults()
        
        new(active_params, jld2_path, now(), sync_interval,
            dirty_filters, 0, default_config)
    end
end

# =============================================================================
# GA OPTIMIZATION STRUCTURES (NOW CAN USE WriteThruStorage)
# =============================================================================

# Stub for individual filter GA (Chunk 1 will implement fully)
mutable struct SingleFilterGA
    # Filter identity
    period::Int32                            # Fibonacci period
    filter_index::Int32                      # Position in bank
    
    # GA population (13 parameters per individual)
    population::Matrix{Float32}              # population_size × 13
    fitness::Vector{Float32}                 # population_size
    
    # Best solution tracking
    best_chromosome::Vector{Float32}         # 13 parameters
    best_fitness::Float32
    generations_since_improvement::Int32
    
    # Evolution state
    generation::Int32
    converged::Bool
    
    # Parameter bounds
    param_ranges::ParameterRanges
    
    # Constructor stub
    function SingleFilterGA(period::Int32, filter_index::Int32, pop_size::Int32)
        population = randn(Float32, pop_size, 13)
        fitness = zeros(Float32, pop_size)
        best_chromosome = zeros(Float32, 13)
        param_ranges = ParameterRanges()
        
        new(period, filter_index, population, fitness, 
            best_chromosome, 0.0f0, 0, 0, false, param_ranges)
    end
end

# Stub for filter bank GA (Chunk 1 will implement fully)
mutable struct FilterBankGA
    instrument::String                        # "YM", "ES", etc.
    num_filters::Int32                       # Total filters
    population_size::Int32                   # Same for all filters
    
    # Independent GA for each filter
    filter_gas::Vector{SingleFilterGA}       # Length = num_filters
    
    # Shared configuration
    ga_params::GAParameters                  # Mutation rate, etc.
    
    # Write-through storage
    storage::WriteThruStorage                # Now this is defined!
    
    # Performance tracking
    generation::Int32
    total_evaluations::Int64
    best_fitness_history::Vector{Float32}
end

# =============================================================================
# MULTI-INSTRUMENT MANAGEMENT STRUCTURES
# =============================================================================

# Per-instrument configuration
@with_kw struct InstrumentConfig
    symbol::String                           # "YM", "ES", etc.
    num_filters::Int32                       # 20-256
    population_size::Int32                   # Same for all filters in instrument
    
    # Storage paths
    parameter_path::String                   # "data/YM/parameters/active.jld2"
    ga_workspace_path::String               # "data/YM/ga_workspace/"
    config_path::String                     # "data/YM/config.toml"
    
    # Filter specifications
    fibonacci_periods::Vector{Int32}        # [1,2,3,5,8,13,21,34,55...]
    
    # Optimization settings
    max_generations::Int32 = 500
    convergence_threshold::Float32 = 0.001f0
    
    # Cross-instrument initialization
    initialization_source::Union{String, Nothing} = nothing  # "YM" to copy from
    
    # GA parameters
    ga_params::GAParameters = GAParameters()
end

# Master system managing all instruments
mutable struct InstrumentGASystem
    # Active instruments and their configurations
    instruments::Dict{String, InstrumentConfig}  # "YM" => config
    active_instruments::Vector{String}           # ["YM", "ES", "NQ"]
    
    # Currently optimizing instrument (sequential processing)
    current_instrument::Union{String, Nothing}
    
    # Master configuration
    master_config_path::String                   # "config/master.toml"
    
    # Global settings
    gpu_enabled::Bool
    max_memory_gb::Float32
    checkpoint_interval::Int32
    
    # Constructor
    function InstrumentGASystem(master_config_path::String = "data/master_config.toml")
        new(
            Dict{String, InstrumentConfig}(),
            String[],
            nothing,
            master_config_path,
            false,  # GPU disabled by default
            12.0f0, # 12GB memory limit
            50      # Checkpoint every 50 generations
        )
    end
end

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

# Get default chromosome for a filter period
function get_default_chromosome(period::Int32, defaults::FilterDefaults)::Vector{Float32}
    # Check for period-specific overrides
    if haskey(defaults.period_overrides, period)
        return defaults.period_overrides[period]
    end
    
    # Return default parameters as chromosome
    return Float32[
        defaults.default_q_factor,
        Float32(log(defaults.default_batch_size)),  # Log scale
        Float32(log(defaults.default_pll_gain)),    # Log scale
        Float32(log(defaults.default_loop_bandwidth)), # Log scale
        defaults.default_lock_threshold,
        defaults.default_ring_decay,
        defaults.default_enable_clamping ? 1.0f0 : 0.0f0,  # Binary
        Float32(log(defaults.default_clamping_threshold)),  # Log scale
        Float32(log(defaults.default_volume_scaling)),      # Log scale
        defaults.default_max_frequency_deviation,
        Float32(defaults.default_phase_error_history_length),
        defaults.default_complex_weight_real,
        defaults.default_complex_weight_imag
    ]
end

# Validate instrument configuration
function validate_instrument_config(config::InstrumentConfig)::Bool
    # Check filter count
    if config.num_filters < 20 || config.num_filters > 256
        @error "Invalid filter count: $(config.num_filters) (must be 20-256)"
        return false
    end
    
    # Check population size
    if config.population_size < 10 || config.population_size > 1000
        @error "Invalid population size: $(config.population_size) (must be 10-1000)"
        return false
    end
    
    # Check Fibonacci periods
    if isempty(config.fibonacci_periods)
        @error "No Fibonacci periods specified"
        return false
    end
    
    # Verify paths exist
    if !isdir(dirname(config.parameter_path))
        @warn "Parameter directory does not exist: $(dirname(config.parameter_path))"
    end
    
    if !isdir(config.ga_workspace_path)
        @warn "GA workspace directory does not exist: $(config.ga_workspace_path)"
    end
    
    return true
end

end # module GATypes