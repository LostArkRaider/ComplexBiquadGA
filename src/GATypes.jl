module GATypes

using Dates
using Parameters
using Random

export InstrumentConfig, InstrumentGASystem, WriteThruStorage, FilterDefaults,
       GAParameters, ParameterRanges, validate_instrument_config, PeriodOverride

# --- Struct to replace Dictionaries ---
struct PeriodOverride
    period::Int32
    parameters::Vector{Float32}
end

# --- CORE GA PARAMETER STRUCTURES ---
@with_kw struct GAParameters
    mutation_rate::Float32 = 0.1f0
    crossover_rate::Float32 = 0.7f0
    elite_size::Int32 = 10
    tournament_size::Int32 = 5
    max_generations::Int32 = 500
    convergence_threshold::Float32 = 0.001f0
    early_stopping_patience::Int32 = 20
end

@with_kw struct ParameterRanges
    q_factor_range::Tuple{Float32, Float32} = (0.5f0, 10.0f0)
    lock_threshold_range::Tuple{Float32, Float32} = (0.0f0, 1.0f0)
    ring_decay_range::Tuple{Float32, Float32} = (0.9f0, 1.0f0)
    max_frequency_deviation_range::Tuple{Float32, Float32} = (0.01f0, 0.5f0)
    batch_size_range::Tuple{Int32, Int32} = (100, 5000)
    phase_detector_gain_range::Tuple{Float32, Float32} = (0.001f0, 1.0f0)
    loop_bandwidth_range::Tuple{Float32, Float32} = (0.0001f0, 0.1f0)
    clamping_threshold_range::Tuple{Float32, Float32} = (1.0f-8, 1.0f-3)
    volume_scaling_range::Tuple{Float32, Float32} = (0.1f0, 10.0f0)
    enable_clamping_options::Tuple{Bool, Bool} = (false, true)
    phase_error_history_length_options::Vector{Int32} = Int32[5, 10, 15, 20, 30, 40, 50]
    complex_weight_mag_range::Tuple{Float32, Float32} = (0.0f0, 2.0f0)
    complex_weight_phase_range::Tuple{Float32, Float32} = (0.0f0, Float32(2π))
end

# --- STORAGE STRUCTURES ---
@with_kw struct FilterDefaults
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
    default_complex_weight_real::Float32 = 1.0f0
    default_complex_weight_imag::Float32 = 0.0f0
    period_overrides::Vector{PeriodOverride} = PeriodOverride[]
end

mutable struct WriteThruStorage{M<:AbstractMatrix{Float32}, B<:AbstractVector{Bool}}
    active_params::M
    jld2_path::String
    last_sync::DateTime
    sync_interval::Int32
    dirty_filters::B
    pending_updates::Int32
    default_config::FilterDefaults
end

# --- MULTI-INSTRUMENT MANAGEMENT STRUCTURES ---
@with_kw struct InstrumentConfig
    symbol::String
    num_filters::Int32
    population_size::Int32
    parameter_path::String
    ga_workspace_path::String
    config_path::String
    fibonacci_periods::Vector{Int32}
    max_generations::Int32 = 500
    convergence_threshold::Float32 = 0.001f0
    initialization_source::Union{String, Nothing} = nothing
    ga_params::GAParameters = GAParameters()
end

mutable struct InstrumentGASystem
    instruments::Vector{InstrumentConfig}
    active_instruments::Vector{String}
    current_instrument::Union{String, Nothing}
    master_config_path::String
    gpu_enabled::Bool
    max_memory_gb::Float32
    checkpoint_interval::Int32
end

function validate_instrument_config(config::InstrumentConfig)::Bool
    if !(20 <= config.num_filters <= 256)
        @error "Invalid filter count: $(config.num_filters) (must be 20-256)"
        return false
    end
    if !(10 <= config.population_size <= 1000)
        @error "Invalid population size: $(config.population_size) (must be 10-1000)"
        return false
    end
    if isempty(config.fibonacci_periods)
        @error "No Fibonacci periods specified"
        return false
    end
    return true
end

end # module GATypes