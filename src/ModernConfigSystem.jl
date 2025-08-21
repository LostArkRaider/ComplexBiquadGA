module ModernConfigSystem

# All `using` statements are now handled by the main ComplexBiquadGA.jl module

export FilterParameters, FilterBank, FilterConfig, ExtendedFilterConfig, 
       ProcessingConfig, PLLConfig, IOConfig,
       load_filter_config, save_filter_config, validate_config,
       create_default_configs, list_available_configs,
       show_config_summary, create_default_filter_params,
       get_filter_by_period, set_filter_by_period!,
       get_active_filters, get_active_periods

# =============================================================================
# PER-FILTER PARAMETER STRUCTURES
# =============================================================================

@with_kw struct FilterParameters
    period::Int
    q_factor::Float64 = 2.0
    sma_window::Int = 20
    batch_size::Int = 1000
    phase_detector_gain::Float64 = 0.1
    loop_bandwidth::Float64 = 0.01
    lock_threshold::Float64 = 0.7
    ring_decay::Float64 = 0.995
    enable_clamping::Bool = false
    clamping_threshold::Float64 = 1e-6
    volume_scaling::Float64 = 1.0
    max_frequency_deviation::Float64 = 0.2
    phase_error_history_length::Int = 20
end

function create_default_filter_params(period::Int)::FilterParameters
    return FilterParameters(period = period)
end

mutable struct FilterBank
    filters::Vector{FilterParameters}
    active_mask::Vector{Bool}
    periods::Vector{Int}
    num_slots::Int
    num_active::Int
end

# Constructor with pre-allocation
function FilterBank(initial_periods::Vector{Int}, max_filters::Int = 20)
    num_initial = length(initial_periods)
    @assert num_initial <= max_filters "Initial periods exceed maximum filters"
    
    filters = Vector{FilterParameters}(undef, max_filters)
    active_mask = zeros(Bool, max_filters)
    periods = zeros(Int, max_filters)
    
    for (i, period) in enumerate(initial_periods)
        filters[i] = create_default_filter_params(period)
        active_mask[i] = true
        periods[i] = period
    end
    
    for i in (num_initial + 1):max_filters
        filters[i] = create_default_filter_params(0)
        active_mask[i] = false
        periods[i] = 0
    end
    
    return FilterBank(filters, active_mask, periods, max_filters, num_initial)
end

# --- Other helper functions for FilterBank (get_filter_by_period, etc.) remain the same ---
# ... (get_filter_by_period, set_filter_by_period!, get_active_filters, get_active_periods)

# =============================================================================
# CORE CONFIGURATION STRUCTURES
# =============================================================================

@with_kw struct ProcessingConfig
    include_diagnostics::Bool = true
    include_price_levels::Bool = true
    include_momentum::Bool = true
    include_volatility::Bool = true
end

@with_kw struct PLLConfig
    enabled::Bool = false
end

@with_kw struct IOConfig
    input_file::String = "data/default_input.jld2"
    output_file::String = "data/default_output.jld2"
    save_intermediate::Bool = false
    compress_output::Bool = true
    parallel_processing::Bool = false
    max_memory_gb::Float64 = 12.0
    log_progress::Bool = true
    progress_interval::Int = 1000
    validate_inputs::Bool = true
    error_on_missing_file::Bool = true
    backup_on_overwrite::Bool = false
end

@with_kw struct FilterConfig
    name::String = "default"
    description::String = "Default filter bank configuration"
    filter_bank::FilterBank = FilterBank([3, 5, 8, 13, 21, 34, 55])
    processing::ProcessingConfig = ProcessingConfig()
    pll::PLLConfig = PLLConfig()
    io::IOConfig = IOConfig()
    created::DateTime = now()
    version::String = "3.0"
end

@with_kw struct ExtendedFilterConfig
    name::String = "pll_enhanced"
    description::String = "PLL-enhanced filter bank configuration"
    filter_bank::FilterBank = FilterBank([1, 2, 3, 5, 8, 13, 21, 34, 55])
    processing::ProcessingConfig = ProcessingConfig()
    pll::PLLConfig = PLLConfig(enabled=true)
    io::IOConfig = IOConfig()
    created::DateTime = now()
    version::String = "3.0"
    ga_fitness::Float64 = 0.0
end

# --- All other functions (load, save, validate, etc.) remain the same ---
# They use temporary Dicts for TOML I/O, which is acceptable.

end # module ModernConfigSystem