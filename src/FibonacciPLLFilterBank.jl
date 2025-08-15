# FibonacciPLLFilterBank.jl - Main package module
# Place this in src/FibonacciPLLFilterBank.jl
# This is the main module that properly manages all submodules

module FibonacciPLLFilterBank

# Standard library dependencies
using Random
using Statistics
using Parameters
using TOML
using Dates

# Include and load core module
include("core/ModernConfigSystem.jl")
using .ModernConfigSystem

# Include and load GA optimization module
include("ga_optimization/FilterParameterGA.jl")
using .FilterParameterGA

# Re-export everything users need from submodules
# From ModernConfigSystem
export FilterBank, FilterParameters, FilterConfig, ExtendedFilterConfig,
       ProcessingConfig, PLLConfig, IOConfig,
       save_filter_config, load_filter_config, validate_config,
       create_default_configs, list_available_configs, show_config_summary,
       get_filter_by_period, set_filter_by_period!,
       get_active_filters, get_active_periods,
       create_default_filter_params

# From FilterParameterGA
export Chromosome, Population, GAConfig,
       ParameterType, ParameterSpec,
       config_to_chromosome, chromosome_to_config,
       initialize_population, evolve!, evolve_generation,
       save_best_config, get_best_config, get_best_chromosome,
       save_ga_state, load_ga_state,
       decode_chromosome, encode_chromosome,
       crossover, mutate!, tournament_select,
       update_fitness!, get_parameter_specs

# Module initialization
function __init__()
    # Create necessary directories if they don't exist
    for dir in ["config", "config/filters", "config/ga", "data", "data/results"]
        if !isdir(dir)
            mkpath(dir)
        end
    end
end

# Convenience function to show module info
function show_module_info()
    println("="^60)
    println("FibonacciPLLFilterBank Module Loaded")
    println("="^60)
    println("\nAvailable Submodules:")
    println("  • ModernConfigSystem - Configuration management")
    println("  • FilterParameterGA - Genetic algorithm optimization")
    println("\nKey Types:")
    println("  • ExtendedFilterConfig - Main configuration type")
    println("  • FilterBank - Collection of filter parameters")
    println("  • Chromosome, Population - GA types")
    println("\nKey Functions:")
    println("  • load_filter_config - Load configurations")
    println("  • config_to_chromosome - Convert for GA")
    println("  • initialize_population - Start GA")
    println("  • evolve! - Run GA optimization")
end

end # module FibonacciPLLFilterBank