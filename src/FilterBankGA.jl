module FilterBankGA

using Statistics
using Printf
using Dates
using ..GATypes
using ..SingleFilterGA
using ..StorageSystem
using ..PopulationInit

export FilterBankGAComplete, evolve_instrument!, evolve_generation!,
       sync_with_storage!, load_from_storage!, get_best_parameters,
       get_convergence_status, BankConvergenceStatus, FilterConvergenceStatus,
       print_status

# --- Moved struct definition here from GATypes ---
mutable struct FilterBankGAComplete{M<:AbstractMatrix{Float32}, V<:AbstractVector{Float32}}
    instrument::String
    num_filters::Int32
    population_size::Int32
    filter_gas::Vector{SingleFilterGAComplete{M, V}}
    ga_params::GAParameters
    param_ranges::ParameterRanges
    storage::WriteThruStorage
    generation::Int32
    total_evaluations::Int64
    best_fitness_history::Vector{Float32}
    mean_fitness_history::Vector{Float32}
    convergence_history::Vector{Float32}
    start_time::DateTime
    last_sync_time::DateTime
    master_seed::Union{Int, Nothing}
end

# --- Structs for status reporting ---
struct FilterConvergenceStatus
    period::Int32
    converged::Bool
    best_fitness::Float32
    generation::Int32
    generations_since_improvement::Int32
end

struct BankConvergenceStatus
    instrument::String
    generation::Int32
    total_evaluations::Int64
    filter_statuses::Vector{FilterConvergenceStatus}
    converged_filters::Int
    total_filters::Int32
    convergence_rate::Float32
    current_best_fitness::Float32
    current_mean_fitness::Float32
end

# =============================================================================
# FILTER BANK GA CONSTRUCTOR
# =============================================================================

function FilterBankGAComplete(
    config::InstrumentConfig;
    ArrayType::Type=Array{Float32},
    storage::Union{WriteThruStorage, Nothing} = nothing,
    master_seed::Union{Int, Nothing} = nothing,
    init_strategy::Symbol = :random
)
    instrument = config.symbol
    num_filters = config.num_filters
    population_size = config.population_size
    fibonacci_periods = config.fibonacci_periods
    ga_params = config.ga_params
    param_ranges = ParameterRanges()

    if storage === nothing
        cpu_dirty_filters = zeros(Bool, num_filters)
        storage = WriteThruStorage(
            ArrayType(undef, num_filters, 13),
            config.parameter_path,
            now(), Int32(10),
            ArrayType(cpu_dirty_filters),
            Int32(0), FilterDefaults()
        )
    end
    
    TempPopType = similar(ArrayType(undef, 0, 0), 0, 13)
    TempFitType = similar(ArrayType(undef, 0, 0), 0)
    M = typeof(TempPopType)
    V = typeof(TempFitType)

    filter_gas = Vector{SingleFilterGAComplete{M, V}}(undef, num_filters)
    
    for i in 1:num_filters
        period = i <= length(fibonacci_periods) ? fibonacci_periods[i] : Int32(i)
        filter_seed = master_seed === nothing ? nothing : master_seed + i
        
        params_view = @view storage.active_params[i, :]
        init_chromosome = any(params_view .!= 0) ? params_view : nothing
        
        filter_gas[i] = SingleFilterGAComplete(
            period, Int32(i), population_size, param_ranges, ga_params,
            ArrayType=ArrayType,
            seed=filter_seed,
            init_strategy = init_chromosome === nothing ? init_strategy : :seeded,
            init_chromosome = init_chromosome
        )
    end
    
    return FilterBankGAComplete{M, V}(
        instrument, num_filters, population_size,
        filter_gas, ga_params, param_ranges, storage,
        0, 0, Float32[], Float32[], Float32[],
        now(), now(), master_seed
    )
end

# =============================================================================
# EVOLUTION FUNCTIONS
# =============================================================================

function evolve_generation!(fb_ga::FilterBankGAComplete; fitness_function::Function, verbose::Bool=false, parallel::Bool=false)
    fb_ga.generation += 1
    generation_evaluations = 0
    
    if parallel && Threads.nthreads() > 1
        Threads.@threads for filter_ga in fb_ga.filter_gas
            if !filter_ga.converged
                evals_before = filter_ga.total_evaluations
                evolve!(filter_ga, fitness_function=fitness_function, verbose=false)
                generation_evaluations += (filter_ga.total_evaluations - evals_before)
            end
        end
    else
        for filter_ga in fb_ga.filter_gas
            if !filter_ga.converged
                evals_before = filter_ga.total_evaluations
                evolve!(filter_ga, fitness_function=fitness_function, verbose=false)
                generation_evaluations += (filter_ga.total_evaluations - evals_before)
            end
        end
    end
    
    fb_ga.total_evaluations += generation_evaluations
    
    best_fitnesses = [f.best_fitness for f in fb_ga.filter_gas]
    n_converged = sum(f.converged for f in fb_ga.filter_gas)
    
    push!(fb_ga.best_fitness_history, maximum(best_fitnesses))
    push!(fb_ga.mean_fitness_history, mean(best_fitnesses))
    push!(fb_ga.convergence_history, Float32(n_converged / fb_ga.num_filters))
    
    if fb_ga.generation % fb_ga.storage.sync_interval == 0
        sync_with_storage!(fb_ga)
    end
    
    if verbose
        @printf("Best fitness: %.4f | Mean: %.4f | Converged: %d/%d\n",
                maximum(best_fitnesses), mean(best_fitnesses), 
                n_converged, fb_ga.num_filters)
    end
end

function evolve_instrument!(fb_ga::FilterBankGAComplete; generations::Int = 100,
                          fitness_function::Function,
                          verbose::Bool = true,
                          parallel::Bool = false,
                          checkpoint_interval::Int = 50)
    
    if verbose
        println("\n" * "╔" * "═"^60 * "╗")
        println("║ Starting Evolution: $(fb_ga.instrument)" * " "^(38-length(fb_ga.instrument)) * "║")
        println("║ Filters: $(fb_ga.num_filters) | Population: $(fb_ga.population_size)" * " "^(30-length(string(fb_ga.num_filters))-length(string(fb_ga.population_size))) * "║")
        println("╚" * "═"^60 * "╝")
    end
    
    for gen in 1:generations
        if all(f.converged for f in fb_ga.filter_gas)
            if verbose
                println("\n✅ All filters converged at generation $(fb_ga.generation)")
            end
            break
        end
        
        evolve_generation!(fb_ga, fitness_function=fitness_function, 
                         verbose=verbose, parallel=parallel)
        
        if fb_ga.generation % checkpoint_interval == 0 && verbose
             println("💾 Checkpoint saved at generation $(fb_ga.generation)")
        end
    end
    
    sync_with_storage!(fb_ga)
    
    if verbose
        println("\n" * "╔" * "═"^60 * "╗")
        println("║ Evolution Complete: $(fb_ga.instrument)" * " "^(36-length(fb_ga.instrument)) * "║")
        println("║ Total generations: $(fb_ga.generation)" * " "^(37-length(string(fb_ga.generation))) * "║")
        println("║ Total evaluations: $(fb_ga.total_evaluations)" * " "^(37-length(string(fb_ga.total_evaluations))) * "║")
        println("╚" * "═"^60 * "╝")
    end
end

# =============================================================================
# STORAGE INTEGRATION
# =============================================================================

function sync_with_storage!(fb_ga::FilterBankGAComplete)
    for i in 1:fb_ga.num_filters
        filter_ga = fb_ga.filter_gas[i]
        set_active_parameters!(fb_ga.storage, Int32(i), Array(filter_ga.best_chromosome))
    end
    sync_to_storage!(fb_ga.storage)
    fb_ga.last_sync_time = now()
end

function load_from_storage!(fb_ga::FilterBankGAComplete)
    if load_from_storage!(fb_ga.storage)
        for i in 1:fb_ga.num_filters
            params = get_active_parameters(fb_ga.storage, Int32(i))
            if any(params .!= 0)
                filter_ga = fb_ga.filter_gas[i]
                copyto!(filter_ga.best_chromosome, params)
                filter_ga.population = initialize_from_seed(
                    filter_ga.best_chromosome, Int32(size(filter_ga.population, 1)), 
                    filter_ga.param_ranges, diversity=0.1f0, rng=filter_ga.rng
                )
            end
        end
        return true
    end
    return false
end

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

function get_best_parameters(fb_ga::FilterBankGAComplete)
    params = similar(fb_ga.filter_gas[1].population, fb_ga.num_filters, 13)
    for i in 1:fb_ga.num_filters
        params[i, :] = fb_ga.filter_gas[i].best_chromosome
    end
    return params
end

function get_convergence_status(fb_ga::FilterBankGAComplete)::BankConvergenceStatus
    filter_statuses = [
        FilterConvergenceStatus(
            f.period, f.converged, f.best_fitness,
            f.generation, f.generations_since_improvement
        ) for f in fb_ga.filter_gas
    ]
    
    n_converged = sum(f.converged for f in fb_ga.filter_gas)
    best_fitnesses = [f.best_fitness for f in fb_ga.filter_gas]
    
    return BankConvergenceStatus(
        fb_ga.instrument,
        fb_ga.generation,
        fb_ga.total_evaluations,
        filter_statuses,
        n_converged,
        fb_ga.num_filters,
        Float32(n_converged / fb_ga.num_filters),
        isempty(best_fitnesses) ? 0.0f0 : maximum(best_fitnesses),
        isempty(best_fitnesses) ? 0.0f0 : mean(best_fitnesses)
    )
end

function print_status(fb_ga::FilterBankGAComplete)
    println("\n📊 Filter Bank Status: $(fb_ga.instrument)")
    println("="^50)
    println("  Generation: $(fb_ga.generation)")
    println("  Total evaluations: $(fb_ga.total_evaluations)")
    
    n_converged = sum(f.converged for f in fb_ga.filter_gas)
    println("  Converged: $n_converged/$(fb_ga.num_filters)")
    
    best_fitnesses = [f.best_fitness for f in fb_ga.filter_gas]
    println("  Best fitness: $(maximum(best_fitnesses))")
    println("  Mean fitness: $(mean(best_fitnesses))")
    
    println("\n  Filter Details:")
    for filter_ga in fb_ga.filter_gas
        status = filter_ga.converged ? "✓" : "○"
        @printf("    %s Filter %3d: Fitness=%.4f, Gen=%3d\n",
                status, filter_ga.period, filter_ga.best_fitness, 
                filter_ga.generation)
    end
    println("="^50)
end

end # module FilterBankGA