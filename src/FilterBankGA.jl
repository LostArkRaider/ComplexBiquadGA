# src/FilterBankGA.jl - Container Managing Multiple SingleFilterGA Instances
# Coordinates evolution across all filters while maintaining independence

module FilterBankGA

using Statistics
using Printf
using Dates

export FilterBankGAComplete, evolve_instrument!, evolve_generation!,
       sync_with_storage!, load_from_storage!, get_best_parameters,
       get_convergence_status, print_status, save_checkpoint

# Include dependencies
if !isdefined(Main, :SingleFilterGA)
    include("SingleFilterGA.jl")
end
if !isdefined(Main, :StorageSystem)
    include("StorageSystem.jl")
end

using Main.SingleFilterGA
using Main.StorageSystem

# =============================================================================
# FILTER BANK GA STRUCTURE
# =============================================================================

"""
Complete GA system for one instrument's entire filter bank
"""
mutable struct FilterBankGAComplete
    instrument::String                        # "YM", "ES", etc.
    num_filters::Int32                       # Total filters
    population_size::Int32                   # Same for all filters
    
    # Independent GA for each filter
    filter_gas::Vector{SingleFilterGAComplete}  # Length = num_filters
    
    # Shared configuration
    ga_params                                # GAParameters
    param_ranges                             # ParameterRanges
    
    # Write-through storage
    storage                                  # WriteThruStorage
    
    # Performance tracking
    generation::Int32
    total_evaluations::Int64
    best_fitness_history::Vector{Float32}   # Max fitness across all filters
    mean_fitness_history::Vector{Float32}   # Mean of best fitness per filter
    convergence_history::Vector{Float32}    # Percentage of converged filters
    
    # Timing
    start_time::DateTime
    last_sync_time::DateTime
    
    # Random seed for reproducibility
    master_seed::Union{Int, Nothing}
end

"""
Constructor for FilterBankGAComplete
"""
function FilterBankGAComplete(config;  # InstrumentConfig
                            storage = nothing,
                            master_seed::Union{Int, Nothing} = nothing,
                            init_strategy::Symbol = :random)
    
    # Extract configuration
    instrument = config.symbol
    num_filters = config.num_filters
    population_size = config.population_size
    fibonacci_periods = config.fibonacci_periods
    ga_params = config.ga_params
    
    # Create parameter ranges
    param_ranges = Main.GATypes.ParameterRanges()
    
    # Initialize or use provided storage
    if storage === nothing
        storage = Main.GATypes.WriteThruStorage(
            num_filters,
            config.parameter_path,
            Int32(10)  # Convert to Int32 for sync interval
        )
    end
    
    # Create individual filter GAs
    filter_gas = SingleFilterGAComplete[]
    
    for i in 1:num_filters
        period = i <= length(fibonacci_periods) ? fibonacci_periods[i] : Int32(i)
        
        # Deterministic seed for each filter if master seed provided
        filter_seed = master_seed === nothing ? nothing : master_seed + i
        
        # Check if we have existing parameters in storage
        init_chromosome = nothing
        if any(storage.active_params[i, :] .!= 0)
            init_chromosome = storage.active_params[i, :]
            init_strat = :seeded
        else
            init_strat = init_strategy
        end
        
        # Create filter GA
        filter_ga = SingleFilterGAComplete(
            period,
            Int32(i),
            population_size,
            param_ranges,
            ga_params,
            seed = filter_seed,
            init_strategy = init_strat,
            init_chromosome = init_chromosome
        )
        
        push!(filter_gas, filter_ga)
    end
    
    # Initialize tracking
    best_fitness_history = Float32[]
    mean_fitness_history = Float32[]
    convergence_history = Float32[]
    
    return FilterBankGAComplete(
        instrument,
        num_filters,
        population_size,
        filter_gas,
        ga_params,
        param_ranges,
        storage,
        0,
        0,
        best_fitness_history,
        mean_fitness_history,
        convergence_history,
        now(),
        now(),
        master_seed
    )
end

# =============================================================================
# EVOLUTION FUNCTIONS
# =============================================================================

"""
Evolve all filters for one generation
"""
function evolve_generation!(fb_ga;
                          fitness_function::Union{Function, Nothing} = nothing,
                          verbose::Bool = false,
                          parallel::Bool = false)
    
    fb_ga.generation += 1
    
    if verbose
        println("\n" * "="^60)
        println("Generation $(fb_ga.generation) - $(fb_ga.instrument)")
        println("="^60)
    end
    
    # Track convergence
    n_converged = 0
    best_fitnesses = Float32[]
    
    # Track evaluations THIS generation only (FIX)
    generation_evaluations = 0
    
    # Evolve each filter independently
    if parallel && Threads.nthreads() > 1
        # Parallel evolution
        Threads.@threads for filter_ga in fb_ga.filter_gas
            if !filter_ga.converged
                # Store evaluations before evolution
                evals_before = filter_ga.total_evaluations
                evolve!(filter_ga, fitness_function=fitness_function, verbose=false)
                # Track only new evaluations this generation
                generation_evaluations += (filter_ga.total_evaluations - evals_before)
            end
        end
    else
        # Sequential evolution
        for filter_ga in fb_ga.filter_gas
            if !filter_ga.converged
                # Store evaluations before evolution
                evals_before = filter_ga.total_evaluations
                evolve!(filter_ga, fitness_function=fitness_function, verbose=false)
                # Track only new evaluations this generation
                generation_evaluations += (filter_ga.total_evaluations - evals_before)
            end
        end
    end
    
    # Collect statistics
    for filter_ga in fb_ga.filter_gas
        push!(best_fitnesses, filter_ga.best_fitness)
        if filter_ga.converged
            n_converged += 1
        end
    end
    
    # Add only this generation's evaluations to total (FIX)
    fb_ga.total_evaluations += generation_evaluations
    
    # Update history
    push!(fb_ga.best_fitness_history, maximum(best_fitnesses))
    push!(fb_ga.mean_fitness_history, mean(best_fitnesses))
    push!(fb_ga.convergence_history, Float32(n_converged / fb_ga.num_filters))
    
    # Sync to storage if needed
    if fb_ga.generation % fb_ga.storage.sync_interval == 0
        sync_with_storage!(fb_ga)
    end
    
    if verbose
        @printf("Best fitness: %.4f | Mean: %.4f | Converged: %d/%d\n",
                maximum(best_fitnesses), mean(best_fitnesses), 
                n_converged, fb_ga.num_filters)
    end
end

"""
Evolve instrument for multiple generations
"""
function evolve_instrument!(fb_ga;
                          generations::Int = 100,
                          fitness_function::Union{Function, Nothing} = nothing,
                          verbose::Bool = true,
                          parallel::Bool = false,
                          checkpoint_interval::Int = 50)
    
    if verbose
        println("\n" * "╔" * "═"^60 * "╗")
        println("║ Starting Evolution: $(fb_ga.instrument)" * " "^(60-22-length(fb_ga.instrument)) * "║")
        println("║ Filters: $(fb_ga.num_filters) | Population: $(fb_ga.population_size)" * " "^(60-30-length(string(fb_ga.num_filters))-length(string(fb_ga.population_size))) * "║")
        println("╚" * "═"^60 * "╝")
    end
    
    start_gen = fb_ga.generation
    
    for gen in 1:generations
        # Check if all filters converged
        all_converged = all(f.converged for f in fb_ga.filter_gas)
        if all_converged
            if verbose
                println("\n✅ All filters converged at generation $(fb_ga.generation)")
            end
            break
        end
        
        # Evolve one generation
        evolve_generation!(fb_ga, fitness_function=fitness_function, 
                         verbose=verbose, parallel=parallel)
        
        # Checkpoint if needed
        if fb_ga.generation % checkpoint_interval == 0
            save_checkpoint(fb_ga)
            if verbose
                println("💾 Checkpoint saved at generation $(fb_ga.generation)")
            end
        end
        
        # Progress update
        if verbose && gen % 10 == 0
            elapsed = now() - fb_ga.start_time
            print_progress(fb_ga, elapsed)
        end
    end
    
    # Final sync
    sync_with_storage!(fb_ga)
    
    if verbose
        println("\n" * "╔" * "═"^60 * "╗")
        println("║ Evolution Complete: $(fb_ga.instrument)" * " "^(60-24-length(fb_ga.instrument)) * "║")
        println("║ Total generations: $(fb_ga.generation)" * " "^(60-23-length(string(fb_ga.generation))) * "║")
        println("║ Total evaluations: $(fb_ga.total_evaluations)" * " "^(60-23-length(string(fb_ga.total_evaluations))) * "║")
        println("╚" * "═"^60 * "╝")
    end
end

# =============================================================================
# STORAGE INTEGRATION
# =============================================================================

"""
Sync best parameters to storage
"""
function sync_with_storage!(fb_ga)
    for i in 1:fb_ga.num_filters
        filter_ga = fb_ga.filter_gas[i]
        
        # Update storage with best chromosome
        Main.StorageSystem.set_active_parameters!(fb_ga.storage, Int32(i), filter_ga.best_chromosome)
    end
    
    # Perform actual sync to disk
    Main.StorageSystem.sync_to_storage!(fb_ga.storage)
    fb_ga.last_sync_time = now()
    
    println("💾 Synced $(fb_ga.num_filters) filters to storage")
end

"""
Load parameters from storage
"""
function load_from_storage!(fb_ga)
    # Load from disk
    if Main.StorageSystem.load_from_storage!(fb_ga.storage)
        # Update each filter GA with loaded parameters
        for i in 1:fb_ga.num_filters
            params = Main.StorageSystem.get_active_parameters(fb_ga.storage, Int32(i))
            
            if any(params .!= 0)
                filter_ga = fb_ga.filter_gas[i]
                
                # Set as best chromosome
                filter_ga.best_chromosome .= params
                
                # Reinitialize population around loaded parameters
                filter_ga.population = Main.PopulationInit.initialize_from_seed(
                    params, Int32(size(filter_ga.population, 1)), 
                    filter_ga.param_ranges, diversity=0.1f0, rng=filter_ga.rng
                )
                
                println("  Loaded parameters for filter $(filter_ga.period)")
            end
        end
        
        println("✅ Loaded parameters from storage")
        return true
    else
        println("⚠️  No existing parameters found in storage")
        return false
    end
end

"""
Save checkpoint
"""
function save_checkpoint(fb_ga)
    # Calculate overall fitness
    best_fitnesses = [f.best_fitness for f in fb_ga.filter_gas]
    overall_fitness = mean(best_fitnesses)
    
    # Create checkpoint
    checkpoint_file = Main.StorageSystem.create_checkpoint(fb_ga.storage, fb_ga.generation, overall_fitness)
    
    return checkpoint_file
end

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

"""
Get best parameters for all filters
"""
function get_best_parameters(fb_ga)::Matrix{Float32}
    params = Matrix{Float32}(undef, fb_ga.num_filters, 13)
    
    for i in 1:fb_ga.num_filters
        params[i, :] = fb_ga.filter_gas[i].best_chromosome
    end
    
    return params
end

"""
Get convergence status
"""
function get_convergence_status(fb_ga)::Dict{String, Any}
    status = Dict{String, Any}()
    
    status["instrument"] = fb_ga.instrument
    status["generation"] = fb_ga.generation
    status["total_evaluations"] = fb_ga.total_evaluations
    
    # Per-filter status
    filter_status = Dict{Int32, Dict{String, Any}}()
    
    for filter_ga in fb_ga.filter_gas
        filter_status[filter_ga.period] = Dict(
            "converged" => filter_ga.converged,
            "best_fitness" => filter_ga.best_fitness,
            "generation" => filter_ga.generation,
            "generations_since_improvement" => filter_ga.generations_since_improvement
        )
    end
    
    status["filters"] = filter_status
    
    # Overall statistics
    n_converged = sum(f.converged for f in fb_ga.filter_gas)
    status["converged_filters"] = n_converged
    status["total_filters"] = fb_ga.num_filters
    status["convergence_rate"] = Float32(n_converged / fb_ga.num_filters)
    
    if !isempty(fb_ga.best_fitness_history)
        status["current_best"] = fb_ga.best_fitness_history[end]
        status["current_mean"] = fb_ga.mean_fitness_history[end]
    end
    
    return status
end

"""
Print status summary
"""
function print_status(fb_ga)
    println("\n📊 Filter Bank Status: $(fb_ga.instrument)")
    println("="^50)
    println("  Generation: $(fb_ga.generation)")
    println("  Total evaluations: $(fb_ga.total_evaluations)")
    
    # Convergence summary
    n_converged = sum(f.converged for f in fb_ga.filter_gas)
    println("  Converged: $n_converged/$(fb_ga.num_filters)")
    
    # Fitness summary
    best_fitnesses = [f.best_fitness for f in fb_ga.filter_gas]
    println("  Best fitness: $(maximum(best_fitnesses))")
    println("  Mean fitness: $(mean(best_fitnesses))")
    
    # Per-filter details
    println("\n  Filter Details:")
    for filter_ga in fb_ga.filter_gas
        status = filter_ga.converged ? "✓" : "○"
        @printf("    %s Filter %3d: Fitness=%.4f, Gen=%3d\n",
                status, filter_ga.period, filter_ga.best_fitness, 
                filter_ga.generation)
    end
    
    println("="^50)
end

"""
Print progress during evolution
"""
function print_progress(fb_ga, elapsed::Millisecond)
    n_converged = sum(f.converged for f in fb_ga.filter_gas)
    best_fitnesses = [f.best_fitness for f in fb_ga.filter_gas]
    
    println("\n📈 Progress Update - Generation $(fb_ga.generation)")
    println("  Elapsed time: $(elapsed)")
    println("  Converged: $n_converged/$(fb_ga.num_filters) ($(round(100*n_converged/fb_ga.num_filters, digits=1))%)")
    println("  Best: $(round(maximum(best_fitnesses), digits=4)) | Mean: $(round(mean(best_fitnesses), digits=4))")
    
    # Show top 5 filters
    top_indices = sortperm(best_fitnesses, rev=true)[1:min(5, fb_ga.num_filters)]
    println("  Top filters:")
    for idx in top_indices
        filter_ga = fb_ga.filter_gas[idx]
        @printf("    Filter %3d: %.4f\n", filter_ga.period, filter_ga.best_fitness)
    end
end

# =============================================================================
# ADVANCED FEATURES
# =============================================================================

"""
Apply different evolution strategies to different filters
"""
function apply_adaptive_strategies!(fb_ga)
    for filter_ga in fb_ga.filter_gas
        if filter_ga.converged
            continue
        end
        
        # Check stagnation
        if filter_ga.generations_since_improvement > 20
            # Inject diversity
            Main.SingleFilterGA.inject_diversity!(filter_ga, injection_rate=0.2f0)
        elseif filter_ga.generations_since_improvement > 10
            # Increase mutation
            Main.SingleFilterGA.adapt_parameters!(filter_ga)
        end
    end
end

"""
Cross-filter learning (optional - breaks independence but can help)
"""
function cross_filter_seeding!(fb_ga;
                              migration_rate::Float32 = 0.05f0)
    
    # Find best performing filter
    best_fitnesses = [f.best_fitness for f in fb_ga.filter_gas]
    best_filter_idx = argmax(best_fitnesses)
    best_chromosome = fb_ga.filter_gas[best_filter_idx].best_chromosome
    
    # Seed struggling filters with best chromosome
    for (i, filter_ga) in enumerate(fb_ga.filter_gas)
        if i != best_filter_idx && !filter_ga.converged
            if filter_ga.generations_since_improvement > 15
                # Replace some individuals with perturbed best
                pop_size = size(filter_ga.population, 1)
                n_migrate = round(Int, pop_size * migration_rate)
                
                for j in 1:n_migrate
                    filter_ga.population[j, :] = Main.PopulationInit.add_noise_to_chromosome(
                        best_chromosome, filter_ga.param_ranges, 0.2f0, filter_ga.rng
                    )
                end
            end
        end
    end
end

end # module FilterBankGA