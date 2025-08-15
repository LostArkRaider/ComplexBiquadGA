# FilterParameterGA.jl - Genetic Algorithm Core for Filter Parameter Optimization
# DIRECTLY INTEGRATES with ModernConfigSystem - NO SEPARATE CONFIGURATION SYSTEM

module FilterParameterGA

using Random
using Statistics
using Parameters
using TOML
using Dates

# CRITICAL: Import ModernConfigSystem from parent module
# This module expects to be included from FibonacciPLLFilterBank which has ModernConfigSystem
using ..ModernConfigSystem

export ParameterType, ParameterSpec,
       Chromosome, Population, GAConfig,
       initialize_population, evolve_generation, evolve!,
       decode_chromosome, encode_chromosome, 
       chromosome_to_config, config_to_chromosome,
       crossover, mutate, tournament_select,
       save_best_config, load_ga_state, save_ga_state,
       get_best_chromosome, update_fitness!,
       get_parameter_specs

# =============================================================================
# PARAMETER TYPE SYSTEM
# =============================================================================

@enum ParameterType begin
    LINEAR
    LOGARITHMIC
    BINARY
    DISCRETE
end

struct ParameterSpec
    name::Symbol
    type::ParameterType
    min_val::Float64        # For continuous types
    max_val::Float64        # For continuous types
    options::Vector{Int}    # For discrete types
    description::String
end

# Define parameter specifications for all 12 parameters
const PARAMETER_SPECS = [
    ParameterSpec(:q_factor, LINEAR, 0.5, 10.0, Int[], "Filter Q factor/bandwidth"),
    ParameterSpec(:sma_window, LOGARITHMIC, 1.0, 200.0, Int[], "Simple moving average window"),
    ParameterSpec(:batch_size, LOGARITHMIC, 100.0, 5000.0, Int[], "Processing batch size"),
    ParameterSpec(:phase_detector_gain, LOGARITHMIC, 0.001, 1.0, Int[], "PLL phase detector sensitivity"),
    ParameterSpec(:loop_bandwidth, LOGARITHMIC, 0.0001, 0.1, Int[], "PLL loop filter bandwidth"),
    ParameterSpec(:lock_threshold, LINEAR, 0.0, 1.0, Int[], "PLL lock quality threshold"),
    ParameterSpec(:ring_decay, LINEAR, 0.9, 1.0, Int[], "Ringing decay factor"),
    ParameterSpec(:enable_clamping, BINARY, 0.0, 1.0, Int[], "Enable signal clamping"),
    ParameterSpec(:clamping_threshold, LOGARITHMIC, 1e-8, 1e-3, Int[], "Clamping activation threshold"),
    ParameterSpec(:volume_scaling, LOGARITHMIC, 0.1, 10.0, Int[], "Volume component scaling"),
    ParameterSpec(:max_frequency_deviation, LINEAR, 0.01, 0.5, Int[], "Maximum frequency deviation"),
    ParameterSpec(:phase_error_history_length, DISCRETE, 5.0, 50.0, [5, 10, 15, 20, 30, 40, 50], "Phase error buffer size")
]

# =============================================================================
# ENCODING/DECODING FUNCTIONS
# =============================================================================

# Linear scaling: direct mapping
function linear_encode(value::Float64, min_val::Float64, max_val::Float64)::Float64
    return clamp((value - min_val) / (max_val - min_val), 0.0, 1.0)
end

function linear_decode(gene::Float64, min_val::Float64, max_val::Float64)::Float64
    return min_val + clamp(gene, 0.0, 1.0) * (max_val - min_val)
end

# Logarithmic scaling: exponential mapping
function log_encode(value::Float64, min_val::Float64, max_val::Float64)::Float64
    @assert min_val > 0 && max_val > 0 "Logarithmic scaling requires positive values"
    return clamp(log(value / min_val) / log(max_val / min_val), 0.0, 1.0)
end

function log_decode(gene::Float64, min_val::Float64, max_val::Float64)::Float64
    @assert min_val > 0 && max_val > 0 "Logarithmic scaling requires positive values"
    return min_val * (max_val / min_val) ^ clamp(gene, 0.0, 1.0)
end

# Binary encoding
function binary_encode(value::Bool)::Float64
    return value ? 1.0 : 0.0
end

function binary_decode(gene::Float64)::Bool
    return gene >= 0.5
end

# Discrete enumeration
function discrete_encode(value::Int, options::Vector{Int})::Float64
    idx = findfirst(==(value), options)
    if isnothing(idx)
        # Default to middle option if value not found
        return 0.5
    end
    return (idx - 1) / max(1, length(options) - 1)
end

function discrete_decode(gene::Float64, options::Vector{Int})::Int
    if isempty(options)
        return 20  # Default value
    end
    idx = 1 + floor(Int, clamp(gene, 0.0, 0.9999) * length(options))
    return options[clamp(idx, 1, length(options))]
end

# =============================================================================
# CHROMOSOME STRUCTURE
# =============================================================================

mutable struct Chromosome
    genes::Vector{Float64}           # Normalized [0,1] values
    num_filters::Int                 # Number of active filters
    active_periods::Vector{Int}      # Which filter periods are active
    fitness::Float64                 # Fitness score
    evaluated::Bool                  # Has fitness been calculated?
end

# Create chromosome from ExtendedFilterConfig (DIRECT INTEGRATION)
function config_to_chromosome(config::ExtendedFilterConfig)::Chromosome
    bank = config.filter_bank
    active_filters = get_active_filters(bank)
    num_filters = length(active_filters)
    genes = Vector{Float64}(undef, num_filters * 12)
    active_periods = [fp.period for fp in active_filters]
    
    for (filter_idx, fp) in enumerate(active_filters)
        encode_filter_to_genes!(genes, filter_idx, fp)
    end
    
    return Chromosome(genes, num_filters, active_periods, config.ga_fitness, config.ga_fitness > 0)
end

# Create chromosome from FilterBank (USES ModernConfigSystem.FilterBank directly)
function Chromosome(bank::FilterBank)
    active_filters = get_active_filters(bank)
    num_filters = length(active_filters)
    genes = Vector{Float64}(undef, num_filters * 12)
    active_periods = [fp.period for fp in active_filters]
    
    for (filter_idx, fp) in enumerate(active_filters)
        encode_filter_to_genes!(genes, filter_idx, fp)
    end
    
    return Chromosome(genes, num_filters, active_periods, 0.0, false)
end

# Create chromosome from filter parameters array
function Chromosome(filter_params::Vector{FilterParameters})
    num_filters = length(filter_params)
    genes = Vector{Float64}(undef, num_filters * 12)
    active_periods = [fp.period for fp in filter_params]
    
    for (filter_idx, fp) in enumerate(filter_params)
        encode_filter_to_genes!(genes, filter_idx, fp)
    end
    
    return Chromosome(genes, num_filters, active_periods, 0.0, false)
end

# Encode a single filter's parameters into genes
function encode_filter_to_genes!(genes::Vector{Float64}, filter_idx::Int, fp::FilterParameters)
    base_idx = (filter_idx - 1) * 12
    
    genes[base_idx + 1] = linear_encode(fp.q_factor, PARAMETER_SPECS[1].min_val, PARAMETER_SPECS[1].max_val)
    genes[base_idx + 2] = log_encode(Float64(fp.sma_window), PARAMETER_SPECS[2].min_val, PARAMETER_SPECS[2].max_val)
    genes[base_idx + 3] = log_encode(Float64(fp.batch_size), PARAMETER_SPECS[3].min_val, PARAMETER_SPECS[3].max_val)
    genes[base_idx + 4] = log_encode(fp.phase_detector_gain, PARAMETER_SPECS[4].min_val, PARAMETER_SPECS[4].max_val)
    genes[base_idx + 5] = log_encode(fp.loop_bandwidth, PARAMETER_SPECS[5].min_val, PARAMETER_SPECS[5].max_val)
    genes[base_idx + 6] = linear_encode(fp.lock_threshold, PARAMETER_SPECS[6].min_val, PARAMETER_SPECS[6].max_val)
    genes[base_idx + 7] = linear_encode(fp.ring_decay, PARAMETER_SPECS[7].min_val, PARAMETER_SPECS[7].max_val)
    genes[base_idx + 8] = binary_encode(fp.enable_clamping)
    genes[base_idx + 9] = log_encode(fp.clamping_threshold, PARAMETER_SPECS[9].min_val, PARAMETER_SPECS[9].max_val)
    genes[base_idx + 10] = log_encode(fp.volume_scaling, PARAMETER_SPECS[10].min_val, PARAMETER_SPECS[10].max_val)
    genes[base_idx + 11] = linear_encode(fp.max_frequency_deviation, PARAMETER_SPECS[11].min_val, PARAMETER_SPECS[11].max_val)
    genes[base_idx + 12] = discrete_encode(fp.phase_error_history_length, PARAMETER_SPECS[12].options)
end

# Decode chromosome to filter parameters
function decode_chromosome(chr::Chromosome)::Vector{FilterParameters}
    filter_params = Vector{FilterParameters}(undef, chr.num_filters)
    
    for filter_idx in 1:chr.num_filters
        filter_params[filter_idx] = decode_filter_from_genes(chr.genes, filter_idx, chr.active_periods[filter_idx])
    end
    
    return filter_params
end

# Decode a single filter from genes
function decode_filter_from_genes(genes::Vector{Float64}, filter_idx::Int, period::Int)::FilterParameters
    base_idx = (filter_idx - 1) * 12
    
    return FilterParameters(
        period = period,
        q_factor = linear_decode(genes[base_idx + 1], PARAMETER_SPECS[1].min_val, PARAMETER_SPECS[1].max_val),
        sma_window = round(Int, log_decode(genes[base_idx + 2], PARAMETER_SPECS[2].min_val, PARAMETER_SPECS[2].max_val)),
        batch_size = round(Int, log_decode(genes[base_idx + 3], PARAMETER_SPECS[3].min_val, PARAMETER_SPECS[3].max_val)),
        phase_detector_gain = log_decode(genes[base_idx + 4], PARAMETER_SPECS[4].min_val, PARAMETER_SPECS[4].max_val),
        loop_bandwidth = log_decode(genes[base_idx + 5], PARAMETER_SPECS[5].min_val, PARAMETER_SPECS[5].max_val),
        lock_threshold = linear_decode(genes[base_idx + 6], PARAMETER_SPECS[6].min_val, PARAMETER_SPECS[6].max_val),
        ring_decay = linear_decode(genes[base_idx + 7], PARAMETER_SPECS[7].min_val, PARAMETER_SPECS[7].max_val),
        enable_clamping = binary_decode(genes[base_idx + 8]),
        clamping_threshold = log_decode(genes[base_idx + 9], PARAMETER_SPECS[9].min_val, PARAMETER_SPECS[9].max_val),
        volume_scaling = log_decode(genes[base_idx + 10], PARAMETER_SPECS[10].min_val, PARAMETER_SPECS[10].max_val),
        max_frequency_deviation = linear_decode(genes[base_idx + 11], PARAMETER_SPECS[11].min_val, PARAMETER_SPECS[11].max_val),
        phase_error_history_length = discrete_decode(genes[base_idx + 12], PARAMETER_SPECS[12].options)
    )
end

# CRITICAL: Convert chromosome DIRECTLY to ExtendedFilterConfig
function chromosome_to_config(chr::Chromosome, 
                             name::String = "ga_optimized",
                             description::String = "GA-optimized filter configuration")::ExtendedFilterConfig
    
    # Create FilterBank from chromosome
    filter_params = decode_chromosome(chr)
    bank = FilterBank(Int[], 20)  # Start with empty bank
    
    for fp in filter_params
        set_filter_by_period!(bank, fp)
    end
    
    # Return ExtendedFilterConfig directly
    return ExtendedFilterConfig(
        name = name,
        description = description,
        filter_bank = bank,
        processing = ProcessingConfig(),  # Use defaults
        pll = PLLConfig(enabled = true),  # Enable PLL for extended config
        io = IOConfig(),                  # Use defaults
        ga_fitness = chr.fitness
    )
end

# Helper to encode a chromosome from a FilterBank
function encode_chromosome(bank::FilterBank)::Chromosome
    return Chromosome(bank)
end

# =============================================================================
# GENETIC OPERATORS
# =============================================================================

# Crossover with hybrid strategy (Option C)
function crossover(parent1::Chromosome, parent2::Chromosome, crossover_rate::Float64 = 0.8)::Tuple{Chromosome, Chromosome}
    @assert parent1.num_filters == parent2.num_filters "Parents must have same number of filters"
    @assert parent1.active_periods == parent2.active_periods "Parents must have same active periods"
    
    if rand() > crossover_rate
        # No crossover, return copies
        child1 = Chromosome(copy(parent1.genes), parent1.num_filters, copy(parent1.active_periods), 0.0, false)
        child2 = Chromosome(copy(parent2.genes), parent2.num_filters, copy(parent2.active_periods), 0.0, false)
        return (child1, child2)
    end
    
    child1_genes = copy(parent1.genes)
    child2_genes = copy(parent2.genes)
    
    # Decide crossover strategy
    if rand() < 0.5
        # Strategy A: Swap entire filter configurations
        for filter_idx in 1:parent1.num_filters
            if rand() < 0.5
                base_idx = (filter_idx - 1) * 12
                range = (base_idx + 1):(base_idx + 12)
                child1_genes[range], child2_genes[range] = child2_genes[range], child1_genes[range]
            end
        end
    else
        # Strategy B: Mix within filter parameters (arithmetic crossover)
        for filter_idx in 1:parent1.num_filters
            base_idx = (filter_idx - 1) * 12
            for param_idx in 1:12
                gene_idx = base_idx + param_idx
                if rand() < 0.5
                    # Arithmetic crossover for this parameter
                    alpha = rand()
                    new_val1 = alpha * parent1.genes[gene_idx] + (1 - alpha) * parent2.genes[gene_idx]
                    new_val2 = (1 - alpha) * parent1.genes[gene_idx] + alpha * parent2.genes[gene_idx]
                    child1_genes[gene_idx] = clamp(new_val1, 0.0, 1.0)
                    child2_genes[gene_idx] = clamp(new_val2, 0.0, 1.0)
                end
            end
        end
    end
    
    child1 = Chromosome(child1_genes, parent1.num_filters, copy(parent1.active_periods), 0.0, false)
    child2 = Chromosome(child2_genes, parent2.num_filters, copy(parent2.active_periods), 0.0, false)
    
    return (child1, child2)
end

# Type-aware mutation
function mutate!(chr::Chromosome, mutation_rate::Float64 = 0.1, mutation_strength::Float64 = 0.1)
    for filter_idx in 1:chr.num_filters
        base_idx = (filter_idx - 1) * 12
        
        for param_idx in 1:12
            if rand() < mutation_rate
                gene_idx = base_idx + param_idx
                spec = PARAMETER_SPECS[param_idx]
                
                if spec.type == LINEAR || spec.type == LOGARITHMIC
                    # Gaussian mutation for continuous parameters
                    noise = randn() * mutation_strength
                    chr.genes[gene_idx] = clamp(chr.genes[gene_idx] + noise, 0.0, 1.0)
                    
                elseif spec.type == BINARY
                    # Bit flip for binary parameters
                    chr.genes[gene_idx] = chr.genes[gene_idx] >= 0.5 ? 0.0 : 1.0
                    
                elseif spec.type == DISCRETE
                    # Jump to adjacent or random option
                    if rand() < 0.7  # 70% chance of adjacent jump
                        # Move to adjacent option
                        current_val = chr.genes[gene_idx]
                        num_options = length(spec.options)
                        step = 1.0 / max(1, num_options - 1)
                        if rand() < 0.5
                            chr.genes[gene_idx] = clamp(current_val - step, 0.0, 1.0)
                        else
                            chr.genes[gene_idx] = clamp(current_val + step, 0.0, 1.0)
                        end
                    else
                        # Random jump
                        chr.genes[gene_idx] = rand()
                    end
                end
            end
        end
    end
    
    chr.evaluated = false  # Mark as needing re-evaluation
end

# Tournament selection
function tournament_select(population::Vector{Chromosome}, tournament_size::Int = 5)::Chromosome
    @assert !isempty(population) "Population cannot be empty"
    @assert all(c.evaluated for c in population) "All chromosomes must be evaluated"
    
    tournament_size = min(tournament_size, length(population))
    
    # Use randperm to select random indices instead of sample
    indices = randperm(length(population))[1:tournament_size]
    contestants = population[indices]
    
    # Select best (highest fitness)
    best_idx = argmax([c.fitness for c in contestants])
    return contestants[best_idx]
end

# =============================================================================
# POPULATION MANAGEMENT
# =============================================================================

mutable struct Population
    chromosomes::Vector{Chromosome}
    generation::Int
    best_fitness::Float64
    fitness_history::Vector{Float64}
end

@with_kw struct GAConfig
    population_size::Int = 100
    generations::Int = 200
    mutation_rate::Float64 = 0.1
    mutation_strength::Float64 = 0.1
    crossover_rate::Float64 = 0.8
    elitism_count::Int = 5
    tournament_size::Int = 5
end

# Initialize population with diverse parameters
function initialize_population(periods::Vector{Int}, config::GAConfig)::Population
    chromosomes = Vector{Chromosome}(undef, config.population_size)
    
    for i in 1:config.population_size
        # Create random filter parameters for each period
        filter_params = FilterParameters[]
        for period in periods
            # Generate random parameters with appropriate distributions
            q_factor = rand() * 9.5 + 0.5  # Linear [0.5, 10.0]
            sma_window = round(Int, exp(log(200/1) * rand()) * 1)  # Log [1, 200]
            batch_size = round(Int, exp(log(5000/100) * rand()) * 100)  # Log [100, 5000]
            phase_detector_gain = exp(log(1.0/0.001) * rand()) * 0.001  # Log [0.001, 1.0]
            loop_bandwidth = exp(log(0.1/0.0001) * rand()) * 0.0001  # Log [0.0001, 0.1]
            lock_threshold = rand()  # Linear [0.0, 1.0]
            ring_decay = rand() * 0.1 + 0.9  # Linear [0.9, 1.0]
            enable_clamping = rand() < 0.5
            clamping_threshold = exp(log(1e-3/1e-8) * rand()) * 1e-8  # Log [1e-8, 1e-3]
            volume_scaling = exp(log(10.0/0.1) * rand()) * 0.1  # Log [0.1, 10.0]
            max_frequency_deviation = rand() * 0.49 + 0.01  # Linear [0.01, 0.5]
            phase_error_history_length = rand([5, 10, 15, 20, 30, 40, 50])
            
            push!(filter_params, FilterParameters(
                period = period,
                q_factor = q_factor,
                sma_window = sma_window,
                batch_size = batch_size,
                phase_detector_gain = phase_detector_gain,
                loop_bandwidth = loop_bandwidth,
                lock_threshold = lock_threshold,
                ring_decay = ring_decay,
                enable_clamping = enable_clamping,
                clamping_threshold = clamping_threshold,
                volume_scaling = volume_scaling,
                max_frequency_deviation = max_frequency_deviation,
                phase_error_history_length = phase_error_history_length
            ))
        end
        
        chromosomes[i] = Chromosome(filter_params)
    end
    
    return Population(chromosomes, 0, -Inf, Float64[])
end

# Alternative: Initialize from existing ExtendedFilterConfig
function initialize_population(base_config::ExtendedFilterConfig, config::GAConfig)::Population
    chromosomes = Vector{Chromosome}(undef, config.population_size)
    
    # First chromosome is the base configuration
    chromosomes[1] = config_to_chromosome(base_config)
    
    # Create variations for the rest
    periods = get_active_periods(base_config.filter_bank)
    for i in 2:config.population_size
        # Create random filter parameters for each period
        filter_params = FilterParameters[]
        for period in periods
            # Use base config as starting point with random variations
            base_fp = get_filter_by_period(base_config.filter_bank, period)
            if !isnothing(base_fp)
                # Add random noise to base parameters
                q_factor = clamp(base_fp.q_factor * (0.5 + rand()), 0.5, 10.0)
                sma_window = clamp(round(Int, base_fp.sma_window * (0.5 + rand())), 1, 200)
                batch_size = clamp(round(Int, base_fp.batch_size * (0.5 + rand())), 100, 5000)
                phase_detector_gain = clamp(base_fp.phase_detector_gain * (0.1 + rand() * 1.9), 0.001, 1.0)
                loop_bandwidth = clamp(base_fp.loop_bandwidth * (0.1 + rand() * 1.9), 0.0001, 0.1)
                lock_threshold = clamp(base_fp.lock_threshold + randn() * 0.2, 0.0, 1.0)
                ring_decay = clamp(base_fp.ring_decay + randn() * 0.01, 0.9, 1.0)
                enable_clamping = rand() < 0.5
                clamping_threshold = clamp(base_fp.clamping_threshold * (0.1 + rand() * 1.9), 1e-8, 1e-3)
                volume_scaling = clamp(base_fp.volume_scaling * (0.5 + rand()), 0.1, 10.0)
                max_frequency_deviation = clamp(base_fp.max_frequency_deviation + randn() * 0.1, 0.01, 0.5)
                phase_error_history_length = rand([5, 10, 15, 20, 30, 40, 50])
                
                push!(filter_params, FilterParameters(
                    period = period,
                    q_factor = q_factor,
                    sma_window = sma_window,
                    batch_size = batch_size,
                    phase_detector_gain = phase_detector_gain,
                    loop_bandwidth = loop_bandwidth,
                    lock_threshold = lock_threshold,
                    ring_decay = ring_decay,
                    enable_clamping = enable_clamping,
                    clamping_threshold = clamping_threshold,
                    volume_scaling = volume_scaling,
                    max_frequency_deviation = max_frequency_deviation,
                    phase_error_history_length = phase_error_history_length
                ))
            else
                # Create default if not found
                push!(filter_params, create_default_filter_params(period))
            end
        end
        
        chromosomes[i] = Chromosome(filter_params)
    end
    
    return Population(chromosomes, 0, -Inf, Float64[])
end

# Placeholder fitness function (to be replaced in Chunk 2)
function evaluate_fitness!(chr::Chromosome)
    if chr.evaluated
        return chr.fitness
    end
    
    # Simple placeholder: random fitness with slight preference for certain parameters
    fitness = 0.0
    filter_params = decode_chromosome(chr)
    
    for fp in filter_params
        # Prefer moderate Q factors
        fitness += exp(-abs(fp.q_factor - 3.0) / 2.0)
        
        # Prefer moderate SMA windows
        fitness += exp(-abs(log(fp.sma_window) - log(30)) / 2.0)
        
        # Prefer PLL parameters in middle ranges
        fitness += exp(-abs(log(fp.phase_detector_gain) - log(0.05)) / 2.0)
        fitness += exp(-abs(log(fp.loop_bandwidth) - log(0.005)) / 2.0)
        
        # Small bonus for reasonable lock threshold
        fitness += fp.lock_threshold > 0.5 ? 0.5 : 0.0
        
        # Add some randomness
        fitness += rand() * 0.5
    end
    
    chr.fitness = fitness / length(filter_params)
    chr.evaluated = true
    
    return chr.fitness
end

# Update fitness for entire population
function update_fitness!(pop::Population)
    for chr in pop.chromosomes
        evaluate_fitness!(chr)
    end
end

# Evolve one generation
function evolve_generation(pop::Population, config::GAConfig)::Population
    # Ensure all chromosomes are evaluated
    update_fitness!(pop)
    
    # Sort by fitness (descending)
    sort!(pop.chromosomes, by=c->c.fitness, rev=true)
    
    # Track best fitness
    best_fitness = pop.chromosomes[1].fitness
    push!(pop.fitness_history, best_fitness)
    pop.best_fitness = max(pop.best_fitness, best_fitness)
    
    # Create new population
    new_chromosomes = Vector{Chromosome}(undef, config.population_size)
    
    # Elitism: keep best chromosomes
    for i in 1:config.elitism_count
        new_chromosomes[i] = Chromosome(
            copy(pop.chromosomes[i].genes),
            pop.chromosomes[i].num_filters,
            copy(pop.chromosomes[i].active_periods),
            pop.chromosomes[i].fitness,
            true
        )
    end
    
    # Generate rest through crossover and mutation
    idx = config.elitism_count + 1
    while idx <= config.population_size
        # Tournament selection for parents
        parent1 = tournament_select(pop.chromosomes, config.tournament_size)
        parent2 = tournament_select(pop.chromosomes, config.tournament_size)
        
        # Crossover
        child1, child2 = crossover(parent1, parent2, config.crossover_rate)
        
        # Mutation
        mutate!(child1, config.mutation_rate, config.mutation_strength)
        if idx + 1 <= config.population_size
            mutate!(child2, config.mutation_rate, config.mutation_strength)
        end
        
        # Add to new population
        new_chromosomes[idx] = child1
        if idx + 1 <= config.population_size
            new_chromosomes[idx + 1] = child2
            idx += 2
        else
            idx += 1
        end
    end
    
    return Population(new_chromosomes, pop.generation + 1, pop.best_fitness, pop.fitness_history)
end

# Main evolution loop
function evolve!(pop::Population, config::GAConfig, target_fitness::Float64 = Inf;
                 verbose::Bool = true, callback::Function = (p, g) -> nothing)
    
    for generation in 1:config.generations
        pop = evolve_generation(pop, config)
        
        if verbose && generation % 10 == 0
            println("Generation $generation: Best fitness = $(round(pop.best_fitness, digits=4))")
        end
        
        # Call user callback
        callback(pop, generation)
        
        # Check for convergence
        if pop.best_fitness >= target_fitness
            if verbose
                println("Target fitness reached at generation $generation!")
            end
            break
        end
    end
    
    return pop
end

# =============================================================================
# CONFIGURATION I/O - DIRECTLY USES ModernConfigSystem
# =============================================================================

# Save best configuration using ModernConfigSystem's save_filter_config
function save_best_config(pop::Population, filename::String)
    sort!(pop.chromosomes, by=c->c.fitness, rev=true)
    best_chr = pop.chromosomes[1]
    
    # Convert to ExtendedFilterConfig
    config = chromosome_to_config(best_chr, "ga_optimized", 
                                 "GA-optimized configuration (fitness: $(round(best_chr.fitness, digits=4)))")
    
    # Use ModernConfigSystem's save function
    save_filter_config(config, filename)
    
    println("Best configuration saved to $filename (fitness: $(round(best_chr.fitness, digits=4)))")
end

# Save GA state for checkpoint/resume
function save_ga_state(pop::Population, config::GAConfig, filename::String)
    # Create state structure for TOML
    state_data = Dict{String, Any}()
    state_data["generation"] = pop.generation
    state_data["best_fitness"] = pop.best_fitness
    state_data["fitness_history"] = pop.fitness_history
    
    state_data["ga_config"] = Dict(
        "population_size" => config.population_size,
        "generations" => config.generations,
        "mutation_rate" => config.mutation_rate,
        "mutation_strength" => config.mutation_strength,
        "crossover_rate" => config.crossover_rate,
        "elitism_count" => config.elitism_count,
        "tournament_size" => config.tournament_size
    )
    
    state_data["chromosomes"] = []
    for chr in pop.chromosomes
        push!(state_data["chromosomes"], Dict(
            "genes" => chr.genes,
            "num_filters" => chr.num_filters,
            "active_periods" => chr.active_periods,
            "fitness" => chr.fitness,
            "evaluated" => chr.evaluated
        ))
    end
    
    open(filename, "w") do f
        TOML.print(f, state_data)
    end
    
    println("GA state saved to $filename")
end

# Load GA state from checkpoint
function load_ga_state(filename::String)::Tuple{Population, GAConfig}
    state_data = TOML.parsefile(filename)
    
    # Reconstruct GA config
    gc = state_data["ga_config"]
    config = GAConfig(
        population_size = gc["population_size"],
        generations = gc["generations"],
        mutation_rate = gc["mutation_rate"],
        mutation_strength = gc["mutation_strength"],
        crossover_rate = gc["crossover_rate"],
        elitism_count = gc["elitism_count"],
        tournament_size = gc["tournament_size"]
    )
    
    # Reconstruct chromosomes
    chromosomes = Chromosome[]
    for chr_data in state_data["chromosomes"]
        chr = Chromosome(
            Vector{Float64}(chr_data["genes"]),
            chr_data["num_filters"],
            Vector{Int}(chr_data["active_periods"]),
            chr_data["fitness"],
            chr_data["evaluated"]
        )
        push!(chromosomes, chr)
    end
    
    # Reconstruct population
    pop = Population(
        chromosomes,
        state_data["generation"],
        state_data["best_fitness"],
        Vector{Float64}(state_data["fitness_history"])
    )
    
    return (pop, config)
end

# Get best chromosome from population
function get_best_chromosome(pop::Population)::Chromosome
    @assert !isempty(pop.chromosomes) "Population cannot be empty"
    update_fitness!(pop)
    return pop.chromosomes[argmax([c.fitness for c in pop.chromosomes])]
end

# Get best configuration as ExtendedFilterConfig
function get_best_config(pop::Population)::ExtendedFilterConfig
    best_chr = get_best_chromosome(pop)
    return chromosome_to_config(best_chr)
end

# Get parameter specifications (for external use)
function get_parameter_specs()::Vector{ParameterSpec}
    return PARAMETER_SPECS
end

end # module FilterParameterGA