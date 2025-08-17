# src/PopulationInit.jl - Population Initialization for GA
# Random and seeded initialization strategies

module PopulationInit

using Random
using Statistics

export initialize_population, initialize_from_seed, initialize_from_defaults,
       add_noise_to_chromosome, validate_population, repair_population!

# =============================================================================
# RANDOM INITIALIZATION
# =============================================================================

"""
Initialize population with random chromosomes within bounds
"""
function initialize_population(pop_size::Int32, ranges;
                              rng::AbstractRNG = Random.default_rng())::Matrix{Float32}
    
    population = Matrix{Float32}(undef, pop_size, 13)
    
    for i in 1:pop_size
        population[i, :] = random_chromosome(ranges, rng)
    end
    
    return population
end

"""
Generate a single random chromosome
"""
function random_chromosome(ranges, rng::AbstractRNG)::Vector{Float32}
    chromosome = Vector{Float32}(undef, 13)
    
    for j in 1:13
        bounds = Main.ParameterEncoding.get_parameter_bounds(Int32(j), ranges)
        
        if j == 7  # Binary parameter (enable_clamping)
            chromosome[j] = rand(rng) > 0.5 ? 1.0f0 : 0.0f0
            
        elseif j == 11  # Discrete parameter (phase_error_history_length)
            n_options = Int32(bounds[2])
            chromosome[j] = Float32(rand(rng, 1:n_options))
            
        else  # Continuous parameters
            # Use uniform distribution within bounds
            chromosome[j] = bounds[1] + (bounds[2] - bounds[1]) * rand(rng, Float32)
        end
    end
    
    return chromosome
end

# =============================================================================
# SEEDED INITIALIZATION
# =============================================================================

"""
Initialize population around a seed chromosome with controlled diversity
"""
function initialize_from_seed(seed_chromosome::Vector{Float32},
                             pop_size::Int32,
                             ranges;
                             diversity::Float32 = 0.1f0,
                             rng::AbstractRNG = Random.default_rng())::Matrix{Float32}
    
    if length(seed_chromosome) != 13
        error("Seed chromosome must have 13 elements")
    end
    
    population = Matrix{Float32}(undef, pop_size, 13)
    
    # First individual is the seed
    population[1, :] = seed_chromosome
    
    # Generate rest with perturbations
    for i in 2:pop_size
        population[i, :] = add_noise_to_chromosome(seed_chromosome, ranges, 
                                                  diversity, rng)
    end
    
    return population
end

"""
Add controlled noise to a chromosome
"""
function add_noise_to_chromosome(chromosome::Vector{Float32},
                                ranges,
                                noise_level::Float32 = 0.1f0,
                                rng::AbstractRNG = Random.default_rng())::Vector{Float32}
    
    noisy = copy(chromosome)
    
    for i in 1:13
        bounds = Main.ParameterEncoding.get_parameter_bounds(Int32(i), ranges)
        
        if i == 7  # Binary parameter
            # Flip with probability proportional to noise_level
            if rand(rng, Float32) < noise_level
                noisy[i] = noisy[i] > 0.5f0 ? 0.0f0 : 1.0f0
            end
            
        elseif i == 11  # Discrete parameter
            # Random walk
            n_options = Int32(bounds[2])
            current = round(Int32, noisy[i])
            
            if rand(rng, Float32) < noise_level
                step = rand(rng, [-1, 1])
                new_val = clamp(current + step, 1, n_options)
                noisy[i] = Float32(new_val)
            end
            
        else  # Continuous parameters
            # Gaussian noise scaled by range and noise_level
            range_width = bounds[2] - bounds[1]
            noise = randn(rng, Float32) * noise_level * range_width
            noisy[i] = clamp(noisy[i] + noise, bounds...)
        end
    end
    
    return noisy
end

# =============================================================================
# INITIALIZATION FROM DEFAULTS
# =============================================================================

"""
Initialize population from default parameters with diversity
"""
function initialize_from_defaults(defaults, 
                                 pop_size::Int32,
                                 ranges;
                                 diversity::Float32 = 0.15f0,
                                 rng::AbstractRNG = Random.default_rng())::Matrix{Float32}
    
    # Create default chromosome
    default_params = [
        defaults.default_q_factor,
        defaults.default_batch_size,
        defaults.default_pll_gain,
        defaults.default_loop_bandwidth,
        defaults.default_lock_threshold,
        defaults.default_ring_decay,
        defaults.default_enable_clamping,
        defaults.default_clamping_threshold,
        defaults.default_volume_scaling,
        defaults.default_max_frequency_deviation,
        defaults.default_phase_error_history_length,
        defaults.default_complex_weight_real,
        defaults.default_complex_weight_imag
    ]
    
    # Encode to chromosome
    default_chromosome = Main.ParameterEncoding.encode_chromosome(default_params, ranges)
    
    # Initialize population around defaults
    return initialize_from_seed(default_chromosome, pop_size, ranges, 
                               diversity=diversity, rng=rng)
end

# =============================================================================
# SPECIALIZED INITIALIZATION STRATEGIES
# =============================================================================

"""
Initialize with Latin Hypercube Sampling for better coverage
"""
function initialize_lhs(pop_size::Int32, ranges;
                       rng::AbstractRNG = Random.default_rng())::Matrix{Float32}
    
    population = Matrix{Float32}(undef, pop_size, 13)
    
    # For each parameter
    for j in 1:13
        bounds = Main.ParameterEncoding.get_parameter_bounds(Int32(j), ranges)
        
        if j == 7  # Binary parameter
            # Roughly half true, half false
            n_true = div(pop_size, 2)
            values = vcat(ones(Float32, n_true), zeros(Float32, pop_size - n_true))
            shuffle!(rng, values)
            population[:, j] = values
            
        elseif j == 11  # Discrete parameter
            n_options = Int32(bounds[2])
            # Distribute evenly across options
            values = Float32[]
            for opt in 1:n_options
                n_copies = div(pop_size, n_options)
                append!(values, fill(Float32(opt), n_copies))
            end
            # Fill remainder
            while length(values) < pop_size
                push!(values, Float32(rand(rng, 1:n_options)))
            end
            shuffle!(rng, values)
            population[:, j] = values[1:pop_size]
            
        else  # Continuous parameters
            # Create Latin hypercube for this dimension
            min_val, max_val = bounds
            segment_size = (max_val - min_val) / pop_size
            
            values = Float32[]
            for i in 1:pop_size
                # Random point within segment i
                seg_min = min_val + (i - 1) * segment_size
                seg_max = min_val + i * segment_size
                push!(values, seg_min + rand(rng, Float32) * (seg_max - seg_min))
            end
            shuffle!(rng, values)
            population[:, j] = values
        end
    end
    
    return population
end

"""
Initialize with opposition-based learning
"""
function initialize_opposition(base_pop_size::Int32, ranges;
                              rng::AbstractRNG = Random.default_rng())::Matrix{Float32}
    
    # Generate base population
    base_population = initialize_population(base_pop_size, ranges, rng=rng)
    
    # Generate opposite population
    opposite_population = Matrix{Float32}(undef, base_pop_size, 13)
    
    for i in 1:base_pop_size
        for j in 1:13
            bounds = Main.ParameterEncoding.get_parameter_bounds(Int32(j), ranges)
            
            if j == 7  # Binary - flip
                opposite_population[i, j] = base_population[i, j] > 0.5f0 ? 0.0f0 : 1.0f0
                
            elseif j == 11  # Discrete - opposite within range
                n_options = Int32(bounds[2])
                current = round(Int32, base_population[i, j])
                opposite = n_options - current + 1
                opposite_population[i, j] = Float32(clamp(opposite, 1, n_options))
                
            else  # Continuous - opposite point
                min_val, max_val = bounds
                opposite_population[i, j] = min_val + max_val - base_population[i, j]
            end
        end
    end
    
    # Combine base and opposite populations
    combined = vcat(base_population, opposite_population)
    
    return combined
end

# =============================================================================
# VALIDATION AND REPAIR
# =============================================================================

"""
Validate that all chromosomes in population are within bounds
"""
function validate_population(population::Matrix{Float32}, ranges)::Bool
    pop_size = size(population, 1)
    
    for i in 1:pop_size
        if !Main.ParameterEncoding.validate_chromosome(population[i, :], ranges)
            return false
        end
    end
    
    return true
end

"""
Repair population by enforcing bounds
"""
function repair_population!(population::Matrix{Float32}, ranges)
    pop_size = size(population, 1)
    
    for i in 1:pop_size
        for j in 1:13
            bounds = Main.ParameterEncoding.get_parameter_bounds(Int32(j), ranges)
            population[i, j] = clamp(population[i, j], bounds...)
        end
    end
    
    return population
end

# =============================================================================
# DIVERSITY ENHANCEMENT
# =============================================================================

"""
Ensure minimum diversity in population
"""
function ensure_diversity!(population::Matrix{Float32}, 
                         ranges,
                         min_diversity::Float32 = 0.05f0;
                         rng::AbstractRNG = Random.default_rng())
    
    pop_size = size(population, 1)
    
    # Calculate pairwise distances
    for i in 1:(pop_size-1)
        for j in (i+1):pop_size
            distance = sqrt(sum((population[i, :] .- population[j, :]).^2))
            
            # If too similar, perturb one
            if distance < min_diversity
                population[j, :] = add_noise_to_chromosome(population[j, :], 
                                                          ranges, 0.2f0, rng)
            end
        end
    end
    
    return population
end

"""
Create diverse initial population using multiple strategies
"""
function initialize_diverse(pop_size::Int32, 
                          ranges,
                          defaults = nothing;
                          rng::AbstractRNG = Random.default_rng())::Matrix{Float32}
    
    # Allocate populations for different strategies
    n_strategies = 4
    sub_pop_size = div(pop_size, n_strategies)
    remainder = pop_size - (sub_pop_size * n_strategies)
    
    populations = Matrix{Float32}[]
    
    # Strategy 1: Pure random
    push!(populations, initialize_population(Int32(sub_pop_size), ranges, rng=rng))
    
    # Strategy 2: Latin Hypercube
    push!(populations, initialize_lhs(Int32(sub_pop_size), ranges, rng=rng))
    
    # Strategy 3: Opposition-based (generates 2x, so halve the size)
    opp_size = div(sub_pop_size, 2)
    opp_pop = initialize_opposition(Int32(opp_size), ranges, rng=rng)
    push!(populations, opp_pop[1:sub_pop_size, :])
    
    # Strategy 4: From defaults or random
    if defaults !== nothing
        push!(populations, initialize_from_defaults(defaults, Int32(sub_pop_size), 
                                                    ranges, rng=rng))
    else
        push!(populations, initialize_population(Int32(sub_pop_size + remainder), 
                                                ranges, rng=rng))
    end
    
    # Combine all strategies
    combined = vcat(populations...)
    
    # Ensure we have exactly pop_size individuals
    if size(combined, 1) > pop_size
        combined = combined[1:pop_size, :]
    elseif size(combined, 1) < pop_size
        # Add random individuals to fill
        extra = initialize_population(Int32(pop_size - size(combined, 1)), 
                                     ranges, rng=rng)
        combined = vcat(combined, extra)
    end
    
    return combined
end

end # module PopulationInit