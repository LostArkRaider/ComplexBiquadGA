# test/test_chunk1.jl - Unit Tests for Core GA Infrastructure (Chunk 1)
# Tests for genetic operators, parameter encoding, population init, and GA evolution
# Run this file from the project root: julia test/test_chunk1.jl
# Or from Julia REPL at project root: include("test/test_chunk1.jl")

using Test
using Random
using Statistics
using LinearAlgebra  # Add this for norm function

# =============================================================================
# TEST UTILITIES
# =============================================================================

"""
Create a test fitness function
"""
function test_fitness_function(chromosome::Vector{Float32}, period::Int32)::Float32
    # Simple fitness: prefer middle values for most parameters
    fitness = 1.0f0
    
    for i in 1:13
        if i == 7  # Binary parameter
            continue
        elseif i == 11  # Discrete parameter
            continue
        else
            # Prefer middle range values
            normalized = (chromosome[i] - minimum(chromosome)) / 
                        (maximum(chromosome) - minimum(chromosome) + 1e-6)
            fitness *= (1.0f0 - abs(normalized - 0.5f0))
        end
    end
    
    return fitness
end

"""
Create temporary test directory
"""
function setup_test_dir()::String
    test_dir = joinpath(tempdir(), "ga_chunk1_test_$(randstring(8))")
    mkpath(test_dir)
    return test_dir
end

"""
Clean up test directory
"""
function cleanup_test_dir(test_dir::String)
    if isdir(test_dir)
        rm(test_dir, recursive=true)
    end
end

# =============================================================================
# PARAMETER ENCODING TESTS
# =============================================================================

@testset "ParameterEncoding Module Tests" begin
    ranges = ParameterRanges()
    
    @testset "Individual Parameter Encoding/Decoding" begin
        # Test linear parameter (q_factor)
        @test encode_parameter(2.0, Int32(1), ranges) ≈ 2.0f0
        @test decode_parameter(2.0f0, Int32(1), ranges) ≈ 2.0f0
        
        # Test logarithmic parameter (batch_size)
        encoded = encode_parameter(1000, Int32(2), ranges)
        @test encoded ≈ log(1000)
        decoded = decode_parameter(encoded, Int32(2), ranges)
        @test decoded == 1000
        
        # Test binary parameter (enable_clamping)
        @test encode_parameter(true, Int32(7), ranges) == 1.0f0
        @test encode_parameter(false, Int32(7), ranges) == 0.0f0
        @test decode_parameter(1.0f0, Int32(7), ranges) == true
        @test decode_parameter(0.0f0, Int32(7), ranges) == false
        
        # Test discrete parameter (phase_error_history_length)
        encoded = encode_parameter(20, Int32(11), ranges)
        @test encoded == 4.0f0  # Index of 20 in options
        @test decode_parameter(4.0f0, Int32(11), ranges) == 20
    end
    
    @testset "Complex Weight Encoding" begin
        # Test magnitude and phase encoding
        real, imag = 1.0f0, 1.0f0
        mag, phase = encode_complex_weight(real, imag)
        @test mag ≈ sqrt(2.0f0)
        @test phase ≈ π/4
        
        # Test decoding
        decoded_real, decoded_imag = decode_complex_weight(mag, phase)
        @test decoded_real ≈ real atol=1e-5
        @test decoded_imag ≈ imag atol=1e-5
    end
    
    @testset "Full Chromosome Operations" begin
        # Create test parameters - USE proper types for each parameter
        params = Any[
            Float32(2.0),      # q_factor
            Int32(1000),       # batch_size  
            Float32(0.1),      # phase_detector_gain
            Float32(0.01),     # loop_bandwidth
            Float32(0.7),      # lock_threshold
            Float32(0.995),    # ring_decay
            true,              # enable_clamping (Bool)
            Float32(1e-6),     # clamping_threshold
            Float32(1.0),      # volume_scaling
            Float32(0.2),      # max_frequency_deviation
            Int32(20),         # phase_error_history_length
            Float32(1.0),      # complex_weight_real
            Float32(0.0)       # complex_weight_imag
        ]
        
        # Encode and decode
        chromosome = encode_chromosome(params, ranges)
        @test length(chromosome) == 13
        
        decoded = decode_chromosome(chromosome, ranges)
        @test decoded[1] ≈ 2.0f0 atol=1e-5
        @test decoded[2] == 1000
        @test decoded[7] == true
        @test decoded[11] == 20
    end
    
    @testset "Bounds Validation" begin
        chromosome = randn(Float32, 13) * 100  # Random values, likely out of bounds
        
        # Apply bounds
        apply_bounds!(chromosome, ranges)
        
        # Validate
        @test validate_chromosome(chromosome, ranges) == true
        
        # Check specific bounds
        @test ranges.q_factor_range[1] <= chromosome[1] <= ranges.q_factor_range[2]
        @test chromosome[7] in [0.0f0, 1.0f0]
    end
end

# =============================================================================
# GENETIC OPERATORS TESTS
# =============================================================================

@testset "GeneticOperators Module Tests" begin
    ranges = ParameterRanges()
    rng = MersenneTwister(42)
    
    @testset "Tournament Selection" begin
        pop_size = 20
        population = randn(Float32, pop_size, 13)
        fitness = rand(Float32, pop_size)
        
        # Run tournament
        winner_idx = tournament_selection(population, fitness, Int32(5), rng=rng)
        
        @test 1 <= winner_idx <= pop_size
        
        # Run multiple tournaments and check that higher fitness individuals win more often
        wins = zeros(Int, pop_size)
        for _ in 1:100
            idx = tournament_selection(population, fitness, Int32(5), rng=rng)
            wins[idx] += 1
        end
        
        # Check that the best individual has won at least once
        best_idx = argmax(fitness)
        @test wins[best_idx] > 0
    end
    
    @testset "Elite Selection" begin
        fitness = Float32[0.1, 0.5, 0.3, 0.9, 0.7, 0.2]
        elite_indices = elite_selection(fitness, Int32(3))
        
        @test length(elite_indices) == 3
        @test elite_indices[1] == 4  # Highest fitness
        @test elite_indices[2] == 5  # Second highest
        @test elite_indices[3] == 2  # Third highest
    end
    
    @testset "Uniform Crossover" begin
        parent1 = ones(Float32, 13)
        parent2 = zeros(Float32, 13)
        offspring1 = Vector{Float32}(undef, 13)
        offspring2 = Vector{Float32}(undef, 13)
        
        uniform_crossover!(offspring1, offspring2, parent1, parent2, 1.0f0, rng=rng)
        
        # Check that genes were mixed
        @test any(offspring1 .== 0.0f0)
        @test any(offspring1 .== 1.0f0)
        @test any(offspring2 .== 0.0f0)
        @test any(offspring2 .== 1.0f0)
    end
    
    @testset "Gaussian Mutation" begin
        chromosome = ones(Float32, 13) * 0.5f0
        original = copy(chromosome)
        
        gaussian_mutation!(chromosome, 1.0f0, ranges, rng=rng)  # 100% mutation rate
        
        # Check that mutations occurred
        @test chromosome != original
        
        # Check bounds are respected
        @test validate_chromosome(chromosome, ranges)
        
        # Test binary mutation
        chromosome[7] = 1.0f0
        gaussian_mutation!(chromosome, 1.0f0, ranges, rng=rng)
        @test chromosome[7] in [0.0f0, 1.0f0]
    end
    
    @testset "Population Evolution" begin
        pop_size = 10
        population = Main.PopulationInit.initialize_population(Int32(pop_size), ranges, rng=rng)
        fitness = rand(Float32, pop_size)
        ga_params = GAParameters()
        
        original_pop = copy(population)
        
        evolve_population!(population, fitness, ga_params, ranges, rng=rng)
        
        # Population should change
        @test population != original_pop
        
        # Elite should be preserved
        best_idx = argmax(fitness)
        best_chromosome = original_pop[best_idx, :]
        
        # At least one individual should match the best (elite)
        found_elite = false
        for i in 1:pop_size
            if all(population[i, :] .≈ best_chromosome)
                found_elite = true
                break
            end
        end
        @test found_elite
    end
    
    @testset "Population Diversity" begin
        # Identical population should have zero diversity
        population = ones(Float32, 5, 13)
        @test Main.GeneticOperators.population_diversity(population) == 0.0f0
        
        # Different population should have positive diversity
        population = randn(Float32, 5, 13)
        @test Main.GeneticOperators.population_diversity(population) > 0.0f0
    end
end

# =============================================================================
# POPULATION INITIALIZATION TESTS
# =============================================================================

@testset "PopulationInit Module Tests" begin
    ranges = ParameterRanges()
    rng = MersenneTwister(42)
    
    @testset "Random Initialization" begin
        pop_size = Int32(20)
        population = initialize_population(pop_size, ranges, rng=rng)
        
        @test size(population) == (20, 13)
        @test validate_population(population, ranges)
        
        # Check diversity
        @test population_diversity(population) > 0.0f0
    end
    
    @testset "Seeded Initialization" begin
        seed_chromosome = ones(Float32, 13) * 0.5f0
        pop_size = Int32(10)
        
        population = initialize_from_seed(seed_chromosome, pop_size, ranges, 
                                        diversity=0.1f0, rng=rng)
        
        @test size(population) == (10, 13)
        @test population[1, :] == seed_chromosome
        
        # Others should be similar but not identical
        for i in 2:10
            distance = norm(population[i, :] - seed_chromosome)
            @test distance > 0.0f0
            @test distance < 10.0f0  # Adjusted to be more realistic
        end
    end
    
    @testset "LHS Initialization" begin
        pop_size = Int32(20)
        population = Main.PopulationInit.initialize_lhs(pop_size, ranges, rng=rng)
        
        @test size(population) == (20, 13)
        @test validate_population(population, ranges)
        
        # Check coverage for continuous parameters
        for j in [1, 5, 6, 10]  # Linear parameters
            values = population[:, j]
            bounds = get_parameter_bounds(Int32(j), ranges)
            @test minimum(values) <= bounds[1] + 0.2f0 * (bounds[2] - bounds[1])
            @test maximum(values) >= bounds[2] - 0.2f0 * (bounds[2] - bounds[1])
        end
    end
    
    @testset "Opposition-Based Initialization" begin
        base_size = Int32(5)
        population = Main.PopulationInit.initialize_opposition(base_size, ranges, rng=rng)
        
        @test size(population) == (10, 13)  # 2x base size
        @test validate_population(population, ranges)
    end
    
    @testset "Population Repair" begin
        population = randn(Float32, 10, 13) * 1000  # Likely out of bounds
        
        repair_population!(population, ranges)
        
        # Debug: Check what's failing validation
        is_valid = validate_population(population, ranges)
        if !is_valid
            # Check each chromosome to find the issue
            for i in 1:10
                if !Main.ParameterEncoding.validate_chromosome(population[i, :], ranges)
                    println("Chromosome $i failed validation")
                    for j in 1:13
                        bounds = Main.ParameterEncoding.get_parameter_bounds(Int32(j), ranges)
                        val = population[i, j]
                        if val < bounds[1] || val > bounds[2]
                            println("  Param $j: value=$val, bounds=$bounds")
                        end
                    end
                end
            end
        end
        
        @test is_valid
    end
end

# =============================================================================
# SINGLE FILTER GA TESTS
# =============================================================================

@testset "SingleFilterGA Module Tests" begin
    ranges = ParameterRanges()
    ga_params = GAParameters(max_generations=Int32(50))
    
    @testset "GA Creation" begin
        ga = Main.SingleFilterGA.SingleFilterGAComplete(Int32(21), Int32(1), Int32(20),
                                   ranges, ga_params, seed=42)
        
        @test ga.period == 21
        @test ga.filter_index == 1
        @test size(ga.population) == (20, 13)
        @test length(ga.fitness) == 20
        @test ga.generation == 0
        @test ga.converged == false
    end
    
    @testset "Fitness Evaluation" begin
        ga = Main.SingleFilterGA.SingleFilterGAComplete(Int32(21), Int32(1), Int32(10),
                                   ranges, ga_params, seed=42)
        
        # Test with stub fitness
        Main.SingleFilterGA.evaluate_fitness!(ga)
        
        @test all(ga.fitness .>= 0.0f0)
        @test all(ga.fitness .<= 1.0f0)
        @test ga.total_evaluations == 10
        
        # Test with custom fitness function
        Main.SingleFilterGA.evaluate_fitness!(ga, test_fitness_function)
        @test ga.total_evaluations == 20
    end
    
    @testset "Evolution" begin
        ga = Main.SingleFilterGA.SingleFilterGAComplete(Int32(21), Int32(1), Int32(10),
                                   ranges, ga_params, seed=42)
        
        # Evolve for one generation
        Main.SingleFilterGA.evolve!(ga, verbose=false)
        
        @test ga.generation == 1
        @test ga.best_fitness >= 0.0f0
        @test length(ga.fitness_history) == 1
        @test length(ga.diversity_history) == 1
        
        # Evolve for multiple generations
        for _ in 1:10
            Main.SingleFilterGA.evolve!(ga, verbose=false)
        end
        
        @test ga.generation == 11
        @test length(ga.fitness_history) == 11
    end
    
    @testset "Convergence Detection" begin
        ga = Main.SingleFilterGA.SingleFilterGAComplete(Int32(21), Int32(1), Int32(10),
                                   ranges, ga_params, seed=42)
        
        # Initially not converged
        @test Main.SingleFilterGA.check_convergence(ga) == false
        
        # Simulate stagnation
        ga.generation = 20
        ga.generations_since_improvement = 25
        @test Main.SingleFilterGA.check_convergence(ga) == true
        
        # Test max generations
        ga2 = Main.SingleFilterGA.SingleFilterGAComplete(Int32(21), Int32(1), Int32(10),
                                    ranges, ga_params, seed=42)
        ga2.generation = 50
        @test Main.SingleFilterGA.check_convergence(ga2) == true
    end
    
    @testset "Best Solution Tracking" begin
        ga = Main.SingleFilterGA.SingleFilterGAComplete(Int32(21), Int32(1), Int32(10),
                                   ranges, ga_params, seed=42)
        
        # Set up fake fitness
        ga.fitness = Float32[0.1, 0.5, 0.9, 0.3, 0.7, 0.2, 0.4, 0.6, 0.8, 0.35]
        ga.population[3, :] = ones(Float32, 13) * 0.5f0  # Best individual
        
        Main.SingleFilterGA.update_best!(ga)
        
        @test ga.best_fitness == 0.9f0
        @test ga.best_chromosome == ga.population[3, :]
        @test ga.best_generation == 0
    end
    
    @testset "GA Reset" begin
        ga = Main.SingleFilterGA.SingleFilterGAComplete(Int32(21), Int32(1), Int32(10),
                                   ranges, ga_params, seed=42)
        
        # Evolve and then reset
        for _ in 1:5
            Main.SingleFilterGA.evolve!(ga, verbose=false)
        end
        
        original_best = copy(ga.best_chromosome)
        Main.SingleFilterGA.reset_ga!(ga, keep_best=true)
        
        @test ga.generation == 0
        @test ga.converged == false
        @test isempty(ga.fitness_history)
        @test ga.best_chromosome == original_best
    end
    
    @testset "Statistics" begin
        ga = Main.SingleFilterGA.SingleFilterGAComplete(Int32(21), Int32(1), Int32(10),
                                   ranges, ga_params, seed=42)
        
        Main.SingleFilterGA.evolve!(ga, verbose=false)
        stats = Main.SingleFilterGA.get_statistics(ga)
        
        @test stats["period"] == 21
        @test stats["generation"] == 1
        @test haskey(stats, "best_fitness")
        @test haskey(stats, "current_mean_fitness")
        @test haskey(stats, "current_diversity")
    end
end

# =============================================================================
# FILTER BANK GA TESTS
# =============================================================================

@testset "FilterBankGA Module Tests" begin
    test_dir = setup_test_dir()
    
    try
        @testset "FilterBankGA Creation" begin
            # Create test configuration
            config = InstrumentConfig(
                symbol = "TEST",
                num_filters = Int32(3),
                population_size = Int32(10),
                parameter_path = joinpath(test_dir, "TEST/parameters/active.jld2"),
                ga_workspace_path = joinpath(test_dir, "TEST/ga_workspace/"),
                config_path = joinpath(test_dir, "TEST/config.toml"),
                fibonacci_periods = Int32[3, 5, 8],
                ga_params = GAParameters(max_generations=Int32(20))
            )
            
            fb_ga = Main.FilterBankGA.FilterBankGAComplete(config, master_seed=42)
            
            @test fb_ga.instrument == "TEST"
            @test fb_ga.num_filters == 3
            @test length(fb_ga.filter_gas) == 3
            @test fb_ga.generation == 0
        end
        
        @testset "Evolution Generation" begin
            config = InstrumentConfig(
                symbol = "TEST",
                num_filters = Int32(2),
                population_size = Int32(5),
                parameter_path = joinpath(test_dir, "TEST/parameters/active.jld2"),
                ga_workspace_path = joinpath(test_dir, "TEST/ga_workspace/"),
                config_path = joinpath(test_dir, "TEST/config.toml"),
                fibonacci_periods = Int32[3, 5],
                ga_params = GAParameters()
            )
            
            fb_ga = Main.FilterBankGA.FilterBankGAComplete(config, master_seed=42)
            
            # Evolve one generation
            Main.FilterBankGA.evolve_generation!(fb_ga, verbose=false)
            
            @test fb_ga.generation == 1
            @test length(fb_ga.best_fitness_history) == 1
            @test length(fb_ga.mean_fitness_history) == 1
            @test length(fb_ga.convergence_history) == 1
            
            # Check that filters evolved
            for filter_ga in fb_ga.filter_gas
                @test filter_ga.generation >= 1
            end
        end
        
        @testset "Storage Integration" begin
            # Ensure directory exists
            mkpath(joinpath(test_dir, "TEST/parameters"))
            
            config = InstrumentConfig(
                symbol = "TEST",
                num_filters = Int32(2),
                population_size = Int32(5),
                parameter_path = joinpath(test_dir, "TEST/parameters/active.jld2"),
                ga_workspace_path = joinpath(test_dir, "TEST/ga_workspace/"),
                config_path = joinpath(test_dir, "TEST/config.toml"),
                fibonacci_periods = Int32[3, 5],
                ga_params = GAParameters()
            )
            
            fb_ga = Main.FilterBankGA.FilterBankGAComplete(config, master_seed=42)
            
            # Evolve and sync
            Main.FilterBankGA.evolve_generation!(fb_ga, verbose=false)
            Main.FilterBankGA.sync_with_storage!(fb_ga)
            
            @test isfile(config.parameter_path)
            
            # Test loading
            fb_ga2 = Main.FilterBankGA.FilterBankGAComplete(config, master_seed=43)
            Main.FilterBankGA.load_from_storage!(fb_ga2)
            
            # Check parameters were loaded
            params1 = Main.FilterBankGA.get_best_parameters(fb_ga)
            params2 = Main.FilterBankGA.get_best_parameters(fb_ga2)
            
            @test size(params1) == size(params2)
        end
        
        @testset "Convergence Status" begin
            config = InstrumentConfig(
                symbol = "TEST",
                num_filters = Int32(2),
                population_size = Int32(5),
                parameter_path = joinpath(test_dir, "TEST/parameters/active.jld2"),
                ga_workspace_path = joinpath(test_dir, "TEST/ga_workspace/"),
                config_path = joinpath(test_dir, "TEST/config.toml"),
                fibonacci_periods = Int32[3, 5],
                ga_params = GAParameters()
            )
            
            fb_ga = Main.FilterBankGA.FilterBankGAComplete(config, master_seed=42)
            
            # Mark one filter as converged
            fb_ga.filter_gas[1].converged = true
            
            status = Main.FilterBankGA.get_convergence_status(fb_ga)
            
            @test status["instrument"] == "TEST"
            @test status["converged_filters"] == 1
            @test status["total_filters"] == 2
            @test status["convergence_rate"] == 0.5f0
        end
        
        @testset "Multi-Generation Evolution" begin
            config = InstrumentConfig(
                symbol = "TEST",
                num_filters = Int32(2),
                population_size = Int32(5),
                parameter_path = joinpath(test_dir, "TEST/parameters/active.jld2"),
                ga_workspace_path = joinpath(test_dir, "TEST/ga_workspace/"),
                config_path = joinpath(test_dir, "TEST/config.toml"),
                fibonacci_periods = Int32[3, 5],
                ga_params = GAParameters(max_generations=Int32(10))
            )
            
            fb_ga = Main.FilterBankGA.FilterBankGAComplete(config, master_seed=42)
            
            # Evolve for multiple generations
            Main.FilterBankGA.evolve_instrument!(fb_ga, generations=5, verbose=false)
            
            @test fb_ga.generation == 5
            @test length(fb_ga.best_fitness_history) == 5
            @test fb_ga.total_evaluations > 0
        end
        
    finally
        cleanup_test_dir(test_dir)
    end
end

# =============================================================================
# INTEGRATION TESTS
# =============================================================================

@testset "Chunk 1 Integration Tests" begin
    test_dir = setup_test_dir()
    
    try
        @testset "Complete GA Pipeline" begin
            # 1. Create configuration
            config = InstrumentConfig(
                symbol = "INTEGRATION",
                num_filters = Int32(3),
                population_size = Int32(10),
                parameter_path = joinpath(test_dir, "INT/parameters/active.jld2"),
                ga_workspace_path = joinpath(test_dir, "INT/ga_workspace/"),
                config_path = joinpath(test_dir, "INT/config.toml"),
                fibonacci_periods = Int32[3, 5, 8],
                ga_params = GAParameters(
                    mutation_rate = 0.15f0,
                    crossover_rate = 0.8f0,
                    elite_size = Int32(2),
                    max_generations = Int32(20)
                )
            )
            
            # 2. Create filter bank GA
            fb_ga = Main.FilterBankGA.FilterBankGAComplete(config, master_seed=42, init_strategy=:lhs)
            
            # 3. Run evolution
            Main.FilterBankGA.evolve_instrument!(fb_ga, generations=10, 
                             fitness_function=test_fitness_function,
                             verbose=false)
            
            # 4. Check results
            @test fb_ga.generation == 10
            
            # FIXED EXPECTATION:
            # Initial evaluation: 3 filters × 10 population = 30
            # Per generation: 3 filters × 10 population = 30
            # Total for 10 generations: 30 + (10 × 30) = 330
            # Note: Each generation evaluates the new population created
            @test fb_ga.total_evaluations == 330
            
            # 5. Get best parameters
            best_params = Main.FilterBankGA.get_best_parameters(fb_ga)
            @test size(best_params) == (3, 13)
            
            # 6. Decode parameters
            ranges = ParameterRanges()
            for i in 1:3
                decoded = decode_chromosome(best_params[i, :], ranges)
                @test length(decoded) == 13
            end
            
            # 7. Check convergence
            status = Main.FilterBankGA.get_convergence_status(fb_ga)
            @test status["generation"] == 10
            @test haskey(status["filters"], Int32(3))
        end
        
        @testset "Filter Independence Verification" begin
            # Create two identical configurations
            config = InstrumentConfig(
                symbol = "INDEP",
                num_filters = Int32(2),
                population_size = Int32(5),
                parameter_path = joinpath(test_dir, "INDEP/parameters/active.jld2"),
                ga_workspace_path = joinpath(test_dir, "INDEP/ga_workspace/"),
                config_path = joinpath(test_dir, "INDEP/config.toml"),
                fibonacci_periods = Int32[3, 5],
                ga_params = GAParameters()
            )
            
            fb_ga = Main.FilterBankGA.FilterBankGAComplete(config, master_seed=42)
            
            # Modify one filter's population
            fb_ga.filter_gas[1].population .= 1.0f0
            fb_ga.filter_gas[2].population .= 0.0f0
            
            # Evolve
            Main.FilterBankGA.evolve_generation!(fb_ga, verbose=false)
            
            # Check that populations remain different
            pop1_mean = mean(fb_ga.filter_gas[1].population)
            pop2_mean = mean(fb_ga.filter_gas[2].population)
            
            @test abs(pop1_mean - pop2_mean) > 0.1  # Significant difference
        end
        
    finally
        cleanup_test_dir(test_dir)
    end
end

println("\n✅ All Chunk 1 tests completed successfully!")