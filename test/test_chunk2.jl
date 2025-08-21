# test/test_chunk2.jl - Unit Tests for Multi-Instrument Support and Storage Architecture
# Tests for GATypes, InstrumentManager, StorageSystem, and ConfigurationLoader

using Test
using Dates
using TOML
using JLD2

# =============================================================================
# TEST UTILITIES
# =============================================================================

"""
Create a temporary test directory
"""
function setup_test_dir()::String
    test_dir = joinpath(tempdir(), "ga_test_$(randstring(8))")
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
# GATYPES TESTS
# =============================================================================

@testset "GATypes Module Tests" begin
    @testset "Parameter Ranges" begin
        ranges = ParameterRanges()
        
        @test ranges.q_factor_range == (0.5f0, 10.0f0)
        @test ranges.batch_size_range == (100, 5000)
        @test ranges.phase_detector_gain_range == (0.001f0, 1.0f0)
        @test length(ranges.phase_error_history_length_options) == 7
        @test ranges.complex_weight_mag_range == (0.0f0, 2.0f0)
    end
    
    @testset "FilterDefaults" begin
        defaults = FilterDefaults()
        
        @test defaults.default_q_factor == 2.0f0
        @test defaults.default_batch_size == 1000
        @test defaults.default_enable_clamping == false
        @test defaults.default_complex_weight_real == 1.0f0
        @test defaults.default_complex_weight_imag == 0.0f0
    end
    
    @testset "InstrumentConfig" begin
        config = InstrumentConfig(
            symbol = "YM",
            num_filters = Int32(50),
            population_size = Int32(100),
            parameter_path = "data/YM/parameters/active.jld2",
            ga_workspace_path = "data/YM/ga_workspace/",
            config_path = "data/YM/config.toml",
            fibonacci_periods = Int32[1, 2, 3, 5, 8, 13, 21, 34, 55]
        )
        
        @test config.symbol == "YM"
        @test config.num_filters == 50
        @test config.population_size == 100
        @test length(config.fibonacci_periods) == 9
        @test validate_instrument_config(config) == true
    end
    
    @testset "SingleFilterGA Stub" begin
        ga = SingleFilterGA(Int32(21), Int32(1), Int32(100))
        
        @test ga.period == 21
        @test ga.filter_index == 1
        @test size(ga.population) == (100, 13)
        @test length(ga.fitness) == 100
        @test length(ga.best_chromosome) == 13
        @test ga.converged == false
    end
    
    @testset "WriteThruStorage" begin
        storage = WriteThruStorage(Int32(50), "test.jld2", Int32(10))
        
        @test size(storage.active_params) == (50, 13)
        @test length(storage.dirty_filters) == 50
        @test all(.!storage.dirty_filters)
        @test storage.pending_updates == 0
        @test storage.sync_interval == 10
    end
    
    @testset "Default Chromosome Generation" begin
        defaults = FilterDefaults()
        chromosome = get_default_chromosome(Int32(21), defaults)
        
        @test length(chromosome) == 13
        @test chromosome[1] == defaults.default_q_factor
        @test chromosome[12] == defaults.default_complex_weight_real
        @test chromosome[13] == defaults.default_complex_weight_imag
    end
end

# =============================================================================
# INSTRUMENT MANAGER TESTS
# =============================================================================

@testset "InstrumentManager Module Tests" begin
    test_dir = setup_test_dir()
    
    try
        @testset "InstrumentGASystem Creation" begin
            system = InstrumentGASystem(joinpath(test_dir, "master_config.toml"))
            
            @test system.current_instrument === nothing
            @test isempty(system.instruments)
            @test isempty(system.active_instruments)
            @test system.gpu_enabled == false
            @test system.max_memory_gb == 12.0f0
            @test system.checkpoint_interval == 50
        end
        
        @testset "Master Config Creation and Loading" begin
            master_path = joinpath(test_dir, "master_config.toml")
            system = InstrumentGASystem(master_path)
            
            # Load should create default config
            load_master_config!(system)
            @test isfile(master_path)
            
            # Add an instrument
            config = InstrumentConfig(
                symbol = "TEST",
                num_filters = Int32(30),
                population_size = Int32(50),
                parameter_path = joinpath(test_dir, "TEST/parameters/active.jld2"),
                ga_workspace_path = joinpath(test_dir, "TEST/ga_workspace/"),
                config_path = joinpath(test_dir, "TEST/config.toml"),
                fibonacci_periods = Int32[1, 2, 3, 5, 8, 13]
            )
            
            @test add_instrument!(system, config) == true
            @test "TEST" in system.active_instruments
            @test haskey(system.instruments, "TEST")
        end
        
        @testset "Instrument Switching" begin
            master_path = joinpath(test_dir, "master_config.toml")
            system = InstrumentGASystem(master_path)
            
            # Add multiple instruments
            for symbol in ["YM", "ES"]
                config = InstrumentConfig(
                    symbol = symbol,
                    num_filters = Int32(25),
                    population_size = Int32(50),
                    parameter_path = joinpath(test_dir, "$symbol/parameters/active.jld2"),
                    ga_workspace_path = joinpath(test_dir, "$symbol/ga_workspace/"),
                    config_path = joinpath(test_dir, "$symbol/config.toml"),
                    fibonacci_periods = Int32[1, 2, 3, 5, 8]
                )
                add_instrument!(system, config)
            end
            
            @test switch_instrument!(system, "YM") == true
            @test system.current_instrument == "YM"
            
            @test switch_instrument!(system, "ES") == true
            @test system.current_instrument == "ES"
            
            @test switch_instrument!(system, "INVALID") == false
        end
        
        @testset "Memory Estimation" begin
            config = InstrumentConfig(
                symbol = "MEM_TEST",
                num_filters = Int32(50),
                population_size = Int32(100),
                parameter_path = "test",
                ga_workspace_path = "test",
                config_path = "test",
                fibonacci_periods = Int32[1, 2, 3]
            )
            
            memory_mb = estimate_memory_usage(config)
            @test memory_mb > 0
            @test memory_mb < 100  # Should be reasonable
        end
        
        @testset "Directory Creation" begin
            config = InstrumentConfig(
                symbol = "DIR_TEST",
                num_filters = Int32(10),
                population_size = Int32(20),
                parameter_path = joinpath(test_dir, "DIR_TEST/parameters/active.jld2"),
                ga_workspace_path = joinpath(test_dir, "DIR_TEST/ga_workspace/"),
                config_path = joinpath(test_dir, "DIR_TEST/config.toml"),
                fibonacci_periods = Int32[1, 2, 3]
            )
            
            create_instrument_directories(config)
            
            @test isdir(joinpath(test_dir, "DIR_TEST"))
            @test isdir(joinpath(test_dir, "DIR_TEST/parameters"))
            @test isdir(joinpath(test_dir, "DIR_TEST/ga_workspace"))
        end
        
    finally
        cleanup_test_dir(test_dir)
    end
end

# =============================================================================
# STORAGE SYSTEM TESTS
# =============================================================================

@testset "StorageSystem Module Tests" begin
    test_dir = setup_test_dir()
    
    try
        @testset "Storage Initialization" begin
            storage_path = joinpath(test_dir, "test_storage.jld2")
            storage = WriteThruStorage(Int32(10), storage_path, Int32(5))
            
            @test size(storage.active_params) == (10, 13)
            @test storage.jld2_path == storage_path
            @test storage.sync_interval == 5
            @test storage.pending_updates == 0
        end
        
        @testset "Parameter Get/Set" begin
            storage_path = joinpath(test_dir, "params.jld2")
            storage = WriteThruStorage(Int32(5), storage_path, Int32(10))
            
            # Set parameters for filter 1
            params = randn(Float32, 13)
            set_active_parameters!(storage, Int32(1), params)
            
            @test storage.dirty_filters[1] == true
            @test storage.pending_updates == 1
            
            # Get parameters back
            retrieved = get_active_parameters(storage, Int32(1))
            @test retrieved ≈ params
        end
        
        @testset "Storage Sync" begin
            storage_path = joinpath(test_dir, "sync_test.jld2")
            storage = WriteThruStorage(Int32(3), storage_path, Int32(10))
            
            # Modify some parameters
            for i in 1:3
                params = randn(Float32, 13)
                set_active_parameters!(storage, Int32(i), params)
            end
            
            @test storage.pending_updates == 3
            
            # Sync to disk
            sync_to_storage!(storage)
            
            @test isfile(storage_path)
            @test storage.pending_updates == 0
            @test all(.!storage.dirty_filters)
            
            # Load back
            storage2 = WriteThruStorage(Int32(3), storage_path, Int32(10))
            load_from_storage!(storage2)
            
            @test storage2.active_params ≈ storage.active_params
        end
        
        @testset "Checkpointing" begin
            storage_path = joinpath(test_dir, "checkpoint/active.jld2")
            mkpath(dirname(storage_path))
            storage = WriteThruStorage(Int32(5), storage_path, Int32(10))
            
            # Set some parameters
            test_params = randn(Float32, 5, 13)
            storage.active_params .= test_params
            
            # Create checkpoint
            checkpoint_file = create_checkpoint(storage, Int32(100), 0.95f0)
            @test isfile(checkpoint_file)
            
            # Modify parameters
            storage.active_params .= randn(Float32, 5, 13)
            
            # Restore from checkpoint
            @test restore_from_checkpoint(storage, checkpoint_file) == true
            @test storage.active_params ≈ test_params
            
            # List checkpoints
            checkpoints = list_checkpoints(storage)
            @test length(checkpoints) >= 1
        end
        
        @testset "Filter Defaults" begin
            defaults_path = joinpath(test_dir, "defaults.toml")
            defaults = FilterDefaults(
                default_q_factor = 3.0f0,
                default_batch_size = 2000,
                default_pll_gain = 0.2f0
            )
            
            save_filter_defaults(defaults, defaults_path)
            @test isfile(defaults_path)
            
            loaded_defaults = load_filter_defaults(defaults_path)
            @test loaded_defaults.default_q_factor == 3.0f0
            @test loaded_defaults.default_batch_size == 2000
            @test loaded_defaults.default_pll_gain == 0.2f0
        end
        
        @testset "Apply Defaults" begin
            storage_path = joinpath(test_dir, "apply_defaults.jld2")
            storage = WriteThruStorage(Int32(3), storage_path, Int32(10))
            
            periods = Int32[1, 2, 3]
            apply_defaults!(storage, periods)
            
            # Check that parameters were set
            for i in 1:3
                params = get_active_parameters(storage, Int32(i))
                @test length(params) == 13
                @test params[1] == storage.default_config.default_q_factor
            end
            
            # Should have been synced
            @test isfile(storage_path)
        end
        
    finally
        cleanup_test_dir(test_dir)
    end
end

# =============================================================================
# CONFIGURATION LOADER TESTS
# =============================================================================

@testset "ConfigurationLoader Module Tests" begin
    test_dir = setup_test_dir()
    
    try
        @testset "System Initialization" begin
            master_path = joinpath(test_dir, "data/master_config.toml")
            mkpath(dirname(master_path))
            
            system = initialize_ga_system(master_path)
            
            @test isa(system, InstrumentGASystem)
            @test system.master_config_path == master_path
            @test isfile(master_path)
        end
        
        @testset "Load or Create Instrument" begin
            master_path = joinpath(test_dir, "data/master_config.toml")
            system = InstrumentGASystem(master_path)
            
            # Create new instrument
            config = load_or_create_instrument("TEST", system)
            
            @test config.symbol == "TEST"
            @test config.num_filters == 50  # Default
            @test "TEST" in system.active_instruments
            
            # Load existing instrument
            config2 = load_or_create_instrument("TEST", system)
            @test config2.symbol == config.symbol
        end
        
        @testset "Migration from Legacy" begin
            # Create legacy TOML file
            legacy_path = joinpath(test_dir, "legacy.toml")
            legacy_config = Dict(
                "filters" => Dict(
                    "3" => Dict("q_factor" => 2.5, "batch_size" => 1500),
                    "5" => Dict("q_factor" => 2.0, "batch_size" => 1000),
                    "8" => Dict("q_factor" => 2.2, "batch_size" => 1200)
                )
            )
            
            open(legacy_path, "w") do io
                TOML.print(io, legacy_config)
            end
            
            # Perform migration
            master_path = joinpath(test_dir, "data/master_config.toml")
            system = InstrumentGASystem(master_path)
            
            success = migrate_from_legacy_config(legacy_path, "MIGRATED", system)
            @test success == true
            
            # Check migrated instrument
            @test haskey(system.instruments, "MIGRATED")
            config = system.instruments["MIGRATED"]
            @test config.num_filters == 3
            @test config.fibonacci_periods == Int32[3, 5, 8]
        end
        
        @testset "System Report Generation" begin
            master_path = joinpath(test_dir, "data/master_config.toml")
            system = initialize_ga_system(master_path)
            
            # Add a test instrument
            config = InstrumentConfig(
                symbol = "REPORT_TEST",
                num_filters = Int32(10),
                population_size = Int32(20),
                parameter_path = joinpath(test_dir, "REPORT_TEST/parameters/active.jld2"),
                ga_workspace_path = joinpath(test_dir, "REPORT_TEST/ga_workspace/"),
                config_path = joinpath(test_dir, "REPORT_TEST/config.toml"),
                fibonacci_periods = Int32[1, 2, 3]
            )
            add_instrument!(system, config)
            
            report = generate_system_report(system)
            
            @test haskey(report, "timestamp")
            @test haskey(report, "instruments")
            @test haskey(report["instruments"], "REPORT_TEST")
            @test report["instruments"]["REPORT_TEST"]["num_filters"] == 10
            @test haskey(report, "total_memory_mb")
        end
        
    finally
        cleanup_test_dir(test_dir)
    end
end

# =============================================================================
# INTEGRATION TESTS
# =============================================================================

@testset "Integration Tests" begin
    test_dir = setup_test_dir()
    
    try
        @testset "Complete Workflow" begin
            # 1. Initialize system
            master_path = joinpath(test_dir, "data/master_config.toml")
            mkpath(dirname(master_path))
            system = initialize_ga_system(master_path)
            
            # 2. Add an instrument
            config = InstrumentConfig(
                symbol = "WORKFLOW",
                num_filters = Int32(5),
                population_size = Int32(10),
                parameter_path = joinpath(test_dir, "data/WORKFLOW/parameters/active.jld2"),
                ga_workspace_path = joinpath(test_dir, "data/WORKFLOW/ga_workspace/"),
                config_path = joinpath(test_dir, "data/WORKFLOW/config.toml"),
                fibonacci_periods = Int32[1, 2, 3, 5, 8]
            )
            
            @test add_instrument!(system, config) == true
            
            # 3. Switch to instrument
            @test switch_instrument!(system, "WORKFLOW") == true
            
            # 4. Initialize storage
            storage = initialize_storage(config)
            @test size(storage.active_params) == (5, 13)
            
            # 5. Modify parameters
            for i in 1:5
                params = randn(Float32, 13)
                set_active_parameters!(storage, Int32(i), params)
            end
            
            # 6. Create checkpoint
            checkpoint = create_checkpoint(storage, Int32(50), 0.85f0)
            @test isfile(checkpoint)
            
            # 7. Save system
            save_master_config(system)
            save_instrument_config(config)
            
            # 8. Verify everything persisted
            @test isfile(master_path)
            @test isfile(config.config_path)
            @test isfile(storage.jld2_path)
        end
        
    finally
        cleanup_test_dir(test_dir)
    end
end

println("\n✅ All tests completed successfully!")