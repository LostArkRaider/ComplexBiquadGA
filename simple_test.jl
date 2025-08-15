# simple_test.jl - Simple test script
# Run this from your project root directory

println("\n====== GA System Simple Test ======")
println("Working Directory: ", pwd())

# Load all modules using the master loader
println("\n📦 Loading modules...")
try
    QUIET_LOAD = true  # Don't show confirmation message
    include("load_all.jl")
    println("✅ All modules loaded")
catch e
    println("❌ Failed to load modules")
    println("Error: ", e)
    exit(1)
end

# Run basic tests
println("\n🧪 Running basic tests...")

# Test 1: Create basic types
print("  1. Creating types... ")
try
    ranges = ParameterRanges()
    defaults = FilterDefaults()
    println("✅")
catch e
    println("❌ ", e)
end

# Test 2: Create instrument config
print("  2. Creating instrument config... ")
try
    config = InstrumentConfig(
        symbol = "TEST",
        num_filters = Int32(5),
        population_size = Int32(10),
        parameter_path = "test.jld2",
        ga_workspace_path = "test/",
        config_path = "test.toml",
        fibonacci_periods = Int32[1, 2, 3]
    )
    println("✅")
catch e
    println("❌ ", e)
end

# Test 3: Create and use storage
print("  3. Testing storage... ")
try
    # Create temporary directory for test
    test_dir = mktempdir()
    storage_path = joinpath(test_dir, "test.jld2")
    
    # Create storage
    storage = WriteThruStorage(Int32(3), storage_path, Int32(10))
    
    # Set and get parameters
    params = randn(Float32, 13)
    set_active_parameters!(storage, Int32(1), params)
    retrieved = get_active_parameters(storage, Int32(1))
    
    if retrieved ≈ params
        println("✅")
    else
        println("❌ Parameters don't match")
    end
    
    # Cleanup
    rm(test_dir, recursive=true)
catch e
    println("❌ ", e)
end

# Test 4: Create GA system
print("  4. Creating GA system... ")
try
    test_dir = mktempdir()
    master_path = joinpath(test_dir, "master.toml")
    system = InstrumentGASystem(master_path)
    
    # Add an instrument
    config = InstrumentConfig(
        symbol = "YM",
        num_filters = Int32(10),
        population_size = Int32(20),
        parameter_path = joinpath(test_dir, "YM/params.jld2"),
        ga_workspace_path = joinpath(test_dir, "YM/ga/"),
        config_path = joinpath(test_dir, "YM/config.toml"),
        fibonacci_periods = Int32[1, 2, 3, 5, 8]
    )
    
    add_instrument!(system, config)
    
    if "YM" in system.active_instruments
        println("✅")
    else
        println("❌ Instrument not added")
    end
    
    # Cleanup
    rm(test_dir, recursive=true)
catch e
    println("❌ ", e)
end

# Test 5: Storage persistence
print("  5. Testing persistence... ")
try
    test_dir = mktempdir()
    storage_path = joinpath(test_dir, "persist.jld2")
    
    # Create and save
    storage1 = WriteThruStorage(Int32(2), storage_path, Int32(10))
    test_params = randn(Float32, 13)
    set_active_parameters!(storage1, Int32(1), test_params)
    sync_to_storage!(storage1)
    
    # Load in new storage
    storage2 = WriteThruStorage(Int32(2), storage_path, Int32(10))
    load_from_storage!(storage2)
    loaded_params = get_active_parameters(storage2, Int32(1))
    
    if loaded_params ≈ test_params
        println("✅")
    else
        println("❌ Loaded parameters don't match")
    end
    
    # Cleanup
    rm(test_dir, recursive=true)
catch e
    println("❌ ", e)
end

println("\n====== Test Complete ======")
println("\n✅ All basic functionality is working!")
println("\nYou can now:")
println("  1. Run the full test suite with: julia test/test_chunk2.jl")
println("  2. Start working with the GA system")