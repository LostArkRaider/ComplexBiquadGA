# diagnostic.jl - Quick diagnostic script to check system setup
# Run this from your project root directory

println("\n" * "="^60)
println("GA OPTIMIZATION SYSTEM - DIAGNOSTIC CHECK")
println("="^60)

# Check Julia version
println("\n📌 Julia Version: ", VERSION)
if VERSION < v"1.8"
    println("  ⚠️ Warning: Julia 1.8+ recommended")
else
    println("  ✅ Julia version OK")
end

# Check working directory (should be project root)
println("\n📁 Working Directory: ", pwd())

# Check for Project.toml in current directory
if !isfile("Project.toml")
    println("  ⚠️ Project.toml not found - make sure you're in the project root directory!")
    return  # Don't exit, just return
end

# Check for source files (paths relative to project root)
println("\n📦 Checking for source files:")
required_files = [
    "src/GATypes.jl",
    "src/InstrumentManager.jl",
    "src/StorageSystem.jl",
    "src/ConfigurationLoader.jl",
    "load_all.jl"
]

missing_files = String[]
for file in required_files
    if isfile(file)
        println("  ✅ $file")
    else
        println("  ❌ $file - MISSING")
        push!(missing_files, file)
    end
end

if !isempty(missing_files)
    println("\n⚠️ Missing files detected!")
    println("Please ensure all files are present.")
    return  # Don't exit, just return
end

# Check for required packages
println("\n📚 Checking required packages:")
using Pkg

required_packages = ["TOML", "JLD2", "Parameters", "Dates", "Test"]
missing_packages = String[]

for pkg_name in required_packages
    try
        eval(Meta.parse("using $pkg_name"))
        println("  ✅ $pkg_name available")
    catch
        println("  ❌ $pkg_name - NOT INSTALLED")
        push!(missing_packages, pkg_name)
    end
end

if !isempty(missing_packages)
    println("\n📥 Installing missing packages...")
    for pkg in missing_packages
        println("  Installing $pkg...")
        Pkg.add(pkg)
    end
    println("\n✅ Packages installed! Please run the diagnostic again.")
    return  # Don't exit, just return
end

# Load all modules using the master loader
println("\n🔧 Loading all modules...")
module_load_success = false  # Initialize the variable
try
    # When running from include("scripts/diagnostic.jl"), we need to go up one level
    # Check if we can find load_all.jl in current dir or parent dir
    load_path = ""
    if isfile("load_all.jl")
        load_path = "load_all.jl"
    elseif isfile("../load_all.jl")
        load_path = "../load_all.jl"
    else
        error("Cannot find load_all.jl")
    end
    
    include(load_path)
    module_load_success = true  # Set to true if successful
    println("  ✅ All modules loaded successfully")
catch e
    println("  ❌ Failed to load modules")
    println("  Error: ", e)
    println("\n  Debug info:")
    println("    Current working directory: ", pwd())
    println("    Script location: ", @__DIR__)
    println("    load_all.jl in pwd: ", isfile("load_all.jl"))
    println("    load_all.jl in parent: ", isfile("../load_all.jl"))
    # Don't exit! Just return so REPL stays alive
    return
end

# Quick functionality test
println("\n🧪 Quick functionality test:")

try
    # Create basic types
    print("  Creating basic types... ")
    ranges = ParameterRanges()
    defaults = FilterDefaults()
    println("✅")
    
    # Create instrument config
    print("  Creating instrument config... ")
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
    
    # Create storage
    print("  Creating storage... ")
    storage = WriteThruStorage(Int32(5), "test.jld2", Int32(10))
    println("✅")
    
    # Test parameter operations
    print("  Testing parameter operations... ")
    params = randn(Float32, 13)
    set_active_parameters!(storage, Int32(1), params)
    retrieved = get_active_parameters(storage, Int32(1))
    if retrieved ≈ params
        println("✅")
    else
        println("❌ Parameter mismatch")
    end
    
    println("\n✅ All functionality tests passed!")
    
catch e
    println("❌")
    println("  Error: ", e)
end

# System info
println("\n💻 System Information:")
println("  OS: ", Sys.KERNEL)
println("  CPU Threads: ", Threads.nthreads())
println("  Julia Threads: ", Threads.nthreads())

# Final status
println("\n" * "="^60)
println("DIAGNOSTIC COMPLETE")
println("="^60)

println("\n✅ System is ready for testing!")
println("\nNext steps:")
println("  1. Run the full test suite: julia test/test_chunk2.jl")
println("  2. Start using the GA system with: include(\"load_all.jl\")")