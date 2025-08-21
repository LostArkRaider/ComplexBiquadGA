using ComplexBiquadGA
using Test
using Random

function run_all_tests()
    @testset "Chunk 1: Core GA Infrastructure" begin
        println("\nRunning Chunk 1 tests...")
        include("test_chunk1.jl")
    end

    @testset "Chunk 2: Multi-Instrument & Storage" begin
        println("\nRunning Chunk 2 tests...")
        include("test_chunk2.jl")
    end

    @testset "Chunk 3: Filter Fitness Evaluation" begin
        println("\nRunning Chunk 3 tests...")
        include("test_chunk3.jl")
    end

    @testset "Chunk 4: Weight Optimization & Prediction" begin
        println("\nRunning Chunk 4 tests...")
        include("test_chunk4.jl")
    end

    @testset "Supporting Module: Synthetic Signal Generator" begin
        println("\nRunning Synthetic Signal Generator tests...")
        include("test_synthetic_signal_generator.jl")
    end
    
    @testset "Supporting Module: TickHotLoop Conversion" begin
        println("\nRunning TickHotLoop Conversion tests...")
        include("test_tickhotloop_conversion.jl")
    end
end

# This is the key: the logic inside this block only runs
# when you are in an interactive Julia session (REPL).
if isinteractive()
    println("Select test(s) to run:")
    println("  [1] Chunk 1: Core GA Infrastructure")
    println("  [2] Chunk 2: Multi-Instrument & Storage")
    println("  [3] Chunk 3: Filter Fitness Evaluation")
    println("  [4] Chunk 4: Weight Optimization & Prediction")
    println("  [A] All Tests (Default)")
    
    print("Enter selection: ")
    choice = uppercase(strip(readline()))
    
    println("-"^50)

    if choice == "1"
        @testset "Chunk 1 Tests" begin include("test_chunk1.jl") end
    elseif choice == "2"
        @testset "Chunk 2 Tests" begin include("test_chunk2.jl") end
    elseif choice == "3"
        @testset "Chunk 3 Tests" begin include("test_chunk3.jl") end
    elseif choice == "4"
        @testset "Chunk 4 Tests" begin include("test_chunk4.jl") end
    else
        println("Running all tests...")
        run_all_tests()
    end
    
    println("-"^50)
    println("Test run complete.")

# When run by an automated process (like `pkg> test`),
# `isinteractive()` is false, and this block will be executed.
else
    run_all_tests()
end