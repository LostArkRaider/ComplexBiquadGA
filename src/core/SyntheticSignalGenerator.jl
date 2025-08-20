# test_synthetic_signal_generator.jl
# Interactive test script for SyntheticSignalGenerator module
# Tests real-valued signal generation

println("\n" * "="^80)
println("SYNTHETIC SIGNAL GENERATOR - INTERACTIVE TEST SUITE")
println("Testing Real-Valued Signal Generation")
println("="^80 * "\n")

# ============================================================================
# SETUP AND IMPORTS
# ============================================================================

println("📦 Loading required packages...")
using Pkg

# Check and install required packages
required_packages = ["Plots", "Statistics", "FFTW", "DataFrames", "Dates", "Random", "Test"]
for pkg in required_packages
    if !haskey(Pkg.project().dependencies, pkg)
        println("  Installing $pkg...")
        Pkg.add(pkg)
    end
end

using Plots
using Statistics
using FFTW
using DataFrames
using Dates
using Random
using Test

# Load the SyntheticSignalGenerator module
println("📂 Loading SyntheticSignalGenerator module...")
include("../src/core/SyntheticSignalGenerator.jl")
using .SyntheticSignalGenerator

println("✅ Setup complete!\n")

# ============================================================================
# TEST UTILITIES
# ============================================================================

# Color codes for terminal output
const GREEN = "\033[32m"
const RED = "\033[31m"
const YELLOW = "\033[33m"
const RESET = "\033[0m"

function wait_for_user()
    println("\n$(YELLOW)Press Enter to continue...$(RESET)")
    readline()
end

function ask_yes_no(prompt::String)
    while true
        print("$(YELLOW)$prompt (y/n)? $(RESET)")
        response = lowercase(readline())
        if response == "y"
            return true
        elseif response == "n"
            return false
        else
            println("$(RED)Invalid input. Please enter 'y' or 'n'.$(RESET)")
        end
    end
end

# ============================================================================
# TEST SUITE
# ============================================================================

function run_all_tests()
    println("Running all tests...")
    
    # Define Fibonacci numbers for testing
    fibonacci_numbers = Int32[3, 5, 8, 13, 21, 34]
    
    @testset "Signal Generation Tests" begin
        # Test pure sine wave generation
        @testset "Pure Sine Wave" begin
            fib_num = Int32(13)
            signal = generate_synthetic_signal(
                n_bars = 100,
                ticks_per_bar = Int64(fib_num),
                signal_type = :pure_sine,
                signal_params = SyntheticSignalGenerator.SignalParams(100.0, Float64(fib_num))
            )
            @test length(signal) == 100 * fib_num
            @test eltype(signal) == Float32
            # Check if the signal is approximately sinusoidal
            # The exact peak may not be sampled due to discrete ticks, so we use a wider tolerance
            @test maximum(signal) ≈ 100.0f0 atol=1.0f0
            @test minimum(signal) ≈ -100.0f0 atol=1.0f0
        end
        
        # Test Fibonacci mixture signal generation
        @testset "Fibonacci Mixture" begin
            params = FibonacciMixtureParams(fibonacci_numbers)
            signal = generate_synthetic_signal(
                n_bars = 10,
                ticks_per_bar = Int64(89),
                signal_type = :fibonacci_mixture,
                signal_params = params
            )
            @test length(signal) == 10 * 89
            @test eltype(signal) == Float32
            # Check if the mixture is a sum of its parts
            @test sum(signal) != 0.0 # Just a sanity check
        end
        
        # Test the main test signal generation function
        @testset "generate_test_signals_complex_iq" begin
            # Using an n_ticks value that is a multiple of most of the fibonacci numbers to prevent InexactError
            signals_and_keys = generate_test_signals_complex_iq(fibonacci_numbers, n_ticks=54600)
            
            # Now test the properties of the returned struct
            @test typeof(signals_and_keys) == TestSignalSet
            @test length(signals_and_keys.signals) == length(fibonacci_numbers) * 3 + 1
            @test length(signals_and_keys.keys) == length(fibonacci_numbers) * 3 + 1
            
            # Check that the combined signal exists and has the correct type
            @test "fib_combined_mixture" in signals_and_keys.keys
            combined_signal_index = findfirst(isequal("fib_combined_mixture"), signals_and_keys.keys)
            combined_signal = signals_and_keys.signals[combined_signal_index]
            @test eltype(combined_signal) == ComplexF32
        end
    end
end

function interactive_menu()
    println("\n$(GREEN)Interactive Menu:$(RESET)")
    println("1. Run all automated tests")
    println("2. Exit")
    
    while true
        print("\n$(YELLOW)Enter your choice (1-2): $(RESET)")
        choice = readline()
        
        if choice == "1"
            run_all_tests()
            wait_for_user()
        elseif choice == "2"
            println("\n$(GREEN)Exiting test suite. Goodbye!$(RESET)")
            break
        else
            println("$(RED)Invalid option. Please select 1 or 2.$(RESET)")
        end
    end
end

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

println("$(GREEN)Welcome to the SyntheticSignalGenerator Test Suite!$(RESET)")
println("\nThis test script will now run automated tests on the signal generator.")

if ask_yes_no("\nWould you like to use the interactive menu?")
    interactive_menu()
else
    run_all_tests()
end