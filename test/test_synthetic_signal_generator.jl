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

# Function to visualize a signal
function visualize_signal(signal::Vector{Float32}, title::String)
    plot_title = "Generated Signal: " * title
    p = plot(signal, label="Signal", title=plot_title, xlabel="Tick", ylabel="Amplitude", legend=false)
    display(p)
end

# ============================================================================
# TEST FUNCTIONS
# ============================================================================

function test_pure_sine_wave()
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
        @test maximum(signal) ≈ 100.0f0 atol=1.0f0
        @test minimum(signal) ≈ -100.0f0 atol=1.0f0
    end
end

function test_fibonacci_mixture()
    @testset "Fibonacci Mixture" begin
        fibonacci_numbers = Int32[3, 5, 8, 13, 21, 34]
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
        @test sum(signal) != 0.0
    end
end

function run_all_tests()
    println("Running all tests...")
    @testset "Signal Generation Tests" begin
        test_pure_sine_wave()
        test_fibonacci_mixture()
    end
end

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

function interactive_menu()
    println("\n$(GREEN)Interactive Menu:$(RESET)")
    println("1. Run all automated tests")
    println("2. Run Pure Sine Wave Test")
    println("3. Run Fibonacci Mixture Test")
    println("4. Display Pure Sine Wave Plot")
    println("5. Display Fibonacci Mixture Plot")
    println("6. Exit")

    while true
        print("\n$(YELLOW)Enter your choice (1-6): $(RESET)")
        choice = readline()
        
        if choice == "1"
            run_all_tests()
        elseif choice == "2"
            test_pure_sine_wave()
        elseif choice == "3"
            test_fibonacci_mixture()
        elseif choice == "4"
            fib_num = Int32(13)
            signal = generate_synthetic_signal(
                n_bars = 100,
                ticks_per_bar = Int64(fib_num),
                signal_type = :pure_sine,
                signal_params = SyntheticSignalGenerator.SignalParams(100.0, Float64(fib_num))
            )
            # Visualize a portion of the signal, up to 1000 ticks
            visualize_signal(signal[1:min(length(signal), 1000)], "Pure Sine Wave")
        elseif choice == "5"
            fibonacci_numbers = Int32[3, 5, 8, 13, 21, 34]
            params = FibonacciMixtureParams(fibonacci_numbers)
            signal = generate_synthetic_signal(
                n_bars = 10,
                ticks_per_bar = Int64(89),
                signal_type = :fibonacci_mixture,
                signal_params = params
            )
            # Visualize a portion of the signal, up to 1000 ticks
            visualize_signal(signal[1:min(length(signal), 1000)], "Fibonacci Mixture Signal")
        elseif choice == "6"
            println("\n$(GREEN)Exiting test suite. Goodbye!$(RESET)")
            break
        else
            println("$(RED)Invalid option. Please select 1-6.$(RESET)")
        end
        
        # Display the menu again after each action
        println("\n" * "="^80)
        println("SYNTHETIC SIGNAL GENERATOR - INTERACTIVE TEST SUITE")
        println("Testing Real-Valued Signal Generation")
        println("="^80 * "\n")
        println("\n$(GREEN)Interactive Menu:$(RESET)")
        println("1. Run all automated tests")
        println("2. Run Pure Sine Wave Test")
        println("3. Run Fibonacci Mixture Test")
        println("4. Display Pure Sine Wave Plot")
        println("5. Display Fibonacci Mixture Plot")
        println("6. Exit")
        
    end
end

println("$(GREEN)Welcome to the SyntheticSignalGenerator Test Suite!$(RESET)")
println("\nThis test script will now run automated tests on the signal generator.")

if ask_yes_no("\nWould you like to use the interactive menu?")
    interactive_menu()
else
    run_all_tests()
end