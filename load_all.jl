# load_all.jl - Master module loader for GA Optimization System
# Chunk 4 Testing Version - Includes all required modules in correct dependency order
# Must be run from project root directory

# Verify we're in project root
if !isfile("Project.toml")
    error("Not in project root! Please run from the directory containing Project.toml")
end

# ============================================================================
# LOAD REQUIRED PACKAGES
# ============================================================================

println("Loading required packages...")

using Pkg

# Add required packages if not already present
required_packages = [
    "TOML",
    "JLD2",
    "Parameters",
    "Dates",
    "Printf",
    "Statistics",
    "LinearAlgebra",
    "Random",
    "DSP",
    "FFTW",
    "DataFrames",
    "CircularArrays",
    "ProgressMeter"
]

for pkg in required_packages
    if !haskey(Pkg.project().dependencies, pkg)
        println("  Adding package: $pkg")
        Pkg.add(pkg)
    end
end

# Load all packages
using TOML
using JLD2
using Parameters
using Dates
using Printf
using Statistics
using LinearAlgebra
using Random
using DSP
using FFTW
using DataFrames
using CircularArrays
using ProgressMeter

# ============================================================================
# MODULE LOADING WITH DEPENDENCY ORDER
# ============================================================================

println("\nLoading GA Optimization System modules...")
println("="^60)

# Track loaded modules
loaded_modules = String[]

# Helper function to load a module
function load_module(name::String, path::String)
    if !isdefined(Main, Symbol(name))
        if !isfile(path)
            @warn "Module file not found: $path"
            return false
        end
        try
            include(path)
            push!(loaded_modules, name)
            println("  ✅ Loaded: $name")
            return true
        catch e
            @error "Failed to load $name: $e"
            return false
        end
    else
        println("  ✓ Already loaded: $name")
        push!(loaded_modules, name)
        return true
    end
end

# ============================================================================
# CHUNK 1-2: CORE GA INFRASTRUCTURE
# ============================================================================

println("\nChunk 1-2: Core GA Infrastructure")
println("-"^40)

# Core types must be loaded first
load_module("GATypes", "src/GATypes.jl")
using Main.GATypes

# Parameter encoding/decoding
load_module("ParameterEncoding", "src/ParameterEncoding.jl")
using Main.ParameterEncoding

# Management modules
load_module("InstrumentManager", "src/InstrumentManager.jl")
using Main.InstrumentManager

load_module("StorageSystem", "src/StorageSystem.jl")
using Main.StorageSystem

load_module("ConfigurationLoader", "src/ConfigurationLoader.jl")
using Main.ConfigurationLoader

# GA Operations
load_module("GeneticOperators", "src/GeneticOperators.jl")
using Main.GeneticOperators

load_module("PopulationInit", "src/PopulationInit.jl")
using Main.PopulationInit

# ============================================================================
# CHUNK 3: FILTER CONFIGURATION AND FITNESS
# ============================================================================

println("\nChunk 3: Filter Configuration and Fitness")
println("-"^40)

# Configuration system
load_module("ModernConfigSystem", "src/core/ModernConfigSystem.jl")
using Main.ModernConfigSystem

# Filter bank implementation
load_module("ProductionFilterBank", "src/core/ProductionFilterBank.jl")
using Main.ProductionFilterBank

# Filter integration bridge
load_module("FilterIntegration", "src/FilterIntegration.jl")
using Main.FilterIntegration

# Signal metrics
load_module("SignalMetrics", "src/SignalMetrics.jl")
using Main.SignalMetrics

# Fitness evaluation
load_module("FitnessEvaluation", "src/FitnessEvaluation.jl")
using Main.FitnessEvaluation

# GA-Fitness bridge
load_module("GAFitnessBridge", "src/GAFitnessBridge.jl")
using Main.GAFitnessBridge

# ============================================================================
# CHUNK 4: WEIGHT OPTIMIZATION AND PREDICTION
# ============================================================================

println("\nChunk 4: Weight Optimization and Prediction")
println("-"^40)

# Prediction metrics
load_module("PredictionMetrics", "src/PredictionMetrics.jl")
using Main.PredictionMetrics

# Unified weighted prediction module (merged from WeightOptimization + PricePrediction)
load_module("WeightedPrediction", "src/WeightedPrediction.jl")
using Main.WeightedPrediction

# ============================================================================
# SUPPORTING MODULES
# ============================================================================

println("\nSupporting Modules")
println("-"^40)

# Synthetic signal generation
load_module("SyntheticSignalGenerator", "src/core/SyntheticSignalGenerator.jl")
using Main.SyntheticSignalGenerator

# Market data processing
load_module("TickHotLoopF32", "src/core/TickHotLoopF32.jl")
using Main.TickHotLoopF32

# ============================================================================
# EXPORT COMMONLY USED TYPES AND FUNCTIONS
# ============================================================================

# This makes them available without module prefix
const IGASystem = Main.GATypes.InstrumentGASystem
const IConfig = Main.GATypes.InstrumentConfig
const WTStorage = Main.GATypes.WriteThruStorage

# Chunk 4 specific exports for convenience
const WeightSet = Main.WeightedPrediction.WeightSet
const PredictionSystem = Main.WeightedPrediction.PredictionSystem
const StreamingPredictor = Main.WeightedPrediction.StreamingPredictor

# ============================================================================
# CONFIRMATION MESSAGE
# ============================================================================

# Can be disabled by setting QUIET_LOAD=true before including
if !@isdefined(QUIET_LOAD) || !QUIET_LOAD
    println("\n" * "="^60)
    println("✅ GA Optimization System loaded successfully!")
    println("="^60)
    println("\nLoaded modules ($(length(loaded_modules)) total):")
    
    # Group modules by chunk
    chunk1_2 = ["GATypes", "ParameterEncoding", "InstrumentManager", "StorageSystem", 
                "ConfigurationLoader", "GeneticOperators", "PopulationInit"]
    chunk3 = ["ModernConfigSystem", "ProductionFilterBank", "FilterIntegration", 
              "SignalMetrics", "FitnessEvaluation", "GAFitnessBridge"]
    chunk4 = ["PredictionMetrics", "WeightedPrediction"]
    supporting = ["SyntheticSignalGenerator", "TickHotLoopF32"]
    
    println("\n  Chunks 1-2 (Core GA):")
    for m in chunk1_2
        if m in loaded_modules
            println("    • $m")
        end
    end
    
    println("\n  Chunk 3 (Filter Fitness):")
    for m in chunk3
        if m in loaded_modules
            println("    • $m")
        end
    end
    
    println("\n  Chunk 4 (Prediction):")
    for m in chunk4
        if m in loaded_modules
            println("    • $m")
        end
    end
    
    println("\n  Supporting:")
    for m in supporting
        if m in loaded_modules
            println("    • $m")
        end
    end
    
    println("\nReady for Chunk 4 testing!")
    println("Run: include(\"tests/test_chunk4.jl\")")
end