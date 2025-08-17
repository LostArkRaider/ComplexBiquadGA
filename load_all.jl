# load_all.jl - Master module loader for GA Optimization System
# Include this at the start of any script or test to load all modules in correct order
# Must be run from project root directory

# Verify we're in project root
if !isfile("Project.toml")
    error("Not in project root! Please run from the directory containing Project.toml")
end

# Load required packages
using TOML
using JLD2
using Parameters
using Dates
using Printf

# Module loading with dependency order
# GATypes has no dependencies, load first
if !isdefined(Main, :GATypes)
    include("src/GATypes.jl")
end
using Main.GATypes

# InstrumentManager depends on GATypes
if !isdefined(Main, :InstrumentManager)
    include("src/InstrumentManager.jl")
end
using Main.InstrumentManager

# StorageSystem depends on GATypes
if !isdefined(Main, :StorageSystem)
    include("src/StorageSystem.jl")
end
using Main.StorageSystem

# ConfigurationLoader depends on all above
if !isdefined(Main, :ConfigurationLoader)
    include("src/ConfigurationLoader.jl")
end
using Main.ConfigurationLoader

# Export commonly used types and functions for convenience
# This makes them available without module prefix
const IGASystem = Main.GATypes.InstrumentGASystem
const IConfig = Main.GATypes.InstrumentConfig
const WTStorage = Main.GATypes.WriteThruStorage

# Confirmation message (can be disabled by setting QUIET_LOAD=true before including)
if !@isdefined(QUIET_LOAD) || !QUIET_LOAD
    println("✅ GA Optimization System modules loaded successfully")
    println("   Available modules: GATypes, InstrumentManager, StorageSystem, ConfigurationLoader")
end