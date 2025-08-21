module ComplexBiquadGA

# External Dependencies
using Dates, Parameters, TOML, JLD2, Printf, Statistics, LinearAlgebra, Random, DSP, FFTW, DataFrames, CircularArrays, ProgressMeter

#== Correct Include Order ==#

# --- Core Types ---
include("GATypes.jl")

# --- Primitives ---
include("ParameterEncoding.jl")
include("PopulationInit.jl")
include("GeneticOperators.jl")

# --- GA Controllers & Storage ---
# CORRECTED ORDER: StorageSystem must be loaded BEFORE FilterBankGA
include("StorageSystem.jl") 
include("SingleFilterGA.jl")
include("FilterBankGA.jl")

# --- Configuration ---
include("InstrumentManager.jl")
include("ConfigurationLoader.jl")
include("ModernConfigSystem.jl")

# --- Filter & Fitness (Chunk 3) ---
include("ProductionFilterBank.jl")
include("SignalMetrics.jl")
include("FilterIntegration.jl")
include("FitnessEvaluation.jl")
include("GAFitnessBridge.jl")

# --- Prediction (Chunk 4) ---
include("PredictionMetrics.jl")
include("WeightedPrediction.jl")

# --- Supporting Modules ---
# Note: These are not submodules, so we don't use a leading dot.
include("core/SyntheticSignalGenerator.jl")
include("core/TickHotLoopF32.jl")


#== Public API Exports ==#

# --- Core GA Types (from GATypes.jl) ---
export InstrumentConfig, InstrumentGASystem, WriteThruStorage, FilterDefaults,
       GAParameters, ParameterRanges,
       validate_instrument_config, PeriodOverride

# --- GA Controller Types ---
export SingleFilterGAComplete, FilterBankGAComplete

# --- Weight & Prediction Types (from WeightedPrediction.jl) ---
export WeightSet, PredictionSystem, StreamingPredictor, HorizonPrediction,
       initialize_weights_rms, optimize_weights, evaluate_weight_fitness,
       create_prediction_system, create_streaming_predictor, process_tick!

# --- High-level Functions (from ConfigurationLoader.jl) ---
export initialize_ga_system, create_default_configs, load_or_create_instrument

# --- Modern Configuration System (from ModernConfigSystem.jl) ---
export FilterParameters, FilterBank, FilterConfig, ExtendedFilterConfig,
       load_filter_config, save_filter_config

end # module ComplexBiquadGA