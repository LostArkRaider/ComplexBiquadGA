# GA Optimization System for ComplexBiquad PLL Filter Bank - Specification v0.3

## Revision History
- **v0.3**: Added parameter type specifications, per-filter configuration support, filter registry system, removed all dictionary-based implementations
- **v0.2**: Initial MVP specification

---

## 1. Executive Overview

### 1.1 Project Purpose
This specification defines a Genetic Algorithm (GA) optimization system for a ComplexBiquad PLL filter bank that processes real YM futures tick data. The system optimizes both filter parameters and prediction weights to forecast price changes at future tick indices.

### 1.2 Core Innovation
The system treats market data as superposed rotating phasors extracted by Fibonacci-period filters. Each filter output represents a complex rotating vector that can be extrapolated to future time points. By optimizing real-valued weights that combine these phasor predictions while preserving their phase information, the system achieves accurate long-range price forecasting.

### 1.3 Two-Stage Optimization Architecture

**Stage 1: Filter/PLL Parameter Optimization**
- Optimizes Q factors, PLL gains, loop bandwidths, lock thresholds
- **Each filter has independent parameters** (12 parameters per filter)
- Goal: Extract clean, stable rotating phasors from noisy market data
- Uses modified ExtendedFilterConfig with per-filter parameter arrays
- Fitness based on signal quality metrics (SNR, lock quality, frequency selectivity)

**Stage 2: Prediction Weight Optimization**
- Optimizes real-valued weights for combining filter outputs
- Weights preserve phase information (magnitude scaling only)
- Different weights for different prediction horizons
- Goal: Accurate price change predictions at 100-2000+ tick horizons

### 1.4 Key Technical Constraints

1. **Real-Valued Weights**: Weights must be real numbers to preserve phase information extracted by filters
2. **Phase Preservation**: The temporal relationships encoded in filter phases are critical and must not be altered
3. **Long-Range Focus**: Predictions target hundreds to thousands of ticks ahead, not short-range (1-5 tick) forecasts
4. **Real Data Primary**: System uses TickHotLoopF32 to process actual YM tick files; synthetic signals for validation only
5. **GPU-Ready Design**: Architecture supports future GPU acceleration though MVP is CPU-based
6. **Per-Filter Independence**: Each filter maintains completely independent parameter sets
7. **Configurable Filter Count**: Number of filters is dynamic with auto-generation of defaults
8. **No Dictionaries**: All runtime data structures use direct struct access for performance

### 1.5 Existing Codebase Integration

The system leverages four existing modules:
- **TickHotLoopF32.jl**: Ultra-low-latency tick processing to ComplexF32 signals
- **ProductionFilterBank.jl**: ComplexBiquad and PLLFilterState implementations
- **ModernConfigSystem.jl**: Type-safe TOML-based configuration management (modified for per-filter params)
- **SyntheticSignalGenerator.jl**: Test signal generation for validation only

### 1.6 Mathematical Foundation

Each filter i produces a complex phasor that evolves as:
```
z_i(t+Δt) = |z_i(t)| * exp(i*(φ_i(t) + ω_i*Δt))
```

Price change prediction via weighted vector sum:
```
price_prediction(t+Δt) = Real(Σ w_i * z_i(t+Δt))
```
where w_i ∈ ℝ (real weights preserve phase)

---

## 2. Parameter Type Specifications (NEW in v0.3)

### 2.1 Parameter Types and Scaling

The GA system recognizes four distinct parameter types, each with specific encoding/decoding strategies:

| Parameter Type | Description | Gene Encoding | Mutation Strategy |
|---------------|-------------|---------------|-------------------|
| **LINEAR** | Direct linear mapping | [0,1] → [min,max] | Gaussian noise |
| **LOGARITHMIC** | Exponential scaling | [0,1] → log space | Gaussian noise |
| **BINARY** | Boolean values | 0.5 threshold | Bit flip |
| **DISCRETE** | Enumerated options | Index mapping | Adjacent/random jump |

### 2.2 Complete Parameter Specification Table

| # | Parameter | Type | Scaling | Range/Options | Rationale |
|---|-----------|------|---------|---------------|-----------|
| 1 | **q_factor** | Float64 | LINEAR | [0.5, 10.0] | Linear response to bandwidth changes |
| 2 | **sma_window** | Int | LOGARITHMIC | [1, 200] | More resolution needed at lower values |
| 3 | **batch_size** | Int | LOGARITHMIC | [100, 5000] | Exponential performance impact |
| 4 | **phase_detector_gain** | Float64 | LOGARITHMIC | [0.001, 1.0] | 3 orders of magnitude range |
| 5 | **loop_bandwidth** | Float64 | LOGARITHMIC | [0.0001, 0.1] | 3 orders of magnitude range |
| 6 | **lock_threshold** | Float64 | LINEAR | [0.0, 1.0] | Direct probability mapping |
| 7 | **ring_decay** | Float64 | LINEAR | [0.9, 1.0] | Narrow range, linear response |
| 8 | **enable_clamping** | Bool | BINARY | {false, true} | Simple on/off switch |
| 9 | **clamping_threshold** | Float64 | LOGARITHMIC | [1e-8, 1e-3] | 5 orders of magnitude |
| 10 | **volume_scaling** | Float64 | LOGARITHMIC | [0.1, 10.0] | 2 orders of magnitude |
| 11 | **max_frequency_deviation** | Float64 | LINEAR | [0.01, 0.5] | Linear frequency response |
| 12 | **phase_error_history_length** | Int | DISCRETE | {5,10,15,20,30,40,50} | Specific buffer sizes only |

### 2.3 Encoding/Decoding Functions

```julia
# Linear scaling
linear_decode(gene, min, max) = min + gene * (max - min)
linear_encode(value, min, max) = (value - min) / (max - min)

# Logarithmic scaling
log_decode(gene, min, max) = min * (max/min)^gene
log_encode(value, min, max) = log(value/min) / log(max/min)

# Binary
binary_decode(gene) = gene >= 0.5
binary_encode(value) = value ? 1.0 : 0.0

# Discrete enumeration
discrete_decode(gene, options) = options[1 + floor(Int, gene * length(options) * 0.9999)]
discrete_encode(value, options) = (findfirst(==(value), options) - 1) / max(1, length(options) - 1)
```

---

## 3. Per-Filter Configuration System (NEW in v0.3)

### 3.1 Architecture Changes

**Previous Design**: Single set of parameters applied to all filters
**New Design**: Each filter maintains independent parameter sets with struct-based storage

### 3.2 Configuration Structure (NO DICTIONARIES)

```julia
# Filter Bank using direct struct access
mutable struct FilterBank
    filters::Vector{FilterParameters}      # All filter configurations
    active_mask::Vector{Bool}             # Which filters are active
    periods::Vector{Int}                  # Period for each filter slot
    num_slots::Int                        # Total allocated slots
    num_active::Int                       # Number of active filters
end

# Per-filter parameters
@with_kw struct FilterParameters
    period::Int                          # Fibonacci period
    q_factor::Float64                    # Parameter 1
    sma_window::Int                      # Parameter 2
    batch_size::Int                      # Parameter 3
    phase_detector_gain::Float64         # Parameter 4
    loop_bandwidth::Float64              # Parameter 5
    lock_threshold::Float64              # Parameter 6
    ring_decay::Float64                  # Parameter 7
    enable_clamping::Bool                # Parameter 8
    clamping_threshold::Float64          # Parameter 9
    volume_scaling::Float64              # Parameter 10
    max_frequency_deviation::Float64     # Parameter 11
    phase_error_history_length::Int      # Parameter 12
end
```

### 3.3 Filter Registry System

**Purpose**: Maintain configurations for all filters (active and inactive) without dictionaries

**Features**:
- Auto-generation of default configurations for new periods
- Retention of configurations for unused filters
- TOML persistence of complete registry
- Direct struct access for all operations

**Implementation**:
```julia
mutable struct FilterRegistry
    bank::FilterBank
    last_modified::DateTime
end

# Usage
registry = FilterRegistry(initial_periods, max_filters)
filter = get_filter_by_period(registry.bank, period)  # Linear search
set_filter_by_period!(registry.bank, params)          # Direct array update
```

---

## 4. Development Chunks

The project is divided into six independent mini-projects, each building on the previous while maintaining standalone functionality.

---

## 5. Chunk 1: Filter Parameter GA Core (COMPLETED)

### 5.1 Purpose
Implement the genetic algorithm infrastructure for optimizing filter and PLL parameters with comprehensive per-filter parameter tuning and type-aware encoding.

### 5.2 Implementation Status
✅ **COMPLETED** - FilterParameterGA.jl module implemented with:
- Full 12-parameter encoding per filter
- Type-aware mutation strategies
- Hybrid crossover (filter-level and parameter-level)
- Struct-based filter registry (no dictionaries)
- TOML persistence for GA state and configurations

### 5.3 Key Features Delivered

**Parameter Encoding System:**
- Mixed-type chromosome encoding (Float64, Int, Bool)
- Type-specific mutation strategies
- Logarithmic scaling for wide-range parameters
- Discrete enumeration for buffer sizes

**Genetic Operators:**
- Hybrid crossover strategy (Option C implemented)
- Uniform mutation rates with type-aware application
- Tournament selection with configurable pressure
- Elitism preservation (top 5-10%)

**Population Management:**
- Dynamic chromosome length based on filter count
- Batch evaluation structure for GPU readiness
- Convergence tracking across 108+ dimensions
- Checkpoint/resume capability

### 5.4 Chromosome Structure
- **Length**: `num_filters × 12` parameters
- **Encoding**: All genes normalized to [0,1] range
- **Example**: 9 Fibonacci filters = 108 parameters total
- **Organization**: Parameters for filter i at indices [(i-1)×12+1 : i×12]

### 5.5 Validation Results
- ✅ All 12 parameters per filter remain within valid ranges
- ✅ Type-specific mutations maintain parameter validity
- ✅ Crossover preserves parameter independence
- ✅ Population diversity maintained across evolution
- ✅ TOML serialization preserves full precision
- ✅ No dictionaries used in runtime data structures

---

## 6. Chunk 2: Filter Fitness Evaluation

### 6.1 Purpose
Implement fitness evaluation for filter parameters using real tick data, measuring signal extraction quality without requiring prediction accuracy.

### 6.2 Dependencies
- FilterParameterGA.jl (from Chunk 1) ✅
- ProductionFilterBank.jl (for filter processing)
- TickHotLoopF32.jl (for real tick data)

### 6.3 Deliverables

**FilterFitnessEvaluator.jl Module:**
- Signal quality metrics (SNR, spectral purity, stability)
- PLL lock quality assessment
- Frequency selectivity measurement
- Batch fitness evaluation pipeline
- Type-aware parameter validation

**Key Metrics:**
- Signal-to-noise ratio per filter
- PLL lock quality and stability over time
- Frequency response sharpness (Q factor effectiveness)
- Phase coherence and angular velocity stability
- Ringing suppression and transient response

### 6.4 Integration with Chunk 1
- Replace placeholder fitness function
- Utilize decode_chromosome for parameter extraction
- Batch evaluation of population chromosomes
- Cache fitness values for unchanged chromosomes

### 6.5 Validation
- Process sample tick data through filter configurations
- Verify fitness metrics correlate with signal quality
- Validate batch evaluation produces consistent results
- Test with both standard and PLL-enhanced configurations

### 6.6 Success Criteria
- Fitness function differentiates good from poor parameters
- Evaluation completes in reasonable time for large populations
- Metrics provide actionable feedback for GA evolution
- System handles real tick data robustly

---

## 7. Chunk 3: Phasor Tracking Module

### 7.1 Purpose
Extract and track magnitude, phase, and angular velocity from filter outputs, preparing for phasor-based prediction.

### 7.2 Dependencies
- Optimized filter configurations (from Chunks 1-2)
- ProductionFilterBank.jl (for filter outputs)
- TickHotLoopF32.jl (for continuous tick processing)

### 7.3 Deliverables

**PhasorTracker.jl Module:**
- Real-time magnitude extraction: |z(t)|
- Instantaneous phase tracking: φ(t) = angle(z(t))
- Angular velocity estimation: ω(t) = dφ/dt
- Phase unwrapping and continuity management
- Phasor stability detection

**Data Structures:**
- PhasorState: magnitude, phase, angular velocity per filter
- PhasorHistory: rolling window for stability analysis
- PhasorMetrics: quality indicators for each filter's phasor

### 7.4 Validation
- Verify phase unwrapping handles 2π discontinuities
- Validate angular velocity estimation accuracy
- Test phasor stability under various market conditions
- Confirm magnitude tracking follows envelope correctly

### 7.5 Success Criteria
- Smooth, continuous phase tracking without jumps
- Accurate angular velocity estimation
- Reliable detection of stable vs unstable phasors
- Efficient processing of tick streams

---

## 8. Chunk 4: Phasor Prediction Engine

### 8.1 Purpose
Implement phasor extrapolation to future tick indices and combine predictions using real-valued weights.

### 8.2 Dependencies
- PhasorTracker.jl (from Chunk 3)
- Filter configurations and outputs

### 8.3 Deliverables

**PhasorPredictor.jl Module:**
- Phasor extrapolation: z(t+Δt) = |z(t)| * exp(i*(φ(t) + ω*Δt))
- Multi-horizon prediction support (100, 500, 1000, 2000+ ticks)
- Real-weight application preserving phase
- Vector summation for price prediction
- Prediction confidence estimation

**Key Functions:**
- Extrapolate single phasor to future tick
- Apply real weights without phase corruption
- Combine weighted phasors into price prediction
- Handle missing or unstable phasors gracefully

### 8.4 Validation
- Verify phase preservation with real weights
- Test extrapolation accuracy on synthetic signals
- Validate vector sum produces reasonable price predictions
- Confirm predictions degrade gracefully with horizon

### 8.5 Success Criteria
- Accurate phasor extrapolation for stable signals
- Phase information preserved through weighting
- Reasonable price predictions at various horizons
- Computational efficiency for real-time use

---

## 9. Chunk 5: Weight Optimization GA

### 9.1 Purpose
Implement genetic algorithm for optimizing real-valued prediction weights at multiple horizons.

### 9.2 Dependencies
- PhasorPredictor.jl (from Chunk 4)
- FilterParameterGA.jl (for GA infrastructure)
- Historical tick data for training

### 9.3 Deliverables

**WeightOptimizationGA.jl Module:**
- Real-valued weight chromosome: W[num_filters, num_horizons]
- Prediction accuracy fitness function
- Multi-horizon optimization strategy
- Weight regularization options
- Convergence detection

**Optimization Features:**
- Separate weights for each prediction horizon
- Real-value constraint enforcement
- Adaptive mutation rates based on prediction error
- Elitism to preserve best weight sets
- Weight magnitude penalties to prevent overfitting

### 9.4 Validation
- Verify weights remain real-valued throughout evolution
- Test prediction accuracy improvement over generations
- Validate different horizons receive appropriate weights
- Confirm convergence to stable weight configurations

### 9.5 Success Criteria
- Prediction accuracy improves significantly over random weights
- Different horizons develop distinct weight patterns
- GA converges reliably within reasonable generations
- Weights generalize to out-of-sample data

---

## 10. Chunk 6: Integration and Horizon Discovery

### 10.1 Purpose
Integrate all components and empirically determine maximum useful prediction horizons through systematic testing.

### 10.2 Dependencies
- All previous chunks
- Extended historical tick data for validation

### 10.3 Deliverables

**PredictionSystem.jl Module:**
- Complete pipeline from ticks to predictions
- Horizon accuracy analysis tools
- Walk-forward validation framework
- Performance metrics and reporting
- Configuration management for optimal parameters

**Analysis Tools:**
- Accuracy vs horizon curves
- Prediction confidence intervals
- Error decomposition (magnitude vs phase errors)
- Market regime detection and adaptation
- Maximum useful horizon determination

### 10.4 Validation
- End-to-end testing on historical data
- Walk-forward analysis to prevent overfitting
- Comparison with baseline prediction methods
- Stress testing under various market conditions

### 10.5 Success Criteria
- Complete system processes tick data to predictions
- Clear identification of maximum useful prediction horizon
- Performance metrics meet practical trading requirements
- System maintains stability over extended operation

---

## 11. GPU-Ready Design Patterns

### 11.1 Architecture Principles
While GPU implementation is not part of MVP, the design incorporates patterns for future acceleration:

**Batch Processing:**
- Population-parallel GA evaluation
- Multiple chromosomes evaluated simultaneously
- Vectorized fitness calculations

**Memory Layout:**
- Coalesced access patterns for filter states
- Structure-of-arrays for phasor data
- Minimal host-device transfer requirements
- Direct struct access (no dictionary overhead)

**Computational Kernels:**
- Filter processing as parallel kernel per filter
- Phasor extrapolation as vectorized operation
- Weight application as matrix multiplication

### 11.2 CUDA.jl Preparation
- Data structures aligned for GPU memory
- Algorithms expressed as map/reduce operations
- Minimal branching in core loops
- Pre-allocated memory buffers
- Struct-based storage for efficient GPU transfer

---

## 12. Performance Targets

### 12.1 MVP Performance Goals
- Filter optimization: < 1 hour for 100 generations, population 50
- Weight optimization: < 30 minutes per horizon
- Tick processing: > 10,000 ticks/second
- Prediction generation: < 1ms per horizon

### 12.2 Accuracy Targets
- Signal extraction: SNR > 20dB for primary frequencies
- PLL lock quality: > 70% average across filters
- Prediction accuracy: Directional accuracy > 55% at 100-tick horizon
- Horizon discovery: Identify useful range within 5% accuracy

---

## 13. Risk Mitigation

### 13.1 Technical Risks
- **Filter instability**: Implement stability checks in GA constraints
- **Phase unwrapping errors**: Use robust unwrapping algorithms
- **Overfitting**: Employ regularization and walk-forward validation
- **Computational bottlenecks**: Profile and optimize critical paths

### 13.2 Data Risks
- **Tick data quality**: Implement robust cleaning in TickHotLoopF32
- **Market regime changes**: Detect and adapt to regime shifts
- **Sparse data periods**: Handle low-activity periods gracefully

---

## 14. Module Structure

### 14.1 Directory Layout
```
ComplexBiquadGA/
├── src/
│   ├── core/
│   │   ├── ProductionFilterBank.jl          # Existing: Filter implementations
│   │   ├── ModernConfigSystem.jl            # Existing: Configuration management
│   │   ├── TickHotLoopF32.jl               # Existing: Tick processing
│   │   └── SyntheticSignalGenerator.jl     # Existing: Test signals
│   │
│   ├── ga_optimization/
│   │   ├── FilterParameterGA.jl            # Chunk 1: GA for filter params
│   │   ├── FilterFitnessEvaluator.jl       # Chunk 2: Signal quality metrics
│   │   ├── PhasorTracker.jl                # Chunk 3: Phasor extraction
│   │   ├── PhasorPredictor.jl              # Chunk 4: Phasor extrapolation
│   │   ├── WeightOptimizationGA.jl         # Chunk 5: Weight evolution
│   │   └── PredictionSystem.jl             # Chunk 6: Integration
│   │
│   ├── analysis/
│   │   ├── BacktestEngine.jl               # Performance validation
│   │   ├── MetricsCalculator.jl            # Accuracy metrics
│   │   └── HorizonAnalyzer.jl              # Prediction range discovery
│   │
│   └── utils/
│       ├── DataLoader.jl                   # Tick file management
│       ├── Visualization.jl                # Plotting utilities
│       └── Logging.jl                      # Performance logging
│
├── config/
│   ├── ga/
│   │   ├── filter_ga_default.toml          # Default filter GA settings
│   │   ├── filter_ga_aggressive.toml       # High mutation rates
│   │   ├── weight_ga_default.toml          # Weight optimization settings
│   │   └── weight_ga_conservative.toml     # Low risk settings
│   │
│   ├── filters/
│   │   ├── filter_config_default.toml      # Standard filter config
│   │   ├── filter_config_pll.toml          # PLL-enhanced config
│   │   └── filter_config_optimized.toml    # Best evolved config
│   │
│   └── prediction/
│       ├── horizons_standard.toml          # Standard prediction horizons
│       ├── horizons_extended.toml          # Long-range horizons
│       └── weights_optimal.toml            # Best weight matrices
│
├── data/
│   ├── ticks/
│   │   ├── YM_training.csv                 # Training tick data
│   │   ├── YM_validation.csv               # Validation data
│   │   └── YM_test.csv                     # Out-of-sample test
│   │
│   └── results/
│       ├── ga_runs/                        # GA evolution history
│       ├── predictions/                    # Prediction outputs
│       └── metrics/                        # Performance logs
│
├── test/
│   ├── unit/
│   │   ├── test_filter_ga.jl
│   │   ├── test_phasor_tracker.jl
│   │   └── test_weight_optimization.jl
│   │
│   └── integration/
│       ├── test_end_to_end.jl
│       └── test_real_data.jl
│
├── scripts/
│   ├── run_filter_optimization.jl          # REPL script for filter GA
│   ├── run_weight_optimization.jl          # REPL script for weight GA
│   ├── analyze_predictions.jl              # Performance analysis script
│   ├── parameter_tuning.jl                 # Interactive parameter experiments
│   └── visualize_results.jl                # Plotting and visualization
│
├── Project.toml                            # Julia package dependencies
├── Manifest.toml                           # Locked dependencies
└── README.md                               # Project documentation
```

### 14.2 Module Dependencies Graph
```
TickHotLoopF32 ──────┐
                     ├──> FilterFitnessEvaluator ──> FilterParameterGA
ProductionFilterBank ┘                                      │
                                                           ↓
ModernConfigSystem ←───────────────────────────── [Optimized Configs]
        │                                                   │
        └──> PhasorTracker ──> PhasorPredictor ──> WeightOptimizationGA
                                      │                     │
                                      └───────┬─────────────┘
                                              ↓
                                        PredictionSystem
                                              │
                                              ↓
                                    [Predictions & Analysis]
```

---

## 15. Data Specifications

### 15.1 Tick Data Format
Input data from TickHotLoopF32 processing:
- **Raw format**: CSV with timestamp;bid;ask;last;volume
- **Processed format**: ComplexF32 where real = price change, imag = volume
- **Timing**: ~0.8056 seconds per tick (market average)
- **Volume**: Filtered to single-contract trades only

### 15.2 Data Requirements
- **Training**: Minimum 500,000 ticks (~5 trading days)
- **Validation**: 20% of training data size
- **Test**: Separate out-of-sample period
- **Quality**: Pre-cleaned via TickHotLoopF32 robust EMA bands

---

## 16. Fitness Function Specifications

### 16.1 Stage 1: Filter Quality Metrics

**Signal-to-Noise Ratio (SNR):**
```
SNR = 10 * log10(signal_power / noise_power)
Target: > 20 dB for primary frequencies
```

**PLL Lock Quality:**
```
lock_quality = exp(-2.0 * mean(phase_error_history))
Target: > 0.7 average across filters
```

**Frequency Selectivity:**
```
selectivity = peak_magnitude / mean(sideband_magnitude)
Target: > 10 for good isolation
```

### 16.2 Stage 2: Prediction Accuracy Metrics

**Mean Squared Error (MSE):**
```
MSE = mean((predicted_price - actual_price)^2)
Normalized by price variance for comparability
```

**Directional Accuracy:**
```
accuracy = count(sign(predicted_Δ) == sign(actual_Δ)) / total
Target: > 55% for tradeable signals
```

**Maximum Horizon Discovery:**
```
useful_horizon = max{h : directional_accuracy(h) > 52%}
```

---

## 17. GA Parameter Recommendations (Updated)

### 17.1 Filter Optimization GA

| Parameter | Range | Default | Notes |
|-----------|-------|---------|-------|
| Population Size | 50-200 | 100 | Larger due to 108+ parameters |
| Generations | 100-500 | 200 | More generations for high-dimensional search |
| Mutation Rate | 0.05-0.15 | 0.10 | **Uniform across all parameters** |
| Mutation Strength | 0.05-0.20 | 0.10 | Controls Gaussian noise magnitude |
| Crossover Rate | 0.60-0.90 | 0.80 | **Hybrid strategy (Option C)** |
| Crossover Strategy | - | Hybrid | 50% filter-swap, 50% arithmetic |
| Elitism | 5-10% | 5 | Preserve best configurations |
| Tournament Size | 3-7 | 5 | Higher pressure for convergence |
| Chromosome Length | Dynamic | n×12 | n = number of active filters |

### 17.2 Weight Optimization GA

| Parameter | Range | Default | Notes |
|-----------|-------|---------|-------|
| Population Size | 50-200 | 100 | Larger for weight complexity |
| Generations | 100-500 | 200 | Longer for fine-tuning |
| Mutation Rate | 0.01-0.10 | 0.05 | Lower for weight stability |
| Weight Range | [-10, 10] | [-5, 5] | Prevent extreme values |
| Regularization | 0.0001-0.01 | 0.001 | L2 penalty on weights |

### 17.3 Parameter-Specific Mutation Guidelines

| Parameter Type | Mutation Method | Typical σ | Notes |
|---------------|----------------|-----------|-------|
| LINEAR | Gaussian noise | 0.1 | In normalized [0,1] space |
| LOGARITHMIC | Gaussian noise | 0.1 | Applied in log-transformed space |
| BINARY | Bit flip | - | Direct flip with mutation probability |
| DISCRETE | Jump | - | 70% adjacent, 30% random |

---

## 18. Configuration File Structure (Updated)

### 18.1 Per-Filter Configuration Format

**Example: config/ga/optimized_filters.toml**
```toml
[metadata]
name = "ga_optimized"
description = "GA-optimized per-filter configuration"
created = "2024-01-15T10:30:00"
fitness = 0.8734
active_periods = [1, 2, 3, 5, 8, 13, 21, 34, 55]

[filters.1]
q_factor = 3.2
sma_window = 25
batch_size = 1500
phase_detector_gain = 0.08
loop_bandwidth = 0.008
lock_threshold = 0.72
ring_decay = 0.996
enable_clamping = true
clamping_threshold = 2.5e-7
volume_scaling = 1.8
max_frequency_deviation = 0.25
phase_error_history_length = 20

[filters.2]
q_factor = 2.8
sma_window = 30
# ... (unique parameters for each filter)

# Unused filters retained for future use
[filters.89]
q_factor = 2.0  # Default values
sma_window = 20
# ...
```

### 18.2 GA State Checkpoint Format

**Example: ga_state.toml**
```toml
[ga_state]
generation = 150
best_fitness = 0.8734
population_size = 100

[ga_config]
mutation_rate = 0.1
mutation_strength = 0.1
crossover_rate = 0.8
elitism_count = 5
tournament_size = 5

[[chromosomes]]
genes = [0.534, 0.234, ...]  # 108 values for 9 filters
fitness = 0.8734
evaluated = true
num_filters = 9
active_periods = [1, 2, 3, 5, 8, 13, 21, 34, 55]
```

---

## 19. Testing Strategy (Updated)

### 19.1 Unit Testing - Chunk 1 Specific

**Parameter Type Testing:**
- Linear encoding/decoding accuracy
- Logarithmic scaling preservation
- Binary threshold consistency
- Discrete option mapping correctness

**Per-Filter Independence:**
- Verify filters maintain independent parameters
- Cross-filter contamination checks
- Registry isolation testing
- Configuration persistence validation

**Type-Aware Mutation:**
- Gaussian noise for continuous types
- Bit flip for binary parameters
- Jump behavior for discrete parameters
- Boundary condition handling

**Struct-Based Storage:**
- FilterBank access performance
- No dictionary overhead verification
- Direct array manipulation correctness
- Pre-allocation efficiency

### 19.2 Integration Testing

**FilterParameterGA Integration:**
- Chromosome → FilterParameters → FilterBank pipeline
- Registry integration with configuration system
- TOML round-trip accuracy
- Population evolution with type constraints

**End-to-End Pipeline:**
- Tick data → Filter processing → Phasor extraction → Prediction
- Verify no phase corruption through pipeline
- Validate prediction at multiple horizons
- Confirm weight optimization improves accuracy

### 19.3 Performance Benchmarks

**Tick Processing:**
- Rate > 10,000 ticks/second
- Memory usage < 4GB for standard configuration
- No dictionary lookup overhead

**GA Operations:**
- Generation time < 30 seconds for population 50
- Prediction generation < 1ms per horizon
- Struct access < 10ns per field

### 19.4 Validation Testing

**Walk-Forward Analysis:**
- Train on period T, test on T+1
- Rolling window validation
- No lookahead bias verification
- Generalization metrics

**Market Regime Testing:**
- High volatility periods
- Low activity periods
- Trend vs ranging markets
- News event responses

---

## 20. Implementation Timeline (Updated)

### 20.1 Development Schedule (6-8 weeks)

**Week 1: ✅ COMPLETED - Chunk 1 - Filter Parameter GA**
- Designed chromosome encoding with type system
- Implemented genetic operators with type awareness
- Created population management with per-filter params
- Developed struct-based filter registry
- Unit testing and validation

**Week 2: IN PROGRESS - Chunk 2 - Filter Fitness Evaluation**
- Implement signal quality metrics
- Integrate with ProductionFilterBank
- Process sample tick data
- Validate fitness differentiation

**Week 3: Chunk 3 - Phasor Tracking**
- Extract magnitude/phase/frequency
- Implement phase unwrapping
- Track phasor stability
- Test with real data

**Week 4: Chunk 4 - Phasor Prediction**
- Implement extrapolation algorithm
- Apply real-weight scaling
- Vector summation logic
- Validate phase preservation

**Week 5: Chunk 5 - Weight Optimization**
- Design weight chromosome
- Implement prediction fitness
- Multi-horizon optimization
- Convergence testing

**Week 6-8: Chunk 6 - Integration**
- Complete pipeline assembly
- Horizon discovery analysis
- Performance optimization
- Documentation and cleanup

### 20.2 Milestones

1. **Filter GA Complete**: ✅ Optimized filter parameters achieved
2. **Phasor Extraction Working**: Stable phasor tracking from real data
3. **Predictions Generated**: First price predictions produced
4. **Weights Optimized**: Improved accuracy via weight evolution
5. **MVP Complete**: Full pipeline with discovered optimal horizon

---

## 21. Design Decisions and Rationale (NEW in v0.3)

### 21.1 Parameter Type Selection Rationale

**Logarithmic Parameters:**
- Used for parameters spanning multiple orders of magnitude
- Provides finer control at lower values
- Natural for frequency-domain parameters
- Examples: PLL gains, bandwidths, thresholds

**Linear Parameters:**
- Used for parameters with uniform sensitivity
- Direct intuitive mapping
- Narrow range parameters
- Examples: Q factor, lock threshold, ring decay

**Discrete Parameters:**
- Used for computationally significant values
- Prevents invalid buffer sizes
- Reduces search space
- Example: phase_error_history_length

### 21.2 Crossover Strategy Decision

**Hybrid Approach (Option C) Selected:**
- 50% probability of filter-level swap
- 50% probability of parameter-level mixing
- Rationale: Balances exploration and exploitation
- Maintains both coarse and fine-grained search

### 21.3 Mutation Rate Uniformity

**Uniform Rate Chosen:**
- Simplifies hyperparameter tuning
- Type-aware application provides sufficient differentiation
- Easier to analyze convergence behavior
- Can be refined in future iterations

### 21.4 No-Dictionary Design Decision

**Struct-Based Storage Selected:**
- **Performance**: Direct memory access, no hash overhead
- **Type Safety**: Compile-time verification
- **Cache Efficiency**: Better spatial locality
- **GPU Ready**: Simpler memory transfer patterns
- **Predictability**: No resize operations or hash collisions

---

## 22. Future Extensions (Post-MVP)

### 22.1 Advanced GA Features
- Adaptive mutation rates based on fitness plateau detection
- Island model GA for parallel evolution
- Coevolution of filter count and parameters
- Multi-objective optimization (accuracy vs computational cost)

### 22.2 Parameter Coupling Analysis
- Correlation detection between parameters
- Linkage learning for improved crossover
- Parameter importance ranking
- Sensitivity analysis tools

### 22.3 Online Adaptation
- Real-time parameter adjustment during trading
- Regime-specific parameter sets
- Continuous learning from prediction errors
- Automatic filter addition/removal

### 22.4 GPU Implementation
- CUDA.jl kernel development
- Multi-GPU scaling
- Real-time streaming processing
- Cloud-based distributed evolution

### 22.5 Trading Integration
- Risk-adjusted position sizing
- Transaction cost optimization
- Portfolio-level optimization
- Real-time execution framework

---

## Appendix A: Parameter Quick Reference

| Param # | Name | Type | Range | GA Encoding |
|---------|------|------|-------|-------------|
| 1 | q_factor | LINEAR | [0.5, 10.0] | Direct |
| 2 | sma_window | LOG | [1, 200] | Exponential |
| 3 | batch_size | LOG | [100, 5000] | Exponential |
| 4 | phase_detector_gain | LOG | [0.001, 1.0] | Exponential |
| 5 | loop_bandwidth | LOG | [0.0001, 0.1] | Exponential |
| 6 | lock_threshold | LINEAR | [0.0, 1.0] | Direct |
| 7 | ring_decay | LINEAR | [0.9, 1.0] | Direct |
| 8 | enable_clamping | BINARY | {0, 1} | Threshold |
| 9 | clamping_threshold | LOG | [1e-8, 1e-3] | Exponential |
| 10 | volume_scaling | LOG | [0.1, 10.0] | Exponential |
| 11 | max_frequency_deviation | LINEAR | [0.01, 0.5] | Direct |
| 12 | phase_error_history_length | DISCRETE | {5,10,15,20,30,40,50} | Enumerated |

---

## Appendix B: Struct-Based Architecture Benefits

### Performance Comparison: Struct vs Dictionary

| Operation | Struct Access | Dictionary Access | Improvement |
|-----------|--------------|-------------------|-------------|
| Field Read | ~1-2ns | ~20-50ns | 10-25x faster |
| Field Write | ~1-2ns | ~30-70ns | 15-35x faster |
| Memory Layout | Contiguous | Scattered | Better cache |
| Type Safety | Compile-time | Runtime | Fewer bugs |
| GPU Transfer | Direct | Requires conversion | Simpler |

### Memory Layout Example
```julia
# Struct: Contiguous memory
FilterBank: [filter1][filter2][filter3]...[filterN]
            [mask1][mask2][mask3]...[maskN]
            [period1][period2][period3]...[periodN]

# Dictionary: Scattered with overhead
Dict: hash_table → bucket → key_value_pair → data
      (indirect)   (search)  (comparison)   (access)
```

---

## End of Document

*This specification represents the complete design for the GA Optimization System for ComplexBiquad PLL Filter Bank, incorporating all updates from v0.3 including parameter type specifications, per-filter configuration support, and struct-based implementation without dictionaries.*