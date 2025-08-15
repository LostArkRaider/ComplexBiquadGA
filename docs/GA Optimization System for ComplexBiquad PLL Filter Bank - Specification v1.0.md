# GA Optimization System for ComplexBiquad PLL Filter Bank - Specification v1.0

## Revision History
- **v1.0**: Major architecture revision - Per-filter independent populations, multi-instrument support, clarified weight application
- **v0.6**: Added full vectorization design, GPU-readiness patterns, Float32 optimization
- **v0.5**: Removed sma_window parameter, added dual parameter space architecture
- **v0.4**: Added Hybrid JLD2+TOML storage architecture, complex weight parameter

---

## 1. Executive Overview

### 1.1 Project Purpose
This specification defines a Genetic Algorithm (GA) optimization system for ComplexBiquad PLL filter banks that process real futures tick data across multiple financial instruments (YM, ES, NQ, etc.). Each instrument maintains its own independent filter bank with separate GA populations. The system optimizes both filter parameters and complex prediction weights to forecast price changes at future tick indices.

### 1.2 Core Innovation: Per-Filter Independent Evolution
The system treats each filter as an independent optimization problem with its own GA population. This fundamental architecture eliminates interference between filters - optimizing Filter 1 never degrades Filter 50's performance. Each filter evolves at its own rate, converging independently in a tractable 13-dimensional search space rather than attempting to optimize thousands of parameters simultaneously.

**Key Architectural Principles:**
- **Complete Filter Independence**: No genetic crossover between filters
- **Multi-Instrument Support**: Separate populations for YM, ES, NQ, etc.
- **Tractable Search Space**: 13 parameters per filter vs. 2600+ in monolithic approach
- **Parallel Evolution**: All filters can optimize simultaneously
- **Write-Through Persistence**: Memory-resident with automatic JLD2 backing

### 1.3 Two-Stage Optimization Architecture

**Stage 1: Filter Parameter Optimization**
- Optimizes 12 parameters per filter independently
- Each filter has its own population of candidate solutions
- Goal: Extract clean, stable rotating phasors from noisy market data
- Fitness based on signal quality metrics (SNR, lock quality, frequency selectivity)
- No genetic material exchange between filters

**Stage 2: Prediction Weight Optimization**
- Optimizes complex-valued weight for each filter
- Complex weight allows magnitude scaling and phase adjustment
- Applied ONLY to price change (real) component, not volume
- Goal: Accurate price change predictions via weighted vector sum
- Different weights for different prediction horizons

### 1.4 Complex Signal Structure and Weight Application

**Input Signal Format:**
```
z(t) = price_change(t) + i * 1.0
```
- Real part: tick-to-tick price change
- Imaginary part: volume (always 1.0)

**Weight Application:**
```
weighted_output = complex_weight * real(filter_output) + i * imag(filter_output)
```
- Complex weight modifies ONLY the price change component
- Volume component passes through unchanged
- Enables both magnitude scaling and phase rotation of price signal

**Prediction via Vector Sum:**
```
price_prediction(t+Δt) = Real(Σ weighted_outputs)
```

### 1.5 Multi-Instrument Architecture

The system supports multiple financial instruments with complete isolation:

```
InstrumentGASystem
├── YM/
│   ├── FilterBankGA (e.g., 50 filters)
│   ├── Parameters (50 × 13 params each)
│   └── Storage (YM-specific JLD2/TOML)
├── ES/
│   ├── FilterBankGA (e.g., 100 filters)
│   ├── Parameters (100 × 13 params each)
│   └── Storage (ES-specific JLD2/TOML)
└── master_config.toml (lists active instruments)
```

### 1.6 Key Technical Constraints

1. **Per-Filter Independence**: Each filter maintains completely separate GA population
2. **Configurable Population Size**: Same population size across all filters within an instrument
3. **Complex Weight Application**: Modifies price change only, preserves unit volume
4. **Multi-Instrument Support**: Separate parameter sets per market symbol
5. **Variable Filter Count**: Each instrument can have different number of filters (20-256)
6. **Write-Through Persistence**: Automatic save to JLD2 on parameter updates
7. **No Inter-Filter Crossover**: Complete genetic isolation between filters
8. **GPU-Ready Vectorization**: 3D tensor operations for parallel evolution
9. **Float32 Precision**: Throughout for GPU efficiency
10. **Sequential Instrument Processing**: One instrument optimizes at a time

---

## 2. Parameter Type Specifications

### 2.1 Complete Parameter Specification (13 Parameters Per Filter)

| # | Parameter | Type | Scaling | Range/Options | Purpose |
|---|-----------|------|---------|---------------|---------|
| 1 | **q_factor** | Float32 | LINEAR | [0.5, 10.0] | Filter bandwidth |
| 2 | **batch_size** | Int32 | LOGARITHMIC | [100, 5000] | Processing batch |
| 3 | **phase_detector_gain** | Float32 | LOGARITHMIC | [0.001, 1.0] | PLL sensitivity |
| 4 | **loop_bandwidth** | Float32 | LOGARITHMIC | [0.0001, 0.1] | PLL response |
| 5 | **lock_threshold** | Float32 | LINEAR | [0.0, 1.0] | Lock quality |
| 6 | **ring_decay** | Float32 | LINEAR | [0.9, 1.0] | Ringing factor |
| 7 | **enable_clamping** | Bool | BINARY | {false, true} | Clamp enable |
| 8 | **clamping_threshold** | Float32 | LOGARITHMIC | [1e-8, 1e-3] | Clamp level |
| 9 | **volume_scaling** | Float32 | LOGARITHMIC | [0.1, 10.0] | Volume factor |
| 10 | **max_frequency_deviation** | Float32 | LINEAR | [0.01, 0.5] | Freq limits |
| 11 | **phase_error_history_length** | Int32 | DISCRETE | {5,10,15,20,30,40,50} | Buffer size |
| 12-13 | **complex_weight** | ComplexF32 | COMPLEX | mag:[0,2], phase:[0,2π] | Price weighting |

### 2.2 Chromosome Structure

Each filter's chromosome consists of 13 genes:
- Genes 1-11: Filter and PLL parameters
- Genes 12-13: Complex weight (real and imaginary components)

Total chromosome size: 13 × Float32 = 52 bytes per individual

---

## 3. Multi-Instrument Data Structures

### 3.1 Top-Level Instrument Management

```julia
# Master system managing all instruments
struct InstrumentGASystem
    # Active instruments and their configurations
    instruments::Dict{String, InstrumentConfig}  # "YM" => config
    active_instruments::Vector{String}           # ["YM", "ES", "NQ"]
    
    # Currently optimizing instrument (sequential processing)
    current_instrument::Union{String, Nothing}
    
    # Master configuration
    master_config_path::String                   # "config/master.toml"
    
    # Global settings
    gpu_enabled::Bool
    max_memory_gb::Float32
    checkpoint_interval::Int32
end

# Per-instrument configuration
struct InstrumentConfig
    symbol::String                    # "YM", "ES", etc.
    num_filters::Int32                # 20-256
    population_size::Int32            # Same for all filters in instrument
    
    # Storage paths
    parameter_path::String            # "data/YM/parameters/active.jld2"
    ga_workspace_path::String         # "data/YM/ga_workspace/"
    config_path::String              # "data/YM/config.toml"
    
    # Filter specifications
    fibonacci_periods::Vector{Int32}  # [1,2,3,5,8,13,21,34,55...]
    
    # Optimization settings
    max_generations::Int32
    convergence_threshold::Float32
    
    # Cross-instrument initialization
    initialization_source::Union{String, Nothing}  # "YM" to copy from
end
```

### 3.2 Per-Instrument Filter Bank GA

```julia
# GA system for one instrument's entire filter bank
struct FilterBankGA
    instrument::String                        # "YM"
    num_filters::Int32                       # Total filters
    population_size::Int32                   # Same for all filters
    
    # Independent GA for each filter
    filter_gas::Vector{SingleFilterGA}       # Length = num_filters
    
    # Shared configuration
    ga_params::GAParameters                  # Mutation rate, etc.
    
    # Write-through storage
    storage::WriteThruStorage
    
    # Performance tracking
    generation::Int32
    total_evaluations::Int64
    best_fitness_history::Vector{Float32}
end

# Individual filter GA (small, tractable)
struct SingleFilterGA
    # Filter identity
    period::Int32                            # Fibonacci period
    filter_index::Int32                      # Position in bank
    
    # GA population (small!)
    population::Matrix{Float32}              # population_size × 13
    fitness::Vector{Float32}                 # population_size
    
    # Best solution tracking
    best_chromosome::Vector{Float32}         # 13 parameters
    best_fitness::Float32
    generations_since_improvement::Int32
    
    # Evolution state
    generation::Int32
    converged::Bool
    
    # Parameter bounds (period-specific in future versions)
    param_ranges::ParameterRanges
end
```

### 3.3 Write-Through Storage System

```julia
# Automatic persistence to JLD2
mutable struct WriteThruStorage
    # Memory-resident parameters
    active_params::Matrix{Float32}           # num_filters × 13
    
    # JLD2 backing store
    jld2_path::String
    last_sync::DateTime
    sync_interval::Int32                     # Generations between syncs
    
    # Change tracking
    dirty_filters::BitVector                 # Which filters changed
    pending_updates::Int32
    
    # TOML defaults for new filters
    default_config::FilterDefaults
end

# Default configuration for new filters
struct FilterDefaults
    # Loaded from TOML
    default_q_factor::Float32
    default_batch_size::Int32
    default_pll_gain::Float32
    default_loop_bandwidth::Float32
    # ... other defaults
    
    # Period-specific overrides (optional)
    period_overrides::Dict{Int32, Vector{Float32}}
end
```

### 3.4 Vectorized Operations for Parallel Evolution

```julia
# 3D tensor for vectorized operations across all filters
struct VectorizedFilterBankOps
    # Stacked populations: [filter_id, individual, parameter]
    all_populations::Array{Float32, 3}       # num_filters × pop_size × 13
    all_fitness::Matrix{Float32}             # num_filters × pop_size
    
    # Pre-allocated workspace
    selection_buffer::Array{Float32, 3}
    crossover_buffer::Array{Float32, 3}
    mutation_buffer::Array{Float32, 3}
    
    # GPU arrays (if enabled)
    gpu_populations::Union{Nothing, CuArray{Float32, 3}}
    gpu_fitness::Union{Nothing, CuArray{Float32, 2}}
end
```

---

## 4. Storage Architecture

### 4.1 Directory Structure

```
data/
├── master_config.toml              # Lists all instruments
├── YM/
│   ├── config.toml                # YM-specific configuration
│   ├── parameters/
│   │   ├── active.jld2            # Current best parameters
│   │   └── checkpoint_*.jld2      # Historical checkpoints
│   ├── ga_workspace/
│   │   ├── population.jld2        # Current populations
│   │   ├── fitness_history.jld2   # Fitness tracking
│   │   └── convergence.jld2       # Convergence metrics
│   └── defaults.toml              # Default parameters for new filters
├── ES/
│   ├── config.toml
│   ├── parameters/
│   └── ...
└── NQ/
    └── ...
```

### 4.2 Master Configuration File

```toml
# data/master_config.toml
[instruments]
active = ["YM", "ES", "NQ"]
default_population_size = 100
default_generations = 500

[YM]
num_filters = 50
fibonacci_periods = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
population_size = 100
initialization_source = ""  # No source, random init

[ES]
num_filters = 75
fibonacci_periods = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
population_size = 100
initialization_source = "YM"  # Initialize from YM's parameters

[NQ]
num_filters = 100
fibonacci_periods = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233]
population_size = 128
initialization_source = "ES"  # Initialize from ES's parameters

[storage]
sync_interval = 10  # Write to JLD2 every 10 generations
checkpoint_interval = 50  # Create checkpoint every 50 generations
compression = false  # No compression for write-through
```

---

## 5. Genetic Algorithm Operations

### 5.1 Independent Filter Evolution

```julia
# Evolve all filters in an instrument independently
function evolve_instrument!(fb_ga::FilterBankGA, generations::Int)
    for gen in 1:generations
        # Each filter evolves independently
        for filter_ga in fb_ga.filter_gas
            if !filter_ga.converged
                # Evaluate fitness for this filter only
                evaluate_filter_fitness!(filter_ga)
                
                # Genetic operations on this filter's population
                selection!(filter_ga)
                crossover!(filter_ga)
                mutate!(filter_ga)
                
                # Update best solution
                update_best!(filter_ga)
                
                # Check convergence
                check_convergence!(filter_ga)
            end
        end
        
        # Write-through to storage
        if gen % fb_ga.storage.sync_interval == 0
            sync_to_storage!(fb_ga.storage)
        end
    end
end
```

### 5.2 Vectorized Parallel Evolution

```julia
# Vectorized operations for GPU efficiency
function evolve_instrument_vectorized!(fb_ga::FilterBankGA, vops::VectorizedFilterBankOps)
    # Stack all populations into 3D tensor
    for (i, fga) in enumerate(fb_ga.filter_gas)
        vops.all_populations[i, :, :] = fga.population
    end
    
    # Evaluate all filters in parallel
    evaluate_all_filters_vectorized!(vops.all_fitness, vops.all_populations)
    
    # Vectorized genetic operations
    selection_vectorized!(vops.selection_buffer, vops.all_populations, vops.all_fitness)
    crossover_vectorized!(vops.crossover_buffer, vops.selection_buffer)
    mutate_vectorized!(vops.mutation_buffer, vops.crossover_buffer)
    
    # Unstack back to individual populations
    for (i, fga) in enumerate(fb_ga.filter_gas)
        fga.population = vops.mutation_buffer[i, :, :]
        fga.fitness = vops.all_fitness[i, :]
    end
end
```

### 5.3 Weight Application in Fitness Evaluation

```julia
# Evaluate prediction accuracy with proper weight application
function evaluate_prediction_fitness(filter_outputs::Vector{ComplexF32}, 
                                    weights::Vector{ComplexF32},
                                    target_price_change::Float32)::Float32
    # Apply weights ONLY to price change (real part)
    weighted_sum = ComplexF32(0, 0)
    
    for (output, weight) in zip(filter_outputs, weights)
        # Weight multiplies real part only, imaginary preserved
        weighted_output = weight * real(output) + im * imag(output)
        weighted_sum += weighted_output
    end
    
    # Prediction is real part of sum
    prediction = real(weighted_sum)
    
    # Fitness is negative MSE (higher is better)
    error = prediction - target_price_change
    return -error^2
end
```

---

## 6. Initialization Strategies

### 6.1 Cross-Instrument Initialization

```julia
# Initialize new instrument from successful one
function initialize_from_instrument!(new_config::InstrumentConfig, 
                                    source_symbol::String)
    source_path = "data/$(source_symbol)/parameters/active.jld2"
    
    if isfile(source_path)
        source_params = JLD2.load(source_path, "parameters")
        
        # Map source filters to new instrument's filters
        for (i, period) in enumerate(new_config.fibonacci_periods)
            # Find matching period in source
            source_idx = findfirst(==(period), source_periods)
            
            if source_idx !== nothing
                # Copy parameters with small random perturbation
                base_params = source_params[source_idx, :]
                
                # Initialize population around successful parameters
                for j in 1:new_config.population_size
                    perturbation = randn(Float32, 13) * 0.1f0
                    initial_params = base_params + perturbation
                    # Store in new instrument's population
                end
            else
                # Random initialization for unmatched periods
                initialize_random_filter!(period)
            end
        end
    end
end
```

### 6.2 Default Parameter Loading

```julia
# Load defaults for new filters
function load_filter_defaults(instrument::String)::FilterDefaults
    default_path = "data/$(instrument)/defaults.toml"
    
    if isfile(default_path)
        config = TOML.parsefile(default_path)
        return parse_filter_defaults(config)
    else
        # Use system-wide defaults
        return FilterDefaults(
            default_q_factor = 2.0f0,
            default_batch_size = 1000,
            default_pll_gain = 0.1f0,
            default_loop_bandwidth = 0.01f0,
            # ... other defaults
        )
    end
end
```

---

## 7. Performance Optimization

### 7.1 Memory Layout for Multi-Instrument System

```julia
# Memory calculation per instrument
# For YM with 50 filters, population 100:
# - Populations: 50 × 100 × 13 × 4 bytes = 260 KB
# - Fitness: 50 × 100 × 4 bytes = 20 KB
# - Working buffers: ~3× population = 780 KB
# Total per instrument: ~1 MB

# For 10 instruments: ~10 MB total GA memory
# Highly cache-efficient, fits in L3 cache
```

### 7.2 Convergence Expectations

| Metric | Monolithic (2600D) | Per-Filter (13D) | Improvement |
|--------|-------------------|------------------|-------------|
| Search space | 2600 dimensions | 13 dimensions | 200× smaller |
| Convergence time | 5000+ generations | 50-200 generations | 25× faster |
| Population needed | 1000+ | 100 | 10× smaller |
| Memory usage | 10+ MB | 260 KB/filter | 40× less |
| Parallelization | Limited | Perfect | N× speedup |

### 7.3 GPU Optimization Patterns

```julia
# 3D tensor operations for all filters simultaneously
# Dimensions: [filter_id, individual, parameter]

# Single CUDA kernel evaluates all filters
@cuda threads=256 blocks=num_filters evaluate_filters_kernel!(
    all_populations,  # 3D tensor
    all_fitness,      # 2D matrix
    test_signals      # Shared test data
)

# Memory access pattern optimized for coalescing
# Each thread block handles one filter
# Threads within block handle individuals
```

---

## 8. Development Chunks (Revised for v1.0 Architecture)

### Chunk 1: Core GA Infrastructure for Single-Filter Populations
**Purpose**: Establish the foundation for per-filter independent GA populations with proper data structures and basic genetic operations.

**Deliverables**:
- `SingleFilterGA` struct with 13-parameter chromosomes
- `FilterBankGA` container managing n independent filter GAs
- Basic genetic operators (selection, crossover, mutation) for 13D search space
- Parameter encoding/decoding for all 12 filter params + complex weight
- Population initialization with configurable size
- Fitness evaluation interface (stub for now)
- Basic TOML configuration loading
- Unit tests for genetic operators

**Key Features**:
- Population size configurable across all filters (e.g., 100 for all)
- No inter-filter genetic exchange
- Complex weight stored as 2 Float32 genes
- Independent evolution rate per filter
- Memory-efficient 13D chromosome structure

**Success Criteria**:
- Can create and evolve single filter populations independently
- Genetic operators work correctly on 13-parameter chromosomes
- No memory leaks or filter cross-contamination

---

### Chunk 2: Multi-Instrument Support and Storage Architecture
**Purpose**: Add multi-instrument capability with separate parameter sets per market symbol and implement write-through persistence.

**Deliverables**:
- `InstrumentGASystem` top-level container
- `InstrumentConfig` with per-instrument settings
- Master configuration file support (`master_config.toml`)
- Per-instrument directory structure creation
- Write-through storage system to JLD2
- Automatic parameter persistence on updates
- TOML defaults for new/uninitialized filters
- Instrument switching logic (sequential processing)
- Storage unit tests

**Key Features**:
- Each instrument (YM, ES, NQ) has separate filter banks
- Variable filter counts per instrument (20-256)
- Automatic directory creation: `data/YM/`, `data/ES/`, etc.
- Memory-resident parameters with JLD2 backing
- Configurable sync intervals
- Checkpoint/recovery system

**Success Criteria**:
- Can manage multiple instruments with different configurations
- Parameters persist automatically to disk
- Can recover from crashes without data loss
- Proper isolation between instruments

---

### Chunk 3: Filter Fitness Evaluation System
**Purpose**: Implement comprehensive fitness evaluation for filter parameter optimization (Stage 1 of two-stage optimization).

**Deliverables**:
- Signal quality metrics (SNR, lock quality, ringing)
- PLL performance evaluation
- Frequency selectivity measurement
- Integration with existing filter bank modules
- Batch fitness evaluation for populations
- Fitness caching and update strategies
- Performance benchmarking tools
- Synthetic signal testing framework

**Key Features**:
- Uses real tick data or synthetic signals
- Per-filter independent fitness calculation
- No cross-filter fitness dependencies
- Efficient batch evaluation of 100 individuals
- Fitness history tracking

**Success Criteria**:
- Accurate fitness reflects filter quality
- Can process real YM tick data
- Fitness evaluation < 10ms per population
- Reproducible results with same data

---

### Chunk 4: Complex Weight Optimization for Prediction
**Purpose**: Implement Stage 2 optimization for complex weights that combine filter outputs for price prediction.

**Deliverables**:
- Complex weight application (real part only modification)
- Vector summation with proper weight handling
- Price prediction at multiple horizons (100-2000 ticks)
- Prediction accuracy fitness metrics
- Weight magnitude and phase optimization
- Integration with filter outputs
- Backtesting framework

**Key Features**:
- Weight affects price change (real) only, not volume
- Complex multiplication for phase adjustment
- Multi-horizon evaluation (different weights per horizon)
- MSE/MAE fitness metrics for prediction
- Preserves unit volume in imaginary component

**Success Criteria**:
- Correct weight application verified mathematically
- Improved prediction accuracy vs. unweighted
- Can optimize for different time horizons
- Real-time prediction capability

---

### Chunk 5: Vectorized Operations and GPU Acceleration
**Purpose**: Optimize performance through vectorization and optional GPU support for parallel filter evolution.

**Deliverables**:
- 3D tensor operations for all filters simultaneously
- Vectorized genetic operators (selection, crossover, mutation)
- Batch fitness evaluation across all filters
- GPU kernel implementations (CUDA.jl)
- CPU SIMD optimizations
- Memory pool management
- Performance profiling tools
- Benchmark suite

**Key Features**:
- Stack all filter populations into 3D tensors
- Single operation processes all filters
- Minimal CPU↔GPU transfers
- Float32 precision throughout
- Pre-allocated buffers
- Zero-allocation hot paths

**Success Criteria**:
- 5-10x speedup with vectorization
- Additional 5-10x with GPU (if available)
- Can evolve 200 filters in parallel
- Memory usage < 100MB for 200 filters
- No performance degradation over time

---

### Chunk 6: Cross-Instrument Initialization and Convergence
**Purpose**: Implement intelligent initialization strategies and convergence detection for production deployment.

**Deliverables**:
- Cross-instrument parameter seeding (YM → ES → NQ)
- Success-based initialization with perturbation
- Per-filter convergence detection
- Early stopping mechanisms
- Adaptive mutation rates
- Parameter range discovery
- Best practice templates
- Production deployment scripts

**Key Features**:
- Initialize new instruments from successful ones
- Small random perturbations around good solutions
- Independent convergence per filter
- Automatic parameter range adjustment
- Stagnation detection and restart
- Parameter stability metrics

**Success Criteria**:
- Faster convergence with seeded initialization
- Robust convergence detection
- Can identify optimal parameter ranges
- Production-ready with monitoring
- Automatic recovery from poor initializations

---

### Chunk 7: Integration with Production Filter Bank
**Purpose**: Integrate GA optimization with existing ProductionFilterBank.jl and real-time tick processing.

**Deliverables**:
- Bridge between GA parameters and filter bank configuration
- Real-time parameter updates without restart
- Live fitness evaluation during market hours
- A/B testing framework for parameters
- Performance impact assessment
- Rollback mechanisms
- Integration tests with tick data

**Key Features**:
- Hot-swappable parameters
- Zero-downtime updates
- Side-by-side parameter comparison
- Real-time fitness monitoring
- Automatic rollback on degradation

**Success Criteria**:
- Seamless integration with existing codebase
- No disruption to live trading
- Can update parameters during market hours
- Measurable improvement in predictions
- Safe rollback capability

---

### Chunk 8: Monitoring, Visualization, and Analysis
**Purpose**: Build comprehensive monitoring and analysis tools for GA optimization and parameter evolution.

**Deliverables**:
- Real-time fitness dashboards
- Parameter evolution visualization
- Convergence plots per filter
- Population diversity metrics
- Fitness landscape analysis
- Parameter correlation matrices
- Web-based monitoring interface
- Automated reporting system

**Key Features**:
- Per-instrument dashboards
- Individual filter tracking
- Historical parameter evolution
- Fitness improvement trends
- Population health metrics
- Alert system for anomalies

**Success Criteria**:
- Clear visibility into optimization progress
- Can identify problematic filters
- Historical analysis capabilities
- Exportable reports for analysis
- Real-time monitoring without performance impact

---

### Development Notes:
- Each chunk is designed to be completed in 1-2 focused sessions
- Chunks 1-2 are foundational and must be completed first
- Chunks 3-4 can be developed in parallel after 1-2
- Chunk 5 is optional but recommended for scale
- Chunks 6-8 are production-readiness features
- Total estimated development time: 4-6 weeks

---

## 9. Configuration Examples

### 9.1 Instrument-Specific Configuration

```toml
# data/YM/config.toml
[instrument]
symbol = "YM"
description = "E-mini Dow Jones"
tick_size = 1.0
contract_size = 5.0

[filters]
count = 50
periods = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]

[ga]
population_size = 100
mutation_rate = 0.1
crossover_rate = 0.7
elite_size = 10
tournament_size = 5

[optimization]
stage1_generations = 200  # Filter parameters
stage2_generations = 100  # Weights
convergence_threshold = 0.001
early_stopping_patience = 20

[storage]
sync_interval = 10
checkpoint_interval = 50
keep_best_n = 5
```

### 9.2 Filter Defaults Configuration

```toml
# data/YM/defaults.toml
[default_parameters]
q_factor = 2.0
batch_size = 1000
phase_detector_gain = 0.1
loop_bandwidth = 0.01
lock_threshold = 0.7
ring_decay = 0.995
enable_clamping = false
clamping_threshold = 1e-6
volume_scaling = 1.0
max_frequency_deviation = 0.2
phase_error_history_length = 20

# Period-specific overrides
[period_overrides.1]
q_factor = 1.5  # Lower Q for period 1

[period_overrides.89]
q_factor = 3.0  # Higher Q for longer periods
loop_bandwidth = 0.005
```

---

## 10. Key Design Decisions (v1.0)

### 10.1 Per-Filter Independence
- **Rationale**: Eliminates interference, enables parallel evolution
- **Trade-off**: No information sharing between filters
- **Benefit**: 200× reduction in search space complexity

### 10.2 Multi-Instrument Architecture
- **Rationale**: Different markets need different parameters
- **Trade-off**: Increased storage and management complexity
- **Benefit**: Optimal parameters per market

### 10.3 Weight Application to Real Part Only
- **Rationale**: Volume is always 1, only price changes matter
- **Trade-off**: Slightly more complex weight application
- **Benefit**: Correct signal processing, better predictions

### 10.4 Write-Through Persistence
- **Rationale**: Never lose optimization progress
- **Trade-off**: Periodic I/O overhead
- **Benefit**: Crash recovery, parameter history

### 10.5 Sequential Instrument Processing
- **Rationale**: Simplifies resource management
- **Trade-off**: Can't optimize YM and ES simultaneously
- **Benefit**: Predictable memory usage, easier debugging

---

## 11. Testing Strategy

### 11.1 Unit Tests Per Component

```julia
@testset "Per-Filter GA Tests" begin
    # Test filter independence
    @test filter1.population !== filter2.population
    
    # Test weight application
    output = apply_weight(ComplexF32(1.5, 1.0), ComplexF32(0.5, 0.5))
    @test real(output) ≈ 0.75  # 0.5 * 1.5
    @test imag(output) ≈ 1.0   # Unchanged
    
    # Test convergence
    @test filter_ga.generations_since_improvement < 50
end
```

### 11.2 Integration Tests

```julia
@testset "Multi-Instrument Tests" begin
    # Test instrument isolation
    optimize_instrument!("YM", 10)
    optimize_instrument!("ES", 10)
    @test load_params("YM") !== load_params("ES")
    
    # Test cross-initialization
    init_from_instrument!("NQ", "YM")
    @test has_similar_ranges("NQ", "YM")
end
```

---

## 12. Implementation Timeline

### Phase 1: Foundation (Week 1)
- Day 1-2: Multi-instrument infrastructure
- Day 3-4: Per-filter GA implementation
- Day 5: Write-through storage

### Phase 2: Optimization (Week 2)
- Day 1-2: Vectorized operations
- Day 3-4: GPU kernels
- Day 5: Performance tuning

### Phase 3: Integration (Week 3)
- Day 1-2: Weight application and prediction
- Day 3-4: Cross-instrument initialization
- Day 5: Testing and validation

### Phase 4: Production (Week 4)
- Day 1-2: Monitoring and logging
- Day 3-4: Documentation
- Day 5: Deployment preparation

---

## Appendix A: Mathematical Foundations

### A.1 Complex Weight Application

Given:
- Filter output: `F = F_r + i*F_i`
- Complex weight: `W = W_r + i*W_i`
- Input volume: Always 1.0

Weight application:
```
Weighted = W * F_r + i*F_i
         = (W_r + i*W_i) * F_r + i*F_i
         = W_r*F_r + i*(W_i*F_r + F_i)
```

Only the real part (price change) is weighted, preserving unit volume in imaginary.

### A.2 Prediction Calculation

For n filters with outputs F₁...Fₙ and weights W₁...Wₙ:
```
Prediction = Real(Σ(Wᵢ * Real(Fᵢ) + i*Imag(Fᵢ)))
          = Σ(Real(Wᵢ) * Real(Fᵢ))
```

The imaginary components cancel in the final sum, leaving only weighted price changes.

---

## Appendix B: Performance Benchmarks

### B.1 Expected Performance Metrics

| Operation | CPU (Single Filter) | GPU (All Filters) | Speedup |
|-----------|-------------------|-------------------|---------|
| Population evaluation | 1ms | 10ms (50 filters) | 5× |
| Genetic operations | 0.5ms | 5ms (50 filters) | 5× |
| Full generation | 2ms | 20ms (50 filters) | 5× |
| 100 generations | 200ms | 2s (50 filters) | 5× |
| Convergence (typical) | 10s | 100s (50 filters) | N/A |

### B.2 Memory Footprint

| Component | Single Filter | 50 Filters | 200 Filters |
|-----------|--------------|------------|-------------|
| Population | 52 KB | 2.6 MB | 10.4 MB |
| Fitness arrays | 400 B | 20 KB | 80 KB |
| Working buffers | 156 KB | 7.8 MB | 31.2 MB |
| Total | ~210 KB | ~10.5 MB | ~42 MB |

---

## End of Specification v1.0

*This document represents the complete design for the GA Optimization System with per-filter independent populations, multi-instrument support, and clarified weight application for complex signal processing.*