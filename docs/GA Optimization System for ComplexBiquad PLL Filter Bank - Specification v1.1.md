# GA Optimization System for ComplexBiquad PLL Filter Bank - Specification v1.1

## Revision History
- **v1.1**: Integrated TickHotLoopF32.jl documentation, clarified data pipeline from raw ticks to filter optimization
- **v1.0**: Major architecture revision - Per-filter independent populations, multi-instrument support, clarified weight application
- **v0.6**: Added full vectorization design, GPU-readiness patterns, Float32 optimization
- **v0.5**: Removed sma_window parameter, added dual parameter space architecture
- **v0.4**: Added Hybrid JLD2+TOML storage architecture, complex weight parameter

---

## 1. Executive Overview

### 1.1 Project Purpose
This specification defines a Genetic Algorithm (GA) optimization system for ComplexBiquad PLL filter banks that process real futures tick data across multiple financial instruments (YM, ES, NQ, etc.). The system uses TickHotLoopF32.jl to transform raw tick data into complex-valued signals, which are then processed by optimized filter banks. Each instrument maintains its own independent filter bank with separate GA populations. The system optimizes both filter parameters and complex prediction weights to forecast price changes at future tick indices.

### 1.2 Data Processing Pipeline

The complete data flow from raw ticks to predictions:

```
Raw Tick Data (semicolon-delimited file)
    ↓
TickHotLoopF32.jl (Data Cleaning & Normalization)
    ├─ Parse & validate tick data
    ├─ Apply robust cleaning (jump guards, winsorization)
    ├─ AGC normalization to ±1 range
    └─ 4-phase complex rotation
    ↓
Complex Signal: z(t) = price_change(t) + i*1.0
    ↓
ComplexBiquad PLL Filter Bank (GA-Optimized)
    ├─ Multiple filters at Fibonacci periods
    ├─ Each filter extracts rotating phasors
    └─ Parameters optimized by GA
    ↓
Weighted Combination (GA-Optimized Weights)
    └─ Predict future price changes
```

### 1.3 Core Innovation: Per-Filter Independent Evolution
The system treats each filter as an independent optimization problem with its own GA population. This fundamental architecture eliminates interference between filters - optimizing Filter 1 never degrades Filter 50's performance. Each filter evolves at its own rate, converging independently in a tractable 13-dimensional search space rather than attempting to optimize thousands of parameters simultaneously.

**Key Architectural Principles:**
- **Complete Filter Independence**: No genetic crossover between filters
- **Multi-Instrument Support**: Separate populations for YM, ES, NQ, etc.
- **Tractable Search Space**: 13 parameters per filter vs. 2600+ in monolithic approach
- **Parallel Evolution**: All filters can optimize simultaneously
- **Write-Through Persistence**: Memory-resident with automatic JLD2 backing
- **Robust Input Processing**: TickHotLoopF32.jl handles noisy market data

### 1.4 Two-Stage Optimization Architecture

**Stage 1: Filter Parameter Optimization**
- Optimizes 12 parameters per filter independently
- Each filter has its own population of candidate solutions
- Goal: Extract clean, stable rotating phasors from complex signals produced by TickHotLoopF32.jl
- Fitness based on signal quality metrics (SNR, lock quality, frequency selectivity)
- No genetic material exchange between filters

**Stage 2: Prediction Weight Optimization**
- Optimizes complex-valued weight for each filter
- Complex weight allows magnitude scaling and phase adjustment
- Applied ONLY to price change (real) component, not volume
- Goal: Accurate price change predictions via weighted vector sum
- Different weights for different prediction horizons

### 1.5 Complex Signal Structure and Weight Application

**TickHotLoopF32.jl Output Format:**
```julia
(tick_idx::Int64, ts::SubString, z::ComplexF32, Δ::Int32, flag::UInt8)
```
Where `z` is the complex signal:
```
z(t) = normalized_price_change(t) + i * 1.0
```
- Real part: Normalized tick-to-tick price change (±1 range after AGC)
- Imaginary part: Volume (always 1.0 for valid ticks)

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

### 1.6 Multi-Instrument Architecture

The system supports multiple financial instruments with complete isolation:

```
InstrumentGASystem
├── YM/
│   ├── FilterBankGA (e.g., 50 filters)
│   ├── Parameters (50 × 13 params each)
│   ├── TickHotLoopF32 Config (YM-specific bounds)
│   └── Storage (YM-specific JLD2/TOML)
├── ES/
│   ├── FilterBankGA (e.g., 100 filters)
│   ├── Parameters (100 × 13 params each)
│   ├── TickHotLoopF32 Config (ES-specific bounds)
│   └── Storage (ES-specific JLD2/TOML)
└── master_config.toml (lists active instruments)
```

### 1.7 Key Technical Constraints

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
11. **Robust Input Processing**: TickHotLoopF32.jl handles market noise and outliers

---

## 2. Tick Data Processing Layer (TickHotLoopF32.jl)

### 2.1 Input Data Requirements

The GA system processes tick data that has been cleaned and normalized by TickHotLoopF32.jl. Raw input files must follow this schema:

**Expected File Format:**
```
timestamp;field2;field3;last_price_ticks;volume
```

**Field Descriptions:**
1. **timestamp** - Trading timestamp (ISO format)
2. **field2** - Unused field (typically bid price)
3. **field3** - Unused field (typically ask price)
4. **last_price_ticks** - Last traded price in ticks (Int32)
5. **volume** - Number of contracts (must be 1 for valid ticks)

**Example Input:**
```
2024-01-15T09:30:00.123;41250;41251;41251;1
2024-01-15T09:30:00.456;41251;41252;41252;1
2024-01-15T09:30:00.789;41252;41253;41254;1
```

### 2.2 TickHotLoopF32 Processing Pipeline

The module applies sophisticated cleaning and normalization before data reaches the filter bank:

1. **Parse & Validate**: Check format and volume=1 constraint
2. **Absolute Price Range Check**: Market-specific bounds (e.g., 40000-43000 for YM)
3. **Delta Calculation & Hard Jump Guard**: Limit maximum per-tick movement
4. **Robust EMA Band (Winsorization)**: Soft clipping to adaptive bands
5. **AGC Normalization**: Scale to ±1 range based on recent volatility
6. **4-Phase Complex Rotation**: Distribute signal energy across complex plane

### 2.3 Instrument-Specific Cleaning Configuration

Each instrument requires tailored cleaning parameters:

```julia
# YM-specific configuration
ym_clean_cfg = CleanCfgInt(
    min_ticks = 40000,        # YM typical range lower bound
    max_ticks = 43000,        # YM typical range upper bound
    max_jump_ticks = 50,      # Maximum allowed single-tick move
    z_cut = 7.0f0,           # Robust z-score multiplier
    agc_guard_c = 7,         # AGC headroom factor
    agc_Smin = 4,            # Minimum AGC scale
    agc_Smax = 50            # Maximum AGC scale
)

# ES-specific configuration  
es_clean_cfg = CleanCfgInt(
    min_ticks = 4000,         # ES typical range lower bound
    max_ticks = 4500,         # ES typical range upper bound
    max_jump_ticks = 25,      # ES typically less volatile than YM
    z_cut = 7.0f0,
    agc_guard_c = 7,
    agc_Smin = 2,
    agc_Smax = 25
)
```

### 2.4 Output Signal Properties

After TickHotLoopF32 processing, each tick produces:

```julia
(tick_idx, ts, z, Δ, flag)
```

Where:
- **tick_idx**: Sequential tick counter (used for 4-phase rotation)
- **ts**: Original timestamp (for alignment)
- **z**: Complex signal with normalized price change (real) and unit volume (imaginary)
- **Δ**: Cleaned price change in ticks (for analysis)
- **flag**: Audit flags (HOLDLAST=0x01, CLAMPED=0x02, WINSORIZED=0x04)

The complex signal `z` has critical properties:
- Real part: Normalized to approximately ±1/7 range (with guard factor)
- Imaginary part: Always 1.0 for valid ticks
- 4-phase rotation applied based on tick_idx
- AGC-adapted to market volatility

---

## 3. Parameter Type Specifications

### 3.1 Complete Parameter Specification (13 Parameters Per Filter)

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

### 3.2 Chromosome Structure

Each filter's chromosome consists of 13 genes:
- Genes 1-11: Filter and PLL parameters
- Genes 12-13: Complex weight (real and imaginary components)

Total chromosome size: 13 × Float32 = 52 bytes per individual

---

## 4. Multi-Instrument Data Structures

### 4.1 Top-Level Instrument Management

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
    
    # TickHotLoopF32 configuration
    clean_cfg::CleanCfgInt           # Instrument-specific cleaning params
    tick_file_path::String           # Path to tick data file
    
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

### 4.2 Per-Instrument Filter Bank GA

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
    
    # TickHotLoopF32 integration
    tick_streamer::Function                  # Reference to stream_complex_ticks_f32
    clean_cfg::CleanCfgInt                  # Cleaning configuration
    
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

### 4.3 Integration with TickHotLoopF32

```julia
# Create tick data streamer for fitness evaluation
function create_tick_streamer(instrument::String, clean_cfg::CleanCfgInt)
    tick_file = get_tick_file_path(instrument)
    return () -> stream_complex_ticks_f32(tick_file, clean_cfg)
end

# Evaluate filter fitness using cleaned tick data
function evaluate_filter_fitness!(filter_ga::SingleFilterGA, 
                                 tick_streamer::Function,
                                 eval_ticks::Int = 10000)
    
    # Stream cleaned, normalized tick data
    tick_channel = tick_streamer()
    
    for chromosome in eachrow(filter_ga.population)
        # Create filter with chromosome parameters
        filter = create_filter_from_chromosome(chromosome)
        
        # Process ticks through filter
        filter_outputs = ComplexF32[]
        for (i, (tick_idx, ts, z, Δ, flag)) in enumerate(tick_channel)
            if i > eval_ticks
                break
            end
            
            # Feed normalized complex signal to filter
            output = process_tick!(filter, z)
            push!(filter_outputs, output)
        end
        
        # Calculate fitness metrics
        fitness = calculate_signal_quality(filter_outputs)
    end
end
```

---

## 5. Storage Architecture

### 5.1 Directory Structure

```
data/
├── master_config.toml              # Lists all instruments
├── YM/
│   ├── config.toml                # YM-specific configuration
│   ├── clean_config.toml          # TickHotLoopF32 parameters
│   ├── tick_data/
│   │   └── YM_20240115.txt        # Raw tick data files
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
│   ├── clean_config.toml
│   ├── tick_data/
│   ├── parameters/
│   └── ...
└── NQ/
    └── ...
```

### 5.2 Master Configuration File

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
tick_file = "data/YM/tick_data/YM_20240115.txt"

[YM.clean_config]
min_ticks = 40000
max_ticks = 43000
max_jump_ticks = 50
z_cut = 7.0
agc_guard_c = 7
agc_Smin = 4
agc_Smax = 50

[ES]
num_filters = 75
fibonacci_periods = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
population_size = 100
initialization_source = "YM"  # Initialize from YM's parameters
tick_file = "data/ES/tick_data/ES_20240115.txt"

[ES.clean_config]
min_ticks = 4000
max_ticks = 4500
max_jump_ticks = 25
z_cut = 7.0
agc_guard_c = 7
agc_Smin = 2
agc_Smax = 25

[storage]
sync_interval = 10  # Write to JLD2 every 10 generations
checkpoint_interval = 50  # Create checkpoint every 50 generations
compression = false  # No compression for write-through
```

---

## 6. Genetic Algorithm Operations

### 6.1 Independent Filter Evolution with Tick Data

```julia
# Evolve all filters in an instrument independently
function evolve_instrument!(fb_ga::FilterBankGA, generations::Int)
    for gen in 1:generations
        # Stream tick data for this generation
        tick_streamer = create_tick_streamer(fb_ga.instrument, fb_ga.clean_cfg)
        
        # Each filter evolves independently
        for filter_ga in fb_ga.filter_gas
            if !filter_ga.converged
                # Evaluate fitness using cleaned tick data
                evaluate_filter_fitness!(filter_ga, tick_streamer)
                
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

### 6.2 Fitness Evaluation with Normalized Signals

```julia
# Evaluate filter performance on TickHotLoopF32 output
function calculate_signal_quality(filter::ComplexBiquadFilter,
                                 tick_stream::Channel)::Float32
    
    quality_metrics = Float32[]
    
    for (tick_idx, ts, z, Δ, flag) in tick_stream
        # Process normalized complex signal
        output = process_tick!(filter, z)
        
        # Metrics on cleaned, normalized data
        snr = calculate_snr(output, z)
        lock_quality = calculate_lock_quality(filter)
        stability = calculate_stability(output)
        
        push!(quality_metrics, snr + lock_quality + stability)
    end
    
    return mean(quality_metrics)
end
```

### 6.3 Weight Application in Fitness Evaluation

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
    
    # De-normalize prediction using AGC scale from TickHotLoopF32
    # (In practice, would track AGC scale during processing)
    
    # Fitness is negative MSE (higher is better)
    error = prediction - target_price_change
    return -error^2
end
```

---

## 7. Initialization Strategies

### 7.1 Cross-Instrument Initialization with Cleaning Parameters

```julia
# Initialize new instrument from successful one
function initialize_from_instrument!(new_config::InstrumentConfig, 
                                    source_symbol::String)
    source_path = "data/$(source_symbol)/parameters/active.jld2"
    
    # Also consider copying cleaning parameters if markets are similar
    source_clean_path = "data/$(source_symbol)/clean_config.toml"
    
    if isfile(source_path)
        source_params = JLD2.load(source_path, "parameters")
        
        # Optionally adapt cleaning parameters
        if should_copy_clean_params(new_config.symbol, source_symbol)
            source_clean = load_clean_config(source_clean_path)
            adapt_clean_config!(new_config.clean_cfg, source_clean)
        end
        
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

### 7.2 Cleaning Parameter Adaptation

```julia
# Adapt cleaning parameters between similar instruments
function adapt_clean_config!(target::CleanCfgInt, source::CleanCfgInt, 
                            instrument_pair::Tuple{String,String})
    
    scaling_factors = Dict(
        ("YM", "ES") => 0.1,    # ES prices ~1/10 of YM
        ("ES", "NQ") => 3.0,     # NQ prices ~3x ES
        ("YM", "NQ") => 0.3      # NQ prices ~3/10 of YM
    )
    
    if haskey(scaling_factors, instrument_pair)
        scale = scaling_factors[instrument_pair]
        
        # Scale price bounds
        target.min_ticks = Int32(round(source.min_ticks * scale))
        target.max_ticks = Int32(round(source.max_ticks * scale))
        
        # Scale jump limits proportionally
        target.max_jump_ticks = Int32(round(source.max_jump_ticks * scale))
        
        # AGC parameters may need adjustment
        target.agc_Smin = Int32(round(source.agc_Smin * sqrt(scale)))
        target.agc_Smax = Int32(round(source.agc_Smax * sqrt(scale)))
        
        # Other parameters typically remain the same
        target.z_cut = source.z_cut
        target.agc_guard_c = source.agc_guard_c
    end
end
```

---

## 8. Performance Optimization

### 8.1 Memory Layout for Multi-Instrument System

```julia
# Memory calculation per instrument including TickHotLoopF32 buffers
# For YM with 50 filters, population 100:
# - Populations: 50 × 100 × 13 × 4 bytes = 260 KB
# - Fitness: 50 × 100 × 4 bytes = 20 KB
# - Working buffers: ~3× population = 780 KB
# - Tick stream buffer: 256 × (8+4+8+4+1) = 6.4 KB
# Total per instrument: ~1.1 MB

# For 10 instruments: ~11 MB total GA memory
# Highly cache-efficient, fits in L3 cache
```

### 8.2 Tick Processing Performance

| Component | Time per Tick | Throughput |
|-----------|--------------|------------|
| TickHotLoopF32 parsing | 0.5 μs | 2M ticks/sec |
| Cleaning & normalization | 1.0 μs | 1M ticks/sec |
| AGC update | 0.2 μs | 5M ticks/sec |
| 4-phase rotation | 0.1 μs | 10M ticks/sec |
| **Total preprocessing** | **1.8 μs** | **550K ticks/sec** |

### 8.3 Convergence Expectations

| Metric | Monolithic (2600D) | Per-Filter (13D) | Improvement |
|--------|-------------------|------------------|-------------|
| Search space | 2600 dimensions | 13 dimensions | 200× smaller |
| Convergence time | 5000+ generations | 50-200 generations | 25× faster |
| Population needed | 1000+ | 100 | 10× smaller |
| Memory usage | 10+ MB | 260 KB/filter | 40× less |
| Parallelization | Limited | Perfect | N× speedup |

---

## 9. Development Chunks (Unchanged from v1.0)

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

**Integration Points**:
- Will use TickHotLoopF32.jl output format for fitness evaluation
- CleanCfgInt parameters will be loaded from TOML

### Chunk 2: Multi-Instrument Support and Storage Architecture
**Purpose**: Add multi-instrument capability with separate parameter sets per market symbol and implement write-through persistence.

**Deliverables**:
- `InstrumentGASystem` top-level container
- `InstrumentConfig` with per-instrument settings including `CleanCfgInt`
- Master configuration file support (`master_config.toml`)
- Per-instrument directory structure creation
- Write-through storage system to JLD2
- Automatic parameter persistence on updates
- TOML defaults for new/uninitialized filters
- Instrument switching logic (sequential processing)
- Storage unit tests

**Integration Points**:
- Store instrument-specific TickHotLoopF32 configurations
- Link to appropriate tick data files per instrument

### Chunk 3: Filter Fitness Evaluation System
**Purpose**: Implement comprehensive fitness evaluation for filter parameter optimization (Stage 1 of two-stage optimization).

**Deliverables**:
- Signal quality metrics (SNR, lock quality, ringing)
- PLL performance evaluation
- Frequency selectivity measurement
- Integration with existing filter bank modules
- Integration with TickHotLoopF32.jl for tick streaming
- Batch fitness evaluation for populations
- Fitness caching and update strategies
- Performance benchmarking tools
- Synthetic signal testing framework

**Integration Points**:
- Direct integration with `stream_complex_ticks_f32`
- Use instrument-specific `CleanCfgInt` parameters
- Process normalized complex signals from TickHotLoopF32

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

**Integration Points**:
- Account for AGC normalization from TickHotLoopF32
- Denormalization of predictions back to tick units

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

### Chunk 6: Cross-Instrument Initialization and Convergence
**Purpose**: Implement intelligent initialization strategies and convergence detection for production deployment.

**Deliverables**:
- Cross-instrument parameter seeding (YM → ES → NQ)
- Cleaning parameter adaptation between instruments
- Success-based initialization with perturbation
- Per-filter convergence detection
- Early stopping mechanisms
- Adaptive mutation rates
- Parameter range discovery
- Best practice templates
- Production deployment scripts

**Integration Points**:
- Copy and adapt TickHotLoopF32 configurations
- Scale parameters based on instrument characteristics

### Chunk 7: Integration with Production Filter Bank
**Purpose**: Integrate GA optimization with existing ProductionFilterBank.jl and real-time tick processing.

**Deliverables**:
- Bridge between GA parameters and filter bank configuration
- Real-time parameter updates without restart
- Live fitness evaluation during market hours
- Integration with TickHotLoopF32 for live data
- A/B testing framework for parameters
- Performance impact assessment
- Rollback mechanisms
- Integration tests with tick data

**Integration Points**:
- Use `run_from_ticks_f32` for production integration
- Pass optimized CleanCfgInt to TickHotLoopF32

### Chunk 8: Monitoring, Visualization, and Analysis
**Purpose**: Build comprehensive monitoring and analysis tools for GA optimization and parameter evolution.

**Deliverables**:
- Real-time fitness dashboards
- Parameter evolution visualization
- Convergence plots per filter
- Population diversity metrics
- Fitness landscape analysis
- Parameter correlation matrices
- TickHotLoopF32 statistics display (holdlast, clamped, winsorized counts)
- Web-based monitoring interface
- Automated reporting system

---

## 10. Configuration Examples

### 10.1 Instrument-Specific Configuration with Cleaning Parameters

```toml
# data/YM/config.toml
[instrument]
symbol = "YM"
description = "E-mini Dow Jones"
tick_size = 1.0
contract_size = 5.0

[tick_processing]
data_file = "data/YM/tick_data/YM_20240115.txt"
min_ticks = 40000
max_ticks = 43000
max_jump_ticks = 50
z_cut = 7.0
agc_guard_c = 7
agc_Smin = 4
agc_Smax = 50
a_shift = 4  # α = 2^-4 for EMA(Δ)
b_shift = 4  # β = 2^-4 for EMA(|Δ-emaΔ|)
b_shift_agc = 6  # β_agc = 2^-6 for AGC envelope

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

---

## 11. Key Design Decisions (v1.1)

### 11.1 TickHotLoopF32 Integration
- **Rationale**: Robust preprocessing essential for noisy market data
- **Trade-off**: Additional processing layer before filters
- **Benefit**: Clean, normalized signals improve filter convergence

### 11.2 Per-Instrument Cleaning Parameters
- **Rationale**: Different markets have different price ranges and volatilities
- **Trade-off**: More configuration parameters to manage
- **Benefit**: Optimal preprocessing for each instrument

### 11.3 4-Phase Complex Rotation
- **Rationale**: Distributes signal energy across complex plane
- **Trade-off**: Adds phase tracking complexity
- **Benefit**: Better frequency resolution in filter bank

### 11.4 AGC Normalization
- **Rationale**: Adaptive scaling handles varying market volatility
- **Trade-off**: Must track/reverse normalization for predictions
- **Benefit**: Consistent signal amplitudes for filter processing

### 11.5 Audit Flags
- **Rationale**: Track data quality and processing decisions
- **Trade-off**: Additional metadata to manage
- **Benefit**: Debugging and quality assessment capabilities

[Previous sections 11.1-11.5 from v1.0 renumbered to 11.6-11.10]

### 11.6 Per-Filter Independence
- **Rationale**: Eliminates interference, enables parallel evolution
- **Trade-off**: No information sharing between filters
- **Benefit**: 200× reduction in search space complexity

### 11.7 Multi-Instrument Architecture
- **Rationale**: Different markets need different parameters
- **Trade-off**: Increased storage and management complexity
- **Benefit**: Optimal parameters per market

### 11.8 Weight Application to Real Part Only
- **Rationale**: Volume is always 1, only price changes matter
- **Trade-off**: Slightly more complex weight application
- **Benefit**: Correct signal processing, better predictions

### 11.9 Write-Through Persistence
- **Rationale**: Never lose optimization progress
- **Trade-off**: Periodic I/O overhead
- **Benefit**: Crash recovery, parameter history

### 11.10 Sequential Instrument Processing
- **Rationale**: Simplifies resource management
- **Trade-off**: Can't optimize YM and ES simultaneously
- **Benefit**: Predictable memory usage, easier debugging

---

## 12. Testing Strategy

### 12.1 Unit Tests Per Component

```julia
@testset "TickHotLoopF32 Integration Tests" begin
    # Test cleaning configuration
    cfg = CleanCfgInt(min_ticks=40000, max_ticks=43000)
    @test cfg.min_ticks < cfg.max_ticks
    
    # Test signal normalization
    for (idx, ts, z, Δ, flag) in stream_complex_ticks_f32(test_file, cfg)
        @test abs(real(z)) <= 1.0  # Normalized range
        @test imag(z) == 1.0  # Unit volume
    end
    
    # Test audit flags
    @test (flag & HOLDLAST) == 0 || Δ == 0
    @test (flag & CLAMPED) == 0 || abs(Δ) == cfg.max_jump_ticks
end

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

### 12.2 Integration Tests

```julia
@testset "Multi-Instrument Tests" begin
    # Test instrument isolation
    optimize_instrument!("YM", 10)
    optimize_instrument!("ES", 10)
    @test load_params("YM") !== load_params("ES")
    
    # Test cleaning parameter adaptation
    ym_cfg = load_clean_config("YM")
    es_cfg = load_clean_config("ES")
    @test es_cfg.min_ticks < ym_cfg.min_ticks  # ES has lower prices
    
    # Test cross-initialization
    init_from_instrument!("NQ", "YM")
    @test has_similar_ranges("NQ", "YM")
end
```

---

## 13. Implementation Timeline

### Phase 1: Foundation (Week 1)
- Day 1-2: Multi-instrument infrastructure with TickHotLoopF32 configs
- Day 3-4: Per-filter GA implementation
- Day 5: Write-through storage

### Phase 2: Integration (Week 2)
- Day 1-2: TickHotLoopF32 integration for fitness evaluation
- Day 3-4: Cleaning parameter management
- Day 5: Testing with real tick data

### Phase 3: Optimization (Week 3)
- Day 1-2: Vectorized operations
- Day 3-4: GPU kernels
- Day 5: Performance tuning

### Phase 4: Production (Week 4)
- Day 1-2: Weight application and prediction
- Day 3-4: Cross-instrument initialization with cleaning params
- Day 5: Deployment preparation

---

## Appendix A: Mathematical Foundations

### A.1 Signal Flow Through System

```
Raw Price P(t) in ticks
    ↓
Delta: Δ(t) = P(t) - P(t-1)
    ↓
Cleaning: Δ'(t) = winsorize(clamp(Δ(t)))
    ↓
AGC Normalization: n(t) = Δ'(t) / S(t) where S(t) adapts to volatility
    ↓
Complex Signal: z(t) = n(t) * phase_rotate(t) + i*1.0
    ↓
Filter Bank: F_k(z) for k = 1...N filters
    ↓
Weighted Sum: Prediction = Real(Σ W_k * Real(F_k(z)) + i*Imag(F_k(z)))
```

### A.2 Complex Weight Application

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

### A.3 AGC Scaling Reversal

To convert predictions back to tick units:
```
tick_prediction = normalized_prediction * S(t) * agc_guard_c
```
Where S(t) is the AGC scale factor at time t.

---

## Appendix B: Performance Benchmarks

### B.1 Expected Performance Metrics with TickHotLoopF32

| Operation | Time | Throughput |
|-----------|------|------------|
| Tick parsing & cleaning | 1.8 μs/tick | 550K ticks/sec |
| Filter processing (50 filters) | 50 μs/tick | 20K ticks/sec |
| Population evaluation (100 individuals) | 5 ms | 200 evals/sec |
| Full generation (50 filters) | 250 ms | 4 gen/sec |
| Typical convergence | 50-200 gen | 12-50 seconds |

### B.2 Memory Footprint

| Component | Single Filter | 50 Filters | 200 Filters |
|-----------|--------------|------------|-------------|
| Population | 52 KB | 2.6 MB | 10.4 MB |
| Fitness arrays | 400 B | 20 KB | 80 KB |
| Working buffers | 156 KB | 7.8 MB | 31.2 MB |
| Tick stream buffer | 6.4 KB | 6.4 KB | 6.4 KB |
| Total | ~215 KB | ~10.5 MB | ~42 MB |

---

## End of Specification v1.1

*This document represents the complete design for the GA Optimization System with integrated TickHotLoopF32.jl preprocessing, per-filter independent populations, multi-instrument support, and clarified weight application for complex signal processing.*