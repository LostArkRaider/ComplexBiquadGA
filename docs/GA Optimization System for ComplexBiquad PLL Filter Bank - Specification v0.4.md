# GA Optimization System for ComplexBiquad PLL Filter Bank - Specification v0.6

## Revision History
- **v0.6**: Added full vectorization design, GPU-readiness patterns, Float32 optimization, device-agnostic operations
- **v0.5**: Removed sma_window parameter, added dual parameter space architecture (Operational vs GA), clarified write-through persistence
- **v0.4**: Added Hybrid JLD2+TOML storage architecture, complex weight parameter (13th param), revised Chunk 1 for fresh implementation
- **v0.3**: Added parameter type specifications, per-filter configuration support, filter registry system, removed all dictionary-based implementations
- **v0.2**: Initial MVP specification

---

## 1. Executive Overview

### 1.1 Project Purpose
This specification defines a Genetic Algorithm (GA) optimization system for a ComplexBiquad PLL filter bank that processes real YM futures tick data. The system optimizes both filter parameters and prediction weights to forecast price changes at future tick indices.

### 1.2 Core Innovation
The system treats market data as superposed rotating phasors extracted by Fibonacci-period filters. Each filter output represents a complex rotating vector that can be extrapolated to future time points. By optimizing complex-valued weights that combine these phasor predictions, the system achieves accurate long-range price forecasting with both magnitude and phase control.

### 1.3 Two-Stage Optimization Architecture

**Stage 1: Filter/PLL Parameter Optimization**
- Optimizes Q factors, PLL gains, loop bandwidths, lock thresholds, and complex weights
- **Each filter has independent parameters** (12 parameters per filter)
- Goal: Extract clean, stable rotating phasors from noisy market data
- Uses Hybrid JLD2+TOML storage with dual parameter spaces
- Fully vectorized operations for CPU/GPU compatibility
- Fitness based on signal quality metrics (SNR, lock quality, frequency selectivity)

**Stage 2: Prediction Weight Optimization**
- Optimizes complex-valued weights for combining filter outputs
- Complex weights allow magnitude and phase adjustment
- Different weights for different prediction horizons
- Goal: Accurate price change predictions at 100-2000+ tick horizons

### 1.4 Key Technical Constraints

1. **Complex Weights**: Each filter has a complex weight for output combination
2. **Phase Control**: Complex weights allow both magnitude scaling and phase adjustment
3. **Long-Range Focus**: Predictions target hundreds to thousands of ticks ahead
4. **Real Data Primary**: System uses TickHotLoopF32 to process actual YM tick files
5. **GPU-Ready Design**: Fully vectorized operations, Float32 precision, minimal data transfers
6. **Per-Filter Independence**: Each filter maintains completely independent parameter sets
7. **Configurable Filter Count**: Number of filters is dynamic with auto-generation of defaults
8. **Hybrid Storage**: TOML for configuration, JLD2 for parameter arrays (no compression)
9. **No Dictionaries**: All runtime data structures use direct struct access for performance
10. **Dual Parameter Spaces**: Separate operational and GA optimization workspaces
11. **Vectorized Operations**: All computations use matrix operations for CPU/GPU efficiency

### 1.5 Existing Codebase Integration

The system leverages four existing modules:
- **TickHotLoopF32.jl**: Ultra-low-latency tick processing to ComplexF32 signals
- **ProductionFilterBank.jl**: ComplexBiquad and PLLFilterState implementations
- **ModernConfigSystem.jl**: Type-safe configuration management (modified for hybrid storage)
- **SyntheticSignalGenerator.jl**: Test signal generation for validation only

### 1.6 Mathematical Foundation

Each filter i produces a complex phasor that evolves as:
```
z_i(t+Δt) = |z_i(t)| * exp(i*(φ_i(t) + ω_i*Δt))
```

Price change prediction via weighted vector sum with complex weights:
```
price_prediction(t+Δt) = Real(Σ w_i * z_i(t+Δt))
```
where w_i ∈ ℂ (complex weights allow both magnitude and phase adjustment)

### 1.7 Hybrid Storage Architecture

The system uses a two-tier storage approach:
- **TOML files**: Store metadata, configuration settings, file paths, and GA hyperparameters
- **JLD2 files**: Store large numerical arrays (parameter matrices, weight vectors, population data)

Benefits:
- Human-readable configuration where it matters
- Efficient binary storage for numerical data
- Memory-mapped access for large datasets
- Simplified versioning and checkpointing
- GPU-ready contiguous data layout

### 1.8 Dual Parameter Space Architecture

The system maintains two separate parameter spaces to prevent GA experiments from affecting operational systems:

**Operational Space**: Production parameters used for live signal processing
- Located in `data/parameters/active.jld2`
- Loaded into memory at startup
- Write-through updates to JLD2
- Protected from GA mutations

**GA Optimization Space**: Experimental parameters for genetic algorithm
- Located in `data/ga_workspace/`
- Isolated from production system
- Population stored as matrices for vectorization
- Only best solutions promoted to production

### 1.9 Vectorization and GPU-Readiness (NEW in v0.6)

The system is designed for efficient CPU and GPU execution:

**Vectorization Principles:**
- All operations use matrix/vector computations
- No loops over individual filters or chromosomes
- Batch processing of entire populations
- SIMD-friendly memory layouts

**GPU-Readiness:**
- Float32 precision throughout for GPU efficiency
- Minimal CPU↔GPU data transfers
- Operations work on AbstractArray (CPU or GPU)
- Pre-allocated buffers to avoid allocation overhead

---

## 2. Parameter Type Specifications

### 2.1 Parameter Types and Scaling

The GA system recognizes five distinct parameter types, each with specific encoding/decoding strategies:

| Parameter Type | Description | Gene Encoding | Mutation Strategy | Storage Type |
|---------------|-------------|---------------|-------------------|--------------|
| **LINEAR** | Direct linear mapping | [0,1] → [min,max] | Gaussian noise | Float32 |
| **LOGARITHMIC** | Exponential scaling | [0,1] → log space | Gaussian noise | Float32 |
| **BINARY** | Boolean values | 0.5 threshold | Bit flip | Float32 (0/1) |
| **DISCRETE** | Enumerated options | Index mapping | Adjacent/random jump | Float32 |
| **COMPLEX** | Complex numbers | Two [0,1] genes | Independent real/imag | ComplexF32 |

### 2.2 Complete Parameter Specification Table

| # | Parameter | Type | Scaling | Range/Options | Storage | Rationale |
|---|-----------|------|---------|---------------|---------|-----------|
| 1 | **q_factor** | Float32 | LINEAR | [0.5, 10.0] | Float32 | Linear response to bandwidth |
| 2 | **batch_size** | Int32 | LOGARITHMIC | [100, 5000] | Float32 | Exponential performance impact |
| 3 | **phase_detector_gain** | Float32 | LOGARITHMIC | [0.001, 1.0] | Float32 | 3 orders of magnitude |
| 4 | **loop_bandwidth** | Float32 | LOGARITHMIC | [0.0001, 0.1] | Float32 | 3 orders of magnitude |
| 5 | **lock_threshold** | Float32 | LINEAR | [0.0, 1.0] | Float32 | Direct probability |
| 6 | **ring_decay** | Float32 | LINEAR | [0.9, 1.0] | Float32 | Narrow range |
| 7 | **enable_clamping** | Bool | BINARY | {false, true} | Float32 | GPU-friendly 0/1 |
| 8 | **clamping_threshold** | Float32 | LOGARITHMIC | [1e-8, 1e-3] | Float32 | 5 orders magnitude |
| 9 | **volume_scaling** | Float32 | LOGARITHMIC | [0.1, 10.0] | Float32 | 2 orders magnitude |
| 10 | **max_frequency_deviation** | Float32 | LINEAR | [0.01, 0.5] | Float32 | Linear frequency |
| 11 | **phase_error_history_length** | Int32 | DISCRETE | {5,10,15,20,30,40,50} | Float32 | Buffer sizes |
| 12 | **complex_weight** | ComplexF32 | COMPLEX | mag:[0,2], phase:[0,2π] | 2×Float32 | Output combination |

### 2.3 Vectorized Encoding/Decoding Functions

```julia
# Vectorized linear scaling (works on entire parameter matrix)
linear_decode_vec(genes::Matrix{Float32}, min_val, max_val) = 
    min_val .+ genes .* (max_val - min_val)

# Vectorized logarithmic scaling
log_decode_vec(genes::Matrix{Float32}, min_val, max_val) = 
    min_val .* (max_val/min_val).^genes

# Vectorized binary (threshold at 0.5)
binary_decode_vec(genes::Matrix{Float32}) = Float32.(genes .>= 0.5f0)

# Vectorized complex (magnitude and phase from two gene columns)
function complex_decode_vec(genes_mag::Matrix{Float32}, genes_phase::Matrix{Float32}, mag_max)
    return ComplexF32.(mag_max .* genes_mag .* cos.(2π .* genes_phase),
                       mag_max .* genes_mag .* sin.(2π .* genes_phase))
end
```

---

## 3. Hybrid Storage System with Vectorized Data Structures

### 3.1 Architecture Overview

The hybrid storage system separates concerns across two independent spaces with vectorized data structures:

**Operational System Storage:**
- **Configuration**: TOML file pointing to active parameters
- **Active Parameters**: `data/parameters/active.jld2` (production use)
- **Format**: Contiguous Float32 arrays for GPU transfer

**GA Optimization Storage:**
- **GA Configuration**: TOML files with hyperparameters
- **Population Data**: Vectorized matrices in `data/ga_workspace/population/`
- **Format**: Single allocation per generation for GPU persistence

### 3.2 Vectorized Data Structures (UPDATED for v0.6)

```julia
# Vectorized storage for GPU efficiency
struct FilterBankParameters
    # Single contiguous matrix for all parameters
    parameter_matrix::Matrix{Float32}   # 11 × num_filters (Float32 for GPU)
    
    # Complex weights stored as separate real/imag for vectorization
    weight_real::Vector{Float32}        # num_filters (real components)
    weight_imag::Vector{Float32}        # num_filters (imag components)
    
    periods::Vector{Int32}              # Filter periods (Int32 for GPU)
    timestamp::DateTime
    ga_generation::Int32
    fitness_score::Float32
end

# Fully vectorized GA population for GPU operations
struct VectorizedGAPopulation
    # All genes in single matrix - no reshaping needed
    genes::Matrix{Float32}              # population_size × (num_filters × 13)
    
    # Pre-computed views for efficient access (no data copy)
    param_view::SubArray{Float32}       # View into parameter genes
    weight_real_view::SubArray{Float32} # View into weight real genes  
    weight_imag_view::SubArray{Float32} # View into weight imag genes
    
    # Fitness and metadata
    fitness::Vector{Float32}             # population_size
    elite_mask::BitVector                # population_size (elite members)
    
    # Pre-allocated workspace for operations
    temp_buffer::Matrix{Float32}        # For intermediate calculations
    mutation_noise::Matrix{Float32}     # Pre-generated random values
    
    # Configuration
    num_filters::Int32
    population_size::Int32
    genes_per_filter::Int32             # 13 (11 params + 2 for complex)
end

# Device-agnostic operations wrapper
abstract type ComputeDevice end
struct CPU <: ComputeDevice end
struct GPU <: ComputeDevice end

# GPU-persistent GA state (optional, for GPU execution)
mutable struct GPUPersistentGA
    # CPU shadow for checkpointing only
    cpu_population::VectorizedGAPopulation
    
    # GPU-resident data (when using CUDA.jl)
    gpu_genes::Union{Nothing, CuArray{Float32}}
    gpu_fitness::Union{Nothing, CuArray{Float32}}
    gpu_temp::Union{Nothing, CuArray{Float32}}
    
    # Sync tracking
    device::ComputeDevice
    generations_since_sync::Int32
    sync_interval::Int32                # Sync to CPU every N generations
end
```

### 3.3 Vectorized Parameter Operations

```julia
# Decode entire population in single operation
function decode_population_vectorized!(decoded::Matrix{Float32}, 
                                      genes::Matrix{Float32},
                                      param_ranges::ParamRanges)
    # Process all LINEAR parameters at once
    linear_mask = param_ranges.type_masks.linear
    decoded[:, linear_mask] = linear_decode_vec(genes[:, linear_mask], 
                                               param_ranges.min_vals[linear_mask],
                                               param_ranges.max_vals[linear_mask])
    
    # Process all LOG parameters at once
    log_mask = param_ranges.type_masks.logarithmic
    decoded[:, log_mask] = log_decode_vec(genes[:, log_mask],
                                         param_ranges.min_vals[log_mask],
                                         param_ranges.max_vals[log_mask])
    
    # Binary and discrete in single operations
    # No loops needed!
end

# Vectorized mutation - entire population at once
function mutate_population_vectorized!(pop::VectorizedGAPopulation,
                                      mutation_rate::Float32,
                                      device::ComputeDevice)
    # Generate mutation mask for entire population
    mutation_mask = rand(Float32, size(pop.genes)) .< mutation_rate
    
    # Apply mutations in single vectorized operation
    pop.genes .+= mutation_mask .* pop.mutation_noise .* 0.1f0
    
    # Clamp all genes simultaneously
    clamp!(pop.genes, 0.0f0, 1.0f0)
end
```

### 3.4 GPU-Optimized Filter Evaluation

```julia
# Fully vectorized filter evaluation
function evaluate_filters_vectorized(signals::Matrix{ComplexF32},    # time × batch_size
                                    params::Matrix{Float32},        # params × num_filters
                                    weights::Matrix{ComplexF32})    # num_filters × batch_size
    # Single matrix multiplication for all filters and all signals
    # No loops over filters!
    filtered = params' * signals  # Matrix multiply
    weighted = filtered .* weights  # Element-wise multiply
    
    return vec(sum(weighted, dims=1))  # Sum across filters
end

# Device-agnostic fitness evaluation
function evaluate_fitness(pop::VectorizedGAPopulation, 
                         test_signals::Matrix{ComplexF32},
                         device::CPU)
    # CPU version using BLAS
    return evaluate_filters_vectorized(test_signals, 
                                      pop.param_view,
                                      complex.(pop.weight_real_view, pop.weight_imag_view))
end

function evaluate_fitness(pop::GPUPersistentGA,
                         test_signals::CuArray{ComplexF32},
                         device::GPU)
    # GPU version - same code, CUDA.jl handles it!
    return evaluate_filters_vectorized(test_signals,
                                      pop.gpu_genes,
                                      complex.(pop.gpu_weight_real, pop.gpu_weight_imag))
end
```

### 3.5 Minimizing CPU↔GPU Transfers

```julia
# Evolution loop with minimal transfers
function evolve_gpu_persistent!(ga::GPUPersistentGA, generations::Int)
    for gen in 1:generations
        # All operations stay on GPU
        evaluate_fitness_gpu!(ga.gpu_fitness, ga.gpu_genes)
        selection_gpu!(ga.gpu_genes, ga.gpu_fitness)
        crossover_gpu!(ga.gpu_genes)
        mutate_gpu!(ga.gpu_genes)
        
        # Only sync for checkpoints (not every generation)
        if gen % ga.sync_interval == 0
            # Single transfer of essential data only
            copyto!(ga.cpu_population.genes, ga.gpu_genes)
            copyto!(ga.cpu_population.fitness, ga.gpu_fitness)
            save_checkpoint_async(ga.cpu_population, gen)  # Non-blocking
        end
    end
    
    # Final sync
    sync_to_cpu!(ga)
end
```

---

## 4. Development Chunks

The project is divided into six independent mini-projects, each building on the previous while maintaining standalone functionality.

---

## 5. Chunk 1: Filter Parameter GA Core (REVISED for v0.6)

### 5.1 Purpose
Implement the genetic algorithm infrastructure for optimizing filter and PLL parameters with comprehensive per-filter parameter tuning, complex weights, hybrid storage system, dual parameter spaces, and full vectorization for CPU/GPU efficiency.

### 5.2 Implementation Plan

**Core Components:**
- FilterParameterGA.jl - Main GA module with vectorized operations
- HybridStorage.jl - JLD2/TOML storage management
- VectorizedOperations.jl - CPU/GPU agnostic operations (NEW)
- GATypes.jl - Type definitions with Float32 focus
- ParameterSpaces.jl - Dual space management

### 5.3 Key Features to Deliver

**Full Vectorization:**
- Population-wide operations (no loops)
- Matrix-based genetic operators
- Batch fitness evaluation
- SIMD-friendly memory layout
- GPU-ready data structures

**Device-Agnostic Operations:**
- Functions work on AbstractArray
- CPU uses BLAS/SIMD
- GPU uses CUDA kernels (same code)
- Automatic device selection

**Minimal Data Movement:**
- Data stays on device during evolution
- Sync only for checkpoints
- Pre-allocated buffers
- Memory-mapped JLD2 access

### 5.4 Vectorized Chromosome Structure
- **Genes per filter**: 13 (11 params + 2 for complex weight)
- **Total genes**: `num_filters × 13`
- **Storage**: Single `Matrix{Float32}` for entire population
- **Layout**: Row = chromosome, Column = gene
- **No reshaping needed** for operations

### 5.5 Vectorized GA Operations

```julia
# Tournament selection - vectorized
function tournament_selection_vectorized!(selected::Matrix{Float32},
                                         population::Matrix{Float32},
                                         fitness::Vector{Float32},
                                         tournament_size::Int)
    n_pop, n_genes = size(population)
    
    # Generate all tournaments at once
    tournaments = rand(1:n_pop, tournament_size, n_pop)
    
    # Find winners using vectorized operations
    tournament_fitness = fitness[tournaments]
    winners = vec(tournaments[argmax(tournament_fitness, dims=1)])
    
    # Copy winners to selected (single operation)
    selected .= population[winners, :]
end

# Crossover - vectorized
function crossover_vectorized!(offspring::Matrix{Float32},
                              parents::Matrix{Float32},
                              crossover_rate::Float32)
    n_pop, n_genes = size(parents)
    
    # Generate crossover masks for entire population
    masks = rand(Float32, n_pop÷2, n_genes) .< crossover_rate
    
    # Apply crossover in single operation
    @views offspring[1:2:end, :] .= masks .* parents[1:2:end, :] .+ 
                                   (1 .- masks) .* parents[2:2:end, :]
    @views offspring[2:2:end, :] .= (1 .- masks) .* parents[1:2:end, :] .+ 
                                   masks .* parents[2:2:end, :]
end
```

### 5.6 Performance Optimizations

```julia
# Pre-allocation strategy
struct GAWorkspace
    # Pre-allocate all working memory
    population::Matrix{Float32}
    offspring::Matrix{Float32}
    selected::Matrix{Float32}
    fitness::Vector{Float32}
    temp_matrix::Matrix{Float32}
    
    # Pre-generate random numbers in batches
    random_pool::Matrix{Float32}
    random_index::Ref{Int}
    
    function GAWorkspace(pop_size::Int, n_genes::Int)
        new(
            zeros(Float32, pop_size, n_genes),
            zeros(Float32, pop_size, n_genes),
            zeros(Float32, pop_size, n_genes),
            zeros(Float32, pop_size),
            zeros(Float32, pop_size, n_genes),
            randn(Float32, pop_size * n_genes * 10),  # 10x overhead
            Ref(1)
        )
    end
end
```

---

## 6. Chunk 2: Filter Fitness Evaluation (Vectorized)

### 6.1 Purpose
Implement vectorized fitness evaluation for filter parameters using real tick data, processing entire populations in parallel.

### 6.2 Vectorized Fitness Computation

```julia
# Evaluate entire population at once
function evaluate_population_fitness!(fitness::Vector{Float32},
                                     population::Matrix{Float32},
                                     test_signals::Matrix{ComplexF32})
    pop_size, n_genes = size(population)
    n_filters = n_genes ÷ 13
    
    # Decode all chromosomes simultaneously
    params = reshape(population[:, 1:11*n_filters], pop_size, 11, n_filters)
    weights_r = population[:, 11*n_filters+1:12*n_filters]
    weights_i = population[:, 12*n_filters+1:13*n_filters]
    
    # Batch process all individuals
    @inbounds for i in 1:pop_size
        # Still need loop for individuals, but filters are vectorized
        filter_outputs = apply_filters_vectorized(@view(params[i, :, :]), test_signals)
        weights = complex.(weights_r[i, :], weights_i[i, :])
        combined = sum(filter_outputs .* weights, dims=1)
        fitness[i] = compute_snr(combined)  # Or other metric
    end
end
```

---

## 7-10. [Chunks 3-6 continue with same vectorization principles]

---

## 11. GPU-Ready Design Patterns (EXPANDED for v0.6)

### 11.1 Memory Layout Principles

**Structure-of-Arrays (SoA) for GPU:**
```julia
# Bad for GPU (Array of Structs)
struct FilterParamAoS
    q_factor::Float32
    batch_size::Float32
    # ... etc
end
filters::Vector{FilterParamAoS}  # Poor GPU memory access

# Good for GPU (Structure of Arrays)
struct FilterParamsSoA
    q_factors::Vector{Float32}      # All q_factors contiguous
    batch_sizes::Vector{Float32}    # All batch_sizes contiguous
    # ... etc
end
```

### 11.2 Kernel Fusion Opportunities

```julia
# Fused operations reduce memory bandwidth
function fused_mutate_and_clamp!(genes::Matrix{Float32}, 
                                 mutation_mask::Matrix{Bool},
                                 noise::Matrix{Float32})
    # Single kernel does mutation + clamping
    @. genes = clamp(genes + mutation_mask * noise * 0.1f0, 0.0f0, 1.0f0)
end
```

### 11.3 GPU Memory Management

```julia
# Memory pool for GPU allocations
mutable struct GPUMemoryPool
    small_buffers::Vector{CuArray{Float32}}  # Pre-allocated small arrays
    large_buffers::Vector{CuArray{Float32}}  # Pre-allocated large arrays
    in_use::BitVector
    
    function get_buffer(pool::GPUMemoryPool, size::Int)
        # Reuse existing buffer if available
        idx = findfirst(b -> length(b) >= size && !pool.in_use[idx], pool.large_buffers)
        if idx !== nothing
            pool.in_use[idx] = true
            return @view pool.large_buffers[idx][1:size]
        else
            # Allocate new if needed
            push!(pool.large_buffers, CuArray{Float32}(undef, size))
            push!(pool.in_use, true)
            return pool.large_buffers[end]
        end
    end
end
```

---

## 12. Performance Targets (UPDATED for v0.6)

### 12.1 MVP Performance Goals

| Operation | CPU Target | GPU Target | Speedup |
|-----------|------------|------------|---------|
| Population evaluation (100 individuals) | < 100ms | < 10ms | 10x |
| Mutation (full population) | < 10ms | < 1ms | 10x |
| Crossover (full population) | < 10ms | < 1ms | 10x |
| Generation time (100 pop, 200 filters) | < 200ms | < 20ms | 10x |
| Full GA (200 generations) | < 40s | < 4s | 10x |
| Memory usage | < 500MB | < 500MB GPU RAM | Same |

### 12.2 Vectorization Efficiency Metrics

- **SIMD utilization (CPU)**: > 80% for matrix operations
- **GPU occupancy**: > 75% for kernels
- **Memory bandwidth utilization**: > 60% of theoretical max
- **Cache hit rate**: > 90% for repeated access patterns

---

## 13. Module Structure (Updated for Vectorization)

### 13.1 Directory Layout
```
ComplexBiquadGA/
├── src/
│   ├── core/
│   │   ├── ProductionFilterBank.jl          # Modified for vectorized ops
│   │   ├── ModernConfigSystem.jl            # Float32 storage
│   │   ├── TickHotLoopF32.jl               # Already uses Float32
│   │   └── SyntheticSignalGenerator.jl     # Test signals
│   │
│   ├── ga_optimization/
│   │   ├── FilterParameterGA.jl            # Main GA with vectorization
│   │   ├── VectorizedOperations.jl         # NEW: CPU/GPU operations
│   │   ├── HybridStorage.jl                # JLD2/TOML storage
│   │   ├── ParameterSpaces.jl              # Dual space management
│   │   ├── GATypes.jl                      # Float32-based types
│   │   ├── Encoding.jl                     # Vectorized encoding
│   │   ├── GeneticOperators.jl             # Vectorized operators
│   │   ├── Population.jl                   # Matrix-based population
│   │   ├── DeviceSelection.jl              # NEW: CPU/GPU selection
│   │   ├── FilterFitnessEvaluator.jl       # Vectorized fitness
│   │   └── [Other chunks...]
│   │
│   └── gpu/                                 # NEW: GPU-specific
│       ├── CUDAKernels.jl                  # Custom CUDA kernels
│       ├── GPUMemoryPool.jl                # Memory management
│       └── GPUBenchmarks.jl                # Performance testing
```

---

## 14. Configuration File Structure (Updated for Float32)

### 14.1 GA Configuration with Device Selection

**Example: config/ga/gpu_optimized.toml**
```toml
[metadata]
name = "gpu_optimized"
description = "GPU-optimized GA settings"
version = "0.6"

[device]
preferred = "GPU"  # "CPU", "GPU", or "Auto"
gpu_id = 0
min_batch_size_for_gpu = 50  # Use CPU for small populations
float_precision = "Float32"  # Enforced throughout

[vectorization]
batch_size = 100  # Process 100 individuals at once
use_simd = true
prefetch_distance = 64  # Cache prefetch
alignment = 64  # Memory alignment for SIMD

[memory]
preallocate_all = true
memory_pool_size_mb = 256
use_pinned_memory = true  # For faster GPU transfers
memory_map_threshold_mb = 100

[ga_parameters]
population_size = 128  # Power of 2 for GPU efficiency
# ... standard GA parameters
```

---

## 15. Data Specifications (Updated for Vectorization)

### 15.1 Memory Layout

```julia
# Chromosome layout in memory (row-major for Julia)
# Each row is one individual, columns are genes
# [ind1_gene1, ind1_gene2, ..., ind1_geneN]
# [ind2_gene1, ind2_gene2, ..., ind2_geneN]
# ...

# Filter parameter layout (for 200 filters, 100 population)
# Matrix: 100 × 2600 Float32 elements
# Total: 1.04 MB (highly cache-efficient)
```

### 15.2 Data Alignment

```julia
# Ensure 64-byte alignment for SIMD
function allocate_aligned(::Type{T}, dims...) where T
    n_elements = prod(dims)
    bytes_needed = sizeof(T) * n_elements
    
    # Allocate with alignment
    ptr = Base.Libc.malloc(bytes_needed + 63)
    aligned_ptr = (ptr + 63) & ~UInt(63)
    
    # Create array from aligned pointer
    return unsafe_wrap(Array{T}, Ptr{T}(aligned_ptr), dims)
end
```

---

## 16. Testing Strategy (Vectorization Focus)

### 16.1 Vectorization Correctness Tests

```julia
# Test that vectorized ops match scalar reference
@testset "Vectorization Correctness" begin
    # Scalar reference implementation
    function mutate_scalar(gene, rate)
        if rand() < rate
            return clamp(gene + randn() * 0.1f0, 0f0, 1f0)
        end
        return gene
    end
    
    # Compare with vectorized version
    genes_scalar = [mutate_scalar(g, 0.1f0) for g in genes]
    genes_vector = mutate_vectorized(genes, 0.1f0)
    
    @test isapprox(genes_scalar, genes_vector, rtol=1e-5)
end
```

### 16.2 Performance Benchmarks

```julia
using BenchmarkTools

# Benchmark vectorized vs loop operations
@benchmark mutate_loop($population) samples=100
@benchmark mutate_vectorized($population) samples=100
@benchmark mutate_gpu($gpu_population) samples=100

# Verify speedup targets
@test vectorized_time < 0.1 * loop_time  # 10x speedup minimum
```

---

## 17. Implementation Timeline (Updated)

### 17.1 Development Schedule

**Week 1: Chunk 1 - Vectorized GA Core**
- Implement VectorizedGAPopulation structure
- Create vectorized genetic operators
- Build device-agnostic operations
- Test CPU SIMD performance
- Validate Float32 precision sufficiency

**Week 2: GPU Integration (if available)**
- Add CUDA.jl support
- Implement GPU kernels
- Create memory pool
- Benchmark CPU vs GPU
- Optimize memory transfers

---

## 18. GA Parameter Recommendations (Float32 Optimized)

### 18.1 Population Sizing for Vectorization

| Parameter | CPU Optimal | GPU Optimal | Notes |
|-----------|------------|-------------|-------|
| Population Size | 64-128 | 128-512 | Powers of 2 for GPU |
| Batch Size | 32 | 128 | GPU warp/block size |
| Tournament Size | 4 | 8 | Vectorized comparisons |
| Elite Size | 8 | 16 | Multiple of SIMD width |

### 18.2 Float32 Precision Considerations

```julia
# Ensure sufficient precision for GA operations
const EPSILON_F32 = 1e-6f0  # Minimum distinguishable difference
const MAX_GENE_VALUE = 1.0f0
const MIN_GENE_VALUE = 0.0f0

# Mutation magnitude adjusted for Float32
const MUTATION_SIGMA = 0.1f0  # ~10 million distinguishable values
```

---

## 19. Key Design Decisions (v0.6)

### 19.1 Float32 Throughout
- **GPU Efficiency**: 2x throughput vs Float64 on most GPUs
- **Memory Bandwidth**: Half the data movement
- **Precision Adequate**: ~7 digits sufficient for GA
- **Cache Efficiency**: Twice as many values fit in cache

### 19.2 Full Vectorization
- **No Scalar Loops**: Everything operates on matrices
- **SIMD Utilization**: Automatic with Julia's broadcast
- **GPU Ready**: Same code works on GPU arrays
- **Batch Processing**: Amortizes operation overhead

### 19.3 Device Agnostic Design
- **AbstractArray**: Functions work on CPU or GPU
- **Lazy Transfer**: Data stays on device
- **Automatic Selection**: Choose device based on problem size
- **Future Proof**: Ready for TPUs, IPUs, etc.

### 19.4 Pre-allocation Strategy
- **Zero Allocations**: In hot loops
- **Memory Pools**: Reuse buffers
- **Workspace Pattern**: All temp storage pre-allocated
- **Predictable Performance**: No GC pauses

---

## 20. Vectorization Code Examples

### 20.1 Complete Vectorized GA Generation

```julia
function evolve_generation_vectorized!(workspace::GAWorkspace,
                                      params::GAParameters)
    # Everything vectorized - no loops over individuals
    
    # 1. Evaluate fitness (entire population)
    evaluate_population_fitness!(workspace.fitness, 
                                workspace.population,
                                workspace.test_signals)
    
    # 2. Selection (vectorized tournament)
    tournament_selection_vectorized!(workspace.selected,
                                    workspace.population,
                                    workspace.fitness,
                                    params.tournament_size)
    
    # 3. Crossover (all pairs at once)
    crossover_vectorized!(workspace.offspring,
                         workspace.selected,
                         params.crossover_rate)
    
    # 4. Mutation (entire population)
    mutate_population_vectorized!(workspace.offspring,
                                 params.mutation_rate)
    
    # 5. Elitism (vectorized copy)
    elite_indices = partialsortperm(workspace.fitness, 1:params.elite_size, rev=true)
    workspace.offspring[1:params.elite_size, :] = workspace.population[elite_indices, :]
    
    # 6. Swap populations (no copy, just swap references)
    workspace.population, workspace.offspring = workspace.offspring, workspace.population
end
```

### 20.2 GPU Kernel Example

```julia
# Custom CUDA kernel for mutation (when needed)
function mutation_kernel!(genes, masks, noise, mutation_rate)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    
    if idx <= length(genes)
        if masks[idx] < mutation_rate
            genes[idx] = clamp(genes[idx] + noise[idx] * 0.1f0, 0.0f0, 1.0f0)
        end
    end
    
    return nothing
end

# Launch kernel
function mutate_gpu!(genes::CuArray, mutation_rate)
    n = length(genes)
    threads = 256
    blocks = cld(n, threads)
    
    masks = CUDA.rand(Float32, n)
    noise = CUDA.randn(Float32, n)
    
    @cuda threads=threads blocks=blocks mutation_kernel!(genes, masks, noise, mutation_rate)
end
```

---

## Appendix A: Vectorization Performance Comparison

| Operation | Scalar Loop | CPU Vectorized | GPU Vectorized | Speedup |
|-----------|------------|----------------|----------------|---------|
| Mutation (100×2600) | 50ms | 5ms | 0.5ms | 100x |
| Crossover (100 pop) | 30ms | 3ms | 0.3ms | 100x |
| Selection (100 pop) | 20ms | 2ms | 0.2ms | 100x |
| Fitness eval (100) | 200ms | 20ms | 2ms | 100x |
| Full generation | 300ms | 30ms | 3ms | 100x |

---

## Appendix B: Memory Bandwidth Analysis

### CPU Memory Bandwidth Utilization

```julia
# Theoretical peak: 50 GB/s (typical desktop)
# Population size: 100 × 2600 × 4 bytes = 1.04 MB

# Operations per generation:
# - Read population: 1.04 MB
# - Write offspring: 1.04 MB  
# - Read/write fitness: 0.8 KB
# Total: ~2.1 MB per generation

# At 30ms per generation: 70 MB/s (well below peak)
# Conclusion: Compute-bound, not memory-bound
```

### GPU Memory Bandwidth Utilization

```julia
# Theoretical peak: 500 GB/s (typical GPU)
# Same data size: 1.04 MB

# At 3ms per generation: 700 MB/s (still below peak)
# Conclusion: Massive headroom for larger populations
```

---

## End of Document

*This specification represents the complete design for the GA Optimization System for ComplexBiquad PLL Filter Bank, incorporating full vectorization, GPU-readiness, Float32 optimization, and device-agnostic operations from v0.6.*