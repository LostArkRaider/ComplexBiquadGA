# GA Optimization System for ComplexBiquad PLL Filter Bank - Specification v1.5

**Version:** 1.5  
**Date:** August 17, 2025  
**Status:** Chunks 1-4 Complete with Phase Extrapolation

## Executive Summary

This specification defines a two-stage genetic algorithm (GA) optimization system for a ComplexBiquad PLL filter bank used in futures tick data forecasting. The system optimizes both filter parameters (Stage 1) and prediction weights (Stage 2) using phase-based extrapolation for accurate future price change prediction.

### Key Innovation in v1.5
**Phase-Based Extrapolation**: Each filter's rotating phasor is projected forward n ticks using its design frequency before applying weights and summing for prediction. This provides more accurate future predictions by accounting for the natural oscillation of each filter.

## 1. System Overview

### 1.1 Purpose
Optimize a bank of ComplexBiquad filters with PLL enhancement to predict normalized price changes [-1, +1] from futures tick data using phase-extrapolated vector summation.

### 1.2 Core Components

1. **Filter Bank**: Multiple ComplexBiquad filters with different Fibonacci periods
2. **GA Optimizer**: Two-stage optimization (filter parameters, then weights)
3. **Phase Extrapolator**: Projects filter phasors forward in time
4. **Prediction System**: Weighted vector sum of projected filter outputs
5. **Storage System**: Multi-instrument persistent state management

### 1.3 Two-Stage Optimization

#### Stage 1: Filter Parameter Optimization
- 13 parameters per filter
- Energy-based fitness metrics
- Independent GA population per filter

#### Stage 2: Weight Optimization (REVISED)
- Scalar weights [0,1] per filter
- Phase-based extrapolation before weighting
- MSE minimization on I-component

### 1.4 Signal Processing Pipeline

```
Tick Data → Complex Signal → Filters → Phase Projection → Weighted Sum → Prediction
    ↓            ↓                           ↓                  ↓            ↓
  price      I=Δprice[-1,+1]          magnitude*e^(iφ)    weight*proj    real()
  volume      Q=volume                  φ'=φ+ωn
```

### 1.5 Complex Signal Structure

**Input Signal:**
- Real (I): Normalized price change [-1, +1]
- Imaginary (Q): Volume/tick count (encodes phase)

**4-Phase Rotation:**
- Each tick advances π/2 (90°)
- Provides phase reference frame
- Q tracks rotation position

**Phase Extrapolation (NEW in v1.5):**
- Project each filter's phasor forward n ticks
- Use design frequency ω = 2π/period
- Constant magnitude (no decay)
- Linear phase model: φ(t+n) = φ(t) + ωn

## 2. Technical Architecture

### 2.1 GA System Architecture

```
FilterBankGA
├── Filter 1 GA (13D parameters)
│   ├── Population (100 individuals)
│   ├── Filter Fitness (energy-based)
│   └── Weight GA (scalar weights)
│       ├── Population (50 individuals)
│       └── Prediction Fitness (MSE)
├── Filter 2 GA
│   └── ... (independent)
└── Filter N GA
    └── ... (independent)
```

### 2.2 Parameter Space

#### Filter Parameters (13D)
1. `q_factor` [0.5, 10.0] - Filter bandwidth
2. `batch_size` [100, 5000] - Processing batch
3. `phase_detector_gain` [0.001, 1.0] - PLL sensitivity
4. `loop_bandwidth` [0.0001, 0.1] - PLL bandwidth
5. `lock_threshold` [0.0, 1.0] - Lock quality threshold
6. `ring_decay` [0.9, 1.0] - Ringing decay
7. `enable_clamping` {0, 1} - Binary flag
8. `clamping_threshold` [1e-8, 1e-3] - Clamp level
9. `volume_scaling` [0.1, 10.0] - Q scaling
10. `max_frequency_deviation` [0.01, 0.5] - Freq limits
11. `phase_error_history_length` {5,10,15,20,30,40,50} - Buffer size
12. `weight_magnitude` [0, 1] - Scalar weight (Stage 2)
13. `reserved` - Future use

### 2.3 Phase Extrapolation Model

#### Frequency Calculation
```julia
frequency[k] = 2π / period[k]  # Design frequency
# where period is doubled Fibonacci (e.g., 13 → 26)
```

#### Phase Projection
```julia
magnitude[k] = abs(filter_output[k])
phase[k] = angle(filter_output[k])
projected_phase[k] = phase[k] + frequency[k] * n_ticks
projected[k] = magnitude[k] * exp(im * projected_phase[k])
```

#### Prediction
```julia
prediction = real(Σ(weight[k] * projected[k]))
```

## 3. Implementation Modules

### 3.1 Core GA Infrastructure (Chunk 1) ✅
- `GATypes.jl`: Core data structures
- `GAOperations.jl`: Selection, crossover, mutation
- `GAPopulation.jl`: Population management

### 3.2 Multi-Instrument Support (Chunk 2) ✅
- `InstrumentManager.jl`: YM, ES, NQ support
- `StorageSystem.jl`: JLD2 persistence
- `ConfigurationLoader.jl`: TOML configuration

### 3.3 Filter Fitness Evaluation (Chunk 3) ✅
- `FilterIntegration.jl`: GA ↔ Filter bridge
- `SignalMetrics.jl`: Energy-based metrics
- `FitnessEvaluation.jl`: Weighted scoring

### 3.4 Weight Optimization (Chunk 4) ✅ REVISED
- `WeightOptimization.jl`: Scalar weight GA with phase projection
- `PricePrediction.jl`: Phase extrapolation and vector sum
- `PredictionMetrics.jl`: I-component MSE/MAE

### 3.5 System Integration (Chunk 5) 🔜
- `GAController.jl`: Two-stage orchestration
- `TickDataProcessor.jl`: Real tick processing
- `ProductionPipeline.jl`: End-to-end workflow

## 4. Fitness Functions

### 4.1 Stage 1: Filter Fitness (Energy-Based)
```julia
fitness_filter = w₁*SNR + w₂*lock_quality + w₃*(1-ringing) + w₄*selectivity
```

### 4.2 Stage 2: Prediction Fitness (Phase-Based)
```julia
# With phase extrapolation:
projected[k] = project_forward(filter[k], frequency[k], n_ticks)
prediction = real(Σ(weight[k] * projected[k]))
MSE = mean((prediction - actual)²)
fitness_weight = 1/(1 + MSE) * (0.7 + 0.3*directional_accuracy)
```

## 5. Data Flow

### 5.1 Tick Processing
```
Raw Tick → Parse → Normalize → Complex Signal → Filter Bank
   ↓         ↓         ↓            ↓              ↓
 price    validate  [-1,+1]    I=Δprice      filter outputs
 volume              Q=volume
```

### 5.2 Prediction Pipeline (REVISED)
```
Filter Outputs → Extract Phase/Mag → Project Forward → Apply Weights → Sum → Real Part
      ↓               ↓                    ↓              ↓           ↓        ↓
   z[k,t]         |z|,∠z            z[k,t+n]         w[k]*z      Σ(w*z)   prediction
```

### 5.3 4-Phase Rotation
```
Tick 1: Phase = 0°    (1, 0)
Tick 2: Phase = 90°   (0, 1)
Tick 3: Phase = 180°  (-1, 0)
Tick 4: Phase = 270°  (0, -1)
Tick 5: Phase = 0°    (cycle repeats)
```

## 6. Performance Requirements

### 6.1 Optimization Performance
- Population size: 100 (filters), 50 (weights)
- Generations: Adaptive (convergence-based)
- Evaluation time: <10ms per filter
- Memory: <1GB per instrument

### 6.2 Prediction Performance
- Latency: <1ms per prediction ✅
- Throughput: >5000 predictions/second ✅
- Phase projection overhead: <0.02ms/filter ✅
- Horizons: 100-2000 ticks (continuous range)

### 6.3 Accuracy Targets
- Directional accuracy: >55%
- Correlation: >0.3
- Sharpe ratio: >1.0

## 7. Storage and Persistence

### 7.1 State Management
```julia
InstrumentState
├── filter_parameters: Matrix{Float32}  # per filter
├── weights: Matrix{Float32}            # per horizon
├── fitness_history: Vector{Float32}
├── best_individuals: Dict
└── metadata: Dict
```

### 7.2 File Structure
```
data/
├── populations/
│   ├── YM_filter_1.jld2
│   ├── YM_weights_1.jld2
│   └── ...
├── configs/
│   ├── YM.toml
│   └── ...
└── results/
    └── predictions.jld2
```

## 8. Configuration

### 8.1 Weight Initialization (RMS-Based)
```julia
rms[k] = sqrt(mean(abs²(filter[k])))
target_rms = mean(rms)
weight[k] = clamp(target_rms/rms[k], 0, 1)
```

### 8.2 Horizon Configuration
```toml
[prediction]
min_horizon = 100
max_horizon = 2000
optimization_points = [100, 250, 500, 1000, 1500, 2000]
interpolation = "linear"
```

### 8.3 Phase Model Configuration
```toml
[phase_extrapolation]
frequency_source = "design"  # or "pll_vco", "instantaneous"
magnitude_model = "constant"  # or "decay"
phase_model = "linear"        # or "nonlinear"
four_phase_tracking = true
```

## 9. Testing Requirements

### 9.1 Unit Tests
- [x] GA operations (crossover, mutation)
- [x] Filter parameter encoding/decoding
- [x] Energy metrics calculation
- [x] Phase extrapolation accuracy
- [x] Weight optimization convergence

### 9.2 Integration Tests
- [x] End-to-end filter optimization
- [x] Weight optimization with projection
- [x] Multi-instrument handling
- [ ] Real tick data processing
- [ ] Production pipeline

### 9.3 Performance Tests
- [x] <1ms prediction latency
- [x] Memory usage under limits
- [ ] Concurrent instrument optimization
- [ ] Streaming prediction throughput

## 10. Development Phases

### Phase 1: Foundation ✅
- Chunks 1-3: Core GA, storage, filter fitness
- Status: Complete

### Phase 2: Prediction ✅
- Chunk 4: Weight optimization with phase extrapolation
- Status: Complete (Revised in v1.5)

### Phase 3: Integration 🔜
- Chunk 5: System integration
- Chunk 6: Production deployment

### Phase 4: Enhancement 📋
- Multi-instrument parallelization
- Real-time streaming
- Advanced phase models

## Appendix A: Mathematical Details

### A.1 Complex Biquad Transfer Function
```
H(z) = (b₀ + b₁z⁻¹ + b₂z⁻²)/(1 + a₁z⁻¹ + a₂z⁻²)
```

### A.2 Phase Extrapolation Mathematics
```
Given: z(t) = r(t)e^(iφ(t))
Model: φ(t+n) = φ(t) + ωn + θ₄(n)
Where: ω = 2π/T (design frequency)
       θ₄(n) = πn/2 (4-phase rotation)
       
Projection: z(t+n) = r(t)e^(i(φ(t) + ωn + θ₄(n)))
```

### A.3 Prediction Error
```
e(n) = real(Σw[k]·z[k](t+n)) - real(actual(t+n))
MSE = E[e²(n)]
```

## Appendix B: Critical Design Decisions

### B.1 Scalar vs Complex Weights
**Decision**: Scalar weights only
**Rationale**: Preserves phase relationships, simpler optimization

### B.2 Phase Extrapolation
**Decision**: Linear phase model with constant magnitude
**Rationale**: Predicting price, not filter state; avoid decay assumptions

### B.3 Frequency Source
**Decision**: Design frequency (2π/period)
**Rationale**: Stable, predictable, matches filter design

### B.4 Comparison Metric
**Decision**: I-component only
**Rationale**: Price change encoded in real part

## Version History

- **v1.0**: Initial specification
- **v1.1**: Added two-stage optimization
- **v1.2**: Corrected complex weight application
- **v1.3**: Added 4-phase rotation details
- **v1.4**: Scalar weights, RMS initialization
- **v1.5**: Phase-based extrapolation for prediction

---

*End of Specification v1.5 - Phase Extrapolation Complete*