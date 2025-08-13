# GA Optimization System for ComplexBiquad PLL Filter Bank - Specification

## 1. Executive Overview

### 1.1 Project Purpose
This specification defines a Genetic Algorithm (GA) optimization system for a ComplexBiquad PLL filter bank that processes real YM futures tick data. The system optimizes both filter parameters and prediction weights to forecast price changes at future tick indices.

### 1.2 Core Innovation
The system treats market data as superposed rotating phasors extracted by Fibonacci-period filters. Each filter output represents a complex rotating vector that can be extrapolated to future time points. By optimizing real-valued weights that combine these phasor predictions while preserving their phase information, the system achieves accurate long-range price forecasting.

### 1.3 Two-Stage Optimization Architecture

**Stage 1: Filter/PLL Parameter Optimization**
- Optimizes Q factors, PLL gains, loop bandwidths, lock thresholds
- Goal: Extract clean, stable rotating phasors from noisy market data
- Uses existing ProductionFilterBank and ModernConfigSystem modules
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

### 1.5 Existing Codebase Integration

The system leverages four existing modules:
- **TickHotLoopF32.jl**: Ultra-low-latency tick processing to ComplexF32 signals
- **ProductionFilterBank.jl**: ComplexBiquad and PLLFilterState implementations
- **ModernConfigSystem.jl**: Type-safe TOML-based configuration management
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

## 2. Development Chunks

The project is divided into six independent mini-projects, each building on the previous while maintaining standalone functionality.

---

## 3. Chunk 1: Filter Parameter GA Core

### 3.1 Purpose
Implement the genetic algorithm infrastructure for optimizing filter and PLL parameters. This establishes the foundation for Stage 1 optimization.

### 3.2 Dependencies
- ModernConfigSystem.jl (for FilterConfig and ExtendedFilterConfig structures)
- ProductionFilterBank.jl (for parameter constraints and validation)

### 3.3 Deliverables

**FilterParameterGA.jl Module:**
- Chromosome encoding for filter/PLL parameters
- Genetic operators (crossover, mutation) respecting parameter constraints
- Population management with GPU-ready batch structure
- Integration with ModernConfigSystem for parameter persistence

**Key Components:**
- Parameter chromosome mapping to ExtendedFilterConfig fields
- Constraint enforcement (Q factor ranges, PLL stability limits)
- Batch evaluation structure for future GPU parallelization
- TOML serialization of optimal configurations

### 3.4 Validation
- Parameter ranges remain within ModernConfigSystem constraints
- Mutated parameters maintain filter stability criteria
- Crossover preserves parameter relationships
- Population diversity metrics

### 3.5 Success Criteria
- GA can generate valid FilterConfig/ExtendedFilterConfig instances
- Genetic operators maintain parameter validity
- Population evolution framework established
- Batch structure supports parallel evaluation

---

## 4. Chunk 2: Filter Fitness Evaluation

### 4.1 Purpose
Implement fitness evaluation for filter parameters using real tick data, measuring signal extraction quality without requiring prediction accuracy.

### 4.2 Dependencies
- FilterParameterGA.jl (from Chunk 1)
- ProductionFilterBank.jl (for filter processing)
- TickHotLoopF32.jl (for real tick data)

### 4.3 Deliverables

**FilterFitnessEvaluator.jl Module:**
- Signal quality metrics (SNR, spectral purity, stability)
- PLL lock quality assessment
- Frequency selectivity measurement
- Batch fitness evaluation pipeline

**Key Metrics:**
- Signal-to-noise ratio per filter
- PLL lock quality and stability over time
- Frequency response sharpness (Q factor effectiveness)
- Phase coherence and angular velocity stability
- Ringing suppression and transient response

### 4.4 Validation
- Process sample tick data through filter configurations
- Verify fitness metrics correlate with signal quality
- Validate batch evaluation produces consistent results
- Test with both standard and PLL-enhanced configurations

### 4.5 Success Criteria
- Fitness function differentiates good from poor parameters
- Evaluation completes in reasonable time for large populations
- Metrics provide actionable feedback for GA evolution
- System handles real tick data robustly

---

## 5. Chunk 3: Phasor Tracking Module

### 5.1 Purpose
Extract and track magnitude, phase, and angular velocity from filter outputs, preparing for phasor-based prediction.

### 5.2 Dependencies
- Optimized filter configurations (from Chunks 1-2)
- ProductionFilterBank.jl (for filter outputs)
- TickHotLoopF32.jl (for continuous tick processing)

### 5.3 Deliverables

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

### 5.4 Validation
- Verify phase unwrapping handles 2π discontinuities
- Validate angular velocity estimation accuracy
- Test phasor stability under various market conditions
- Confirm magnitude tracking follows envelope correctly

### 5.5 Success Criteria
- Smooth, continuous phase tracking without jumps
- Accurate angular velocity estimation
- Reliable detection of stable vs unstable phasors
- Efficient processing of tick streams

---

## 6. Chunk 4: Phasor Prediction Engine

### 6.1 Purpose
Implement phasor extrapolation to future tick indices and combine predictions using real-valued weights.

### 6.2 Dependencies
- PhasorTracker.jl (from Chunk 3)
- Filter configurations and outputs

### 6.3 Deliverables

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

### 6.4 Validation
- Verify phase preservation with real weights
- Test extrapolation accuracy on synthetic signals
- Validate vector sum produces reasonable price predictions
- Confirm predictions degrade gracefully with horizon

### 6.5 Success Criteria
- Accurate phasor extrapolation for stable signals
- Phase information preserved through weighting
- Reasonable price predictions at various horizons
- Computational efficiency for real-time use

---

## 7. Chunk 5: Weight Optimization GA

### 7.1 Purpose
Implement genetic algorithm for optimizing real-valued prediction weights at multiple horizons.

### 7.2 Dependencies
- PhasorPredictor.jl (from Chunk 4)
- FilterParameterGA.jl (for GA infrastructure)
- Historical tick data for training

### 7.3 Deliverables

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

### 7.4 Validation
- Verify weights remain real-valued throughout evolution
- Test prediction accuracy improvement over generations
- Validate different horizons receive appropriate weights
- Confirm convergence to stable weight configurations

### 7.5 Success Criteria
- Prediction accuracy improves significantly over random weights
- Different horizons develop distinct weight patterns
- GA converges reliably within reasonable generations
- Weights generalize to out-of-sample data

---

## 8. Chunk 6: Integration and Horizon Discovery

### 8.1 Purpose
Integrate all components and empirically determine maximum useful prediction horizons through systematic testing.

### 8.2 Dependencies
- All previous chunks
- Extended historical tick data for validation

### 8.3 Deliverables

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

### 8.4 Validation
- End-to-end testing on historical data
- Walk-forward analysis to prevent overfitting
- Comparison with baseline prediction methods
- Stress testing under various market conditions

### 8.5 Success Criteria
- Complete system processes tick data to predictions
- Clear identification of maximum useful prediction horizon
- Performance metrics meet practical trading requirements
- System maintains stability over extended operation

---

## 9. GPU-Ready Design Patterns

### 9.1 Architecture Principles
While GPU implementation is not part of MVP, the design incorporates patterns for future acceleration:

**Batch Processing:**
- Population-parallel GA evaluation
- Multiple chromosomes evaluated simultaneously
- Vectorized fitness calculations

**Memory Layout:**
- Coalesced access patterns for filter states
- Structure-of-arrays for phasor data
- Minimal host-device transfer requirements

**Computational Kernels:**
- Filter processing as parallel kernel per filter
- Phasor extrapolation as vectorized operation
- Weight application as matrix multiplication

### 9.2 CUDA.jl Preparation
- Data structures aligned for GPU memory
- Algorithms expressed as map/reduce operations
- Minimal branching in core loops
- Pre-allocated memory buffers

---

## 10. Performance Targets

### 10.1 MVP Performance Goals
- Filter optimization: < 1 hour for 100 generations, population 50
- Weight optimization: < 30 minutes per horizon
- Tick processing: > 10,000 ticks/second
- Prediction generation: < 1ms per horizon

### 10.2 Accuracy Targets
- Signal extraction: SNR > 20dB for primary frequencies
- PLL lock quality: > 70% average across filters
- Prediction accuracy: Directional accuracy > 55% at 100-tick horizon
- Horizon discovery: Identify useful range within 5% accuracy

---

## 11. Risk Mitigation

### 11.1 Technical Risks
- **Filter instability**: Implement stability checks in GA constraints
- **Phase unwrapping errors**: Use robust unwrapping algorithms
- **Overfitting**: Employ regularization and walk-forward validation
- **Computational bottlenecks**: Profile and optimize critical paths

### 11.2 Data Risks
- **Tick data quality**: Implement robust cleaning in TickHotLoopF32
- **Market regime changes**: Detect and adapt to regime shifts
- **Sparse data periods**: Handle low-activity periods gracefully

---

## 12. Module Structure

### 12.1 Directory Layout
```
fibonacci_pll_filter_bank/
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

### 12.2 Module Dependencies Graph
```
TickHotLoopF32 ──────┐
                     ├──> FilterFitnessEvaluator ──> FilterParameterGA
ProductionFilterBank ┘                                      │
                                                           ↓
ModernConfigSystem ←──────────────────────────── [Optimized Configs]
        │                                                   │
        └──> PhasorTracker ──> PhasorPredictor ──> WeightOptimizationGA
                                      │                     │
                                      └─────────┬───────────┘
                                                ↓
                                        PredictionSystem
                                                │
                                                ↓
                                    [Predictions & Analysis]
```

### 12.3 Configuration File Structure

**Example: config/ga/filter_ga_default.toml**
```toml
[ga_parameters]
population_size = 50
generations = 100
mutation_rate = 0.1
crossover_rate = 0.8
elitism_count = 5
tournament_size = 3

[chromosome]
encoding = "real"
length = 27  # 9 filters × 3 params (Q, gain, bandwidth)
bounds_q = [0.5, 10.0]
bounds_gain = [0.001, 1.0]
bounds_bandwidth = [0.0001, 0.1]

[fitness]
metrics = ["snr", "lock_quality", "stability"]
weights = [0.4, 0.4, 0.2]
evaluation_ticks = 10000
parallel_evaluations = true
```

**Example: config/prediction/horizons_standard.toml**
```toml
[horizons]
test_points = [100, 200, 500, 1000, 2000]
max_horizon = 5000
validation_split = 0.2

[weights]
initialization = "random"
range = [-10.0, 10.0]
regularization = 0.001
```

## 13. Data Specifications

### 13.1 Tick Data Format
Input data from TickHotLoopF32 processing:
- **Raw format**: CSV with timestamp;bid;ask;last;volume
- **Processed format**: ComplexF32 where real = price change, imag = volume
- **Timing**: ~0.8056 seconds per tick (market average)
- **Volume**: Filtered to single-contract trades only

### 13.2 Data Requirements
- **Training**: Minimum 500,000 ticks (~5 trading days)
- **Validation**: 20% of training data size
- **Test**: Separate out-of-sample period
- **Quality**: Pre-cleaned via TickHotLoopF32 robust EMA bands

## 14. Fitness Function Specifications

### 14.1 Stage 1: Filter Quality Metrics

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

### 14.2 Stage 2: Prediction Accuracy Metrics

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

## 15. GA Parameter Recommendations

### 15.1 Filter Optimization GA

| Parameter | Range | Default | Notes |
|-----------|-------|---------|-------|
| Population Size | 30-100 | 50 | Balance diversity vs computation |
| Generations | 50-200 | 100 | Early stopping if converged |
| Mutation Rate | 0.05-0.20 | 0.10 | Adaptive reduction over time |
| Crossover Rate | 0.60-0.90 | 0.80 | Arithmetic crossover preferred |
| Elitism | 5-10% | 5 | Preserve best configurations |
| Tournament Size | 2-5 | 3 | Selection pressure control |

### 15.2 Weight Optimization GA

| Parameter | Range | Default | Notes |
|-----------|-------|---------|-------|
| Population Size | 50-200 | 100 | Larger for weight complexity |
| Generations | 100-500 | 200 | Longer for fine-tuning |
| Mutation Rate | 0.01-0.10 | 0.05 | Lower for weight stability |
| Weight Range | [-10, 10] | [-5, 5] | Prevent extreme values |
| Regularization | 0.0001-0.01 | 0.001 | L2 penalty on weights |

## 16. Testing Strategy

### 16.1 Unit Testing
Each module requires comprehensive unit tests:

**FilterParameterGA:**
- Chromosome validity after genetic operations
- Constraint preservation during mutation
- Crossover producing valid offspring
- Population diversity maintenance

**PhasorTracker:**
- Phase unwrapping correctness
- Angular velocity estimation accuracy
- Magnitude tracking stability
- Edge case handling (zero crossings, discontinuities)

**WeightOptimizationGA:**
- Real-value constraint enforcement
- Weight matrix dimension consistency
- Convergence behavior verification
- Regularization effectiveness

### 16.2 Integration Testing

**End-to-End Pipeline:**
- Tick data → Filter processing → Phasor extraction → Prediction
- Verify no phase corruption through pipeline
- Validate prediction at multiple horizons
- Confirm weight optimization improves accuracy

**Performance Benchmarks:**
- Tick processing rate > 10,000 ticks/second
- GA generation time < 30 seconds for population 50
- Prediction generation < 1ms per horizon
- Memory usage < 4GB for standard configuration

### 16.3 Validation Testing

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

## 17. Implementation Timeline

### 17.1 Development Schedule (6-8 weeks)

**Week 1-2: Chunk 1 - Filter Parameter GA**
- Design chromosome encoding
- Implement genetic operators
- Create population management
- Unit testing

**Week 2-3: Chunk 2 - Filter Fitness Evaluation**
- Implement signal quality metrics
- Integrate with ProductionFilterBank
- Process sample tick data
- Validate fitness differentiation

**Week 3-4: Chunk 3 - Phasor Tracking**
- Extract magnitude/phase/frequency
- Implement phase unwrapping
- Track phasor stability
- Test with real data

**Week 4-5: Chunk 4 - Phasor Prediction**
- Implement extrapolation algorithm
- Apply real-weight scaling
- Vector summation logic
- Validate phase preservation

**Week 5-6: Chunk 5 - Weight Optimization**
- Design weight chromosome
- Implement prediction fitness
- Multi-horizon optimization
- Convergence testing

**Week 6-8: Chunk 6 - Integration**
- Complete pipeline assembly
- Horizon discovery analysis
- Performance optimization
- Documentation and cleanup

### 17.2 Milestones

1. **Filter GA Complete**: Optimized filter parameters achieved
2. **Phasor Extraction Working**: Stable phasor tracking from real data
3. **Predictions Generated**: First price predictions produced
4. **Weights Optimized**: Improved accuracy via weight evolution
5. **MVP Complete**: Full pipeline with discovered optimal horizon

## 18. Future Extensions (Post-MVP)

### 13.1 Advanced Features
- Adaptive filter bank (dynamic period selection)
- Online learning for weight updates
- Multi-asset correlation exploitation
- Regime-specific weight sets

### 13.2 GPU Implementation
- CUDA.jl kernel development
- Multi-GPU scaling
- Real-time streaming processing
- Cloud-based distributed evolution

### 13.3 Trading Integration
- Risk-adjusted position sizing
- Transaction cost optimization
- Portfolio-level optimization
- Real-time execution framework