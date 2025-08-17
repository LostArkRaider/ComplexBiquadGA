# TickHotLoopF32.jl - Complete Technical Documentation

## Overview

TickHotLoopF32.jl is an ultra-low-latency tick data processing module designed for financial market data analysis. It reads raw tick data, applies sophisticated cleaning and normalization algorithms, and outputs complex-valued samples suitable for feeding into a Fibonacci PLL (Phase-Locked Loop) filter bank.

### Key Features
- **Real-time tick processing** with minimal latency
- **Robust data cleaning** with multiple defense layers
- **Adaptive normalization** using AGC (Automatic Gain Control)
- **4-phase complex rotation** for signal processing
- **Integer-optimized algorithms** for performance

## Input Data Schema

### Expected File Format
The module expects a semicolon-delimited text file with the following structure:
```
timestamp;field2;field3;last_price_ticks;volume
```

### Field Descriptions
1. **timestamp** - Trading timestamp (kept as substring for efficiency)
2. **field2** - Unused field (typically contains bid price)
3. **field3** - Unused field (typically contains ask price)
4. **last_price_ticks** - Last traded price in ticks (Int32)
5. **volume** - Number of contracts (must be 1 for valid ticks)

### Example Input
```
2024-01-15T09:30:00.123;41250;41251;41251;1
2024-01-15T09:30:00.456;41251;41252;41252;1
2024-01-15T09:30:00.789;41252;41253;41254;1
```

## Output Data Schema

Each processed tick produces a tuple with 5 elements:
```julia
(tick_idx::Int64, ts::SubString, z::ComplexF32, Δ::Int32, flag::UInt8)
```

### Output Fields
- **tick_idx** - Sequential tick counter (1-based)
- **ts** - Original timestamp (as substring)
- **z** - Complex-valued normalized signal
- **Δ** - Cleaned price change in ticks
- **flag** - Audit flags indicating processing actions

## Core Processing Pipeline

### 1. Parse & Validate
```julia
# Parse line into components
parts = split(line, ';')
if length(parts) != 5
    stats_skipped += 1
    continue
end

# Extract and validate fields
ts = SubString(parts[1])
last_ticks = parse(Int32, parts[4])
vol = parse(Int, parts[5])

# Enforce single-contract constraint
if vol != 1
    stats_skipped += 1
    continue
end
```

### 2. Absolute Price Range Check
```julia
# Sanity check against configured bounds
if last_ticks < cfg.min_ticks || last_ticks > cfg.max_ticks
    if last_clean !== nothing
        flag |= HOLDLAST  # 0x01
        stats_holdlast += 1
        # Emit with Δ=0 (hold last value)
        Δ = Int32(0)
        normalized_ratio = Float32(0)
        # Continue with 4-phase rotation...
    end
    continue
end
```

### 3. Delta Calculation & Hard Jump Guard
```julia
# Calculate raw price change from last clean price
raw_delta = last_ticks - last_clean

# Apply hard limit on maximum per-tick movement
Δ = raw_delta
if abs(Δ) > cfg.max_jump_ticks
    # Clamp to maximum allowed jump
    Δ = Δ > 0 ? cfg.max_jump_ticks : -cfg.max_jump_ticks
    flag |= CLAMPED  # 0x02
    stats_clamped += 1
end
```

### 4. Robust EMA Band (Winsorization)
```julia
# Update delta EMAs using integer arithmetic
if !has_delta_ema
    # Initialize on first delta
    ema_delta = Δ
    ema_delta_dev = abs(Δ)
    has_delta_ema = true
else
    # Update EMA(Δ) with α = 2^-a_shift
    delta_diff = Δ - ema_delta
    ema_delta += delta_diff >> cfg.a_shift  # Efficient division by 2^a_shift
    
    # Update EMA(|Δ - emaΔ|) with β = 2^-b_shift
    abs_dev = abs(Δ - ema_delta)
    dev_diff = abs_dev - ema_delta_dev
    ema_delta_dev += dev_diff >> cfg.b_shift
    
    # Prevent zero deviation
    ema_delta_dev = max(ema_delta_dev, Int32(1))
end

# Apply winsorization after warmup period
if ticks_accepted > delta_warmup && has_delta_ema
    lo, hi = band_from_delta_ema(ema_delta, ema_delta_dev, cfg.z_cut)
    if Δ < lo
        Δ = lo
        flag |= WINSORIZED  # 0x04
    elseif Δ > hi
        Δ = hi
        flag |= WINSORIZED  # 0x04
    end
end
```

### 5. AGC Normalization
```julia
# Update slow EMA of absolute delta (envelope follower)
absΔ = abs(Δ)
agc_diff = absΔ - emaAbsΔ
emaAbsΔ += agc_diff >> cfg.b_shift_agc  # β_agc = 2^-6 = 1/64

# Ensure non-zero envelope
emaAbsΔ = max(emaAbsΔ, Int32(1))

# Calculate normalization scale
S_raw = cfg.agc_guard_c * emaAbsΔ  # Guard factor × envelope
S_raw = clamp(S_raw, cfg.agc_Smin, cfg.agc_Smax)

# Normalize price change to ±1 range
normalized_ratio = Float32(Δ) / Float32(S_raw)

# Apply guard factor scaling
# With agc_guard_c=7, typical values are in ±1/7 range
# Scale up to use full ±1 dynamic range
normalized_ratio = normalized_ratio * Float32(cfg.agc_guard_c)

# Final clamp to prevent overshooting
normalized_ratio = clamp(normalized_ratio, -1.0f0, 1.0f0)
```

### 6. 4-Phase Complex Rotation
```julia
# Determine phase position (1,2,3,4) based on tick index
@inline phase_pos_global(tick_idx::Int64)::Int32 = 
    Int32(((tick_idx - 1) & 0x3) + 1)

# Complex multipliers for quadrants: 0°, +90°, 180°, -90°
const QUAD4 = (
    ComplexF32(1,0),   # 0°:   1
    ComplexF32(0,1),   # 90°:  i
    ComplexF32(-1,0),  # 180°: -1
    ComplexF32(0,-1)   # 270°: -i
)

# Apply rotation to normalized value
@inline function apply_quad_phase(normalized_value::Float32, pos::Int32)::ComplexF32
    q = QUAD4[pos]
    return ComplexF32(
        normalized_value * real(q), 
        normalized_value * imag(q)
    )
end

# In processing pipeline:
pos = phase_pos_global(tick_idx)
z = apply_quad_phase(normalized_ratio, pos)
```

## Configuration Parameters

### CleanCfgInt Structure
```julia
Base.@kwdef mutable struct CleanCfgInt
    # Absolute price bounds (market-specific)
    min_ticks::Int32              # Minimum viable price (e.g., 40000 for YM)
    max_ticks::Int32              # Maximum viable price (e.g., 43000 for YM)
    
    # Jump protection
    max_jump_ticks::Int32 = 50   # Max allowed single-tick move
    
    # EMA parameters for robust band
    a_shift::Int = 4              # α = 2^-4 = 1/16 for EMA(Δ)
    b_shift::Int = 4              # β = 2^-4 = 1/16 for EMA(|Δ-emaΔ|)
    
    # Winsorization width
    z_cut::Float32 = 7f0          # Robust z-score multiplier (≈ 1.253 * MAD * z_cut)
    
    # AGC (normalization) parameters
    b_shift_agc::Int = 6          # β_agc = 2^-6 = 1/64 (slow envelope)
    agc_guard_c::Int32 = 7        # Guard/headroom factor
    agc_Smin::Int32 = 4           # Minimum scale (prevents over-amplification)
    agc_Smax::Int32 = 50          # Maximum scale (synchronized with max_jump_ticks)
end
```

## Audit Flags

The system uses bit flags to track processing actions:

```julia
const HOLDLAST   = 0x01  # Price outside bounds → held last (Δ=0)
const CLAMPED    = 0x02  # Exceeded jump guard → clamped to max_jump_ticks
const WINSORIZED = 0x04  # Soft clamp to robust EMA band
```

## Key Algorithms

### Robust Band Calculation
```julia
@inline function band_from_delta_ema(ema_delta::Int32, ema_delta_dev::Int32, zcut::Float32)
    # Ensure minimum deviation
    safe_dev = max(Int32(1), ema_delta_dev)
    
    # Integer approximation: 1.253 ≈ 5/4 = 1.25
    z_int = Int32(zcut)
    w = (Int32(5) * z_int * safe_dev) >> 2  # Multiply by 5/4
    
    # Minimum band width
    w = max(w, Int32(2))
    
    # Calculate band limits
    lo = ema_delta - w
    hi = ema_delta + w
    
    return lo, hi
end
```

### Next Power of Two
```julia
@inline function next_pow2_i32(x::Int32)::Int32
    if x <= 1
        return Int32(1)
    end
    if x >= Int32(1) << 30  # Prevent overflow
        return Int32(1) << 30
    end
    
    # Bit manipulation to find next power of 2
    ux = UInt32(x - 1)
    ux |= ux >> 1
    ux |= ux >> 2
    ux |= ux >> 4
    ux |= ux >> 8
    ux |= ux >> 16
    return Int32(ux + 0x00000001)
end
```

## Main Entry Points

### stream_complex_ticks_f32
Creates a Channel that streams processed tick data:
```julia
function stream_complex_ticks_f32(tickfile::AbstractString, cfg::CleanCfgInt)
    return Channel{Tuple{Int64,SubString{String},ComplexF32,Int32,UInt8}}(256) do ch
        # Initialize state variables
        # Process file line by line
        # Emit processed ticks to channel
    end
end
```

### run_from_ticks_f32
High-level runner that integrates with external configuration system:
```julia
function run_from_ticks_f32(config_name::AbstractString,
                           tickfile::AbstractString;
                           init_bank::Function,      # Filter bank constructor
                           on_tick::Function,         # Tick handler
                           cfg_clean::Union{Nothing,CleanCfgInt}=nothing)
    
    # Load configuration
    config = Main.ModernConfigSystem.load_filter_config(config_name)
    
    # Build cleaning config (use defaults if not provided)
    c = cfg_clean === nothing ? CleanCfgInt(...) : cfg_clean
    
    # Initialize filter bank
    bank = init_bank(config)
    
    # Stream and process ticks
    for rec in stream_complex_ticks_f32(tickfile, c)
        on_tick(rec, config, bank)
    end
end
```

## Performance Optimizations

### Integer Arithmetic
- All delta calculations use Int32 to avoid floating-point overhead
- EMA updates use bit shifts instead of division (e.g., `>> 4` instead of `/16`)
- Band calculations use integer approximations (e.g., `5/4` for `1.253`)

### Memory Efficiency
- Timestamps kept as SubString to avoid string allocation
- Channel buffering (256 elements) for smooth streaming
- Single-pass processing with minimal state

### Computational Efficiency
- Inline functions for hot path operations
- Bit manipulation for phase position calculation
- Power-of-two operations optimized with bit tricks

## Statistical Tracking

The module maintains running statistics:
- **ticks_processed** - Total lines read
- **ticks_accepted** - Valid ticks emitted
- **stats_skipped** - Invalid format or volume ≠ 1
- **stats_holdlast** - Prices outside valid range
- **stats_clamped** - Hard jump guard triggered
- **stats_winsorized** - Soft band clipping applied

Progress reports every 100,000 ticks show:
- Processing counts
- Current EMA values
- AGC scale factor

## Error Handling

### Parse Errors
- Silently skip malformed lines
- Increment skip counter for statistics

### Range Violations
- Hold last valid price when outside bounds
- Set HOLDLAST flag for audit trail

### First Tick Initialization
- Emit with Δ=0 until baseline established
- No flags set on initialization

## Integration Points

### Required External Module
- `Main.ModernConfigSystem` - Configuration loader

### Callback Functions
- `init_bank(config)` - Initialize filter bank
- `on_tick(rec, config, bank)` - Process each tick

### Export Interface
- `run_from_ticks_f32` - Main entry point
- `stream_complex_ticks_f32` - Low-level streamer
- `CleanCfgInt` - Configuration structure

## Typical Usage Example

```julia
using TickHotLoopF32

# Define configuration
cfg = CleanCfgInt(
    min_ticks = 40000,
    max_ticks = 43000,
    max_jump_ticks = 50,
    z_cut = 7.0f0
)

# Process tick file
run_from_ticks_f32(
    "my_config",
    "ticks_20240115.txt",
    init_bank = my_filter_bank_init,
    on_tick = my_tick_handler,
    cfg_clean = cfg
)
```

## Design Philosophy

1. **Robustness First** - Multiple layers of defense against bad data
2. **Performance Critical** - Integer math and bit operations where possible
3. **Market Adaptive** - AGC adjusts to changing volatility
4. **Signal Integrity** - Proper normalization preserves price/volume relationship
5. **Audit Trail** - Flags track every processing decision

## Mathematical Foundation

### Normalization Concept
The module treats each tick as a price change relative to volume:
- Price change (Δ) represents the signal
- Volume (always 1) represents the reference
- Normalization scales Δ to ±1 range based on recent volatility
- 4-phase rotation distributes signal energy across complex plane

### EMA Formulation
Using powers of 2 for EMA coefficients enables shift operations:
- `EMA[n] = EMA[n-1] + (X[n] - EMA[n-1]) >> shift`
- Equivalent to `α = 2^-shift` in traditional EMA
- No floating-point operations required

### Robust Statistics
The winsorization band approximates:
- Center: EMA of deltas
- Width: z_cut × 1.253 × MAD
- MAD approximated by EMA(|Δ - EMA(Δ)|)

This provides outlier resistance while maintaining responsiveness to market regime changes.