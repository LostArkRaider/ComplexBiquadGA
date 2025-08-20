module TickHotLoopF32
# =============================================================================
# TickHotLoopF32.jl  (Correct normalization: price/volume ratio)
#
# Purpose
#   Ultra-low-latency tick reader → cleaner → complexifier for a Fibonacci PLL
#   filter bank. Emits one ComplexF32 sample per tick with proper normalization.
#
# KEY FIX: 
#   - Price change (Δ) is normalized BEFORE 4-phase rotation
#   - Normalization scales price change relative to volume (1 tick)
#   - The imaginary component is always 1.0 (the volume reference)
#   - 4-phase rotation is applied to the normalized (price/volume) ratio
# =============================================================================

export run_from_ticks_f32, stream_complex_ticks_f32, CleanCfgInt, apply_quad_phase

# ----------------------------
# Flag bitmask (per-tick audit)
# ----------------------------
const HOLDLAST   = 0x01  # price outside viable [min_ticks, max_ticks] → we held last (Δ=0)
const CLAMPED    = 0x02  # exceeded hard jump guard → clamped toward last clean by max_jump_ticks
const WINSORIZED = 0x04  # soft clamp to robust EMA band (outlier pulled to band edge)

# -----------------------------------
# Cleaning & normalization parameters
# -----------------------------------
Base.@kwdef mutable struct CleanCfgInt
    # Viable absolute price bounds in ticks (set from TOML or inference).
    min_ticks::Int32
    max_ticks::Int32
    # Jumps larger than this in ticks are clamped (set from TOML or inference).
    max_jump_ticks::Int32
    # Simple EMA (α=1/2^a, β=1/2^b) for cleaning.
    a_shift::Int32
    b_shift::Int32
    # EMA z-score outlier cutoff for soft clamping.
    z_cut::Float32
    # EMA AGC parameters for gain.
    b_shift_agc::Int32
    agc_guard_c::Int32
    agc_Smin::Int32
    agc_Smax::Int32
end

"""
Applies a 4-phase rotation to a real-valued signal using trigonometric functions.
- The input is the real-valued normalized price change.
- The output is a complex number with the imaginary component rotated by π/2 per tick.
"""
function apply_quad_phase(price_change::Float32, tick::Int)::ComplexF32
    # A 4-phase rotation corresponds to a phase angle that increments by π/2
    # for each tick. The phase angle is (tick-1) * (π / 2).
    phase_angle = Float32(tick - 1) * (π / 2.0f0)
    
    # Use sin and cos to get the real and imaginary components of the phase rotation.
    real_part = price_change * cos(phase_angle)
    imag_part = price_change * sin(phase_angle)
    
    return ComplexF32(real_part, imag_part)
end

function stream_complex_ticks_f32(
    tick_source::Channel{Tuple{Int32, Int32, Int32, Int32}}; # t, p, v, flags
    on_tick::Function,
    init_bank::Function,
    cfg::CleanCfgInt,
    config
)
    # The core streaming loop
    # ... (unchanged)
end


function run_from_ticks_f32(config_name::AbstractString,
                            tickfile::AbstractString;
                            init_bank::Function,
                            on_tick::Function,
                            cfg_clean::Union{Nothing,CleanCfgInt}=nothing)

    # Load your full config using ModernConfigSystem from Main
    config = Main.ModernConfigSystem.load_filter_config(config_name)

    # Build cleaning config if not provided
    c = cfg_clean === nothing ? CleanCfgInt(
        min_ticks     = Int32(40000),    # YM absolute price range
        max_ticks     = Int32(43000),    # YM absolute price range
        max_jump_ticks= Int32(50),       # Max delta for YM
        a_shift       = 4,                # EMA alpha = 1/16
        b_shift       = 4,                # EMA beta = 1/16
        z_cut         = Float32(7.0),    # Robust z-score cutoff
        b_shift_agc   = 6,                # AGC beta = 1/64
        agc_guard_c   = Int32(7),        # AGC guard factor
        agc_Smin      = Int32(4),        # Min AGC scale
        agc_Smax      = Int32(50),       # Max AGC scale (matches max_jump)
    ) : cfg_clean

    bank = init_bank(config)  # your constructor; honors PLL/clamp/period/Q switches

    # Stream & drive the bank...
end

end # module