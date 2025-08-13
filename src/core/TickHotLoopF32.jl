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

export run_from_ticks_f32, stream_complex_ticks_f32, CleanCfgInt

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

    # Hard jump guard: max allowed per-tick move (ticks) before clamping.
    max_jump_ticks::Int32 = 50  # Increased default for YM

    # EMA parameters for soft robust band (winsorization) on DELTAS.
    # α = 2^-a_shift for EMA(Δ), β = 2^-b_shift for EMA(|Δ-emaΔ|).
    a_shift::Int = 4
    b_shift::Int = 4

    # Robust z width multiplier for the EMA band (≈ 1.253 * MAD * z_cut).
    z_cut::Float32 = 7f0

    # Normalization (AGC) params (integer-friendly).
    # emaAbsΔ is a SLOW EMA of |Δ|; scale S_raw = c * emaAbsΔ is clamped to [Smin, Smax],
    # then rounded UP to the next power-of-two (S2) so Δ/S2 is a shift, not a divide.
    b_shift_agc::Int = 6               # β_agc = 1/64 → slow, stable envelope for PD
    agc_guard_c::Int32 = 7             # guard factor (typ. 6..8)
    agc_Smin::Int32 = 4                # min scale in ticks (prevents over-amplification)
    agc_Smax::Int32 = 50               # default cap for YM; synced at runtime
end

# ----------------------------
# Robust EMA band for DELTAS (integer math, no rounding)
# ----------------------------
@inline function band_from_delta_ema(ema_delta::Int32, ema_delta_dev::Int32, zcut::Float32)
    # EMA(|Δ-emaΔ|) as MAD proxy; ensure ema_delta_dev is at least 1
    safe_dev = max(Int32(1), ema_delta_dev)
    
    # Calculate band width using integer approximation
    # 1.253 ≈ 5/4 = 1.25, close enough for our purposes
    z_int = Int32(zcut)  # Truncate z_cut
    w = (Int32(5) * z_int * safe_dev) >> 2  # Divide by 4 using shift
    
    # Ensure minimum band width for deltas
    w = max(w, Int32(2))  # Minimum band of ±2 ticks for price changes
    
    # Band limits for delta (much smaller than absolute prices)
    lo = ema_delta - w
    hi = ema_delta + w
    
    return lo, hi
end

# ----------------------------
# Next power-of-two helper (integer only)
# ----------------------------
@inline function next_pow2_i32(x::Int32)::Int32
    if x <= 1
        return Int32(1)
    end
    if x >= Int32(1) << 30  # Prevent overflow
        return Int32(1) << 30
    end
    ux = UInt32(x - 1)
    ux |= ux >> 1; ux |= ux >> 2; ux |= ux >> 4; ux |= ux >> 8; ux |= ux >> 16
    return Int32(ux + 0x00000001)
end

# ----------------------------
# Global 4-phase rotation helpers
# ----------------------------
# Position ∈ {1,2,3,4} based on sequential tick_idx (1-based), never resets.
@inline phase_pos_global(tick_idx::Int64)::Int32 = Int32(((tick_idx - 1) & 0x3) + 1)

# Unit complex multipliers for the four quadrants: 0°, +90°, 180°, −90°
const QUAD4 = (ComplexF32(1,0), ComplexF32(0,1), ComplexF32(-1,0), ComplexF32(0,-1))

# Apply quadrant rotation to normalized value (price_change/volume ratio)
@inline function apply_quad_phase(normalized_value::Float32, pos::Int32)::ComplexF32
    q = QUAD4[pos]  # q ∈ {1, i, −1, −i}
    return ComplexF32(normalized_value * real(q), normalized_value * imag(q))
end

# ---------------------------------------------------------
# Core streamer: parse → clean → normalize → 4-phase ComplexF32 out
# ---------------------------------------------------------
function stream_complex_ticks_f32(tickfile::AbstractString, cfg::CleanCfgInt)
    return Channel{Tuple{Int64,SubString{String},ComplexF32,Int32,UInt8}}(256) do ch
        # -------- Persistent state (hot) --------
        tick_idx::Int64 = 0
        ticks_processed::Int64 = 0
        ticks_accepted::Int64 = 0

        # EMA state for DELTA robust band
        ema_delta::Int32 = 0       # EMA of price changes
        ema_delta_dev::Int32 = 1   # EMA of |Δ - emaΔ|
        has_delta_ema = false
        delta_warmup = 20  # Warmup period before applying winsorization

        # AGC (amplitude normalization) state: slow EMA of |Δ|
        emaAbsΔ::Int32 = 1  # start at 1 to avoid division by zero

        # Last accepted clean price in ticks
        last_clean::Union{Nothing,Int32} = nothing

        # Sync AGC cap with hard guard
        if cfg.agc_Smax != cfg.max_jump_ticks
            cfg.agc_Smax = cfg.max_jump_ticks
        end

        # Statistics tracking
        stats_holdlast = 0
        stats_clamped = 0
        stats_winsorized = 0
        stats_skipped = 0

        # -------- Stream the file, one line at a time --------
        for line in eachline(tickfile)
            ticks_processed += 1
            
            # Parse with minimal work; ignore malformed lines
            parts = split(line, ';')
            if length(parts) != 5
                stats_skipped += 1
                continue
            end
            
            ts = SubString(parts[1])  # Keep as substring for efficiency
            last_ticks = try 
                parse(Int32, parts[4]) 
            catch
                stats_skipped += 1
                continue 
            end
            vol = try 
                parse(Int, parts[5]) 
            catch
                stats_skipped += 1
                continue 
            end
            
            if vol != 1  # enforce one contract per record
                stats_skipped += 1
                continue
            end

            tick_idx += 1
            flag::UInt8 = 0x00

            # ---- 1) Absolute price range check (sanity check) ----
            if last_ticks < cfg.min_ticks || last_ticks > cfg.max_ticks
                # Price way out of range - hold last if we have one
                if last_clean !== nothing
                    flag |= HOLDLAST
                    stats_holdlast += 1
                    # Emit with Δ=0
                    Δ = Int32(0)
                    # Normalized price/volume ratio = 0/1 = 0
                    normalized_ratio = Float32(0)
                    pos = phase_pos_global(tick_idx)
                    z = apply_quad_phase(normalized_ratio, pos)
                    put!(ch, (tick_idx, ts, z, Δ, flag))
                    ticks_accepted += 1
                end
                continue
            end
            
            # ---- 2) Initialize on first good tick ----
            if last_clean === nothing
                last_clean = last_ticks
                # Emit first tick with Δ=0
                Δ = Int32(0)
                normalized_ratio = Float32(0)
                pos = phase_pos_global(tick_idx)
                z = apply_quad_phase(normalized_ratio, pos)
                put!(ch, (tick_idx, ts, z, Δ, flag))
                ticks_accepted += 1
                continue
            end
            
            # ---- 3) Compute raw delta ----
            raw_delta = last_ticks - last_clean
            
            # ---- 4) Hard jump guard on DELTA ----
            Δ = raw_delta
            if abs(Δ) > cfg.max_jump_ticks
                # Clamp delta to max_jump_ticks
                Δ = Δ > 0 ? cfg.max_jump_ticks : -cfg.max_jump_ticks
                flag |= CLAMPED
                stats_clamped += 1
            end
            
            # ---- 5) Update DELTA EMAs (integer math) ----
            if !has_delta_ema
                # Initialize delta EMAs
                ema_delta = Δ
                ema_delta_dev = abs(Δ)
                has_delta_ema = true
            else
                # Update EMA(Δ) with α = 2^-a_shift
                delta_diff = Δ - ema_delta
                ema_delta += delta_diff >> cfg.a_shift
                
                # Update EMA(|Δ - emaΔ|) with β = 2^-b_shift
                abs_dev = abs(Δ - ema_delta)
                dev_diff = abs_dev - ema_delta_dev
                ema_delta_dev += dev_diff >> cfg.b_shift
                
                # Ensure ema_delta_dev doesn't go to zero
                ema_delta_dev = max(ema_delta_dev, Int32(1))
            end
            
            # ---- 6) Winsorization on DELTA (after warmup) ----
            if ticks_accepted > delta_warmup && has_delta_ema
                lo, hi = band_from_delta_ema(ema_delta, ema_delta_dev, cfg.z_cut)
                if Δ < lo
                    Δ = lo
                    flag |= WINSORIZED
                    stats_winsorized += 1
                elseif Δ > hi
                    Δ = hi
                    flag |= WINSORIZED
                    stats_winsorized += 1
                end
            end
            
            # ---- 7) Update AGC envelope ----
            absΔ = abs(Δ)
            agc_diff = absΔ - emaAbsΔ
            emaAbsΔ += agc_diff >> cfg.b_shift_agc
            
            # Ensure emaAbsΔ doesn't go to zero
            emaAbsΔ = max(emaAbsΔ, Int32(1))
            
            # ---- 8) Normalize price change to ±1 range ----
            # The AGC scale represents the typical magnitude of price changes
            # We want to map ±S_raw to ±1.0 for proper signal strength
            
            # S_raw is the expected scale of price changes
            S_raw = cfg.agc_guard_c * emaAbsΔ
            S_raw = clamp(S_raw, cfg.agc_Smin, cfg.agc_Smax)
            
            # Normalize so that a price change of ±S_raw maps to ±1.0
            # This ensures typical price movements produce strong signals
            normalized_ratio = Float32(Δ) / Float32(S_raw)
            
            # The guard factor (agc_guard_c) acts as a headroom multiplier
            # With agc_guard_c=7, we expect most values in ±1/7 range
            # So we scale up to use the full ±1 range
            normalized_ratio = normalized_ratio * Float32(cfg.agc_guard_c)
            
            # Clamp to [-1, 1] to prevent outliers from overshooting
            normalized_ratio = clamp(normalized_ratio, -1.0f0, 1.0f0)
            
            # ---- 9) Apply 4-phase rotation to the normalized ratio ----
            pos = phase_pos_global(tick_idx)
            z = apply_quad_phase(normalized_ratio, pos)
            
            # ---- 10) Emit complex sample ----
            put!(ch, (tick_idx, ts, z, Δ, flag))
            ticks_accepted += 1
            
            # ---- 11) Update last_clean with the actual new price ----
            # Important: use original price plus cleaned delta
            last_clean = last_clean + Δ
            
            # Progress reporting every 100k ticks
            if ticks_processed % 100000 == 0
                println("   Processed $(ticks_processed) ticks, accepted $(ticks_accepted)")
                println("   Stats: skip=$stats_skipped, hold=$stats_holdlast, clamp=$stats_clamped, winsor=$stats_winsorized")
                println("   Delta EMA: $ema_delta ± $ema_delta_dev, AGC scale: $(S_raw)")
            end
        end
        
        # Final statistics
        println("\nTick processing complete:")
        println("  Total lines: $ticks_processed")
        println("  Skipped: $stats_skipped")
        println("  Accepted: $ticks_accepted")
        println("  Holdlast: $stats_holdlast")
        println("  Clamped: $stats_clamped")
        println("  Winsorized: $stats_winsorized")
        println("  Final delta EMA: $ema_delta ± $ema_delta_dev")
    end
end

# ---------------------------------------------------------
# Runner: loads your config, builds bank, streams to on_tick
# ---------------------------------------------------------
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

    # Stream & drive the bank
    for rec in stream_complex_ticks_f32(tickfile, c)
        # rec = (tick_idx, ts, z::ComplexF32, Δ::Int32, flag::UInt8)
        on_tick(rec, config, bank)  # you: filters → (PD clamp if enabled) → PLL/NCO
    end
    return nothing
end

end # module TickHotLoopF32