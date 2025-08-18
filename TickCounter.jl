using CSV
using DataFrames
using Dates
using Statistics

# --- Configuration ---
base_dir   = "data/input"
file_name  = "YM 06-25.Last.txt"   # exact file name (with space)
delim_char = ';'

# Daytime trading session (CME Equity Index RTH, Central Time) — adjust if needed
session_start = Time(8, 30, 0)   # 08:30
session_end   = Time(15, 0, 0)   # 15:00 (exclusive)

# --- Helpers ---

"""
    parse_timestamp(ts::AbstractString) -> DateTime

Parse timestamps like "20250319 070000 0520000" into a DateTime (second precision).
The microsecond part is ignored for counting & session filtering.
"""
function parse_timestamp(ts::AbstractString)
    parts = split(ts)  # ["YYYYMMDD","HHMMSS","uuuuuu?"]
    @assert length(parts) ≥ 2 "Bad timestamp field: $ts"
    date_str, time_str = parts[1], parts[2]
    y = parse(Int, date_str[1:4])
    m = parse(Int, date_str[5:6])
    d = parse(Int, date_str[7:8])
    hh = parse(Int, time_str[1:2])
    mm = parse(Int, time_str[3:4])
    ss = parse(Int, time_str[5:6])
    return DateTime(y, m, d, hh, mm, ss)
end

"""
    in_day_session(t::Time; start=session_start, stop=session_end) -> Bool

True if time t is within [start, stop) — inclusive of start, exclusive of stop.
"""
in_day_session(t::Time; start::Time=session_start, stop::Time=session_end) =
    (t >= start) && (t < stop)

# --- Load & compute ---

full_path = joinpath(base_dir, file_name)

tbl = CSV.File(
    full_path;
    header=false,
    delim=delim_char,
    ignoreemptylines=true,
    normalizenames=false,
    stripwhitespace=true,
)

counts_by_day = Dict{Date, Int}()

for row in tbl
    ts_str = row.Column1::AbstractString
    dt = parse_timestamp(ts_str)
    t  = Time(dt)
    if in_day_session(t)
        d = Date(dt)
        counts_by_day[d] = get(counts_by_day, d, 0) + 1
    end
end

# Filter out days with fewer than 100 ticks
filtered_counts = Dict(d => c for (d,c) in counts_by_day if c ≥ 100)

if isempty(filtered_counts)
    println("No days with at least 100 ticks in the configured day session.")
else
    dates_sorted   = sort(collect(keys(filtered_counts)))
    counts_sorted  = [filtered_counts[d] for d in dates_sorted]

    daily_min = minimum(counts_sorted)
    daily_max = maximum(counts_sorted)
    daily_avg = mean(counts_sorted)

    df_counts = DataFrame(Date = dates_sorted, Count = counts_sorted)

    println("\nFile: $full_path")
    println("Session window: $(string(session_start))–$(string(session_end)) (inclusive/exclusive)")
    println("Trading days (≥100 ticks) found: $(length(dates_sorted))")

    println("\nPer-day tick counts (≥100 ticks) within day session:")
    show(df_counts, allrows=true, allcols=true)

    println("\n\nSummary within day session (days with ≥100 ticks):")
    println("  Daily MIN ticks : $daily_min")
    println("  Daily MAX ticks : $daily_max")
    println("  Daily AVG ticks : $(round(daily_avg; digits=2))")
end
