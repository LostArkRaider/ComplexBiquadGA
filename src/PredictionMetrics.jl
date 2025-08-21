module PredictionMetrics

# All `using` statements are handled by the main ComplexBiquadGA.jl module

export calculate_mse, calculate_mae, calculate_rmse,
       calculate_directional_accuracy, calculate_correlation,
       calculate_sharpe_ratio, calculate_max_drawdown,
       calculate_all_metrics, HorizonMetrics, MetricSuite,
       create_metric_suite, update_metric_suite!,
       generate_performance_report

# =============================================================================
# METRIC STRUCTURES
# =============================================================================

struct HorizonMetrics
    horizon::Int32
    mse::Float32
    mae::Float32
    rmse::Float32
    directional_accuracy::Float32
    correlation::Float32
    r_squared::Float32
    sharpe_ratio::Float32
    max_drawdown::Float32
    n_samples::Int32
end

mutable struct MetricSuite
    horizons::Vector{Int32}
    horizon_metrics::Vector{HorizonMetrics}
    weighted_mse::Float32
    weighted_mae::Float32
    weighted_rmse::Float32
    weighted_directional_accuracy::Float32
    weighted_correlation::Float32
    average_sharpe_ratio::Float32
    max_drawdown_overall::Float32
    total_samples::Int32
end

# =============================================================================
# CORE METRIC CALCULATIONS (Hardware-Agnostic)
# =============================================================================

# Helper function to ensure calculations are done on the CPU
function to_cpu(arr::AbstractVector)
    return Array(arr)
end

function calculate_mse(predicted::AbstractVector{T}, actual::AbstractVector{T})::Float32 where T
    pred_cpu, actual_cpu = to_cpu(predicted), to_cpu(actual)
    @assert length(pred_cpu) == length(actual_cpu) "Length mismatch"
    isempty(pred_cpu) && return Float32(0)
    return mean((pred_cpu .- actual_cpu) .^ 2)
end

function calculate_mae(predicted::AbstractVector{T}, actual::AbstractVector{T})::Float32 where T
    pred_cpu, actual_cpu = to_cpu(predicted), to_cpu(actual)
    @assert length(pred_cpu) == length(actual_cpu) "Length mismatch"
    isempty(pred_cpu) && return Float32(0)
    return mean(abs.(pred_cpu .- actual_cpu))
end

function calculate_rmse(predicted::AbstractVector{T}, actual::AbstractVector{T})::Float32 where T
    return sqrt(calculate_mse(predicted, actual))
end

function calculate_directional_accuracy(predicted::AbstractVector{T}, actual::AbstractVector{T})::Float32 where T
    pred_cpu, actual_cpu = to_cpu(predicted), to_cpu(actual)
    @assert length(pred_cpu) == length(actual_cpu) "Length mismatch"
    isempty(pred_cpu) && return Float32(0.5)
    
    correct = sum(sign.(pred_cpu) .== sign.(actual_cpu))
    return Float32(correct / length(pred_cpu))
end

function calculate_correlation(predicted::AbstractVector{T}, actual::AbstractVector{T})::Float32 where T
    pred_cpu, actual_cpu = to_cpu(predicted), to_cpu(actual)
    @assert length(pred_cpu) == length(actual_cpu) "Length mismatch"
    length(pred_cpu) < 2 && return Float32(0)
    
    (std(pred_cpu) < 1e-10 || std(actual_cpu) < 1e-10) && return Float32(0)
    
    return cor(pred_cpu, actual_cpu)
end

function calculate_sharpe_ratio(predicted::AbstractVector{T}, actual::AbstractVector{T};
                               risk_free_rate::Float32 = 0.0f0,
                               annualization_factor::Float32 = 252.0f0)::Float32 where T
    pred_cpu, actual_cpu = to_cpu(predicted), to_cpu(actual)
    @assert length(pred_cpu) == length(actual_cpu) "Length mismatch"
    length(pred_cpu) < 2 && return Float32(0)
    
    returns = sign.(pred_cpu) .* actual_cpu
    excess_returns = returns .- risk_free_rate / annualization_factor
    
    mean_excess = mean(excess_returns)
    std_excess = std(excess_returns)
    std_excess < 1e-10 && return Float32(0)
    
    return Float32(mean_excess / std_excess * sqrt(annualization_factor))
end

# ... (Other calculation functions like calculate_max_drawdown refactored similarly) ...

# =============================================================================
# METRIC SUITE MANAGEMENT
# =============================================================================

function create_metric_suite(horizons::Vector{Int32} = Int32[])::MetricSuite
    # ... (Implementation unchanged) ...
end

function update_metric_suite!(suite::MetricSuite, horizon::Int32,
                             predicted::AbstractVector{Float32}, actual::AbstractVector{Float32})
    # ... (Implementation unchanged) ...
end

function generate_performance_report(suite::MetricSuite)::String
    # ... (Implementation unchanged) ...
end

end # module PredictionMetrics