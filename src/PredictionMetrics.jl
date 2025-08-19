# src/PredictionMetrics.jl - Performance Metrics for Price Prediction
# Clean version without module conflicts or dictionary usage

module PredictionMetrics

using Statistics
using LinearAlgebra

export calculate_mse, calculate_mae, calculate_rmse,
       calculate_directional_accuracy, calculate_correlation,
       calculate_sharpe_ratio, calculate_max_drawdown,
       calculate_all_metrics, HorizonMetrics, MetricSuite,
       create_metric_suite, update_metric_suite!,
       generate_performance_report

# =============================================================================
# METRIC STRUCTURES (NO DICTIONARIES)
# =============================================================================

"""
Metrics for a specific prediction horizon
"""
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

"""
Complete metric suite (using struct fields instead of dictionaries)
"""
mutable struct MetricSuite
    horizons::Vector{Int32}
    horizon_metrics::Vector{HorizonMetrics}
    
    # Aggregate metrics as fields
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
# CORE METRIC CALCULATIONS
# =============================================================================

"""
Calculate Mean Squared Error
"""
function calculate_mse(predicted::Vector{Float32}, actual::Vector{Float32})::Float32
    @assert length(predicted) == length(actual) "Length mismatch"
    isempty(predicted) && return Float32(0)
    return mean((predicted .- actual) .^ 2)
end

"""
Calculate Mean Absolute Error
"""
function calculate_mae(predicted::Vector{Float32}, actual::Vector{Float32})::Float32
    @assert length(predicted) == length(actual) "Length mismatch"
    isempty(predicted) && return Float32(0)
    return mean(abs.(predicted .- actual))
end

"""
Calculate Root Mean Squared Error
"""
function calculate_rmse(predicted::Vector{Float32}, actual::Vector{Float32})::Float32
    return sqrt(calculate_mse(predicted, actual))
end

"""
Calculate directional accuracy (percentage of correct sign predictions)
"""
function calculate_directional_accuracy(predicted::Vector{Float32}, actual::Vector{Float32})::Float32
    @assert length(predicted) == length(actual) "Length mismatch"
    isempty(predicted) && return Float32(0.5)
    
    correct = sum(sign.(predicted) .== sign.(actual))
    return Float32(correct / length(predicted))
end

"""
Calculate correlation coefficient
"""
function calculate_correlation(predicted::Vector{Float32}, actual::Vector{Float32})::Float32
    @assert length(predicted) == length(actual) "Length mismatch"
    length(predicted) < 2 && return Float32(0)
    
    pred_std = std(predicted)
    actual_std = std(actual)
    
    (pred_std < 1e-10 || actual_std < 1e-10) && return Float32(0)
    
    return cor(predicted, actual)
end

"""
Calculate R-squared
"""
function calculate_r_squared(predicted::Vector{Float32}, actual::Vector{Float32})::Float32
    @assert length(predicted) == length(actual) "Length mismatch"
    isempty(predicted) && return Float32(0)
    
    actual_mean = mean(actual)
    ss_tot = sum((actual .- actual_mean) .^ 2)
    ss_tot < 1e-10 && return Float32(0)
    
    ss_res = sum((actual .- predicted) .^ 2)
    r_squared = 1.0f0 - ss_res / ss_tot
    
    return clamp(r_squared, -1.0f0, 1.0f0)
end

"""
Calculate Sharpe ratio
"""
function calculate_sharpe_ratio(predicted::Vector{Float32}, actual::Vector{Float32};
                               risk_free_rate::Float32 = 0.0f0,
                               annualization_factor::Float32 = 252.0f0)::Float32
    @assert length(predicted) == length(actual) "Length mismatch"
    length(predicted) < 2 && return Float32(0)
    
    # Trading returns based on prediction signals
    returns = sign.(predicted) .* actual
    excess_returns = returns .- risk_free_rate / annualization_factor
    
    mean_excess = mean(excess_returns)
    std_excess = std(excess_returns)
    std_excess < 1e-10 && return Float32(0)
    
    return Float32(mean_excess / std_excess * sqrt(annualization_factor))
end

"""
Calculate maximum drawdown
"""
function calculate_max_drawdown(predicted::Vector{Float32}, actual::Vector{Float32})::Float32
    @assert length(predicted) == length(actual) "Length mismatch"
    isempty(predicted) && return Float32(0)
    
    returns = sign.(predicted) .* actual
    cumulative = cumsum(returns)
    running_max = accumulate(max, cumulative)
    drawdowns = (cumulative .- running_max) ./ (running_max .+ 1e-10)
    
    return abs(minimum(drawdowns))
end

"""
Calculate all metrics for a horizon
"""
function calculate_all_metrics(predicted::Vector{Float32}, actual::Vector{Float32}, 
                              horizon::Int32)::HorizonMetrics
    
    if isempty(predicted) || isempty(actual)
        return HorizonMetrics(horizon, 0.0f0, 0.0f0, 0.0f0, 0.5f0,
                            0.0f0, 0.0f0, 0.0f0, 0.0f0, 0)
    end
    
    mse = calculate_mse(predicted, actual)
    mae = calculate_mae(predicted, actual)
    rmse = calculate_rmse(predicted, actual)
    dir_acc = calculate_directional_accuracy(predicted, actual)
    corr = calculate_correlation(predicted, actual)
    r2 = calculate_r_squared(predicted, actual)
    sharpe = calculate_sharpe_ratio(predicted, actual)
    max_dd = calculate_max_drawdown(predicted, actual)
    
    return HorizonMetrics(horizon, mse, mae, rmse, dir_acc,
                         corr, r2, sharpe, max_dd, length(predicted))
end

# =============================================================================
# METRIC SUITE MANAGEMENT
# =============================================================================

"""
Create empty metric suite
"""
function create_metric_suite(horizons::Vector{Int32} = Int32[])::MetricSuite
    return MetricSuite(
        horizons,
        HorizonMetrics[],
        0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, Int32(0)
    )
end

"""
Update metric suite with new evaluation
"""
function update_metric_suite!(suite::MetricSuite, horizon::Int32,
                             predicted::Vector{Float32}, actual::Vector{Float32})
    
    metrics = calculate_all_metrics(predicted, actual, horizon)
    
    # Find or add horizon
    idx = findfirst(h -> h == horizon, suite.horizons)
    if idx === nothing
        push!(suite.horizons, horizon)
        push!(suite.horizon_metrics, metrics)
    else
        suite.horizon_metrics[idx] = metrics
    end
    
    # Update aggregates
    update_aggregates!(suite)
end

"""
Update aggregate metrics
"""
function update_aggregates!(suite::MetricSuite)
    isempty(suite.horizon_metrics) && return
    
    total_samples = sum(m.n_samples for m in suite.horizon_metrics)
    total_samples == 0 && return
    
    # Weighted averages
    suite.weighted_mse = sum(m.mse * m.n_samples for m in suite.horizon_metrics) / total_samples
    suite.weighted_mae = sum(m.mae * m.n_samples for m in suite.horizon_metrics) / total_samples
    suite.weighted_rmse = sqrt(suite.weighted_mse)
    suite.weighted_directional_accuracy = sum(m.directional_accuracy * m.n_samples for m in suite.horizon_metrics) / total_samples
    suite.weighted_correlation = sum(m.correlation * m.n_samples for m in suite.horizon_metrics) / total_samples
    
    # Simple averages
    suite.average_sharpe_ratio = mean(m.sharpe_ratio for m in suite.horizon_metrics)
    suite.max_drawdown_overall = maximum(m.max_drawdown for m in suite.horizon_metrics)
    suite.total_samples = Int32(total_samples)
end

"""
Generate performance report
"""
function generate_performance_report(suite::MetricSuite)::String
    report = "PREDICTION PERFORMANCE REPORT\n"
    report *= "="^60 * "\n\n"
    
    # Overall metrics
    report *= "AGGREGATE METRICS:\n"
    report *= "-"^30 * "\n"
    report *= "Weighted MSE: $(round(suite.weighted_mse, digits=6))\n"
    report *= "Weighted MAE: $(round(suite.weighted_mae, digits=6))\n"
    report *= "Weighted RMSE: $(round(suite.weighted_rmse, digits=6))\n"
    report *= "Directional Accuracy: $(round(suite.weighted_directional_accuracy * 100, digits=1))%\n"
    report *= "Correlation: $(round(suite.weighted_correlation, digits=3))\n"
    report *= "Avg Sharpe Ratio: $(round(suite.average_sharpe_ratio, digits=2))\n"
    report *= "Max Drawdown: $(round(suite.max_drawdown_overall * 100, digits=1))%\n"
    report *= "Total Samples: $(suite.total_samples)\n\n"
    
    # Per-horizon metrics
    report *= "METRICS BY HORIZON:\n"
    report *= "-"^30 * "\n"
    
    for m in suite.horizon_metrics
        report *= "\nHorizon $(m.horizon) ticks (n=$(m.n_samples)):\n"
        report *= "  MSE: $(round(m.mse, digits=6))\n"
        report *= "  Dir Accuracy: $(round(m.directional_accuracy * 100, digits=1))%\n"
        report *= "  Correlation: $(round(m.correlation, digits=3))\n"
        report *= "  Sharpe: $(round(m.sharpe_ratio, digits=2))\n"
    end
    
    return report
end

end # module PredictionMetrics