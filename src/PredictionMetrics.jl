# src/PredictionMetrics.jl - Performance Metrics for Price Prediction

"""
Prediction Metrics Module - Chunk 4

Calculates performance metrics for price change predictions.
Focuses on I-component (real part) comparison between predicted and actual values.

Key Metrics:
- MSE/MAE for prediction accuracy
- Directional accuracy (sign agreement)
- Sharpe ratio for trading signals
- Correlation and R-squared
- Multi-horizon performance tracking
"""

module PredictionMetrics

using Statistics
using LinearAlgebra
using DataFrames

export PredictionMetric, MetricResult, MetricSuite,
       calculate_mse, calculate_mae, calculate_rmse,
       calculate_directional_accuracy, calculate_correlation,
       calculate_sharpe_ratio, calculate_max_drawdown,
       evaluate_prediction_performance, create_metric_suite,
       aggregate_metrics, generate_performance_report,
       HorizonMetrics, calculate_all_metrics

# =============================================================================
# METRIC STRUCTURES
# =============================================================================

"""
Individual metric result
"""
struct MetricResult
    name::String
    value::Float32
    horizon::Int32
    n_samples::Int32
    timestamp::Int64
end

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
Complete metric suite for evaluation
"""
mutable struct MetricSuite
    horizons::Vector{Int32}
    metrics_by_horizon::Dict{Int32, HorizonMetrics}
    aggregate_metrics::Dict{String, Float32}
    
    # Time series tracking
    metric_history::Vector{MetricResult}
    evaluation_count::Int64
end

# =============================================================================
# CORE METRIC CALCULATIONS (I-COMPONENT ONLY)
# =============================================================================

"""
Calculate Mean Squared Error on I-components
"""
function calculate_mse(predicted::Vector{Float32}, actual::Vector{Float32})::Float32
    @assert length(predicted) == length(actual) "Length mismatch"
    
    if isempty(predicted)
        return Float32(0)
    end
    
    errors = predicted .- actual
    return mean(errors .^ 2)
end

"""
Calculate Mean Absolute Error on I-components
"""
function calculate_mae(predicted::Vector{Float32}, actual::Vector{Float32})::Float32
    @assert length(predicted) == length(actual) "Length mismatch"
    
    if isempty(predicted)
        return Float32(0)
    end
    
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
function calculate_directional_accuracy(predicted::Vector{Float32}, 
                                       actual::Vector{Float32})::Float32
    @assert length(predicted) == length(actual) "Length mismatch"
    
    if isempty(predicted)
        return Float32(0.5)  # Random guess baseline
    end
    
    correct = sum(sign.(predicted) .== sign.(actual))
    return Float32(correct / length(predicted))
end

"""
Calculate correlation coefficient
"""
function calculate_correlation(predicted::Vector{Float32}, 
                              actual::Vector{Float32})::Float32
    @assert length(predicted) == length(actual) "Length mismatch"
    
    if length(predicted) < 2
        return Float32(0)
    end
    
    # Handle zero variance cases
    pred_std = std(predicted)
    actual_std = std(actual)
    
    if pred_std < 1e-10 || actual_std < 1e-10
        return Float32(0)
    end
    
    return cor(predicted, actual)
end

"""
Calculate R-squared (coefficient of determination)
"""
function calculate_r_squared(predicted::Vector{Float32}, 
                            actual::Vector{Float32})::Float32
    @assert length(predicted) == length(actual) "Length mismatch"
    
    if isempty(predicted)
        return Float32(0)
    end
    
    # Total sum of squares
    actual_mean = mean(actual)
    ss_tot = sum((actual .- actual_mean) .^ 2)
    
    if ss_tot < 1e-10
        return Float32(0)
    end
    
    # Residual sum of squares
    ss_res = sum((actual .- predicted) .^ 2)
    
    r_squared = 1.0f0 - ss_res / ss_tot
    return clamp(r_squared, -1.0f0, 1.0f0)  # Can be negative for bad models
end

# =============================================================================
# TRADING METRICS
# =============================================================================

"""
Calculate Sharpe ratio for trading signals based on predictions
"""
function calculate_sharpe_ratio(predicted::Vector{Float32}, 
                               actual::Vector{Float32};
                               risk_free_rate::Float32 = 0.0f0,
                               annualization_factor::Float32 = 252.0f0)::Float32
    @assert length(predicted) == length(actual) "Length mismatch"
    
    if length(predicted) < 2
        return Float32(0)
    end
    
    # Generate returns based on prediction signals
    # Trade direction based on predicted sign, magnitude based on actual
    returns = sign.(predicted) .* actual
    
    # Calculate excess returns
    excess_returns = returns .- risk_free_rate / annualization_factor
    
    # Calculate Sharpe ratio
    mean_excess = mean(excess_returns)
    std_excess = std(excess_returns)
    
    if std_excess < 1e-10
        return Float32(0)
    end
    
    # Annualized Sharpe ratio
    sharpe = mean_excess / std_excess * sqrt(annualization_factor)
    
    return Float32(sharpe)
end

"""
Calculate maximum drawdown
"""
function calculate_max_drawdown(predicted::Vector{Float32}, 
                               actual::Vector{Float32})::Float32
    @assert length(predicted) == length(actual) "Length mismatch"
    
    if isempty(predicted)
        return Float32(0)
    end
    
    # Calculate cumulative returns based on predictions
    returns = sign.(predicted) .* actual
    cumulative = cumsum(returns)
    
    # Calculate running maximum
    running_max = accumulate(max, cumulative)
    
    # Calculate drawdowns
    drawdowns = (cumulative .- running_max) ./ (running_max .+ 1e-10)
    
    # Maximum drawdown (most negative)
    max_dd = minimum(drawdowns)
    
    return abs(max_dd)
end

# =============================================================================
# COMPREHENSIVE EVALUATION
# =============================================================================

"""
Calculate all metrics for predictions at a specific horizon
"""
function calculate_all_metrics(predicted::Vector{Float32},
                              actual::Vector{Float32},
                              horizon::Int32)::HorizonMetrics
    
    # Ensure we have data
    if isempty(predicted) || isempty(actual)
        return HorizonMetrics(
            horizon, 0.0f0, 0.0f0, 0.0f0, 0.5f0,
            0.0f0, 0.0f0, 0.0f0, 0.0f0, 0
        )
    end
    
    # Calculate all metrics
    mse = calculate_mse(predicted, actual)
    mae = calculate_mae(predicted, actual)
    rmse = calculate_rmse(predicted, actual)
    dir_acc = calculate_directional_accuracy(predicted, actual)
    corr = calculate_correlation(predicted, actual)
    r2 = calculate_r_squared(predicted, actual)
    sharpe = calculate_sharpe_ratio(predicted, actual)
    max_dd = calculate_max_drawdown(predicted, actual)
    
    return HorizonMetrics(
        horizon, mse, mae, rmse, dir_acc,
        corr, r2, sharpe, max_dd, length(predicted)
    )
end

"""
Evaluate prediction performance across multiple horizons
"""
function evaluate_prediction_performance(predictions::Dict{Int32, Vector{Float32}},
                                        actuals::Dict{Int32, Vector{Float32}})::MetricSuite
    
    horizons = sort(collect(keys(predictions)))
    metrics_by_horizon = Dict{Int32, HorizonMetrics}()
    
    for horizon in horizons
        if haskey(actuals, horizon)
            pred = predictions[horizon]
            actual = actuals[horizon]
            
            # Ensure same length
            min_len = min(length(pred), length(actual))
            if min_len > 0
                metrics = calculate_all_metrics(
                    pred[1:min_len],
                    actual[1:min_len],
                    horizon
                )
                metrics_by_horizon[horizon] = metrics
            end
        end
    end
    
    # Calculate aggregate metrics
    aggregate_metrics = aggregate_metrics(metrics_by_horizon)
    
    return MetricSuite(
        horizons,
        metrics_by_horizon,
        aggregate_metrics,
        Vector{MetricResult}(),
        0
    )
end

"""
Aggregate metrics across horizons
"""
function aggregate_metrics(metrics_by_horizon::Dict{Int32, HorizonMetrics})::Dict{String, Float32}
    
    if isempty(metrics_by_horizon)
        return Dict{String, Float32}()
    end
    
    # Collect all metrics
    all_metrics = collect(values(metrics_by_horizon))
    
    # Weight by number of samples
    total_samples = sum(m.n_samples for m in all_metrics)
    
    if total_samples == 0
        return Dict{String, Float32}()
    end
    
    # Weighted averages
    weighted_mse = sum(m.mse * m.n_samples for m in all_metrics) / total_samples
    weighted_mae = sum(m.mae * m.n_samples for m in all_metrics) / total_samples
    weighted_dir_acc = sum(m.directional_accuracy * m.n_samples for m in all_metrics) / total_samples
    weighted_corr = sum(m.correlation * m.n_samples for m in all_metrics) / total_samples
    
    # Simple averages for some metrics
    avg_sharpe = mean(m.sharpe_ratio for m in all_metrics)
    max_drawdown = maximum(m.max_drawdown for m in all_metrics)
    
    return Dict{String, Float32}(
        "weighted_mse" => weighted_mse,
        "weighted_mae" => weighted_mae,
        "weighted_rmse" => sqrt(weighted_mse),
        "weighted_directional_accuracy" => weighted_dir_acc,
        "weighted_correlation" => weighted_corr,
        "average_sharpe_ratio" => avg_sharpe,
        "max_drawdown" => max_drawdown,
        "total_samples" => Float32(total_samples)
    )
end

# =============================================================================
# REPORTING
# =============================================================================

"""
Generate performance report
"""
function generate_performance_report(suite::MetricSuite)::String
    report = "PREDICTION PERFORMANCE REPORT\n"
    report *= "="^60 * "\n\n"
    
    # Overall metrics
    report *= "AGGREGATE METRICS:\n"
    report *= "-"^30 * "\n"
    
    if haskey(suite.aggregate_metrics, "weighted_mse")
        report *= "Weighted MSE: $(round(suite.aggregate_metrics["weighted_mse"], digits=6))\n"
        report *= "Weighted MAE: $(round(suite.aggregate_metrics["weighted_mae"], digits=6))\n"
        report *= "Weighted RMSE: $(round(suite.aggregate_metrics["weighted_rmse"], digits=6))\n"
        report *= "Directional Accuracy: $(round(suite.aggregate_metrics["weighted_directional_accuracy"] * 100, digits=1))%\n"
        report *= "Correlation: $(round(suite.aggregate_metrics["weighted_correlation"], digits=3))\n"
        report *= "Avg Sharpe Ratio: $(round(suite.aggregate_metrics["average_sharpe_ratio"], digits=2))\n"
        report *= "Max Drawdown: $(round(suite.aggregate_metrics["max_drawdown"] * 100, digits=1))%\n"
    end
    
    report *= "\n"
    
    # Per-horizon metrics
    report *= "METRICS BY HORIZON:\n"
    report *= "-"^30 * "\n"
    
    for horizon in sort(suite.horizons)
        if haskey(suite.metrics_by_horizon, horizon)
            m = suite.metrics_by_horizon[horizon]
            report *= "\nHorizon $horizon ticks (n=$(m.n_samples)):\n"
            report *= "  MSE: $(round(m.mse, digits=6))\n"
            report *= "  MAE: $(round(m.mae, digits=6))\n"
            report *= "  RMSE: $(round(m.rmse, digits=6))\n"
            report *= "  Dir Accuracy: $(round(m.directional_accuracy * 100, digits=1))%\n"
            report *= "  Correlation: $(round(m.correlation, digits=3))\n"
            report *= "  R²: $(round(m.r_squared, digits=3))\n"
            report *= "  Sharpe: $(round(m.sharpe_ratio, digits=2))\n"
            report *= "  Max DD: $(round(m.max_drawdown * 100, digits=1))%\n"
        end
    end
    
    return report
end

"""
Create metric suite for tracking
"""
function create_metric_suite(horizons::Vector{Int32})::MetricSuite
    return MetricSuite(
        horizons,
        Dict{Int32, HorizonMetrics}(),
        Dict{String, Float32}(),
        Vector{MetricResult}(),
        0
    )
end

"""
Update metric suite with new evaluation
"""
function update_metric_suite!(suite::MetricSuite,
                             horizon::Int32,
                             predicted::Vector{Float32},
                             actual::Vector{Float32})
    
    # Calculate metrics
    metrics = calculate_all_metrics(predicted, actual, horizon)
    
    # Store in suite
    suite.metrics_by_horizon[horizon] = metrics
    
    # Add to history
    push!(suite.metric_history, MetricResult(
        "mse", metrics.mse, horizon, metrics.n_samples, time()
    ))
    push!(suite.metric_history, MetricResult(
        "directional_accuracy", metrics.directional_accuracy, 
        horizon, metrics.n_samples, time()
    ))
    
    # Update aggregate metrics
    suite.aggregate_metrics = aggregate_metrics(suite.metrics_by_horizon)
    suite.evaluation_count += 1
end

# =============================================================================
# DATA FRAME SUPPORT
# =============================================================================

"""
Convert metric suite to DataFrame for analysis
"""
function metrics_to_dataframe(suite::MetricSuite)::DataFrame
    
    # Collect all metrics into vectors
    horizons = Int32[]
    mse_vals = Float32[]
    mae_vals = Float32[]
    rmse_vals = Float32[]
    dir_acc_vals = Float32[]
    corr_vals = Float32[]
    r2_vals = Float32[]
    sharpe_vals = Float32[]
    dd_vals = Float32[]
    n_samples_vals = Int32[]
    
    for horizon in sort(suite.horizons)
        if haskey(suite.metrics_by_horizon, horizon)
            m = suite.metrics_by_horizon[horizon]
            push!(horizons, horizon)
            push!(mse_vals, m.mse)
            push!(mae_vals, m.mae)
            push!(rmse_vals, m.rmse)
            push!(dir_acc_vals, m.directional_accuracy)
            push!(corr_vals, m.correlation)
            push!(r2_vals, m.r_squared)
            push!(sharpe_vals, m.sharpe_ratio)
            push!(dd_vals, m.max_drawdown)
            push!(n_samples_vals, m.n_samples)
        end
    end
    
    return DataFrame(
        horizon = horizons,
        mse = mse_vals,
        mae = mae_vals,
        rmse = rmse_vals,
        directional_accuracy = dir_acc_vals,
        correlation = corr_vals,
        r_squared = r2_vals,
        sharpe_ratio = sharpe_vals,
        max_drawdown = dd_vals,
        n_samples = n_samples_vals
    )
end

"""
Calculate performance percentiles for benchmarking
"""
function calculate_performance_percentiles(suite::MetricSuite)::Dict{String, Vector{Float32}}
    
    if isempty(suite.metrics_by_horizon)
        return Dict{String, Vector{Float32}>()
    end
    
    # Collect metrics
    dir_accs = [m.directional_accuracy for m in values(suite.metrics_by_horizon)]
    correlations = [m.correlation for m in values(suite.metrics_by_horizon)]
    sharpes = [m.sharpe_ratio for m in values(suite.metrics_by_horizon)]
    
    # Calculate percentiles [0, 25, 50, 75, 100]
    percentiles = [0, 25, 50, 75, 100]
    
    return Dict{String, Vector{Float32}}(
        "directional_accuracy" => [quantile(dir_accs, p/100) for p in percentiles],
        "correlation" => [quantile(correlations, p/100) for p in percentiles],
        "sharpe_ratio" => [quantile(sharpes, p/100) for p in percentiles]
    )
end

end # module PredictionMetrics