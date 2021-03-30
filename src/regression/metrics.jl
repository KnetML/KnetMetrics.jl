
using LinearAlgebra
using Statistics: mean, median

#TODO
# R2-Score, Variance-Score, Deviance related functions

export max_error, mean_absolute_error, mean_squared_error, mean_squared_log_error, median_absolute_error, mean_absolute_percentage_error

max_error(y_true, y_pred) = maximum(abs.(convert_1d(y_true) .- convert_1d(y_pred)))

function mean_absolute_error(y_true, y_pred; weights = nothing)
    _validate_distance_input(y_true, y_pred, weights)
    result = abs.(y_true .- y_pred)
    result = weights == nothing ? mean(result) : mean(result .* weights)
    return result
end

function mean_squared_error(y_true, y_pred; weights = nothing, squared = false)
    _validate_distance_input(y_true, y_pred, weights)
    result = (y_true .- y_pred) .^ 2
    result = !squared ? sqrt.(result) : result
    return weights == nothing ? mean(result) : mean(result .* weights)
end

mean_squared_log_error(y_true, y_pred; weights = nothing) = mean_squared_error(log.(1 .+ y_true), log.(1 .+ y_pred); weights = weights)

function median_absolute_error(y_true, y_pred; weights = nothing)
    _validate_distance_input(y_true, y_pred, weights)
    result = median(abs.(y_pred .- y_true) ; dims = 1)
    return weights == nothing ? mean(result) : mean(result .* weights)
end

function mean_absolute_percentage_error(y_true, y_pred; weights = nothing)
    _validate_distance_input(y_true, y_pred, weights)
    result = abs.(y_true .- y_pred) ./ max.(abs.(y_true),eps())
    return weights == nothing ? mean(result) : mean(result .* weights)
end
