module utils

using LinearAlgebra
using Statistics: mean
export check_index,clear_output,_average_helper,convert2array,convert_1d,_check_curve,_calculate_for_curves, _calculate_for_curves_with_matrices, _trapz, _validate_distance_input

global CONVERT_ARRAY_TYPE = true
global ARRAY_TYPE = AbstractArray
global SUPRESS_WARNINGS = false

function check_index(x, none_accepted; class_name = nothing, ith_class = nothing, valid_modes = nothing, average = "macro")
    if average == "sample-weights"; @assert weights != nothing && length(weights) == length(c.Labels) """If the average mode is weighted, weights that are the same size as the labels must be provided!
        @assert average in valid_modes "Unknown averaging mode. This function only supports the following types: " * join(valid_modes, ", ")
    If no precalculated weights can be provided but the class imbalance is to be taken into account try ' average = "weighted" ' or  ' average = "micro" ' """; end

    if !none_accepted; @assert class_name != nothing || ith_class != nothing "No class name or class indexing value provided"; end
    if none_accepted && class_name == nothing == ith_class
        return -1
    elseif class_name != nothing
        @assert class_name in x "There is no such class in the labels of the given confusion matrix"
        index = findfirst(x -> x == class_name, x)
        return index
    else
        @assert ith_class >= 0 && ith_class <= length(x) "ith_class value is not in range"
        return ith_class
    end
end

function clear_output(x, zero_division)
    if true in [isnan(i) || isinf(i) for i in x]
        if zero_division == "warn" || zero_division == "0"
            if zero_division == "warn"; @warn "Zero division, replacing NaN or Inf with 0"; end;
            if length(x) > 1
                return replace(x, NaN => 0)
            else
                return 0
            end
        else
            if length(x) > 1
                return replace(x, NaN => 1)
            else
                return 1
            end
        end
    else return x
    end
end

function _average_helper(numerator, denominator, weights, average, zero_division, normalize)
    if average == "macro"
        x = clear_output(denominator == nothing ? mean(numerator) : mean(numerator ./ denominator), zero_division)
    elseif average == "micro"
        x = clear_output(denominator == nothing ? sum(numerator) : sum(numerator) / sum(denominator) , zero_division)
    elseif average == "weighted" || average == "sample-weights"
        x = clear_output(denominator == nothing ? (sum(numerator .* weights) / sum(weights)) : (sum((numerator ./ denominator) .* weights) / sum(weights)), zero_division)
    else
        x = clear_output(denominator == nothing ? numerator : numerator ./ denominator, zero_division)
    end
    return normalize && length(x) > 1 ? LinearAlgebra.normalize(x) : x
end

function convert2array(x)
     if CONVERT_ARRAY_TYPE && !(typeof(x) <: ARRAY_TYPE)
         !SUPRESS_WARNINGS && @warn "Input has been converted to KnetMetrics.ARRAY_TYPE ($ARRAY_TYPE) . If this behaviour is not wanted,
         set KnetMetrics.CONVERT_ARRAY_TYPE to false. Warning can be supressed via setting Knet.SUPRESS_WARNINGS to
         true."
         return ARRAY_TYPE(x)
    else
        return x
    end
end

convert_1d(x) = reshape(convert2array(x), (length(x)))

function _check_curve(y_true, thresholding_func = nothing, thresholds = [], first_func = false_positive_rate, second_func = true_positive_rate; ith_class = nothing,
        class_name = nothing, average = "macro", normalize = false, weights = nothing)

    thresholding_func == nothing || thresholds == [] ? throw(DomainError("'thresholding_func' cannot be equal to nothing and thresholds must be provided.")) : nothing
    y_true, thresholds = convert_1d(y_true),  convert_1d(thresholds)

    @assert average in ["macro", "weighted", "micro", "sample-weights"] "Unknown averaging mode. This function only supports the following types: macro, weighted, sample-weights"
    if average == "sample-weights"; @assert weights != nothing && length(weights) == length(c.Labels) """If the average mode is weighted, weights that are the same size as the labels must be provided!
    If no precalculated weights can be provided but the class imbalance is to be taken into account try ' average = "weighted" ' or  ' average = "micro" ' """; end
end

function _calculate_for_curves(y_true, y_pred, thresholding_func = nothing, thresholds = [], first_func = false_positive_rate, second_func = true_positive_rate; ith_class = nothing,
        class_name = nothing, average = "macro", normalize = false, weights = nothing)
    print("CALCULATE FOR CURVES : ", thresholding_func, thresholds,"\n\n\n\n\n")

    vals = [thresholding_func(y_pred, i) for i in thresholds]
    matrices = [confusion_matrix(y_true, predictions) for predictions in vals]

    return _calculate_for_curves_with_matrices(matrices, thresholding_func, thresholds, first_func, second_func; ith_class = ith_class,
            class_name = class_name, average = average, normalize = normalize, weights = weights)

end

function _calculate_for_curves_with_matrices(c, thresholding_func = nothing, thresholds = [], first_func = false_positive_rate, second_func = true_positive_rate; ith_class = nothing,
        class_name = nothing, average = "macro", normalize = false, weights = nothing)

    options = (ith_class = ith_class, class_name = class_name, average = average, normalize = normalize, weights = weights)
    ratios = []

    ratios = [(first_func(c[i];options...), second_func(c[i];options...)) for i in 1:length(c)]
    print("OPTIONS : ", options)
    return RocCurve(thresholds, ratios, options)

end

_trapz(y,x, dx = 1.0) = sum(diff(x) .* (y[1:end-1] .+ y[2:end] ) / 2.0 )


function _validate_distance_input(u,v,w; p = nothing, p_is_used = false, check_weight_length=false)
    if p_is_used; @assert p >= 1 "'p' value must be greater than or equal to one"; end
    check_weight_length && w != nothing && @assert length(w) == length(u) == length(v) "Given 'w' values must be of same size with distance vectors"
end

end
