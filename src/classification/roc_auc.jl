export auc, RocCurve, PrecisionRecallCurve, ratio_at_threshold, fpr_at_threshold, tpr_at_threshold, auc_at_threshold, precision_at_threshold, recall_at_threshold

using LinearAlgebra
using Statistics


function auc(x,y)
    direction = 1
    dx = [x[i] - x[i-1] for i in 2:length(x)]
    if any(x-> x < 0, dx)
        if all(x-> x < 0, dx)
            direction = -1
        else
            throw(DomainError("Function is neither increasing nor decreasing"))
        end
    end
    return direction * _trapz(y,x)
end

auc(c::confusion_matrix) = auc(false_positive_rate(c; average="binary"), true_positive_rate(c; average="binary"))

struct RocCurve
    thresholds::Vector{Number}
    ratios::Vector{Tuple{Number, Number}}
    options::NamedTuple
end

function RocCurve(y_true, y_pred, thresholding_func, thresholds; ith_class = nothing, class_name = nothing, average = "macro", normalize = false, weights = nothing)
    return _calculate_for_curves(y_true, y_pred, thresholding_func, thresholds; ith_class = ith_class, class_name = class_name,
        average = average, normalize = normalize, weights = weights)
end

function RocCurve(c::Vector{confusion_matrix} , thresholds, first_func = false_positive_rate, second_func = true_positive_rate; ith_class = nothing,
        class_name = nothing, average = "macro", normalize = false, weights = nothing)
        return _calculate_for_curves_with_matrices(c, thresholding_func, thresholds; ith_class = ith_class, class_name = class_name,
            average = average, normalize = normalize, weights = weights)
end

ratio_at_threshold(r::RocCurve, threshold) = r.ratios[findfirst(x -> x == threshold, r.thresholds)]
fpr_at_threshold(r::RocCurve, threshold) = ratio_at_threshold(r, threshold)[1]
tpr_at_threshold(r::RocCurve, threshold) = ratio_at_threshold(r, threshold)[2]

function Base.show(io::IO, ::MIME"text/plain", r::RocCurve)
    println(io,"Roc Curve:")
    println(io, repeat("=", 60))
    for i in 1:length(r.thresholds)
        println(io, lpad("Threshold : ", 30), r.thresholds[i],  lpad("False Positive Rate : ", 30), r.ratios[i][1], lpad("True Positive Rate : ", 30), r.ratios[i][2])
    end
    println("These values have been calculated with the following parameters : ", r.options)
end

##

struct PrecisionRecallCurve
    thresholds::Vector{Number}
    ratios::Vector{Tuple{Number, Number}}
    options::NamedTuple
end

function PrecisionRecallCurve(y_true, y_pred, thresholding_func, thresholds; ith_class = nothing, class_name = nothing, average = "macro", normalize = false, weights = nothing)
    return _calculate_for_curves(y_true, y_pred, thresholding_func, thresholds, precision_score, recall_score; ith_class = ith_class, class_name = class_name,
        average = average, normalize = normalize, weights = weights)
end

function PrecisionRecallCurve(c::Vector{confusion_matrix} , thresholds, first_func = precision_score, second_func = recall_score; ith_class = nothing,
        class_name = nothing, average = "macro", normalize = false, weights = nothing)
        return _calculate_for_curves_with_matrices(c, thresholding_func, thresholds, precision_score, recall_score; ith_class = ith_class, class_name = class_name,
            average = average, normalize = normalize, weights = weights)
end

ratio_at_threshold(r::PrecisionRecallCurve, threshold) =  r.ratios[findfirst(x -> x == threshold, r.thresholds)]
precision_at_threshold(r::PrecisionRecallCurve, threshold) = ratio_at_threshold(r, threshold)[1]
recall_at_threshold(r::PrecisionRecallCurve, threshold) = ratio_at_threshold(r, threshold)[2]

function Base.show(io::IO, ::MIME"text/plain", r::PrecisionRecallCurve)
    println(io,"Precision Recall Curve:")
    println(io, repeat("=", 60))
    for i in 1:length(r.thresholds)
        println(io, lpad("Threshold : ", 30), r.thresholds[i],  lpad("Precision Score : ", 30), r.ratios[i][1], lpad("Recall Score : ", 30), r.ratios[i][2])
    end
    println("These values have been calculated with the following parameters : ", r.options)
end

##

#TODO
# roc_auc_score and other metrics
