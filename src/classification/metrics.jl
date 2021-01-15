export classification_report, condition_positive, condition_negative, predicted_positive,predicted_negative, correctly_classified, incorrectly_classified, sensitivity_score, recall_score, specificity_score, precision_score, positive_predictive_value, accuracy_score,  _score, negative_predictive_value, false_negative_rate, false_positive_rate, false_discovery_rate, false_omission_rate, f1_score, prevalence_threshold, threat_score, matthews_correlation_coeff, fowlkes_mallows_index, informedness, markedness, cohen_kappa_score, hamming_loss, jaccard_score, confusion_params

using Statistics: mean

function check_index(x, none_accepted; class_name = nothing, ith_class = nothing)
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

"""
```classification_report(c::confusion_matrix; keywords)```
```classification_report(y_true, y_pred; keywords)```

Return all the values listed below if `return_dict` is true. Else, write the values to the given IO element.

Returned dictionary:
```
    "true-positives" => c.true_positives
    "false-positives" => c.false_positives
    "true-negatives" => c.true_negatives
    "false-negatives" => c.false_negatives
    "condition-positive" => condition_positive(c, average = "binary")
    "condition-negative" => condition_negative(c, average = "binary")
    "predicted-positive" => predicted_positive(c, average = "binary")
    "predicted-negative" => predicted_negative(c, average = "binary")
    "correctly-classified" => correctly_classified(c, average = "binary")
    "incorrectly-classified" => incorrectly_classified(c, average = "binary")
    "sensitivity" => sensitivity_score(c, average = "binary")
    "specificity" => specificity_score(c, average = "binary")
    "precision" => precision_score(c, average = "binary")
    "accuracy-score" => accuracy_score(c, average = "binary")
    "balanced-accuracy" =>  balanced_accuracy_score(c, average = "binary")
    "positive-predictive-value" =>  positive_predictive_value(c, average = "binary")
    "negative-predictive-value" => negative_predictive_value(c, average = "binary")
    "false-negative-rate"  => false_negative_rate(c, average = "binary")
    "false-positive-rate"  => false_positive_rate(c, average = "binary")
    "false-discovery-rate" => false_discovery_rate(c, average = "binary")
    "false-omission-rate"  => false_omission_rate(c, average = "binary")
    "f1-score" => f1_score(c, average = "binary")
    "prevalence-threshold" => prevalence_threshold(c, average = "binary")
    "threat-score" => threat_score(c, average = "binary")
    "matthews-correlation-coefficient" => matthews_correlation_coeff(c, average = "binary")
    "fowlkes-mallows-index" => fowlkes_mallows_index(c, average = "binary")
    "informedness" => informedness(c, average = "binary")
    "markedness" => markedness(c, average = "binary")
    "jaccard-score-macroaverage" => jaccard_score(c, average = "macro")
    "jaccard-score-microaverage" => jaccard_score(c, average = "micro")
    "hamming-loss" => hamming_loss(c)
    "cohen-kappa-score" => cohen_kappa_score(c)
```

For a sample output to the given IO element, see Example section.

## Arguments
- `c::confusion_matrix` : The confusion matrix to report for
- `y_true::Vector` : True values for classification
- `y_pred::Vector` : Predicted values for classification

## Keywords

- `io::IO = Base.stdout` : IO element to write to. If 'return_dict' is true, this value will be ignored.
- `return_dict::Bool = false` :  Return a dictionary instead of printing if true.
- `target_names::Vector = nothing` : If not nothing, replace the labels of the given confusion matrix object whilst printing. If 'return_dict' is true, this will be ignored.
- `digits::Int = 2` : Determines how the rounding procedure will be digitized. If `return_dict` is true, this will be ignored and the values
    will be placed into the dictionary with full precision.

## Example

```julia-repl

julia> y_true = [ 4, 4, 1, 4, 4, 2, 1, 1, 2, 4, 1, 3, 3, 3, 1, 1, 3, 1, 4, 4, 3, 3, 3, 1, 4, 1, 2, 3, 2, 2];

julia> y_pred = [ 1, 4, 4, 1, 2, 3, 1, 3, 2, 1, 2, 2, 2, 1, 4, 4, 2, 1, 3, 2, 2, 3, 2, 1, 2, 3, 4, 1, 2, 1];

julia> classification_report(x)
Summary:
confusion_matrix
True Positives: [3, 2, 1, 1]
False Positives: [6, 9, 4, 4]
True Negatives: [15, 16, 18, 18]
False Negatives: [6, 3, 7, 7]

    Labelwise Statistics

                                          1       2       3       4
                Condition Positive:     9.0     5.0     8.0     8.0
                Condition Negative:    21.0    25.0    22.0    22.0
                Predicted Positive:     9.0    11.0     5.0     5.0
                Predicted Negative:    21.0    19.0    25.0    25.0
              Correctly Classified:    18.0    18.0    19.0    19.0
            Incorrectly Classified:    12.0    12.0    11.0    11.0
                       Sensitivity:    0.33     0.4    0.12    0.12
                       Specificity:    0.71    0.64    0.82    0.82
                         Precision:    0.33    0.18     0.2     0.2
                    Accuracy Score:    0.33    0.18     0.2     0.2
                 Balanced Accuracy:    0.33     0.4    0.12    0.12
         Negative Predictive Value:    0.71    0.84    0.72    0.72
               False Negative Rate:    0.67     0.6    0.88    0.88
               False Positive Rate:    0.29    0.36    0.18    0.18
              False Discovery Rate:    0.29    0.36    0.18    0.18
               False Omission Rate:    0.29    0.16    0.28    0.28
                          F1 Score:    0.33    0.25    0.15    0.15
                     Jaccard Score:     0.2    0.14    0.08    0.08
              Prevalence Threshold:    0.48    0.49    0.55    0.55
                      Threat Score:     0.2    0.14    0.08    0.08
  Matthews Correlation Coefficient:    0.11    0.09    0.03    0.03
             Fowlkes Mallows Index:    0.82    0.76    0.57    0.57
                      Informedness:    0.05    0.04   -0.06   -0.06
                        Markedness:    0.05    0.02   -0.08   -0.08

      General Statistics

              Accuracy Score:   0.22878787878787876
           Cohen Kappa Score:   -0.00877192982456143
                Hamming Loss:   0.7666666666666667
               Jaccard Score:   0.12738095238095237
```
"""
function classification_report(c::confusion_matrix; io::IO = Base.stdout, return_dict = false, target_names = nothing, digits = 2, normalize = false)
    if return_dict
        result_dict = Dict()
        result_dict["true-positives"] = c.true_positives
        result_dict["false-positives"] = c.false_positives
        result_dict["true-negatives"] = c.true_negatives
        result_dict["false-negatives"] = c.false_negatives
        result_dict["condition-positive"] = condition_positive(c, average = "binary", normalize = normalize)
        result_dict["condition-negative"] = condition_negative(c, average = "binary", normalize = normalize)
        result_dict["predicted-positive"] = predicted_positive(c, average = "binary", normalize = normalize)
        result_dict["predicted-negative"] = predicted_negative(c, average = "binary", normalize = normalize)
        result_dict["correctly-classified"] = correctly_classified(c, average = "binary", normalize = normalize)
        result_dict["incorrectly-classified"] = incorrectly_classified(c, average = "binary", normalize = normalize)
        result_dict["sensitivity"] = sensitivity_score(c, average = "binary", normalize = normalize)
        result_dict["specificity"] = specificity_score(c, average = "binary", normalize = normalize)
        result_dict["precision"] = precision_score(c, average = "binary", normalize = normalize)
        result_dict["accuracy-score"] = accuracy_score(c, average = "binary", normalize = normalize)
        result_dict["balanced Accuracy"] = balanced_accuracy_score(c, average = "binary", normalize = normalize)
        result_dict["positive-predictive-value"] =  positive_predictive_value(c, average = "binary", normalize = normalize)
        result_dict["negative-predictive-value"] = negative_predictive_value(c, average = "binary", normalize = normalize)
        result_dict["false-negative-rate"]  = false_negative_rate(c, average = "binary", normalize = normalize)
        result_dict["false-positive-rate"]  = false_positive_rate(c, average = "binary", normalize = normalize)
        result_dict["false-discovery-rate"] = false_discovery_rate(c, average = "binary", normalize = normalize)
        result_dict["f1-score"] = f1_score(c, average = "binary", normalize = normalize)
        result_dict["false-omission-rate"]  = false_omission_rate(c , average = "binary", normalize = normalize)
        result_dict["prevalence-threshold"] = prevalence_threshold(c, average = "binary", normalize = normalize)
        result_dict["threat-score"] = threat_score(c, average = "binary", normalize = normalize)
        result_dict["matthews-correlation-coefficient"] = matthews_correlation_coeff(c, average = "binary", normalize = normalize)
        result_dict["fowlkes-mallows-index"] = fowlkes_mallows_index(c, average = "binary", normalize = normalize)
        result_dict["informedness"] = informedness(c, average = "binary", normalize = normalize)
        result_dict["markedness"] = markedness(c, average = "binary", normalize = normalize)
        result_dict["jaccard-score-macroaverage"] = jaccard_score(c, average = "macro", normalize = normalize)
        result_dict["jaccard-score-microaverage"] = jaccard_score(c, average = "micro", normalize = normalize)
        result_dict["hamming-loss"] = hamming_loss(c)
        result_dict["cohen-kappa-score"] = cohen_kappa_score(c)
        return result_dict
    else
        labels = target_names != nothing && length(target_names) == length(c.Labels) ? target_names : c.Labels
        len = maximum([length(string(i)) for i in labels])
        label_size = length(c.Labels)
        label_len = len + digits + 5
        println(io,"Summary:\n", summary(c))
        println(io,"True Positives: ", c.true_positives)
        println(io,"False Positives: ", c.false_positives)
        println(io,"True Negatives: ", c.true_negatives)
        println(io,"False Negatives: ", c.false_negatives)
        println(io,"\n",lpad("Labelwise Statistics", label_len * Int(round(size(c.matrix)[1] / 2)+1)), "\n")
        println(io,lpad(" ", 35), [lpad(i, label_len) for i in labels]...)
        println(io,lpad("Condition Positive:", 35), [lpad(round(i, digits = digits), label_len) for i in condition_positive(c, average = "binary", normalize = normalize)]...)
        println(io,lpad("Condition Negative:", 35), [lpad(round(i, digits = digits), label_len) for i in condition_negative(c, average = "binary", normalize = normalize)]...)
        println(io,lpad("Predicted Positive:", 35), [lpad(round(i, digits = digits), label_len) for i in predicted_positive(c, average = "binary", normalize = normalize)]...)
        println(io,lpad("Predicted Negative:", 35), [lpad(round(i, digits = digits), label_len) for i in predicted_negative(c, average = "binary", normalize = normalize)]...)
        println(io,lpad("Correctly Classified:", 35), [lpad(round(i, digits = digits), label_len) for i in correctly_classified(c, average = "binary", normalize = normalize)]...)
        println(io,lpad("Incorrectly Classified:", 35), [lpad(round(i, digits = digits), label_len) for i in incorrectly_classified(c, average = "binary", normalize = normalize)]...)
        println(io,lpad("Sensitivity:", 35), [lpad(round(i, digits = digits), label_len) for i in sensitivity_score(c, average = "binary", normalize = normalize)]...)
        println(io,lpad("Specificity:", 35), [lpad(round(i, digits = digits), label_len) for i in specificity_score(c, average = "binary", normalize = normalize)]...)
        println(io,lpad("Precision:", 35) , [lpad(round(i, digits = digits), label_len) for i in precision_score(c, average = "binary", normalize = normalize)]...)
        println(io,lpad("Accuracy Score:", 35 ) , [lpad(round(i, digits = digits), label_len) for i in accuracy_score(c, average = "binary", normalize = normalize)]...)
        println(io,lpad("Balanced Accuracy:", 35), [lpad(round(i, digits = digits), label_len) for i in balanced_accuracy_score(c, average = "binary", normalize = normalize)]...)
        println(io,lpad("Negative Predictive Value:", 35), [lpad(round(i, digits = digits), label_len) for i in negative_predictive_value(c, average = "binary", normalize = normalize)]...)
        println(io,lpad("False Negative Rate:", 35), [lpad(round(i, digits = digits), label_len) for i in false_negative_rate(c, average = "binary", normalize = normalize)]...)
        println(io,lpad("False Positive Rate:", 35), [lpad(round(i, digits = digits), label_len) for i in false_positive_rate(c, average = "binary", normalize = normalize)]...)
        println(io,lpad("False Discovery Rate:", 35), [lpad(round(i, digits = digits), label_len) for i in false_discovery_rate(c, average = "binary", normalize = normalize)]...)
        println(io,lpad("False Omission Rate:", 35), [lpad(round(i, digits = digits), label_len) for i in false_omission_rate(c, average = "binary", normalize = normalize)]...)
        println(io,lpad("F1 Score:", 35), [lpad(round(i, digits = digits), label_len) for i in f1_score(c, average = "binary", normalize = normalize)]...)
        println(io,lpad("Jaccard Score:", 35), [lpad(round(i, digits = digits), label_len) for i in jaccard_score(c, average = "binary", normalize = normalize)]...)
        println(io,lpad("Prevalence Threshold:", 35), [lpad(round(i, digits = digits), label_len) for i in prevalence_threshold(c, average = "binary", normalize = normalize)]...)
        println(io,lpad("Threat Score:", 35), [lpad(round(i, digits = digits), label_len) for i in threat_score(c, average = "binary", normalize = normalize)]...)
        println(io,lpad("Matthews Correlation Coefficient:", 35), [lpad(round(i, digits = digits), label_len) for i in matthews_correlation_coeff(c, average = "binary", normalize = normalize)]...)
        println(io,lpad("Fowlkes Mallows Index:", 35), [lpad(round(i, digits = digits), label_len) for i in fowlkes_mallows_index(c, average = "binary", normalize = normalize)]...)
        println(io,lpad("Informedness:", 35), [lpad(round(i, digits = digits), label_len) for i in informedness(c, average = "binary", normalize = normalize)]...)
        println(io,lpad("Markedness:", 35), [lpad(round(i, digits = digits), label_len) for i in markedness(c, average = "binary", normalize = normalize)]...)
        println(io,"\n",lpad("General Statistics", label_len * Int(round(size(c.matrix)[1] / 2)+1)), "\n")
        println(io, lpad("Accuracy Score:\t",30), accuracy_score(c, average = "macro"))
        println(io, lpad("Cohen Kappa Score:\t", 30), cohen_kappa_score(c))
        println(io, lpad("Hamming Loss:\t", 30), hamming_loss(c))
        println(io, lpad("Jaccard Score:\t", 30), jaccard_score(c, average = "macro"))
    end
end

classification_report(expected, predicted; io::IO = Base.stdout, return_dict = false, target_names = nothing, digits = 2) =
classification_report(confusion_matrix(expected, predicted); io = Base.stdout, return_dict = false, target_names = nothing, digits = 2)

##

# Confusion Matrix Evaluation Functions

"""
```condition_positive(c::confusion_matrix; keywords)```
```condition_positive(y_true, y_pred; keywords)```

Return condition positive values of either the whole confusion matrix or the classes specified by `class_name` or `ith_class`
arguments.

    Condition Positives = True Positives + False Negatives

## Arguments
- `c::confusion_matrix` : The confusion matrix to report for
- `y_true::Vector` : True values for classification
- `y_pred::Vector` : Predicted values for classification

## Keywords

- `ith_class::Int = nothing` : Return the results for the ith class, the ith label in the label list of the given confusion matrix object.

- `class_name` : Return the results for the class of the specified value in the label list of the given confusion matrix.

If both `class_name` and `ith_class` arguments are equal to `nothing`, return condition positive values for all the elements in the labels arrays

- `average::String = "binary"` :
\n\t   `"binary"` : Return the classwise values.
\n\t   `"macro"` : Return the macro average (mean) of the classwise values.
\n\t    `"micro"` : Return micro average (sum of the numerator divided by sum of the denominator instead of elementwise division) of the classwise values
\n\t    `"weighted"` :
        Return the weighted average (weighted mean with true positives per class) of the classwise values.
\n\t    `"sample-weights"` : Return the weighted average (weighted mean with given weights per class) of the classwise values.

- `weights::Vector  nothing` :  Use the given weights whilst calculating 'sample_weights' option. If average is not 'sample-weigts' this will be ignored.

- `normalize::Bool = false` : If true, normalize the result.

## Example

First example no indexing:\n\n

```julia-repl
julia> y_true = [ "4", "4", "1", "4", "4", "2", "1", "1", "2", "4", "1", "3", "3", "3", "1", "1", "3", "1", "4", "4", "3", "3", "3", "1", "4", "1", "2", "3", "2", "2"];

julia> y_pred = [ "1", "4", "4", "1", "2", "3", "1", "3", "2", "1", "2", "2", "2", "1", "4", "4", "2", "1", "3", "2", "2", "3", "2", "1", "2", "3", "4", "1", "2", "1"];

julia> x = confusion_matrix(y_true, y_pred);

julia> condition_positive(x)
4-element Array{Int64,1}:
 9
 5
 8
 8

 julia> condition_positive(x, average = "macro")
 7.5

 julia> condition_positive(x, normalize = true)
 4-element Array{Float64,1}:
  0.588348405414552
  0.32686022523030667
  0.5229763603684907
  0.5229763603684907

 julia> condition_positive(y_true, y_pred)
 4-element Array{Int64,1}:
  9
  5
  8
  8

julia> condition_positive(x, class_name = 2)
5

julia> condition_positive(x) == condition_positive(y_true, y_pred)
true

```
See also : [`confusion_matrix`](@ref), [`condition_negative`](@ref), [`predicted_positive`](@ref), [`predicted_negative`](@ref)

"""
function condition_positive(c::confusion_matrix; ith_class = nothing, class_name = nothing, average = "binary", weights = nothing, normalize = false)
    @assert average in ["binary", "macro", "weighted", "micro", "sample-weights"] "Unknown averaging mode. This function only supports the following types: binary, macro, weighted, sample-weights"
    if average == "sample-weights"; @assert weights != nothing && length(weights) == length(c.Labels) """If the average mode is weighted, weights that are the same size as the labels must be provided!
    If no precalculated weights can be provided but the class imbalance is to be taken into account try ' average = "weighted" ' or  ' average = "micro" ' """; end
    index = check_index(c.Labels, true, ith_class = ith_class, class_name = class_name)
    if index == -1
        x = c.true_positives + c.false_negatives
    else
        x = c.true_positives[index] + c.false_negatives[index]
    end
    if average == "weighted"; weights = c.true_positives .+ c.false_negatives ; end
    return _average_helper(x, nothing, weights, average, c.zero_division, normalize)
end

condition_positive(expected, predicted; ith_class = nothing, class_name = nothing, average = "binary", normalize = false) =
condition_positive(confusion_matrix(expected,predicted), ith_class = ith_class, class_name = class_name, average = average, normalize = normalize)

##
"""
```condition_negative(c::confusion_matrix; ith_class = nothing, class_name = nothing)```
```condition_negative(y_true, y_pred; ith_class = nothing, class_name = nothing)```

Return condition negative values of either the whole confusion matrix or the classes specified by `class_name` or `ith_class`
arguments.

    Condition Negatives: True Negatives + False Positives

## Arguments
- `c::confusion_matrix` : The confusion matrix to report for
- `y_true::Vector` : True values for classification
- `y_pred::Vector` : Predicted values for classification

## Keywords

- `ith_class::Int = nothing` : Return the results for the ith class, the ith label in the label list of the given confusion matrix object.

- `class_name` : Return the results for the class of the specified value in the label list of the given confusion matrix.

If both `class_name` and `ith_class` arguments are equal to `nothing`, return condition positive values for all the elements in the labels arrays

- `average::String = "binary"` :
\n\t   `"binary"` : Return the classwise values.
\n\t   `"macro"` : Return the macro average (mean) of the classwise values.
\n\t    `"micro"` : Return micro average (sum of the numerator divided by sum of the denominator instead of elementwise division) of the classwise values
\n\t    `"weighted"` : Return the weighted average (weighted mean with true positives per class) of the classwise values.
\n\t    `"sample-weights"` : Return the weighted average (weighted mean with given weights per class) of the classwise values.

- `weights::Vector  nothing` :  Use the given weights whilst calculating 'sample_weights' option. If average is not 'sample-weigts' this will be ignored.

- `normalize::Bool = false` : If true, normalize the result.

## Example

First example no indexing:\n\n

```julia-repl
julia> y_pred = [ 1, 4, 4, 1, 2, 3, 1, 3, 2, 1, 2, 2, 2, 1, 4, 4, 2, 1, 3, 2, 2, 3, 2, 1, 2, 3, 4, 1, 2, 1];

julia> y_true = [ 4, 4, 1, 4, 4, 2, 1, 1, 2, 4, 1, 3, 3, 3, 1, 1, 3, 1, 4, 4, 3, 3, 3, 1, 4, 1, 2, 3, 2, 2];

julia> x = confusion_matrix(y_true, y_pred, labels = [1,2,3,4]);

julia> condition_negative(x)
4-element Array{Int64,1}:
 21
 25
 22
 22

julia> condition_negative(x, average = "macro")
22.5

julia> condition_negative(x, normalize = true)
4-element Array{Float64,1}:
 0.4656330736664175
 0.5543250876981161
 0.48780607717434216
 0.48780607717434216

julia> condition_negative(x, ith_class = 2)
25

julia> condition_negative(x) == condition_negative(y_true, y_pred)
true

```

See also : [`confusion_matrix`](@ref), [`condition_positive`](@ref), [`predicted_positive`](@ref), [`predicted_negative`](@ref)

"""
function condition_negative(c::confusion_matrix; ith_class = nothing, class_name = nothing, average = "binary", weights = nothing, normalize = false)
    @assert average in ["binary", "macro", "weighted", "micro", "sample-weights"] "Unknown averaging mode. This function only supports the following types: binary, macro, weighted, sample-weights"
    if average == "sample-weights"; @assert weights != nothing && length(weights) == length(c.Labels) """If the average mode is weighted, weights that are the same size as the labels must be provided!
    If no precalculated weights can be provided but the class imbalance is to be taken into account try ' average = "weighted" ' or  ' average = "micro" ' """; end
    index = check_index(c.Labels, true, ith_class = ith_class, class_name = class_name)
    if index == -1
        x = c.true_negatives .+ c.false_positives
    else
        x = c.true_negatives[index] + c.false_positives[index]
    end
    if average == "weighted"; weights = c.true_positives .+ c.false_negatives ; end
    return _average_helper(x, nothing, weights, average, c.zero_division, normalize)
end

condition_negative(expected, predicted; ith_class = nothing, class_name = nothing, average = "binary", normalize = false)  =
condition_negative(confusion_matrix(expected,predicted), ith_class = ith_class, class_name = class_name, average = average, normalize = normalize)

"""
```predicted_positive(c::confusion_matrix; keywords)```
```predicted_positive(y_true, y_pred; keywords)```

Return predicted positive values of either the whole confusion matrix or the classes specified by `class_name` or `ith_class`
arguments.

    Predicted Positives: True Positives + False Positives

## Arguments
- `c::confusion_matrix` : The confusion matrix to report for
- `y_true::Vector` : True values for classification
- `y_pred::Vector` : Predicted values for classification

## Keywords

- `ith_class::Int = nothing` : Return the results for the ith class, the ith label in the label list of the given confusion matrix object.

- `class_name` : Return the results for the class of the specified value in the label list of the given confusion matrix.

If both `class_name` and `ith_class` arguments are equal to `nothing`, return condition positive values for all the elements in the labels arrays

- `average::String = "binary"` :
\n\t   `"binary"` : Return the classwise values.
\n\t   `"macro"` : Return the macro average (mean) of the classwise values.
\n\t    `"micro"` : Return micro average (sum of the numerator divided by sum of the denominator instead of elementwise division) of the classwise values
\n\t    `"weighted"` :
        Return the weighted average (weighted mean with true positives per class) of the classwise values.
\n\t    `"sample-weights"` : Return the weighted average (weighted mean with given weights per class) of the classwise values.

- `weights::Vector  nothing` :  Use the given weights whilst calculating 'sample_weights' option. If average is not 'sample-weigts' this will be ignored.

- `normalize::Bool = false` : If true, normalize the result.

## Example

First example no indexing:\n\n

```julia-repl
julia> y_pred = [ 1, 4, 4, 1, 2, 3, 1, 3, 2, 1, 2, 2, 2, 1, 4, 4, 2, 1, 3, 2, 2, 3, 2, 1, 2, 3, 4, 1, 2, 1];

julia> y_true = [ 4, 4, 1, 4, 4, 2, 1, 1, 2, 4, 1, 3, 3, 3, 1, 1, 3, 1, 4, 4, 3, 3, 3, 1, 4, 1, 2, 3, 2, 2];

julia> x = confusion_matrix(y_true, y_pred);

julia> predicted_positive(x)
4-element Array{Int64,1}:
  9
 11
  5
  5

julia> predicted_positive(x, average = "macro")
7.5

julia> predicted_positive(x, average = "micro")
30

julia> predicted_positive(x, normalize = true)
4-element Array{Float64,1}:
 0.5669467095138407
 0.6929348671835831
 0.31497039417435596
 0.31497039417435596

julia> predicted_positive(x, class_name = 3)
5

julia> predicted_positive(x) == predicted_positive(y_true, y_pred)
true

```

See also : [`confusion_matrix`](@ref), [`condition_negative`](@ref), [`predicted_positive`](@ref), [`predicted_negative`](@ref)

"""
function predicted_positive(c::confusion_matrix; ith_class = nothing, class_name = nothing, average = "binary", weights = nothing, normalize = false)
    @assert average in ["binary", "macro", "weighted", "micro", "sample-weights"] "Unknown averaging mode. This function only supports the following types: binary, macro, weighted, sample-weights"
    if average == "sample-weights"; @assert weights != nothing && length(weights) == length(c.Labels) """If the average mode is weighted, weights that are the same size as the labels must be provided!
    If no precalculated weights can be provided but the class imbalance is to be taken into account try ' average = "weighted" ' or  ' average = "micro" ' """; end

    index = check_index(c.Labels, true, ith_class = ith_class, class_name = class_name)
    if index == -1
        x = c.true_positives .+ c.false_positives
    else
        x = c.true_positives[index] + c.false_positives[index]
    end
    if average == "weighted"; weights = c.true_positives .+ c.false_negatives ; end
    return _average_helper(x, nothing, weights, average, c.zero_division, normalize)
end

predicted_positive(expected, predicted; ith_class = nothing, class_name = nothing, average = "binary", normalize = false)  =
predicted_positive(confusion_matrix(expected,predicted), ith_class = ith_class, class_name = class_name, average = average, normalize = normalize)

"""
```predicted_negative(c::confusion_matrix; keywords)```
```predicted_negative(y_true, y_pred; keywords)```

Return predicted negative values of either the whole confusion matrix or the classes specified by `class_name` or `ith_class`
arguments.

    Predicted Negatives: Negatives + False Negatives

## Arguments
- `c::confusion_matrix` : The confusion matrix to report for
- `y_true::Vector` : True values for classification
- `y_pred::Vector` : Predicted values for classification

## Keywords

- `ith_class::Int = nothing` : Return the results for the ith class, the ith label in the label list of the given confusion matrix object.

- `class_name` : Return the results for the class of the specified value in the label list of the given confusion matrix.

If both `class_name` and `ith_class` arguments are equal to `nothing`, return condition positive values for all the elements in the labels arrays

- `average::String = "binary"` :
\n\t   `"binary"` : Return the classwise values.
\n\t   `"macro"` : Return the macro average (mean) of the classwise values.
\n\t    `"micro"` : Return micro average (sum of the numerator divided by sum of the denominator instead of elementwise division) of the classwise values
\n\t    `"weighted"` :
            Return the weighted average (weighted mean with true positives per class) of the classwise values.
\n\t    `"sample-weights"` : Return the weighted average (weighted mean with given weights per class) of the classwise values.

- `weights::Vector  nothing` :  Use the given weights whilst calculating 'sample_weights' option. If average is not 'sample-weigts' this will be ignored.

- `normalize::Bool = false` : If true, normalize the result.

## Example

First example no indexing:\n\n

```julia-repl
julia> y_pred = [ 1, 4, 4, 1, 2, 3, 1, 3, 2, 1, 2, 2, 2, 1, 4, 4, 2, 1, 3, 2, 2, 3, 2, 1, 2, 3, 4, 1, 2, 1];

julia> y_true = [ 4, 4, 1, 4, 4, 2, 1, 1, 2, 4, 1, 3, 3, 3, 1, 1, 3, 1, 4, 4, 3, 3, 3, 1, 4, 1, 2, 3, 2, 2];

julia> x = confusion_matrix(y_true, y_pred)

julia> predicted_negative(x)
4-element Array{Int64,1}:
 21
 19
 25
 25

julia> predicted_negative(x, average= "macro")
22.5

julia> predicted_negative(x, normalize = true)
4-element Array{Float64,1}:
 0.46358632497276536
 0.41943524640393054
 0.551888482110435
 0.551888482110435

julia> predicted_negative(x, ith_class = 3)
25

julia> predicted_negative(x) == predicted_negative(y_true, y_pred)
true

```

See also : [`confusion_matrix`](@ref), [`condition_negative`](@ref), [`predicted_positive`](@ref), [`condition_positive`](@ref)

"""
function predicted_negative(c::confusion_matrix; ith_class = nothing, class_name = nothing, average = "binary", weights = nothing, normalize = false)
    @assert average in ["binary", "macro", "weighted", "micro", "sample-weights"] "Unknown averaging mode. This function only supports the following types: binary, macro, weighted, sample-weights"
    if average == "sample-weights"; @assert weights != nothing && length(weights) == length(c.Labels) """If the average mode is weighted, weights that are the same size as the labels must be provided!
    If no precalculated weights can be provided but the class imbalance is to be taken into account try ' average = "weighted" ' or  ' average = "micro" ' """; end

    index = check_index(c.Labels, true, ith_class = ith_class, class_name = class_name)
    if index == -1
        x = c.true_negatives .+ c.false_negatives
    else
        x = c.true_negatives[index] + c.false_negatives[index]
    end
    if average == "weighted"; weights = c.true_positives .+ c.false_negatives ; end
    return _average_helper(x, nothing, weights, average, c.zero_division, normalize )
end

predicted_negative(expected, predicted; ith_class = nothing, class_name = nothing, average = "binary", normalize = false) =
predicted_negative(confusion_matrix(expected,predicted); ith_class = ith_class, class_name = class_name, average = average, normalize)

"""
```correctly_classified(c::confusion_matrix; keywords)```
```correctly_classified(y_true, y_pred; keywords)```

Return number of correctly classified instances of either the whole confusion matrix or the classes specified by `class_name` or `ith_class`
arguments.

    Correctly Classified Values: True Positives + True Negatives

## Arguments
- `c::confusion_matrix` : The confusion matrix to report for
- `y_true::Vector` : True values for classification
- `y_pred::Vector` : Predicted values for classification

## Keywords

- `ith_class::Int = nothing` : Return the results for the ith class, the ith label in the label list of the given confusion matrix object.

- `class_name` : Return the results for the class of the specified value in the label list of the given confusion matrix.

If both `class_name` and `ith_class` arguments are equal to `nothing`, return condition positive values for all the elements in the labels arrays

- `average::String = "binary"` :
\n\t   `"binary"` : Return the classwise values.
\n\t   `"macro"` : Return the macro average (mean) of the classwise values.
\n\t    `"micro"` : Return micro average (sum of the numerator divided by sum of the denominator instead of elementwise division) of the classwise values
\n\t    `"weighted"` :
        Return the weighted average (weighted mean with true positives per class) of the classwise values.
\n\t    `"sample-weights"` : Return the weighted average (weighted mean with given weights per class) of the classwise values.

- `weights::Vector  nothing` :  Use the given weights whilst calculating 'sample_weights' option. If average is not 'sample-weigts' this will be ignored.

- `normalize::Bool = false` : If true, normalize the result.


## Example

First example no indexing:\n\n

```julia-repl
julia> y_pred = [ 1, 4, 4, 1, 2, 3, 1, 3, 2, 1, 2, 2, 2, 1, 4, 4, 2, 1, 3, 2, 2, 3, 2, 1, 2, 3, 4, 1, 2, 1];

julia> y_true = [ 4, 4, 1, 4, 4, 2, 1, 1, 2, 4, 1, 3, 3, 3, 1, 1, 3, 1, 4, 4, 3, 3, 3, 1, 4, 1, 2, 3, 2, 2];

julia> x = confusion_matrix(y_true, y_pred)

julia> correctly_classified(x)
4-element Array{Int64,1}:
 18
 18
 19
 19

julia> correctly_classified(x, normalize = true)
4-element Array{Float64,1}:
 0.48630890426246925
 0.48630890426246925
 0.5133260656103842
 0.5133260656103842

julia> correctly_classified(x, average = "micro")
74

julia> correctly_classified(x, weights = rand(4))
4-element Array{Int64,1}:
 18
 18
 19
 19

julia> correctly_classified(x, ith_class = 3)
19

julia> correctly_classified(x) == correctly_classified(y_true, y_pred)
true

```

See also : [`confusion_matrix]`(@ref), [`predicted_negative`](@ref), [`predicted_positive`](@ref), [`incorrectly_classified`](@ref)

"""
function correctly_classified(c::confusion_matrix; ith_class = nothing, class_name = nothing, average = "binary", weights = nothing, normalize = false)
    @assert average in ["binary", "macro", "weighted", "micro", "sample-weights"] "Unknown averaging mode. This function only supports the following types: binary, macro, weighted, sample-weights"
    if average == "sample-weights"; @assert weights != nothing && length(weights) == length(c.Labels) """If the average mode is weighted, weights that are the same size as the labels must be provided!
    If no precalculated weights can be provided but the class imbalance is to be taken into account try ' average = "weighted" ' or  ' average = "micro" ' """; end

    index = check_index(c.Labels, true, ith_class = ith_class, class_name = class_name)
    if index == -1
        x = c.true_positives .+ c.true_negatives
    else
        x = c.true_positives[index] + c.true_negatives[index]
    end
    if average == "weighted"; weights = c.true_positives .+ c.false_negatives ; end
    return _average_helper(x, nothing, weights, average, c.zero_division, normalize)
end


correctly_classified(expected, predicted; ith_class = nothing, class_name = nothing, average = "binary", normalize = false)  =
correctly_classified(confusion_matrix(expected,predicted); ith_class = ith_class, class_name = class_name, average = average, normalize = normalize)

"""
```incorrectly_classified(c::confusion_matrix; keywords)```
```incorrectly_classified(y_true, y_pred; keywords)```

Return number of incorrectly classified instances of either the whole confusion matrix or the classes specified by `class_name` or `ith_class`
arguments.

    Inorrectly Classified: False Negatives + False Positives

## Arguments
- `c::confusion_matrix` : The confusion matrix to report for
- `y_true::Vector` : True values for classification
- `y_pred::Vector` : Predicted values for classification

## Keywords

- `ith_class::Int = nothing` : Return the results for the ith class, the ith label in the label list of the given confusion matrix object.

- `class_name` : Return the results for the class of the specified value in the label list of the given confusion matrix.

If both `class_name` and `ith_class` arguments are equal to `nothing`, return condition positive values for all the elements in the labels arrays

- `average::String = "binary"` :
\n\t   `"binary"` : Return the classwise values.
\n\t   `"macro"` : Return the macro average (mean) of the classwise values.
\n\t    `"micro"` : Return micro average (sum of the numerator divided by sum of the denominator instead of elementwise division) of the classwise values
\n\t    `"weighted"` :
            Return the weighted average (weighted mean with true positives per class) of the classwise values.
\n\t    `"sample-weights"` : Return the weighted average (weighted mean with given weights per class) of the classwise values.

- `weights::Vector  nothing` :  Use the given weights whilst calculating 'sample_weights' option. If average is not 'sample-weigts' this will be ignored.

- `normalize::Bool = false` : If true, normalize the result.

## Example

First example no indexing:\n\n

```julia-repl
julia> y_pred = [ 1, 4, 4, 1, 2, 3, 1, 3, 2, 1, 2, 2, 2, 1, 4, 4, 2, 1, 3, 2, 2, 3, 2, 1, 2, 3, 4, 1, 2, 1];

julia> y_true = [ 4, 4, 1, 4, 4, 2, 1, 1, 2, 4, 1, 3, 3, 3, 1, 1, 3, 1, 4, 4, 3, 3, 3, 1, 4, 1, 2, 3, 2, 2];

julia> x = confusion_matrix(y_true, y_pred)

julia> incorrectly_classified(x)
4-element Array{Int64,1}:
 12
 12
 11
 11

julia> incorrectly_classified(x, average = "macro")
11.5

julia> incorrectly_classified(x, average = "micro")
46

julia> incorrectly_classified(x, normalize = true)
4-element Array{Float64,1}:
 0.5212466913156832
 0.5212466913156832
 0.4778094670393763
 0.4778094670393763

julia> incorrectly_classified(x, average = "sample-weights", weights = [0.25, 0.1, 0.1, 0.55])
11.350000000000001

julia> incorrectly_classified(x, ith_class = 3)
11

julia> incorrectly_classified(x) == incorrectly_classified(y_true, y_pred)
true

```

See also : [`confusion_matrix`](@ref), [`predicted_negative`](@ref), [`predicted_positive`](@ref), [`correctly_classified`](@ref)

"""
function incorrectly_classified(c::confusion_matrix; ith_class = nothing, class_name = nothing, average = "binary", weights = nothing, normalize = false)
    @assert average in ["binary", "macro", "weighted", "micro", "sample-weights"] "Unknown averaging mode. This function only supports the following types: binary, macro, weighted, sample-weights"
    if average == "sample-weights"; @assert weights != nothing && length(weights) == length(c.Labels) """If the average mode is weighted, weights that are the same size as the labels must be provided!
    If no precalculated weights can be provided but the class imbalance is to be taken into account try ' average = "weighted" ' or  ' average = "micro" ' """; end

    index = check_index(c.Labels, true, ith_class = ith_class, class_name = class_name)
    if index == -1
        x = c.false_positives .+ c.false_negatives
    else
        x = c.false_positives[index] + c.false_negatives[index]
    end
    if average == "weighted"; weights = c.true_positives .+ c.false_negatives ; end
    return _average_helper(x, nothing, weights, average, c.zero_division, normalize)
end

incorrectly_classified(expected, predicted; ith_class = nothing, class_name = nothing, average = "binary", normalize = false) =
incorrectly_classified(confusion_matrix(expected,predicted), ith_class = ith_class, class_name = class_name, average = average, normalize = normalize)

"""
```sensitivity_score(c::confusion_matrix; keywords)```
```sensitivity_score(y_true, y_pred; keywords)```

Return sensitivity (recall) score of either the whole confusion matrix or the classes specified by `class_name` or `ith_class`
arguments.

## Arguments
- `c::confusion_matrix` : The confusion matrix to report for
- `y_true::Vector` : True values for classification
- `y_pred::Vector` : Predicted values for classification

## Keywords

- `ith_class::Int = nothing` : Return the results for the ith class, the ith label in the label list of the given confusion matrix object.

- `class_name` : Return the results for the class of the specified value in the label list of the given confusion matrix.

If both `class_name` and `ith_class` arguments are equal to `nothing`, return condition positive values for all the elements in the labels arrays

- `average::String = "macro"` :
\n\t   `"binary"` : Return the classwise values.
\n\t   `"macro"` : Return the macro average (mean) of the classwise values.
\n\t    `"micro"` : Return micro average (sum of the numerator divided by sum of the denominator instead of elementwise division) of the classwise values
\n\t    `"weighted"` :
        Return the weighted average (weighted mean with true positives per class) of the classwise values.
\n\t    `"sample-weights"` : Return the weighted average (weighted mean with given weights per class) of the classwise values.

- `weights::Vector  nothing` :  Use the given weights whilst calculating 'sample_weights' option. If average is not 'sample-weigts' this will be ignored.

- `normalize::Bool = false` : If true, normalize the result.

## Examples

```julia-repl
julia> y_pred = [ 1, 4, 4, 1, 2, 3, 1, 3, 2, 1, 2, 2, 2, 1, 4, 4, 2, 1, 3, 2, 2, 3, 2, 1, 2, 3, 4, 1, 2, 1];

julia> y_true = [ 4, 4, 1, 4, 4, 2, 1, 1, 2, 4, 1, 3, 3, 3, 1, 1, 3, 1, 4, 4, 3, 3, 3, 1, 4, 1, 2, 3, 2, 2];

julia> x = confusion_matrix(y_true, y_pred)

julia> sensitivity_score(x)
0.24583333333333335

julia> sensitivity_score(x, average = "binary")
4-element Array{Float64,1}:
 0.3333333333333333
 0.4
 0.125
 0.125

julia> sensitivity_score(x, average = "binary", normalize = true)
4-element Array{Float64,1}:
 0.6061997863600779
 0.7274397436320935
 0.2273249198850292
 0.2273249198850292

julia> sensitivity_score(x, average = "micro")
0.23333333333333334

julia> sensitivity_score(x, average = "sample-weights", weights= [1,2,3,4])
0.20083333333333334

julia> sensitivity_score(x, class_name = 2)
0.4

julia> sensitivity_score(x) == sensitivity_score(y_true, y_pred)
true

```

See also : [`confusion_matrix`](@ref) , [`recall_score`](@ref) ,  [`balanced_accuracy_score`](@ref), [`specificity_score`](@ref)
"""
function sensitivity_score(c::confusion_matrix; ith_class = nothing, class_name = nothing , average = "macro", weights = nothing, normalize = false)
    @assert average in ["binary", "macro", "weighted", "micro", "sample-weights"] "Unknown averaging mode. This function only supports the following types: binary, macro, weighted, sample-weights"
    if average == "sample-weights"; @assert weights != nothing && length(weights) == length(c.Labels) """If the average mode is weighted, weights that are the same size as the labels must be provided!
    If no precalculated weights can be provided but the class imbalance is to be taken into account try ' average = "weighted" ' or  ' average = "micro" ' """; end

    index = check_index(c.Labels, true, ith_class = ith_class, class_name = class_name)
    if index == -1
        numerator = c.true_positives
        denominator = condition_positive(c)
    else
        numerator = c.true_positives[index]
        denominator = condition_positive(c, ith_class = index)
    end
    if average == "weighted"; weights = c.true_positives .+ c.false_negatives ; end
    return _average_helper(numerator, denominator, weights, average, c.zero_division, normalize)
end

sensitivity_score(expected, predicted; ith_class = nothing, class_name = nothing, average = "macro", normalize = false)  =
sensitivity_score(confusion_matrix(expected,predicted), ith_class = ith_class, class_name = class_name, average = average, normalize = normalize)

"""
```recall_score(c::confusion_matrix; keywords)```
```recall_score(y_true, y_pred; keywords)```

Return recall(sensitivity) score of either the whole confusion matrix or the classes specified by `class_name` or `ith_class`
arguments.

## Arguments
- `c::confusion_matrix` : The confusion matrix to report for
- `y_true::Vector` : True values for classification
- `y_pred::Vector` : Predicted values for classification

## Keywords

- `ith_class::Int = nothing` : Return the results for the ith class, the ith label in the label list of the given confusion matrix object.

- `class_name` : Return the results for the class of the specified value in the label list of the given confusion matrix.

If both `class_name` and `ith_class` arguments are equal to `nothing`, return condition positive values for all the elements in the labels arrays

- `average::String = "macro"` :
\n\t   `"binary"` : Return the classwise values.
\n\t   `"macro"` : Return the macro average (mean) of the classwise values.
\n\t    `"micro"` : Return micro average (sum of the numerator divided by sum of the denominator instead of elementwise division) of the classwise values
\n\t    `"weighted"` :
        Return the weighted average (weighted mean with true positives per class) of the classwise values.
\n\t    `"sample-weights"` : Return the weighted average (weighted mean with given weights per class) of the classwise values.

- `weights::Vector  nothing` :  Use the given weights whilst calculating 'sample_weights' option. If average is not 'sample-weigts' this will be ignored.

- `normalize::Bool = false` : If true, normalize the result.

## Examples

```julia-repl
julia> y_pred = [ 1, 4, 4, 1, 2, 3, 1, 3, 2, 1, 2, 2, 2, 1, 4, 4, 2, 1, 3, 2, 2, 3, 2, 1, 2, 3, 4, 1, 2, 1];

julia> y_true = [ 4, 4, 1, 4, 4, 2, 1, 1, 2, 4, 1, 3, 3, 3, 1, 1, 3, 1, 4, 4, 3, 3, 3, 1, 4, 1, 2, 3, 2, 2];

julia> x = confusion_matrix(y_true, y_pred)

julia> recall_score(x)
0.24583333333333335

julia> recall_score(x, average = "binary")
4-element Array{Float64,1}:
 0.3333333333333333
 0.4
 0.125
 0.125

julia> recall_score(x, average = "binary", normalize = true)
4-element Array{Float64,1}:
 0.6061997863600779
 0.7274397436320935
 0.2273249198850292
 0.2273249198850292

julia> recall_score(x, average = "micro")
0.23333333333333334

julia> recall_score(x, average = "sample-weights", weights = [4,3,2,1])
0.29083333333333333

julia> recall_score(x, class_name = 2)
0.4

julia> recall_score(x) == recall_score(y_true, y_pred)
true

```

See also : [`confusion_matrix`](@ref) , [`sensitivity_score`](@ref) ,  [`balanced_accuracy_score`](@ref), [`specificity_score`](@ref)

"""
function recall_score(c::confusion_matrix; ith_class = nothing, class_name = nothing, average = "macro", normalize = false, weights = nothing)
    return sensitivity_score(c, ith_class = ith_class, class_name = class_name, average = average, normalize = normalize, weights = weights)
end

recall_score(expected, predicted; ith_class = nothing, class_name = nothing, average = "macro", normalize = false, weights = nothing) =
recall_score(confusion_matrix(expected,predicted), ith_class = ith_class, class_name = class_name, average = average, normalize = normalize, weights = weights)

"""
```specificity_score(c::confusion_matrix; keywords)```
```specificity_score(y_true, y_pred; keywords)```

Return specificity score of either the whole confusion matrix or the classes specified by `class_name` or `ith_class`
arguments.

## Arguments
- `c::confusion_matrix` : The confusion matrix to report for
- `y_true::Vector` : True values for classification
- `y_pred::Vector` : Predicted values for classification

## Keywords

- `ith_class::Int = nothing` : Return the results for the ith class, the ith label in the label list of the given confusion matrix object.

- `class_name` : Return the results for the class of the specified value in the label list of the given confusion matrix.

If both `class_name` and `ith_class` arguments are equal to `nothing`, return condition positive values for all the elements in the labels arrays

- `average::String = "macro"` :
\n\t   `"binary"` : Return the classwise values.
\n\t   `"macro"` : Return the macro average (mean) of the classwise values.
\n\t    `"micro"` : Return micro average (sum of the numerator divided by sum of the denominator instead of elementwise division) of the classwise values
\n\t    `"weighted"` :
        Return the weighted average (weighted mean with true positives per class) of the classwise values.
\n\t    `"sample-weights"` : Return the weighted average (weighted mean with given weights per class) of the classwise values.

- `weights::Vector  nothing` :  Use the given weights whilst calculating 'sample_weights' option. If average is not 'sample-weigts' this will be ignored.

- `normalize::Bool = false` : If true, normalize the result.

## Examples

```julia-repl
julia> y_pred = [ 1, 4, 4, 1, 2, 3, 1, 3, 2, 1, 2, 2, 2, 1, 4, 4, 2, 1, 3, 2, 2, 3, 2, 1, 2, 3, 4, 1, 2, 1];

julia> y_true = [ 4, 4, 1, 4, 4, 2, 1, 1, 2, 4, 1, 3, 3, 3, 1, 1, 3, 1, 4, 4, 3, 3, 3, 1, 4, 1, 2, 3, 2, 2];

julia> x = confusion_matrix(y_true, y_pred)

julia> specificity_score(x)
0.7476623376623377

julia> specificity_score(x, average = "binary")
4-element Array{Float64,1}:
 0.7142857142857143
 0.64
 0.8181818181818182
 0.8181818181818182

julia> specificity_score(x, average = "binary", normalize = true)
4-element Array{Float64,1}:
 0.47527807274822387
 0.4258491531824086
 0.5444094287843292
 0.5444094287843292

julia> specificity_score(x, average = "weighted")
0.7573160173160175

julia> specificity_score(x, class_name = 1)
0.7142857142857143

julia> specificity_score(x) == specificity_score(y_true, y_pred)
true

```

See also : [`confusion_matrix`](@ref), [`sensitivity_score`](@ref), [`balanced_accuracy_score`](@ref),[`recall_score`](@ref)

"""
function specificity_score(c::confusion_matrix; ith_class = nothing, class_name = nothing, average = "macro", weights = nothing, normalize = false)
    @assert average in ["binary", "macro", "weighted", "micro", "sample-weights"] "Unknown averaging mode. This function only supports the following types: binary, macro, weighted, sample-weights"
    if average == "sample-weights"; @assert weights != nothing && length(weights) == length(c.Labels) """If the average mode is weighted, weights that are the same size as the labels must be provided!
    If no precalculated weights can be provided but the class imbalance is to be taken into account try ' average = "weighted" ' or  ' average = "micro" ' """; end

    index = check_index(c.Labels, true, ith_class = ith_class, class_name = class_name)
    if index == -1
        numerator = c.true_negatives
        denominator = condition_negative(c)
    else
        numerator = c.true_negatives[index]
        denominator =  condition_negative(c, ith_class = index)
    end
    if average == "weighted"; weights = c.true_positives .+ c.false_negatives ; end
    return _average_helper(numerator, denominator, weights, average, c.zero_division, normalize)
end


specificity_score(expected, predicted; ith_class = nothing, class_name = nothing, average = "macro", normalize = false)  =
specificity_score(confusion_matrix(expected,predicted), ith_class = ith_class, class_name = class_name, average = average, normalize = normalize)

"""
```precision_score(c::confusion_matrix; keywords)```
```precision_score(y_true, y_pred; keywords)```

Return precision score of either the whole confusion matrix or the classes specified by `class_name` or `ith_class`
arguments.

## Arguments
- `c::confusion_matrix` : The confusion matrix to report for
- `y_true::Vector` : True values for classification
- `y_pred::Vector` : Predicted values for classification

## Keywords

- `ith_class::Int = nothing` : Return the results for the ith class, the ith label in the label list of the given confusion matrix object.

- `class_name` : Return the results for the class of the specified value in the label list of the given confusion matrix.

If both `class_name` and `ith_class` arguments are equal to `nothing`, return condition positive values for all the elements in the labels arrays

- `average::String = "macro"` :
\n\t   `"binary"` : Return the classwise values.
\n\t   `"macro"` : Return the macro average (mean) of the classwise values.
\n\t    `"micro"` : Return micro average (sum of the numerator divided by sum of the denominator instead of elementwise division) of the classwise values
\n\t    `"weighted"` :
        Return the weighted average (weighted mean with true positives per class) of the classwise values.
\n\t    `"sample-weights"` : Return the weighted average (weighted mean with given weights per class) of the classwise values.

- `weights::Vector  nothing` :  Use the given weights whilst calculating 'sample_weights' option. If average is not 'sample-weigts' this will be ignored.

- `normalize::Bool = false` : If true, normalize the result.

## Examples

```julia-repl
julia> y_pred = [ 1, 4, 4, 1, 2, 3, 1, 3, 2, 1, 2, 2, 2, 1, 4, 4, 2, 1, 3, 2, 2, 3, 2, 1, 2, 3, 4, 1, 2, 1];

julia> y_true = [ 4, 4, 1, 4, 4, 2, 1, 1, 2, 4, 1, 3, 3, 3, 1, 1, 3, 1, 4, 4, 3, 3, 3, 1, 4, 1, 2, 3, 2, 2];

julia> x = confusion_matrix(y_true, y_pred)

julia> precision_score(x)
0.22878787878787876

julia> precision_score(x, average = "binary")
4-element Array{Float64,1}:
 0.3333333333333333
 0.18181818181818182
 0.2
 0.2

julia> precision_score(x, average = "micro")
0.23333333333333334

julia> precision_score(x, average = "weighted")
0.23696969696969697

julia>  precision_score(x, class_name = 3)
0.3333333333333333

julia> precision_score(x) == precision_score(y_true, y_pred)
true

```

See Also : [`confusion_matrix`](@ref), [`sensitivity_score`](@ref) , [`balanced_accuracy_score`](@ref), [`recall_score`](@ref)

"""
function precision_score(c::confusion_matrix; ith_class = nothing, class_name = nothing, average = "macro", weights = nothing, normalize = false)
    @assert average in ["binary", "macro", "weighted", "micro", "sample-weights"] "Unknown averaging mode. This function only supports the following types: binary, macro, weighted, sample-weights"
    if average == "sample-weights"; @assert weights != nothing && length(weights) == length(c.Labels) """If the average mode is weighted, weights that are the same size as the labels must be provided!
    If no precalculated weights can be provided but the class imbalance is to be taken into account try ' average = "weighted" ' or  ' average = "micro" ' """; end

    index = check_index(c.Labels, true, ith_class = ith_class, class_name = class_name)
    if index == -1
        numerator = c.true_positives
        denominator = c.true_positives .+ c.false_positives
    else
        numerator = c.true_positives[index]
        denominator = c.true_positives[index] + c.false_positives[index]
    end
    if average == "weighted"; weights = c.true_positives .+ c.false_negatives ; end
    return _average_helper(numerator, denominator, weights, average, c.zero_division, normalize)
end


precision_score(expected, predicted; ith_class = nothing, class_name = nothing, average = "macro", normalize = false) =
precision_score(confusion_matrix(expected,predicted), ith_class = ith_class, class_name = class_name, average = average, normalize = normalize)

"""
```positive_predictive_value(c::confusion_matrix; keywords)```
```positive_predictive_value(y_true, y_pred; keywords)```

Return  score of either the whole confusion matrix or the classes specified by `class_name` or `ith_class`
arguments.

## Arguments
- `c::confusion_matrix` : The confusion matrix to report for
- `y_true::Vector` : True values for classification
- `y_pred::Vector` : Predicted values for classification

## Keywords

- `ith_class::Int = nothing` : Return the results for the ith class, the ith label in the label list of the given confusion matrix object.

- `class_name` : Return the results for the class of the specified value in the label list of the given confusion matrix.

If both `class_name` and `ith_class` arguments are equal to `nothing`, return condition positive values for all the elements in the labels arrays

- `average::String = "macro"` :
\n\t   `"binary"` : Return the classwise values.
\n\t   `"macro"` : Return the macro average (mean) of the classwise values.
\n\t    `"micro"` : Return micro average (sum of the numerator divided by sum of the denominator instead of elementwise division) of the classwise values
\n\t    `"weighted"` :
        Return the weighted average (weighted mean with true positives per class) of the classwise values.
\n\t    `"sample-weights"` : Return the weighted average (weighted mean with given weights per class) of the classwise values.

- `weights::Vector  nothing` :  Use the given weights whilst calculating 'sample_weights' option. If average is not 'sample-weigts' this will be ignored.

- `normalize::Bool = false` : If true, normalize the result.

## Examples

```julia-repl
julia> y_pred = [ 1, 4, 4, 1, 2, 3, 1, 3, 2, 1, 2, 2, 2, 1, 4, 4, 2, 1, 3, 2, 2, 3, 2, 1, 2, 3, 4, 1, 2, 1];

julia> y_true = [ 4, 4, 1, 4, 4, 2, 1, 1, 2, 4, 1, 3, 3, 3, 1, 1, 3, 1, 4, 4, 3, 3, 3, 1, 4, 1, 2, 3, 2, 2];

julia> x = confusion_matrix(y_true, y_pred)

julia> positive_predictive_value(x)
0.22878787878787876

julia> positive_predictive_value(x, average = "binary")
4-element Array{Float64,1}:
 0.3333333333333333
 0.18181818181818182
 0.2
 0.2

julia> positive_predictive_value(x, average = "binary", normalize = true)
4-element Array{Float64,1}:
 0.7040297388442665
 0.3840162211877817
 0.4224178433065599
 0.4224178433065599

julia> positive_predictive_value(x, average = "micro")
0.23333333333333334

julia> positive_predictive_value(x, ith_class = 2)
0.18181818181818182

julia> positive_predictive_value(x) == precision_score(y_true, y_pred)
true

```

See Also : [`negative_predictive_value`](@ref), [`confusion_matrix`](@ref), [`sensitivity_score`](@ref), [`balanced_accuracy_score`](@ref),[`recall_score`](@ref)

"""
function positive_predictive_value(c::confusion_matrix; ith_class = nothing, class_name = nothing, average = "macro", normalize = false, weights = nothing)
   return precision_score(c, class_name = class_name, ith_class = ith_class, average = average, normalize = normalize, weights = weights)
end

positive_predictive_value(expected, predicted; ith_class = nothing, class_name = nothing, average = "macro", normalize = false, weights = nothing) =
positive_predictive_value(confusion_matrix(expected,predicted), ith_class = ith_class, class_name = class_name, average = average, normalize = normalize, weights = weights)

"""
```accuracy_score(c::confusion_matrix; keywords) ```
```accuracy_score(y_true, y_pred; keywords) ```

Return accuracy classification score.

## Arguments
- `c::confusion_matrix` : The confusion matrix to report for
- `y_true::Vector` : True values for classification
- `y_pred::Vector` : Predicted values for classification

## Keywords

- `ith_class::Int = nothing` : Return the results for the ith class, the ith label in the label list of the given confusion matrix object.

- `class_name` : Return the results for the class of the specified value in the label list of the given confusion matrix.

If both `class_name` and `ith_class` arguments are equal to `nothing`, return condition positive values for all the elements in the labels arrays

- `average::String = "macro"` :
\n\t   `"binary"` : Return the classwise values.
\n\t   `"macro"` : Return the macro average (mean) of the classwise values.
\n\t    `"micro"` : Return micro average (sum of the numerator divided by sum of the denominator instead of elementwise division) of the classwise values
\n\t    `"weighted"` :
        Return the weighted average (weighted mean with true positives per class) of the classwise values.
\n\t    `"sample-weights"` : Return the weighted average (weighted mean with given weights per class) of the classwise values.

- `weights::Vector  nothing` :  Use the given weights whilst calculating 'sample_weights' option. If average is not 'sample-weigts' this will be ignored.

- `normalize::Bool = false` : If true, normalize the result.

## Examples

```julia-repl
julia> y_pred = [ 1, 4, 4, 1, 2, 3, 1, 3, 2, 1, 2, 2, 2, 1, 4, 4, 2, 1, 3, 2, 2, 3, 2, 1, 2, 3, 4, 1, 2, 1];

julia> y_true = [ 4, 4, 1, 4, 4, 2, 1, 1, 2, 4, 1, 3, 3, 3, 1, 1, 3, 1, 4, 4, 3, 3, 3, 1, 4, 1, 2, 3, 2, 2];

julia> x = confusion_matrix(y_true, y_pred)

julia> accuracy_score(x)
0.22878787878787876

julia> accuracy_score(x, average = "binary")
4-element Array{Float64,1}:
 0.7040297388442665
 0.3840162211877817
 0.4224178433065599
 0.4224178433065599

julia> accuracy_score(x, average = "binary", normalize = true)
4-element Array{Float64,1}:
 0.7040297388442665
 0.3840162211877817
 0.4224178433065599
 0.4224178433065599

julia> accuracy_score(x, average = "micro")
0.23333333333333334

julia> accuracy_score(x) == accuracy_score(y_true, y_pred)
true

```

See Also : [`jaccard_score`](@ref), [`confusion_matrix`](@ref), [`hamming_loss`](@ref), [`balanced_accuracy_score`](@ref), [`recall_score`](@ref)


"""
function accuracy_score(c::confusion_matrix; ith_class = nothing, class_name = nothing, average = "macro", normalize = true, weights = nothing)
    @assert average in ["binary", "macro", "weighted", "micro", "sample-weights"] "Unknown averaging mode. This function only supports the following types: binary, macro, weighted, sample-weights"
    if average == "sample-weights"; @assert weights != nothing && length(weights) == length(c.Labels) """If the average mode is weighted, weights that are the same size as the labels must be provided!
    If no precalculated weights can be provided but the class imbalance is to be taken into account try ' average = "weighted" ' or  ' average = "micro" ' """; end

    index = check_index(c.Labels, true, ith_class = ith_class, class_name = class_name)
    if index == -1
        numerator = (c.true_positives)
        denominator = (c.true_positives .+ c.false_positives)
    else
        numerator = (c.true_positives[index])
        denominator = (c.true_positives[index] .+ c.false_positives[index])
    end
    if average == "weighted"; weights = c.true_positives .+ c.false_negatives ; end
    return _average_helper(numerator, denominator, weights, average, c.zero_division, normalize)
end

accuracy_score(expected, predicted; ith_class = nothing, class_name = nothing, average = "macro", normalize = true, weights = nothing)  =
accuracy_score(confusion_matrix(expected,predicted), ith_class = ith_class, class_name = class_name, normalize = normalize, weights = weights, average = average)

"""
```balanced_accuracy_score(c::confusion_matrix; keywords) ```
```balanced_accuracy_score(y_true, y_pred; keywords) ```

Return balanced accuracy classification score.

## Arguments
- `c::confusion_matrix` : The confusion matrix to report for
- `y_true::Vector` : True values for classification
- `y_pred::Vector` : Predicted values for classification

## Keywords

- `ith_class::Int = nothing` : Return the results for the ith class, the ith label in the label list of the given confusion matrix object.

- `class_name` : Return the results for the class of the specified value in the label list of the given confusion matrix.

If both `class_name` and `ith_class` arguments are equal to `nothing`, return condition positive values for all the elements in the labels arrays

- `average::String = "macro"` :
\n\t   `"binary"` : Return the classwise values.
\n\t   `"macro"` : Return the macro average (mean) of the classwise values.
\n\t    `"micro"` : Return micro average (sum of the numerator divided by sum of the denominator instead of elementwise division) of the classwise values
\n\t    `"weighted"` :
        Return the weighted average (weighted mean with true positives per class) of the classwise values.
\n\t    `"sample-weights"` : Return the weighted average (weighted mean with given weights per class) of the classwise values.

- `weights::Vector  nothing` :  Use the given weights whilst calculating 'sample_weights' option. If average is not 'sample-weigts' this will be ignored.

- `normalize::Bool = false` : If true, normalize the result.

## Examples

```julia-repl
julia> y_pred = [ 1, 4, 4, 1, 2, 3, 1, 3, 2, 1, 2, 2, 2, 1, 4, 4, 2, 1, 3, 2, 2, 3, 2, 1, 2, 3, 4, 1, 2, 1];

julia> y_true = [ 4, 4, 1, 4, 4, 2, 1, 1, 2, 4, 1, 3, 3, 3, 1, 1, 3, 1, 4, 4, 3, 3, 3, 1, 4, 1, 2, 3, 2, 2];

julia> x = confusion_matrix(y_true, y_pred)

julia> balanced_accuracy_score(x)
0.24583333333333335

julia> balanced_accuracy_score(x, average = "micro")
0.9833333333333334

julia>  balanced_accuracy_score(x, average = "macro")
0.24583333333333335

julia> balanced_accuracy_score(x, average = "weighted")
0.23333333333333334

julia> balanced_accuracy_score(x, average = "binary")
4-element Array{Float64,1}:
 0.3333333333333333
 0.4
 0.125
 0.125

julia> balanced_accuracy_score(x, average = "binary", normalize = true)
4-element Array{Float64,1}:
 0.6061997863600779
 0.7274397436320935
 0.2273249198850292
 0.2273249198850292

julia> balanced_accuracy_score(x) == balanced_accuracy_score(y_true, y_pred)
true

```

See Also : [`accuracy_score`](@ref) , [`confusion_matrix`](@ref) , [`hamming_loss`](@ref), [`balanced_accuracy_score`](@ref) ,[`recall_score`](@ref)

"""
function balanced_accuracy_score(c::confusion_matrix; ith_class = nothing, class_name = nothing , average = "macro", weights = nothing, normalize = false)
    @assert average in ["binary", "macro", "weighted", "micro", "sample-weights"] "Unknown averaging mode. This function only supports the following types: binary, macro, weighted, sample-weights"
    if average == "sample-weights"; @assert weights != nothing && length(weights) == length(c.Labels) """If the average mode is weighted, weights that are the same size as the labels must be provided!
    If no precalculated weights can be provided but the class imbalance is to be taken into account try ' average = "weighted" ' or  ' average = "micro" ' """; end
    index = check_index(c.Labels, true, ith_class = ith_class, class_name = class_name)

    index = check_index(c.Labels, true, ith_class = ith_class, class_name = class_name)
    if index == -1
        numerator = sensitivity_score(c, average = "binary") .+ recall_score(c, average = "binary")
        denominator = 2
    else
        numerator = sensitivity_score(c) .+ recall_score(c)
        denominator = 2
    end
    if average == "weighted"; weights = c.true_positives .+ c.false_negatives ; end
    return _average_helper(numerator, denominator, weights, average, c.zero_division, normalize)
end

balanced_accuracy_score(expected, predicted; ith_class = nothing, class_name = nothing, average = "macro", normalize = false)  =
balanced_accuracy_score(confusion_matrix(expected,predicted), ith_class = ith_class, class_name = class_name, average = average, normalize = normalize)

"""
```negative_predictive_value(c::confusion_matrix; keywords) ```
```negative_predictive_value(y_true, y_pred; keywords) ```

Return negative predictive value for the specified class(es).

## Arguments
- `c::confusion_matrix` : The confusion matrix to report for
- `y_true::Vector` : True values for classification
- `y_pred::Vector` : Predicted values for classification

## Keywords

- `ith_class::Int = nothing` : Return the results for the ith class, the ith label in the label list of the given confusion matrix object.

- `class_name` : Return the results for the class of the specified value in the label list of the given confusion matrix.

If both `class_name` and `ith_class` arguments are equal to `nothing`, return condition positive values for all the elements in the labels arrays

- `average::String = "macro"` :
\n\t   `"binary"` : Return the classwise values.
\n\t   `"macro"` : Return the macro average (mean) of the classwise values.
\n\t    `"micro"` : Return micro average (sum of the numerator divided by sum of the denominator instead of elementwise division) of the classwise values
\n\t    `"weighted"` :
        Return the weighted average (weighted mean with true positives per class) of the classwise values.
\n\t    `"sample-weights"` : Return the weighted average (weighted mean with given weights per class) of the classwise values.

- `weights::Vector  nothing` :  Use the given weights whilst calculating 'sample_weights' option. If average is not 'sample-weigts' this will be ignored.

- `normalize::Bool = false` : If true, normalize the result.

## Examples

```julia-repl
julia> y_pred = [ 1, 4, 4, 1, 2, 3, 1, 3, 2, 1, 2, 2, 2, 1, 4, 4, 2, 1, 3, 2, 2, 3, 2, 1, 2, 3, 4, 1, 2, 1];

julia> y_true = [ 4, 4, 1, 4, 4, 2, 1, 1, 2, 4, 1, 3, 3, 3, 1, 1, 3, 1, 4, 4, 3, 3, 3, 1, 4, 1, 2, 3, 2, 2];

julia> x = confusion_matrix(y_true, y_pred)

julia> negative_predictive_value(x)
0.7490977443609022

julia> negative_predictive_value(x, average = "binary")
4-element Array{Float64,1}:
 0.7142857142857143
 0.8421052631578947
 0.72
 0.72

julia> negative_predictive_value(x, average = "binary", normalize = true)
4-element Array{Float64,1}:
 0.47554150307001175
 0.5606384036193822
 0.4793458350945718
 0.4793458350945718

julia> negative_predictive_value(x, average = "micro")
0.7444444444444445

julia> negative_predictive_value(x, average = "weighted")
0.7386365914786966

julia> negative_predictive_value(x, class_name = 2)
0.8421052631578947

julia> negative_predictive_value(x) == negative_predictive_value(y_true, y_pred)
true

```

See Also :   [`confusion_matrix`](@ref), [`accuracy_score`](@ref), [`positive_predictive_value`](@ref), [`balanced_accuracy_score`](@ref)

"""
function negative_predictive_value(c::confusion_matrix; ith_class = nothing, class_name = nothing , average = "macro", weights = nothing, normalize = false)
    @assert average in ["binary", "macro", "weighted", "micro", "sample-weights"] "Unknown averaging mode. This function only supports the following types: binary, macro, weighted, sample-weights"
    if average == "sample-weights"; @assert weights != nothing && length(weights) == length(c.Labels) """If the average mode is weighted, weights that are the same size as the labels must be provided!
    If no precalculated weights can be provided but the class imbalance is to be taken into account try ' average = "weighted" ' or  ' average = "micro" ' """; end

    index = check_index(c.Labels, true, ith_class = ith_class, class_name = class_name)
    if index == -1
        numerator = c.true_negatives
        denominator = c.true_negatives .+ c.false_negatives
    else
        numerator = c.true_negatives[index]
        denominator = c.true_negatives[index] + c.false_negatives[index]
    end
    if average == "weighted"; weights = c.true_positives .+ c.false_negatives ; end
    return _average_helper(numerator, denominator, weights, average, c.zero_division, normalize)
end

negative_predictive_value(expected, predicted; ith_class = nothing, class_name = nothing, average = "macro", normalize = false)  =
negative_predictive_value(confusion_matrix(expected,predicted), ith_class = ith_class, class_name = class_name, average = average, normalize = normalize)

"""
```false_negative_rate(c::confusion_matrix; keywords) ```
```false_negative_rate(y_true, y_pred; keywords) ```

Return false negative rate for the specified class(es).


## Arguments
- `c::confusion_matrix` : The confusion matrix to report for
- `y_true::Vector` : True values for classification
- `y_pred::Vector` : Predicted values for classification

## Keywords

- `ith_class::Int = nothing` : Return the results for the ith class, the ith label in the label list of the given confusion matrix object.

- `class_name` : Return the results for the class of the specified value in the label list of the given confusion matrix.

If both `class_name` and `ith_class` arguments are equal to `nothing`, return condition positive values for all the elements in the labels arrays

- `average::String = "macro"` :
\n\t   `"binary"` : Return the classwise values.
\n\t   `"macro"` : Return the macro average (mean) of the classwise values.
\n\t    `"micro"` : Return micro average (sum of the numerator divided by sum of the denominator instead of elementwise division) of the classwise values
\n\t    `"weighted"` :
            Return the weighted average (weighted mean with true positives per class) of the classwise values.
\n\t    `"sample-weights"` : Return the weighted average (weighted mean with given weights per class) of the classwise values.

- `weights::Vector  nothing` :  Use the given weights whilst calculating 'sample_weights' option. If average is not 'sample-weigts' this will be ignored.

- `normalize::Bool = false` : If true, normalize the result.

## Examples

```julia-repl
julia> y_pred = [ 1, 4, 4, 1, 2, 3, 1, 3, 2, 1, 2, 2, 2, 1, 4, 4, 2, 1, 3, 2, 2, 3, 2, 1, 2, 3, 4, 1, 2, 1];

julia> y_true = [ 4, 4, 1, 4, 4, 2, 1, 1, 2, 4, 1, 3, 3, 3, 1, 1, 3, 1, 4, 4, 3, 3, 3, 1, 4, 1, 2, 3, 2, 2];

julia> x = confusion_matrix(y_true, y_pred)

julia> false_negative_rate(x)
0.7541666666666667

julia> false_negative_rate(x, average ="binary")
4-element Array{Float64,1}:
 0.6666666666666666
 0.6
 0.875
 0.875

julia> false_negative_rate(x, average ="binary", normalize = true)
 4-element Array{Float64,1}:
  0.43621513219189245
  0.39259361897270323
  0.5725323610018589
  0.5725323610018589

julia> false_negative_rate(x, average ="micro")
 0.7666666666666667

julia> false_negative_rate(x, average ="weighted")
 0.7666666666666667

julia> false_negative_rate(x, ith_class = 2)
 0.6

 julia> false_negative_rate(x) == false_negative_rate(y_true, y_pred)
 true

 ```

 See Also :   [`confusion_matrix`](@ref), [`false_positive_rate`](@ref), [`positive_predictive_value`], [`balanced_accuracy_score`](@ref)

"""
function false_negative_rate(c::confusion_matrix; ith_class = nothing, class_name = nothing, average = "macro", weights = nothing, normalize = false)
    @assert average in ["binary", "macro", "weighted", "micro", "sample-weights"] "Unknown averaging mode. This function only supports the following types: binary, macro, weighted, sample-weights"
    if average == "sample-weights"; @assert weights != nothing && length(weights) == length(c.Labels) """If the average mode is weighted, weights that are the same size as the labels must be provided!
    If no precalculated weights can be provided but the class imbalance is to be taken into account try ' average = "weighted" ' or  ' average = "micro" ' """; end

    index = check_index(c.Labels, true, ith_class = ith_class, class_name = class_name)
    if index == -1
        numerator = c.false_negatives
        denominator = condition_positive(c)
    else
        numerator = c.false_negatives[index]
        denominator = condition_positive(c,ith_class = index)
    end
    if average == "weighted"; weights = c.true_positives .+ c.false_negatives ; end
    return _average_helper(numerator, denominator, weights, average, c.zero_division, normalize)
end

false_negative_rate(expected, predicted; ith_class = nothing, class_name = nothing, average = "macro", normalize = false)  =
false_negative_rate(confusion_matrix(expected,predicted), ith_class = ith_class, class_name = class_name, average = average, normalize = normalize)

"""
```false_positive_rate(c::confusion_matrix; keywords) ```
```false_positive_rate(y_true, y_pred; keywords) ```

Return false positive rate for the specified class(es).

## Arguments
- `c::confusion_matrix` : The confusion matrix to report for
- `y_true::Vector` : True values for classification
- `y_pred::Vector` : Predicted values for classification

## Keywords

- `ith_class::Int = nothing` : Return the results for the ith class, the ith label in the label list of the given confusion matrix object.

- `class_name` : Return the results for the class of the specified value in the label list of the given confusion matrix.

If both `class_name` and `ith_class` arguments are equal to `nothing`, return condition positive values for all the elements in the labels arrays

- `average::String = "macro"` :
\n\t   `"binary"` : Return the classwise values.
\n\t   `"macro"` : Return the macro average (mean) of the classwise values.
\n\t    `"micro"` : Return micro average (sum of the numerator divided by sum of the denominator instead of elementwise division) of the classwise values
\n\t    `"weighted"` :
        Return the weighted average (weighted mean with true positives per class) of the classwise values.
\n\t    `"sample-weights"` : Return the weighted average (weighted mean with given weights per class) of the classwise values.

- `weights::Vector  nothing` :  Use the given weights whilst calculating 'sample_weights' option. If average is not 'sample-weigts' this will be ignored.

- `normalize::Bool = false` : If true, normalize the result.

## Examples

```julia-repl
julia> y_pred = [ 1, 4, 4, 1, 2, 3, 1, 3, 2, 1, 2, 2, 2, 1, 4, 4, 2, 1, 3, 2, 2, 3, 2, 1, 2, 3, 4, 1, 2, 1];

julia> y_true = [ 4, 4, 1, 4, 4, 2, 1, 1, 2, 4, 1, 3, 3, 3, 1, 1, 3, 1, 4, 4, 3, 3, 3, 1, 4, 1, 2, 3, 2, 2];

julia> x = confusion_matrix(y_true, y_pred)

julia> false_positive_rate(x)
0.25233766233766236

julia> false_positive_rate(x, average = "binary")
4-element Array{Float64,1}:
 0.2857142857142857
 0.36
 0.18181818181818182
 0.18181818181818182

julia> false_positive_rate(x, average = "binary", normalize = true)
4-element Array{Float64,1}:
 0.5425242535003962
 0.6835805594104992
 0.34524270677297947
 0.34524270677297947

julia> false_positive_rate(x, average = "micro")
0.25555555555555554

julia> false_positive_rate(x, average = "weighted")
0.24268398268398267

julia> false_positive_rate(x, class_name = 3)
0.18181818181818182

julia> false_positive_rate(x) == false_positive_rate(y_true, y_pred)
true

```

See Also :   [`confusion_matrix`](@ref), [`false_negative_rate`](@ref), [`positive_predictive_value`](@ref), [`balanced_accuracy_score`](@ref)

"""
function false_positive_rate(c::confusion_matrix; ith_class = nothing, class_name = nothing, average = "macro", weights = nothing, normalize = false)
    @assert average in ["binary", "macro", "weighted", "micro", "sample-weights"] "Unknown averaging mode. This function only supports the following types: binary, macro, weighted, sample-weights"
    if average == "sample-weights"; @assert weights != nothing && length(weights) == length(c.Labels) """If the average mode is weighted, weights that are the same size as the labels must be provided!
    If no precalculated weights can be provided but the class imbalance is to be taken into account try ' average = "weighted" ' or  ' average = "micro" ' """; end

    index = check_index(c.Labels, true, ith_class = ith_class, class_name = class_name)
    if index == -1
        numerator = c.false_positives
        denominator =  condition_negative(c)
    else
        numerator = c.false_positives[index] / condition_negative(c,ith_class = index)
        return clear_output(numerator,c.zero_division)
    end

    if average == "weighted"; weights = c.true_positives .+ c.false_negatives ; end
    return _average_helper(numerator, denominator, weights, average, c.zero_division, normalize)
end

false_positive_rate(expected, predicted; ith_class = nothing, class_name = nothing, average = "macro", normalize = false) =
false_positive_rate(confusion_matrix(expected,predicted), ith_class = ith_class, class_name = class_name, average = average, normalize = normalize)

"""
```false_discovery_rate(c::confusion_matrix; keywords) ```
```false_discovery_rate(y_true, y_pred; keywords) ```

Return false discovery rate for the specified class(es).

## Arguments
- `c::confusion_matrix` : The confusion matrix to report for
- `y_true::Vector` : True values for classification
- `y_pred::Vector` : Predicted values for classification

## Keywords

- `ith_class::Int = nothing` : Return the results for the ith class, the ith label in the label list of the given confusion matrix object.

- `class_name` : Return the results for the class of the specified value in the label list of the given confusion matrix.

If both `class_name` and `ith_class` arguments are equal to `nothing`, return condition positive values for all the elements in the labels arrays

- `average::String = "macro"` :
\n\t   `"binary"` : Return the classwise values.
\n\t   `"macro"` : Return the macro average (mean) of the classwise values.
\n\t    `"micro"` : Return micro average (sum of the numerator divided by sum of the denominator instead of elementwise division) of the classwise values
\n\t    `"weighted"` :
        Return the weighted average (weighted mean with true positives per class) of the classwise values.
\n\t    `"sample-weights"` : Return the weighted average (weighted mean with given weights per class) of the classwise values.

- `weights::Vector  nothing` :  Use the given weights whilst calculating 'sample_weights' option. If average is not 'sample-weigts' this will be ignored.

- `normalize::Bool = false` : If true, normalize the result.

## Examples

```julia-repl
julia> y_pred = [ 1, 4, 4, 1, 2, 3, 1, 3, 2, 1, 2, 2, 2, 1, 4, 4, 2, 1, 3, 2, 2, 3, 2, 1, 2, 3, 4, 1, 2, 1];

julia> y_true = [ 4, 4, 1, 4, 4, 2, 1, 1, 2, 4, 1, 3, 3, 3, 1, 1, 3, 1, 4, 4, 3, 3, 3, 1, 4, 1, 2, 3, 2, 2];

julia> x = confusion_matrix(y_true, y_pred)

julia> false_discovery_rate(x)
0.25233766233766236

julia> false_discovery_rate(x, average = "binary")
4-element Array{Float64,1}:
 0.2857142857142857
 0.36
 0.18181818181818182
 0.18181818181818182

 julia> false_discovery_rate(x, average = "binary", normalize = true)
 4-element Array{Float64,1}:
  0.5425242535003962
  0.6835805594104992
  0.34524270677297947
  0.34524270677297947

 julia> false_discovery_rate(x, average = "micro")
 0.25555555555555554

 julia> false_discovery_rate(x, average = "weighted")
 0.24268398268398267

 julia> false_discovery_rate(x, average = "sample-weights", weights = [1,2,3,4])
 0.22784415584415588

 julia> false_discovery_rate(x, ith_class =2)
 0.36

 julia> false_discovery_rate(x) == false_discovery_rate(y_true,y_pred)
true

```
See Also :   [`confusion_matrix`](@ref), [`accuracy_score`](@ref),  [`positive_predictive_value`](@ref), [`false_omission_rate`](@ref)

"""
function false_discovery_rate(c::confusion_matrix; ith_class = nothing, class_name = nothing, average = "macro", weights = nothing, normalize = false)
    @assert average in ["binary", "macro", "weighted", "micro", "sample-weights"] "Unknown averaging mode. This function only supports the following types: binary, macro, weighted, sample-weights"
    if average == "sample-weights"; @assert weights != nothing && length(weights) == length(c.Labels) """If the average mode is weighted, weights that are the same size as the labels must be provided!
    If no precalculated weights can be provided but the class imbalance is to be taken into account try ' average = "weighted" ' or  ' average = "micro" ' """; end

    index = check_index(c.Labels, true, ith_class = ith_class, class_name = class_name)
    if index == -1
        numerator = c.false_positives
        denominator = c.false_positives .+ c.true_negatives
    else
        numerator = c.false_positives[index]
        denominator =  c.false_positives[index]  + c.true_negatives[index]
    end

    if average == "weighted"; weights = c.true_positives .+ c.false_negatives ; end
    return _average_helper(numerator, denominator, weights, average, c.zero_division, normalize)
end

false_discovery_rate(expected, predicted; ith_class = nothing, class_name = nothing, average = "macro", normalize = false)  =
false_discovery_rate(confusion_matrix(expected,predicted), ith_class = ith_class, class_name = class_name, average = average, normalize = normalize)

"""
```false_omission_rate(c::confusion_matrix; keywords)```
```false_omission_rate(y_true, y_pred; keywords)```

Return false omission rate for the specified class(es).

## Arguments
- `c::confusion_matrix` : The confusion matrix to report for
- `y_true::Vector` : True values for classification
- `y_pred::Vector` : Predicted values for classification

## Keywords

- `ith_class::Int = nothing` : Return the results for the ith class, the ith label in the label list of the given confusion matrix object.

- `class_name` : Return the results for the class of the specified value in the label list of the given confusion matrix.

If both `class_name` and `ith_class` arguments are equal to `nothing`, return condition positive values for all the elements in the labels arrays

- `average::String = "macro"` :
\n\t   `"binary"` : Return the classwise values.
\n\t   `"macro"` : Return the macro average (mean) of the classwise values.
\n\t    `"micro"` : Return micro average (sum of the numerator divided by sum of the denominator instead of elementwise division) of the classwise values
\n\t    `"weighted"` :
        Return the weighted average (weighted mean with true positives per class) of the classwise values.
\n\t    `"sample-weights"` : Return the weighted average (weighted mean with given weights per class) of the classwise values.

- `weights::Vector  nothing` :  Use the given weights whilst calculating 'sample_weights' option. If average is not 'sample-weigts' this will be ignored.

- `normalize::Bool = false` : If true, normalize the result.

## Examples

```julia-repl
julia> y_pred = [ 1, 4, 4, 1, 2, 3, 1, 3, 2, 1, 2, 2, 2, 1, 4, 4, 2, 1, 3, 2, 2, 3, 2, 1, 2, 3, 4, 1, 2, 1];

julia> y_true = [ 4, 4, 1, 4, 4, 2, 1, 1, 2, 4, 1, 3, 3, 3, 1, 1, 3, 1, 4, 4, 3, 3, 3, 1, 4, 1, 2, 3, 2, 2];

julia> x = confusion_matrix(y_true, y_pred)

julia> false_omission_rate(x)
0.25090225563909774

julia> false_omission_rate(x, average = "binary")
4-element Array{Float64,1}:
 0.2857142857142857
 0.1578947368421053
 0.28
 0.28

julia> false_omission_rate(x, average = "binary", normalize = true)
4-element Array{Float64,1}:
 0.55674233261854
 0.30767339434182484
 0.5456074859661693
 0.5456074859661693

julia> false_omission_rate(x, average = "micro")
1.003609022556391

julia> false_omission_rate(x, average = "weighted")
0.2613634085213033

julia> false_omission_rate(x, class_name = 4)
0.28

julia> false_omission_rate(x) == false_omission_rate(y_true, y_pred)
true

```

See Also :   [`confusion_matrix`](@ref), [`accuracy_score`](@ref), [`positive_predictive_value`], [`false_discovery_rate`](@ref)
"""
function false_omission_rate(c::confusion_matrix; ith_class = nothing, class_name = nothing, average = "macro", weights = nothing, normalize = false)
    @assert average in ["binary", "macro", "weighted", "micro", "sample-weights"] "Unknown averaging mode. This function only supports the following types: binary, macro, weighted, sample-weights"
    if average == "sample-weights"; @assert weights != nothing && length(weights) == length(c.Labels) """If the average mode is weighted, weights that are the same size as the labels must be provided!
    If no precalculated weights can be provided but the class imbalance is to be taken into account try ' average = "weighted" ' or  ' average = "micro" ' """; end

    index = check_index(c.Labels, true, ith_class = ith_class, class_name = class_name)
    if index == -1
        x = 1 .- negative_predictive_value(c, average = "binary")
    else
        x = 1 - negative_predictive_value(c, ith_class = index)
    end
    if average == "weighted"; weights = c.true_positives .+ c.false_negatives ; end
    return _average_helper(x, nothing, weights, average, c.zero_division, normalize)
end

false_omission_rate(expected, predicted; ith_class = nothing, class_name = nothing, average = "macro", normalize = false) =
false_omission_rate(confusion_matrix(expected,predicted), ith_class = ith_class, class_name = class_name, average = average, normalize = normalize)

"""
```f1_score(c::confusion_matrix; keywords) ```
```f1_score(y_true, y_pred; keywords) ```

Return f1 score for the specified class(es).

## Arguments
- `c::confusion_matrix` : The confusion matrix to report for
- `y_true::Vector` : True values for classification
- `y_pred::Vector` : Predicted values for classification

## Keywords

- `ith_class::Int = nothing` : Return the results for the ith class, the ith label in the label list of the given confusion matrix object.

- `class_name` : Return the results for the class of the specified value in the label list of the given confusion matrix.

If both `class_name` and `ith_class` arguments are equal to `nothing`, return condition positive values for all the elements in the labels arrays

- `average::String = "macro"` :
\n\t   `"binary"` : Return the classwise values.
\n\t   `"macro"` : Return the macro average (mean) of the classwise values.
\n\t    `"micro"` : Return micro average (sum of the numerator divided by sum of the denominator instead of elementwise division) of the classwise values
\n\t    `"weighted"` :
        Return the weighted average (weighted mean with true positives per class) of the classwise values.
\n\t    `"sample-weights"` : Return the weighted average (weighted mean with given weights per class) of the classwise values.

- `weights::Vector  nothing` :  Use the given weights whilst calculating 'sample_weights' option. If average is not 'sample-weigts' this will be ignored.

- `normalize::Bool = false` : If true, normalize the result.

## Examples

```julia-repl
julia> y_pred = [ 1, 4, 4, 1, 2, 3, 1, 3, 2, 1, 2, 2, 2, 1, 4, 4, 2, 1, 3, 2, 2, 3, 2, 1, 2, 3, 4, 1, 2, 1];

julia> y_true = [ 4, 4, 1, 4, 4, 2, 1, 1, 2, 4, 1, 3, 3, 3, 1, 1, 3, 1, 4, 4, 3, 3, 3, 1, 4, 1, 2, 3, 2, 2];

julia> x = confusion_matrix(y_true, y_pred)

julia> f1_score(x)
0.22275641025641024

julia> f1_score(x, average = "binary")
4-element Array{Float64,1}:
 0.3333333333333333
 0.25
 0.15384615384615385
 0.15384615384615385

julia> f1_score(x, average = "binary", normalize =true)
4-element Array{Float64,1}:
 0.7091421918888378
 0.5318566439166283
 0.3272963962563867
 0.3272963962563867

julia> f1_score(x, average = "micro")
0.23333333333333334

julia> f1_score(x, average = "weighted")
0.22371794871794873

julia> f1_score(x, average = "sample-weights", weights = [0.4, 0.2, 0.2, 0.2])
0.2448717948717949

julia> f1_score(x) == f1_score(y_true, y_pred)
true

```

See Also :   [`confusion_matrix`](@ref), [`accuracy_score`](@ref), [`recall_score`](@ref), [`false_omission_rate`](@ref)

"""
function f1_score(c::confusion_matrix; ith_class = nothing, class_name = nothing, average = "macro", weights = nothing, normalize = false)
    @assert average in ["binary", "macro", "weighted", "micro", "sample-weights"] "Unknown averaging mode. This function only supports the following types: binary, macro, weighted, sample-weights"
    if average == "sample-weights"; @assert weights != nothing && length(weights) == length(c.Labels) """If the average mode is weighted, weights that are the same size as the labels must be provided!
    If no precalculated weights can be provided but the class imbalance is to be taken into account try ' average = "weighted" ' or  ' average = "micro" ' """; end

    index = check_index(c.Labels, true, ith_class = ith_class, class_name = class_name)
    if index == -1
        numerator = 2 .* c.true_positives
        denominator = 2 .* c.true_positives .+ c.false_positives .+ c.false_negatives
    else
        numerator = 2 * c.true_positives[index]
        denominator = 2 * c.true_positives[index] + c.false_positives[index] + c.false_negatives[index]
    end

    if average == "weighted"; weights = c.true_positives .+ c.false_negatives ; end
    return _average_helper(numerator, denominator, weights, average, c.zero_division, normalize)
end

f1_score(expected, predicted; ith_class = nothing, class_name = nothing, average = "macro", normalize = false)  =
f1_score(confusion_matrix(expected,predicted), ith_class = ith_class, class_name = class_name, average = average, normalize = normalize)

"""
```prevalence_threshold(c::confusion_matrix; keywords)```
```prevalence_threshold(y_true, y_pred; keywords)``
`
Return prevalence threshold for the specified class(es).

## Arguments
- `c::confusion_matrix` : The confusion matrix to report for
- `y_true::Vector` : True values for classification
- `y_pred::Vector` : Predicted values for classification

## Keywords

- `ith_class::Int = nothing` : Return the results for the ith class, the ith label in the label list of the given confusion matrix object.

- `class_name` : Return the results for the class of the specified value in the label list of the given confusion matrix.

If both `class_name` and `ith_class` arguments are equal to `nothing`, return condition positive values for all the elements in the labels arrays

- `average::String = "macro"` :
\n\t   `"binary"` : Return the classwise values.
\n\t   `"macro"` : Return the macro average (mean) of the classwise values.
\n\t    `"micro"` : Return micro average (sum of the numerator divided by sum of the denominator instead of elementwise division) of the classwise values
\n\t    `"weighted"` :
        Return the weighted average (weighted mean with true positives per class) of the classwise values.
\n\t    `"sample-weights"` : Return the weighted average (weighted mean with given weights per class) of the classwise values.

- `weights::Vector  nothing` :  Use the given weights whilst calculating 'sample_weights' option. If average is not 'sample-weigts' this will be ignored.

- `normalize::Bool = false` : If true, normalize the result.

## Examples

```julia-repl
julia> y_pred = [ 1, 4, 4, 1, 2, 3, 1, 3, 2, 1, 2, 2, 2, 1, 4, 4, 2, 1, 3, 2, 2, 3, 2, 1, 2, 3, 4, 1, 2, 1];

julia> y_true = [ 4, 4, 1, 4, 4, 2, 1, 1, 2, 4, 1, 3, 3, 3, 1, 1, 3, 1, 4, 4, 3, 3, 3, 1, 4, 1, 2, 3, 2, 2];

julia> x = confusion_matrix(y_true, y_pred)

julia> prevalence_threshold(x)
0.515243503586089

julia> prevalence_threshold(x, average = "binary")
4-element Array{Float64,1}:
 0.48074069840785894
 0.4868329805051359
 0.5467001677156806
 0.5467001677156806

julia> prevalence_threshold(x, average = "binary", normalize = true)
4-element Array{Float64,1}:
 0.46564689315863056
 0.471547896007439
 0.529535434443566
 0.529535434443566

julia> prevalence_threshold(x, average = "micro")
0.7594667188323998

julia> prevalence_threshold(x, average = "weighted")
0.5169344623882434

julia> prevalence_threshold(x, average = "sample-weights", weights = [5,4,2,1])
0.49926132643390675

julia> prevalence_threshold(x, ith_class =2)
0.4868329805051359

julia> prevalence_threshold(x) == prevalence_threshold(y_true, y_pred)
true

```

See Also :  [ `confusion_matrix`](@ref),[ `accuracy_score`](@ref), [`recall_score`](@ref), [`f1_score`](@ref)

"""
function prevalence_threshold(c::confusion_matrix; ith_class = nothing, class_name = nothing, average = "macro", weights = nothing, normalize = false)
    @assert average in ["binary", "macro", "weighted", "micro", "sample-weights"] "Unknown averaging mode. This function only supports the following types: binary, macro, weighted, sample-weights"
    if average == "sample-weights"; @assert weights != nothing && length(weights) == length(c.Labels) """If the average mode is weighted, weights that are the same size as the labels must be provided!
    If no precalculated weights can be provided but the class imbalance is to be taken into account try ' average = "weighted" ' or  ' average = "micro" ' """; end
    index = check_index(c.Labels, true, ith_class = ith_class, class_name = class_name)
    if index == -1
        numerator = sqrt.(sensitivity_score(c, average = "binary") .* (-specificity_score(c, average = "binary") .+ 1 )) .+ specificity_score(c, average = "binary") .- 1
        denominator = sensitivity_score(c, average = "binary") .+ specificity_score(c, average = "binary") .- 1
    else
        numerator = sqrt(sensitivity_score(c, ith_class = index) * (-specificity_score(c, ith_class = index) + 1 )) + specificity_score(c,ith_class = index) - 1
        denominator = sensitivity_score(c,ith_class = index) + specificity_score(c,ith_class = index) - 1
    end
    if average == "weighted"; weights = c.true_positives .+ c.false_negatives ; end
    return _average_helper(numerator, denominator, weights, average, c.zero_division, normalize)
end


prevalence_threshold(expected, predicted; ith_class = nothing, class_name = nothing, average = "macro", normalize = false) =
prevalence_threshold(confusion_matrix(expected,predicted), ith_class = ith_class, class_name = class_name, average = average, normalize = normalize)

"""
```threat_score(c::confusion_matrix; keywords)```
```threat_score(y_true, y_pred; keywords)```

Return threat score for the specified class(es).

## Arguments
- `c::confusion_matrix` : The confusion matrix to report for
- `y_true::Vector` : True values for classification
- `y_pred::Vector` : Predicted values for classification

## Keywords

- `ith_class::Int = nothing` : Return the results for the ith class, the ith label in the label list of the given confusion matrix object.

- `class_name` : Return the results for the class of the specified value in the label list of the given confusion matrix.

If both `class_name` and `ith_class` arguments are equal to `nothing`, return condition positive values for all the elements in the labels arrays

- `average::String = "macro"` :
\n\t   `"binary"` : Return the classwise values.
\n\t   `"macro"` : Return the macro average (mean) of the classwise values.
\n\t    `"micro"` : Return micro average (sum of the numerator divided by sum of the denominator instead of elementwise division) of the classwise values
\n\t    `"weighted"` :
        Return the weighted average (weighted mean with true positives per class) of the classwise values.
\n\t    `"sample-weights"` : Return the weighted average (weighted mean with given weights per class) of the classwise values.

- `weights::Vector  nothing` :  Use the given weights whilst calculating 'sample_weights' option. If average is not 'sample-weigts' this will be ignored.

- `normalize::Bool = false` : If true, normalize the result.

## Examples

```julia-repl
julia> y_pred = [ 1, 4, 4, 1, 2, 3, 1, 3, 2, 1, 2, 2, 2, 1, 4, 4, 2, 1, 3, 2, 2, 3, 2, 1, 2, 3, 4, 1, 2, 1];

julia> y_true = [ 4, 4, 1, 4, 4, 2, 1, 1, 2, 4, 1, 3, 3, 3, 1, 1, 3, 1, 4, 4, 3, 3, 3, 1, 4, 1, 2, 3, 2, 2];

julia> x = confusion_matrix(y_true, y_pred)

julia> threat_score(x)
0.12738095238095237

julia> threat_score(x, average = "binary")
4-element Array{Float64,1}:
 0.2
 0.14285714285714285
 0.08333333333333333
 0.08333333333333333

julia> threat_score(x, average = "binary", normalize = true)
4-element Array{Float64,1}:
 0.7337433939929494
 0.5241024242806781
 0.30572641416372887
 0.30572641416372887

julia> threat_score(x, average = "micro")
0.1320754716981132

julia> threat_score(x, average = "weighted")
0.12825396825396823

julia> threat_score(x, average = "sample-weights", weights = [4,32,1,3])
0.1426190476190476

julia> threat_score(x) == threat_score(y_true, y_pred)
true

```
See Also :   [`confusion_matrix`](@ref), [`accuracy_score`](@ref), [`recall_score`](@ref) , [`f1_score`](@ref)

"""
function threat_score(c::confusion_matrix; ith_class = nothing, class_name = nothing, average = "macro", weights = nothing, normalize = false)
    @assert average in ["binary", "macro", "weighted", "micro", "sample-weights"] "Unknown averaging mode. This function only supports the following types: binary, macro, weighted, sample-weights"
    if average == "sample-weights"; @assert weights != nothing && length(weights) == length(c.Labels) """If the average mode is weighted, weights that are the same size as the labels must be provided!
    If no precalculated weights can be provided but the class imbalance is to be taken into account try ' average = "weighted" ' or  ' average = "micro" ' """; end

    index = check_index(c.Labels, true, ith_class = ith_class, class_name = class_name)
    if index == -1
        numerator = c.true_positives
        denominator = c.true_positives .+ c.false_negatives .+ c.false_positives
    else
        numerator = c.true_positives[index]
        denominator = c.true_positives[index] .+ c.false_negatives[index] .+ c.false_positives[index]
    end
    if average == "weighted"; weights = c.true_positives .+ c.false_negatives ; end
    return _average_helper(numerator, denominator, weights, average, c.zero_division, normalize)
end

threat_score(expected, predicted; ith_class = nothing, class_name = nothing, average = "macro", normalize = false) =
threat_score(confusion_matrix(expected,predicted), ith_class = ith_class, class_name = class_name, average = average, normalize = normalize)

"""
```matthews_correlation_coeff(c::confusion_matrix; keywords)```
```matthews_correlation_coeff(c::confusion_matrix; keywords)```

Return Matthew's Correlation Coefficient for the specified class(es).

## Arguments
- `c::confusion_matrix` : The confusion matrix to report for
- `y_true::Vector` : True values for classification
- `y_pred::Vector` : Predicted values for classification

## Keywords

- `ith_class::Int = nothing` : Return the results for the ith class, the ith label in the label list of the given confusion matrix object.

- `class_name` : Return the results for the class of the specified value in the label list of the given confusion matrix.

If both `class_name` and `ith_class` arguments are equal to `nothing`, return condition positive values for all the elements in the labels arrays

- `average::String = "macro"` :
\n\t   `"binary"` : Return the classwise values.
\n\t   `"macro"` : Return the macro average (mean) of the classwise values.
\n\t    `"micro"` : Return micro average (sum of the numerator divided by sum of the denominator instead of elementwise division) of the classwise values
\n\t    `"weighted"` :
        Return the weighted average (weighted mean with true positives per class) of the classwise values.
\n\t    `"sample-weights"` : Return the weighted average (weighted mean with given weights per class) of the classwise values.

- `weights::Vector  nothing` :  Use the given weights whilst calculating 'sample_weights' option. If average is not 'sample-weigts' this will be ignored.

- `normalize::Bool = false` : If true, normalize the result.

## Examples

```julia-repl
julia> y_pred = [ 1, 4, 4, 1, 2, 3, 1, 3, 2, 1, 2, 2, 2, 1, 4, 4, 2, 1, 3, 2, 2, 3, 2, 1, 2, 3, 4, 1, 2, 1];

julia> y_true = [ 4, 4, 1, 4, 4, 2, 1, 1, 2, 4, 1, 3, 3, 3, 1, 1, 3, 1, 4, 4, 3, 3, 3, 1, 4, 1, 2, 3, 2, 2];

julia> x = confusion_matrix(y_true, y_pred)

julia> matthews_correlation_coeff(x)
0.06582375105362859

julia> matthews_correlation_coeff(x, average = "binary")
4-element Array{Float64,1}:
 0.11339930081752816
 0.08717456993952205
 0.03136056672873207
 0.03136056672873207

julia> matthews_correlation_coeff(x, average = "binary", normalize = true)
4-element Array{Float64,1}:
 0.7572453649544855
 0.5821247446210014
 0.20941613948652785
 0.20941613948652785

julia> matthews_correlation_coeff(x, average = "micro")
0.26329500421451435

julia> matthews_correlation_coeff(x, average = "weighted")
0.06527452082383589

julia> matthews_correlation_coeff(x, ith_class =1)
0.11339930081752816

julia> matthews_correlation_coeff(x) == matthews_correlation_coeff(y_true, y_pred)
true

```

See Also :   [`confusion_matrix`](@ref), [`accuracy_score`](@ref), [`threat_score`](@ref), [`f1_score`](@ref)

"""
function matthews_correlation_coeff(c::confusion_matrix; ith_class = nothing, class_name = nothing, average = "macro", weights = nothing, normalize = false)
    @assert average in ["binary", "macro", "weighted", "micro", "sample-weights"] "Unknown averaging mode. This function only supports the following types: binary, macro, weighted, sample-weights"
    if average == "sample-weights"; @assert weights != nothing && length(weights) == length(c.Labels) """If the average mode is weighted, weights that are the same size as the labels must be provided!
    If no precalculated weights can be provided but the class imbalance is to be taken into account try ' average = "weighted" ' or  ' average = "micro" ' """; end

    index = check_index(c.Labels, true, ith_class = ith_class, class_name = class_name)
    if index == -1
        x = sqrt.( precision_score(c, average = "binary") .* sensitivity_score(c, average = "binary") .* specificity_score(c, average = "binary") .* negative_predictive_value(c, average = "binary")) .- sqrt.(false_discovery_rate(c, average = "binary") .* false_negative_rate(c, average = "binary") .* false_positive_rate(c, average = "binary") .*
        false_omission_rate(c, average = "binary"))
    else
        x = sqrt.( precision_score(c, ith_class = index) .* sensitivity_score(c, ith_class = index) .* specificity_score(c, ith_class = index) .* negative_predictive_value(c, ith_class = index)) .- sqrt.(false_discovery_rate(c, ith_class = index) .* false_negative_rate(c, ith_class = index) .* false_positive_rate(c, ith_class = index) .* false_omission_rate(c, ith_class = index))
    end
    if average == "weighted"; weights = c.true_positives .+ c.false_negatives ; end
    return _average_helper(x, nothing, weights, average, c.zero_division, normalize)
end

matthews_correlation_coeff(expected, predicted; ith_class = nothing, class_name = nothing, average = "macro", normalize = false)  =
matthews_correlation_coeff(confusion_matrix(expected,predicted), ith_class = ith_class, class_name = class_name, average = average, normalize = normalize)

"""
```fowlkes_mallows_index(c::confusion_matrix; keywords)```
```fowlkes_mallows_index(y_true, y_pred; keywords)```

Return Fowlkes Mallows Index for the specified class(es).

## Arguments
- `c::confusion_matrix` : The confusion matrix to report for
- `y_true::Vector` : True values for classification
- `y_pred::Vector` : Predicted values for classification

## Keywords

- `ith_class::Int = nothing` : Return the results for the ith class, the ith label in the label list of the given confusion matrix object.

- `class_name` : Return the results for the class of the specified value in the label list of the given confusion matrix.

If both `class_name` and `ith_class` arguments are equal to `nothing`, return condition positive values for all the elements in the labels arrays

- `average::String = "macro"` :
\n\t   `"binary"` : Return the classwise values.
\n\t   `"macro"` : Return the macro average (mean) of the classwise values.
\n\t    `"micro"` : Return micro average (sum of the numerator divided by sum of the denominator instead of elementwise division) of the classwise values
\n\t    `"weighted"` :
        Return the weighted average (weighted mean with true positives per class) of the classwise values.
\n\t    `"sample-weights"` : Return the weighted average (weighted mean with given weights per class) of the classwise values.

- `weights::Vector  nothing` :  Use the given weights whilst calculating 'sample_weights' option. If average is not 'sample-weigts' this will be ignored.

- `normalize::Bool = false` : If true, normalize the result.

## Examples

```julia-repl
julia> y_pred = [ 1, 4, 4, 1, 2, 3, 1, 3, 2, 1, 2, 2, 2, 1, 4, 4, 2, 1, 3, 2, 2, 3, 2, 1, 2, 3, 4, 1, 2, 1];

julia> y_true = [ 4, 4, 1, 4, 4, 2, 1, 1, 2, 4, 1, 3, 3, 3, 1, 1, 3, 1, 4, 4, 3, 3, 3, 1, 4, 1, 2, 3, 2, 2];

julia> x = confusion_matrix(y_true, y_pred)

julia> fowlkes_mallows_index(x)                                                                                                                         julia> fowlkes_mallows_index(x)
0.6798605193558345

julia> fowlkes_mallows_index(x, average = "binary")
4-element Array{Float64,1}:
 0.816496580927726
 0.7627700713964738
 0.570087712549569
 0.570087712549569

julia> fowlkes_mallows_index(x, average = "binary", normalize=true)
4-element Array{Float64,1}:
 0.592585202874323
 0.5535923457160827
 0.4137501009661282
 0.4137501009661282

julia> fowlkes_mallows_index(x, average = "micro")
2.719442077423338

julia> fowlkes_mallows_index(x, average = "weighted")
0.6761240995375003

julia> fowlkes_mallows_index(x, average = "sample-weights", weights = [4,5,3,21])
0.6291497509661493

julia> fowlkes_mallows_index(x, ith_class =3)
0.6798605193558345

julia> fowlkes_mallows_index(x) == fowlkes_mallows_index(y_true, y_pred)
true

```

See Also :   [`confusion_matrix`](@ref), [`matthews_correlation_coeff`](@ref), [`threat_score`](@ref), [`f1_score`](@ref)
"""
function fowlkes_mallows_index(c::confusion_matrix; ith_class = nothing, class_name = nothing, average = "macro", weights = nothing, normalize = false)
    @assert average in ["binary", "macro", "weighted", "micro", "sample-weights"] "Unknown averaging mode. This function only supports the following types: binary, macro, weighted, sample-weights"
    if average == "sample-weights"; @assert weights != nothing && length(weights) == length(c.Labels) """If the average mode is weighted, weights that are the same size as the labels must be provided!
    If no precalculated weights can be provided but the class imbalance is to be taken into account try ' average = "weighted" ' or  ' average = "micro" ' """; end

    index = check_index(c.Labels, true, ith_class = ith_class, class_name = class_name)
    if index == -1
        x = sqrt.(precision_score(c, average = "binary") .+ sensitivity_score(c, average = "binary"))
    else
        x = sqrt.(precision_score(c, average = "binary") .+ sensitivity_score(c, average = "binary"))
    end
    if average == "weighted"; weights = c.true_positives .+ c.false_negatives ; end
    return _average_helper(x, nothing, weights, average, c.zero_division, normalize )
end


fowlkes_mallows_index(expected, predicted; ith_class = nothing, class_name = nothing, average = "macro", normalize = false)  =
fowlkes_mallows_index(confusion_matrix(expected,predicted), ith_class = ith_class, class_name = class_name, average = average, normalize = normalize)

"""
```informedness(c::confusion_matrix; keywords)```
```informedness(y_true, y_pred; keywords)```

Return informedness value for the specified class(es).

## Arguments
- `c::confusion_matrix` : The confusion matrix to report for
- `y_true::Vector` : True values for classification
- `y_pred::Vector` : Predicted values for classification

## Keywords

- `ith_class::Int = nothing` : Return the results for the ith class, the ith label in the label list of the given confusion matrix object.

- `class_name` : Return the results for the class of the specified value in the label list of the given confusion matrix.

If both `class_name` and `ith_class` arguments are equal to `nothing`, return condition positive values for all the elements in the labels arrays

- `average::String = "macro"` :
\n\t   `"binary"` : Return the classwise values.
\n\t   `"macro"` : Return the macro average (mean) of the classwise values.
\n\t    `"micro"` : Return micro average (sum of the numerator divided by sum of the denominator instead of elementwise division) of the classwise values
\n\t    `"weighted"` :
        Return the weighted average (weighted mean with true positives per class) of the classwise values.
\n\t    `"sample-weights"` : Return the weighted average (weighted mean with given weights per class) of the classwise values.

- `weights::Vector  nothing` :  Use the given weights whilst calculating 'sample_weights' option. If average is not 'sample-weigts' this will be ignored.

- `normalize::Bool = false` : If true, normalize the result.

## Examples

```julia-repl
julia> y_pred = [ 1, 4, 4, 1, 2, 3, 1, 3, 2, 1, 2, 2, 2, 1, 4, 4, 2, 1, 3, 2, 2, 3, 2, 1, 2, 3, 4, 1, 2, 1];

julia> y_true = [ 4, 4, 1, 4, 4, 2, 1, 1, 2, 4, 1, 3, 3, 3, 1, 1, 3, 1, 4, 4, 3, 3, 3, 1, 4, 1, 2, 3, 2, 2];

julia> x = confusion_matrix(y_true, y_pred)

julia> informedness(x)
-0.006504329004328957

julia> informedness(x, average = "binary")
4-element Array{Float64,1}:
  0.04761904761904767
  0.040000000000000036
 -0.05681818181818177
 -0.05681818181818177

julia> informedness(x, average = "binary", normalize = true)
4-element Array{Float64,1}:
  0.468654520667708
  0.39366979736087465
 -0.5591900530694233
 -0.5591900530694233

julia> informedness(x, average = "micro")
-0.026017316017315828

julia> informedness(x, average = "weighted")
-0.009350649350649302

julia> informedness(x, ith_class =2 )
0.040000000000000036

julia> informedness(x) == informedness(y_true, y_pred)
true

```

See Also :   [`confusion_matrix`](@ref), [`matthews_correlation_coeff`](@ref), [`markedness`](@ref), [`f1_score`](@ref)
"""
function informedness(c::confusion_matrix; ith_class = nothing, class_name = nothing, average = "macro", weights = nothing, normalize = false)
    @assert average in ["binary", "macro", "weighted", "micro", "sample-weights"] "Unknown averaging mode. This function only supports the following types: binary, macro, weighted, sample-weights"
    if average == "sample-weights"; @assert weights != nothing && length(weights) == length(c.Labels) """If the average mode is weighted, weights that are the same size as the labels must be provided!
    If no precalculated weights can be provided but the class imbalance is to be taken into account try ' average = "weighted" ' or  ' average = "micro" ' """; end

    index = check_index(c.Labels, true, ith_class = ith_class, class_name = class_name)

    if index == -1
        x = specificity_score(c, average = "binary") .+ sensitivity_score(c, average = "binary") .- 1
    else
        x = specificity_score(c, ith_class = index) .+ sensitivity_score(c, ith_class = index) -1
    end
    if average == "weighted"; weights = c.true_positives .+ c.false_negatives ; end
    return _average_helper(x, nothing, weights, average, c.zero_division, normalize)
end

informedness(expected, predicted; ith_class = nothing, class_name = nothing, average = "macro", normalize = false)  =
informedness(confusion_matrix(expected,predicted), ith_class = ith_class, class_name = class_name, average = average, normalize = normalize)

"""
```markedness(c::confusion_matrix; keywords)```
```markedness(y_true, y_pred; keywords)```

Return markedness value for the specified class(es).

## Arguments
- `c::confusion_matrix` : The confusion matrix to report for
- `y_true::Vector` : True values for classification
- `y_pred::Vector` : Predicted values for classification

## Keywords

- `ith_class::Int = nothing` : Return the results for the ith class, the ith label in the label list of the given confusion matrix object.

- `class_name` : Return the results for the class of the specified value in the label list of the given confusion matrix.

If both `class_name` and `ith_class` arguments are equal to `nothing`, return condition positive values for all the elements in the labels arrays

- `average::String = "macro"` :
\n\t   `"binary"` : Return the classwise values.
\n\t   `"macro"` : Return the macro average (mean) of the classwise values.
\n\t    `"micro"` : Return micro average (sum of the numerator divided by sum of the denominator instead of elementwise division) of the classwise values
\n\t    `"weighted"` :
        Return the weighted average (weighted mean with true positives per class) of the classwise values.
\n\t    `"sample-weights"` : Return the weighted average (weighted mean with given weights per class) of the classwise values.

- `weights::Vector  nothing` :  Use the given weights whilst calculating 'sample_weights' option. If average is not 'sample-weigts' this will be ignored.

- `normalize::Bool = false` : If true, normalize the result.

## Examples

```julia-repl
julia> y_pred = [ 1, 4, 4, 1, 2, 3, 1, 3, 2, 1, 2, 2, 2, 1, 4, 4, 2, 1, 3, 2, 2, 3, 2, 1, 2, 3, 4, 1, 2, 1];

julia> y_true = [ 4, 4, 1, 4, 4, 2, 1, 1, 2, 4, 1, 3, 3, 3, 1, 1, 3, 1, 4, 4, 3, 3, 3, 1, 4, 1, 2, 3, 2, 2];

julia> x = confusion_matrix(y_true, y_pred)

julia> markedness(x)
-0.022114376851218975

julia> markedness(x, average = "binary")
4-element Array{Float64,1}:
  0.04761904761904767
  0.02392344497607657
 -0.08000000000000007
 -0.08000000000000007

julia> markedness(x, average = "binary", normalize = true)
4-element Array{Float64,1}:
  0.38077081282284964
  0.19129634137033105
 -0.6396949655423873
 -0.6396949655423873

julia> markedness(x, average = "micro")
-0.0884575074048759

julia> markedness(x, average = "weighted")
-0.024393711551606308

julia> markedness(x, ith_class=1)
0.04761904761904767

julia> markedness(x) == markedness(y_true, y_pred)
true

```

See Also :   [`confusion_matrix`](@ref), [`matthews_correlation_coeff`](@ref), [`informedness`](@ref), [`f1_score`](@ref)

"""
function markedness(c::confusion_matrix; ith_class = nothing, class_name = nothing, average = "macro", weights = nothing, normalize = false)
    @assert average in ["binary", "macro", "weighted", "micro", "sample-weights"] "Unknown averaging mode. This function only supports the following types: binary, macro, weighted, sample-weights"
    if average == "sample-weights"; @assert weights != nothing && length(weights) == length(c.Labels) """If the average mode is weighted, weights that are the same size as the labels must be provided!
    If no precalculated weights can be provided but the class imbalance is to be taken into account try ' average = "weighted" ' or  ' average = "micro" ' """; end

    index = check_index(c.Labels, true, ith_class = ith_class, class_name = class_name)
    if index == -1
        x = precision_score(c, average = "binary") .+ negative_predictive_value(c, average = "binary") .- 1
    else
        x = precision_score(c, ith_class = index) .+ negative_predictive_value(c, ith_class = index) - 1
    end
    if average == "weighted"; weights = c.true_positives .+ c.false_negatives ; end
    return _average_helper(x, nothing, weights, average, c.zero_division, normalize)
end

markedness(expected, predicted; ith_class = nothing, class_name = nothing, average = "macro", normalize = false) =
markedness(confusion_matrix(expected,predicted), ith_class = ith_class, class_name = class_name, average = average, normalize = normalize)

"""
```cohen_kappa_score(c::confusion_matrix; weights = nothing) ```
```cohen_kappa_score(y_true, y_pred; weights = nothing) ```

Return Cohen's Kappa (a statistic that measures inter-annotator agreement)

## Arguments
- `c::confusion_matrix` : The confusion matrix to report for
- `y_true::Vector` : True values for classification
- `y_pred::Vector` : Predicted values for classification

## Keywords
- `weights::String = nothing` :
\n\t   nothing : not weighted
\n\t   `"linear"` : linear weighted
\n\t   `"quadratic"` : quadratic weighted

## Examples

```julia-repl
julia> y_true = [3, 1, 1, 2, 1, 2, 1, 5, 1, 5];

julia> y_pred = [3, 1, 4, 1, 4, 2, 3, 4, 3, 1];

julia> x = confusion_matrix(y_true, y_pred, labels= [1,2,3,4,5]);

julia> cohen_kappa_score(x)
0.12499999999999989

julia> cohen_kappa_score(x, weights = "linear")
0.012345679012345623

julia> cohen_kappa_score(x, weights = "quadratic")
-0.11111111111111116
```

See Also :  [`confusion_matrix`](@ref), [`accuracy_score`](@ref), [`jaccard_score`](@ref), [`f1_score`](@ref)

"""
function cohen_kappa_score(c::confusion_matrix; weights = nothing)
#reference: scikitlearn.metrics.classification.cohen_kappa_score
    @assert weights in [nothing, "quadratic", "linear"] "Unknown kappa weighting type"
    sum0 = sum(c.matrix, dims = 1)
    sum1 = sum(c.matrix, dims = 2)
    expected = zeros(length(c.Labels), length(c.Labels))
    expected .= kron(sum1, sum0)' ./ sum(sum0)

    if weights == nothing
        w_mat = ones(length(c.Labels),length(c.Labels))
        for i in 1:length(c.Labels)
            w_mat[i,i] = 0
        end
    else
        w_mat = zeros(length(c.Labels),length(c.Labels))
        for i in 1:length(c.Labels); w_mat[:,i] = w_mat[:,i] .+ i; end
        if weights == "linear"
            w_mat = abs.(w_mat - transpose(w_mat))
        else
            w_mat = (w_mat - transpose(w_mat)) .^ 2
        end
    end
    x = sum(w_mat .* c.matrix) ./ sum(w_mat .* expected)
    return clear_output(1 .- x,c.zero_division)
end

cohen_kappa_score(expected, predicted; weights = nothing)  =
cohen_kappa_score(confusion_matrix(expected,predicted), weights = weights)

"""
```hamming_loss(c::confusion_matrix) ```
```hamming_loss(y_true, y_pred) ```

Compute the average Hamming loss.
    The Hamming loss is the fraction of labels that are incorrectly predicted.

## Examples

```julia-repl
julia> y_pred = [ 1, 4, 4, 1, 2, 3, 1, 3, 2, 1, 2, 2, 2, 1, 4, 4, 2, 1, 3, 2, 2, 3, 2, 1, 2, 3, 4, 1, 2, 1];

julia> y_true = [ 4, 4, 1, 4, 4, 2, 1, 1, 2, 4, 1, 3, 3, 3, 1, 1, 3, 1, 4, 4, 3, 3, 3, 1, 4, 1, 2, 3, 2, 2];

julia> x = confusion_matrix(y_true, y_pred)

julia> hamming_loss(x)
0.7666666666666667
```

See Also :   [`confusion_matrix`](@ref), [`accuracy_score`](@ref), [`jaccard_score`](@ref), [`f1_score`](@ref)

"""
function hamming_loss(c::confusion_matrix;)
    x = zeros(sum(c.matrix))
    x[1] = sum(c.false_negatives)
    return clear_output(mean(x), c.zero_division)
end

hamming_loss(expected::Array{T,1}, predicted::Array{T,1};) where T <: Union{Int, String}  =
hamming_loss(confusion_matrix(expected,predicted))

"""
```jaccard_score(c::confusion_matrix; keywords) ```
```jaccard_score(y_true, y_pred; keywords) ```

## Arguments
- `c::confusion_matrix` : The confusion matrix to report for
- `y_true::Vector` : True values for classification
- `y_pred::Vector` : Predicted values for classification

## Keywords

- `ith_class::Int = nothing` : Return the results for the ith class, the ith label in the label list of the given confusion matrix object.

- `class_name` : Return the results for the class of the specified value in the label list of the given confusion matrix.

If both `class_name` and `ith_class` arguments are equal to `nothing`, return condition positive values for all the elements in the labels arrays

- `average::String = "binary"` :
\n\t   `"binary"` : Return the classwise values.
\n\t   `"macro"` : Return the macro average (mean) of the classwise values.
\n\t    `"micro"` : Return micro average (sum of the numerator divided by sum of the denominator instead of elementwise division) of the classwise values
\n\t    `"weighted"` :
        Return the weighted average (weighted mean with true positives per class) of the classwise values.
\n\t    `"sample-weights"` : Return the weighted average (weighted mean with given weights per class) of the classwise values.

- `weights::Vector  nothing` :  Use the given weights whilst calculating 'sample_weights' option. If average is not 'sample-weigts' this will be ignored.

- `normalize::Bool = false` : If true, normalize the result.

## Examples

```julia-repl
julia> y_pred = [ 1, 4, 4, 1, 2, 3, 1, 3, 2, 1, 2, 2, 2, 1, 4, 4, 2, 1, 3, 2, 2, 3, 2, 1, 2, 3, 4, 1, 2, 1];

julia> y_true = [ 4, 4, 1, 4, 4, 2, 1, 1, 2, 4, 1, 3, 3, 3, 1, 1, 3, 1, 4, 4, 3, 3, 3, 1, 4, 1, 2, 3, 2, 2];

julia> x = confusion_matrix(y_true, y_pred)

julia> jaccard_score(x)
0.12738095238095237

julia> jaccard_score(x, average = "binary")
4-element Array{Float64,1}:
 0.2
 0.14285714285714285
 0.08333333333333333
 0.08333333333333333

julia> jaccard_score(x, average = "binary", normalize =true)
4-element Array{Float64,1}:
 0.7337433939929494
 0.5241024242806781
 0.30572641416372887
 0.30572641416372887

julia> jaccard_score(x, average = "micro")
0.1320754716981132

julia> jaccard_score(x, average = "weighted")
0.12825396825396823

julia> jaccard_score(x, ith_class = 2)
0.14285714285714285

julia> jaccard_score(y_true, y_pred) == jaccard_score(x)
true

```

See Also :   [`confusion_matrix`](@ref), [`accuracy_score`](@ref), [`hamming_loss`](@ref), [`f1_score`](@ref)

"""
function jaccard_score(c::confusion_matrix;  ith_class = nothing, class_name = nothing, average = "macro", weights = nothing, normalize = false)
    @assert average in ["binary", "macro", "weighted", "micro", "sample-weights"] "Unknown averaging mode. This function only supports the following types: binary, macro, weighted, sample-weights"
    if average == "sample-weights"; @assert weights != nothing && length(weights) == length(c.Labels) """If the average mode is weighted, weights that are the same size as the labels must be provided!
    If no precalculated weights can be provided but the class imbalance is to be taken into account try ' average = "weighted" ' or  ' average = "micro" ' """; end
    index = check_index(c.Labels, true, ith_class = ith_class, class_name = class_name)
    if index == -1
        numerator = c.true_positives
        denominator =  c.true_positives .+ c.false_negatives .+ c.false_positives
    else
        numerator = c.true_positives[index]
        denominator =  c.true_positives[index] + c.false_negatives[index] + c.false_positives[index]
    end
    if average == "weighted"; weights = c.true_positives .+ c.false_negatives ; end
    return _average_helper(numerator, denominator, weights, average, c.zero_division, normalize)
end

jaccard_score(expected, predicted; ith_class = nothing, class_name = nothing, average = "macro", weights = nothing, normalize = false) =
jaccard_score(confusion_matrix(expected,predicted), ith_class = ith_class, class_name = class_name, average = average, weights = weights, normalize = normalize)
