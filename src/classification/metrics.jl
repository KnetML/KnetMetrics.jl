export classification_report, condition_positive, condition_negative, predicted_positive, predicted_negative, correctly_classified, incorrectly_classified, sensitivity_score, true_positive_rate, recall_score, specificity_score, true_negative_rate, precision_score, positive_predictive_value, accuracy_score, balanced_accuracy_score, negative_predictive_value, false_negative_rate, false_positive_rate, false_discovery_rate, false_omission_rate, fbeta_score, f1_score, prevalence_threshold, threat_score, matthews_correlation_coeff, fowlkes_mallows_index, informedness, markedness, cohen_kappa_score, hamming_loss, jaccard_score
using Statistics: mean

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

julia> y_true = [2, 3, 2, 2, 2, 1, 3, 3, 2, 1, 2, 3, 2, 3, 2];

julia> y_pred = [1, 3, 1, 1, 3, 2, 2, 1, 3, 2, 3, 3, 2, 3, 2];

julia> c = confusion_matrix(y_true, y_pred);

julia> classification_report(c)
Summary:
confusion_matrix
True Positives: [0, 2, 3]
False Positives: [4, 3, 3]
True Negatives: [9, 4, 7]
False Negatives: [2, 6, 2]

    Labelwise Statistics

                                          1       2       3
                Condition Positive:     2.0     8.0     5.0
                Condition Negative:    13.0     7.0    10.0
                Predicted Positive:     4.0     5.0     6.0
                Predicted Negative:    11.0    10.0     9.0
              Correctly Classified:     9.0     6.0    10.0
            Incorrectly Classified:     6.0     9.0     5.0
                       Sensitivity:     0.0    0.25     0.6
                       Specificity:    0.69    0.57     0.7
                         Precision:     0.0     0.4     0.5
                    Accuracy Score:     0.0     0.4     0.5
                 Balanced Accuracy:     0.0    0.25     0.6
         Negative Predictive Value:    0.82     0.4    0.78
               False Negative Rate:     1.0    0.75     0.4
               False Positive Rate:    0.31    0.43     0.3
              False Discovery Rate:    0.31    0.43     0.3
               False Omission Rate:    0.18     0.6    0.22
                          F1 Score:     0.0    0.31    0.55
                     Jaccard Score:     0.0    0.18    0.38
              Prevalence Threshold:     1.0    0.57    0.41
                      Threat Score:     0.0    0.18    0.38
  Matthews Correlation Coefficient:    0.02
             Fowlkes Mallows Index:     0.0    0.81    1.05
                      Informedness:   -0.31   -0.18     0.3
                        Markedness:   -0.18    -0.2    0.28

      General Statistics

              Accuracy Score:   0.3
           Cohen Kappa Score:   -0.020408163265306145
                Hamming Loss:   0.6666666666666666
               Jaccard Score:   0.18560606060606064
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
julia> y_true = [2, 3, 2, 2, 2, 1, 3, 3, 2, 1, 2, 3, 2, 3, 2];

julia> y_pred = [1, 3, 1, 1, 3, 2, 2, 1, 3, 2, 3, 3, 2, 3, 2];

julia> c = confusion_matrix(y_true, y_pred);

julia> condition_positive(c)
3-element Array{Int64,1}:
 2
 8
 5

julia> condition_positive(c, average = "macro")
5.0

julia> condition_positive(c, normalize=True)
ERROR: UndefVarError: True not defined
Stacktrace:
 [1] top-level scope at none:1

julia> condition_positive(c, normalize=true)
3-element Array{Float64,1}:
 0.20739033894608505
 0.8295613557843402
 0.5184758473652127

julia> condition_positive(c, class_name=2)
8

```
See also : [`confusion_matrix`](@ref), [`condition_negative`](@ref), [`predicted_positive`](@ref), [`predicted_negative`](@ref)

"""
function condition_positive(c::confusion_matrix; ith_class = nothing, class_name = nothing, average = "binary", weights = nothing, normalize = false)
    index = check_index(c.Labels, true, ith_class = ith_class, class_name = class_name, valid_modes = ["binary", "macro", "weighted", "micro", "sample-weights"], average = average,weights=weights)
    if index == -1
        x = c.true_positives .+ c.false_negatives
    else
        x = c.true_positives[index] .+ c.false_negatives[index]
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
julia> y_true = [2, 3, 2, 2, 2, 1, 3, 3, 2, 1, 2, 3, 2, 3, 2];

julia> y_pred = [1, 3, 1, 1, 3, 2, 2, 1, 3, 2, 3, 3, 2, 3, 2];

julia> c = confusion_matrix(y_true, y_pred);

julia> condition_positive(c, normalize=true)
3-element Array{Float64,1}:
 0.20739033894608505
 0.8295613557843402
 0.5184758473652127

julia> condition_positive(c, class_name=2)
8

julia> condition_negative(c)
3-element Array{Int64,1}:
 13
  7
 10

julia> condition_negative(c, average = "macro")
10.0

julia> condition_negative(c, normalize=true)
3-element Array{Float64,1}:
 0.7290038003196576
 0.39254050786443107
 0.5607721540920443

julia> condition_negative(c, ith_class=2)
7

```

See also : [`confusion_matrix`](@ref), [`condition_positive`](@ref), [`predicted_positive`](@ref), [`predicted_negative`](@ref)

"""
function condition_negative(c::confusion_matrix; ith_class = nothing, class_name = nothing, average = "binary", weights = nothing, normalize = false)
    index = check_index(c.Labels, true, ith_class = ith_class, class_name = class_name, valid_modes = ["binary", "macro", "weighted", "micro", "sample-weights"], average = average,weights=weights)
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
julia> y_true = [2, 3, 2, 2, 2, 1, 3, 3, 2, 1, 2, 3, 2, 3, 2];

julia> y_pred = [1, 3, 1, 1, 3, 2, 2, 1, 3, 2, 3, 3, 2, 3, 2];

julia> c = confusion_matrix(y_true, y_pred);

julia> predicted_positive(c)
3-element Array{Int64,1}:
 4
 5
 6

julia> predicted_positive(c, average= "micro")
15

julia> predicted_positive(c, normalize=true)
3-element Array{Float64,1}:
 0.4558423058385518
 0.5698028822981898
 0.6837634587578276

julia> predicted_positive(c, class_name=1)
4

```

See also : [`confusion_matrix`](@ref), [`condition_negative`](@ref), [`predicted_positive`](@ref), [`predicted_negative`](@ref)

"""
function predicted_positive(c::confusion_matrix; ith_class = nothing, class_name = nothing, average = "binary", weights = nothing, normalize = false)
    index = check_index(c.Labels, true, ith_class = ith_class, class_name = class_name, valid_modes = ["binary", "macro", "weighted", "micro", "sample-weights"], average = average,weights=weights)
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

julia> y_true = [2, 3, 2, 2, 2, 1, 3, 3, 2, 1, 2, 3, 2, 3, 2];

julia> y_pred = [1, 3, 1, 1, 3, 2, 2, 1, 3, 2, 3, 3, 2, 3, 2];

julia> c = confusion_matrix(y_true, y_pred);

julia> predicted_negative(c)
3-element Array{Int64,1}:
 11
 10
  9

julia> predicted_negative(c, average = "weighted")
9.8

julia> predicted_negative(c, normalize=true)
3-element Array{Float64,1}:
 0.6329788714132796
 0.575435337648436
 0.5178918038835923

julia> predicted_negative(c, ith_class=3)
9

```

See also : [`confusion_matrix`](@ref), [`condition_negative`](@ref), [`predicted_positive`](@ref), [`condition_positive`](@ref)

"""
function predicted_negative(c::confusion_matrix; ith_class = nothing, class_name = nothing, average = "binary", weights = nothing, normalize = false)
    index = check_index(c.Labels, true, ith_class = ith_class, class_name = class_name, valid_modes = ["binary", "macro", "weighted", "micro", "sample-weights"], average = average,weights=weights)
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

julia> y_true = [2, 3, 2, 2, 2, 1, 3, 3, 2, 1, 2, 3, 2, 3, 2];

julia> y_pred = [1, 3, 1, 1, 3, 2, 2, 1, 3, 2, 3, 3, 2, 3, 2];

julia> c = confusion_matrix(y_true, y_pred);

julia> correctly_classified(c)
3-element Array{Int64,1}:
  9
  6
 10

julia> correctly_classified(c, average="macro")
8.333333333333334

julia> correctly_classified(c, normalize=true)
3-element Array{Float64,1}:
 0.6109598099719176
 0.4073065399812784
 0.6788442333021306

julia> correctly_classified(c, ith_class=2)
6

```

See also : [`confusion_matrix]`(@ref), [`predicted_negative`](@ref), [`predicted_positive`](@ref), [`incorrectly_classified`](@ref)

"""
function correctly_classified(c::confusion_matrix; ith_class = nothing, class_name = nothing, average = "binary", weights = nothing, normalize = false)
    index = check_index(c.Labels, true, ith_class = ith_class, class_name = class_name, valid_modes = ["binary", "macro", "weighted", "micro", "sample-weights"], average = average,weights=weights)
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
julia> y_true = [2, 3, 2, 2, 2, 1, 3, 3, 2, 1, 2, 3, 2, 3, 2];

julia> y_pred = [1, 3, 1, 1, 3, 2, 2, 1, 3, 2, 3, 3, 2, 3, 2];

julia> c = confusion_matrix(y_true, y_pred);

julia> incorrectly_classified(c)
3-element Array{Int64,1}:
 6
 9
 5

julia> incorrectly_classified(c, average="macro")
6.666666666666667

julia> incorrectly_classified(c, average="micro")
20

julia> incorrectly_classified(c, normalize=true)
3-element Array{Float64,1}:
 0.5035088149780135
 0.7552632224670202
 0.4195906791483445

julia> incorrectly_classified(c, average = "sample-weights", weights = [1,2,3])
 6.5

julia> incorrectly_classified(c, ith_class=2)
 9

```

See also : [`confusion_matrix`](@ref), [`predicted_negative`](@ref), [`predicted_positive`](@ref), [`correctly_classified`](@ref)

"""
function incorrectly_classified(c::confusion_matrix; ith_class = nothing, class_name = nothing, average = "binary", weights = nothing, normalize = false)
    index = check_index(c.Labels, true, ith_class = ith_class, class_name = class_name, valid_modes = ["binary", "macro", "weighted", "micro", "sample-weights"], average = average,weights=weights)
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
julia> y_true = [2, 3, 2, 2, 2, 1, 3, 3, 2, 1, 2, 3, 2, 3, 2];

julia> y_pred = [1, 3, 1, 1, 3, 2, 2, 1, 3, 2, 3, 3, 2, 3, 2];

julia> c = confusion_matrix(y_true, y_pred);

julia> sensitivity_score(c)
0.2833333333333333

julia> sensitivity_score(c, average="macro")
0.2833333333333333

julia> sensitivity_score(c, average="binary", normalize=true)
3-element Array{Float64,1}:
 0.0
 0.3846153846153846
 0.9230769230769229

julia> sensitivity_score(c, class_name=3)
0.6

```

See also : [`confusion_matrix`](@ref) , [`recall_score`](@ref) ,  [`balanced_accuracy_score`](@ref), [`specificity_score`](@ref)
"""
true_positive_rate, sensitivity_score

function sensitivity_score(c::confusion_matrix; ith_class = nothing, class_name = nothing , average = "macro", weights = nothing, normalize = false)
    index = check_index(c.Labels, true, ith_class = ith_class, class_name = class_name, valid_modes = ["binary", "macro", "weighted", "micro", "sample-weights"], average = average,weights=weights)
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

true_positive_rate(c::confusion_matrix; ith_class = nothing, class_name = nothing , average = "macro", weights = nothing, normalize = false) = sensitivity_score(c; ith_class = ith_class, class_name = class_name , average = average, weights = weights, normalize = normalize)
true_positive_rate(expected, predicted; ith_class = nothing, class_name = nothing , average = "macro", weights = nothing, normalize = false) = sensitivity_score(confusion_matrix(expected,predicted); ith_class = ith_class, class_name = class_name , average = average, weights = weights, normalize = normalize)

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
julia> y_true = [2, 3, 2, 2, 2, 1, 3, 3, 2, 1, 2, 3, 2, 3, 2];

julia> y_pred = [1, 3, 1, 1, 3, 2, 2, 1, 3, 2, 3, 3, 2, 3, 2];

julia> c = confusion_matrix(y_true, y_pred);

julia> recall_score(c)
0.2833333333333333

julia> recall_score(c, average="binary")
3-element Array{Float64,1}:
 0.0
 0.25
 0.6

julia> recall_score(c, average="binary", normalize=true)
3-element Array{Float64,1}:
 0.0
 0.3846153846153846
 0.9230769230769229

julia> recall_score(c, ith_class=2)
0.25

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
julia> y_true = [2, 3, 2, 2, 2, 1, 3, 3, 2, 1, 2, 3, 2, 3, 2];

julia> y_pred = [1, 3, 1, 1, 3, 2, 2, 1, 3, 2, 3, 3, 2, 3, 2];

julia> c = confusion_matrix(y_true, y_pred);

julia> specificity_score(c)
0.6545787545787546

julia> specificity_score(c, average="binary")
3-element Array{Float64,1}:
 0.6923076923076923
 0.5714285714285714
 0.7

julia> specificity_score(c, average="binary", normalize=true)
3-element Array{Float64,1}:
 0.6081724251468739
 0.5019835890101181
 0.6149298965373947

julia> specificity_score(c, average="binary", normalize=true, ith_class=2)
0.5714285714285714

julia> specificity_score(c, average="weighted")
0.6304029304029304

julia> specificity_score(c, class_name=2)
0.5714285714285714

```

See also : [`confusion_matrix`](@ref), [`sensitivity_score`](@ref), [`balanced_accuracy_score`](@ref),[`recall_score`](@ref)

"""
true_negative_rate, specificity_score

function specificity_score(c::confusion_matrix; ith_class = nothing, class_name = nothing, average = "macro", weights = nothing, normalize = false)
    index = check_index(c.Labels, true, ith_class = ith_class, class_name = class_name, valid_modes = ["binary", "macro", "weighted", "micro", "sample-weights"], average = average,weights=weights)
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

true_negative_rate(expected, predicted; ith_class = nothing, class_name = nothing, average = "macro", normalize = false) =
specificity_score(confusion_matrix(expected,predicted), ith_class = ith_class, class_name = class_name, average = average, normalize = normalize)

true_negative_rate(c::confusion_matrix; ith_class = nothing, class_name = nothing, average = "macro", weights = nothing, normalize = false) =
specificity_score(c, ith_class = ith_class, class_name = class_name, average = average, normalize = normalize)

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
julia> y_true = [2, 3, 2, 2, 2, 1, 3, 3, 2, 1, 2, 3, 2, 3, 2];

julia> y_pred = [1, 3, 1, 1, 3, 2, 2, 1, 3, 2, 3, 3, 2, 3, 2];

julia> c = confusion_matrix(y_true, y_pred);

julia> precision_score(c)
0.3

julia> precision_score(c,average="binary",normalize=true)
3-element Array{Float64,1}:
 0.0
 0.6246950475544243
 0.7808688094430303

julia> precision_score(c,average="weighted")
0.38

julia> precision_score(c, class_name=1)
0.0

```

See Also : [`confusion_matrix`](@ref), [`sensitivity_score`](@ref) , [`balanced_accuracy_score`](@ref), [`recall_score`](@ref)

"""
function precision_score(c::confusion_matrix; ith_class = nothing, class_name = nothing, average = "macro", weights = nothing, normalize = false)
    index = check_index(c.Labels, true, ith_class = ith_class, class_name = class_name, valid_modes = ["binary", "macro", "weighted", "micro", "sample-weights"], average = average,weights=weights)
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
julia> y_true = [2, 3, 2, 2, 2, 1, 3, 3, 2, 1, 2, 3, 2, 3, 2];

julia> y_pred = [1, 3, 1, 1, 3, 2, 2, 1, 3, 2, 3, 3, 2, 3, 2];

julia> c = confusion_matrix(y_true, y_pred);

julia> positive_predictive_value(c)
0.3

julia> positive_predictive_value(c, average = "binary")
3-element Array{Float64,1}:
 0.0
 0.4
 0.5

julia> positive_predictive_value(c, average = "binary", normalize=true)
3-element Array{Float64,1}:
 0.0
 0.6246950475544243
 0.7808688094430303

julia> positive_predictive_value(c, average = "binary", normalize=true, ith_class=2)
0.4

julia> positive_predictive_value(c, ith_class=1)
0.0

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
julia> y_true = [2, 3, 2, 2, 2, 1, 3, 3, 2, 1, 2, 3, 2, 3, 2];

julia> y_pred = [1, 3, 1, 1, 3, 2, 2, 1, 3, 2, 3, 3, 2, 3, 2];

julia> c = confusion_matrix(y_true, y_pred);

julia> accuracy_score(c)
0.3

julia> accuracy_score(c, average="binary")
3-element Array{Float64,1}:
 0.0
 0.6246950475544243
 0.7808688094430303

julia> accuracy_score(c, average="binary", normalize=true)
3-element Array{Float64,1}:
 0.0
 0.6246950475544243
 0.7808688094430303

julia> accuracy_score(c, average="binary", normalize=true, class_name=2)
0.4

```

See Also : [`jaccard_score`](@ref), [`confusion_matrix`](@ref), [`hamming_loss`](@ref), [`balanced_accuracy_score`](@ref), [`recall_score`](@ref)

"""
function accuracy_score(c::confusion_matrix; ith_class = nothing, class_name = nothing, average = "macro", normalize = true, weights = nothing)
    index = check_index(c.Labels, true, ith_class = ith_class, class_name = class_name, valid_modes = ["binary", "macro", "weighted", "micro", "sample-weights"], average = average,weights=weights)
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
julia> y_true = [2, 3, 2, 2, 2, 1, 3, 3, 2, 1, 2, 3, 2, 3, 2];

julia> y_pred = [1, 3, 1, 1, 3, 2, 2, 1, 3, 2, 3, 3, 2, 3, 2];

julia> c = confusion_matrix(y_true, y_pred);

julia> balanced_accuracy_score(c)
0.2833333333333333

julia> balanced_accuracy_score(c, average="micro")
0.85

julia> balanced_accuracy_score(c, average="binary")
3-element Array{Float64,1}:
 0.0
 0.25
 0.6

julia> balanced_accuracy_score(c, average="sample-weights", weights=[1,2,3])
0.3833333333333333

julia> balanced_accuracy_score(c, average="binary", normalize=true)
3-element Array{Float64,1}:
 0.0
 0.3846153846153846
 0.9230769230769229

```

See Also : [`accuracy_score`](@ref) , [`confusion_matrix`](@ref) , [`hamming_loss`](@ref), [`balanced_accuracy_score`](@ref) ,[`recall_score`](@ref)

"""
function balanced_accuracy_score(c::confusion_matrix; ith_class = nothing, class_name = nothing , average = "macro", weights = nothing, normalize = false)
    index = check_index(c.Labels, true, ith_class = ith_class, class_name = class_name, valid_modes = ["binary", "macro", "weighted", "micro", "sample-weights"], average = average,weights=weights)
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
julia> y_true = [2, 3, 2, 2, 2, 1, 3, 3, 2, 1, 2, 3, 2, 3, 2];

julia> y_pred = [1, 3, 1, 1, 3, 2, 2, 1, 3, 2, 3, 3, 2, 3, 2];

julia> c = confusion_matrix(y_true, y_pred);

julia> negative_predictive_value(c)
0.6653198653198653

julia> negative_predictive_value(c, average="binary")
3-element Array{Float64,1}:
 0.8181818181818182
 0.4
 0.7777777777777778

julia> negative_predictive_value(c, average="binary", normalize=true)
3-element Array{Float64,1}:
 0.6831574015089602
 0.3339880629599361
 0.6494212335332091

julia> negative_predictive_value(c, average="micro")
0.6666666666666666

julia> negative_predictive_value(c, ith_class=1)
0.8181818181818182

```

See Also :   [`confusion_matrix`](@ref), [`accuracy_score`](@ref), [`positive_predictive_value`](@ref), [`balanced_accuracy_score`](@ref)

"""
function negative_predictive_value(c::confusion_matrix; ith_class = nothing, class_name = nothing , average = "macro", weights = nothing, normalize = false)
    index = check_index(c.Labels, true, ith_class = ith_class, class_name = class_name, valid_modes = ["binary", "macro", "weighted", "micro", "sample-weights"], average = average,weights=weights)
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
julia> y_true = [2, 3, 2, 2, 2, 1, 3, 3, 2, 1, 2, 3, 2, 3, 2];

julia> y_pred = [1, 3, 1, 1, 3, 2, 2, 1, 3, 2, 3, 3, 2, 3, 2];

julia> c = confusion_matrix(y_true, y_pred);

julia> false_negative_rate(c)
0.7166666666666667

julia> false_negative_rate(c, average="binary")
3-element Array{Float64,1}:
 1.0
 0.75
 0.4

julia> false_negative_rate(c, average="binary", normalize=true)
3-element Array{Float64,1}:
 0.7619393177594593
 0.5714544883195944
 0.30477572710378376

julia> false_negative_rate(c, average="weighted")
0.6666666666666666

julia> false_negative_rate(c, ith_class=3)
0.4

 ```

 See Also :   [`confusion_matrix`](@ref), [`false_positive_rate`](@ref), [`positive_predictive_value`], [`balanced_accuracy_score`](@ref)

"""
function false_negative_rate(c::confusion_matrix; ith_class = nothing, class_name = nothing, average = "macro", weights = nothing, normalize = false)
    index = check_index(c.Labels, true, ith_class = ith_class, class_name = class_name, valid_modes = ["binary", "macro", "weighted", "micro", "sample-weights"], average = average,weights=weights)
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
julia> y_true = [2, 3, 2, 2, 2, 1, 3, 3, 2, 1, 2, 3, 2, 3, 2];

julia> y_pred = [1, 3, 1, 1, 3, 2, 2, 1, 3, 2, 3, 3, 2, 3, 2];

julia> c = confusion_matrix(y_true, y_pred);

julia> false_positive_rate(c)
0.34542124542124547

julia> false_positive_rate(c, average="binary")
3-element Array{Float64,1}:
 0.3076923076923077
 0.42857142857142855
 0.3

julia> false_positive_rate(c, average="weighted")
0.36959706959706956

julia> false_positive_rate(c, average="binary", normalize=true)
3-element Array{Float64,1}:
 0.5069760762696655
 0.7061452490898912
 0.4943016743629238

julia> false_positive_rate(c, class_name=1)
0.3076923076923077

```

See Also :   [`confusion_matrix`](@ref), [`false_negative_rate`](@ref), [`positive_predictive_value`](@ref), [`balanced_accuracy_score`](@ref)

"""
function false_positive_rate(c::confusion_matrix; ith_class = nothing, class_name = nothing, average = "macro", weights = nothing, normalize = false)
    index = check_index(c.Labels, true, ith_class = ith_class, class_name = class_name, valid_modes = ["binary", "macro", "weighted", "micro", "sample-weights"], average = average,weights=weights)
    if index == -1
        numerator = c.false_positives
        denominator =  condition_negative(c)
    else
        numerator = c.false_positives[index] ./ condition_negative(c,ith_class = index)
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
julia> y_true = [2, 3, 2, 2, 2, 1, 3, 3, 2, 1, 2, 3, 2, 3, 2];

julia> y_pred = [1, 3, 1, 1, 3, 2, 2, 1, 3, 2, 3, 3, 2, 3, 2];

julia> c = confusion_matrix(y_true, y_pred);

julia> false_discovery_rate(c)
0.34542124542124547

julia> false_discovery_rate(c, average="binary")
3-element Array{Float64,1}:
 0.3076923076923077
 0.42857142857142855
 0.3

julia> false_discovery_rate(c, average="binary", normalize=true)
3-element Array{Float64,1}:
 0.5069760762696655
 0.7061452490898912
 0.4943016743629238

julia> false_discovery_rate(c, average="micro")
0.3333333333333333

julia> false_discovery_rate(c, average="weighted")
0.36959706959706956

julia> false_discovery_rate(c, ith_class=2)
0.42857142857142855

julia> false_discovery_rate(c, average="sample-weights", weights=[3,2,1])
0.34670329670329664

```
See Also :   [`confusion_matrix`](@ref), [`accuracy_score`](@ref),  [`positive_predictive_value`](@ref), [`false_omission_rate`](@ref)

"""
function false_discovery_rate(c::confusion_matrix; ith_class = nothing, class_name = nothing, average = "macro", weights = nothing, normalize = false)
    index = check_index(c.Labels, true, ith_class = ith_class, class_name = class_name, valid_modes = ["binary", "macro", "weighted", "micro", "sample-weights"], average = average,weights=weights)
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
julia> y_true = [2, 3, 2, 2, 2, 1, 3, 3, 2, 1, 2, 3, 2, 3, 2];

julia> y_pred = [1, 3, 1, 1, 3, 2, 2, 1, 3, 2, 3, 3, 2, 3, 2];

julia> c = confusion_matrix(y_true, y_pred);

julia> false_omission_rate(c)
0.3346801346801347

julia> false_omission_rate(c,average="binary")
3-element Array{Float64,1}:
 0.18181818181818177
 0.6
 0.2222222222222222

julia> false_omission_rate(c,average="binary", normalize=true)
3-element Array{Float64,1}:
 0.2733441855753975
 0.902035812398812
 0.33408733792548595

julia> false_omission_rate(c,average="sample-weights", weights=[5,4,3])
0.3313131313131313

julia> false_omission_rate(c, ith_class =2)
0.6

```

See Also :   [`confusion_matrix`](@ref), [`accuracy_score`](@ref), [`positive_predictive_value`], [`false_discovery_rate`](@ref)
"""
function false_omission_rate(c::confusion_matrix; ith_class = nothing, class_name = nothing, average = "macro", weights = nothing, normalize = false)
    index = check_index(c.Labels, true, ith_class = ith_class, class_name = class_name, valid_modes = ["binary", "macro", "weighted", "micro", "sample-weights"], average = average,weights=weights)
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


function fbeta_score(c::confusion_matrix; ith_class = nothing, class_name = nothing, average = "macro", weights = nothing, normalize = false, beta=1)
    index = check_index(c.Labels, true, ith_class = ith_class, class_name = class_name, valid_modes = ["binary", "macro", "weighted", "micro", "sample-weights"], average = average,weights=weights)
    if index == -1
        p = precision_score(c, average="binary")
        r = recall_score(c, average="binary")
        numerator = (1 + beta.*beta) .* p .* r
        denominator = (beta.*beta) .* p .+ r
    else
        p = precision_score(c, ith_class=index)
        r = recall_score(c, ith_class=index)
        numerator = (1 + beta*beta) .* p .* r
        denominator = (beta*beta) .* p .+ r
    end

    if average == "weighted"; weights = c.true_positives .+ c.false_negatives ; end
    return _average_helper(numerator, denominator, weights, average, c.zero_division, normalize)
end

fbeta_score(expected, predicted; ith_class = nothing, class_name = nothing, average = "macro", weights = nothing, normalize = false, beta=1) =
    fbeta_score(confusion_matrix(expected,predicted), ith_class = ith_class, class_name = class_name, average = average, normalize = normalize, beta=beta)


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

```

See Also :   [`confusion_matrix`](@ref), [`accuracy_score`](@ref), [`recall_score`](@ref), [`false_omission_rate`](@ref)

"""
function f1_score(c::confusion_matrix; ith_class = nothing, class_name = nothing, average = "macro", weights = nothing, normalize = false)
    return fbeta_score(c, ith_class = ith_class, class_name = class_name, average = average, weights = weights, normalize = normalize, beta = 1)
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
julia> y_true = [2, 3, 2, 2, 2, 1, 3, 3, 2, 1, 2, 3, 2, 3, 2];

julia> y_pred = [1, 3, 1, 1, 3, 2, 2, 1, 3, 2, 3, 3, 2, 3, 2];

julia> c = confusion_matrix(y_true, y_pred);

julia> prevalence_threshold(c)
0.6603944281302531

julia> prevalence_threshold(c, average="binary")
3-element Array{Float64,1}:
 1.0
 0.5669697220176636
 0.4142135623730953

julia> prevalence_threshold(c, average="binary", normalize=true)
3-element Array{Float64,1}:
 0.8184008853423851
 0.46400852246158186
 0.33899274616696445

julia> prevalence_threshold(c, average="micro")
1.5283320194691157

julia> prevalence_threshold(c, average="weighted")
0.5737883725337857

julia> prevalence_threshold(c, ith_class=2)
0.5669697220176636

```

See Also :  [ `confusion_matrix`](@ref),[ `accuracy_score`](@ref), [`recall_score`](@ref), [`f1_score`](@ref)

"""
function prevalence_threshold(c::confusion_matrix; ith_class = nothing, class_name = nothing, average = "macro", weights = nothing, normalize = false)
    index = check_index(c.Labels, true, ith_class = ith_class, class_name = class_name, valid_modes = ["binary", "macro", "weighted", "micro", "sample-weights"], average = average,weights=weights)
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
julia> y_true = [2, 3, 2, 2, 2, 1, 3, 3, 2, 1, 2, 3, 2, 3, 2];

julia> y_pred = [1, 3, 1, 1, 3, 2, 2, 1, 3, 2, 3, 3, 2, 3, 2];

julia> c = confusion_matrix(y_true, y_pred);

julia> threat_score(c)
0.18560606060606064

julia> threat_score(c, average= "binary")
3-element Array{Float64,1}:
 0.0
 0.18181818181818182
 0.375

julia> threat_score(c, average= "binary", normalize=true)
3-element Array{Float64,1}:
 0.0
 0.43627350651936947
 0.8998141071961996

julia> threat_score(c, average= "micro")
0.2

julia> threat_score(c, average= "weighted")
0.22196969696969698

julia> threat_score(c, ith_class=1)
0.0

```
See Also :   [`confusion_matrix`](@ref), [`accuracy_score`](@ref), [`recall_score`](@ref) , [`f1_score`](@ref)

"""
function threat_score(c::confusion_matrix; ith_class = nothing, class_name = nothing, average = "macro", weights = nothing, normalize = false)
    index = check_index(c.Labels, true, ith_class = ith_class, class_name = class_name, valid_modes = ["binary", "macro", "weighted", "micro", "sample-weights"], average = average,weights=weights)
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
julia> y_true = [2, 3, 2, 2, 2, 1, 3, 3, 2, 1, 2, 3, 2, 3, 2];

julia> y_pred = [1, 3, 1, 1, 3, 2, 2, 1, 3, 2, 3, 3, 2, 3, 2];

julia> c = confusion_matrix(y_true, y_pred);

julia> matthews_correlation_coeff(c)
0.01573112115074754

julia> matthews_correlation_coeff(c, average="binary")
3-element Array{Float64,1}:
 -0.1312004408342218
 -0.13630866504628206
  0.3147024693327465

julia> matthews_correlation_coeff(c, average="micro")
0.04719336345224262

julia> matthews_correlation_coeff(c, average="binary", normalize = true, ith_class=2)
-0.13630866504628206

```

See Also :   [`confusion_matrix`](@ref), [`accuracy_score`](@ref), [`threat_score`](@ref), [`f1_score`](@ref)

"""
function matthews_correlation_coeff(c::confusion_matrix; ith_class = nothing, class_name = nothing, average = "macro", weights = nothing, normalize = false)
    index = check_index(c.Labels, true, ith_class = ith_class, class_name = class_name, valid_modes = ["binary", "macro", "weighted", "micro", "sample-weights"], average = average,weights=weights)
    if index == -1
        x = sqrt.(.*(positive_predictive_value(c, average="binary"), true_positive_rate(c, average="binary"), true_negative_rate(c, average="binary"), negative_predictive_value(c, average="binary"))) .-
            sqrt.(.*(false_discovery_rate(c, average="binary"), false_negative_rate(c, average="binary"), false_positive_rate(c, average="binary"), false_omission_rate(c, average="binary")))
    else
        x = sqrt(*(positive_predictive_value(c, ith_class=index), true_positive_rate(c, ith_class=index), true_negative_rate(c, ith_class=index), negative_predictive_value(c, ith_class=index))) -
            sqrt(*(false_discovery_rate(c, ith_class=index), false_negative_rate(c, ith_class=index), false_positive_rate(c, ith_class=index), false_omission_rate(c, ith_class=index)))
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
julia> y_true = [2, 3, 2, 2, 2, 1, 3, 3, 2, 1, 2, 3, 2, 3, 2];

julia> y_pred = [1, 3, 1, 1, 3, 2, 2, 1, 3, 2, 3, 3, 2, 3, 2];

julia> c = confusion_matrix(y_true, y_pred);

julia> fowlkes_mallows_index(c)
0.6183448743333355

julia> fowlkes_mallows_index(c, average="binary")
3-element Array{Float64,1}:
 0.0
 0.806225774829855
 1.0488088481701516

julia> fowlkes_mallows_index(c, average="binary", normalize=true)
3-element Array{Float64,1}:
 0.0
 0.609449400220044
 0.7928249671720918

julia> fowlkes_mallows_index(c, average="binary", normalize=true, ith_class=2)
3-element Array{Float64,1}:
 0.0
 0.609449400220044
 0.7928249671720918

julia> fowlkes_mallows_index(c, average="micro")
1.8550346230000065

```

See Also :   [`confusion_matrix`](@ref), [`matthews_correlation_coeff`](@ref), [`threat_score`](@ref), [`f1_score`](@ref)
"""
function fowlkes_mallows_index(c::confusion_matrix; ith_class = nothing, class_name = nothing, average = "macro", weights = nothing, normalize = false)
    index = check_index(c.Labels, true, ith_class = ith_class, class_name = class_name, valid_modes = ["binary", "macro", "weighted", "micro", "sample-weights"], average = average,weights=weights)
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
julia> y_true = [2, 3, 2, 2, 2, 1, 3, 3, 2, 1, 2, 3, 2, 3, 2];

julia> y_pred = [1, 3, 1, 1, 3, 2, 2, 1, 3, 2, 3, 3, 2, 3, 2];

julia> c = confusion_matrix(y_true, y_pred);

julia> informedness(c)
-0.062087912087912166

julia> informedness(c, average="binary")
3-element Array{Float64,1}:
 -0.3076923076923077
 -0.1785714285714286
  0.2999999999999998

julia> informedness(c, average="binary", ith_class=2)
-0.1785714285714286

julia> informedness(c, average="binary", normalize=true)
3-element Array{Float64,1}:
 -0.6611883610841327
 -0.38372538812918416
  0.6446586520570289

julia> informedness(c, average="micro")
-0.1862637362637365

```

See Also :   [`confusion_matrix`](@ref), [`matthews_correlation_coeff`](@ref), [`markedness`](@ref), [`f1_score`](@ref)
"""
function informedness(c::confusion_matrix; ith_class = nothing, class_name = nothing, average = "macro", weights = nothing, normalize = false)
    index = check_index(c.Labels, true, ith_class = ith_class, class_name = class_name, valid_modes = ["binary", "macro", "weighted", "micro", "sample-weights"], average = average,weights=weights)
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
julia> y_true = [2, 3, 2, 2, 2, 1, 3, 3, 2, 1, 2, 3, 2, 3, 2];

julia> y_pred = [1, 3, 1, 1, 3, 2, 2, 1, 3, 2, 3, 3, 2, 3, 2];

julia> c = confusion_matrix(y_true, y_pred);

julia> markedness(c)
-0.03468013468013468

julia> markedness(c, average="binary")
3-element Array{Float64,1}:
 -0.18181818181818177
 -0.19999999999999996
  0.2777777777777777

julia> markedness(c, average="binary", normalize=true)
3-element Array{Float64,1}:
 -0.4691112238990908
 -0.5160223462889999
  0.7166977031791665

julia> markedness(c, average="weighted")
-0.03831649831649832

julia> markedness(c, ith_class=3)
0.2777777777777777

```

See Also :   [`confusion_matrix`](@ref), [`matthews_correlation_coeff`](@ref), [`informedness`](@ref), [`f1_score`](@ref)

"""
function markedness(c::confusion_matrix; ith_class = nothing, class_name = nothing, average = "macro", weights = nothing, normalize = false)
    index = check_index(c.Labels, true, ith_class = ith_class, class_name = class_name, valid_modes = ["binary", "macro", "weighted", "micro", "sample-weights"], average = average,weights=weights)
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
julia> y_true = [2, 3, 2, 2, 2, 1, 3, 3, 2, 1, 2, 3, 2, 3, 2];

julia> y_pred = [1, 3, 1, 1, 3, 2, 2, 1, 3, 2, 3, 3, 2, 3, 2];

julia> c = confusion_matrix(y_true, y_pred);

julia> cohen_kappa_score(c)
-0.020408163265306145

julia> cohen_kappa_score(c, weights = "quadratic")
0.19753086419753085

julia> cohen_kappa_score(c, weights = "linear")
0.07821229050279332

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
julia> y_true = [2, 3, 2, 2, 2, 1, 3, 3, 2, 1, 2, 3, 2, 3, 2];

julia> y_pred = [1, 3, 1, 1, 3, 2, 2, 1, 3, 2, 3, 3, 2, 3, 2];

julia> c = confusion_matrix(y_true, y_pred);

julia> hamming_loss(c)
0.6666666666666666

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
julia> y_true = [2, 3, 2, 2, 2, 1, 3, 3, 2, 1, 2, 3, 2, 3, 2];

julia> y_pred = [1, 3, 1, 1, 3, 2, 2, 1, 3, 2, 3, 3, 2, 3, 2];

julia> c = confusion_matrix(y_true, y_pred);

julia> jaccard_score(c)
0.18560606060606064

julia> jaccard_score(c, average="binary")
3-element Array{Float64,1}:
 0.0
 0.18181818181818182
 0.375

julia> jaccard_score(c, average="binary", normalize=true)
3-element Array{Float64,1}:
 0.0
 0.43627350651936947
 0.8998141071961996

julia> jaccard_score(c, average="binary", normalize=true,ith_class=1)
0.0

julia> jaccard_score(c, average="weighted")
0.22196969696969698

julia> jaccard_score(c, class_name=2)
0.18181818181818182

```

See Also :   [`confusion_matrix`](@ref), [`accuracy_score`](@ref), [`hamming_loss`](@ref), [`f1_score`](@ref)

"""
function jaccard_score(c::confusion_matrix;  ith_class = nothing, class_name = nothing, average = "macro", weights = nothing, normalize = false)
    index = check_index(c.Labels, true, ith_class = ith_class, class_name = class_name, valid_modes = ["binary", "macro", "weighted", "micro", "sample-weights"], average = average,weights=weights)
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
