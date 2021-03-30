#Visualization Functions

export visualize

using Plots

gr()

macro _call_func_(f,c, kwargs...)
    z = [esc(a) for a in kwargs]
    return :((eval($f))($c; $(z...)))
end

function _plot(c::confusion_matrix; func = nothing, type = nothing, title = "Visualization", labels = nothing)
    x = @_call_func_ Symbol(type)  c average = "binary"
    l = Array{typeof(labels[1])}(labels)
    x = func(l, x, title = title, labels = permutedims(l))
end

"""
```visualize(c::confusion_matrix, <keywords>)```

Visualize the given properties of the confusion matrix as specified

## Keywords
- `mode` : String, Array of String; represents properties given below of the confusion matrix, can be either an array containing different properties
or a single string.\n
    Supported Modes: \n
    - `matrix`\n
    - `condition_positive`\n
    - `condition_negative`\n
    - `predicted_positive\n
    - `predicted_negative`\n
    - `correctly_classified`\n
    - `incorrectly_classified`\n
    - `sensitivity_score`\n
    - `recall_score`\n
    - `specificity_score`\n
    - `precision_score`\n
    - `positive_predictive-value`\n
    - `accuracy_score`\n
    - `balanced_accuracy-score`\n
    - `negative_predictive-value`\n
    - `false_negative_rate`\n
    - `false_positive_rate`\n
    - `false_discovery_rate`\n
    - `false_omission_rate`\n
    - `f1_score`\n
    - `prevalence_threshold`\n
    - `threat_score`\n
    - `matthews_correlation_coeff`\n
    - `fowlkes_mallows_index`\n
    - `informedness`\n
    - `markedness`\n
    - `cohen_kappa_score`\n
    - `hamming_loss`\n
    - `jaccard_score`\n

- `seriestype::String = "heatmap"` :
    Supported visualization functions:
        - `heatmap`
        - `bar`
        - `histogram`
        - `scatter`
        - `line`

`title::String` : Denotes the title that will displayed above the drawn plot, default: nothing

`labels::Vector` : Denotes the labels that will be used for the plot. If equals to nothing, labels of the given confusion matrix will be used.
"""
function visualize(c::confusion_matrix; mode = "matrix", seriestype::String = "heatmap", title= nothing, labels = nothing)
    @assert seriestype in ["scatter", "heatmap", "line", "histogram", "bar"] "Unknown visualization format"
    labels = labels != nothing ? labels : convert(Array{typeof(c.Labels[1])}, c.Labels)
    if title == nothing; title = mode isa Array ? mode : String(Base.copymutable(mode)); end
    title = title isa Array ? title : [title]
    mode = mode isa Array ? mode : [mode]
    plt = []
    for i in 1:length(mode)
        @assert mode[i] in ["matrix", "condition_positive", "condition_negative", "predicted_positive","predicted_negative", "correctly_classified", "incorrectly_classified", "sensitivity_score", "recall_score", "specificity_score", "precision_score", "positive_predictive_value", "accuracy_score", "balanced_accuracy_score", "negative_predictive_value", "false_negative_rate", "false_positive_rate", "false_discovery_rate",
         "false_omission_rate", "f1_score", "prevalence_threshold", "threat_score", "matthews_correlation_coeff", "fowlkes_mallows_index",
         "informedness", "markedness", "cohen_kappa_score", "hamming_loss", "jaccard_score"] "Unknown visualization mode"
        if mode[i] != "matrix"; @assert seriestype in ["scatter", "line", "histogram", "bar"] "The given mode does not support this visualization format"; end
        x = nothing
        if mode[i] == "matrix"
            if seriestype == "histogram"; x = histogram(labels, c.matrix, labels = permutedims(labels), title = title[i])
            elseif seriestype == "scatter"; x = scatter(labels, c.matrix, labels = permutedims(labels), title = title[i])
            elseif seriestype == "line"; x = plot(labels, c.matrix, labels = permutedims(labels), title = title[i])
            elseif seriestype == "bar"; x = bar(labels, c.matrix, labels = permutedims(labels), title = title[i])
            elseif seriestype == "heatmap"; x = heatmap(labels, labels, c.matrix, labels = permutedims(labels), title = title[i])
            end
        else
            if seriestype == "histogram"; x = _plot(c; func = histogram, type = Symbol(mode[i]), title = title[i], labels = labels)
            elseif seriestype == "scatter"; x = _plot(c; func = scatter, type = Symbol(mode[i]), title = title[i], labels = labels)
            elseif seriestype == "line"; x =  _plot(c; func = plot, type = Symbol(mode[i]), title = title[i], labels = labels)
            elseif seriestype == "bar"; x =  _plot(c; func = bar, type = Symbol(mode[i]), title = title[i], labels = labels)
            elseif seriestype == "heatmap"; x =  _plot(c; func = heatmap, type = Symbol(mode[i]), title = title[i], labels = labels)
            end
        end
        push!(plt, x)
    end
    plot(plt..., layout = (length(plt), 1))
end


visualize(r::RocCurve, title = "Roc Curve", marker = (15, 0.2, :orange)) = scatter(r.ratios, annotations = [(r.ratios[i]..., r.thresholds[i]) for i in 1:length(r.ratios)], title = title, marker = marker)
visualize(r::PrecisionRecallCurve, title = "Precision Recall Curve", marker = (15, 0.2, :orange)) = scatter(r.ratios, annotations = [(r.ratios[i]..., r.thresholds[i]) for i in 1:length(r.ratios)], title = title, marker = marker)
