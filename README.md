# KnetMetrics
A standalone machine learning metrics library implemented in pure [Julia](http://docs.julialang.org) by [Emirhan Kurtuluş](https://github.com/ekurtulus). A vast collection of classification, regression and pairwise metrics and related visualizations are included. This package is created as a part of the [Knet](https://github.com/denizyuret/Knet.jl) ecosystem; however, built-in Julia arrays and all other types that support the same set of operations are compatible.

## Examples
```julia
julia> Pkg.add("KnetMetrics"); using KnetMetrics
# dummy data
julia> y_true = [2, 3, 2, 2, 2, 1, 3, 3, 2, 1, 2, 3, 2, 3, 2, 3, 1,1];
julia> y_pred = [1, 3, 1, 1, 3, 2, 2, 1, 3, 2, 3, 3, 2, 3, 2, 1, 1, 3];
# creating a confusion matrix
julia> c = confusion_matrix( y_true, y_pred, labels = [1,2,3]) #labels are truly optional

            Expected

      1      2      3
_____________________
      1      2      1   │1
      3      2      3   │2
      2      1      3   │3      Predicted

# testing some metrics
julia> f1_score(c) # f1_score(y_true, y_pred)
0.32307692307692304

julia> f1_score(c, average = "binary")
3-element Array{Float64,1}:
 0.2
 0.3076923076923077
 0.4615384615384615

julia> f1_score(c, average = "binary", normalize=true)
3-element Array{Float64,1}:
 0.33918173268560714
 0.5218180502855494
 0.7827270754283241

julia> f1_score(c, class_name = 3)
0.4615384615384615

julia> matthews_correlation_coeff(c, average = "micro")
0.20396752553080869

julia> matthews_correlation_coeff(c, average = "weighted")
0.07138375997792953

julia> matthews_correlation_coeff(c, average = "sample-weights", weights = [3,2,1])
0.03263110671272045

julia> minkowski_distance(y_true, y_pred)
21

julia> mean_absolute_error(y_true, y_pred)
0.8333333333333334
```

## Currently Supported Metrics
Note: (*) symbol denotes that the function has a built-in visualization through ```visualize``` function.
#### Classification
- Confusion Matrix *
- Condition Positive and Negative *
- Predicted Positive and Negative *
- Correctly and Incorrectly Classified
- True Positive Rate (Sensitivity Score, Recall Score) *
- True Negative Rate (Specificity Score) *
- Positive Predictive Value (Precision Score) *
- Accuracy Score *
- Balanced Accuracy Score *
- Negative Predictive Value *
- False Negative Rate *
- False Positive Rate *
- False Discovery Rate *
- False Omission Rate *
- Fbeta Score (F1 Score) *
- Prevalence Threshold *
- Threat Score *
- Matthews Correlation Coefficient *
- Fowlkes Mallows Index *
- Informedness *
- Markedness *
- Cohen Kappa Score
- Hamming Loss
- Jaccard Score *

#### Regression
- Maximum Residual Error
- Mean Absolute Error
- Mean Squared Error
- Mean Squared Log Error
- Median Absolute Error
- Mean Absolute Percentage Error

#### Pairwise
- Minkowski Distance
- Euclidian Distance
- Manhattan Distance
- Chebyshev Distance
- Braycurtis Distance
- Canberra Distance
- Cityblock Distance
- Mahalanobis Distance
- Correlation
- Cosine Distance
- Cosine Similarity


## TO-DO
1. A greater range of Roc Curve related functions
2. A greater range of regression functions
3. A greater range of pairwise functions and kernels
4. Clustering metrics
