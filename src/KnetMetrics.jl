module KnetMetrics

import LinearAlgebra

import Statistics

CONVERT_ARRAY_TYPE = true
ARRAY_TYPE = Array
SUPRESS_WARNINGS = false

include("utilities/utilities.jl")
include("classification/classification.jl"); export confusion_matrix, class_confusion, visualize, classification_report, condition_positive, condition_negative, predicted_positive,predicted_negative, correctly_classified, incorrectly_classified, sensitivity_score, recall_score, specificity_score, precision_score, positive_predictive_value, accuracy_score, balanced_accuracy_score, negative_predictive_value, false_negative_rate, false_positive_rate, false_discovery_rate, false_omission_rate, f1_score, prevalence_threshold, threat_score, matthews_correlation_coeff, fowlkes_mallows_index, informedness, markedness, cohen_kappa_score, hamming_loss, jaccard_score, confusion_params
include("pairwise/pairwise.jl"); export minkowski_distance, euclidian_distance, manhattan_distance, chebyshev_distance, braycurtis_distance, canberra_distance, cityblock_distance, mahalanobis_distance,
correlation, cosine_distance, cosine_similarity
include("regression/regression.jl"); export max_error, mean_absolute_error, mean_squared_error, mean_squared_log_error, median_absolute_error, mean_absolute_percentage_error


using .classification
using .pairwise
using .regression

end
