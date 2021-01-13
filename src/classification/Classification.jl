module Classification

import Plots
import Statistics

include("confusion_matrix.jl"); export confusion_params, confusion_matrix, class_confusion
include("metrics.jl"); export classification_report, condition_positive, condition_negative, predicted_positive,predicted_negative, correctly_classified, incorrectly_classified, sensitivity_score, recall_score, specificity_score, precision_score, positive_predictive_value, accuracy_score, balanced_accuracy_score, negative_predictive_value, false_negative_rate, false_positive_rate, false_discovery_rate, false_omission_rate, f1_score, prevalence_threshold, threat_score, matthews_correlation_coeff, fowlkes_mallows_index, informedness, markedness, cohen_kappa_score, hamming_loss, jaccard_score, confusion_params
include("visualization.jl"); export visualize
include("trackers.jl"); export ConditionPositive,ConditionNegative,PredictedPositive,PredictedNegative,CorrectlyClassified,IncorrectlyClassified,SensitivityScore,RecallScore,SpecificityScore,PrecisionScore,PositivePredictiveValue,AccuracyScore,BalancedAccuracyScore,NegativePredictiveValue,FalseNegativeRate,FalsePositiveRate,FalseDiscoveryRate,FalseOmissionRate,F1Score,PrevalenceThreshold,ThreatScore,MatthewsCorrelationCoeff,FowlkesMallowsIndex,Informedness,Markedness,CohenKappaScore,HammingLoss,JaccardScore,
ClassificationReport,ConfusionMatrix

end
