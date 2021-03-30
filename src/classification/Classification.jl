module classification

#import Plots
import Statistics
using Requires
using ..utils

include("confusion_matrix.jl"); export confusion_params, confusion_matrix, class_confusion
include("metrics.jl"); export classification_report, condition_positive, condition_negative, predicted_positive,predicted_negative, correctly_classified, incorrectly_classified, sensitivity_score, recall_score, specificity_score, precision_score, positive_predictive_value, accuracy_score, balanced_accuracy_score, negative_predictive_value, false_negative_rate, false_positive_rate, false_discovery_rate, false_omission_rate, f1_score, prevalence_threshold, threat_score, matthews_correlation_coeff, fowlkes_mallows_index, informedness, markedness, cohen_kappa_score, hamming_loss, jaccard_score, confusion_params
include("roc_auc.jl"); export auc, RocCurve, PrecisionRecallCurve, ratio_at_threshold, fpr_at_threshold, tpr_at_threshold, auc_at_threshold, precision_at_threshold, recall_at_threshold

function __init__()
    @require Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80" @eval begin
        include("visualization.jl")
        export visualize
    end
end

end
