export ConfusionMatrix, ClassificationReport, HammingLoss, CohenKappaScore,ConditionPositive, ConditionNegative , PredictedPositive , PredictedNegative, CorrectlyClassified, IncorrectlyClassified, SensitivityScore, RecallScore, SpecificityScore, PrecisionScore, PositivePredictiveValue, AccuracyScore, BalancedAccuracyScore, NegativePredictiveValue, FalseNegativeRate, FalsePositiveRate, FalseDiscoveryRate, FalseOmissionRate, F1Score, PrevalenceThreshold, ThreatScore, MatthewsCorrelationCoeff, FowlkesMallowsIndex, Informedness, Markedness, JaccardScore

import Base: getindex

abstract type metricTracker end

##
# Struct definitions

for i in ( :ConditionPositive,:ConditionNegative,:PredictedPositive,:PredictedNegative,:CorrectlyClassified,:IncorrectlyClassified,:SensitivityScore,:RecallScore,:SpecificityScore,:PrecisionScore,:PositivePredictiveValue,:AccuracyScore,:BalancedAccuracyScore,:NegativePredictiveValue,:FalseNegativeRate,:FalsePositiveRate,:FalseDiscoveryRate,:FalseOmissionRate,:F1Score,:PrevalenceThreshold,:ThreatScore,:MatthewsCorrelationCoeff,:FowlkesMallowsIndex,:Informedness,:Markedness,:JaccardScore)
    @eval mutable struct $(i) <: metricTracker
        ith_class
        class_name
        average
        weights
        normalize
        num
        name
        eval_dict
    end
end

mutable struct ClassificationReport <: metricTracker
    normalize
    num
    name
    eval_dict
end

mutable struct ConfusionMatrix <: metricTracker
    labels
    normalize::Bool
    sample_weight::Int
    zero_division
    type
    num
    name
    eval_dict
end

mutable struct HammingLoss <: metricTracker
    num
    name
    eval_dict
end

mutable struct CohenKappaScore <: metricTracker
    weights
    num
    name
    eval_dict
end

##
# Functions

function getindex(s::T, key) where T <: metricTracker
    return s.eval_dict[key]
end

ClassificationReport(; normalize =false, name = "classification-report ", num = 1) = ClassificationReport(normalize, num, name, Dict())
function (s::ClassificationReport)(x::confusion_matrix; name = nothing, num = nothing)
    s.eval_dict[(name == nothing ? s.name : name) * string(num == nothing ? s.num : num)] = classification_report(x, normalize = s.normalize, return_dict = true)
    s.num += 1
end
(s::ClassificationReport)(y_true, y_pred; name = nothing, num = nothing) = s(confusion_matrix(y_true, y_pred), name = name, num = num)

CohenKappaScore(;num = 1, name = "cohen-kappa-score ", weights = nothing) = CohenKappaScore(weights,num,name, Dict())
function (s::CohenKappaScore)(x::confusion_matrix; name = nothing, num = nothing)
    s.eval_dict[(name == nothing ? s.name : name) * string(num == nothing ? s.num : num)] = cohen_kappa_score(confusion_matrix(y_true, y_pred), s.weights)
    s.num += 1
end
(s::CohenKappaScore)(x::confusion_matrix; name = nothing, num = nothing) = s(confusion_matrix(y_true, y_pred), s.weights)

HammingLoss(;name = "hamming-loss ", num = 1) = HammingLoss(num, name, Dict())
function (s::HammingLoss)(x::confusion_matrix; name = nothing, num = nothing)
    s.eval_dict[(name == nothing ? s.name : name) * string(num == nothing ? s.num : num)] = hamming_loss(x)
    s.num += 1
end
(s::ConditionPositive)(y_true, y_pred; name = nothing, num = nothing) = s(confusion_matrix(y_true, y_pred), name = name, num = num)

ConfusionMatrix(;num = 1, name = "confusion-matrix ", labels = nothing, normalize = false, sample_weight = 0, zero_division = "warn", type = Number) =
ConfusionMatrix(labels, normalize, sample_weight, zero_division, type, num, name, Dict{String, confusion_matrix}())
function (s::ConfusionMatrix)(y_true, y_pred; name = nothing, num = nothing)
    s.eval_dict[(name == nothing ? s.name : name) * string(num == nothing ? s.num : num)] = confusion_matrix(y_true, y_pred, labels = s.labels, normalize = s.normalize, sample_weight = s.sample_weight, zero_division = s.zero_division, type = s.type)
    s.num += 1
end

ConditionPositive(;average = "binary", ith_class = nothing, class_name = nothing, weights = nothing, normalize = false, name = "condition-positive " , num = 1) =
ConditionPositive(ith_class, class_name, average, weights, normalize, num, name, Dict())
function (s::ConditionPositive)(x::confusion_matrix; name = nothing, num = nothing)
    s.eval_dict[(name == nothing ? s.name : name) * string(num == nothing ? s.num : num)] = condition_positive(x, ith_class = s.ith_class, class_name = s.class_name, average = s.average, weights = s.weights, normalize = s.normalize)
    s.num += 1
end
(s::ConditionPositive)(y_true, y_pred; name = nothing, num = nothing) = s(confusion_matrix(y_true, y_pred), name = name, num = num)

ConditionNegative(;average = "binary", ith_class = nothing, class_name = nothing, weights = nothing, normalize = false, name = "condition-negative ", num = 1) =
ConditionNegative(ith_class, class_name, average, weights, normalize, num, name, Dict())
function (s::ConditionNegative)(x::confusion_matrix; name = nothing, num = nothing)
    s.eval_dict[(name == nothing ? s.name : name) * string(num == nothing ? s.num : num)] = condition_negative(x, ith_class = s.ith_class, class_name = s.class_name, average = s.average, weights = s.weights, normalize = s.normalize)
    s.num += 1
end
(s::ConditionNegative)(y_true, y_pred; name = nothing, num = nothing) = s(confusion_matrix(y_true, y_pred), name = name, num = num)

PredictedPositive(;average = "binary", ith_class = nothing, class_name = nothing, weights = nothing, normalize = false, name = "predicted-positive ", num = 1) =
PredictedPositive(ith_class, class_name, average, weights, normalize, num, name, Dict())
function (s::PredictedPositive)(x::confusion_matrix; name = nothing, num = nothing)
    s.eval_dict[(name == nothing ? s.name : name) * string(num == nothing ? s.num : num)] = predicted_positive(x, ith_class = s.ith_class, class_name = s.class_name, average = s.average, weights = s.weights, normalize = s.normalize)
    s.num += 1
end
(s::PredictedPositive)(y_true, y_pred; name = nothing, num = nothing) = s(confusion_matrix(y_true, y_pred), name = name, num = num)

PredictedNegative(;average = "binary", ith_class = nothing, class_name = nothing, weights = nothing, normalize = false, name = "predicted-negative ", num = 1) =
PredictedNegative(ith_class, class_name, average, weights, normalize, num, name, Dict())
function (s::PredictedNegative)(x::confusion_matrix; name = nothing, num = nothing)
    s.eval_dict[(name == nothing ? s.name : name) * string(num == nothing ? s.num : num)] = predicted_negative(x, ith_class = s.ith_class, class_name = s.class_name, average = s.average, weights = s.weights, normalize = s.normalize)
    s.num += 1
end
(s::PredictedNegative)(y_true, y_pred; name = nothing, num = nothing) = s(confusion_matrix(y_true, y_pred), name = name, num = num)

CorrectlyClassified(;average = "binary", ith_class = nothing, class_name = nothing, weights = nothing, normalize = false, name = "correctly-classified ", num = 1) =
CorrectlyClassified(ith_class, class_name, average, weights, normalize, num, name, Dict())
function (s::CorrectlyClassified)(x::confusion_matrix; name = nothing, num = nothing)
    s.eval_dict[(name == nothing ? s.name : name) * string(num == nothing ? s.num : num)] = correctly_classified(x, ith_class = s.ith_class, class_name = s.class_name, average = s.average, weights = s.weights, normalize = s.normalize)
    s.num += 1
end
(s::CorrectlyClassified)(y_true, y_pred; name = nothing, num = nothing) = s(confusion_matrix(y_true, y_pred), name = name, num = num)

IncorrectlyClassified(;average = "binary", ith_class = nothing, class_name = nothing, weights = nothing, normalize = false, name = "incorrectly-classified ", num = 1) =
IncorrectlyClassified(ith_class, class_name, average, weights, normalize, num, name, Dict())
function (s::IncorrectlyClassified)(x::confusion_matrix; name = nothing, num = nothing)
    s.eval_dict[(name == nothing ? s.name : name) * string(num == nothing ? s.num : num)] = incorrectly_classified(x, ith_class = s.ith_class, class_name = s.class_name, average = s.average, weights = s.weights, normalize = s.normalize)
    s.num += 1
end
(s::IncorrectlyClassified)(y_true, y_pred; name = nothing, num = nothing) = s(confusion_matrix(y_true, y_pred), name = name, num = num)

SensitivityScore(;average = "binary", ith_class = nothing, class_name = nothing, weights = nothing, normalize = false, name = "sensitivity-score ", num = 1) =
SensitivityScore(ith_class, class_name, average, weights, normalize, num, name, Dict())
function (s::SensitivityScore)(x::confusion_matrix; name = nothing, num = nothing)
    s.eval_dict[(name == nothing ? s.name : name) * string(num == nothing ? s.num : num)] = sensitivity_score(x, ith_class = s.ith_class, class_name = s.class_name, average = s.average, weights = s.weights, normalize = s.normalize)
    s.num += 1
end
(s::SensitivityScore)(y_true, y_pred; name = nothing, num = nothing) = s(confusion_matrix(y_true, y_pred), name = name, num = num)

RecallScore(;average = "binary", ith_class = nothing, class_name = nothing, weights = nothing, normalize = false, name = "recall-score ", num = 1) =
RecallScore(ith_class, class_name, average, weights, normalize, num, name, Dict())
function (s::RecallScore)(x::confusion_matrix; name = nothing, num = nothing)
    s.eval_dict[(name == nothing ? s.name : name) * string(num == nothing ? s.num : num)] = recall_score(x, ith_class = s.ith_class, class_name = s.class_name, average = s.average, weights = s.weights, normalize = s.normalize)
    s.num += 1
end
(s::RecallScore)(y_true, y_pred; name = nothing, num = nothing) = s(confusion_matrix(y_true, y_pred), name = name, num = num)

SpecificityScore(;average = "binary", ith_class = nothing, class_name = nothing, weights = nothing, normalize = false, name = "specificity-score ", num = 1) =
SpecificityScore(ith_class, class_name, average, weights, normalize, num, name, Dict())
function (s::SpecificityScore)(x::confusion_matrix; name = nothing, num = nothing)
    s.eval_dict[(name == nothing ? s.name : name) * string(num == nothing ? s.num : num)] = specificity_score(x, ith_class = s.ith_class, class_name = s.class_name, average = s.average, weights = s.weights, normalize = s.normalize)
    s.num += 1
end
(s::SpecificityScore)(y_true, y_pred; name = nothing, num = nothing) = s(confusion_matrix(y_true, y_pred), name = name, num = num)

PrecisionScore(;average = "binary", ith_class = nothing, class_name = nothing, weights = nothing, normalize = false, name = "precision-score ", num = 1) =
PrecisionScore(ith_class, class_name, average, weights, normalize, num, name, Dict())
function (s::PrecisionScore)(x::confusion_matrix; name = nothing, num = nothing)
    s.eval_dict[(name == nothing ? s.name : name) * string(num == nothing ? s.num : num)] = precision_score(x, ith_class = s.ith_class, class_name = s.class_name, average = s.average, weights = s.weights, normalize = s.normalize)
    s.num += 1
end
(s::PrecisionScore)(y_true, y_pred; name = nothing, num = nothing) = s(confusion_matrix(y_true, y_pred), name = name, num = num)

PositivePredictiveValue(;average = "binary", ith_class = nothing, class_name = nothing, weights = nothing, normalize = false, name = "positive-predictive-value ", num = 1) =
PositivePredictiveValue(ith_class, class_name, average, weights, normalize, num, name, Dict())
function (s::PositivePredictiveValue)(x::confusion_matrix; name = nothing, num = nothing)
    s.eval_dict[(name == nothing ? s.name : name) * string(num == nothing ? s.num : num)] = positive_predictive_value(x, ith_class = s.ith_class, class_name = s.class_name, average = s.average, weights = s.weights, normalize = s.normalize)
    s.num += 1
end
(s::PositivePredictiveValue)(y_true, y_pred; name = nothing, num = nothing) = s(confusion_matrix(y_true, y_pred), name = name, num = num)

AccuracyScore(;average = "binary", ith_class = nothing, class_name = nothing, weights = nothing, normalize = false, name = "accuracy-score ", num = 1) =
AccuracyScore(ith_class, class_name, average, weights, normalize, num, name, Dict())
function (s::AccuracyScore)(x::confusion_matrix; name = nothing, num = nothing)
    s.eval_dict[(name == nothing ? s.name : name) * string(num == nothing ? s.num : num)] = accuracy_score(x, ith_class = s.ith_class, class_name = s.class_name, average = s.average, weights = s.weights, normalize = s.normalize)
    s.num += 1
end
(s::AccuracyScore)(y_true, y_pred; name = nothing, num = nothing) = s(confusion_matrix(y_true, y_pred), name = name, num = num)

BalancedAccuracyScore(;average = "binary", ith_class = nothing, class_name = nothing, weights = nothing, normalize = false, name = "balanced-accuracy-score ", num = 1) =
BalancedAccuracyScore(ith_class, class_name, average, weights, normalize, num, name, Dict())
function (s::BalancedAccuracyScore)(x::confusion_matrix; name = nothing, num = nothing)
    s.eval_dict[(name == nothing ? s.name : name) * string(num == nothing ? s.num : num)] = balanced_accuracy_score(x, ith_class = s.ith_class, class_name = s.class_name, average = s.average, weights = s.weights, normalize = s.normalize)
    s.num += 1
end
(s::BalancedAccuracyScore)(y_true, y_pred; name = nothing, num = nothing) = s(confusion_matrix(y_true, y_pred), name = name, num = num)

NegativePredictiveValue(;average = "binary", ith_class = nothing, class_name = nothing, weights = nothing, normalize = false, name = "negative-predictive-value ", num = 1) =
NegativePredictiveValue(ith_class, class_name, average, weights, normalize, num, name, Dict())
function (s::NegativePredictiveValue)(x::confusion_matrix; name = nothing, num = nothing)
    s.eval_dict[(name == nothing ? s.name : name) * string(num == nothing ? s.num : num)] = negative_predictive_value(x, ith_class = s.ith_class, class_name = s.class_name, average = s.average, weights = s.weights, normalize = s.normalize)
    s.num += 1
end
(s::NegativePredictiveValue)(y_true, y_pred; name = nothing, num = nothing) = s(confusion_matrix(y_true, y_pred), name = name, num = num)

FalseNegativeRate(;average = "binary", ith_class = nothing, class_name = nothing, weights = nothing, normalize = false, name = "false-negative-rate ", num = 1) =
FalseNegativeRate(ith_class, class_name, average, weights, normalize, num, name, Dict())
function (s::FalseNegativeRate)(x::confusion_matrix; name = nothing, num = nothing)
    s.eval_dict[(name == nothing ? s.name : name) * string(num == nothing ? s.num : num)] = false_negative_rate(x, ith_class = s.ith_class, class_name = s.class_name, average = s.average, weights = s.weights, normalize = s.normalize)
    s.num += 1
end
(s::FalseNegativeRate)(y_true, y_pred; name = nothing, num = nothing) = s(confusion_matrix(y_true, y_pred), name = name, num = num)

FalsePositiveRate(;average = "binary", ith_class = nothing, class_name = nothing, weights = nothing, normalize = false, name = "false-positive-rate ", num = 1) =
FalsePositiveRate(ith_class, class_name, average, weights, normalize, num, name, Dict())
function (s::FalsePositiveRate)(x::confusion_matrix; name = nothing, num = nothing)
    s.eval_dict[(name == nothing ? s.name : name) * string(num == nothing ? s.num : num)] = false_positive_rate(x, ith_class = s.ith_class, class_name = s.class_name, average = s.average, weights = s.weights, normalize = s.normalize)
    s.num += 1
end
(s::FalsePositiveRate)(y_true, y_pred; name = nothing, num = nothing) = s(confusion_matrix(y_true, y_pred), name = name, num = num)

FalseDiscoveryRate(;average = "binary", ith_class = nothing, class_name = nothing, weights = nothing, normalize = false, name = "false-discovery-rate ", num = 1) =
FalseDiscoveryRate(ith_class, class_name, average, weights, normalize, num, name, Dict())
function (s::FalseDiscoveryRate)(x::confusion_matrix; name = nothing, num = nothing)
    s.eval_dict[(name == nothing ? s.name : name) * string(num == nothing ? s.num : num)] = false_discovery_rate(x, ith_class = s.ith_class, class_name = s.class_name, average = s.average, weights = s.weights, normalize = s.normalize)
    s.num += 1
end
(s::FalseDiscoveryRate)(y_true, y_pred; name = nothing, num = nothing) = s(confusion_matrix(y_true, y_pred), name = name, num = num)

FalseOmissionRate(;average = "binary", ith_class = nothing, class_name = nothing, weights = nothing, normalize = false, name = "false-omission-rate ", num = 1) =
FalseOmissionRate(ith_class, class_name, average, weights, normalize, num, name, Dict())
function (s::FalseOmissionRate)(x::confusion_matrix; name = nothing, num = nothing)
    s.eval_dict[(name == nothing ? s.name : name) * string(num == nothing ? s.num : num)] = false_omission_rate(x, ith_class = s.ith_class, class_name = s.class_name, average = s.average, weights = s.weights, normalize = s.normalize)
    s.num += 1
end
(s::FalseOmissionRate)(y_true, y_pred; name = nothing, num = nothing) = s(confusion_matrix(y_true, y_pred), name = name, num = num)

F1Score(;average = "binary", ith_class = nothing, class_name = nothing, weights = nothing, normalize = false, name = "f1-score ", num = 1) =
F1Score(ith_class, class_name, average, weights, normalize, num, name, Dict())
function (s::F1Score)(x::confusion_matrix; name = nothing, num = nothing)
    s.eval_dict[(name == nothing ? s.name : name) * string(num == nothing ? s.num : num)] = f1_score(x, ith_class = s.ith_class, class_name = s.class_name, average = s.average, weights = s.weights, normalize = s.normalize)
    s.num += 1
end
(s::F1Score)(y_true, y_pred; name = nothing, num = nothing) = s(confusion_matrix(y_true, y_pred), name = name, num = num)

PrevalenceThreshold(;average = "binary", ith_class = nothing, class_name = nothing, weights = nothing, normalize = false, name = "prevalence-threshold ", num = 1) =
PrevalenceThreshold(ith_class, class_name, average, weights, normalize, num, name, Dict())
function (s::PrevalenceThreshold)(x::confusion_matrix; name = nothing, num = nothing)
    s.eval_dict[(name == nothing ? s.name : name) * string(num == nothing ? s.num : num)] = prevalence_threshold(x, ith_class = s.ith_class, class_name = s.class_name, average = s.average, weights = s.weights, normalize = s.normalize)
    s.num += 1
end
(s::PrevalenceThreshold)(y_true, y_pred; name = nothing, num = nothing) = s(confusion_matrix(y_true, y_pred), name = name, num = num)

ThreatScore(;average = "binary", ith_class = nothing, class_name = nothing, weights = nothing, normalize = false, name = "threat-score ", num = 1) =
ThreatScore(ith_class, class_name, average, weights, normalize, num, name, Dict())
function (s::ThreatScore)(x::confusion_matrix; name = nothing, num = nothing)
    s.eval_dict[(name == nothing ? s.name : name) * string(num == nothing ? s.num : num)] = threat_score(x, ith_class = s.ith_class, class_name = s.class_name, average = s.average, weights = s.weights, normalize = s.normalize)
    s.num += 1
end
(s::ThreatScore)(y_true, y_pred; name = nothing, num = nothing) = s(confusion_matrix(y_true, y_pred), name = name, num = num)

MatthewsCorrelationCoeff(;average = "binary", ith_class = nothing, class_name = nothing, weights = nothing, normalize = false, name = "matthews-correlation-coeff", num = 1) =
MatthewsCorrelationCoeff(ith_class, class_name, average, weights, normalize, num, name, Dict())
function (s::MatthewsCorrelationCoeff)(x::confusion_matrix; name = nothing, num = nothing)
    s.eval_dict[(name == nothing ? s.name : name) * string(num == nothing ? s.num : num)] = matthews_correlation_coeff(x, ith_class = s.ith_class, class_name = s.class_name, average = s.average, weights = s.weights, normalize = s.normalize)
    s.num += 1
end
(s::MatthewsCorrelationCoeff)(y_true, y_pred; name = nothing, num = nothing) = s(confusion_matrix(y_true, y_pred), name = name, num = num)

FowlkesMallowsIndex(;average = "binary", ith_class = nothing, class_name = nothing, weights = nothing, normalize = false, name ="folwkes-mallows-index ", num = 1) =
FowlkesMallowsIndex(ith_class, class_name, average, weights, normalize, num, name, Dict())
function (s::FowlkesMallowsIndex)(x::confusion_matrix; name = nothing, num = nothing)
    s.eval_dict[(name == nothing ? s.name : name) * string(num == nothing ? s.num : num)] = fowlkes_mallows_index(x, ith_class = s.ith_class, class_name = s.class_name, average = s.average, weights = s.weights, normalize = s.normalize)
    s.num += 1
end
(s::FowlkesMallowsIndex)(y_true, y_pred; name = nothing, num = nothing) = s(confusion_matrix(y_true, y_pred), name = name, num = num)

Informedness(;average = "binary", ith_class = nothing, class_name = nothing, weights = nothing, normalize = false, name = "informedness ", num = 1) =
Informedness(ith_class, class_name, average, weights, normalize, num, name, Dict())
function (s::Informedness)(x::confusion_matrix; name = nothing, num = nothing)
    s.eval_dict[(name == nothing ? s.name : name) * string(num == nothing ? s.num : num)] = informedness(x, ith_class = s.ith_class, class_name = s.class_name, average = s.average, weights = s.weights, normalize = s.normalize)
    s.num += 1
end
(s::Informedness)(y_true, y_pred; name = nothing, num = nothing) = s(confusion_matrix(y_true, y_pred), name = name, num = num)

Markedness(;average = "binary", ith_class = nothing, class_name = nothing, weights = nothing, normalize = false, name = "markedness " , num = 1) =
Markedness(ith_class, class_name, average, weights, normalize, num, name, Dict())
function (s::Markedness)(x::confusion_matrix; name = nothing, num = nothing)
    s.eval_dict[(name == nothing ? s.name : name) * string(num == nothing ? s.num : num)] = markedness(x, ith_class = s.ith_class, class_name = s.class_name, average = s.average, weights = s.weights, normalize = s.normalize)
    s.num += 1
end
(s::Markedness)(y_true, y_pred; name = nothing, num = nothing) = s(confusion_matrix(y_true, y_pred), name = name, num = num)

JaccardScore(;average = "binary", ith_class = nothing, class_name = nothing, weights = nothing, normalize = false, name = "jaccard-score ", num = 1) =
JaccardScore(ith_class, class_name, average, weights, normalize, num, name, Dict())
function (s::JaccardScore)(x::confusion_matrix; name = nothing, num = nothing)
    s.eval_dict[(name == nothing ? s.name : name) * string(num == nothing ? s.num : num)] = jaccard_score(x, ith_class = s.ith_class, class_name = s.class_name, average = s.average, weights = s.weights, normalize = s.normalize)
    s.num += 1
end
(s::JaccardScore)(y_true, y_pred; name = nothing, num = nothing) = s(confusion_matrix(y_true, y_pred), name = name, num = num)

##
# Docs
"""
# Trackers
Trackers are utility objects that are defined to keep track of the calls made to the metrics functions. All the trackers and the functions they call:

    ConfusionMatrix : ` confusion_matrix`
    ClassificationReport : `  classification_report`
    HammingLoss : ` hamming_loss`
    CohenKappaScore : ` cohen_kappa_score`
    ConditionPositive : ` condition_positive`
    ConditionNegative : ` condition_negative`
    PredictedPositive : ` predicted_positive`
    PredictedNegative : ` predicted_negative`
    CorrectlyClassified : ` correctly_classified`
    IncorrectlyClassified : ` incorrectly_classified`
    SensitivityScore : ` sensitivity_score`
    RecallScore : ` recall_score`
    SpecificityScore : ` specificity_score`
    PrecisionScore : ` precision_score`
    PositivePredictiveValue : ` positive_predictive_value`
    AccuracyScore : ` accuracy_score`
    BalancedAccuracyScore : ` balanced_accuracy_score`
    NegativePredictiveValue : ` negative_predictive_value`
    FalseNegativeRate : ` false_negative_rate`
    FalsePositiveRate : ` false_positive_rate`
    FalseDiscoveryRate : ` false_discovery_rate`
    FalseOmissionRate : ` false_omission_rate`
    F1Score : ` f1_score`
    PrevalenceThreshold : ` prevalence_threshold`
    ThreatScore : ` threat_score`
    MatthewsCorrelationCoeff : ` matthews_correlation_coeff`
    FowlkesMallowsIndex : ` fowlkes_mallows_index`
    Informedness : ` informedness`
    Markedness : ` markedness`
    JaccardScore : ` jaccard_score`

All trackers have the fields that are the arguments of the called function for instance:

`JaccardScore(; ;average = "binary", ith_class = nothing, class_name = nothing, weights = nothing, normalize = false, name = "jaccard-score ", num = 1)`

When trackers are called, they return the output of the called function and store the output in their `eval_dict::Dict` field. First the num field
which counts the number of times a tracker is called is incremented by one and the `eval_dict` is updated as follows: `eval_dict[name * string(num)] = output`

##Example

```julia-repl

julia> y_pred = [ 1, 4, 4, 1, 2, 3, 1, 3, 2, 1, 2, 2, 2, 1, 4, 4, 2, 1, 3, 2, 2, 3, 2, 1, 2, 3, 4, 1, 2, 1];

julia> y_true = [ 4, 4, 1, 4, 4, 2, 1, 1, 2, 4, 1, 3, 3, 3, 1, 1, 3, 1, 4, 4, 3, 3, 3, 1, 4, 1, 2, 3, 2, 2];

julia>  x = confusion_matrix(y_true, y_pred);

julia> a = JaccardScore(average = "micro", num = 4, name = "Trackers Demo ");

julia> x = confusion_matrix(y_true, y_pred);
[ Info: No labels provided, constructing a label set by union of the unique elements in Expected and Predicted arrays

julia> a = JaccardScore(average = "micro", num = 4, name = "Trackers Demo "); #Start counting from 4

julia> a(x)
0.1320754716981132

julia> a(y_true, y_pred, name = "Names can be specified! ", num = 44)
0.1320754716981132

julia> a.average = "macro" # mutable struct
"macro"

julia> a(x, name = "Tracker ", num = 1) # compute with average = "macro"
0.12738095238095237

julia> a.eval_dict
Dict{Any,Any} with 3 entries:
  "Trackers Demo 5"            => 0.132075
  "Tracker 1"                  => 0.127381
  "Names can be specified! 44" => 0.132075

```
"""
ConfusionMatrix, ClassificationReport, HammingLoss, CohenKappaScore,ConditionPositive, ConditionNegative , PredictedPositive , PredictedNegative, CorrectlyClassified, IncorrectlyClassified, SensitivityScore, RecallScore, SpecificityScore, PrecisionScore, PositivePredictiveValue, AccuracyScore, BalancedAccuracyScore, NegativePredictiveValue, FalseNegativeRate, FalsePositiveRate, FalseDiscoveryRate, FalseOmissionRate, F1Score, PrevalenceThreshold, ThreatScore, MatthewsCorrelationCoeff, FowlkesMallowsIndex, Informedness, Markedness, JaccardScore
