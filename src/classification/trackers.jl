import Base: getindex

abstract type metricTracker end

Names =  ( :ConditionPositive,
:ConditionNegative,
:PredictedPositive,
:PredictedNegative,
:CorrectlyClassified,
:IncorrectlyClassified,
:SensitivityScore,
:RecallScore,
:SpecificityScore,
:PrecisionScore,
:PositivePredictiveValue,
:AccuracyScore,
:BalancedAccuracyScore,
:NegativePredictiveValue,
:FalseNegativeRate,
:FalsePositiveRate,
:FalseDiscoveryRate,
:FalseOmissionRate,
:F1Score,
:PrevalenceThreshold,
:ThreatScore,
:MatthewsCorrelationCoeff,
:FowlkesMallowsIndex,
:Informedness,
:Markedness,
:JaccardScore)

for i in Names
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

function getindex(s::T, key) where T <: metricTracker
    return s.eval_dict[key]
end

ClassificationReport(; normalize =false, name = "classification-report", num = 0) = ClassificationReport(normalize, num, name, Dict())
function (s::ClassificationReport)(x::confusion_matrix; name = nothing, num = nothing)
    s.num += 1
    s.eval_dict[(name == nothing ? s.name : name) * string(num == nothing ? s.num : num)] = classification_report(x, normalize = s.normalize, return_dict = true)
end
(s::ClassificationReport)(y_true, y_pred; name = nothing, num = nothing) = s(confusion_matrix(y_true, y_pred), name = name, num = num)

mutable struct CohenKappaScore <: metricTracker
    weights
    num
    name
    eval_dict
end

CohenKappaScore(;num = 0, name = "cohen-kappa-score", weights = nothing) = CohenKappaScore(weights,num,name, Dict())
function (s::CohenKappaScore)(x::confusion_matrix; name = nothing, num = nothing)
    s.num += 1
    s.eval_dict[(name == nothing ? s.name : name) * string(num == nothing ? s.num : num)] = cohen_kappa_score(confusion_matrix(y_true, y_pred), s.weights)
end
(s::CohenKappaScore)(x::confusion_matrix; name = nothing, num = nothing) = s(confusion_matrix(y_true, y_pred), s.weights)

mutable struct HammingLoss <: metricTracker
    num
    name
    eval_dict
end

HammingLoss(;name = "hamming-loss", num = 0) = HammingLoss(num, name, Dict())
function (s::HammingLoss)(x::confusion_matrix; name = nothing, num = nothing)
    s.num += 1
    s.eval_dict[(name == nothing ? s.name : name) * string(num == nothing ? s.num : num)] = hamming_loss(x)
end
(s::ConditionPositive)(y_true, y_pred; name = nothing, num = nothing) = s(confusion_matrix(y_true, y_pred), name = name, num = num)

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

ConfusionMatrix(;num = 0, name = "confusion-matrix", labels = nothing, normalize = false, sample_weight = 0, zero_division = "warn", type = num) =
ConfusionMatrix(labels, normalize, sample_weight, zero_division, type, num, name, Dict{String, confusion_matrix}())
function (s::ConfusionMatrix)(y_true, y_pred; name = nothing, num = nothing)
    s.num += 1
    s.eval_dict[(name == nothing ? s.name : name) * string(num == nothing ? s.num : num)] = confusion_matrix(y_true, y_pred, labels = s.labels, normalize = s.normalize, sample_weight = s.sample_weight, zero_division = s.zero_division, type = s.type)
end

ConditionPositive(;average = "binary", ith_class = nothing, class_name = nothing, weights = nothing, normalize = false, name = "condition-positive " , num = 0) =
ConditionPositive(ith_class, class_name, average, weights, normalize, num, name, Dict())
function (s::ConditionPositive)(x::confusion_matrix; name = nothing, num = nothing)
    s.num += 1
    s.eval_dict[(name == nothing ? s.name : name) * string(num == nothing ? s.num : num)] = condition_positive(x, ith_class = s.ith_class, class_name = s.class_name, average = s.average, weights = s.weights, normalize = s.normalize)
end
(s::ConditionPositive)(y_true, y_pred; name = nothing, num = nothing) = s(confusion_matrix(y_true, y_pred), name = name, num = num)

ConditionNegative(;average = "binary", ith_class = nothing, class_name = nothing, weights = nothing, normalize = false, name = "condition-negative ", num = 0) =
ConditionNegative(ith_class, class_name, average, weights, normalize, num, name, Dict())
function (s::ConditionNegative)(x::confusion_matrix; name = nothing, num = nothing)
    s.num += 1
    s.eval_dict[(name == nothing ? s.name : name) * string(num == nothing ? s.num : num)] = condition_negative(x, ith_class = s.ith_class, class_name = s.class_name, average = s.average, weights = s.weights, normalize = s.normalize)
end
(s::ConditionNegative)(y_true, y_pred; name = nothing, num = nothing) = s(confusion_matrix(y_true, y_pred), name = name, num = num)

PredictedPositive(;average = "binary", ith_class = nothing, class_name = nothing, weights = nothing, normalize = false, name = "predicted-positive ", num = 0) =
PredictedPositive(ith_class, class_name, average, weights, normalize, num, name, Dict())
function (s::PredictedPositive)(x::confusion_matrix; name = nothing, num = nothing)
    s.num += 1
    s.eval_dict[(name == nothing ? s.name : name) * string(num == nothing ? s.num : num)] = predicted_positive(x, ith_class = s.ith_class, class_name = s.class_name, average = s.average, weights = s.weights, normalize = s.normalize)
end
(s::PredictedPositive)(y_true, y_pred; name = nothing, num = nothing) = s(confusion_matrix(y_true, y_pred), name = name, num = num)

PredictedNegative(;average = "binary", ith_class = nothing, class_name = nothing, weights = nothing, normalize = false, name = "predicted-negative ", num = 0) =
PredictedNegative(ith_class, class_name, average, weights, normalize, num, name, Dict())
function (s::PredictedNegative)(x::confusion_matrix; name = nothing, num = nothing)
    s.num += 1
    s.eval_dict[(name == nothing ? s.name : name) * string(num == nothing ? s.num : num)] = predicted_negative(x, ith_class = s.ith_class, class_name = s.class_name, average = s.average, weights = s.weights, normalize = s.normalize)
end
(s::PredictedNegative)(y_true, y_pred; name = nothing, num = nothing) = s(confusion_matrix(y_true, y_pred), name = name, num = num)

CorrectlyClassified(;average = "binary", ith_class = nothing, class_name = nothing, weights = nothing, normalize = false, name = "correctly-classified ", num = 0) =
CorrectlyClassified(ith_class, class_name, average, weights, normalize, num, name, Dict())
function (s::CorrectlyClassified)(x::confusion_matrix; name = nothing, num = nothing)
    s.num += 1
    s.eval_dict[(name == nothing ? s.name : name) * string(num == nothing ? s.num : num)] = correctly_classified(x, ith_class = s.ith_class, class_name = s.class_name, average = s.average, weights = s.weights, normalize = s.normalize)
end
(s::CorrectlyClassified)(y_true, y_pred; name = nothing, num = nothing) = s(confusion_matrix(y_true, y_pred), name = name, num = num)

IncorrectlyClassified(;average = "binary", ith_class = nothing, class_name = nothing, weights = nothing, normalize = false, name = "incorrectly-classified ", num = 0) =
IncorrectlyClassified(ith_class, class_name, average, weights, normalize, num, name, Dict())
function (s::IncorrectlyClassified)(x::confusion_matrix; name = nothing, num = nothing)
    s.num += 1
    s.eval_dict[(name == nothing ? s.name : name) * string(num == nothing ? s.num : num)] = incorrectly_classified(x, ith_class = s.ith_class, class_name = s.class_name, average = s.average, weights = s.weights, normalize = s.normalize)
end
(s::IncorrectlyClassified)(y_true, y_pred; name = nothing, num = nothing) = s(confusion_matrix(y_true, y_pred), name = name, num = num)

SensitivityScore(;average = "binary", ith_class = nothing, class_name = nothing, weights = nothing, normalize = false, name = "sensitivity-score ", num = 0) =
SensitivityScore(ith_class, class_name, average, weights, normalize, num, name, Dict())
function (s::SensitivityScore)(x::confusion_matrix; name = nothing, num = nothing)
    s.num += 1
    s.eval_dict[(name == nothing ? s.name : name) * string(num == nothing ? s.num : num)] = sensitivity_score(x, ith_class = s.ith_class, class_name = s.class_name, average = s.average, weights = s.weights, normalize = s.normalize)
end
(s::SensitivityScore)(y_true, y_pred; name = nothing, num = nothing) = s(confusion_matrix(y_true, y_pred), name = name, num = num)

RecallScore(;average = "binary", ith_class = nothing, class_name = nothing, weights = nothing, normalize = false, name = "recall-score ", num = 0) =
RecallScore(ith_class, class_name, average, weights, normalize, num, name, Dict())
function (s::RecallScore)(x::confusion_matrix; name = nothing, num = nothing)
    s.num += 1
    s.eval_dict[(name == nothing ? s.name : name) * string(num == nothing ? s.num : num)] = recall_score(x, ith_class = s.ith_class, class_name = s.class_name, average = s.average, weights = s.weights, normalize = s.normalize)
end
(s::RecallScore)(y_true, y_pred; name = nothing, num = nothing) = s(confusion_matrix(y_true, y_pred), name = name, num = num)

SpecificityScore(;average = "binary", ith_class = nothing, class_name = nothing, weights = nothing, normalize = false, name = "specificity-score ", num = 0) =
SpecificityScore(ith_class, class_name, average, weights, normalize, num, name, Dict())
function (s::SpecificityScore)(x::confusion_matrix; name = nothing, num = nothing)
    s.num += 1
    s.eval_dict[(name == nothing ? s.name : name) * string(num == nothing ? s.num : num)] = specificity_score(x, ith_class = s.ith_class, class_name = s.class_name, average = s.average, weights = s.weights, normalize = s.normalize)
end
(s::SpecificityScore)(y_true, y_pred; name = nothing, num = nothing) = s(confusion_matrix(y_true, y_pred), name = name, num = num)

PrecisionScore(;average = "binary", ith_class = nothing, class_name = nothing, weights = nothing, normalize = false, name = "precision-score ", num = 0) =
PrecisionScore(ith_class, class_name, average, weights, normalize, num, name, Dict())
function (s::PrecisionScore)(x::confusion_matrix; name = nothing, num = nothing)
    s.num += 1
    s.eval_dict[(name == nothing ? s.name : name) * string(num == nothing ? s.num : num)] = precision_score(x, ith_class = s.ith_class, class_name = s.class_name, average = s.average, weights = s.weights, normalize = s.normalize)
end
(s::PrecisionScore)(y_true, y_pred; name = nothing, num = nothing) = s(confusion_matrix(y_true, y_pred), name = name, num = num)

PositivePredictiveValue(;average = "binary", ith_class = nothing, class_name = nothing, weights = nothing, normalize = false, name = "positive-predictive-value ", num = 0) =
PositivePredictiveValue(ith_class, class_name, average, weights, normalize, num, name, Dict())
function (s::PositivePredictiveValue)(x::confusion_matrix; name = nothing, num = nothing)
    s.num += 1
    s.eval_dict[(name == nothing ? s.name : name) * string(num == nothing ? s.num : num)] = positive_predictive_value(x, ith_class = s.ith_class, class_name = s.class_name, average = s.average, weights = s.weights, normalize = s.normalize)
end
(s::PositivePredictiveValue)(y_true, y_pred; name = nothing, num = nothing) = s(confusion_matrix(y_true, y_pred), name = name, num = num)

AccuracyScore(;average = "binary", ith_class = nothing, class_name = nothing, weights = nothing, normalize = false, name = "accuracy-score ", num = 0) =
AccuracyScore(ith_class, class_name, average, weights, normalize, num, name, Dict())
function (s::AccuracyScore)(x::confusion_matrix; name = nothing, num = nothing)
    s.num += 1
    s.eval_dict[(name == nothing ? s.name : name) * string(num == nothing ? s.num : num)] = accuracy_score(x, ith_class = s.ith_class, class_name = s.class_name, average = s.average, weights = s.weights, normalize = s.normalize)
end
(s::AccuracyScore)(y_true, y_pred; name = nothing, num = nothing) = s(confusion_matrix(y_true, y_pred), name = name, num = num)

BalancedAccuracyScore(;average = "binary", ith_class = nothing, class_name = nothing, weights = nothing, normalize = false, name = "balanced-accuracy-score ", num = 0) =
BalancedAccuracyScore(ith_class, class_name, average, weights, normalize, num, name, Dict())
function (s::AccuracyScore)(x::confusion_matrix; name = nothing, num = nothing)
    s.num += 1
    s.eval_dict[(name == nothing ? s.name : name) * string(num == nothing ? s.num : num)] = balanced_accuracy_score(x, ith_class = s.ith_class, class_name = s.class_name, average = s.average, weights = s.weights, normalize = s.normalize)
end
(s::BalancedAccuracyScore)(y_true, y_pred; name = nothing, num = nothing) = s(confusion_matrix(y_true, y_pred), name = name, num = num)

NegativePredictiveValue(;average = "binary", ith_class = nothing, class_name = nothing, weights = nothing, normalize = false, name = "negative-predictive-value ", num = 0) =
NegativePredictiveValue(ith_class, class_name, average, weights, normalize, num, name, Dict())
function (s::NegativePredictiveValue)(x::confusion_matrix; name = nothing, num = nothing)
    s.num += 1
    s.eval_dict[(name == nothing ? s.name : name) * string(num == nothing ? s.num : num)] = negative_predictive_value(x, ith_class = s.ith_class, class_name = s.class_name, average = s.average, weights = s.weights, normalize = s.normalize)
end
(s::NegativePredictiveValue)(y_true, y_pred; name = nothing, num = nothing) = s(confusion_matrix(y_true, y_pred), name = name, num = num)

FalseNegativeRate(;average = "binary", ith_class = nothing, class_name = nothing, weights = nothing, normalize = false, name = "false-negative-rate ", num = 0) =
FalseNegativeRate(ith_class, class_name, average, weights, normalize, num, name, Dict())
function (s::FalseNegativeRate)(x::confusion_matrix; name = nothing, num = nothing)
    s.num += 1
    s.eval_dict[(name == nothing ? s.name : name) * string(num == nothing ? s.num : num)] = false_negative_rate(x, ith_class = s.ith_class, class_name = s.class_name, average = s.average, weights = s.weights, normalize = s.normalize)
end
(s::FalseNegativeRate)(y_true, y_pred; name = nothing, num = nothing) = s(confusion_matrix(y_true, y_pred), name = name, num = num)

FalsePositiveRate(;average = "binary", ith_class = nothing, class_name = nothing, weights = nothing, normalize = false, name = "false-positive-rate ", num = 0) =
FalsePositiveRate(ith_class, class_name, average, weights, normalize, num, name, Dict())
function (s::FalsePositiveRate)(x::confusion_matrix; name = nothing, num = nothing)
    s.num += 1
    s.eval_dict[(name == nothing ? s.name : name) * string(num == nothing ? s.num : num)] = false_positive_rate(x, ith_class = s.ith_class, class_name = s.class_name, average = s.average, weights = s.weights, normalize = s.normalize)
end
(s::FalsePositiveRate)(y_true, y_pred; name = nothing, num = nothing) = s(confusion_matrix(y_true, y_pred), name = name, num = num)

FalseDiscoveryRate(;average = "binary", ith_class = nothing, class_name = nothing, weights = nothing, normalize = false, name = "false-discovery-rate ", num = 0) =
FalseDiscoveryRate(ith_class, class_name, average, weights, normalize, num, name, Dict())
function (s::FalseDiscoveryRate)(x::confusion_matrix; name = nothing, num = nothing)
    s.num += 1
    s.eval_dict[(name == nothing ? s.name : name) * string(num == nothing ? s.num : num)] = false_discovery_rate(x, ith_class = s.ith_class, class_name = s.class_name, average = s.average, weights = s.weights, normalize = s.normalize)
end
(s::FalseDiscoveryRate)(y_true, y_pred; name = nothing, num = nothing) = s(confusion_matrix(y_true, y_pred), name = name, num = num)

FalseOmissionRate(;average = "binary", ith_class = nothing, class_name = nothing, weights = nothing, normalize = false, name = "false-omission-rate ", num = 0) =
FalseOmissionRate(ith_class, class_name, average, weights, normalize, num, name, Dict())
function (s::FalseOmissionRate)(x::confusion_matrix; name = nothing, num = nothing)
    s.num += 1
    s.eval_dict[(name == nothing ? s.name : name) * string(num == nothing ? s.num : num)] = false_omission_rate(x, ith_class = s.ith_class, class_name = s.class_name, average = s.average, weights = s.weights, normalize = s.normalize)
end
(s::FalseOmissionRate)(y_true, y_pred; name = nothing, num = nothing) = s(confusion_matrix(y_true, y_pred), name = name, num = num)

F1Score(;average = "binary", ith_class = nothing, class_name = nothing, weights = nothing, normalize = false, name = "f1-score ", num = 0) =
F1Score(ith_class, class_name, average, weights, normalize, num, name, Dict())
function (s::F1Score)(x::confusion_matrix; name = nothing, num = nothing)
    s.num += 1
    s.eval_dict[(name == nothing ? s.name : name) * string(num == nothing ? s.num : num)] = f1_score(x, ith_class = s.ith_class, class_name = s.class_name, average = s.average, weights = s.weights, normalize = s.normalize)
end
(s::F1Score)(y_true, y_pred; name = nothing, num = nothing) = s(confusion_matrix(y_true, y_pred), name = name, num = num)

PrevalenceThreshold(;average = "binary", ith_class = nothing, class_name = nothing, weights = nothing, normalize = false, name = "prevalence-threshold ", num = 0) =
PrevalenceThreshold(ith_class, class_name, average, weights, normalize, num, name, Dict())
function (s::F1Score)(x::confusion_matrix; name = nothing, num = nothing)
    s.num += 1
    eval_dict[(name == nothing ? s.name : name) * string(num == nothing ? s.num : num)] = prevalence_threshold(x, ith_class = s.ith_class, class_name = s.class_name, average = s.average, weights = s.weights, normalize = s.normalize)
end
(s::PrevalenceThreshold)(y_true, y_pred; name = nothing, num = nothing) = s(confusion_matrix(y_true, y_pred), name = name, num = num)

ThreatScore(;average = "binary", ith_class = nothing, class_name = nothing, weights = nothing, normalize = false, name = "threat-score ", num = 0) =
ThreatScore(ith_class, class_name, average, weights, normalize, num, name, Dict())
function (s::ThreatScore)(x::confusion_matrix; name = nothing, num = nothing)
    s.num += 1
    s.eval_dict[(name == nothing ? s.name : name) * string(num == nothing ? s.num : num)] = threat_score(x, ith_class = s.ith_class, class_name = s.class_name, average = s.average, weights = s.weights, normalize = s.normalize)
end
(s::ThreatScore)(y_true, y_pred; name = nothing, num = nothing) = s(confusion_matrix(y_true, y_pred), name = name, num = num)

MatthewsCorrelationCoeff(;average = "binary", ith_class = nothing, class_name = nothing, weights = nothing, normalize = false, name = "matthews-correlation-coeff", num = 0) =
MatthewsCorrelationCoeff(ith_class, class_name, average, weights, normalize, num, name, Dict())
function (s::MatthewsCorrelationCoeff)(x::confusion_matrix; name = nothing, num = nothing)
    s.num += 1
    s.eval_dict[(name == nothing ? s.name : name) * string(num == nothing ? s.num : num)] = matthews_correlation_coeff(x, ith_class = s.ith_class, class_name = s.class_name, average = s.average, weights = s.weights, normalize = s.normalize)
end
(s::MatthewsCorrelationCoeff)(y_true, y_pred; name = nothing, num = nothing) = s(confusion_matrix(y_true, y_pred), name = name, num = num)

FowlkesMallowsIndex(;average = "binary", ith_class = nothing, class_name = nothing, weights = nothing, normalize = false, name ="folwkes-mallows-index ", num = 0) =
FowlkesMallowsIndex(ith_class, class_name, average, weights, normalize, num, name, Dict())
function (s::FowlkesMallowsIndex)(x::confusion_matrix; name = nothing, num = nothing)
    s.num += 1
    s.eval_dict[(name == nothing ? s.name : name) * string(num == nothing ? s.num : num)] = fowlkes_mallows_index(x, ith_class = s.ith_class, class_name = s.class_name, average = s.average, weights = s.weights, normalize = s.normalize)
end
(s::FowlkesMallowsIndex)(y_true, y_pred; name = nothing, num = nothing) = s(confusion_matrix(y_true, y_pred), name = name, num = num)

Informedness(;average = "binary", ith_class = nothing, class_name = nothing, weights = nothing, normalize = false, name = "informedness ", num = 0) =
Informedness(ith_class, class_name, average, weights, normalize, num, name, Dict())
function (s::Informedness)(x::confusion_matrix; name = nothing, num = nothing)
    s.num += 1
    s.eval_dict[(name == nothing ? s.name : name) * string(num == nothing ? s.num : num)] = informedness(x, ith_class = s.ith_class, class_name = s.class_name, average = s.average, weights = s.weights, normalize = s.normalize)
end
(s::Informedness)(y_true, y_pred; name = nothing, num = nothing) = s(confusion_matrix(y_true, y_pred), name = name, num = num)

Markedness(;average = "binary", ith_class = nothing, class_name = nothing, weights = nothing, normalize = false, name = "markedness " , num = 0) =
Markednes(ith_class, class_name, average, weights, normalize, num, name, Dict())
function (s::Markedness)(x::confusion_matrix; name = nothing, num = nothing)
    s.num += 1
    s.eval_dict[(name == nothing ? s.name : name) * string(num == nothing ? s.num : num)] = markedness(x, ith_class = s.ith_class, class_name = s.class_name, average = s.average, weights = s.weights, normalize = s.normalize)
end
(s::Markedness)(y_true, y_pred; name = nothing, num = nothing) = s(confusion_matrix(y_true, y_pred), name = name, num = num)

JaccardScore(;average = "binary", ith_class = nothing, class_name = nothing, weights = nothing, normalize = false, name = "jaccard-score ", num = 0) =
JaccardScore(ith_class, class_name, average, weights, normalize, num, name, Dict())
function (s::JaccardScore)(x::confusion_matrix; name = nothing, num = nothing)
    s.num += 1
    s.eval_dict[(name == nothing ? s.name : name) * string(num == nothing ? s.num : num)] = jaccard_score(x, ith_class = s.ith_class, class_name = s.class_name, average = s.average, weights = s.weights, normalize = s.normalize)
end
(s::JaccardScore)(y_true, y_pred; name = nothing, num = nothing) = s(confusion_matrix(y_true, y_pred), name = name, num = num)


##
# Visualization Tracker
