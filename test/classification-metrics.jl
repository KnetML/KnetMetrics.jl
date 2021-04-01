using Test
using KnetMetrics
using Random

@testset "classification-metrics" begin
    Random.seed!(42)
    y_true = rand(1:3, 100)
    Random.seed!(123)
    y_pred = rand(1:3,100)
    c = confusion_matrix(y_true, y_pred, labels=[1,2,3])

    @testset "class-confusion" begin
        @test class_confusion(c,ith_class =2) == class_confusion(c, class_name = 2)
    end

    @testset "condition-positive" begin
        @test condition_positive(c) == [31, 34, 35]
        @test condition_positive(c, ith_class = 2) == 34
        @test condition_positive(c, class_name = 3) == condition_positive(c, ith_class =3)
    end

    @testset "condition-negative" begin
        @test condition_negative(c) == [69, 66, 65]
        @test condition_negative(c,ith_class = 3) == 65
    end

    @testset "predicted-positive" begin
        @test predicted_positive(c) == [28, 39, 33]
    end

    @testset "predicted-negative" begin
        @test predicted_negative(c) == [72, 61, 67]
    end

    @testset "correctly-classified" begin
        @test correctly_classified(c) == [57, 55, 56]
    end

    @testset "incorrectly-classified" begin
        @test incorrectly_classified(c) == [43, 45, 44]
    end

    @testset "sensitivity-score" begin
        @test isapprox(sensitivity_score(c), 0.33756212162284266)
    end

    @testset "recall-score" begin
        @test isapprox(recall_score(c), 0.33756212162284266)
    end

    @testset "specificity-score" begin
        @test isapprox(specificity_score(c), 0.6694267085571433)
    end

    @testset "precision-score" begin
        @test isapprox(precision_score(c), 0.3361083361083361)
    end

    @testset "accuracy-score" begin
        @test isapprox(accuracy_score(c), 0.3361083361083361)
    end

    @testset "balanced-accuracy-score" begin
        @test isapprox(balanced_accuracy_score(c), 0.33756212162284266)
    end

    @testset "negative-predictive-value" begin
        @test isapprox(negative_predictive_value(c), 0.6698010403356623)
    end

    @testset "false-negative-rate" begin
        @test isapprox(false_negative_rate(c), 0.6624378783771573)
    end

    @testset "false-positive-rate" begin
        @test isapprox(false_positive_rate(c), 0.3305732914428567)
    end

    @testset "fbeta-score" begin
        @test isapprox(fbeta_score(c), 0.33589642032805694)
    end

    @testset "prevalence-threshold" begin
        @test isapprox(prevalence_threshold(c), 0.4988856718477456)
    end

    @testset "threat-threshold" begin
        @test isapprox(threat_score(c), 0.20281219832565797)
    end

    @testset "matthews-correlation-coefficient" begin
        @test isapprox(matthews_correlation_coeff(c), 0.0705714539179035)
    end

    @testset "fowlkes-mallows-index" begin
        @test isapprox(fowlkes_mallows_index(c), 0.8186209097563695)
    end

    @testset "informedness" begin
        @test isapprox(informedness(c), 0.006988830179986009)
    end

    @testset "markedness" begin
        @test isapprox(markedness(c), 0.005909376443998397)
    end

    @testset "cohen-kappa-score" begin
        @test isapprox(cohen_kappa_score(c), 0.007668019846639673)
    end

    @testset "hamming-loss" begin
        @test isapprox(hamming_loss(c), 0.66)
    end

    @testset "jaccard-score" begin
        @test isapprox(jaccard_score(c), 0.20281219832565797)
    end

end
