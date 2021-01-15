using Test
using KnetMetrics

@testset "classification-metrics" begin
    random_true = rand(2:6,1000)
    random_pred = rand(2:6,1000)
    random_labels = [2,3,4,5,6]
    random_confusion_matrix = confusion_matrix(random_true, random_pred, labels = random_labels)

    y_pred = [ 1, 4, 4, 1, 2, 3, 1, 3, 2, 1, 2, 2, 2, 1, 4, 4, 2, 1, 3, 2, 2, 3, 2, 1, 2, 3, 4, 1, 2, 1];
    y_true = [ 4, 4, 1, 4, 4, 2, 1, 1, 2, 4, 1, 3, 3, 3, 1, 1, 3, 1, 4, 4, 3, 3, 3, 1, 4, 1, 2, 3, 2, 2];
    x = confusion_matrix(y_true, y_pred)

    @testset "class-confusion" begin
        @test class_confusion(x,ith_class =2) == [2 9; 3 16]
        @test class_confusion(x,ith_class =2) == class_confusion(x, class_name = 2)
    end

    @testset "condition-positive" begin
        @test condition_positive(x) == [9, 5, 8, 8]
        @test condition_positive(x, ith_class = 2) == 5
        @test condition_positive(random_confusion_matrix, class_name = 3) == condition_positive(random_confusion_matrix, ith_class =2)
    end

    @testset "condition-negative" begin
        @test condition_negative(x) == [21, 25, 22, 22]
        @test condition_negative(x,ith_class = 3) == 22
        @test condition_negative(random_confusion_matrix, class_name = 3) == condition_negative(random_confusion_matrix, ith_class = 2)
    end

    @testset "sensitivity-recall-score" begin
        @test isapprox(sensitivity_score(x), 0.24583333333333335)
        @test sensitivity_score(random_confusion_matrix) == recall_score(random_confusion_matrix)
    end

    @testset "specificity-score" begin
        @test isapprox(specificity_score(x),0.7476623376623377)
    end

    @testset "precision-score" begin
        @test isapprox(precision_score(x), 0.22878787878787876)
    end

    @testset "accuracy-score" begin
        @test isapprox(accuracy_score(x), 0.22878787878787876)
        @test isapprox(accuracy_score(x, average = "binary"), [0.7040297388442665, 0.3840162211877817, 0.4224178433065599, 0.4224178433065599])
    end

    @testset "balanced-accuracy-score" begin
        @test isapprox(balanced_accuracy_score(x),0.24583333333333335)
    end

    @testset "negative-predictive-value" begin
        @test isapprox(negative_predictive_value(x), 0.7490977443609022)
    end

    @testset "false-positive-rate" begin
        @test isapprox(false_positive_rate(x), 0.25233766233766236)
    end

    @testset "false-negative-rate" begin
        @test isapprox(false_negative_rate(x),  0.7541666666666667)
    end

    @testset "false-discovery-rate" begin
        @test isapprox(false_discovery_rate(x) , 0.25233766233766236)
    end

    @testset "false-omission-rate" begin
        @test isapprox(false_omission_rate(x),0.25090225563909774)
    end

    @testset "f1-score" begin
        @test isapprox(f1_score(x),0.22275641025641024)
    end

    @testset "prevalence-threshold" begin
        @test isapprox(prevalence_threshold(x),0.515243503586089)
    end

    @testset "threat-score" begin
        @test isapprox(threat_score(x), 0.12738095238095237)
    end

    @testset "matthews-correlation-coefficient" begin
        @test isapprox(matthews_correlation_coeff(x),0.06582375105362859)
    end

    @testset "fowlkes-mallows-index" begin
        @test isapprox( fowlkes_mallows_index(x),0.6798605193558345)
    end

    @testset "informedness" begin
        @test isapprox(informedness(x) ,-0.006504329004328957)
    end

    @testset "markedness" begin
        @test isapprox(markedness(x), -0.022114376851218975)
    end

    @testset "hamming-loss" begin
        @test isapprox(hamming_loss(x), 0.7666666666666667)
    end

    @testset "jaccard-score" begin
        @test isapprox(jaccard_score(x),0.12738095238095237)
        @test isapprox(0.1320754716981132, jaccard_score(x, average = "micro"))
    end

    @testset "cohen-kappa-score" begin
        @test isapprox(-0.00877192982456143, cohen_kappa_score(x))
    end

end
