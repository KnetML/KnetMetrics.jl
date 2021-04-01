using Test
using KnetMetrics
using Random


@testset "regression-metrics" begin

    Random.seed!(42)
    u = rand(100)
    Random.seed!(123)
    v = rand(100)

    @testset "max-error" begin
        @test isapprox(max_error(u,v), 0.8624100780860133)
    end

    @testset "max-error" begin
        @test isapprox(max_error(u,v), 0.8624100780860133)
    end

    @testset "mean-absolute-error" begin
        @test isapprox(mean_absolute_error(u,v), 0.3666602951971807)
    end

    @testset "mean-squared-error" begin
        @test isapprox(mean_squared_error(u,v), 0.3666602951971807)
    end

    @testset "mean-squared-log-error" begin
        @test isapprox(mean_squared_log_error(u,v), 0.24971305530593674)
    end

    @testset "median-absolute-error" begin
        @test isapprox(median_absolute_error(u,v), 0.3183007014762824)
    end

    @testset "mean-absolute-percentage-error" begin
        @test isapprox(mean_absolute_percentage_error(u,v), 5.329695963484449)
    end


end
