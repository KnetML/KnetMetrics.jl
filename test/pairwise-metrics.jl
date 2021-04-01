using Test
using KnetMetrics
using Random


@testset "pairwise-metrics" begin
    Random.seed!(42)
    u = rand(100)
    Random.seed!(123)
    v = rand(100)

    @testset "minkowski-distance" begin
        @test isapprox(minkowski_distance(u,v), 19.192858878774214)
    end

    @testset "minkowski-distance-p3" begin
        @test isapprox(minkowski_distance(u,v, p=3), 11.841710005717449)
    end

    @testset "euclidian-distance" begin
        @test isapprox(euclidian_distance(u,v), 19.192858878774214)
    end

    @testset "manhattan-distance" begin
        @test isapprox(manhattan_distance(u,v), 36.666029519718066)
    end

    @testset "chebyshev-distance" begin
        @test isapprox(chebyshev_distance(u,v), 0.8624100780860133)
    end

    @testset "braycurtis-distance" begin
        @test isapprox(braycurtis_distance(u,v), 0.3653169890723984)
    end

    @testset "canberra-distance" begin
        @test isapprox(canberra_distance(u,v), 32.09459846314555)
    end

    @testset "cityblock-distance" begin
        @test isapprox(cityblock_distance(u,v), 36.666029519718066)
    end

    @testset "correlation" begin
        @test isapprox(correlation(u,v), 1.1508533815000055)
    end

    @testset "cosine-similarity" begin
        @test isapprox(cosine_similarity(u,v), -0.1508533815000055)
    end

end
