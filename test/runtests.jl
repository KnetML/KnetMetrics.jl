using Test

@testset "KnetMetrics" begin
    include("classification-metrics.jl")
    include("regression-metrics.jl")
    include("pairwise-metrics.jl")
end
