module pairwise

using LinearAlgebra
using Statistics: mean
using ..utils

include("metrics.jl"); export minkowski_distance, euclidian_distance, manhattan_distance, chebyshev_distance, braycurtis_distance, canberra_distance, cityblock_distance, mahalanobis_distance,
correlation, cosine_distance, cosine_similarity

end
