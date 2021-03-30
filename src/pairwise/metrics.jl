
export minkowski_distance, euclidian_distance, manhattan_distance, chebyshev_distance, braycurtis_distance, canberra_distance, cityblock_distance, mahalanobis_distance,
correlation, cosine_distance, cosine_similarity

using LinearAlgebra
using Statistics: mean

#TODO
# scikitlearn-like kernels ?

# Distance methods

function minkowski_distance(u,v; p = 2, w = nothing, rooted = false)
    _validate_distance_input(u, v, w; p = p, p_is_used = true)
    val = abs.(u .- v)
    val = w != nothing ? val .* w : val
    if p == 1
        val = sum(val)
    elseif p == Inf
        val = maximum(val)
    else
        val = sum(val .^ p)
    end
    return rooted ? val .^ (1 / p) : val
end

minkowski_distance(x::Tuple; p = 2, w = nothing) = minkowski_distance(x...; p = p, w = w)

euclidian_distance(u,v; w = nothing, squared = false) = minkowski_distance(u,v; w = w, p = 2, rooted = squared)
euclidian_distance(x::Tuple;w = nothing, squared = false) = euclidian_distance(x...; w = w, rooted = squared)

manhattan_distance(u,v; w = nothing) = minkowski_distance(u,v; w = w, p = 1)
manhattan_distance(x::Tuple;w = nothing) = manhattan_distance(x...; w = w)

chebyshev_distance(u,v; w = nothing) = minkowski_distance(u,v; w = w, p = Inf, rooted=false)
chebyshev_distance(x::Tuple;w = nothing) = chebyshev_distance(x...; w = w)

function braycurtis_distance(u,v; w = nothing)
    _validate_distance_input(u, v, w)
    difference = abs.(u .- v)
    summation = abs.(u .+ v)
    if w != nothing
        difference = w .* difference
        summation = w .* summation
    end
    sum(difference) / sum(summation)
end

braycurtis_distance(x::Tuple; w = nothing) = braycurtis_distance(x...; w = w)

function canberra_distance(u,v; w = nothing)
    _validate_distance_input(u, v, w)
    difference = abs.(u .- v)
    result = difference / (abs.(u) .+ abs.(v))
    if w != nothing
        result = result .* w
    end
    return  sum(result[result .!== Inf])
end
canberra_distance(x::Tuple; w = nothing) = canberra_distance(x...; w = w)

function cityblock_distance(u,v; w = nothing)
    _validate_distance_input(u,v,w)
    result = abs.(u .- v)
    if w != nothing
        result = result .* w
    end
    return sum(result)
end
cityblock_distance(x; w = nothing) = cityblock_distance(x...; w = w)

function mahalanobis_distance(u,v, Vinv)
    _validate_distance_input(u,v,nothing)
    delta = u .- v
    return sqrt.(dot(dot(delta, convert_1d(Vinv)), delta))
end

mahalanobis_distance(x::Tuple, Vinv) = mahalanobis_distance(x..., Vinv)

##

function correlation(u,v; w = nothing, centered = true)
    _validate_distance_input(u,v,w)
    w = w != nothing ? w : ones(length(u))
    _u = convert_1d(u)
    _v = convert_1d(v)
    if centered
        umu = mean(_u .* w)
        vmu = mean(_v .* w)
        _u = _u .- umu
        _v = _v .- vmu
    end
    uv = mean(_u .* _v .* w)
    uu = mean( (_u .^ 2) .* w )
    vv = mean( (_v .^ 2) .* w  )
    return abs(1 - uv / sqrt(uu*vv))
end

cosine_distance(u,v;w=nothing) = correlation(u,v; w = w)
cosine_similarity(u,v; w = nothing) = -cosine_distance(u,v; w = w) + 1
