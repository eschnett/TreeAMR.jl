# The definitions of `ghost_tests.jl`, split off so that each test file runs
# on its own: every test process includes all of `helpers/` ("Parallel
# groups" in CODE.md).

# A group with every stencil marked not unit: the same transfer, computed
# through the weights.
function weighted(g::TreeAMR.TransferGroup{T,D}) where {T,D}
    stencils = map(s -> TreeAMR.Stencil1D{T}(s.targetfirst, s.srcstart, s.weights, false,
                                             s.single),
                   g.stencils)
    return TreeAMR.TransferGroup{T,D}(g.kind, stencils, g.targetblocks, g.sourceblocks,
                                      g.factorcol, g.orientation, g.plane)
end

# Every value a copy could meet that an arithmetic shortcut could change:
# signed zeros, NaN, infinities, subnormals, ordinary numbers.
function awkward_fill!(work::AbstractArray{T}) where {T}
    specials = T[-zero(T), zero(T), T(NaN), -T(Inf), floatmin(T) / 4, -floatmin(T) / 3]
    for (i, I) in enumerate(eachindex(work))
        work[I] = iszero(i % 7) ? specials[1 + (i ÷ 7) % length(specials)] :
                  T(sin(i))
    end
    return work
end

# A group with every column's `single` row cleared: the same transfer,
# computed as the full tensor-product sum over every weight.
function full_sum(g::TreeAMR.TransferGroup{T,D}) where {T,D}
    stencils = map(s -> TreeAMR.Stencil1D{T}(s.targetfirst, s.srcstart, s.weights, s.unit,
                                             zero(s.single)),
                   g.stencils)
    return TreeAMR.TransferGroup{T,D}(g.kind, stencils, g.targetblocks, g.sourceblocks,
                                      g.factorcol, g.orientation, g.plane, g.bounded)
end

# Finite data with what a dropped `±0` term could still change: signed
# zeros and subnormals, beside ordinary numbers of both signs.
function finite_fill!(rng, work::AbstractArray{T}) where {T}
    specials = T[-zero(T), zero(T), floatmin(T) / 4, -floatmin(T) / 3]
    for I in eachindex(work)
        work[I] = rand(rng) < 0.1 ? rand(rng, specials) : T(randn(rng))
    end
    return work
end
