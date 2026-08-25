# Independent, deliberately naive reference implementations. The point
# is that they share no logic with the package: the property tests
# compare the real (fast, bit-twiddling) implementations against
# brute-force geometry in exact rational arithmetic.

using TreeAMR: MortonKey, Forest, MAX_LEVEL, level, padded_coord

"""
Morton comparison by walking every bit plane from the top, most
significant dimension first — the obvious O(32·D) implementation that
`TreeAMR.isless` replaces with an O(D) most-significant-bit search.
"""
function naive_isless(a::MortonKey{D}, b::MortonKey{D}) where {D}
    a.root != b.root && return a.root < b.root
    pa = ntuple(d -> padded_coord(a.coords[d], a.level), D)
    pb = ntuple(d -> padded_coord(b.coords[d], b.level), D)
    for bitplane in (MAX_LEVEL - 1):-1:0
        mask = UInt64(1) << bitplane
        for d in 1:D
            ba, bb = pa[d] & mask, pb[d] & mask
            ba != bb && return ba < bb
        end
    end
    return a.level < b.level
end

"""
The box a leaf covers, in exact `Rational` units of root cells, as
`(lo, hi)`. Independent of the package's floating-point geometry.
"""
function leafbox(forest::Forest{D}, k::MortonKey{D}) where {D}
    rootpos = TreeAMR.root_position(forest, k.root)
    n = 1 << level(k)
    lo = ntuple(d -> rootpos[d] + Int(k.coords[d]) // n, D)
    hi = ntuple(d -> rootpos[d] + (Int(k.coords[d]) + 1) // n, D)
    return lo, hi
end

"""All brick translations that a periodic dimension identifies."""
periodic_shifts(forest::Forest{D}) where {D} =
    Iterators.product(ntuple(d -> forest.periodic[d] ?
                                  (-forest.roots[d], 0, forest.roots[d]) : (0,), D)...)

"""Whether two leaf boxes overlap in volume (touching does not count)."""
function overlaps(forest::Forest{D}, k1::MortonKey{D}, k2::MortonKey{D}) where {D}
    lo1, hi1 = leafbox(forest, k1)
    lo2, hi2 = leafbox(forest, k2)
    return all(d -> lo1[d] < hi2[d] && lo2[d] < hi1[d], 1:D)
end

"""
Whether two leaves touch — sharing a face, an edge, or just a corner
point — accounting for periodic wraparound. A leaf can be its own
neighbor when a periodic dimension has a single root.
"""
function adjacent(forest::Forest{D}, k1::MortonKey{D}, k2::MortonKey{D}) where {D}
    lo1, hi1 = leafbox(forest, k1)
    lo2, hi2 = leafbox(forest, k2)
    for shift in periodic_shifts(forest)
        # A block is not adjacent to itself under the identity shift.
        k1 == k2 && all(==(0), shift) && continue
        all(d -> lo1[d] <= hi2[d] + shift[d] && lo2[d] + shift[d] <= hi1[d], 1:D) && return true
    end
    return false
end

"""
Every direction in which `k2` touches `k1`: `-1`/`+1` where `k2` lies
entirely beyond `k1`'s low/high face in that dimension, `0` where their
extents overlap.

This is a *set*, not a single direction, because a periodic dimension
can make two blocks touch more than one way at once — a block wrapping
onto itself abuts both its own faces, and with a single root a
neighbor's far side wraps back around to the near side.
"""
function adjacency_directions(forest::Forest{D}, k1::MortonKey{D}, k2::MortonKey{D}) where {D}
    lo1, hi1 = leafbox(forest, k1)
    lo2, hi2 = leafbox(forest, k2)
    dirs = Set{NTuple{D,Int}}()
    for shift in periodic_shifts(forest)
        k1 == k2 && all(==(0), shift) && continue
        l2 = ntuple(d -> lo2[d] + shift[d], D)
        h2 = ntuple(d -> hi2[d] + shift[d], D)
        all(d -> lo1[d] <= h2[d] && l2[d] <= hi1[d], 1:D) || continue
        push!(dirs, ntuple(D) do d
            h2[d] <= lo1[d] ? -1 : l2[d] >= hi1[d] ? 1 : 0
        end)
    end
    return dirs
end

"""
A random forest built by repeatedly refining and coarsening at random.
Deliberately left unbalanced, so `balance!` has real work to do.
"""
function random_forest(rng, ::Val{D}; nsteps, maxlvl, maxroot=2) where {D}
    roots = ntuple(_ -> rand(rng, 1:maxroot), D)
    periodic = ntuple(_ -> rand(rng, Bool), D)
    forest = Forest(roots; N=4, G=1, periodic=periodic)
    for _ in 1:nsteps
        k = rand(rng, forest.leaves)
        if level(k) < maxlvl && rand(rng) < 0.75
            refine!(forest, k)
        elseif level(k) > 0
            p = parentkey(k)
            all(c -> isleaf(forest, c), childkeys(p)) && coarsen!(forest, p)
        end
    end
    return forest
end
