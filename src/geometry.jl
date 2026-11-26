# Physical geometry of the forest: mapping keys and cell indices to
# coordinates. Blocks are cubes with uniform spacing, halving per level.
#
# Every function here takes an optional leading element type and does its
# arithmetic *in* that type from the first operation, defaulting to the
# forest's own `floattype`. That is the point of the parameter: a
# `Float64` intermediate narrowed at the end would still need hardware
# fp64, which is exactly what a device may not have.

"""
    root_spacing([T], forest)

The physical width of one cell of a root-level block — the coarsest cell
size in the hierarchy. Isotropic by construction.

As everywhere in this file, the optional leading `T` is the type the
arithmetic is done in; it defaults to [`floattype`](@ref)`(forest)`.
"""
root_spacing(::Type{T}, forest::Forest) where {T} =
    (T(forest.extents[1][2]) - T(forest.extents[1][1])) / forest.roots[1] / forest.N
root_spacing(forest::Forest{D,T}) where {D,T} = root_spacing(T, forest)

"""
    spacing([T], forest, level::Integer)
    spacing([T], forest, k::MortonKey)

The physical cell size at the given refinement level, halving per level.
"""
spacing(::Type{T}, forest::Forest, lvl::Integer) where {T} =
    root_spacing(T, forest) / (1 << Int(lvl))
spacing(forest::Forest{D,T}, lvl::Integer) where {D,T} = spacing(T, forest, lvl)
spacing(::Type{T}, forest::Forest{D}, k::MortonKey{D}) where {T,D} =
    spacing(T, forest, level(k))
spacing(forest::Forest{D,T}, k::MortonKey{D}) where {D,T} = spacing(T, forest, k)

"""
    minimum_spacing([T], forest)

The cell size of the finest leaf present. This is what sets the global
timestep: with no subcycling, the whole hierarchy advances at the CFL
limit of its finest cells.
"""
minimum_spacing(::Type{T}, forest::Forest) where {T} =
    spacing(T, forest, maxlevel(forest))
minimum_spacing(forest::Forest{D,T}) where {D,T} = minimum_spacing(T, forest)

"""
    block_spacings(forest, T=floattype(forest))

The cell size of every block this rank stores, indexed by local block —
what a kernel needs to scale a finite-difference stencil, since blocks
at different levels have different spacings. Serially that is every
leaf; over a distributed forest it is the leaves of
[`blockrange`](@ref), so that entry `b` belongs to block `b` of every
[`FieldSet`](@ref) over `forest`.
"""
block_spacings(forest::Forest{D,R}, ::Type{T}=R) where {D,R,T} =
    T[spacing(T, forest, k) for k in localleaves(forest)]

# The leaves whose blocks this rank stores, in local block order.
localleaves(forest::Forest) = view(forest.leaves, blockrange(forest))

# The integer corner of block `k` along each dimension: the number of
# spacings of its level from its patch's low face to its low face, `g·N`
# for `g` its integer coordinate at its level within the patch. An `Int`,
# since `MAX_LEVEL` levels below a few roots overflow 32 bits. Positions
# are formed from it by `nodeposition` (CODE.md, "Positions" under
# "Centerings").
function block_base(forest::Forest{D}, k::MortonKey{D}) where {D}
    rootpos = root_position(forest, k.root)
    lvl = level(k)
    return ntuple(d -> ((rootpos[d] << lvl) + Int(k.coords[d])) * forest.N, D)
end

# The low corner of block `k`'s patch, in `T`: the forest's own extents'
# low corner on a forest of one brick.
function block_low(::Type{T}, forest::Forest{D}, k::MortonKey{D}) where {T,D}
    extents = patch_extents(forest, patch(forest, k))
    return ntuple(d -> T(extents[d][1]), D)
end

# The box points of patch `p` are located in, in `T`: from its low corner
# to the larger of the high corner it was given and its high face as
# positions form it, `lo + K·h` for `K` its spacings across at level 0
# (the same bits at every level). The two differ by rounding: one cell
# size serves every dimension and every patch, and the extents need only
# agree with it to `check_brick`'s tolerance. The larger one, so that
# every point `coordinates` returns is located, and every point of the
# box the forest was given still is (CODE.md, "Positions" under
# "Centerings").
function location_box(::Type{T}, forest::Forest{D}, p::Integer) where {T,D}
    extents, roots = patch_extents(forest, p), patch_roots(forest, p)
    h = spacing(T, forest, 0)
    return ntuple(D) do d
        lo, hi = T(extents[d][1]), T(extents[d][2])
        top = nodeposition(lo, h, 0, roots[d] * forest.N + 1, 1, nothing)
        return (lo, max(hi, top))
    end
end

"""
    block_origin([T], forest, k::MortonKey)

The physical position of the lower corner of block `k`'s interior (the
outer corner of its first interior cell, not that cell's center). On a
forest of patches (M17) it is in the logical coordinates of `k`'s patch.

It is a node's position, formed by [`nodeposition`](@ref
TreeAMR.nodeposition) as `lo + B·h` from the patch's low corner `lo`,
the block's integer corner `B` and its level's spacing `h`, so that it
is bit for bit the corner the blocks around it see and the first vertex
`coordinates` gives.
"""
function block_origin(::Type{T}, forest::Forest{D}, k::MortonKey{D}) where {T,D}
    lo, B, h = block_low(T, forest, k), block_base(forest, k), spacing(T, forest, k)
    return ntuple(d -> nodeposition(lo[d], h, B[d], 1, 1, nothing), D)
end
block_origin(forest::Forest{D,T}, k::MortonKey{D}) where {D,T} = block_origin(T, forest, k)

"""
    block_origins(forest, T=floattype(forest))

The lower corner of the interior of every block this rank stores,
indexed by local block — the companion of [`block_spacings`](@ref), for
what in a kernel is not a point's position: a point is placed through
[`nodeposition`](@ref TreeAMR.nodeposition) from [`block_lows`](@ref)
and [`block_bases`](@ref) instead, which puts every copy of a point that
several blocks hold at one position bit for bit. The corners themselves
are formed by the same rule, so `origins[b]` is the position of block
`b`'s first vertex.

Together these arrays are the whole geometry a device-side kernel
sees: plain `isbits` arrays indexed by block, no keys and no forest.
"""
block_origins(forest::Forest{D,R}, ::Type{T}=R) where {D,R,T} =
    NTuple{D,T}[block_origin(T, forest, k) for k in localleaves(forest)]

"""
    block_lows(forest, T=floattype(forest))

The low corner of the patch (M17) of every block this rank stores, as an
`NTuple{D,T}` indexed by local block: the forest's own low corner on a
forest of one brick. With [`block_bases`](@ref) and
[`block_spacings`](@ref) it is what [`nodeposition`](@ref
TreeAMR.nodeposition) forms a point's position from, `lo + k·h` with `k`
an integer, and why every block holding a point puts it at the same
position bit for bit. A number per block rather than per patch, so that
a kernel reads it as it reads the spacing.
"""
block_lows(forest::Forest{D,R}, ::Type{T}=R) where {D,R,T} =
    NTuple{D,T}[block_low(T, forest, k) for k in localleaves(forest)]

"""
    block_bases(forest)

The integer corner of every block this rank stores, as an
`NTuple{D,Int}` indexed by local block: along each dimension the number
of spacings of the block's level from its patch's low face to the
block's low face. Owned index `I` of block `b` has the integer
coordinate `bases[b][d] + I − 1` there, which every block holding the
point agrees on and which [`nodeposition`](@ref TreeAMR.nodeposition)
turns into its position. `Int` on every backend, since a deep forest's
coordinates overflow 32 bits.
"""
block_bases(forest::Forest{D}) where {D} =
    NTuple{D,Int}[block_base(forest, k) for k in localleaves(forest)]

"""
    block_patches(forest)

The patch (M17; see [`patch`](@ref)) of every block this rank stores, as
`Int32`, indexed by local block — the companion of
[`block_origins`](@ref) and [`block_spacings`](@ref) for a forest of
patches, whose extents may overlap or coincide: a block's origin then
does not say which patch it is in, and so which of the application's
maps it is under, and this does. All ones on a forest of one brick.
"""
block_patches(forest::Forest) = Int32[patch(forest, k) for k in localleaves(forest)]

"""
    block_extent([T], forest, k::MortonKey)

The `(lo, hi)` physical extent of block `k`'s interior, per dimension.
Both ends are node positions ([`nodeposition`](@ref
TreeAMR.nodeposition)), so the leaves' extents tile the domain bit for
bit: a block's `hi` is its neighbour's `lo`, at one level or across two.
The domain's high face is then at `lo + K·h` for `K` the number of
spacings across it, which is the extent the forest was given up to
rounding (CODE.md, "Positions" under "Centerings").
"""
function block_extent(::Type{T}, forest::Forest{D}, k::MortonKey{D}) where {T,D}
    lo, B, h = block_low(T, forest, k), block_base(forest, k), spacing(T, forest, k)
    N = forest.N
    return ntuple(d -> (nodeposition(lo[d], h, B[d], 1, 1, nothing),
                        nodeposition(lo[d], h, B[d], N + 1, 1, nothing)), D)
end
block_extent(forest::Forest{D,T}, k::MortonKey{D}) where {D,T} = block_extent(T, forest, k)
