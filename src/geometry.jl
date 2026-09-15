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

The cell size of every leaf, indexed by block — what a kernel needs to
scale a finite-difference stencil, since blocks at different levels have
different spacings.
"""
block_spacings(forest::Forest{D,R}, ::Type{T}=R) where {D,R,T} =
    T[spacing(T, forest, k) for k in forest.leaves]

"""
    block_origin([T], forest, k::MortonKey)

The physical position of the lower corner of block `k`'s interior (the
outer corner of its first interior cell, not that cell's center).
"""
function block_origin(::Type{T}, forest::Forest{D}, k::MortonKey{D}) where {T,D}
    rootpos = root_position(forest, k.root)
    scale = 1 << level(k)
    return ntuple(D) do d
        lo, hi = T(forest.extents[d][1]), T(forest.extents[d][2])
        rootwidth = (hi - lo) / forest.roots[d]
        lo + (rootpos[d] + T(k.coords[d]) / T(scale)) * rootwidth
    end
end
block_origin(forest::Forest{D,T}, k::MortonKey{D}) where {D,T} = block_origin(T, forest, k)

"""
    block_origins(forest, T=floattype(forest))

The lower corner of every leaf's interior, indexed by block — the
companion of [`block_spacings`](@ref), and what a kernel needs to turn
its cell index into a position without consulting the tree. Interior
cell `i` of block `b` is centred at
`origins[b][d] + (i - 1/2) * spacings[b]`.

Together the two arrays are the whole geometry a device-side kernel
sees: plain `isbits` arrays indexed by block, no keys and no forest.
"""
block_origins(forest::Forest{D,R}, ::Type{T}=R) where {D,R,T} =
    NTuple{D,T}[block_origin(T, forest, k) for k in forest.leaves]

"""
    block_extent([T], forest, k::MortonKey)

The `(lo, hi)` physical extent of block `k`'s interior, per dimension.
The leaves' extents tile the domain exactly.
"""
function block_extent(::Type{T}, forest::Forest{D}, k::MortonKey{D}) where {T,D}
    origin = block_origin(T, forest, k)
    width = spacing(T, forest, k) * forest.N
    return ntuple(d -> (origin[d], origin[d] + width), D)
end
block_extent(forest::Forest{D,T}, k::MortonKey{D}) where {D,T} = block_extent(T, forest, k)
