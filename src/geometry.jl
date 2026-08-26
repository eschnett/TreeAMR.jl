# Physical geometry of the forest: mapping keys and cell indices to
# coordinates. Blocks are cubes with uniform spacing, halving per level.

"""
    root_spacing(forest)

The physical width of one cell of a root-level block — the coarsest cell
size in the hierarchy. Isotropic by construction.
"""
root_spacing(forest::Forest) =
    (forest.extents[1][2] - forest.extents[1][1]) / forest.roots[1] / forest.N

"""
    spacing(forest, level::Integer)
    spacing(forest, k::MortonKey)

The physical cell size at the given refinement level, halving per level.
"""
spacing(forest::Forest, lvl::Integer) = root_spacing(forest) / (1 << Int(lvl))
spacing(forest::Forest{D}, k::MortonKey{D}) where {D} = spacing(forest, level(k))

"""
    minimum_spacing(forest)

The cell size of the finest leaf present. This is what sets the global
timestep: with no subcycling, the whole hierarchy advances at the CFL
limit of its finest cells.
"""
minimum_spacing(forest::Forest) = spacing(forest, maxlevel(forest))

"""
    block_spacings(forest, T=Float64)

The cell size of every leaf, indexed by block — what a kernel needs to
scale a finite-difference stencil, since blocks at different levels have
different spacings.
"""
block_spacings(forest::Forest, ::Type{T}=Float64) where {T} =
    T[spacing(forest, k) for k in forest.leaves]

"""
    block_origin(forest, k::MortonKey)

The physical position of the lower corner of block `k`'s interior (the
outer corner of its first interior cell, not that cell's center).
"""
function block_origin(forest::Forest{D}, k::MortonKey{D}) where {D}
    rootpos = root_position(forest, k.root)
    scale = 1 << level(k)
    return ntuple(D) do d
        lo, hi = forest.extents[d]
        rootwidth = (hi - lo) / forest.roots[d]
        lo + (rootpos[d] + k.coords[d] / scale) * rootwidth
    end
end

"""
    block_extent(forest, k::MortonKey)

The `(lo, hi)` physical extent of block `k`'s interior, per dimension.
The leaves' extents tile the domain exactly.
"""
function block_extent(forest::Forest{D}, k::MortonKey{D}) where {D}
    origin = block_origin(forest, k)
    width = spacing(forest, k) * forest.N
    return ntuple(d -> (origin[d], origin[d] + width), D)
end

"""
    cell_center(forest, k::MortonKey, idx::NTuple{D,Integer})

The physical position of the center of cell `idx` of block `k`. `idx` is
1-based over the *stored* array, so interior cells are `G+1:G+N` and
values outside that range name ghost cells (whose coordinates are still
well defined, and lie outside the block).
"""
function cell_center(forest::Forest{D}, k::MortonKey{D}, idx::NTuple{D,<:Integer}) where {D}
    origin = block_origin(forest, k)
    h = spacing(forest, k)
    return ntuple(d -> origin[d] + (Int(idx[d]) - forest.G - 0.5) * h, D)
end
