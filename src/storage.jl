"""
    FieldSet{T,D,A}

Block storage for `nvars` variables over every leaf of a
[`Forest`](@ref): one big persistent array holding all leaf blocks
including their ghosts,

    work :: A   # size (N+2G, ..., N+2G, nvars, nblocks)

Cell indices vary fastest (so a GPU reads them coalesced), then the
variable index, then the block index; blocks are ordered by the forest's
Morton key order, so `forest.leaves[b]` is the key of block `b`.

Element type `T` is generic — `Float64` by default, `Float32` for GPUs.

A field set is tied to the forest's *current* leaf array. Block indices
are deliberately not stable across regridding (M4), which compacts the
block slots and rebuilds the storage.

    FieldSet(forest, nvars)          # Float64
    FieldSet{Float32}(forest, nvars)

# Examples

```jldoctest
julia> forest = Forest((2,); N = 4, G = 1);

julia> fs = FieldSet(forest, 3);

julia> size(fs.work)
(6, 3, 2)
```
"""
mutable struct FieldSet{T,D,A<:AbstractArray{T}}
    const forest::Forest{D}
    const nvars::Int
    # Replaced wholesale by regridding, which compacts the block slots
    # into a freshly sized array. Mutable so that references an
    # application already holds stay valid across a regrid.
    work::A
end

function FieldSet{T}(forest::Forest{D}, nvars::Integer) where {T,D}
    nvars > 0 || throw(ArgumentError("nvars must be positive, got $nvars"))
    stored = forest.N + 2 * forest.G
    work = zeros(T, ntuple(_ -> stored, D)..., Int(nvars), nleaves(forest))
    return FieldSet{T,D,typeof(work)}(forest, Int(nvars), work)
end
FieldSet(forest::Forest, nvars::Integer) = FieldSet{Float64}(forest, nvars)

"""
    nblocks(fs::FieldSet)

The number of blocks stored — one per leaf of the underlying forest.
"""
nblocks(fs::FieldSet) = size(fs.work, ndims(fs.work))

"""
    blockkey(fs::FieldSet, b::Integer)

The [`MortonKey`](@ref) of block `b`.
"""
blockkey(fs::FieldSet, b::Integer) = fs.forest.leaves[b]

"""
    blockview(fs::FieldSet, b::Integer)
    blockview(fs::FieldSet, b::Integer, v::Integer)

A view of block `b` **including ghosts** — shape `(N+2G, ..., N+2G,
nvars)`, or `(N+2G, ..., N+2G)` for a single variable `v`.
"""
blockview(fs::FieldSet{T,D}, b::Integer) where {T,D} =
    view(fs.work, ntuple(_ -> Colon(), D + 1)..., b)
blockview(fs::FieldSet{T,D}, b::Integer, v::Integer) where {T,D} =
    view(fs.work, ntuple(_ -> Colon(), D)..., v, b)

"""
    interiorview(fs::FieldSet, b::Integer)
    interiorview(fs::FieldSet, b::Integer, v::Integer)

A view of block `b` **excluding ghosts** — shape `(N, ..., N, nvars)`,
or `(N, ..., N)` for a single variable `v`. This is the part that tiles
the domain and that the ODE state vector holds.
"""
function interiorview(fs::FieldSet{T,D}, b::Integer) where {T,D}
    forest = fs.forest
    inner = (forest.G + 1):(forest.G + forest.N)
    return view(fs.work, ntuple(_ -> inner, D)..., :, b)
end
function interiorview(fs::FieldSet{T,D}, b::Integer, v::Integer) where {T,D}
    forest = fs.forest
    inner = (forest.G + 1):(forest.G + forest.N)
    return view(fs.work, ntuple(_ -> inner, D)..., v, b)
end

"""
    fill_by_coordinates!(f, fs::FieldSet)

Set every interior cell of every block from the callback
`f(x, v) -> value`, where `x` is the cell center (an `NTuple{D,Float64}`,
see [`cell_center`](@ref)) and `v` the variable index. Ghosts are left
untouched — they are filled by the ghost exchange (M2).

Serial and allocation-light; the parallel and GPU versions come with the
KernelAbstractions kernels in M2 and M6.
"""
function fill_by_coordinates!(f, fs::FieldSet{T,D}) where {T,D}
    forest = fs.forest
    G, N = forest.G, forest.N
    for b in 1:nblocks(fs)
        k = blockkey(fs, b)
        for v in 1:fs.nvars
            block = blockview(fs, b, v)
            for idx in CartesianIndices(ntuple(_ -> (G + 1):(G + N), D))
                x = cell_center(forest, k, Tuple(idx))
                block[idx] = f(x, v)
            end
        end
    end
    return fs
end
