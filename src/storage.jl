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

@kernel function coordinates_kernel!(work, f, @Const(origins), @Const(spacings),
                                     ::Val{D}, ::Val{G}) where {D,G}
    I = @index(Global, NTuple)                     # (i1..iD, var, block)
    v, b = I[D + 1], I[D + 2]
    origin, h = origins[b], spacings[b]
    # The stored index of interior cell i is i + G, so `cell_center`'s
    # (idx - G - 1/2) is just (i - 1/2) — the same arithmetic, in the
    # same order, so this reproduces `cell_center` bit for bit.
    x = ntuple(d -> origin[d] + (I[d] - 0.5) * h, Val(D))
    work[ntuple(d -> I[d] + G, Val(D))..., v, b] = f(x, v)
end

"""
    fill_by_coordinates!(f, fs::FieldSet)

Set every interior cell of every block from the callback
`f(x, v) -> value`, where `x` is the cell center (an `NTuple{D,Float64}`,
see [`cell_center`](@ref)) and `v` the variable index. Ghosts are left
untouched — they are filled by the ghost exchange (M2).

`f` is called once per cell from a KernelAbstractions kernel, so it runs
concurrently across blocks (M5) and must be a pure function of its
arguments. The tree is not consulted: the kernel gets the geometry as
the two plain per-block arrays [`block_origins`](@ref) and
[`block_spacings`](@ref), which is also what makes it a device kernel in
M6.
"""
function fill_by_coordinates!(f, fs::FieldSet{T,D}) where {T,D}
    forest = fs.forest
    backend = get_backend(fs.work)
    coordinates_kernel!(backend)(fs.work, f, block_origins(forest),
                                 block_spacings(forest), Val(D), Val(forest.G);
                                 ndrange=(ntuple(_ -> forest.N, D)..., fs.nvars,
                                          nblocks(fs)))
    synchronize(backend)
    return fs
end

@kernel function zero_kernel!(work)
    I = @index(Global, NTuple)
    work[I...] = zero(eltype(work))
end

# Zeroing fresh block storage through a kernel rather than `fill!` is
# not about speed: on a multi-socket node it is the *first touch* that
# decides which NUMA domain each page lands in, and a serial `fill!`
# would park the whole array on whichever domain the driver thread sits
# on. The kernel touches each block from the same chunk of the ndrange
# that will later compute on it.
function zerofill!(work, backend)
    zero_kernel!(backend)(work; ndrange=size(work))
    synchronize(backend)
    return work
end
