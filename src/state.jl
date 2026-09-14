# Coupling to ODE integrators.
#
# The state vector holds leaf *interiors only*: it is what an integrator
# sees, and it never contains ghosts. The working array is scratch,
# refreshed at every RHS evaluation. So a right-hand side reads
#
#     scatter!(fs, u)          # flat vector -> working array interiors
#     fill_ghosts!(fs, sched)  # copies, restrictions, prolongations
#     map_blocks!(...)         # application kernels, writing du directly
#
# The integrator never sees ghosts and the RHS never mutates `u`. The
# cost is one scatter per evaluation, which `CODE.md` accepts in exchange
# for not spending integrator bandwidth on ghost memory.

"""
    statelength(fs::FieldSet)

Number of entries in this field set's state vector: `N^D · nvars ·
nblocks`, counting interior cells only.
"""
statelength(fs::FieldSet{T,D}) where {T,D} = fs.forest.N^D * fs.nvars * nblocks(fs)

"""
    statevector(fs::FieldSet)

A freshly allocated, zeroed state vector for `fs` — the `u` an ODE
integrator advances. Use [`gather!`](@ref) to load the field set's
current interior values into it.
"""
statevector(fs::FieldSet{T}) where {T} = zeros(T, statelength(fs))

"""
    statearray(u, fs::FieldSet)

View a flat state vector as an `(N, ..., N, nvars, nblocks)` array,
sharing memory. This is the layout application kernels write `du` in;
the index order matches the working array, minus the ghosts.
"""
function statearray(u::AbstractVector, fs::FieldSet{T,D}) where {T,D}
    length(u) == statelength(fs) || throw(DimensionMismatch(
        "state vector has $(length(u)) entries but this field set needs " *
        "$(statelength(fs))"))
    return reshape(u, ntuple(_ -> fs.forest.N, D)..., fs.nvars, nblocks(fs))
end

@kernel function scatter_kernel!(work, @Const(state), ::Val{D}, ::Val{G}) where {D,G}
    I = @index(Global, NTuple)                     # (i1..iD, var, block)
    work[ntuple(d -> I[d] + G, Val(D))..., I[D + 1], I[D + 2]] = state[I...]
end

@kernel function gather_kernel!(state, @Const(work), ::Val{D}, ::Val{G}) where {D,G}
    I = @index(Global, NTuple)
    state[I...] = work[ntuple(d -> I[d] + G, Val(D))..., I[D + 1], I[D + 2]]
end

function run_over_interiors!(kernel, fs::FieldSet{T,D}, a, b) where {T,D}
    backend = get_backend(fs.work)
    N, G = fs.forest.N, fs.forest.G
    kernel(backend)(a, b, Val(D), Val(G);
                    ndrange=(ntuple(_ -> N, D)..., fs.nvars, nblocks(fs)))
    synchronize(backend)
    return nothing
end

"""
    scatter!(fs::FieldSet, u::AbstractVector)

Copy a state vector into the working array's interior cells, leaving
ghosts untouched. The first step of every RHS evaluation; follow it with
[`fill_ghosts!`](@ref).
"""
function scatter!(fs::FieldSet{T,D}, u::AbstractVector) where {T,D}
    run_over_interiors!(scatter_kernel!, fs, fs.work, statearray(u, fs))
    return fs
end

"""
    gather!(u::AbstractVector, fs::FieldSet)

Copy the working array's interior cells into a state vector — the
inverse of [`scatter!`](@ref).

An application's RHS kernels normally write `du` in state layout
directly, fusing this away; `gather!` is for setting up initial data and
for reading results back out.
"""
function gather!(u::AbstractVector, fs::FieldSet{T,D}) where {T,D}
    run_over_interiors!(gather_kernel!, fs, statearray(u, fs), fs.work)
    return u
end

"""
    map_blocks!(kernel!, fs::FieldSet, args...)

Launch a KernelAbstractions kernel over every interior cell of every
block, with `ndrange = (N, ..., N, nblocks)`. The kernel's global index
is therefore `(i1, ..., iD, b)` with each `i` running over `1:N`; add
`G` to reach the working array's stored indices.

Blocks are uniform work units, so this is one flat parallel loop. The
CPU backend spreads it over `Threads.nthreads()` as it stands (M5), and
the same launch runs on a device in M6; every work item writes its own
output cell, so the result does not depend on how the loop was split.

```julia
@kernel function rhs!(du, @Const(work), @Const(h), ::Val{D}, ::Val{G}) where {D,G}
    I = @index(Global, NTuple)
    b = I[D + 1]
    c = ntuple(d -> I[d] + G, Val(D))          # stored (ghosted) index
    du[ntuple(d -> I[d], Val(D))..., 1, b] = work[c..., 2, b]
end
```
"""
function map_blocks!(kernel!, fs::FieldSet{T,D}, args...) where {T,D}
    backend = get_backend(fs.work)
    kernel!(backend)(args...;
                     ndrange=(ntuple(_ -> fs.forest.N, D)..., nblocks(fs)))
    synchronize(backend)
    return nothing
end

"""
    volume_weighted_norm(fs::FieldSet, u::AbstractVector; p=2)

The `p`-norm of a state vector with each cell weighted by its volume,
normalized by the domain volume — so the result is a resolution
independent RMS (or, for `p = Inf`, the plain maximum).

Weighting matters on an adaptive mesh: refined regions contribute more
*entries* per unit volume simply for being refined, so an unweighted
norm silently emphasizes them. This is also the shape an adaptive
integrator's `internalnorm` needs; through M3 only fixed-`dt`
integrators are exercised, so it is used here for error measurement.

Threaded over blocks, with the per-block partials combined in block
order, so the value does not depend on the thread count.
"""
function volume_weighted_norm(fs::FieldSet{T,D}, u::AbstractVector; p::Real=2) where {T,D}
    state = statearray(u, fs)
    forest = fs.forest
    colons = ntuple(_ -> Colon(), D)
    R = float(real(T))

    # One partial per block, then a serial pass over them in block
    # order: threaded, and bit-for-bit independent of the thread count,
    # which a running total split across tasks would not be.
    partials = Vector{R}(undef, nblocks(fs))

    if isinf(p)
        threaded_foreach(nblocks(fs)) do b
            block = view(state, colons..., :, b)
            partials[b] = isempty(block) ? zero(R) : maximum(abs, block)
        end
        return isempty(partials) ? zero(R) : maximum(partials)
    end

    volumes = Vector{R}(undef, nblocks(fs))
    threaded_foreach(nblocks(fs)) do b
        block = view(state, colons..., :, b)
        cellvolume = spacing(R, forest, blockkey(fs, b))^D
        partials[b] = cellvolume * sum(x -> abs(x)^p, block)
        volumes[b] = cellvolume * length(block)
    end

    volume = sum(volumes)
    volume == 0 && return zero(R)
    # `inv(R(p))`, not `1 / p`: the latter is a Float64 exponent, which
    # promotes the whole result to Float64 and made this function return a
    # different type from the `isinf(p)` branch above.
    return (sum(partials) / volume)^inv(R(p))
end
