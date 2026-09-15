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

Allocated on the field set's own backend, so a device-resident field set
gets a device-resident state vector and the integrator never touches
host memory (M6).
"""
function statevector(fs::FieldSet{T}) where {T}
    backend = get_backend(fs.work)
    u = allocate(backend, T, (statelength(fs),))
    return zerofill!(u, backend)
end

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

# Per-block reductions.
#
# Every diagnostic in the package has the same shape: one value per
# block, then a serial pass over those values *in block order*. That
# ordering is what makes the result bit-for-bit independent of the
# thread count (M5), and it is preserved here — only how the per-block
# values are produced changes with the backend.
#
# On the CPU they are threaded host reductions over per-block views,
# which is what M5 measured and what the recorded numbers were taken
# with. On a device that same formulation would be one kernel launch and
# one device-to-host synchronization *per block*, issued from several
# host tasks at once — so there it becomes a single launch with one work
# item per block, each looping over its own cells. Every work item owns
# its output slot, so this is deterministic by construction: the same
# discipline as everywhere else.
#
# The two paths share one specification of the reduction, `(f, op,
# init)`. They did not always: an earlier version passed a host
# block-reducer *and* a kernel-form fold, and nothing checked that the
# two agreed. `sum` over a block view is a sequential `mapfoldl` when
# the view is `IndexCartesian` but a *pairwise* one when it is
# `IndexLinear` — which `D = 1` with a scalar `vars` actually is. They
# agreed for every shape the package itself passed, by accident of
# `SubArray`'s `viewindexing` and nothing more (amended in M6).
#
# `array` is either the working array (offset `g = G`, so ghosts are
# skipped) or the state array (`g = 0`). Neither helper below works that
# out for itself; `block_mapreduce` does, which is why it and not these
# is what an application calls.
@kernel function block_reduce_kernel!(values, @Const(array), f, op, init,
                                      firstvar::Int, lastvar::Int,
                                      ::Val{D}, ::Val{G}, ::Val{N}) where {D,G,N}
    b = @index(Global)
    acc = init
    for v in firstvar:lastvar
        for c in CartesianIndices(ntuple(_ -> N, Val(D)))
            acc = op(acc, f(array[ntuple(d -> Tuple(c)[d] + G, Val(D))..., v, b]))
        end
    end
    values[b] = acc
end

# The device path: one launch, one work item per block, one copy back.
function _block_mapreduce_device(f, op, init::R, array, fs::FieldSet{T,D},
                                 backend::Backend, g::Int,
                                 vars::UnitRange{Int}) where {R,T,D}
    n = nblocks(fs)
    values = allocate(backend, R, (n,))
    block_reduce_kernel!(backend)(values, array, f, op, init,
                                  first(vars), last(vars),
                                  Val(D), Val(g), Val(fs.forest.N); ndrange=n)
    synchronize(backend)
    return tohost(values)
end

# The host path. It is split out under its own name rather than
# dispatched on `::CPU` so that both paths stay reachable on a machine
# with no device, which is what lets the suite check that the two
# compute the same fold.
function _block_mapreduce_host(f, op, init::R, array, fs::FieldSet{T,D}, g::Int,
                               vars::UnitRange{Int}) where {R,T,D}
    N = fs.forest.N
    inner = ntuple(_ -> (g + 1):(g + N), D)
    values = Vector{R}(undef, nblocks(fs))
    threaded_foreach(nblocks(fs)) do b
        values[b] = mapreduce(f, op, view(array, inner..., vars, b); init=init)
    end
    return values
end

# `vars` ends up as a pair of kernel arguments, so it has to name a
# contiguous run of variables: there is no way to hand a device an
# arbitrary index vector cell by cell, and silently reducing
# `first:last` instead would be worse than refusing.
function _varrange(vars, nvars::Integer)
    r = if vars isa Integer
        Int(vars):Int(vars)
    elseif vars isa AbstractUnitRange{<:Integer}
        Int(first(vars)):Int(last(vars))
    else
        throw(ArgumentError("vars must be an integer or a contiguous range of " *
                            "variable indices, got a $(typeof(vars)): the " *
                            "selection becomes a kernel argument, and a kernel " *
                            "cannot be handed an arbitrary index vector"))
    end
    isempty(r) || 1 <= first(r) <= last(r) <= nvars ||
        throw(ArgumentError("vars = $vars is out of range for a field set with " *
                            "$nvars variable(s); valid indices are 1:$nvars"))
    return r
end

"""
    block_mapreduce(f, op, init, fs::FieldSet; vars=1:fs.nvars)
    block_mapreduce(f, op, init, fs::FieldSet, u::AbstractVector; vars=1:fs.nvars)

Reduce each block's interior cells to one value: `f` transforms a cell
value and `op` folds the transformed values into an accumulator that
starts at `init`. The result is a host `Vector` of length
`nblocks(fs)`, indexed by block, with element type taken from `init`.

The first form reads the working array's interiors, skipping the ghosts;
the second reads a state vector, which has no ghosts to skip. `vars` is
an integer or a contiguous range of variable indices.

This is the read-side counterpart of [`map_blocks!`](@ref), and it is
the shape every diagnostic here has. Combining the values is left to the
caller, because the useful combination usually weights each block by its
own geometry first — see [`total_mass`](@ref). Combine them **in block
order** (`sum`, `maximum`, a loop over `1:nblocks(fs)`) and the answer
does not depend on the thread count; a running total split across tasks
would not, which is why this returns the per-block values rather than a
number.

The largest value of each variable, which a refinement criterion needs
for its scale:

```julia
scales = [maximum(block_mapreduce(abs, max, zero(eltype(fs.work)), fs; vars=v))
          for v in 1:fs.nvars]
```

It runs wherever the data lives: threaded host reductions over per-block
views on the CPU, one kernel work item per block on a device. It
synchronizes and copies back to the host, so it belongs at diagnostic or
regrid frequency — not inside a right-hand side.

The fold is one specification on both backends, but its *association* is
not: the host's `mapreduce` may reassociate `op` where the kernel's
sequential loop cannot. The guarantee is bit-identical results across
thread counts, which is what `CODE.md` claims; identical results across
backends is not claimed and, for floating-point `op`, not true.

!!! note "Callbacks on a device"
    `f` and `op` become kernel arguments, so everything they close over
    must be `isbits`. A captured `Type` is the usual trip: write
    `oftype(x, 2)` rather than closing over `T` and calling `T(2)`. The
    same rule covers captured arrays and any mutable state.
"""
function block_mapreduce(f, op, init, fs::FieldSet{T,D}; vars=1:fs.nvars) where {T,D}
    return _block_mapreduce(f, op, init, fs.work, fs, Int(fs.forest.G),
                            _varrange(vars, fs.nvars))
end

function block_mapreduce(f, op, init, fs::FieldSet{T,D}, u::AbstractVector;
                         vars=1:fs.nvars) where {T,D}
    return _block_mapreduce(f, op, init, statearray(u, fs), fs, 0,
                            _varrange(vars, fs.nvars))
end

function _block_mapreduce(f, op, init::R, array, fs::FieldSet{T,D}, g::Int,
                          vars::UnitRange{Int}) where {R,T,D}
    backend = get_backend(array)
    return backend isa CPU ?
           _block_mapreduce_host(f, op, init, array, fs, g, vars) :
           _block_mapreduce_device(f, op, init, array, fs, backend, g, vars)
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
    forest = fs.forest
    R = float(real(T))

    # One partial per block, then a serial pass over them in block
    # order: threaded, and bit-for-bit independent of the thread count,
    # which a running total split across tasks would not be.
    if isinf(p)
        partials = block_mapreduce(abs, max, zero(R), fs, u)
        return isempty(partials) ? zero(R) : maximum(partials)
    end

    # An integer exponent stays an integer: `abs(x)^2` is a squaring,
    # while `abs(x)^2.0` would drag a `Float64` operand into the
    # innermost loop — fatal on a device with no hardware fp64, and the
    # exact leak the type-genericity work went after.
    q = p isa Integer ? Int(p) : R(p)
    partials = block_mapreduce(x -> abs(x)^q, +, zero(R), fs, u)
    volumes = Vector{R}(undef, nblocks(fs))
    cells = fs.forest.N^D * fs.nvars
    for b in 1:nblocks(fs)
        cellvolume = spacing(R, forest, blockkey(fs, b))^D
        partials[b] *= cellvolume
        volumes[b] = cellvolume * cells
    end

    volume = sum(volumes)
    volume == 0 && return zero(R)
    # `inv(R(p))`, not `1 / p`: the latter is a Float64 exponent, which
    # promotes the whole result to Float64 and made this function return a
    # different type from the `isinf(p)` branch above.
    return (sum(partials) / volume)^inv(R(p))
end
