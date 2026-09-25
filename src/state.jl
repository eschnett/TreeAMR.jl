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
    zerofill!(statearray(u, fs), backend)          # first touch by block owner
    return u
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
    work[ntuple(d -> I[d] + G[d], Val(D))..., I[D + 1], I[D + 2]] = state[I...]
end

@kernel function gather_kernel!(state, @Const(work), ::Val{D}, ::Val{G}) where {D,G}
    I = @index(Global, NTuple)
    state[I...] = work[ntuple(d -> I[d] + G[d], Val(D))..., I[D + 1], I[D + 2]]
end

function run_over_interiors!(kernel, fs::FieldSet{T,D}, a, b) where {T,D}
    backend = get_backend(fs.work)
    launch_by_owner!(kernel, backend, a, b, Val(D), Val(fs.G);
                     ndrange=(ntuple(_ -> fs.forest.N, D)..., fs.nvars, nblocks(fs)))
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
    map_blocks!(kernel!, fs::FieldSet, args...; closed=false, stored=false)

Launch a KernelAbstractions kernel over every **owned** point of every
block, with `ndrange = (N, ..., N, nblocks)`. The kernel's global index
is therefore `(i1, ..., iD, b)` with each `i` running over `1:N`; add
`G[d]` to reach the working array's stored indices.

With `closed = true` the loop runs over the **closed** range instead —
`N + c[d]` per dimension, so a vertex-like dimension also covers the
shared boundary plane (see [`closedview`](@ref)). That is what a
quantity defined on a block's faces wants: a flux has `N+1` faces per
dimension, not `N`, and the extra one is the block's own high face. In a
cell-centered field set the two are the same loop.

With `stored = true` the loop runs over every **stored** point of every
block, ghosts included — `ndrange = (size(fs.work)[1:D]..., nblocks(fs))`,
that is `N + 2G[d] + c[d]` per dimension.

!!! warning "The three forms index differently"
    Under `stored = true` the kernel's global index **is** the stored
    index: there is nothing to add. Under the default and under
    `closed = true` it is an offset into the owned range and the kernel
    adds `G[d]`. A kernel written for one form is wrong under the other
    — silently so, since both are in bounds — so a kernel meant for the
    stored form should not add `G` anywhere.

`stored = true` together with `closed = true` is an error: the closed
range is a sub-range of the stored one, so asking for both is a
contradiction.

The stored form exists because a pointwise pass sometimes has to cover
the ghosts too. The case that asked for it is a hydrodynamics code
exchanging *conserved* variables and recovering *primitive* ones from
them: its reconstruction reads primitives two cells into the
neighbours, so the recovery has to have run in the ghost cells as well.
Which points a block stores is the mesh's business, not the
application's, so the launch says `stored = true` rather than the
application spelling out `N + 2G + c` for itself (see `CODE.md`,
"Application interface").

Blocks are uniform work units, so this is one flat parallel loop. On
the CPU it runs *by owner* (see [`launch_by_owner!`](@ref TreeAMR.launch_by_owner!)):
every block on the same thread as in every other per-block pass of the
package, which is worth up to 2.4x on a many-core node, and one
workgroup per block — so a kernel handed to `map_blocks!` must not
depend on the workgroup size. The same launch runs on a device; every
work item writes its own output cell, so the result does not depend on
how the loop was split.

```julia
@kernel function rhs!(du, @Const(work), @Const(h), ::Val{D}, ::Val{G}) where {D,G}
    I = @index(Global, NTuple)
    b = I[D + 1]
    c = ntuple(d -> I[d] + G[d], Val(D))       # stored (ghosted) index
    du[ntuple(d -> I[d], Val(D))..., 1, b] = work[c..., 2, b]
end
```

The same pass over the stored extent, which reaches the ghosts and does
not shift:

```julia
@kernel function recover!(prim, @Const(cons), ::Val{D}) where {D}
    I = @index(Global, NTuple)                 # already a stored index
    b = I[D + 1]
    c = ntuple(d -> I[d], Val(D))
    prim[c..., 1, b] = sqrt(cons[c..., 1, b])
end
```
"""
function map_blocks!(kernel!, fs::FieldSet{T,D}, args...;
                     closed::Bool=false, stored::Bool=false) where {T,D}
    stored && closed && throw(ArgumentError(
        "map_blocks! takes `closed = true` or `stored = true`, not both: the " *
        "closed range G+1 … G+N+c is a sub-range of the stored one 1 … N+2G+c, " *
        "so asking for both names two different loops. Pass `stored = true` for " *
        "every stored point, ghosts included, and `closed = true` for the owned " *
        "points plus the shared boundary plane."))
    backend = get_backend(fs.work)
    c = staggers(fs)
    extent = stored ? ntuple(d -> size(fs.work, d), D) :
             ntuple(d -> fs.forest.N + (closed ? c[d] : 0), D)
    launch_by_owner!(kernel!, backend, args...; ndrange=(extent..., nblocks(fs)))
    synchronize(backend)
    return nothing
end

# Per-block reductions.
#
# Every diagnostic in the package has the same shape: one value per
# block, then a pass over those values on the host. Each block's value
# is bit-for-bit independent of the thread count (M5), because it is
# computed from that block's cells alone; a floating-point combination
# of the values is promised to roundoff only (`CODE.md`, "Parallelism",
# narrowed after M8), which is what lets the device path below be
# hierarchical and lets M7 `Allreduce`. Only how the per-block values
# are produced changes with the backend.
#
# On the CPU they are threaded host reductions over per-block views,
# which is what M5 measured and what the recorded numbers were taken
# with. On a device that same formulation would be one kernel launch and
# one device-to-host synchronization *per block*, issued from several
# host tasks at once; and the M6 form — one work item per block, each
# looping its own cells — was the weak row of the M6 table, 960 work
# items on a device that wants tens of thousands. So a device reduces
# in two launches: `REDUCE_LANES` lanes per block, each striding over
# the block's cells and variables in linear order (adjacent lanes read
# adjacent cells) into a private partial, then one work item per block
# folding its lanes in order. There is no barrier and no local memory,
# deliberately: KernelAbstractions realises a barrier on the CPU backend
# by splitting the kernel into separate loops over the workgroup, so a
# local does not survive a `@synchronize` unless it is `@private`, and
# the CPU backend is exactly where the suite cross-checks this path.
# Without a barrier the lanes of a block need not share a workgroup
# either, so none is imposed: a lane is a work item of an ordinary
# launch, and block and lane are read off the global index. (A static
# workgroup of 256 was the first form; Metal.jl launches one without
# checking the pipeline's own limit, so a register-heavy `f` could have
# failed to launch.) Every partial is a function of the block's cells
# and the stride, and the lane fold has a fixed order, so the device
# result is reproducible from run to run and exact for an
# order-independent `op`. `init` starts every lane, so it enters the
# fold once per lane rather than once: it must satisfy
# `op(init, init) == init`, which a neutral element does, and so does
# any `init` under an idempotent `op` such as `max`.
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
# `array` is either the working array (offset `g = fs.G`, so ghosts are
# skipped) or the state array (`g` all zeros). Neither helper below works
# that out for itself; `block_mapreduce` does, which is why it and not
# these is what an application calls.

# Lanes per block in the device reductions. A power of two, so that
# block and lane come out of the global index as a shift and a mask; a
# block with fewer cells times variables than lanes leaves the surplus
# lanes at `init`.
const REDUCE_LANES = 256

# Launch one: lane `l` of block `b` folds entries `l, l + W, l + 2W, …`
# of the block's cells times selected variables, cells fastest, from
# `init`, into its own slot.
@kernel function block_partials_kernel!(partials, @Const(array), f, op, init,
                                        firstvar::Int, lastvar::Int,
                                        ::Val{D}, ::Val{G}, ::Val{N},
                                        ::Val{W}) where {D,G,N,W}
    g = @index(Global)
    b, l = divrem(g - 1, W) .+ 1
    cells = CartesianIndices(ntuple(_ -> N, Val(D)))
    ncells = length(cells)
    acc = init
    for j in l:W:(ncells * (lastvar - firstvar + 1))
        q, r = divrem(j - 1, ncells)
        c = Tuple(cells[r + 1])
        acc = op(acc, f(array[ntuple(d -> c[d] + G[d], Val(D))..., firstvar + q, b]))
    end
    partials[l, b] = acc
end

# Launch two: one work item per block folds its lanes, in lane order,
# starting from the first lane's partial.
@kernel function fold_lanes_kernel!(values, @Const(partials), op, ::Val{W}) where {W}
    b = @index(Global)
    acc = partials[1, b]
    for l in 2:W
        acc = op(acc, partials[l, b])
    end
    values[b] = acc
end

# The device path: two launches, one synchronization, one copy back.
function _block_mapreduce_device(f, op, init::R, array, fs::FieldSet{T,D},
                                 backend::Backend, g::NTuple{D,Int},
                                 vars::UnitRange{Int}) where {R,T,D}
    n = nblocks(fs)
    n == 0 && return R[]
    W = REDUCE_LANES
    partials = allocate(backend, R, (W, n))
    values = allocate(backend, R, (n,))
    block_partials_kernel!(backend)(partials, array, f, op, init,
                                    first(vars), last(vars),
                                    Val(D), Val(g), Val(fs.forest.N), Val(W);
                                    ndrange=W * n)
    fold_lanes_kernel!(backend)(values, partials, op, Val(W); ndrange=n)
    synchronize(backend)
    return tohost(values)
end

# The host path. It is split out under its own name rather than
# dispatched on `::CPU` so that both paths stay reachable on a machine
# with no device, which is what lets the suite check that the two
# compute the same fold.
function _block_mapreduce_host(f, op, init::R, array, fs::FieldSet{T,D},
                               g::NTuple{D,Int},
                               vars::UnitRange{Int}) where {R,T,D}
    N = fs.forest.N
    inner = ntuple(d -> (g[d] + 1):(g[d] + N), D)
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
value and `op` folds the transformed values, starting from `init`. The
result is a host `Vector` of length `nblocks(fs)`, indexed by block,
with element type taken from `init`. `init` must satisfy
`op(init, init) == init` — a neutral element does, as Base's `reduce`
requires of its `init`, and so does any `init` under an idempotent `op`
such as `max` — because on a device the fold starts from it once per
lane, not once per block.

The first form reads the working array's interiors, skipping the ghosts;
the second reads a state vector, which has no ghosts to skip. `vars` is
an integer or a contiguous range of variable indices.

This is the read-side counterpart of [`map_blocks!`](@ref), and it is
the shape every diagnostic here has. Combining the values is left to the
caller, because a refinement criterion wants them per block; for one
number over the whole mesh, weighted or not, use
[`mesh_mapreduce`](@ref). Each block's value is bit-identical whatever
the thread count, since it is computed from that block's cells alone; a
floating-point combination of them is promised to roundoff only,
whichever way it is written.

The largest value of each variable, which a refinement criterion needs
for its scale:

```julia
scales = [maximum(block_mapreduce(abs, max, zero(eltype(fs.work)), fs; vars=v))
          for v in 1:fs.nvars]
```

It runs wherever the data lives: threaded host reductions over per-block
views on the CPU; on a device, one workgroup per block in two launches,
lanes striding over the block's cells and then one work item per block
folding the lanes. It synchronizes and copies back to the host, so it
belongs between steps and not inside a right-hand side.

The fold is one specification on both backends, but its *association* is
not: the host's `mapreduce` may reassociate `op`, and the lanes of the
device fold split the cells differently again. The guarantee is
per-block values that are bit-identical across thread counts and, for a
floating-point `op`, agree to roundoff across backends — identical
across backends is not claimed, and is not true. A device value is
reproducible from run to run, since the lane fold has a fixed order.

!!! note "Callbacks on a device"
    `f` and `op` become kernel arguments, so everything they close over
    must be `isbits`. A captured `Type` is the usual trip: write
    `oftype(x, 2)` rather than closing over `T` and calling `T(2)`. The
    same rule covers captured arrays and any mutable state.
"""
function block_mapreduce(f, op, init, fs::FieldSet{T,D}; vars=1:fs.nvars) where {T,D}
    return _block_mapreduce(f, op, init, fs.work, fs, fs.G,
                            _varrange(vars, fs.nvars))
end

function block_mapreduce(f, op, init, fs::FieldSet{T,D}, u::AbstractVector;
                         vars=1:fs.nvars) where {T,D}
    return _block_mapreduce(f, op, init, statearray(u, fs), fs, ntuple(_ -> 0, D),
                            _varrange(vars, fs.nvars))
end

function _block_mapreduce(f, op, init::R, array, fs::FieldSet{T,D}, g::NTuple{D,Int},
                          vars::UnitRange{Int}) where {R,T,D}
    backend = get_backend(array)
    return backend isa CPU ?
           _block_mapreduce_host(f, op, init, array, fs, g, vars) :
           _block_mapreduce_device(f, op, init, array, fs, backend, g, vars)
end

"""
    mesh_mapreduce(f, op, init, fs::FieldSet; vars=1:fs.nvars, weight=nothing)
    mesh_mapreduce(f, op, init, fs::FieldSet, u::AbstractVector;
                   vars=1:fs.nvars, weight=nothing)

Reduce the whole mesh to one number: the per-block values of
[`block_mapreduce`](@ref), each multiplied by `weight(key)` when a
weight is given, combined with `op`. The two forms and `vars` are those
of `block_mapreduce`; `init` is returned for a mesh with no blocks and
must be a neutral element for `op`.

`weight` is a function of a block's [`MortonKey`](@ref), evaluated on the
host and converted to the type of `init` before it multiplies each
block's value — so `init` has to be a floating-point value when a
weight is given. It scales a block's contribution by its geometry: a
cell volume, `spacing(T, forest, key)^D`, is what
[`volume_weighted_norm`](@ref) and [`total_mass`](@ref) pass to `+`,
and a positive weight is equally meaningful under `max` or `min`, where
`max |v| / h` is a CFL rate.

This is the form a conserved total, a norm, or a CFL speed wants, and
the one place a reduction crosses blocks: the per-block values are
combined on the host, and under MPI (M7) across ranks as well, so an
application never sees a communicator. The floating-point result is
promised to roundoff across thread counts, rank counts and backends,
not bit for bit; see [`block_mapreduce`](@ref) for what *is* exact.

The peak of a variable over the mesh, and the mass of another:

```julia
peak = mesh_mapreduce(abs, max, zero(T), fs; vars=1)
mass = mesh_mapreduce(identity, +, zero(T), fs; vars=2,
                      weight=key -> spacing(T, fs.forest, key)^D)
```
"""
function mesh_mapreduce(f, op, init, fs::FieldSet; vars=1:fs.nvars, weight=nothing)
    values = block_mapreduce(f, op, init, fs; vars=vars)
    return combine_blocks(op, init, fs, values, weight)
end

function mesh_mapreduce(f, op, init, fs::FieldSet, u::AbstractVector; vars=1:fs.nvars,
                        weight=nothing)
    values = block_mapreduce(f, op, init, fs, u; vars=vars)
    return combine_blocks(op, init, fs, values, weight)
end

# The host stage. Two details keep every recorded norm and mass exactly
# where the M3 code left it. The weight multiplies in place, in the
# order `total_mass` always used. And the combination is
# `mapreduce(identity, op, values)` *without* `init`: `sum(v)` and
# `mapreduce(identity, +, v)` are the same pairwise reduction bit for
# bit, whereas `reduce(+, v; init)` is a sequential left fold — an
# explicit `init` changes Base's association (measured, 2026-09-22). The
# M7 `Allreduce` goes here and nowhere else.
function combine_blocks(op, init::R, fs::FieldSet, values::Vector{R}, weight) where {R}
    if weight !== nothing
        for b in eachindex(values)
            values[b] *= oftype(init, weight(blockkey(fs, b)))
        end
    end
    isempty(values) && return init
    return mapreduce(identity, op, values)
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

A [`mesh_mapreduce`](@ref) with the cell volume as the weight. The value
is reproducible to roundoff across thread counts and backends; on the
CPU, where the partials are combined in block order, it is exact across
thread counts as well.
"""
function volume_weighted_norm(fs::FieldSet{T,D}, u::AbstractVector; p::Real=2) where {T,D}
    forest = fs.forest
    R = float(real(T))

    isinf(p) && return mesh_mapreduce(abs, max, zero(R), fs, u)

    # An integer exponent stays an integer: `abs(x)^2` is a squaring,
    # while `abs(x)^2.0` would drag a `Float64` operand into the
    # innermost loop — fatal on a device with no hardware fp64, and the
    # exact leak the type-genericity work went after.
    q = p isa Integer ? Int(p) : R(p)
    # `float(real(T))` inline rather than the local `R`: a closure over a
    # local holding a type stores it as a `DataType` on Julia 1.10 and
    # returns `Any`, which cost the norm a dynamic dispatch per block
    # (found in review, measured at 2x the host time on 1.10.12).
    cellvolume(key) = spacing(float(real(T)), forest, key)^D
    total = mesh_mapreduce(x -> abs(x)^q, +, zero(R), fs, u; weight=cellvolume)

    # The domain volume as the sum of the block volumes, through the same
    # combination as everything else — so `sum`'s association from M3 is
    # kept, and M7 has one place to reduce across ranks.
    cells = forest.N^D * fs.nvars
    volumes = Vector{R}(undef, nblocks(fs))
    for b in 1:nblocks(fs)
        volumes[b] = cellvolume(blockkey(fs, b)) * cells
    end
    volume = combine_blocks(+, zero(R), fs, volumes, nothing)
    volume == 0 && return zero(R)
    # `inv(R(p))`, not `1 / p`: the latter is a Float64 exponent, which
    # promotes the whole result to Float64 and made this function return a
    # different type from the `isinf(p)` branch above.
    return (total / volume)^inv(R(p))
end
