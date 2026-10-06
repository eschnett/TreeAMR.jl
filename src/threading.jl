# Host-side threading, and which thread runs which block.
#
# Per-cell work is KernelAbstractions kernels, and the CPU backend
# spreads a launch over `Threads.nthreads()`, one contiguous chunk of
# the ndrange per task. Since every cell of every kernel in this package
# writes its own output slot, that much of M5 is free: run Julia with
# threads and the kernels are parallel. What remains is the *host-side*
# loops over blocks — neighbor finding when the schedule is built, the
# mark arithmetic in regridding, the boundary hook, the diagnostic
# reductions. Those are ordinary Julia, and they are threaded here.
#
# Every parallel pass over blocks on the CPU also gives each block to
# the same thread, every time: block `b` belongs to the thread whose
# chunk of `threadchunks(nblocks)` contains it, and both the host loops
# below and the kernel launches of `launch_by_owner!` honour that. On a
# many-core node a block's data streams at up to 2.7x the rate when the
# core that last touched it touches it again — most likely because a
# line last held by another core complex costs coherence traffic that
# competes with the data on the fabric (`CODE.md`, "Parallelism", "What
# one process loses"). Launching each phase on whichever thread is free — what
# `Threads.@spawn` and KernelAbstractions' default CPU schedule do —
# moved every block to a new core in every phase, and cost the RHS
# evaluation 2.4x on 64 cores.
#
# Every host loop here is written as "compute per block in parallel, then
# combine in block order", never as "accumulate into shared state". The
# partition below depends only on the item count and the thread count,
# and the combining pass is serial and ordered, so everything these
# loops produce — schedules, marks, per-block values — is **bit-for-bit**
# independent of how many threads Julia was started with, and costs
# nothing but the discipline. Summing floating-point partials in block
# order is how the diagnostics happen to be written and is free here
# too, but since M8 it is not a promise: `CODE.md` ("Parallelism")
# guarantees a floating-point sum to roundoff only, so that a device
# may reduce hierarchically and MPI may `Allreduce`.

"""
    threadchunks(n) -> Vector{UnitRange{Int}}

`1:n` split into one contiguous range per task: as many ranges as there
are threads, never more than there are items, and exactly one range when
Julia was started single-threaded (so nothing is spawned at all).

The split is a function of `n` and `Threads.nthreads()` alone — never of
which thread picks up which range — so anything built from it is
reproducible. Over blocks it is also the *ownership* partition: range
`c` runs on thread `c` in [`threaded_chunks`](@ref) and in every
[`launch_by_owner!`](@ref).
"""
function threadchunks(n::Integer)
    n = Int(n)
    n <= 0 && return UnitRange{Int}[]
    ntasks = min(Threads.nthreads(), n)
    ntasks <= 1 && return [1:n]
    return [equalsplit(n, ntasks, t) for t in 1:ntasks]
end

# `1:n` split into `p` contiguous ranges whose lengths differ by at most
# one, the longer ones first: `divrem`, with the remainder going to the
# first parts, and part `i` returned in O(1) for `i in 1:p` without the
# others being formed. A part beyond `n` is empty. This is the one piece
# of arithmetic behind both the thread split (`threadchunks`) and the
# rank split (`blockrange`, M7), so that the two are the same contiguous
# curve ranges one level apart (`CODE.md`, "Distributed meshes"); a
# cost-weighted split would change only this function.
function equalsplit(n::Int, p::Int, i::Int)
    len, extra = divrem(n, p)
    lo = 1 + (i - 1) * len + min(i - 1, extra)
    return lo:(lo + len - 1 + (i <= extra))
end

# The part of `equalsplit(n, p, ·)` that contains item `i in 1:n`, in
# O(1): the inverse of the split, which is how a rank names the owner of
# a leaf (M7) without forming every rank's range.
function equalsplit_part(n::Int, p::Int, i::Int)
    len, extra = divrem(n, p)
    long = extra * (len + 1)                # the items of the longer parts
    i <= long && return (i - 1) ÷ (len + 1) + 1
    return extra + (i - long - 1) ÷ len + 1
end

"""
    threaded_foreach(f, n)

Call `f(i)` for every `i in 1:n`, one task per chunk of
[`threadchunks`](@ref). On a single thread this is a plain loop — no
task is spawned and no closure is boxed — so the serial path pays
nothing for the parallel one.

`f` must write only to storage belonging to its own `i`: these loops
carry no locks and no atomics by design (see the note at the top of
`threading.jl`).
"""
function threaded_foreach(f, n::Integer)
    return threaded_chunks(n) do _, range
        for i in range
            f(i)
        end
    end
end

"""
    threaded_chunks(f, n)

Call `f(c, range)` once per chunk of [`threadchunks`](@ref), each on its
own task, with `c` the chunk index, and chunk `c` always on the `c`-th
thread of the default pool. A caller that has to *collect* results
rather than write them in place gives each task the slot `c` of a
per-chunk buffer and concatenates the buffers afterwards in chunk order
— which is how every collecting pass in this package stays reproducible
while allocating one buffer per thread rather than one per block.

The tasks are *sticky*, placed on their thread the way `Threads.@threads
:static` places its own, but without entering a "threaded region": so
unlike `@threads :static` this may be called from inside another
parallel loop, or from two tasks at once. It waits for every task before
rethrowing the first error.
"""
function threaded_chunks(f, n::Integer)
    chunks = threadchunks(n)
    if length(chunks) <= 1
        for (c, range) in enumerate(chunks)
            f(c, range)
        end
        return nothing
    end
    offset = Threads.threadpoolsize(:interactive)   # default-pool thread ids follow
    tasks = Vector{Task}(undef, length(chunks))
    for (c, range) in enumerate(chunks)
        task = Task(() -> f(c, range))
        task.sticky = true
        ccall(:jl_set_task_tid, Cint, (Any, Cint), task, offset + c - 1)
        tasks[c] = task
        schedule(task)
    end
    failure = nothing
    for task in tasks
        try
            wait(task)
        catch err
            failure === nothing && (failure = err)
        end
    end
    failure === nothing || throw(innermost_error(failure))
    return nothing
end

# `@sync` reports whatever a task threw as a `TaskFailedException`
# (inside a `CompositeException` when several tasks fail at once).
# Callers should see the error the loop body actually threw — the
# package's `ArgumentError`s say *why* something is wrong, and that
# message must survive being raised on a worker task — so unwrap it.
function innermost_error(err)
    err isa CompositeException && !isempty(err.exceptions) &&
        return innermost_error(first(err.exceptions))
    err isa TaskFailedException && return innermost_error(err.task.exception)
    return err
end

"""
    launch_by_owner!(kernel, backend, args...; ndrange)

Launch the KernelAbstractions `kernel` over a block-shaped `ndrange`,
whose last axis is the block index, so that on the CPU every block runs
on the thread that owns it (see [`threadchunks`](@ref)).

On the CPU that is the backend's static schedule with one block per
workgroup: KernelAbstractions then splits the blocks exactly as
`threadchunks(nblocks)` does and puts chunk `c` on thread `c`, the same
thread [`threaded_chunks`](@ref) uses. A kernel launched this way must
not depend on the workgroup size. The static schedule is
`Threads.@threads :static` underneath, which refuses to run inside
another threaded region, so there — a caller's own `@threads` loop — the
launch falls back to the default schedule and simply loses the affinity.
A device backend is launched as it always was.
"""
function launch_by_owner!(kernel, backend::Backend, args...; ndrange)
    # A rank without blocks (M7) launches nothing.
    any(iszero, ndrange) && return nothing
    kernel(backend)(args...; ndrange=ndrange)
    return nothing
end

function launch_by_owner!(kernel, backend::CPU, args...; ndrange)
    any(iszero, ndrange) && return nothing
    if Threads.nthreads() == 1 || ccall(:jl_in_threaded_region, Cint, ()) != 0
        kernel(backend)(args...; ndrange=ndrange)
    else
        kernel(CPU(; static=true))(args...; ndrange=ndrange,
                                   workgroupsize=(Base.front(ndrange)..., 1))
    end
    return nothing
end

# A launch flattened to one axis on a device, for the copy kernels.
#
# KernelAbstractions forms `@index(Global, NTuple)` on a device by indexing
# two `CartesianIndices` — the workgroup's position in the grid, and the
# item's in the workgroup — with linear ids, which costs two integer
# divisions per axis when the extents are run-time values: a 64-bit
# division, emulated in software, per axis per work item. In a kernel whose
# work is one load and one store that is nearly everything — about 1200
# SASS instructions to copy one value on an H200 ("The copy kernels on a
# device" in CODE.md). A one-dimensional ndrange needs no division at all,
# so on a device the copy kernels launch over one axis and recover their
# position themselves, by multiplying with each extent's precomputed
# inverse (`Base.MultiplicativeInverses`, the arithmetic a compiler applies
# to a division by a constant) rather than dividing.
#
# The CPU keeps the shaped launch. There the backend iterates a
# workgroup's `CartesianIndices` by incrementing, with no division, and
# the innermost axis runs along a contiguous row; a flat launch made the
# CPU scatter 2.3x slower (measured 38.8 → 89 ns a point at `16³`, 20
# variables), half of it the inverses and half the stores no longer
# running along a row. So a kernel launched through `launch_positional!`
# takes its position from `kernel_position(shape, I)`: `shape` is
# `nothing` on the CPU, where `I` is already the position, and a
# `LinearShape` on a device, where the launch is flat. One kernel serves
# both.
#
# `LinearShape` holds the inverses of the first `K` extents of a `K + 1`-
# axis shape; the last extent needs none, since nothing is divided by it.
# It is isbits, so it is a kernel argument on every backend. Building one
# costs about 9 ns an extent, so it is built at each launch. Its arithmetic
# is `Int32` whenever the whole launch fits, and `Int` otherwise: measured
# for a scatter's index (D = 3), the 32-bit form took 3.0 ns a point on
# Metal against 5.0; signed, because Base's unsigned inverse branches. A
# launch past 2³¹ items (a 17 GB state vector of `Float64`) takes the `Int`
# form and is still exact.
struct LinearShape{K,I<:Union{Int32,Int}}
    inverses::NTuple{K,SignedMultiplicativeInverse{I}}
end

function LinearShape(extents::Tuple{Vararg{Integer}})
    I = prod(Int, extents) <= typemax(Int32) ? Int32 : Int
    inverses = map(e -> SignedMultiplicativeInverse{I}(I(e)), Base.front(extents))
    return LinearShape{length(inverses),I}(inverses)
end

# The one-based position `(i₁, …, i_{K+1})` of the one-based linear index
# `n`, column-major as `CartesianIndices` orders it, as `Int`s. Recursion
# over the tuple rather than a loop, so that it unrolls in a device kernel
# too.
@inline linear_position(s::LinearShape{K,I}, n::Integer) where {K,I} =
    map(Int, _position(s.inverses, (n - 1) % I))
@inline _position(::Tuple{}, x::Integer) = (x + one(x),)
@inline function _position(inverses::Tuple, x::Integer)
    q = div(x, first(inverses))
    r = x - q * first(inverses).divisor
    return (r + one(x), _position(Base.tail(inverses), q)...)
end

# A kernel's position from its global index tuple `I`: the tuple itself
# under a shaped launch, the position of its one linear index under a flat
# one.
@inline kernel_position(::Nothing, I::Tuple) = I
@inline kernel_position(s::LinearShape, I::Tuple{Integer}) = linear_position(s, first(I))

# The launch of `kernel_position`'s two forms: the shape and the ndrange for
# `extents`. `nothing` and the extents themselves for a shaped launch, a
# `LinearShape` and the product for a flat one. A device launches flat and
# the CPU shaped (`flatlaunch`); the callers take `flat` as a keyword, so
# that the suite runs the device's form on the CPU, where CI has no device.
flatlaunch(::CPU) = false
flatlaunch(::Backend) = true
launch_shape(flat::Bool, extents::Tuple) =
    flat ? (LinearShape(extents), prod(Int, extents)) : (nothing, extents)

# `launch_by_owner!` for a kernel that takes its position through
# `kernel_position`: `shape` is its first argument, and `extents` the shaped
# ndrange, block index last. A shaped launch goes by owner as it is; a flat
# one on the CPU, which only the suite asks for, is one block per workgroup
# on KernelAbstractions' default schedule, not by owner.
function launch_positional!(kernel, backend::Backend, args...; extents::Tuple,
                            flat::Bool=flatlaunch(backend))
    any(iszero, extents) && return nothing          # a rank without blocks (M7)
    flat || return launch_by_owner!(kernel, backend, nothing, args...; ndrange=extents)
    # The flat launch's shape is `Int32` or `Int` by the launch's size, and
    # on the CPU the flat launch runs only in the suite: hidden from
    # inference, so that a call site is compiled for the launch it makes,
    # not for all three (see `run_group!`).
    shape, ndrange = launch_shape(true, extents)
    launch = Base.inferencebarrier(kernel(backend))
    if backend isa CPU
        launch(shape, args...; ndrange=ndrange, workgroupsize=prod(Int, Base.front(extents)))
    else
        launch(shape, args...; ndrange=ndrange)
    end
    return nothing
end

"""
    threaded_collect(f!, T, n) -> Vector{T}

Run `f!(buffer, i)` for every `i in 1:n`, where `buffer` is this task's
own `Vector{T}` to push onto, and return the concatenation of the
buffers in index order — the same vector a serial loop pushing onto one
buffer would have produced.
"""
function threaded_collect(f!, ::Type{T}, n::Integer) where {T}
    chunks = threadchunks(n)
    isempty(chunks) && return T[]
    buffers = [T[] for _ in chunks]
    threaded_chunks(n) do c, range
        buffer = buffers[c]
        for i in range
            f!(buffer, i)
        end
    end
    out = buffers[1]
    for c in 2:length(buffers)
        append!(out, buffers[c])
    end
    return out
end
