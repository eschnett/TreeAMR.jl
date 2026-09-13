# Host-side threading.
#
# Per-cell work is KernelAbstractions kernels, and the CPU backend
# already spreads a launch over `Threads.nthreads()`, one contiguous
# chunk of the ndrange per task. Since every cell of every kernel in
# this package writes its own output slot, that much of M5 is free: run
# Julia with threads and the kernels are parallel.
#
# What remains is the *host-side* loops over blocks — neighbor finding
# when the schedule is built, the mark arithmetic in regridding, the
# boundary hook, the diagnostic reductions. Those are ordinary Julia,
# and they are threaded here.
#
# Every one of them is written as "compute per block in parallel, then
# combine in block order", never as "accumulate into shared state". The
# partition below depends only on the item count and the thread count,
# and the combining pass is serial and ordered, so results are
# **bit-for-bit** independent of how many threads Julia was started
# with — a stronger property than the "matches serial to roundoff" that
# `CODE.md` asks of M5, and one that costs nothing but the discipline.

"""
    threadchunks(n) -> Vector{UnitRange{Int}}

`1:n` split into one contiguous range per task: as many ranges as there
are threads, never more than there are items, and exactly one range when
Julia was started single-threaded (so nothing is spawned at all).

The split is a function of `n` and `Threads.nthreads()` alone — never of
which thread picks up which range — so anything built from it is
reproducible.
"""
function threadchunks(n::Integer)
    n = Int(n)
    n <= 0 && return UnitRange{Int}[]
    ntasks = min(Threads.nthreads(), n)
    ntasks <= 1 && return [1:n]
    len, extra = divrem(n, ntasks)
    ranges = Vector{UnitRange{Int}}(undef, ntasks)
    lo = 1
    for t in 1:ntasks
        hi = lo + len - 1 + (t <= extra)
        ranges[t] = lo:hi
        lo = hi + 1
    end
    return ranges
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
own task, with `c` the chunk index. A caller that has to *collect*
results rather than write them in place gives each task the slot `c` of
a per-chunk buffer and concatenates the buffers afterwards in chunk
order — which is how every collecting pass in this package stays
reproducible while allocating one buffer per thread rather than one per
block.
"""
function threaded_chunks(f, n::Integer)
    chunks = threadchunks(n)
    if length(chunks) <= 1
        for (c, range) in enumerate(chunks)
            f(c, range)
        end
        return nothing
    end
    try
        @sync for (c, range) in enumerate(chunks)
            Threads.@spawn f(c, range)
        end
    catch err
        rethrow(innermost_error(err))
    end
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
