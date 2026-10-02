# The communicator layer (M7).
#
# One forest can run over several processes. Every rank holds the whole
# forest and only field data are distributed (`CODE.md`, "Distributed
# meshes"), so the package needs very little of a message-passing
# library: who it is, how many there are, three collectives and
# nonblocking point-to-point messages over flat buffers. Those are the
# verbs below. Every verb has a method for `SerialCommunicator`, the
# default, which is rank 0 of 1 and sends nothing; so `src/` never
# branches on whether MPI is loaded, and a serial run takes the
# distributed code path with every message empty.
#
# The MPI methods live in a package extension, `ext/TreeAMRMPIExt.jl`,
# for the reason HDF5's do: an application that never runs distributed
# should not load MPI.

"""
    Communicator

The processes a [`Forest`](@ref) is distributed over, as the package
sees them: an abstract type whose subtypes answer the internal verbs
[`commrank`](@ref TreeAMR.commrank), [`commsize`](@ref TreeAMR.commsize),
[`allgather`](@ref TreeAMR.allgather),
[`allgatherv`](@ref TreeAMR.allgatherv),
[`alltoallv`](@ref TreeAMR.alltoallv) and the nonblocking
[`isend`](@ref TreeAMR.isend) / [`irecv`](@ref TreeAMR.irecv) /
[`waitall`](@ref TreeAMR.waitall).

[`SerialCommunicator`](@ref TreeAMR.SerialCommunicator) is the default.
An application does not construct one of these itself: it passes its own
communicator as `Forest(…; comm)`, and [`communicator`](@ref) converts it
— an `MPI.Comm` once MPI is loaded beside TreeAMR. A subtype that
implements only some verbs is refused at the first one it lacks, with
the verb named; see "Distributed meshes" in `CODE.md`.
"""
abstract type Communicator end

"""
    SerialCommunicator()

The communicator of a forest that is not distributed: rank 0 of 1. Its
collectives return the caller's own contribution and it has no peer to
send to. Every forest built without a `comm` keyword has one.
"""
struct SerialCommunicator <: Communicator end

Base.show(io::IO, ::SerialCommunicator) = print(io, "SerialCommunicator()")

"""
    communicator(comm) -> TreeAMR.Communicator
    communicator(comm::MPI.Comm; deviceaware = false)

The package's view of an application's communicator: what a
[`Forest`](@ref) stores for its `comm` keyword. A
[`TreeAMR.Communicator`](@ref TreeAMR.Communicator) is returned as it is,
and `nothing` — the keyword's default — is a
[`SerialCommunicator`](@ref TreeAMR.SerialCommunicator).
With MPI.jl loaded beside TreeAMR, an `MPI.Comm` such as
`MPI.COMM_WORLD` is converted by the MPI extension, which loads with
MPI. Three things hold for it (see "Distributed meshes" in `CODE.md`):

- The package's messages travel over a *duplicate* of the communicator,
  so they can never match the application's. `MPI_Comm_dup` is
  collective, so the duplicate is made once per communicator and cached
  for the life of the process: every forest over `MPI.COMM_WORLD` shares
  one. It is never freed, since `MPI_Comm_free` is collective too and a
  finalizer runs at a different moment on each rank.
- MPI must be initialized, at `MPI.THREAD_SERIALIZED` or better, which
  is what `MPI.Init()` asks for by default. The package calls MPI from
  the calling task only, never from a threaded loop or a kernel, but a
  Julia task can move between OS threads between two calls.
- Everything over such a forest that communicates is collective: every
  rank makes the same calls, in the same order, from one task at a time.
  That covers the forest mutations, the schedule builds, the ghost fill,
  the interface restriction and the reductions.
- A field set on a device exchanges through host mirrors of its message
  buffers unless the communicator is made with
  `communicator(comm; deviceaware = true)`, which hands MPI the device
  buffers themselves. Say so only for an MPI that reads the device's
  memory (a CUDA-aware MPI for CUDA): a wrong `true` is a crash inside
  the library, a wrong `false` two copies per message. `MPI.has_cuda()`
  answers for Open MPI only, and for MPICH only through the environment
  variable `JULIA_MPI_HAS_CUDA`, so it is a hint for the caller to pass,
  not a default. The ranks need not agree on it, since the bytes sent
  are the same either way.

```julia
# Field sets on CUDABackend(), and an MPI that may be CUDA-aware:
forest = Forest((4, 4); N = 8,
                comm = communicator(MPI.COMM_WORLD; deviceaware = MPI.has_cuda()))
```
"""
communicator(comm::Communicator) = comm
communicator(::Nothing) = SerialCommunicator()
communicator(comm) = throw(ArgumentError(
    "$(typeof(comm)) is not a communicator TreeAMR knows: pass nothing for a " *
    "serial forest, an MPI.Comm with MPI.jl loaded beside TreeAMR (which loads " *
    "its MPI extension), or a TreeAMR.Communicator."))

# A verb a `Communicator` subtype does not implement. A test-only
# communicator that answers rank and size, say, is enough for the
# partition, and anything that would need a message from it says so
# here instead of failing with a `MethodError` somewhere inside a
# reduction.
missing_verb(comm::Communicator, verb::Symbol) = throw(ArgumentError(
    "$(typeof(comm)) does not implement `$verb`, which this operation needs: " *
    "a forest distributed over a communicator exchanges ghosts, reduces and " *
    "regrids through the verbs `commrank`, `commsize`, `allgather`, " *
    "`allgatherv`, `alltoallv`, `isend`, `irecv` and `waitall`, and a " *
    "communicator answering only some of them supports only what needs no more."))

"""
    commrank(comm) -> Int

This process's rank in `comm`, `0:commsize(comm) - 1`.
"""
commrank(::SerialCommunicator) = 0
commrank(comm::Communicator) = missing_verb(comm, :commrank)

"""
    commsize(comm) -> Int

The number of processes in `comm`.
"""
commsize(::SerialCommunicator) = 1
commsize(comm::Communicator) = missing_verb(comm, :commsize)

"""
    allgather(comm, x) -> Vector

The `isbits` value `x` of every rank, in rank order, on every rank:
collective. It is what a reduction is built from (`combine_blocks`
gathers one partial per rank and folds them in rank order), so that the
association is the package's and the same on every rank.
"""
function allgather(::SerialCommunicator, x)
    isbits(x) || throw(ArgumentError(
        "allgather sends one isbits value per rank, got a $(typeof(x))"))
    return [x]
end
allgather(comm::Communicator, x) = missing_verb(comm, :allgather)

"""
    allgatherv(comm, v::AbstractVector) -> Vector

The concatenation, in rank order, of every rank's vector `v`, on every
rank: collective. The lengths may differ between ranks, and may be zero.
"""
allgatherv(::SerialCommunicator, v::AbstractVector) = collect(v)
allgatherv(comm::Communicator, v::AbstractVector) = missing_verb(comm, :allgatherv)

"""
    alltoallv(comm, sendbuf::AbstractVector, sendcounts) -> (recvbuf, recvcounts)

Every rank sends the next `sendcounts[s + 1]` elements of `sendbuf` to
rank `s`, and receives the concatenation, in rank order, of what every
rank sent it: collective. Returns the received elements and their
counts per source rank.
"""
function alltoallv(::SerialCommunicator, sendbuf::AbstractVector,
                   sendcounts::AbstractVector{<:Integer})
    length(sendcounts) == 1 && only(sendcounts) == length(sendbuf) || throw(ArgumentError(
        "a serial communicator has one rank, so sendcounts has one entry, the " *
        "length of sendbuf ($(length(sendbuf))); got $sendcounts"))
    return collect(sendbuf), Int[length(sendbuf)]
end
alltoallv(comm::Communicator, sendbuf::AbstractVector, sendcounts) =
    missing_verb(comm, :alltoallv)

# A serial rank has no peer: the exchange never sends to its own rank,
# since a transfer with both ends on one rank is a local group. So the
# serial point-to-point verbs refuse any message, and `waitall` is
# reached only with nothing to wait for.
no_peer(peer) = throw(ArgumentError(
    "a serial communicator has rank 0 only, so there is no peer $peer to " *
    "exchange with: a transfer with both ends on one rank is a local group and " *
    "never a message"))

"""
    isend(comm, buf::AbstractVector, peer::Integer, tag::Integer) -> request

Start sending the flat buffer `buf` to rank `peer` under `tag`, without
waiting; [`waitall`](@ref TreeAMR.waitall) completes it, and `buf` must
not be touched before then.
"""
isend(::SerialCommunicator, buf::AbstractVector, peer::Integer, tag::Integer) =
    no_peer(peer)
isend(comm::Communicator, buf::AbstractVector, peer::Integer, tag::Integer) =
    missing_verb(comm, :isend)

"""
    irecv(comm, buf::AbstractVector, peer::Integer, tag::Integer) -> request

Start receiving into the flat buffer `buf` from rank `peer` under `tag`,
without waiting; the data are there once [`waitall`](@ref TreeAMR.waitall)
has returned.
"""
irecv(::SerialCommunicator, buf::AbstractVector, peer::Integer, tag::Integer) =
    no_peer(peer)
irecv(comm::Communicator, buf::AbstractVector, peer::Integer, tag::Integer) =
    missing_verb(comm, :irecv)

"""
    waitall(comm, requests::AbstractVector)

Wait until every request of [`isend`](@ref TreeAMR.isend) and
[`irecv`](@ref TreeAMR.irecv) in `requests` has completed.
"""
function waitall(::SerialCommunicator, requests::AbstractVector)
    isempty(requests) || throw(ArgumentError(
        "a serial communicator starts no messages, so there is nothing to wait for"))
    return nothing
end
waitall(comm::Communicator, requests::AbstractVector) = missing_verb(comm, :waitall)

"""
    hoststaging(comm, buffer::AbstractVector) -> Bool

Whether a stage buffer allocated on a device has to pass through a host
mirror before `comm` can send it or receive into it (M7, step 8). The
exchange packs on the field set's backend; when this answers `true` it
copies the packed buffer to a host vector of the same layout before
[`isend`](@ref TreeAMR.isend), and the received host vector to the
device before unpacking, so that the communicator only ever sees host
memory. When it answers `false` the device buffer itself is handed over,
which needs a device-aware MPI.

A host `Array` never needs staging, so the CPU path is the direct one.
Any other buffer is staged unless the communicator says it can take it:
that is the safe default, since a library handed device memory it cannot
read fails, at best, while staging costs two copies. The MPI extension
answers `false` for a device buffer only when its communicator was made
with `communicator(comm; deviceaware = true)`. See "MPI+GPU" under
"Distributed meshes" in `CODE.md`.
"""
hoststaging(::Communicator, buffer::AbstractVector) = !(buffer isa Array)

# The library communicator underneath, for a library that has to be
# handed one itself: parallel HDF5 opens a checkpoint over it (step 6 of
# M7). The MPI extension returns its duplicate; no other communicator has
# one, and the refusal says what the caller needs instead.
librarycomm(comm::Communicator) = throw(ArgumentError(
    "$(typeof(comm)) has no MPI communicator underneath it, and a checkpoint of a " *
    "forest distributed over it would need one: every rank opens the one file " *
    "through parallel HDF5, over MPI. Build the forest with `comm = " *
    "MPI.COMM_WORLD` (or another MPI.Comm), with MPI.jl and HDF5.jl loaded."))

# --- The message-buffer pool (M7) ------------------------------------------
#
# A stage's send and receive buffers, and their host mirrors where the
# messages are staged, are allocated the first time the stage runs and
# kept with it, so a steady run of fills allocates none. A regrid builds
# a new stage every time, though, and the schedules an application
# rebuilds after it start without buffers: so before the pool, every
# regrid of a distributed forest allocated and zero-filled the buffers of
# its transfer stage and of the next schedule's first fill afresh, and,
# when staging, allocated and page-locked their mirrors, which is slow on
# a device's host side (CUDA's `cuMemHostRegister`; measured on
# Symmetry's H200s at 48 ms a staged regrid against 8.7 ms direct, M7
# step 8). The pool keeps that memory and hands it out again.
#
# - It belongs to the forest (`bufferpool`), which outlives every
#   schedule and regrid over it, and is reached from every place that
#   stages: `run_stage!` has the forest. It is made on first use, and a
#   serial forest, which has no stage with messages, never makes one.
# - A *lease* is a buffer of at least the length asked for, handed out as
#   an object of exactly that length over the pooled memory: for a host
#   vector, an `Array` over the same `Memory` (`Base.wrap`), so that it
#   is an `Array` as before — what `hoststaging` and the MPI extension's
#   buffer check test for; for a device array, a contiguous `view`,
#   which CUDA, Metal and the other GPUArrays backends return as an
#   array of the parent's own type. Where a view is not of that type the
#   lease is an ordinary allocation, not pooled.
# - A lease returns to the pool in two ways. The regrid stage, which
#   lives for one call, gives its leases back once its sends have been
#   waited on (`release_stage!`). A schedule's leases are reclaimed once
#   the forest's generation has moved past the one they were leased at —
#   or their holder has been collected — since a stale schedule can never
#   run again (`fill_ghosts!`, `restrict_interfaces!` and `regrid!` refuse
#   one) and every message of the call it last ran in was waited on in
#   that call. Reclaiming deletes the lease from the holder's dictionary
#   too, so that a stale stage asked for its buffers directly (the tests
#   and `bench/mpi.jl` do, through internals) takes new ones rather than
#   memory that is now someone else's. A buffer is therefore never handed
#   out twice at once: it is either free or held by one lease.
# - Best fit: a request takes the smallest free buffer that is long
#   enough. If none is, the smallest free one is dropped (for the GC to
#   free, and CUDA to unpin) and a new one allocated with a quarter's
#   headroom, so the number of buffers is bounded by the most ever leased
#   at once, and their sizes follow the largest requests. Nothing shrinks
#   otherwise: memory kept while unused is the price of not allocating it
#   again (CODE.md, "MPI+GPU").
# - A fresh buffer is zero-filled, as the stage buffers always were; a
#   reused one holds an earlier stage's bytes, which nothing reads, since
#   every element of a send buffer is packed and every element of a
#   receive buffer is received before it is unpacked.
#
# Exchanges over one forest run from one task at a time (the collective
# contract), but the pool is locked anyway: a lease is rare, and the lock
# costs nothing beside the allocation it saves.

# A pooled allocation: `full`, what was allocated (and page-locked, for a
# host mirror), and its `Memory` for a host vector, through which a
# shorter `Array` over it is made; `nothing` for a device array, which is
# sliced with `view`.
struct PooledBuffer
    full::AbstractVector
    mem::Union{Nothing,Memory}
end

capacity(p::PooledBuffer) = length(p.full)

lease_slice(p::PooledBuffer, n::Int) =
    p.mem === nothing ? view(p.full, 1:n) : Base.wrap(Array, p.mem, (n,))

# One buffer handed out: from which free list, at which forest
# generation, and to which holder — the stage's dictionary of buffers or
# of mirrors, weakly, under the key `nvars`.
struct Lease
    buffer::PooledBuffer
    key::Any
    generation::Int
    holder::WeakRef
    nvars::Int
end

mutable struct BufferPool
    const lock::ReentrantLock
    const free::Dict{Any,Vector{PooledBuffer}}
    const leased::IdDict{Any,Lease}              # the handed-out object => its lease
    allocated::Int          # buffers allocated, over the forest's life
    pagelocked::Int         # of them, host mirrors page-locked for a device
    dropped::Int            # free buffers dropped for a larger one
end

BufferPool() = BufferPool(ReentrantLock(), Dict{Any,Vector{PooledBuffer}}(),
                          IdDict{Any,Lease}(), 0, 0, 0)

Base.show(io::IO, p::BufferPool) =
    print(io, "BufferPool(", length(p.leased), " leased, ",
          sum(length, values(p.free); init=0), " free, ", p.allocated, " allocated)")

# Return every lease whose holder is stale (leased at another generation)
# or gone to its free list, deleting it from a living holder. Called
# with the lock held.
function reclaim_stale!(pool::BufferPool, generation::Int)
    stale = Any[]
    for (obj, l) in pool.leased
        holder = l.holder.value
        if l.generation != generation || holder === nothing
            holder === nothing || delete!(holder, l.nvars)
            push!(stale, obj)
        end
    end
    for obj in stale
        l = pop!(pool.leased, obj)
        push!(get!(() -> PooledBuffer[], pool.free, l.key), l.buffer)
    end
    return nothing
end

# A buffer of `n` elements for `holder[nvars]`, at the forest generation
# `generation`. `make(cap)` allocates a fresh `PooledBuffer` of capacity
# `cap`, zero-filled; `A` is the type the lease must have. A zero-length
# request is not pooled: there is nothing to keep.
function lease!(make, ::Type{A}, pool::BufferPool, key, n::Int, generation::Int,
                holder::AbstractDict, nvars::Int) where {A}
    n == 0 && return make(0).full::A
    return lock(pool.lock) do
        reclaim_stale!(pool, generation)
        free = get!(() -> PooledBuffer[], pool.free, key)
        best = 0
        for (i, p) in enumerate(free)
            capacity(p) >= n && (best == 0 || capacity(p) < capacity(free[best])) &&
                (best = i)
        end
        if best == 0
            if !isempty(free)
                deleteat!(free, argmin(capacity.(free)))
                pool.dropped += 1
            end
            buffer = make(n + n ÷ 4)
            pool.allocated += 1
        else
            buffer = free[best]
            deleteat!(free, best)
        end
        obj = lease_slice(buffer, n)
        if !(obj isa A)
            # A backend whose contiguous view is not an array of its own
            # type: hand out an ordinary allocation instead.
            push!(free, buffer)
            return make(n).full::A
        end
        pool.leased[obj] = Lease(buffer, key, generation, WeakRef(holder), nvars)
        return obj
    end
end

# Give the leases among `objs` back to the pool; anything else (a buffer
# allocated outside it) is left to the GC.
function release!(pool::BufferPool, objs)
    lock(pool.lock) do
        for obj in objs
            l = pop!(pool.leased, obj, nothing)
            l === nothing && continue
            push!(get!(() -> PooledBuffer[], pool.free, l.key), l.buffer)
        end
    end
    return nothing
end
