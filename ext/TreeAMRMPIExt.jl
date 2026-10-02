# Distributed meshes over MPI (M7): the MPI methods of the communicator
# verbs declared, and documented, in `src/communicator.jl`.
#
# The package needs very little of MPI ("Distributed meshes" in
# `CODE.md`): its rank and size, three collectives — an allgather of one
# `isbits` value, which every reduction and the forest digest are built
# from, an allgatherv and an alltoallv — and nonblocking point-to-point
# messages over flat buffers. Each verb below is one MPI.jl call over a
# *duplicate* of the application's communicator, so that the package's
# messages, whose tags are its own, can never match the application's.
#
# Three rules shape this file:
#
# - One duplicate per application communicator. `MPI_Comm_dup` is
#   collective, and so is `MPI_Comm_free`, which therefore cannot run
#   from a finalizer: a finalizer runs at a different moment on each
#   rank. So the duplicate is made without the finalizer MPI.jl's
#   `Comm_dup` attaches, cached per communicator for the life of the
#   process, and never freed (MPICH_jll 5.0.2 gives a process 2046
#   duplicates, measured 2026-10-01, so a test that builds a thousand
#   forests must not make a thousand).
# - MPI is called from the calling task only, never from a threaded loop
#   or a kernel. A Julia task may still move between OS threads between
#   two calls, so `MPI_THREAD_SERIALIZED` is required, which is
#   `MPI.Init()`'s default.
# - Buffers are flat and contiguous: an `Array` or a `UnitRange` view of
#   one, which MPI.jl hands to the library as a pointer and a count, not
#   as a derived datatype. A device buffer is handed over only by a
#   communicator made with `deviceaware = true` (step 8 of M7); every
#   other one has the exchange stage its messages through host mirrors
#   (`hoststaging`), so the library sees host memory only.

module TreeAMRMPIExt

using MPI: MPI
using TreeAMR
using TreeAMR: Communicator
import TreeAMR: communicator, commrank, commsize, allgather, allgatherv, alltoallv,
                isend, irecv, waitall, librarycomm, hoststaging

# A forest's communicator over MPI: the duplicate, with its rank and size
# read once, since neither can change, and whether the caller said the
# library reads device memory (`deviceaware`), which is the caller's to
# know: see `communicator`'s docstring and "MPI+GPU" in `CODE.md`.
struct MPICommunicator <: Communicator
    comm::MPI.Comm
    rank::Int
    size::Int
    deviceaware::Bool
end

Base.show(io::IO, c::MPICommunicator) =
    print(io, "MPICommunicator(rank ", c.rank, " of ", c.size,
          c.deviceaware ? ", device-aware)" : ")")

# The duplicates made so far, by the handle of the communicator they
# duplicate. Guarded by a lock, since forests may be built from several
# tasks; the duplication itself is collective, so the contract that
# every rank builds its forests in the same order covers it.
const DUPLICATES = Dict{Any,MPICommunicator}()
const DUPLICATES_LOCK = ReentrantLock()

function communicator(comm::MPI.Comm; deviceaware::Bool=false)
    MPI.Initialized() && !MPI.Finalized() || throw(ArgumentError(
        "MPI is " * (MPI.Finalized() ? "already finalized" : "not initialized") *
        ": a forest over an MPI communicator duplicates it, which is an MPI call. " *
        "Call `MPI.Init()` before building the forest, and build none after " *
        "`MPI.Finalize()`."))
    comm == MPI.COMM_NULL && throw(ArgumentError(
        "MPI.COMM_NULL has no ranks to distribute a forest over; a process outside " *
        "a communicator's group builds its forest without `comm` (serially)."))
    provided = MPI.Query_thread()
    provided >= MPI.THREAD_SERIALIZED || throw(ArgumentError(
        "MPI was initialized at thread level $provided, but TreeAMR needs " *
        "MPI.THREAD_SERIALIZED or better: it calls MPI only from the calling task, " *
        "but a Julia task can move between OS threads between two calls, which " *
        "a lower level forbids. `MPI.Init()` asks for THREAD_SERIALIZED by default; " *
        "pass `threadlevel = :serialized` (or `:multiple`) if you initialize " *
        "with another."))
    return lock(DUPLICATES_LOCK) do
        cached = get(DUPLICATES, comm.val, nothing)
        # A handle can be reused once the application frees its
        # communicator. The cached duplicate is still right if its group
        # is the same ranks in the same order (a duplicate compares
        # CONGRUENT with its original); the comparison is local, and its
        # answer is the same on every rank of the group.
        if cached === nothing ||
           !(MPI.Comm_compare(cached.comm, comm) in (MPI.CONGRUENT, MPI.IDENT))
            dup = MPI.Comm()
            MPI.API.MPI_Comm_dup(comm, dup)
            cached = MPICommunicator(dup, MPI.Comm_rank(dup), MPI.Comm_size(dup), false)
            DUPLICATES[comm.val] = cached
        end
        # The setting is not part of the duplicate: a forest made with it
        # and one made without share the one duplicate, as every forest
        # over the communicator does.
        return MPICommunicator(cached.comm, cached.rank, cached.size, deviceaware)
    end
end

commrank(c::MPICommunicator) = c.rank
commsize(c::MPICommunicator) = c.size

# The duplicate, for parallel HDF5, which opens a checkpoint over it and
# duplicates it once more itself (`TreeAMRHDF5MPIExt`).
librarycomm(c::MPICommunicator) = c.comm

function allgather(c::MPICommunicator, x)
    isbits(x) || throw(ArgumentError(
        "allgather sends one isbits value per rank, got a $(typeof(x))"))
    return MPI.Allgather(x, c.comm)
end

function allgatherv(c::MPICommunicator, v::AbstractVector)
    send = collect(v)
    counts = MPI.Allgather(length(send), c.comm)
    recv = similar(send, sum(counts))
    MPI.Allgatherv!(send, MPI.VBuffer(recv, counts), c.comm)
    return recv
end

function alltoallv(c::MPICommunicator, sendbuf::AbstractVector,
                   sendcounts::AbstractVector{<:Integer})
    length(sendcounts) == c.size && sum(sendcounts; init=0) == length(sendbuf) ||
        throw(ArgumentError(
            "sendcounts has one entry per rank ($(c.size)), summing to the length " *
            "of sendbuf ($(length(sendbuf))); got $sendcounts"))
    send = collect(sendbuf)
    scounts = Cint.(sendcounts)
    rcounts = MPI.Alltoall(MPI.UBuffer(scounts, 1), c.comm)
    recv = similar(send, sum(rcounts; init=0))
    MPI.Alltoallv!(MPI.VBuffer(send, scounts), MPI.VBuffer(recv, rcounts), c.comm)
    return recv, Int.(rcounts)
end

# A message buffer has to be one contiguous run of memory, which is what
# the exchange hands over: a stage buffer or a `UnitRange` view of one
# peer's segment of it. Anything else would reach MPI as a derived
# datatype, or not at all, so it is refused rather than sent. Host memory
# is what every communicator takes; a device-aware one also takes a
# dense device vector or a contiguous view of one, which MPI.jl hands to
# the library by its device pointer (through its CUDA or ROCm extension).
const HostBuffer = Union{Array,Base.FastContiguousSubArray{<:Any,1,<:Array}}
const DenseBuffer = Union{DenseVector,Base.FastContiguousSubArray{<:Any,1,<:DenseArray}}

# Whether a stage buffer goes through host mirrors: a host `Array` never
# does, a device buffer unless the caller said the library reads device
# memory.
hoststaging(c::MPICommunicator, buffer::AbstractVector) =
    !(buffer isa Array) && !c.deviceaware

flatbuffer(c::MPICommunicator, buf::HostBuffer) = buf
flatbuffer(c::MPICommunicator, buf::AbstractVector) =
    c.deviceaware && buf isa DenseBuffer ? buf : throw(ArgumentError(
        "a message buffer must be a contiguous host vector (an Array, or a UnitRange " *
        "view of one)" *
        (c.deviceaware ? ", or a dense device vector or a contiguous view of one" :
         "; a device buffer is sent directly only over a communicator made with " *
         "`communicator(comm; deviceaware = true)`, and otherwise staged through " *
         "host memory by the exchange") * ", got a $(typeof(buf))"))

checkpeer(c::MPICommunicator, peer) =
    0 <= peer < c.size && peer != c.rank || throw(ArgumentError(
        "rank $(c.rank) has no peer $peer: a message goes to another rank of the " *
        "$(c.size), and a transfer with both ends on one rank is a local group"))

function isend(c::MPICommunicator, buf::AbstractVector, peer::Integer, tag::Integer)
    checkpeer(c, peer)
    return MPI.Isend(flatbuffer(c, buf), c.comm; dest=Int(peer), tag=Int(tag))
end

function irecv(c::MPICommunicator, buf::AbstractVector, peer::Integer, tag::Integer)
    checkpeer(c, peer)
    return MPI.Irecv!(flatbuffer(c, buf), c.comm; source=Int(peer), tag=Int(tag))
end

function waitall(c::MPICommunicator, requests::AbstractVector)
    isempty(requests) && return nothing
    MPI.Waitall(convert(Vector{MPI.Request}, requests))
    return nothing
end

end
