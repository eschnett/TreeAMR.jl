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
# The MPI methods will live in a package extension, `TreeAMRMPIExt`,
# for the reason HDF5 does: an application that never runs distributed
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

The package's view of an application's communicator: what a
[`Forest`](@ref) stores for its `comm` keyword. A
[`TreeAMR.Communicator`](@ref TreeAMR.Communicator) is returned as it is,
and `nothing` — the keyword's default — is a
[`SerialCommunicator`](@ref TreeAMR.SerialCommunicator).
With MPI.jl loaded beside TreeAMR, an `MPI.Comm` such as
`MPI.COMM_WORLD` is converted by the MPI extension (M7, from its step 3
on), which duplicates it once per communicator so that the package's
messages can never match the application's.
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
