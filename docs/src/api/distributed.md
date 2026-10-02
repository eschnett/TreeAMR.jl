# Distributed meshes

M7 runs one forest over several processes. Every rank holds the whole
forest, and only field data are distributed: each rank stores the blocks
of one contiguous run of the leaves, and every block index is local to
the rank. The design is "Distributed meshes" in `CODE.md`; M7 is being
implemented in steps, and so far the partition, the local block
indexing, the distributed schedule, the ghost exchange and the interface
restriction over MPI, the reductions across ranks, the regrid and point
interpolation are in place. Checkpoints still refuse a distributed
forest.

A distributed run loads MPI.jl beside TreeAMR, which loads the package's
MPI extension, and passes its communicator to the forest:

```julia
using MPI, TreeAMR
MPI.Init()
forest = Forest((4, 4); N = 8, comm = MPI.COMM_WORLD)
```

Every forest mutation and every schedule build is then collective. A
[`GhostSchedule`](@ref) or [`InterfaceSchedule`](@ref) built over forests
that differ between ranks — a `refine!` made on one rank only, say — is
refused on every rank together, with the reason, as is one built for a
different layout on some rank. [`interpolate`](@ref) is collective too:
every rank passes the same field set and arguments and its own points,
any number of them, and gets their values back in its own order.

```@docs
communicator
blockrange
```

## Communicators

Not exported: an application passes its own communicator as
`Forest(…; comm)`, and the package talks to it through these verbs, each
of which has a serial method.

```@docs
TreeAMR.Communicator
TreeAMR.SerialCommunicator
TreeAMR.commrank
TreeAMR.commsize
TreeAMR.allgather
TreeAMR.allgatherv
TreeAMR.alltoallv
TreeAMR.isend
TreeAMR.irecv
TreeAMR.waitall
```

## The distributed schedule

Not exported. Over a distributed forest a [`GhostSchedule`](@ref) or an
[`InterfaceSchedule`](@ref) holds its exchange as stages, one per
ordering point of the serial one, and each stage what this rank computes
for another rank's targets and what it receives for its own, with the
layout both ends of each message derive from the replicated forest.
Serially every stage is a phase of the serial exchange and sends nothing.

```@docs
TreeAMR.ExchangeStage
TreeAMR.RemoteStage
TreeAMR.LayoutEntry
```
