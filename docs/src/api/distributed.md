# Distributed meshes

M7 runs one forest over several processes. Every rank holds the whole
forest, and only field data are distributed: each rank stores the blocks
of one contiguous run of the leaves, and every block index is local to
the rank. The design is "Distributed meshes" in `CODE.md`; M7 is being
implemented in steps, and so far the partition, the local block
indexing and the distributed schedule are in place, with no messages
yet.

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
