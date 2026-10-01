# Distributed meshes

M7 runs one forest over several processes. Every rank holds the whole
forest, and only field data are distributed: each rank stores the blocks
of one contiguous run of the leaves, and every block index is local to
the rank. The design is "Distributed meshes" in `CODE.md`; M7 is being
implemented in steps, and so far the partition and the local block
indexing are in place, with no messages yet.

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
