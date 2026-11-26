# Distributed meshes

M7 runs one forest over several processes. Every rank holds the whole
forest, and only field data are distributed: each rank stores the blocks
of one contiguous run of the leaves, and every block index is local to
the rank. The design is "Distributed meshes" in `CODE.md`, and the guide
to writing a distributed application is
[Running distributed](@ref) on the home page. Everything a serial run
does works across ranks: the ghost exchange for every centering, the
interface restriction, the reductions, the regrid with the
repartitioning it implies, and point interpolation; checkpoints too,
through the companion package TreeIOHDF5.

A distributed run loads MPI.jl beside TreeAMR, which loads the package's
MPI extension, and passes its communicator to the forest:

```julia
using MPI, TreeAMR
MPI.Init()
forest = Forest((4, 4); N = 8, comm = MPI.COMM_WORLD)
```

Every forest mutation and every schedule build is then collective, and
every block index of a field set is local: [`blockkey`](@ref)`(fs, b)`
names block `b`'s leaf, and [`nblocks`](@ref)`(fs)` sizes a per-block
array, while [`nleaves`](@ref) and `forest.leaves` are the whole mesh. A
[`GhostSchedule`](@ref) or [`UnifySchedule`](@ref) built over forests
that differ between ranks — a `refine!` made on one rank only, say — is
refused on every rank together, with the reason, as is one built for a
different layout on some rank. [`interpolate`](@ref) is collective too:
every rank passes the same field set and arguments and its own points,
any number of them, and gets their values back in its own order.

Field sets on a device backend work the same way. Their message buffers
live on the device, and by default every message is staged through
page-locked host mirrors, which any MPI can send. An MPI that reads
device memory — a CUDA-aware one, for CUDA — can be handed the device
buffers instead, which the application says when it converts its
communicator; see [`communicator`](@ref):

```julia
forest = Forest((4, 4); N = 8,
                comm = communicator(MPI.COMM_WORLD; deviceaware = true))
```

```@docs
communicator
blockrange
```

## Combining values over the ranks

An application passes its own communicator as `Forest(…; comm)`, and
the package talks to it through verbs, each of which has a serial
method. Four of them an application may call too, on
[`communicator`](@ref)`(forest)`, to combine a host value of its own
over the ranks — a diagnostic's sum, or a decision taken from the clock
or the file system that has to be the same on every rank. The two
collectives return the same vector on every rank, in rank order, so a
fold over it is the same on every rank too; serially they return the
caller's own contribution. They are public but not exported, since
their names are generic: call them qualified, as `TreeAMR.allgather`,
or import them by name.

```julia
comm = communicator(forest)
stop = any(TreeAMR.allgather(comm, time() > deadline))   # the same on every rank
```

```@docs
TreeAMR.Communicator
TreeAMR.SerialCommunicator
TreeAMR.commrank
TreeAMR.commsize
TreeAMR.allgather
TreeAMR.allgatherv
```

## Communicators

Not public: the verbs the package uses for itself — the all-to-all of
point interpolation, the messages of the exchange, and the two that
only TreeIOHDF5's checkpoints use. Their messages travel on the
package's own duplicate of the application's communicator, so a
message an application sent there could match one of the package's.

```@docs
TreeAMR.alltoallv
TreeAMR.bcast
TreeAMR.commnodes
TreeAMR.isend
TreeAMR.irecv
TreeAMR.waitall
TreeAMR.hoststaging
```

## The distributed schedule

Not exported. Over a distributed forest a [`GhostSchedule`](@ref) or a
[`UnifySchedule`](@ref) holds its exchange as stages, one per
ordering point of the serial one, and each stage what this rank computes
for another rank's targets and what it receives for its own, with the
layout both ends of each message derive from the replicated forest.
Serially every stage is a phase of the serial exchange and sends nothing.

```@docs
TreeAMR.ExchangeStage
TreeAMR.RemoteStage
TreeAMR.LayoutEntry
```
