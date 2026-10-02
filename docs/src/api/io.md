# Checkpoint and restart

Saving a run at a chunk boundary and resuming it, bit for bit, in a fresh
process on any thread count and any backend (M9a). HDF5 is a weak
dependency: these functions are implemented by the package extension
`TreeAMRHDF5Ext`, which loads with `using HDF5`, so that an application
that never checkpoints does not load HDF5 and its binaries. Without it
they have no methods, and the error says to load HDF5.

A checkpoint stores what cannot be recomputed — the forest's parameters
and leaf list, and the owned points of the field sets the application
evolves — plus the application's own run state as plain data. Everything
derived is rebuilt after a load: the schedules, and the ghosts, by the
application's own `fill_ghosts!`. The file layout, format version 2, is
specified in
[CODE.md](https://github.com/eschnett/TreeAMR.jl/blob/main/CODE.md#checkpoint-and-restart),
"Checkpoint and restart", together with the reasons for it; version 1,
which every version before M7's part files wrote, is still read.

Serially a checkpoint is one HDF5 file. Over a distributed forest it is
an index file at `path`, written by rank 0, and a part file per I/O
process beside it, `path.<saveid>.<j>.h5`; no file is written or opened
by more than one process, so nothing depends on parallel I/O ("Checkpoints
without parallel I/O" in `CODE.md`). The index holds an HDF5 external
link to each part, `/TreeAMR.jl/parts/<jjjj>`, so that `h5dump`, h5py or
HDFView can navigate from the index into the parts; TreeAMR's own loader
never follows them, and finds the parts from the index's part table.

```@docs
save_checkpoint
load_checkpoint
write_plain
read_plain
checkpoint_environment
```
