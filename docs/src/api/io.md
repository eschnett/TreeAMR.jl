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
application's own `fill_ghosts!`. The file layout, format version 1, is
specified in
[CODE.md](https://github.com/eschnett/TreeAMR.jl/blob/main/CODE.md#checkpoint-and-restart),
"Checkpoint and restart", together with the reasons for it.

```@docs
save_checkpoint
load_checkpoint
write_plain
read_plain
checkpoint_environment
```
