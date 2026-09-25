# Internals

Not exported, and not part of the public interface, but documented
because they define the shape of the schedule.

```@docs
TreeAMR.Stencil1D
TreeAMR.TransferGroup
TreeAMR.BoundaryRegion
TreeAMR.BoundaryBatch
TreeAMR.BoundaryPlan
TreeAMR.lagrange_weights
TreeAMR.unit_lagrange_weights
TreeAMR.ghost_layers_read
```

The host-side threading primitives, for the same reason — the shape of
every parallel pass over blocks in the package:

```@docs
TreeAMR.threadchunks
TreeAMR.threaded_foreach
TreeAMR.threaded_chunks
TreeAMR.threaded_collect
TreeAMR.launch_by_owner!
```

The device-residency helpers behind the `backend` keyword:

```@docs
TreeAMR.todevice
TreeAMR.tohost
```
