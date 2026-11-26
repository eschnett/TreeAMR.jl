# ODE coupling

The flat interiors-only state vector, per-block kernels over it, and the
reductions.

```@docs
statelength
staterange
statevector
stateview
statearray
scatter!
gather!
map_blocks!
block_mapreduce
mesh_mapreduce
volume_weighted_norm
```

A fold over points yields many results from one sweep: its callback sees
a whole point — the variables, the position, the block's spacing, level
and tag — and folds it into any `isbits` accumulator, such as bins of
norms by region.

```@docs
mesh_fold
block_fold
TreeAMR.FoldPoint
Sum
Max
Min
TreeAMR.combine
addto
```
