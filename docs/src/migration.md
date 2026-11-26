# Moving to closed blocks (0.3)

TreeAMR 0.3 changes how a vertex-like dimension is owned (M22, "Closed
blocks" in `CODE.md`). Under half-open ownership a block owned the
points `G+1 … G+N` of a vertex-like dimension and its high face plane
was a ghost; now it owns its **closed range**, `G+1 … G+N+1`, both face
planes, and a point on a block face is a *shared point* with a copy on
every block holding it, which the fill makes one number. Cell-centered
and element sets (`c = 0`) are unchanged, except where this page says.
This page lists every change an application or a companion package has
to make, in one place.

## The state and its size

- A block's state is its closed box: `∏_d (N + c_d)` points a variable,
  with `c_d = 1` along a vertex-like dimension (`staggers(fs)`), where
  it was `N^D`. Size arrays and loops by `N .+ staggers(fs)`, never by
  `N`. `statelength`, `statevector`, `statearray` (a plain `reshape` to
  `(N+c_1, …, N+c_D, nvars, nblocks)`), `scatter!` and `gather!` follow.
- `interiorview` is the closed box. `map_blocks!` launches over it,
  every copy of every shared point; its `closed` keyword is gone, and
  passing it is refused with the reason.
- The invariant is `N ≥ 2G` in every dimension (it was `N ≥ 2G + 2`
  along a stagger).
- Form every position through `nodeposition` (with `block_lows` and
  `block_bases`, or `block_origins` and `block_spacings`), so that the
  copies of a point are at one position bit for bit; a right-hand side
  that forms `origin + (I − 1) h` itself gives the copies positions that
  differ in the last bit.

## Each point once: `primaries`

- `block_mapreduce`, `block_fold`, `mesh_mapreduce` and `mesh_fold`
  over a set with a vertex-like dimension require `primaries = true`
  (each point once, at its primary copy) or `false` (every copy); there
  is no default. `max`, `min` and their like are right either way; a sum
  or a count needs `true`.
- `volume_weighted_norm` and `total_mass` need no keyword: they take
  the trapezoid's weights within each block, half on a face plane, so a
  shared point has its weight once in total.
- `primary_mask(fs)` and `isprimary(mask, i…, b)` say which copies are
  primaries. `map_blocks!` does not skip secondaries; a kernel that
  skips them through the mask must call `unify!(du, …)` afterwards,
  since the entries it leaves are not defined.

## Walls

- The `walls` keyword of `FieldSet` and `GhostSchedule` is gone, and so
  are symmetric walls and the derived wall row: the wall plane is an own
  point on both sides of a domain, evolved by default, the boundary
  hook filling the `G` ghosts beyond. A Dirichlet condition writes the
  wall too: `CellBoundary(g; wall = :prescribed)` (and
  `boundary_by_coordinates(f; wall = :prescribed)`), which writes the
  wall's primaries before the copies are unified. A downstream's own
  wall work-around (TreeGRRMHD's `walls_to_state!`, a zeroed wall
  `du`) goes.
- At a reflecting face the wall plane is projected onto each variable's
  parity: an odd variable is zero there after every fill.

## Unifying: `unify!` and `UnifySchedule`

- `restrict_interfaces!(flux, isched)` and `InterfaceSchedule` are gone:
  the interface restriction of a conservative scheme is
  `unify!(flux, UnifySchedule(flux))`, which gives every secondary copy
  of a face value its primary's (the fine faces' area-weighted sum at a
  coarse-fine face). It takes no `Operators`.
- `unify!(v, fs, schedule)` applies the same to a state vector; it is
  optional inside a run (the fill unifies the working array on every
  call, so no right-hand side reads a stale copy) but keeps a state's
  secondaries equal to their primaries, and is required after a
  right-hand side that skips secondaries.
- **Unify a state before measuring it.** The state vector an integrator
  returns holds each secondary as integrated, with its own right-hand
  side, and only the next fill overwrites it. An error norm, an output,
  a symmetry check or a comparison of two runs taken straight from
  `sol.u[end]` reads those stale copies: at a coarse-fine face of a
  vertex set the coarse copy's error is the coarse stencil's, and
  TreeWave's sine wave measured an L∞ error of `O(1)` there until its
  drivers unified first, `unify!(copy(u), fs, schedule)`. Across a glue
  the set is interpolated at, the same call makes each glued point's
  copies their mean (stage A).
- A hand-written loop over `interiorview` that counts or sums visits
  every copy; skip the secondaries with `isprimary(primary_mask(fs), i…,
  b)`, as `primaries = true` does.
- After `regrid!`: rebuild the schedule, then fill or `unify!`, then
  `gather!`. `regrid!` leaves the unification to the next fill.

## Seams, glues and pairs

- On a rotating seam (a quarter turn) a set whose layout is not
  symmetric in the seam's plane and is vertex-like along either of its
  dimensions — a flux normal to one of them, with or without ghosts —
  has its seam copies in its partner set, so it is filled, unified and
  regridded only as a `RotationPair`; `unify!` of such a set alone is
  refused ("Unify it as a RotationPair"). Before 0.3 a flux without
  ghosts needed no partner.
- A seam point's copies are one number through the turn, and a point on
  the axis is projected onto the turn's invariant part (a turned
  vector's in-plane components are zero there).
- Across a glue that copies, the lowest-numbered patch's copy is the
  primary; a glue may now reverse a vertex-like axis and join a high
  face to a high face. A set vertex-like along both dimensions of a
  patch corner (three or more patches meeting) is refused, whatever its
  ghost widths; interpolate it across its glues instead.
- Across a glue a set is interpolated at (`glue = :interpolate`), every
  patch holding a point evolves its own copy, and the fill makes the
  copies their mean (stage A). M20's appendix is gone:
  `statearray(u, fs; appendix = true)` and `map_blocks!(…; appendix)`
  are refused with the reason, and `isowned`, `hasappendix`,
  `corelength`, `appendix_entries`, `AppendixArray` and the fold masks
  no longer exist; read the closed state as any other, and count with
  `primaries = true`.

## Regridding

- `firing_boxes` sweeps the closed box, so a point on a face fires in
  every block holding it, and along a vertex-like dimension it reports
  the cells touching the firing nodes: flag boxes stay in cell indices
  `1:N`. A box an application builds by hand from a vertex-like
  dimension's nodes must likewise be the cells touching them.

## Smaller changes

- The fill runs its stages as A (kinked glues), W (prescribed walls),
  P (wall planes and a seam's axis), 0 (secondaries), 1, the hook, the
  interpolated glue ghosts, then 2; the boundary hook sees outer faces
  only.
- A zero that a seam's turn or a wall's parity negates is `+0`, not
  `−0`.
- A cell-centered set's numbers are unchanged; a vertex-like set's
  change where half-open ownership made a mesh asymmetric (refined
  regions, walls, seams), by the scheme's error there.
- Checkpoints: TreeIOHDF5 stores a vertex-like set's closed blocks under
  the format feature `closed_blocks`, and refuses, saying why, a file
  written before 0.3 that holds such a set (M20's appendix and AMR-8's
  symmetric walls included); its cell-like sets still load through
  `load_checkpoint(…; fieldsets)`. Cell and element files load as
  before. Whether a half-open file can be converted is open.
