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
- **Settle a state before reading it.** The state vector an integrator
  returns holds each secondary as integrated, with its own block's
  stencil; only the next fill overwrites it. At a coarse-fine face the
  fine primary's stencil reads a prolongated ghost and the coarse
  secondary's does not, so their `du` differ by `O(h^min(q, p−2))`
  (scheme order `q`, prolongation order `p`), which the secondary
  integrates with no feedback: on TreeWave's sine wave after a quarter
  period, π/2 at every `N` at `p = 2`, `O(h²)` at `p = 4` and 6. An
  error norm, `volume_weighted_norm` or `total_mass` of a state, an
  output, a checkpoint, a symmetry check or a `scatter!` into `regrid!`
  without a fill reads those stale copies. Unify first,
  `unify!(copy(u), fs, schedule)` (across an interpolated glue it makes
  the copies their mean); where the hook prescribes a wall, `unify!`
  cannot write it, so `scatter!`, `fill_ghosts!` with the hook and
  `gather!` instead. An adaptive integrator's error estimate is not
  affected (measured: the same steps raw or unified).
- **Decide what a Dirichlet hook does with the wall.** Under half-open
  ownership the low outer plane was evolved and the high one was a ghost
  the hook wrote; now both are own points, evolved unless the hook says
  `wall = :prescribed`. A port that keeps its hook unchanged evolves
  both. Where the data are the boundary condition itself, prescribe:
  TreeWave's closed ball, whose energy rose once by 7 % with an evolved
  sphere, follows its M20 history prescribed. Where the hook only
  supplies exact data around a scheme that converges anyway, evolving
  may keep more order: TreeBSSN's Kerr–Schild box and TreeWaveGR's holes
  converge faster evolved than prescribed, with somewhat larger errors.
  Say which, and why.
- **Write a state over the closed box.** A helper that writes only the
  old owned range `1:N` leaves the high-face copies as they were, and
  where those are the primaries the fill spreads the old values
  (TreeGRRMHD's gauge offset). Random data drawn per entry gives the
  copies of a point different values; draw from the point's position,
  or unify after.
- **Positions are `nodeposition`'s, in tests too.** A test comparing
  positions bit for bit with `block_origins + (I − 1) h` fails against
  the kernels' `nodeposition`. Every position is now formed from the
  patch's low corner and the block's integer corner, so two forests
  whose patches start at different corners place the same physical point
  alike only to rounding (TreeWaveOLDG's nested boxes against a uniform
  box over the finest cube, element nodes off the integers: 3e-15 of the
  field). Where zero is on a patch's grid, positions are anchored at
  zero instead (step 10b): two domains that both have zero on their grid
  — a quadrant, bitant or half-turn domain and its full box — place a
  point alike bit for bit at any spacing, and a box centred on zero is
  exactly antisymmetric. Its edges are then `∓(n/2)·h`, one ulp off the
  extents in about 6 % of boxes, always each other's negation. Domains
  without zero on their grid keep the low corner, and compare to
  rounding.
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
- A reduced domain (quadrant, bitant, half turn) and its full box have
  the same positions bit for bit where zero is on both grids, and the
  same values to rounding: the reduced domain makes its seam's copies
  one number, where the box keeps the two values its data or right-hand
  side give at `p` and at its image. A test comparing the two compares
  to a bound that names that source; TreeGeneralizedHarmonic's spinning
  hole on the rotating quadrant measured 136–610 ulps on the seam planes
  after the first fill, amplified to `4.65e−9` by a moving layer.
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

- TreeAMR precompiles a fill (PrecompileTools, a new hard dependency):
  the package's own precompilation takes longer (2.3 → 16–21 s on a
  laptop, once per environment), and the first fill in a fresh process
  comes sooner (9.3 → 2.5 s after the start). An environment that
  takes TreeAMR from a checkout through `[sources]` needs `Pkg.resolve()`
  once to pick up the new dependency; `Pkg.instantiate()` alone leaves
  `using TreeAMR` failing to load PrecompileTools.

- The fill runs its stages as A (kinked glues), W (prescribed walls),
  P (wall planes and a seam's axis), 0 (secondaries), 1, the hook, the
  interpolated glue ghosts, then 2; the boundary hook sees outer faces
  only.
- A zero that a seam's turn or a wall's parity negates is `+0`, not
  `−0`.
- `interpolate!` and `locate_point` read a point lying exactly on a
  face, edge or corner that leaves of two levels share from the finest
  leaf holding it, on either side of the face; before 0.3 the
  half-open rule gave it the upper leaf, coarse or fine. Among leaves
  of one level the upper one still holds it. A value interpolated at
  such a point moves by the scheme's error there, and its mirror image
  now comes from the same level.
- A cell-centered set's numbers are unchanged; a vertex-like set's
  change where half-open ownership made a mesh asymmetric (refined
  regions, walls, seams), by the scheme's error there.
- Checkpoints: TreeIOHDF5 stores a vertex-like set's closed blocks under
  the format feature `closed_blocks`, and refuses, saying why, a file
  written before 0.3 that holds such a set (M20's appendix and AMR-8's
  symmetric walls included); its cell-like sets still load through
  `load_checkpoint(…; fieldsets)`. Cell and element files load as
  before. Erik decided (2026-11-26): every file written from 0.3 with a
  vertex-like set carries the feature `closed_blocks`; a half-open file
  with such a set, M20's `interpolated_glue` files included, is refused
  and not converted, since the planes it lacks are what a fill with the
  application's hook computes; and TreeIOHDF5 0.2.0, which bounds
  TreeAMR at `=0.3.0`, is released right after TreeAMR 0.3.0.
