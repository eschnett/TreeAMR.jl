# TreeAMR.jl — Design

TreeAMR.jl implements a tree-based (octree-style) AMR discretization for
Julia. It provides the mesh, the storage, and the inter-grid operations —
no physics.

## Goals

- Simple design, as far as the problem allows.
- Highly efficient on HPC systems: multi-threading, GPUs, MPI (including
  MPI+GPU).
- Staged implementation: serial CPU first, then multi-threaded, then GPU,
  then MPI.
- Interoperate with standard Julia packages: time integration
  (OrdinaryDiffEq.jl), elliptic solvers, I/O, visualization.

Motivating applications (all external to this package): toy codes (wave
equation, simple hydro), the Einstein equations, relativistic GRMHD.

## Scope and non-goals

- **No physics.** The package provides the AMR mesh and its operations.
  Applications supply the equations, fluxes, and any physics-specific
  interpolation operators (see [Operators](#operators)).
- **No subcycling, by design.** All cells advance with the same global
  timestep, set by the finest level. This is a permanent commitment, not a
  temporary simplification: it removes time interpolation from ghost
  filling, makes the entire hierarchy a single state vector for standard
  ODE integrators, and reduces conservation at coarse-fine faces to a
  purely spatial condition. The cost is wasted coarse-level work, which we
  accept.
- **Every centering, from M8 on.** Through M6 the package was
  cell-centered only, deliberately: the original plan scheduled
  face-centered variables for M8 and left vertex and edge centering
  unscheduled, accepting that the retrofit would touch the core (array
  shapes, ghost rules). The M8 design (decided) does all `2^D` centerings
  at once — in 3D: cell, vertex, three faces, three edges — because once
  the storage and the one-dimensional stencil builders know about one
  staggered dimension they know about any subset of them, and
  constrained-transport MHD needs faces *and* edges. See
  [Centerings](#centerings). What the package still does not do is
  divergence-preserving (constrained-transport) prolongation of face
  fields: that operator is not a tensor product — the `x`-face
  reconstruction depends on the `y`- and `z`-face data to keep
  `∇·B = 0` — and is an application operator in the sense of
  [Operators](#operators).

## Core concepts

### Blocks

The unit of storage and computation is a **block**: a Cartesian grid of
`N^D` cells, covering a cube in physical space with uniform spacing that
halves per refinement level. Every block has the same size. A field set
over the blocks stores `G_d` ghost layers on each side in dimension `d`,
so a cell-centered field set holds `(N+2G)^D` cells per variable per
block; other centerings are one plane longer per staggered dimension
(see [Centerings](#centerings)).

- `D` is generic (1, 2, 3, ...) and compile-time (it is the array rank).
- `N` is likely 32 or 64 and belongs to the **forest**: cells are the
  tree's geometry. `G` is small (0–4) and belongs to the **field set**,
  per dimension (amended in the M8 design; through M6 it was a forest
  parameter). It says how far a stencil reaches into a neighbor's data,
  which is a property of what is stored, not of how space is cut up: a
  flux field reaches nowhere, an evolved vertex-centered field reaches
  differently from a cell-centered one at the same order, and two field
  sets over one forest with different `G` is the normal case from M8 on.
  Both are runtime values (no recompilation when they change), validated
  at construction against the invariants below.

Ghosts are stored on all sides of every block (decided; the earlier
ideas of x-only or unstored ghosts are dropped). Ghost cells exist only
in the working array, not in the ODE state vector (see
[Time integration](#time-integration)).

Invariants tying the parameters together (per dimension, against that
dimension's `G_d`, from M8 on):

- `N` is even (a fine block covers `N/2` cells at its parent's spacing).
- Shifted restriction of order `p` reads fine interior cells up to depth
  `2G + p/2 − 1`: `N ≥ 2G + p/2 − 1` (at `p = 2` this is the basic
  `N ≥ 2G`).
- The restriction window itself must fit inside a fine block's interior:
  `N ≥ p` (found in M2; binding when `G` is small).
- Symmetric prolongation of order `p` reads up to `p/2` of the source
  block's own ghost layers: `G ≥ p/2`. (Worst case is the first fine
  ghost layer: its target sits a quarter coarse cell from the interface,
  between the coarse nodes straddling it, and the node across the
  interface is already ghost layer 1.)
- The two restriction bullets and the `p/2` above are for the
  point-value family. The conservative family (measured in its
  implementation): prolongation orders are *odd* (the reconstruction is
  centered on a cell, not a quarter cell off one) with `G ≥ (p−1)/2` —
  one fewer ghost layer at comparable order — and restriction is the
  fixed 2-cell exact average, needing only `N ≥ 2G`.
- All of the above is for a cell-centered dimension. In a vertex-like
  dimension (M8 design): the exchange region must fit within one ring of
  finer neighbors, `N ≥ 2G + 2` (the shared boundary plane makes the
  high-side region one plane longer, and `N` is even); restriction is
  injection and needs nothing further; point-value prolongation reads
  `p/2 − 1` planes beyond the shared plane, so `G ≥ p/2 − 1`. The
  conservative family is refused along a vertex-like dimension (see
  [Operators](#operators)). Derivations under [Centerings](#centerings)
  and [Ghost filling](#ghost-filling).

### Centerings

*(Designed in M8, before implementation; "decided" below records the
design discussion, "measured" is still to come.)*

**The model.** Per dimension, a variable lives either at **cell centers**
— `N` values per block, at half-integer positions — or at **cell
boundaries** — the integer positions, `N + 1` per block counting both
boundary planes. A centering is therefore a `D`-tuple of `:cell` /
`:vertex`, and the familiar names are spellings of tuples: in 3D, cell is
`(:cell, :cell, :cell)`, vertex `(:vertex, :vertex, :vertex)`, the
`x`-face `(:vertex, :cell, :cell)`, the `x`-edge `(:cell, :vertex,
:vertex)` — an `x`-edge runs *along* `x`, so it is cell-like there and the
complement of the `x`-face. `cellcentered(D)`, `vertexcentered(D)`,
`facecentered(D, d)` and `edgecentered(D, d)` construct them. A tuple
rather than an enumeration of the `2^D` cases (decided) because of the
tensor-product structure everything rests on: every transfer is a product
of `D` one-dimensional stencils, and the stencil for dimension `d`
depends on the centering *in that dimension* alone.

**Shared points and half-open ownership** (decided). A vertex-like
dimension's boundary points are shared — block `b`'s point `N` is the
next block's point `0` — and the state vector must hold every degree of
freedom exactly once. In every dimension a block **owns its points
`0 … N−1`**, whatever the centering; the boundary point `N` belongs to the
neighbor on that side. Geometrically a point belongs to the block whose
half-open cell box contains it, and those boxes tile the domain exactly,
so the rule is unambiguous at edges, corners and across levels with no
tie-breaking. Consequences:

- The state layout is `N^D` per block per variable for *every*
  centering; `statelength`, `scatter!`, `gather!`, `statearray`,
  `map_blocks!` and `volume_weighted_norm` do not change.
- The non-owned boundary plane is stored and filled by the exchange
  exactly as a ghost is — copy, prolongation or restriction — and the
  block needs it: as a stencil source for its last owned point, and as
  the place where its own high-face flux lives.
- At a non-periodic boundary the domain's *upper* boundary points belong
  to nobody; they are the first plane of an outward-facing region and the
  boundary hook fills them, while the lower boundary points are owned and
  evolved. A vertex-centered application with physical boundaries sees
  this asymmetry; a periodic domain has no boundary and none. Accepted
  (decided): treating both boundary planes alike would have the lowest
  blocks own `N − 1` points in that dimension and the uniform state
  layout would be gone.

Rejected: closed ownership, both copies of a shared point in the state
vector. At a coarse-fine face the two copies see different ghosts — one
side's injected, the other's prolongated — so their right-hand sides
differ and the copies drift; keeping them consistent needs an exchange of
`du`, or of `u` inside the RHS, which the RHS contract forbids (AMReX's
`OverrideSync` exists to repair exactly this). The volume-weighted norm
would also count every shared point twice.

**Stored layout.** Per dimension `d`, with `c_d = 1` for `:vertex` and
`0` for `:cell`, a field set with ghost width `G_d` stores `N + 2G_d +
c_d` points (writing `G` for `G_d`):

| stored index `i` | role | count |
|---|---|---|
| `1 … G` | low ghosts | `G` |
| `G+1 … G+N` | **owned** (what the state vector holds) | `N` |
| `G+N+1 … G+N+c_d` | shared boundary plane (vertex-like dimensions only) | `c_d` |
| `G+N+c_d+1 … N+2G+c_d` | high ghosts | `G` |

at positions `origin + (i − G − 1/2)·h` (cell) or `origin + (i − G − 1)·h`
(vertex). Three ranges have names: the **owned** range `G+1 … G+N` (what
`interiorview` returns, unchanged); the **closed** range `G+1 … G+N+c_d`,
the block's `N+1` faces or vertices including both boundary planes
(`closedview`, and `map_blocks!(…; closed = true)`); and the **exchange**
regions, `1 … G` below and `G+N+1 … N+2G+c_d` above — `G` and `G + c_d`
points. Everything above the owned range is exchange-filled, so the
asymmetry is bookkeeping in the target ranges and nothing more.
`coordinates(fs, b, idx)` maps a stored index to a position for any
centering and replaces `cell_center`, which took the forest's `G` and
assumed cell centering; `fill_by_coordinates!` and
`boundary_by_coordinates` evaluate their callbacks at the field set's own
points — face centers for a face field.

Rejected: one shape for every centering, `(N+2G)^D`, with the shared
plane doubling as the first high ghost. Every index convention would
have stayed literally as it is, but a ghost-free face field becomes
impossible — there is no slot for the high face when `G = 0` — and
ghost-free fluxes are the reason `G` moved to the field set at all. The
extra plane costs `(N+2G+1)/(N+2G)` per vertex-like dimension and buys
`G = 0`.

**Invariant.** The exchange region on a block's high side spans positions
`N … N+G_d+c_d−1` and must lie within one ring of neighbors, including
when those neighbors are finer and each spans only `N/2` coarse cells:
`N/2 ≥ G_d + c_d`, i.e. **`N ≥ 2G_d + 2c_d`** — `N ≥ 2G` in a cell-centered
dimension, `N ≥ 2G + 2` in a vertex-like one (`N` is even). It is the
same condition as "restriction reads owned points only": the last coarse
ghost at position `N+G` is fine point `2G`, stored at `3G + 1`, owned iff
`3G + 1 ≤ G + N`. Checked when the field set is built.

**`G` is per dimension** (decided): an `NTuple{D,Int}`, with a plain
integer as the uniform shorthand. Nothing in the package needs the ghost
width to be the same in every dimension — every target range, stencil
and invariant is written per dimension already — so the tuple costs a
broadcast at construction and keeps the layout as general as the
centering is. Which staggered quantities use it:

- **Computed staggered quantities need no ghosts at all.** A flux `F_d`
  (face-centered in `d`) is consumed only by the divergence of owned
  cells, which reads faces `i` and `i+1` — the closed range in `d`, the
  owned range across. A constrained-transport EMF `E_d` (edge-centered) is
  consumed only by the curl at owned faces, which reads the edges
  bounding them — again the closed ranges. What makes those closed-range
  values consistent across levels is the interface restriction under
  [Conservation](#conservation-at-coarse-fine-faces), not a ghost fill.
  `G = 0` in every dimension.
- **Evolved staggered quantities need ghosts in every dimension, the
  vertex-like ones included.** A face-centered `B_d` or an edge-centered
  vector potential `A_d` is ghost-filled, and its prolongation reads the
  coarse source's exchange region *across* the shared plane: `G_d ≥ p/2 −
  1` along the stagger (zero at `p = 2`, one at `p = 4`) and `G ≥ p/2`
  across it. Independently of the mesh, an upwind constrained-transport
  scheme reconstructs `B_y` to an edge from faces `j − ½` and `j + ³⁄₂`,
  i.e. across the block's own boundary face.

So the two bounds genuinely differ by dimension for an evolved staggered
field, and a hypothetical prolongated flux would be the extreme case —
zero along, `(p−1)/2` across for a conservative tangential
reconstruction. A face field's memory is the visible payoff:
`(N+1)·(N+2G)^{D−1}` instead of `(N+2G+1)·(N+2G)^{D−1}` per block per
variable. The Burgers fluxes are the first user, with `G = 0`.

A field set is thus the unit of **(centering, `G`, operators)**, and an
application partitions its variables by that triple. This is also how
the long-deferred "per-variable operator selection" is delivered:
conservative operators for a density and point-value ones for a velocity
are two field sets over one forest, each with its own schedule.

### Tree structure

The domain is a **forest**: a brick of `M_1 × ... × M_D` root blocks,
each the root of an octree (a single root, all `M_i = 1`, is the
simplest case). This admits non-cubic rectangular domains. Morton keys
carry the root index; neighbor finding across root boundaries is
arithmetic on the brick. Refinement is
**all-or-nothing**: a block is either a leaf or is fully refined into
exactly `2^D` children of half the spacing. *(Amended from the original
sketch's "up to 2^D children": leaf-only storage requires it.)*

- **Leaf-only data.** Only leaves carry data; the leaves tile the domain
  exactly, with no overlapping coarse data underneath refined regions.
  There is never a question of which copy of a region is valid.
- **Linear octree.** The tree is represented as a sorted flat array of
  **Morton keys** (level + interleaved coordinate bits), one per leaf —
  the p4est/Dendro model. No pointers: neighbor finding is key arithmetic
  plus binary search; refinement replaces a key by its `2^D` children;
  coarsening replaces `2^D` sibling keys by their parent; MPI
  partitioning is splitting the sorted curve into contiguous ranges.
- **2:1 balance** is enforced across faces, edges, and corners. Every
  ghost region then touches at most one level up or down, which bounds
  the ghost-filling cases and prolongation stencils.

**Key encoding** (decided): a key is an `isbits` struct — root index
(`Int32`), level (`Int8`), and per-dimension coordinates
(`NTuple{D, UInt32}`, the block's integer position at its own level).
Curve order is root index first, then Morton order, with the bit
interleaving computed on the fly during comparison — no packed-integer
format, and no practical depth limit (32 levels). The curve is plain
Morton; Hilbert was considered and rejected (better MPI partition
locality, but the rotation arithmetic is not worth it at realistic rank
counts).

**Neighbor asymmetry** (recorded from M1): neighbor finding is not
symmetric under reversing the direction when levels differ — a coarse
block found across a fine block's corner also spans the face beyond it,
so the reversed direction from the coarse side points elsewhere. Exact
reciprocity holds only between same-level neighbors; adjacency is always
mutually discoverable, just not necessarily across the opposite
direction. Under periodicity two blocks can additionally be adjacent in
several directions at once (a single periodic root abuts both of its own
faces). Ghost filling must therefore be formulated as each block asking
for its own ghost sources — never as reversing a neighbor lookup.

### Domain and boundaries

The root brick maps to a rectangular physical domain: root block
`(i_1, ..., i_D)` covers the corresponding sub-box of the user-given
physical extents. Blocks are cubes (isotropic spacing), so the brick
dimensions must match the domain's aspect ratio. Boundary behavior is
declared up front:

- **Periodic** (per dimension): handled in the tree itself — neighbor-key
  arithmetic wraps modulo the brick, so periodic ghost filling is the
  ordinary copy/prolongation/restriction machinery with no special
  boundary code. With `M_i = 1` a block can be its own periodic
  neighbor; this is supported.
- **Physical** (per face): ghost cells are filled by a user-supplied
  boundary condition hook (which also covers e.g. reflection
  symmetries). For a vertex-like dimension the domain's upper boundary
  plane is the first plane of an outward-facing region and is filled by
  the hook too (M8; see [Centerings](#centerings)).

### Data layout

Two arrays exist:

- the **state vector**: a flat vector of length `N^D · nvars · nblocks`
  holding leaf interiors only — this is what ODE integrators see;
- the **working array**: one big persistent `(D+2)`-dimensional array
  holding all leaf blocks including ghosts:

      work :: Array{T, D+2}   # size (N+2G₁+c₁, ..., N+2G_D+c_D, nvars, nblocks)

  with `c_d = 1` in a vertex-like dimension and `0` in a cell-centered
  one, and `G_d` the field set's ghost width in dimension `d` (M8; see
  [Centerings](#centerings)). The state vector holds `N^D` per block per
  variable for every centering.

- One array for all variables of one field set (decided against
  per-variable arrays). Variables that differ in centering, ghost width,
  or operator choice go in different field sets (M8).
- Cell indices vary fastest (GPU coalescing); blocks are ordered by
  Morton key.
- Element type `T` is generic; `Float64` default, `Float32` relevant for
  GPUs. See "Precision" below for what carries `T` (amended in M5).
- Block slots are compacted at each regridding; block indices are **not**
  stable across regridding, and no stable region identifiers are offered
  (applications refer to space via keys or coordinates, not block
  slots).

Applications can allocate additional **field sets** — further block
arrays with the same layout over the same forest — for non-evolved data:
analysis quantities (constraint monitors), background/coordinate fields,
ghost-bearing temporaries. Field sets are transferred across regridding
(or re-evaluated, at the application's choice) but are not part of the
ODE state vector.

### Precision

**The mesh is generic in its floating-point type, and the arithmetic is
generic, not just the storage** (decided; amended in M5, when the
implementation showed the original one-line rule was not enough).

The original specification said only that the working array's element
type is generic. That is too weak. Geometry was computed from `Float64`
extents and *converted* at the end, so a `Float32` field set still needed
hardware fp64 to find a cell center — and fp64 is exactly what a device
may not have. So:

- **`Forest{D,T}` carries the geometry type**, and every geometry
  function computes in it from the first operation. Each takes an
  optional leading type, defaulting to the forest's, so a caller can ask
  for something else without the intermediate ever being `Float64`.
- **`FieldSet` and `GhostSchedule` default their element type to the
  forest's.** They remain free to differ — storing `Float32` fields over
  `Float64` coordinates is a legitimate mixed-precision configuration —
  but the two must agree with *each other*, which `fill_ghosts!` enforces
  with a message rather than a `MethodError`.
- **No floating-point literal may appear in per-cell arithmetic.** `0.5`
  is an fp64 operand and drags the expression with it; `1//2` converted
  to the working type is exact and folds at compile time. This is what
  makes `fill_by_coordinates!` reproduce `cell_center` bit for bit at
  every type, rather than only when both happened to be `Float64`.
- **Interpolation weights are computed exactly and rounded once.** Every
  position a stencil is evaluated at is an integer or a quarter integer,
  so the Lagrange products are exact in `Rational`; the single conversion
  into `Stencil1D{T}` is the only rounding in the construction.
  `Rational{BigInt}`, not a fixed width: the running products outgrow a
  64-bit numerator at order 16 (measured: 5.79e20 there, 1.78e17 at order
  14), and a bignum removes the ceiling rather than moving it, at a cost
  paid once per stencil entry at schedule-build time. It also buys
  correct rounding on the way out, since `T(::Rational{BigInt})` divides
  through `BigFloat` instead of rounding numerator and denominator to `T`
  first.

Two things this bought beyond portability. The conservative family's
defining property — the two subcell weight vectors average to the unit
vector on the center cell, so children always average back to their
parent — is now an algebraic identity that holds *exactly* at every
order, where it was previously only assertable to `atol = 1e-12`. And
tolerances that were `Float64`-calibrated constants became type relative:
the cube check's `rtol = 1e-12` is `4096·eps(T)` (1e-12/eps(Float64) ≈
4500), which reproduces the old behavior at `Float64` instead of sitting
below `eps(Float32)`.

Exercised in the test suite at `Float64`, `Float32`, and `Float32x2` —
MultiFloats.jl's double-`Float32`, a software type with no hardware
support at all, and therefore the strongest available evidence that no
fp64 path is load-bearing. The two non-default types catch opposite
faults: `Float32` is the leak detector, since a stray `Float64` operand
widens the result, while `Float32x2` promotes `Float64` *downward* and so
absorbs leaks silently — it tests instead that nothing depends on a
hardware float at all. Note that `eps(Float32x2) = 1.4e-14` is *coarser*
than `Float64`: it is not an "at least as accurate" drop-in.

One consequence worth stating: a negative accuracy assertion ("order `p`
does *not* reproduce degree `p`") does not rescale with the type. It
measures a truncation error, which is the same number in every
precision, while the roundoff floor it must clear moves. At `Float32`
and order 4 the two are only a decade apart, so the sharp
order-boundary claims stay in the `Float64` tests and the type-generic
tests assert a ratio instead.

## Operations

### Ghost filling

Under 2:1 balance there are exactly three cases per ghost region:

1. **Same-level neighbor:** direct copy.
2. **Coarser neighbor** (fine ghosts): **prolongation** — interpolation
   from coarse cells.
3. **Finer neighbor** (coarse ghosts): **restriction** — interpolation
   from fine cells.

**In a vertex-like dimension** (M8 design; see [Centerings](#centerings))
the three cases keep their names and change their one-dimensional
stencils. A copy is a shift by `N` as before, with the high-side target
one plane longer. Restriction is **injection**: a coarse point at
position `X` coincides with fine point `2X`, the stencil has width one and
weight one, it is exact for any data, carries no order, and never has to
shift — the circularity that forces cell-centered restriction to shift
its window cannot arise, because the coincident fine point is always
owned by the fine block (the `N ≥ 2G + 2` invariant). Prolongation puts
fine point `φ` at coarse coordinate `φ/2`: an integer for even `φ`, where
the Lagrange weights through any window containing that node are the unit
vector *exactly* (they are built in rational arithmetic), and a
half-integer for odd `φ`, where a symmetric window of even width `p`
applies — the same builder serves both. It reads the coarse source's
exchange region *across* the shared plane: the fine ghost nearest the
interface sits at coarse `−1/2`, its upper nodes are `0, 1, …, p/2 − 1`,
so the source's high exchange region (`G + 1` points) must hold `p/2` of
them, `G ≥ p/2 − 1`, against `G ≥ p/2` for cell centering. Tangentially
the window is clamped inward at the source's edges exactly as today.

**Stencil widths are per dimension** from M8 on: the transfer kernel
takes a `Val{NTuple{D,Int}}` where it took one `Val{P}`. Not cosmetic —
an injection padded to width `p` with zero weights reads `p − 1` slots
that need not hold data (a `G = 0` face field has no slot at all beyond
its high face), and `0 × NaN = NaN`. The interface restriction under
[Conservation](#conservation-at-coarse-fine-faces) is width one normally
and width two tangentially and needs this in any case.

Ghost filling is **phased**. Phase 1: all same-level copies and all
restrictions (each reads only interior cells of other blocks — race
free). Phase 2: prolongations, swept **level by level, coarsest targets
first**. The sweep is required because a prolongation stencil may read
the coarse source block's own ghosts, which may themselves be
prolongated from a still-coarser block (levels l−2, l−1, l side by side
are legal under 2:1 balance). Each phase/sweep step is an embarrassingly
parallel loop over blocks, with barriers in between.

Periodic boundaries need no special handling here — the tree wraps
around (see [Domain and boundaries](#domain-and-boundaries)). Physical
(outer) boundaries are filled by the user-supplied boundary hook,
invoked per outward-facing ghost region (faces, edges, and corners)
**between phase 1 and the prolongation sweep** — amended in M2: a block
at the domain edge has prolongation stencils that reach *tangentially*
past the edge into the coarse source's own outer ghosts, so running the
hook last would feed unwritten memory into the interpolation. The hook
may therefore read its block's interior (as reflecting and extrapolating
conditions do) but not other blocks' ghosts, and not ghosts that
prolongation has yet to fill.

Edge and corner ghost regions are always filled — some stencils don't
need them, but filling unconditionally is simpler, and cross-derivative
stencils do. Application kernels are strictly block-local: neighbor data
is visible only through ghost cells.

Transfers are **batched by stencil**: everything sharing a kind, a
direction and a child offset shares one set of one-dimensional stencils
and so one kernel launch. Prolongations are additionally batched by
*target level* (amended in M5): the batch is the unit the phase-2 sweep
schedules, so a batch spanning two levels would be filed under one of
them and the coarsest-target-first order would be quietly lost wherever
three levels meet. The original keying did span levels; it was found
while threading the schedule build, and no configuration could be
constructed in which it actually produced wrong ghosts — but the
ordering the sweep exists to guarantee was not in fact enforced, which
is enough reason to fix it. Batching is an implementation detail of a
phase, not the unit of parallelism: see
[Parallelism](#parallelism) for how a phase is actually run.

Neighbor finding is a regridding-frequency operation; ghost filling runs
at every RHS evaluation. The ghost-fill **exchange schedule** — the flat
list of copy/prolongation/restriction source–target region pairs — is
therefore precomputed whenever the tree changes and cached;
`fill_ghosts!` only replays it. Tree queries (`neighbor_keys` and
friends) must never appear in the per-evaluation path. The forest
carries a generation counter, bumped on every leaf-array change, so a
stale schedule is detected in O(1) — a leaf count is not enough, since a
refine-then-coarsen returns to the same size with different leaves.

Restriction is otherwise only needed when coarsening during regridding
and for analysis/output — there is no periodic "restrict fine onto
coarse" step, since no overlapping coarse data exists.

### Operators

Both families' weights are built in exact rational arithmetic and rounded
once into the stencil's element type; see "Precision" above for why, and
for what that makes exact.

Prolongation and restriction are **symmetric**: both are interpolation
operators with a configurable accuracy order, matched to the
application's discretization order. Plain `2^D` averaging is a valid
restriction only at 2nd order — it is exact for finite-volume cell
*averages*, but only 2nd-order accurate for point values at cell
centers; high-order finite differencing (e.g. for the Einstein
equations) needs correspondingly high-order restriction at coarse-fine
interfaces, which are everywhere on a tree mesh. Whether an operator is
conservative is a property of the operator, not of the mesh.

The package defines the operator **interface** and is required to ship
two operator families:

- **Polynomial point-value interpolation** (finite-difference semantics:
  data are point samples at cell centers), order configurable. Order 2
  is linear interpolation, whose restriction counterpart is the `2^D`
  average.
- **Conservative prolongation/restriction** (finite-volume semantics:
  data are cell averages). Restriction is the exact volume average — it
  is exact for *any* field, so it carries no order knob (implemented as
  the fixed 2-cell stencil the point-value family shares at order 2).
  Prolongation reconstructs a polynomial over each coarse cell,
  constrained to preserve that cell's average, and evaluates fine values
  as subcell averages — locally conservative by construction because the
  two subcell weight vectors average to the unit vector on the center
  cell, independent of the data. Orders are **odd** (`1` is piecewise
  constant, `3` the familiar `±1/8` slope), built via the primitive
  function so reconstruction-from-averages reduces to ordinary Lagrange
  interpolation. Both operators are still *linear* with per-cell
  weights, so they run through the same tensor-product stencil
  machinery; the family-specific invariants are under
  [Blocks](#blocks). How the interface-order rule below transfers to
  this family is predicted below and measured in M8.

There is **no default order** (amended after M3, which showed the
original default of 2 silently producing first-order convergence): the
right order depends on the application's differencing order via the
interface-order rule below, which the mesh library cannot know, so
`Operators` requires both orders explicitly. The application chooses
`G`; the package verifies at construction that `N`, `G`, and the
requested operator orders satisfy the invariants listed under
[Blocks](#blocks). Physics-specific operators —
hydro-aware limited interpolation, primitive-variable-based prolongation
— live in application packages and plug into the same interface. *(This
resolves the original sketch's open question about where hydro-specific
operators belong.)*

Operators are configured per field set, not per variable — and from M8
on that *is* per-variable selection (decided): a field set is the unit of
centering, ghost width and operators alike, so conservative operators for
a density and point-value ones for a velocity are two field sets over one
forest, each with its own schedule (see [Centerings](#centerings)). The
schedule is built from a field set, `GhostSchedule(fs, operators)`, and
records the layout it was built for; field sets with the same layout may
share it, and `fill_ghosts!` checks.

**Interface-order rule** (measured in M3): the interpolation order `p`
must exceed the application's differencing order by two, for *both*
operators. A ghost filled by an order-`p` operator carries an `O(hᵖ)`
error; a second-derivative stencil divides it by `h²`, leaving an
`O(h^{p−2})` truncation error along the interface, so the global rate is
`min(interior order, p − m + 1)` with `m` the highest derivative order.
The M3 wave equation (2nd-order Laplacian) measured L2 rates of 1.0 /
0.9 / 1.0 / 2.0 for (prolongation, restriction) orders (2,2) / (4,2) /
(2,4) / (4,4): raising one operator alone does not help, because each
side of the interface gets its ghosts from a different one. The same
mesh unrefined converges at 2.0 with order-2 operators, which pins this
on the interface rather than the scheme. The tests assert all four
rates.

The `+2` is the second-derivative case, `m = 2`. A flux divergence has
`m = 1`, so a first-order-in-derivatives scheme needs `p` to exceed its
order by *one* only — the caveat on Erik's list that the rule "is only
true if second derivatives are taken". For the conservative family, whose
orders are odd, the prediction is therefore rates **1, 2, 2** for
prolongation orders `p = 1, 3, 5` under a second-order finite-volume
scheme, and `p = 3` is the first order that does not degrade it. This is
what the M8 Burgers study measures (see [Milestones](#milestones)); the
table goes here when it exists.

**Interface stencils** (decided in M2): near a coarse-fine interface the
symmetric restriction window cannot exist — fine data across the
interface would itself be prolongated coarse data, a circularity.
Restriction therefore **shifts** its window inward (by `p/2 − 1` fine
cells at the ghost layer nearest the interface). Shifting preserves
polynomial exactness (Lagrange interpolation through any `p` distinct
nodes is exact for degree `< p`) and keeps the target inside the node
hull — interpolation, never extrapolation; the implementation asserts
this. Reducing the order instead was rejected: a symmetric window at the
first ghost layer collapses to order 2 regardless of `p` — and by the
interface-order rule above, order-2 ghost data feeding a
second-derivative stencil leaves an `O(1)` interface truncation error,
capping global convergence at *first* order (measured in M3). Prolongation, by
contrast, stays symmetric: it may read the source block's ghosts (that
is what the level-ordered sweep guarantees), at the cost of `G ≥ p/2`.
The conservative family never shifts at all: its restriction window is
exactly a cell's own children, and its prolongation *cannot* shift —
an off-center reconstruction would no longer preserve the containing
cell's average, so the `G ≥ (p−1)/2` bound has no shifted fallback.
Stability does not discriminate between the choices here
(global `dt`, 2:1 balance); damping high-frequency interface modes
remains the job of the application's usual Kreiss–Oliger dissipation.

**Operators per centering** (M8 design). Because the operator is a
tensor product, a family is a rule giving one-dimensional operators per
dimension's centering, and every constraint is checked per dimension
against that dimension's `G_d`:

| centering of `d` | family | restriction | prolongation | needs, in `d` |
|---|---|---|---|---|
| cell | `PointValue` | shifted Lagrange, even `p` | Lagrange at quarter offsets, even `p` | `G ≥ p/2`, `N ≥ 2G + p/2 − 1`, `N ≥ p` |
| cell | `Conservative` | 2-cell average | subcell averages of the reconstruction, odd `p` | `G ≥ (p−1)/2` |
| vertex | `PointValue` | injection | Lagrange at integers / half-integers, even `p` | `G ≥ p/2 − 1` |
| vertex | `Conservative` | *refused* | *refused* | — |

The first two rows are today's, the third follows from
[Ghost filling](#ghost-filling). The last row is deliberately empty
(decided). What a face- or edge-centered quantity stores is an *average*
along its cell-like dimensions and a *point value* along its vertex-like
ones, so along a vertex dimension the conservative family has nothing to
conserve and would merely interpolate — at an even order that the
family's odd `p` does not name (`p + 1` was the candidate). Rather than
fix that rule before anything exercises it, `GhostSchedule` refuses a
conservative field set with a vertex-like dimension, with a message
saying why. Nothing in M8 needs it: fluxes and EMFs are never
ghost-filled or transferred, and constrained-transport `B` needs a
divergence-preserving operator that is not a tensor product in any case.

### Conservation at coarse-fine faces

With a global timestep, conservation requires only that the flux a
coarse cell sees on a coarse-fine face equals the area-weighted sum of
the `2^(D-1)` fine-face fluxes — a purely spatial condition, enforced
within a single RHS evaluation. No flux registers or time-accumulated
corrections are needed (they only exist to bridge subcycled timesteps).

Mechanically this makes a conservative RHS **three steps** (the "two
phases" of the original sketch, with the fixup named): (i) all blocks
compute face fluxes, (ii) fluxes at coarse-fine faces are restricted
onto the coarse side, (iii) all blocks apply the flux divergence. The
original plan tied this to M8 and left applications non-conservative at
coarse-fine interfaces until then (fine for the wave equation and the
Einstein equations); the M8 design of step (ii) follows.

**Interface restriction** (M8 design; the "flux fixup"). A mesh
operation on any field set with a vertex-like dimension:

    isched = InterfaceSchedule(flux)             # rebuilt when the tree changes
    restrict_interfaces!(flux, isched)

For every block, every **face direction** `±e_d` with `d` a vertex-like
dimension of the field set (so that the field has points *on* that
boundary face), and every *finer* neighbor across it: the block's
boundary plane in `d` is overwritten with the restriction of the finer
neighbor's coincident boundary plane — injection along vertex-like
dimensions, the exact two-cell average along cell-like ones, weights
`1/2^{D−1}`, exact in binary. Tangentially the target is the block's
*owned* range in a cell-like dimension and its *closed* range in a
vertex-like one, split half-open between the finer neighbors with the
top point going to the top neighbor. On the low side (`δ_d = −1`) the
target is the block's **owned** boundary plane `G+1`, on the high side
its shared plane `G+N+1`; the source is the finer neighbor's opposite
boundary plane. Both sides *computed* those values one step earlier —
this is the one operation in the package that overwrites a block's own
closed-range values, and doing so is its purpose. It is a targeted
transfer, not a ghost fill: nothing else is touched and no ghosts are
needed, which is what lets a flux field have `G = 0`. For a
cell-centered field set there is no admissible direction, and the
constructor says so rather than silently building nothing.

*Face directions only* (decided, correcting a first draft that also
listed edge directions for edge-centered fields). The
constrained-transport consistency condition — a coarse cell's `∇·B`
stays zero only if, on every edge of a coarse face that is a
*coarse-fine* face, the coarse EMF equals the average of the fine EMFs —
lives on the edges of coarse-fine **faces**, including the boundary lines
of those faces, and nowhere else. A finer neighbor across an edge alone,
with same-level neighbors across both adjacent faces, imposes nothing:
the coarse blocks sharing that edge all compute the same EMF there from
the same data. Hence the closed tangential range for edge fields, and no
edge-direction transfers.

**One phase per direction.** Within a direction's phase every target
point is written exactly once. A point on the line where two coarse-fine
faces of the same block meet is written in two phases, by two same-level
fine blocks that computed the same value there (they hold the same data
in the stencil's footprint — the same obligation as for same-level
fluxes, below), so the result does not depend on the order and the
thread-independence discipline holds unchanged. `D` small launches
instead of one, over interface planes only.

**Why conservation follows, exactly.** For a finite-volume update
`h^D·∂ₜū_i = −h^{D−1} Σ_d (F_{d,i+½} − F_{d,i−½})` the total `Σ h^D ∂ₜū`
telescopes to face fluxes at block boundaries with alternating signs, and
vanishes iff the two sides of every shared face agree on the
*area-weighted* flux. Same level: both blocks compute the flux at the
shared face from the same cell values — their own and their ghosts, which
are bit-for-bit copies — with the same kernel, so the two fluxes are
identical *provided the flux stencil at a boundary face is covered by the
cell field's ghosts*. That is the application's one obligation: `G` of
the state set is at least the flux stencil's half-width (2 for a linear
reconstruction); no exchange is needed. Coarse-fine: the coarse flux is
replaced by the average of the `2^{D−1}` fine ones, and
`h_c^{D−1}·F_c = Σ h_f^{D−1}·F_f` by construction of the weights.
Periodic faces are same-level or coarse-fine faces like any other;
physical faces carry whatever boundary flux the application computes.
Hence the domain integral of every conserved variable changes only by
roundoff per RHS evaluation, in any Runge–Kutta stage. No flux registers,
no time accumulation: that is what the global `dt` bought.

**Construction.** The interface schedule comes out of the same neighbor
search as the ghost schedule — its transfers are the `:restrict` cases
of the neighbor walk for the admissible directions, with the target
range cut down to one plane — and it is batched, sliced and replayed by
the same `TransferGroup` / `run_phase!` machinery on any backend. It
records the forest generation and the field set's layout, and refuses to
run stale or on a different layout, as the ghost schedule does.

### Regridding

1. The application supplies a flagging function marking each block
   `Refine`, `Coarsen`, or `Keep` (per block; per-cell criteria are
   reduced to a block verdict inside the application's flag function —
   the same shape the device-side flagging kernel takes in M6).
2. **Buffering** (added post-M4; amended to *box-as-source* after the
   first implementation measured Refine-keyed dilation to be inert at a
   steady-state frontier): a flagging function may report, with **any**
   flag, the **bounding box of the cells where its criterion fired**
   (interior indices). A block is a **dilation source** iff it
   explicitly reports a box — with a bare flag, only `Refine` is a
   source (box defaulting to the whole interior, the conservative
   isotropic case); a bare `Keep` never is, else quiescent blocks would
   recruit their neighbors and coarsening would die globally; `Coarsen`
   is never a source. Each source has a **requested level** `L`
   (`Refine` → `l+1`, `Keep` → `l`). Its box, dilated by `buffer`
   *cells* at the block's own resolution, reaches a direction only when
   it exits the block along all of that direction's nonzero components
   (the edge/corner conjunction comes from the geometry); every leaf
   reached joins the buffer — promoted to `Refine` if its level is
   below `L`, and its `Coarsen` demoted to `Keep` if its level is at or
   below `L` (an over-fine neighbor may still coarsen toward `L`). A
   feature-holding block at target level thus returns `(Keep, box)` and
   holds an equal-level margin that travels with the feature — the
   proactive margin the original Refine-keyed rule could not provide
   (measured: buffers narrower than most of a block width were inert,
   because a freshly refined block's box hugs the face the feature
   entered through). `buffer ≤ N`: recruitment is a single pass reaching
   one ring, the cap is rejected loudly rather than silently truncated,
   and one ring covers any sane regrid cadence. The *width* is the
   application's choice (feature speed × regrid cadence — physics the
   mesh cannot know); the margin arithmetic and neighbor lookup are the
   mesh's job. Measured guidance: the width must *exceed* the feature's
   travel per regrid interval (in cells at the feature's level) — a
   margin narrower than the motion it must cover measured slightly
   worse than no buffer at all. The box costs the application only a
   min/max reduction over firing cells — the same shape the M6
   device-side flagging kernel produces. Accepted approximation: the box is convex, so disconnected
   flagged clusters within one block inflate it. 2:1 completion (next
   step) independently adds a graded one-level-coarser ring; the buffer
   provides the equal-level margin that keeps a feature away from the
   coarse-fine interface. Buffers exist here only for feature motion and
   interface distance — the subcycling rationale for Carpet-style buffer
   zones does not apply under a global `dt`.
3. Marks are requests, not commands: they are completed to maintain 2:1
   balance (refinement ripples outward; coarsening happens only when all
   `2^D` siblings ask for it, and is undone where balance cannot support
   it).
4. The new sorted key list is built; a new data array is allocated.
5. Data transfer: same-level blocks are copied (bit for bit), newly
   refined blocks are prolongated from their parent, coarsened blocks
   are restricted from their children — the `δ = 0` cases of the same
   stencil machinery ghost filling uses. Ghosts must be filled
   immediately before the transfer, because prolongation from a parent
   reads that parent's ghost layers. From M8 on each field set moves
   with its own schedule — `regrid!(forest, (state => schedule, aux =>
   aux_schedule); flags, …)` — since operators are per field set; a set
   paired with `nothing` is **resized without transfer**, which is what
   flux sets want (fluxes are recomputed at the next RHS evaluation).
   For a staggered field the target of every transfer is the *owned*
   range: the shared plane and the ghosts of a fresh block are left to
   the next `fill_ghosts!`, as ghosts are today. Coarsening a vertex-like
   dimension is injection from the children's even points, all owned.

A single regrid moves a block by **at most one level**: marks move a
block by one, and balance completion against an already-balanced tree
adds at most one more, never two. This is what makes parent/child-only
transfer sufficient; the transfer asserts it rather than trusting the
argument.

**Conservation of the transfer** (measured in M4): coarsening conserves
the volume integral of *any* field exactly when restriction is the
order-2 average; untouched blocks are copied bit for bit; refinement
conserves exactly those fields the operators reproduce exactly, and
**not** others — prolongation is not locally conservative, and at a
refinement boundary the fine region draws on neighbor values through the
parent's ghosts without those neighbors giving anything up
(percent-level drift measured for a field discontinuous across a
periodic seam). Selecting the **conservative operator family** (see
[Operators](#operators)) makes the transfer exactly conservative for
any field; that family landed early (pre-M5), verified exactly
conservative for random data under random regrids where the point-value
family drifts at the percent level.

**Initialization** iterates the same machinery: fill initial data →
flag → regrid → *re-evaluate* the initial data on the new mesh (rather
than prolongating it) → repeat until the hierarchy stops changing.

When driven by an ODE integrator, regridding changes both the size and
the meaning of the state vector. DiffEq callbacks support `resize!`, but
multistep history and dense output become invalid when entries are
reinterpreted, and `u_modified!` must be signaled; in practice
regridding means stop → rebuild → `reinit!` for anything beyond simple
Runge–Kutta schemes.

## Application interface (sketch)

Indicative only — names and signatures will evolve (updated for the M8
design; through M6 `G` was a forest keyword and `regrid!` took bare
field sets):

    # mesh: cells are the tree's geometry, ghosts are not
    forest = Forest(roots; N, periodic, extents)
    refine!(forest, keys); coarsen!(forest, keys)  # with 2:1 completion

    # fields: block arrays over the forest; each carries its centering
    # and its per-dimension ghost width
    state  = FieldSet(forest, nvars; G = 2)                       # cell-centered
    fluxes = ntuple(d -> FieldSet(forest, nvars; G = 0,
                                  centering = facecentered(D, d)), D)
    ops    = Operators(family = Conservative, prolongation = 3, restriction = 2)
    sched  = GhostSchedule(state, ops)                # per layout, not per forest
    isched = ntuple(d -> InterfaceSchedule(fluxes[d]), D)

    # global time step (application's choice)
    dt = cfl * minimum_spacing(forest)

    # the conservative RHS: three steps, written out by the application
    function rhs!(du, u, p, t)
        scatter!(p.state, u)
        fill_ghosts!(p.state, p.sched)
        for d in 1:D
            map_blocks!(flux_kernel!, p.fluxes[d], ...; closed = true)   # N+1 faces
            restrict_interfaces!(p.fluxes[d], p.isched[d])              # the fixup
        end
        map_blocks!(divergence_kernel!, p.state, statearray(du, p.state), ...)
    end

    # regridding: each set with its own schedule; `nothing` = resize only
    if regrid!(forest, (state => sched, fluxes[1] => nothing, …); flags, buffer)
        sched = GhostSchedule(state, ops); isched = …    # then reinit! the integrator
    end

There is still no `semidiscretize`-style wrapper. The flux kernel reads
two field sets with different `G`: face `i` (in `1 … N+1`) lies between
cells `i−1` and `i`, stored at `i − 1 + G_u` and `i + G_u`, while the
face itself is stored at `i + G_f` — the off-by-`G` sharp edge TreeWave's
notes already warn about, now with two `G`s; the Burgers kernels in the
tests are the worked example.

## Time integration

The whole hierarchy advances with one global `dt` (finest-level CFL). The
state is one flat vector; standard integrators (OrdinaryDiffEq.jl) drive
it unmodified.

**The state vector contains interiors only** (decided). Each RHS
evaluation:

1. scatters `u` into the working array,
2. fills ghosts (copies, restrictions, prolongations),
3. runs the application's kernels,
4. writes `du` in state layout — the gather can be fused into the
   compute kernels, since they write interior cells only.

The integrator never sees ghosts and the RHS never mutates `u`; the cost
is one scatter per RHS evaluation, which we accept. (The alternative —
handing the padded working array to the integrator, with `du = 0` in
ghost cells — was rejected: it makes the RHS mutate `u` and spends
integrator bandwidth on ghost memory.)

**Several field sets in one state vector** (specified in the M8 design,
implemented with the first application that needs it). Burgers has one
evolved set, but constrained-transport MHD evolves cell-centered hydro
variables *and* face-centered `B`, and the integrator must see both as
one vector. The state functions accept a tuple of field sets and lay them
out contiguously in tuple order — `statevector((hydro, Bx, By, Bz))`,
`scatter!(sets, u)`, `statearray(du, sets, i)` for the `i`-th set's view.
Each set contributes `N^D · nvars · nblocks` entries whatever its
centering; half-open ownership (see [Centerings](#centerings)) is what
makes this a plain concatenation.

Because external integrators own the stages, ghosts are filled at
*every* RHS evaluation; the classic wide-ghost/fewer-exchanges
optimization is unavailable by construction. This is an accepted cost,
alongside the wasted coarse-level work.

Coupling details (decided): the application writes `f!(du, u, p, t)`
itself, calling `scatter!` → `fill_ghosts!` → `map_blocks!` explicitly —
no `semidiscretize`-style wrapper until the pattern has stabilized. The
flat vector `u` is the authoritative data; the working array is scratch,
refreshed at every RHS evaluation (output and analysis scatter and
ghost-fill first). Through M3 only fixed-`dt` integrators are exercised,
with `dt` chosen by the application from a minimum-spacing query.
Adaptive integrators need a volume-weighted `internalnorm` — the default
norm weights fine regions more, simply because they contribute more
entries per volume — documented here, implemented post-M3.

## Parallelism

- **Multi-threading:** parallelize over blocks. Blocks are uniform-sized
  work units; RHS kernels are a single parallel loop, and ghost filling
  is a short sequence of parallel loops with barriers between the phases
  described in [Ghost filling](#ghost-filling). There is no switch: the
  KernelAbstractions CPU backend spreads a launch over
  `Threads.nthreads()`, and the host-side passes over blocks (neighbor
  finding when the schedule is built, the mark arithmetic in
  regridding, the boundary hook, the diagnostic reductions) are
  threaded the same way.

  **Bit-identical results, not merely equal to roundoff** (decided in
  M5). No parallel loop shares an accumulator: each writes its own slot,
  and every reduction forms one partial per block and sums the partials
  in block order. The chunking is a function of the item count and the
  thread count alone. So a 64-thread run reproduces a serial one exactly
  — worth the discipline, because it makes "the thread count" something
  a debugging session never has to consider. Collecting passes (the
  neighbor search, the balance scan, the buffer dilation) follow the
  same rule: each task fills a buffer of its own, and the buffers are
  concatenated in block order.

  **Application callbacks therefore run concurrently**: the `f(x, v)` of
  `fill_by_coordinates!`, the `f(b, key)` of `flag_blocks`, and the
  boundary hook. They must be pure functions of their arguments (the
  hook may write the region it was handed, and nothing else). This is
  the same contract M6 imposes anyway, since two of the three become
  device kernels.

  **A phase is one parallel loop, not a sequence of launches** (amended
  in M5). Ghost transfers are batched by stencil, and the batches differ
  in size by orders of magnitude — a face slab is `G·N^(D-1)` cells, a
  corner `G^D`. Launching the batches one after another leaves the small
  ones with a single workgroup each, i.e. serial, which measured as a
  hard ceiling of ~2.5x on the ghost fill however many threads were
  available, while the single-launch parts of the same step scaled
  fine. Each phase is therefore flattened into slices of roughly equal
  cell count, never crossing a batch, dealt out largest first, one task
  per thread, each slice launching as a single inline workgroup. The
  regrid transfer uses the same machinery for the same reason. A device
  backend keeps the plain per-batch launches: there a launch *is* the
  parallel unit.

  **Page placement dominates everything else on a NUMA node** (measured
  in M5, 64-core AMD EPYC 7532, 8 NUMA domains; 960 blocks of `32^3`,
  31.5M cells, a 1.2 GB working set). The same kernels partition the
  *same* arrays differently from one launch to the next — the working
  array by stored cell, the state vector by interior cell, a ghost slab
  by target region — so no first-touch pattern can serve them all, and
  the default first-touch placement leaves most accesses off-domain.
  Speedups on 64 cores against one:

  | phase                 | first touch | pages interleaved |
  |---|---|---|
  | RHS evaluation        | 10.2 | **36.3** |
  | ghost fill            |  9.9 | **36.6** |
  | scatter               |  5.7 | **36.1** |
  | initial data          | 22.4 | **59.5** |
  | volume-weighted norm  | 21.8 | **37.8** |
  | schedule build        |  2.8 |   2.4 |

  Interleaving the pages (`numactl --interleave=all`) is thus worth
  2–6x at high thread counts, and is a process-level policy the library
  cannot set for itself — so it is documented as the way to run rather
  than implemented. Pinning KernelAbstractions to its *static* schedule,
  so that a chunk of an ndrange always lands on the same thread, was
  measured as the alternative and rejected: with first-touch placement
  it reproduced the left-hand column to within noise (RHS 10.0, scatter
  5.5, norm 19.3), for exactly the reason above — stability within one
  kernel does not make a page local to all the kernels that touch it.

  The compute-bound pass (initial data, a sine per cell) scales past the
  memory-bound ones, as it should. Two pieces do not scale, both by
  construction and both negligible in absolute terms: building the
  schedule saturates below 3x because its tail — merging the per-task
  transfer lists into groups — is serial, and
  `complete_marks` gets *slower* with threads (80 microseconds to 460)
  because the pass is shorter than the cost of spawning the tasks. Both
  are regrid-frequency and orders of magnitude below the regrid's own
  data movement, so neither is worth a grain-size heuristic.
- **GPU:** all kernels (ghost fill, prolongation, restriction,
  application RHS) are written with **KernelAbstractions.jl** from the
  start, so the CPU implementation is already the GPU implementation.
  The leaf data array lives resident on the device. Regridding splits
  cleanly: only the flagging kernel runs on the device; the driver logic
  (mark completion, 2:1 balance, key rebuild) runs on the host; block
  data transfer (copy/prolongate/restrict into the new array) runs on
  the device.

  **The backend is chosen once, at allocation** (decided in M6). It is
  a keyword on `FieldSet` and on `GhostSchedule`, and nothing else takes
  one: every kernel in the package already read its backend off the
  storage it was handed (`get_backend(fs.work)`), so the storage
  decision *is* the backend decision. `statevector` allocates where its
  field set lives and `regrid!` reallocates there, which is what keeps
  an application's RHS — `scatter!` → `fill_ghosts!` → `map_blocks!` —
  literally the same code on a device. No device package is a dependency
  of TreeAMR; `KernelAbstractions.allocate` is the whole interface.

  **The schedule has to move with the data** (found in M6; the paragraph
  above did not anticipate it). "All kernels are KA kernels" is
  necessary but not sufficient: a kernel also dereferences things that
  are not field data. The transfer kernel reads the 1D stencils'
  `srcstart` and `weights` and the group's block-index vectors, and
  those were host arrays. So `Stencil1D` and `TransferGroup` became
  parametric in their array type, and a `GhostSchedule` uploads them
  once at construction. This costs nothing per ghost fill and is the
  natural residency point, since a schedule is already rebuilt whenever
  the tree changes — the same argument that put the exchange in a cached
  schedule, one level down. The weights are still built on the host in
  exact `Rational{BigInt}` arithmetic, which is a strength here: bignum
  interpolation is exactly the work a device should not be asked to do.

  **Two application callbacks needed a second form** (found in M6). M5
  already required every callback to be a pure function of its
  arguments, and expected that to be enough for M6 — "the same contract
  M6 imposes anyway, since two of the three become device kernels". It
  was not. Purity is about *concurrency*; what a device additionally
  requires is that the callback never touch host memory, and two of the
  three callbacks were handed structures that only exist on the host:

  - The **boundary hook** received `(fs, b, key, δ, region)` and wrote
    the region cell by cell — the one scalar-index path left in the
    package. It gains a cell-wise form, `CellBoundary(g)` with
    `g(x, v, δ)`, which the package launches as a kernel over the
    outward-facing ghost cells, batched by region shape exactly as
    transfers are batched by stencil (a region's extent is `G` along
    each nonzero component of `δ` and `N` along each zero one, so there
    are only a handful of shapes). `boundary_by_coordinates` is now one
    of these, and it reproduces the old host loop *bit for bit*: the
    kernel forms the position from the same per-block origin and spacing
    `cell_center` uses, so M5's thread-independence digests did not
    move. The region form is kept and is still what a condition reading
    the block's interior needs — reflecting, extrapolating outflow — and
    is CPU-only, which it says if handed a device field set. That
    limitation is real and is not papered over: outer boundaries that
    read their own interior are a CPU-only capability until the cell
    form grows an interior accessor.
  - The **flagging function** received `(b, key)` and no data, so an
    application closed over its field set and read it on the host.
    `firing_boxes(fires, fs)` is the device form: `fires(work, idx, b, x)`
    is a per-cell predicate (given the *stored* index, so a stencil may
    cross a block face into the ghosts), evaluated over every block in
    one launch, returning each block's firing-cell count and the
    bounding box of its firing cells. The *verdict* — which flag,
    against which maximum level — stays with the application, because
    that is physics. This is exactly the split step 1 of
    [Regridding](#regridding) describes, and the box is exactly what
    step 2 dilates. One work item per block, each looping its own cells:
    integer min/max is order-independent and every item owns its output
    slots, so the M5 determinism discipline carries over with nothing
    added.

  **Reductions got a device method, not a rewrite.** The diagnostics
  (`volume_weighted_norm`, `total_mass`) form one partial per block and
  sum the partials in block order. On a device the per-block host
  reduction would be one launch and one synchronization *per block*,
  issued from several host tasks at once; so the partials are formed in
  a single launch there instead. The CPU path is untouched — the M5
  numbers were measured with it — and the ordered combination, which is
  what the bit-identity rests on, is shared.

  **The reduction became public, as `block_mapreduce` (amended).** It
  was an internal helper, on the reasoning that the package ships the
  diagnostics an application needs. That reasoning was wrong, and the
  downstream application found it from the outside: the diagnostics it
  needs are *its* diagnostics — the per-variable scale a refinement
  criterion divides by, a coverage count, a pulse tracker — and there
  was no supported way to build one that is threaded, backend-agnostic
  and thread-count deterministic. What the mesh owns here is not the
  reduction but the *determinism discipline*, and a second copy of that
  argument downstream is a second thing to get wrong. It is the same
  split `firing_boxes` already makes, and the read-side counterpart of
  `map_blocks!`, which was exported from M3.

  Exporting it meant fixing it first, and the fix is the point:

  - **One specification of the reduction, not two.** The internal form
    took a host block-reducer `f` *and* a kernel-form fold `(op, init)`,
    which had to agree and which nothing checked. They do not agree in
    general: `sum` over a block view is a sequential `mapfoldl` when the
    view is `IndexCartesian` but a *pairwise* one when it is
    `IndexLinear`, and `D = 1` with a scalar variable selection is the
    latter — measured, at 3.0e-15 relative over 4096 cells. Every shape
    the package itself passed happened to be `IndexCartesian`, so the
    two paths agreed by accident of `SubArray`'s `viewindexing` and
    nothing more. The public form is `mapreduce`-shaped — `f` a
    per-cell transform, `(op, init)` the fold — so the host can still
    reassociate while both backends compute the same thing. A second
    latent copy of this went with it: the host path raised `abs(x)^p`
    where the kernel raised `abs(x)^q`, the `Float64`-exponent leak the
    comment two lines above it warns against.
  - **No ghost offset in the signature.** The internal form took a bare
    array plus the `g` that matched it; omitting `g` silently reduced
    the wrong cells. The public form takes the field set (working array,
    ghosts skipped) or the field set and a state vector (no ghosts), and
    works `g` out itself. This is the same off-by-`G` that `cell_center`
    taking stored indices invites, and one the caller should not be
    asked to get right twice.
  - **A variable selection a kernel cannot take is refused.** The host
    used `vars` as a view index and the device used
    `first(vars):last(vars)`, so a non-contiguous selection silently
    meant different things on the two backends. It is now an
    `ArgumentError` saying why.

  The guarantee is stated as what it is: bit-identical across thread
  counts, because every block owns its output slot and combining happens
  in block order. Identical across *backends* is not claimed and, for a
  floating-point `op`, is not true — the association differs. The CPU
  numbers did not move: `volume_weighted_norm` at `p = 1, 2, 3, ∞` and
  `total_mass` reproduce their pre-M6 values bit for bit in `D = 1, 2, 3`
  and in both `Float64` and `Float32`. Both paths are reachable on
  `CPU()`, so the suite checks that they compute the same fold on every
  run and not only where there is a device.

  **Measured on an H200** (960 blocks of `32^3`, 31.5M cells, `Float64`;
  the host column is the same node's 16 allocated cores under
  `numactl --interleave=all`, not the 64-core EPYC of the M5 table, so
  the ratios are a device-versus-a-socket comparison and not a
  device-versus-a-node one):

  | phase                 | H200 (s) | 16 cores (s) | ratio |
  |---|---|---|---|
  | RHS evaluation        | 0.0070 | 0.249 | **35.5** |
  | ghost fill            | 0.0053 | 0.170 | **32.2** |
  | scatter               | 0.0010 | 0.034 | 35.0 |
  | initial data          | 0.0013 | 0.083 | 63.4 |
  | regrid transfer       | 0.0013 | 0.115 | 91.7 |
  | volume-weighted norm  | 0.0232 | 0.058 | 2.5 |
  | `firing_boxes`        | 0.0160 | 0.088 | 5.5 |
  | schedule build        | 0.1114 | 0.105 | 0.9 |
  | triad reference       | 3842 GB/s | 205 GB/s | 18.7 |

  The per-evaluation path — the only part that runs at every RHS
  evaluation — tracks the bandwidth ratio, which is what a mesh library
  should deliver and is the whole claim. Three rows deserve their
  explanation rather than a footnote:

  - **The two per-block reductions are the weak rows, by choice.**
    `volume_weighted_norm` and `firing_boxes` both run one work item per
    *block*, so 960 work items on a device that wants tens of thousands.
    A hierarchical reduction would fix that and would give up the
    property that makes these functions trustworthy: one work item per
    block, each accumulating its own cells in its own order, is
    deterministic without a word of extra care, which is the M5
    discipline. Neither is on the per-evaluation path — one is a
    diagnostic, the other runs at regrid frequency — so the trade is
    paid where it is cheap. It would have to be revisited if
    `volume_weighted_norm` were ever wired in as an adaptive
    integrator's `internalnorm`, which is still an open question above.
  - **Building the schedule does not speed up, and should not.** It is
    the host-side neighbor search, which M5 already measured as
    saturating below 3x; the device upload added to it is small enough
    to disappear into the noise (0.111 s against the host's 0.105 s).
  - **The regrid transfer's 91.7 is not a device win over a host
    one**, it is the transfer alone against a host path that is
    bandwidth-bound in a worse access pattern; the honest reading is
    that the transfer costs about one RHS evaluation on either.

  **Metal, on an Apple M3 Pro, is the portability evidence rather than a
  speed result.** The full suite passes there in `Float32` on a backend
  that reports no hardware fp64 at all — which is the strongest
  available check that no fp64 path is load-bearing, the same role
  `Float32x2` plays for the type genericity. It is not a speedup: at
  3.9M cells the RHS takes 0.0187 s against the same chip's 8 CPU
  threads at 0.0217 s, and the two triad references agree (107 against
  113 GB/s), because on that part it is one memory system either way.
- **MPI:** the sorted Morton curve is split into contiguous per-rank
  ranges. Ghost exchange communicates face/edge/corner cell data between
  ranks; prolongation/restriction happen on the owner of the finer data.
  Regridding rebuilds and repartitions the curve. With a global `dt` and
  uniform blocks every block costs the same, so partitioning by equal
  block counts along the curve is already load-balanced. The exchange is
  layout-generic before MPI exists (M8 precedes M7, decided), so ghost
  fill, interface restriction and regrid transfer for every centering
  are distributed by one design. CUDA-aware MPI
  for GPU+MPI.

## Ecosystem integration

- **Time integration:** OrdinaryDiffEq.jl via the flat state vector (see
  above).
- **Elliptic solvers:** no solver-specific machinery in the package; the
  ghost/operator infrastructure suffices to build composite-grid
  operators. (Multigrid on the tree hierarchy would require overlapping
  coarse data, which leaf-only storage does not provide — out of scope.)
- **I/O:** HDF5.jl output and checkpoint/restart; possibly ADIOS2 later.
- **Visualization:** VTK export (non-overlapping AMR / multiblock
  formats) via WriteVTK.jl or similar.

## Open questions

All design questions through M3 are resolved in the sections above.
Remaining, none blocking before their milestone:

- Wiring `volume_weighted_norm` (implemented in M3) into adaptive
  integrators as `internalnorm` (post-M3).
- The one-dimensional operators of the conservative family along a
  vertex-like dimension: refused until an application needs them, with
  order-`(p+1)` Lagrange as the recorded candidate (see
  [Operators](#operators)).
- A state vector spanning several field sets: specified under
  [Time integration](#time-integration), implemented with the first
  application that needs it (constrained-transport MHD).

## Milestones

Each milestone has a concrete acceptance test; serial correctness is
established before any parallelism. The numbers are the order the
milestones were planned in; **M8 is done before M7** (decided in the M8
design, see the M8 entry), and the list below is in execution order.

- **M0 — Scaffolding.** Package skeleton, test harness, CI, docs stub.
  *(Skeleton exists.)*
- **M1 — Tree core (serial, D-generic).** Morton keys over a brick of
  roots, sorted leaf array, neighbor finding, refine/coarsen, 2:1
  balance enforcement, block storage, periodic wraparound. *Accept:*
  hand-rolled property tests with a seeded RNG (tiling, balance,
  neighbor soundness/completeness — exact reciprocity only at equal
  levels, see neighbor asymmetry above — and periodicity) on random
  refinement patterns in D = 1, 2, 3. *(Done.)*
- **M2 — Ghost exchange and default operators.** The cached exchange
  schedule, the phased ghost fill (three cases, level-ordered
  prolongation), periodic boundaries, physical-boundary hooks, default
  operators of configurable order with the `G`-sufficiency check —
  written as KernelAbstractions kernels (CPU backend). *Accept:*
  polynomial data reproduced exactly up to operator order (degree
  `p − 1` and no further) across all face/edge/corner and three-level
  configurations. Periodic wraparound is tested by its definition
  instead — an `M`-root periodic domain reproduces the middle tile of an
  explicit `3M`-root tiling bit for bit (amended in M2: polynomials are
  not periodic, so polynomial exactness across the seam is unattainable;
  only constants are periodic polynomials). *(Done.)*
- **M3 — Wave equation + OrdinaryDiffEq.** Scalar wave in 2nd-order
  form (state `(u, ∂ₜu)`, 2nd-order centered Laplacian), periodic cube,
  static two-level refinement over a sub-box, manual `f!`, fixed-`dt`
  RK4. *Accept:* volume-weighted L2/L∞ errors against the exact
  sine-mode solution converge at 2nd order — which requires **order-4
  operators and `G = 2`** (amended in M3; see the interface-order rule
  under [Operators](#operators)), verified in D = 1, 2 with a 3D smoke
  test. The wave equation lives in the tests: the package has no
  physics. *(Done.)*
- **M4 — Regridding.** Flag → balance → rebuild → transfer; the
  initial-data cycle; integrator reinit. *Accept:* the initial-data
  cycle converges to a fixed-point hierarchy; a moving refined region
  tracks a travelling pulse with the accuracy of the *uniformly finest*
  mesh at fewer cells (measured error ratio 1.00 — matching that
  reference is what "without artifacts" means operationally); transfer
  conservation as stated under [Regridding](#regridding) (amended in M4
  from the original blanket "conservation of transferred data", which
  refinement cannot deliver without conservative operators). *(Done.)*
- **M5 — Multi-threading.** Threaded loops over blocks. *Accept:*
  results match serial to roundoff; scaling measurement on a many-core
  node. Delivered stronger than asked on the first count: results are
  **bit-identical** across thread counts, checked by running a full
  adapt/evolve/regrid/evolve cycle in subprocesses at different thread
  counts and comparing digests of the state vector, the leaf array, the
  schedule shape and the reductions. Scaling on a 64-core AMD EPYC 7532
  (8 NUMA domains, 960 blocks of `32^3`): **36.3x** on the RHS path,
  59.5x on the compute-bound initial-data pass, with the table and the
  two findings that got it there — a phase must be one parallel loop,
  and pages must be interleaved — under [Parallelism](#parallelism).
  `bench/scan.sh` reproduces the measurement. *(Done.)*
- **M6 — GPU.** CUDA backend via KernelAbstractions; device-resident
  data. Floating-point-type genericity *(landed early, after M5)* is a
  prerequisite that is now in place: the geometry and the interpolation
  weights no longer evaluate in `Float64` on their way into a `Float32`
  field, so nothing on the per-cell path needs hardware fp64. See
  "Precision" under [Core concepts](#core-concepts). *Accept:* M3
  convergence results reproduced on GPU; kernel benchmarks. The backend
  is a keyword on `FieldSet` and `GhostSchedule` and nothing else; what
  the milestone did *not* anticipate, and what most of the work was, is
  that the exchange schedule has to become device-resident and that two
  application callbacks — the boundary hook and the flagging function —
  needed a second, cell-wise form, both recorded under
  [Parallelism](#parallelism). The whole suite passes on CUDA (NVIDIA
  H200) in `Float64` *and* `Float32`, and on Metal (Apple M3 Pro) in
  `Float32`, a backend with no hardware fp64 at all. The M3 convergence
  result is reproduced on both: L2 rate **1.99** in `Float64` and
  **1.99** in `Float32`, with order-4 operators and `G = 2` over the
  two-level mesh. The `Float32` study has to be run at coarser
  resolutions, and for a reason worth recording: the error measured is a
  truncation error, the same number in every precision, while the
  roundoff floor it must clear moves — and the step count grows with
  `N`, so in `D = 1` the floor is already reached at `N = 64` (the rate
  over `N = 16…128` collapses to 0.73). That is the positive-assertion
  form of the caveat "Precision" records for negative ones.
  `bench/gpu.jl` produces the kernel benchmarks in the format
  `bench/threads.jl` uses, so a device run and a host run read side by
  side; `bench/symmetry_gpu.sh` is the cluster job that runs both.
  *(Done.)*
- **M8 — Every centering, per-field-set ghost width, conservation,
  Burgers.** Done before M7 (decided): the MPI exchange is built over
  the schedule, and with the schedule layout-generic first, M7
  distributes ghost fill, interface restriction and regrid transfer for
  every centering in one design, instead of building the cell-centered
  exchange and retrofitting it twice. The design is under
  [Centerings](#centerings), [Ghost filling](#ghost-filling),
  [Operators](#operators) and
  [Conservation](#conservation-at-coarse-fine-faces); the conservative
  cell-data operator family *(landed early, pre-M5, with its
  regrid-transfer conservation already verified)* is its foundation. The
  test problem is Burgers' equation, in the tests, as the wave equation
  is; an Euler hydro toy is a separate package, as TreeWave is for the
  wave equation. Two halves:
  - **M8a — layout.** `G` (per dimension) and `centering` on the field
    set, `Forest` without `G`, `coordinates` in place of `cell_center`,
    `GhostSchedule(fs, ops)`, per-dimension stencil widths, the vertex
    rows of the operator table, `regrid!` over `fs => schedule` pairs,
    and the wave test split into vertex- and cell-centered halves.
    *Accept:* the M2 exactness test (degree `p − 1`, face/edge/corner,
    three levels) over all `2^D` centerings in `D = 1, 2, 3`, the oracle
    averaging along cell-like dimensions and sampling along vertex-like
    ones; vertex restriction bit-for-bit injection on random data; **the
    wave equation is vertex-centered from here on**, with the M3 study
    repeated on it — predicted rates **1 and 2** for prolongation orders
    2 and 4 at *any* restriction order, since restriction is exact, and
    `G = 1` sufficient at order 4 where cell centering needs 2 — while
    the cell-centered study is kept verbatim as `wave_cell` so the M3
    numbers stay under test; the thread digests. The cell-centered
    stencils are the same rational weights as before, so M8a changes no
    measured number.
  - **M8b — conservation.** `InterfaceSchedule` / `restrict_interfaces!`,
    `map_blocks!(…; closed = true)`, and Burgers' equation
    `∂ₜu + Σ_d ∂_d(u²/2) = 0` on a periodic box in `test/burgers.jl`:
    finite volume, linear reconstruction (unlimited for the smooth
    studies, so the measured rate isolates the interface and not a
    limiter's clipping at extrema; minmod wherever there is a shock),
    Rusanov flux, `SSPRK33` from `OrdinaryDiffEqSSPRK` (a new test
    dependency — conservation to roundoff holds for any Runge–Kutta
    method, since every stage's `du` sums to zero, but only a
    strong-stability-preserving one keeps the shock monotone), `G = 2` on
    the state and `G = 0` on the fluxes, a gradient criterion through
    `firing_boxes` with the M4 travelling margin. Data depending on
    `s = Σ_d x_d` solve the one-dimensional equation in `s` with speed
    factor `D`, so `u = u₀(s − D·u·t)` with `u₀ = ū + a·sin(2πs/L)` is
    exact in any `D` up to `t_b = L/(2πaD)`, solved per cell by Newton and
    compared as *cell averages*; after `t_b` a shock travels at `D·ū`.
    *Accept:* **total mass conserved to roundoff across coarse-fine
    faces** — a shock crossing a refined region that follows it, regrids
    in between, `|Δ Σ h^D u| ≤ c·eps(T)·Σ h^D|u|·nsteps` — with the
    negative control (fixup skipped) measured and its drift recorded
    here; the interface-order rule for the conservative family measured
    (predicted rates 1, 2, 2 for `p = 1, 3, 5`) and tabulated under
    [Operators](#operators); a tracked shock matching the uniformly fine
    reference at fewer cells, as the M4 pulse did; the interface
    restriction equal to a hand-computed average over the finer
    neighbors, and each target written exactly once per phase; the
    Burgers cycle in the thread-independence workload and in the device
    suite.
- **M7 — MPI.** Curve partitioning, distributed ghost exchange (for
  every centering, and the interface restriction with it, since both are
  transfers over the same schedule machinery), distributed regridding.
  *Accept:* results match serial; weak-scaling smoke test; then MPI+GPU
  with CUDA-aware MPI.
- **M9 — I/O and visualization.** HDF5 output, checkpoint/restart, VTK
  export.
