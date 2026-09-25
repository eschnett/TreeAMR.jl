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
- **No subcycling.** All cells advance with the same global
  timestep, set by the finest level. This removes time interpolation from ghost
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
  dimension (measured in M8a step 2): the exchange region must fit
  within one ring of finer neighbors, `N ≥ 2G + 2` (the shared boundary
  plane makes the high-side region one plane longer, and `N` is even);
  restriction is injection and needs nothing further; point-value
  prolongation reads `p/2 − 1` planes beyond the shared plane, so
  `G ≥ p/2 − 1` — and nothing else, in particular neither of the two
  restriction bullets above, which are about a window that no longer
  exists. The conservative family is refused along a vertex-like
  dimension (see [Operators](#operators)). Derivations under
  [Centerings](#centerings) and [Ghost filling](#ghost-filling).

### Centerings

*(Designed in M8, before implementation, and implemented in M8a steps
1, 2 and 3. "Decided" below records the design discussion; the three
"Implemented in M8a" notes at the end of this section record what the
implementation settled or had to correct.)*

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
  this asymmetry; a periodic domain has no boundary and none. At a
  *reflecting* upper face the exchange derives the wall points instead
  of the hook (M10; see [Ghost filling](#ghost-filling)), and the
  asymmetry remains: the low wall evolves, the high one interpolates.
  Accepted
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

**Implemented in M8a step 1** (`G` alone; centering follows in step 2).
`G` is a required `FieldSet` keyword with no default — an integer or an
`NTuple{D,Integer}`, stored as `NTuple{D,Int}` — for the same reason
`Operators` has no default order, and refusing it says so. `Forest` no
longer takes `G`, and still accepts the keyword only in order to throw a
message naming where it went; `regrid!` likewise keeps the M6
three-argument form as a method that throws. Four things the design left
open, settled by the implementation:

- **`GhostSchedule` belongs to a layout, not to a forest.** The
  documented form is `GhostSchedule(fs, ops)`. The forest form survives
  as `GhostSchedule(forest, ops; G, T, backend)` — `centering` joined
  its keywords in step 2 — because it costs nothing: it is the body, and
  the field-set form is one line spelling the triple out of `fs`, and it
  is what a caller with no field set in hand uses. `fill_ghosts!`
  compares `fs.G` against the schedule's and refuses a mismatch,
  alongside the existing forest, generation, element type and backend
  checks: every target range in a schedule is wrong for another `G`, and
  nothing else would have caught it. Step 2 added the centering to that
  same check, for the same reason — the stored extent, the target ranges
  and the one-dimensional operators all differ between a cell-centered
  and a vertex-like dimension.
- **`coordinates` defaults to the *field set's* element type**, not the
  forest's `floattype`, with the leading-type form
  `coordinates(S, fs, b, idx)` as elsewhere in the geometry. That is the
  type `fill_by_coordinates!` and `boundary_by_coordinates` already hand
  their callbacks, and the bit-for-bit agreement between those kernels
  and `coordinates` is under test at two ghost widths — the kernel never
  forms `G` (its index *is* the owned index) while `coordinates`
  subtracts it, so one ghost width could not have distinguished them.
- **`check_operators` takes the field set** and checks every constraint
  per dimension against that dimension's `G_d`, naming the dimension in
  the message. It kept M2's blanket `G_d ≥ 1` in every dimension, so
  that a ghost-free field set could not have a ghost schedule at all;
  step 2 relaxed that along a stagger and a review after step 2 removed
  it altogether — see the step-2 record below for why it was never
  needed.
- **`regrid!` allocates per field set from that set's own `G`**, and
  fills ghosts per field set from that set's own schedule, rather than
  once from a shared one.

**Implemented in M8a step 2** (the centering itself). `centering` is a
`FieldSet` keyword defaulting to `cellcentered(D)` — unlike `G` it *does*
get a default, because cell-centered is what a field set was through M6
and what everything not deliberately staggered wants. It is validated
into an `NTuple{D,Symbol}`; `staggers` turns it into the `c_d` tuple the
arithmetic uses, and is exported so that an application can size its own
loops. `closedview` and `map_blocks!(…; closed = true)` are the closed
range's two faces. The cell-centered stencils are the same rational
weights as before, so no measured number moved: the whole suite passes
unchanged, the M3 wave tables included, and the thread-independence
digests still agree byte for byte. What the step settled or corrected:

- **There is no blanket `G_d ≥ 1` at all** (amended after step 2; the
  step itself only relaxed it along a stagger). Step 1's rule — a
  dimension without ghosts has nothing to exchange, so a ghost-filled
  field set needs `G_d ≥ 1` everywhere — was M2's uniform-`G` check made
  per dimension, and it is redundant with the operator table wherever
  interpolation reads a neighbor: point-value prolongation needs
  `G_d ≥ p/2 ≥ 1`, conservative needs `G_d ≥ (p−1)/2`, which is 1 from
  `p = 3` on, and restriction never reads ghosts in either family. The
  one order it is not redundant for is conservative `p = 1`, where it
  is *wrong*: piecewise-constant prolongation reads only the coarse
  cell containing the fine one, tangentially as well as normally and in
  the regrid transfer, so a cell-centered dimension with `G_d = 0` is a
  legal layout for `Conservative` `(1, 2)` operators. That matters for
  an auxiliary field set with no ghosts that must be carried across
  regrids: `regrid!` needs a schedule, and the schedule needs
  `check_operators` to pass. Its ghost exchange is simply empty (the
  builder already skipped empty regions), and the conservative regrid
  test now runs `p = 1` at `G = 0`. Along a stagger the reasoning is the
  one step 2 gave: the block still has its shared plane, which is
  exactly the whole exchange of a second-order evolved face field —
  `G = (0, g, g)` at `p = 2` builds a schedule and fills it, and in
  `D = 1` that schedule is *entirely* injection, so the exchange is then
  exact for arbitrary data rather than to an order.
- **`N ≥ p` is a cell-centered constraint**, not a global one. It was
  checked once against the forest's `N` because it does not mention `G`;
  it is about the restriction *window* fitting inside a fine block's
  interior, and along a stagger there is no window. It moved into the
  per-dimension loop with everything else.
- **An empty target region is skipped when the schedule is built**, not
  filtered at launch. A cell-centered dimension with `G_d = 0` has no
  slab on either side, so every direction that leaves the block along it
  has nothing to fill; testing the region for emptiness before the
  neighbor search spares the tree query as well as the zero-size launch.
  (Reachable through the front door since the blanket `G_d ≥ 1` went: a
  ghost-free cell-centered set with conservative `(1, 2)` operators has
  an entirely empty schedule.)
- **The oracle generalization the plan expected was not needed.** The
  plan anticipated teaching `cell_average` to average along cell-like
  dimensions only, so that a staggered field set could be checked as the
  mixed average-and-point-value object it is. Nothing needs it: that
  reading belongs to the conservative family, which is refused along a
  stagger, so every staggered set under test is point-value throughout
  and is compared at `coordinates`. The oracle was left alone.

**Implemented in M8a step 3** (the application). The wave equation is
vertex-centered from here on — `centering` is a keyword on every entry
point of `test/wave.jl`, defaulting to `vertexcentered(D)`, and the M3
study survives verbatim in `test/wave_cell_tests.jl` saying
`cellcentered(D)` out loud at every call. The measured table is under
[Operators](#operators); no cell-centered number moved. What the step
settled:

- **Nothing in the application knows the centering.**
  `wave_rhs_kernel!` takes no `Val(C)`, as the plan predicted: it reads
  its own point and its neighbours a spacing away, which is the same
  stencil wherever those points sit. `WaveProblem` needs no centering
  either, since it carries the field set. So the whole staggered study
  is the cell-centered one with one keyword changed at the
  `FieldSet` calls — which is the claim a centering that is "a property
  of the field set and nothing else" has to make good on.
- **`wave_forest` deliberately does not take a centering**, against the
  plan's list. It builds the forest, and how space is cut into blocks is
  exactly what a centering does not change; a keyword that is accepted
  and ignored would say otherwise.
- **The staggered `Float32` wave case lives in `gpu_tests.jl`, not
  `type_tests.jl`.** `wave.jl` is structurally `Float64`/`Float32` only
  — MultiFloats implements no trigonometric functions at all — and
  `gpu_tests.jl` runs its whole matrix on the `CPU()` backend in both,
  so the vertex convergence study is already measured in `Float32`
  there. `type_tests.jl`'s vertex case is the *layout* one: the
  staggered testset gained `vertexcentered(D)` beside `facecentered(D,
  1)`, which is the first case that is vertex-like in every dimension at
  once.
- **The thread workload gains two staggered cycles, not a rerun.** A
  vertex-like dimension lengthens every exchange region by the shared
  plane, replaces the restriction stencils with injection and narrows
  the prolongation window, so `run_phase!` deals out differently shaped
  slices; the D = 1 periodic cycle also regrids and transfers, and the
  D = 2 non-periodic one exercises the boundary hook on the upper
  boundary plane that belongs to nobody.

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
- **Reflecting** (per face; M10): a face across which the solution is
  its own mirror image. That covers a symmetry plane and, equally, a
  hydrodynamic solid wall, which does the same thing to the ghosts. It is
  declared on the forest as `reflecting = ((lo, hi), …)`, one pair per
  dimension, shaped like `extents`; a dimension cannot be both periodic
  and reflecting. Every variable of a field set over such a forest has a
  **parity** per dimension, `EvenParity` or `OddParity`: a scalar is
  even everywhere, the `d` component of a vector is odd in `d` and even
  elsewhere, and a product of components takes the product of their
  parities. The parity is required, with no default, since it is physics.
  `NoParity` exists for dimensions without a reflecting face and is
  refused in one that has one (decided). A variable without a parity has
  no value beyond the wall, and ghosts without a value do not stay in the
  ghosts: the first regrid prolongation that reads them carries them into
  the interior.

  Like periodicity, reflection needs no boundary code, though it lives
  one layer up. The tree is unchanged — `neighbor_keys` still finds
  nothing across the face — and the *schedule* turns every ghost region
  crossing a reflecting face into an ordinary copy, restriction or
  prolongation from a mirrored source (see [Ghost filling](#ghost-filling)).
  So reflection runs on every backend, in the same kernel and the same
  phases as every other transfer, and the boundary hook never sees a
  reflecting face.
- **Physical** (per face, neither periodic nor reflecting): ghost cells
  are filled by a user-supplied boundary condition hook. For a
  vertex-like dimension the domain's upper boundary plane is the first
  plane of an outward-facing region and is filled by the hook too (M8;
  see [Centerings](#centerings)). Until M10 the hook was also the only
  way to express a reflection, which it could not do correctly at every
  edge and corner; see [Ghost filling](#ghost-filling).

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

**In a vertex-like dimension** (implemented in M8a step 2; see
[Centerings](#centerings)) the three cases keep their names and change
their one-dimensional stencils. A copy is a shift by `N` as before, with
the high-side target one plane longer. Restriction is **injection**: a coarse point at
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
may therefore read its block's interior (as extrapolating conditions
do) but not other blocks' ghosts, and not ghosts that prolongation has
yet to fill.

**Reflecting faces** (M10) are transfers, not hook calls. A ghost region
in direction `δ` that crosses reflecting faces in the dimensions of a
mask `m` is the mirror image of a region that lies inside the domain.
Zero the masked components of `δ` to get `δ′`. The mirror image is then
the block's own interior next to the wall when `δ′ = 0`, and otherwise
it lies in the node adjacent to the block in direction `δ′`, at the
block's own extent along every masked dimension. The source is whatever
the ordinary search finds in `δ′`:

- the block itself (a copy);
- a same-level neighbor (a copy);
- a coarser one (a prolongation);
- finer ones (a restriction, from the children on the wall side only:
  the others cover the half of the tangential extent that the mirror
  image does not reach).

If `δ′` leaves the domain through an outer face, the region is the
hook's, as before.

Along each masked dimension, the one-dimensional stencil is the ordinary
*tangential* one (the `δ_d = 0` stencil of the same kind and child
offset) with its **target rows remapped** by the mirror `j ↦ j*`:

| | low wall | high wall |
|---|---|---|
| cell-centered | `j* = 2G + 1 − j` | `j* = 2(G + N) + 1 − j` |
| vertex-like | `j* = 2(G + 1) − j` | `j* = 2(G + N + 1) − j` |

The source windows and weights are the ordinary ones, so `Stencil1D`
and the transfer kernel do not change. The kernel only multiplies the
result by a per-variable factor, the product of the variable's parities
over the masked dimensions. It keeps a table of these factors on the
field set, not in the schedule: a schedule still belongs to a layout,
and parity belongs to the variables. The mirror image of a ghost slab
lies inside the owned range and within its wall-side half, since
`G ≤ N/2`, and `G + 1 ≤ N/2` along a stagger, which is exactly
`N ≥ 2G + 2c`. That is asserted when the schedule is built.

Phasing needs no new rule. A mirror copy or restriction reads interiors
only and joins phase 1; a mirror prolongation targets the block's own
level and joins the sweep at that level. This is what the hook could not
do. At a mixed edge or corner region whose tangential neighbor is
coarser, the mirrored values are the block's *own* prolongated ghosts,
and the hook, which runs before the sweep, would have read them stale.

**The upper wall point in a vertex-like dimension** (decided in the M10
design). On a high reflecting face the wall plane `G + N + 1` belongs to
nobody (see [Centerings](#centerings)), and the mirror maps it onto
itself, so reflection alone does not determine it. It is derived:

- An odd variable is exactly zero there.
- An even variable takes the symmetric Lagrange interpolant at the wall
  through the `p` points `±1, …, ±p/2` beside it. By symmetry that folds
  to the `p/2` one-sided points with weights `2w_k`, which at `p = 4` is
  `(4u₁ − u₂)/3`.

`p` is the prolongation order, so the error is `O(h^p)`, the same as a
prolongated ghost's, and the interface-order rule covers the point. Every
source has its own wall at `G + N + 1` in its own frame, so the row is
the same whatever kind the transfer is, and it reads the source's own
points beside the wall. A coarser source's points are further apart,
`O(H^p)`, as prolongation's are. The row is its own batch, because its
parity factor is `(1 + s)/2` where the mirrored rows' is `s`. The low
wall point is owned and evolved, as on any face. So a vertex-centered
reflecting box is not symmetric under exchanging its two walls: the low
wall evolves, the high one interpolates. Nor is an odd variable's low
wall point forced to zero. It stays zero if the right-hand side respects
the parity, as a centered stencil does, since the mirrored ghosts give
`u(−h) = −u(h)` exactly.

Edge and corner ghost regions are always filled — some stencils don't
need them, but filling unconditionally is simpler, and cross-derivative
stencils do. Application kernels are strictly block-local: neighbor data
is visible only through ghost cells.

Transfers are **batched by stencil**: everything sharing a kind, a
direction, a child offset and (M10) a mirror state per dimension — none,
mirrored rows, or the vertex-like upper wall row — shares one set of
one-dimensional stencils and so one kernel launch. Prolongations are additionally batched by
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

### What the ghost fill costs

Profiling a downstream solve (TreeGeneralizedHarmonic, a uniform mesh of
120 blocks of `8^3`) found that **the ghost machinery owned most of the
non-compilation run time and essentially all of the remaining heap
allocation**: the floating-point work of the physics was about 22 % of
run self time and the transfer kernel's addressing about 19 %. That is
the right order of magnitude to care about — and it understates the
refined case, where the same application measures the fill at 79 % of an
evaluation at a coarse-fine face against 22 % uniform. Four costs were
named; all four are real, one was diagnosed wrongly, and a fifth turned
out to be the largest. Measured here with `bench/ghosts.jl`, one thread,
`D = 3`, `N = 8`, 4 roots per edge, 10 variables, `p = 4`, flat *self*
time:

| | before | after |
|---|---|---|
| uniform mesh, 64 blocks, one fill | 5.12 ms, 17552 B | **2.90 ms, 10896 B** |
| two-level mesh, 120 blocks, one fill | 28.4 ms, 157408 B | **15.4 ms, 97504 B** |
| two-level mesh, schedule build | 19.6 ms, 39.1 MiB | **3.15 ms, 4.28 MiB** |
| `p = 6` schedule build | 89.5 ms, 104.4 MiB | **4.2 ms, 6.28 MiB** |

The same fills at four threads, where a phase runs as slices rather than
as one launch per group, gain slightly more: 1.01 ms → **0.53 ms**
uniform and 6.57 ms → **2.85 ms** two-level. So this is a change to what
the kernel does per point, not an accident of the serial schedule.

**Bounds checking was the largest cost, and was not on the list**
(found here). `checkbounds_indices` and its `size` calls were 15 % of
the uniform fill's self time and 25 % of the two-level one — more than
the interpolation arithmetic. Every index the transfer kernel forms is
constructed by the schedule, whose whole job is to guarantee it: a
target offset runs over the stencil's own `ntarget`, a source window is
clamped into the stored extent when the stencil is built, and
KernelAbstractions' CPU emitter guards the body with `__validindex`, so
the global index never leaves the `ndrange`. So the kernel's reads and
writes are `@inbounds`. That is an assertion, and it is checked rather
than believed: CI runs `julia-runtest` with its default
`check_bounds=yes`, which overrides `@inbounds` package-wide, so a
stencil that walks out of its block fails there. The user's boundary
hook is deliberately left outside the `@inbounds` region — whatever it
indexes is checked as it would be anywhere else.

**The launch is over the target box, not over a flattened copy of it.**
The kernel took `ndrange = (prod(boxlen), nvars, ntransfers)` and
recovered the per-axis position with an integer `div` and `rem` per
dimension, per ghost point, per variable — 9 % of the uniform fill's
self time, and the largest entry after the bounds checks. It is now
`ndrange = (boxlen…, nvars, ntransfers)`, so the backend supplies the
position and no division happens at all. `map_blocks!` already launched
a `D + 1`-dimensional ndrange, so this is not new ground for the device
backends; on a GPU the division moves into KernelAbstractions' own
`expand`, where it belongs. The two boundary kernels had the same
flattening and got the same treatment. Since this is the one change that
alters launch geometry, it was checked on real hardware and not only
argued: the whole suite passes on Metal (Apple M3 Pro, `Float32`).

**The tensor-product sum is generated, not iterated.** `for m in
CartesianIndices(Ps)` cost its trip count in `__inc` even though `Ps` is
a compile-time constant, and it recomputed the full `D`-fold weight
product at every stencil point. It is now a generated loop nest with
literal trip counts, which hoists `D − 1` of the `D` weight loads out of
the inner loops. The nest reproduces the old loop *exactly*, not merely
to the same accuracy: `m_1` runs innermost, as column-major
`CartesianIndices` iteration did, so the contributions are summed in the
same order, and the weight product is still formed as
`((w₁ · w₂) · …) · w_D`, since floating-point multiplication does not
associate. Only the loads move. Ghost fills are bit-identical to the
previous implementation across `D = 1, 2, 3`, both centerings and both
operator families — checked by digest against the previous commit, not
inferred.

**The per-launch allocation is KernelAbstractions', not ours**
(corrects the downstream brief, which put it on `run_group!`
reassembling the group's geometry). Rebuilding the geometry tuples costs
nothing measurable; what allocates is the argument tuple that
`Kernel{CPU}`'s varargs call boxes on the way into KA's `__run`
inference barrier — 672 B per launch, of which the group geometry
accounted for 32. Passing a slice as an offset into the group's block
lists instead of as two `SubArray`s, and dropping the two box-shape
tuples the flat launch needed, brings it to 416 B. The rest is KA's and
would take either fewer launches or a leaner launch path to remove;
neither is worth doing for allocation alone, since at ~200 launches per
fill it is well under a microsecond of time.

**The exact rational weights are memoised.** Building them in
`Rational{BigInt}` is deliberate — it is what makes them exact before
the single rounding into `T` (see [Precision](#precision)) — and it was
never in the per-evaluation path. It *is* in the per-schedule path,
which a regridding run pays at every regrid and a test suite pays once
per problem it builds, and it was recomputing the same few weight
vectors thousands of times: 39 MiB of bignums for one `p = 4` schedule,
104 MiB at `p = 6`. Lagrange weights are translation invariant — shift
every node and the target together and every difference is unchanged,
which in exact rational arithmetic is an identity — so a window of `p`
consecutive integer nodes is fully described by `p` and by where the
target falls inside it, and the package only ever asks for integer or
quarter-integer targets. A cache keyed on that pair (and one on `p` for
the conservative subcell weights) cuts a `p = 6` schedule build by 21x
and its allocation by 17x, with the weights unchanged to the last bit
because nothing about the arithmetic changed.

What is left, at one thread, is loads, floating-point work, and
KernelAbstractions' own `CartesianIndices` iteration over the workgroup
— which is now the single largest entry in a copy-dominated fill, where
each ghost point does exactly one stencil point of work. That is KA's
loop, not ours, and removing it would mean not using KA's CPU emitter.

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

**The same rule along a stagger** (measured in M8a step 3). Repeating
that study on a **vertex-centered** field set — same equation, same
Laplacian, same two-level mesh, values moved from the cell centers to
the cell boundaries — changes the table in exactly the way the vertex
row of the operator table predicts, and in no other way. Restriction is
injection, which has no order, so only the prolongation order enters;
`min(2, p − 1)` is then 1 at `p = 2` and 2 at `p = 4`:

| prolongation | restriction | L2 rate, `D = 1` | L2 rate, `D = 2` |
|---|---|---|---|
| 2 | 2 | 0.99 | 1.01 |
| 2 | 4 | 0.99 | 1.01 |
| 4 | 2 | 1.99 | 1.99 |
| 4 | 4 | 1.99 | 1.99 |

all at **`G = 1`**, where the cell-centered study needs `G = 2` and
`check_operators` refuses one less; the unrefined control converges at
2.00 in both dimensions. The rows come in pairs because the two runs
are *the same computation*: a vertex-like restriction stencil is width
one with weight one whatever order is asked for, so the errors agree bit
for bit, and the tests assert that identity rather than the two rates —
it is the claim that would break first if a builder ever started using
`p` along a stagger. `G = 2` at `p = 4` likewise reproduces `G = 1` bit
for bit: the relaxed bound is not "nearly enough", it is the whole
requirement. The cell-centered study is kept verbatim beside it
(`test/wave_cell_tests.jl`) so the M3 numbers above stay under test.

The `+2` is the second-derivative case, `m = 2`. A flux divergence has
`m = 1`, so a first-order-in-derivatives scheme needs `p` to exceed its
order by *one* only — the caveat on Erik's list that the rule "is only
true if second derivatives are taken". For the conservative family, whose
orders are odd, the prediction is therefore rates **1, 2, 2** for
prolongation orders `p = 1, 3, 5` under a second-order finite-volume
scheme, and `p = 3` is the first order that does not degrade it.

**Measured in M8b step 5**, with the Burgers study of `test/burgers.jl`:
the smooth sine run to half its breaking time on the M3 two-level mesh,
unlimited linear reconstruction, Rusanov flux, `SSPRK33`, conservative
restriction (which is exact and therefore never enters), over
`N = 8 … 64` in `D = 1` and `N = 8 … 32` in `D = 2`:

| prolongation | L∞ rate, `D = 1` | L∞ rate, `D = 2` | L1 rate, `D = 1` | L1 rate, `D = 2` |
|---|---|---|---|---|
| 1 | **1.00** | **0.84** | 1.98 | 1.79 |
| 3 | 1.97 | 1.81 | 1.96 | 1.88 |
| 5 | 1.97 | 1.77 | 1.97 | 1.88 |
| *unrefined control* | 1.93 | 1.76 | 1.97 | 1.90 |

The prediction holds: order 1 costs a full order, order 3 recovers the
scheme's own rate, and order 5 buys nothing further — the refined runs at
`p = 3` and `p = 5` land on the *unrefined control's* rate, which is the
sharper statement that the interface has stopped being what limits them.

**The norm is part of the result** (measured in M8b step 5; the design
did not anticipate it). The rule shows in `L∞` and **not** in an integral
norm: every L1 column above is the scheme's own rate, `p = 1` included.
An order-`p` prolongation leaves a flux defect on the coarse-fine face
and nowhere else, and the solution error it causes stays in a
neighbourhood of the face whose measure shrinks with `h`; a
volume-weighted norm multiplies the two and sees `O(h²)` whatever `p` is,
while `L∞` sees the defect's own `O(h^p)`. M3's wave study measured the
same rule in L2 because a second-derivative stencil divides the ghost
error by `h²` and radiates it over the whole domain; there the choice of
norm did not matter, and here it decides whether the effect is visible at
all. Both norms are asserted in `burgers_tests.jl`, in both directions,
precisely because picking one and believing it is the easy mistake.

**Why the defect stays local is conservation, not the flux-divergence
form** (amended after step 5, when the record credited the form; the
prediction below was made before the run). The residual the defect
creates is a *dipole*: after the fixup both sides of the face use the
same flux, so the fine cell loses exactly what the coarse cell gains and
the residual has zero net mass. A first-order hyperbolic operator carries
a zero-mean residual nowhere — the contributions injected at successive
steps travel the same characteristic and telescope — which is what
leaves an `O(h)` bump on an `O(h)` neighbourhood and nothing downstream.
Without the fixup the fine flux error is `O(h)` at `p = 1` while the
coarse side's is `O(h²)`, so the residual has net mass `O(h)` per unit
time (the leak the drift test measures), and the equation transports it
downstream as an `O(h)` plateau. Measured: the order-1 L1 rate **falls
from 1.98 to 1.12** in `D = 1` and **from 1.80 to 1.25** in `D = 2` when
the fixup is skipped, while `L∞` stays at 1.0 either way; at `p = 3` the
leak is `O(h³)` and the rate does not see it (2.00 without the fixup).
So "the rule shows only in `L∞`" is a property of a *conservative*
scheme: a non-conservative flux-divergence scheme radiates its interface
defect as the wave equation does, and its L1 rate says so. The negative
control on the rate is asserted alongside the one on the drift.

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

**Operators per centering** (decided in the M8 design; the vertex rows
measured in M8a steps 2 and 3). Because the operator is a tensor
product, a family is a rule giving one-dimensional operators per
dimension's centering, and every constraint is checked per dimension
against that dimension's `G_d`:

| centering of `d` | family | restriction | prolongation | needs, in `d` |
|---|---|---|---|---|
| cell | `PointValue` | shifted Lagrange, even `p` | Lagrange at quarter offsets, even `p` | `G ≥ p/2`, `N ≥ 2G + p/2 − 1`, `N ≥ p` |
| cell | `Conservative` | 2-cell average | subcell averages of the reconstruction, odd `p` | `G ≥ (p−1)/2` |
| vertex | `PointValue` | injection | Lagrange at integers / half-integers, even `p` | `G ≥ p/2 − 1` |
| vertex | `Conservative` | *refused* | *refused* | — |

The first two rows are the cell-centered ones; the third follows from
[Ghost filling](#ghost-filling) and is measured twice — as exactness to
degree `p − 1` and no further, over all `2^D` centerings in `D = 1, 2, 3`
at `p = 2` and `p = 4`, with `G = p/2 − 1` accepted where `G = p/2 − 2`
is refused (M8a step 2); and as the convergence rate of an application
that reads those ghosts, in the vertex-centered wave table under
[the interface-order rule](#operators) above (M8a step 3). The last row
is deliberately empty (decided). What a face- or
edge-centered quantity stores is an *average* along its cell-like
dimensions and a *point value* along its vertex-like ones, so along a
vertex dimension the conservative family has nothing to conserve and
would merely interpolate — at an even order that the family's odd `p`
does not name (`p + 1` was the candidate). Rather than
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

**Interface restriction** (M8 design, implemented in M8b step 4; the
"flux fixup"). A mesh operation on any field set with a vertex-like
dimension:

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

**Implemented in M8b step 4.** `InterfaceSchedule(fs)` and
`restrict_interfaces!(fs, isched)`, in `src/interfaces.jl`; no new
kernel, no new struct beyond the schedule itself. The design above
survived contact with the code; four things it left open, settled here:

- **The schedule takes no `Operators`**, unlike the ghost schedule. The
  transfer is injection and the exact two-cell average, both fixed by
  the geometry, so there is no order for a caller to choose and no
  family to select — the point-value builder at `p = 2` *is* the exact
  average, which is also what the conservative family's restriction is.
  `InterfaceSchedule(fs)` is therefore the whole signature.
- **A phase is a face *dimension*, not a signed direction** — `D`
  launches, as the design's count said, but arrived at differently. The
  two signs of one dimension target different planes (`G+1` and
  `G+N+1`), so they never collide and share a phase; it is *dimensions*
  that have to be separated, because the line where two coarse-fine
  faces of a block meet is a target of both.
- **The phases commute, for two reasons, and 2:1 balance is one of
  them** (measured). No plane the fixup writes is a plane it reads, over
  all phases: a point where that could happen lies on the line where two
  faces of a fine block meet, so the level-`l+2` block that wrote it and
  the level-`l` block that would read it touch across an edge or a
  *corner*, which `balance!` forbids — it walks all `3^D − 1`
  directions, not just the faces. That is stronger than the design
  claimed and is under test (`isdisjoint(targets, sources)` over a
  three-level forest, every centering that has an interface, `G = 0`
  and `1`). The design's own argument remains the *other* reason: a
  point on the line where two coarse-fine faces of the same block meet
  is written in two phases, once from each finer neighbor, and the result
  is order-independent only because both neighbors computed the same
  value there from the same data — the application's obligation, the one
  that also makes same-level fluxes agree. For a face field the
  tangential range is owned, so no point is written twice and balance
  alone suffices; for an edge field with its closed tangential range
  both conditions are needed, and the oracle test satisfies the second
  by construction, its data being a function of position. Together they
  make the fixup a pure function of the fluxes it is handed, whatever
  order the phases run in and however `run_phase!` deals its slices out.
- **`target_range` grew a `closed` flag rather than the interface
  schedule growing a range of its own.** Tangentially the fixup wants
  the closed range where the ghost exchange wants the owned one, and
  that is one `od * c` on the top half of the halved case; the two
  boundary planes are the first points of ranges the function already
  names. For the same reason the stencils come from the ghost
  restriction's builder over an explicit target range
  (`restrict_stencil_over`), so the fixup and the exchange cannot
  disagree about where a coincident fine point is.

A linear field is *invariant* under the fixup — injection reproduces it
and so does the two-cell average, since the coarse point is the midpoint
of the two fine ones — which is the analytic claim the device test rests
on. On Metal in `Float32` the fixup reproduces the CPU result bit for
bit: it adds no arithmetic, only target ranges.

**Measured in M8b step 5.** Burgers' equation (`test/burgers.jl`) is the
application that puts a number on all of this, and every number below has
its negative control — the identical run with step (ii) skipped, which is
a keyword on the test problem and the *only* difference between the two.
Mass is `Σ hᴰ u`, and the domain integral is exactly 1 here, so an "ulp"
below is `eps(T)` of the answer itself.

| configuration | mass drift | without the fixup |
|---|---|---|
| static two-level mesh, smooth, `D = 1` (62 steps) | 1.1e-16 (0.5 ulp) | 3.2e-4 |
| the same, `D = 2` (123 steps) | 0 | 4.4e-5 |
| the same, `D = 3` (92 steps) | 1.1e-16 (0.5 ulp) | 6.1e-5 |
| shock tracked through regrids, `D = 1` (242 steps) | 3.3e-16 (1.5 ulp) | 3.8e-5 |
| the same, `D = 2` (337 steps) | 5.6e-16 (2.5 ulp) | 5.1e-5 |
| the same in `Float32`, `D = 1` (146 steps) | 1.2e-7 (1 ulp) | 2.5e-5 |

Three things in that table are worth saying out loud. The drift is one or
two ulp of the total mass and does **not** grow with the step count, so
`c·eps(T)·Σ hᴰ|u|·nsteps` is a bound the runs sit ten orders inside
rather than a rate they approach. The leak without the fixup is a
*discretization* error, hence the same number in `Float64` and in
`Float32` (2.466e-5 against 2.468e-5 in the same configuration) — which
is why the separation is eleven orders of magnitude in double precision
and a factor of 207 in single, and why the Float32 assertion has to be
written against `eps(T)` rather than against a constant. And the
**uniform** mesh conserves to roundoff with or without the fixup, since
every face there is a same-level face: that is the control that pins all
of the above on the coarse-fine faces rather than on the scheme.

The same-level half of the argument is the application's obligation, and
Burgers is where it becomes concrete: `G = 2` on the state, because the
linear reconstruction at a block's own boundary face reads cells
`i-2 … i+1`, and with one ghost the two sides of that face would
reconstruct from different numbers. On Metal in `Float32` the whole
three-step right-hand side reproduces the CPU bit for bit — the drift,
the control's drift and the error norms are the same numbers — which is
the strongest available statement that nothing in it is fp64-dependent.

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
field sets; `reflecting` and `parity` are M10's):

    # mesh: cells are the tree's geometry, ghosts are not
    forest = Forest(roots; N, periodic, reflecting, extents)   # (lo, hi) walls
    refine!(forest, keys); coarsen!(forest, keys)  # with 2:1 completion

    # fields: block arrays over the forest; each carries its centering
    # and its per-dimension ghost width
    state  = FieldSet(forest, nvars; G = 2)                       # cell-centered
    # over a forest with reflecting faces every set also says how each
    # variable mirrors: parity = [EvenParity, (OddParity, EvenParity), …]
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

**Three launch ranges, not two** (amended for TreeHydro). `map_blocks!`
launched over the owned range `N` or, with `closed = true`, the closed
range `N + c_d`. It gains a third: `stored = true` launches over every
*stored* point of every block, ghosts included — `N + 2G_d + c_d` per
dimension, which is exactly `size(fs.work)`. The consumer is the
primitive recovery of a finite-volume hydrodynamics code: the exchange
carries *conserved* variables, its reconstruction reads *primitives* two
cells into the neighbours, so the pointwise conversion between them has
to have run in the ghost cells as well. Nothing about that is
hydrodynamics — it is a pass over the storage — and which points a block
stores is the mesh's business, so the mesh spells the range rather than
the application recomputing `N + 2G + c` for itself.

Under `stored = true` the kernel's global index **is** the stored index;
under the other two it is an offset into the owned range and the kernel
adds `G_d`. That is a real trap, because a kernel written for the wrong
form still writes in bounds — it writes the wrong cells — so the
docstring says it first and the test asserts the index each cell holds
and not merely how many were written. `stored = true` with
`closed = true` is refused: the closed range is a sub-range of the
stored one, so asking for both names two different loops rather than an
intersection.

**An all-variables form of the coordinate callbacks** (amended for
TreeHydro). `fill_by_coordinates!(f, fs)` calls `f(x, v)` once per point
*and variable* — its kernel has a variable axis in the ndrange — and
`CellBoundary(g)` calls `g(x, v, δ)` the same way. Beside those, and
without changing them, there is now a form called once per *point* that
returns every variable at once, selected by wrapping the callback in
`AllVariables`:

    fill_by_coordinates!(AllVariables(f), fs)       # f(x)    -> NTuple{nvars}
    CellBoundary(AllVariables(g))                   # g(x, δ) -> NTuple{nvars}
    adapt_to_initial_data!(fs, ops; initial = AllVariables(f), …)
    boundary_by_coordinates(AllVariables(f))
        # = CellBoundary(AllVariables((x, δ) -> f(x)))

The reason is that some states are only definable as a whole. A
hydrodynamics code states its initial and boundary data as *primitive*
variables and stores *conserved* ones, and the conversion needs all the
primitives of a point together. Per variable, the whole conversion would
run `nvars` times per point with all but one number thrown away — at
setup for the initial data, and at **every** right-hand side evaluation
for the boundary hook, which is the case that decided it.

A wrapper type rather than arity detection on the callback: a closure
does not advertise its arity reliably, and the package already spells a
hook's form as a type (`CellBoundary`). One field, so it is `isbits`
whenever the callback is and can be a kernel argument unchanged. The
all-variables kernels form the position with the same expression as the
per-variable ones, on the same origin and spacing in the same order, so
the two forms fill a field set with bit-for-bit the same numbers and
switching between them is not a numerical change — the same claim, for
the same reason, as the M6 cell-wise boundary hook's.

The tuple's length has to be `nvars`, and a kernel cannot say so
usefully, so it is checked **once on the host** before the launch, by
evaluating the callback at one owned point of block 1 (and, for the
boundary form, with the sample direction `δ = (−1, 0, …, 0)`); the
`ArgumentError` names both numbers. Calling the callback on the host is
legal, because everything it closes over is `isbits` by the device rule
— that is what lets it be a kernel argument at all. The boundary form
pays that one host call per ghost fill, against a launch over every
outward-facing ghost cell.

The one case the host call does not cover is a callback closing over a
*device array*, which a kernel argument may legally do (the backend
adapts it on the way in) and which the host call would scalar-index.
That is a narrow gap and it is documented rather than worked around:
such a callback uses the per-variable form, which has no host call.
Silently skipping the check instead would trade a named error for a
wrong number of variables written into the storage.

The region form of the boundary hook is untouched: `AllVariables` is
about how many values a *cell-wise* callback returns, not about which
form the hook takes.

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
  M5; narrowed after M8, next paragraph). No parallel loop shares an
  accumulator: each writes its own slot, and every reduction forms one
  partial per block and sums the partials in block order. The chunking
  is a function of the item count and the thread count alone. So a
  64-thread run reproduces a serial one exactly — worth the discipline,
  because it makes "the thread count" something a debugging session
  never has to consider. Collecting passes (the neighbor search, the
  balance scan, the buffer dilation) follow the same rule: each task
  fills a buffer of its own, and the buffers are concatenated in block
  order.

  **Floating-point sums are promised to roundoff only** (narrowed after
  M8). The paragraph above bundles two rules under one name, and only
  one of them is about reductions. That every work item owns its output
  slot, and every collecting pass fills one buffer per task and
  concatenates in block order, is race freedom: it costs nothing, it
  stays, and it is what makes the state vector, the leaf array, the
  schedule, and every max, min or integer reduction bit-identical across
  thread counts — and, once M7 exists, across rank counts. That the
  partials of a floating-point *sum* are combined in block order is the
  rule that is no longer a guarantee. It was free on the CPU, where one
  `mapreduce` per block and a serial pass over the partials is how the
  diagnostics would be written anyway — the M5 table was measured that
  way, and the CPU fold is unchanged — but it prices two things the
  design now wants: a hierarchical reduction on a device, where one work
  item per block is the weak row of the M6 table below, and a plain
  `Allreduce` under MPI, where identity across rank counts would mean
  every rank seeing the same association wherever the partition falls,
  i.e. a global gather of block partials on every reduction. What did
  *not* force the change is worth recording, because it is the usual
  argument: SIMD inside a block leaves thread-count identity intact,
  since a block's partial depends on its data and the compilation target
  and not on which thread ran it. What SIMD breaks is identity across
  *machines*, which was never claimed and which TreeHydro measured at
  1–4 ulp across CI runners. The claim is therefore: everything that is
  not a floating-point sum is bit-identical across thread counts; a
  floating-point sum is reproducible to roundoff across thread counts,
  rank counts, backends and machines, and exactly from run to run at a
  fixed configuration wherever the reduction underneath uses a fixed
  fold order and no atomics. `block_mapreduce` keeps returning per-block
  values, each still bit-identical; how a caller combines them is the
  caller's. What the narrowing permits is specified and measured under
  "**Implemented**" below, after the M6 paragraphs it revises.

  **Application callbacks therefore run concurrently**: the `f(x, v)` of
  `fill_by_coordinates!` (or the `f(x)` of its `AllVariables` form), the
  `f(b, key)` of `flag_blocks`, and the boundary hook. They must be pure
  functions of their arguments (the hook may write the region it was
  handed, and nothing else). This is the same contract M6 imposes
  anyway, since two of the three become device kernels.

  **A phase is one parallel loop, not a sequence of launches** (amended
  in M5). Ghost transfers are batched by stencil, and the batches differ
  in size by orders of magnitude — a face slab is `G·N^(D-1)` cells, a
  corner `G^D`. Launching the batches one after another leaves the small
  ones with a single workgroup each, i.e. serial, which measured as a
  hard ceiling of ~2.5x on the ghost fill however many threads were
  available, while the single-launch parts of the same step scaled
  fine. Each phase was therefore flattened into slices of roughly equal
  cell count, never crossing a batch, dealt out largest first, one task
  per thread, each slice launching as a single inline workgroup. The
  regrid transfer uses the same machinery for the same reason. A device
  backend keeps the plain per-batch launches: there a launch *is* the
  parallel unit. *(Amended 2026-09-23:* the phase is still one parallel
  loop and every batch is still split across all threads, but by owner
  rather than by size. Each thread takes the transfers whose target
  blocks it owns, because slices dealt to whichever task was free moved
  every block's ghosts to a new core in every fill; see "What one
  process loses".)

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
  than implemented. *(Amended 2026-09-23:* since every per-block
  pass now runs each block on its owner's thread, first touch lands a
  block on the domain that computes on it, and with pinned threads that
  beats interleaving; see "What one process loses".) Pinning KernelAbstractions to its *static* schedule,
  so that a chunk of an ndrange always lands on the same thread, was
  measured as the alternative and rejected: with first-touch placement
  it reproduced the left-hand column to within noise (RHS 10.0, scatter
  5.5, norm 19.3), for exactly the reason above — stability within one
  kernel does not make a page local to all the kernels that touch it.
  *(Revisited 2026-09-23:* right for the reason given and for one kernel
  alone, but most of what it was meant to fix was not placement. With
  the ghost fill given the same block-to-thread map as the kernels, the
  static schedule recovers 2.4x on the RHS; see "What one process
  loses" below.)

  The compute-bound pass (initial data, a sine per cell) scales past the
  memory-bound ones, as it should. Two pieces do not scale, both by
  construction and both negligible in absolute terms: building the
  schedule saturates below 3x because its tail — merging the per-task
  transfer lists into groups — is serial, and
  `complete_marks` gets *slower* with threads (80 microseconds to 460)
  because the pass is shorter than the cost of spawning the tasks. Both
  are regrid-frequency and orders of magnitude below the regrid's own
  data movement, so neither is worth a grain-size heuristic.

  **Where the 64-thread efficiency goes: the process, not the pages**
  (measured 2026-09-23 on Symmetry, `bench/symmetry_numa.sh`, jobs
  562390–562402; a 64-core AMD EPYC 7543 node — two sockets of four
  NUMA domains, distance 12 within a socket and 32 across, two DDR4
  channels per domain, so about 410 GB/s for the node — same 960 blocks
  of `32^3` as the M5 table, Julia 1.13). The question was how much of
  the gap to 64x a domain-local layout could recover — one MPI rank per
  NUMA domain, or the same block ownership imitated with pinned thread
  groups in one process. Eight independent 8-thread copies of the
  benchmark, each bound to one domain with its own memory and holding
  an eighth of the blocks, have no cross-domain traffic at all and so
  bound *any* such scheme from above. Nanoseconds per cell for the whole
  node, the slowest copy's time over all cells; five repeats of the
  first column spread 2.05–2.45 on the RHS:

  | phase                 | 1 × 64, interleaved | 1 × 64, first touch | 2 × 32, socket-local | 8 × 8, domain-local | 8 × 8, interleaved |
  |---|---|---|---|---|---|
  | RHS evaluation        | 2.41 | 2.42 | 1.21 | **0.73** | 0.89 |
  | ghost fill            | 0.92 | 1.00 | 0.36 | **0.29** | 0.36 |
  | scatter               | 0.68 | 0.73 | 0.29 | **0.16** | 0.17 |
  | initial data          | 0.46 | 0.46 | 0.45 | 0.39 | 0.41 |
  | volume-weighted norm  | 0.27 | 0.44 | 0.28 | 0.26 | — |
  | triad reference       | 0.71 | 0.92 | 0.34 | 0.28 | 0.18 |

  *(Amended the same day.)* The copies of the last three columns were
  started together but not synchronized, and each reports its own best
  of twenty repetitions per phase, so a copy whose repetitions fell
  while the others were compiling or in another phase reports bandwidth
  the node never gave all of them at once. Re-measured in shared
  wall-clock windows ("What one process loses", below), the RHS gap to
  eight copies is 2.2x rather than 3x, and for a plain static loop the
  stream gap between one process and eight disappears. The conclusions
  below are amended where they depend on the difference.

  Three things follow, in decreasing order of surprise.

  - *The loss is not memory placement.* The last column is the control:
    eight 8-thread processes whose pages are interleaved over the whole
    node, the same non-local placement as the first column, run the
    per-evaluation path 2.7–4x faster than the one 64-thread process
    and within 20 % of the domain-local copies. Placement itself is
    worth about 1.2x: the same 8-thread process on domain 0 runs the
    RHS at 5.8 ns/cell with local pages, 5.9 interleaved over the node,
    5.6 on a same-socket domain and 8.2 on a cross-socket one. First
    touch is within 10 % of interleaving here, against 3.6x on the
    Rome node of the M5 table; the benchmark's first touch is parallel,
    and Milan's fabric is forgiving. The kernel's NUMA-balancing
    counters did not move during the runs, so page migration plays no
    part either.
  - *One process stops scaling at 32 threads.* At fixed placement and
    fixed problem the RHS costs 3.02, 2.34, 2.41 and 2.37 ns/cell at
    16, 32, 48 and 64 threads; the ghost fill 1.64, 1.11, 0.86, 0.97;
    scatter 0.74, 0.70, 0.63, 0.68 — while the compute-bound initial
    data goes 1.55, 0.77, 0.56, 0.47. Every memory-streaming phase
    saturates, and the compute-bound one does not. Not the garbage
    collector (0 ms per call, 1.7 MB allocated per RHS evaluation, all
    of it per-task bookkeeping), not the GC or interactive thread
    counts (`--gcthreads=1` and `-t 64,0` change nothing), not the
    footprint (an 8-thread process over all 960 blocks runs the RHS at
    4.7 ns/cell, no slower than over 120). Pinning (`JULIA_EXCLUSIVE=1`)
    helps scatter 1.5x and the ghost fill 1.15x and leaves the RHS
    where it was.
  - *The stream itself localizes it in the launch path.* `bench/stream.jl`
    moves the same three 717 MB arrays through four mechanisms. In one
    64-thread interleaved process: `Threads.@threads :static` over
    per-thread chunks 205 GB/s; one `Threads.@spawn` per chunk under
    `@sync`, which is what the KernelAbstractions CPU backend does,
    96 GB/s; the KernelAbstractions triad kernel 62 GB/s; the same kernel
    under the backend's static scheduling 69 GB/s. Pinned: 138, 123,
    115 and 131. In an 8-thread process over the same arrays all four
    are equal at 48 GB/s, bandwidth-bound; from eight 8-thread
    processes the KernelAbstractions kernel aggregates about 290 GB/s
    and the static loop about 385, the latter 94 % of the node
    *(unsynchronized; in shared windows 169 and 224, the static loop
    exactly what one process reaches)*. So the per-thread streaming
    rate of a KernelAbstractions CPU kernel falls from 4–6 GB/s in an
    8-thread process to 1 GB/s in a 64-thread one with the same node
    load and the same pages, and a plain loop falls by half *(the
    8-thread rates were taken on an otherwise idle node; at equal node
    load the plain loop does not fall at all)*. The mechanism was not
    identified here. The candidates were thread wake-up cascades on
    spawn, migration of unpinned threads across sockets between
    launches, and something in the backend's per-workgroup loop that
    contends only at high thread counts. It is none of the three; the
    next paragraph but one has it.

  What this means for the plan (decided): NUMA-aware page placement
  inside one process is not worth building; it buys 1.2x and
  duplicates M7. One MPI rank per NUMA domain would recover the gap,
  but so does one process: *(amended the same day)* the gap is 2.2x,
  not 3x, and it is not a per-process limit but the loss of
  data-to-core affinity between launches, which a block-ownership
  launch policy recovers entirely inside one process (next paragraph).
  M7 therefore gains no bandwidth argument from this, and loses none:
  its partition of blocks over ranks and the ownership partition over
  threads are the same contiguous Morton ranges, one level apart. The
  M5 finding that pages must be interleaved stands as advice for Rome
  and is harmless on Milan. And
  the host `triad reference` figures quoted in the tables below, taken
  before 2026-09-23, are inflated: the benchmarks left the two input
  arrays unwritten, and on Linux an untouched allocation reads from the
  kernel's shared zero page, so the "triad" was a write stream reporting
  three times its bandwidth. Both benchmarks now write their inputs
  first; the device figures, whose memory is real, are unaffected, and
  the time ratios stand.

  **What one process loses: data-to-core affinity** (measured
  2026-09-23 on Symmetry, jobs 562406–562425 on nodes cn099–cn103, all
  EPYC 7543; `bench/symmetry_affinity.sh` reproduces the comparisons
  with `bench/affinity.jl`, `bench/affinity_mesh.jl` and
  `bench/owner.jl`; the `perf` and IBS counters needed
  `kernel.perf_event_paranoid=-1`, set by hand on cn099). The first step
  was to repair the comparison: processes given one start time now run
  each mode in the same wall-clock window and report what they moved
  inside it. In shared windows eight domain-bound 8-thread processes
  stream the static triad at 221–224 GB/s together, and one 64-thread
  process at 174–218, pinned or not; the node's ceiling for the same
  count is 244, eight processes on their own domains' memory. For a
  plain static loop there is no per-process limit. What remains depends
  on the launch path, and it is large (GB/s, chunks of 1.4M elements,
  pages interleaved, ranges over the runs):

  | launch                                   | 1 × 64 unpinned | 1 × 64 pinned | 8 × 8, shared windows |
  |---|---|---|---|
  | `@threads :static`, chunk t on thread t  | 200–218 | 174–207 | 221–224 |
  | the same, chunk map rotated every launch |  90–99  |  80–86  | 160 |
  | one `Threads.@spawn` per chunk           |  76–95  |  98–149 | 164–169 |
  | KernelAbstractions, default schedule     |  46–60  |  84–126 | 166–172 |
  | KernelAbstractions, static schedule      |  90–125 | 136–166 | 207–211 |

  Everything the earlier paragraph suspected is ruled out:

  - *Not the launch.* Long-lived tasks that spin on a counter, never
    created or woken per launch, stream no faster than spawn (83–202
    GB/s). Staggering their starts by 2 µs per thread brings them to
    220–224, which `@threads :static` gets for free by waking its
    threads one after another, about 9 µs apart. Setting
    `JULIA_THREAD_SLEEP_THRESHOLD=0` changes nothing, and making idle
    threads spin forever halves the static loop.
  - *Not the thread being off its core.* Every chunk's thread CPU time
    equals its wall time (99–100 %), with under 0.2 involuntary context
    switches per chunk. `perf record` puts 97–99 % of all samples on
    the loop's own loads and stores in every mode, and the scheduler
    under 1 %.
  - *Not the code.* A scalar loop the compiler may not vectorize
    streams as fast as the SIMD one (196–210 against 207–218). Retired
    instructions and DRAM fills per launch agree between the modes to
    within 13 %, the KernelAbstractions kernel included.

  The slow modes move the same bytes with the same instructions, and
  every load simply waits longer. IBS puts the mean L1-miss latency at
  1816 cycles in the static loop and at 3218–4256 in the spawned,
  rotated and cross-socket ones, in proportion to the rate.

  What differs is whether a chunk meets the core that streamed it the
  launch before. With threads pinned (slot t on CPU t − 1; eight CPUs
  per CCD, each CCD its own NUMA domain, 32 per socket) and chunk t
  alternating between threads t and t + K on successive launches:

  | K    | 0 | 1 | 4 | 8 | 16 | 32 |
  |---|---|---|---|---|---|---|
  | GB/s | 196 | 166 | 108 | 114 | 84 | 72 |

  K = 1 moves one chunk in eight to the next CCD, K = 4 half of them,
  K = 8 all of them within the socket, K = 32 all of them to the other
  socket. The farther data travels between two launches, the slower it
  streams. This is what "the process" stood for: an 8-thread process
  bound to one NUMA domain is one CCD on this node, so however its
  tasks are shuffled, its data never leaves that CCD.

  Two more observations say what kind of cost this is. On an idle node
  it does not exist: a core chasing pointers through, or streaming, a
  buffer last written or read by itself, by a core of its own CCD, of
  another CCD or of the other socket sees the same 74–79 ns and
  27–29 GB/s with the buffer in its own domain (`bench/owner.jl`). And with interleaved pages it outlives the
  scrambling: after one window of the rotated map the static loop runs
  at 110, 137, 153 and 150 GB/s in the next four three-second windows,
  against 195 before, while with first-touch pages it is back at 210
  at once. Both fit coherence traffic in the data fabric. A line last
  held by another CCX costs its home directory a probe to that CCX. A
  lone reader never notices, which suggests the DRAM read goes out in
  parallel with the probe. With 64 cores streaming, the probes and
  their answers compete with the data for the same links, and a
  directory entry left pointing at the wrong CCX stays until something
  evicts it. The fabric's own counters would show this directly, but
  the uncore PMU is not loaded on these nodes, so this step is
  inferred, not counted.

  *Where the package lost it.* Every per-evaluation phase broke
  affinity. `map_blocks!` and `scatter!` launched on KernelAbstractions'
  default CPU schedule, one `@spawn` per thread-chunk, so a chunk ran
  on whichever thread took it. `run_phase!` dealt ghost slices largest
  first to spawned tasks, so a thread's ghost targets lay in blocks
  that other threads scattered into and computed on. And the phases
  partitioned the blocks differently in any case. Three launch policies
  on the 960-block mesh, in shared windows, nanoseconds per cell
  averaged over the window (one-process columns on cn102, eight-process
  columns on cn103; the V0 unpinned RHS on cn103 was 2.56, so the nodes
  agree):

  | phase          | V0 unpinned | V0 pinned | V1 pinned | V3 pinned | V3 pinned, first touch | V3 unpinned | 8 × 8 interleaved | 8 × 8 domain-local |
  |---|---|---|---|---|---|---|---|---|
  | RHS evaluation | 2.49 | 2.45 | 2.26 | **1.01** | **0.94** | 1.23 | 1.16 | 0.99 |
  | ghost fill     | 1.01 | 0.92 | 0.95 | **0.39** | 0.42 | 0.51 | 0.44 | 0.40 |
  | scatter        | 0.75 | 0.44 | 0.30 | **0.27** | 0.27 | 0.28 | 0.30 | 0.28 |
  | RHS kernel     | 0.75 | 0.40 | 0.29 | **0.28** | 0.28 | 0.29 | 0.28 | 0.27 |

  V0 is the package as it is. V1 puts every package launch on the
  backend's static schedule, so repeated launches of one kernel give
  thread t the same blocks. V3 adds the ghost fill by owner: each
  group's transfers sorted by target block, and the task on thread c
  filling the ghosts of exactly the blocks thread c owns under V1. Both
  were installed by overwriting methods in the benchmark script (which
  now compares the implemented policy against a `spawn` control
  instead). V1 alone recovers each kernel repeated on its own, since
  scatter and the RHS kernel reach the eight-process figures. It does
  not recover the RHS evaluation (2.45 to 2.26), because the ghost fill
  in the middle re-scrambles the working array every time; under V0 the
  evaluation (77 ms) even costs more than its three phases run
  separately (55). V3 recovers all of it: 1.01 ns/cell, as fast as
  eight domain-local processes, and 0.94 with first-touch pages, which
  the ownership makes domain-local without `numactl`. Unpinned, V3
  still gains 2x (1.23), and pinning adds the rest.

  *What changed (implemented the same day, measured below).* One
  partition of blocks over threads, `threadchunks(nblocks)`, is used by
  every per-block pass for the lifetime of a forest generation:

  1. `map_blocks!`, `scatter!`/`gather!` (`run_over_interiors!`),
     `fill_by_coordinates!` and `zerofill!`, the first touch of every
     working array and state vector, go through `launch_by_owner!`. On
     the CPU that is KernelAbstractions' static schedule with one block
     per workgroup
     (`workgroupsize` the whole ndrange except a 1 on the block axis;
     no kernel in `src/` uses workgroup features). KernelAbstractions
     then splits the blocks exactly as `threadchunks` does
     (`divrem(nblocks, nthreads)`, the remainder to the first threads),
     and `@threads :static` puts chunk t on thread t every time. The
     package must substitute `CPU(; static = true)` itself, because
     `get_backend(::Array)` returns the default `CPU()` and the flag
     cannot travel on the storage. A device backend is untouched.
     Nothing upstream is required. KernelAbstractions' default is
     reasonable for a kernel run once and wrong for one run every step
     over the same data, which is worth an issue there, not a
     dependency.
  2. `run_phase!` in `ghosts.jl`, on the CPU, goes by owner instead of
     largest first, for the ghost fill, the interface fixup and the
     regrid transfer alike. Every builder already collects a group's
     transfers in block order, so `targetblocks` is non-decreasing (now
     documented and tested), and each thread finds its share of every
     group by bisection at fill time. `PhaseSlice` and `phase_plan` are
     gone. M5's point survives, since every group, face slab or corner,
     is still split across all threads, now by owner rather than by
     size.
  3. `threaded_chunks` in `threading.jl` puts chunk c on thread c on
     every call, so that the host loops over blocks (the reductions,
     the region boundary hook, the regrid bookkeeping) inherit the same
     map. It does so with sticky tasks placed the way `@threads :static`
     places its own, but without entering a threaded region, so unlike
     `@threads :static` it nests and may be called from two tasks at
     once.
  4. Documented: pin the threads (`JULIA_EXCLUSIVE=1` or
     ThreadPinning.jl) and then drop `numactl --interleave=all`, since
     first touch now lands each block on its owner's domain.

  None of this touches the M5 determinism rules: every work item still
  owns its output slot, and only which thread runs it changes. The
  thread-count digest test passes unchanged, on Julia 1.10 and 1.13.
  Three constraints come with it. First, KernelAbstractions' static
  schedule is `@threads :static`, which refuses to run nested inside
  another threaded loop, so `launch_by_owner!` checks
  `jl_in_threaded_region` and falls back to the default schedule there.
  A caller launching `map_blocks!` from inside their own `@threads` loop
  keeps working and only loses the affinity (tested). Second, a kernel
  handed to `map_blocks!` sees one workgroup per block on the CPU, which
  matters only to a kernel that uses local memory or `@groupsize`; the
  docstring says so. Third, ownership by
  block count balances cells, not ghost work, and ghost work
  concentrates at level boundaries. At 15 blocks per thread the owner
  fill was 2.3x faster than today's. A small in-cache smoke run at 7.5
  blocks per thread (16 threads, 120 blocks of `16^3`) had it 1.7x
  slower. If that case matters, the fix is a contiguous partition
  weighted by per-block cost, which is the partition M7 needs across
  ranks anyway. Ownership is by block index, so a multi-block forest
  changes nothing here.

  *Measured after the change* (job 562462, cn102, the same windows; the
  "before" column is the previous `src/` on the same node). Nanoseconds
  per cell, averaged over the window:

  | phase          | before, pinned, interleaved | after, pinned, interleaved | after, pinned, first touch | before, unpinned, interleaved | after, unpinned, interleaved | after, unpinned, first touch | spawn control, pinned | 8 × 8 domain-local, after |
  |---|---|---|---|---|---|---|---|---|
  | RHS evaluation | 2.42 | 1.00 | **0.92** | 2.55 | 1.30 | 1.46 | 1.70 | 0.87 |
  | ghost fill     | 0.91 | 0.39 | **0.36** | 1.02 | 0.56 | 0.60 | 0.57 | 0.32 |
  | scatter        | 0.44 | 0.30 | **0.26** | 0.76 | 0.28 | 0.33 | 0.41 | 0.26 |
  | RHS kernel     | 0.40 | 0.29 | **0.27** | 0.76 | 0.29 | 0.32 | 0.40 | 0.26 |

  The implementation reproduces the V3 prototype: 2.4x on the RHS
  evaluation pinned, and one pinned process with first-touch pages is
  within 6 % of eight domain-local processes, which gained a little
  themselves because ownership also helps inside a CCD. The spawn
  control keeps the new partitions but lets every launch land anywhere,
  and it gives back most of the gain. Unpinned, the policy still halves
  the RHS, but first touch is then worse than interleaving (1.46
  against 1.30), since an unpinned thread need not stay on the domain
  where it first touched its pages. Hence the advice now reads: pin,
  and then drop the interleaving; if you cannot pin, keep it.
  `bench/threads.jl` (best of 20, pinned, first touch) puts the 64-thread
  speedup over one thread bound to one domain at 38.6x on the RHS path,
  36.8x on the ghost fill, 52x on scatter and 51x on the norm, and the
  RHS at 0.81 ns/cell against 2.41 in the first table of "Where the
  64-thread efficiency goes". The per-evaluation path allocates 2.3 MB
  per RHS at 64 threads, against 1.7 MB before, for the sticky tasks and
  closures, and the collector still spends no measurable time on it.

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
    form grows an interior accessor. *(Amended in M10: reflection has
    left this list. It is a property of the domain and a transfer in
    the schedule, so it runs on every backend; see
    [Ghost filling](#ghost-filling). Extrapolating outflow, and anything
    else that reads the interior, is what remains CPU-only.)*
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
    added. (Amended after M8: it is now the two-launch reduction under
    "**Implemented**" below, 256 lanes per block, and still exact.)

  **The coordinate callbacks got a third form, over the variable axis**
  (amended for TreeHydro). The two forms above are about *where* a
  callback runs; this one is about *how much it returns*. `AllVariables(f)`
  makes `fill_by_coordinates!` and `CellBoundary` call their callback
  once per point instead of once per point and variable, with the whole
  `NTuple{nvars}` coming back at once — because a conserved state built
  from a primitive one cannot be produced a variable at a time, and
  producing it `nvars` times per point and keeping one number is what the
  per-variable form would cost, at every RHS evaluation in the hook's
  case. The all-variables kernels have no variable axis in their ndrange
  and unroll the write over the variables with a `Val`; they form the
  position with the same expression as the per-variable ones, so the two
  agree bit for bit and the M5 digests do not move. The concurrency
  contract is unchanged — one work item per point, each owning every
  variable slot of that point — and so is the `isbits` rule, which the
  single-field wrapper preserves. See
  [Application interface](#application-interface-sketch) for the
  argument and for the host-side length check.

  **Reductions got a device method, not a rewrite.** The diagnostics
  (`volume_weighted_norm`, `total_mass`) form one partial per block and
  sum the partials in block order. On a device the per-block host
  reduction would be one launch and one synchronization *per block*,
  issued from several host tasks at once; so the partials are formed in
  a single launch there instead. The CPU path is untouched — the M5
  numbers were measured with it — and the ordered combination is
  shared. (That combination was what the bit-identity of the sums
  rested on; since the narrowing above it is how the code happens to be
  written, not a promise.)

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

  The guarantee is stated as what it is (and narrowed after M8, see
  above): each block's value is bit-identical across thread counts,
  because every block owns its output slot; the combination is the
  caller's, and a floating-point one is promised to roundoff. Identical
  across *backends* was never claimed and, for a floating-point `op`, is
  not true — the association differs. The CPU numbers did not move:
  `volume_weighted_norm` at `p = 1, 2, 3, ∞` and `total_mass` reproduce
  their pre-M6 values bit for bit in `D = 1, 2, 3` and in both `Float64`
  and `Float32`. Both paths are reachable on `CPU()`, so the suite checks
  that they compute the same fold on every run and not only where there
  is a device.

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
  | triad reference       | 3842 GB/s | 205 GB/s (inflated, see below) | 18.7 |

  The host triad figure is the write-stream artefact described under
  "Where the 64-thread efficiency goes" above and overstates the host by
  up to 3x, which would make the last ratio larger, not smaller.

  The per-evaluation path — the only part that runs at every RHS
  evaluation — tracks the bandwidth ratio, which is what a mesh library
  should deliver and is the whole claim. Three rows deserve their
  explanation rather than a footnote:

  - **The two per-block reductions are the weak rows, by choice**
    (revisited after M8, below). `volume_weighted_norm` and
    `firing_boxes` both run one work item per *block*, so 960 work items
    on a device that wants tens of thousands. A hierarchical reduction
    would fix that and would give up the property that makes these
    functions trustworthy: one work item per block, each accumulating
    its own cells in its own order, is deterministic without a word of
    extra care, which is the M5 discipline. Neither is on the
    per-evaluation path — one is a diagnostic, the other runs at regrid
    frequency — so the trade is paid where it is cheap. It would have to
    be revisited if `volume_weighted_norm` were ever wired in as an
    adaptive integrator's `internalnorm`, which is still an open
    question above.

    *Revisited after M8.* The trade is no longer cheap and the property
    is no longer promised. TreeHydro takes a signal-speed maximum at
    every step for its CFL condition, through `block_mapreduce`; at the
    table's sizes that is 3.3 RHS evaluations per step on the device,
    which under a three-stage integrator doubles the step. And the
    argument above was only half right on its own terms: `firing_boxes`
    reduces an integer count and integer min/max, which are
    order-independent, so a hierarchical form of it is bit-identical
    anyway — it was one item per block by simplicity, not by necessity.
    Both were rewritten as specified under "**Implemented**" below, and
    re-measured on the H200 there: 0.459 ms and 0.970 ms against this
    table's 23.2 and 16.0.
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
  113 GB/s, both taken with the unwritten inputs noted above; whether
  macOS shares a zero page the same way was not checked), because on
  that part it is one memory system either way.

  **Implemented: the two-launch device reduction and the global scalar
  form** (decided after M8, amended before implementation, implemented
  and measured 2026-09-22). Two pieces, both made admissible by the
  narrowing of the bit-identity claim above and both wanted before M7.

  - **`block_mapreduce` stays as it is**: per-block, local to the
    process, a host `Vector`. Refinement criteria are per-block by
    nature — TreeWave's per-variable peaks and hot-cell counts,
    TreeHydro's floor counts — and under MPI they need no communication,
    which is why the per-block form must survive as the local one. Only
    its contract changed, as stated above; its *device path* is what
    gets rewritten.
  - **The device path becomes 256 lanes per block, in two launches and
    with no barrier** (amended before implementation; the first version of
    this block said a tree in local memory, see below; and amended after
    review, see the end of this bullet). The first launch has an `ndrange`
    of lanes times blocks, and reads block and lane off the global index.
    Lane `l` of block `b` strides over the block's `N^D · nvars` entries —
    cells fastest, then variables — in linear order, `l, l + 256, …`,
    converting each linear index to a Cartesian one so that adjacent lanes
    read adjacent cells, folds them from `init` into a private accumulator,
    and writes its partial to a scratch array of shape lanes by blocks. The
    second launch is one work item per block, folding that block's 256
    partials in lane order into its slot — a few hundred thousand operations
    at the table's size, so it costs nothing, and it keeps the copy back at
    `nblocks` values. Each partial is a function of the block's cells and
    the stride alone, and the lane fold has a fixed order, so the result is
    reproducible from run to run and exact for `max`, `min` and integer
    sums. *After review:* the lanes were first a static workgroup of 256 and
    the stride restarted for each variable. Metal.jl launches a static
    workgroup without checking the pipeline's own thread limit, so a
    register-heavy `f` could have failed to launch; with no barrier the
    lanes of a block need not share a workgroup, and now do not. And
    restarting per variable left only `N^D` lanes at work below 256 cells,
    whatever `nvars`; the stride now runs over cells and variables together,
    as this bullet had said all along. The first launch is the
    bandwidth-bound shape the RHS already has, and the acceptance is that
    the norm row tracks the RHS row's ratio in the M6 table instead of
    sitting at 2.5 — the norm reads the state vector once, a fraction of a
    millisecond at the H200's triad rate, against 23.2 ms measured. The
    kernels are KernelAbstractions, not an array library's `mapreduce`: the
    package depends on KernelAbstractions alone. The CPU keeps the threaded
    per-block `mapreduce` it has — that is what M5 measured, and a workgroup
    on the CPU backend is one task looping — and both paths stay reachable
    on `CPU()`, so the suite keeps checking that they compute the same fold
    to roundoff. `firing_boxes` gets the same two launches, its three
    order-independent accumulators (count, low corner, high corner) folded
    the same way, and stays bit-identical while doing so.

    *Why not a tree in local memory.* The barrier is the problem, on
    the one backend where a tree gains nothing. KernelAbstractions
    0.9.42 — the floor of this package's compat bound — implements
    `@synchronize` on the CPU backend by splitting the kernel into
    separate loops over the workgroup at each barrier, so a local that
    is written before a barrier and read after it does not survive
    unless it is declared `@private`, and a barrier inside a loop splits
    the loop. That is a documented pitfall, it is invisible on CUDA and
    Metal, and the CPU backend is exactly where the suite cross-checks
    the device fold. Two launches have no barrier and nothing to get
    wrong; the second could be folded into local memory later if it ever
    showed in a profile, which at `nblocks` items of 256 values it will
    not.
  - **`init` must satisfy `op(init, init) == init`.** The first launch
    starts every lane from `init`, so it enters the result once per
    lane rather than once (the second launch starts from the first
    lane's partial, so not once more). A neutral element satisfies
    this, which is what Base's `reduce` asks of its `init`; so does any
    `init` under an idempotent `op`, which matters because
    `block_mapreduce(identity, max, zero(R), …)` over signed data —
    which the downstream packages write — is not a neutral `init` and
    was never wrong. The docstring of `block_mapreduce` said the
    accumulator "starts at `init`", which a one-item-per-block kernel
    made true and a hierarchical one does not; it states the condition
    now.
  - **A global scalar form, `mesh_mapreduce`.** `mesh_mapreduce(f, op,
    init, fs[, u]; vars, weight = nothing)` returns one number: the
    per-block values of `block_mapreduce`, each scaled by `weight(key)`
    when a weight is given, combined over the local blocks and — once
    M7 exists — across ranks with `Allreduce`. The weight is a host
    function of the block's key, applied on the host to the per-block
    values before they are combined, because that is where the geometry
    is and because the cross-block stage is `nblocks` numbers and not
    worth a launch; it is meant for sums (a cell volume,
    `spacing(key)^D`) and is documented as such. `volume_weighted_norm`
    and `total_mass` become calls to it, which is how they turn global
    in M7 without changing signature; the M7 `Allreduce` lives in
    `mesh_mapreduce`'s combination step and nowhere else — the norm's
    domain volume goes through the same step (amended after review,
    which found it summed separately) — since the mesh owns the
    communicator and an application must not be asked to. `+`, `max`
    and `min` map to the builtin operations, anything else to a custom
    one. The name sits beside `block_mapreduce`: one returns a value per
    block, the other one value for the mesh.

    *The combination is `mapreduce(identity, op, values)` with no
    `init`, and `init` is returned only for an empty vector.* This is
    what keeps every recorded norm and mass where it is. `sum(v)` and
    `mapreduce(identity, +, v)` agree bit for bit — both are Base's
    pairwise `@simd` reduction — but `reduce(+, v; init = 0.0)` does
    not, because an explicit `init` turns the combination into a
    sequential left fold (measured on 960 values, 2026-09-22). The
    weight is applied as `values[b] *= weight(key)`, the multiplication
    order `total_mass` uses today, so that stays exact too.
  - **What this changed around it.** The guidance that `block_mapreduce`
    belongs at diagnostic or regrid frequency relaxed, with the device
    path bandwidth-bound, to "not inside a right-hand side": a per-step
    reduction is a fraction of an RHS evaluation on either backend. The
    open question of wiring `volume_weighted_norm` in as an adaptive
    integrator's `internalnorm` lost its cost objection at the same
    time. The thread-independence digests did not move — the CPU fold
    is untouched — and that was the acceptance for the rewiring: the 56
    lines `test/thread_workload.jl` prints, the `l2` and `mass` sums
    included, were recorded before the change and are byte for byte
    the same after it. The device tests compare the reductions to
    roundoff as before and gained three checks: two device calls return
    identical bits, a block smaller than a workgroup and one whose cell
    count is not a multiple of the lane count agree with the host, and
    `firing_boxes` matches the host sweep exactly at those sizes too.

  *Measured on the Apple M3 Pro under Metal, in `Float32`* (120 blocks
  of `32^3`, 3.9M cells, two variables; `bench/gpu.jl`, best of 20 —
  best of 5 for `firing_boxes`, as the benchmark times it — before and
  after the change on the same day, the "after" column being the form
  after review):

  | phase                 | one item per block | two launches | speedup |
  |---|---|---|---|
  | RHS evaluation        | 13.0 ms | 11.5 ms | — |
  | volume-weighted norm  | 16.3 ms | **0.85 ms** | **19x** |
  | `firing_boxes`        |  9.2 ms | **1.76 ms** | **5.2x** |
  | triad reference       |  1.24 ms | 1.14 ms | — |

  The norm went from 1.3 RHS evaluations to 7 % of one. Its cost is half
  fixed: the same call on blocks of `4^3`, with almost nothing to read,
  takes 0.38 ms — two launches, two allocations, the synchronization and the
  copy back — and the other 0.4 ms reads the 31 MB state vector at about two
  thirds of the triad rate, the bandwidth-bound shape the design asked for.
  (As a static workgroup, before the review, the norm measured 0.67 ms; the
  consecutive-lane form with the flattened stride is slower by that much and
  is kept for the reason given above.) `firing_boxes` is 5x faster and still
  well above its bandwidth floor, and the same small-block measurement says
  why: 0.79 ms at `4^3` against 1.76 ms at `32^3`, so nearly half of it is
  per-call overhead — two uploads of the block origins and spacings, six
  allocations, two launches, three copies back. That is what to trim if it
  ever matters; it does not at regrid frequency, so it is left as is. The
  review also found that on the *CPU backend* the one-item form had been
  serial all along: an ndrange of up to 1024 items is a single workgroup
  there, i.e. one task. Measured at 120 blocks of `32^3` in `Float64`,
  `firing_boxes` took 67 ms at one thread and 68 ms at eight before; it
  takes 78 ms and 14.9 ms now. So the `firing_boxes` row of the H200 table
  above compares a device against one core, not sixteen, and its 5.5x was
  against a serial sweep. Two things the implementation turned up are
  recorded here rather than lost.
  The lane fold in `firing_boxes` first failed to compile on Metal — a
  closure inside `ntuple` capturing the running corner, which the loop
  reassigns, is boxed and becomes a dynamic call — which is the trap
  `widen_lo`/`widen_hi` already exist for, so the fold goes through them
  too. And `fill_by_coordinates!` evaluating `sin` on the device gives a
  value one ulp from the host's at many points in `Float32` — 8.7 % of the
  entries, in the review's count over 11M — which the one-item form had
  never exposed because no test compared per-block maxima at those
  positions; the device tests now reduce a copy of the device's data on the
  host instead of recomputing it, since the reduction is what is under
  test, not the transcendental.

  *Measured on the H200* (`bench/symmetry_gpu.sh`, job 562304 on
  Symmetry, 2026-09-23, at release 0.1.2; the same 960 blocks of `32^3`,
  31.5M cells, `Float64`, and the same node's 16 cores under
  `numactl --interleave=all` as the M6 table). The suite passed on CUDA
  first — 90202 tests in `Float64` and `Float32`, 7m58 on 8 threads — so
  the two-launch kernels have now compiled and passed on every backend
  the package runs on. The two rows the change was for, with the M6
  one-item-per-block numbers beside them:

  | phase                 | M6, one item | two launches | 16 cores | ratio |
  |---|---|---|---|---|
  | volume-weighted norm  | 23.2 ms | **0.459 ms** | 53.1 ms | **116** |
  | `firing_boxes`        | 16.0 ms | **0.970 ms** | 21.3 ms | **22.0** |
  | RHS evaluation        |  7.0 ms | 4.57 ms | 104 ms | 22.8 |
  | triad reference       |  0.56 ms | 0.56 ms | 8.7 ms (inflated, as in the M6 table) | 15.6 |

  The norm is 51x faster than the one-item form and costs 10 % of an
  RHS evaluation (0.38 ms and 8.6 % in `Float32`); `firing_boxes` is 16x
  faster. The norm's ratio now exceeds the bandwidth ratio, which says
  more about the host than the device: the host norm reads 503 MB in
  53 ms, 26x above its own bandwidth floor, because a threaded
  `mapreduce` over per-block `IndexCartesian` views is a sequential fold
  per block — the other half of the M6 observation that the CPU numbers
  were "what M5 measured". It is not on the per-evaluation path and is
  left as is. Two rows moved for reasons that have nothing to do with
  reductions and are recorded because this is their first H200
  measurement: the RHS evaluation and the ghost fill went from 7.0 and
  5.3 ms to 4.6 and 2.8 ms, which is the ghost-fill work under
  [What the ghost fill costs](#what-the-ghost-fill-costs) arriving on
  the device, and the host `firing_boxes` went from 88 to 21 ms, which
  is the serial-workgroup finding above. The M6 table stays as the M6
  measurement.
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
  integrators as `internalnorm` (post-M3). Its cost objection on a
  device — the norm at one work item per block cost three RHS
  evaluations — went with the two-launch reduction: the norm is 7 % of
  an RHS evaluation on Metal and 10 % on the H200, measured under
  [Parallelism](#parallelism).
- The one-dimensional operators of the conservative family along a
  vertex-like dimension: refused until an application needs them, with
  order-`(p+1)` Lagrange as the recorded candidate (see
  [Operators](#operators)).
- A state vector spanning several field sets: specified under
  [Time integration](#time-integration), implemented with the first
  application that needs it (constrained-transport MHD).
- **The integrator's own passes are not owner-based** (raised by
  TreeGeneralizedHarmonic, 2026-09-25, after it adopted the ownership
  policy of [Parallelism](#parallelism); not decided here). An external
  integrator forms its stage vectors itself, and OrdinaryDiffEq's `RK4()`
  does so with FastBroadcast's `@..` on one thread (`thread = Serial()`),
  over the whole state, between every two `map_blocks!` launches — so on
  a many-core node it is both a serial pass and the migration the
  ownership policy removed from `src/`. It also allocates its buffers
  (seven `similar` for RK4's cache, six `recursivecopy`s) on the calling
  thread at every `solve`, so they are first-touched there and not by
  owner. *Measured* (Symmetry job 563749, cn079, one exclusive 64-thread
  process; TreeGeneralizedHarmonic's gauge wave at `q = 4`, 512 blocks of
  `16³`, 20 variables, a 320 MB state vector; each configuration twice,
  the second pass in reverse order, agreeing to a few percent; milliseconds,
  minimum per call):

  | | unpinned, first touch | unpinned, interleaved | pinned, first touch | pinned, interleaved |
  |---|---|---|---|---|
  | one RHS evaluation | 431–435 | 438–439 | **377–383** | 414–416 |
  | `u + a k`, OrdinaryDiffEq's serial `@..` | 48 | 58 | 53 | 58 |
  | the same, `launch_by_owner!` kernel | 30 | **6.1** | 19–22 | **6.2** |
  | the same, `@.. thread = True()` (Polyester) | 26–30 | 6.0 | 17–26 | 6.0 |
  | one RK4 step through `solve` (4 steps a call) | 3273–3378 | 3317–3350 | **2986–2996** | 3171–3249 |
  | 4 RHS + the serial stage updates | 1957–1973 | 2032–2035 | 1753–1781 | 1930–1942 |

  The serial updates are 12–14 % of a step at 64 threads (1 % at four on
  the development machine), and an owner-mapped kernel does them 8–9×
  faster on interleaved pages. `solve` itself adds about 1.2 s a step over
  its parts at four steps a call — the allocation and serial first touch
  of about fifteen state-sized vectors and the one extra RHS of the FSAL
  initialisation — which an application that calls `solve` for a few
  steps at a time (TreeGeneralizedHarmonic's moving hole does) pays in
  full. Pinning with first touch is still the fastest configuration, as
  "What one process loses" says, but *under first touch the owner-mapped
  update is 3–5× slower than under interleaving*: the pattern of vectors
  that live on one domain. The candidate cause, not proven, is Linux's
  automatic NUMA balancing, which is on for cn079 (`numa_balancing = 1`):
  in each process the serial passes ran before the owner-mapped ones, and
  the kernel migrates default-policy pages toward the core that keeps
  touching them — core 0 — and leaves interleaved pages where they are.
  A rerun with the serial passes last, and `numastat -p` beside it, would
  settle it; if it holds, the first-touch advice needs "and nothing serial
  touches the state" attached.

  *Polyester is not the way out*, measured in the same job. Its stage
  update is no faster than the owner-mapped kernel (6.0 ms against 6.2),
  and it does not compose with this package's launches: after every
  `@batch`, ThreadingUtilities' workers spin for `2²⁰` `pause`s before
  yielding (`threadtasks.jl`), and the sticky tasks of `threaded_chunks`
  and the static KernelAbstractions schedule wait behind them. An
  owner-mapped launch right after a Polyester loop took 33 ms against
  12.5 ms after another owner-mapped one on cn079, and 9.8 ms against
  0.06 ms on the development machine (Apple silicon); an RHS evaluation
  right after one ranged from 275 to 598 ms against a steady 414.
  `RK4(; thread = True())` gains 0–6 % a step on Symmetry and loses 50 %
  on the development machine. It is also CPU-only, does not nest, and
  matches the block ownership only by coincidence (when every block has
  the same size).

  *The suggestion is an owner-aware state vector type, here.* FastBroadcast
  handles only `DefaultArrayStyle` itself and hands every other style to
  `Base.materialize!`, which ends in `copyto!(dest, bc::Broadcasted)` —
  the hook `CuArray` uses. A wrapper around the flat vector that carries
  the block shape `(N…, nvars, nblocks)`, with its own `BroadcastStyle`,
  a `copyto!` that runs `dest[I] = bc[I]` through `launch_by_owner!` over
  the block-shaped view, and a `similar` that allocates and zero-fills by
  owner as `statevector` does, would make every stage update *and* every
  integrator buffer owner-based with no change to the integrator. A
  host-side prototype (`ownedstate.jl` in the directory below: a plain
  loop in `copyto!`, not yet a kernel) confirms the routing:
  OrdinaryDiffEq's `RK4` accepts it, returns it, sends exactly its four
  combinations a step through the overridden `copyto!`, allocates its
  thirteen buffers through its `similar` (once per `solve`, never per
  step), and gives bit for bit the answer of a plain `Vector`. What it
  leaves open: `statearray`, `scatter!` and `gather!` would accept it (or
  its parent); host reductions over it fall back to the generic
  `AbstractVector` methods; on a device the `Broadcasted` has to be
  adapted with its arrays; and TreeWave and TreeHydro, which have the
  same serial updates, would pass it to `solve` too. The per-`solve`
  allocation stays either way: that one is the application's, whose fix
  is to keep one integrator per mesh generation rather than call `solve`
  per chunk. The script, its log and the prototype are in
  `/mnt/beegfs/eschnetter/claude/TreeGeneralizedHarmonic/threading-bench`
  on Symmetry (`threadbench.jl`, `threadbench.sbatch`,
  `out/threadbench-563749.log`).

## Milestones

Each milestone has a concrete acceptance test; serial correctness is
established before any parallelism. The numbers are the order the
milestones were planned in; **M8 is done before M7** (decided in the M8
design, see the M8 entry), and so is M10, which was added after M8. The
list below is in execution order.

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
  physics. *(Done.)* *(Amended in M8a: the wave study is
  **vertex-centered** from M8 on — `test/wave_tests.jl`, with its own
  rate table under [Operators](#operators) — and this cell-centered
  study is kept verbatim as `test/wave_cell_tests.jl` so the M3 numbers
  stay under test.)*
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
  schedule shape and the reductions — the floating-point sums among
  those narrowed to a roundoff promise after M8, see
  [Parallelism](#parallelism); the digests still agree, because the CPU
  fold did not change. Scaling on a 64-core AMD EPYC 7532
  (8 NUMA domains, 960 blocks of `32^3`): **36.3x** on the RHS path,
  59.5x on the compute-bound initial-data pass, with the table and the
  two findings that got it there — a phase must be one parallel loop,
  and pages must be interleaved — under [Parallelism](#parallelism).
  `bench/scan.sh` reproduces the measurement. Re-measured on a Milan
  node on 2026-09-23 ("Where the 64-thread efficiency goes" and "What
  one process loses", same section): the remaining gap is not page
  placement but the loss of data-to-core affinity between launches,
  which the block-ownership launch policy recovers in one process
  (implemented the same day: 38.6x on the RHS path at 64 threads,
  pinned). *(Done.)*
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
  Burgers.** *(Done.)* Done before M7 (decided): the MPI exchange is
  built over the schedule, and with the schedule layout-generic first, M7
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
    *(Done.)* See "**Implemented in M8a step 1**", "**step 2**" and
    "**step 3**" under [Centerings](#centerings) for what the design
    left open and the implementation settled. Nothing changed a measured
    cell-centered number, as predicted: the whole suite passes at one
    and eight threads with the M3 wave tables unchanged, and the
    thread-independence digests still agree byte for byte. The
    centering's own acceptance tests pass — the M2 exactness claim over
    all `2^D` centerings in `D = 1, 2, 3` at `p = 2` and `p = 4`, the
    write-count partition of the stored points including the shared
    plane and with `G = 0` along a stagger, bit-for-bit injection on
    position-determined data (with the cell-centered control showing
    *no* coincident points across levels at all), the `G ≥ p/2 − 1`
    bound, the conservative refusal, and the regrid transfer exact for
    every centering — and so does the wave equation on top of them: the
    predicted rates **1 and 2** at prolongation orders 2 and 4, measured
    0.99 / 1.99 in `D = 1` and 1.01 / 1.99 in `D = 2`, at **`G = 1`**,
    with the restriction order and the second ghost plane both
    bit-for-bit inert, the M4 pulse tracked to the uniformly finest
    mesh's accuracy (0.0240 against 0.0233, at 176 cells against 256),
    and two staggered cycles in the thread workload. The table is under
    [Operators](#operators).
    *Accept:* the M2 exactness test (degree `p − 1`, face/edge/corner,
    three levels) over all `2^D` centerings in `D = 1, 2, 3`, the oracle
    averaging along cell-like dimensions and sampling along vertex-like
    ones *(amended in M8a step 2: the averaging half has no consumer,
    since the conservative family is refused along a stagger — see
    [Centerings](#centerings))*; vertex restriction bit-for-bit
    injection on arbitrary data; **the wave equation is vertex-centered
    from here on**, with the M3 study
    repeated on it — predicted rates **1 and 2** for prolongation orders
    2 and 4 at *any* restriction order, since restriction is exact, and
    `G = 1` sufficient at order 4 where cell centering needs 2 — while
    the cell-centered study is kept verbatim as `wave_cell` so the M3
    numbers stay under test; the thread digests. The cell-centered
    stencils are the same rational weights as before, so M8a changes no
    measured number. *(All of this is measured; see above.)*
  - **M8b — conservation.** *(Done.)* `InterfaceSchedule` /
    `restrict_interfaces!`
    *(step 4; see "**Implemented in M8b step 4**" under
    [Conservation](#conservation-at-coarse-fine-faces))*,
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
    suite. *(All of this is measured, in M8b step 5;
    `test/burgers.jl` and `test/burgers_tests.jl` are the study.)*
    **Mass is conserved to 0.5-2.5 ulp** of the domain integral, and the
    drift does not grow with the step count; the negative control leaks
    3.8e-5 to 3.2e-4 in the same runs, eleven orders of magnitude more,
    with the full table and the `Float32` caveat under
    [Conservation](#conservation-at-coarse-fine-faces). A **uniform**
    mesh conserves with or without the fixup, which is what pins that on
    the coarse-fine faces rather than on the scheme. The interface-order
    rule comes out as predicted — L∞ rates **1.00 / 1.97 / 1.97** in
    `D = 1` and **0.84 / 1.81 / 1.77** in `D = 2` for `p = 1, 3, 5`,
    with `p = 3` and `p = 5` matching the *unrefined control's* own rate
    — and the **norm turned out to be part of the result**: an integral
    norm sees none of it, because a flux divergence leaves the defect on
    the interface instead of radiating it as a second derivative does
    (both under [Operators](#operators)). The tracked shock matches the
    uniformly fine reference at **6.6e-4** against that mesh, where the
    uniform coarse mesh is **5.2e-3** — 7.9x worse — using 80 cells
    against 128, and with the travelling buffer removed the refined
    region falls off the shock entirely. The Burgers cycle is in the
    thread workload (bit-identical at 1 and 8 threads) and in the device
    suite, where Metal in `Float32` reproduces the CPU numbers bit for
    bit.
- **M10 — Reflecting boundaries.** *(Done.)* Done before M7, for M8's reason: M7
  then distributes mirror transfers as the ordinary transfers they are,
  rather than retrofitting them. `reflecting` per face on the forest,
  `parity` per variable and dimension on the field set, and the mirror
  transfers and the derived upper wall point under
  [Ghost filling](#ghost-filling). Until now a reflection could only be
  written as a region-form boundary hook. That form is CPU-only, has to
  repeat the mirror index arithmetic per centering, and reads stale
  ghosts at a mixed edge or corner region whose tangential neighbor is
  coarser. *Accept:*
  - refusals, each saying why: periodic and reflecting in one dimension,
    a missing parity, `NoParity` in a reflected dimension;
  - the M2 exactness test with data even or odd about the wall, of
    per-dimension degree `p − 1`, over three levels touching the wall,
    at `p = 2, 4`, for every centering. It covers the derived upper wall
    point: zero for odd data, exact for even. The conservative family
    is tested on cell averages;
  - a reflecting domain reproducing the mirrored doubled domain on
    arbitrary data, to roundoff, including a coarser tangential neighbor
    at the wall;
  - every ghost written exactly once, the hook never handed a
    reflecting region;
  - **no ghost undefined, and none read before it is defined**, for
    every combination of periodic, outer and reflecting faces in
    `D = 1, 2, 3` and every centering: `NaN`-prefilled storage with
    finite owned data, checked for `NaN` after one fill;
  - the regrid transfer exact at a wall;
  - the wave equation with odd and with even data on a reflecting half
    domain agreeing with the full domain, and converging at the
    predicted rate at a vertex-centered upper wall;
  - a reflecting cycle in the thread workload and in the device suite.

  *(All of this is measured; `test/reflect_tests.jl` is the study, with
  its oracles at the end of `test/ghost_oracles.jl`.)* The design needed
  no amendment: the mechanism above went in as specified, and every
  mirror point landed inside the tangential stencil's range, as
  `N ≥ 2G + 2c` promises. Four things are worth recording.

  - **The doubled domain is reproduced to roundoff.** It is bit for bit
    in `D = 1`, and within 4.4e-16 in `D = 2` and 1.6e-15 in `D = 3` over
    up to 221184 stored values, on arbitrary data. That holds cell-centered
    at either wall and vertex-centered at the low one. The residue is
    summation order: a mirrored stencil adds the same terms as the
    doubled domain's, reversed.
  - **The `NaN` test catches the ordering bug it is there for.** It
    covers 235 cases: 10 in `D = 1`, 100 in `D = 2`, and in `D = 3` the
    125 face combinations with the centering rotating. All are clean, and
    the values are exact to 1.2e-14. As a check that the test is not
    vacuous, mirrored prolongations were moved into phase 1, where the
    hook used to run: it then failed 39 to 41 of the 42 2D cases with a
    reflecting face, and all 234 in 3D. 1D has no mirrored prolongation
    to misplace. The 2D count varies between processes because the order
    of the groups within a phase does: a misplaced prolongation reads its
    source stale only if it runs before the group that fills it. That
    order is free precisely because phase 1 is order independent. It costs about 11 s warm, most of it the 3D sweep.
  - **The wave equation keeps its rate.** Order-4 operators on a box
    with reflecting walls at `x₁ = 0` and `L/2`, `N = 8, 16, 32`, give
    L2 rates **1.91 / 2.02** (vertex, odd / even) and **1.99 / 2.00**
    (cell) in `D = 1`, and **1.96 / 1.99** and **1.99 / 2.00** in
    `D = 2`. The periodic box with the same refinement gives 1.99–2.00.
    The derived upper wall point therefore costs no order, as the
    interface-order rule predicts. The lowest of the eight, vertex-odd in
    1D at 1.91, is inside the 0.15 the periodic studies are held to; its
    L∞ rate is 1.95. Why that one case sits lowest was not
    investigated.

    The half box also evolves as the periodic box it folds.
    Cell-centered, at `N = 16`, the L∞ errors of the two agree to a
    relative 3.0e-11 (`D = 1`) and 2.6e-13 (`D = 2`) after 128 and 91
    RK4 steps, on half the blocks.
  - **Cost.** The suite went from 89168 tests in 3m31 to 89961 in 4m49,
    at 8 threads. The M10 testsets add up to about 42 s of that, and the
    device subset (Float32 compilation on the CPU backend) and the two
    reflecting cycles in each thread-workload subprocess about 8 s more.
    The rest is inside the run-to-run noise at this length. On Julia 1.10,
    in one thread, it was 97744 tests in 2m53. The 3D order-4 exactness
    sweep was then dropped as a duplicate of the `NaN` test's, which left
    89836 tests in 3m37 in one thread on the current Julia.
- **M7 — MPI.** Curve partitioning, distributed ghost exchange (for
  every centering, and the interface restriction with it, since both are
  transfers over the same schedule machinery), distributed regridding,
  and the `Allreduce` inside `mesh_mapreduce` (the planned global
  reduction under [Parallelism](#parallelism)), the one place a
  communicator appears in a reduction. *Accept:* results match serial
  to roundoff — bit-identical for everything but floating-point sums,
  as [Parallelism](#parallelism) states; weak-scaling smoke test; then
  MPI+GPU with CUDA-aware MPI.
- **M9 — I/O and visualization.** HDF5 output, checkpoint/restart, VTK
  export.
