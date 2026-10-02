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

### Point interpolation

(Added in M11, for the horizon finder of TreeGeneralizedHarmonic, which
carried a stopgap of its own; `src/interpolate.jl`.) Everything above
moves data between the mesh's own points. An analysis needs the other
direction as well — the field, and its gradient, at points the mesh did
not choose: a horizon finder's trial surface, asked for some 500 points
about 55 times per find; a tracer; a sampled ray. `interpolate(fs, xs,
basis; derivs, vars, exclude)` answers that for a whole batch in one
launch on the field set's backend, and `locate_point(forest, x)` is its
first step on its own. Decided:

- **The stencil of a query** is the `n^D` stored points of one block the
  interpolant reads: `n` consecutive stored indices per dimension, in the
  block containing the point, **ghosts included**. It never crosses into
  another block's array, which keeps a query a gather from one block, so
  the ghosts must be current, filled with the boundary hook, as for any
  stencil. Every stored point is exchange- or hook-filled (see
  [Blocks](#blocks)), so a stencil may use all of `1 : N + 2G_d + c_d`.
- **Location is one binary search.** The point is mapped to the node at
  the finest level present that contains it — the root brick position,
  then the node's coordinates bit by bit from the fraction within the
  root, doubling being exact in any binary type, so `2^L` never has to be
  an integer in `T` — and the covering leaf is the *last leaf not after
  that node* in curve order. That is correct because the leaves tile the
  domain and an ancestor sorts immediately before its contiguous subtree,
  so no leaf lies between the covering leaf and the node. The stopgap
  searched each ancestor in turn, `maxlevel` searches per point. The
  comparison is `curve_less` on `(root, padded coordinates, level)`, the
  same function `isless` on keys now calls, because the checking key
  constructor cannot run in a kernel.
- **Folding at faces.** Along a periodic dimension the point is wrapped
  into the domain — and the *wrapped* point is what the stencil is built
  from (the stopgap wrapped for location only, a latent bug no Dirichlet
  case could see). Beyond a reflecting face it is mirrored once, `x →
  2w − x` (decided with Erik: a symmetric run's horizon finder queries
  across the wall), and the value takes the variable's parity sign from
  `fs.factors` — the table the mirrored transfers already multiply by —
  and each derivative across the wall one sign more. A point outside the
  domain after that is refused rather than taken from the nearest block,
  which would extrapolate without saying so.
- **The basis is the extension point.** `Lagrange(n)` is the one
  implemented. The kernel knows a basis only through `stencilwidth`,
  `stencilstart` (which `n` points, from the query's continuous stored
  index) and `basisweights` (the weights and their first `M` derivatives
  at the offset `ξ`), so a smooth basis built from the same nodal data —
  Lagrange interpolation is only `C⁰` — is a new type and three methods.
- **Lagrange's stencil changes only at stored points.** It starts at
  `floor(s) − floor((n−1)/2)`: centered for even `n`, half a point off for
  odd `n`. The naive centering `floor(s − (n−1)/2)` switches between
  nodes for even `n`, where the two stencils disagree, and the
  interpolant jumps there; switching *at* a node, where every stencil
  interpolates the same value, makes it continuous inside a block. The
  start is then clamped into the stored array, which moves the stencil
  *toward* the point, so it never extrapolates and only goes off center
  at the array's edge — never when `G ≥ n/2` (cell) or `G ≥ n/2 − 1`
  (vertex), and exact on the same polynomials either way, which is what
  makes a `G = 0` field set interpolable at all. The same idea as
  point-value restriction's shift.
- **Weights by truncated Taylor products.** The numerator `∏_{j≠k}(ξ −
  j)` is carried as a series in `ε`, `∏_{j≠k}((ξ − j) + ε)` to order `M`,
  so derivatives come with it; the denominators are integers. Nothing
  divides by `ξ − j`, so a query on a node is exact (the barycentric
  form's `0/0` there is the first thing an exactness test hits). The
  schedule's exact rational weights do not serve: they are cached per
  offset, and a stream of arbitrary offsets would grow the cache without
  bound.
- **Derivatives as multi-indices, first order for now.** `derivs` is a
  tuple of `D`-component multi-indices in physical units. The weights,
  the contraction, the `h^{|m|}` scaling and the mirror signs are all
  written for any order; one check refuses `|m| ≥ 2` until tests claim
  it, so second derivatives need no change of interface.
- **Sum factorization along dimension 1.** Each row of `n` points is
  contracted once per derivative order in `x₁`, and only the partial
  sums meet the other dimensions' weights, whose products — one per row
  and requested derivative — are formed once per point rather than once
  per variable. For the value and gradient in 3D at `n = 6` that is 16
  multiplications per row and variable instead of 72. Both were
  measured to matter, on the M11 benchmark below: forming the outer
  products per variable took 1.9 ms per batch against 1.2, and a
  reassigned tuple captured by an `ntuple` closure (boxed, so one
  allocation per update, and not compilable for a device) made the first
  version take 505 ms.
- **Excluded regions flag, they do not throw.** An optional `exclude`
  region marks every query whose stencil has a point inside it; the
  value is computed either way and the caller decides. The stopgap's
  guard threw, because a horizon inside the damping layer is a bug
  there; a sampler may want to ignore the flag. The region is an
  axis-aligned `Ellipsoid` (a ball is the round case), and its test is
  exact and `O(D·n)` rather than `O(n^D)`: the scaled distance is a sum
  of per-dimension terms, so the stencil point nearest the center is the
  per-dimension nearest, and the stencil reaches inside iff that point
  does. It agrees bit for bit with enumerating the stencil, since the
  sum at the nearest point adds the same terms in the same order.
  `Region` is abstract, with enumeration as the default for other
  shapes. Stencil positions are those `coordinates` gives, evaluated by
  the same expression.
- **Errors after the launch.** A point outside the domain writes block
  `0` into its own slot, and the host reports the first such point once
  the batch is done. A device kernel cannot throw, and on the CPU an
  exception would arrive wrapped in a `TaskFailedException` — the
  stopgap moved to `threaded_foreach` precisely so that its refusal
  reached the caller readable.
- **One launch over points.** Points are not blocks, so this is a plain
  launch, not a by-owner one; on the CPU the workgroup is sized to give
  every thread a share, since a batch of a few hundred points would
  otherwise fit one default workgroup and run on one thread. Every query
  writes only its own slots, so the result is bit-identical across
  thread counts (a line of `test/thread_workload.jl` says so). The
  leaves, origins and spacings are uploaded per call, which at analysis
  cadence costs nothing that matters.

Under M7 the leaves a rank holds are its own, so a query must first be
routed to the rank that owns its block — the one place this design will
change. The location already produces exactly the block index that
routing needs. *(Designed 2026-10-01 under
[Distributed meshes](#distributed-meshes): the location stays on the
host against the replicated leaves, two `alltoallv`s route the points
to their owners and the values back, and the kernel is unchanged.
Implemented in M7's step 5, with the kernel unchanged except for one
argument, the block offset, which is 0 serially. Over a distributed
forest an outside point is found on the host before anything is
routed, and refused on every rank there, rather than after the launch.
A device batch goes through the host for the messages. The serial path
measured the same before and after.)*

### Checkpoint and restart

*(Designed 2026-09-29 as M9a, before implementation, and decided with
Erik the same day except where marked; implemented, and its throughput
measured, the same day — see "Throughput and filters" below.)*
Long runs outlast a queue's day: TreeHydro's showcase on an H200 at
10–20 levels, and TreeGeneralizedHarmonic's production runs, estimated
at 38–149 h. An application calls one function at a chunk boundary to
save the forest, its evolved field sets and its own plain data; a fresh
process loads them exactly, on any thread count and any backend, and
continues **bit-identically** at any thread count (across backends,
see "Loading" below). A file this version cannot interpret is refused
with the reason, and with a way to recreate the environment that wrote
it.

**What is saved and what is rebuilt** (decided). A checkpoint holds
what cannot be recomputed, and nothing else:

| Item | Handling | Why |
|---|---|---|
| Forest: `D`, the geometry type, `roots`, `periodic`, `reflecting`, `extents` (bitwise, in the geometry type), `N` | saved | the inputs the forest was built from |
| Forest: the leaf list | saved | the only record of the mesh's history |
| Forest: `generation` | not saved | a staleness counter, meaningless in another process |
| Field set: element type, `nvars`, `G`, `centering`, `parity` | saved | its layout |
| Field set: the **owned** points, in state-vector layout `(N, …, N, nvars, nblocks)` | saved | the authoritative data |
| Ghosts, shared vertex planes, the derived wall plane | rebuilt by `scatter!` and `fill_ghosts!` with the application's hook | derived from the owned points |
| `GhostSchedule`, `InterfaceSchedule`, `Operators`, the parity factors | rebuilt | derived, or the application's inputs |
| Regridding | nothing to save | it keeps no state between calls |
| Scratch field sets (fluxes, primitives) | the application's choice | it passes only the sets it evolves |

The flat state vector is the authoritative data and the working array
is scratch (see [Time integration](#time-integration)), so what is
stored is what an integrator holds, and restoring it is `scatter!`
followed by the ordinary ghost fill. Ghosts are a function of the owned
points, the operators and the boundary hook; all three are back after a
load, and a stored copy of the ghosts could only disagree with the one
the fill produces. The leaf list is the one thing that cannot be
recomputed, since a mesh is the product of every regrid since the
start. Regridding itself keeps nothing between calls — the buffer, the
marks and their completion are recomputed from the flags each time —
so there is no hidden mesh state to lose. (Parthenon, by contrast,
warns that an AMR run restarted without its per-block derefinement
counters may not be bitwise exact.)

**File layout, format version 1** (decided). One HDF5 file. Everything
TreeAMR writes lives under one top-level group, `/TreeAMR.jl`, and the
application gets a top-level group of its own, named after it:

    /TreeAMR.jl/         attrs: format = "TreeAMR checkpoint",
                                format_version = 1,
                                features = ["brick"]  (must understand),
                                application = "<app name>"
      provenance/        scalar datasets: treeamr_version,
                         julia_version, created (UTC, ISO 8601),
                         hostname, nthreads, nranks (M7; 1 if absent),
                         project (Project.toml text, or ""),
                         manifest (Manifest.toml text, or "")
      forest/            attrs: D, N, connectivity = "brick",
                                roots (D), periodic (D),
                                reflecting (2, D), geometry_type
                                [, geometry_limbtype, geometry_nlimbs]
        extents          (limbs?, 2, D), in the geometry type, bitwise
        root             Int32[nleaves]
        level            Int8[nleaves]
        coords           UInt32[D, nleaves]
      fieldsets/<name>/  attrs: eltype [, limbtype, nlimbs], nvars,
                                G (D), centering (D: "cell" | "vertex"),
                                parity ((D, nvars): "even" | "odd" |
                                "none"; absent when the set has none),
                                range = "owned"
        data             (limbs?, N, …, N, nvars, nblocks)
    /<app name>/         attr: format_version (the application's own)
      data               the plain-data tree; beside it, whatever the
                         do-block writes
    /                    otherwise free, for sidecars (M9b)

- **One top-level group per package.** Neither TreeAMR nor the
  application writes at the root or into the other's group, so the two
  layouts, and their format versions, evolve independently. The
  `application` attribute names the application's group; a name that
  would collide with `TreeAMR.jl` is refused.
- **The root stays free for sidecars** (M9b). A visualization view can
  be added to the same file and point at these datasets without copying
  them: an HDF5 hard or soft link where the layout already matches, a
  virtual dataset where it has to be remapped (VTKHDF's per-level
  arrays, say). An external link from a separate file is the fallback.
- **Leaves are stored as columns, in global curve order** — the order
  of `forest.leaves` — so block `b` of every dataset is leaf `b`, and
  any contiguous range of blocks, such as an M7 rank's, is one
  hyperslab of every dataset.
- **No coordinates, and no partition.** Coordinates follow exactly from
  the key, the extents, `N` and the centering (see
  [Precision](#precision)), so a stored copy could only disagree. The
  partition belongs to the process that wrote the file; a reader on
  another thread or rank count chooses its own.
- **The view from C.** HDF5.jl reverses dimensions, so a C or Python
  reader sees a field set as `(nblocks, nvars, N_D, …, N_1)`, slowest
  index first, as Parthenon stores its variables, and the coordinates
  as `(nleaves, D)`.
- **Contiguous when unfiltered; with filters, one chunk per block and
  variable**, `(limbs?, N, …, N, 1, 1)`, so that reading a block
  decompresses that block and no other. The filters are the caller's
  HDF5.jl filter objects, none by default, and TreeAMR depends on no
  filter package. **None is also the recommendation, and `Shuffle()`
  followed by H5Zzstd's `ZstdFilter(1)` the one filter to name when
  size matters** (measured 2026-09-29; the numbers and the reasons are
  under "Throughput and filters" below).
- `range = "owned"` says the data are the owned points only, and leaves
  room for a file that stores more, which a version-1 reader would
  refuse by value.
- **Spellings** (fixed by the implementation). Shapes above are in
  Julia's order, which C sees reversed. Every `Bool` in the file —
  `periodic`, `reflecting`, a plain-data `Bool` — is a `UInt8`, 0 or 1,
  because HDF5.jl would write an HDF5 bitfield, which other readers
  handle poorly. A `(lo, hi)` pair per dimension is a `(2, D)` array,
  which C sees as `(D, 2)`. The other scalars and small arrays are
  `Int64`, the leaf columns excepted. A `Float16`, which HDF5.jl does
  not predefine, is the IEEE half type built as h5py builds it, and a
  `Complex` is the compound `(r, i)`, as HDF5.jl and h5py spell it.

**Element types** (decided).

- **Native types** — `Float16`, `Float32`, `Float64`, the signed and
  unsigned integers, `Bool`, and `Complex` of those — are stored as
  themselves, bit for bit.
- **An `isbits` type made of one native type throughout**, with no
  padding, is stored as **limbs**: a leading dimension of that native
  type, with `eltype` naming the type, `limbtype` the native type and
  `nlimbs` the count. MultiFloats' `Float32x2`, which the test suite
  uses (see [Precision](#precision)), is two `Float32` limbs. The name
  is the type as a module importing nothing but Base prints it,
  `MultiFloats.MultiFloat{Float32, 2}` (amended in the implementation:
  the design said `string(T)`, which qualifies a name or not according
  to what the writer had imported into `Main`, so one type would be
  recorded under two names depending on how the run was started). The
  loader cannot name such a type without its package, so the caller
  passes it — `load_checkpoint(path; types = (Float32x2,))` — and the
  loader matches it by name and checks its size and its limbs against
  the file's. The name is a label to match against,
  not a recipe: without the type the data still read as plain `Float32`
  limbs, which is what a converter or a Python reader needs. The
  geometry type is treated the same way, for `extents`.
- **No conversion on load** in version 1. A field set comes back in the
  type it was saved in, since exactness is the point of a checkpoint;
  loading into a narrower type is an open question.

**Versioning** (decided). This is also the answer to how files from
older versions are treated: what cannot be interpreted is rejected,
with the reason, and defaults are supplied where they are obvious.

- **Compatibility is decided by the file's `format_version` and its
  `features`, never by package versions.** Two releases may write the
  same format, and a release reads every format version it knows,
  whichever release wrote the file.
- **An additive change with an obvious default needs no bump**; the
  reader supplies the default. Had the format existed before M10, its
  files would have had no `reflecting`, and "no reflecting faces" is
  the only reading of that.
- **A change of meaning bumps `format_version`**, and a reader refuses
  a version newer than it knows.
- **A new capability that an old reader must not silently ignore** — a
  multi-block connectivity, say — adds a name to `features`, and a
  reader refuses a file that names a feature it does not know. This is
  Zarr v3's `must_understand` idea, with every listed feature
  must-understand: what may be ignored is simply not listed. Version 1
  lists one, `brick`.
- **Refusals are `ArgumentError`s that say why.** They name the TreeAMR
  version that wrote the file, from `provenance`, and point to
  `checkpoint_environment(path, dir)`. That writes the stored
  `Project.toml` and `Manifest.toml` into `dir`, so that `julia
  --project=dir` recreates the writer's environment, and "use the older
  version" is one command.
- **Why the layout is plain and documented.** Two versions of TreeAMR
  cannot be loaded into one Julia process, so a converter from an old
  format cannot call the old package: it reads the old *file*. That
  works only if the file is plain HDF5 described here, with no Julia
  type a reader has to resolve. It is also why the format is neither
  JLD2 nor `Serialization`, both of which tie a file to the definitions
  of the types that wrote it.
- **The application's group carries its own `format_version`.** TreeAMR
  stores it and returns it and does not interpret it; the application
  checks it. Plain data written as NamedTuples come back as
  NamedTuples, so keyword defaults on the application's side are the
  good-default path for a field it adds later.

**Writing** (decided).

- **Atomically.** The file is written to `path * ".partial"` and moved
  over `path` when it is complete; on any error the partial file is
  removed and the error rethrown. A crash while writing then cannot
  destroy the previous checkpoint, which is the one a restart needs.
- **Durably, by default** (`sync = true`; added 2026-09-29, after the
  first implementation, which left it as an open question). Closing a
  file only hands its data to the operating system's page cache, which
  survives the process but not a power loss or a kernel crash; and the
  rename is a separate update of the directory, which can reach the
  disk before the data it points to, leaving `path` naming a truncated
  file with the previous checkpoint already gone. So the partial file
  is flushed to stable storage before the rename, and its directory
  after it, so that the rename is durable too. On Linux the flush is
  `fsync`; on macOS it is `fcntl(F_FULLFSYNC)`, because `fsync` there
  hands the data to the drive without waiting for the drive's own
  cache (1 ms against 125 ms for 540 MB), falling back to `fsync` on a
  file system without it; a directory that refuses an `fsync`
  (`EINVAL`) is accepted. HDF5 cannot do this itself: its flush, too,
  ends in the page cache. `sync = false` is for files that need not
  outlive the machine, such as a test's. What the flush costs is under
  "Throughput and filters" below.
- **Two forms per field set.** `name => (fs, u)` writes the state
  vector `u`, and is the recommended form: after `solve` the working
  array holds whatever the last right-hand side scattered, which is a
  stage, not the solution. `name => fs` gathers the owned points from
  `fs.work` first, which is right for a set whose working array is
  current by construction — an auxiliary set filled by
  `fill_by_coordinates!`, say.
- **Through the host.** Device data are copied to the host (`tohost`,
  in `src/device.jl`) before they are written.
- **One forest.** Every field set must be over the forest being saved
  (`===`), as for `regrid!`.
- **Filters are the caller's**, as above.

**Loading** (decided).

1. The forest is built through the validated leaves path,
   `Forest{R}(roots; N, periodic, reflecting, extents, leaves)`, which
   refuses a list that does not tile the brick exactly or is not
   balanced, so that a damaged or hand-edited file cannot yield a mesh
   the schedule would silently get wrong.
2. `FieldSet{T}(forest, nvars; G, centering, parity, backend)`.
3. `u = statevector(fs)`: allocated on the backend and first-touched by
   the block owners, so NUMA placement is right although the HDF5 read
   that fills it is serial — placement is decided at the first touch,
   not by the later write. (Whether automatic NUMA balancing then moves
   pages that one thread touches is the unproven candidate in the
   [Open questions](#open-questions) item on the integrator's own
   passes; a load touches them once.)
   The read goes into `u`, through a host buffer on a device.
4. `scatter!(fs, u)`.

Ghosts are left to the application's `fill_ghosts!(fs,
GhostSchedule(fs, ops); boundary)`, with its own operators and hook.
The file does not hold those because they are the application's
inputs, the hook often a closure. Nothing stored depends on the thread
count or the backend, so a checkpoint loads on any of them, and the
load itself is exact everywhere. What follows is bit-identical at any
thread count, by the invariant under [Parallelism](#parallelism);
across backends it is as close as a host run and a device run ever are
— bit for bit in the Burgers study on Metal, not promised in general.

**Throughput and filters** (measured 2026-09-29, `bench/checkpoint.jl`).
The machine is the development laptop — an Apple M3 Pro, 6 performance
and 6 efficiency cores, 36 GB, its internal SSD under APFS — with Julia
1.13.1 and HDF5.jl 0.17.4 over libhdf5 2.2.2. The mesh is 3D, 3296
blocks of `16³`, refined twice around the sphere `r = 1/2` (56, 1000
and 2240 blocks on levels 0, 1 and 2). It carries two `Float64` states,
each with its feature on the refined shell:

- `pulse`, a smooth outgoing spherical Gaussian `(u, ∂ₜu)`,
  vertex-centered, 216 MB — TreeWave's and TreeGeneralizedHarmonic's
  kind of data;
- `blast`, a blast wave in the five conserved variables,
  cell-centered, 540 MB: a smooth interior behind a discontinuity and a
  uniform atmosphere outside it, the same bits everywhere — TreeHydro's
  kind.

Throughput is in GB/s of state data (owned points, 1 GB = 10⁹ bytes),
the best of five calls after one that compiles. `save` is
`save_checkpoint(…; sync = false)`, which ends in the page cache;
`sync` is the default, `sync = true`, which adds the flushes of the
file and of its directory to stable storage (on macOS
`fcntl(F_FULLFSYNC)`: a plain `fsync` there returned in 1 ms for 540
MB, against 125 ms for the full flush, so it proves nothing). `load` is
all of `load_checkpoint`, at one thread and at six; the saves are the
one-thread run's. One chunk is one block and variable, 32 KB. The
ratio is the state's size over the file's.

| data | filter | ratio | save | sync | load, 1 thread | load, 6 threads |
|---|---|---|---|---|---|---|
| pulse | none | 1.00 | 5.3 | 4.0 | 1.11 | 3.65 |
| pulse | `Shuffle` + `Deflate(1)` | 1.34 | 0.10 | 0.10 | 0.25 | 0.30 |
| pulse | `Shuffle` + zstd 1 | 1.32 | 0.62 | 0.61 | 0.73 | 1.37 |
| pulse | `Shuffle` + zstd 3 | 1.34 | 0.45 | 0.44 | 0.72 | 1.34 |
| pulse | `Shuffle` + LZ4 | 1.30 | 0.82 | 0.80 | 0.74 | 1.55 |
| pulse | bitshuffle + LZ4 | 1.19 | 0.58 | 0.59 | 0.52 | 0.77 |
| blast | none | 1.00 | 6.4 | 3.9 | 1.20 | 3.98 |
| blast | `Shuffle` + `Deflate(1)` | 6.00 | 0.32 | 0.32 | 0.49 | 0.68 |
| blast | `Shuffle` + zstd 1 | 6.14 | 1.06 | 1.02 | 0.76 | 1.36 |
| blast | `Shuffle` + zstd 3 | 6.23 | 0.90 | 0.88 | 0.76 | 1.37 |
| blast | `Shuffle` + LZ4 | 5.68 | 1.20 | 1.18 | 0.86 | 1.75 |
| blast | bitshuffle + LZ4 | 4.85 | 0.64 | 0.63 | 0.52 | 0.76 |

zstd is H5Zzstd's `ZstdFilter(level)`, LZ4 H5Zlz4's `Lz4Filter()`, and
bitshuffle H5Zbitshuffle's `BitshuffleFilter(compressor = :lz4)`. Blosc
was not measured. **Bitshuffle with zstd cannot be measured with the
registered H5Zbitshuffle** (0.1.3): it calls `bshuf_compress_zstd`
without its last argument, `comp_lvl`, so zstd compresses at whatever
level the argument register happens to hold. The same data and the
same `comp_level = 1` came out 175.3 MB at 0.02 GB/s in one process and
181.6 MB at 0.35 GB/s in another — though bit for bit readable in both.
HDF5.jl's unreleased 0.18, where the filter moves into an extension,
passes the level. This table is the second of two measurements the
same day; across the six runs, the unfiltered save varied between 4.6
and 7.3 GB/s and its flush to stable storage between 2.9 and 5.4, while
a filtered save agreed within 15 %, with no trend in the thread count.
The laptop was in use, with a video call and an endpoint-security
agent that scans written files taking about a core, so these are its
numbers, not a quiet node's. The flush costs an unfiltered save of
540 MB about 50 ms here, and a filtered one nothing measurable, which
is why `sync = true` is the default.

- **No filter is the recommendation.** Unfiltered, a save reaches the
  page cache at 5–7 GB/s and stable storage at 3–5, and a load runs at
  2.8–4.0 GB/s on six threads. A smooth field compresses 1.3-fold at
  best, because the low mantissa bits of smooth data are noise to a
  lossless coder, and every filter buys that with 9 to 70 times the
  save time. A checkpoint is written to be read once, if at all.
- **When size matters, `Shuffle()` then `ZstdFilter(1)`.** Data that
  are mostly a uniform atmosphere are where a filter pays: 6.1-fold at
  1.0 GB/s saved, within 1.5 % of level 3's ratio and 1.2 times its
  speed. `Shuffle` with LZ4 is 12 % faster at 8 % less ratio, the
  choice if time matters more. The built-in `Deflate(1)`, the only
  filter every HDF5 has, gets less ratio at a third of the speed on
  such data, and on smooth data it is six times slower than zstd, at
  0.10 GB/s. Bitshuffle with LZ4 loses to byte shuffle with LZ4 on both
  counts at this chunk size.
- **A filtered file needs its filter to be read.** In Julia the
  filter's package is loaded before `load_checkpoint` (`using
  H5Zzstd`), and in C or Python the HDF5 plugin is installed (h5py's
  through hdf5plugin). Without it the read fails with HDF5's own error,
  a plugin it cannot find, and not with a refusal that says why (an
  open question).
- **Compression is serial.** HDF5 runs the filter pipeline chunk by
  chunk on the calling thread, so a filtered save gains nothing from
  threads, and a filtered load gains only the part of it that is not
  HDF5's. Under M7 each rank compresses its own chunks, which is where
  a filter parallelizes.
- **Where an unfiltered load goes.** For the 540 MB state at one
  thread, 470 ms: the field set's allocation and zero fill 157, the
  state vector's first touch 84, the HDF5 read 87, `scatter!` 134, the
  validated forest 20. At six threads the threaded parts fall to 27,
  13, 30 and 6 ms, and the read, 82 ms, is half of the 156.
- **The loads read the page cache.** A load right after a save reads
  the file the save just wrote, so the load numbers are decompression
  and memory bandwidth, not the SSD's read rate. A cold-cache load
  needs the cache dropped first (`purge` on macOS, `drop_caches` on
  Linux, both root-only) and was not measured. The numbers that matter
  for a production run are the cluster file system's, which
  `TREEAMR_BENCH_DIR` points the benchmark at, and belong to M7's
  parallel-I/O measurement.

**Parallel I/O and M7** (the facts checked 2026-09-29 against the HDF
Group's "Collective Calling Requirements in Parallel HDF5
Applications", "A Brief Introduction to Parallel HDF5" and "HDF5
Parallel Compression", and HDF5.jl's MPI page; the design is not
decided).

- In parallel HDF5 only raw data transfers — `H5Dwrite`, `H5Dread` —
  may run independently per rank, and any number of them. Every call
  that creates or changes the file's structure or metadata is
  collective: creating, opening, flushing and closing the file;
  creating groups, datasets and attributes; writing an attribute;
  extending a dataset. All ranks make the same call with the same
  arguments, so every rank pays for every object, and that cost does
  not fall as ranks are added. A layout whose object count grows with
  the blocks or the ranks — one dataset per variable, level and
  component, as in CarpetIOHDF5 — makes metadata the part that does not
  scale.
- Writing a filtered (compressed) dataset in parallel needs HDF5 ≥
  1.10.2 and collective writes. A chunk that several ranks write is
  given one owner, and the others send it their parts.
- This layout keeps the object count fixed: about a dozen, two more per
  field set, plus the application's plain data, and none of it depends
  on the number of blocks or ranks. Each rank's blocks are one
  contiguous hyperslab of every dataset, since an M7 rank holds a
  contiguous range of the curve and the curve is the block axis. And a
  chunk is one block, so under compression every chunk has one writer
  and none moves between ranks.
- The alternative many codes use is one file per I/O process plus a
  wrapper file (CarpetIOHDF5's per-process output, AthenaK's per-rank
  restart, a Conduit Blueprint root file). It is easy to write and
  needs no collective metadata, but it is awkward to read back on a
  different rank count, where each reader must find the files that
  hold its range; SAMRAI's restart requires the same process count
  unless a separate redistribution tool is run.
- **Decided:** M9a is serial. Which parallel design M7 uses is
  benchmarked first, on Symmetry and other HPC systems; this layout is
  the candidate, not a commitment. If the shared file does not hold up,
  M7 bumps `format_version`, and that is accepted. *(Amended
  2026-10-01, decided with Erik: parallel HDF5 into this layout is part
  of M7, with `format_version` unchanged, and the benchmark became M7's
  step 6 instead of a precondition — the shared file is built and then
  measured, and the per-process files remain the fallback if it does
  not hold up. The design, and the check that the stock HDF5_jll is
  already a parallel build, are under
  [Distributed meshes](#distributed-meshes).)* *(Step 6 of M7 built it:
  the version-1 layout written and read by every rank of a distributed
  forest at once, a file loading on any rank count; what it settled is
  under "Parallel checkpoints" there. The throughput on Symmetry's
  parallel file system is still to be measured.)*

**Multi-block** (checked). Keys are relative to their root,
`connectivity = "brick"` is a tagged record rather than an assumption,
and no coordinates are stored. A conforming multi-block forest, the
likely route to spherical domains, is then a new connectivity kind,
with whatever describes its roots stored beside the leaves, plus a
feature name. Nothing in the format obstructs it. Parthenon's layout
would: it stores global tree locations in one virtual tree over the
root grid, which a multi-block forest cannot express.

**Formats considered** (a survey, 2026-09-29). No existing standard
fits a leaf-only octree checkpoint. VTKHDF has no non-overlapping AMR
type: its `OverlappingAMR` must be sorted by level, and VTK's
non-overlapping AMR exists only in the XML `.vthb` format. Conduit
Blueprint associates fields with vertices or elements only, so it has
no face or edge centering. openPMD's mesh-refinement extension is still
an open pull request. AMReX, Chombo and Carpet store overlapping
hierarchies, with coarse data under fine. For checkpoints the norm is a
code's own versioned HDF5 schema — Parthenon, FLASH, Athena++,
CarpetIOHDF5 — or its own binary format (AMReX, AthenaK, p4est).
Parthenon's `.rhdf` is the closest model, and the one followed here:
one dataset per variable over all blocks in Z-order, a block table,
collective hyperslab writes with one block per chunk, restart on any
rank count, and one integer format version. What is done differently
is keys relative to their root, no stored coordinates, and a features
list beside the version. JLD2 is set aside because it records Julia
type names, which ties a file to the definitions that wrote it, and
because it has no MPI.

**The API in brief** (as implemented). HDF5 is a **package extension**,
`TreeAMRHDF5Ext` (`ext/TreeAMRHDF5Ext.jl`) over the weak dependency
HDF5 (`[compat]` 0.17), because the only hard dependency today is
KernelAbstractions, and an application that never checkpoints should
not load HDF5 and its binaries. The core, `src/checkpoint.jl`, holds
the docstrings, stubs without methods, and an error hint that says to
load HDF5 while the extension is not loaded. `fieldsets` and
`application` have no defaults, and the refusals say why.

    # core: a forest from a validated leaf list
    Forest{T}(roots; N, periodic, reflecting, extents, leaves)

    # the extension, loaded by `using HDF5`
    save_checkpoint(path, forest;                    # returns path
                    fieldsets   = ("U" => (U, u), "aux" => aux),  # or ()
                    application = "TreeHydro" => 1,  # name => its version
                    data = (; t, chunk, recipe), filters = (),
                    sync = true)                     # flush to stable storage
    save_checkpoint(path, forest; …) do app::HDF5.Group
        # further datasets in the application's group, beside `data`
    end
    ck = load_checkpoint(path; backend = CPU(), types = (),
                         fieldsets = nothing,        # or names, a subset
                         comm = nothing)             # or an MPI.Comm (M7)
        # ck.forest, ck.fieldsets["U"].fieldset, ck.fieldsets["U"].state,
        # ck.application, ck.data, ck.provenance
    load_checkpoint(path; …) do app … end        # result in ck.result
    write_plain(parent, name, value); read_plain(parent, name)
    checkpoint_environment(path, dir; force = false)  # returns dir

- **The `leaves` keyword validates**, in the tone of the other
  refusals: every root index is below `prod(roots)`; the keys are
  strictly sorted, which also refuses duplicates; the leaves tile the
  brick exactly, checked by one walk along the curve (each leaf starts
  where the previous leaf's subtree ends, and every root is covered);
  and the list is balanced (`isbalanced`). An unbalanced list is
  refused rather than rebalanced, because it cannot have come from a
  forest, and `block_sources!` would silently build wrong prolongations
  from it. `MortonKey`'s own checks cover each key.
- **Plain data are a closed, documented set of types**: `Bool`, `Int8`
  to `Int64` and `UInt8` to `UInt64`, `Float16`, `Float32`, `Float64`,
  `Complex` of those, `Rational`s of the native integers, `String`,
  `Symbol`, `VersionNumber` and `Nothing`; tuples, NamedTuples, and
  `AbstractDict`s with `String` or `Symbol` keys; and arrays of native
  numbers or strings. Every item is a dataset or group with a `type`
  attribute from a closed vocabulary — `number`, `rational`, `string`,
  `symbol`, `version`, `nothing`, `array`, `tuple`, `namedtuple`,
  `dict` — with `eltype` naming a number's or an array's element type
  (`"String"` for strings) and `keytype` (`"String"` or `"Symbol"`) on
  a dict. A Rational is the dataset `[numerator, denominator]` in its
  integer type; a nonempty tuple of one native number type is one
  array, and any other tuple a group of the items `"1"`, `"2"`, …; and
  groups keep their order (`track_order`), so a NamedTuple keeps its
  field order. A NamedTuple comes back as a NamedTuple and a Dict as a
  `Dict{String,Any}` or `Dict{Symbol,Any}`, an array as an `Array`.
  Anything else is an `ArgumentError` that says to convert it to a
  NamedTuple: structs are the application's to convert (`to_plain`,
  `from_plain` on its side), so no type name reaches the file.
  Rationals store exactly, and TreeHydro already states its parameters
  as Rationals, so a case recipe round-trips exactly.

**Where a checkpoint belongs in a chunked driver** (decided). The
downstream drivers integrate in chunks: `solve` over a fixed number of
steps, then flag, regrid, rebuild the schedules and restart the
integrator (see [Regridding](#regridding)). A checkpoint goes at a
chunk boundary, *after* the regrid. There the integrator holds nothing
but `(t, u)`, for the fixed-step explicit methods every downstream
application uses, so restoring `t` and `u` restores the integrator —
a step size computed from the forest or the state comes out the same —
and a restarted run begins the next chunk with exactly what the
uninterrupted one began it with. Mid-chunk, the integrator's own state
(its stages and cached derivatives) would have to be saved too; before
the regrid, a restart would have to replay the regrid and whatever the
application does after it, such as TreeHydro's atmosphere reset. The
application's run state — the time or chunk index, histories, trackers
— goes in its plain data.

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
  M7 therefore gains no bandwidth argument from this, and loses none
  *(confirmed in M7's step 7: on one Symmetry node one pinned 64-thread
  process runs the RHS 1.5–1.6 times as fast as 8 ranks of 8 threads
  over the same mesh, the ranks paying the packs and unpacks of their
  halos)*:
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
    block, the other one value for the mesh. *(Amended 2026-10-01: the
    site stays, but what it does across ranks is an `allgather` of one
    partial per rank, folded in rank order, with no MPI operation at
    all — custom operations fail on ARM, and a builtin one exists only
    for MPI's native types. See "Reductions" under
    [Distributed meshes](#distributed-meshes).)*

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
  for GPU+MPI. *(Amended 2026-10-01, when M7 was specified under
  [Distributed meshes](#distributed-meshes) below: a transfer is
  evaluated on the owner of its **source**, not of the finer data —
  the sender computes — so a restriction runs on the fine side, as
  this said, but a prolongation runs on the coarse side.)*

### Distributed meshes

*(Designed 2026-10-01 as M7, before implementation. Four choices were
decided with Erik that day and are marked so; the rest follows from
them and from the sections cited, and is open to amendment as the
implementation measures it. The steps are in the M7 entry under
[Milestones](#milestones). Implemented in them by 2026-10-02, the
amendments marked where they were made. Measured on Symmetry the same
day: steps 7 and 8 in full, step 6 on one node; the findings are in
the steps, and amend the bullets below where marked.)* M7 runs one forest over several
processes. It came after M8 and M10 on purpose: every centering, the
interface restriction and the mirrored transfers at reflecting faces
are now entries of one schedule, so distributing the schedule
distributes all of them at once. The claim extends the one
[Parallelism](#parallelism) makes for threads. The leaf array, the
schedule, the state vector gathered in curve order, and every max, min
and integer reduction are bit-identical to a serial run at any rank
count. A floating-point sum is reproducible to roundoff across rank
counts, and at one rank it is exactly the serial value.

**What is replicated and what is distributed** (decided 2026-10-01 with
Erik). Every rank holds the whole forest, `forest.leaves` included, and
only field data are distributed.

- *Why it is affordable.* A leaf is a key of about 20 bytes in 3D (see
  "Key encoding" under [Tree structure](#tree-structure)), so 10⁵
  leaves are 2 MB per rank. Everything that reads the whole leaf
  array — the neighbor search, the buffered flags, mark completion,
  balance — runs at regrid frequency, `O(nleaves)` per rank. AMReX
  replicates its `BoxArray` and `DistributionMapping` the same way, and
  Parthenon, after Athena++, its block tree.
- *What the alternative costs.* p4est distributes the forest, and then
  balance and neighbor search are themselves parallel algorithms with
  a ghost layer of the tree. That is the hard part of p4est, and it
  buys memory a mesh of up to some 10⁶ leaves does not need.
- *What it buys.* Every tree query stays a local call: the schedule
  build, `locate_point` and the regrid's classification of new leaves
  send no messages. The same arithmetic on the same leaves gives every
  rank the same answer, and the message layouts below rely on that.

What replication costs at thousands of ranks — the replicated regrid
bookkeeping above all — is for the weak-scaling smoke test (step 7) to
measure, not for this section to assume. *(Measured in step 7, in one
process at up to 180224 leaves: the schedule build stays flat at a fixed
number of blocks per rank, and of the replicated passes two grow to
dominate a rank's regrid — the buffer's neighbour search when many
blocks report boxes, and the classification of every new leaf — both of
which can be made `O(local)` without changing a result; see step 7 under
[Milestones](#milestones). Both were, the same day: each rank now
searches the buffer from its own sources and classifies only the new
leaves it needs, and a rank's regrid stayed at its 1408-leaf cost up to
180224 leaves.)* *(On Symmetry, step 7: at up to 32 ranks on four nodes
the schedule build and the refining regrid are flat from 2 ranks on, and
`bench/replicated.jl` on a rank's domain reproduces the flat regrid to
180224 leaves; the replicated completion after the buffer, 20 ms there
when every block is a source, is what still grows.)*

**The communicator layer** (MPI a weak dependency, decided 2026-10-01
with Erik). A new file, `src/communicator.jl`, sits after `device.jl`
in the layer order.

- It defines an abstract `Communicator` and the default
  `SerialCommunicator`, rank 0 of 1, which sends nothing.
- The package talks to it through a few internal verbs: `commrank`,
  `commsize`, `allgather` of one `isbits` value, `allgatherv` of a
  vector, `alltoallv`, and nonblocking `isend` / `irecv` with
  `waitall` over flat buffers.
- Every verb has a serial method, so `src/` never branches on whether
  MPI is there. A serial run takes the distributed code path with every
  message empty, and the existing suite is its test.
- The MPI methods live in a package extension, `TreeAMRMPIExt`,
  triggered by MPI.jl, as HDF5 is for checkpoints: an application that
  never runs distributed does not load MPI. It adds `MPICommunicator`,
  which holds `MPI.Comm_dup(comm)` of the application's communicator,
  so that TreeAMR's messages can never match the application's.
- An application writes `Forest(…; comm = MPI.COMM_WORLD)`, and
  `communicator(::MPI.Comm)` converts it.

Three details follow from this:

- *One duplicate per application communicator.* `Comm_dup` is
  collective, and so is `Comm_free`, which therefore cannot run from a
  finalizer, since a finalizer runs at a different moment on each rank.
  `communicator` caches the duplicate per communicator it is given, so
  a test that builds a thousand forests over `COMM_WORLD` holds one
  duplicate and not a thousand. MPICH_jll 5.0.2 gives a process 2046
  duplicates and then fails (measured 2026-10-01, on Julia 1.11 and
  1.13). *(Step 3: MPI.jl's own `Comm_dup` attaches a finalizer that
  frees the duplicate, so the extension calls `MPI_Comm_dup` through
  `MPI.API` and attaches none. The cache is keyed by the handle; since
  MPI may reuse a handle once the application frees its communicator, a
  hit is used only while `MPI_Comm_compare` still finds the duplicate
  `CONGRUENT` with the communicator passed — a local call whose answer
  is the same on every rank of the group.)*
- *MPI is called from the calling task only*, never inside a threaded
  loop or a kernel. A Julia task may still migrate between OS threads
  between two calls, so the extension requires
  `MPI_THREAD_SERIALIZED` or better — `MPI.Init()`'s default — and
  refuses a communicator initialized with less, saying why.
- *Why the field is abstract.* `Forest` gains `comm::Communicator`,
  abstract-typed, so `Forest{D,T}` keeps its two parameters and every
  `FieldSet{…}` and `GhostSchedule{…}` signature downstream is
  unchanged. The price is a dynamic dispatch per verb: a few per ghost
  fill, and none in a kernel.

**Every forest mutation is collective.** `refine!`, `coarsen!`,
`balance!`, `regrid!` and the constructors are called on every rank
with the same arguments. Every host pass in `src/` is a deterministic
function of its inputs — the thread-count invariant under
[Parallelism](#parallelism) — so the calls produce the same forest
everywhere without a message. Checking that at every call would cost a
collective each, so the operations that would go wrong silently on a
diverged forest check it instead: `GhostSchedule`, `InterfaceSchedule`
and `regrid!` `allgather` a digest of the forest — its generation,
`nleaves` and `hash(forest.leaves)`, which for `MortonKey` is defined
from the fields. A forest whose digest differs between ranks is
refused, naming the ranks and saying that every forest mutation is
collective. At regrid frequency that is one `O(nleaves)` pass and one
collective.

*(Amended in step 3, where the check was implemented.)*

- *Not `hash(forest.leaves)`.* Base's hash of a long vector samples its
  elements rather than reading them all: on Julia 1.11 and 1.13 alike,
  zeroing element 50001 of `collect(1:100_000)` leaves the hash
  unchanged (checked). So the digest folds `hash(key, h)` over every
  leaf.
- *It carries the brick and the layout too.* The brick (roots, `N`,
  the periodic and reflecting faces, the extents) is hashed through its
  printed form. Beside it goes a hash of the layout the schedule is
  built for — the ghost widths, the centering, the operator family and
  orders, the element type and the backend's name — since a rank that
  built its stencils for other operators would deliver wrong ghosts as
  silently as a diverged forest. Every hash is of integers and strings,
  never of a `Symbol` or an object identity, which differ between
  processes. A layout that differs is refused with a reason of its own.
- *A refusal on some ranks is a refusal on all.* A schedule's argument
  checks run before the gather; a rank whose checks refuse sets a flag
  in its digest instead of throwing at once, and after the gather it
  throws its own error while every other rank throws one naming the
  ranks that refused. Without that, one rank's `ArgumentError` would
  leave the others waiting in the gather, or in their first exchange.
  Only `ArgumentError`s are agreed this way; anything else is a bug and
  is rethrown at once. *(Amended in step 4: `regrid!`'s checks run
  through the same gather, and the `DimensionMismatch` it raises
  serially for a flag vector of the wrong length is agreed too, so that
  error keeps its type on the rank that raised it.)*
- *Serially nothing is gathered*: a forest over one rank skips the
  digest and its `O(nleaves)` pass, so a serial build is unchanged.
- *The verdict is the same on every rank* because it is computed from
  the same gathered digests: the message names rank 0's forest and the
  ranks that differ from it, whichever rank prints it.

**The partition** (decided). Rank `r` owns the contiguous leaf range
`blockrange(forest)`, an equal-count split of `1:nleaves(forest)` over
`commsize`. The arithmetic is `threadchunks`'s — `divrem`, the
remainder to the first ranks — and both are written over one shared
helper. The rank split and the thread split under it are then the same
contiguous Morton ranges one level apart, as "Where the 64-thread
efficiency goes" anticipated. Equal counts balance the load because
every block costs the same under a global `dt` (the MPI bullet above).
The one imbalance is ghost work concentrating at level boundaries,
noted under "What one process loses"; a cost-weighted split would
change only the helper, at both levels.

- **Block index `b` becomes local, everywhere.** `nblocks(fs)` is the
  rank's count, the last axis of `fs.work`, and `blockkey(fs, b)` is
  `forest.leaves[first(blockrange(forest)) + b - 1]`. The state
  vector, `statelength`, `block_spacings`, `block_origins`, the flags
  of `flag_blocks` and the values of `block_mapreduce` and
  `firing_boxes` are all per local block.
- **Tree queries stay global.** `nleaves(forest)` is the global count,
  and `find_leaf`, `locate_point` and `neighbor_keys` answer in global
  leaf indices, since they are about the forest and not about a field
  set.
- **No API break.** Serially local and global coincide, so nothing
  downstream changes and no 0.2 is needed. What becomes wrong only
  under MPI is code that sizes a per-block array by `nleaves`, or
  indexes `forest.leaves` by a block index. The docstrings say "local
  block", and step 9 audits the three downstreams for both.
- **Ranks without blocks are allowed.** Such a rank takes part in every
  collective, enters no exchange stage, and contributes nothing to a
  reduction (below). Every launch guards an empty `ndrange`. Empty
  ranks occur whenever ranks outnumber leaves, which a coarse initial
  mesh on many ranks does for a while, and the workload tests one.
- **What an application must make global itself.** `block_mapreduce`
  stays local by design, so a value an application combines from it —
  TreeHydro's CFL signal speed, TreeWave's per-variable refinement
  scale — becomes a per-rank value under MPI. A `dt` that differs
  between ranks desynchronizes the run, and a rank-dependent scale
  makes the mesh depend on the rank count. Such a value goes through
  `mesh_mapreduce`. So does an adaptive integrator's error norm: it has
  to be a global one (`volume_weighted_norm`, the open `internalnorm`
  question), or each rank chooses its own step.

**The exchange: the sender computes** (decided 2026-10-01 with Erik). A
transfer whose source and target blocks live on different ranks is
evaluated on the source's rank, into a packed buffer, which is sent.
The target's rank only copies it into place. Three reasons:

- *The source rank has everything the transfer reads*, including the
  source's own ghost layers, which a prolongation reads (in phase 2,
  and in the regrid from a parent). Evaluated on the receiver, the
  transfer would need those ghosts shipped too, as a second and wider
  exchange.
- *The message is the transfer's target box.* A restriction's source
  footprint is `2^D` times its target and more, so sending the result
  is much smaller. For a prolongation the two are about the same size:
  in 3D at `N = 16`, `G = 2` and order 4, a fine face slab is 512
  points and the coarse footprint it is computed from about 480.
- *The receiver needs nothing but a layout*: no stencils, no second
  schedule, no knowledge of the source's ghosts. Unpacking is a copy.

**Building the distributed schedule.** Each rank runs the existing
`block_sources!` for its own targets, as today, and also for the
*candidate remote targets*: the remote leaves that touch one of its
own, i.e. the union of `neighbor_keys` over all directions `δ` around
its leaves.

- *The candidates are complete.* Adjacency is mutually discoverable
  (`neighbor_keys`' docstring): a target that takes data from a local
  block is found from that block across some direction, if not
  necessarily the opposite one.
- *Mirrored transfers need nothing extra.* A mirrored transfer's source
  is found across `δ′` from the target, so it is a neighbor or the
  target itself. A self-transfer — a block's reflection into its own
  ghosts, the derived wall plane — is always local.
- *Three classes.* A *local* transfer has target and source on this
  rank; it is today's group, run as today. A *recv* transfer has a
  local target and a remote source, and becomes an unpack. A *send*
  transfer has a remote target and a local source, and becomes a pack.
- *What is checked.* The union over ranks of the local, recv and send
  transfers is the serial schedule's transfer set exactly; that is
  step 2's acceptance.
- *Cost.* The build is `O(local + halo)` per rank, the halo being the
  remote neighbors. The digest check above is the one `O(nleaves)`
  pass.

**Stages.** The serial fill is ordered by its phases because a later
phase reads what an earlier one wrote: copies and restrictions, then
the hook, then prolongations by target level, coarsest first. Each such
ordering point becomes a *stage*, and a stage completes on a rank
before the next one starts there:

- phase 1 is one stage. The boundary hook then runs locally, on each
  rank's own blocks, after that stage's unpacks — exactly where it runs
  today;
- phase 2 is one stage per target level, ascending;
- `restrict_interfaces!` is one stage per face dimension, ascending.
  Its serial phases exist because a point on the line where two
  coarse-fine faces meet is written in two of them (`InterfaceSchedule`'s
  docstring), and with three levels a later dimension can also read,
  on a fine block's plane, a point an earlier one wrote there, where
  that fine block is itself the coarse side of a face. Either makes
  the phases ordering points like the ghost fill's;
- the regrid transfer is one stage.

A stage runs in five steps:

1. post an `irecv` per peer into the stage's receive buffer;
2. pack every send transfer, then synchronize the backend;
3. `isend` per peer;
4. run the local groups, which overlap the messages in flight;
5. `waitall` on the receives, then unpack.

The sends are waited on at the end of the call, before any buffer is
reused. *(Step 2: `run_stage!` in `ghosts.jl` runs these five steps
through the communicator verbs. A stage without messages is the serial
phase, the same launch and the same barrier.)* Within a stage, nothing a pack or a local group reads is
written by an unpack:

- phase 1 reads interiors and writes ghosts;
- a level-`ℓ` stage reads level-`ℓ − 1` sources and their ghosts, and
  writes level-`ℓ` ghosts;
- an interface stage reads fine-side planes and writes coarse-side
  ones. A block that is both, at different faces, plays the two roles
  on different planes.

So the order inside a stage is free, as the order inside a serial
phase is, and only the stage boundaries carry the serial ordering.

**Deadlock freedom and message matching.** Messages are nonblocking
and matched by peer and tag, and the tag names the stage: phase 1, the
target level, the face dimension, or the regrid. A rank enters only the
stages in which it has a message or a local group. A skipped stage
cannot be confused with another, since its tag is its own.

- *Progress, by induction on the stage.* Suppose every rank has
  completed the stages before `k`. A pack reads only what those stages
  completed locally, so every rank can pack and post its stage-`k`
  sends without waiting for anyone. Then every stage-`k` receive has
  its send posted, and the `waitall` returns.
- *Across calls*, MPI's non-overtaking rule — two messages from one
  sender with one tag arrive in the order they were sent — matches each
  fill's message to the same fill's receive, provided every rank makes
  the collective calls in the same order. That is the contract, and it
  implies one more rule: two exchanges over one forest must not run
  concurrently from two tasks.

**Buffer layout.** Each stage has one send buffer and one receive
buffer, each divided into one contiguous segment per peer.

- Within a segment the transfers are ordered by `GroupKey` (under a
  total order over its fields), then by global target leaf, then by
  global source leaf. Each transfer occupies its target box times
  `nvars`, in the kernel's index order.
- Both ends derive this order from the replicated forest, the sender
  from its send list and the receiver from its recv list, which hold
  the same transfers. So no layout descriptor is ever exchanged; only
  the stage's data move.
- Phase 1's groups come out of a `Dict` today, in an order that does
  not matter there. The remote groups are sorted explicitly, since for
  them it does.
- Step 2 checks that rank `r`'s send layout to `s` is `s`'s receive
  layout from `r`.

**Pack and unpack are transfers.** Both go through `transfer_kernel!`,
which keeps "one kernel for every transfer" true.

- *The packed buffer is a kernel argument.* It is passed in place of
  the working array as a small `NamedTuple` — the flat buffer and each
  transfer's offset into it — which KernelAbstractions adapts to a
  device without a new dependency. *(Amended in step 2: the kernel's
  tuple has a third field, `dims`, the target box times `nvars`, since
  a slot's linear index needs the box's shape. The offsets are stored
  in points, not elements, and multiplied by `nvars` when the buffer is
  addressed, because a schedule belongs to a layout and serves field
  sets of any variable count. A driver passes `(buf, offsets)`, and
  `run_group!` adds the group's `dims`. The "block" index the kernel
  hands the accessor is the slot.)*
- *Three accessor methods, not two.* The kernel reaches `dest` and
  `src` through a load, a store and the element type that
  `stencil_sum` starts its accumulator from, `zero(eltype(src))` today.
  `eltype` of a `NamedTuple` is not the buffer's, so it needs a method
  of its own. Each accessor has a method for an array, today's indexing
  unchanged, and one for a packed buffer.
- *Groups.* A pack group is a send group whose target is a buffer
  slot. An unpack group is a width-1, weight-1 transfer from a buffer
  slot into the recv target.
- *Ownership.* Pack groups are sorted by source block and run by the
  owner of the source block. Unpack groups run by the owner of the
  target, as every group does today. On the CPU, `run_phase!`'s
  bisection is then over whichever block list says who owns the
  transfer.
- **The parity factor is applied when unpacking** (amended while
  specifying; the first draft applied it when packing). An unpack goes
  through the kernel's sum, `0 + 1·x`. That is the identity on every
  value a stencil sum produces, because the sum starts at `+0` and so
  never yields `−0`. It is not the identity on `−0` itself, which it
  turns into `+0`. A mirrored transfer is the one transfer that can
  produce `−0`: an odd variable that is zero at the wall, such as the
  normal momentum of a fluid at rest, gives `+0 · (−1)`. A pack that
  applied the factor would therefore deliver `+0` where the serial
  fill writes `−0`, and the byte-for-byte digests would see it. So a
  pack computes the unscaled sum, and the unpack group carries the
  mirrored group's `factorcol`. The value written is then
  `scaled(0 + 1·acc) == scaled(acc)`, bit for bit what the serial
  kernel writes. Step 2's round trip checks this bitwise, `Float32x2`
  included, for which it also needs `0 + 1·x` to be the identity on
  normalized limbs. *(Measured in step 2. With the factor moved to the
  pack, the `Float64` round trip fails in every 2D case tried, as
  argued. `Float32x2` cannot tell the two apart: MultiFloats' product
  returns `+0` for `0 · (−1)`, so its serial fill has no `−0` to lose.
  Nor is `0 + 1·x` the identity on every limb pair — it maps `(−0, −0)`
  to `(+0, +0)` — but it held on every value a pack produced in the
  round trip, which is a check over the tested cases and not a proof.
  In 1D every mirrored transfer is the block's own reflection,
  so the remote `−0` case exists from 2D on.)*
- **Why the ghosts are bit-identical.** Every ghost is written once, by
  the same kernel, from the same source values and the same stencil,
  summed in the same order as in the serial fill; the unpack adds an
  identity. So the ghosts are bit-identical to serial for any
  partition, and so is everything computed from them. It is the
  argument the thread invariant rests on, with a rank boundary in place
  of a thread boundary. It needs the transfer arithmetic not to depend
  on the destination type: no `@fastmath` or `@simd` reassociation in
  `transfer_kernel!`. That holds today, since "What the ghost fill
  costs" fixed its summation order.

**MPI+GPU.** The pack and receive buffers live on the field set's
backend and are allocated once per schedule (amended in step 2: once
per schedule *and variable count*, on first use, for the reason the
offsets are in points). They go to MPI directly
when MPI is device-aware (`MPI.has_cuda()` for CUDA). Otherwise they
are staged through host buffers of the same layout, which is the path
Metal always takes and the one the suite tests. The synchronization
after the pack is what makes a device buffer safe to send.
`MPI.has_cuda()` asks only Open MPI (and answers `true` for IBM
Spectrum MPI); for any MPICH it is `false` unless the environment
variable `JULIA_MPI_HAS_CUDA` says otherwise (MPI.jl 0.20.27's
`environment.jl`). So on a CUDA-aware MPICH the direct path is opted
into by that variable, and step 8 records whether the MPI on
Symmetry's H200 nodes is CUDA-aware at all. *(It is: HPC-X 2.20's Open
MPI 4.1.7, for which `MPI.has_cuda()` is `true`; both paths pass there,
and the direct one saves the two copies, 0.16–0.2 ms of a 2.5 ms fill
on four H200s. See step 8.)*

*(Step 8, where this was implemented; what it settled, and where it
amends the paragraph above:)*

- *Who decides* (amended: the paragraph had `MPI.has_cuda()` decide). The
  caller does, when it converts its communicator:
  `communicator(comm; deviceaware = true)` hands MPI the device buffers,
  and the default, `false`, stages every device message through host
  mirrors. Four reasons. A wrong "yes" is a crash, or silent garbage,
  inside the MPI library, and a wrong "no" costs two copies a message, so
  the safe answer is the default. `MPI.has_cuda()` is not a safe answer:
  it is `false` for every MPICH, CUDA-aware or not, unless an environment
  variable says otherwise, and it is about CUDA only, while the package
  cannot tell, without naming a device package, whether a buffer is a
  `CuArray`, a `ROCArray` or an `MtlArray` — so a `true` from an Open MPI
  built with CUDA would also send a ROCm buffer's device pointer to a
  library that cannot read it. The application knows its backend and its
  MPI, and `deviceaware = MPI.has_cuda()` is one line for it to write on
  CUDA (the docstring shows it). And the setting belongs with the
  communicator, since it is a property of the MPI library: not a keyword
  of `Forest`, which would put it beside the mesh's parameters, and not a
  TreeAMR environment variable or preference, which would be state the
  call does not show and, for a preference, a new dependency.
- *The ranks need not agree.* Staged or not, the bytes on the wire are
  the stage buffer's, so one rank may stage while its peer sends from
  the device. Nothing is gathered for it, and it is not part of the
  layout hash.
- *A verb, not a type test in the driver.* `hoststaging(comm, buffer)`
  is a new verb of the communicator layer, and the only one whose
  fallback on the abstract type answers rather than refuses: `true` for anything but a host `Array`.
  The MPI extension answers `false` for a device buffer only when the
  communicator was made device-aware. The CPU's buffers are `Array`s, so
  its path is the direct one whatever the setting, and the serial path
  never asks, since a stage without messages has no buffers. The setting
  is a field of `MPICommunicator` beside the duplicate, and not part of
  it: a device-aware forest and an ordinary one over the same
  communicator share the one cached duplicate.
- *The mirrors* are `Vector`s of the element type with the stage
  buffers' layout, one send and one receive mirror per stage and
  variable count, allocated on first use and kept in the `RemoteStage`
  beside the buffers (a new field, `mirrors`, and a fifth type parameter,
  `HB`), so a staged fill allocates no buffer after its first. They are
  page-locked for the backend through `KernelAbstractions.pagelock!`
  (since KernelAbstractions 0.9.40, inside the `[compat]` floor of
  0.9.42): CUDA pins them, which lets the copies run at the bus's rate
  rather than through the driver's bounce buffer; the CPU and Metal do
  nothing, Metal's memory being unified. A regrid stage is built per
  regrid, so its mirrors are too, and CUDA unpins them when they are
  collected. No weak dependency on a device package was needed.
- *The order of a staged stage*: post the receives into the receive
  mirror; pack and synchronize; copy the send buffer down (`copyto!`
  into an `Array`, which returns once the host holds the data — CUDA.jl
  synchronizes before a download); send from the send mirror; run the
  local groups; wait for the receives; copy the receive mirror up; unpack;
  synchronize. The upload is asynchronous on CUDA from pinned memory, but
  it is queued before the unpack on the backend's queue, and the final
  synchronization completes it before the next call can receive into the
  mirror again. The send mirror is rewritten only by the next call's
  download, after this call's sends have been waited on.
- *The direct path* is the CPU's code with the device buffers' views in
  place of the mirrors'. The extension's buffer check, which accepted any
  contiguous view before — a view of a device array included — now
  accepts host memory only, `Array` or a contiguous view of one, and a
  dense device vector or a contiguous view of one only over a
  device-aware communicator; the refusal names the setting. MPI.jl hands
  a `CuArray` to the library by its device pointer through its own CUDA
  extension.
- *Interpolation and checkpoints* stay as steps 5 and 6 left them: both
  go through the host, at analysis and checkpoint cadence.

**Reductions: an allgather of per-rank partials** (decided).
`combine_blocks` in `state.jl` is already the single site where M7's
communication was to go. It folds the local blocks as today,
`mapreduce(identity, op, values)`, then `allgather`s one partial per
rank, which every rank folds in rank order. This supersedes the
`Allreduce` that the `mesh_mapreduce` bullet above planned, for four
reasons:

- *The association is the package's, not the library's.* MPI leaves
  the order in which an `Allreduce` combines floating-point values to
  the implementation and only recommends that it be reproducible, so
  whether every rank receives the same bits, and whether a rerun does,
  is a property of the MPI library. A rank-order fold of gathered
  partials is defined here and is the same on every rank by
  construction, which a `dt` or a norm-driven decision needs.
- *Any `op` and any `isbits` partial work*, with no custom MPI
  operation. Custom operations do not work on this development
  machine: MPI.jl 0.20.27 refuses every user-defined reduction operator
  on non-Intel architectures, a closure and a named function alike,
  unless it is registered statically with `@RegisterOp` (measured
  2026-10-01, Apple M3 Pro). A builtin operation exists only for MPI's
  native types, which `Float32x2` and a tuple partial are not, while
  `MPI.Allgather` of a tuple and of a struct with padding both worked.
- *At one rank the result is exactly the serial one.*
- *It costs what an `Allreduce` of one value costs*: `log P` latency
  steps, plus `P` small values per rank, about 160 KB at 10⁴ ranks for
  a `Float64` with its flag (below).

Three consequences:

- **An empty rank contributes no partial, not `init`.** `init` need
  only satisfy `op(init, init) == init` (the `init` bullet under
  "**Implemented**" above). So `max` over negative data from
  `init = 0` is right serially, and would be wrong if an empty rank's
  `0` entered the fold. Each rank therefore gathers `(hasvalue, value)`, and the fold
  skips the empty ones; `init` is returned only if every rank is empty.
  *(Amended in step 3: the example is wrong. `init` starts every
  block's fold, so with an associative `op` and `op(init, init) ==
  init`, one more `init` cannot change the result: `max` from 0 over
  negative data is 0 on every rank count. What an empty rank's `init`
  would change is a weighted reduction, since the weight scales a
  block's value and not `init`: `max` from 1 over values below 1,
  weight ½, is ½, and 1 with an empty rank's `init` in the fold. The
  workload checks exactly that, at a rank count with an empty rank.)*
- **What is exact.** Max, min and integer reductions are bit-identical
  across rank counts. A floating-point sum is reproducible to roundoff,
  since each rank's partial is associated differently when the
  partition moves, which the narrowing above permits. Gathering every
  block's partial instead would make sums independent of the rank
  count too, at `nleaves` values per reduction on every rank — the
  price the narrowing declined.
- **What stays as it is.** `volume_weighted_norm` reduces twice (the
  sum and the domain volume), so it costs two collectives; folding both
  into one tuple partial is the obvious change if that ever shows.
  `block_mapreduce` and `firing_boxes` stay local.

**Regridding.** Flagging is local and the decision is replicated.

1. `flag_blocks` and `firing_boxes` iterate the local blocks, and
   `regrid!` takes the local flags.
2. Each rank reduces its flags to the canonical mark, `markbox`'s flag
   and box, which is an `isbits` value, and `allgatherv` assembles the
   global vector in curve order. A caller's flag vector may be
   heterogeneous, so it is not gathered as it is.
3. `buffered_flags`, `complete_marks` and `balance!` run replicated,
   `O(nleaves)` per rank, as AMReX and Parthenon do. Every rank arrives
   at the same new leaves, so `regrid!` returns the same `Bool`
   everywhere. `buffered_flags` and `complete_marks` remain public over
   the *global* flag vector, which serially is the only one. *(Amended
   after step 7: the buffer's neighbour search is not replicated. Each
   rank searches from its own sources, and the recruits are gathered in
   a second `allgatherv` and applied on every rank, which gives the
   marks the replicated search gives; the rest of the completion is
   replicated as described.)*
4. The ghost fill before the transfer is the distributed fill, so a
   parent's ghosts are current on its owner.
5. The transfer is one stage over the new partition. `transfer_groups`
   classifies every new leaf as today, from the replicated old and new
   leaf arrays. The old owner of each source evaluates the transfer —
   a copy, a prolongation from the parent, or one child's share of a
   restriction — into a buffer shaped like that transfer's target box,
   and the new owner unpacks it. *(Amended while specifying: the
   first draft said a buffer shaped like the new block's owned box.
   That is right for a copy and a prolongation; a restriction's target
   is one child's part of it, and a coarsened block's children may have
   had up to `2^D` different owners.)* Sender computes is what makes this
   simple: the prolongation reads the parent's ghost layers, which only
   the parent's old owner holds. *(Amended after step 7: a rank
   classifies only the new leaves it will own and those whose sources
   it owns, not every new leaf.)*
6. That stage is also the repartitioning: a kept block whose owner
   changes is a copy between ranks. When `k` blocks are added near the
   start of the curve, an equal-count split shifts every later rank's
   boundary, by `k(1 − r/P)` blocks for rank `r`: up to `k` blocks
   moved per rank, `k/2` on average, the same order as the change
   itself.
7. `fs => nothing` reallocates to the new local count and moves
   nothing.

`adapt_to_initial_data!` and `total_mass` follow unchanged: the first
is a loop of collective calls, the second a `mesh_mapreduce`.

*(Step 4, where this was implemented; what it settled:)*

- *The canonical mark* is `RegridMark{D}`: the flag, whether the caller
  reported a box (`explicit`, which decides whether a `Keep` is a
  dilation source), and the box as two `NTuple{D,Int32}` corners,
  validated by `markbox` when the mark is made. `buffered_flags` reads
  it through methods of `markflag`, `markbox` and `issource`, so the
  public functions keep taking either caller form, and a gathered
  vector is concretely typed. `regrid!` makes the marks serially too:
  one code path, and a serial regrid measures the same (below).
- *The checks ride the digest gather.* `regrid!`'s argument checks —
  the pairs, each field set's forest and block count, each schedule's
  forest, staleness, ghost width, centering, element type and backend,
  `buffer`'s limits, the flag count and every flag's box — run inside
  `collective_checks`, with a layout hash of what the ranks must also
  agree on: each pair's variable count, ghost width, centering, element
  type, backend and operators, in order, `buffer`, `transfer`, and
  whether there is a hook. A rank that passed other field sets, or the
  same ones in another order, would match its regrid messages — all
  under tag 50 — to the wrong field set's. So one `allgather` of the
  digest carries every refusal, and the flags follow in an
  `allgatherv` only once all ranks have passed. Serially nothing is
  gathered and every refusal keeps its serial type and message.
- *The stage is `remote_stage`'s*, as step 2 left it to be.
  `regrid_sources` classifies every new leaf once per regrid (amended
  after step 7: only the new leaves this rank needs), not once
  per field set as before, into a `TransferPairs` under `GroupKey`s
  with direction zero and level 0 — targets new leaves, sources old,
  both global — and `regrid_stage` splits it by the new partition for
  the targets and the old for the sources, builds the local groups and
  calls `remote_stage` with the two owners and ranges. A coarsened
  block's `2^D` restrictions are `2^D` transfers with one target and
  their own sources, so children with different owners are simply
  received from different peers; nothing about them is special. Over
  one rank both ranges are every leaf, nothing is split, and the stage
  is the serial groups with no messages. `run_stage!` gained a form
  over a separate `dest` and `src` — the new mesh's array and the old
  one's — which the exchange's form now calls with `fs.work` twice.
  `transfer_groups` keeps its signature, as the serial stage's groups,
  for `thread_tests.jl` and `bench/gpu.jl`.
- *The ghost fill's checks stay rank-local* (decided in step 4).
  `fill_ghosts!` refuses a stale schedule, a field set over another
  forest, a block count that does not match, another layout or
  backend, each on the rank that sees it, and does not agree. Two
  reasons. Each check is a function of objects the collective contract
  keeps identical on every rank — the forest's generation and leaves,
  a schedule whose build was digest-checked, a field set's layout — so
  under the contract they agree without a message, and a check that
  fires on one rank only means the contract was already broken, which
  the digest catches at the next schedule build or regrid. And agreeing
  would cost a collective per fill, which is per right-hand-side
  evaluation: a global synchronization of every rank on the path "Two
  time scales" keeps free of everything but the neighbour messages.
  What a one-rank refusal does is make that rank throw before it posts
  a message, so the others wait in their first stage: a hang, never
  wrong data. An MPI application turns it into a job failure the usual
  way — `MPI.Abort` from a handler, as `test/mpi_workload.jl` does.

**Point interpolation.** `interpolate` becomes collective. Each rank
passes its own points, possibly none.

1. Each rank locates its points on the host against the replicated
   leaves, which gives each point's global leaf index. The owner
   follows from the split arithmetic in `O(1)`.
2. An `alltoallv` routes the points to their owners.
3. The existing kernel runs over the received points, with the block
   offset subtracted to reach the local block.
4. A second `alltoallv` returns the values and the `exclude` flags, in
   the caller's order.

Each value comes from the same kernel, block and stencil as serially,
so it is bit-identical. Outside points are agreed by an `allgather` of
each rank's first one, so every rank throws the same `ArgumentError`
together, instead of one rank throwing while the others wait in the
next collective. Ghosts must be current, as now.

*(Step 5, where this was implemented; what it settled, and where it
amends the four steps above:)*

- *The contract.* Every rank passes the same field set, basis, `derivs`,
  `vars` and `exclude`, and its own points, any number or none; it gets
  the answers for its own points, in its own order, on the field set's
  backend. An owner evaluates the points it receives with its *own*
  arguments, so those that shape the evaluation must agree, and are
  hashed for the check below; the points are the only per-rank input.
- *One `allgather` before anything is sent* (amended: the four steps had
  the outside points agreed after the values returned). The location runs
  on the host, before the routing, so a rank knows its outside points
  before it sends anything, and the same gather that agrees on them can
  carry everything else a refusal on one rank would leave hanging. It is
  a `ForestDigest` — this rank's refusal flag, the generation, `nleaves`
  and a hash of the arguments above, with the leaf fold and the brick
  zeroed — beside the rank's first outside point, if any, and that point
  as `NTuple{D,T}`. `digest_verdict` gives the verdict on the first part,
  whose layout message now names interpolation's arguments too. So a
  refused batch, for any reason, routes nothing and writes nothing;
  serially the in-domain values are written before the throw, which the
  docstring never promised either way.
- *Not the full forest digest.* Its fold over every leaf costs about
  14 ns a leaf — 10 µs at 288 leaves, 38 µs at 2304, 260 µs at 18432
  (measured, `D = 3`) — against 0.11 ms for the 496-point horizon batch
  at four threads. Per call at analysis cadence it would cost as much as
  the interpolation, while the generation and the leaf count catch every
  mutation made on one rank only, and the schedule builds and regrids
  check the leaves themselves.
- *Not the same message on every rank* (amended: the design said every
  rank throws the same `ArgumentError`). A rank that passed an outside
  point raises the serial message for its own first one, with a sentence
  naming the ranks that passed any; every other rank names the first such
  rank, its point index and the point. A rank's own point is what its
  caller can act on, and the others still all throw, together.
- *The messages.* Points travel as `NTuple{D,T}` — converted on the host
  by the same `T(x[d])` the serial kernel applies first, so the owner's
  fold, location and stencil start from the same bits. MPI.jl sends an
  `isbits` element type through a derived datatype it builds and caches
  (`Datatype(T)` in its `datatypes.jl`), which covers `Float32x2` points
  and values as it does in the regrid stage. Then the values come back as
  a flat vector of `T`: the received points are concatenated in rank
  order, so each asking rank's answers are one contiguous segment of the
  owner's `(nvars, K, n)` output. The `exclude` flags take a third
  `alltoallv`, of `Bool` (`MPI_C_BOOL`), only when there is a region;
  every rank knows whether there is one, since it is agreed. Each
  `alltoallv` is an `Alltoall` of the counts and the `Alltoallv`, so a
  batch costs one `allgather` and four or six all-to-all calls; the
  return could reuse the known counts, which was not worth a verb.
- *The order.* A stable counting sort by owner gives each owner a rank's
  points in that rank's order, and the inverse permutation puts the
  answers back. Bit for bit, the value of a point is the serial one: the
  same kernel, the same point, the same block's array (ghosts included,
  which the distributed fill made bit-identical) and the same stencil.
- *One kernel, one more argument.* `interpolate_kernel!` takes the block
  offset, `first(blockrange) − 1`, which is 0 serially. The owner
  re-folds and re-locates each point it receives — the kernel's own
  search, against the replicated leaves — rather than receive the leaf:
  that keeps the serial kernel unchanged, and it makes the kernel's
  per-point leaf a check. Every received point must locate in the
  owner's own blocks, and otherwise the owner raises an error saying the
  forests differ. That error is raised on one rank only, but it can
  only follow a broken collective contract that the generation check
  missed.
- *A device backend goes through the host.* The points are copied to the
  host for the location and the routing, the received ones uploaded for
  the kernel, its results downloaded for the return, and the answers
  uploaded into the caller's arrays. At analysis cadence, hundreds of
  points of `nvars × K` values, those copies are small, and the device
  location stays the serial path's. Step 8's device buffers are for the
  exchange; nothing here needs them. Checked once on Metal in process
  (`Float32`, three simulated ranks, one with no points, value, gradient
  and an excluded ball): bitwise equal to the serial Metal call. That
  check is a scratch script, not part of the suite, which has no device
  in its environment.
- *The host location* is one search per point at tens of nanoseconds, so
  it runs serially below 4096 points and through `threaded_foreach`
  above, every point writing only its own slot.

**Parallel checkpoints** (parallel HDF5 in M7, decided 2026-10-01 with
Erik). One shared file, with the version-1 layout unchanged:
`format_version` stays 1, and a file restarts on any rank count, serial
included, which is what "No coordinates, and no partition" was for.

- *Where the code goes.* The MPI-specific calls —
  `h5open(path, mode, comm, info)`, the `mpio` file access, the
  collective transfer property — come from HDF5.jl's own MPI
  extension, which loads only with MPI. So they go into a second
  extension, `TreeAMRHDF5MPIExt`, triggered by HDF5 and MPI together.
  `TreeAMRHDF5Ext` gains hooks, dispatched on the forest's communicator
  type, for opening the file, for creating objects and for the slab
  each rank reads and writes, with today's code as the serial methods.
- *Collective metadata.* Every rank creates every group, dataset and
  attribute with the same arguments, which the layout keeps to a fixed
  dozen or so at any rank count (see "Parallel I/O and M7" under
  [Checkpoint and restart](#checkpoint-and-restart)). Attribute values
  must agree, so the provenance is formed on rank 0 and broadcast,
  since `created`, `hostname` and `nthreads` differ between ranks. It
  gains `nranks` (decided 2026-10-01 with Erik), an additive field
  whose obvious default for an older file is 1, so the format does not
  change.
- *Raw data by hyperslab.* Each rank writes and reads the last-axis
  hyperslab of its own blocks, `blockrange`, in every dataset: the leaf
  columns `root`, `level` and `coords`, and each field set's `data`.
  The transfers are collective, as a filtered dataset requires. With
  filters a chunk is one block and variable, so every chunk has exactly
  one writer. A rank with no blocks takes part with an empty slab.
- *Durability.* The write still goes to `path * ".partial"`. Under
  `sync = true` the flush has to reach every rank's writes, not only
  rank 0's: on a parallel file system each client caches its own, and
  an `fsync` on rank 0 flushes none of the others'. So the ranks make a
  collective `H5Fflush`, which the MPI-IO driver turns into
  `MPI_File_sync` on every rank (confirmed in step 6 against libhdf5
  2.2.0's source, below), and then close the file collectively. Rank 0 then
  flushes the file itself as today (`F_FULLFSYNC` on macOS, where a
  plain `fsync` does not reach the drive), renames it and flushes the
  directory. A barrier follows, so that no rank returns before the
  rename.
- *Errors.* Everything that can be refused — the field sets, the
  forest, the path, the plain data — is checked before the first
  collective HDF5 call, and the verdict is agreed by `allgather`, so a
  refusal is raised on every rank with the same reason. An error inside
  a collective HDF5 call on one rank cannot be recovered portably,
  since the others are waiting in it. It is fatal to the job, as in any
  MPI code, and documented as such.
- *Plain data and the do-block are collective.* `write_plain` and the
  `f(app)` callback run on every rank with the same data, because an
  attribute written collectively has one value. `write_plain` checks
  this by gathering a digest of the encoded bytes, and refuses data
  that differ between ranks; that is step 6's refusal test.
- *Loading.* Every rank reads all of the leaf columns, since the forest
  is replicated, builds the forest with `comm` through the validated
  `leaves` path, and reads its own slab of each field set into its
  state vector.

*(Step 6, where this was implemented; what it settled, and where it
amends the bullets above:)*

- *Where the code goes* (amended). `TreeAMRHDF5MPIExt` holds one call,
  `open_parallel_file(::MPI.Comm, path, mode)`, which is HDF5.jl's
  `h5open(path, mode, comm, info)`. The collective transfer property is
  HDF5.jl's own `dxpl_mpio = :collective`, which needs no MPI, and
  everything else is in `TreeAMRHDF5Ext` beside the serial code. The
  hooks dispatch not on the communicator type, which the HDF5 extension
  cannot name (it is `TreeAMRMPIExt`'s, and the load order of two
  extensions is not defined), but on an `Access` chosen by `commsize`:
  `Alone` for one rank, whose methods are M9a's calls unchanged, and
  `Ranked` otherwise. The extension reaches MPI through two stubs in
  `src/`: `librarycomm(comm)`, whose method in `TreeAMRMPIExt` returns
  the duplicate, and `open_parallel_file`, whose fallback refuses with
  the reason. A forest over a one-rank MPI communicator takes the
  serial path, and its file is a serial file with `nranks = 1`.
- *Who writes what.* Every rank creates every group, dataset and
  attribute with the same arguments, and writes every attribute. A
  dataset over the blocks — `root`, `level`, `coords` and each `data` —
  is written by last-axis hyperslab in one collective transfer, an
  empty rank with an empty selection (`H5Sselect_none`, which HDF5.jl
  does not wrap). Every other dataset — the extents, the provenance,
  the plain data — is written by rank 0 alone with an independent
  transfer, and read by every rank. That is legal because parallel
  HDF5 allocates an unfiltered dataset's storage when it creates it
  (`H5D__create` in `H5Dint.c`), and it keeps a value from being written
  `P` times over to the same bytes.
- *No variable-length data in a parallel file* (found in step 6).
  libhdf5 refuses to write variable-length data through the MPI-IO
  driver (`H5D__write`: "Parallel IO does not support writing VL or
  region reference datatypes yet"), from any number of ranks, and
  HDF5.jl stores an array of strings as variable-length. Attributes are
  not affected (the `features`, `centering` and `parity` arrays wrote
  and read back), nor are scalar strings, which HDF5.jl stores at a
  fixed length. So in a parallel file a plain-data array of strings is
  fixed-length UTF-8, NUL-padded to its longest string; `read` returns
  the same `Array{String}`, so the item reads back the same, with the
  same `type` and `eltype`, and the format version is unchanged. A
  string in such an array that holds a NUL is refused before anything
  is written, since the padding would lose it. The M9a reader (0.1.4)
  reads a parallel file, fixed-length strings and filtered data
  included (checked).
- *Provenance* is rank 0's, gathered to every rank by an `allgatherv` to
  which the other ranks contribute nothing, so no broadcast verb was
  added. `nranks` follows `nthreads`; a serial file carries `nranks = 1`
  (decided here: the field is then in every file M7 writes, and a file
  without it is from before M7, which `read_provenance` reads as 1).
- *The agreement.* Before the file is created every rank runs the
  serial checks — the field sets, `application`, the geometry type —
  and walks `data` the way `write_plain` will write it, refusing what it
  would refuse, with the same messages, and folding a hash of every
  item's name, type and bits. One `allgather` carries a `ForestDigest`
  (the full one: a forest that differs would write a corrupt leaf
  list), the hash of the path and every keyword but `data` as its
  layout, and the data hash, and `digest_verdict` gives the verdict, the
  data a message of its own. A load agrees on its arguments the same
  way, with rank 0 alone checking that the file exists and is HDF5;
  what the file holds is the same for every rank, so a refusal of its
  contents comes on every rank at the same point without a message.
  Serially the walk runs too, so a value outside the plain-data types is
  now refused before the file is created rather than while it is
  written, with the same message.
- *How much of the plain data is checked* (decided here): all of it, by
  that hash, once per `save_checkpoint` and once per `write_plain` in
  the do-block. Two reasons. Under the collective contract a value that
  differs between ranks need not fail: an attribute written with
  different values is undefined in parallel HDF5, which does not check
  it, and a dataset that rank 0 writes alone would silently record rank
  0's value, so a restart would read one rank's run state as
  everyone's. And the
  cost is a walk over values that are small next to the field data, plus
  one collective per call. What the do-block writes through HDF5 itself
  is not checked; the docstring says it must be the same on every rank.
- *Durability, confirmed.* In libhdf5 2.2.0, `H5Fflush` reaches
  `H5F__flush` (`H5VLnative_file.c`), which calls `H5F__flush_phase2`
  with `closing = false` and so `H5FD_flush` and the driver's
  `H5FD__mpio_flush`, which calls `MPI_File_sync` unless the file is
  closing (`H5FDmpio.c`). So a flush as the file closes does not sync,
  and the explicit collective flush before the close is needed. In
  MPICH's ROMIO, `MPI_File_sync` is `ADIOI_GEN_Flush`, an `fsync` on
  each rank that wrote through its own descriptor, which under
  collective buffering are the aggregators that did the writing. Then
  rank 0 does what a serial save does — `F_FULLFSYNC` on macOS, where
  the ranks' `fsync` stops short of the drive, then the rename, then the
  directory — and an `allgather` of whether it succeeded is the barrier,
  so every rank returns once the checkpoint is in place, or throws.
  Whether BeeGFS honours each client's `fsync` is the file system's,
  and is not something a test here can see.
- *Collective metadata reads are not enabled.* Each rank reads the
  file's metadata independently, which is always correct, and keeps a
  do-block that reads on rank 0 alone legal. At thousands of ranks the
  independent reads of one small object header may become the cost
  (`H5Pset_all_coll_metadata_ops` is the remedy); the Symmetry
  measurement is where that would show. *(At up to 8 ranks on one node,
  step 6, it did not; the multi-node run is open.)*
- *Errors inside a collective call* remain fatal, as above. A
  `write_plain` refused in the do-block is agreed, so every rank leaves
  the block together and closes the file collectively, and the partial
  file is removed (tested).

**What the feasibility check found** (2026-10-01, in a scratch
environment on the development machine: Apple M3 Pro, HDF5.jl 0.17.4,
HDF5_jll 2.2.2+0, which is libhdf5 2.2.0, and MPI.jl 0.20.27).

- *Parallel HDF5 needs nothing chosen.* Every one of HDF5_jll 2.2.2's
  60 artifacts is an MPI build, tagged by MPI ABI: `mpich`, `mpiabi`,
  `openmpi` and `mpitrampoline`, all four on `x86_64-linux-gnu` and on
  `aarch64-apple-darwin`, and `microsoftmpi` on Windows. MPIPreferences
  selects the one matching the MPI binary. So `HDF5.has_parallel()` is
  `true` out of the box, and has been all along under M9a, which loads
  the same MPI build and never opens a file in parallel.
- *Both binaries work.* On Julia 1.11.9 with the default binary,
  MPICH_jll 5.0.2, the `mpich` artifact loads. On Julia 1.13.1 it is
  MPIABI_jll with the `mpiabi` artifact, because a `LocalPreferences.toml`
  in the global v1.13 environment selects it on this machine. That
  preference stacks into every project, and subprocesses must see the
  same one as the `mpiexec` that launches them; launching them with
  `MPI.mpiexec()` from a parent with the same load path ensures that.
- *What was run.* Under `mpiexec -n 2` and `-n 3`,
  `h5open(path, "w", comm, info)` created a file with an attribute and
  a dataset written by column, on both Julia versions. On 1.13, a
  contiguous dataset and a chunked one with `Deflate(1)` were written
  collectively by last-axis hyperslab, one chunk per column, with one
  rank's slab empty, and read back correctly, serially and in
  parallel.
- *What was not.* Linux CI was not run. Its artifact exists for the
  default binary, and step 3, which adds MPI to the test environment,
  will show it there. `MPI.has_cuda()` exists in MPI.jl 0.20.27 and
  returns `false` for MPICH_jll here.

**What an MPI test costs** (measured the same day). One `mpiexec -n 3`
subprocess, two threads per rank, that loads MPI and TreeAMR, builds a
forest and does one `Allreduce`, takes 1.7–1.8 s of wall clock in three
runs, 1.04 s of it inside the script on rank 0. The M5 workload,
`test/thread_workload.jl`, run under `-n 3` with each rank doing the
whole of it, takes 29.8 s, against 24.0 s for one process at two
threads. The ranks compile in parallel, so a rank count costs about one
serial workload plus a quarter. Two such subprocesses (`-n 2`, `-n 3`)
would add about a minute to the suite, and the serial reference can run
in-process. CI runners have three or four cores, while `-n 3` at two
threads wants six, and MPICH polls while it waits; on CI the ranks
should therefore run one thread each.

**The multi-block check** (the standing instruction, checked
2026-10-01). Nothing here obstructs a conforming multi-block forest.

- Ownership is by curve index, whatever the roots' connectivity.
- Candidate peers come from `neighbor_keys`, in `forest.jl`, which is
  where a multi-block connectivity would live.
- A remote transfer is still a stencil built in `schedule.jl` and
  evaluated by the one kernel, so an oriented root face would change
  only the stencils. The pack evaluates them, and the unpack is a copy
  into a target box whatever the orientation.
- Face-resident quantities cross ranks through the interface
  restriction's path.
- The checkpoint stores no partition.

## Ecosystem integration

- **Time integration:** OrdinaryDiffEq.jl via the flat state vector (see
  above).
- **Elliptic solvers:** no solver-specific machinery in the package; the
  ghost/operator infrastructure suffices to build composite-grid
  operators. (Multigrid on the tree hierarchy would require overlapping
  coarse data, which leaf-only storage does not provide — out of scope.)
- **I/O:** checkpoint and restart through HDF5.jl, as a package
  extension — M9a, done, as specified under
  [Checkpoint and restart](#checkpoint-and-restart). Visualization
  export is M9b; an ADIOS2 backend only if parallel HDF5 does not scale
  at M7 (see [Open questions](#open-questions)).
- **Visualization:** export is M9b, after M7, with its candidates
  listed under [Milestones](#milestones). The original plan here, VTK's
  non-overlapping AMR format, does not exist in VTKHDF (see "Formats
  considered" under [Checkpoint and restart](#checkpoint-and-restart)).

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
- **What M9a leaves open** (2026-09-29; see
  [Checkpoint and restart](#checkpoint-and-restart)). None of it is
  needed to restart a run.
  - Conversion on load, such as `Float64` into `Float32` for a device
    without fp64. Version 1 loads the saved type only, since exactness
    is the point of a checkpoint.
  - Partial loads: a subset of a field set's variables, or a range of
    blocks.
  - Appendable time series: several states in one file.
  - The parallel I/O design for M7: one shared file, as the version-1
    layout allows, against one file per I/O process plus a wrapper
    file. It is benchmarked on Symmetry and other HPC systems first.
    *(Decided 2026-10-01 with Erik: the shared file, specified under
    [Distributed meshes](#distributed-meshes). What stays open is its
    throughput on a cluster file system, measured in M7's step 6.)*
  - An ADIOS2 backend, if parallel HDF5 does not scale at M7. The data
    model maps one-to-one onto ADIOS2 variables (the datasets) and
    attributes.
  - A refusal for a missing filter (found 2026-09-29, measuring M9a). A
    file saved with H5Zzstd's filter and loaded without `using H5Zzstd`
    fails with HDF5's `H5Error`, a plugin it cannot find in a build
    directory, rather than with a reason. The loader could read the
    data set's filter pipeline first and name the filter and the
    package that provides it.
- **The integrator's own passes are not owner-based** (raised by
  TreeGeneralizedHarmonic, 2026-09-25, after it adopted the ownership
  policy of [Parallelism](#parallelism); decided the same day not to
  optimise it further, below). An external
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

  *Polyester: not pursued further* (decided 2026-09-25), for what the
  same job measured. Its stage
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

  *Not pursued further* (decided 2026-09-25). The integrator's passes
  stay as measured above: no Polyester (`thread = True()`), no
  owner-aware state vector type to route OrdinaryDiffEq's broadcasts
  through `launch_by_owner!` (the suggestion this item first made,
  dropped), and no further work on OrdinaryDiffEq's performance. What the
  time integration has to get right is
  correctness, and above all that limiters are applied where the
  method's stability argument puts them. That is the part still open.
  OrdinaryDiffEq's `SSPRK33`, which TreeHydro and `test/burgers.jl` use,
  is written in Shu–Osher form: it limits every stage value and carries
  the limited value into the next stage, so each stage is a convex
  combination of forward Euler steps from limited states. IMEXRungeKutta
  limits a stage value only where `f_exp!` reads it, and builds the next
  stage from `uⁿ` and the tendencies, so for an explicit SSP method the
  convexity is lost. The script and its log are in
  `/mnt/beegfs/eschnetter/claude/TreeGeneralizedHarmonic/threading-bench`
  on Symmetry (`threadbench.jl`, `threadbench.sbatch`,
  `out/threadbench-563749.log`).

  *IMEXRungeKutta as the second integrator* (measured 2026-09-25, in
  `test/imex_tests.jl`; its 1.1 has explicit tableaus, `Euler`, `RK4` and
  `SSPRK33`, with `solve_imp! = nothing`). With both limiters passed —
  the stage limiter for what `f_exp!` reads, the step limiter for what
  persists — and the initial data and every regrid's output limited by
  the application, every right-hand side sees a limited state and every
  stored state is limited. What the Butcher form gives up is the
  Shu–Osher carry-forward, which only a positivity proof built on
  limited stage values needs; neither TreeHydro's atmosphere nor
  TreeGRRMHD's has one. What it gains is exact bookkeeping: a stage
  correction never enters the state, so with a conservative right-hand
  side the drift of a conserved total *is* the step limiter's injection.
  On Burgers with a cap below the initial maximum and an unlimited
  reconstruction, that holds to 6e-16 in `D = 1` and 4e-15 in `D = 2`;
  through OrdinaryDiffEq's `SSPRK33` with the same two limiters the step
  limiter injects nothing (its last stage-limiter call already capped
  the result) and the drift is the weighted stage injections instead —
  TreeHydro's "a bound, not an equality" under `:stage`. The rest of the
  file: `RK4` by owner, with a partition from `threadchunks`, is bitwise
  its broadcast and within 7e-14 of OrdinaryDiffEq's `RK4` on the wave;
  the vertex-centered wave keeps its rates (1.99 in `D = 1, 2`); Burgers
  conserves mass to 0.0 through `SSPRK33` across coarse-fine faces, against
  a leak of 1e-3 without the fixup; and each block's state entries are
  handled by the thread `map_blocks!` runs the block on. The partition
  function itself stays open (IMEXRungeKutta's `CODE.md` asks TreeAMR
  for it); `state_partition` in the test file is the candidate.
  `bench/stepping.jl` times a step by integrator. On the development
  machine at 6 threads (960 blocks of `16³`, a 63 MB state) the RHS is
  most of a step and the two integrators are within 6 %; it is the
  Symmetry run that will say what the serial passes cost there.

## Milestones

Each milestone has a concrete acceptance test; serial correctness is
established before any parallelism. The numbers are the order the
milestones were planned in; **M8 is done before M7** (decided in the M8
design, see the M8 entry), and so are M10, which was added after M8,
and M11, added after M10 for a downstream horizon finder. **So is M9a,
checkpoint and restart** (decided 2026-09-29), because the downstream
applications need to restart long runs before they need MPI. M9 is
split for it: its second half, M9b (visualization export), stays after
M7. M9a's layout was chosen so that M7 need not change it, subject to
M7's benchmarks. The list below is in execution order. M7 is implemented
(2026-10-02), with its cluster measurements open, so M9b is next.

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
- **M11 — Point interpolation.** *(Done.)* `interpolate` and
  `locate_point`, as specified under [Point
  interpolation](#point-interpolation). Asked for by
  TreeGeneralizedHarmonic, whose apparent-horizon provider carried a
  stopgap — 3D, vertex-centered, host-only, `maxlevel` searches per
  point — and listed it as its first upstream prerequisite; porting it
  onto this is that package's next step. *Accept:*
  - `locate_point` agreeing with exact rational leaf boxes (half-open,
    the upper domain face to the last leaf, a periodic upper face to the
    first) on random forests in `D = 1, 2, 3`, on the finest lattice as
    well as at random;
  - exactness on tensor polynomials of degree `n − 1`, value and every
    first derivative, through three levels, at nodes, on block faces and
    at the domain's corners, for `n = 2…6` in `D = 1, 2` and `n = 3, 4`
    in `D = 3`, for cell, vertex and face centering; and at `G = 0` and
    `G = 1`, where the stencil shifts;
  - not exact one degree higher, and converging at rate `n` (value) and
    `n − 1` (gradient);
  - continuity inside a block on random data, across nodes and
    midpoints;
  - a point one period away agreeing to roundoff; beyond a reflecting
    face the parity-signed mirror, gradients included, exactly for data
    of definite parity, over six combinations of face kinds and
    centerings;
  - the ellipsoid test agreeing with enumeration bit for bit, and the
    flags through `interpolate` agreeing with the stencils
    `query_stencil` names;
  - the kernel's value equal to the contraction, with exact rational
    weights, of the block's own stored points over that stencil, on
    random data;
  - refusals with reasons; `Float32` and `Float32x2` exact and never
    widened; device agreement with the host; a line in the thread
    workload.

  *(Measured 2026-09-25; `test/interpolate_tests.jl`, `bench/interpolate.jl`.)*
  The design needed no amendment. Five things are worth recording.

  - **Allocation and hidden arithmetic, audited.** For `Float64` and
    `Float32` a batch allocates a fixed ~3 KB of host setup (the origins
    and spacings vectors, the variable list) plus the 4-byte block index
    per point, and nothing per point in the kernel; the kernel's LLVM
    IR calls no `Rational`, `BigInt` or `BigFloat` code and no generic
    dispatch. `Float32x2` did not pass at first: 535 bytes and 15 µs per
    point, from `pointoffsets` (`storage.jl`) converting `1//2` through
    `BigFloat` at run time — which every position-forming kernel in the
    package shares, `fill_by_coordinates!` included. It is now `one(h) /
    2`, the same value in every binary type, and `Float32x2` allocates
    what `Float64` does; with the innermost closures marked `@inline` it
    takes 12 µs per point, 14 times `Float64`, against 10.5 times for a
    bare double-float multiply-add loop.

  - **Rates.** A smooth 2D field, `N = 8 → 16`, over a three-level
    mesh: value **2.96 / 3.20** at `n = 3` (cell / vertex), **3.97 /
    4.08** at `n = 4`, **5.04 / 5.11** at `n = 5`; gradient **1.79 /
    2.06**, **3.05 / 3.27** and **3.86 / 4.01**. The suite holds them to
    `n − 0.4` and `n − 1.4`.
  - **Cost against the stopgap** (`bench/interpolate.jl`,
    `bench/symmetry_interpolate.sh`; the stopgap is a verbatim copy of
    TreeGeneralizedHarmonic's core run on the same batches). The
    horizon finder's workload: points in a shell through three levels,
    20 variables, value and gradient, `Lagrange(4)`, vertex-centered with
    `G = 2`, 176 leaves, a 249 MiB working array; best of 50 whole
    `interpolate!` calls. On a Symmetry AMD EPYC 7532 node (64 cores, 8
    NUMA domains, threads pinned), in ns per point:

    | threads | 496 points | stopgap | 49600 points | stopgap |
    |---|---|---|---|---|
    | 1 | 2083 | 1926 | 2214 | 1904 |
    | 8 | 356 | 312 | 267 | 258 |
    | 16 | 256 | | 135 | |
    | 64 | 288 | 643 | 42.1 | 43.9 |

    So the finder's own batch takes **0.13–0.14 ms from 16 threads up**,
    against the stopgap's 0.32 ms at 64; a large batch scales 53-fold
    to 64 threads. Serially this is 8–16 % behind the stopgap on the
    EPYC, and 25 % ahead on an Apple laptop (781 against 1042 ns): the
    stopgap accumulates all 20 variables in one `SVector`, which AVX2
    vectorizes across the variables, where this contracts one variable
    at a time. Contracting the variables in static chunks is the
    recorded way to close that if it ever matters; at eight threads the
    gap is 3 %, and at 64 it has reversed. Three measured causes were
    fixed on the way: a `Val` for the derivative multi-indices (run-time
    ones made the selections dynamic tuple indexing, 21 % serially), the
    closure boxing above, and a floor of 32 points per CPU task (the
    496-point batch took 0.29 ms at 64 threads without it, 0.14 at 16).

    **NUMA does not matter here.** A point is not owned by any thread,
    so a query reads its block from wherever first touch put it — and
    eight threads bound to one domain with their memory there run at
    the pinned eight-thread rate (264 against 267 ns), and 64 pinned
    threads beat `numactl --interleave=all` (42.1 against 49.7). A batch
    touches too little of the array for remote reads to show.

    `locate_point` is 113 ns per point on the laptop and 125 on the
    EPYC, against 249 for the per-level descent. The allocation is a
    fixed ~10–20 KB of host setup per batch plus four bytes per point,
    against the stopgap's 890 bytes per point.
  - **Suite cost.** 92054 tests in 4m35–4m54 at eight threads after M11,
    against 4m49 for M10 (within the run-to-run noise at this length);
    the new file takes about 20 s on its own, almost all of it
    compiling one kernel per (dimension, order, centering) case, and
    the thread-workload subprocesses about 2 s each. On Julia 1.10, in
    one thread, 99962 tests in 3m10. The device suite passes on Metal
    (`Float32`) and on an H200 (`bench/symmetry_gpu.sh`: 93348 tests,
    `Float64` and `Float32`, in 10m09), the interpolation's agreement
    with the host included.
  - **On a device** (one H200, `bench/symmetry_interpolate.sh cuda`), the
    same batches: 496 points in **0.26 ms** (`Float64`) and 0.28
    (`Float32`), which is latency — the small uploads of the leaves and
    the geometry, the launch, the on-device check for outside points,
    the synchronization — and so no better than the node's 16 host
    cores (0.12 ms). Large batches are throughput: **6.4 ns per point**
    in `Float64` and 4.7 in `Float32` at 496000 points, against 178 on
    the 16 cores. For a device-resident run the comparison that matters
    is with the stopgap's `hostcopy` of the whole field set per find;
    caching the uploads per forest generation would take the small
    batch down further, and is not done. The host allocates a constant
    17 KB per batch: the outside check is reduced on the device, so only
    a batch with an outside point copies its block indices back (the
    first version copied them always, 2 MB at 496000 points, and ran at
    12.2 ns per point before the `Val` for the multi-indices).
- **M9a — Checkpoint and restart.** *(Done.)* Done before M7
  (decided): TreeHydro's long runs and TreeGeneralizedHarmonic's
  production runs, estimated at 38–149 h, outlast any queue's day and
  need to stop and resume before they need MPI, and
  TreeGeneralizedHarmonic and TreeGRRMHD were waiting for M9.
  Serial. A forest from a validated leaf list, and `save_checkpoint`,
  `load_checkpoint`, `write_plain` / `read_plain` and
  `checkpoint_environment` in the package extension `TreeAMRHDF5Ext`,
  as specified under [Checkpoint and restart](#checkpoint-and-restart).
  *Accept:*
  - the forest from a leaf list: random refinement patterns round-trip,
    checked against the oracles, and a gap, an overlap, a duplicate, an
    unsorted list, an unbalanced list and a root out of range are each
    refused with the reason;
  - **the round trip is bitwise**, in `D = 1, 2, 3`, for cell, vertex,
    face and edge centering, in `Float64`, `Float32` and `Float32x2`,
    with periodic, reflecting (with parity) and outer faces: after
    `scatter!` and `fill_ghosts!`, the leaves, the forest's parameters,
    the field-set metadata, the state vector byte for byte and the
    working arrays all equal the saved ones;
  - **a restart continues bit-identically through regrids**: a chunked
    driver saves after chunk `k`, drops every object, loads into fresh
    ones and continues, and its leaves and state vector equal an
    uninterrupted run's byte for byte. This is checked for the
    vertex-centered wave study (`test/wave.jl`, RK4) and for the
    conservative Burgers study (`test/burgers.jl`, `SSPRK33`, the
    interface fixup), whose face flux sets with `G = 0` are not saved.
    The time is part of the plain data;
  - refusals, each saying why: a newer `format_version`, an unknown
    feature, a `Float32x2` file loaded without `types`, a field set
    over another forest, a value outside the plain-data types;
  - an atomic write: a do-block that throws leaves the earlier file
    intact and no `.partial` file behind;
  - plain data round-tripped exactly (`Rational`s, `Symbol`s, nested
    NamedTuples and Dicts, tuples, arrays, `nothing`,
    `VersionNumber`s), and `checkpoint_environment` writing the
    `Project.toml` and `Manifest.toml` that were stored;
  - a device round trip in the device suite;
  - **throughput measured** (`bench/checkpoint.jl`): save and load rates
    and file sizes, uncompressed and with `Shuffle` + `Deflate(1)`, and
    with zstd and bitshuffle where those packages are available, on a
    smooth wave pulse and on a Burgers shock, recorded here with the
    recommended filter chosen from them;
  - TreeWave and TreeHydro still green against it, the change being
    additive.

  *(Measured 2026-09-29; `test/checkpoint_tests.jl`,
  `bench/checkpoint.jl`.)* The design needed one amendment, the name a
  limb type is recorded under (see "Element types" under
  [Checkpoint and restart](#checkpoint-and-restart)); the
  implementation fixed the spellings the design had left to it, and
  chose `rename` over `mv(…; force = true)`, which on Julia 1.11 removes
  the old file before the new one is in place. Four things are worth
  recording.

  - **Restarts are bit-identical, and the test would see it if they
    were not.** Both studies continue byte for byte through regrids
    after the restart point, and both fail on a one-ulp error in the
    restored time, which was checked. The device round trip passes on
    the CPU backend in the suite, and on Metal in `Float32` (the testset
    run on its own).
  - **Throughput** (the table and the reasons are under "Throughput and
    filters" in [Checkpoint and restart](#checkpoint-and-restart)). On
    the development laptop an unfiltered save reaches the page cache at
    5–7 GB/s and stable storage at 3–5, and a load of the 540 MB state
    runs at 3.6 GB/s on six threads, half of that time the HDF5 read. The
    recommendation is no filter, and `Shuffle()` with `ZstdFilter(1)`
    when size matters: 6.1-fold at 1.0 GB/s on a state that is mostly
    atmosphere, and 1.3-fold on a smooth one.
  - **Suite cost.** 93686 tests in 4m16 at one thread and 93738 in 4m47
    at eight threads, against 92054 in 4m35–4m54 at eight after M11;
    93686 in 3m48 on Julia 1.11.9 (with `sync`, 2026-09-29; 93677,
    93729 and 93677 before it). `checkpoint_tests.jl` (541 tests) adds
    about 30 s at one thread inside the suite, and takes about 57 s on
    its own, most of it compilation.
  - **The downstreams are green.** Against the M9a checkout, developed
    into scratch copies, at one thread: TreeWave 310 tests in 1m12, and
    TreeHydro 11893 in 4m16. Neither calls the new functions yet; each
    gains them with `using HDF5` once a release carries them.
- **M7 — MPI.** *(Implemented 2026-10-02. Measured on Symmetry the same
  day: the weak-scaling table of step 7 at up to four nodes and the H200
  run of step 8, both paths, pass; the checkpoint throughput of step 6
  ran on one node, and its multi-node part is open, so M7 is not yet
  marked done.)* Curve partitioning, distributed ghost exchange (for
  every centering, and the interface restriction with it, since both are
  transfers over the same schedule machinery), distributed regridding,
  and the `Allreduce` inside `mesh_mapreduce` (the planned global
  reduction under [Parallelism](#parallelism)), the one place a
  communicator appears in a reduction. *(Amended 2026-10-01: the
  reduction is an `allgather` of per-rank partials folded in rank
  order, not an `Allreduce`, and parallel checkpoints are part of M7;
  both are under [Distributed meshes](#distributed-meshes), which
  specifies the milestone. Interpolation routing is added with them.)*
  *Accept:* results match serial to roundoff — bit-identical for
  everything but floating-point sums, as [Parallelism](#parallelism)
  states; weak-scaling smoke test; then MPI+GPU with CUDA-aware MPI. In
  steps, each ending green and committed, with what it measured recorded
  here and in the commit body. Every step runs the full suite at one
  thread and at eight, with the thread-independence digests byte for
  byte, and from step 3 on the MPI tests with it:
  - **Step 0 — specification.** *(Done, 2026-10-01.)*
    [Distributed meshes](#distributed-meshes), after two feasibility
    checks recorded there: the stock HDF5_jll is a parallel build and
    works under `mpiexec` with MPICH_jll on Julia 1.11 and MPIABI_jll
    on 1.13, and an `mpiexec -n 3` subprocess costs about a serial
    workload and a quarter.
  - **Step 1 — local blocks, no messages.** `communicator.jl`, the
    forest's `comm` field and keyword, the split helper shared with
    `threadchunks`, and `blockrange`. Every block-index site becomes
    local: allocation, `blockkey` and the coordinate and fill origins in
    `storage.jl`; `block_origins` and `block_spacings`; the schedule's
    block count and `BoundaryPlan`; the block-count checks in
    `ghosts.jl` and `interfaces.jl`; `combine_blocks` and the volume
    loop in `state.jl`; `flag_blocks`, `regrid!`, the scratch forest and
    `firing_boxes` in `regrid.jl`; and `interpolate.jl`. *Accept:* the
    whole suite passes unchanged. New in-process tests, through a
    test-only `PartitionCommunicator(rank, size)` that answers rank and
    size and sends nothing, check the partition: the ranges tile
    `1:nleaves` in order and differ in length by at most one, ranks
    beyond `nleaves` are empty, and `blockkey` on every rank names the
    leaf that rank owns. *(Done, 2026-10-01.)* What it settled, and
    where it went beyond the plan:
    - *The split.* `equalsplit(n, p, i)` in `threading.jl`, the closed
      form of `threadchunks`' old loop, is the one helper:
      `threadchunks` returns its parts, and `blockrange(forest)` is part
      `commrank + 1` of `commsize`. The test compares it against the old
      loop for every `n ≤ 40` and `p ≤ 9`, and `threadchunks` against it
      at the suite's thread count.
    - *The keyword.* `Forest(…; comm = nothing)`, and
      `communicator(nothing)` is the `SerialCommunicator`, so the default
      is spelled the same way the MPI case will be. A `Communicator`
      subtype lacking a verb is refused at that verb by name, through a
      fallback method on the abstract type, rather than with a
      `MethodError` inside a reduction; that is what lets the test-only
      communicator answer rank and size alone. The serial `isend` and
      `irecv` refuse any peer, since a serial rank has none, and the
      serial `waitall` accepts only an empty list.
    - *The geometry per local block.* `block_origins(forest)` and
      `block_spacings(forest)` take the forest, not a field set, and are
      per local block through `blockrange`, so every kernel that reads
      them is unchanged. Tree queries — `nleaves`, `find_leaf`,
      `neighbor_keys`, `locate_point` — stay global, and `flag_blocks`
      walks the local range and passes the local index.
    - *The schedules build their local part* (amended: the plan had them
      untouched until step 2). `GhostSchedule` and `InterfaceSchedule`
      walk the rank's own targets in global leaf indices and then keep
      the transfers with both ends on the rank, shifted to local
      indices, together with the boundary regions of the rank's blocks.
      That is step 2's *local* class already; the test checks it against
      the serial schedule restricted to each rank's range, mirrored
      transfers and three levels included, over 2 and 3 fake ranks.
    - *What refuses a distributed forest until its step*: `fill_ghosts!`
      and `restrict_interfaces!` (step 3), `regrid!` (step 4),
      `interpolate` (step 5) and `save_checkpoint` (step 6), each with an
      `ArgumentError` naming the step; a reduction is refused by the
      missing `allgather` of a communicator that lacks one.
    - *The reduction is already written* (amended: the plan put it in
      step 3). `combine_blocks` gathers `(hasvalue, partial)` per rank
      and folds the ranks that have a value, in rank order, as
      "Reductions" above specifies — but at one rank it returns its own
      fold without calling `allgather`, so the serial value is unchanged
      by construction and a serial reduction keeps accepting a
      non-`isbits` partial, which `allgather` refuses. The multi-rank
      fold is first exercised by step 3's workload.
    - *Empty ranks.* `launch_by_owner!` skips an empty `ndrange` on a
      device too, `firing_boxes` returns an empty vector, and the
      `AllVariables` fill returns before checking its callback at a
      block that does not exist.
    - *Docs.* The `docs/src/api/distributed.md` page that step 9 was to
      add exists now, with `communicator`, `blockrange` and the internal
      verbs, since Documenter refuses a docstring that is on no page.
    - *Suite cost.* 98818 tests in 4m44 at one thread and 98870 in 4m51
      at eight; the difference from M9a's 93686 and 93738 is exactly
      `partition_tests.jl`'s 5132, so no existing test moved. The docs
      build.
  - **Step 2 — the distributed schedule.** The candidate remote targets,
    the three classes, the sorted layouts, the stages, and the
    packed-buffer accessors. *Accept,* in process, over 1–5 fake ranks,
    in `D = 1, 2, 3`, with periodic, reflecting and outer faces, every
    centering, three levels and the interface schedule:
    - the union over ranks of the local, send and recv transfers
      equals the serial schedule's transfer set exactly;
    - rank `r`'s send layout to `s` equals `s`'s receive layout from
      `r`;
    - every target is written exactly once;
    - a pack-then-unpack round trip, with every rank's buffers wired
      directly in one process, reproduces the serial `fill_ghosts!`
      bitwise, in `Float64`, `Float32` and `Float32x2`, including an
      odd variable that is zero at a reflecting wall (the `−0` case).

    *(Done, 2026-10-01.)* What it settled, and where it went beyond the
    plan:
    - *The data structures.* `GhostSchedule` keeps `phase1`, `phase2`
      and `levels`, which now hold the transfers *local* to the rank,
      and gains `stages`, a vector of `ExchangeStage`s in tag order:
      phase 1, then one per phase-2 target level. A stage holds its
      `tag`, its local groups and a `remote` part, which is `nothing`
      when the stage has no messages on this rank. Serially every stage
      is `nothing`-remote and its local groups *are* the phase (the same
      vector). `InterfaceSchedule` gains `stages` the same way, one per
      face dimension. A `RemoteStage` holds, per direction, the peers
      ascending with their segment lengths in points; one `LayoutEntry`
      per buffer slot (peer, `GroupKey`, global target and source,
      offset and size in points), kept on the host; the pack and unpack
      groups; the slot offsets on the backend; and the buffers. A
      `GroupKey` has an explicit total order (`keyorder`: kind ranked,
      then direction, offset, level, mirror state), so a layout never
      depends on `Dict` order.
    - *The tags*: phase 1 is 1, the level-`ℓ` stage `2 + ℓ`, interface
      dimension `d` is `40 + d`, and 50 is reserved for the regrid.
      They ascend in the order the stages run.
    - *The classification.* `split_received!` splits the transfers
      found for the rank's own targets into local ones, which are
      shifted to local indices, and received ones, kept global.
      `sent_transfers` runs the builder's own search (`block_sources!`
      or `interface_sources!`) for every candidate remote target,
      `remote_neighbors` in `forest.jl`, and keeps those with a local
      source. `leafowner` inverts the split in `O(1)`, through
      `equalsplit_part` beside `equalsplit`. `remote_stage` builds one
      stage's messages. It takes the target and source owners and
      ranges separately, so that step 4's regrid stage — targets in the
      new partition, sources in the old — is a call of it, and is not
      built here.
    - *Packs and unpacks.* A pack is the serial group's stencils with
      every target range moved to start at 1, `targetblocks` the slots,
      `sourceblocks` the local sources, sorted by (source, slot), and
      no parity column. An unpack is a width-1, weight-1 group from the
      slot into the serial target box, sorted by (target, slot), with
      the parity column. On the CPU, `run_phase!` bisects
      `sourceblocks` when the destination is a packed buffer, through
      `ownerblocks` and `ownercount`; the device path is the per-group
      one, unchanged.
    - *The driver.* `run_stage!` runs the five steps through the
      communicator verbs. `exchange_ghosts!` and `exchange_interfaces!`
      are the bodies of `fill_ghosts!` and `restrict_interfaces!` after
      their checks, and a serial fill now goes through them. The two
      public functions still refuse a distributed forest, so step 3
      removes the refusal and adds the MPI methods of the verbs.
      `run_stage!` takes the forest rather than its communicator: the
      `comm` field is abstractly typed, and passing it made every stage
      of a serial fill a run-time dispatch of 48 bytes (measured, then
      removed).
    - *Tests*, in `test/exchange_tests.jl`. They cover 1–5 simulated
      ranks (and `nleaves + 1` in 1D, which leaves a rank empty), `D =
      1, 2, 3`, periodic, outer and reflecting faces, `faces_forest`'s
      three levels, every centering, both operator families, and the
      interface schedule over every centering with a face dimension.
      They check the candidates against the serial schedule's readers.
      They check the union of the local transfers and the matched
      sent and received halves against the serial set, per stage, with
      every slot consistent with its layout entry. They check both
      ends' layouts entry by entry and the per-stage write counts
      against the serial ones, at most one write per point. They run
      the bitwise lockstep round trip, the boundary hook between the
      stages included: `Float64` over every centering, `Float32` and
      `Float32x2` over a subset, and the interface restriction in
      `Float64` and `Float32x2`. Finally the staged driver itself runs
      over an in-process mailbox communicator, one task per rank, in
      `D = 2, 3` at 3 and 5 ranks, for the ghosts and the interfaces,
      bitwise against serial. `gpu_tests.jl` runs the lockstep round
      trip on every backend; on Metal (Float32) it passes, with the
      rest of that file.
    - *Serial cost* (`bench/ghosts.jl` at its defaults: `D = 3`,
      `N = 8`, 4³ roots, 10 variables, order 4; best of 400 fills and 40
      builds, HEAD before and after, alternated). The fill is unchanged
      at one thread: 2.71–2.78 ms against 2.71–2.73 ms on the uniform
      mesh, 11.8–14.5 ms against 12.0–15.0 ms on the two-level one, with
      identical allocations (10912 and 97536 bytes). At four threads
      the uniform fill measured 0.51–0.52 ms against 0.64–0.65 ms; the
      cause was not traced. The schedule build is within noise,
      0.40–0.43 ms against 0.41 ms and 3.0–3.3 ms against 3.0–3.2 ms.
      It allocates 2.8 % more on the uniform mesh (389 KB against 379 KB)
      and 3.1 % more on the two-level one (4.65 MB against 4.51 MB),
      nearly all of it the per-key stencil memo below.
    - *Distributed build cost* (one thread, `D = 3`, `N = 8`, `G = 2`,
      order 4, the bench's two-level periodic mesh, slowest rank of `P`
      fake ranks). At 120 leaves the serial build is 3.4 ms and a rank's
      is 3.6, 3.1 and 2.6 ms at `P = 2`, 4 and 8. At 960 leaves (8³
      roots) the serial build is 9.7 ms and a rank's 11.0, 9.2 and
      7.3 ms. So at these sizes a rank's build costs about the serial
      one, not `1/P` of it. A profile of one of the eight ranks at 960
      leaves puts a third of the time in the search for the sent
      transfers and another third in building the remote stages. At
      120 blocks a rank, a 3D halo is about as many blocks as the
      rank's own, so that is the `O(local + halo)` the spec predicts,
      not a defect. It is a regrid-frequency cost, about 1,300–2,100
      sent transfers a rank here. The host stencils are memoized per
      group key, which the stage builder asks for again (10 % of the
      rank build).
    - *What was not done.* The regrid's stage, which is step 4, has
      only its builder's interface, as above.
    - *Suite cost.* 108338 tests at one thread, in 5m29 and 5m51 over
      two runs, and 108390 at eight, in 5m35 and 5m54. The increase over
      step 1's 98818 and 98870 is exactly the 9506 tests of
      `exchange_tests.jl` and the 14 of the CPU entry of the new
      `gpu_tests.jl` testset (7380 of the 9506 check the owner inverse),
      so no existing test moved. The thread-independence digests are
      unchanged. The new file takes about 50 s standalone, almost all of
      it compilation of the `Float32` and `Float32x2` kernels and of the
      3D centerings; that is most of the 45–65 s the suite grew by, and
      what to trim first if the suite has to shrink. The docs build.
  - **Step 3 — the MPI extension and the exchange.** `TreeAMRMPIExt`,
    the staged `fill_ghosts!` and `restrict_interfaces!`, the
    `combine_blocks` allgather, and the forest digest check; MPI in
    `[weakdeps]` and `[extensions]`, and in `test/Project.toml`. A new
    `test/mpi_workload.jl`, a standalone script like
    `thread_workload.jl`, prints on rank 0 digests of the leaves and of
    the state gathered in curve order, the integer and max reductions
    exactly, and the sums to `%.17g`. A new `test/mpi_tests.jl` runs it
    under `MPI.mpiexec()` at `-n 2` and `-n 3`. *Accept:* against a
    serial run, every line byte-identical except the sum lines, which
    agree to roundoff, over the vertex and cell waves, Burgers with the
    fixup, a reflecting box, three levels, and a rank with no blocks;
    the digest check refuses a forest mutated on one rank only, on
    every rank; and what the subprocesses add to the suite is measured,
    with the rank counts trimmed if it is large.

    *(Done, 2026-10-01.)* What it settled, and where it went beyond the
    plan:
    - *The extension.* `ext/TreeAMRMPIExt.jl` defines `MPICommunicator`
      (the duplicate, with its rank and size read once) and one method
      per verb, each a single MPI.jl call: `MPI.Allgather` of the
      `isbits` value, `Allgatherv!` after a gather of the lengths,
      `Alltoallv!` after an `Alltoall` of the counts, `Isend`, `Irecv!`
      and `Waitall`. `communicator(::MPI.Comm)` refuses MPI not
      initialized (or finalized), `COMM_NULL` and a thread level below
      `THREAD_SERIALIZED`, each saying what to do; the duplicate and its
      cache are as amended under "The communicator layer". A message
      buffer must be an `Array` or a `UnitRange` view of one, which
      MPI.jl passes as a pointer and a count: the exchange's per-peer
      segments are exactly such views, and anything else is refused
      rather than sent as a derived datatype. The fill and the
      interface restriction lose their refusals and run through
      `run_stage!` unchanged from step 2; the reduction needed nothing,
      since step 1 had written it.
    - *The digest*, in `forest.jl` (`ForestDigest`, `agree_on_forest`,
      `digest_verdict`, `collective_checks`), as amended under "Every
      forest mutation is collective": a fold over every leaf, the brick,
      the layout, and a refusal flag, so that a refusal on some ranks is
      raised on all. `GhostSchedule` and `InterfaceSchedule` run their
      argument checks through `collective_checks`; `regrid!` will in
      step 4. The in-process fakes of steps 1 and 2 answer the digest
      gather alone, by replication — their ranks are copies of one
      forest — so they still refuse every other verb they lack.
    - *The workload.* `test/mpi_workload.jl` prints, on rank 0, lines
      that do not depend on the rank count: per case the leaf count, the
      maximum level and a digest of the leaves; the serial transfer
      count, as the sum over ranks of local and received transfers, and
      whether the totals sent and received agree; digests, gathered to
      every rank in block order, of the working arrays *with their
      ghosts* after a fill, of the state after the steps, and of the
      working arrays after a last fill; and the reductions — a count,
      `linf`, `min`, a `max` of negative data and a weighted `max` from a
      non-neutral `init` exactly, the norm and the mass on `sum` lines.
      The cases: the cell-centered wave in 1D on two periodic leaves,
      which leaves a rank empty at `-n 3`, and on three levels against
      outer faces; the vertex-centered wave in 2D on three levels with
      the hook; the cell-centered wave in 3D, periodic, three levels; a
      reflecting box in 2D, vertex-centered with walls at both ends
      (the derived wall plane) and cell-centered with low walls, the
      first variable odd across x₁; Burgers in 2D on three levels with
      the fixup, the fluxes digested after it and mass conservation
      asserted; and every centering through one fill and, where a
      dimension is vertex-like, one interface restriction — cell,
      vertex and both faces in 2D, vertex, a face and an edge in 3D —
      over pseudo-random data that is a function of the global leaf and
      the stored index, so that no stencil reproduces it by accident
      and the injection at a vertex-like face is not a no-op. Lines
      starting with `#` depend on the rank count and are checked on
      their own: the refusals, the duplicate cache, `alltoallv` and
      `allgatherv` with empty contributions, and the negative control.
      Without the argument `mpi` the forests are serial.
    - *The test.* `test/mpi_tests.jl` runs the serial reference in
      process (the script in a module of its own, printing into a
      buffer), then `MPI.mpiexec()` at `-n 2` and `-n 3`, one thread a
      rank, with `Base.julia_cmd()` and the active project, under a
      deadline, and compares: every line byte for byte except the `sum`
      lines, within `rtol = 1e-12`. It checks the comparison itself on
      the serial output (a flipped digest character is caught, a sum
      moved by `10⁻⁹` is caught, one moved by three ulps is not). The
      negative control rewrites one rank's received ghosts from its last
      receive buffer with one value moved by an ulp, and asserts that
      the gathered digest changes. The refusals are a `refine!` on rank
      1 only (`GhostSchedule`) and on rank 0 only
      (`InterfaceSchedule`), other operators on rank 1, and an argument
      only rank 1's checks refuse; each is raised on every rank, and the
      test asserts the reasons. In process, the verdict is tested on
      digests of real forests, and a forest over `COMM_WORLD` in a
      process without `MPI.Init` is refused.
    - *The launcher's environment.* `MPI.mpiexec()` returns a `Cmd`
      carrying an environment — the library paths of the MPI binary —
      which interpolating it into a larger command drops. The test
      launches `setenv(cmd, mpiexec().env)`.
    - *What was measured.* At `-n 2`, `-n 3` and `-n 4` with one thread
      a rank, and at `-n 3` with two, every line agrees with the serial
      run byte for byte except the `sum` lines, which differ in the last
      one or two digits of 17 where they differ at all (`l2`, `mass`;
      Burgers' mass 16 against 15.999999999999998); at `-n 1` the output
      is the serial output exactly. Each `mpiexec` run of the workload
      takes 27–29 s of wall clock at any of these rank counts, and the
      serial run 25 s, nearly all of it compilation. Inside the suite
      `mpi_tests.jl` takes 58.4 s at one thread and 58.9 s at eight
      (timed around its `include`): the serial reference in process,
      which reuses the suite's compiled kernels, and the two `mpiexec`
      runs. That is the minute "What an MPI test costs" predicted, so the
      rank counts were not trimmed.
    - *Julia 1.11.* In the manifest-free copy, Julia 1.11.9 resolves
      MPI.jl 0.20.27 with MPICH_jll (its global environment's preference),
      against MPIABI_jll on 1.13.1 here, where the global v1.13
      `LocalPreferences.toml` selects it; `Pkg.test` copies the merged
      preferences of the load path into its sandbox (Pkg's
      `Operations.jl`, read, not measured), so the suite's ranks run the
      same binary as its `MPI.mpiexec()` either way. `mpi_tests.jl` alone
      passes there (32 tests, 1m04), and so does the whole suite, 108376
      tests in 5m08.
    - *Suite cost.* 108376 tests at one thread, in 6m10 and 6m00 over
      two runs, and 108428 at eight, in 6m01. The increase over step 2's
      108338 and 108390 is 38: the 32 of `mpi_tests.jl` and the 16 of the
      in-process verdict test, less a net 10 in the refusal assertions
      steps 1 and 2 made of `fill_ghosts!` and `restrict_interfaces!`
      over a distributed forest (11 removed, and one added: a fill over
      a communicator that answers rank and size alone now stops at
      `irecv`, by name). The thread-independence digests are
      unchanged. The docs build.
  - **Step 4 — the distributed regrid.** The flag gather, the transfer
    stage with the repartitioning, and `adapt_to_initial_data!`.
    *Accept:* the tracked pulse and the Burgers shock through regrid
    cycles in the workload, with leaves and state byte-identical to
    serial and mass conserved to roundoff across ranks; a regrid that
    migrates blocks between ranks, and one that coarsens a block whose
    children had different owners.

    *(Done, 2026-10-01.)* What it settled, and where it went beyond the
    plan (the design decisions are recorded under "Regridding" in
    [Distributed meshes](#distributed-meshes)):
    - *The driver.* `regrid!` runs its checks through
      `collective_checks`, gathers the canonical marks with one
      `allgatherv`, completes them replicated, and per field set fills
      the old mesh's ghosts (the distributed fill), zero-fills the new
      local array and runs the regrid stage through `run_stage!` from
      the old array into the new one, waiting for its sends before the
      next field set; then `rebuild_leaves!`. `fs => nothing`
      reallocates to the new local count, `transfer = false` moves
      nothing, and "nothing changed" returns `false` on every rank,
      since every rank decides it from the same gathered marks. The
      step-1 refusal is gone; `adapt_to_initial_data!`, `firing_boxes`
      and `total_mass` needed no change.
    - *The workload.* `test/mpi_workload.jl` gains the tracked pulse (a
      right-moving Gaussian, vertex-centered, outer faces with the hook,
      flagged through `firing_boxes` with a buffer, three regrids with
      RK4 steps between), Burgers' shock through three regrids with the
      fixup (conservative operators, the fluxes `=> nothing`; mass
      conserved across each regrid and the run, and on `sum` lines),
      `adapt_to_initial_data!` from a single leaf with the host callback
      and with a `firing_boxes` flag vector (two of three ranks start
      empty at `-n 3`), and a case that refines the first blocks of a
      uniform mesh and coarsens them back, and refines three of four
      roots and coarsens them back, over three field sets at once (cell,
      `nvars = 2`, `G = 2`, order 4; vertex, `nvars = 1`, `G = 1`,
      order 2; a face, `nvars = 3`, `G = 2`, order 4), then a regrid
      that changes nothing. The second half runs again in `Float32` and
      `Float32x2` over the cell-centered set, so both types cross MPI in
      the regrid stage and the exchange. After each regrid the leaves
      and every moved field set's array, ghosts zero, are digested
      before any fill, so the transfer itself is compared bit for bit.
      `#` lines count the blocks that moved up and down the ranks and
      the coarsened blocks whose children had more than one owner, and
      the test asserts at each rank count that both directions and the
      straddling case occur. Four more refusals: a bad flag box on rank
      1, a flag vector of the wrong length on rank 0, another `buffer`
      on rank 1, and a forest refined on rank 1, each refused on every
      rank, the forest untouched by the first three.
    - *In process*, `test/regrid_exchange_tests.jl`. The lockstep test
      builds the regrid stage of every simulated rank, packs from its
      slice of the old array, delivers every segment, runs the local
      groups and unpacks, and requires the result bit for bit equal to
      the serial `regrid!` on every rank's new blocks; the union of the
      local and received transfers equal to the serial transfer set;
      both ends' layouts entry for entry; and every sent transfer naming
      its source's old owner and its target's new one. It covers `D =
      1, 2, 3`, every centering, the periodic, outer and reflecting
      faces of `exchange_tests.jl`, PointValue and Conservative, 1–5
      ranks and one more than the leaves in 1D (empty ranks), and, in
      3D, 7 ranks, which is the first count at which a coarsened group
      of that mesh straddles a rank boundary; `Float32` and `Float32x2`
      over a 2D subset. Then `regrid!` and `adapt_to_initial_data!`
      themselves run with one task per rank over a communicator whose
      collectives are rendezvous between the tasks and whose messages
      go through step 2's mailbox: two field sets and a flux `=> nothing`
      with mixed bare and boxed flags and a buffer, in 2D and 3D at 3
      and 5 ranks, bitwise against serial, and `transfer = false`; both
      criterion forms from a single leaf at 3 ranks, with the serial
      passes and data; and the refusals.
    - *Measured.* The workload at `-n 2`, `3` and `4` with one thread a
      rank, and at `-n 3` with two, prints the serial lines byte for
      byte except the `sum` lines, which differ in the last one or two
      of 17 digits where they differ (`l2`; Burgers' mass 16 against
      15.999999999999998 and 16.000000000000004); 39–42 s of wall clock
      per run, against 27–29 s in step 3. Blocks moved up and down at
      every rank count (the uniform mesh: 3, 6 and 5 up on the refine,
      as many down on the coarsen, at 2, 3 and 4 ranks), and 1, 2 and
      2 coarsened blocks had children of more than one owner.
    - *Serial cost* (a scratch script, best of 15: `D = 3`, `N = 16`,
      the 4³-root mesh of `bench/threads.jl` with its middle refined, a
      field set of 2 variables, `G = 2`, order 4, and a face-centered
      flux `=> nothing`, refining a slab of 128 blocks and coarsening it
      back; HEAD before and after, alternated). At one thread the
      refine took 37.1–38.4 ms before and 37.2–38.3 ms after, the
      coarsening 23.2–23.7 against 23.4–23.7; at four threads the
      refine 12.6–13.2 ms against 12.7–14.3 and the coarsening 7.4–7.9
      against 7.5–7.9. On a host-heavier mesh (`N = 6`, 8³ roots, 1856
      leaves after the refine) 26.3–27.0 ms against 26.4–27.2 and
      28.2–28.3 against 28.0–30.7. A regrid allocates 0.1–0.8 % more
      (38.55 MB against 38.50, 34.86 against 34.57): the marks and the
      layout hash. `bench/ghosts.jl`'s fill is unchanged, 2.57–2.59 ms
      against 2.57–2.58 and 11.45–11.46 against 11.48, with 16 and 32
      bytes less allocated per fill.
    - *Julia 1.11.* In the manifest-free copy, Julia 1.11.9 passes
      `partition_tests.jl`, `exchange_tests.jl`,
      `regrid_exchange_tests.jl`, `regrid_tests.jl` and `mpi_tests.jl`
      (2m38 together); the whole suite was not run there.
    - *Suite cost.* 108827 tests at one thread in 6m57 (7m02 in a
      second run, timed by file) and 108879 at eight in 6m57: 451 more
      than step 3 at each — the 430 of `regrid_exchange_tests.jl`, 22
      more in `mpi_tests.jl`, and one fewer in `partition_tests.jl`,
      whose two regrid refusals became one. `mpi_tests.jl` now takes
      96.6 s inside the suite, against step 3's 58.4 s, and
      `regrid_exchange_tests.jl` 12.4 s: about 50 s more in all. The
      `Float32x2` and `Float32` workload cases cost about 2 s each per
      run and the three-field-set case the most compilation; those are
      what to trim first. The thread-independence digests are
      unchanged. The docs build.
  - **Step 5 — interpolation routing.** *Accept:* an interpolation line
    in the workload, bit-identical to serial; a rank that passes no
    points; and an outside point refused on every rank with the same
    reason. *(Amended in step 5: every rank refuses together, but a rank
    that passed an outside point names its own, and the others name the
    first rank's; see "Point interpolation" in
    [Distributed meshes](#distributed-meshes).)*

    *(Done, 2026-10-01.)* What it settled, and where it went beyond the
    plan (the design decisions are recorded under "Point interpolation"
    in [Distributed meshes](#distributed-meshes)):
    - *The driver.* `interpolate!` over a distributed forest goes to
      `interpolate_distributed!` in `interpolate.jl`: the serial
      argument checks, factored into `check_interpolation`, run inside
      one agreed gather together with the cheap forest check, a hash of
      the arguments that must agree and the rank's first outside point;
      then the host location, the route by a stable counting sort and
      `alltoallv`, the kernel over the received points through
      `launch_interpolation!` — the serial launch, factored out, with
      the block offset — and the return of the values, and of the flags
      when there is a region, into the caller's order. `interpolate`
      itself defers a refusal of `derivs` to `interpolate!` over a
      distributed forest, so that it too is agreed. The step-1 refusal
      is gone; `refuse_distributed` now serves only the checkpoint.
    - *In process*, `test/interpolate_exchange_tests.jl`, over
      `regrid_exchange_tests.jl`'s rendezvous communicator, which gains
      an `alltoallv`. Every simulated rank queries an uneven slice of
      one point list — rank 1 none — and must reproduce its part of the
      serial `interpolate` bit for bit: value and gradient of two
      variables in reverse order with an excluded ball, the value alone
      of every variable without a region, and `interpolate!` into
      caller-supplied outputs from points given as vectors. The points
      run beyond periodic dimensions and reflecting walls, sit on every
      leaf's lower corner (so on every rank boundary) and on the
      domain's corners, over random data in every stored point. The
      cases are `D = 1` with a reflecting wall at 2 and 3 ranks and
      periodic at `nleaves + 2` ranks (two of them without blocks);
      `D = 2` reflecting at both ends and periodic, vertex-centered, at
      3 ranks, outer and cell-centered at 5, and a face-centered
      `Float32` set at 3; `D = 3` with a reflecting, an outer and a
      periodic dimension at 4. The refusals at 3 ranks: an outside
      point on rank 1 (its message and the others'), outside points on
      ranks 0 and 2 with rank 1 passing none, `derivs` only rank 0's
      checks refuse, `vars` and `exclude` that differ on one rank, and
      a forest refined on one rank.
    - *The workload.* `test/mpi_workload.jl` gains a 2D case, periodic
      in x₁, reflecting below in x₂ and outer above, three levels and an
      odd variable, over pseudo-random data with the ghosts filled:
      every rank asks for its own uneven slice (rank 1 none) of 301
      points running half a period and half a domain beyond the faces,
      and rank 0 digests the gathered answers, which are then in
      global order — value and gradient of two variables with an
      excluded ellipse and its flags in `Float64`, the value alone in
      `Float64` and `Float32x2`. A `#` line refuses an outside point on
      rank 1, and `mpi_tests.jl` asserts it is refused on every rank
      with rank 0 naming rank 1's point.
    - *Measured.* The workload at `-n 2`, `3` and `4`, one thread a
      rank, prints the serial lines byte for byte except the `sum`
      lines, which differ in the last one or two of 17 digits where they
      differ, as in step 4; the five interpolation lines are
      byte-identical at every rank count. 44–47 s of wall clock per
      run, against 39–42 s in step 4, the serial run 38 s. On Metal, a
      scratch script found the device path bitwise equal to the serial
      Metal call (the device bullet under "Point interpolation" in
      [Distributed meshes](#distributed-meshes)); it is not in the suite.
    - *Serial cost* (`bench/interpolate.jl` with `N = 8`, 4³ roots and
      176 leaves, 20 variables, value and gradient with an excluded
      ball, best of 200 calls; HEAD before and after, alternated, four
      runs each). At one thread the 496-point batch took 0.353–0.368 ms
      before and 0.363–0.370 ms after, 4960 points 3.61–3.66 ms against
      3.64–3.71 ms; at four threads 0.109–0.114 ms against
      0.110–0.114 ms and 0.981–0.998 ms against 0.979–0.992 ms. That is
      within the run-to-run spread. A call allocates 32 bytes more at
      one thread (10320 against 10288) and 96 more at four (13232
      against 13136), the block offset in the kernel's arguments.
    - *Julia 1.11.* In the manifest-free copy, Julia 1.11.9 passes `interpolate_tests.jl`, `partition_tests.jl`,
      `exchange_tests.jl`, `regrid_exchange_tests.jl`,
      `interpolate_exchange_tests.jl` and `mpi_tests.jl` (3m06
      together); the whole suite was not run there.
    - *Suite cost.* 109038 tests at one thread in 7m49 and 109090 at
      eight in 8m09: 211 more than step 4 at each — the 204 of
      `interpolate_exchange_tests.jl` and 7 more in `mpi_tests.jl`;
      `partition_tests.jl`'s interpolation refusal became a refusal by
      verb, one test for one. The thread-independence digests are
      unchanged. The docs build. The step-4 tree, run the same day on
      the same machine, took 7m02 at one thread, so the suite grew by
      about 47 s; of that the new tests account for about 24 s when run
      standalone — `interpolate_exchange_tests.jl` 13.5 s, nearly all of
      it compilation, and `mpi_tests.jl` 125 s against the step-4 tree's
      114 s — and the rest was not separated from the run-to-run spread.
      The `Float32` and 3D cases of the new file are what to trim first.
  - **Step 6 — parallel checkpoints.** `TreeAMRHDF5MPIExt` and the hooks
    in `TreeAMRHDF5Ext`. *Accept:* save at `-n 3`, then load at `-n 2`,
    at `-n 1` and serially, each continuation byte-identical to the
    uninterrupted run; a serial file loaded at `-n 3`; a rank with no
    blocks; plain data that differ between ranks refused on every rank;
    `MPI_File_sync` confirmed under the collective flush; and
    `bench/checkpoint.jl` run under MPI on Symmetry, with the shared
    file's throughput recorded on its parallel file system — the
    benchmark "Parallel I/O and M7" asked for. The per-process files
    are revisited only if the shared file does not hold up.

    *(Done, 2026-10-01, except the Symmetry measurement, which ran on
    one node on 2026-10-02 and not yet on several; see the last items.)*
    What it settled, and where it went beyond the
    plan (the design decisions are recorded under "Parallel checkpoints"
    in [Distributed meshes](#distributed-meshes)):
    - *The code.* `ext/TreeAMRHDF5MPIExt.jl` (`[extensions]`
      `TreeAMRHDF5MPIExt = ["HDF5", "MPI"]`) opens the shared file;
      `TreeAMRHDF5Ext` gains the `Access` hooks (`open_file`,
      `write_whole`, `write_slab` / `read_slab`, `put_value`,
      `agree_values`, `flush_ranks`, `publish`), the agreement
      (`agreed`), the plain-data walk (`plain_hash`) and the
      fixed-length string arrays; `src/` gains the stubs `librarycomm`
      and `open_parallel_file`, `load_checkpoint`'s `comm` keyword, and
      the docstrings' collective contract. `refuse_distributed` is gone,
      its last caller with it.
    - *The serial file is unchanged but for `nranks`.* A scratch script
      wrote the same checkpoint — two field sets, `Float64` with parity
      and a `Float32x2` vertex set, a reflecting face, every kind of
      plain data and a do-block `write_plain` — with HEAD before the step
      and after it, unfiltered and with `Shuffle` + `Deflate(1)`; `h5dump`
      of the two files differs in `created` and the new `nranks = 1`
      only, and `h5dump -p` also in the storage offsets, which the 8-byte
      dataset shifts.
    - *The workload.* `test/mpi_workload.jl` gains `checkpoint_case`:
      the tracked pulse as a chunked driver (RK4 steps, the flags and a
      regrid per chunk), run uninterrupted for three chunks, then saved
      after the first — the bare `name => fs` form, plain data with a
      Rational and an array of strings including an empty one —
      unfiltered and with `Shuffle` + `Deflate(1)`, dropped, loaded at
      the same rank count and continued; the mesh refines again after
      the save (39 leaves, then 87). Its lines must be the serial
      run's, and the continued digests the uninterrupted run's. A 1D
      forest of two leaves is saved and loaded too, which at three ranks
      leaves one rank without blocks both times. `checkpoint_cross` then
      loads every file the earlier runs wrote at another rank count, at
      this one and, under MPI, a second time over `MPI.COMM_SELF`, on
      `#` lines. `mpi_tests.jl` runs the serial reference, `-n 3` and
      `-n 2` (in that order now) over one `TREEAMR_CHECKPOINT_DIR`, then
      the serial loads in process: so the serial file loads at 3 ranks
      and at one; the 3-rank files at 2, at one inside the 2-rank job
      (a one-rank MPI communicator — what a separate `mpiexec -n 1`
      would test, without a third launch) and serially; the 2-rank files
      serially. Every load reproduces the saved digests and every
      continuation the uninterrupted ones. The refusals, at every rank
      count: plain data that differ between ranks; an `application` only
      rank 1 refuses; filters that differ on rank 1; a `write_plain` in
      the do-block of a value that differs, after which the earlier file
      is intact and no partial file is left; a load whose `fieldsets`
      differ on rank 1; a load of a missing file, which rank 0 alone
      looks for. Each is raised on every rank.
    - *In process*: `checkpoint_tests.jl` passes unchanged but for two
      deliberate assertions, `nranks == 1` and a file without `nranks`
      reading as 1; `partition_tests.jl`'s save over the test-only
      communicator now stops at its missing `allgather`, by name, like
      the interpolation.
    - *Measured by hand.* The workload at `-n 2`, `3` and `4`, one thread
      a rank, prints the serial lines byte for byte except the `sum`
      lines, which agree to roundoff, as in step 5; every cross load at
      every count reproduces the digests. 53–57 s of wall clock per run,
      against 44–47 s in step 5, and the serial run 44 s, against 38;
      most of the difference is compiling HDF5.jl's paths in each rank.
      `mpi_tests.jl` alone takes 2m34, against step 5's 125 s.
    - *Throughput, locally only* (`bench/checkpoint.jl`, now runnable
      under `mpiexec` with the argument `mpi`; the development laptop of
      "Throughput and filters", its internal SSD, one thread a rank,
      MPICH_jll 5.0.2, the same day and the same in-use conditions).
      GB/s of state, aggregate over the ranks, the slowest rank's time;
      `-n 1` is the serial path over a one-rank communicator:

      | data | filter | serial | `-n 1` | `-n 2` | `-n 4` |
      |---|---|---|---|---|---|
      | blast, save | none | 7.49 | 7.50 | 6.66 | 6.90 |
      | blast, sync | none | 4.58 | 4.66 | 4.52 | 4.47 |
      | blast, load | none | 1.24 | 1.28 | 2.28 | 3.89 |
      | blast, save | `Shuffle` + `Deflate(1)` | 0.34 | 0.34 | 0.67 | 0.92 |
      | blast, load | `Shuffle` + `Deflate(1)` | 0.51 | 0.51 | 0.96 | 1.57 |
      | pulse, save | none | 5.66 | 7.46 | 6.51 | 4.55 |
      | pulse, load | none | 1.20 | 1.21 | 2.03 | 3.09 |
      | pulse, save | `Shuffle` + `Deflate(1)` | 0.11 | 0.11 | 0.21 | 0.38 |
      | pulse, save | bitshuffle + LZ4 | 0.62 | 0.62 | 1.12 | 1.80 |

      So on one SSD the unfiltered save stays at the page cache's rate
      and the flush at the drive's whatever the rank count, while what
      is per rank — compression, the load's allocation, first touch and
      `scatter!` — divides among the ranks: the filtered save 2.7 times
      faster at 4 ranks, the unfiltered load 3.1 times. This is the
      point "Compression is serial" made, that under M7 a filter
      parallelizes. File sizes are the serial ones (90.0 MB for blast
      with Deflate). H5Zzstd and H5Zlz4 were not in the environment, so
      their rows are missing; H5Zbitshuffle was, from the default
      environment. One run each; the M9a table varied by 15–60 % run to
      run on this machine. These are not the parallel file system's
      numbers, which are the point of the measurement.
    - *`MPI_File_sync`* is confirmed by reading libhdf5 2.2.0's and
      MPICH 5.0's sources (the chain is under "Parallel checkpoints"),
      not by tracing a run: the call is made, but no test here can see
      a write reach stable storage, and macOS's `fsync` does not.
    - *Symmetry, one node only* (measured 2026-10-02, job 567854 on
      cn106, AMD EPYC 7543, 64 cores, 8 NUMA domains; Julia 1.13.1,
      MPICH_jll 5.0.2 through `srun --mpi=pmi2`, HDF5_jll's MPICH build
      of libhdf5 2.2.0 with `HDF5.has_parallel()`; the files on BeeGFS
      under `/mnt/beegfs/eschnetter/claude`; `bench/symmetry_checkpoint_mpi.sh`
      as amended in the step's script commit, `TREEAMR_BENCH_REPS = 3`,
      `64 / P` threads a rank, ranks in blocks so that 8 are one per
      domain). GB/s of state, aggregate, the slowest rank's time, save
      (`sync = false`) / sync / load; the default mesh, 3296 blocks,
      216 MB of `pulse` and 540 MB of `blast`; `-n 1` is the serial
      path over a one-rank communicator:

      | data | filter | `-n 1` | `-n 2` | `-n 4` | `-n 8` |
      |---|---|---|---|---|---|
      | blast | none | 1.53 / 1.60 / 1.52 | 1.56 / 1.26 / 2.82 | 1.44 / 0.96 / 4.03 | 1.37 / 0.84 / 4.56 |
      | blast | `Shuffle` + `Deflate(1)` | 0.22 / 0.22 / 0.42 | 0.40 / 0.38 / 0.70 | 0.52 / 0.52 / 1.40 | 0.81 / 0.77 / 2.51 |
      | blast | shuffle + zstd(1) | 0.64 / 0.64 / 0.70 | 0.89 / 0.85 / 1.02 | 1.18 / 1.01 / 2.50 | 1.46 / 1.28 / 3.83 |
      | blast | bitshuffle + zstd(1) | 0.51 / 0.51 / 0.68 | 0.77 / 0.74 / 1.28 | 1.01 / 0.96 / 2.41 | 1.15 / 1.06 / 3.65 |
      | pulse | none | 1.26 / 1.34 / 1.20 | 1.35 / 1.11 / 2.31 | 1.34 / 0.90 / 2.36 | 1.15 / 0.77 / 3.27 |
      | pulse | `Shuffle` + `Deflate(1)` | 0.07 / 0.07 / 0.19 | 0.12 / 0.12 / 0.45 | 0.19 / 0.18 / 0.74 | 0.25 / 0.23 / 1.33 |
      | pulse | shuffle + zstd(1) | 0.30 / 0.30 / 0.42 | 0.35 / 0.34 / 1.05 | 0.42 / 0.36 / 1.79 | 0.42 / 0.34 / 2.43 |

      File sizes are the serial ones (blast 89.9 MB with Deflate, 87.9
      with zstd(1); pulse 161–164 MB). The mesh with 6³ → 12³ roots has
      12648 blocks (the refinement follows a surface, so 3.8 times the
      blocks, not 8; 829 MB of pulse and 2.07 GB of blast): at `-n 1`
      pulse unfiltered 1.25 / 1.48 / 1.35, blast zstd(1) 0.62 / 0.63 /
      0.69; at `-n 2` pulse unfiltered 1.48 / 1.32 / 2.55, blast
      unfiltered 1.48 / 1.13 / 2.96 — the default mesh's rates, so the
      table is not a small-file effect. What it shows:
      1. **An unfiltered save from one node is 1.2–1.6 GB/s at every
         rank count**, and the flush costs more with more ranks (blast
         sync 1.60 → 0.84 GB/s from 1 to 8), every rank's
         `MPI_File_sync` being a BeeGFS client round trip. One node's
         client, or the servers, set that rate; which, only the
         multi-node run can say.
      2. **A filter parallelizes**, as "Compression is serial"
         predicted: blast with zstd(1) saves at 1.46 GB/s at 8 ranks,
         2.3 times its one-rank rate and about the unfiltered rate,
         for a file 6.1 times smaller; Deflate 3.7 times faster. Pulse,
         whose data compress to 1.3 only, gains less (zstd(1) 1.4
         times).
      3. **The load scales** (blast unfiltered 1.52 → 4.56 GB/s), but it
         reads a file the node just wrote, so this is the client's
         cache and the per-rank allocation and `scatter!`, not BeeGFS's
         read rate.
      4. Collective metadata reads were not needed at these counts.

      **Not run: the multi-node part.** Job 567855 (4 nodes, 8 ranks a
      node, `TREEAMR_CKPT_ONENODE=0`, so 2 and 4 nodes) was queued
      behind 567854, and the tunnel to the cluster went down before
      either finished, so their logs were not copied; the table above is
      read from 567854's log while it ran, its 4- and 8-rank rows on the
      larger mesh and the shuffle + LZ4 and bitshuffle + LZ4 rows above
      one rank not included. Whether the shared file's aggregate rate
      grows with nodes past the one-node 1.5 GB/s — the question
      "Parallel I/O and M7" asked — therefore stays open, and with it
      whether the per-process files need revisiting.
    - *Julia 1.11.* In the manifest-free copy, Julia 1.11.9 passes
      `partition_tests.jl`, `checkpoint_tests.jl` and `mpi_tests.jl`
      (2m53 together), the parallel checkpoints and every cross load
      included; the whole suite was not run there.
    - *Suite cost.* 109107 tests at one thread in 8m10 and 109159 at
      eight in 8m02, against step 5's 109038 in 7m49 and 109090 in 8m09:
      69 more at each — 67 in `mpi_tests.jl` (the checkpoint lines, the
      cross loads at each rank count and in process, the refusals, and
      the extension's presence) and the two `nranks` assertions of
      `checkpoint_tests.jl`. `mpi_tests.jl` takes about 28 s more than in
      step 5 standalone (2m34 against 125 s), which is the HDF5 code
      compiled in each `mpiexec` run and the in-process loads; at one
      thread the suite grew by about 20 s. The thread-independence
      digests are unchanged. The docs build.
  - **Step 7 — weak-scaling smoke test.** `bench/mpi.jl` holds the
    blocks per rank fixed and times the RHS, the ghost fill, a regrid
    and the norm; `bench/symmetry_mpi.sh` runs one rank per NUMA domain
    at eight threads, on one, two and four nodes, with the system MPI
    through MPIPreferences. *Accept:* the table recorded here, with
    what limits it named, the replicated regrid bookkeeping and the
    overlap of the exchange with the local groups among them.

    *(Done, 2026-10-02: built and run locally, then on Symmetry at up to
    four nodes, 32 ranks; the table and what limits it are the
    "Symmetry" items below.)* What it settled:
    - *The benchmark.* `bench/mpi.jl` (under `mpiexec` with the
      argument `mpi`, in the test environment) builds a mesh of `TILES`
      identical tiles of `ROOTS^D` roots stacked along `x_D`, periodic,
      `TILES` the rank count unless set. The roots are numbered with
      `x_D` slowest and every tile has the same leaves, so the
      equal-count split gives rank `r` exactly tile `r`: the blocks per
      rank are fixed by construction, and the halo per rank is constant
      from three ranks on. Two tiles: *two-level*, the leaves with `x₁`
      in the lower half and `x_D` in the upper half of the tile refined
      once (176 blocks at `ROOTS = 4` in 3D), so a coarse-fine face
      crosses every rank boundary and phase 1, the prolongation stage
      and the interface restriction all carry messages; and *uniform*
      (64 blocks), copies only. Timed in synchronized windows — an
      `MPI.Barrier`, the call, the slowest rank's time by `allgather` —
      as the minimum and the median over the repetitions: the wave RHS
      (`scatter!` → `fill_ghosts!` → `map_blocks!`), the fill alone and
      its three parts run separately through the internals (the stages'
      local groups, the packs, the unpacks), `scatter!`, the norm, a
      `mesh_mapreduce` max, `restrict_interfaces!` on a set
      face-centered along `x_D`, `interpolate!` of 1000 points a rank
      spread over the whole domain, the two schedule builds, a regrid
      that refines the lowest layer of roots of tile 0 — the start of
      the curve, so every later rank's range shifts and blocks migrate
      — and one that coarsens it back, and the triad. `#` lines give per
      rank, as min/mean/max, the messages, bytes, transfers and peers of
      a fill and of an interface restriction (read from the stages),
      the blocks a regrid made a rank take from another, and ns per cell
      of the whole job. `TREEAMR_BENCH_BACKEND` and
      `TREEAMR_BENCH_DEVICEAWARE` choose a device and its message path,
      `TREEAMR_BENCH_TILES` the mesh of a `P`-rank run for a serial
      control, `TREEAMR_BENCH_LABEL` the column name. The tab-separated
      lines carry the minimum *and* the median, so `bench/scan.sh`'s
      awk does not read them; `bench/mpitable.awk` is the matching
      parser (efficiency against the first run, then both times) and
      `bench/mpiscan.sh P…` launches each count through
      `MPI.mpiexec()` and prints the table.
    - *Laptop smoke run, not a scaling result* (Apple M3 Pro, 6
      performance and 6 efficiency cores, one memory system; MPIABI_jll
      through the global preference; `D = 3`, `N = 16`, `ROOTS = 4`, 2
      variables, `G = 2`, order 4, Float64, 10 windows; `bench/mpiscan.sh
      1 2 4`, one thread a rank). Minimum ms, two-level mesh, with the
      weak-scaling efficiency `t(1)/t(P)`:

      | phase | 1 rank | 2 ranks | 4 ranks | eff. 2 | eff. 4 |
      |---|---|---|---|---|---|
      | rhs | 16.65 | 20.12 | 22.56 | 0.83 | 0.74 |
      | fill_ghosts | 10.22 | 14.00 | 16.38 | 0.73 | 0.62 |
      | — its local groups | 9.91 | 11.43 | 14.16 | | |
      | — its packs | – | 1.63 | 1.68 | | |
      | — its unpacks | – | 0.29 | 0.30 | | |
      | scatter | 2.56 | 2.61 | 2.69 | 0.98 | 0.95 |
      | norm | 6.15 | 6.25 | 6.46 | 0.98 | 0.95 |
      | interfaces | 0.025 | 0.046 | 0.073 | | |
      | interpolate (1000 pts/rank) | 0.148 | 0.272 | 0.396 | 0.54 | 0.37 |
      | ghost_schedule | 3.82 | 6.30 | 6.57 | 0.61 | 0.58 |
      | interface_schedule | 0.13 | 1.23 | 1.35 | | |
      | regrid, refine | 46.0 | 57.4 | 65.5 | 0.80 | 0.70 |
      | regrid, coarsen | 30.4 | 45.5 | 44.7 | 0.67 | 0.68 |
      | triad reference | 0.61 | 1.22 | 2.20 | 0.50 | 0.28 |

      A fill sends, per rank, 2 messages at 2 ranks and 3 at 4 (phase 1
      to both neighbours, the prolongation stage to one), 717 KB each way
      in both cases, 560 transfers sent and 560 received against 4224
      local (4784 serially); the interface restriction one message of
      32 KB, 32 transfers. The uniform mesh: one stage, 410 KB, 288
      transfers; its rhs 3.26 / 4.25 / 4.56 ms, fill 1.19 / 1.70 / 2.51,
      build 0.40 / 1.20 / 1.26, refine 32.2 / 39.1 / 46.1. The refine
      makes ranks take 0/28/56 blocks from another (min/mean/max) at 2
      ranks and 0/42/84 at 4, the coarsening as many. At two threads a
      rank the two-level rhs is 8.72 / 11.53 / 22.47 ms (efficiency 0.76
      and 0.39) — eight threads on this machine reach its efficiency
      cores, and the window is the slowest rank's.
    - *What limits it here, named.* (1) **Memory bandwidth is shared**:
      the triad's aggregate is about 120 GB/s at any rank count, one
      core nearly saturates it, and so the per-rank triad halves with
      each doubling. Whatever streams memory cannot weak-scale on this
      machine, and the local groups of the fill do not (9.9 → 14.2 ms
      for *fewer* transfers); only the phases bound by one core's
      arithmetic do (`scatter`, the norm, 0.95). That is the reason the
      numbers are not a scaling result. (2) **The sender computes**: the
      packs are 1.6–1.7 ms, about a tenth of the fill. They are the
      halo's copies, restrictions and prolongations, evaluated into the
      send buffer rather than into ghosts, and the unpacks (0.3 ms) are
      the extra pass that copies them into place. (3) **The overlap**: the fill less its local groups, packs
      and unpacks is 0.7 ms at 2 ranks and 0.2 ms at 4, so over shared
      memory almost nothing of the messages is left unhidden behind the
      local groups; between nodes is what Symmetry will show. (4)
      **Per-call collectives on small work**: an interpolation of 1000
      points a rank costs an `allgather` and four `alltoallv`, 0.40 ms
      at 4 ranks against 0.06 ms in one 4-thread process over the same
      4-tile mesh (`TREEAMR_BENCH_TILES=4`); there the rhs and fill were
      within 5 % of the 4-rank run (22.1 against 22.7 ms, 15.9 against
      16.8), the schedule build 3.9 against 6.7 ms, the interface
      schedule 0.14 against 1.35 ms and the refine 48.8 against 65.4 ms.
      The interface schedule's growth is `remote_neighbors`, which
      searches all `3^D − 1` directions of every local leaf (about 1 ms
      here) for a builder that needs only faces, and which each
      schedule build repeats.
    - *Does the schedule build stay flat with `P`?* Yes. Step 2's
      finding — a rank's build costs about the serial one at 120 blocks
      a rank — was strong scaling: the halo is as large as the local
      part, so the build does not shrink with `P`. At a *fixed* number
      of blocks per rank it does not grow either. `bench/replicated.jl`
      simulates rank `P÷2` of `P` in one process, over a communicator
      that answers the digest gather by replication, hands `regrid!` the
      global marks and makes every message a no-op, over this tile mesh
      (176 blocks a rank, `N = 16`); best of 3–5, ms, at one thread and
      at four:

      | leaves (`P`) | 1408 (8) | 11264 (64) | 90112 (512) | 180224 (1024) |
      |---|---|---|---|---|
      | digest, fold over every leaf | 0.024 | 0.15 | 1.14 | 2.27 |
      | `GhostSchedule` on the rank, 1 thread | 6.3 | 6.7 | 8.1 | 9.1 |
      | the same, 4 threads | 4.3 | 4.4 | 5.6 | 6.8 |
      | `remote_neighbors`, 1 thread | 1.04 | 1.19 | 1.29 | 1.34 |
      | `complete_marks`, the slab, no buffer | 0.025 | 0.13 | 1.01 | 1.90 |
      | `balance!` alone | 0.005 | 0.033 | 0.26 | 0.52 |
      | `regrid_sources`, 1 thread | 0.17 | 1.27 | 12.1 | 29.0 |
      | the same, 4 threads | 0.17 | 1.01 | 8.7 | 19.4 |
      | `split_regrid` | 0.015 | 0.026 | 0.11 | 0.21 |
      | `complete_marks`, every block a `(Keep, box)` source, buffer 4, 1 thread | 8.2 | 74 | 672 | 1407 |
      | the same, 4 threads | 2.3 | 21 | 204 | 436 |
      | `regrid!` on the rank, refine, 1 thread | 21.7 | 21.9 | 51.7 | 67.2 |
      | `regrid!` on the rank, refine, 4 threads | 11.4 | 12.7 | 17.7 | 44.0 |
      | marks gathered per rank and regrid | 0.05 MB | 0.36 MB | 2.9 MB | 5.8 MB |

      Over 128 times the leaves the rank's build grows by 2.8 ms at one
      thread, 2.2 ms of it the digest, the rest the `log n` of the
      searches; so it is weak-scaling-flat, and the digest is the only
      `O(nleaves)` part of it (13 ns a leaf, single-threaded, folded up
      to three times per generation: `regrid!`, each `GhostSchedule`,
      each `InterfaceSchedule`). The `regrid!` times are noisy, the
      garbage collector's share included; a profile of one rank at 90112
      leaves (one thread, 20 regrids, 29 ms each) put 38 % in
      `regrid_sources`, 25 % in the fill before the transfer, 22 % in
      allocating and zeroing the new array (local, first touch), 7 % in
      `complete_marks` and
      7 % in the transfer stage.
    - *The replicated regrid bookkeeping, and what to do about it*
      (proposed here, and the first two implemented the same day; what
      was done, and measured, is the next item). Two passes become the
      regrid's cost long before the forest's memory does:
      1. **The buffer's recruit search** in `buffered_flags`, about
         5.8–7.8 µs per source leaf at one thread and 1.6–2.4 at four,
         over every source of the whole forest on every rank. With
         every block a source it overtakes the whole local regrid
         between 1408 and 11264 leaves (8.2 and 74 ms against 22) and
         is 13 times it at 90112 (672 against 52); at 32 ranks
         of 176 blocks (four Symmetry nodes) it would be some 35–40 ms
         at one thread, by the 1408- and 11264-leaf points, against a
         local regrid of 11–22 ms — an extrapolation, not measured
         there. The share of sources is the share of blocks whose
         criterion fired (`firing_boxes`), so a run with a wide feature
         pays it. *Remedy:* the recruits of a source are found from the
         tree alone, so each rank can search for its own sources only
         and gather the `(leaf, level)` pairs beside the marks; applying
         them only raises a mark to `Refine` below a level or lifts a
         `Coarsen` at or below one, which does not depend on their
         order, so the result is bit for bit the replicated one, at
         `O(local sources)` per rank and one `allgatherv` of the pairs.
      2. **`regrid_sources`** classifies every new leaf on every rank,
         with a `Dict` of every old leaf and a vector per new one: 12 ms
         at 90112 leaves, 29 ms at 180224, 38 % of a rank's regrid at
         512 ranks. *Remedy:* a rank needs only the new leaves it will
         own and those whose sources it owns now. Refinement and
         coarsening keep the curve order, so the second set is one
         contiguous range of new indices, found by two binary searches
         over the new leaves at the keys of the rank's first and last
         old leaf, and an old leaf is found by `searchsortedfirst`
         instead of the `Dict`: `O(local · log n)`. The layouts are
         sorted explicitly (`stage_layout`), so restricting the list
         changes no message.
      3. Smaller and left as they are: the digest (cache its leaf fold
         per generation if it ever shows; 1.1 ms at 90112 leaves is
         14 % of a build), `complete_marks`' own passes and `balance!`
         (1–2 ms at 10⁵ leaves, replicated in AMReX and Parthenon too),
         and the marks' `allgatherv`, 32 bytes a leaf to every rank,
         which a gather of the marks that are not a bare `Keep` would
         shrink when it matters.
    - *What was done about it* (2026-10-02, after the measurement; the
      results are unchanged, bit for bit):
      1. **The recruits.** `buffered_flags` is now two passes, both
         internal: `buffer_recruits`, the neighbour search over a run of
         sources, which returns `Recruit(leaf, level)` pairs in source
         order, and `apply_recruits!`, the serial rewrite of the marks.
         The public function runs both over every leaf, as before.
         `regrid!` runs `regrid_marks` instead: each rank searches from
         its own sources, reduces its recruits with
         `strongest_recruits`, gathers them in a second `allgatherv`,
         and applies the union to the gathered marks; then the
         completion, which is `complete_marks` after the buffer
         (`completed_leaves`). Serially nothing is gathered and the
         recruits are applied as found, which is `buffered_flags`.
         *Why the order does not matter*, checked against the code
         rather than assumed: the rewrite is two statements per recruit
         `(j, L)` at a leaf of level `l` — a `Coarsen` with `l ≤ L`
         becomes `Keep`, then anything with `l < L` becomes `Refine`.
         `Refine` is never undone, and a `Keep` made from `Coarsen` is
         only ever raised to `Refine`. So a leaf ends as `Refine` if any
         recruit asks it for more than its level, as `Keep` if it was
         `Coarsen` and some recruit asks for its level exactly, and as
         it was otherwise: a function of its own reported mark and of
         the largest level asked of it, whatever the order of the
         recruits and however often one repeats. Recruitment reads the
         *reported* marks, never the rewritten ones, so the search
         itself does not depend on the order either. That is also what
         lets a rank keep, per leaf, only its largest request, and drop
         a request at a leaf already finer, which changes no mark; so
         gathering in global source order, the fallback had the rewrite
         been order-dependent, was not needed. *Why a second gather and
         not the marks' own*: merging the two would need a
         byte-level encoding of a vector of padded structs with its
         per-rank counts, to save one collective whose cost is below
         the marks' gather. Measured over MPI on this laptop (one tile
         a rank, one thread, every block a `(Keep, box)` source, buffer
         4), at 2, 4 and 8 ranks: the recruits' `allgatherv` 0.015,
         0.034 and 0.078 ms with 1600 bytes from a rank, the marks'
         0.021, 0.074 and 0.188 ms with 5632; the whole `regrid_marks`
         0.97, 1.12 and 1.93 ms against the replicated `buffered_flags`
         1.88, 4.11 and 11.8 ms, with the same leaves at every count.
      2. **`regrid_sources`** takes `oldrange` and `newrange` and
         classifies `newrange` and `overlapping_leaves(oldrange)`: the
         new leaves from the one that covers the start of the rank's
         first old leaf (that leaf or an ancestor, at or before it in the
         curve's pre-order, else its first descendant, directly after
         it) through the last one at or before the deepest last
         descendant of its last old leaf. Without the ranges it
         classifies every leaf, as before, which is what the serial
         `regrid!`, `transfer_groups` and the tests use. A plain
         `searchsortedfirst` in place of the `Dict` was *slower* for the
         full classification — 14.1 against 11.6 ms at 90112 leaves,
         30.9 against 29.6 at 180224 — so the serial regrid would have
         paid for the distributed one. The search therefore starts at the
         previous target's sources (`findfrom`: four steps along the
         curve, then a binary search over the rest), since the sources
         of ascending new leaves are non-decreasing old leaves; that
         made the full classification faster than the `Dict`. The
         in-process lockstep test now builds every rank's stage from its
         own classification and checks that `split_regrid` keeps the
         same transfers from it as from the full one.
      3. **The digest's fold is not cached.** A cache per forest and
         generation goes stale whenever the leaves change without
         `rebuild_leaves!`, and nothing prevents that: `forest.leaves`
         is a public `Vector`, and `test/regrid_tests.jl` itself
         empties and refills one at generation 0. The digest exists to
         catch a forest that changed outside the collective contract,
         so it should not trust the generation to say the leaves are
         unchanged. Where to keep the cache is a second problem:
         `Forest` is immutable, and its positional constructor is called
         from `complete_marks`. The saving would be at most 1.1 ms a
         build at 90112 leaves.

      `bench/replicated.jl` (rank `P÷2`, 176 blocks a rank, ms, best of
      3–5) now also times the rank's own buffer and classification.
      Before and after on the same machine, run back to back, a stale
      copy of the tree for the "before":

      | leaves (`P`) | 1408 (8) | 11264 (64) | 90112 (512) | 180224 (1024) |
      |---|---|---|---|---|
      | every block a source, buffer 4: replicated `complete_marks`, 1 thread | 7.6 | 71 | 650 | 1365 |
      | on the rank: `regrid_marks` / with the completion, 1 thread | 1.0 / 1.1 | 1.1 / 1.6 | 1.4 / 5.4 | 1.6 / 9.6 |
      | replicated, 4 threads | 2.3 | 21.7 | 212 | 422 |
      | on the rank, 4 threads | 0.45 / 0.52 | 0.53 / 0.93 | 0.64 / 3.8 | 0.78 / 7.5 |
      | `regrid_sources`, every leaf, `Dict` (before), 1 thread | 0.165 | 1.26 | 11.6 | 29.6 |
      | the same by `findfrom` | 0.134 | 0.99 | 8.1 | 18.1 |
      | the rank's own (what `regrid!` runs) | 0.021 | 0.019 | 0.017 | 0.016 |
      | `regrid!` on the rank, refine / coarsen, 1 thread, before | 21.6 / 22.7 | 22.4 / 23.0 | 33.6 / 42.6 | 73.1 / 68.1 |
      | after | 21.9 / 22.9 | 22.7 / 22.8 | 20.7 / 21.0 | 23.2 / 23.6 |
      | the same, 4 threads, before | 11.1 / 11.0 | 11.4 / 11.7 | 22.5 / 17.4 | 35.7 / 41.4 |
      | after | 11.3 / 9.9 | 9.8 / 11.4 | 17.0 / 9.4 | 15.4 / 11.4 |

      So a rank's regrid no longer grows with the forest here: at
      180224 leaves it costs what it costs at 1408, within the noise
      the four-thread column shows. What remains replicated is the
      completion. It is the 4–8 ms above at 90112–180224 leaves when
      every block is a source, since these marks then refine every
      coarse leaf beside a fine one and `balance!` has a level 2 to
      check; with the slab's marks it is the 0.9–2.0 ms of item 3. A
      recruit gathers as 8 bytes, 0.82 MB a rank at 90112 leaves in
      that case, against 2.9 MB of marks.

      *Tests.* A new in-process testset in
      `test/regrid_exchange_tests.jl` runs `regrid_marks`, the
      completion and `regrid!` itself (one task per simulated rank, over
      the rendezvous communicator) on random forests with random bare and
      boxed flags, `D = 1, 2, 3`, buffer 0–4 and 1–7 ranks, a roots-only
      forest among them so that most ranks are empty, and requires every
      rank's marks and leaves to be the serial `buffered_flags` and
      `complete_marks` ones. Each of four deliberate breakages fails it
      or the lockstep test: skipping the gather, dropping a recruit at a
      leaf of the requested level, and moving either end of
      `overlapping_leaves`. The suite passes at one thread (110511
      tests, 8m08) and at eight (110563, 7m57), and the MPI workload at
      2, 3 and 4 ranks prints the serial lines byte for byte, apart from
      the 20 `sum` lines, which agree to roundoff (11, 11 and 9 of them
      byte for byte).
    - *A device.* `TREEAMR_BENCH_BACKEND=metal TREEAMR_BENCH_N=8
      TREEAMR_BENCH_REPS=3 TREEAMR_BENCH_PROJECT=<env> bench/mpiscan.sh
      1 2` (a scratch environment that develops this checkout and adds
      Metal, MPI and KernelAbstractions) runs in Float32 through the
      staging path at 2 ranks, every phase included; its numbers, two
      ranks sharing one GPU, say only that the path works.
    - *Symmetry* (measured 2026-10-02, job 567847 on cn092–cn095: AMD
      EPYC 7532, the Rome nodes of `amdq`, 64 cores and 8 NUMA domains
      each, ConnectX-6 HDR100 InfiniBand; Julia 1.13.1; HPC-X 2.20's Open
      MPI 4.1.7 over UCX 1.17, `nvhpc-hpcx-cuda12/24.9`, through
      MPIPreferences' system binary). `bench/symmetry_mpi.sh` as amended
      in the same step's script commit: one rank per NUMA domain bound by
      `numactl`, 8 threads, `D = 3`, `ROOTS = 4`, 2 variables, Float64,
      10 windows, the minimum over them. Launching took four fixes to
      the script and none to the package: `amddebugq` takes one node a
      job, so the job ran on `amdq`; SLURM 21.08 ignores
      `--ntasks-per-node` in a step unless `--ntasks` is given;
      MPICH_jll starts under `srun --mpi=pmi2` but runs TCP over IPoIB,
      2.1 GB/s and 40 µs ping-pong between nodes (job 567845), while
      HPC-X runs 12.0 GB/s and 2.2 µs (job 567846) and is not built with
      SLURM's PMI, so it is launched by its own `mpiexec`; and Open MPI
      hands each rank a pseudo-terminal, into which the juliaup launcher
      writes a terminal title, so the ranks run the julia binary itself.
      Minimum ms, two-level mesh (176 blocks a rank), one node at 1–8
      ranks, then 2 and 4 nodes at 8 a node; efficiency `t(1)/t(P)`:

      | `N = 16` | 1 | 2 | 4 | 8 | 16 | 32 | eff. 32 |
      |---|---|---|---|---|---|---|---|
      | rhs | 7.75 | 11.98 | 12.99 | 12.78 | 13.59 | 14.41 | 0.54 |
      | fill_ghosts | 4.37 | 8.40 | 8.86 | 8.87 | 8.93 | 9.64 | 0.45 |
      | — its local groups | 4.33 | 5.18 | 3.90 | 5.29 | 5.31 | 5.55 | |
      | — its packs | – | 2.73 | 2.97 | 2.98 | 2.95 | 2.93 | |
      | — its unpacks | – | 0.51 | 0.78 | 0.80 | 0.79 | 0.80 | |
      | scatter | 1.84 | 1.84 | 1.84 | 1.95 | 1.91 | 1.91 | 0.97 |
      | norm | 2.96 | 2.09 | 2.09 | 2.08 | 3.34 | 3.00 | 0.99 |
      | interfaces | 0.13 | 0.70 | 0.62 | 0.74 | 0.72 | 0.52 | |
      | interpolate (1000 pts/rank) | 0.24 | 0.77 | 0.83 | 0.61 | 0.69 | 10.24 | 0.02 |
      | ghost_schedule | 8.38 | 11.79 | 11.65 | 12.10 | 11.70 | 11.73 | 0.71 |
      | interface_schedule | 0.33 | 1.54 | 1.23 | 1.63 | 1.36 | 1.36 | |
      | regrid, refine | 25.6 | 28.9 | 26.4 | 27.0 | 24.9 | 32.5 | 0.79 |
      | regrid, coarsen | 21.9 | 28.9 | 33.9 | 30.5 | 33.7 | 38.8 | 0.56 |
      | triad reference | 1.70 | 1.74 | 2.13 | 2.18 | 2.26 | 2.15 | 0.79 |

      | `N = 32` | 1 | 2 | 4 | 8 | 16 | 32 | eff. 32 |
      |---|---|---|---|---|---|---|---|
      | rhs | 37.9 | 47.7 | 49.4 | 58.2 | 55.0 | 55.9 | 0.68 |
      | fill_ghosts | 18.7 | 29.2 | 28.6 | 27.7 | 29.8 | 30.1 | 0.62 |
      | — its local groups | 18.7 | 17.0 | 17.0 | 17.0 | 16.9 | 17.1 | |
      | — its packs | – | 8.58 | 8.58 | 8.55 | 8.96 | 9.07 | |
      | — its unpacks | – | 2.04 | 1.73 | 1.77 | 1.99 | 2.09 | |
      | scatter | 8.02 | 8.69 | 8.71 | 9.38 | 9.96 | 11.98 | 0.67 |
      | norm | 15.5 | 15.2 | 15.9 | 16.2 | 16.0 | 16.9 | 0.92 |
      | interpolate | 0.19 | 0.78 | 0.94 | 0.63 | 0.86 | 8.28 | 0.02 |
      | ghost_schedule | 13.2 | 16.0 | 16.4 | 16.2 | 16.2 | 16.3 | 0.81 |
      | regrid, refine | 162 | 174 | 192 | 163 | 154 | 162 | 1.00 |
      | regrid, coarsen | 134 | 172 | 224 | 225 | 217 | 217 | 0.62 |
      | triad reference | 13.3 | 13.6 | 13.8 | 14.5 | 14.4 | 13.8 | 0.96 |

      A fill sends, per rank, 2 messages at 2 ranks and 3 from 4 on
      (716800 bytes each way at `N = 16`, 2.3 MB at 32; 560 transfers
      sent and 560 received against 4224 local); the regrid makes ranks
      take 0/50/108 blocks from another (min/mean/max) at 32 ranks. The
      uniform mesh (64 blocks a rank, one stage): rhs 2.05 / 3.75 / 2.62 /
      3.92 / 3.85 / 3.83 ms at `N = 16` (efficiency 0.53 at 32) and 9.84
      / 13.4 / 15.3 / 16.3 / 16.2 / 16.9 at 32 (0.58). What it shows:
      1. **The step from one rank to two is the whole loss, and the
         network is not in it.** From 2 to 32 ranks, across one, two and
         four nodes, the two-level rhs grows by 20 % at `N = 16` (12.0 →
         14.4 ms) and 17 % at 32 (47.7 → 55.9): efficiency 0.83 and 0.85
         against 2 ranks. The fill less its local groups, packs and
         unpacks — what the messages left unhidden — is −0.2 to 0.4 ms
         at `N = 16` and 0.4 to 2.0 ms at 32 at every count, inside the
         error of subtracting separately measured minima; 2.3 MB over
         UCX at 12 GB/s is 0.2 ms. The overlap does its job, and
         between nodes too.
      2. **The packs are the cost**: 2.7–3.0 ms of an 8.9 ms fill at
         `N = 16`, 8.6–9.1 of 30 at 32, against local groups of 5.3 and
         17.0 ms for 7.5 times as many transfers. So a packed transfer
         costs about 4 times a local one. The likely reason, not
         measured: pack groups run by the owner of the *source* block
         ("Ownership" under "Pack and unpack are transfers"), and the
         sources of a rank's halo are the bottom and top layers of its
         tile, which sit at the two ends of its curve range and so in
         the first and last thread chunks: at 8 threads the packs for
         the rank below run on one thread, those for the rank above on
         about four. The sender-computes rule moves the halo's
         arithmetic to the sender, as designed; what costs is that the
         block-ownership partition then gives it to a few threads. A
         pack split over all threads by transfer, not by owner, would
         trade affinity for balance, which is a design question.
      3. **The schedule build and the regrid are flat from 2 ranks on**,
         as `bench/replicated.jl` predicted: the build 11.7–12.1 ms at
         `N = 16` and 16.0–16.4 at 32 at every count from 2 to 32; the
         refining regrid 25–33 ms and 154–192 ms. The coarsening regrid
         rises from 1 to 4 ranks, by 55 % at `N = 16` and 67 % at 32,
         and is flat after at `N = 32`; at 16 it rises again at 32 ranks
         (not traced).
      4. **`interpolate!` at 32 ranks jumps to 8–10 ms**, from 0.6–0.9
         ms at 2–16 ranks; at 16 ranks (two nodes) its *median* is
         already 9.9 ms. Nothing else in the table does this, and
         MPICH_jll over TCP at 16 ranks takes 2.0 ms (below), so it is
         the collectives — an `allgather` and four `alltoallv` — inside
         this Open MPI at that scale, not the work. A job that repeats
         the 16- and 32-rank runs with HPC-X's HCOLL collectives on and
         off (567858) was queued and its result not collected; the cause
         is open.
      5. **Phases without messages slow by the slowest of 32 ranks**:
         `scatter` 8.0 → 12.0 ms at `N = 32` and the uniform mesh's
         local groups 2.5 → 4.3, while the triad stays at 0.96. The
         window is the slowest rank's.
    - *One node, the same mesh: 8 ranks against one process* (the same
      job; the control is 64 threads over the 8-tile mesh in one
      process). The one process wins the per-evaluation phases. Minimum
      ms, two-level, 8 ranks / one process pinned with first touch
      (`JULIA_EXCLUSIVE=1`) / one process interleaved:

      | | `N = 16` | `N = 32` |
      |---|---|---|
      | rhs | 12.78 / 7.98 / 11.97 | 58.2 / 39.4 / 50.5 |
      | fill_ghosts | 8.87 / 4.78 / 7.82 | 27.7 / 19.8 / 28.2 |
      | ghost_schedule | 12.1 / 16.7 / 17.1 | 16.2 / 23.0 / 19.9 |
      | regrid, refine | 27.0 / 48.1 / 44.3 | 163 / 233 / 245 |
      | regrid, coarsen | 30.5 / 48.9 / 41.4 | 225 / 221 / 218 |

      One pinned process runs the rhs 1.6 times as fast at `N = 16` and
      1.5 times at 32 (the uniform mesh 1.3 and 1.6), because the 8-rank
      run pays the packs and unpacks of item 2 and the one process pays
      nothing for the same halo: its local groups are the 8 ranks' local
      groups plus the halo, at about the same time (4.95 against 5.29
      ms at `N = 16`). That settles what "What one process loses" left
      for this step: the ownership policy recovers inside one process
      everything one rank per domain was expected to, and on one node
      ranks are a loss for the rhs. The replicated host passes go the
      other way — the schedule build and the refining regrid are
      1.4–1.8 times faster over 8 ranks, being per rank `O(local)` work
      on fewer threads — but they run at regrid frequency. So the
      layout to recommend is one process per node, its threads pinned,
      and ranks between nodes.
    - *The default binary between nodes* (job 567857, cn109–cn110, the
      Milan EPYC 7543 nodes, MPICH_jll 5.0.2 through `srun --mpi=pmi2`,
      `N = 16`; compare within the job only). On one node it matches
      HPC-X's shape (two-level rhs 6.32 / 11.83 / 12.31 / 13.66 ms at
      1–8 ranks); on two nodes the rhs is 18.52 ms and the fill 13.42,
      against 13.66 and 9.48 at 8 ranks, so TCP over IPoIB adds 4 ms a
      fill that UCX does not (efficiency 0.34 at 16 ranks, against 0.57
      for HPC-X at 16 in job 567847). Between nodes the system MPI is
      not optional.
    - *The replicated costs on a rank's domain* (`bench/replicated.jl`
      in job 567847, one domain of cn092, 8 threads, 176 blocks a rank;
      ms):

      | leaves (`P`) | 1408 (8) | 11264 (64) | 90112 (512) | 180224 (1024) |
      |---|---|---|---|---|
      | digest | 0.036 | 0.19 | 1.40 | 2.79 |
      | `GhostSchedule` on the rank | 11.9 | 11.7 | 13.0 | 14.3 |
      | `regrid_sources`, every leaf / the rank's own | 0.40 / 0.17 | 1.94 / 0.18 | 15.8 / 0.16 | 32.3 / 0.17 |
      | every block a source: replicated `complete_marks` | 3.6 | 33 | 290 | 596 |
      | on the rank: `regrid_marks` / the completion | 0.78 / 1.04 | 0.84 / 2.15 | 0.89 / 12.2 | 1.26 / 20.4 |
      | `regrid!` on the rank, refine / coarsen | 23.1 / 18.5 | 19.2 / 22.8 | 19.5 / 19.4 | 22.4 / 23.9 |

      So the laptop's finding holds on the cluster: a rank's regrid
      costs at 180224 leaves what it costs at 1408. What remains
      replicated and grows is the completion after the buffer, 20 ms at
      180224 leaves when every block is a source, about a regrid's
      worth; the build grows by 2.4 ms over 128 times the leaves, the
      digest most of it.
    - *What was not done.* The benchmark itself changed nothing in
      `src/`, so the suite was not rerun for it; the follow-up above
      did, and it was.
      The fill's overlap is read from the difference of its parts, not
      traced inside `run_stage!`. The benchmark's mesh is periodic, so
      the boundary hook is not timed.
  - **Step 8 — MPI+GPU.** Device-resident buffers, the device-aware
    check and host staging. *Accept:* the workload on Metal through the
    staging path, and on an H200 on Symmetry, agreeing with the serial
    device run as [Parallelism](#parallelism) states for a device.

    *(Done locally, 2026-10-01, and on Symmetry's H200s, both paths, on
    2026-10-02.)*
    What it settled, and where it went beyond the plan (the design
    decisions are recorded under "MPI+GPU" in
    [Distributed meshes](#distributed-meshes)):
    - *The code.* `run_stage!` in `ghosts.jl` is the one place every
      stage runs — the ghost fill, the interface restriction and the
      regrid transfer — so it is the one place that stages: it asks the
      new verb `hoststaging(comm, buffer)`, and when the answer is yes it
      posts the receives into host mirrors, downloads the packed buffer
      after the pack's synchronization, sends from the mirror, and
      uploads the receive mirror before the unpack. `stagemirrors`
      allocates the mirrors once per stage and variable count, kept in
      `RemoteStage`'s new `mirrors` field, and page-locks them with
      `KernelAbstractions.pagelock!`. The MPI extension gains the
      `deviceaware` keyword of `communicator` and a field of
      `MPICommunicator` for it, answers `hoststaging`, and checks a
      message buffer against the setting. No new dependency, weak or
      otherwise.
    - *An inference fix found by the new test.* The in-process staging
      test defines a fourth test `Communicator` with an `allgather`
      method, and with it `type_tests.jl`'s `@inferred total_mass` and
      `@inferred volume_weighted_norm` failed: `combine_blocks` calls
      `allgather` on the abstractly typed `comm` field, and with that
      many methods inference returns `Any` for the call, and the fold
      over its result with it. Nothing about a reduction's type should
      depend on how many communicator types a process has loaded — an
      application with its own would have hit it too — so the gather is
      now asserted, `allgather(comm, partial)::Vector{typeof(partial)}`,
      which every `allgather` returns.
    - *The device workload.* `test/mpi_device_workload.jl`, standalone and
      not in `Pkg.test`, since the test environment has no device package
      and must not gain one: the vertex-centered wave on three levels with
      the hook, a reflecting box with an odd variable, a periodic 3D
      mesh, each filled and stepped with RK4; Burgers with the interface
      fixup and its conservation; the tracked pulse through two regrids
      flagged by `firing_boxes` on the device; and refinements of the
      first blocks and their coarsening, which move blocks up and down
      the ranks and coarsen siblings with different owners, over a
      conservative cell-centered set and a vertex-centered one. Rank 0
      prints digests of the arrays downloaded and gathered in block
      order, ghosts included, the exact reductions, the sums, and `#`
      lines with how many of each schedule's stages had messages and how
      many were staged, and how blocks migrated. Every callback computes
      in the coordinates' type (the forests' extents are in `T`), since
      Metal has no Float64. `TREEAMR_TEST_BACKEND` chooses `cpu`, `metal`
      or `cuda`, `TREEAMR_TEST_T` the type, and
      `TREEAMR_TEST_DEVICEAWARE=1` the direct path; on a node with several
      devices each rank takes device `local rank mod devices`.
      `test/mpi_device_tests.jl` runs it serially and under `mpiexec` at
      `TREEAMR_TEST_RANKS` (default `2 3`), one thread a rank, and
      requires every line to be the serial device run's, the `sum` lines
      to `rtol = 10⁻⁵` in Float32 and `10⁻¹²` in Float64; every stage with
      messages to have been staged on a device that is not device-aware,
      and none otherwise; and the migrations to have happened.
    - *Measured on Metal* (Apple M3 Pro, Metal.jl in a scratch environment
      that develops this checkout and adds Metal, MPI, KernelAbstractions,
      SHA and Test; MPICH_jll 5.0.2; two and three ranks sharing the one
      GPU). `TREEAMR_TEST_BACKEND=metal julia --project=<env>
      test/mpi_device_tests.jl` passes, 36 tests in 1m45: the serial run
      31 s, `-n 2` 36 s and `-n 3` 38 s of wall clock, nearly all of it
      compilation. In Float32 every line but the sums is the serial Metal
      run's byte for byte, at both counts; at `-n 3` the sums move in the
      seventh or eighth of 9 digits (Burgers' mass 15.999999 against 16,
      `l2` by 1–2 × 10⁻⁷ relative). At `-n 3` 9, 8, 6, 9 and 5 stages of
      the five schedules had messages, and all of them were staged;
      6 blocks moved up the ranks on the refinement, 6 back down on the
      coarsening, and 2 coarsened blocks had children of two owners. As a
      negative control, with the upload of the receive mirror removed the
      `-n 2` run differs from the serial one in 53 lines. The same driver
      on the CPU (the direct path, nothing staged) passes in 59 s.
    - *In process*, the staging path on the CPU: a communicator in
      `regrid_exchange_tests.jl` that answers `hoststaging` with `true`
      for every buffer, over the rendezvous communicator, and records the
      arrays every message is handed out of. At 3 ranks in 2D, vertex-
      centered, outer and reflecting faces with the hook, the staged ghost
      fill, the interface restriction and `regrid!` reproduce the serial
      results bit for bit; every buffer handed to the communicator is one
      of the stages' mirrors and none is a stage buffer; a second fill
      reuses the mirrors and gives the same bits. With the upload removed
      it fails, 7 of its 31 tests. `mpi_workload.jl` gains a `#` line that
      checks, under MPI, that a device-aware communicator shares the
      duplicate, and that a buffer which is not contiguous host memory is
      refused before anything is sent, naming the setting.
    - *The CPU path is unchanged.* `bench/ghosts.jl` at its defaults, one
      thread, best of 200, HEAD before and after, alternated twice: the
      serial fill allocates exactly what it did, 10896 bytes on the
      uniform mesh and 97504 on the two-level one, and the schedule build
      392832 and 4656880; the fill took 2.660–2.719 ms against
      2.667–2.670 ms and 12.09–12.22 ms against 11.87–11.92 ms, the second
      1.5–3 % slower in both rounds, though the serial path runs no
      changed line (the stage without messages returns before the new
      code); not traced further. A distributed CPU fill (the same
      two-level mesh, 10 variables, `mpiexec` with one thread a rank,
      best of 50) allocates exactly what it did, 130320 bytes on each rank
      at `-n 2` and 151856 / 189168 / 151856 at `-n 3`, in 7.0 ms and
      6.1–6.9 ms both before and after.
    - *The CPU workload by hand*, `test/mpi_workload.jl` at `-n 3` against
      the serial run (before the inference fix above, which changes no
      value): every line but the `sum` lines and the `#` lines identical,
      the sums agreeing to the last one or two of 17 digits, as in step 6.
    - *Symmetry* (measured 2026-10-02 on cn111, one node of 8 H200s
      with 4 of them allocated, NVLink between every pair; Julia 1.13.1,
      CUDA.jl's CUDACore 6.4.1; `bench/symmetry_mpi_gpu.sh` as amended
      in the step's script commit, on `h200q`, since `h200debugq`'s QOS
      allows a group 2 GPUs and 24 CPUs). One rank per GPU, one thread a
      rank, `TREEAMR_TEST_RANKS = "2 3 4"`:
      - **Host staging, MPICH_jll 5.0.2** (job 567852):
        `test/mpi_device_tests.jl` passes in Float64 and in Float32, 52
        tests each, every line of every distributed run the serial CUDA
        run's but the sums, every stage with messages staged. The first
        job (567850) had failed at `-n 2`: CUDA refuses to page-lock an
        empty range, and a stage this rank only sends in, or only
        receives in, has an empty mirror, which Metal and the CPU never
        refused. `pagelock_mirror!` now skips an empty mirror (the
        step's fix commit), and nothing else changed between the jobs.
      - **Symmetry has a CUDA-aware MPI**: HPC-X 2.20's Open MPI 4.1.7
        (`nvhpc-hpcx-cuda12/24.9`, through MPIPreferences' system
        binary), whose `ompi_info` says `opal_built_with_cuda_support`
        and for which `MPI.has_cuda()` is `true`. Over it (job 567856)
        the test passes through host staging and through the direct
        path, `communicator(COMM_WORLD; deviceaware = true)`, in
        Float64 and in Float32: four times 52 tests, the direct runs
        with no stage staged. So the direct path, which no device here
        could run, is checked — MPI.jl hands the `CuArray` views to the
        library by device pointer, and UCX moves them. The probe of
        `MPI.has_cuda()` has to run under `mpiexec -n 1`: a singleton
        `MPI.Init()` of this Open MPI inside a SLURM job waits forever
        for the daemon it spawns (job 567853, cancelled).
      - **What the two paths cost** (`bench/mpi.jl` on CUDA through
        `bench/mpiscan.sh 1 2 4`, `N = 32`, `ROOTS = 4`, Float64, 5.8 M
        cells a rank, the four GPUs of one node, HPC-X; minimum ms,
        staged / direct):

        | phase | 1 rank | 2 ranks | 4 ranks |
        |---|---|---|---|
        | two-level rhs | 2.07 / 2.06 | 2.89 / 2.72 | 2.92 / 2.76 |
        | two-level fill_ghosts | 1.71 / 1.70 | 2.49 / 2.29 | 2.52 / 2.35 |
        | — its local groups | 1.69 / 1.67 | 1.44 / 1.41 | 1.44 / 1.43 |
        | — its packs | – | 0.52 / 0.51 | 0.51 / 0.50 |
        | — its unpacks | – | 0.44 / 0.44 | 0.44 / 0.44 |
        | uniform rhs | 0.31 / 0.31 | 0.63 / 0.61 | 0.71 / 0.63 |
        | two-level regrid, refine | 5.2 / 5.2 | 25.4 / 7.1 | 48.0 / 8.7 |
        | two-level regrid, coarsen | 3.6 / 3.6 | 26.1 / 7.3 | 52.6 / 8.2 |
        | interpolate (1000 pts/rank) | 0.11 / 0.11 | 0.55 / 0.55 | 0.47 / 0.50 |

        The rhs weak-scales at 0.71 staged and 0.74 direct to 4 GPUs.
        As on the CPU, the messages are not the cost: the fill less its
        parts is under 0.2 ms either way. The packs and unpacks are,
        0.95 ms of a 2.5 ms fill, and on a device that is launches —
        one per group, for groups of a few thousand points (the
        uniform mesh's single stage has 0.12 + 0.12 ms of them for a
        fill of 0.16 ms serially). The direct path saves 0.16–0.2 ms a
        fill, the two copies. **The staged regrid is the surprise**: 48
        ms at 4 ranks against 8.7 direct and 5.2 serially. A regrid
        stage is built per regrid, so its host mirrors are allocated,
        zero-filled and page-locked afresh each time, for up to 84
        blocks of `32³ × 2` values; that is the likely cost (not
        traced), and MPICH_jll's staging (job 567852) shows the same
        48.4 ms. A regrid that staged through pageable memory, or kept
        its mirrors, would not pay it; left open, since a regrid runs at
        regrid frequency and the direct path exists.
    - *What was not checked* (before the Symmetry run, which checked the
      direct path on CUDA). The direct path on any device: Metal has no
      device-aware MPI, and the CPU's direct path is not the device's
      code path for MPI.jl, which hands a `CuArray` over through its
      own CUDA extension. A checkpoint or an interpolation from a device
      field set under MPI was not run (both go through the host, as steps
      5 and 6 recorded).
    - *Julia 1.11.* In the manifest-free copy, Julia 1.11.9 passes
      `partition_tests.jl`, `exchange_tests.jl`,
      `regrid_exchange_tests.jl` (the staging test included),
      `type_tests.jl` after them (the inference fix) and `mpi_tests.jl`
      (3m21 together); the whole suite was not run there.
    - *Suite cost.* 109140 tests at one thread in 7m55 and 109192 at eight
      in 7m51, against step 6's 109107 in 8m10 and 109159 in 8m02: 33
      more at each, the 31 of the staging test and the device-aware line
      at two rank counts. The thread-independence digests are unchanged.
      The docs build.
  - **Step 9 — wrap-up.** A `docs/src/api/distributed.md` page
    (`communicator`, `blockrange`), a guide section in `index.md`, the
    status there and in `README.md`, and CLAUDE.md's architecture row,
    MPI test commands and suite cost. *Accept:* the docs build; TreeWave
    and TreeHydro green in scratch copies against the checkout, with
    the audit for `nleaves`-sized per-block arrays done in all three
    downstreams; M7 marked *(Done.)*. Tagging the release is left to
    Erik. *(Amended in step 9: M7 is marked implemented rather than
    done, since the acceptance's weak-scaling table and the H200 run
    are scripted and not run; see the milestone's heading.)*

    *(Done, 2026-10-02.)* What it settled:
    - *Suite cost, and what was trimmed.* After step 8 the suite took
      about 8 minutes at one thread, against M9a's 4m16, and
      `mpi_tests.jl` was the largest file: 123–128 s, the serial
      reference in process and then `mpiexec -n 3` and `-n 2` one after
      the other, each launch about 55 s of wall clock and, by a timed
      copy of the workload, almost all of it compilation (a fresh serial
      process spends about 20 s compiling `main`'s call tree before its
      first case runs, and the cases' own time is about 19 s). Nothing
      orders the launches but the checkpoint cross loads, so they now
      overlap. Each run leaves an empty marker beside its checkpoints
      once they are written, and `TREEAMR_CHECKPOINT_FROM` names the
      rank counts whose files a run waits for and loads; every rank
      waits on its own, sleeping, so none spins in MPI meanwhile, and a
      run by hand without the variable loads whatever it finds, as
      before. A new helper, `test/mpi_jobs.jl`, starts both jobs at the
      start of the suite where the machine has room for five ranks
      beside it — eight threads and 24 GB, since a rank measured about
      2 GB resident — and `mpi_tests.jl` collects them; elsewhere, a CI
      runner among them, the three-rank job runs beside the serial
      reference and the two-rank job after it.
      `TREEAMR_TEST_MPI_CONCURRENT=0` or `1` overrides the choice. Two
      type repeats went, each of a path covered elsewhere: the
      workload's `Float32` regrid case (`Float32` crosses MPI as a
      native type, its regrid stage is checked bitwise in process by
      `regrid_exchange_tests.jl`, and `Float32x2`, which MPI sends
      through a derived datatype, stays), and `exchange_tests.jl`'s
      pack/unpack round trip in `Float32` in 1D and 3D and in `Float32x2`
      in 1D, 29 tests (`Float32` stays in 2D, `Float32x2` in 2D and 3D,
      where remote mirrored transfers exist). No claim of an acceptance
      list went with them: every case the step-3 to step-6 acceptances
      name is still in the workload, at both rank counts, and every type
      the step-2 round trip names is still in it. The thread-independence
      test was not touched.

      Per file, one thread, timed around each `include` in a scratch
      copy of `runtests.jl` (s; before and after on the same machine,
      the same day):

      | file | before | after |
      |---|---|---|
      | inline M1 tests | 10.7 | 14.7 |
      | `ghost_tests.jl` | 14.2 | 18.0 |
      | `centering_tests.jl` | 25.7 | 30.5 |
      | `reflect_tests.jl` | 28.7 | 29.5 |
      | `interpolate_tests.jl` | 18.0 | 23.4 |
      | `interface_tests.jl` | 5.2 | 5.1 |
      | `partition_tests.jl` | 8.8 | 8.2 |
      | `exchange_tests.jl` | 40.1 | 32.8 |
      | `regrid_exchange_tests.jl` | 20.3 | 19.0 |
      | `interpolate_exchange_tests.jl` | 13.2 | 12.9 |
      | `allvariables_tests.jl` | 8.3 | 8.1 |
      | `state_tests.jl` | 7.6 | 7.5 |
      | `regrid_tests.jl` | 10.4 | 10.8 |
      | `wave_tests.jl` | 4.2 | 4.2 |
      | `wave_cell_tests.jl` | 2.3 | 2.3 |
      | `burgers_tests.jl` | 16.6 | 17.2 |
      | `imex_tests.jl` | 5.6 | 5.3 |
      | `type_tests.jl` | 12.8 | 13.4 |
      | `checkpoint_tests.jl` | 26.6 | 28.6 |
      | `thread_tests.jl` | 50.2 | 50.3 |
      | `mpi_tests.jl` | 127.8 | 15.0 |
      | `gpu_tests.jl` | 27.5 | 27.0 |
      | whole run | 489.6 | 386.5 |

      The files that run while the jobs compile lose 4–6 s each, about
      20 s in all, to the five ranks on the other cores (an M3 Pro, six
      performance and six efficiency cores); the net is 103 s. On the
      sequential path, which is CI's, `mpi_tests.jl` alone takes 1m47
      (130 tests), against step 6's 2m34 for three launches in a row,
      and the whole suite, forced onto that path here with
      `TREEAMR_TEST_MPI_CONCURRENT=0`, 7m52 (110484 tests), within the
      run-to-run spread of the 7m55 before (steps 5–8 measured 7m49 to
      8m10). So the trim is a local saving: a CI cell, whose runner has
      no room for the jobs beside the suite, is not expected to get
      faster, and launching them there anyway was not tried, since a
      `macos-latest` runner has 7 GB for the main process and five ranks
      of about 2 GB.
    - *CI.* `.github/workflows/CI.yml` needed no change for MPI: MPI.jl's
      default binary is MPICH_jll, an artifact on both runners'
      platforms, and the parallel HDF5_jll build matching it is chosen
      by MPIPreferences with nothing configured ("What the feasibility
      check found"); the ranks run at one thread whatever
      `JULIA_NUM_THREADS` says, and inherit `--check-bounds=yes` and,
      on the single-threaded cells, `--code-coverage` through
      `Base.julia_cmd()`, so their `.cov` files are merged with the
      rest. The job gained a `timeout-minutes` of 120, since a process
      stuck outside the test's own deadlines would hold a runner for
      GitHub's default six hours. **None of it has run on GitHub**: the
      branch is not pushed. What is unverified there: that MPICH's
      launcher starts on the hosted macOS runner; the memory of a
      `macos-latest` runner (7 GB) against the main process and three
      ranks of about 2 GB each, which may swap; and the time of the
      coverage cells, whose ranks are instrumented too (the deadline is
      900 s a launch, against about 55 s uninstrumented here).
      *(Linux, 2026-10-02: in a fresh clone of the branch at 12a6cd6 on
      one Symmetry node each — jobs 567843 on cn106 and 567844 on cn109,
      AMD EPYC 7543 — `Pkg.test()` at one thread passes on Julia 1.11.9,
      110484 tests in 13m34 (`--check-bounds=yes`, which 1.11's
      `Pkg.test` sets), and on 1.13.1, 110484 in 11m18, the MPI jobs
      started early on both, with MPICH_jll 5.0.2 resolved by default.
      `HYDRA_LAUNCHER=fork` was set, so that MPICH's launcher forked its
      ranks as on a runner without SLURM, rather than starting them
      through `srun`. So the MPI tests work on Linux; GitHub's runners
      are what CI is for.)*
    - *Docs.* A guide section, "Running distributed", in
      `docs/src/index.md`; the status there and in `README.md`; the
      distributed API page pointing at the guide. Two examples that
      were right only serially were fixed: `firing_boxes`' verdict used
      `level(forest.leaves[b])`, and the regridding section named
      `forest.leaves[b]` as the way to tell blocks apart; both are
      `blockkey(fs, b)`. The docs build.
    - *Fixed upstream from the audit.* `buffered_flags` now refuses a
      flag vector that is not one per leaf, as `complete_marks` already
      did: TreeGeneralizedHarmonic calls it with `flag_blocks`' flags,
      which over a distributed forest are a rank's local blocks, and
      without the check it buffered around leaves `1:nblocks` and wrote
      global recruit indices into the local vector — silently. The
      `DimensionMismatch` says to pass the local flags to `regrid!` with
      `buffer` instead. And `block_mapreduce`'s docstring had shown
      `maximum(block_mapreduce(…))` as a refinement scale, the
      rank-local pattern two downstreams copied; it shows
      `mesh_mapreduce` now and says why.
    - *The downstreams*, each copied to scratch with this checkout
      developed into the copy, at one thread: TreeWave 310 tests in 1m22
      (`Pkg.test`'s summary; 310 in 1m12 against the M9a checkout);
      TreeHydro 12447 in 5m00 (11893 in 4m16 then; it has grown since);
      TreeGeneralizedHarmonic, whose whole suite takes about 19 minutes,
      a subset of 1760 tests in 7m04 — `precision_`, `prerequisite_`,
      `stencils_`, `stepping_`, `interface_`, `refinement_`, `horizon_`,
      `checkpoint_` and `type_tests.jl`, the files that reach TreeAMR
      most directly (its names, the partition, the exchange, the
      buffer, `interpolate`, checkpoints, the element types). All pass,
      so the local block indices, the new check in `buffered_flags` and
      the rest of M7 change nothing serially for them.
    - *The audit* (read-only, of each downstream's `src/` and `bin/`
      against the list under "What an application must make global
      itself"). None of the three passes `comm` to a `Forest` yet, so
      today an `mpiexec` launch runs a full serial copy per rank, and
      nothing below is wrong until one does. None indexes
      `forest.leaves` by a block index or sizes a per-block array by
      `nleaves`, with the exceptions named. What would go wrong once it
      is distributed:
      - *TreeWave.* `field_scales` (`src/refinement.jl:112`) is
        `maximum(block_mapreduce(…))`, so the Löhner reference amplitude
        becomes per rank: the mesh would depend on the rank count, and an
        empty rank would throw on `maximum` of nothing and leave the
        others in the next collective. `blast_coverage`
        (`src/blast.jl:281–295`) and `track_pulse`'s tracking measure
        (`src/supergaussian.jl:188–195`) are rank-local diagnostics.
        Its integrators are fixed-step, so no adaptive norm. The viewers
        in `bin/` would plot one rank's blocks.
      - *TreeHydro.* Five functions combine `block_mapreduce` on the host
        — `max_signal_speed` (`src/evolution.jl:477`), `floor_hits`
        (`:505`), `ghost_floor_hits` (`:578–585`), `indicator_scales`
        (`src/refinement.jl:189–190`), `peak_compression`
        (`src/sedov.jl:512`) — and the first is the CFL speed: `evolve!`
        derives each chunk's step count from it (`src/driver.jl:857–860`),
        so the ranks would take different numbers of steps and hang in
        the exchange; `entropywave_errors`, `sedov_static` and the
        `check_cfl` calls do the same. Its time-based checkpoint triggers
        read each rank's own clock (`src/driver.jl:910–917`), so some
        ranks would enter the collective `save_checkpoint` and others
        not. The run state it checkpoints holds the rank-local counts and
        speeds, which `write_plain`'s agreement would refuse on every
        rank. `tracked_share`, `reduce_to_grid`, `mode_amplitude`,
        `max_y_kinetic_energy`, `shock_radius` are rank-local
        diagnostics, and `src/kelvinhelmholtz.jl:544` records
        `nblocks` where the mesh's block count is meant.
      - *TreeGeneralizedHarmonic.* Its reductions are already
        `mesh_mapreduce` (the speed, the bounds, the constraint norms),
        and its integrator is fixed-step. But `indicator_flags` passes
        local flags to the public `buffered_flags`
        (`src/refinement.jl:739`) — now refused, above — and
        `clamp_marks` (`:762`) and `refinement_centroid` (`:816`) index
        `forest.leaves[b]` by a local block, the centroid's sums staying
        per rank too; `nfiring` (`:743`) is a local count. Its
        time-based checkpoint triggers read each rank's clock
        (`src/driver.jl:1287–1293`), its run state holds wall-clock fit
        costs and the rank-local centroids (`:1312`), which the agreement
        would refuse, and its non-finite check (`:741`) reads the local
        state only, so one rank would throw alone. Its horizon finder
        passes every point on every rank, which is correct under M7's
        collective `interpolate` and only redundant.
      None of these is fixed here: they are the downstreams' to fix
      when they distribute.
    - *Julia 1.11.* In the manifest-free copy, Julia 1.11.9 passes the
      whole suite, 110484 tests in 6m46 at one thread, with the MPI jobs
      started early as on 1.13. It resolves MPICH_jll there, the binary
      CI's runners get, against MPIABI_jll on 1.13.1 here (the global
      preference), and IMEXRungeKutta 1.3.0 from `main`, against the 1.1.0
      the 1.13 test manifest holds.
    - *Suite cost.* 110484 tests at one thread in 6m25 and 110536 at
      eight in 6m21 (`Pkg.test`), against 7m55 and 7m51 after step 8.
      The differences from the step-7 follow-up's 110511 and 110563 are the
      29 tests of the round trip and two new `buffered_flags`
      assertions. The thread-independence digests are unchanged. The
      docs build.
- **M9b — Visualization export.** *(Split from M9, "I/O and
  visualization", on 2026-09-29, when its checkpoint half became M9a;
  not designed.)* After M7. The candidates:
  - an XDMF sidecar that describes the checkpoint's own datasets as
    hyperslabs, for ParaView and VisIt (it grows by one grid per block);
  - VTKHDF `OverlappingAMR` with restricted parents added, which turns
    leaf-only data into a real overlapping hierarchy and gives level of
    detail for about 1/7 more storage in 3D (ParaView only);
  - Conduit Blueprint through Conduit.jl, for VisIt;
  - a Parthenon-compatible export, which yt reads directly;
  - sampling onto output grids (slices, a uniform box) through M11's
    `interpolate`.

  An in-file view goes in a top-level group of its own and points at
  the checkpoint's datasets without copying them, through links or
  virtual datasets, as the layout under
  [Checkpoint and restart](#checkpoint-and-restart) leaves room for.
