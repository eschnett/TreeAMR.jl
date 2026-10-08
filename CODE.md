# TreeAMR.jl — Design

TreeAMR.jl implements a tree-based (octree-style) AMR discretization for
Julia. It provides the mesh, the storage, and the inter-grid operations —
no physics.

**Status.** Milestones M0–M8, M10, M11, M9a, M7 and M12 are done;
next is visualization export (M9b). The plan is in [PLAN.md](PLAN.md).
How the package got here — who decided what and when, the alternatives
tried, measurements that later ones replaced, and the milestone records
— is in [HISTORY.md](HISTORY.md); a step number such as "M7 step 6b" or
"step 8" below names a step of the milestone records there
([HISTORY.md](HISTORY.md#milestones)).

**Contents**

- [Goals](#goals)
- [Scope and non-goals](#scope-and-non-goals)
- [Core concepts](#core-concepts)
  - [Blocks](#blocks)
  - [Centerings](#centerings)
  - [Tree structure](#tree-structure)
  - [Domain and boundaries](#domain-and-boundaries)
  - [Data layout](#data-layout)
  - [Precision](#precision)
- [Operations](#operations)
  - [Ghost filling](#ghost-filling)
  - [What the ghost fill costs](#what-the-ghost-fill-costs)
  - [The copy kernels on a device](#the-copy-kernels-on-a-device)
  - [Operators](#operators)
  - [Conservation at coarse-fine faces](#conservation-at-coarse-fine-faces)
  - [Regridding](#regridding)
  - [Point interpolation](#point-interpolation)
  - [Checkpoint and restart](#checkpoint-and-restart)
- [Application interface (sketch)](#application-interface-sketch)
- [Time integration](#time-integration)
- [Parallelism](#parallelism)
  - [Distributed meshes](#distributed-meshes)
- [Code structure](#code-structure)
  - [Rules that span several files](#rules-that-span-several-files)
  - [Index conventions](#index-conventions)
- [Tests](#tests)
- [Ecosystem integration](#ecosystem-integration)
  - [Downstream applications](#downstream-applications)
- [Open questions](#open-questions)

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
- **Every centering.** The package does all `2^D` centerings
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
  per dimension. It says how far a stencil reaches into a neighbor's data,
  which is a property of what is stored, not of how space is cut up: a
  flux field reaches nowhere, an evolved vertex-centered field reaches
  differently from a cell-centered one at the same order, and two field
  sets over one forest with different `G` is the normal case.
  Both are runtime values (no recompilation when they change), validated
  at construction against the invariants below.

Ghosts are stored on all sides of every block. Ghost cells exist only
in the working array, not in the ODE state vector (see
[Time integration](#time-integration)).

Invariants tying the parameters together (per dimension, against that
dimension's `G_d`):

- `N` is even (a fine block covers `N/2` cells at its parent's spacing).
- Shifted restriction of order `p` reads fine interior cells up to depth
  `2G + p/2 − 1`: `N ≥ 2G + p/2 − 1` (at `p = 2` this is the basic
  `N ≥ 2G`).
- The restriction window itself must fit inside a fine block's interior:
  `N ≥ p` (binding when `G` is small).
- Symmetric prolongation of order `p` reads up to `p/2` of the source
  block's own ghost layers: `G ≥ p/2`. (Worst case is the first fine
  ghost layer: its target sits a quarter coarse cell from the interface,
  between the coarse nodes straddling it, and the node across the
  interface is already ghost layer 1.)
- The two restriction bullets and the `p/2` above are for the
  point-value family. The conservative family: prolongation orders are *odd* (the reconstruction is
  centered on a cell, not a quarter cell off one) with `G ≥ (p−1)/2` —
  one fewer ghost layer at comparable order — and restriction is the
  fixed 2-cell exact average, needing only `N ≥ 2G`.
- All of the above is for a cell-centered dimension. In a vertex-like
  dimension: the exchange region must fit
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
rather than an enumeration of the `2^D` cases because of the
tensor-product structure everything rests on: every transfer is a product
of `D` one-dimensional stencils, and the stencil for dimension `d`
depends on the centering *in that dimension* alone.

**Shared points and half-open ownership**. A vertex-like
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
  of the hook (see [Ghost filling](#ghost-filling)), and the
  asymmetry remains: the low wall evolves, the high one interpolates.
  Accepted: treating both boundary planes alike would have the lowest
  blocks own `N − 1` points in that dimension and the uniform state
  layout would be gone.

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
centering; `fill_by_coordinates!` and
`boundary_by_coordinates` evaluate their callbacks at the field set's own
points — face centers for a face field.

The alternatives rejected here — closed ownership, and one shape for every centering — are recorded with their reasons in [HISTORY.md](HISTORY.md#centerings).

**Invariant.** The exchange region on a block's high side spans positions
`N … N+G_d+c_d−1` and must lie within one ring of neighbors, including
when those neighbors are finer and each spans only `N/2` coarse cells:
`N/2 ≥ G_d + c_d`, i.e. **`N ≥ 2G_d + 2c_d`** — `N ≥ 2G` in a cell-centered
dimension, `N ≥ 2G + 2` in a vertex-like one (`N` is even). It is the
same condition as "restriction reads owned points only": the last coarse
ghost at position `N+G` is fine point `2G`, stored at `3G + 1`, owned iff
`3G + 1 ≤ G + N`. Checked when the field set is built.

**`G` is per dimension**: an `NTuple{D,Int}`, with a plain
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

`G` is a required `FieldSet` keyword with no default — an integer or an
`NTuple{D,Integer}`, stored as `NTuple{D,Int}` — for the same reason
`Operators` has no default order, and refusing it says so. `Forest` no
longer takes `G`, and still accepts the keyword only in order to throw a
message naming where it went; `regrid!` likewise keeps the old
three-argument form as a method that throws. Four things the design left
open, settled by the implementation:

- **`GhostSchedule` belongs to a layout, not to a forest.** The
  documented form is `GhostSchedule(fs, ops)`. The forest form survives
  as `GhostSchedule(forest, ops; G, T, backend)` because it costs nothing: it is the body, and
  the field-set form is one line spelling the triple out of `fs`, and it
  is what a caller with no field set in hand uses. `fill_ghosts!`
  compares `fs.G` against the schedule's and refuses a mismatch,
  alongside the existing forest, generation, element type and backend
  checks: every target range in a schedule is wrong for another `G`, and
  nothing else would have caught it. The check covers the centering too,
  for the same reason — the stored extent, the target ranges
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
  the message. There is no blanket `G_d ≥ 1` (see below for why
  none is needed).
- **`regrid!` allocates per field set from that set's own `G`**, and
  fills ghosts per field set from that set's own schedule, rather than
  once from a shared one.

`centering` is a
`FieldSet` keyword defaulting to `cellcentered(D)` — unlike `G` it *does*
get a default, because cell-centered is what a field set was through M6
and what everything not deliberately staggered wants. It is validated
into an `NTuple{D,Symbol}`; `staggers` turns it into the `c_d` tuple the
arithmetic uses, and is exported so that an application can size its own
loops. `closedview` and `map_blocks!(…; closed = true)` are the closed
range's two faces. The cell-centered stencils are the same rational
weights as before.

- **There is no blanket `G_d ≥ 1` at all**. Step 1's rule — a
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
  test now runs `p = 1` at `G = 0`. Along a stagger: the block still has its shared plane, which is
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

The test wave equation is vertex-centered — `centering` is a keyword on every entry
point of `test/wave.jl`, defaulting to `vertexcentered(D)`, and the M3
study survives verbatim in `test/wave_cell_tests.jl` saying
`cellcentered(D)` out loud at every call. The measured table is under
[Operators](#operators). What follows from it:

- **Nothing in the application knows the centering.**
  `wave_rhs_kernel!` takes no `Val(C)`: it reads
  its own point and its neighbours a spacing away, which is the same
  stencil wherever those points sit. `WaveProblem` needs no centering
  either, since it carries the field set. So the whole staggered study
  is the cell-centered one with one keyword changed at the
  `FieldSet` calls — which is the claim a centering that is "a property
  of the field set and nothing else" has to make good on.
- **`wave_forest` deliberately does not take a centering**. It builds the forest, and how space is cut into blocks is
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
exactly `2^D` children of half the spacing.

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

**Key encoding**: a key is an `isbits` struct — root index
(`Int32`), level (`Int8`), and per-dimension coordinates
(`NTuple{D, UInt32}`, the block's integer position at its own level).
Curve order is root index first, then Morton order, with the bit
interleaving computed on the fly during comparison — no packed-integer
format, and no practical depth limit (32 levels). The curve is plain
Morton.

**Neighbor asymmetry**: neighbor finding is not
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
- **Reflecting** (per face): a face across which the solution is
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
  refused in one that has one. A variable without a parity has
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
- **Rotating** (a pair of dimensions): a quarter-plane symmetry. Only one quadrant of the
  `(d1, d2)` plane is simulated, and the other three are its images
  under rotations by 90° about the line where the low faces of `d1` and
  `d2` meet. The models are Cactus's RotatingSymmetry90 and a spinning
  black hole, for which a reflection in `x` or in `y` is not a
  symmetry; together with M10's reflection at the low face of a third
  dimension it gives an octant. It is declared on the forest as
  `rotating = (d1, d2)`, default `nothing`. The rotation `R` takes
  `e_{d1}` to `e_{d2}` and `e_{d2}` to `−e_{d1}`, a quarter turn
  counterclockwise in the `(d1, d2)` plane, and leaves every other
  dimension alone; its axis is the line where the two low faces meet,
  at the domain's low corner in the plane. The order of the pair fixes
  the sense of `R`, and with it the meaning of the variable map below.
  The forest refuses, each time saying why:

  - `D < 2`;
  - `d1 == d2`, or either dimension out of range;
  - `roots[d1] ≠ roots[d2]`, since the low face of `d1` must map onto
    the low face of `d2` (once the root counts agree, the cube check
    makes the extents' lengths agree);
  - `d1` or `d2` periodic;
  - a reflecting face on either side of `d1` or `d2`. The two low faces
    are the seam, and the two high faces stay outer, the hook's;
    reflecting high walls together with the rotation are an open
    question (see [Open questions](#open-questions)).

  Every field set over such a forest declares its **rotation**, a
  signed variable map with one entry per variable, required with no
  default, since it is physics, as `parity` is. Variable `v`
  at `Rp` equals `sign(rotation[v])` times variable `|rotation[v]|` at
  `p`, of the set it rotates from: the set itself when its layout is
  symmetric under exchanging `d1` and `d2`, its partner otherwise (see
  "Rotating seams" under [Ghost filling](#ghost-filling)). For
  `(ρ, vx, vy, vz)` with `rotating = (1, 2)` it is
  `rotation = (1, −3, 2, 4)`: `ρ` and `vz` are unchanged, and the
  rotated velocity has `vx′ = −vy` and `vy′ = vx`. A 90° rotation maps
  every Cartesian tensor component, of any rank, to plus or minus
  another component, so a signed map is enough for any variable an
  application stores in Cartesian components. The field set refuses,
  saying why:

  - a map that is not a signed permutation;
  - a map of the wrong length;
  - for a set that rotates into itself, a map `Q` with `Q⁴ ≠ I`, since
    four quarter turns are the identity;
  - a map that sends a variable to one of another parity in a
    reflecting dimension. The rotation and a reflection outside its
    plane commute, so the two declarations must agree.

  The fourth-power and parity checks need the
  set the map reads from, so the field set makes them for a symmetric
  layout and `RotationPair` makes them, as `Q_a Q_b Q_a Q_b = I` and the
  parities across the pair, for an asymmetric one; a lone asymmetric
  set is checked for its shape only. Over a forest without a seam the
  map may be omitted, and if given is checked for its shape, kept and
  otherwise ignored, as `parity` is over a forest without reflecting
  faces, so `factors` and `rotvars` are `nothing` there exactly as
  before. A `RotationPair` of two symmetric sets is refused: each turns
  into itself.

  Unlike reflection, **the tree sees the seam.** The low face of `d1`
  is glued to the low face of `d2`, and across it neighbor finding
  returns the real leaves, each with an **orientation**, the number of
  quarter turns that carry it to where the asking block sees it.
  `balance!` and `isbalanced` see the seam through the neighbor search,
  as they see a periodic wrap, and a block at the axis is its own
  neighbor in three directions of the plane, as a single periodic root
  is its own neighbor.
  Balance gains one rule, **conformity at the seam**: a leaf on the low face of `d1` and its image
  on the low face of `d2` are at the same level, so no coarse-fine face
  crosses the seam. Why, and what it leaves, is under "Rotating seams"
  in [Ghost filling](#ghost-filling). A leaf list handed to
  `Forest(roots; …, leaves)` that is not conforming is refused there,
  as an unbalanced one is.

  The forest has a field for it, `rotating`, with `(0, 0)` for none:
  two `Int8`s, which fit in the struct's alignment padding (what a
  field of `Forest` costs is under "The buffer pool" in
  [Distributed meshes](#distributed-meshes)).
- **Physical** (per face, neither periodic, reflecting nor a rotating
  seam): ghost cells are filled by a user-supplied boundary condition
  hook. For a vertex-like dimension the domain's upper boundary plane
  is the first plane of an outward-facing region and is filled by the
  hook too (see [Centerings](#centerings)).

### Data layout

Two arrays exist:

- the **state vector**: a flat vector of length `N^D · nvars · nblocks`
  holding leaf interiors only — this is what ODE integrators see;
- the **working array**: one big persistent `(D+2)`-dimensional array
  holding all leaf blocks including ghosts:

      work :: Array{T, D+2}   # size (N+2G₁+c₁, ..., N+2G_D+c_D, nvars, nblocks)

  with `c_d = 1` in a vertex-like dimension and `0` in a cell-centered
  one, and `G_d` the field set's ghost width in dimension `d` (see
  [Centerings](#centerings)). The state vector holds `N^D` per block per
  variable for every centering.

- One array for all variables of one field set, not per-variable
  arrays. Variables that differ in centering, ghost width, or operator
  choice go in different field sets.
- Cell indices vary fastest (GPU coalescing); blocks are ordered by
  Morton key.
- Element type `T` is generic; `Float64` default, `Float32` relevant for
  GPUs. See "Precision" below for what carries `T`.
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
generic, not just the storage**.

A generic storage type alone is too weak, since fp64 is exactly what a device may not have; how that was found is in [HISTORY.md](HISTORY.md#precision). So:

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

**In a vertex-like dimension** (see
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

**Stencil widths are per dimension**: the transfer kernel
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
**between phase 1 and the prolongation sweep**, because a block
at the domain edge has prolongation stencils that reach *tangentially*
past the edge into the coarse source's own outer ghosts, so running the
hook last would feed unwritten memory into the interpolation. The hook
may therefore read its block's interior (as extrapolating conditions
do) but not other blocks' ghosts, and not ghosts that prolongation has
yet to fill.

**Reflecting faces** are transfers, not hook calls. A ghost region
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

**The upper wall point in a vertex-like dimension**. On a high reflecting face the wall plane `G + N + 1` belongs to
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

**Rotating seams.**
A rotating seam (see [Domain and boundaries](#domain-and-boundaries))
is harder than a reflecting face in two ways. It is **non-local**: the
low face of `d1` is glued to the low face of `d2`, so the source of a
ghost lies elsewhere in the tree and can be at another level. And it
**mixes variables**: `vx′ = −vy` and `vy′ = vx`. The transfer kernel as
M10 left it can express neither. Its stencils are separable — the
source index along `d` depends on the target index along `d` alone —
so it cannot exchange axes, and it reads target variable `v` from
source variable `v`. The design keeps the stencils, the phases and the
kernel's arithmetic, and changes only where the kernel loads from.

*Orientation.* A ghost region of a block lies in the image `R^r` of
real data, `r ∈ {0, 1, 2, 3}`: `r = 1` when it lies beyond the low face
of `d1` only, `r = 3` beyond the low face of `d2` only, `r = 2` beyond
both, and `r = 0` otherwise. Since `u(R^r q) = Q^r u(q)`, with `Q` the
field set's rotation map, the ghost at `p` is `Q^r` applied to the data
at `R^{−r} p`. In global level coordinates `g` — a node's integer
position at its level, counted across the brick, with the axis at 0 —
the naive neighbor that the brick arithmetic gives maps back to a real
node as follows, every other coordinate unchanged:

| `r` | the naive neighbor `(g_{d1}, g_{d2})` | the real node |
|---|---|---|
| 1 | `g_{d1} < 0 ≤ g_{d2}` | `(g_{d2}, −1 − g_{d1})` |
| 3 | `g_{d2} < 0 ≤ g_{d1}` | `(−1 − g_{d2}, g_{d1})` |
| 2 | both negative | `(−1 − g_{d1}, −1 − g_{d2})` |

A real node beyond a high face leaves the domain, and the region is the
hook's, as for M10's `δ′`: the hook sees outer faces only, never the
seam. All the neighbors of one block in one direction share one `r`,
since the seam lies on a root boundary at every level, so the
orientation belongs to a (block, direction) pair and the neighbor
search returns it with the keys. In the real frame the search runs in
the rotated direction `R^{−r} δ`, and a finer neighbor's child offset,
which a restriction's stencils depend on, is taken back into the
virtual frame by the same permutation and flip as the coordinates.
With `(a, b)` the components
along `(d1, d2)`, the direction from the real leaves (`real_direction`)
and the virtual child offset of a real offset `o` (`virtual_offset`)
are

| `r` | `R^{−r} δ` | virtual offset |
|---|---|---|
| 1 | `(δ_b, −δ_a)` | `(1 − o_b, o_a)` |
| 2 | `(−δ_a, −δ_b)` | `(1 − o_a, 1 − o_b)` |
| 3 | `(−δ_b, δ_a)` | `(o_b, 1 − o_a)` |

The offset follows because the parent of a node maps to the parent of
its image (`−1 − g` halves to `−1 − ⌊g/2⌋`), so a child turns about its
parent's centre. The test derives both from turned `Rational` boxes,
not from these formulas. Where the step finds no leaf, the search
still returns the orientation of the region it stepped into, the one
the hook's region lies in.

*The virtual frame.* A transfer's stencils are built exactly as though
the source sat at its **virtual position**, where the asking block sees
it across the seam, with the target's layout. A same-level source is
then a copy, a coarser one a prolongation and finer ones a restriction,
with the ordinary one-dimensional stencils for that kind, direction
and child offset, so the stencil builders do not change. What is left
is to read the virtual source's points out of the real array.

*The axis map.* The kernel turns a source point's virtual stored index
`k` into the real stored index with an `AxisMap`, a permutation of the
dimensions and a flip per dimension, applied in the source load. Along
a flipped dimension `k ↦ n_d + 1 − k`, with `n_d` the source array's
stored size along `d`. One formula serves cell and vertex centering
alike, because the flip reverses the stored array, ghosts onto ghosts
and the closed range onto itself. In the plane:

| `r` | real index along `d1` | real index along `d2` |
|---|---|---|
| 1 | `k[d2]` | `n + 1 − k[d1]` |
| 2 | `n + 1 − k[d1]` | `n + 1 − k[d2]` |
| 3 | `n + 1 − k[d2]` | `k[d1]` |

and the identity elsewhere. The flipped dimensions are always the real
source's dimensions normal to the seam. The variable map is applied in
the same load: for target variable `v` the kernel reads source variable
`σ_r(v)`, from a table `rotvars` on the field set, `nvars × 4`, of the
composed map per orientation, which lives beside the parity factors
and moves to the device with them.

*Why no fix-up pass.* Filling the seam ghosts as ordinary copies and
then mixing them by a second pass is not needed. A 90° rotation of any Cartesian tensor component is a **signed
permutation** of components, so `Q^r` splits into two halves, each of
which goes where M10 already put something:

- the **permutation** is a variable remap on the source load, beside
  the axis map;
- the **sign** is a column of the factor table on the target, where M10
  puts the parity. The table gains the orientation as a second axis,
  column `mirror column + 3^D·r`, so `r = 0` keeps every existing
  column, and a seam transfer that is also mirrored (a reflecting low
  face in `z` with the rotation in `xy`) multiplies one factor, the
  parity times the rotation's sign.

So a rotated transfer is part of one replay of the schedule, in the
ordinary phases, on every backend; no ghost is written twice, and none
is read in a provisional state. A fix-up pass would have had to run
inside the phases, not after them, since a prolongation may read a
coarse source's rotated ghosts. The kernel reaches a rotated source
through an accessor, as it reaches M7's packed buffers, and an ordinary
group launches it unchanged. Under MPI the pack applies the axis map
and the permutation and computes the unscaled sum, and the unpack
applies the sign through the factor column, as it applies the parity,
so the serial fill's `−0` is kept bit for bit (see "The parity factor
is applied when unpacking" under
[Distributed meshes](#distributed-meshes)).

*Conformity*. Leaves across a seam face
are at the same level: `balance!` refines a leaf across a seam face
that is coarser by any amount, not only by two levels or more, and the
checked `leaves` path and `isbalanced` enforce the same rule. A family
on one seam face is coarsened only together with its image, since
completion undoes any other coarsening there, as it undoes every
coarsening balance cannot support. Two things follow:

- No coarse-fine face crosses the seam, so the interface restriction
  never does (see [Conservation](#conservation-at-coarse-fine-faces)).
  Otherwise it would have to overwrite a block's owned seam plane from
  the rotated image of a finer neighbor's, a transfer with no other
  use.
- The two owned seam planes of a vertex-like set (below) are evolved at
  one resolution, from same-level ghosts that are each other's images,
  so the objection M8 raised against two copies of a point — at a
  coarse-fine face their ghosts differ and they drift apart — does not
  arise.

The cost is that refinement at one seam face refines its image at the
other, which a problem with this symmetry wants anyway: a feature at
the seam is at both faces. Edge and corner regions across the seam,
such as a block on the low face of `d1` asking in direction `(−1, +1)`
in the plane, are not covered by the rule, and can still meet a
difference of one level. They are prolongations and restrictions
through the rotation, which the virtual frame builds like any other.
A single regrid still moves a block by at most one level. Before it the
forest is conforming, so a leaf and its image are at one level; the
marks move each of them by at most one; and the seam rule raises a
leaf only to its image's new level, at most one above its own old one.
The transfer asserts this, as it does today, and a test exercises it.

*Phase 1 stays race free.* A rotated copy or restriction reads only the
real source's **owned** points, as an ordinary one does. The flipped
dimension is the real source's dimension normal to the seam, and the
target reads the virtual source's points nearest itself, which are the
real source's points nearest the seam: its low owned points. In a
vertex-like set the virtual source's high shared plane, flipped, is the
real source's owned seam plane. It lies where the target's own owned
seam plane lies, which is not a ghost, so no copy targets that position
and none reads the plane. The tangential dimension is not flipped and is
read as an ordinary transfer reads it. That was checked by hand in the
design, and the `NaN` test checks it again. Phase 2 needs no new rule: a
rotated prolongation targets the block's own level and joins the sweep
there. The ghosts it reads in its coarse source are the virtual
source's, which the flip maps to the real source's ghosts — the virtual
high ghosts facing the target are the real low ghosts beyond the seam —
and these are filled in phase 1 or in an earlier step of the sweep, as
for any prolongation.

*Two owned seam planes*. In a vertex-like set the low plane
of `d1` and the low plane of `d2` are both owned, as every low boundary
plane is under M8's half-open ownership, and both are evolved, although
they are the same points under `R`. The alternative, deriving one from
the other as M10 derives the upper wall point, would have the blocks on
one seam face own `N − 1` points across it, and the uniform state
layout would be gone; that is the reason M8 gave for accepting
asymmetric walls. Under conformity the two copies see same-level
ghosts that are each other's images, so a covariant right-hand side
evolves them identically: to roundoff in general, and bit for bit when
its arithmetic is itself invariant under the rotation. IEEE addition
commutes, so `a + b` against `b + a` is exact, but a longer sum taken
in another order is not. The wave test asserts what it measures. The
axis, a point in 2D and a line along the third dimension in 3D, is
its own image and is owned once, by the blocks along it. A sum
over owned points, such as `volume_weighted_norm`, holds the seam's
points once per face, so it is not exactly a quarter of the full
domain's. A vertex-like set carries no conserved total (the
conservative family is refused along a stagger), so this is recorded
rather than corrected. The choice is about ownership only and assumes
nothing about aligned axes.

*Asymmetric layouts, and pairs*. The virtual frame needs the rotated source to have the target's
layout. A set whose layout is symmetric under exchanging `d1` and `d2`
— `G[d1] == G[d2]` and `c[d1] == c[d2]`, which holds for cell- and
vertex-centered sets and for any set staggered alike along `d1` and
`d2`, such as the `z`-face and the `z`-edge, at equal ghost widths
there — rotates into itself. One that is not has its rotated image
in the set with the swapped layout: `B_x`, face-centered in `x`, takes
the value `−B_y` across the low face of `x`. Two cases:

- **`G = 0` in both plane dimensions**, as for TreeHydro's and Burgers'
  fluxes: the set has no seam ghost region at all. None of its
  exchange regions has a negative component in the plane — along its
  stagger the one region is the shared plane at a block's high face —
  so none crosses the seam. An asymmetric layout
  with `G = 0` along both plane dimensions is cell-centered along one
  of them, and `check_operators` refuses `G = 0` along a cell-centered
  dimension for every point-value order and the conservative family
  along a vertex-like one, so no ghost schedule serves such a set: it
  is never ghost-filled, and is regridded as `fs => nothing`, as the
  fluxes are today. The field set takes it without a partner.
- **Otherwise**, as for `B_x` and `B_y` of constrained transport with
  `G > 0` in the plane: the two sets are filled as a
  **`RotationPair(a, b)`**, each from the other across the seam. An
  orientation of 1 or 3 reads the partner's working array, and 2 reads
  the set's own, since two quarter turns map a layout onto itself.
  `RotationPair` is an exported immutable value: two field sets over
  one forest with each other's swapped layout, the same `nvars`,
  element type and backend, and maps with `Q_a Q_b Q_a Q_b = I`; it
  refuses anything else, with the reason. It builds the composed tables
  — for `a`'s targets `Q_a`, `Q_a Q_b` and `Q_a Q_b Q_a` for
  `r = 1, 2, 3`, the first and last reading `b` — and symmetrically for
  `b`. `regrid!` swaps `work` in place, so a pair stays valid across
  regrids. Each set keeps its own schedule and operators; a transfer's
  stencils are its target's.

  `fill_ghosts!(pair, (sa, sb); boundary)`, with one hook for both sets
  or one each, runs the two schedules' stages merged by (stage,
  member): phase 1 for `a` and then `b`, then the hooks, then each
  phase-2 target level for `a` and then `b`. A prolongation may read
  its partner's coarser ghosts, which an earlier step of the merged
  sweep has filled. Under MPI each stage completes before the next
  starts, so the two members share a stage's tag, MPI's
  non-overtaking order keeping their messages apart. What a stage
  completes before the next is its receives — its sends are waited on
  at the end of the fill, as in every fill since M7 — and that is
  enough. A stage sends at most one message to each peer, and every
  rank runs the merged stages in one order, so between two ranks `a`'s
  message of a tag is sent before `b`'s, and `a`'s receive of it is
  posted, and completed, before `b`'s is posted; MPI matches two
  messages of one tag between one pair of ranks in the order they were
  sent, so `a`'s receive takes `a`'s message. The regrid's transfers of
  the two members share tag 50 for the same reason, as several field
  sets' have since M7. A
  plain `fill_ghosts!` on an asymmetric set whose schedule has odd
  orientations refuses, saying to fill it as a `RotationPair`.
  The schedule records nothing for
  it. The fill decides from the layout, `G[d1] > 0 || G[d2] > 0`,
  which holds exactly when the serial schedule has transfers of odd
  orientation — every block on a seam face then has a nonempty region
  beyond it — and is the same on every rank, where a schedule's own
  groups are not: a rank whose blocks are off the seam has none, and a
  refusal on some ranks only would leave the others waiting in the
  exchange.
  `regrid!` takes the element
  `pair => (sa, sb)` beside `fs => schedule` and `fs => nothing`, fills
  the pair once before either set moves, and moves each set with its
  own schedule's operators; the transfers themselves are
  `δ = 0` and never cross the seam. A plain `fs => schedule` of an asymmetric
  set with ghosts in the plane is refused there, among the collective
  checks, with the pair named, since its fill would be refused further
  down on every rank alike; `fs => nothing` fills nothing and is
  allowed. `adapt_to_initial_data!` builds its schedules itself, so it
  takes no `pair => …` element: it has a pair form,
  `adapt_to_initial_data!(pair, operators; initial, flag | flags, …)`,
  with `initial` and `boundary` one for both sets or a tuple of two,
  `flags` called with the pair, and the two schedules returned. It
  fills ghosts before it flags, for a criterion that reads them, which
  is why it needs the pair at all.

*90° only*. A 180° rotation, the
π-symmetry, needs the same machinery and one thing more, a face glued
to itself and flipped about the domain's centre line; it is an open
question (see [Open questions](#open-questions)).

Edge and corner ghost regions are always filled — some stencils don't
need them, but filling unconditionally is simpler, and cross-derivative
stencils do. Application kernels are strictly block-local: neighbor data
is visible only through ghost cells.

Transfers are **batched by stencil**: everything sharing a kind, a
direction, a child offset, a mirror state per dimension — none,
mirrored rows, or the vertex-like upper wall row — and an
orientation shares one set of one-dimensional stencils and so one
kernel launch. The orientation joins `keyorder` as well, so both ends
of an MPI message still derive one layout. A group carries its
orientation and the seam's plane
`(d1, d2)`, two `Int8`s and a pair of them, and `run_group!` derives the
axis map at the launch, the identity for an ordinary group; `factorcol` is an `Int32`, so the three
share the eight bytes it had alone and a group is no larger than
it would be without them. Prolongations are
additionally batched by *target level*: the batch is
the unit the phase-2 sweep
schedules, so a batch spanning two levels would be filed under one of
them and the coarsest-target-first order would be quietly lost wherever
three levels meet. Batching is an implementation detail of a
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

The downstream profile that led to the changes below, and their before-and-after numbers, are in [HISTORY.md](HISTORY.md#what-the-ghost-fill-costs). The kernel's costs, and what was done about each:

**Bounds checking was the largest cost.** `checkbounds_indices` and its `size` calls were 15 % of
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
backends; on a GPU it does not
belong in KernelAbstractions' own `expand`. KernelAbstractions forms the index with run-time 64-bit
divisions, emulated in software, and that held the fill and the scatter
to a fifth of an H200's bandwidth; a device now launches flat and
divides by precomputed inverses, while the CPU keeps this launch — see
[The copy kernels on a device](#the-copy-kernels-on-a-device). The two boundary kernels launch the same way.

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
associate. Only the loads move. 

**The per-launch allocation is KernelAbstractions', not ours.** Rebuilding the geometry tuples costs
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

### The copy kernels on a device

TreeGeneralizedHarmonic made its
right-hand-side kernel 8× faster on an H200, 8.5 → 1.1 ns a point, and
then found what was left of an evaluation to be TreeAMR's: `scatter!`
and `fill_ghosts!` were 0.73 of its 1.81 ns a point at `32³`, and 1.39 of
2.61 at `16³` (its `CODE.md`, "Upstream prerequisites", item 5, and "The
right-hand side on an H200"). The scatter ran at 0.96 TB/s and the
transfer kernel at 0.65, of the H200's 4.8.

**The index was the cost.** Both kernels launched over a `D + 2`-axis
ndrange with run-time extents and read `@index(Global, NTuple)`. On a
device KernelAbstractions forms that by indexing two `CartesianIndices`
— the workgroup's place in the grid, the item's in the workgroup — with
linear ids (`expand` in its `nditeration.jl`): two integer divisions per
axis, in 64 bits, which a GPU emulates in software. The downstream
counted about 1200 SASS instructions to copy one value. On the CPU none
of this happens: the backend iterates a workgroup's `CartesianIndices` by
incrementing.

**A device launches flat; the CPU keeps the shaped launch.** One kernel
serves both (`launch_positional!` and `kernel_position` in
`threading.jl`). On a device the ndrange is one axis, which `expand`
forms without dividing, and the kernel recovers its position by
multiplying with each extent's precomputed inverse
(`Base.MultiplicativeInverses.SignedMultiplicativeInverse`, the
arithmetic a compiler applies to a division by a constant), carried in a
`LinearShape` kernel argument built at each launch (about 9 ns an
extent). Its arithmetic is `Int32` when the launch has at most 2³¹ − 1
items and `Int` otherwise. On the CPU the launch is the shaped one by
owner, as before, because a flat launch there cost the scatter 2.3×
(38.8 → 89 ns a point at `16³`, 20 variables, one thread): half of that
was the inverses, and half was stores that no longer ran along a row. The
suite runs the flat form on the CPU too (`flat = true`), since CI has no
device, and requires it to fill bit for bit as the shaped form does.

**A copy skips its weights.** `Stencil1D` records whether it is one point
of weight exactly one (`unit`, decided on the host from the weights), and
the kernel of a group whose stencils are all unit takes a branch, chosen
by a run-time flag that is the same for every item of a launch: one
load, no `D` weight loads, no product. It computes `0 + x`, not `x`,
because the general sum starts from zero and `0 + 1·x` turns `−0` into
`+0`; so fills are bit-identical to 0.1.7, which was checked by digest of
the whole working array (ghosts included) over `D = 1, 2, 3`, both
centerings, both operator families, reflecting walls and rotating seams,
in `Float64` and `Float32`, and by `thread_workload.jl` and the serial
`mpi_workload.jl`, whose outputs are unchanged. `unit` comes from the
weights, not from the kind: a mirrored copy at a vertex-like high wall
interpolates the wall plane it does not own (`wall_stencil`), and a
vertex-centered `PointValue` restriction is an injection, so it is unit.

**Every run-time choice of type is compiled at every launch site.**
Inference follows each member of a union at every call site. So the copy is a
`Bool` argument (folded away for any stencil wider than one point, `Ps`
being a type parameter), the CPU's shaped launch is type-stable, and
only the flat launch hides its kernel from inference
(`Base.inferencebarrier`), which costs one dynamic call in a launch that
KernelAbstractions already dispatches dynamically. A barrier on every
launch had doubled the CPU fill's allocation, and a closure over the
launch's arguments raised it further; the form kept allocates what 0.1.7
did, plus the flag (11312 B a uniform fill against 10896). Compilation
is 65.4 s against 63.0, within its noise, and the workload's output is
unchanged.

**What it buys on an H200** (job 570290, `bench/copies.jl` from
`bench/symmetry_copies.sh`: a uniform periodic mesh, 20 variables,
`G = 3`, vertex-centered, `Float64`; 0.1.7 measured in the same job
reproduces the downstream's numbers to the digit):

| ns per owned point | 512 × `16³` | 512 × `32³` | 8 × `128³` |
|---|---|---|---|
| `scatter!`, 0.1.7 → now | 0.358 → **0.182** | 0.335 → **0.140** | 0.329 → **0.116** |
| `gather!` | 0.382 → **0.147** | 0.441 → **0.136** | 0.330 → **0.134** |
| `fill_ghosts!` | 1.031 → **0.816** | 0.399 → **0.296** | 0.081 → **0.060** |
| a linear copy of the state, the floor | 0.100 | 0.093 | 0.093 |

On a two-level mesh (960 blocks of `32³`) the fill went 0.481 → 0.362.
The scatter reaches the downstream's static-stride prototype (0.14 at
`32³`) without a block size in the type. On Metal (Apple M3 Pro,
`Float32`, 64 blocks of `16³`) the scatter went 12.4 → 3.1 ns a point
against a floor of 1.9, and the fill 38.9 → 11.8. On the CPU the shaped
launch is kept, and the copies still gained, from reading the state
vector linearly and `@inbounds` (re-timed 2026-10-06 on a quiet machine,
alternating with 0.1.7; Apple M3 Pro, `bench/copies.jl` at 64 blocks of
`16³`, ns a point):

| CPU | scatter | gather | fill | copy floor |
|---|---|---|---|---|
| one thread, 0.1.7 → now | 35.8 → **11.2** | 38.2 → **11.4** | 175 → **138** | 3.0 |
| six threads | 6.7 → **6.0** | 7.0 → **4.5** | 46.7 → **44** | 2.5 |

`bench/ghosts.jl` (10 variables, `8³`): the uniform fill 2.78 → 1.98 ms at
one thread and 0.64 → 0.41 at four; the two-level fill 14.2–15.1 → 12.8
and 4.1–4.6 → 3.73. The full suite took 9m49 for 0.1.7 and 9m57 here at
one thread (1627 more tests), and 9m20 against 8m00 at eight.

**How the index is formed no longer matters** (`bench/copy_index.jl`,
the scatter's copy written seven ways, H200, `16³`, `Float64`): the
`NTuple` index 0.341 ns a point, 64-bit inverses 0.232, 32-bit inverses
0.181, the same with the store's linear index in 32 bits 0.181, shifts
and masks for a power-of-two `N` 0.181, and no arithmetic at all, the
store index read from a table, 0.189; a plain copy 0.100. At `32³` the same
seven are 0.395, 0.225, 0.140, 0.139, 0.139, 0.148 and 0.093. So a
power-of-two block size would buy nothing, and the remaining 1.8× is the
store pattern: rows of `N` values written at an offset of `G` into rows
of `N + 2G + c`. That is the layout's, an open question below
([Open questions](#open-questions), "The working array's layout").

**The fill gained less, and the layout is why** (job 570298,
`bench/copy_groups.jl`: each of the 26 groups of the uniform fill alone,
flat, shaped as through 0.1.7, and through its weights). By the length
of a group's innermost run, which is what the memory system sees:

| H200, `Float64`, 20 variables | 512 × `16³` | 512 × `32³` |
|---|---|---|
| innermost run `N` (faces and edges along the first dimension) | 579 µs, 1.24 TB/s | 1690 µs, 1.54 TB/s |
| innermost run `G + 1` = 4 | 637 µs, 0.54 TB/s | 1562 µs, 0.64 TB/s |
| innermost run `G` = 3 | 619 µs, 0.42 TB/s | 1862 µs, 0.40 TB/s |
| the whole fill | 1716 µs | 4969 µs |

- The groups with long rows are where the index cost was: shaped, the
  `16 × 16 × 4` face took 281 µs, flat 126, and the `32 × 32 × 4` one 1009
  against 424. They now run at a third to a half of the floor.
- The groups with short rows took as long shaped as flat — the index was
  hidden behind their memory traffic — and run at a tenth of the floor.
  They are two thirds of the fill. A warp's 32 values there lie in about
  eleven rows a whole stored row apart, 24 or 32 bytes of each, on both
  the read and the write side, so most of every sector moved is not
  wanted. No way of forming the index changes that; a different layout
  could, which is the open question.
- Skipping a copy's weights is worth 3–10 % of a group (`16 × 16 × 4`: 131
  through the weights, 126 without).

So the downstream's "a fill at copy bandwidth, about 4× faster" is not
reachable by the kernel alone: what the kernel could gain it has, 1.26×
on the fill at `16³` and 1.35× at `32³`, and the rest is the layout's.

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
  this family is predicted and measured below.

There is **no default order**: the
right order depends on the application's differencing order via the
interface-order rule below, which the mesh library cannot know, so
`Operators` requires both orders explicitly. The application chooses
`G`; the package verifies at construction that `N`, `G`, and the
requested operator orders satisfy the invariants listed under
[Blocks](#blocks). Physics-specific operators —
hydro-aware limited interpolation, primitive-variable-based prolongation
— live in application packages and plug into the same interface.

Operators are configured per field set, not per variable — and that
*is* per-variable selection: a field set is the unit of
centering, ghost width and operators alike, so conservative operators for
a density and point-value ones for a velocity are two field sets over one
forest, each with its own schedule (see [Centerings](#centerings)). The
schedule is built from a field set, `GhostSchedule(fs, operators)`, and
records the layout it was built for; field sets with the same layout may
share it, and `fill_ghosts!` checks.

**Interface-order rule:** the interpolation order `p`
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

**The same rule along a stagger.** Repeating
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

**Measured** with the Burgers study of `test/burgers.jl`:
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

**The norm is part of the result**. The rule shows in `L∞` and **not** in an integral
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
form.** The residual the defect
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

**Interface stencils:** near a coarse-fine interface the
symmetric restriction window cannot exist — fine data across the
interface would itself be prolongated coarse data, a circularity.
Restriction therefore **shifts** its window inward (by `p/2 − 1` fine
cells at the ghost layer nearest the interface). Shifting preserves
polynomial exactness (Lagrange interpolation through any `p` distinct
nodes is exact for degree `< p`) and keeps the target inside the node
hull — interpolation, never extrapolation; the implementation asserts
this. 

Prolongation, by
contrast, stays symmetric: it may read the source block's ghosts (that
is what the level-ordered sweep guarantees), at the cost of `G ≥ p/2`.
The conservative family never shifts at all: its restriction window is
exactly a cell's own children, and its prolongation *cannot* shift —
an off-center reconstruction would no longer preserve the containing
cell's average, so the `G ≥ (p−1)/2` bound has no shifted fallback.
Stability does not discriminate between the choices here
(global `dt`, 2:1 balance); damping high-frequency interface modes
remains the job of the application's usual Kreiss–Oliger dissipation.

(Why the order is not reduced instead is in [HISTORY.md](HISTORY.md#operators).)

**Operators per centering**. Because the operator is a tensor
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
is refused; and as the convergence rate of an application
that reads those ghosts, in the vertex-centered wave table under
[the interface-order rule](#operators) above. The last row
is deliberately empty. What a face- or
edge-centered quantity stores is an *average* along its cell-like
dimensions and a *point value* along its vertex-like ones, so along a
vertex dimension the conservative family has nothing to conserve and
would merely interpolate — at an even order that the family's odd `p`
does not name (`p + 1` was the candidate). Rather than
fix that rule before anything exercises it, `GhostSchedule` refuses a
conservative field set with a vertex-like dimension, with a message
saying why. Nothing needs it: fluxes and EMFs are never
ghost-filled or transferred, and constrained-transport `B` needs a
divergence-preserving operator that is not a tensor product in any case.

### Conservation at coarse-fine faces

With a global timestep, conservation requires only that the flux a
coarse cell sees on a coarse-fine face equals the area-weighted sum of
the `2^(D-1)` fine-face fluxes — a purely spatial condition, enforced
within a single RHS evaluation. No flux registers or time-accumulated
corrections are needed (they only exist to bridge subcycled timesteps).

Mechanically this makes a conservative RHS **three steps**: (i) all blocks
compute face fluxes, (ii) fluxes at coarse-fine faces are restricted
onto the coarse side, (iii) all blocks apply the flux divergence.

**Interface restriction** (the
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

*Face directions only*. The
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

**Across a rotating seam**. Nothing crosses it
here. Conformity at the seam (see "Rotating seams" under
[Ghost filling](#ghost-filling)) makes every seam face a same-level
face, so the interface schedule records no transfer across it; it is
built from the oriented neighbor search, and asserts that no
restriction with a nonzero orientation was recorded. Edge directions
are absent from it by design (above), so the seam's level differences
at edges and corners never reach it. What remains is the same-level
argument with the rotation in it. The block on the low face of `d1`
computes the flux through that face from its cells and its rotated
ghosts, and its image on the low face of `d2` computes the flux
through its face from the same numbers; the two are each other's
image, `F_{d1}(0, s) = −F_{d2}(s, 0)` in the plane, so the mass that
leaves through one face enters through the other. The application's
obligation therefore gains one clause: its flux must be covariant
under the rotation, which a flux written in Cartesian components is.
The flux sets themselves, with `G = 0`, have no seam ghosts and fill
alone. The rigid-rotation advection test of M12 puts a number on it.
The interface schedule searches with
`oriented_neighbors`, and a finer neighbor with a nonzero orientation
is an `error` — a bug, since only a forest that bypassed the checked
paths can have one; the test makes one with `refine!` and no
`balance!`. Over a balanced rotating forest it is, restriction for
restriction and bit for bit on random data, the schedule of the same
leaves without the seam, whose seam faces are outer faces. The
advection test, a compact bump carried by `v = (−y, x)` across the seam
and across a refinement boundary, with a linear reconstruction and an
upwind flux chosen by the sign of the normal velocity, conserves the
total to a relative 6e-15 over 102 SSPRK3 steps while 55 % of the mass
crosses the seam, and drifts by 1.9 % without the fixup. The covariance
is exact there: the two seam faces' fluxes are each other's negatives
bit for bit, because `a − b = −(b − a)` in IEEE arithmetic and the
upwind side follows the normal velocity, which the turn negates on one
side. One sharp edge, the application's, not the seam's: a velocity
stored on faces must be set on both of a block's faces, the closed
range; `fill_by_coordinates!` sets owned points only, and a block whose
high face carries a zero velocity leaks 20 % in the same run.

**Construction.** The interface schedule comes out of the same neighbor
search as the ghost schedule — its transfers are the `:restrict` cases
of the neighbor walk for the admissible directions, with the target
range cut down to one plane — and it is batched, sliced and replayed by
the same `TransferGroup` / `run_phase!` machinery on any backend. It
records the forest generation and the field set's layout, and refuses to
run stale or on a different layout, as the ghost schedule does.

`InterfaceSchedule(fs)` and
`restrict_interfaces!(fs, isched)`, in `src/interfaces.jl`; no new
kernel, no new struct beyond the schedule itself. Four things the design above leaves
open are settled in it:

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
  them**. No plane the fixup writes is a plane it reads, over
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

**Measured.** Burgers' equation (`test/burgers.jl`) is the
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
2. **Buffering**: a flagging function may report, with **any**
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
   reads that parent's ghost layers. Each field set moves
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

**Conservation of the transfer**: coarsening conserves
the volume integral of *any* field exactly when restriction is the
order-2 average; untouched blocks are copied bit for bit; refinement
conserves exactly those fields the operators reproduce exactly, and
**not** others — prolongation is not locally conservative, and at a
refinement boundary the fine region draws on neighbor values through the
parent's ghosts without those neighbors giving anything up
(percent-level drift measured for a field discontinuous across a
periodic seam). Selecting the **conservative operator family** (see
[Operators](#operators)) makes the transfer exactly conservative for
any field; that family is verified exactly
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

(`src/interpolate.jl`.) Everything above
moves data between the mesh's own points. An analysis needs the other
direction as well — the field, and its gradient, at points the mesh did
not choose: a horizon finder's trial surface, asked for some 500 points
about 55 times per find; a tracer; a sampled ray. `interpolate(fs, xs,
basis; derivs, vars, exclude)` answers that for a whole batch in one
launch on the field set's backend, and `locate_point(forest, x)` is its
first step on its own.

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
  so no leaf lies between the covering leaf and the node. The
  comparison is `curve_less` on `(root, padded coordinates, level)`, the
  same function `isless` on keys now calls, because the checking key
  constructor cannot run in a kernel.
- **Folding at faces.** Along a periodic dimension the point is wrapped
  into the domain — and the *wrapped* point is what the stencil is built
  from. Beyond a reflecting face it is mirrored once, `x →
  2w − x` (a symmetric run's horizon finder queries
  across the wall), and the value takes the variable's parity sign from
  `fs.factors` — the table the mirrored transfers already multiply by —
  and each derivative across the wall one sign more. A point outside the
  domain after that is refused rather than taken from the nearest block,
  which would extrapolate without saying so.
- **Folding through a rotating seam**. After
  the periodic and reflecting folds, a point beyond the seam, in
  `R^r` of the domain, is rotated back, `q = R^{−r} p`, and the stencil
  is built at `q`. The rotating dimensions are neither periodic nor
  reflecting, so the folds act on different coordinates and their order
  is a convention. The value of variable `v` is variable `σ_r(v)` at `q`
  times the sign from the factor table, the two tables the rotated
  transfers use. Each first derivative is remapped as well, by the
  chain rule through `R^{−r}`: for `r = 1`, `∂_{d1}` at `p` is
  `−∂_{d2}` at `q` and `∂_{d2}` is `∂_{d1}`; for `r = 2` both are
  negated; for `r = 3`, `∂_{d1}` is `∂_{d2}` and `∂_{d2}` is `−∂_{d1}`.
  A derivative of any order turns once per order, so a second one
  takes the product of its two signs: for `r = 1`, `∂_{d1}∂_{d2}` is
  `−∂_{d2}∂_{d1}` and `∂_{d1}²` is `∂_{d2}²`; the kernel's sign is per
  order, and `rotate_tests.jl` checks the Hessian of the metric's
  `g_xz` beyond the seam bit for bit.
  `PointGeometry` carries the rotation, so `locate_point` and M7's host
  routing share the fold, and a point that the rotation takes beyond a
  high face is outside and refused. A set with an asymmetric layout,
  one of a `RotationPair` or a ghost-free flux set, refuses a point
  beyond the seam with the reason: its value there is its partner's,
  which the kernel does not read (an open question, see
  [Open questions](#open-questions)).
  `fold_point` returns the orientation with the folded point, and the
  turn is taken about the low corner of the plane, `(a, b) ↦ (b, −a)`,
  `(−a, −b)` and `(−b, a)` for `r = 1, 2, 3`. An odd `r` contracts with
  the requested multi-indices with `d1` and `d2` exchanged, a second
  compile-time tuple beside `derivs`, so that the selections stay
  static (run-time ones cost M11 21 %); each derivative takes the turn's
  sign `(−1)^{m_{d1}}`, `(−1)^{m_{d1}+m_{d2}}` or `(−1)^{m_{d2}}`, which
  holds at any order, so second derivatives need nothing more here. A set that cannot turn a point — an asymmetric layout — gets no
  variable table in the kernel, which marks such a point `−1` beside an
  outside point's `0`; the host decides which refusal it is with the
  same fold (`outside_error`, whose face labels now name the seam). M7's
  host routing marks it the same way. `exclude` is tested where the
  stencil is read, as for a mirrored point. One finding about the
  comparison, not the fold: the full plane agrees with the quadrant at a
  turned point to roundoff for cell centering, but only to
  interpolation accuracy for vertex centering, since a vertex-like full
  plane is not itself covariant under the turn where levels meet — its
  ghost stencils depend on the side, half-open ownership putting the
  shared plane on a block's high one — while the quadrant is covariant
  by construction. The test therefore compares with the full plane at
  the preimage, turned; the numbers are under step 5 of M12 in [HISTORY.md](HISTORY.md#m12--rotating-symmetry).
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
- **Derivatives as multi-indices, up to second order**. `derivs` is a tuple of
  `D`-component multi-indices in physical units. The weights, the
  contraction, the `h^{|m|}` scaling and the mirror signs are all
  written for any order; one check refused `|m| ≥ 2` until tests
  claimed it, and now refuses `|m| ≥ 3` for the same reason. Second
  derivatives needed no change of interface and none of the kernel:
  only the check moved. A derivative across a reflecting wall takes one
  sign per order across it, so `∂ₓ²` across a wall in `x` keeps the
  variable's parity and `∂ₓ∂ᵧ` flips it. The rate of an `m`-th
  derivative is **`min(n − max_a m_a, p − |m|)`**: `Lagrange(n)` loses
  one order per derivative along a dimension, so a mixed `∂ₓ∂ᵧ` keeps
  `n − 1`, but near a coarse-fine face the stencil reads ghosts with
  the exchange's `O(hᵖ)` error, which any derivative of total order
  `|m|` divides by `h^{|m|}`. So the full mixed rate through
  `Lagrange(n)` needs `p ≥ n + 1`, and a pure second derivative
  `p ≥ n`; the usual `p = n` for even `n` holds `∂ₓ∂ᵧ` to `n − 2`
  (measured below, under M11).
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
  value is computed either way and the caller decides. A sampler may want to ignore the flag. The region is an
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
  exception would arrive wrapped in a `TaskFailedException`.
- **One launch over points.** Points are not blocks, so this is a plain
  launch, not a by-owner one; on the CPU the workgroup is sized to give
  every thread a share, since a batch of a few hundred points would
  otherwise fit one default workgroup and run on one thread. Every query
  writes only its own slots, so the result is bit-identical across
  thread counts (a line of `test/thread_workload.jl` says so). The
  leaves, origins and spacings are uploaded per call, which at analysis
  cadence costs nothing that matters.

Over a distributed forest the leaves a rank holds are its own, so a
query must first be routed to the rank that owns its block. The
location already produces exactly the block index that routing needs:
the location stays on the host against the replicated leaves, two
`alltoallv`s route the points to their owners and the values back, and
the kernel is unchanged except for one argument, the block offset,
which is 0 serially (see [Distributed meshes](#distributed-meshes)).
Over a distributed forest an outside point is found on the host before
anything is routed, and refused on every rank there, rather than after
the launch. A device batch goes through the host for the messages.

### Checkpoint and restart

Long runs outlast a queue's day: TreeHydro's showcase on an H200 at
10–20 levels, and TreeGeneralizedHarmonic's production runs, estimated
at 38–149 h. An application calls one function at a chunk boundary to
save the forest, its evolved field sets and its own plain data; a fresh
process loads them exactly, on any thread count and any backend, and
continues **bit-identically** at any thread count (across backends,
see "Loading" below). A file this version cannot interpret is refused
with the reason, and with a way to recreate the environment that wrote
it.

**What is saved and what is rebuilt**. A checkpoint holds
what cannot be recomputed, and nothing else:

| Item | Handling | Why |
|---|---|---|
| Forest: `D`, the geometry type, `roots`, `periodic`, `reflecting`, `rotating`, `extents` (bitwise, in the geometry type), `N` | saved | the inputs the forest was built from |
| Forest: the leaf list | saved | the only record of the mesh's history |
| Forest: `generation` | not saved | a staleness counter, meaningless in another process |
| Field set: element type, `nvars`, `G`, `centering`, `parity`, `rotation` | saved | its layout |
| Field set: the **owned** points, in state-vector layout `(N, …, N, nvars, nblocks)` | saved | the authoritative data |
| Ghosts, shared vertex planes, the derived wall plane | rebuilt by `scatter!` and `fill_ghosts!` with the application's hook | derived from the owned points |
| `GhostSchedule`, `InterfaceSchedule`, `Operators`, the parity factors, the rotation tables, a `RotationPair` | rebuilt | derived, or the application's inputs |
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

**File layout, format version 1** (read by every later version — version 2, the files per I/O
process, follows the version-1 bullets). One HDF5 file. Everything
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
                                [, geometry_limbtype, geometry_nlimbs],
                                leaves_crc32c (M7; absent before)
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
        data_crc32c      UInt32[nblocks] (M7; absent before)
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
  size matters** (the numbers and the reasons are
  under "Throughput and filters" below).
- `range = "owned"` says the data are the owned points only, and leaves
  room for a file that stores more, which a version-1 reader would
  refuse by value.
- **Checksums** (why they were added is under "Parallel
  checkpoints" in [HISTORY.md](HISTORY.md#distributed-meshes)).
  `leaves_crc32c` is the CRC-32C (Castagnoli) of the bytes of `root`,
  then `level`, then `coords`, each column whole, as stored; entry `b`
  of `data_crc32c` is the CRC-32C of block `b`'s bytes in `data`, the
  slice `data[…, b]` with all its variables and limbs, in the file's
  order. Both are over the little-endian bytes the file holds, so a C or
  Python reader can check them. A load verifies both and refuses a
  mismatch, with the reason that the file was damaged while or after it
  was written; over several ranks each rank checks the blocks it reads,
  and the verdict is agreed. A file without them, from before, loads
  unchecked.
  - *No format change*. They are an additive change with an
    obvious default, "not checked", so `format_version` stays 1 and
    nothing joins `features`: a reader that ignores them reads the file
    exactly as before. A feature would have made every file M7 writes
    unreadable to 0.1.4 for nothing a reader must understand. The 0.1.4
    reader was checked to load files with them, filtered and not.
  - *CRC-32C*, because it is in Julia's standard library (`CRC32c`,
    which TreeAMR now depends on), runs in hardware on x86-64 and
    AArch64 (13.7 GB/s on one thread of the development laptop, against
    the 1–2 GB/s of a save), and is defined outside Julia.
  - *Per block*, not per rank or per file, because the file stores no
    partition and loads on any rank count: each rank checks exactly the
    blocks it reads, and the checksums are written beside the data by
    the same slab. The leaf list, which every rank reads whole, has one.
  - *Not covered*: the extents, the provenance and the plain data, which
    are small and written by rank 0 alone, and HDF5's own metadata.
  - *Not HDF5's own checksums*. HDF5's Fletcher-32 filter was considered and is not
    used, for four reasons. It accepts an all-zero chunk, trailer
    included: on libhdf5 2.2.0 a chunk overwritten with zeros through
    `H5Dwrite_chunk` read back without an error, while one nonzero byte
    with a zero trailer was refused — and whole zeroed slabs are exactly
    the loss seen on BeeGFS, whereas the CRC-32C of zeros is not zero. Its
    checksum lives inside the chunk, so an intact chunk in the wrong
    place, or one left from an earlier write, verifies; ours are stored
    apart from the data and indexed by block. It needs chunked storage,
    which would leave the contiguous, unfiltered datasets uncovered. And
    it is weaker: 16-bit words, with 0x0000 and 0xFFFF ambiguous. HDF5's
    checksums of its own metadata, which the newer format structures
    carry (through the library-version bounds), are not enabled either.
  - *Over the parts*. A part file stores each field set's
    `data_crc32c` for its blocks, and the index's part table stores a
    CRC-32C of each of those arrays, so the index vouches for the
    per-block checksums and they for the data: a part from another save,
    or one whose checksums were damaged, is refused before its data are
    read.
- **Spellings**. Shapes above are in
  Julia's order, which C sees reversed. Every `Bool` in the file —
  `periodic`, `reflecting`, a plain-data `Bool` — is a `UInt8`, 0 or 1,
  because HDF5.jl would write an HDF5 bitfield, which other readers
  handle poorly. A `(lo, hi)` pair per dimension is a `(2, D)` array,
  which C sees as `(D, 2)`. The other scalars and small arrays are
  `Int64`, the leaf columns excepted. A `Float16`, which HDF5.jl does
  not predefine, is the IEEE half type built as h5py builds it, and a
  `Complex` is the compound `(r, i)`, as HDF5.jl and h5py spell it.

**File layout, format version 2** (the design is "Checkpoints without parallel I/O" under
[Distributed meshes](#distributed-meshes)). A checkpoint is an *index
file* `path` and one *part file* per I/O process, `path.<saveid>.<j>.h5`;
with one I/O process the single part lives inside the index, so a
serial checkpoint is one file. The index is the version-1 file with the
field data moved out:

    path:
    /TreeAMR.jl/         attrs: format = "TreeAMR checkpoint",
                                format_version = 2, features = ["brick"],
                                application, save_id (32 hex digits)
      provenance/        as in version 1, and nparts (1 if absent)
      forest/            as in version 1, leaves_crc32c included
      fieldsets/<name>/  the layout attributes of version 1; no datasets
      parttable/         file        String[k]  (relative; "" for the
                                                 part inside the index)
                         first_block Int64[k]   (global leaf index, from 1)
                         last_block  Int64[k]   (first_block − 1 if empty)
                         bytes       Int64[k]   (the part file's size; 0
                                                 for the inline part)
                         fieldsets   String[s]  (the field sets, in order)
                         data_crc32c UInt32[s, k] (CRC-32C of each part's
                                                 data_crc32c, per set)
      parts/<jjjj>       an external link to `<file>:/TreeAMR.jl` per part
                         file, for tools; with one part, the part group
                         itself, inline (`jjjj` zero-padded to 4 digits)
    /<app name>/         as in version 1

    path.<saveid>.<j>.h5:
    /TreeAMR.jl/         attrs: format = "TreeAMR checkpoint part",
                                format_version = 2, save_id, part = j,
                                first_block, last_block
      fieldsets/<name>/
        data             (limbs?, N, …, N, nvars, last − first + 1)
        data_crc32c      UInt32[last − first + 1]

The inline part, `/TreeAMR.jl/parts/0000` of the index, has the part
file's `/TreeAMR.jl` attributes and contents. The datasets keep version
1's spellings, chunking and filters; a part's block axis is its own
range, so block `b` of the forest is entry `b − first_block + 1` of the
part that holds it. The parts tile `1:nleaves` in order, as the ranks'
ranges do.

**Rotating forests** (no format bump). A forest
with a rotating seam writes a `rotating` attribute in `forest/`, the
pair `(d1, d2)`, beside `reflecting`, and each of its field sets a
`rotation` attribute in `fieldsets/<name>/`, the signed map as
`Int64[nvars]`, beside `parity`. Such a file also lists `rotating` in
`features`, so a reader from before M12 refuses it, with the reason,
instead of loading the seam as two outer faces; a forest without a seam
writes neither the attributes nor the feature, so every file it writes
is readable by every earlier reader of its version. A file without the
attributes loads as non-rotating, which is the only reading of their
absence, so the version-1 fixtures still load. The load builds the
forest through the checked `leaves` path, which refuses a leaf list that
is not conforming at the seam, and the field set through its
constructor, which refuses a map that is not a valid rotation. A
`RotationPair` is not saved: like the operators, it is the application's
input, rebuilt from the two loaded sets. The attribute and the feature go together, and a file with
one and not the other — which no TreeAMR writes — is refused as
damaged, rather than read with the seam or without it. The pair is
`Int64[2]`, checked by the forest's constructor as a caller's would be;
a map is checked for its length here and as a map by the field set's.
The map joins what the ranks agree on about a field set before a save,
beside the parity.

**Element types**.

- **Native types** — `Float16`, `Float32`, `Float64`, the signed and
  unsigned integers, `Bool`, and `Complex` of those — are stored as
  themselves, bit for bit.
- **An `isbits` type made of one native type throughout**, with no
  padding, is stored as **limbs**: a leading dimension of that native
  type, with `eltype` naming the type, `limbtype` the native type and
  `nlimbs` the count. MultiFloats' `Float32x2`, which the test suite
  uses (see [Precision](#precision)), is two `Float32` limbs. The name
  is the type as a module importing nothing but Base prints it,
  `MultiFloats.MultiFloat{Float32, 2}` (not `string(T)`, which qualifies a name or not according
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

**Versioning**. This is also the answer to how files from
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
  lists one, `brick`. M12 adds `rotating`, listed only by a file whose
  forest has the seam (see "Rotating forests" above).
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

**Writing**.

- **Atomically.** The file is written to `path * ".partial"` and moved
  over `path` when it is complete; on any error the partial file is
  removed and the error rethrown. A crash while writing then cannot
  destroy the previous checkpoint, which is the one a restart needs.
  With part files, the parts are written and flushed first,
  under a fresh save id that no index names, and the rename of the index
  is the commit point; the previous index's parts are deleted only after
  it. See "Checkpoints without parallel I/O".
- **Durably, by default** (`sync = true`). Closing a
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

**Loading**.

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
  HDF5's. Over a distributed forest each I/O process compresses its
  group's chunks, so a filter parallelizes over the I/O processes —
  every rank under `io = :all`, one per node under the default.
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
  parallel-I/O measurement. (They are under M7's steps 6, for the shared
  file, and 6b, for the part files that replaced it, in
  [HISTORY.md](HISTORY.md#milestones).)

The facts about parallel HDF5 checked before M7, the shared-file design they led to, and why it was replaced by the part files are in [HISTORY.md](HISTORY.md#checkpoint-and-restart) and under "Checkpoints without parallel I/O" in [Distributed meshes](#distributed-meshes).

**Multi-block**. Keys are relative to their root,
`connectivity = "brick"` is a tagged record rather than an assumption,
and no coordinates are stored. A conforming multi-block forest, the
likely route to spherical domains, is then a new connectivity kind,
with whatever describes its roots stored beside the leaves, plus a
feature name. Nothing in the format obstructs it. Parthenon's layout
would: it stores global tree locations in one virtual tree over the
root grid, which a multi-block forest cannot express.

The survey of existing formats that this layout follows from (formats considered, and the one modelled) is in [HISTORY.md](HISTORY.md#checkpoint-and-restart).

**The API in brief**. HDF5 is a **package extension**,
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
                    sync = true,                     # flush to stable storage
                    io = :node)                      # I/O processes
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

**Where a checkpoint belongs in a chunked driver**. The
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

Indicative only — names and signatures will evolve:

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

**Three launch ranges, not two.** `map_blocks!`
launches over the owned range `N` or, with `closed = true`, the closed
range `N + c_d`, and a third: `stored = true` launches over every
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

**An all-variables form of the coordinate callbacks**. `fill_by_coordinates!(f, fs)` calls `f(x, v)` once per point
*and variable* — its kernel has a variable axis in the ndrange — and
`CellBoundary(g)` calls `g(x, v, δ)` the same way. Beside those, and
without changing them, there is a form called once per *point* that
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

**The state vector contains interiors only**. Each RHS
evaluation:

1. scatters `u` into the working array,
2. fills ghosts (copies, restrictions, prolongations),
3. runs the application's kernels,
4. writes `du` in state layout — the gather can be fused into the
   compute kernels, since they write interior cells only.

The integrator never sees ghosts and the RHS never mutates `u`; the cost
is one scatter per RHS evaluation, which we accept. (The rejected alternative is in
[HISTORY.md](HISTORY.md#time-integration).)

**Several field sets in one state vector**. Burgers has one
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

Coupling details: the application writes `f!(du, u, p, t)`
itself, calling `scatter!` → `fill_ghosts!` → `map_blocks!` explicitly —
no `semidiscretize`-style wrapper until the pattern has stabilized. The
flat vector `u` is the authoritative data; the working array is scratch,
refreshed at every RHS evaluation (output and analysis scatter and
ghost-fill first). 

Adaptive integrators need a volume-weighted `internalnorm` — the default
norm weights fine regions more, simply because they contribute more
entries per volume.

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

  **Bit-identical results, not merely equal to roundoff** (narrowed by the next paragraph). No parallel loop shares an
  accumulator: each writes its own slot, and every reduction forms one
  partial per block and sums the partials in block order. The chunking
  is a function of the item count and the thread count alone. So a
  64-thread run reproduces a serial one exactly — worth the discipline,
  because it makes "the thread count" something a debugging session
  never has to consider. Collecting passes (the neighbor search, the
  balance scan, the buffer dilation) follow the same rule: each task
  fills a buffer of its own, and the buffers are concatenated in block
  order.

  **Floating-point sums are promised to roundoff only**. The paragraph above bundles two rules under one name, and only
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

  **A phase is one parallel loop, not a sequence of launches.** Ghost
  transfers are batched by stencil, and the batches differ
  in size by orders of magnitude — a face slab is `G·N^(D-1)` cells, a
  corner `G^D`. Launching the batches one after another leaves the small
  ones with a single workgroup each, i.e. serial, which measured as a
  hard ceiling of ~2.5x on the ghost fill however many threads were
  available, while the single-launch parts of the same step scaled
  fine. So the phase is one parallel
  loop and every batch is split across all threads, by owner. Each thread takes the transfers whose target
  blocks it owns, because slices dealt to whichever task was free moved
  every block's ghosts to a new core in every fill; see "What one
  process loses". The
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
  than implemented. (Since every per-block
  pass runs each block on its owner's thread, first touch lands a
  block on the domain that computes on it, and with pinned threads that
  beats interleaving; see "What one process loses".)

  (Pinning KernelAbstractions to its static schedule alone, the alternative measured and rejected before the ownership policy, is in [HISTORY.md](HISTORY.md#parallelism).)

  **Inconsistent:** CLAUDE.md said that running unpinned under
  `numactl --interleave=all`, "which is what the M5 numbers in CODE.md
  were taken with", "is worth 3–7x over unpinned first touch there",
  against the 2–6x stated above.

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

  The copies of the last three columns were
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

  What this means for the plan: NUMA-aware page placement
  inside one process is not worth building; it buys 1.2x and
  duplicates M7. One MPI rank per NUMA domain would recover the gap,
  but so does one process: the gap is 2.2x,
  not 3x, and it is not a per-process limit but the loss of
  data-to-core affinity between launches, which a block-ownership
  launch policy recovers entirely inside one process (next paragraph).
  M7 therefore gains no bandwidth argument from this, and loses none
  (on one Symmetry node one pinned 64-thread
  process runs the RHS 1.5–1.6 times as fast as 8 ranks of 8 threads
  over the same mesh, the ranks paying the packs and unpacks of their
  halos):
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

  V0 is the package before the change below. V1 puts every package launch on the
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

  *What changed (measured below).* One
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

  **The backend is chosen once, at allocation**. It is
  a keyword on `FieldSet` and on `GhostSchedule`, and nothing else takes
  one: every kernel in the package already read its backend off the
  storage it was handed (`get_backend(fs.work)`), so the storage
  decision *is* the backend decision. `statevector` allocates where its
  field set lives and `regrid!` reallocates there, which is what keeps
  an application's RHS — `scatter!` → `fill_ghosts!` → `map_blocks!` —
  literally the same code on a device. No device package is a dependency
  of TreeAMR; `KernelAbstractions.allocate` is the whole interface.

  **The schedule has to move with the data**. "All kernels are KA kernels" is
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

  **Two application callbacks needed a second form**. M5
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
    the block's interior needs — extrapolating outflow — and
    is CPU-only, which it says if handed a device field set. That
    limitation is real and is not papered over: outer boundaries that
    read their own interior are a CPU-only capability until the cell
    form grows an interior accessor. Reflection is not among them: it
    is a property of the domain and a transfer in
    the schedule, so it runs on every backend; see
    [Ghost filling](#ghost-filling). Extrapolating outflow, and anything
    else that reads the interior, is what remains CPU-only.
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
    step 2 dilates. It is the two-launch reduction under
    "**Implemented**" below, 256 lanes per block, and exact: integer
    min/max is order-independent.

  **The coordinate callbacks got a third form, over the variable axis**. The two forms above are about *where* a
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

  **The reduction became public, as `block_mapreduce`.** It
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

  The guarantee is stated as what it is (narrowed, see
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

  - **The two per-block reductions were the weak rows.** One work item
    per block was chosen for determinism (the argument is in
    [HISTORY.md](HISTORY.md#parallelism)); the trade is no longer cheap
    and the property is no longer promised. TreeHydro takes a signal-speed maximum at
    every step for its CFL condition, through `block_mapreduce`; at the
    table's sizes that is 3.3 RHS evaluations per step on the device,
    which under a three-stage integrator doubles the step. And the
    argument was only half right on its own terms: `firing_boxes`
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
  form.** Two pieces, both made admissible by the
  narrowing of the bit-identity claim above and both wanted before M7.

  - **`block_mapreduce` stays as it is**: per-block, local to the
    process, a host `Vector`. Refinement criteria are per-block by
    nature — TreeWave's per-variable peaks and hot-cell counts,
    TreeHydro's floor counts — and under MPI they need no communication,
    which is why the per-block form must survive as the local one. Only
    its contract changed, as stated above; its *device path* is what
    gets rewritten.
  - **The device path becomes 256 lanes per block, in two launches and
    with no barrier**. The first launch has an `ndrange`
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
    sums. The lanes were first a static workgroup of 256 and
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
    when a weight is given, combined over the local blocks and across
    ranks. The weight is a host
    function of the block's key, applied on the host to the per-block
    values before they are combined, because that is where the geometry
    is and because the cross-block stage is `nblocks` numbers and not
    worth a launch; it is meant for sums (a cell volume,
    `spacing(key)^D`) and is documented as such. `volume_weighted_norm`
    and `total_mass` become calls to it, which is how they are global
    without changing signature; the cross-rank step lives in
    `mesh_mapreduce`'s combination step and nowhere else — the norm's
    domain volume goes through the same step — since the mesh owns the
    communicator and an application must not be asked to. The name sits beside `block_mapreduce`: one returns a value per
    block, the other one value for the mesh. What it does across ranks is an `allgather` of one
    partial per rank, folded in rank order, with no MPI operation at
    all — custom operations fail on ARM, and a builtin one exists only
    for MPI's native types. See "Reductions" under
    [Distributed meshes](#distributed-meshes).

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
  ranks; a transfer is evaluated on the owner of its **source** — the sender
  computes — so a restriction runs on the fine side and a prolongation
  on the coarse side (see [Distributed meshes](#distributed-meshes)).
  Regridding rebuilds and repartitions the curve. With a global `dt` and
  uniform blocks every block costs the same, so partitioning by equal
  block counts along the curve is already load-balanced. The exchange is
  layout-generic before MPI exists, so ghost
  fill, interface restriction and regrid transfer for every centering
  are distributed by one design. CUDA-aware MPI
  for GPU+MPI.

### Distributed meshes

M7 runs one forest over several
processes. It came after M8 and M10 on purpose: every centering, the
interface restriction and the mirrored transfers at reflecting faces
are now entries of one schedule, so distributing the schedule
distributes all of them at once. The claim extends the one
[Parallelism](#parallelism) makes for threads. The leaf array, the
schedule, the state vector gathered in curve order, and every max, min
and integer reduction are bit-identical to a serial run at any rank
count. A floating-point sum is reproducible to roundoff across rank
counts, and at one rank it is exactly the serial value.

**What is replicated and what is distributed**. Every rank holds the whole forest, `forest.leaves` included, and
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

What replication costs at thousands of ranks, measured in one process
at up to 180224 leaves: the schedule build stays flat at a fixed
number of blocks per rank; each rank searches the buffer from its own
sources and classifies only the new leaves it needs, and a rank's
regrid stayed at its 1408-leaf cost up to 180224 leaves. On Symmetry,
at up to 32 ranks on four nodes the schedule build and the refining
regrid are flat from 2 ranks on, and `bench/replicated.jl` on a rank's
domain reproduces the flat regrid to 180224 leaves; the replicated
completion after the buffer, 20 ms there when every block is a source,
is what still grows.

**The communicator layer** (MPI a weak dependency). A file, `src/communicator.jl`, sits after `device.jl`
in the layer order.

- It defines an abstract `Communicator` and the default
  `SerialCommunicator`, rank 0 of 1, which sends nothing.
- The package talks to it through a few internal verbs: `commrank`,
  `commsize`, `allgather` of one `isbits` value, `allgatherv` of a
  vector, `alltoallv`, and nonblocking `isend` / `irecv` with
  `waitall` over flat buffers. Two more,
  `commnodes`, the number of shared-memory nodes, and `bcast` of one
  rank's vector, serve the checkpoints without parallel I/O.
- Every verb has a serial method, so `src/` never branches on whether
  MPI is there. A serial run takes the distributed code path with every
  message empty, and the existing suite is its test.
- The MPI methods live in a package extension, `TreeAMRMPIExt`,
  triggered by MPI.jl, as HDF5 is for checkpoints: an application that
  never runs distributed does not load MPI. It adds `MPICommunicator`,
  which holds a duplicate of the application's communicator (below),
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
  1.13). MPI.jl's own `Comm_dup` attaches a finalizer that
  frees the duplicate, so the extension calls `MPI_Comm_dup` through
  `MPI.API` and attaches none. The cache is keyed by the handle; since
  MPI may reuse a handle once the application frees its communicator, a
  hit is used only while `MPI_Comm_compare` still finds the duplicate
  `CONGRUENT` with the communicator passed — a local call whose answer
  is the same on every rank of the group.
- *MPI is called from the calling task only*, never inside a threaded
  loop or a kernel. A Julia task may still migrate between OS threads
  between two calls, so the extension requires
  `MPI_THREAD_SERIALIZED` or better — `MPI.Init()`'s default — and
  refuses a communicator initialized with less, saying why.
- *Why the field is abstract.* `Forest` has `comm::Communicator`,
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
`nleaves` and a fold of the leaves' hashes (below). A forest whose digest differs between ranks is
refused, naming the ranks and saying that every forest mutation is
collective. At regrid frequency that is one `O(nleaves)` pass and one
collective.

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
  is rethrown at once. `regrid!`'s checks run
  through the same gather, and the `DimensionMismatch` it raises
  serially for a flag vector of the wrong length is agreed too, so that
  error keeps its type on the rank that raised it.
- *Serially nothing is gathered*: a forest over one rank skips the
  digest and its `O(nleaves)` pass, so a serial build is unchanged.
- *The verdict is the same on every rank* because it is computed from
  the same gathered digests: the message names rank 0's forest and the
  ranks that differ from it, whichever rank prints it.

**The partition**. Rank `r` owns the contiguous leaf range
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
  block".
- **Ranks without blocks are allowed.** Such a rank takes part in every
  collective, enters no exchange stage, and contributes nothing to a
  reduction (below). Every launch guards an empty `ndrange`. Empty
  ranks occur whenever ranks outnumber leaves, which a coarse initial
  mesh on many ranks does for a while, and the workload tests one.


The workload has a case, `E2`, with the same reach
  at three ranks: two leaves against a wall and outer faces, through
  both all-variables forms, the wave equation, the interface
  restriction, interpolation, regrids that give the empty rank blocks
  and take them away, the initial-data cycle and a checkpoint.

- **What an application must make global itself.** `block_mapreduce`
  stays local by design, so a value an application combines from it —
  TreeHydro's CFL signal speed, TreeWave's per-variable refinement
  scale — becomes a per-rank value under MPI. A `dt` that differs
  between ranks desynchronizes the run, and a rank-dependent scale
  makes the mesh depend on the rank count. Such a value goes through
  `mesh_mapreduce`. So does an adaptive integrator's error norm: it has
  to be a global one (`volume_weighted_norm`, the open `internalnorm`
  question), or each rank chooses its own step.

**The exchange: the sender computes**. A
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
  what the tests check.
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
reused. `run_stage!` in `ghosts.jl` runs these five steps
through the communicator verbs. A stage without messages is the serial
phase, the same launch and the same barrier. Within a stage, nothing a pack or a local group reads is
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
- The tests check that rank `r`'s send layout to `s` is `s`'s receive
  layout from `r`.

**Pack and unpack are transfers.** Both go through `transfer_kernel!`,
which keeps "one kernel for every transfer" true.

- *The packed buffer is a kernel argument.* It is passed in place of
  the working array as a small `NamedTuple` — the flat buffer and each
  transfer's offset into it — which KernelAbstractions adapts to a
  device without a new dependency. The kernel's
  tuple has a third field, `dims`, the target box times `nvars`, since
  a slot's linear index needs the box's shape. The offsets are stored
  in points, not elements, and multiplied by `nvars` when the buffer is
  addressed, because a schedule belongs to a layout and serves field
  sets of any variable count. A driver passes `(buf, offsets)`, and
  `run_group!` adds the group's `dims`. The "block" index the kernel
  hands the accessor is the slot.
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
- **The parity factor is applied when unpacking**. An unpack goes
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
  kernel writes. The round trip test checks this bitwise, `Float32x2`
  included, for which it also needs `0 + 1·x` to be the identity on
  normalized limbs. (Measured: with the factor moved to the
  pack, the `Float64` round trip fails in every 2D case tried, as
  argued. `Float32x2` cannot tell the two apart: MultiFloats' product
  returns `+0` for `0 · (−1)`, so its serial fill has no `−0` to lose.
  Nor is `0 + 1·x` the identity on every limb pair — it maps `(−0, −0)`
  to `(+0, +0)` — but it held on every value a pack produced in the
  round trip, which is a check over the tested cases and not a proof.
  In 1D every mirrored transfer is the block's own reflection,
  so the remote `−0` case exists from 2D on.)
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
backend and are allocated once per schedule and variable count, on
first use, for the reason the offsets are in points. They go to MPI
directly when the communicator is device-aware (below). Otherwise they
are staged through host buffers of the same layout, which is the path
Metal always takes and the one the suite tests. The synchronization
after the pack is what makes a device buffer safe to send.
`MPI.has_cuda()` asks only Open MPI (and answers `true` for IBM
Spectrum MPI); for any MPICH it is `false` unless the environment
variable `JULIA_MPI_HAS_CUDA` says otherwise (MPI.jl 0.20.27's
`environment.jl`). The MPI on Symmetry's H200 nodes is CUDA-aware:
HPC-X 2.20's Open MPI 4.1.7, for which `MPI.has_cuda()` is `true`; both
paths pass there, and the direct one saves the two copies, 0.16–0.2 ms
of a 2.5 ms fill on four H200s.

- *Who decides*. The
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
  is a verb of the communicator layer, and the only one whose
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
  nothing, Metal's memory being unified. No weak dependency on a device package was needed.
  A regrid stage's buffers and mirrors, and those
  of the schedules rebuilt after a regrid, come from the forest's
  buffer pool, the next bullet, so they are allocated and page-locked
  once rather than at every regrid.
- *The buffer pool* (for the staged
  regrid, which Symmetry measured at 48 ms against 8.7 ms direct).
  Every regrid built its transfer stage afresh, and the schedules an
  application rebuilds after it start without buffers, so every
  distributed regrid allocated and zero-filled the device buffers of
  its own stage and of the next schedule's first fill, and, staged,
  allocated and page-locked their host mirrors. The pool keeps them.
  - *Where it lives.* In the forest, whose leaves every schedule and
    regrid is over and which `run_stage!` already has: a mutable
    `ForestState` holds the generation, which was a `Ref` of its own
    before, and the pool, made on first use. One object for both keeps
    `Forest` the size it was. A ninth field made the schedule build
    allocate 80 and 4816 bytes more in `bench/ghosts.jl` (presumably
    closures that capture a `Forest` by value growing by a word; not
    traced), and with the fold its 392832 and 4656880 are unchanged. A
    serial forest never makes a pool. The cost is the struct's size, as the guess about closures implies,
    not its field count. A ninth field of 16 bytes, `NTuple{2,Int}`,
    added 160 and 4896 bytes, twice the 80 the 8-byte one had added to
    the uniform build; one of 2 bytes, `NTuple{2,Int8}`, fits in the
    padding before `extents` for `D ≤ 4` and adds nothing. So the ninth
    field stays, as two `Int8`s. The same measurement found the other
    half of the guess: a closure that captures a `Forest` copies it,
    and a seam check that read the forest inside `GhostSchedule`'s
    argument closure added 240 bytes, twice `sizeof(Forest{3,Float64})`,
    until it read the pair taken outside it.
  - *What it hands out.* A *lease* of at least the length asked for, as
    an object of exactly that length over pooled memory, keyed by role
    (buffer or mirror), array type and backend type. A host vector — a
    CPU stage buffer or any mirror — is an `Array` made by `Base.wrap`
    over the pooled `Memory`, so `hoststaging`'s test for an `Array` and
    the MPI extension's contiguous-host-vector check see what they saw
    before; a device buffer is a contiguous `view`, which CUDA and Metal
    (and every GPUArrays backend) return as an array of the parent's own
    type, so the kernels and the device-aware path see a `CuArray` or an
    `MtlArray` as before. A backend whose view is of another type gets
    an ordinary allocation. A pooled mirror is page-locked once, when it
    is allocated.
  - *When a lease comes back.* The regrid stage gives its leases back
    after its sends have been waited on, at the end of the field set's
    transfer. A schedule's leases are reclaimed at the next lease after
    the forest's generation has moved past the one they were taken at,
    or after their holder has been collected: a stale schedule can never
    run again, since every entry point refuses one, and every message of
    the call it last ran in was waited on in that call. Reclaiming also
    deletes the lease from the stale stage's dictionary, so that a stale
    stage asked for its buffers through the internals takes new ones
    rather than memory another stage holds. So no buffer is held by two
    leases, and none is reused while a message is in flight; the
    in-process test checks that the live schedule's buffers and mirrors
    never overlap.
  - *Sizes.* Best fit: the smallest free buffer that is long enough. If
    none is, the smallest free one is dropped and a new one allocated
    with a quarter's headroom, so the number of buffers per key is
    bounded by the most leased at once and their sizes follow the
    largest requests. Nothing shrinks otherwise: what a forest keeps is
    about the buffers of one live schedule plus one regrid stage, free
    between regrids. That is the memory a staged regrid on CUDA would
    otherwise allocate and pin again each time.
  - *Zeros.* A fresh buffer is zero-filled, as before; a reused one holds
    an earlier stage's bytes, which nothing reads, since every element
    of a send buffer is packed and every element of a receive buffer
    received before it is unpacked. The suite's bitwise comparisons
    run through reused buffers.
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

**Reductions: an allgather of per-rank partials**.
`combine_blocks` in `state.jl` is already the single site where M7's
communication was to go. It folds the local blocks as today,
`mapreduce(identity, op, values)`, then `allgather`s one partial per
rank, which every rank folds in rank order. It is not an
`Allreduce`, for four
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
  "**Implemented**" above). `init` starts every
  block's fold, so with an associative `op` and `op(init, init) ==
  init`, one more `init` cannot change the result: `max` from 0 over
  negative data is 0 on every rank count. What an empty rank's `init`
  would change is a weighted reduction, since the weight scales a
  block's value and not `init`: `max` from 1 over values below 1,
  weight ½, is ½, and 1 with an empty rank's `init` in the fold. The
  workload checks exactly that, at a rank count with an empty rank.
  Each rank therefore gathers `(hasvalue, value)`, and the fold
  skips the empty ones; `init` is returned only if every rank is empty.
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
   the *global* flag vector, which serially is the only one. The buffer's neighbour search is not replicated. Each
   rank searches from its own sources, and the recruits are gathered in
   a second `allgatherv` and applied on every rank, which gives the
   marks the replicated search gives; the rest of the completion is
   replicated as described.
4. The ghost fill before the transfer is the distributed fill, so a
   parent's ghosts are current on its owner.
5. The transfer is one stage over the new partition. A rank
   classifies only the new leaves it will own and those whose sources
   it owns, from the replicated old and new leaf arrays. The old owner of each source evaluates the transfer —
   a copy, a prolongation from the parent, or one child's share of a
   restriction — into a buffer shaped like that transfer's target box,
   and the new owner unpacks it. Sender computes is what makes this
   simple: the prolongation reads the parent's ghost layers, which only
   the parent's old owner holds.
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
- *The stage is `remote_stage`'s*.
  `regrid_sources` classifies the new leaves this rank needs once per
  regrid, not once per field set, into a `TransferPairs` under `GroupKey`s
  with direction zero and level 0 — targets new leaves, sources old,
  both global — and `regrid_stage` splits it by the new partition for
  the targets and the old for the sources, builds the local groups and
  calls `remote_stage` with the two owners and ranges. A coarsened
  block's `2^D` restrictions are `2^D` transfers with one target and
  their own sources, so children with different owners are simply
  received from different peers; nothing about them is special. Over
  one rank both ranges are every leaf, nothing is split, and the stage
  is the serial groups with no messages. `run_stage!` has a form
  over a separate `dest` and `src` — the new mesh's array and the old
  one's — which the exchange's form now calls with `fs.work` twice.
  `transfer_groups` keeps its signature, as the serial stage's groups,
  for `thread_tests.jl` and `bench/gpu.jl`.
- *The ghost fill's checks stay rank-local*.
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
each rank's first one, so every rank throws an `ArgumentError`
together, instead of one rank throwing while the others wait in the
next collective. Ghosts must be current, as now.

- *The contract.* Every rank passes the same field set, basis, `derivs`,
  `vars` and `exclude`, and its own points, any number or none; it gets
  the answers for its own points, in its own order, on the field set's
  backend. An owner evaluates the points it receives with its *own*
  arguments, so those that shape the evaluation must agree, and are
  hashed for the check below; the points are the only per-rank input.
- *One `allgather` before anything is sent*. The location runs
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
- *Not the same message on every rank*. A rank that passed an outside
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
  location stays the serial path's. The device buffers of MPI+GPU are for the
  exchange; nothing here needs them. Checked once on Metal in process
  (`Float32`, three simulated ranks, one with no points, value, gradient
  and an excluded ball): bitwise equal to the serial Metal call. That
  check is a scratch script, not part of the suite, which has no device
  in its environment.
- *The host location* is one search per point at tens of nanoseconds, so
  it runs serially below 4096 points and through `threaded_foreach`
  above, every point writing only its own slot.

**Parallel checkpoints.** The shared-file checkpoint that preceded
"Checkpoints without parallel I/O" — one file written through parallel
HDF5 and MPI-IO — its four-node account on Symmetry and the ROMIO
findings that led to its replacement are in
[HISTORY.md](HISTORY.md#parallel-checkpoints). What carries over from it:

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
- *How much of the plain data is checked*: all of it, by
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

- *Checksums*. The leaf list and every
  block of every field set carry a CRC-32C, verified on load, with the
  verdict agreed across ranks; the format and the reasons are under
  "Checksums" in [Checkpoint and restart](#checkpoint-and-restart).
  They are not the remedy — damage refused is still a lost checkpoint
  — but they are the defence for file systems and MPI-IO
  implementations this was not measured on: the next such loss is
  refused with its reason instead of restored into a wrong state. A
  damaged field set would have been restored: only the leaf list's own
  validation caught this one, and only because zeros broke the curve
  order.

**Checkpoints without parallel I/O** (it
replaces the shared file of "Parallel checkpoints" above, whose
four-node account is the reason). Writing one file
from several nodes was not reliable on Symmetry, and the remedy found
there — two ROMIO hints — is specific to one MPI-IO implementation on
one file system, while the same measurement showed NFS losing whole
pages to plain `pwrite`s from several nodes. Smaller clusters may fail
in ways of their own, and a checkpoint that can be lost is not one. So
no file of a checkpoint is ever shared between processes:

- **One writer and one opener per file**. Every file a
  checkpoint consists of is created and written by exactly one process,
  and on reading opened by exactly one process. HDF5 is used serially
  only, so nothing depends on MPI-IO, its hints, or a file system's
  coherence between clients. `TreeAMRHDF5MPIExt`, `open_parallel_file`,
  `librarycomm` and the hints are removed, and HDF5_jll need
  no longer be a parallel build or match the MPI: the stock one still
  is, which is harmless, and any other serves.
- **I/O groups.** The ranks are split into `k` contiguous groups in rank
  order, by the equal-count split that partitions the blocks
  (`equalsplit`); the first rank of each group is its *I/O process*, and
  the others its members. The ranks own contiguous runs of the curve in
  rank order, so a group's blocks are one contiguous run of the curve,
  and its part of every field set is one range of blocks. The keyword
  `io` of `save_checkpoint` chooses `k`:
  - `io = :node`, the default: one per shared-memory node.
    `k` is the number of nodes, which a verb, `commnodes`, counts
    from `MPI_Comm_split_type(MPI_COMM_TYPE_SHARED)`; its serial method
    answers 1. A node's client is the unit a cluster file system caches
    and is fed by, and one file per node keeps the number of files, and
    the metadata server's load, at the number of nodes.
  - `io = :all`: every rank writes its own part, and nothing is sent.
  - An integer `k ≥ 1`, used as it is up to the rank count and clamped
    to it beyond (an I/O process with no member is a rank
    writing its own part, so a larger `k` can mean nothing else).
  - *Nodes whose ranks are not contiguous*. The groups
    are always contiguous, equal-count rank ranges; `:node` sets `k`
    only. Under the usual block placement of ranks (SLURM's `block`
    distribution, `mpiexec`'s by-slot default) the groups are then
    exactly the nodes and every message stays on its node. Under a
    round-robin placement, or with unequal counts per node, a group
    spans nodes and its messages cross the network, which costs time
    and nothing else. Contiguity is worth more than locality: it is what
    makes a part one range of blocks, which one reader can send back
    out as contiguous ranges at any other rank count.
- **Files**. Beside the *index file* `path`, which rank 0
  writes, each I/O process writes one *part file*,
  `path.<saveid>.<j>.h5` for its group `j in 0:k-1`, in the index's
  directory. The save id is 128 random bits drawn by rank 0
  (`RandomDevice`) and written as 32 lowercase hex digits; it is
  recorded in the index and in every part, with the part's number and
  block range, so that a part from another save — the previous one, or
  one that crashed — is refused rather than read. With one I/O process
  (a serial run, `io = 1`, or one node under `:node`) the single part
  lives inside the index file, so a serial checkpoint is still one
  file. The layout is format version 2, under "File layout, format
  version 2" in [Checkpoint and restart](#checkpoint-and-restart).
- **Writing.**
  1. Every refusal is decided before anything is created and agreed
     across the ranks.
  2. Rank 0 draws the save id and broadcasts it (the verb `bcast`,
     with a serial method).
  3. Each member sends its owned data — the state vector, on the host;
     a device field set through `tohost` as before — to its I/O
     process: the members in curve order, a member's per-block
     CRC-32Cs first and then its blocks, in messages of whole blocks of
     at most 1 GiB each (one block if a block is larger; 64 MiB as
     implemented, see below). The I/O
     process writes each message as it arrives, by hyperslab, with the
     next receive already posted, so it holds at most two messages and
     never its group's data whole. It checks every block it receives
     against the member's CRC before writing it, so a block damaged in
     transit is refused rather than written.
  4. The I/O process writes its part's `data_crc32c`, closes the part
     and, under `sync = true`, flushes it and then its directory to
     stable storage (`flush_to_storage`). It reports success, the
     part's size in bytes, and a checksum of each field set's
     `data_crc32c` to every rank, in one `allgatherv`, which is also the
     verdict: if any I/O process failed, every rank throws and each I/O
     process removes its new part. Nothing is committed.
  5. Rank 0 writes `path * ".partial"`: the format attributes, the
     provenance, the forest, the field sets' layouts, the part table,
     the external links, and the application's group, the do-block's
     writes included. Under `sync = true` it flushes the file, renames
     it over `path` (`Base.Filesystem.rename`), which is the commit
     point, and flushes the directory; the verdict on that is agreed.
  6. Only then does rank 0 delete the parts the previous index named,
     read from it before the commit, and every *orphan*: a file in the
     index's directory whose name is exactly `basename(path)`, a dot,
     32 lowercase hex digits, a dot, decimal digits and `.h5`, with a
     save id other than this one. A failure to delete is a warning, not
     an error, since the checkpoint is in place.

  A crash at any point before the rename leaves the previous index and
  every part it names intact, plus at worst new parts that no index
  names, which the next save removes; a crash after it leaves the new
  checkpoint complete and at worst old parts, removed the same way. An
  I/O process whose writing fails goes on receiving everything its
  members send, without writing it, so that no member waits forever in
  a send, and the failure arrives in the verdict. An error of the MPI
  library itself remains fatal to the job. The cost of the cleanup rule
  is that a copy of the index under another name in the same directory
  does not keep its parts: an earlier checkpoint is kept by saving to
  another path, or by moving the index and its parts to another
  directory together, which keeps their relative names valid.
- **Reading.**
  1. The arguments are agreed, and only rank 0 opens the index. With
     one rank it reads it directly. Otherwise it copies everything but
     the part data — the `/TreeAMR.jl` group's attributes, the
     provenance, the forest, the field sets' layouts, the part table
     and the application's group — into an in-memory HDF5 file (the
     core driver, without a backing store) and broadcasts its bytes,
     and every rank opens that image (HDF5.jl's `h5open(bytes)`, which
     sets the file image). HDF5 itself serializes the forest and the
     plain data, so no serializer is needed. A refusal of the file
     itself — it is missing, or is not a checkpoint — is rank 0's, and
     is broadcast in place of the image, so it is raised on every rank;
     what the image holds is the same everywhere, so a refusal of its
     contents comes on every rank at the same point without a message.
  2. Each part is read by exactly one rank: the owner, under the new
     partition, of the part's first block, moved on to the next rank
     while that one already reads `cld(k, P)` parts, so that no rank
     reads more than its share when that is possible. The assignment is
     a function of the replicated part table, the same on every rank.
  3. Each reading rank opens its parts and checks each against the
     index: that it is a part, its save id, its number and block range,
     its size in bytes, and the checksum of each field set's
     `data_crc32c`. The verdict is agreed before any data move, so a
     missing part, a part from another save, or a truncated one is
     refused on every rank.
  4. Each owner posts its receives, straight into its state vector. The
     reader reads, for each field set, the part's blocks in the range
     each owner needs — contiguous, so a part goes to a contiguous run
     of ranks — by hyperslab, in pieces of at most 1 GiB (64 MiB as
     implemented), checks each
     block's CRC-32C, and sends it, with at most two pieces in flight; a
     piece for itself it reads in place. A reader whose read fails goes
     on sending, so no owner waits forever, and the damage and failure
     verdict is agreed afterwards: a block that fails its checksum is
     refused on every rank.

  A version-1 file is read as one inline part covering every block, by
  rank 0, which already has it open; so a version-1 file, written
  serially or by the shared-file writer, loads on any rank count.
  Loading on a different rank count, and serially, works for every
  combination, as it did.
- **The do-block runs on every rank**, as it did. Rank 0's application
  group is the index's; every other rank gets the same group in an
  in-memory scratch file, discarded afterwards, so that a block that
  calls collective operations still runs on every rank. `write_plain`
  in it still agrees its value across the ranks, so what rank 0 writes
  is what every rank passed; a dataset the block writes through HDF5
  itself is kept from rank 0 alone and must be the same on every rank.
  On loading, each rank's block gets the application group of its
  image.
- **External links for tools**. For
  each part file the index holds an HDF5 external link,
  `/TreeAMR.jl/parts/0003` → `ckpt.h5.<saveid>.3.h5:/TreeAMR.jl`, so
  that `h5dump`, h5py and HDFView navigate from the index into every
  part. TreeAMR's loader never traverses them: traversing an external
  link opens its target, which would have rank 0 open every part, and
  the other ranks hold the index only as an image. It takes the part
  names from the part table and resolves them against the index's
  directory itself. *The shadowing caveat*, for external tools: HDF5
  resolves a relative external link by trying a prefix
  (`HDF5_EXT_PREFIX`, or the link access property), the current
  working directory and the directory of the file holding the link, and
  HDF5's documentation lists the working directory before the parent's
  directory — so a tool started in another directory may open a
  same-named file there instead of the part. (Checked 2026-10-02:
  libhdf5 2.2.0, HDF5_jll's, and h5dump 2.1.1 resolved the parent's
  directory first, and fell back to the working directory only when the
  part was missing there; older libraries may not.) TreeAMR's own
  reader cannot be misled either way.
- **Format version 2; the reader reads 1 and 2.** The parts change what
  a file is, so the version is bumped, as "If the shared file does not
  hold up, M7 bumps `format_version`" accepted. 0.1.4 refuses a
  version-2 file with its format-version refusal, saying it was written
  by a newer TreeAMR and naming `checkpoint_environment` (accepted, by
  the versioning rules).
- **What carries over from the shared file**: the CRC-32C checksums per block and
  of the leaf list, now with a checksum per part and field set in the
  index above them; the refusals agreed before anything is created; the
  plain-data agreement; the self-checking benchmark, which loads every
  save it times; and the BeeGFS reproducers, whose `save_checkpoint`
  modes now exercise this writer and whose MPI-IO and POSIX modes
  remain as regression jobs for the file system. The shared-file
  layout, the collective transfers, the fixed-length string arrays
  (parallel HDF5 wrote no variable-length data) and the hints do not.

- *The save id* is drawn from `Base.Libc.getrandom!`, which is what
  `RandomDevice` reads, and not from `rand`: an application that seeds
  the global generator the same way in every run would draw, in a
  restarted run, the id of the checkpoint it restarted from, and its
  parts would overwrite that checkpoint's before the commit. An I/O
  process also refuses to create a part file that exists already.
- *The gathering's messages.* Per field set and member, first a vector
  of `UInt32`: a status word, then the member's per-block CRC-32Cs; then
  its blocks in pieces of whole blocks. A member that cannot prepare
  its data (gather it, copy it to the host) sends the status 1 and no
  blocks, and its own error comes in the verdict, so its I/O process
  never waits for blocks that are not coming. The I/O process holds two
  receive buffers of at most one piece each and writes its own blocks
  straight from its state vector. The I/O process receives every member's
  checksums first, queues all the members' pieces in curve order, and
  keeps the next piece in flight while it writes the current one — the
  first while it writes its own blocks — and a piece is at most 64 MiB,
  so a member's blocks are several. It is kept for the memory it
  bounds (the measurements are under M7 step 6b in
  [HISTORY.md](HISTORY.md#m7--mpi)).
- *An empty part is stored contiguously*, without filters: HDF5 refuses
  a chunk larger than a fixed dataset whose extent is 0, which a group
  whose ranks hold no blocks would otherwise need.
- *Version 2 requires its checksums*: a part without `data_crc32c`, or
  an index without `leaves_crc32c`, is refused as damaged; in a version-1
  file they stay optional.
- *The agreements.* Every step that runs on some ranks only — the parts,
  the index and the do-block, the commit, opening the parts, reading
  them — ends in one `allgather` of whether each rank failed and an
  `allgatherv` of the messages (`agree_errors`). A rank that failed
  throws its own error; every other rank throws one that quotes the
  first failing rank's message, an `ArgumentError` when that was a
  refusal, so the reason is on every rank's screen. The commit's verdict
  also carries whether the rename happened: if only the directory flush
  after it failed, the new checkpoint is the one at `path`, and its
  parts are kept.
- *The cleanup runs on rank 0 after the others have returned.* It
  touches only files of earlier saves, and the next collective call
  orders it before anything else the ranks do with the directory (the
  MPI test waits at a barrier before it lists the directory). Of the
  names the previous index lists, only those of the part form — a
  `.`, 32 hex digits, a `.`, digits and `.h5`, no `/` — are removed, so
  a damaged index cannot make a save remove any other file.
- *Flushing.* Each I/O process flushes its part and then the directory,
  so the part's directory entry is durable before an index names it;
  rank 0 flushes the index, renames it, and flushes the directory.
- *The index image.* The `/TreeAMR.jl` group's attributes and the field
  sets' layout attributes are copied attribute by attribute, as HDF5.jl
  reads and writes them, and the provenance, the forest, the part table
  and the application's group by `H5Ocopy` (HDF5.jl's `copy_object`),
  which keeps a group's creation order, so a NamedTuple's fields come
  back in order (checked). The image is created with the core driver
  and no backing store, so nothing touches the disk.
- *Reading.* A reader reads a piece for itself straight into its state
  vector, and for another rank into one of two send buffers, waiting on
  that buffer's previous send first; the pieces are cut as the
  gathering's are, so both ends agree on them without a message.
- *Plain data.* Without parallel HDF5 an array of strings is
  variable-length again, as in M9a. A string or `Symbol` holding a NUL is
  now refused by the plain-data walk, serially too, before anything is
  written: HDF5 stores a C string, and HDF5.jl refused it in the middle
  of the write.

The feasibility check of parallel HDF5 and the first measurement of what an MPI test costs, both from before M7 was built, are in [HISTORY.md](HISTORY.md#distributed-meshes).

**Performance work left for later** **(open)** (each item is to be measured
before it is built):

- *The H200 re-measurement of the staged regrid with the pool* is
  pending: `bench/symmetry_mpi_gpu.sh`'s `bench/mpi.jl` over CUDA at 2
  and 4 ranks, staged against direct, against step 8's 25.4 / 48.0 ms
  staged and 7.1 / 8.7 ms direct. The pool's local effect (step 8) says
  nothing about `cuMemHostRegister`, which neither the CPU nor Metal
  calls, so whether the 40 ms were the page-locking is still the
  untraced guess it was.
- *The pool's own rules.* It never shrinks: a forest keeps about one
  live schedule's and one regrid stage's buffers, free between regrids.
  A rule that drops buffers unused for some regrids, or a pool shared
  between forests over one communicator, waits for a run where the
  memory matters.
- *Fewer copies and launches.* Packing straight into page-locked host
  memory (or, on Metal, into the shared buffer the mirror copies to)
  would drop the staging copies; one launch per stage for all its packs,
  and one for its unpacks, would drop most of the 0.95 ms of launches in
  a 2.5 ms H200 fill (step 8); a copy between same-shaped boxes could be
  an MPI subarray datatype with no pack at all, on the host path.
- *One-sided and persistent communication.* A fill repeats the same
  messages until the next regrid, so persistent requests
  (`MPI_Send_init` / `MPI_Recv_init`) built with the schedule would
  save the matching, and RMA (`MPI_Put` into a window over the peer's
  receive buffer, exposed once per schedule) or GPU-initiated transfers
  (NVSHMEM, NCCL) would let the sender write into the peer's memory
  without a receive.
- *`interpolate!` at 32 ranks* (step 7, item 4): 8–10 ms against
  0.6–0.9 at 2–16 ranks under HPC-X's Open MPI, and a median of 9.9 ms
  already at 16; the HCOLL on/off job (567858) was never collected, and
  the cause is open.
- *Compression where the data are*. With `io =
  :node` one process per node runs HDF5's filter pipeline for the
  node's blocks, serially, so a filtered save is bounded by one core's
  compression per node (blast with zstd(1) on four nodes: 1.45 GB/s,
  against 4.50 with `io = :all`; step 6b's record). The members could
  compress their own chunks and send them compressed, for the I/O
  process to write with `H5Dwrite_chunk`, and a load could send
  compressed chunks to their owners (`H5Dread_chunk`); either needs the
  filter pipeline outside HDF5's own write, which HDF5.jl does not wrap.
  Until then `io = :all`, or a larger integer, is the setting for a
  filtered save; whether that should be the default is Erik's to decide.
- *What only many more ranks will show.* Measured to 32 ranks on four
  nodes and 4 GPUs on one: the replicated forest's memory and its
  `O(nleaves)` completion (20 ms at 32 ranks when every block is a
  source, step 7), the flags' `allgatherv`, the forest digest gathered
  by every checked call, the number of peers and messages per rank on
  a deeper hierarchy, and GPUs across nodes, which no run has used.

**The multi-block check** (the standing instruction). Nothing here obstructs a conforming multi-block forest.

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

M12's rotating seam is a
step toward a multi-block forest, not an obstacle to one: it removes
the assumption that a source's axes line up with its target's. An
oriented neighbor search, which returns the real leaves with the
quarter turns that carry them to where the asking block sees them, and
an axis-permuting source map in the kernel's load are the p4est
orientation mechanism in miniature, for the one gluing of two root
faces that a quadrant needs. It also corrects the third bullet above
in one respect: an oriented face does not change the stencils, which
are built in the virtual frame as for an aligned neighbor, but the
source index map, a permutation and a flip per dimension. The unpack
is still a copy into a target box whatever the orientation, carrying
the sign as it carries a parity. The one new ownership rule, both seam
planes of a vertex-like set owned, is about ownership only and assumes
nothing about aligned axes; it is recorded as a choice under "Rotating
seams" in [Ghost filling](#ghost-filling). Conformity at the seam is
the seam's own rule and not something a multi-block forest inherits:
there a glued face would be 2:1 balanced like any other, and the
interface restriction would have to cross it with an orientation,
which the seam avoids.

## Code structure

Fifteen source files, included in dependency order from `src/TreeAMR.jl`, plus
two package extensions in `ext/`; each layer uses only the ones before it:

| layer | files | what |
|---|---|---|
| threading | `threading.jl` | `threadchunks` (the block-ownership partition), the three host-side parallel-loop helpers everything else is built on, and `launch_by_owner!` |
| residency | `device.jl` | `todevice` (host-built metadata uploaded once, where it is already being rebuilt) and `check_floattype` |
| communicator | `communicator.jl`; `ext/TreeAMRMPIExt.jl` | M7: abstract `Communicator`, `SerialCommunicator` (rank 0 of 1), `communicator`, and the internal verbs (`commrank`, `commsize`, `allgather`, `allgatherv`, `alltoallv`, `bcast`, `commnodes`, `isend`/`irecv`/`waitall`, `hoststaging`), each with a serial method; the MPI extension (weak dependency MPI.jl) adds `MPICommunicator` — a cached `MPI_Comm_dup` with its rank, size and `deviceaware` setting — and one MPI.jl call per verb |
| tree | `morton.jl`, `forest.jl` | `MortonKey{D}` (root, level, coords; curve order computed on the fly), `Forest{D}` = sorted leaf vector + `generation` counter; neighbor finding (oriented across a rotating seam), `refine!`/`coarsen!`, `balance!` (conformity at the seam) |
| geometry | `geometry.jl` | key + stored cell index → physical coordinates |
| storage | `storage.jl` | `FieldSet`: one `(N+2G₁+c₁, …, N+2G_D+c_D, nvars, nblocks)` array over all leaves, ghosts included; the per-dimension `G` and the centering live here, not on the forest, and so do `parity`, `rotation` and their tables; `RotationPair` |
| operators | `operators.jl` | `Operators` (family + orders), `check_operators`, Lagrange weights |
| exchange | `schedule.jl`, `ghosts.jl` | `GhostSchedule` (built when the tree changes) and `fill_ghosts!` (replays it) |
| conservation | `interfaces.jl` | `InterfaceSchedule` and `restrict_interfaces!`: the flux fixup at coarse-fine faces, over the same `TransferGroup`/`run_phase!` machinery |
| ODE | `state.jl` | flat interior-only state vector, `scatter!`/`gather!`, `map_blocks!`, the reductions `block_mapreduce` (per block) and `mesh_mapreduce` (one number; its cross-rank step is the rank-order fold of gathered partials in `combine_blocks`), `volume_weighted_norm` |
| regrid | `regrid.jl` | flags → `buffered_flags` → `complete_marks` → rebuild → transfer; `adapt_to_initial_data!` |
| interpolation | `interpolate.jl` | `locate_point` (one binary search) and `interpolate`: a batch of arbitrary points, tensor-product `Lagrange(n)` over one block's stored array, first and second derivatives, periodic wrap and reflecting fold, `exclude` region flags |
| checkpoint | `checkpoint.jl`; `ext/TreeAMRHDF5Ext.jl` | `save_checkpoint`, `load_checkpoint`, `write_plain`/`read_plain`, `checkpoint_environment`: the stubs, docstrings and the load-HDF5 error hint in `src/`, the HDF5 implementation in the extension — serial HDF5 only, over a distributed forest the I/O groups, part files and index of M7 step 6b, the data moved by the communicator verbs |

### Rules that span several files

The ideas that span several files and are easy to violate:

- **Linear octree, leaf-only data.** `forest.leaves` *is* the tree: no node
  objects, no pointers, no coarse data under refined regions. Block `b` of
  any `FieldSet` is `blockkey(fs, b)`, leaf `first(blockrange(forest)) + b
  - 1`: since M7 step 1 every block index is **local to the rank**, while
  tree queries (`nleaves`, `find_leaf`, `locate_point`, `neighbor_keys`)
  stay global. Serially the two coincide, so a site that indexes
  `forest.leaves` by a block index, or sizes a per-block array by
  `nleaves`, passes every serial test and is wrong under MPI; walk
  `blockrange` and size by `nblocks`. Block indices are **not** stable
  across a regrid (slots are compacted), and `FieldSet` is a `mutable struct`
  precisely so that `regrid!` can swap `fs.work` wholesale while callers keep
  their reference.
- **Staleness is by generation, not size.** `rebuild_leaves!` bumps
  `Forest.generation` on every leaf change; a `GhostSchedule` records the
  generation it was built for, and `fill_ghosts!`/`regrid!` refuse a stale
  one. A refine-then-coarsen returns to the same leaf count with different
  leaves, which is why a count check is not enough.
- **Two time scales.** Neighbor finding (`neighbor_keys`, `find_leaf`) is a
  regrid-frequency operation; ghost filling runs at every RHS evaluation.
  Tree queries must never appear in the per-evaluation path — that is the
  whole point of the schedule.
- **One kernel for every transfer.** Same-level copy, restriction,
  prolongation, and the regrid transfer (the `δ = 0` case) are all tensor
  products of `D` one-dimensional `Stencil1D`s, so `transfer_kernel!` in
  `ghosts.jl` serves all of them; a copy is a width-1 stencil with weight 1.
  Transfers sharing `(kind, direction, child offset)` share stencils and are
  batched into one `TransferGroup` = one kernel launch. Stencil construction
  lives in `schedule.jl`; `regrid.jl` reuses `prolongation_stencil` and
  `restriction_stencil` so the two cannot drift apart. A group whose
  stencils are all `unit` (one point, weight exactly one — decided from
  the weights, not the kind) passes the kernel `unit = true` and does no
  stencil arithmetic, computing `0 + x` so that `−0` still becomes `+0`
  as the weighted sum makes it. A flag, not `weights = nothing`: **a
  run-time choice of an argument's type is compiled at every launch site
  for every member of the union**, which made the compilation-bound MPI
  workload a fifth slower and timed out CI's macOS cells (see "The
  copy kernels on a device"). Keep the CPU launch path type-stable, and
  hide unavoidable unions behind `Base.inferencebarrier` on the device
  path only (a barrier or a closure on the CPU path raises the fill's
  allocation). Compare `--trace-compile-timing` of `test/mpi_workload.jl`
  against `main` for any change to a launch path.
- **The copy kernels launch flat on a device, shaped on the CPU**
  (see "The copy kernels on a device"). `transfer_kernel!`,
  `scatter_kernel!` and `gather_kernel!` take a `shape` first and read
  their position through `kernel_position(shape, @index(Global,
  NTuple))`: `nothing` and the block-shaped ndrange on the CPU, a
  `LinearShape` (precomputed `Int32` inverses) and a one-axis ndrange on
  a device, where KernelAbstractions' own `NTuple` index costs two 64-bit
  divisions per axis per item. A flat CPU launch was 2.3x slower, which
  is why the CPU keeps the shaped one. `launch_positional!` and
  `run_group!` take `flat` so that the suite runs the device form on the
  CPU and compares bit for bit; a new copy-like kernel goes the same way.
  KA's CPU emitter only rewrites `@index` at statement level, so assign
  it before passing it to a function.
- **Phased ghost fill.** Phase 1: copies and restrictions (they read
  interiors only, so they are race free). Then the physical-boundary hook.
  Phase 2: prolongations, coarsest target level first, because a
  prolongation may read its coarse source's *own ghosts*, which an earlier
  sweep filled. The hook runs *between* the phases, not last: prolongation
  stencils at a domain edge reach tangentially into the source's outer
  ghosts.
- **Reflecting faces are transfers, not hooks** (M10). `reflecting` is a
  per-face `(lo, hi)` property of the `Forest`, `parity` a per-variable,
  per-dimension property of the `FieldSet` (required when the forest has
  a reflecting face; `NoParity` refused in a reflected dimension). The
  tree does not see the walls — `neighbor_keys` finds nothing across
  them — but `block_sources!` asks `reflect_direction` (in `forest.jl`,
  since it is brick knowledge) for `δ′`, the direction with the masked
  components zeroed, and takes the source that `δ′` finds. Along a
  masked dimension the stencil is the *tangential* one with its target
  rows remapped across the wall (`mirror_rows`), and the kernel
  multiplies by a parity factor from `fs.factors` (column
  `TransferGroup.factorcol`; 0 = an ordinary transfer, left unscaled).
  So mirrored transfers sit in the ordinary phases and run on devices.
  The unowned upper wall plane of a vertex-like dimension is derived by
  `wall_stencil` (odd → 0, even → the folded order-`p` interpolant).
  The boundary hook sees only *outer* faces.
- **Rotating seams are oriented transfers** (M12, see "Rotating
  seams" under "Ghost filling"). `rotating = (d1, d2)` is a property of
  the `Forest` (a ninth field, `NTuple{2,Int8}` because an `Int` pair
  made the schedule build allocate more; `(0, 0)` for none), `rotation`
  a signed permutation per variable on the `FieldSet` (required on a
  rotating forest; variable `v` beyond the seam is `sign(rotation[v])`
  times variable `abs(rotation[v])` at the preimage). The low faces of
  `d1` and `d2` are glued: `neighbor_anchor` is the one place the seam's
  arithmetic lives, and `oriented_neighbors` returns the real leaves
  with the orientation `r ∈ 0:3` (quarter turns), of which
  `neighbor_keys` returns the keys, so everything built on it sees the
  seam. `balance!` and the checked `leaves` path keep the seam
  **conforming** (equal levels across a seam face), so no coarse-fine
  face crosses it and the interface restriction never does. A turned
  transfer's stencils are built in the *virtual frame*, as if the
  source sat at its turned position (`virtual_offset`, `seam_offset`),
  so the stencil builders are unchanged; `GroupKey` carries the
  orientation (last in `keyorder`) and `TransferGroup` the orientation
  and the plane, and the kernel reads the source through the accessor
  `RotatedSource` (`(src, perm, flip, len, vars, col)`: the axis map
  from virtual to real stored index, a run-time `perm` read through
  `tuplepick`, and the variable from `fs.rotvars`, `nvars × 4`), built
  by `run_group!` only for `orientation ≠ 0` so an ordinary launch is
  unchanged. The sign goes through `fs.factors`, grown to `3^D·4`
  columns, column `mirror + 3^D·r` (`r = 0` is M10's table): a 90°
  turn of any Cartesian component is a signed permutation, so it is
  the permutation on the load and the sign on the target, and nothing
  is fixed up afterwards. A set whose layout is not symmetric under
  `d1 ↔ d2` (and has ghosts in the plane) turns into a partner and is
  filled as a `RotationPair`, whose odd turns read the partner's array
  (`altsrc`) and whose two schedules' stages run merged
  (`exchange_pair!`); a plain fill refuses it. Under MPI the pack
  permutes and the unpack, a plain width-1 copy, applies the sign, which
  keeps the serial `−0`. `interpolate` folds a point beyond the seam
  back and turns its value and gradient; a checkpoint writes `rotating`
  and each set's `rotation`, and lists the `"rotating"` feature only
  when used, so unrotated files still load in 0.1.6. A vertex-like set
  owns both seam planes (the same points under the turn) and evolves
  them at one level.
- **Neighbor finding is asymmetric across levels** (see "Neighbor
  asymmetry"). Ghost filling is formulated as each block asking for its own
  sources, never as reversing a neighbor lookup.
- **Two operator families.** `PointValue`: even orders, both operators are
  Lagrange interpolation at the target center, and restriction *shifts* its
  window inward near an interface. `Conservative`: odd prolongation orders
  built through the primitive function, restriction fixed at the exact
  2-cell average, nothing ever shifts. `Operators` has **no default order** —
  the right one follows from the application's differencing order (the
  interface-order rule: `p` must exceed it by two, for *both* operators). Do
  not add one. `check_operators` ties the orders to `N` and `G`; the
  constraints are listed under [Blocks](#blocks).
- **Interiors-only state vector.** An RHS is `scatter!` → `fill_ghosts!` →
  `map_blocks!`, with kernels writing `du` in state layout via `statearray`.
  `u` is never mutated; the working array is scratch. There is deliberately
  no `semidiscretize`-style wrapper.
- **No subcycling.** One global `dt` from `minimum_spacing`.
- **A regrid moves a block by at most one level**, which is what makes
  parent/child-only transfer sufficient; the transfer asserts it. After
  `regrid!` the schedule is stale and the state vector has a new length, so
  callers rebuild both and `reinit!` (or restart) the integrator.
  `adapt_to_initial_data!` *re-evaluates* the initial data on each new mesh
  rather than interpolating it.
- **KernelAbstractions from the start.** All per-cell work in `src/` is a
  `@kernel` launched through `get_backend(fs.work)`, with `D` and `G` as
  `Val` parameters. The CPU implementation is meant to already be the GPU
  implementation (M6). Don't write plain nested loops for cell work in
  `src/`; host-side driver logic (mark completion, balance, key rebuild)
  stays ordinary Julia — but *threaded* ordinary Julia, via
  `threading.jl`.
- **Bit-identical across thread counts, except floating-point sums.**
  Every work item owns its output slot, no parallel loop shares an
  accumulator, and collecting passes fill one buffer per task and
  concatenate in block order. That is race freedom and a hard invariant:
  the state vector, the leaf array, the schedule and every max or
  integer reduction are bit-identical whatever the thread count.
  Floating-point sums are promised to roundoff only (narrowed after M8,
  so that a device can reduce hierarchically — it does, in two launches
  with 256 lanes per block and no barrier — and ranks can fold their own
  blocks, which M7 does);
  the CPU fold is still one `mapreduce` per block summed in block
  order, and so still exact, which is why `test/thread_tests.jl`'s
  acceptance test — `test/thread_workload.jl` in subprocesses at two
  thread counts, digests compared byte for byte — passes unchanged. A
  new parallel loop that races on a slot breaks it; a reassociated sum
  would move only its `l2` and `mass` lines, which are the ones to give
  a tolerance if that day comes.
- **A ghost phase is one parallel loop, by owner.** `run_phase!` in
  `ghosts.jl` gives each thread the part of *every* transfer batch
  whose target blocks it owns; a batch is *not* the unit of
  parallelism, because batch sizes differ by orders of magnitude (face
  slab vs corner) and per-batch launches capped the ghost fill at
  ~2.5x. That needs each group's `targetblocks` non-decreasing, which
  every builder guarantees by collecting in block order and
  `test/thread_tests.jl` asserts. Only the CPU backend does this —
  `run_phase!` has a generic method that keeps per-batch launches for
  devices.
- **Every per-block pass runs a block on its owner's thread.** Block
  `b` belongs to the thread whose chunk of `threadchunks(nblocks)`
  contains it. `threaded_chunks` puts chunk `c` on thread `c` with
  sticky tasks, and `launch_by_owner!` runs block-shaped kernels on
  KernelAbstractions' static schedule with one block per workgroup,
  which splits the blocks identically. A new block-shaped CPU launch
  in `src/` goes through `launch_by_owner!`, and a new host loop over
  blocks through `threaded_chunks`. A bare `kernel(backend)(…)` or
  `Threads.@spawn` puts blocks on arbitrary cores and costs up to 2.4x
  on a many-core node (see "What one process loses").

- **Point interpolation reads one block** (M11, see "Point
  interpolation"). A query's *stencil* is `n^D` consecutive stored
  points of the block `locate_point` finds, ghosts included, so ghosts
  must be current. The kernel sees a basis only through `stencilwidth`,
  `stencilstart` and `basisweights` — the extension point for smooth
  bases — and `derivs` are multi-indices with only `|m| ≤ 2` accepted
  until higher orders are tested. A derivative's rate is
  `min(n − max mₐ, p − |m|)`: the ghosts carry the exchange's `O(hᵖ)`
  error, so `∂ₓ∂ᵧ` through `Lagrange(4)` with `p = 4` converges at 2,
  not 3. Outside points are reported by the
  host after the launch (no throwing in kernels), and `exclude` flags
  rather than throws. `locate_point` and `isless` share `curve_less`
  in `morton.jl`, so the search and the leaf order cannot disagree.

- **A checkpoint stores what cannot be recomputed** (M9a, see "Checkpoint and restart"): the forest's parameters and its leaves, in
  curve order, and each field set's layout and **owned points only**,
  in state-vector layout. Ghosts, shared planes and schedules are the
  application's to rebuild with its own operators and hook. A load
  builds the forest through the validated `leaves` path
  (`Forest(roots; …, leaves)`, which refuses a list that does not tile
  the brick or is not balanced) and the field set through its
  constructor, so nothing read is trusted before they check it.
  Compatibility is the file's `format_version` plus a must-understand
  `features` list, never package versions; each refusal says why and
  points to `checkpoint_environment`. TreeAMR writes only under
  `/TreeAMR.jl` and the application only under its own top-level group,
  leaving the root free for M9b's sidecars. The write goes to
  `path * ".partial"` and is renamed over `path` with
  `Base.Filesystem.rename` — not `mv(…; force = true)`, which on 1.11
  removes the target first — and, under the default `sync = true`, the
  partial file is flushed to stable storage before the rename and its
  directory after (`F_FULLFSYNC` on macOS, `fsync` elsewhere), since
  closing a file ends in the page cache and a rename can reach the disk
  before the data it names. Element types are HDF5 natives or *limbs*
  (`Float32x2` as two `Float32`), named as a Base-only module prints
  them and matched against the loader's `types`. No Julia type
  definition reaches the file, so a converter can read an old file
  without the old package; do not add JLD2 or `Serialization`. Every
  file carries CRC-32C checksums — the leaf list's (`leaves_crc32c`)
  and one per block of each field set (`data_crc32c`), and the index
  one per part and field set over those — verified on load; HDF5's own
  Fletcher-32 is deliberately not used (it accepts an all-zero chunk).
- **No file of a checkpoint has two writers or two openers** (M7 step
  6b, see "Checkpoints without parallel I/O"). A shared file
  written from several nodes through MPI-IO lost data on Symmetry's
  BeeGFS (step 6), so HDF5 is used serially only and the data travel
  as messages: the ranks form `k` contiguous I/O groups (`io = :node`,
  the default, by `commnodes`; `:all`; or a number), each group's first
  rank writes a part file `path.<saveid>.<j>.h5` from its members'
  streamed blocks, and rank 0 writes the index (format version 2) last;
  its rename is the commit point, after which rank 0 removes the
  previous parts and orphans. Loading, only rank 0 opens the index and
  broadcasts an in-memory image of it without the data (`bcast`), each
  part is read by one rank, which sends the blocks to their owners, and
  every step that runs on some ranks only ends in `agree_errors`, so a
  failure anywhere throws everywhere and nobody waits on a message. The
  index's external links to the parts are for tools; the loader must
  never follow them (that opens the target). One part lives inside the
  index, so a serial checkpoint is one file; version-1 files (the
  fixtures in `test/fixtures/`) still load. Do not reintroduce parallel
  HDF5 or MPI-IO.
- **The forest is replicated, the blocks are distributed** (M7, see "Distributed meshes"). Every rank holds `forest.leaves` whole; rank
  `r` stores the contiguous curve range `blockrange(forest)`
  (`equalsplit`, the arithmetic `threadchunks` uses one level down), and
  every block index of a field set is local (the first bullet). Every
  forest mutation is **collective**: the same call with the same
  arguments on every rank, which is what keeps the copies equal without
  a message, since every host pass is deterministic. `GhostSchedule`,
  `InterfaceSchedule` and `regrid!` check it through
  `collective_checks` in `forest.jl` (`interpolate` and the checkpoint
  functions through the same `ForestDigest` and `digest_verdict`): one
  `allgather` of a
  `ForestDigest` (generation, `nleaves`, a fold of `hash(key, h)` over
  every leaf — not `hash(leaves)`, which samples a long vector — the
  brick, a layout hash and a refusal flag), so a diverged forest, a
  layout that differs, or an `ArgumentError` on some ranks is refused on
  all of them together. `fill_ghosts!`'s checks stay rank-local on
  purpose (a collective per fill would sit on the per-evaluation path);
  a one-rank refusal there is a hang, never wrong data. Serially nothing
  is gathered. Any new check that can fire on some ranks only goes
  through `collective_checks`.
- **The sender computes; messages are stages of packed buffers.** A
  transfer whose source and target live on different ranks is evaluated
  on the source's rank by the one kernel into a packed buffer (a
  `NamedTuple` `(buf, offsets, dims)` that `transfer_kernel!` reaches
  through three accessor methods), sent, and unpacked on the target's
  rank by a width-1, weight-1 transfer that carries the mirrored
  group's **parity factor**: applied when packing, the factor would
  turn the serial fill's `−0` into `+0`. Every ordering point of the
  serial fill is a stage with its own tag — phase 1 (1), each phase-2
  target level (`2 + ℓ`), each interface dimension (`40 + d`), the
  regrid (50) — run by `run_stage!` in five steps (post receives, pack
  and synchronize, send, run the local groups, wait and unpack). Both
  ends derive a message's layout from the replicated forest, sorted by
  `keyorder`, so no descriptor is ever sent and `Dict` order must never
  reach a layout. Device buffers are staged through page-locked host
  mirrors unless `communicator(comm; deviceaware = true)` (`hoststaging`).
  Stage buffers and mirrors are **leased from the forest's buffer pool**
  (`bufferpool(forest)`, in `forest.state` beside the generation): the
  regrid stage releases its leases after its sends are waited on, and a
  schedule's are reclaimed once the generation has moved on, so a
  regrid reuses what earlier stages held instead of allocating and
  page-locking it again. A lease is an `Array` (`Base.wrap` over pooled
  `Memory`) or a contiguous device `view`, never a `SubArray` of host
  memory. Do not add a field to `Forest`: put mutable state in
  `ForestState` (a ninth field made the schedule build allocate more;
  M12's `rotating` is one only because two `Int8`s fit in padding, and
  `bench/ghosts.jl` measured it at zero bytes).
  MPI is called from the calling task only, never in a threaded loop or
  kernel, and needs `THREAD_SERIALIZED`.
- **Reductions are an allgather of per-rank partials**, folded in rank
  order on every rank (`combine_blocks` in `state.jl`, the only site),
  not an `Allreduce`: the association is the package's, so every rank
  gets the same bits, any `op` and `isbits` partial works (MPI.jl
  refuses custom operators off Intel), and one rank is exactly serial. An
  empty rank contributes nothing, not `init`. `block_mapreduce` and
  `firing_boxes` stay per local block; a number combined from them is
  rank-local, which is the downstream hazard the step-9 audit found
  everywhere (HISTORY.md, M7 step 9).
- **MPI is a weak dependency** (`[weakdeps]`, `[compat]` 0.20), like
  HDF5: `src/` never names MPI and never branches on it — a serial run
  takes the distributed code path with every message empty, and the
  serial suite is its test. Do not make MPI a hard dependency, do not
  call MPI from `src/`, and do not use `MPI.Comm_dup` (MPI.jl's attaches
  a finalizer that frees collectively at different moments on each
  rank; the extension calls `MPI.API.MPI_Comm_dup` and caches one
  duplicate per communicator, checked with `MPI_Comm_compare`).

### Index conventions

Index conventions: per dimension `d`, stored indices run `1:N+2G_d+c_d`
(`c_d = 1` in a vertex-like dimension, `0` in a cell-centered one); the
**owned** range is `G_d+1:G_d+N` and the **closed** range `G_d+1:G_d+N+c_d`.
`coordinates(fs, b, idx)` — which replaced `cell_center` in M8, and takes the
field set rather than the forest — takes **stored** indices, so owned point
`i` is `idx = i + G_d`. Kernels launched by `map_blocks!` get the global index
`(i1, …, iD, b)` with each `i` in `1:N` (or `1:N+c_d` under `closed = true`)
and add `G_d` to reach the working array — **except** under
`stored = true`, the third range, where the loop runs over `1:N+2G_d+c_d`
and the global index already *is* the stored index, so the kernel adds
nothing. A kernel written for one form is wrong under the other and still
in bounds. Directions are
`δ ∈ {-1, 0, 1}^D` from `alldirections(Val(D))`; `+1` names the high
ghost slab.

The API reference is split by layer, `docs/src/api/{tree,storage,exchange,ode,regrid,interpolate,io,distributed,internals}.md`;
`docs/src/index.md` is the guide (prose and doctests, plus the status) and
holds no `@docs` blocks. The split is there because Documenter's HTML writer
fails the build on any page over 200 KiB (`size_threshold`), and the single
page had reached 178 KiB; the largest page is now about 40 KiB.

HDF5 and MPI are weak dependencies (`[weakdeps]`): the checkpoint
implementation is the package extension `TreeAMRHDF5Ext`, which Julia
loads only once HDF5 is loaded beside TreeAMR, and the MPI methods are
`TreeAMRMPIExt`. The test and docs environments list HDF5, and
`docs/make.jl` does `using HDF5`, so the suite and the checkpoint doctest
see the extension. Without it the five checkpoint functions have no
methods, and an error hint registered in TreeAMR's `__init__` says to
load HDF5.

## Tests

`test/runtests.jl` holds the M1 tests inline and `include`s
`ghost_tests.jl`, `centering_tests.jl`, `reflect_tests.jl` (M10),
`rotate_tests.jl` (M12),
`interpolate_tests.jl` (M11), `interface_tests.jl`, the four in-process
M7 files `partition_tests.jl`, `exchange_tests.jl`,
`regrid_exchange_tests.jl` and `interpolate_exchange_tests.jl`,
`allvariables_tests.jl`, `state_tests.jl`, `regrid_tests.jl`, `wave_tests.jl`,
`wave_cell_tests.jl`, `burgers_tests.jl`, `imex_tests.jl`, `type_tests.jl`,
`checkpoint_tests.jl` (M9a), `thread_tests.jl`, `mpi_tests.jl` (M7) and
`gpu_tests.jl` (M2–M8).

The M7 tests come in two kinds. **In process**, simulated ranks over
test-only communicators, each answering only the verbs its test needs:
`PartitionCommunicator(rank, size)` (`partition_tests.jl`: rank and size,
and the digest gather by replication) for the partition and each rank's
local schedule; `exchange_tests.jl`'s lockstep, which builds every
simulated rank's stages and wires their buffers directly, for the
classification, the layouts, write-once and the bitwise pack/unpack
round trip (the `−0` parity case included), and its
`MailboxCommunicator`, one task per rank with a channel per message,
for the staged driver itself; `regrid_exchange_tests.jl`'s
`GatherCommunicator` (collectives as rendezvous between the tasks, the
mailbox for messages) for `regrid!`, `adapt_to_initial_data!`, the
per-rank buffer search and the refusals, and its `StagingCommunicator`
for the host-mirror path; `interpolate_exchange_tests.jl` over the same
for routed interpolation. **Over MPI**, `mpi_tests.jl` runs the
standalone `mpi_workload.jl` serially in process (the reference) and
under `MPI.mpiexec()` at `-n 3` and `-n 2`, one thread a rank, and
requires every line byte for byte except the `sum` lines (`rtol =
1e-12`); `#` lines depend on the rank count and carry the refusals,
the checkpoint cross loads, the migrations and the negative control,
each asserted on its own. The launches are compilation-bound (about
55 s each), so `test/mpi_jobs.jl` starts both at the start of the
suite where the machine has room (`concurrent_launches`: 8+ threads and
24+ GB, or `TREEAMR_TEST_MPI_CONCURRENT=0/1`), and otherwise runs the
serial reference, the three-rank job and the two-rank job one after the
other (the three ranks beside the reference outgrew CI's 7 GB macOS
runners and timed out); the workload's `TREEAMR_CHECKPOINT_FROM`
and its marker files order the cross loads either way. Keep
`mpi_workload.jl` self-contained like `thread_workload.jl`, its
non-`#` output independent of the rank count, and a new case's lines
deterministic. `mpi_device_workload.jl` and `mpi_device_tests.jl` are
its device counterpart, run by hand (see Commands), never by
`Pkg.test`.
`checkpoint_tests.jl` checks a bitwise round trip over every centering,
`Float64`/`Float32`/`Float32x2` and every face kind, restarts of the
wave and Burgers studies that continue byte for byte through regrids,
the refusals with their reasons, the checksums (a damaged block, a
recomputed one, missing ones), the atomic write and a failing I/O
process, plain data and `checkpoint_environment`, and that the
version-1 files in `test/fixtures/` (written by the step-6 writer)
still load; the leaf-list `Forest` it loads through is tested inline in
`runtests.jl`, against the oracles. The part files themselves are
`mpi_workload.jl`'s: saves with `io = :all`, `2` and `:node`, a part
from another save, a missing part, orphans, a failed I/O process, and
the `OPEN_LOG` hook showing that no file is opened by two processes. The wave
study comes in two halves: `wave_tests.jl` is the **vertex-centered**
one (M8a), and `wave_cell_tests.jl` is the M3 cell-centered study kept
verbatim so its numbers stay under test. `imex_tests.jl` runs the wave
and Burgers studies a second time through IMEXRungeKutta's explicit
`RK4` and `SSPRK33`, by owner, with a `state_partition` helper built
from `threadchunks`; it also asserts that a stage limiter's correction
never reaches the state (the drift of a conserved total is the step
limiter's injection) and that OrdinaryDiffEq's Shu–Osher SSPRK33 is
different there. Its names clash with OrdinaryDiffEq's (`RK4`,
`SSPRK33`), so it uses `import IMEXRungeKutta as IRK`. Seven helper
files are not tests:

- `oracles.jl`, `ghost_oracles.jl` — deliberately naive, independent
  reference implementations (bit-plane Morton comparison, exact `Rational`
  box geometry, analytic polynomials, Gauss–Legendre cell averages). Property
  tests compare the package against *these*, never against the package's own
  neighbor search or geometry. Keep that independence when adding oracles.
  The M10 oracles live at the end of `ghost_oracles.jl`: `faces_forest`
  (three levels against the walls), `parity_data` (polynomials of
  definite parity), `undefined_ghosts` (the `NaN`-prefill check that
  catches a ghost read before it is written) and `reflecting_vs_doubled`
  (a half domain against the doubled domain it folds). The M12 oracles
  follow them: `unfolded_forest` (the full plane built from the
  quadrant's leaves turned in exact `Rational` boxes),
  `seam_neighbor_mismatches` and `seam_conforming` against it;
  `rotating_data` (a scalar and a vector covariant under the turn and
  under no mirror, smooth or polynomial, its constants in the argument's
  real type so a `Float32` device evaluates it), `rotating_forest`,
  `undefined_rotated_ghosts`, `rotating_vs_quadrupled` and
  `rotating_pair_vs_quadrupled` (the quadrant against the full plane
  `quadrant_and_full` builds, every stored point), `hook_regions_leave`,
  and the regrid oracle `rotating_regrid_vs_quadrupled`. Beyond those,
  `rotate_tests.jl` holds the rigid-rotation advection (conservation
  through the seam), interpolation beyond it, and the wave on a quadrant
  against the full plane (`rotating_wave`, scalar and vector, which the
  vertex-centered seam planes keep equal bit for bit).
- `wave.jl` — the scalar wave equation as an application of the mesh
  (`WaveProblem`, `wave_rhs!`, `wave_errors`, `track_pulse`,
  `uniform_pulse`). It lives in the tests because the package has no physics.
  Every entry point takes `centering`, defaulting to `vertexcentered(D)`;
  `wave_forest` does not, because a centering does not change how space is
  cut into blocks.
- `burgers.jl` — Burgers' equation as a **conservative** application
  (`BurgersProblem`, `burgers_rhs!`, `burgers_errors`, `track_shock`,
  `uniform_shock`), the M8b counterpart of `wave.jl`: the three-step
  right-hand side, a cell-centered state with `G = 2` and `D`
  face-centered flux sets with `G = 0`, and `fixup = false` as the
  negative control for conservation. It reuses `to_backend` and
  `convergence_rate` from `wave.jl` and `cell_average` from
  `ghost_oracles.jl`, so `runtests.jl` includes it after both.
- `thread_workload.jl` — a standalone script, not `include`d. The thread
  count is a command-line argument to Julia, so the M5 acceptance test runs
  this in subprocesses at two thread counts and compares their output byte
  for byte. It is deliberately self-contained (its own RK4 and SSPRK3, no
  ODE package) so a subprocess starts in a couple of seconds; keep it that
  way, and keep everything it prints deterministic. Its cycles cover
  periodic and outer boxes, both centerings, reflecting walls (`…r`),
  rotating quadrants with a `RotationPair` fill (`…q`, M12) and the
  conservative Burgers cycle (`B…`).
- `mpi_workload.jl` — the M7 counterpart, standalone too: the argument
  `mpi` makes its forests distributed over `MPI.COMM_WORLD`, and rank 0
  prints the digests gathered in block order. `mpi_tests.jl` also runs
  it in process, in a module of its own, printing into the `IOBuffer`
  it defines as `WORKLOAD_IO`.
- `mpi_jobs.jl` — the launcher of those `mpiexec` jobs
  (`launch_workload`, `finish_workload`, `concurrent_launches`), which
  `runtests.jl` includes before the first test so that the jobs can
  start early.

**The suite is compilation-bound, not kernel-bound.** Measured: annotating the test
applications' kernels `@inbounds` made `wave_rhs_kernel!` 6.8x faster
and left the suite at 3m11 against 3m14, inside the 7 % run-to-run
noise; the same treatment of `test/thread_workload.jl` moved its 18.8 s
by 2 %. CI gains nothing from such a change in any case, since
`check_bounds: yes` overrides `@inbounds` package-wide.

**CI and coverage.** `julia-actions/julia-runtest` defaults `coverage`
to `true`; CI.yml ties coverage to `threads == 1`, which keeps
it on the four cheap cells (both operating systems, both Julia
versions, merged by Codecov) and drops it from the expensive threaded
ones. Instrumentation cost **2.61x** locally
(measured: 3m23.6s → 8m51.8s, 88819 tests, 4 threads both times) —
though that is not the CI factor, because the action also defaults
`check_bounds` to `yes` and a plain `Pkg.test` no longer does.
`Base.julia_cmd()` forwards `--code-coverage` to the subprocesses the
thread-independence test spawns, so those are instrumented too — three
PIDs write `.cov` files and `julia-processcoverage` merges them per
source file, which is why a single-threaded cell still sees the
threaded code paths. The MPI test's five ranks launch through
`Base.julia_cmd()` too, so the MPI extension's coverage comes from them
(not measured in step 9).

## Ecosystem integration

- **Time integration:** OrdinaryDiffEq.jl via the flat state vector (see
  above).
- **Elliptic solvers:** no solver-specific machinery in the package; the
  ghost/operator infrastructure suffices to build composite-grid
  operators. (Multigrid on the tree hierarchy would require overlapping
  coarse data, which leaf-only storage does not provide — out of scope.)
- **I/O:** checkpoint and restart through HDF5.jl, as a package
  extension, as specified under
  [Checkpoint and restart](#checkpoint-and-restart). Visualization
  export is M9b; an ADIOS2 backend only if parallel HDF5 does not scale
  at M7 (see [Open questions](#open-questions)).

  **Inconsistent:** the I/O bullet makes an ADIOS2 backend conditional on parallel HDF5 not scaling at M7, but M7 replaced the parallel-HDF5 checkpoint by part files written serially ("Checkpoints without parallel I/O" under [Distributed meshes](#distributed-meshes)).

- **Visualization:** export is M9b, with its candidates
  listed in [PLAN.md](PLAN.md#m9b--visualization-export). The original plan here, VTK's
  non-overlapping AMR format, does not exist in VTKHDF (see "Formats
  considered" under "Checkpoint and restart" in
  [HISTORY.md](HISTORY.md#checkpoint-and-restart)).

### Downstream applications

Three applications use TreeAMR; mesh machinery belongs here and physics
there. What each calls is what a rename or a re-signature here breaks.
How each was checked against this package's milestones, and the audit of
what each would get wrong once it distributes (M7 step 9), are in
[HISTORY.md](HISTORY.md#downstream-checks).

**TreeWave** (`~/src/jl/TreeWave`, github.com/eschnett/TreeWave.jl) is the sample
application: the scalar wave equation with a Löhner refinement criterion,
ported from `test/wave.jl`. It exists so that there is a real downstream user
of the *public API only*. Facts that matter here:

- It takes TreeAMR from the **General registry**, not from `main` and not
  from this checkout: `Project.toml` bounds it with `TreeAMR = "0.1.0"`
  under `[compat]`, and `bin/Project.toml` inherits that bound through
  its `TreeWave = {path = ".."}` source. So neither a push to `main` nor an uncommitted change
  here reaches it; a change arrives with the next release.
- It takes `FieldSet(forest,
  nvars; G, centering, backend)`, the `fs => schedule` form of `regrid!`,
  `GhostSchedule(fs, ops)`, and `coordinates(fs, b, idx)` in `bin/`.
  `bin/` is outside `src/` and `test/`, so `Pkg.test` never runs it; its
  CI's `viewer` job renders every figure on each push, which is what
  catches breakage there.
- Its `src/` calls: `Forest`, `refine!`, `balance!`, `nleaves`, `level`,
  `maxlevel`, `spacing`, `minimum_spacing`, `block_spacings`,
  `block_extent`, `MortonKey`, `FieldSet`, `nblocks`, `blockkey`,
  `fill_by_coordinates!`, `Operators`, `GhostSchedule`, `fill_ghosts!`,
  `statelength`, `statevector`, `statearray`, `scatter!`, `gather!`,
  `map_blocks!`, `block_mapreduce`, `volume_weighted_norm`, `flag_blocks`,
  `firing_boxes`, `regrid!`, `adapt_to_initial_data!`, `cellcentered`,
  `vertexcentered`, the `RegridFlag` values, and the `(flag, box)` flag
  form; `bin/` adds `blockview`, `interiorview` and `coordinates`, and
  its tests add `buffered_flags`, `complete_marks` and `block_origin`.
  `hostcopy` is its own, in its `device.jl`, and it does not use
  `todevice`. Renaming or re-signaturing any of these breaks it.
- `using HDF5` beside TreeAMR loads the checkpoint extension, and
  nothing else is needed. It calls none of the checkpoint functions yet.
- It does not pass `comm` to any `Forest` yet, so under `mpiexec` each
  rank would run a whole serial copy. Once it does, the fix for what
  the audit found is `mesh_mapreduce`. It indexes nothing by a global
  `b` and sizes nothing by `nleaves`.
- Its `CLAUDE.md` and `CODE.md` record API sharp edges found from the
  outside — a keyword named `maxlevel` shadows the exported
  `maxlevel(forest)` inside a function body; `coordinates` taking stored
  indices is an easy off-by-`G`. Read them when changing anything
  user-facing.

**TreeHydro** (`~/src/jl/TreeHydro`, github.com/eschnett/TreeHydro.jl) is the second
worked application: Newtonian ideal hydrodynamics with a
high-resolution shock-capturing finite-volume scheme — the
*conservative* counterpart of TreeWave, and the heaviest downstream user
of M8.

It pins **neither `main` nor this checkout**: its `Project.toml` and `bin/Project.toml` have no
`[sources]` entry for TreeAMR, only `TreeAMR = "0.1.3"` under `[compat]`.
A change here reaches its tests only once it is tagged and registered,
a higher bar than a push; to try one sooner, `Pkg.develop` this checkout
into a scratch copy of TreeHydro, never the real one.

It is the only caller of several things, which makes it the only test
of them outside this repo: `InterfaceSchedule` / `restrict_interfaces!`,
the physical-boundary hook at all (`boundary_by_coordinates` over an
`AllVariables` callback; TreeWave is periodic throughout),
`map_blocks!(…; stored = true)`, `facecentered` field sets with `G = 0`,
`total_mass`, and `regrid!` over several `fs => schedule` pairs including
`fs => nothing`. Beyond those its `src/` calls `Forest`, `refine!`,
`balance!`, `nleaves`, `level`, `spacing`, `minimum_spacing`,
`block_spacings`, `block_extent`, `FieldSet`, `nblocks`, `blockkey`,
`coordinates`, `fill_by_coordinates!`, `Operators`, `GhostSchedule`,
`fill_ghosts!`, `statelength`, `statevector`, `statearray`, `scatter!`,
`gather!`, `map_blocks!`, `block_mapreduce`, `volume_weighted_norm`,
`firing_boxes` with the `(flag, box)` form of `Refine`/`Keep`/`Coarsen`,
and `adapt_to_initial_data!`. Its tests add `Conservative` (every
`Operators` they build), `maxlevel`, `block_origin`, `cellcentered`,
`staggers`, `RegridFlag` and `buffered_flags`; `bin/` adds
`interiorview`. It uses neither `todevice` nor a TreeAMR `hostcopy`:
`to_backend` and `hostcopy` are its own, in its `device.jl`.

Which of its tests reach TreeAMR: `prerequisite_tests.jl` tests the mesh
directly — `AllVariables` runs once per point, a `stored = true` launch
reaches every ghost, and a name list asserts the resolved release still
exports the M8 surface. The rest go through the scheme:
`interface_tests.jl` (the fixup is what conserves at a coarse-fine face;
the interface-order rule for a system), `sod_tests.jl` and
`sedov_tests.jl` (the Dirichlet hook on a face, and at a corner and an
edge — the M2 ordering case), `refinement_tests.jl` (`firing_boxes`, the
four marks, `buffered_flags`), `driver_tests.jl` and
`kelvinhelmholtz_tests.jl` (`regrid!`, `adapt_to_initial_data!`,
conservation through the regrids), `evolution_tests.jl` and
`reset_tests.jl` (`fill_ghosts!` and `map_blocks!` over `statearray`,
through the right-hand side and the reset). `eos_`, `riemann_`,
`exact_riemann_` and `precision_tests.jl` are pointwise physics and touch
nothing here. Its `bin/`, like TreeWave's, is run by its CI's `viewer`
job rather than by `Pkg.test`.

It does not pass `comm` to a `Forest` yet. Once it does, it is the most
exposed of the three: its CFL speed, its floor counts and its refinement
scales combine `block_mapreduce` on the host, and its wall-clock
checkpoint triggers decide per rank whether to enter the collective
`save_checkpoint`. The fixes are `mesh_mapreduce`, and an agreed trigger.
It needs TreeAMR 0.1.6 or later over MPI, for the fix to the
`AllVariables` boundary hook on a rank without blocks (see "Ranks without
blocks are allowed" in [Distributed meshes](#distributed-meshes)).

**TreeGeneralizedHarmonic** (`~/src/jl/TreeGeneralizedHarmonic`,
github.com/eschnett/TreeGeneralizedHarmonic.jl) is the third application:
the vacuum Einstein equations in the generalized harmonic formulation, a
black hole on the octree. It takes TreeAMR from **General**,
`TreeAMR = "0.1.4"` under `[compat]`, so a change here reaches it only
with a release. Every kernel it has goes through `map_blocks!`. Its
`test/prerequisite_tests.jl` names the four unexported TreeAMR names it
relies on — `threadchunks` (its integrator's partition) and M11's
`Region` extension points `inside`, `stencil_hits` and
`stencil_position` — so renaming one breaks that suite at the top. Its
restart stores, beyond `(t, u)`, its horizon tracking and interior fits,
through its `src/checkpoint.jl`. Of the three it is the closest to ready
for MPI: its reductions are already `mesh_mapreduce` and its integrator
is fixed-step. It does not pass `comm` yet (`gh_forest`,
`src/initialdata.jl`; `load_run`, `src/checkpoint.jl`). Its horizon
finder passing every point on every rank is correct under the
collective `interpolate`, only redundant.

## Open questions

All design questions through M3 are resolved in the sections above.
Remaining, none blocking before their milestone:

- Wiring `volume_weighted_norm` into adaptive
  integrators as `internalnorm`. Its cost objection on a
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
- **What M9a leaves open** (see
  [Checkpoint and restart](#checkpoint-and-restart)). None of it is
  needed to restart a run.
  - Conversion on load, such as `Float64` into `Float32` for a device
    without fp64. Version 1 loads the saved type only, since exactness
    is the point of a checkpoint.
  - Partial loads: a subset of a field set's variables, or a range of
    blocks.
  - Appendable time series: several states in one file.
  - An ADIOS2 backend, if parallel HDF5 does not scale at M7. The data
    model maps one-to-one onto ADIOS2 variables (the datasets) and
    attributes.
    **Inconsistent:** this item is conditional on parallel HDF5 at M7,
    which M7 replaced by part files written serially.
  - A refusal for a missing filter. A
    file saved with H5Zzstd's filter and loaded without `using H5Zzstd`
    fails with HDF5's `H5Error`, a plugin it cannot find in a build
    directory, rather than with a reason. The loader could read the
    data set's filter pipeline first and name the filter and the
    package that provides it.
- **What M12 leaves open** (see "Rotating seams" under
  [Ghost filling](#ghost-filling)).
  - *A 180° rotation, the π-symmetry* (deferred).
    Half of the plane is simulated, and the low face of `d1` is glued to
    itself, flipped about the domain's centre line in `d2`. It needs the
    machinery of M12 — the oriented search, the virtual frame, the axis
    map with orientation 2 only, the signed variable map — and one thing
    more, a face that is its own neighbor across the seam, which a
    leaf's image on the same face can make coarse-fine unless
    conformity is extended to it.
  - *Reflecting high walls together with the rotation*, a symmetric box
    rather than a quadrant open at its high faces. `R` carries the high
    face of `d1` onto that of `d2`, so both would have to reflect, the
    parities along `d1` and `d2` would have to agree with the map (`vx`
    odd in `x` exactly when `vy` is odd in `y`), and a mirrored transfer
    across one high wall can then also cross the seam. Refused for now.
  - *Interpolation of a paired set beyond the seam.* The value there is
    the partner's, so the kernel would need both working arrays, or the
    host would route the folded point to the partner and swap the
    variables. Refused, with the reason, until an application asks.
- **The working array's layout** (from
  [The copy kernels on a device](#the-copy-kernels-on-a-device); to be
  revisited with Erik as future work). Once their index was cheap, the
  copy kernels were left at what the layout `(i₁ … i_D, var, block)`
  allows. The scatter writes rows of `N` at an offset of `G` into rows of
  `N + 2G + c` — 1.8× a plain copy on an H200, and no way of forming the
  index changes that — and a ghost slab normal to the first dimension
  is runs of `G` values, a stride of a whole row apart.
  Candidates, none measured: padding the stored first extent so that
  owned rows start on a 128-byte boundary (the downstream predicted
  about 10 % on its stencil kernels from the same padding); a
  block-of-blocks or variable-innermost layout for the ghost slabs; a
  separate, packed store for the ghost layers. Any of them changes
  `blockview`, `statearray`, the checkpoint's in-memory side and every
  application kernel's indexing, so it is a design decision, not a
  tuning.
- **A state that lives in the working array** (the third step
  of TreeGeneralizedHarmonic's item 5). An integrator whose stage vectors
  are field sets needs no scatter at all, and a native stepper could write
  the next stage's input straight into a second working array from the
  right-hand side's epilogue. It costs `(N + 2G + c)^D / N^D` of memory
  per stage vector (1.81 at `32³`, `G = 3`) unless the stage arithmetic
  skips the ghosts, and it changes "Interiors-only state vector" under
  [Time integration](#time-integration). The downstream estimated it at
  10–15 % of a step once the kernel and the copies are fast.
- **The integrator's own passes are not owner-based** (raised by
  TreeGeneralizedHarmonic after it adopted the ownership
  policy of [Parallelism](#parallelism); not to be optimised further,
  below). An external
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

  *Polyester: not pursued further*, for what the
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

  *Not pursued further.* The integrator's passes
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
