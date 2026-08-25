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
- **Cell-centered only, initially.** Face-centered variables (needed for
  constrained-transport MHD and for conservative flux matching) are
  scheduled: they arrive in milestone M8 together with conservation
  support. Vertex- and edge-centered variables remain unscheduled future
  extensions. We deliberately do not abstract centering before then; the
  retrofit will touch the core (array shapes, ghost rules) and we accept
  that.

## Core concepts

### Blocks

The unit of storage and computation is a **block**: a Cartesian grid of
`N^D` cells with `G` ghost cells on each side, so `(N+2G)^D` stored cells
per variable. Every block has the same size.

- `D` is generic (1, 2, 3, ...) and compile-time (it is the array rank).
- `N` is likely 32 or 64; `G` is small (1–4). Both are runtime values
  (no recompilation when they change), validated at construction against
  the invariants below.
- Cell-centered data; the block covers a cube in physical space, uniform
  spacing within the block, spacing halves per refinement level.

Ghosts are stored on all sides of every block (decided; the earlier
ideas of x-only or unstored ghosts are dropped). Ghost cells exist only
in the working array, not in the ODE state vector (see
[Time integration](#time-integration)).

Invariants tying the parameters together:

- `N` is even (a fine block covers `N/2` cells at its parent's spacing).
- `N ≥ 2G` (filling `G` coarse ghost layers by restriction reads `2G`
  fine interior layers).
- The prolongation stencil for `G` fine ghost layers must fit within the
  coarse neighbor's interior plus ghosts; this couples `G` to the
  operator order — higher-order operators need larger `G`.

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
  symmetries).

### Data layout

Two arrays exist:

- the **state vector**: a flat vector of length `N^D · nvars · nblocks`
  holding leaf interiors only — this is what ODE integrators see;
- the **working array**: one big persistent `(D+2)`-dimensional array
  holding all leaf blocks including ghosts:

      work :: Array{T, D+2}   # size (N+2G, ..., N+2G, nvars, nblocks)

- One array for all variables (decided against per-variable arrays).
- Cell indices vary fastest (GPU coalescing); blocks are ordered by
  Morton key.
- Element type `T` is generic; `Float64` default, `Float32` relevant for
  GPUs.
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

## Operations

### Ghost filling

Under 2:1 balance there are exactly three cases per ghost region:

1. **Same-level neighbor:** direct copy.
2. **Coarser neighbor** (fine ghosts): **prolongation** — interpolation
   from coarse cells.
3. **Finer neighbor** (coarse ghosts): **restriction** — interpolation
   from fine cells.

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
invoked per boundary face after the inter-block phases.

Edge and corner ghost regions are always filled — some stencils don't
need them, but filling unconditionally is simpler, and cross-derivative
stencils do. Application kernels are strictly block-local: neighbor data
is visible only through ghost cells.

Neighbor finding is a regridding-frequency operation; ghost filling runs
at every RHS evaluation. The ghost-fill **exchange schedule** — the flat
list of copy/prolongation/restriction source–target region pairs — is
therefore precomputed whenever the tree changes and cached;
`fill_ghosts!` only replays it. Tree queries (`neighbor_keys` and
friends) must never appear in the per-evaluation path.

Restriction is otherwise only needed when coarsening during regridding
and for analysis/output — there is no periodic "restrict fine onto
coarse" step, since no overlapping coarse data exists.

### Operators

Prolongation and restriction are **symmetric**: both are interpolation
operators with a configurable accuracy order, matched to the
application's discretization order. Plain `2^D` averaging is a valid
restriction only at 2nd order — it is exact for finite-volume cell
*averages*, but only 2nd-order accurate for point values at cell
centers; high-order finite differencing (e.g. for the Einstein
equations) needs correspondingly high-order restriction at coarse-fine
interfaces, which are everywhere on a tree mesh. Whether an operator is
conservative is a property of the operator, not of the mesh.

The package defines the operator **interface** and ships polynomial
operators of configurable order; the default order is 2 (linear
interpolation, whose restriction counterpart is the `2^D` average). The
application chooses `G`; the package verifies at construction that `N`,
`G`, and the requested operator orders satisfy the invariants listed
under [Blocks](#blocks). Physics-specific operators —
hydro-aware limited interpolation, primitive-variable-based prolongation
— live in application packages and plug into the same interface. *(This
resolves the original sketch's open question about where hydro-specific
operators belong.)*

Operators are configured per field set, not per variable. Per-variable
selection (e.g. conservative for density, plain for velocity) is
deferred to M8, whose face-centered support forces an interface
extension anyway.

### Conservation at coarse-fine faces

With a global timestep, conservation requires only that the flux a
coarse cell sees on a coarse-fine face equals the area-weighted sum of
the `2^(D-1)` fine-face fluxes — a purely spatial condition, enforced
within a single RHS evaluation. No flux registers or time-accumulated
corrections are needed (they only exist to bridge subcycled timesteps).

Mechanically this makes a conservative RHS **two-phase**: (i) all blocks
compute face fluxes, (ii) fluxes at coarse-fine faces are restricted
onto the coarse side (crossing rank boundaries under MPI), (iii) all
blocks apply the flux divergence. Fluxes live on faces, so this
machinery — face-centered field storage, copy/restriction/prolongation
of face data, and the corresponding MPI exchange — arrives together with
face-centered variable support (milestone M8), not before. Until then
applications are non-conservative at coarse-fine interfaces (fine for
the wave equation and the Einstein equations).

### Regridding

1. The application supplies a flagging function (per cell or per block)
   marking blocks for refinement or coarsening.
2. Marks are completed to maintain 2:1 balance (refinement ripples
   outward; coarsening happens only when all `2^D` siblings are marked
   and balance permits).
3. The new sorted key list is built; a new data array is allocated.
4. Data transfer: same-level blocks are copied, newly refined blocks are
   prolongated from their parent, coarsened blocks are restricted from
   their children.

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

Indicative only — names and signatures will evolve:

    # mesh
    forest = Forest{D}(roots=(M₁,...,M_D), extents, periodic, N, G)
    refine!(forest, keys); coarsen!(forest, keys)  # with 2:1 completion

    # fields: block arrays over the forest, ghosts included
    state = FieldSet(forest, nvars)   # evolved variables
    aux   = FieldSet(forest, naux)    # non-evolved fields

    # ghost filling (phased; operators configurable per field set)
    fill_ghosts!(state; prolongate=..., restrict=..., boundary=...)

    # global time step (application's choice)
    dt = cfl * minimum_spacing(forest)

    # ODE coupling
    u = flatvector(state)             # interiors only
    scatter!(state, u)                # working array ← flat vector
    # application kernels write du directly in flat (interior) layout

    # regridding
    regrid!(forest, state, aux; flag=...)  # flag → balance → transfer

    # kernels: launched per block via KernelAbstractions
    map_blocks!(kernel!, state, aux)

The conservative two-phase RHS (face-centered field sets, face-flux
restriction) extends this interface in M8.

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
  described in [Ghost filling](#ghost-filling).
- **GPU:** all kernels (ghost fill, prolongation, restriction,
  application RHS) are written with **KernelAbstractions.jl** from the
  start, so the CPU implementation is already the GPU implementation.
  The leaf data array lives resident on the device. Regridding splits
  cleanly: only the flagging kernel runs on the device; the driver logic
  (mark completion, 2:1 balance, key rebuild) runs on the host; block
  data transfer (copy/prolongate/restrict into the new array) runs on
  the device.
- **MPI:** the sorted Morton curve is split into contiguous per-rank
  ranges. Ghost exchange communicates face/edge/corner cell data between
  ranks; prolongation/restriction happen on the owner of the finer data.
  Regridding rebuilds and repartitions the curve. With a global `dt` and
  uniform blocks every block costs the same, so partitioning by equal
  block counts along the curve is already load-balanced. Exchange of
  face-centered data (for flux matching) is added in M8. CUDA-aware MPI
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

- Volume-weighted `internalnorm` for adaptive integrators (post-M3).
- Per-variable (rather than per-field-set) operator selection (M8).
- The detailed design of face-centered field sets (deliberately deferred
  to M8).

## Milestones

Each milestone has a concrete acceptance test; serial correctness is
established before any parallelism.

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
  polynomial data reproduced exactly up to operator order across all
  face/edge/corner configurations, including periodic wraparound and
  three-level corners.
- **M3 — Wave equation + OrdinaryDiffEq.** Scalar wave in 2nd-order
  form (state `(u, ∂ₜu)`, 2nd-order centered Laplacian), periodic cube,
  static two-level refinement over a sub-box, manual `f!`, fixed-`dt`
  RK4. *Accept:* volume-weighted L2/L∞ errors against the exact
  sine-mode solution converge at 2nd order.
- **M4 — Regridding.** Flag → balance → rebuild → transfer; the
  initial-data cycle; integrator reinit. *Accept:* the initial-data
  cycle converges to a stable hierarchy; a moving refined region tracks
  a propagating pulse without artifacts; conservation of transferred
  data.
- **M5 — Multi-threading.** Threaded loops over blocks. *Accept:*
  results match serial to roundoff; scaling measurement on a many-core
  node.
- **M6 — GPU.** CUDA backend via KernelAbstractions; device-resident
  data. *Accept:* M3 convergence results reproduced on GPU; kernel
  benchmarks.
- **M7 — MPI.** Curve partitioning, distributed ghost exchange,
  distributed regridding. *Accept:* results match serial; weak-scaling
  smoke test; then MPI+GPU with CUDA-aware MPI.
- **M8 — Face-centered variables, conservation, hydro toy.**
  Face-centered field sets (copy/restriction/prolongation and MPI
  exchange of face data), face-flux restriction, the two-phase
  conservative RHS, simple hydro test. *Accept:* global conservation to
  roundoff across coarse-fine faces.
- **M9 — I/O and visualization.** HDF5 output, checkpoint/restart, VTK
  export.
