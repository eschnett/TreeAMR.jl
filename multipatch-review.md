# Review of M20: finite differences across a glue

A review of the multi-patch design ("Finite differences across a glue"
in `CODE.md`) and its implementation on branch
`eschnett-claude/multi-patch-system-design-3c1205`, as of commit
`9b344c8`, 2026-11-24. The reviewer was a separate agent (Claude Fable),
given the design, plan, history and code but not the implementer's
opinions, and asked to confirm or reject each decision, Erik's and the
implementer's alike. Its experiment scripts are in the session's
scratch directory and are not kept. The implementer reran the bug
reproductions (V1–V7, P1) and confirmed each.

Timings are from an Apple M3 Pro: trust the ratios, and re-measure the
absolute numbers on Symmetry.

**Status.** The five changes recommended before downstream work are
done (`HISTORY.md`, M20, *Step 4, the review*); the rest stand as
written.

**Contents**

- [Overall verdict](#overall-verdict)
- [The seventeen decisions](#the-seventeen-decisions)
- [Complications, ranked by cost](#complications-ranked-by-cost)
- [Recommended changes, ranked](#recommended-changes-ranked)
- [Bugs and gaps](#bugs-and-gaps)

## Overall verdict

The design is sound, and its structure need not change. No decision is
rejected; six are confirmed with changes (9 in a small way, and 10, 11,
12, 14 and 15). Nothing in the design blocks later refinement at a glue,
per-patch face kinds, gradient sets or GR codes with many variables.

The implementation was not ready for downstream use on the CPU. There
were four correctness bugs and three performance defects, all cheap to
fix:

- every access through `AppendixArray` allocated on the CPU;
- `fill_by_coordinates!` never set the appendix;
- a glue stencil could read a plane that phase 2 fills later;
- a fold lost points for mixed centerings across an axis-swapping glue;
- the glue kernel was 5× slower than a nested loop, and alignment saved
  2× instead of 6× or 36×;
- `regrid!` accepted meshes that `GhostSchedule` then refused.

Most of M20's complexity is the appendix. It is the price of vertex
centering on a cubed sphere, not of a particular ownership rule.
Cell-centered interpolated sets have no appendix and work today
(a polynomial is exact to 3e−14 on the torus and the ball), but no test
filled one.

## The seventeen decisions

Decisions 1–8 are Erik's, 9–17 the implementer's.

1. **Package split** (TreeMultiPatch owns the layouts, maps and atlases;
   TreeAMR only moves data) — *confirm.* TreeAMR calls the atlas only at
   schedule build. Costs: 158 lines of ball layout duplicated in
   `test/helpers/ball.jl`, and an atlas contract TreeAMR cannot fully
   check (maps defined past the box, `chart_locate` clamped into the
   closed box, value-hashable).
2. **Interpolated ghosts** rather than fluxes, penalties, characteristic
   coupling or copies — *confirm.* The prototype's copied ghosts do not
   converge (L2 error 0.12 at every resolution). Penalties and
   characteristics fit a system second-order in space badly. Closed
   ownership is also the layout a penalty scheme would need later, so
   the appendix is not wasted if the coupling changes.
3. **Global Cartesian tensor components** — *confirm.* It is the only
   choice that keeps TreeAMR free of physics. TreeAMR cannot check it,
   and it does not cover staggered vector components (see V4).
4. **Closed ownership** — *confirm*, with a caveat on the evidence. From
   TreeAMR's side it is the simplest rule: no orientation rule on
   layouts, no owner table, no copy-back stage, and symmetric to
   rounding. Any single-owner alternative still needs storage for the
   unowned high planes or corner lines, so the appendix survives either
   way. The caveat: the evidence is a 2D scalar wave. On the torus
   closed ownership needs dissipation ε = 0.5 against 0.05 for
   half-open. The 3D ball's cube–shell faces and four-patch corners are
   unmeasured until PLAN step 5.
5. **The appendix after the cores** — *confirm.* The cores stay `N^D`,
   and brick code is untouched. The appendix is contiguous per block
   range, so a checkpoint needs one extra one-dimensional dataset. The
   same mechanism could own outer high planes later, which a radiative
   outer sphere needs. Cost: 14 `hasappendix` branch sites, 11 of them
   in `src/state.jl`, and every future function of the state layer must
   remember it.
6. **The sender computes** — *confirm.* Evaluating on the receiver would
   ship `k^D` source values per ghost. It follows the package's existing
   rule and is bit for bit over simulated ranks.
7. **No refinement boundary at a glue, for now** — *confirm.* Since the
   lateral glued faces run the full radius of every shell, shells can
   have no radial refinement structure at all, only interior islands;
   resolution there must come from roots and grading. Lifting it needs
   per-level glue stages, cross-level stencils, the appendix in
   prolongation and restriction, and per-point fold masks. None of that
   is blocked.
8. **`chart_aligned`, declared and verified** — *confirm.* It is also the
   lever for the largest remaining cost: equiangular shells in
   TreeMultiPatch would share an angle between neighbouring shells,
   giving 36 taps on the seven-patch ball and 6 on the outer shells
   instead of 216, with no TreeAMR change. Two implementation defects
   are under 14 and P2.
9. **The keywords** (`glue` a `FieldSet` property, `Operators(…; glue =
   k)` without a default, the atlas passed to `GhostSchedule`) —
   *confirm, small changes.* `glue` belongs to the set, since it changes
   the state's layout. Three gaps:
   - `adapt_to_initial_data!` builds `GhostSchedule(fs, operators)` with
     no atlas, so it throws for an interpolated set (V6).
   - `hash(atlas)` enters the layout hash. On Julia 1.13.4 a mutable
     atlas, or one holding a `Vector`, hashes differently in each
     process, so it would be refused across ranks, with a message that
     does not name the atlas. TreeMultiPatch's `isbits` layouts are
     fine.
   - `statearray(u, sets, i)` has no `appendix` keyword.
10. **Clamped stencils, `k ≤ 2(G + c)`** — *confirm with changes.* The
    window and the bound are right. The overhang rule is wrong in both
    directions: it misses a read of plane `G+N+1` (V3) and refuses a
    finer neighbour within the patch, whose ghosts phase 1 has already
    restricted (V5).
11. **`FoldMask`** — *confirm with changes.* One bit per piece is the
    right shape, and the masks are cheap (0.6 µs per block). Two
    defects: `glue_copies` follows a glue into a patch that has no node
    there (V4), and reductions over a state vector were about 17× slower
    than on a brick (P3).
12. **`regrid!`'s guard** — *confirm with changes.* Dropping marks is
    right. Two problems: the guard protects only leaves on a glued face,
    which is less than the schedule requires (V5); and throwing on what
    balance carries in makes a gradient-driven criterion near a glue a
    crash. Offer clipping, a compatible later addition.
13. **A stage of its own between phase 1 and phase 2** — *confirm.*
    Phase 1 has waited and unpacked before it, and the round waits for
    its own sends. One rule to document: the stencils read the hook's
    outer high plane, so `fill_ghosts!` without the hook leaves stale
    values in glue ghosts near the outer sphere, unchecked (S3).
14. **Location by the target's rank** — *confirm with changes* to its
    cost. The protocol is right. The build is serial host code: 5.2 µs
    and 5.3 kB allocated per ghost (`hasmethod` per ghost, three vectors
    per ghost, `oriented_neighbors` twice per direction), about 85 % of
    the schedule build on the test ball. Extrapolated to a ball of 8
    roots per side at `N = 32`, about 5 million face ghosts, that is
    roughly 26 s and 27 GB of allocation per schedule, per regrid, per
    interpolated set. Since glued blocks never change level, the located
    ghosts are invariant across regrids up to renumbering.
15. **Device support** (an untyped layout cache in `FieldSet`,
    `AppendixArray` through Adapt, `statearray(…; appendix = true)`
    uninferred, `Base.wrap` for the cores, appendix-only code behind
    barriers) — *confirm with changes.* It is consistent with the
    package's conventions. `statearray(u, fs)` infers `Array{Float64,4}`;
    a run-time appendix flag infers `Any` for every set. The defect is
    V1: `getindex` and `setindex!` were not inlined on the CPU. Also,
    `AppendixArray` is not an `AbstractArray`, and its `eltype` is
    `Any`. The 5 % compile-time figure, the Metal tests and real MPI
    were not re-checked by the reviewer.
16. **Refusing a ghost in no patch** — *confirm.* The message blames the
    map for not reaching past the box; on a coarse ball the cause is
    geometry, and it should say so.
17. **Reflecting and outflow faces refused; element sets copy** —
    *confirm for M20.* Bitant and octant balls need per-patch face kinds
    first. The unglued slots of the glue table can hold them, and an
    edge that is reflected and then interpolated fits `GluePlan` as a
    ghost at the mirrored position with a parity factor.

## Complications, ranked by cost

1. **The appendix.** It reaches `statelength`, `statevector`,
   `statearray`, `scatter!`, `gather!`, `map_blocks!`, both reductions,
   folds, norms, `GroupKey.top`, the regrid transfers, the device path,
   the checkpoint format and downstream right-hand sides. It is inherent
   to vertex-centered points on the glued faces of a cubed sphere; only
   the fold masks come from closed ownership specifically. Not yet
   updated for it at the time of review: `fill_by_coordinates!`,
   `adapt_to_initial_data!`, the multi-set `statearray`, `interpolate`,
   TreeIOHDF5.
2. **The cost of the glue stage.** `k^D` taps per ghost per variable is
   inherent to 3D interpolation with no shared coordinate. Measured: 525
   ns per ghost and variable at `k = 6`. On the test ball with 20
   variables the fill took 2.38 s at one thread, 2.32 s of it the glue
   stage. Extrapolated to `N = 32`, one glued block face (3267 ghosts)
   costs 1.7 ms per variable, against 0.66 ms for that block's
   right-hand side at 20 ns per point and variable: the fill would rival
   or exceed the right-hand side until the kernel and the layouts are
   fixed.
3. **Two guards that disagree**: `regrid!`'s (on a glued face) and
   `GhostSchedule`'s (within reach of a stencil), from decisions 7, 10
   and 12.
4. **A run-time switch behind barriers**: 17 new `inferencebarrier`
   sites and an uninferred public accessor. This is the package's
   convention, and it makes defects like V1 easy to miss, since the
   device path forces inlining and the CPU path does not.
5. **The plan's build at regrid frequency** (decision 14).
6. **`GroupKey.top`** and its keyword through `target_range`,
   `copy_stencil` and `kind_stencil`, on every set's build path. Every
   call site was read; no key has `top` set outside a copy.
7. **Documentation.** The rule lists have no M20 entry (`CODE.md`,
   "Rules that span several files"; `CLAUDE.md`, "Rules and sharp
   edges"). `CODE.md` still gives the state length as `N^D · nvars ·
   nblocks` in several places, its section header still says "not
   implemented", and `PLAN.md` and `HISTORY.md` each have a line saying
   "not designed".

## Recommended changes, ranked

Before any downstream work:

1. Inline `AppendixArray`'s `getindex` and `setindex!`.
2. Make `fill_by_coordinates!` cover the appendix, and give
   `adapt_to_initial_data!` an atlas keyword.
3. Fix the overhang rule: count plane `G+N+1` as overhang unless the
   source owns it or it is an outer face, and accept same-or-finer
   neighbours. Make `regrid!`'s guard keep what the current plan reads.
4. Have `glue_copies` skip an image that is not a node of the
   neighbour's grid, or refuse such sets.
5. Rewrite `glue_value` as nested loops bounded by the widths, and
   launch by owner (5.2× measured on full stencils; aligned ghosts
   would drop from about 250 ns to a few ns).

Before release:

6. The plan's build: hoist `hasmethod`, drop the per-ghost allocations,
   thread it, and reuse located ghosts across regrids keyed by leaf.
7. TreeIOHDF5: refuse a set with an appendix in the same release that
   admits the new TreeAMR version, then add the feature. Add
   `corelength` and the appendix layout to the companion interface.
8. API polish while it is still free: an exported "does block `b` own
   point `I`" predicate, so a right-hand side can skip the planes it
   would compute and discard (about 9 % of the work at `N = 32`);
   `eltype` for `AppendixArray`; the multi-set keyword; say that
   `appendix = false` leaves `du`'s appendix entries unwritten, not
   zero.
9. Tests: a cell-centered and a mixed-centered interpolated fill, `k`
   other than 6, and a refined second layer.
10. Documentation: one rule bullet in each list, the stale status lines,
    the state-length statements.

After M20, none blocked:

11. Equiangular shells in TreeMultiPatch.
12. Lagrange interpolation on interpolated sets (`interpolate` refuses
    it for a reason that no longer holds for them); a horizon finder on
    a finite-difference ball needs it.
13. A clipping mode for the regrid guard.
14. Per-patch face kinds, and the appendix at outer high faces.
15. Refinement at a glue.

## Bugs and gaps

### Verified by running (and reproduced by the implementer)

- **V1.** `AppendixArray` access allocated on the CPU: a read 160 B and
  90 ns, a write 96 B and 20 ns. Through `map_blocks!(…; appendix =
  true)` a kernel writing one value per point cost 96 B and 15 ns per
  point, against 0 B and 0.2 ns through the core array; with `@inline`
  at the call site, 0 B and 1.2 ns. Folds over a state vector likewise:
  71 ns and 148 B per entry, against 6 ns for the working-array form.
- **V2.** `fill_by_coordinates!` launched over `N^D` only: on the
  seven-patch ball, after it and `gather!`, all 2401 appendix entries
  were zero.
- **V3.** A stale read in the glue stage. Two patches glued at one face,
  spacing ratio about 4.5, `N = 8`, `G = 3`, `k = 6`, both root columns
  at the glue refined: the schedule builds, and 53 of 318 glue ghosts
  come out NaN after a fill from NaN-poisoned ghosts. The stencil ends
  on the source block's plane `G+N+1`, which a phase-2 prolongation
  fills later; the overhang rule did not count that plane. A second
  fill gives no NaN, so in production the ghost silently carries the
  previous fill's value. It needs the source in the second block layer,
  so it is unlikely at `N = 32`.
- **V4.** A fold lost points for mixed centering across an axis-swapping
  glue. Two patches, face (1, dim 1, high) glued to (2, dim 2, low),
  centering `(:cell, :vertex)`: `mesh_fold` counted 496 of 512 owned
  points. `glue_copies` attributed patch 2's face points to a copy in
  patch 1 that does not exist.
- **V5.** `regrid!` succeeded and the next `GhostSchedule` threw. Two
  patches, spacing ratio 2, `N = 8`, every block flagged `Refine`:
  `regrid!` returned true with the glued blocks at level 0 and the rest
  at level 1, and `GhostSchedule` then refused ("reads the ghosts of
  its source block … where no block of the same level and patch fills
  them"), leaving a regridded forest with no schedule. The refused read
  is of ghosts phase 1 restricts, so the data was valid.
- **V6.** `adapt_to_initial_data!` threw "needs the application's
  atlas" for an interpolated set.
- **V7.** `interpolate` with `Lagrange(4)` and an atlas was refused on an
  interpolated set.
- **P1.** The glue kernel: 525 ns per ghost and variable at `k = 6`,
  `D = 3`; nested loops 101 ns.
- **P2.** Alignment saved little: on the thirteen-patch ball a 36-tap
  ghost cost 294 ns and a 6-tap one 254 ns, against 534 ns for 216
  taps, because the loop still ran `K^D` iterations.
- **P3.** `volume_weighted_norm` over a set with an appendix cost 72–80
  ns per entry against 4.3 ns on a brick, most of it V1.

### Verified by reading

- **R1.** `run_glue!` and `move_appendix!` launched with a bare
  `kernel(backend)(…)`, not by owner, against both `CODE.md`'s claim
  ("on the CPU by the target's owner") and the launch-by-owner rule.
- **R2.** TreeIOHDF5 as it stands would drop the appendix silently: it
  checksums and writes `nblocks × N^D · nvars` entries and reloads the
  set without `glue`. Its exact version pins prevent this today; the
  risk is a routine version bump.
- **R3.** The comment on `glue_plan` says a ghost located back in its
  own patch is refused; `locate_glue_ghosts` has no such check.

### Suspected, not run

- **S1.** A mutable or array-holding atlas refused across ranks (the
  hash behaviour is verified, the refusal under MPI is not).
- **S2.** Following R3, a sloppy atlas would yield a value clamped onto
  the face, silently.
- **S3.** `fill_ghosts!` without the hook leaves glue ghosts near an
  outer high face reading a stale plane.
- **S4.** With `k = 2(G + 1)` on a vertex set, a located point that
  rounds onto its block's high boundary could trip "reaches outside the
  stored array" spuriously.
