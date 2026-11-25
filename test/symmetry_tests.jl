# Symmetries on forests of patches (M21; "Symmetries on patches" in
# CODE.md). Step 2: the kinds of a patch's own faces, a mirror face's label
# and the parity declared under it. The oracle is a brick with reflecting
# and outflow faces, laid out as two patches whose axes are swapped, so
# that a mirror plane is a different logical face on each patch; M10's
# tests hold the brick's faces against the full domain.

using TreeAMR: patch_keywords, mirror_label, mirrored_labels, ForestDigest, facekind,
               ReflectingFace, OutflowFace, OuterFace

# The layouts: a name, the brick, the two patches' axes (`laid_out`), the
# brick's reflecting and outflow faces. The mirrored brick dimensions run
# forward in both patches, so that every mirror face of the brick is the
# same side of a patch, and the vertex-like wall rows agree.
const MIRROR_LAYOUTS = [
    ("2D swapped", (4, 3), ((1, 2), (2, 1)),
     ((false, false), (true, true)), ((false, false), (false, false))),
    ("2D swapped, outflow above", (4, 3), ((1, 2), (2, 1)),
     ((false, false), (true, false)), ((false, false), (false, true))),
    ("3D cyclic", (4, 2, 3), ((1, 2, 3), (2, 3, 1)),
     ((false, false), (true, false), (true, false)),
     ((false, false), (false, false), (false, true)))]

# The patches' own faces for the brick's `reflecting` and `outflow`, through
# the layout: axis `a` of patch `p` is the brick's axis `|σ[a]|`, so its
# faces are that axis' faces, reversed where `σ[a] < 0`, and a mirror face
# carries the brick axis as its label. The cut, along brick axis 1, is
# glued and has no kind.
function mirror_fields(σ, reflecting, outflow)
    D = length(σ)
    side(a, s) = σ[a] > 0 ? s : 3 - s
    mirror = ntuple(a -> ntuple(s -> (b = abs(σ[a]); b != 1 &&
                                      reflecting[b][side(a, s)] ? b : 0), 2), D)
    out = ntuple(a -> ntuple(s -> (b = abs(σ[a]); b != 1 && outflow[b][side(a, s)]), 2), D)
    return mirror, out
end

# The layout's forest of patches with the brick's faces as the patches' own,
# and the brick with them as its own, on the images of the patches' leaves.
function mirrored_layout(roots, σs, reflecting, outflow)
    laid, place = laid_out(roots, σs)
    kw = patch_keywords(laid)
    patches = [merge(p, NamedTuple{(:mirror, :outflow)}(mirror_fields(σ, reflecting,
                                                                      outflow)))
               for (p, σ) in zip(kw.patches, σs)]
    patched = Forest(patches; N=4, glue=kw.glue)
    return patched, place
end
mirrored_brick(patched, place, roots, reflecting, outflow) =
    Forest(roots; N=4, reflecting=reflecting, outflow=outflow,
           leaves=laid_out_leaves(patched, place, roots))

# Two variables, the second odd under mirror label 2 (the brick's axis 2)
# and even elsewhere, so that a parity taken along the patch's logical
# dimension rather than its label has the wrong sign.
mirror_parity(D) = [ntuple(_ -> EvenParity, D),
                    ntuple(d -> d == 2 ? OddParity : EvenParity, D)]
mirror_fieldset(forest, C) =
    FieldSet(forest, 2; G=1, centering=C, P=first(C) === :element ? 1 : nothing,
             parity=mirror_parity(length(C)))

# The image of a leaf's box in the brick, its low corner.
leaf_image(forest, place, k) =
    (b = patch_leafbox(forest, k); p = patch(forest, k);
     min.(place(p, b[1]), place(p, b[2])))

@testset "A mirror face is the patch's own, with its label: $name" for (name, roots, σs,
                                                                       refl, out) in
                                                                      MIRROR_LAYOUTS
    # The label must reach the face of every patch it was given on, and the
    # forest-wide kinds must be left alone: a label stored on the wrong face,
    # or a patch's face read as the forest's, mirrors the wrong plane.
    D = length(roots)
    patched, _ = mirrored_layout(roots, σs, refl, out)
    for (p, σ) in enumerate(σs), a in 1:D, s in 1:2
        b = abs(σ[a])
        bs = σ[a] > 0 ? s : 3 - s
        kind = facekind(patched, p, a, s)
        if b != 1 && refl[b][bs]
            @test kind == ReflectingFace && mirror_label(patched, p, a, s) == b
        elseif b != 1 && out[b][bs]
            @test kind == OutflowFace && mirror_label(patched, p, a, s) == 0
        else
            @test kind == OuterFace
        end
    end
    @test mirrored_labels(patched) == ntuple(b -> b != 1 && any(refl[b]), D)
    @test !any(any, patched.reflecting) && !any(any, patched.outflow)
    # The keywords rebuild the forest, faces and all.
    kw = patch_keywords(patched)
    again = Forest(kw.patches; N=4, glue=kw.glue)
    @test ForestDigest(again, UInt(0), false).brick ==
          ForestDigest(patched, UInt(0), false).brick
end

@testset "Mirror faces patch by patch fill as the brick's faces: $name" for (name, roots,
                                                                            σs, refl,
                                                                            out) in
                                                                           MIRROR_LAYOUTS
    # The brick's reflecting and outflow faces, given patch by patch with
    # the swapped axes: the mirrored ghosts must be the brick's bit for bit,
    # the odd variable's sign included, in every centering. Refined against
    # the mirror plane, where a mirrored prolongation reads the parent's
    # mirrored ghosts, to rounding (the sums run in the patch's order of
    # dimensions). No ghost may be left unwritten.
    D = length(roots)
    for refined in (false, true), C in patch_centerings(Val(D))
        patched, place = mirrored_layout(roots, σs, refl, out)
        refined && patch_refine_one!(patched, place)
        fsp = mirror_fieldset(patched, C)
        if any(σ -> map(abs, σ) != ntuple(identity, D), σs) && !all(==(C[1]), C)
            # A layout the swap does not carry is the glue's refusal, as before.
            @test_throws "hold their points alike" GhostSchedule(fsp, patch_ops(fsp))
            continue
        end
        nnan, _ = undefined_patch_ghosts(fsp, patch_ops(fsp), place)
        @test nnan == 0
        brick = mirrored_brick(patched, place, roots, refl, out)
        fsb = mirror_fieldset(brick, C)
        fill_patch_arbitrary!(fsb, (p, x) -> x)
        fill_ghosts!(fsb, GhostSchedule(fsb, patch_ops(fsb));
                     boundary=patch_arbitrary_hook((p, x) -> x))
        result = compare_through_layout(fsp, fsb, place)
        @test result !== nothing
        result === nothing && continue
        worst, npoints = result
        @test refined ? worst < 1e-14 : worst == 0
        @test npoints > 0
    end
end

# A few explicit steps of `u ← u + dt Δu` over the owned points of `fs`,
# its ghosts filled before each: the second differences, summed over the
# dimensions in the forest's own order, at every owned point.
function laplace_steps!(fs::FieldSet{T,D}, schedule, hook; steps=3, dt=T(1) / 64) where {T,D}
    owned = CartesianIndices(ntuple(d -> (fs.G[d] + 1):(fs.G[d] + fs.forest.N), D))
    for _ in 1:steps
        fill_ghosts!(fs, schedule; boundary=hook)
        new = copy(fs.work)
        for b in 1:nblocks(fs), v in 1:fs.nvars, I in owned
            lap = zero(T)
            for d in 1:D
                e = CartesianIndex(ntuple(a -> a == d ? 1 : 0, D))
                lap += fs.work[I + e, v, b] - 2 * fs.work[I, v, b] + fs.work[I - e, v, b]
            end
            new[I, v, b] = fs.work[I, v, b] + dt * lap
        end
        fs.work .= new
    end
    fill_ghosts!(fs, schedule; boundary=hook)
    return fs
end

@testset "Mirror faces patch by patch regrid and evolve as the brick's: $name" for
        (name, roots, σs, refl, out) in MIRROR_LAYOUTS
    # Every block on the mirror plane of brick axis 2 refined by `regrid!`,
    # on both sides of the cut, so that the glue stays conforming as the
    # brick does without a rule; then coarsened back. The children at the
    # wall are prolongated from their parents' mirrored ghosts, which a
    # wrong label or a face read from the wrong patch would get wrong in
    # the interior. Then a few explicit steps of a Laplacian, whose ghosts
    # are the mirror's: the patches must stay the brick to rounding.
    D = length(roots)
    for C in (cellcentered(D), vertexcentered(D))
        patched, place = mirrored_layout(roots, σs, refl, out)
        hookp, hookb = patch_arbitrary_hook(place), patch_arbitrary_hook((p, x) -> x)
        brick = mirrored_brick(patched, place, roots, refl, out)
        fsp, fsb = mirror_fieldset(patched, C), mirror_fieldset(brick, C)
        fill_patch_arbitrary!(fsp, place)
        fill_patch_arbitrary!(fsb, (p, x) -> x)
        for phase in (Refine, Coarsen)
            onplane(forest, pl, k) = leaf_image(forest, pl, k)[2] == 0
            flagged(forest, pl, fs) =
                [phase == Refine ? (onplane(forest, pl, blockkey(fs, b)) ? Refine : Keep) :
                 (level(blockkey(fs, b)) > 0 ? Coarsen : Keep) for b in 1:nblocks(fs)]
            @test regrid!(patched, fsp => GhostSchedule(fsp, patch_ops(fsp));
                          flags=flagged(patched, place, fsp), boundary=hookp)
            @test regrid!(brick, fsb => GhostSchedule(fsb, patch_ops(fsb));
                          flags=flagged(brick, (p, x) -> x, fsb), boundary=hookb)
            @test laid_out_leaves(patched, place, roots) == brick.leaves
            fill_ghosts!(fsp, GhostSchedule(fsp, patch_ops(fsp)); boundary=hookp)
            fill_ghosts!(fsb, GhostSchedule(fsb, patch_ops(fsb)); boundary=hookb)
            worst, npoints = compare_through_layout(fsp, fsb, place)
            @test worst < 1e-13
            @test npoints > 0
        end
        laplace_steps!(fsp, GhostSchedule(fsp, patch_ops(fsp)), hookp)
        laplace_steps!(fsb, GhostSchedule(fsb, patch_ops(fsb)), hookb)
        worst, _ = compare_through_layout(fsp, fsb, place)
        @test worst < 1e-12
    end
end

@testset "A patch's own faces are refused where they cannot be, with the reason" begin
    # Each refusal names what the face would contradict: a label outside
    # the axes, a face both mirrored and extrapolated, a glued face made a
    # wall, a face set by the forest-wide keyword too, one label on two
    # logical axes of a patch, a shape that is not one pair per dimension,
    # and a field set without a parity under a label some face mirrors.
    glue = patch_glue_both((1, 1, 2), (2, 1, 1), (1, 2))
    two(m1, m2; kw...) = Forest([(roots=(2, 2), m1...), (roots=(2, 2), m2...)]; N=4,
                                glue=glue, kw...)
    @test_throws "outside 1:2" two((mirror=((0, 0), (3, 0)),), ())
    @test_throws "both reflecting" two((mirror=((0, 0), (2, 0)),
                                        outflow=((false, false), (true, false))), ())
    @test_throws "is glued to face" two((mirror=((0, 1), (0, 0)),), ())
    @test_throws "is set one way" two((mirror=((0, 0), (2, 0)),), ();
                                      reflecting=((false, false), (true, false)))
    @test_throws "two logical axes of one patch" two((mirror=((1, 0), (1, 0)),), ())
    @test_throws "is not 2 pairs" two((mirror=((0, 0),),), ())
    @test_throws "is not 2 pairs" two((outflow=((0, 0), (0, 0)),), ())
    @test_throws "and the kinds of its own faces" two((walls=1,), ())
    forest = two((mirror=((0, 0), (2, 0)),), ())
    @test_throws "has NoParity under mirror label 2" FieldSet(forest, 1; G=1,
                                                              parity=[(EvenParity,
                                                                       NoParity)])
    # The label, not the dimension: no parity is needed under label 1.
    @test FieldSet(forest, 1; G=1, parity=[(NoParity, EvenParity)]) isa FieldSet
end

# --- Step 3: turns as glue entries -------------------------------------------

# A brick's seam written as glue entries of the one patch it is: the quarter
# turn's low face of axis 1 glued to the low face of axis 2 and back, turned
# by 1 and 3; the half turn's low face of axis 1 glued to itself, reversed
# along axis 2, turned by 2.
function seam_glue(D, kind)
    if kind === :quarter
        σ = ntuple(e -> e == 1 ? -2 : e == 2 ? 1 : e, D)
        τ = ntuple(e -> e == 1 ? 2 : e == 2 ? -1 : e, D)
        return [(from=(1, 1, 1), to=(1, 2, 1), map=σ, turn=1),
                (from=(1, 2, 1), to=(1, 1, 1), map=τ, turn=3)]
    end
    return [(from=(1, 1, 1), to=(1, 1, 1), map=ntuple(e -> e <= 2 ? -e : e, D), turn=2)]
end

# The brick with its seam, in 3D with a mirror at the low face of axis 3
# beneath the turn, as an octant's, and the one patch with the same leaves.
function seam_forests(D, kind; refine=false, N=4)
    roots = ntuple(_ -> 2, D)
    zlow = ntuple(d -> d == 3 ? (true, false) : (false, false), D)
    turn = kind === :quarter ? (rotating=(1, 2),) : (halfturn=(1, 2),)
    brick = Forest(roots; N=N, reflecting=zlow, turn...)
    if refine
        refine!(brick, brick.leaves[1])
        balance!(brick)
        refine!(brick, brick.leaves[end])
        balance!(brick)
    end
    patched = Forest([(roots=roots, mirror=ntuple(d -> d == 3 ? (3, 0) : (0, 0), D))];
                     N=N, glue=seam_glue(D, kind), leaves=brick.leaves, turn...)
    return brick, patched
end

# Three variables, a scalar odd under the mirror of axis 3 and a vector in the
# plane, with the map of each turn.
seam_fieldset(forest, kind, C; walls=:half_open, G=1) =
    FieldSet(forest, 3; G=G, centering=C, walls=walls,
             parity=[ntuple(d -> d == 3 ? OddParity : EvenParity, length(C)),
                     ntuple(_ -> EvenParity, length(C)), ntuple(_ -> EvenParity, length(C))],
             (kind === :quarter ? (rotation=(1, -3, 2),) : (halfturn=(1, -2, -3),))...)
seam_data(x) = (sin(x[1] + 2x[2] + x[end] / 3), cos(3x[1] - x[2]), x[1] * x[2] + x[end])
seam_hook() = boundary_by_coordinates((x, v) -> seam_data(x)[v])

function fill_seam!(fs, ops=PATCH_OPS)
    fill!(fs.work, NaN)
    fill_by_coordinates!((x, v) -> seam_data(x)[v], fs)
    schedule = GhostSchedule(fs, ops)
    fill_ghosts!(fs, schedule; boundary=seam_hook())
    return schedule
end

@testset "A brick's seam written as a turned glue fills as the seam: D=$D, $kind" for
        D in (2, 3), kind in (:quarter, :half)
    # The glue form's whole oracle: the same leaves, the same data, every
    # stored point equal bit for bit to the seam's, the corner at the axis
    # (two turned faces composed) and the half turn's shifted slab included,
    # uniform and refined at both ends of the seam, half-open and with
    # symmetric walls. A turn's variable map or sign taken from the wrong
    # column, a composition that differs from the seam's `r = 2`, or a slab
    # partition that is not the image of the blocks' owned points, differs.
    for refine in (false, true), C in (cellcentered(D), vertexcentered(D)),
        walls in (:half_open, :symmetric)
        walls === :symmetric && C[1] === :cell && continue
        brick, patched = seam_forests(D, kind; refine=refine)
        fb = seam_fieldset(brick, kind, C; walls=walls)
        fp = seam_fieldset(patched, kind, C; walls=walls)
        fill_seam!(fb)
        sp = fill_seam!(fp)
        @test isequal(fb.work, fp.work)
        @test !any(isnan, fp.work)
        @test any(g -> g.turned && g.orientation != 0, sp.phase1)
    end
end

@testset "A brick's seam written as a turned glue regrids and evolves as the seam: $kind" for
        kind in (:quarter, :half)
    # The leaves on the seam refined by `regrid!`, conformity carrying each
    # one's image with it on both forms, then coarsened back; between and
    # after, three explicit Laplacian steps. The glue form must keep the
    # seam's leaves and values bit for bit throughout.
    D = 2
    for C in (cellcentered(D), vertexcentered(D))
        brick, patched = seam_forests(D, kind)
        fb, fp = seam_fieldset(brick, kind, C), seam_fieldset(patched, kind, C)
        fill_seam!(fb)
        fill_seam!(fp)
        for phase in (Refine, Coarsen)
            flags(fs) = [phase == Refine ?
                         (any(iszero, Base.front(Base.front((leafbox(fs.forest,
                                                                     blockkey(fs, b))[1]...,
                                                             0, 0)))[1:2]) ? Refine : Keep) :
                         (level(blockkey(fs, b)) > 0 ? Coarsen : Keep) for b in 1:nblocks(fs)]
            @test regrid!(brick, fb => GhostSchedule(fb, PATCH_OPS); flags=flags(fb),
                          boundary=seam_hook())
            @test regrid!(patched, fp => GhostSchedule(fp, PATCH_OPS); flags=flags(fp),
                          boundary=seam_hook())
            @test patched.leaves == brick.leaves
            laplace_steps!(fb, GhostSchedule(fb, PATCH_OPS), seam_hook())
            laplace_steps!(fp, GhostSchedule(fp, PATCH_OPS), seam_hook())
            @test isequal(fb.work, fp.work)
        end
    end
end

@testset "A turned glue between two patches fills as the brick's seam" begin
    # The quadrant cut into four patches, so that the seam joins patch 3's
    # low face of axis 1 to patch 2's low face of axis 2, two different
    # patches, beside patch 1's seam with itself, and four patches meet at
    # the cut's corner, composed as the axis is. Every stored point must be
    # the brick seam's at the same position, bit for bit, uniform and
    # refined once everywhere.
    cut, _ = brick_cut((2, 2), ((1,), (1,)))
    kw = patch_keywords(cut)
    glue = [[merge(g, (turn=0,)) for g in kw.glue];
            [(from=(1, 1, 1), to=(1, 2, 1), map=(-2, 1), turn=1),
             (from=(1, 2, 1), to=(1, 1, 1), map=(2, -1), turn=3),
             (from=(3, 1, 1), to=(2, 2, 1), map=(-2, 1), turn=1),
             (from=(2, 2, 1), to=(3, 1, 1), map=(2, -1), turn=3)]]
    for refined in (false, true), C in (cellcentered(2), vertexcentered(2))
        patched = Forest(kw.patches; N=4, glue=glue, rotating=(1, 2))
        brick = Forest((2, 2); N=4, rotating=(1, 2))
        if refined
            patch_refine_all!(patched)
            patch_refine_all!(brick)
        end
        fp, fb = seam_fieldset(patched, :quarter, C), seam_fieldset(brick, :quarter, C)
        fill_seam!(fp)
        fill_seam!(fb)
        result = compare_matching_blocks(fp, fb)
        @test result !== nothing && result[1] == 0 && result[2] > 0
    end
end

@testset "A turned glue keeps conformity, and an interpolated set regrids freely at it" begin
    # `balance!` must refine a leaf's image across a turned face, as across a
    # brick's seam. And a turned glue is not a kinked one: an interpolated
    # set copies across it, owns no appendix there, needs no atlas where no
    # glue kinks, and is not held to M20's rule that a leaf on a glued face
    # keeps its level — the central box of a symmetric ball regrids freely.
    brick, patched = seam_forests(2, :quarter)
    k = patched.leaves[findfirst(k -> leafbox(patched, k)[1] == (0, 1), patched.leaves)]
    refine!(patched, k)
    refine!(brick, k)
    balance!(patched)
    balance!(brick)
    @test patched.leaves == brick.leaves
    @test isbalanced(patched)
    _, patched = seam_forests(2, :quarter; N=8)
    fs = FieldSet(patched, 1; G=2, centering=vertexcentered(2), rotation=(1,),
                  glue=:interpolate)
    @test !TreeAMR.hasappendix(fs)
    ops = Operators(prolongation=4, restriction=4, glue=4)
    hook = boundary_by_coordinates((x, v) -> seam_data(x)[1])
    fill_by_coordinates!((x, v) -> seam_data(x)[1], fs)
    before = length(patched.leaves)
    onseam = [any(iszero, leafbox(patched, blockkey(fs, b))[1]) ? Refine : Keep
              for b in 1:nblocks(fs)]
    @test regrid!(patched, fs => GhostSchedule(fs, ops); flags=onseam, boundary=hook)
    @test length(patched.leaves) > before
    @test any(k -> level(k) == 1 && any(iszero, leafbox(patched, k)[1]), patched.leaves)
end

@testset "A turn on patches is refused where it cannot be, with the reason" begin
    # Each refusal names what the turn would contradict.
    roots = (2, 2)
    q = seam_glue(2, :quarter)
    one(glue; kw...) = Forest([(roots=roots,)]; N=4, glue=glue, kw...)
    @test_throws "declares no turn" one(q)
    @test_throws "are both given" one(q; rotating=(1, 2), halfturn=(1, 2))
    @test_throws "is not two different axes" one(q; rotating=(1, 1))
    @test_throws "turn = 2, only" one(q; halfturn=(1, 2))
    @test_throws "is not 0, 1, 2 or 3" one([merge(q[1], (turn=4,)), q[2]]; rotating=(1, 2))
    @test_throws "turned by -r" one([q[1], merge(q[2], (turn=1,))]; rotating=(1, 2))
    high = [(from=(1, 1, 2), to=(1, 2, 2), map=(2, -1), turn=1),
            (from=(1, 2, 2), to=(1, 1, 2), map=(-2, 1), turn=3)]
    @test_throws "glues a low face to a low face" one(high; rotating=(1, 2))
    @test_throws "in the plane" Forest([(roots=(2, 2, 2), mirror=((0, 0), (0, 0), (2, 0)))];
                                       N=4, glue=seam_glue(3, :quarter), rotating=(1, 2))
    forest = one(q; rotating=(1, 2))
    @test_throws "needs `rotation`" FieldSet(forest, 1; G=1)
    # The keywords rebuild a turned forest, turns and plane included.
    kw = patch_keywords(forest)
    again = Forest(kw.patches; N=4, glue=kw.glue, rotating=kw.rotating)
    @test ForestDigest(again, UInt(0), false).brick ==
          ForestDigest(forest, UInt(0), false).brick
end

# --- Step 4: the fold --------------------------------------------------------

@testset "Interpolated ghosts beside a symmetry plane are the full layout's: $sym, $(C[1])" for
        sym in (:mirror, :ownmirror, :half), C in (vertexcentered(2), cellcentered(2))
    # The ghosts of a set interpolated across the strip's kinked glues, beside
    # its mirror and beside its half turn's seam: a ghost whose point lies
    # past the plane is located there by the atlas, folded back into the
    # domain through the mirror or the turned glue, and is the symmetry's
    # image of the value there — the parity's sign, or the turned vector's
    # components negated — and a stencil at a folded point reads the
    # source's own mirrored or turned ghosts. The reduced strip must hold
    # the full strip's values at every stored point, to rounding. A ghost
    # not folded, a sign or a variable map left out, a stencil that cannot
    # reach the symmetry ghosts, or a half turn's slab partitioned without
    # the appendix's plane, differs at the scheme's order or is refused.
    reduced, full = strip_forest(sym), strip_forest(:full)
    fsr, sr = strip_set(reduced, sym, C)
    fsf, _ = strip_set(full, sym, C; full=true)
    @test !any(isnan, fsr.work)
    worst, n = strip_difference(fsr, fsf)
    @test worst < 1e-12
    @test n > 0
    @test any(!iszero, sr.glueplan.fcol) || any(!=(1), sr.glueplan.vcol)
end

@testset "interpolate! past a symmetry plane gives the full layout's values: $sym" for
        sym in (:mirror, :ownmirror, :half)
    # Physical points on both sides of the strip's mirror or half turn's
    # seam, through the atlas, interpolated in an element set (a forest of
    # patches interpolates element sets, whose polynomials read no ghost):
    # a point past the plane is folded back into the reduced strip through
    # its patch's own faces and turned glues, its value the symmetry's image
    # of the value there, and its first and second derivatives carried back
    # in physical space. The reduced strip must answer as the full one, to
    # rounding. A `PatchPoint` past the plane, without an atlas, gets its
    # value; past a patch's own symmetry face its derivatives, which only
    # the atlas's maps can carry back, are refused with the reason, while
    # past a forest-wide mirror they are flipped logically, as before.
    function element_strip(forest, full)
        nv = length(strip_data(Val(sym), (0.0, 0.0)))
        fs = FieldSet(forest, nv; G=1, centering=elementcentered(2), P=3,
                      (full ? (;) : strip_keywords(Val(sym)))...)
        fill_by_coordinates!(WithPatch((ξ, p, v) ->
                                           strip_data(Val(sym),
                                                      TreeAMR.chart_map(STRIP, ξ, p))[v]), fs)
        return fs
    end
    fsr = element_strip(strip_forest(sym), false)
    fsf = element_strip(strip_forest(:full), true)
    xs = [(-1.9 + 3.8 * ((7j) % 31) / 31, -0.75 + 1.5 * ((11j) % 29) / 29) for j in 1:60]
    @test count(x -> x[2] < 0, xs) > 20
    derivs = ((0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2))
    basis = ElementPolynomial()
    r = interpolate(fsr, xs, basis; derivs=derivs, atlas=STRIP)
    f = interpolate(fsf, xs, basis; derivs=derivs, atlas=STRIP)
    @test maximum(abs, r.values - f.values) < 1e-10 * maximum(abs, f.values)
    ps = [PatchPoint((x[1] / 2, -abs(x[2]) / 2), 1) for x in xs[1:10]]
    @test maximum(abs, interpolate(fsr, ps, basis).values -
                       interpolate(fsf, ps, basis).values) < 1e-12
    if sym === :mirror
        @test maximum(abs, interpolate(fsr, ps, basis; derivs=((1, 0), (0, 1))).values -
                           interpolate(fsf, ps, basis; derivs=((1, 0), (0, 1))).values) <
              1e-10
    else
        @test_throws "derivatives are asked for without an atlas" interpolate(
            fsr, ps, basis; derivs=((1, 0),))
    end
end

@testset "interpolate! through a turned glue is the full plane's: $kind" for
        kind in (:quarter, :half)
    # A brick's quadrant (or half plane) written as one patch glued to
    # itself by a turn, through the identity atlas, against the full plane:
    # points beyond the seam are folded logically across the turned entry
    # and their derivatives carried back by the turn in physical space. The
    # data have the symmetry — a scalar a quarter turn leaves alone and its
    # gradient, a vector that turns with the plane — so the folded answer
    # must be the full plane's, to rounding. The quarter turn exchanges the
    # plane's axes, so a wrong permutation of the derivatives' multi-indices,
    # or a wrong sign, differs.
    # The quarter turn is about the origin; the half turn about `(0, 1)`,
    # the centre line of the brick's `y ∈ [0, 2]`, so its full domain is the
    # strip `y ∈ [0, 2]` and its data are symmetric about that point.
    _, patched = seam_forests(2, kind; N=8)
    full = kind === :quarter ? Forest((4, 4); N=8, extents=((-2.0, 2.0), (-2.0, 2.0))) :
           Forest((4, 2); N=8, extents=((-2.0, 2.0), (0.0, 2.0)))
    c = kind === :quarter ? 0.0 : 1.0
    s(x) = x[1]^2 * (x[2] - c)^2 + cos(x[1]) + cos(x[2] - c)
    f(x) = (s(x), 2x[1] * (x[2] - c)^2 - sin(x[1]), 2x[1]^2 * (x[2] - c) - sin(x[2] - c))
    kw = kind === :quarter ? (rotation=(1, -3, 2),) : (halfturn=(1, -2, -3),)
    fp = FieldSet(patched, 3; G=1, centering=elementcentered(2), P=3, kw...)
    ff = FieldSet(full, 3; G=1, centering=elementcentered(2), P=3)
    fill_by_coordinates!((x, v) -> f(x)[v], fp)
    fill_by_coordinates!((x, v) -> f(x)[v], ff)
    xs = [(3.6 * ((7j) % 31) / 31 - 1.8, 3.6 * ((11j) % 29) / 29 - 1.8 + c)
          for j in 1:60]
    filter!(x -> kind === :quarter ? (x[1] < 0 || x[2] < 0) : x[1] < 0 && 0 < x[2] < 2, xs)
    @test length(xs) > 15
    derivs = ((0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2))
    p = interpolate(fp, xs, ElementPolynomial(); derivs=derivs, atlas=IdentityAtlas{2}())
    q = interpolate(ff, xs, ElementPolynomial(); derivs=derivs)
    @test maximum(abs, p.values - q.values) < 1e-10 * maximum(abs, q.values)
end
