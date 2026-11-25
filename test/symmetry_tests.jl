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
