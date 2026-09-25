# M10: reflecting boundaries.
#
# A reflecting face is a property of the domain, like periodicity, and
# the schedule fills its ghosts by mirrored transfers — the tangential
# stencil of the source's kind with its target rows remapped across the
# wall, times each variable's parity sign. Nothing here reaches into
# the mechanism: every claim is stated against an oracle — polynomials
# of definite parity, the doubled domain the half domain folds, a count
# of writes, and `NaN` that no stencil can wash out.

using TreeAMR: reflect_direction

const OPS4 = Operators(prolongation=4, restriction=4)

@testset "Reflecting boundaries are refused where they cannot mean anything" begin
    # A reflection declared somewhere it has no meaning, or data whose
    # mirror image is undefined, must be refused with the reason, not
    # discovered later as a wrong ghost.
    @test_throws "both periodic and reflecting" Forest((2, 2); N=8,
                                                        periodic=(true, false),
                                                        reflecting=((true, false),
                                                                    (false, false)))
    forest = Forest((2, 2); N=8, reflecting=((true, false), (false, false)))
    @test forest.reflecting == ((true, false), (false, false))
    @test Forest((2, 2); N=8).reflecting == ((false, false), (false, false))

    @test_throws "needs `parity`" FieldSet(forest, 2; G=1)
    @test_throws "one entry per variable" FieldSet(forest, 2; G=1,
                                                   parity=[EvenParity])
    @test_throws "NTuple{2,Parity}" FieldSet(forest, 1; G=1,
                                             parity=[(EvenParity,)])
    @test_throws "vector or tuple" FieldSet(forest, 1; G=1, parity=EvenParity)
    # NoParity is legal in the dimension without a reflecting face and
    # refused in the one with.
    @test FieldSet(forest, 1; G=1, parity=[(EvenParity, NoParity)]).parity ==
          [(EvenParity, NoParity)]
    @test_throws "has NoParity in dimension 1" FieldSet(forest, 1; G=1,
                                                        parity=[(NoParity, EvenParity)])
    # A single Parity stands for every dimension.
    @test FieldSet(forest, 2; G=1, parity=[OddParity, EvenParity]).parity ==
          [(OddParity, OddParity), (EvenParity, EvenParity)]
    # Over a forest without reflecting faces the parity may be omitted,
    # and is ignored when given, so nothing written before M10 changes.
    plain = Forest((2, 2); N=8)
    @test FieldSet(plain, 1; G=1).parity === nothing
    @test FieldSet(plain, 1; G=1).factors === nothing
    @test FieldSet(plain, 1; G=1, parity=[NoParity]).factors === nothing
end

@testset "The mirror splits a direction at the reflecting faces it crosses" begin
    # The schedule looks for a mirrored ghost's source in direction δ′,
    # so a wrong split sends it to the wrong block. Only a step that
    # actually leaves through a reflecting face is masked.
    forest = Forest((2, 2); N=8, reflecting=((true, false), (false, false)))
    lowleft = forest.leaves[findfirst(k -> root_position(forest, k.root) == (0, 0),
                                      forest.leaves)]
    lowright = forest.leaves[findfirst(k -> root_position(forest, k.root) == (1, 0),
                                       forest.leaves)]
    @test reflect_direction(forest, lowleft, (-1, 0)) == ((0, 0), (true, false))
    @test reflect_direction(forest, lowleft, (-1, 1)) == ((0, 1), (true, false))
    # The y faces are outer: not masked, even where they leave the domain.
    @test reflect_direction(forest, lowleft, (-1, -1)) == ((0, -1), (true, false))
    @test reflect_direction(forest, lowleft, (0, -1)) == ((0, -1), (false, false))
    # The high x face is outer, and an interior step crosses nothing.
    @test reflect_direction(forest, lowright, (1, 0)) == ((1, 0), (false, false))
    @test reflect_direction(forest, lowright, (-1, 0)) == ((-1, 0), (false, false))
end

@testset "Every ghost is defined and none is read undefined: D=$D" for D in (1, 2, 3)
    # The ordering failure the region-form hook had: at an edge or corner
    # region crossing a reflecting face, the mirrored values may be the
    # block's own prolongated ghosts, and filling them too early reads
    # them undefined. With every stored value `NaN` beforehand, an
    # unwritten ghost stays `NaN` and a read before the write makes one,
    # so a clean result over every combination of periodic, outer and
    # reflecting faces, for every centering, rules both out — and the
    # values then equal the parity polynomial everywhere.
    #
    # Every combination of face kinds, and every centering in D = 1, 2.
    # In D = 3 the full product is 1000 cases; each face-kind combination
    # there is run once, the centering rotating through all eight, which
    # keeps each centering in about sixteen combinations.
    centerings = allcenterings(Val(D))
    cases = vec(collect(Iterators.product(ntuple(_ -> FACE_KINDS, D)...)))
    bad = []
    for (i, kinds) in enumerate(cases)
        Cs = D == 3 ? [centerings[mod1(i, length(centerings))]] : centerings
        for C in Cs
            nnan, worst = undefined_ghosts(kinds, C)
            (nnan == 0 && worst < 1e-10) || push!(bad, (kinds, C, nnan, worst))
        end
    end
    @test isempty(bad)
    isempty(bad) || foreach(println, bad)
end

@testset "Parity polynomials are exact across a reflecting face: D=$D" for D in (1, 2, 3)
    # The M2 exactness claim with a wall in every dimension: data of
    # degree p-1 per dimension with the declared parity is reproduced in
    # every stored point, mirrored rows and the derived upper wall point
    # included. The same data declared with the *other* parity must
    # fail, which shows the mirror, and not the hook, fills those ghosts;
    # and all three mirrored kinds occur, so exactness is not vacuous.
    # (In 3D order 4 is left to the NaN test above, which asserts the
    # same exactness over the same walls.)
    for C in allcenterings(Val(D)), p in (D == 3 ? (2,) : (2, 4))
        for kinds in Iterators.product(ntuple(_ -> (:reflect_lo, :reflect_hi), D)...)
            nnan, worst = undefined_ghosts(kinds, C; p=p)
            @test nnan == 0
            @test worst < 1e-10
        end
    end
    kinds = ntuple(_ -> :reflect_lo, D)
    forest = faces_forest(kinds)
    f, parity = parity_data(Float64, kinds, 4)
    fs = FieldSet(forest, 2; G=ghosts_for(cellcentered(D), 4), parity=parity)
    schedule = GhostSchedule(fs, OPS4)
    counts = mirror_counts(schedule)
    @test occursin("mirrored transfers", sprint(show, schedule))
    # In 1D every mirrored region is the block's own reflection; from 2D
    # on, an edge region's mirror image lies in a tangential neighbor.
    @test counts[:copy] > 0
    D >= 2 && @test counts[:restrict] > 0 && counts[:prolong] > 0

    _, wrong = parity_data(Float64, kinds, 4; flip=true)
    fs = FieldSet(forest, 2; G=ghosts_for(cellcentered(D), 4), parity=wrong)
    fill_ghosts!(fill_by_coordinates!(f, fs), GhostSchedule(fs, OPS4))
    @test max_deviation(fs, f) > 1e-2
end

@testset "The upper wall point of a vertex-like dimension is derived: D=$D" for
        D in (1, 2)
    # Beyond a high reflecting face the wall point is shared with nobody
    # and the mirror maps it onto itself. It is derived: exactly zero for
    # an odd variable, and for an even one the folded symmetric
    # interpolant, (4u₁ - u₂)/3 at order 4 — exact for an even quadratic
    # and not for an even quartic, which is what an order-4 interpolant
    # of an even function is.
    forest = Forest(ntuple(_ -> 1, D); N=8, extents=ntuple(_ -> (0.0, 1.0), D),
                    reflecting=ntuple(d -> d == 1 ? (false, true) : (false, false), D))
    fs = FieldSet(forest, 3; G=1, centering=vertexcentered(D),
                  parity=[EvenParity, OddParity, EvenParity])
    f(x, v) = v == 1 ? 1 + (x[1] - 1)^2 : v == 2 ? (x[1] - 1) : (x[1] - 1)^4
    fill_by_coordinates!(f, fs)
    fill_ghosts!(fs, GhostSchedule(fs, OPS4); boundary=boundary_by_coordinates(f))
    wall = fs.G[1] + forest.N + 1
    row(v) = fs.work[wall, ntuple(_ -> fs.G[1] + 1, D - 1)..., v, 1]
    @test row(1) ≈ 1.0 atol = 1e-14
    @test row(2) == 0
    h = 1 / forest.N
    @test row(3) ≈ (4 * h^4 - (2h)^4) / 3 atol = 1e-14   # the interpolant, not 0
end

@testset "A reflecting face reproduces the mirrored doubled domain: D=$D" for
        D in (1, 2, 3)
    # The definitional test, for arbitrary data: a half domain with a
    # reflecting wall holds, in every stored point, what the doubled
    # domain holds with the mirrored data written out — to roundoff,
    # since a mirrored stencil sums the same terms in the other order.
    # The refinement reaches the wall with a coarse-fine face meeting it
    # tangentially. Not for a vertex-like upper wall, whose wall point
    # the half domain derives while the doubled domain owns it.
    N = 8                                        # order 4 restricts from N >= 5
    for (side, C) in ((:lo, cellcentered(D)), (:hi, cellcentered(D)),
                      (:lo, vertexcentered(D)))
        result = reflecting_vs_doubled(Val(D); side=side, centering=C, N=N)
        @test result !== nothing
        result === nothing && continue
        worst, ncells = result
        @test ncells > 0
        @test worst < 1e-13
    end
end

@testset "The schedule partitions the ghosts at reflecting faces: D=$D" for D in (1, 2, 3)
    # Every stored point outside the owned range is written exactly once —
    # by a mirrored transfer, an ordinary one, or the hook — and the hook
    # is handed only regions whose mirror image still leaves the domain
    # through an outer face. A double write would race; a gap would be
    # stale.
    N = 8
    for kinds in ((:reflect_lo, :outer, :periodic), (:reflect_both, :reflect_hi, :outer),
                  (:outer, :reflect_lo, :reflect_hi))
        ks = kinds[1:D]
        forest = faces_forest(ks; N=N)
        _, parity = parity_data(Float64, ks, 4)
        for C in allcenterings(Val(D))
            G = ghosts_for(C, 4)
            c = staggers(C)
            fs = FieldSet(forest, 2; G=G, centering=C, parity=parity)
            schedule = GhostSchedule(fs, OPS4)
            counts = write_counts(schedule)
            owned = ntuple(d -> (G[d] + 1):(G[d] + N), D)
            ok = true
            for b in 1:nleaves(forest), idx in CartesianIndices(size(counts)[1:D])
                inside = all(d -> Tuple(idx)[d] in owned[d], 1:D)
                counts[idx, b] == (inside ? 0 : 1) || (ok = false)
            end
            @test ok
            # The hook's regions leave the domain even after the mirror.
            @test all(schedule.boundaries) do r
                k = forest.leaves[r.block]
                δ′, _ = reflect_direction(forest, k, r.direction)
                any(!iszero, δ′) && isempty(neighbor_keys(forest, k, δ′))
            end
        end
    end
    # A corner shared by a reflecting and an outer face reaches the hook.
    if D >= 2
        forest = faces_forest((:reflect_lo, :outer); N=N)
        _, parity = parity_data(Float64, (:reflect_lo, :outer), 4)
        schedule = GhostSchedule(FieldSet(forest, 2; G=2, parity=parity), OPS4)
        @test any(schedule.boundaries) do r
            any(reflect_direction(forest, forest.leaves[r.block], r.direction)[2])
        end
    end
end

@testset "The conservative family is exact across a reflecting face: D=$D" for
        D in (1, 2)
    # Cell averages of a parity polynomial are themselves even or odd
    # about the wall, since the cells are mirror images of each other, so
    # the conservative family reproduces them there as it does anywhere.
    for kinds in Iterators.product(ntuple(_ -> (:reflect_lo, :reflect_hi, :outer), D)...)
        forest = faces_forest(kinds)
        f, parity = parity_data(Float64, kinds, 3)
        fs = FieldSet(forest, 2; G=2, parity=parity)
        for b in 1:nblocks(fs), v in 1:2
            h = spacing(forest, blockkey(fs, b))
            blk = blockview(fs, b, v)
            for idx in CartesianIndices(ntuple(d -> (fs.G[d] + 1):(fs.G[d] + forest.N), D))
                blk[idx] = cell_average(x -> f(x, v), coordinates(fs, b, Tuple(idx)), h)
            end
        end
        hook = function (fs, b, key, δ, region)
            h = spacing(fs.forest, key)
            for v in 1:2, idx in region
                blockview(fs, b, v)[idx] =
                    cell_average(x -> f(x, v), coordinates(fs, b, Tuple(idx)), h)
            end
        end
        ops = Operators(prolongation=3, restriction=2, family=Conservative)
        fill_ghosts!(fs, GhostSchedule(fs, ops); boundary=hook)
        worst = 0.0
        for b in 1:nblocks(fs), v in 1:2
            h = spacing(forest, blockkey(fs, b))
            blk = blockview(fs, b, v)
            for idx in CartesianIndices(blk)
                exact = cell_average(x -> f(x, v), coordinates(fs, b, Tuple(idx)), h)
                worst = max(worst, abs(blk[idx] - exact))
            end
        end
        @test worst < 1e-10
    end
end

@testset "The regrid transfer is exact at a reflecting face: D=$D" for D in (1, 2, 3)
    # A prolongation onto a fresh child at the wall reads its parent's
    # mirrored ghosts, which `regrid!` fills first; wrong ghosts there
    # would put a wrong value into the child's *interior*.
    N = D == 3 ? 4 : 8
    kinds = ntuple(d -> isodd(d) ? :reflect_lo : :reflect_hi, D)
    for C in allcenterings(Val(D))
        forest = faces_forest(kinds; N=N)
        f, parity = parity_data(Float64, kinds, 2)
        hook = boundary_by_coordinates(f)            # for the outer faces
        fs = FieldSet(forest, 2; G=1, centering=C, parity=parity)
        fill_by_coordinates!(f, fs)
        flags = flag_blocks(forest) do b, k
            level(k) == 0 ? Refine : level(k) == maxlevel(forest) ? Coarsen : Keep
        end
        @test regrid!(forest, fs => GhostSchedule(fs, OPS2C); flags=flags,
                      boundary=hook)
        fill_ghosts!(fs, GhostSchedule(fs, OPS2C); boundary=hook)
        @test max_deviation(fs, f) < 1e-12
    end
end

# The wave equation on a box with reflecting walls at x₁ = 0 and L/2,
# periodic in the other dimensions: the same two-level refinement as
# `wave_forest`'s, which is symmetric about both walls, so the half box
# is exactly the fold of the periodic box.
function reflecting_wave_forest(::Val{D}, N; L=1.0) where {D}
    forest = Forest(ntuple(d -> d == 1 ? 2 : 4, D); N=N,
                    periodic=ntuple(d -> d != 1, D),
                    reflecting=ntuple(d -> d == 1 ? (true, true) : (false, false), D),
                    extents=ntuple(d -> d == 1 ? (0.0, L / 2) : (0.0, L), D))
    targets = filter(forest.leaves) do k
        ext = block_extent(forest, k)
        all(d -> L / 4 < (ext[d][1] + ext[d][2]) / 2 < 3L / 4, 1:D)
    end
    refine!(forest, targets)
    balance!(forest)
    return forest
end

wave_parity(D, p) = [ntuple(d -> d == 1 ? p : NoParity, D) for _ in 1:2]

@testset "A reflecting half box evolves as the periodic box it folds: D=$D" for
        D in (1, 2)
    # A standing mode odd (sin) or even (cos) about both walls, evolved
    # on the half box and on the full periodic one: cell-centered, the
    # half box owns exactly the full box's points in its half, so the two
    # runs agree to roundoff and so do their errors. (Vertex-centered, the
    # half box derives the point on its upper wall instead of evolving
    # it, and the two differ at truncation level; that is the next
    # testset.)
    for (exact, par) in ((wave_exact, OddParity), (wave_exact_even, EvenParity))
        C = cellcentered(D)
        full = wave_errors(Val(D); N=16, G=2, ops=OPS4, centering=C, exact=exact)
        half = wave_errors(Val(D); N=16, G=2, ops=OPS4, centering=C, exact=exact,
                           forest=reflecting_wave_forest(Val(D), 16),
                           parity=wave_parity(D, par))
        @test half.nsteps == full.nsteps
        @test half.nblocks < full.nblocks
        @test half.linf ≈ full.linf rtol = 1e-8
    end
end

@testset "The reflecting wave study converges at the interface-order rate: D=$D" for
        D in (1, 2)
    # Order-4 operators against the 2nd-order Laplacian: rate 2, the
    # rate of the periodic study, for both centerings and both parities.
    # For a vertex-like dimension that includes the derived upper wall
    # point, an O(h⁴) interpolant read through an h⁻² stencil — the same
    # budget as a prolongated ghost's, so it may not cost the rate.
    for C in (vertexcentered(D), cellcentered(D))
        for (exact, par) in ((wave_exact, OddParity), (wave_exact_even, EvenParity))
            hs, l2, linf = Float64[], Float64[], Float64[]
            for N in (8, 16, 32)
                r = wave_errors(Val(D); N=N, G=ghosts_for(C, 4), ops=OPS4,
                                centering=C, exact=exact,
                                forest=reflecting_wave_forest(Val(D), N),
                                parity=wave_parity(D, par))
                push!(hs, r.h)
                push!(l2, r.l2)
                push!(linf, r.linf)
            end
            @test convergence_rate(hs, l2) ≈ 2.0 atol = 0.15
            @test convergence_rate(hs, linf) ≈ 2.0 atol = 0.2
        end
    end
end
