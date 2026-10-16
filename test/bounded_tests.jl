# The bounded conservative prolongation.
#
# `Operators(…; bounded = true)` makes the order-3 conservative
# prolongation keep every child between the extremes of its parent and
# the parent's face neighbours. Every claim is stated against an oracle
# that knows only the formula and the geometry — the coarse cell a fine
# cell lies in and which side of it — never the schedule's stencils or
# the kernel's way of reading the child's side off them.

# The formula, independently of the kernel: the bounded child of the
# coarse cell `I` of the periodic coarse grid `U` on sides `σ`, with the
# monotonized central slope written out as the smallest of three
# candidates that agree in sign.
oracle_mc(a, b) = sign(a) == sign(b) ?
                  sign(a) * min(abs(2a), abs(2b), abs((a + b) / 2)) : zero(a)

function bounded_child(U::AbstractArray{<:Real,D}, I::CartesianIndex{D}, σ) where {D}
    n = size(U)
    wrap(J) = CartesianIndex(ntuple(d -> mod1(J[d], n[d]), D))
    e(d) = CartesianIndex(ntuple(k -> k == d ? 1 : 0, D))
    u = U[I]
    lo = [U[wrap(I - e(d))] for d in 1:D]
    hi = [U[wrap(I + e(d))] for d in 1:D]
    s = [oracle_mc(hi[d] - u, u - lo[d]) for d in 1:D]
    M, m = max(u, lo..., hi...), min(u, lo..., hi...)
    S = sum(abs, s) / 4
    φ = S > 0 ? min(1, (M - u) / S, (u - m) / S) : 1
    return u + φ * sum(σ[d] * s[d] for d in 1:D) / 4
end

const BOUNDED = Operators(prolongation=3, restriction=2, family=Conservative,
                          bounded=true)

# The coarse cell (of spacing `h`) a point lies in, wrapped onto a
# periodic grid of `n` cells per dimension, and the side of its centre
# the point is on.
function coarse_cell(x, h, n)
    D = length(x)
    I = CartesianIndex(ntuple(d -> mod1(floor(Int, x[d] / h) + 1, n), D))
    σ = ntuple(d -> x[d] - (floor(x[d] / h) + 1 / 2) * h > 0 ? 1 : -1, D)
    return I, σ
end

@testset "A bound is refused where it means nothing, and shown only when set" begin
    # A bound on an operator it cannot bound must be refused with the
    # reason, not ignored; and the unbounded operator must print as it
    # always has.
    @test BOUNDED.bounded
    @test !Operators(prolongation=3, restriction=2, family=Conservative).bounded
    @test_throws "conservative family needs prolongation = 3 or 1" Operators(
        prolongation=5, restriction=2, family=Conservative, bounded=true)
    # Every point-value order has an interpolant to bound.
    @test Operators(prolongation=6, restriction=4, bounded=true).bounded
    @test sprint(show, Operators(prolongation=4, restriction=4, bounded=true)) ==
          "Operators(prolongation=4, restriction=4, family=PointValue, bounded=true)"
    @test sprint(show, Operators(prolongation=3, restriction=2, family=Conservative)) ==
          "Operators(prolongation=3, restriction=2, family=Conservative)"
    @test sprint(show, BOUNDED) ==
          "Operators(prolongation=3, restriction=2, family=Conservative, bounded=true)"
end

@testset "At order 1 the bound is accepted and changes nothing: D=$D" for D in (1, 2, 3)
    # Order 1 copies the parent, so it is bounded already: `bounded = true`
    # must neither be refused there nor send the copy through the order-3
    # kernel, which would read a window the order-1 stencil does not have.
    # So the fill and a regrid must be bit for bit the unbounded ones.
    plain = Operators(prolongation=1, restriction=2, family=Conservative)
    ops = Operators(prolongation=1, restriction=2, family=Conservative, bounded=true)
    @test sprint(show, ops) ==
          "Operators(prolongation=1, restriction=2, family=Conservative, bounded=true)"
    results = map((plain, ops)) do o
        forest = nested_forest(Val(D); N=D == 3 ? 4 : 8, periodic=ntuple(_ -> true, D))
        fs = FieldSet(forest, 1; G=1)
        # Rough data that are a function of position alone: the fill runs
        # threaded, so a random stream would differ between the two runs.
        fill_by_coordinates!((x, v) -> mod(sum(d -> 97.13 * (d + 0.37) * x[d], 1:D), 1), fs)
        sched = GhostSchedule(fs, o)
        @test all(g -> g.bounded == TreeAMR.UNBOUNDED, Iterators.flatten(sched.phase2))
        fill_ghosts!(fs, sched)
        filled = copy(fs.work)
        flags = flag_blocks((b, k) -> isodd(b) ? Refine : Keep, forest)
        regrid!(forest, fs => GhostSchedule(fs, o); flags=flags)
        return filled, copy(fs.work)
    end
    @test results[1][1] == results[2][1]
    @test results[1][2] == results[2][2]
end

@testset "A refined child is the bounded reconstruction of its parent: D=$D" for
        D in (1, 2, 3)
    # The definition. A periodic brick refined everywhere at once: every
    # new child is the formula applied to the coarse grid, whatever the
    # data, so a wrong side, a wrong neighbour or a wrong scaling shows.
    N = D == 3 ? 4 : 8
    forest = Forest(ntuple(_ -> 2, D); N=N, periodic=ntuple(_ -> true, D),
                    extents=ntuple(_ -> (0.0, 2.0), D))
    n = 2N
    h = 2 / n
    U = rand(MersenneTwister(40 + D), n * ones(Int, D)...)
    fs = FieldSet(forest, 1; G=1)
    fill_by_coordinates!((x, v) -> U[coarse_cell(x, h, n)[1]], fs)
    @test regrid!(forest, fs => GhostSchedule(fs, BOUNDED);
                  flags=fill(Refine, nleaves(forest)))
    worst = 0.0
    for b in 1:nblocks(fs)
        blk = blockview(fs, b, 1)
        for idx in CartesianIndices(ntuple(_ -> 2:(N + 1), D))
            I, σ = coarse_cell(coordinates(fs, b, Tuple(idx)), h, n)
            worst = max(worst, abs(blk[idx] - bounded_child(U, I, σ)))
        end
    end
    @test worst < 1e-14
end

@testset "A ghost across a coarse-fine face is the bounded reconstruction: D=$D" for
        D in (1, 2, 3)
    # The same definition for the ghost fill: on a periodic two-level
    # mesh, every fine ghost that lies in a coarse leaf is the formula
    # applied to the coarse picture of the mesh — coarse leaves as they
    # are, refined roots as the averages of their children, which is
    # what a coarse block's restricted ghosts hold.
    N = 4
    roots = 4
    forest = Forest(ntuple(_ -> roots, D); N=N, periodic=ntuple(_ -> true, D),
                    extents=ntuple(_ -> (0.0, Float64(roots)), D))
    refined(x) = all(d -> mod(x[d], roots) < 2, 1:D)
    refine!(forest, filter(k -> refined(block_extent(forest, k) .|> first), forest.leaves))
    @test maxlevel(forest) == 1
    n, h = roots * N, 1 / N
    rng = MersenneTwister(50 + D)
    U = rand(rng, n * ones(Int, D)...)
    F = rand(rng, 2n * ones(Int, D)...)
    for I in CartesianIndices(U)
        x = ntuple(d -> (I[d] - 1 / 2) * h, D)
        refined(x) || continue
        kids = CartesianIndices(ntuple(d -> (2I[d] - 1):(2I[d]), D))
        U[I] = sum(F[J] for J in kids) / 2^D
    end
    fs = FieldSet(forest, 1; G=1)
    fill_by_coordinates!(fs) do x, v
        refined(x) ? F[coarse_cell(x, h / 2, 2n)[1]] : U[coarse_cell(x, h, n)[1]]
    end
    fill_ghosts!(fs, GhostSchedule(fs, BOUNDED))
    worst, nchecked = 0.0, 0
    for b in 1:nblocks(fs)
        level(blockkey(fs, b)) == 1 || continue
        blk = blockview(fs, b, 1)
        for idx in CartesianIndices(blk)
            x = coordinates(fs, b, Tuple(idx))
            refined(x) && continue                  # a copy or a restriction
            I, σ = coarse_cell(x, h, n)
            worst = max(worst, abs(blk[idx] - bounded_child(U, I, σ)))
            nchecked += 1
        end
    end
    @test nchecked > 0
    @test worst < 1e-14
end

@testset "The bounded prolongation reproduces linear data: D=$D" for D in (1, 2, 3)
    # A bound that clipped a linear profile would cost the scheme its
    # order everywhere a smooth gradient crosses a coarse-fine face.
    forest = nested_forest(Val(D); N=D == 3 ? 4 : 8)
    ops = BOUNDED
    @test conservative_exchange_error(forest, ops, scalarpoly(D, 1); G=1) < 1e-12
    # And through a regrid, the prolongation onto fresh children.
    f = scalarpoly(D, 1)
    fs = FieldSet(forest, 1; G=1)
    fill_cell_averages!(fs, f)
    hook = boundary_cell_averages(f)
    flags = flag_blocks((b, k) -> level(k) < maxlevel(forest) ? Refine : Keep, forest)
    @test regrid!(forest, fs => GhostSchedule(fs, ops); flags=flags, boundary=hook)
    fill_ghosts!(fs, GhostSchedule(fs, ops); boundary=hook)
    @test max_average_deviation(fs, f) < 1e-12
end

@testset "Bounded regrids create no new extrema and conserve: D=$D" for D in (1, 2, 3)
    # The point of the bound: data stay within their initial range through
    # the ghost fill and any sequence of regrids, while the total is kept
    # to roundoff. A step, which the unbounded operator overshoots by an
    # eighth, shows that the bound is not vacuous, and a surface, a drop
    # over thirteen orders of magnitude, that positive data stay positive.
    initial = Dict(:random => (x, rng) -> rand(rng),
                   :step => (x, rng) -> x[1] < 1.3 ? 0.0 : 1.0,
                   :surface => (x, rng) -> x[1] < 1.3 ? 1.0 : 1.0e-13)
    for data in (:random, :step, :surface)
        forest = nested_forest(Val(D); N=D == 3 ? 4 : 8, periodic=ntuple(_ -> true, D))
        fs = FieldSet(forest, 1; G=1)
        rng = MersenneTwister(60 + D)
        fill_by_coordinates!((x, v) -> initial[data](x, rng), fs)
        lower = data === :surface ? 1.0e-13 : 0.0
        before = total_mass(fs)
        ops = BOUNDED
        for pass in 1:3
            flags = flag_blocks(forest) do b, k
                r = rand(rng)
                r < 0.4 && level(k) < 3 ? Refine : r < 0.7 ? Coarsen : Keep
            end
            regrid!(forest, fs => GhostSchedule(fs, ops); flags=flags)
            fill_ghosts!(fs, GhostSchedule(fs, ops))
            lo, hi = extrema(fs.work)
            @test lower - eps() <= lo && hi <= 1 + eps()
            data === :surface && @test lo > 0
        end
        @test total_mass(fs) ≈ before rtol = 1e-13
    end
    # The unbounded operator does overshoot a step it refines across.
    forest = nested_forest(Val(D); N=D == 3 ? 4 : 8, periodic=ntuple(_ -> true, D))
    fs = FieldSet(forest, 1; G=1)
    fill_by_coordinates!((x, v) -> x[1] < 1.3 ? 0.0 : 1.0, fs)
    unbounded = Operators(prolongation=3, restriction=2, family=Conservative)
    regrid!(forest, fs => GhostSchedule(fs, unbounded);
            flags=fill(Refine, nleaves(forest)))
    lo, hi = extrema(fs.work)
    @test lo < -0.1 || hi > 1.1
end

@testset "A bounded prolongation is mirrored, turned and clamped as the data are: D=$D" for
        D in (1, 2, 3)
    # The child's side is read off the stencil column, which a mirrored,
    # rotated or clamped row takes from its image. If that were wrong the
    # half domain would not be the fold of the doubled one, the quadrant
    # not the fold of the full plane — for arbitrary data — and an
    # outflow corner would read a ghost before it is written.
    ops = BOUNDED
    for side in (:lo, :hi)
        result = reflecting_vs_doubled(Val(D); side=side, centering=cellcentered(D), N=8,
                                       ops=ops)
        @test result !== nothing
        result === nothing && continue
        worst, ncells = result
        @test ncells > 0 && worst < 1e-13
    end
    if D >= 2
        result = rotating_vs_quadrupled(Val(D); rotating=(1, 2), centering=cellcentered(D),
                                        ops=ops)
        @test result !== nothing
        if result !== nothing
            worst, npoints = result
            @test npoints > 0 && worst < 1e-13
        end
    end
    kinds = ntuple(d -> isodd(d) ? :outflow_lo : :reflect_hi, D)
    nnan, _ = undefined_ghosts(kinds, cellcentered(D); p=3, family=Conservative,
                               bounded=true)
    @test nnan == 0
end

@testset "The bounded prolongation runs in Float32 too" begin
    # The kernel's arithmetic is generic in the element type: no Float64
    # literal may leak into a Float32 field set's bounded prolongation.
    D = 2
    N = 8
    forest = Forest{Float32}((2, 2); N=N, periodic=(true, true),
                             extents=((0, 2), (0, 2)))
    n, h = 2N, 2.0f0 / 2N
    U = rand(MersenneTwister(70), Float32, n, n)
    fs = FieldSet{Float32}(forest, 1; G=1)
    fill_by_coordinates!((x, v) -> U[coarse_cell(x, h, n)[1]], fs)
    @test regrid!(forest, fs => GhostSchedule(fs, BOUNDED);
                  flags=fill(Refine, nleaves(forest)))
    @test eltype(fs.work) === Float32
    worst = 0.0
    for b in 1:nblocks(fs), idx in CartesianIndices((2:(N + 1), 2:(N + 1)))
        I, σ = coarse_cell(coordinates(fs, b, Tuple(idx)), h, n)
        worst = max(worst, abs(blockview(fs, b, 1)[idx] - bounded_child(U, I, σ)))
    end
    @test worst < 1e-6
end

# --- Bounded point values (AMR-10) ------------------------------------------
#
# `Operators(…; family = PointValue, bounded = true)` clamps every
# interpolated value into the range of the source points that enclose it,
# its bracket. The oracle below knows only that definition and the
# geometry of a periodic coarse grid: where a fine point sits in it, the
# centered Lagrange window of width `p` around it, and the bracket.

const BOUNDED_PV = Operators(prolongation=4, restriction=4, bounded=true)
const PLAIN_PV = Operators(prolongation=4, restriction=4)

# The bounded value at position `x` of a periodic coarse grid `U` with
# spacing `h`, centering `cen` (`:cell`, values at `(I − 1/2)h`, or
# `:vertex`, at `(I − 1)h`), from the width-`p` interpolant.
function bounded_point(U::AbstractArray{<:Real,D}, x, h, cen, p; raw=false) where {D}
    n = size(U)
    # Along each dimension the window's first node, its weights, and the
    # bracket: two nodes, or the one the point coincides with.
    axis = ntuple(D) do d
        ξ = Rational{Int}(round(Int, 4 * x[d] / h), 4) + (cen === :cell ? 1 // 2 : 1)
        if isinteger(ξ)
            return Int(ξ), [1.0], [Int(ξ)]
        end
        f = floor(Int, ξ)
        lo = f - p ÷ 2 + 1
        w = TreeAMR.lagrange_weights([Rational(j) for j in lo:(lo + p - 1)], ξ)
        return lo, Float64.(w), [f, f + 1]
    end
    wrap(J) = CartesianIndex(ntuple(d -> mod1(J[d], n[d]), D))
    acc = 0.0
    for m in CartesianIndices(ntuple(d -> length(axis[d][2]), D))
        wt = prod(axis[d][2][m[d]] for d in 1:D)
        acc += wt * U[wrap(ntuple(d -> axis[d][1] + m[d] - 1, D))]
    end
    raw && return acc
    corners = [U[wrap(Tuple(c))] for c in Iterators.product(ntuple(d -> axis[d][3], D)...)]
    return clamp(acc, minimum(corners), maximum(corners))
end

# Monotone in every coordinate, and a cubic, so the order-4 interpolant
# reproduces it and its value lies strictly inside every bracket.
monotone_cubic(D) = (x, v) -> v + sum(d -> 0.4d * x[d] + 0.05 * x[d]^3, 1:D)

@testset "Bounded point values: width 2 is bounded already and changes nothing: D=$D" for
        D in (1, 2, 3)
    # Linear interpolation is a convex combination of its bracket, so
    # `bounded = true` must take the plain kernel at width 2 — every group
    # unbounded — and fill and regrid bit for bit as without it.
    plain = Operators(prolongation=2, restriction=2)
    ops = Operators(prolongation=2, restriction=2, bounded=true)
    for C in (cellcentered(D), vertexcentered(D))
        results = map((plain, ops)) do o
            forest = nested_forest(Val(D); N=8, periodic=ntuple(_ -> true, D))
            fs = FieldSet(forest, 1; G=1, centering=C)
            fill_by_coordinates!((x, v) -> mod(sum(d -> 97.13 * (d + 0.37) * x[d], 1:D), 1),
                                 fs)
            sched = GhostSchedule(fs, o)
            @test all(g -> g.bounded == TreeAMR.UNBOUNDED,
                      Iterators.flatten((sched.phase1, Iterators.flatten(sched.phase2))))
            fill_ghosts!(fs, sched)
            filled = copy(fs.work)
            flags = flag_blocks((b, k) -> isodd(b) ? Refine : Keep, forest)
            regrid!(forest, fs => GhostSchedule(fs, o); flags=flags)
            return filled, copy(fs.work)
        end
        @test results[1][1] == results[2][1]
        @test results[1][2] == results[2][2]
    end
end

@testset "A refined point is the bounded interpolant of the coarse grid: D=$D, $cen" for
        D in (1, 2, 3), cen in (:cell, :vertex)
    # The definition. A periodic brick refined everywhere at once: every
    # new point is the oracle's clamped interpolant of the coarse grid,
    # whatever the data, so a wrong bracket, a missing clamp or a clamp
    # into the wrong window shows. Random data clamp often.
    N = 8
    forest = Forest(ntuple(_ -> 2, D); N=N, periodic=ntuple(_ -> true, D),
                    extents=ntuple(_ -> (0.0, 2.0), D))
    n = 2N
    h = 2 / n
    U = rand(MersenneTwister(80 + D), n * ones(Int, D)...)
    C = ntuple(_ -> cen, D)
    fs = FieldSet(forest, 1; G=2, centering=C)
    at(x) = CartesianIndex(ntuple(d -> mod1(round(Int, x[d] / h + (cen === :cell ? 0.5 : 1)),
                                            n), D))
    fill_by_coordinates!((x, v) -> U[at(x)], fs)
    @test regrid!(forest, fs => GhostSchedule(fs, BOUNDED_PV);
                  flags=fill(Refine, nleaves(forest)))
    worst, nclamped = 0.0, 0
    for b in 1:nblocks(fs)
        blk = blockview(fs, b, 1)
        for idx in CartesianIndices(ntuple(_ -> 3:(N + 2), D))
            x = coordinates(fs, b, Tuple(idx))
            want = bounded_point(U, x, h, cen, 4)
            worst = max(worst, abs(blk[idx] - want))
            want == bounded_point(U, x, h, cen, 4; raw=true) || (nclamped += 1)
        end
    end
    @test worst < 1e-14
    @test nclamped > 0
end

@testset "Bounded point values reproduce a monotone cubic bit for bit: D=$D" for
        D in (1, 2, 3)
    # Where no clamp acts the bounded value is the order-p interpolant
    # itself, so the bound costs no order on smooth monotone data: the
    # cubic is reproduced, and every value is the unbounded fill's, bit
    # for bit, on every centering.
    forest = nested_forest(Val(D); N=8)
    f = monotone_cubic(D)
    for C in allcenterings(Val(D))
        G = ghosts_for(C, 4)
        works = map((PLAIN_PV, BOUNDED_PV)) do ops
            fs = FieldSet(forest, 2; G=G, centering=C)
            fill_ghosts!(fill_by_coordinates!(f, fs), GhostSchedule(fs, ops);
                         boundary=boundary_by_coordinates(f))
            @test max_deviation(fs, f) < 1e-11
            fs.work
        end
        @test works[1] == works[2]
    end
end

@testset "Bounded point values create no new extrema: D=$D, $cen" for
        D in (1, 2, 3), cen in (:cell, :vertex)
    # The point of the bound: through the ghost fill — prolongations,
    # interpolating restrictions — and random regrids, data stay within
    # their initial range, and a surface dropping over thirteen orders of
    # magnitude stays positive. The unbounded control overshoots.
    initial = Dict(:random => (x, rng) -> rand(rng),
                   :step => (x, rng) -> x[1] < 1.3 ? 0.0 : 1.0,
                   :surface => (x, rng) -> x[1] < 1.3 ? 1.0 : 1.0e-13)
    C = ntuple(_ -> cen, D)
    G = ghosts_for(C, 4)
    for data in (:random, :step, :surface)
        forest = nested_forest(Val(D); N=8, periodic=ntuple(_ -> true, D))
        fs = FieldSet(forest, 1; G=G, centering=C)
        rng = MersenneTwister(90 + D)
        fill_by_coordinates!((x, v) -> initial[data](x, rng), fs)
        fill_ghosts!(fs, GhostSchedule(fs, BOUNDED_PV))
        lower = data === :surface ? 1.0e-13 : 0.0
        lo, hi = extrema(fs.work)
        @test lower <= lo && hi <= 1
        for pass in 1:3
            flags = flag_blocks(forest) do b, k
                r = rand(rng)
                r < 0.4 && level(k) < 3 ? Refine : r < 0.7 ? Coarsen : Keep
            end
            regrid!(forest, fs => GhostSchedule(fs, BOUNDED_PV); flags=flags)
            fill_ghosts!(fs, GhostSchedule(fs, BOUNDED_PV))
            lo, hi = extrema(fs.work)
            @test lower <= lo && hi <= 1
        end
    end
    forest = nested_forest(Val(D); N=8, periodic=ntuple(_ -> true, D))
    fs = FieldSet(forest, 1; G=G, centering=C)
    fill_by_coordinates!((x, v) -> x[1] < 1.3 ? 1.0 : 1.0e-13, fs)
    regrid!(forest, fs => GhostSchedule(fs, PLAIN_PV); flags=fill(Refine, nleaves(forest)))
    @test minimum(fs.work) < 0
end

@testset "A bounded derived wall point stays in its bracket" begin
    # The derived upper wall point of a reflecting vertex-like dimension is
    # an interpolation at the prolongation order, `(4u₁ − u₂)/3` at order 4,
    # whatever kind of transfer fills it. Its bracket is the point beside
    # the wall and that point's mirror image, so a bounded wall of an even
    # variable is that point: a surface at the wall stays positive where
    # the unbounded derivation goes negative.
    N = 8
    h = 2 / 2N
    forest = Forest((2,); N=N, reflecting=((false, true),), extents=((0.0, 2.0),))
    f(x, v) = x[1] > 2 - 1.5h ? 1.0e-13 : 1.0
    walls = map((PLAIN_PV, BOUNDED_PV)) do ops
        fs = FieldSet(forest, 1; G=1, centering=vertexcentered(1), parity=[(EvenParity,)])
        fill_by_coordinates!(f, fs)
        fill_ghosts!(fs, GhostSchedule(fs, ops); boundary=boundary_by_coordinates(f))
        blockview(fs, nblocks(fs), 1)[1 + N + 1]
    end
    @test walls[1] < 0
    @test walls[2] == 1.0e-13
end

@testset "Bounded point values are mirrored, turned and clamped as the data are: D=$D" for
        D in (1, 2, 3)
    # The bracket is read off the multilinear rows, which a mirrored,
    # rotated or clamped row takes from its image with the rest of its
    # column. If that went wrong the half domain would not be the fold of
    # the doubled one, the quadrant not the fold of the full plane, and an
    # outflow or reflecting corner would read a ghost before it is written.
    for (C, sides) in ((cellcentered(D), (:lo, :hi)), (vertexcentered(D), (:lo,)))
        for side in sides
            result = reflecting_vs_doubled(Val(D); side=side, centering=C, N=8,
                                           ops=BOUNDED_PV)
            @test result !== nothing
            result === nothing && continue
            worst, ncells = result
            @test ncells > 0 && worst < 1e-13
        end
        if D >= 2
            result = rotating_vs_quadrupled(Val(D); rotating=(1, 2), centering=C,
                                            ops=BOUNDED_PV)
            @test result !== nothing
            if result !== nothing
                worst, npoints = result
                @test npoints > 0 && worst < 1e-13
            end
        end
        kinds = ntuple(d -> isodd(d) ? :outflow_lo : :reflect_hi, D)
        nnan, _ = undefined_ghosts(kinds, C; p=4, bounded=true)
        @test nnan == 0
    end
end

@testset "Bounded point values run in Float32 too" begin
    # No Float64 literal may leak into the clamp or its extremes.
    D = 2
    forest = nested_forest(Val(D); T=Float32, N=8, periodic=(true, true))
    fs = FieldSet{Float32}(forest, 1; G=2)
    fill_by_coordinates!((x, v) -> x[1] < 1.3f0 ? 1.0f0 : 1.0f-13, fs)
    fill_ghosts!(fs, GhostSchedule(fs, BOUNDED_PV))
    @test eltype(fs.work) === Float32
    lo, hi = extrema(fs.work)
    @test 1.0f-13 <= lo && hi <= 1
end
