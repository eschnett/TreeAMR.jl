# The relative boundary hook (TreeGRRMHD's AMR-7).
#
# `CellBoundary(Relative(f))` sets a ghost beyond an outer face to
# `f(x_g) + (u(x_r) − f(x_r))`, `x_r` the nearest evolved point: clamped
# along the components that leave the domain, the ghost's own index along
# every other one, where it lies in the block's own ghosts across a block
# face. The oracle computes `x_r` from the geometry alone — which side of
# the domain a block touches — never from the schedule's `clamps`.

# The stored index of the nearest evolved point of ghost `idx` of block
# `b` in a region of direction `δ`, by the oracle's geometry: a component
# leaves the domain when the block's box touches the domain's face on that
# side and the dimension is not periodic.
function oracle_relative_point(fs::FieldSet{T,D}, b, δ, idx) where {T,D}
    forest = fs.forest
    box = block_extent(forest, blockkey(fs, b))
    G, N = fs.G, forest.N
    return ntuple(D) do d
        forest.periodic[d] && return idx[d]
        δ[d] > 0 && box[d][2] == forest.extents[d][2] && return G[d] + N
        if δ[d] < 0 && box[d][1] == forest.extents[d][1]
            symmetric = fs.walls === :symmetric && fs.centering[d] === :vertex
            return G[d] + 1 + symmetric
        end
        return idx[d]
    end
end

# The worst departure of a filled set from the relative condition, over
# every ghost of every hook region, and how many ghosts were checked, with
# how many of those read a point across a block face (a corner row).
function relative_defect(fs::FieldSet{T,D}, schedule, f) where {T,D}
    worst, n, ncorner = 0.0, 0, 0
    for region in schedule.boundaries, idx in region.region
        b = Int(region.block)
        i = Tuple(idx)
        r = oracle_relative_point(fs, b, region.direction, i)
        want = f(coordinates(fs, b, i)) + (fs.work[r..., 1, b] - f(coordinates(fs, b, r)))
        worst = max(worst, abs(fs.work[i..., 1, b] - want))
        n += 1
        any(d -> region.direction[d] != 0 && r[d] == i[d], 1:D) && (ncorner += 1)
    end
    return worst, n, ncorner
end

relative_case(D) = x -> sum(d -> sin(1.3 * x[d] + d), 1:D)

@testset "Every hook ghost is the relative condition, corner rows included: D=$D" for
        D in (1, 2, 3)
    # On a three-level mesh whose coarse-fine faces meet the outer faces,
    # on data with no structure: every ghost the hook writes is the
    # condition, bit for bit, with its nearest evolved point taken across
    # a block face where its region crosses one.
    forest = nested_forest(Val(D); N=8)
    f = relative_case(D)
    for C in (cellcentered(D), vertexcentered(D))
        fs = FieldSet(forest, 1; G=ghosts_for(C, 4), centering=C)
        fill_arbitrary!(fs)
        sched = GhostSchedule(fs, Operators(prolongation=4, restriction=4))
        fill_ghosts!(fs, sched; boundary=CellBoundary(Relative(f)))
        worst, n, ncorner = relative_defect(fs, sched, f)
        @test n > 0
        @test worst == 0
        D == 1 || @test ncorner > 0
    end
end

@testset "A single block's relative hook is the region form, bit for bit: D=$D" for
        D in (1, 2, 3)
    # Where every region leaves the domain along every outward component,
    # clamping the whole region onto the block — what an application's
    # region-form hook does — is the condition, so the two agree.
    forest = Forest(ntuple(_ -> 1, D); N=8)
    f = relative_case(D)
    ops = Operators(prolongation=4, restriction=4)
    for C in (cellcentered(D), vertexcentered(D))
        works = map((:relative, :region)) do form
            fs = FieldSet(forest, 1; G=ghosts_for(C, 4), centering=C)
            fill_arbitrary!(fs)
            G, N = fs.G, forest.N
            region_hook = function (fs, b, key, δ, region)
                for idx in region
                    i = Tuple(idx)
                    r = ntuple(d -> δ[d] > 0 ? G[d] + N : δ[d] < 0 ? G[d] + 1 : i[d], D)
                    fs.work[i..., 1, b] = f(coordinates(fs, b, i)) +
                                          (fs.work[r..., 1, b] - f(coordinates(fs, b, r)))
                end
            end
            hook = form === :relative ? CellBoundary(Relative(f)) : region_hook
            fill_ghosts!(fs, GhostSchedule(fs, ops); boundary=hook)
            return fs.work
        end
        @test works[1] == works[2]
    end
end

@testset "A linear case is reproduced exactly by the relative hook: D=$D" for
        D in (1, 2, 3)
    # With `u = f` linear the condition returns `f(x_g)`: exactly at a
    # hook ghost, since `u(x_r) − f(x_r)` is zero, and to rounding where
    # the interpolated ghosts it reads come in. Every centering, refined.
    forest = nested_forest(Val(D); N=8)
    f = x -> 0.3 + sum(d -> (0.2 + 0.1d) * x[d], 1:D)
    for C in allcenterings(Val(D))
        fs = FieldSet(forest, 1; G=ghosts_for(C, 4), centering=C)
        fill_by_coordinates!((x, v) -> f(x), fs)
        fill_ghosts!(fs, GhostSchedule(fs, Operators(prolongation=4, restriction=4));
                     boundary=CellBoundary(Relative(f)))
        @test max_deviation(fs, (x, v) -> f(x)) < 1e-12
    end
    # And the all-variables form, two variables with different cases.
    fs = FieldSet(forest, 2; G=2)
    g(x) = (f(x), 2 * f(x) - 1)
    fill_by_coordinates!((x, v) -> g(x)[v], fs)
    fill_ghosts!(fs, GhostSchedule(fs, Operators(prolongation=4, restriction=4));
                 boundary=CellBoundary(Relative(AllVariables(g))))
    @test max_deviation(fs, (x, v) -> g(x)[v]) < 1e-12
end

@testset "One fill is the relative condition's fixed point: D=$D" for D in (2, 3)
    # The condition reads ghosts that transfers fill, prolongated ones
    # included, so it must run after them. Every value NaN but the owned
    # data: one fill leaves nothing undefined, and a second fill changes
    # nothing, bit for bit, on a mesh whose coarse-fine faces meet the
    # outer faces.
    forest = nested_forest(Val(D); N=8)
    f = relative_case(D)
    for C in (cellcentered(D), vertexcentered(D), facecentered(D, 1))
        fs = FieldSet(forest, 1; G=ghosts_for(C, 4), centering=C)
        fill!(fs.work, NaN)
        fill_arbitrary!(fs)
        sched = GhostSchedule(fs, Operators(prolongation=4, restriction=4))
        hook = CellBoundary(Relative(f))
        fill_ghosts!(fs, sched; boundary=hook)
        @test count(isnan, fs.work) == 0
        once = copy(fs.work)
        fill_ghosts!(fs, sched; boundary=hook)
        @test fs.work == once
    end
end

@testset "A regrid with the relative hook transfers what a complete fill gives: D=$D" for
        D in (2, 3)
    # `regrid!` fills before its transfer. With the hook that fill must be
    # complete: the same regrid after a complete fill, with no hook, which
    # leaves the hook's ghosts as they are, must give the same bits.
    f = relative_case(D)
    ops = Operators(prolongation=4, restriction=4)
    works = map((:hook, :prefilled)) do how
        forest = nested_forest(Val(D); N=8)
        fs = FieldSet(forest, 1; G=2, centering=vertexcentered(D))
        fill_arbitrary!(fs)
        hook = CellBoundary(Relative(f))
        flags = flag_blocks((b, k) -> level(k) == 1 ? Refine : Keep, forest)
        if how === :hook
            regrid!(forest, fs => GhostSchedule(fs, ops); flags=flags, boundary=hook)
        else
            fill_ghosts!(fs, GhostSchedule(fs, ops); boundary=hook)
            regrid!(forest, fs => GhostSchedule(fs, ops); flags=flags)
        end
        return fs.work
    end
    @test works[1] == works[2]
end

@testset "A closed box with relative walls is mirror-symmetric" begin
    # TreeGRRMHD's case: the potentials' Dirichlet data in the field, with
    # symmetric walls, on a uniform box. Point-symmetric data and case fill
    # point-symmetrically, bit for bit.
    D, L = 2, 2
    f = x -> cos(1.7 * (x[1] - L / 2)) * cos(1.1 * (x[2] - L / 2))
    forest = faces_forest((:outer, :outer); N=8, refined=false)
    fs = FieldSet(forest, 1; G=1, centering=vertexcentered(2), walls=:symmetric)
    fill_by_coordinates!((x, v) -> f(x) + 0.2 * (x[1] - 1)^2 * (x[2] - 1)^2, fs)
    fill_ghosts!(fs, GhostSchedule(fs, Operators(prolongation=4, restriction=4));
                 boundary=CellBoundary(Relative(f)))
    result = point_mirror_error(fs, L)
    @test result !== nothing && result[1] == 0
end

@testset "The relative hook's forms are checked, and a fresh one compiles nothing" begin
    # The per-variable form names no variable, so it is refused on a set
    # with several; the all-variables form must return one value per
    # variable. A hook rebuilt for every fill with a new time captured is
    # the same type, so its kernel is not compiled again.
    forest = Forest((2, 2); N=8)
    ops = Operators(prolongation=2, restriction=2)
    two = FieldSet(forest, 2; G=1)
    @test_throws "wrap the function in AllVariables" fill_ghosts!(
        two, GhostSchedule(two, ops); boundary=CellBoundary(Relative(x -> 0.0)))
    @test_throws "one value per variable" fill_ghosts!(
        two, GhostSchedule(two, ops); boundary=CellBoundary(Relative(AllVariables(x -> (0.0,)))))
    one = FieldSet(forest, 1; G=1)
    sched = GhostSchedule(one, ops)
    hookat(t) = CellBoundary(Relative(x -> sin(x[1] - t)))
    @test typeof(hookat(0.1)) === typeof(hookat(0.2))
    fill_ghosts!(one, sched; boundary=hookat(0.1))
    if isdefined(Base, :cumulative_compile_timing)
        Base.cumulative_compile_timing(true)
        before = Base.cumulative_compile_time_ns()[1]
        fill_ghosts!(one, sched; boundary=hookat(0.2))
        after = Base.cumulative_compile_time_ns()[1]
        Base.cumulative_compile_timing(false)
        @test after == before
    end
end
