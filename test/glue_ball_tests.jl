# Ghosts interpolated across the glues of the seven-patch and the
# thirteen-patch ball (M20): cube edges where three shells meet, patch
# corners, quarter turns and reversed normals, an outer sphere, and on the
# thirteen-patch ball the shared coordinates `chart_aligned` declares.
# "Finite differences across a glue" in CODE.md; the balls are in
# `helpers/ball.jl`, the fill and the ghost list in `helpers/glue.jl`.

# A polynomial of degree five in each logical coordinate, different in
# each patch, which a six-point stencil reproduces exactly; its part that
# differs between the patches vanishes on every face of the patch's box,
# so that the copies of a glued point agree bit for bit and stage A's mean
# (M22) leaves them as they are.
function ball_poly(ball)
    return function (ξ, p)
        u = ξ ./ ball_roots(ball, p)
        return 0.5 + p * prod(ntuple(d -> u[d]^2 * (1 - u[d])^2 * (1 + u[d]), 3))
    end
end

# The largest difference between each interpolated ghost and `f` at the
# patch and position the atlas locates it at.
function ball_ghost_error(fs, atlas, f)
    return maximum(glue_ghosts(fs)) do (b, idx, ξ, p)
        ξ′, q = TreeAMR.chart_locate(atlas, TreeAMR.chart_map(atlas, ξ, p))
        return abs(fs.work[idx..., 1, b] - f(ξ′, q))
    end
end

@testset "On the balls every interpolated ghost of a patch's polynomial is exact" begin
    # Guards the location and the stencil where a cubed sphere is hardest:
    # across the cube's edges three shells meet, at its corners four
    # patches, the shells are glued by quarter turns, the cube to the
    # shells over its low faces by a reversed normal, and the outermost
    # shells' stencils reach the outer sphere's plane, which the hook
    # fills. Data that is a polynomial of degree glue - 1 in each patch's
    # logical coordinates is interpolated exactly wherever a ghost lands.
    for (ball, atlas) in ((seven_ball(2, 1), seven_ball(2, 1)),
                          (seven_ball(2, 2), seven_ball(2, 2)),
                          (thirteen_ball(2, 1, 1), AlignedBall(thirteen_ball(2, 1, 1))),
                          (thirteen_ball(2, 1, 1), thirteen_ball(2, 1, 1)))
        forest = ball_forest(ball)
        fs = ball_set(forest)
        sched = GhostSchedule(fs, BALL_OPS; atlas=atlas)
        f = ball_poly(ball)
        fill_from_owned!(fs, sched, f; boundary=ball_hook(f))
        ghosts = glue_ghosts(fs)
        @test length(ghosts) == length(sched.glueplan)
        # Every ghost across a glue, at faces, edges and patch corners.
        @test all(((b, idx, ξ, p),) -> isfinite(fs.work[idx..., 1, b]), ghosts)
        @test ball_ghost_error(fs, atlas, f) < 1e-11
    end
    # Too coarse a cube: its corner ghosts, `G` spacings out along the
    # diagonal, lie past the sphere, in no patch, where no stencil can
    # reach them and no outer face makes them the hook's.
    fs = ball_set(ball_forest(seven_ball(1, 1)))
    @test_throws "in no patch" GhostSchedule(fs, BALL_OPS; atlas=seven_ball(1, 1))
end

# The planes of points of a vertex-centered ball that no patch's own
# grid tells apart: the cube's closed `(M + 1)³`, and each radial layer
# above it, a closed cubed sphere of `6M² + 2` points, the outer sphere
# included, whose points are the outermost blocks' own (M22).
ball_distinct(b::TestBall, N) =
    (b.n * N + 1)^3 + (b.m * N + b.m′ * N) * (6 * (b.n * N)^2 + 2)

# The physical position, quantized, as a key: no two nodes of these balls
# are closer than 10⁻⁹, and rounding moves a node far less.
ball_key(x) = round.(Int, x .* 1e9)

@testset "The seven-patch ball's copies are one number, and the primaries each point once" begin
    # Guards closed blocks on a cubed sphere (M22): every block's closed box
    # is in the state, which round-trips through scatter! and gather!;
    # after a fill every copy of a physical point, on every patch and block
    # holding it, is one number, stage A's mean across the glues; and a
    # fold over the primaries counts each physical point exactly once,
    # which here is the ball's number of distinct nodes, three shells
    # holding a point of a cube edge and four patches a cube corner.
    ball = seven_ball(2, 1)
    forest = ball_forest(ball)
    N = forest.N
    fs = ball_set(forest)
    @test statelength(fs) == nblocks(fs) * (N + 1)^3
    u = rand(Xoshiro(20), statelength(fs))
    scatter!(fs, u)
    v = zeros(statelength(fs))
    gather!(v, fs)
    @test v == u
    # The fill makes every copy of a point one number, from random data.
    sched = GhostSchedule(fs, BALL_OPS; atlas=ball)
    fill_ghosts!(fs, sched; boundary=ball_hook((x, p) -> 0.0))
    copies = Dict{NTuple{3,Int},Vector{Float64}}()
    for b in 1:nblocks(fs), I in CartesianIndices(ntuple(_ -> (N + 1), 3))
        idx = Tuple(I) .+ fs.G
        x = TreeAMR.chart_map(ball, coordinates(fs, b, idx), patch(forest, blockkey(fs, b)))
        push!(get!(copies, ball_key(x), Float64[]), fs.work[idx..., 1, b])
    end
    @test length(copies) == ball_distinct(ball, N)
    @test all(c -> all(==(c[1]), c), values(copies))
    # Members of up to four patches occur, at the cube's corners.
    @test maximum(sched.means.targets.mcount) == 4
    # The fold of one over the primaries counts the distinct points, and a
    # function of the position, summed, is its sum over them.
    count1(acc, p) = acc + 1
    @test mesh_fold(count1, +, 0, fs, u; primaries=true) == ball_distinct(ball, N)
    @test mesh_mapreduce(x -> 1, +, 0, fs, u; primaries=true) == ball_distinct(ball, N)
    g(x) = 2 + sin(1.3x[1] + 0.7x[2]) + x[3]^3 - x[1] * x[3] + 0.1x[2]^2
    A = statearray(u, fs)
    distinct = Dict{NTuple{3,Int},Float64}()
    for b in 1:nblocks(fs)
        p = patch(forest, blockkey(fs, b))
        for I in CartesianIndices(ntuple(_ -> (N + 1), 3))
            x = TreeAMR.chart_map(ball, coordinates(fs, b, Tuple(I) .+ fs.G), p)
            A[Tuple(I)..., 1, b] = g(x)
            distinct[ball_key(x)] = g(x)
        end
    end
    @test mesh_fold((acc, p) -> acc + p.u[1], +, 0.0, fs, u; primaries=true) ≈
          sum(values(distinct)) rtol = 1e-13
    @test volume_weighted_norm(fs, fill(2.0, statelength(fs))) ≈ 2.0 rtol = 1e-14
end

# The ball's symmetries, as maps of physical space: the point inversion,
# a reflection, a swap of two axes and a cycle of all three, which
# generate the cube's 48.
const BALL_SYMMETRIES = (x -> .-x, x -> (-x[1], x[2], x[3]), x -> (x[2], x[1], x[3]),
                         x -> (x[3], x[1], x[2]))

@testset "The fill of data with the seven-patch ball's symmetries has them to rounding" begin
    # Guards the orientation of every glue and stencil: a quarter turn or a
    # reversal applied the wrong way, or a window chosen differently on
    # the two sides of a symmetry, breaks the symmetry of the ghosts of
    # symmetric data by far more than rounding, which is all the fill may
    # add under closed ownership.
    ball = seven_ball(2, 1)
    forest = ball_forest(ball)
    fs = ball_set(forest)
    sched = GhostSchedule(fs, BALL_OPS; atlas=ball)
    f(x) = cos(0.7 * (x[1]^2 + x[2]^2 + x[3]^2)) + 0.1 * (x[1]^4 + x[2]^4 + x[3]^4) +
           x[1]^2 * x[2]^2 * x[3]^2
    fp(ξ, p) = f(TreeAMR.chart_map(ball, ξ, p))
    fill_from_owned!(fs, sched, fp; boundary=ball_hook(fp))
    ghosts = glue_ghosts(fs)
    value = Dict{NTuple{3,Int},Float64}()
    for (b, idx, ξ, p) in ghosts
        value[ball_key(TreeAMR.chart_map(ball, ξ, p))] = fs.work[idx..., 1, b]
    end
    for σ in BALL_SYMMETRIES
        missing_ = 0
        err = 0.0
        for (b, idx, ξ, p) in ghosts
            y = ball_key(σ(TreeAMR.chart_map(ball, ξ, p)))
            haskey(value, y) || (missing_ += 1; continue)
            err = max(err, abs(fs.work[idx..., 1, b] - value[y]))
        end
        @test missing_ == 0
        @test err < 1e-12
    end
    # The interpolation error itself is far above rounding, so that the
    # symmetry is not that of an exact fill.
    @test ball_ghost_error(fs, ball, fp) > 1e-6
end

@testset "On the thirteen-patch ball the shared coordinates take no stencil" begin
    # Guards `chart_aligned` on the layout it was made for: between two
    # outer shells a ghost has the radius of a node of the shell across,
    # so its stencil has width one along the radius; between an inner and
    # an outer shell over one face it has both angles, width one along
    # both; and everywhere else it is the full width. An atlas that
    # declares a coordinate its maps do not share is refused.
    ball = thirteen_ball(2, 1, 1)
    atlas = AlignedBall(ball)
    forest = ball_forest(ball)
    fs = ball_set(forest)
    sched = GhostSchedule(fs, BALL_OPS; atlas=atlas)
    plan = sched.glueplan
    k = BALL_OPS.glue
    kinds = Dict{Symbol,Int}()
    wrong = 0
    for j in eachindex(plan.target)
        p = patch(forest, forest.leaves[plan.target[j]])
        q = patch(forest, forest.leaves[plan.source[j]])
        kp, sp, op = ball_shell(ball, p)
        kq, sq, oq = ball_shell(ball, q)
        kind = op && oq ? :outer : kp == kq && sp == sq && kp != 0 && op != oq ?
               :layers : :other
        want = kind == :outer ? ntuple(e -> e == kq ? 1 : k, 3) :
               kind == :layers ? ntuple(e -> e == kq ? k : 1, 3) : (k, k, k)
        wrong += plan.width[j] != want
        kinds[kind] = get(kinds, kind, 0) + 1
    end
    @test wrong == 0
    # Each kind of glue occurs.
    @test all(kind -> get(kinds, kind, 0) > 0, (:outer, :layers, :other))
    @test_throws "not at a node" GhostSchedule(fs, BALL_OPS; atlas=OverAlignedBall(ball))
end
