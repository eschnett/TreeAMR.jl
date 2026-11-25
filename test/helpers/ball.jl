# The seven-patch and the thirteen-patch ball of TreeMultiPatch's
# `balls.jl` (M20), ungraded, with their glue, maps and inverses, written
# out again here because TreeMultiPatch depends on TreeAMR and the test
# environment cannot depend on it. A cube `[-Lc, Lc]³` of `n³` roots
# (patch 1); six shells over its faces `(k, s)` in the order of
# `BALL_SHELLS` (patches 2–7), of `m` roots along their radial dimension
# `k`, which increases outward, and `n` along the other two, which run
# along `x_j`; for the thirteen-patch ball six outer shells over the same
# faces (patches 8–13), of `m′` radial roots, whose radial coordinate is
# the radius. The outer sphere is every outermost shell's high face along
# `k`, an outer face; every other face is glued.

const BALL_SHELLS = ((1, -1), (1, 1), (2, -1), (2, 1), (3, -1), (3, 1))

ball_shell_index(k, s) = 1 + 2(k - 1) + (s == 1 ? 2 : 1)
ball_side(s) = s == 1 ? 2 : 1

struct TestBall
    Lc::Float64
    Rm::Float64          # the sphere the inner shells reach, R1 without outer shells
    R1::Float64
    n::Int
    m::Int
    m′::Int              # 0: the seven-patch ball
end
seven_ball(n, m) = TestBall(1.0, 2.5, 2.5, n, m, 0)
thirteen_ball(n, m, m′) = TestBall(1.0, 2.5, 4.0, n, m, m′)
ball_patches(b::TestBall) = b.m′ == 0 ? 7 : 13

# The cube face `(k, s)` patch `p` lies over, `k = 0` for the cube, and
# whether it is an outer shell.
function ball_shell(b::TestBall, p)
    p == 1 && return 0, 1, false
    p <= 7 && return BALL_SHELLS[p - 1]..., false
    return BALL_SHELLS[p - 7]..., true
end

function ball_roots(b::TestBall, p)
    k, _, outer = ball_shell(b, p)
    k == 0 && return (b.n, b.n, b.n)
    return ntuple(j -> j == k ? (outer ? b.m′ : b.m) : b.n, 3)
end

# The glue, both directions of each gluing: the cube's face onto the
# inner shell's inner face, high to low by the identity or low to low by
# reversing `k`; an inner shell's outer face onto the outer shell's inner
# face; and the quarter turns between the shells of a layer, the radial
# `k` onto the radial `l`, the normal `l` onto `k`, reversed where both
# faces are high or both low.
function ball_glue(b::TestBall)
    glue = NamedTuple{(:from, :to, :map),Tuple{NTuple{3,Int},NTuple{3,Int},NTuple{3,Int}}}[]
    layers = b.m′ == 0 ? (ball_shell_index,) :
             (ball_shell_index, (k, s) -> 6 + ball_shell_index(k, s))
    for (k, s) in BALL_SHELLS
        p = ball_shell_index(k, s)
        σ = ntuple(e -> e == k ? s * k : e, 3)
        push!(glue, (from=(1, k, ball_side(s)), to=(p, k, 1), map=σ))
        push!(glue, (from=(p, k, 1), to=(1, k, ball_side(s)), map=σ))
        if b.m′ > 0
            push!(glue, (from=(p, k, 2), to=(p + 6, k, 1), map=(1, 2, 3)))
            push!(glue, (from=(p + 6, k, 1), to=(p, k, 2), map=(1, 2, 3)))
        end
        for index in layers, (l, t) in BALL_SHELLS
            l == k && continue
            σ = ntuple(e -> e == k ? l : e == l ? (s == t ? -k : k) : e, 3)
            push!(glue, (from=(index(k, s), l, ball_side(t)),
                         to=(index(l, t), k, ball_side(s)), map=σ))
        end
    end
    return glue
end

# The vertex-centered set the ball tests fill, its operators, and the
# hook for the outer sphere's ghosts and face plane, which the stencils of
# the outermost shells read, from the same data as the owned points.
const BALL_OPS = Operators(prolongation=4, restriction=4, glue=6)
ball_set(forest; nvars=1) =
    FieldSet(forest, nvars; G=3, centering=vertexcentered(3), glue=:interpolate)
ball_hook(f) = CellBoundary(WithPatch((x, p, v, δ) -> v * f(x, p)))

ball_forest(b::TestBall; N=8, kw...) =
    Forest([(roots=ball_roots(b, p),) for p in 1:ball_patches(b)]; N=N,
           glue=ball_glue(b), kw...)

# Shell `(k, s)`'s face point `f = Lc c` over logical `ξ`, continued past
# the face's edges.
ball_face_point(b::TestBall, ξ, k, s) =
    ntuple(j -> j == k ? s * b.Lc : b.Lc * (2ξ[j] / b.n - 1), 3)

function TreeAMR.chart_map(b::TestBall, ξ, p)
    p == 1 && return ntuple(j -> (2b.Lc / b.n) * ξ[j] - b.Lc, 3)
    k, s, outer = ball_shell(b, p)
    f = ball_face_point(b, ξ, k, s)
    nf = sqrt(f[1]^2 + f[2]^2 + f[3]^2)
    if outer
        r = b.Rm + (b.R1 - b.Rm) * ξ[k] / b.m′
        return ntuple(i -> r * f[i] / nf, 3)
    end
    ρ = ξ[k] / b.m
    return ntuple(i -> (1 - ρ) * f[i] + ρ * b.Rm * f[i] / nf, 3)
end

# The cube if `x` is in it; else the shell over the face `x` is parallel
# to, `k` the first dimension of largest `|x_k|`, with the tangential
# coordinates from `c = x/|x_k|` and the radial one from `|x_k|` (inner)
# or `|x|` (outer); nothing past the outer sphere.
function TreeAMR.chart_locate(b::TestBall, x)
    n, Lc = b.n, b.Lc
    ax = abs.(x)
    k = ax[1] >= ax[2] ? (ax[1] >= ax[3] ? 1 : 3) : (ax[2] >= ax[3] ? 2 : 3)
    ax[k] <= Lc && return ntuple(j -> clamp((x[j] + Lc) * n / (2Lc), 0.0, n * 1.0), 3), 1
    s = x[k] < 0 ? -1 : 1
    c = ntuple(j -> x[j] / ax[k], 3)
    nc = sqrt(c[1]^2 + c[2]^2 + c[3]^2)
    r2 = x[1]^2 + x[2]^2 + x[3]^2
    ξ = ntuple(j -> clamp(n * (c[j] + 1) / 2, 0.0, n * 1.0), 3)
    radial(ξk) = ntuple(j -> j == k ? ξk : ξ[j], 3)
    if b.m′ == 0 || r2 <= b.Rm^2
        r2 <= (b.Rm * (1 + 8eps()))^2 || return nothing
        # |x_k| = Lc (1 + ρ (Rm/(Lc |c|) - 1))
        ρ = clamp((ax[k] / Lc - 1) / (b.Rm / (Lc * nc) - 1), 0.0, 1.0)
        return radial(b.m * ρ), ball_shell_index(k, s)
    end
    r2 <= (b.R1 * (1 + 8eps()))^2 || return nothing
    ξk = clamp(b.m′ * (sqrt(r2) - b.Rm) / (b.R1 - b.Rm), 0.0, b.m′ * 1.0)
    return radial(ξk), 6 + ball_shell_index(k, s)
end

# The thirteen-patch ball's shared coordinates: the radius between two
# outer shells, both angles between an inner and an outer shell over one
# face. The seven-patch ball shares none, and declares nothing.
struct AlignedBall
    ball::TestBall
end
TreeAMR.chart_map(a::AlignedBall, ξ, p) = TreeAMR.chart_map(a.ball, ξ, p)
TreeAMR.chart_locate(a::AlignedBall, x) = TreeAMR.chart_locate(a.ball, x)
function TreeAMR.chart_aligned(a::AlignedBall, p, q)
    p == q && return (true, true, true)
    kp, sp, op = ball_shell(a.ball, p)
    kq, sq, oq = ball_shell(a.ball, q)
    op && oq && return ntuple(e -> e == kq, 3)
    kp != 0 && (kp, sp) == (kq, sq) && op != oq && return ntuple(e -> e != kq, 3)
    return (false, false, false)
end

# An atlas declaring the radius shared between inner shells too, which
# their blended maps do not share.
struct OverAlignedBall
    ball::TestBall
end
TreeAMR.chart_map(a::OverAlignedBall, ξ, p) = TreeAMR.chart_map(a.ball, ξ, p)
TreeAMR.chart_locate(a::OverAlignedBall, x) = TreeAMR.chart_locate(a.ball, x)
function TreeAMR.chart_aligned(a::OverAlignedBall, p, q)
    kp, _, op = ball_shell(a.ball, p)
    kq, _, oq = ball_shell(a.ball, q)
    !op && !oq && kp != 0 && kq != 0 && kp != kq && return ntuple(e -> e == kq, 3)
    return TreeAMR.chart_aligned(AlignedBall(a.ball), p, q)
end
