# The definitions of `glue_tests.jl` (M20): forests of patches whose maps
# join with a kink, with their atlases, split off so that each test file
# runs on its own ("Parallel groups" in CODE.md).

# The two-patch torus of `bench/coupling/torus.jl`: patch 1 is affine,
# `x = -1 + ξ₁`, `y = ξ₂`; patch 2 is `x = aη + bη²`, `y = ζ + cη(1 - η)`,
# so that the map's Jacobian jumps at both glues. Each patch is `[0, 1]²`
# of `roots` roots, glued to the other at both faces of dimension 1 and
# to itself across dimension 2, a wrap.
struct KinkedTorus
    a::Float64
    b::Float64
    c::Float64
end
const KINKED = KinkedTorus(0.8, 0.4, 0.3)

function kinked_forest(n; N=8, kw...)
    glue = vcat(patch_glue_both((1, 1, 2), (2, 1, 1), (1, 2)),
                patch_glue_both((2, 1, 2), (1, 1, 1), (1, 2)),
                [(from=(p, 2, 2), to=(p, 2, 1), map=(1, 2)) for p in 1:2],
                [(from=(p, 2, 1), to=(p, 2, 2), map=(1, 2)) for p in 1:2])
    patches = [(roots=(n, n), extents=((0.0, 1.0), (0.0, 1.0))) for _ in 1:2]
    return Forest(patches; N=N, glue=glue, kw...)
end

kinked_period(t::KinkedTorus) = 1 + t.a + t.b

function TreeAMR.chart_map(t::KinkedTorus, ξ, p)
    p == 1 && return (-1 + ξ[1], ξ[2])
    η, ζ = ξ
    return (t.a * η + t.b * η^2, ζ + t.c * η * (1 - η))
end

function TreeAMR.chart_locate(t::KinkedTorus, x)
    L = kinked_period(t)
    X = mod(x[1] + 1, L) - 1                 # into [-1, a + b)
    X < 0 && return ((X + 1, mod(x[2], 1.0)), 1)
    η = (-t.a + sqrt(t.a^2 + 4t.b * X)) / (2t.b)
    return ((η, mod(x[2] - t.c * η * (1 - η), 1.0)), 2)
end

# A smooth periodic function on the torus, with its value at a physical
# point.
kinked_wave(t::KinkedTorus, x) = sin(2π * (2x[1] / kinked_period(t) + x[2]) + 0.3)

# The vertex-centered set, its operators and its schedule on the torus.
function kinked_set(forest; G=3, glue=:interpolate)
    return FieldSet(forest, 1; G=G, centering=vertexcentered(2), glue=glue)
end
const KINKED_OPS = Operators(prolongation=4, restriction=4, glue=6)

# Every stored point of every block of `fs` beyond a glued face of its
# patch and nowhere else, past the high face planes the block owns, with
# its logical position and patch: what the glue plan fills. (A set with a
# vertex-like dimension is refused over a glued forest until M22 step 8.)
function glue_ghosts(fs)
    forest = fs.forest
    conn = forest.conn
    G, c = fs.G, staggers(fs.centering)
    D = length(G)
    out = Tuple{Int,NTuple{D,Int},NTuple{D,Float64},Int}[]
    for b in 1:nblocks(fs)
        k = blockkey(fs, b)
        p = patch(forest, k)
        for I in CartesianIndices(size(fs.work)[1:D])
            idx = Tuple(I)
            δ = ntuple(d -> idx[d] <= G[d] ? -1 :
                            idx[d] > G[d] + fs.forest.N + c[d] ? 1 : 0, D)
            all(iszero, δ) && continue
            TreeAMR.interpolated_region(conn, k, δ) || continue
            ξ = coordinates(floattype(forest), fs, b, idx)
            push!(out, (b, idx, ξ, p))
        end
    end
    return out
end

# Own points of a set, from `f(ξ, p)` at their
# positions, times `v` for variable `v`, every other stored point NaN,
# then the fill with `boundary`.
function fill_from_owned!(fs, sched, f; boundary=nothing)
    forest = fs.forest
    G, N = fs.G, forest.N
    D = length(G)
    c = staggers(fs.centering)
    fs.work .= NaN
    for b in 1:nblocks(fs)
        k = blockkey(fs, b)
        p = patch(forest, k)
        for I in CartesianIndices(ntuple(d -> (G[d] + 1):(G[d] + N + c[d]), D))
            ξ = coordinates(floattype(forest), fs, b, Tuple(I))
            for v in 1:fs.nvars
                fs.work[I, v, b] = v * f(ξ, p)
            end
        end
    end
    fill_ghosts!(fs, sched; boundary=boundary)
    return fs
end

# Two patches side by side along dimension 1, glued at one face, with
# affine maps of different spacings: patch 1 is `x = -1 + ξ₁`, patch 2
# `x = a η`, both `y = ξ₂`. Every other face is outer. With `a` small a
# ghost of patch 1 lands deep in patch 2, in its second block layer.
struct StretchedPair
    a::Float64
end
TreeAMR.chart_map(t::StretchedPair, ξ, p) = p == 1 ? (-1 + ξ[1], ξ[2]) : (t.a * ξ[1], ξ[2])
function TreeAMR.chart_locate(t::StretchedPair, x)
    0 <= x[2] <= 1 || return nothing
    -1 <= x[1] < 0 && return ((x[1] + 1, x[2]), 1)
    0 <= x[1] <= t.a && return ((x[1] / t.a, x[2]), 2)
    return nothing
end
stretched_forest(n; N=8) =
    Forest([(roots=(n, n), extents=((0.0, 1.0), (0.0, 1.0))) for _ in 1:2]; N=N,
           glue=patch_glue_both((1, 1, 2), (2, 1, 1), (1, 2)))
