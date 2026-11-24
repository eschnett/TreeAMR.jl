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
# patch and nowhere else, past the high face planes the block owns, with its logical position and patch: what the
# glue plan fills.
function glue_ghosts(fs)
    forest = fs.forest
    conn = forest.conn
    G, c = fs.G, staggers(fs.centering)
    out = Tuple{Int,NTuple{2,Int},NTuple{2,Float64},Int}[]
    dims = TreeAMR.appendix_layout(fs).dims          # the owned high planes (M20)
    for b in 1:nblocks(fs)
        k = forest.leaves[b]
        p = patch(forest, k)
        h = spacing(forest, k)
        o = block_origin(forest, k)
        for I in CartesianIndices(size(fs.work)[1:2])
            idx = Tuple(I)
            δ = ntuple(d -> idx[d] <= G[d] ? -1 :
                            idx[d] > G[d] + fs.forest.N + dims[b][d] ? 1 : 0, 2)
            all(iszero, δ) && continue
            TreeAMR.interpolated_region(conn, k, δ) || continue
            ξ = ntuple(d -> o[d] + (idx[d] - G[d] - 1) * h, 2)
            push!(out, (b, idx, ξ, p))
        end
    end
    return out
end
