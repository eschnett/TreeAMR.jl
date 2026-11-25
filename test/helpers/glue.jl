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
# its logical position and patch: what the glue plan fills.
function glue_ghosts(fs)
    forest = fs.forest
    conn = forest.conn
    G, c = fs.G, staggers(fs.centering)
    D = length(G)
    out = Tuple{Int,NTuple{D,Int},NTuple{D,Float64},Int}[]
    dims = TreeAMR.appendix_layout(fs).dims          # the owned high planes (M20)
    for b in 1:nblocks(fs)
        k = blockkey(fs, b)
        p = patch(forest, k)
        h = spacing(forest, k)
        o = block_origin(forest, k)
        for I in CartesianIndices(size(fs.work)[1:D])
            idx = Tuple(I)
            δ = ntuple(d -> idx[d] <= G[d] ? -1 :
                            idx[d] > G[d] + fs.forest.N + dims[b][d] ? 1 : 0, D)
            all(iszero, δ) && continue
            TreeAMR.interpolated_region(conn, k, δ) || continue
            ξ = ntuple(d -> o[d] + (idx[d] - G[d] - 1 + (1 - c[d]) / 2) * h, D)
            push!(out, (b, idx, ξ, p))
        end
    end
    return out
end

# Owned points of a set, the appendix included, from `f(ξ, p)` at their
# positions, times `v` for variable `v`, every other stored point NaN,
# then the fill with `boundary`.
function fill_from_owned!(fs, sched, f; boundary=nothing)
    forest = fs.forest
    G, N = fs.G, forest.N
    D = length(G)
    c = staggers(fs.centering)
    dims = TreeAMR.appendix_layout(fs).dims
    fs.work .= NaN
    for b in 1:nblocks(fs)
        k = blockkey(fs, b)
        p, h, o = patch(forest, k), spacing(forest, k), block_origin(forest, k)
        for I in CartesianIndices(ntuple(d -> (G[d] + 1):(G[d] + N + dims[b][d]), D))
            ξ = ntuple(d -> o[d] + (I[d] - G[d] - 1 + (1 - c[d]) / 2) * h, D)
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

# M21: the kinked strip, for the fold of ghosts interpolated beside a
# symmetry plane (`symmetry_tests.jl`, the device suite).

# A strip of three patches whose maps kink where they meet: patch 1, the
# centre, is affine over `x ∈ [-1, 1]`; patches 2 and 3 continue it to
# `x > 1` and `x < -1` by `|x| = 1 + aη + bη²`, `y = ζ(1 + cη(1 - η))`, so
# that the Jacobian jumps at both glues. The layout is symmetric under the
# mirror `y → -y` and under the half turn `(x, y) → (-x, -y)`, which maps
# patch 2 onto patch 3. The atlas locates any point, past a patch's box
# too, so that a reduced layout's ghost beyond its symmetry plane comes
# back past the plane, for the fold.
struct KinkedStrip{T}
    a::T
    b::T
    c::T
end
const STRIP = KinkedStrip(0.8, 0.4, 0.3)
# The strip in another number type, for a hook that runs on a device.
strip_in(::Type{T}) where {T} = KinkedStrip{T}(STRIP.a, STRIP.b, STRIP.c)
function TreeAMR.chart_map(t::KinkedStrip, ξ, p)
    p == 1 && return (ξ[1], ξ[2])
    η, ζ = ξ
    x = 1 + t.a * η + t.b * η^2
    return (p == 2 ? x : -x, ζ * (1 + t.c * η * (1 - η)))
end
# The maps' derivatives, for an interpolation's physical derivatives.
function TreeAMR.chart_jacobian(t::KinkedStrip, ξ, p)
    z = zero(ξ[1])
    p == 1 && return ((one(z), z), (z, one(z)))
    η, ζ = ξ
    sx = p == 2 ? 1 : -1
    return ((sx * (t.a + 2t.b * η), z), (ζ * t.c * (1 - 2η), 1 + t.c * η * (1 - η)))
end
function TreeAMR.chart_hessian(t::KinkedStrip, ξ, p)
    z = zero(ξ[1])
    p == 1 && return (((z, z), (z, z)), ((z, z), (z, z)))
    η, ζ = ξ
    sx = p == 2 ? 1 : -1
    cross = t.c * (1 - 2η)
    return (((sx * 2t.b, z), (z, z)), ((-2t.c * ζ, cross), (cross, z)))
end

function TreeAMR.chart_locate(t::KinkedStrip, x)
    abs(x[1]) <= 1 && return ((x[1], x[2]), 1)
    X = abs(x[1]) - 1
    η = (-t.a + sqrt(t.a^2 + 4t.b * X)) / (2t.b)
    return ((η, x[2] / (1 + t.c * η * (1 - η))), x[1] > 0 ? 2 : 3)
end

# The strip over `y ∈ [-1, 1]` (`:full`), or its upper half with a mirror at
# `y = 0` (`:mirror`) or with the half turn's seam there (`:half`): the
# centre's low face glued to itself reversed along `x`, and the two kinked
# patches' low faces to each other, both turned by 2.
function strip_forest(kind; n=1, N=8, T=Float64)
    lo = kind === :full ? -1.0 : 0.0
    m = kind === :full ? 2n : n
    patches = [(roots=(2n, m), extents=((-1.0, 1.0), (lo, 1.0))),
               (roots=(n, m), extents=((0.0, 1.0), (lo, 1.0))),
               (roots=(n, m), extents=((0.0, 1.0), (lo, 1.0)))]
    glue = [patch_glue_both((1, 1, 2), (2, 1, 1), (1, 2));
            patch_glue_both((1, 1, 1), (3, 1, 1), (-1, 2))]
    kind === :full && return Forest{T}(patches; N=N, glue=glue)
    kind === :mirror &&
        return Forest{T}(patches; N=N, glue=glue, reflecting=((false, false), (true, false)))
    # The same mirror given patch by patch, label 2 on each low face.
    kind === :ownmirror &&
        return Forest{T}([merge(q, (mirror=((0, 0), (2, 0)),)) for q in patches]; N=N,
                         glue=glue)
    turned = [(from=(1, 2, 1), to=(1, 2, 1), map=(-1, -2), turn=2),
              (from=(2, 2, 1), to=(3, 2, 1), map=(1, -2), turn=2),
              (from=(3, 2, 1), to=(2, 2, 1), map=(1, -2), turn=2)]
    return Forest{T}(patches; N=N, glue=[glue; turned], halfturn=(1, 2))
end

# Data with each symmetry: for the mirror a scalar even in `y` and one odd;
# for the half turn a scalar it leaves alone and a vector it negates.
# In the type of the point, so that a device without `Float64` runs it.
function strip_data(::Val{:mirror}, x)
    k(v) = oftype(x[1], v)
    return (cos(k(1.3) * x[2]) * sin(k(0.9) * x[1] + k(0.3)),
            sin(k(1.1) * x[2]) * cos(k(0.8) * x[1]))
end
function strip_data(::Val{:half}, x)
    k(v) = oftype(x[1], v)
    return (cos(k(1.3) * x[2]) * cos(k(0.9) * x[1]) + sin(k(0.7) * x[1]) * sin(k(1.1) * x[2]),
            sin(k(0.9) * x[1] + k(1.1) * x[2]), sin(k(0.5) * x[1] - k(0.7) * x[2]))
end
strip_keywords(::Val{:mirror}) = (parity=[(EvenParity, EvenParity), (EvenParity, OddParity)],)
strip_keywords(::Val{:ownmirror}) = strip_keywords(Val(:mirror))
strip_data(::Val{:ownmirror}, x) = strip_data(Val(:mirror), x)
strip_keywords(::Val{:half}) = (halfturn=(1, -2, -3),)

# A set over the strip filled from the data at its owned points, the
# appendix included, and its ghosts filled, the outer faces by the data.
function strip_set(forest, sym, C; full=false, T=Float64, backend=CPU())
    nv = length(strip_data(Val(sym), (T(0), T(0))))
    kw = full ? (;) : strip_keywords(Val(sym))
    fs = FieldSet{T}(forest, nv; G=3, centering=C, glue=:interpolate, backend=backend, kw...)
    # Plain data only, for a device: the symmetry as a `Val`, the atlas in `T`.
    sv, atlas = Val(sym), strip_in(T)
    f(ξ, p, v) = strip_data(sv, TreeAMR.chart_map(atlas, ξ, p))[v]
    hook = CellBoundary(WithPatch((ξ, p, v, δ) -> f(ξ, p, v)))
    sched = GhostSchedule(fs, KINKED_OPS; atlas=STRIP)
    G, N = fs.G, forest.N
    c = staggers(fs.centering)
    dims = TreeAMR.appendix_layout(fs).dims
    work = fill(T(NaN), size(fs.work))
    for b in 1:nblocks(fs)
        k = blockkey(fs, b)
        p, h, o = patch(forest, k), spacing(forest, k), block_origin(forest, k)
        for I in CartesianIndices(ntuple(d -> (G[d] + 1):(G[d] + N + dims[b][d]), 2)),
            v in 1:nv
            work[I, v, b] = f(ntuple(d -> o[d] + (I[d] - G[d] - 1 + (1 - c[d]) / 2) * h,
                                     2), p, v)
        end
    end
    copyto!(fs.work, work)
    fill_ghosts!(fs, sched; boundary=hook)
    return fs, sched
end

# The largest difference between every stored point of `fsr`, over a
# reduced strip, and the full strip's at the same patch and position, and
# the number of points compared.
function strip_difference(fsr, fsf)
    key(fs, b) = (patch(fs.forest, blockkey(fs, b)), level(blockkey(fs, b)),
                  block_origin(fs.forest, blockkey(fs, b)))
    index = Dict(key(fsf, b) => b for b in 1:nblocks(fsf))
    worst, n = 0.0, 0
    for b in 1:nblocks(fsr)
        bf = index[key(fsr, b)]
        for I in CartesianIndices(size(fsr.work)[1:2]), v in 1:fsr.nvars
            x, y = fsr.work[I, v, b], fsf.work[I, v, bf]
            worst = max(worst, abs(x - y))
            n += 1
        end
    end
    return worst, n
end
