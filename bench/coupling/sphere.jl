# Prototype of half-open inter-patch coupling on the cubed sphere's
# surface (TreeAMR PLAN.md, M20 step 3b): the shells' tangential structure
# of the seven-patch ball, with three patches at every corner.
#
# Six gnomonic patches, face (k, s) the cube face x_k = s, logical
# (ξ₁, ξ₂) ∈ [0,1]², A = 2ξ₁ - 1, B = 2ξ₂ - 1 along the other two axes
# l < m with orientation signs, f_k = s, f_l = o_l A, f_m = o_m B,
# X = f/|f| on the unit sphere. The wave ∂ₜu = v, ∂ₜv = Δu with the
# Laplace–Beltrami operator, in every patch
#     Δu = r²[(1+A²)u_AA + 2AB u_AB + (1+B²)u_BB + 2A u_A + 2B u_B],
# r² = 1 + A² + B², √g = r⁻³; fourth-order stencils, mixed as products,
# sixth-order Kreiss–Oliger, RK4, G = 3, vertex-centered, half-open
# ownership: a patch owns its points 0…n-1 along each axis. The
# orientations are the ones that give every edge exactly one owner (each
# ring of four faces around an axis turns one way), all three rings the
# same way, so that the two corners no face owns are (−,−,−) and (+,+,+).
# Ghosts: face planes copied from their owner; the two unowned corners
# extrapolated along the edges into them (`corner = :avg`, the mean of the
# three edges, or `:single`, one edge); every other ghost interpolated at
# its physical position from the face holding it, with an m-point
# Lagrange stencil per dimension inside that face's points 0…n.

using Printf, Random, LinearAlgebra

const G = 6      # storage margin past the box
const GH = 3     # ghost width the scheme reads

struct Face
    k::Int; s::Int; l::Int; m::Int; ol::Int; om::Int
end
faceindex(k, s) = 2(k - 1) + (s > 0 ? 2 : 1)

@inline function pos(F::Face, A, B)
    f = zeros(3)
    f[F.k] = F.s; f[F.l] = F.ol * A; f[F.m] = F.om * B
    return f ./ norm(f)
end
@inline function coords(F::Face, X)
    c = F.s / X[F.k]
    return (F.ol * c * X[F.l], F.om * c * X[F.m])
end

# All orientations with one low side per edge, and the unowned corners of
# each.
function orientations()
    shells = [(k, s) for k in 1:3 for s in (-1, 1)]
    unk = [((k, s), l) for (k, s) in shells for l in 1:3 if l != k]
    sols = []
    for bits in 0:(2^12-1)
        o = Dict(unk[i] => ((bits >> (i - 1)) & 1 == 1 ? 1 : -1) for i in 1:12)
        low(k, s, l, t) = o[((k, s), l)] * t == -1
        all(low(k, s, l, t) != low(l, t, k, s)
            for (k, s) in shells for (l, t) in shells if k < l) || continue
        unowned = [c for c in Iterators.product((-1, 1), (-1, 1), (-1, 1))
                   if !any(k -> all(l -> l == k || low(k, c[k], l, c[l]), 1:3), 1:3)]
        push!(sols, (o, unowned))
    end
    return sols
end

function faces_for(o)
    F = Vector{Face}(undef, 6)
    for k in 1:3, s in (-1, 1)
        l, m = filter(!=(k), 1:3)
        F[faceindex(k, s)] = Face(k, s, l, m, o[((k, s), l)], o[((k, s), m)])
    end
    return F
end

mutable struct SPatch
    F::Face
    u::Matrix{Float64}
    v::Matrix{Float64}
    c::Array{Float64,3}     # c11 c12 c22 d1 d2 w, per stored point
end
@inline ix(i) = i + G + 1

struct FEntry
    tp::Int; ti::Int; tj::Int
    src::Vector{Tuple{Int,Int,Int,Float64}}
end

function lagrange(t, s, m)
    w = zeros(m)
    for k in 0:m-1
        l = 1.0
        for q in 0:m-1
            q == k && continue
            l *= (t - (s + q)) / (k - q)
        end
        w[k+1] = l
    end
    return w
end

struct Sphere
    n::Int
    m::Int
    P::Vector{SPatch}
    copies::Vector{FEntry}       # face planes from their owner
    corners::Vector{FEntry}      # the unowned corners
    ghosts::Vector{FEntry}       # everything else
    avg::Bool
    eps::Float64
    extra::Vector{Tuple{Int,Int,Int}}   # evolved points past 0…n-1 (corner = :own)
    dlo::Int                            # evolved range dlo:dhi along each axis
    dhi::Int
end

# The owner of the unit vector X among the faces: the face holding it as
# one of its points 0…n-1 along both axes, as (face, i, j), or nothing.
function owner(F, n, X; skip=0)
    for q in 1:6
        q == skip && continue
        Fq = F[q]
        Fq.s * X[Fq.k] > 0 || continue
        A, B = coords(Fq, X)
        ti, tj = (A + 1) * n / 2, (B + 1) * n / 2
        ri, rj = round(Int, ti), round(Int, tj)
        abs(ti - ri) < 1e-8 && abs(tj - rj) < 1e-8 || continue
        0 <= ri <= n - 1 && 0 <= rj <= n - 1 && return (q, ri, rj)
    end
    return nothing
end

function Sphere(n; m=6, corner=:avg, mc=m, avg=false, eps=0.0, overlap=-1)
    # overlap = o ≥ 0: every face evolves its points -o…n+o, both face
    # planes and o planes past each edge, and every other ghost is
    # interpolated from the face holding it within that face's -o…n+o.
    ov = overlap >= 0
    dlo, dhi = ov ? (-overlap, n + overlap) : (0, n - 1)
    wlo, whi = ov ? (dlo, dhi) : (0, n)
    o = first(s for s in orientations() if Set(s[2]) == Set([(-1, -1, -1), (1, 1, 1)]))[1]
    F = faces_for(o)
    h = n / 2
    P = map(1:6) do p
        NI = n + 1 + 2G
        c = zeros(6, NI, NI)
        for i in -G:n+G, j in -G:n+G
            A, B = 2i / n - 1, 2j / n - 1
            r2 = 1 + A^2 + B^2
            c[:, ix(i), ix(j)] .= (r2 * (1 + A^2) * h^2, 2r2 * A * B * h^2,
                                   r2 * (1 + B^2) * h^2, 2A * r2 * h, 2B * r2 * h,
                                   r2^(-1.5) / h^2)
        end
        SPatch(F[p], zeros(NI, NI), zeros(NI, NI), c)
    end
    copies, cornerpts, ghosts = FEntry[], Tuple{Int,Int,Int}[], FEntry[]
    top = (ov ? dhi : n) + GH     # half-open: G planes past the face plane n
    for p in 1:6, i in dlo-GH:top, j in dlo-GH:top
        (dlo <= i <= dhi && dlo <= j <= dhi) && continue
        X = pos(F[p], 2i / n - 1, 2j / n - 1)
        if !ov && 0 <= i <= n && 0 <= j <= n
            own = owner(F, n, X; skip=p)
            if own === nothing
                push!(cornerpts, (p, i, j))
            else
                push!(copies, FEntry(p, i, j, [(own..., 1.0)]))
            end
        else
            # The faces holding X: the largest |X_k|, all of them on a tie.
            a = abs.(X)
            src = Tuple{Int,Int,Int,Float64}[]
            qs = [faceindex(k, X[k] > 0 ? 1 : -1) for k in 1:3 if a[k] > maximum(a) - 1e-12]
            for q in qs
                @assert q != p
                A, B = coords(F[q], X)
                ti, tj = (A + 1) * n / 2, (B + 1) * n / 2
                si = clamp(floor(Int, ti) - m ÷ 2 + 1, wlo, whi - m + 1)
                sj = clamp(floor(Int, tj) - m ÷ 2 + 1, wlo, whi - m + 1)
                wi, wj = lagrange(ti, si, m), lagrange(tj, sj, m)
                for a1 in 1:m, b1 in 1:m
                    push!(src, (q, si + a1 - 1, sj + b1 - 1, wi[a1] * wj[b1] / length(qs)))
                end
            end
            push!(ghosts, FEntry(p, i, j, src))
        end
    end
    # The unowned corners: each lies at a box corner of three faces; along
    # each face's two edges into it, extrapolate from the m nearest points.
    corners = FEntry[]
    extra = Tuple{Int,Int,Int}[]
    ov && return Sphere(n, m, P, copies, corners, ghosts, avg, eps, extra, dlo, dhi)
    @assert length(cornerpts) == 6     # two corners, three faces each
    wx = lagrange(0.0, 1, mc)           # nodes at distances 1…mc, value at 0
    if corner === :own
        # A hypothetical: the first face holding each unowned corner owns
        # and evolves it, as TreeAMR's layout cannot, and the others copy.
        for (p, i, j) in cornerpts
            X = pos(F[p], 2i / n - 1, 2j / n - 1)
            o1 = first(c for c in cornerpts
                       if norm(pos(F[c[1]], 2c[2] / n - 1, 2c[3] / n - 1) - X) < 1e-12)
            if o1 == (p, i, j)
                push!(extra, o1)
            else
                push!(corners, FEntry(p, i, j, [(o1..., 1.0)]))
            end
        end
        return Sphere(n, m, P, copies, corners, ghosts, avg, eps, extra, dlo, dhi)
    end
    for (p, i, j) in cornerpts
        X = pos(F[p], 2i / n - 1, 2j / n - 1)
        ests = Vector{Tuple{Int,Int,Int,Float64}}[]
        for (q, qi, qj) in cornerpts
            norm(pos(F[q], 2qi / n - 1, 2qj / n - 1) - X) < 1e-12 || continue
            di, dj = qi == 0 ? 1 : -1, qj == 0 ? 1 : -1
            push!(ests, [(q, qi + d * di, qj, wx[d]) for d in 1:mc])
            push!(ests, [(q, qi, qj + d * dj, wx[d]) for d in 1:mc])
        end
        @assert length(ests) == 6
        sel = corner === :avg ? ests : ests[1:1]
        push!(corners, FEntry(p, i, j,
                              [(q, a, b, w / length(sel)) for e in sel for (q, a, b, w) in e]))
    end
    return Sphere(n, m, P, copies, corners, ghosts, avg, eps, extra, dlo, dhi)
end

function apply!(S::Sphere, list)
    @inbounds for e in list
        su = 0.0; sv = 0.0
        for (q, a, b, w) in e.src
            su += w * S.P[q].u[ix(a), ix(b)]
            sv += w * S.P[q].v[ix(a), ix(b)]
        end
        S.P[e.tp].u[ix(e.ti), ix(e.tj)] = su
        S.P[e.tp].v[ix(e.ti), ix(e.tj)] = sv
    end
end
function fill_ghosts!(S::Sphere)
    apply!(S, S.copies); apply!(S, S.corners); apply!(S, S.ghosts)
end

@inline function rhs_point!(du, dv, T::SPatch, i, j, eps, h)
    u, v, c = T.u, T.v, T.c
    r, q = ix(i), ix(j)
    @inbounds begin
        ui = (u[r-2, q] - 8u[r-1, q] + 8u[r+1, q] - u[r+2, q]) / 12
        uj = (u[r, q-2] - 8u[r, q-1] + 8u[r, q+1] - u[r, q+2]) / 12
        uii = (-u[r-2, q] + 16u[r-1, q] - 30u[r, q] + 16u[r+1, q] - u[r+2, q]) / 12
        ujj = (-u[r, q-2] + 16u[r, q-1] - 30u[r, q] + 16u[r, q+1] - u[r, q+2]) / 12
        dj(rr) = (u[rr, q-2] - 8u[rr, q-1] + 8u[rr, q+1] - u[rr, q+2]) / 12
        uij = (dj(r - 2) - 8dj(r - 1) + 8dj(r + 1) - dj(r + 2)) / 12
        du[r, q] = v[r, q]
        dv[r, q] = c[1, r, q] * uii + c[2, r, q] * uij + c[3, r, q] * ujj +
                   c[4, r, q] * ui + c[5, r, q] * uj
        if eps != 0
            ko(f) = (f[r-3, q] - 6f[r-2, q] + 15f[r-1, q] - 20f[r, q] + 15f[r+1, q] -
                     6f[r+2, q] + f[r+3, q] + f[r, q-3] - 6f[r, q-2] + 15f[r, q-1] -
                     20f[r, q] + 15f[r, q+1] - 6f[r, q+2] + f[r, q+3]) / 64
            du[r, q] += eps * h * ko(u)
            dv[r, q] += eps * h * ko(v)
        end
    end
end

function rhs!(K, S::Sphere, cnt)
    fill_ghosts!(S)
    n, h = S.n, S.n / 2
    for p in 1:6, j in S.dlo:S.dhi, i in S.dlo:S.dhi
        rhs_point!(K[p][1], K[p][2], S.P[p], i, j, S.eps, h)
    end
    for (p, i, j) in S.extra
        rhs_point!(K[p][1], K[p][2], S.P[p], i, j, S.eps, h)
    end
    if S.avg
        # Each face-plane copy evaluates its own right-hand side, and the
        # owner takes the mean of its own and its copies'.
        for e in S.copies
            rhs_point!(K[e.tp][1], K[e.tp][2], S.P[e.tp], e.ti, e.tj, S.eps, h)
        end
        for p in 1:6
            fill!(cnt[p], 1)
        end
        for e in S.copies
            (q, a, b, _) = e.src[1]
            for f in 1:2
                K[q][f][ix(a), ix(b)] += K[e.tp][f][ix(e.ti), ix(e.tj)]
            end
            cnt[q][ix(a), ix(b)] += 1
        end
        for p in 1:6, f in 1:2
            K[p][f] ./= cnt[p]
        end
    end
end

function rk4!(S::Sphere, dt, W)
    U0, K1, K2, K3, K4, cnt = W
    R = ix(S.dlo):ix(S.dhi)
    for p in 1:6
        U0[p][1] .= S.P[p].u; U0[p][2] .= S.P[p].v
    end
    rhs!(K1, S, cnt)
    for (K, c, Kn) in ((K1, dt / 2, K2), (K2, dt / 2, K3), (K3, dt, K4))
        for p in 1:6
            @views S.P[p].u[R, R] .= U0[p][1][R, R] .+ c .* K[p][1][R, R]
            @views S.P[p].v[R, R] .= U0[p][2][R, R] .+ c .* K[p][2][R, R]
        end
        for (p, i, j) in S.extra
            a, b = ix(i), ix(j)
            S.P[p].u[a, b] = U0[p][1][a, b] + c * K[p][1][a, b]
            S.P[p].v[a, b] = U0[p][2][a, b] + c * K[p][2][a, b]
        end
        rhs!(Kn, S, cnt)
    end
    for p in 1:6, f in 1:2
        X = f == 1 ? S.P[p].u : S.P[p].v
        @views X[R, R] .= U0[p][f][R, R] .+ (dt / 6) .* (K1[p][f][R, R] .+
            2 .* K2[p][f][R, R] .+ 2 .* K3[p][f][R, R] .+ K4[p][f][R, R])
    end
    for (p, i, j) in S.extra, f in 1:2
        X = f == 1 ? S.P[p].u : S.P[p].v
        a, b = ix(i), ix(j)
        X[a, b] = U0[p][f][a, b] + (dt / 6) * (K1[p][f][a, b] + 2K2[p][f][a, b] +
                                               2K3[p][f][a, b] + K4[p][f][a, b])
    end
end

workspace(S) = (ntuple(_ -> [(zero(T.u), zero(T.v)) for T in S.P], 5)...,
                [zeros(size(T.u)) for T in S.P])

function timestep(S::Sphere, cfl)
    # The largest logical speed: √ of the largest eigenvalue of g^{ab}
    # over the evolved points, in index units.
    λ = 0.0
    for T in S.P, i in S.dlo:S.dhi, j in S.dlo:S.dhi
        c11, c12, c22 = T.c[1, ix(i), ix(j)], T.c[2, ix(i), ix(j)] / 2, T.c[3, ix(i), ix(j)]
        tr, dt = c11 + c22, c11 * c22 - c12^2
        λ = max(λ, sqrt((tr + sqrt(max(tr^2 - 4dt, 0))) / 2))
    end
    return cfl / λ
end

function evolve!(S::Sphere, tend, cfl; every=0, monitor=nothing)
    nsteps = ceil(Int, tend / timestep(S, cfl))
    dt = tend / nsteps
    W = workspace(S)
    for k in 1:nsteps
        rk4!(S, dt, W)
        if monitor !== nothing && every > 0 && (k % every == 0 || k == nsteps)
            monitor(k * dt) || return k * dt
        end
    end
    return tend
end

# u = Re[(X⋅e₁ + i X⋅e₂)^l e^{-iωt}], a harmonic of degree l rotating about
# e₁ × e₂, ω = √(l(l+1)).
struct Harmonic
    e1::Vector{Float64}; e2::Vector{Float64}; l::Int; ω::Float64
end
function Harmonic(axis, l)
    e3 = normalize(axis)
    e1 = normalize(cross(e3, abs(e3[1]) < 0.9 ? [1.0, 0, 0] : [0, 1.0, 0]))
    return Harmonic(e1, cross(e3, e1), l, sqrt(l * (l + 1)))
end
function exact(w::Harmonic, X, t)
    z = (dot(X, w.e1) + im * dot(X, w.e2))^w.l * exp(-im * w.ω * t)
    return real(z), real(-im * w.ω * z)
end

function set_state!(S::Sphere, f)
    n = S.n
    for T in S.P, i in -G:n+G, j in -G:n+G
        T.u[ix(i), ix(j)], T.v[ix(i), ix(j)] = f(pos(T.F, 2i / n - 1, 2j / n - 1))
    end
end

function errors(S::Sphere, w, t)
    n = S.n
    l2 = 0.0; linf = 0.0; ledge = 0.0
    for T in S.P, i in 0:n-1, j in 0:n-1
        e = abs(T.u[ix(i), ix(j)] - exact(w, pos(T.F, 2i / n - 1, 2j / n - 1), t)[1])
        l2 += T.c[6, ix(i), ix(j)] * e^2
        linf = max(linf, e)
        (min(i, j) < 4 || max(i, j) >= n - 4) && (ledge = max(ledge, e))
    end
    return sqrt(l2), linf, ledge
end

function energy(S::Sphere)
    fill_ghosts!(S)
    n, h = S.n, S.n / 2
    E = 0.0
    @inbounds for T in S.P, i in 0:n-1, j in 0:n-1
        r, q = ix(i), ix(j)
        u = T.u
        A, B = 2i / n - 1, 2j / n - 1
        r2 = 1 + A^2 + B^2
        ua = h * (u[r-2, q] - 8u[r-1, q] + 8u[r+1, q] - u[r+2, q]) / 12
        ub = h * (u[r, q-2] - 8u[r, q-1] + 8u[r, q+1] - u[r, q+2]) / 12
        E += T.c[6, r, q] * (T.v[r, q]^2 +
                             r2 * ((1 + A^2) * ua^2 + 2A * B * ua * ub + (1 + B^2) * ub^2))
    end
    return E
end

# The largest difference between u at every owned point and at its image
# under the linear map M, relative to the largest |u|.
function asymmetry(S::Sphere, M)
    n = S.n
    d = 0.0; a = 0.0
    for p in 1:6, i in 0:n-1, j in 0:n-1
        X = pos(S.P[p].F, 2i / n - 1, 2j / n - 1)
        own = owner([T.F for T in S.P], n, M * X)
        own === nothing && continue          # the image is an unowned corner
        q, a1, b1 = own
        d = max(d, abs(S.P[p].u[ix(i), ix(j)] - S.P[q].u[ix(a1), ix(b1)]))
        a = max(a, abs(S.P[p].u[ix(i), ix(j)]))
    end
    return d / a
end
