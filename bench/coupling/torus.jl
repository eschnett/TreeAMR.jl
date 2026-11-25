# Prototype of the inter-patch coupling (TreeAMR PLAN.md, M20 step 3b).
#
# A 2D torus of two patches, periodic along ζ, glued at both normal faces.
# Patch 1 is affine, x = -1 + ξ, y = ζ. Patch 2 is curved and sheared,
# x = aη + bη², y = ζ + cη(1-η), so the map's Jacobian jumps at both glues
# (spacing a vs 1 and a+2b vs 1, grid lines meeting at an angle). Both
# patches have n cells of logical spacing 1/n along each axis, so the glue
# is conforming. Vertex-centered, the scalar wave in second-order form,
# ∂ₜu = v, ∂ₜv = Δu = g^{ab}∂_a∂_b u + (Δξ^b)∂_b u, with fourth-order
# centered stencils (mixed derivatives as products of first ones) and
# sixth-order Kreiss–Oliger dissipation, RK4, as TreeGeneralizedHarmonic
# discretizes. Ghosts G = 3.
#
# Couplings across the glues:
#   :copy      TreeAMR today: the ghost takes the value at the same index
#              continued into the patch across, ignoring the kink.
#   :halfopen  each patch owns its points 0…n-1 along the normal; its high
#              face plane is the low face of the patch across, copied (a
#              node); every other ghost is interpolated at its physical
#              position from the patch across, with an m-point Lagrange
#              stencil per dimension restricted to that patch's points
#              0…n (lop-sided near the face). `avg` averages the two
#              right-hand sides on a face plane.
#   :overlap   each patch owns its points -o…n+o, both face planes and o
#              planes past each face; ghosts are interpolated from the
#              patch across within its -o…n+o.

using Printf, Random

abstract type Torus end
struct Setup <: Torus      # the kinked torus
    a::Float64
    b::Float64
    c::Float64
end
period_x(s::Setup) = 1 + s.a + s.b
box_centre(s::Setup, p) = p == 1 ? -0.5 : (s.a + s.b) / 2

# The control: one smooth periodic map, x = σ + α sin πσ, y = ζ + γ sin πσ,
# σ = ξ - 1 on patch 1 and η on patch 2, so that the glues do not kink
# and the coefficients still vary.
struct Wavy <: Torus
    α::Float64
    γ::Float64
end
period_x(::Wavy) = 2.0
box_centre(::Wavy, p) = p == 1 ? -0.5 : 0.5
@inline σof(p, ξ) = p == 1 ? ξ - 1 : ξ
@inline function position(s::Wavy, p, ξ, ζ)
    σ = σof(p, ξ)
    return (σ + s.α * sin(π * σ), ζ + s.γ * sin(π * σ))
end
@inline function logical(s::Wavy, p, x, y)
    σ = x
    for _ in 1:50
        f = σ + s.α * sin(π * σ) - x
        σ -= f / (1 + s.α * π * cos(π * σ))
        abs(f) < 1e-15 && break
    end
    return (p == 1 ? σ + 1 : σ, y - s.γ * sin(π * σ))
end
@inline function metric(s::Wavy, p, ξ)
    σ = σof(p, ξ)
    xs, ys = 1 + s.α * π * cos(π * σ), s.γ * π * cos(π * σ)
    xss, yss = -s.α * π^2 * sin(π * σ), -s.γ * π^2 * sin(π * σ)
    J = xs                                   # M = [xs 0; ys 1]
    # M⁻¹ = [1/J 0; -ys/J 1]; gᵃᵇ = M⁻¹ M⁻ᵀ; Δξᵇ = -(M⁻¹)ᵇᵢ gᵃᶜ ∂ₐ∂꜀Xⁱ
    g11 = 1 / J^2
    g12 = -ys / J^2
    g22 = ys^2 / J^2 + 1
    L1 = -(xss / J) * g11
    L2 = -((-ys / J) * xss + yss) * g11
    return (g11, g12, g22, L1, L2, J)
end

@inline function position(s::Setup, p, ξ, ζ)
    p == 1 && return (-1 + ξ, ζ)
    return (s.a * ξ + s.b * ξ^2, ζ + s.c * ξ * (1 - ξ))
end

@inline function logical(s::Setup, p, x, y)
    p == 1 && return (x + 1, y)
    η = s.b == 0 ? x / s.a : (-s.a + sqrt(s.a^2 + 4s.b * x)) / (2s.b)
    return (η, y - s.c * η * (1 - η))
end

# (g¹¹, g¹², g²², Δξ¹, Δξ², √g) at normal coordinate η of patch p; the map
# does not depend on ζ beyond a shift, so neither does the metric.
@inline function metric(s::Setup, p, η)
    p == 1 && return (1.0, 0.0, 1.0, 0.0, 0.0, 1.0)
    J = s.a + 2s.b * η
    q = s.c * (1 - 2η)
    return (1 / J^2, -q / J^2, 1 + q^2 / J^2, -2s.b / J^3,
            2s.c / J^2 + 2s.b * q / J^3, J)
end

struct Variant
    name::String
    mode::Symbol
    o::Int
    avg::Bool
    m::Int
end

mutable struct Patch
    p::Int
    n::Int
    dlo::Int      # owned (evolved) normal indices dlo:dhi
    dhi::Int
    ilo::Int      # stored normal indices ilo:ihi
    ihi::Int
    u::Matrix{Float64}
    v::Matrix{Float64}
    met::Matrix{Float64}   # 6 × stored normal index
end
@inline row(P, i) = i - P.ilo + 1

struct Entry
    tp::Int; ti::Int; tj::Int       # target patch, normal, tangential
    sp::Int; sn::Int; st::Int       # source patch, stencil starts
    wn::Vector{Float64}; wt::Vector{Float64}
end

const G = 3

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

struct Sim
    s::Torus
    var::Variant
    n::Int
    P::Vector{Patch}
    fill::Vector{Entry}
    eps::Float64
end

function Sim(s::Torus, var::Variant, n::Int; eps=0.0)
    dlo, dhi = var.mode in (:overlap, :overlapcore) ? (-var.o, n + var.o) : (0, n - 1)
    ilo, ihi = dlo - G, max(dhi, n) + G   # half-open: the face plane n and G past it
    P = map(1:2) do p
        NI = ihi - ilo + 1
        met = zeros(6, NI)
        for i in ilo:ihi
            met[:, i-ilo+1] .= metric(s, p, i / n)
        end
        Patch(p, n, dlo, dhi, ilo, ihi, zeros(NI, n), zeros(NI, n), met)
    end
    fill = Entry[]
    var.mode === :copy || build_fill!(fill, s, var, n, P)
    return Sim(s, var, n, P, fill, eps)
end

# The interpolated ghosts, located once. Half-open: the face planes first,
# from owned points, then the rest from owned points and face planes.
function build_fill!(fill, s, var, n, P)
    m = var.m
    centre = (box_centre(s, 1), box_centre(s, 2))
    Lx = period_x(s)
    for pass in (1, 2), p in 1:2
        q = 3 - p
        T, S = P[p], P[q]
        for i in T.ilo:T.ihi
            T.dlo <= i <= T.dhi && continue
            isface = i == 0 || i == n
            if var.mode === :halfopen
                (pass == 1) == isface || continue
                wlo, whi = 0, pass == 1 ? n - 1 : n
            elseif var.mode === :overlapcore
                # Donors from the patch across's own box 0…n only, never
                # its overlap points, which are its ghosts' neighbours.
                pass == 1 || continue
                wlo, whi = 0, n
            else
                pass == 1 || continue
                wlo, whi = S.dlo, S.dhi
            end
            for j in 0:n-1
                x, y = position(s, p, i / n, j / n)
                x -= Lx * round((x - centre[q]) / Lx)
                η, ζ = logical(s, q, x, y)
                tn, tt = η * n, ζ * n
                sn = clamp(floor(Int, tn) - m ÷ 2 + 1, wlo, whi - m + 1)
                st = floor(Int, tt) - m ÷ 2 + 1
                push!(fill, Entry(p, i, j, q, sn, st, lagrange(tn, sn, m),
                                  lagrange(tt, st, m)))
            end
        end
    end
    return fill
end

function fill_ghosts!(sim::Sim)
    n = sim.n
    if sim.var.mode === :copy
        for p in 1:2
            T, S = sim.P[p], sim.P[3-p]
            for g in 0:G, j in 1:n           # past the high face, the face too
                T.u[row(T, n + g), j] = S.u[row(S, g), j]
                T.v[row(T, n + g), j] = S.v[row(S, g), j]
            end
            for g in 1:G, j in 1:n
                T.u[row(T, -g), j] = S.u[row(S, n - g), j]
                T.v[row(T, -g), j] = S.v[row(S, n - g), j]
            end
        end
        return
    end
    @inbounds for e in sim.fill
        T, S = sim.P[e.tp], sim.P[e.sp]
        su = 0.0
        sv = 0.0
        for l in eachindex(e.wt)
            jj = mod(e.st + l - 1, n) + 1
            au = 0.0
            av = 0.0
            for k in eachindex(e.wn)
                r = row(S, e.sn + k - 1)
                au += e.wn[k] * S.u[r, jj]
                av += e.wn[k] * S.v[r, jj]
            end
            su += e.wt[l] * au
            sv += e.wt[l] * av
        end
        T.u[row(T, e.ti), e.tj+1] = su
        T.v[row(T, e.ti), e.tj+1] = sv
    end
end

# Right-hand side at the normal indices lo:hi of patch T into (du, dv).
function rhs_patch!(du, dv, T::Patch, lo, hi, n, eps)
    u, v = T.u, T.v
    h1, h2 = Float64(n), Float64(n)^2
    @inbounds for i in lo:hi
        r = row(T, i)
        g11, g12, g22, Γ1, Γ2, _ = T.met[1, r], T.met[2, r], T.met[3, r],
                                   T.met[4, r], T.met[5, r], T.met[6, r]
        for j in 1:n
            jm1 = j == 1 ? n : j - 1
            jm2 = jm1 == 1 ? n : jm1 - 1
            jm3 = jm2 == 1 ? n : jm2 - 1
            jp1 = j == n ? 1 : j + 1
            jp2 = jp1 == n ? 1 : jp1 + 1
            jp3 = jp2 == n ? 1 : jp2 + 1
            ui = (u[r-2, j] - 8u[r-1, j] + 8u[r+1, j] - u[r+2, j]) / 12
            uj = (u[r, jm2] - 8u[r, jm1] + 8u[r, jp1] - u[r, jp2]) / 12
            uii = (-u[r-2, j] + 16u[r-1, j] - 30u[r, j] + 16u[r+1, j] - u[r+2, j]) / 12
            ujj = (-u[r, jm2] + 16u[r, jm1] - 30u[r, j] + 16u[r, jp1] - u[r, jp2]) / 12
            dj(rr) = (u[rr, jm2] - 8u[rr, jm1] + 8u[rr, jp1] - u[rr, jp2]) / 12
            uij = (dj(r - 2) - 8dj(r - 1) + 8dj(r + 1) - dj(r + 2)) / 12
            lap = h2 * (g11 * uii + 2g12 * uij + g22 * ujj) + h1 * (Γ1 * ui + Γ2 * uj)
            du[r, j] = v[r, j]
            dv[r, j] = lap
            if eps != 0
                ko(f) = (f[r-3, j] - 6f[r-2, j] + 15f[r-1, j] - 20f[r, j] +
                         15f[r+1, j] - 6f[r+2, j] + f[r+3, j] +
                         f[r, jm3] - 6f[r, jm2] + 15f[r, jm1] - 20f[r, j] +
                         15f[r, jp1] - 6f[r, jp2] + f[r, jp3]) / 64
                du[r, j] += eps * h1 * ko(u)
                dv[r, j] += eps * h1 * ko(v)
            end
        end
    end
end

function rhs!(K, sim::Sim)
    fill_ghosts!(sim)
    n = sim.n
    avg = sim.var.mode === :halfopen && sim.var.avg
    for p in 1:2
        T = sim.P[p]
        rhs_patch!(K[p][1], K[p][2], T, T.dlo, avg ? n : T.dhi, n, sim.eps)
    end
    if avg
        # Patch 1's face plane n is patch 2's owned plane 0, and patch 2's
        # face plane n is patch 1's owned plane 0, tangent for tangent.
        for (o, c) in ((2, 1), (1, 2))
            O, C = sim.P[o], sim.P[c]
            for f in 1:2, j in 1:n
                K[o][f][row(O, 0), j] = (K[o][f][row(O, 0), j] + K[c][f][row(C, n), j]) / 2
            end
        end
    end
end

function rk4!(sim::Sim, dt, work)
    U0, K1, K2, K3, K4 = work
    n = sim.n
    rows(T) = row(T, T.dlo):row(T, T.dhi)
    for p in 1:2
        T = sim.P[p]
        U0[p][1] .= T.u; U0[p][2] .= T.v
    end
    rhs!(K1, sim)
    for (K, c) in ((K1, dt / 2), (K2, dt / 2), (K3, dt))
        for p in 1:2
            T = sim.P[p]; R = rows(T)
            @views T.u[R, :] .= U0[p][1][R, :] .+ c .* K[p][1][R, :]
            @views T.v[R, :] .= U0[p][2][R, :] .+ c .* K[p][2][R, :]
        end
        rhs!(K === K1 ? K2 : K === K2 ? K3 : K4, sim)
    end
    for p in 1:2
        T = sim.P[p]; R = rows(T)
        @views T.u[R, :] .= U0[p][1][R, :] .+ (dt / 6) .* (K1[p][1][R, :] .+
            2 .* K2[p][1][R, :] .+ 2 .* K3[p][1][R, :] .+ K4[p][1][R, :])
        @views T.v[R, :] .= U0[p][2][R, :] .+ (dt / 6) .* (K1[p][2][R, :] .+
            2 .* K2[p][2][R, :] .+ 2 .* K3[p][2][R, :] .+ K4[p][2][R, :])
    end
end

workspace(sim) = ntuple(_ -> [(zero(T.u), zero(T.v)) for T in sim.P], 5)

struct PlaneWave
    kx::Float64; ky::Float64; ω::Float64; φ::Float64
end
PlaneWave(s::Torus, mx, my, φ) =
    let kx = 2π * mx / period_x(s), ky = 2π * my
        PlaneWave(kx, ky, hypot(kx, ky), φ)
    end
exact(w::PlaneWave, x, y, t) =
    (sin(w.kx * x + w.ky * y - w.ω * t + w.φ), -w.ω * cos(w.kx * x + w.ky * y - w.ω * t + w.φ))

function set_state!(sim::Sim, f)
    n = sim.n
    for T in sim.P, i in T.ilo:T.ihi, j in 0:n-1
        x, y = position(sim.s, T.p, i / n, j / n)
        T.u[row(T, i), j+1], T.v[row(T, i), j+1] = f(x, y)
    end
end

# Errors over the core points 0…n-1 of each patch, which tile the torus
# once: √g-weighted L2, L∞, and L∞ within `near` points of a glue.
function errors(sim::Sim, w::PlaneWave, t; near=4)
    n = sim.n
    l2 = 0.0; linf = 0.0; lnear = 0.0
    for T in sim.P, i in 0:n-1
        r = row(T, i)
        J = T.met[6, r]
        for j in 0:n-1
            x, y = position(sim.s, T.p, i / n, j / n)
            e = abs(T.u[r, j+1] - exact(w, x, y, t)[1])
            l2 += J * e^2 / n^2
            linf = max(linf, e)
            (i < near || i >= n - near) && (lnear = max(lnear, e))
        end
    end
    return sqrt(l2), linf, lnear
end

# √g-weighted L2 norm of (u, v) over the core.
function statenorm(sim::Sim)
    n = sim.n
    s = 0.0
    for T in sim.P, i in 0:n-1
        r = row(T, i)
        J = T.met[6, r]
        for j in 1:n
            s += J * (T.u[r, j]^2 + T.v[r, j]^2) / n^2
        end
    end
    return sqrt(s)
end

function timestep(sim::Sim, cfl)
    λ = 0.0
    for T in sim.P, r in axes(T.met, 2)
        g11, g12, g22 = T.met[1, r], T.met[2, r], T.met[3, r]
        tr, dt = g11 + g22, g11 * g22 - g12^2
        λ = max(λ, sqrt((tr + sqrt(max(tr^2 - 4dt, 0))) / 2))
    end
    return cfl / (sim.n * λ)
end

function evolve!(sim::Sim, tend, cfl; monitor=nothing, every=0)
    dt0 = timestep(sim, cfl)
    nsteps = ceil(Int, tend / dt0)
    dt = tend / nsteps
    work = workspace(sim)
    t = 0.0
    for k in 1:nsteps
        rk4!(sim, dt, work)
        t = k * dt
        if monitor !== nothing && every > 0 && (k % every == 0 || k == nsteps)
            monitor(t) || return t
        end
    end
    return t
end

# Discrete energy Σ √g (v² + g^{ab} D_a u D_b u) over the core, with the
# fourth-order first derivatives; blind to the constant mode of u, which
# the mean of v drives linearly in time.
function energy(sim::Sim)
    fill_ghosts!(sim)
    n = sim.n
    E = 0.0
    @inbounds for T in sim.P, i in 0:n-1
        r = row(T, i)
        g11, g12, g22, J = T.met[1, r], T.met[2, r], T.met[3, r], T.met[6, r]
        u = T.u
        for j in 1:n
            jm1 = j == 1 ? n : j - 1; jm2 = jm1 == 1 ? n : jm1 - 1
            jp1 = j == n ? 1 : j + 1; jp2 = jp1 == n ? 1 : jp1 + 1
            ui = n * (u[r-2, j] - 8u[r-1, j] + 8u[r+1, j] - u[r+2, j]) / 12
            uj = n * (u[r, jm2] - 8u[r, jm1] + 8u[r, jp1] - u[r, jp2]) / 12
            E += J * (T.v[r, j]^2 + g11 * ui^2 + 2g12 * ui * uj + g22 * uj^2) / n^2
        end
    end
    return E
end
