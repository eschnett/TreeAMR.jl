# Burgers' equation as an application of the mesh -- the conservative
# counterpart of `wave.jl`, and the test problem of M8b.
#
#     ∂ₜu + Σ_d ∂_d (u²/2) = 0
#
# on a periodic box. This lives in the tests, not in the package: TreeAMR
# supplies the mesh, the exchange and the interface fixup, never physics.
#
# It exists to exercise the one thing the wave equation cannot: a
# **conservative** right-hand side across coarse-fine faces. That right-hand
# side has three steps, exactly as CODE.md's "Application interface"
# sketches them --
#
#     scatter! -> fill_ghosts!                     # (0) the state's ghosts
#     map_blocks!(flux_kernel!, F_d; closed=true)  # (i) fluxes, N+1 per dim
#     restrict_interfaces!(F_d, isched[d])         # (ii) the fixup
#     map_blocks!(divergence_kernel!, state, du)   # (iii) the divergence
#
# -- and step (ii) is what makes the two sides of every coarse-fine face
# agree on the area-weighted flux, hence what makes the domain integral
# of `u` constant to roundoff under a global timestep. Without it the
# scheme is still stable and still converges; it just leaks mass at the
# interfaces, which is what the negative control in `burgers_tests.jl`
# measures.
#
# Two field sets with different layouts is the normal case here, and the
# reason `G` moved onto the field set in M8a:
#
#   * the state, cell-centered, `G = 2` -- two ghosts because the flux at
#     a block's own boundary face reconstructs from cells `i-2 … i+1`,
#     and both sides of that face must compute it from the same numbers
#     for a same-level face to conserve;
#   * `D` flux sets, `facecentered(D, d)`, `G = 0` -- a flux is computed
#     over the closed range and never exchanged, so it carries no ghosts
#     at all. The fixup reads and writes closed-range values only, which
#     is what makes `G = 0` legal.
#
# The off-by-`G` sharp edge CODE.md warns about, with two `G`s: face `i`
# (in `1 … N+1`) lies between cells `i-1` and `i`, stored at `i - 1 + G_u`
# and `i + G_u`, while the face itself is stored at `i + G_f`.
#
# Everything is generic in the element type and in the backend, as
# `wave.jl` is. `to_backend` and `convergence_rate` are wave.jl's; both
# files are included by `runtests.jl`, and duplicating them here would be
# a method redefinition rather than a second opinion. `burgers_errors`
# additionally uses `cell_average` from `ghost_oracles.jl`: what a
# conservative scheme stores is a cell average, so the reference it is
# judged against has to be one too.

import KernelAbstractions
using KernelAbstractions: @kernel, @index, @Const, get_backend, CPU
using OrdinaryDiffEqSSPRK: SSPRK33
using SciMLBase: ODEProblem, solve

# --- the scheme ----------------------------------------------------------

# The slope in one cell, from its two one-sided differences `back =
# u_i - u_{i-1}` and `forw = u_{i+1} - u_i`. `:none` is the plain
# centered slope, which keeps the reconstruction linear everywhere and so
# keeps the truncation error a clean `O(h²)` -- what a convergence study
# wants, since a limiter clips at smooth extrema and would hide the
# interface behind its own first-order footprint. `:minmod` is what a
# shock wants.
#
# The limiter travels as a `Val`, so the kernel specializes on it and the
# branch disappears; a `Symbol` argument would be neither `isbits`-safe
# nor constant-folded.
@inline burgers_slope(::Val{:none}, back, forw) = (back + forw) / 2
@inline function burgers_slope(::Val{:minmod}, back, forw)
    z = zero(back)
    back * forw <= z && return z
    return abs(back) < abs(forw) ? back : forw
end

"""
The Rusanov (local Lax-Friedrichs) flux for `f(u) = u²/2`, from the two
reconstructed face states.

    F = ½(f_L + f_R) - ½ max(|u_L|, |u_R|) (u_R - u_L)

Written with integer literals only: a `0.5` here would be an fp64 operand
in the innermost loop and would need hardware fp64 on a device, which is
the leak the package's own "no floating-point literal in per-cell
arithmetic" rule exists to stop.
"""
@inline function rusanov(uL, uR)
    fL = uL * uL / 2
    fR = uR * uR / 2
    a = max(abs(uL), abs(uR))
    return (fL + fR - a * (uR - uL)) / 2
end

# The fluxes normal to dimension `d`, one per face of the closed range.
# Launched over the *flux* set with `closed = true`, so `I[d]` runs over
# `1 … N+1` and the other indices over `1 … N`: a block computes the flux
# on both of its own faces, and making the two sides of a coarse-fine face
# agree afterwards is `restrict_interfaces!`'s job.
@kernel function burgers_flux_kernel!(flux, @Const(work), ::Val{D}, ::Val{GU},
                                      ::Val{GF}, ::Val{d}, lim) where {D,GU,GF,d}
    I = @index(Global, NTuple)                     # (i1..iD, block)
    b = I[D + 1]
    # Face `I[d]` lies between cells `I[d]-1` and `I[d]`; cell `i` of the
    # state is stored at `i + GU[d]`. Transversally a face index *is* a
    # cell index, in both field sets.
    c = ntuple(e -> I[e] + GU[e], Val(D))
    m1 = Base.setindex(c, c[d] - 1, d)
    m2 = Base.setindex(c, c[d] - 2, d)
    p1 = Base.setindex(c, c[d] + 1, d)

    um2 = work[m2..., 1, b]
    um1 = work[m1..., 1, b]
    u0 = work[c..., 1, b]
    up1 = work[p1..., 1, b]

    uL = um1 + burgers_slope(lim, um1 - um2, u0 - um1) / 2
    uR = u0 - burgers_slope(lim, u0 - um1, up1 - u0) / 2

    flux[ntuple(e -> I[e] + GF[e], Val(D))..., 1, b] = rusanov(uL, uR)
end

# `du = -Σ_d (F_d[i+1] - F_d[i]) / h`, over the state's owned cells and
# written straight into the state layout.
#
# `fluxes` is an `NTuple{D}` of identically typed arrays, so indexing it
# with the loop variable is type stable; all `D` flux sets share one `GF`,
# which is what lets one stored index serve every direction.
@kernel function burgers_divergence_kernel!(du, fluxes, @Const(spacings),
                                            ::Val{D}, ::Val{GF}) where {D,GF}
    I = @index(Global, NTuple)                     # (i1..iD, block)
    b = I[D + 1]
    c = ntuple(e -> I[e] + GF[e], Val(D))
    acc = zero(eltype(du))
    for d in 1:D
        hi = Base.setindex(c, c[d] + 1, d)
        acc += fluxes[d][hi..., 1, b] - fluxes[d][c..., 1, b]
    end
    du[ntuple(e -> I[e], Val(D))..., 1, b] = -acc / spacings[b]
end

"""
Everything a Burgers right-hand side needs, built once per mesh: the
state set and its ghost schedule, the `D` flux sets and their interface
schedules, and the per-block spacings on whatever backend the data lives
on.

`fixup = false` skips [`restrict_interfaces!`](@ref) and nothing else --
the negative control for the conservation test, and the only difference
between a scheme that conserves to roundoff and one that does not.

After a [`regrid!`](@ref) this is rebuilt rather than mutated, since both
kinds of schedule and the per-block spacings are derived from the leaf
array. Pass the existing `fluxes` when doing so: `regrid!` resized them
in place (`fs => nothing`), so they are already the right shape for the
new mesh, and reallocating them would throw that away.
"""
struct BurgersProblem{T,D,GU,GF,LIM,FS,FL,S,IS,V}
    state::FS
    fluxes::FL                   # NTuple{D,FieldSet}, facecentered(D, d)
    schedule::S
    ischeds::IS                  # NTuple{D,InterfaceSchedule}
    spacings::V                  # per block, wherever the kernels run
    fixup::Bool
    valD::Val{D}
    valGU::Val{GU}
    valGF::Val{GF}
    limiter::Val{LIM}
end

function BurgersProblem(state::FieldSet{T,D}, ops::Operators;
                        Gf::Integer=0, limiter::Symbol=:none,
                        fixup::Bool=true, fluxes=nothing) where {T,D}
    forest = state.forest
    backend = get_backend(state.work)
    all(==(:cell), state.centering) || throw(ArgumentError(
        "the Burgers state is cell-centered; got $(state.centering)"))
    all(>=(2), state.G) || throw(ArgumentError(
        "the flux reconstruction reads cells i-2 … i+1, so the state needs G >= 2 " *
        "in every dimension for the two sides of a same-level face to compute the " *
        "same number; got G=$(state.G)"))
    limiter === :none || limiter === :minmod || throw(ArgumentError(
        "limiter must be :none or :minmod, got :$limiter"))

    fluxes = fluxes === nothing ?
             ntuple(d -> FieldSet{T}(forest, 1; G=Gf, centering=facecentered(D, d),
                                     backend=backend), D) : fluxes
    schedule = GhostSchedule(state, ops)
    ischeds = ntuple(d -> InterfaceSchedule(fluxes[d]), D)
    spacings = to_backend(backend, block_spacings(forest, T))
    GU = state.G
    GF = first(fluxes).G
    return BurgersProblem{T,D,GU,GF,limiter,typeof(state),typeof(fluxes),
                          typeof(schedule),typeof(ischeds),typeof(spacings)}(
        state, fluxes, schedule, ischeds, spacings, fixup,
        Val(D), Val(GU), Val(GF), Val(limiter))
end

"""
The three-step conservative right-hand side, written out by the
application. There is no `semidiscretize`-style wrapper, here as
everywhere.

The `ntuple` over `Val(D)` is not decoration: it unrolls the direction
loop so that each launch gets a *constant* `Val(d)`, which is what the
flux kernel specializes its face dimension on.
"""
function burgers_rhs!(du, u, p::BurgersProblem{T,D}, t) where {T,D}
    scatter!(p.state, u)
    fill_ghosts!(p.state, p.schedule)
    ntuple(Val(D)) do d
        map_blocks!(burgers_flux_kernel!, p.fluxes[d], p.fluxes[d].work,
                    p.state.work, p.valD, p.valGU, p.valGF, Val(d), p.limiter;
                    closed=true)
        p.fixup && restrict_interfaces!(p.fluxes[d], p.ischeds[d])
        nothing
    end
    map_blocks!(burgers_divergence_kernel!, p.state, statearray(du, p.state),
                map(f -> f.work, p.fluxes), p.spacings, p.valD, p.valGF)
    return nothing
end

# --- the exact solution --------------------------------------------------

"""Breaking time of `ū + a sin(2πs/L)` in `D` dimensions."""
burgers_breaktime(D, L::T, a::T) where {T} = L / (2 * T(π) * a * D)

"""
The exact solution, pointwise.

Data depending only on `s = Σ_d x_d` reduce Burgers' equation to the
one-dimensional one in `s` with the speed multiplied by `D`:
`∂_d(u²/2) = u ∂_d u = u u'(s)`, summed over `d`, gives `∂ₜu + D u ∂_s u = 0`.
So the characteristic through `s` carries `u` at speed `D u`, and

    u = u₀(s - D u t),   u₀(s) = ū + a sin(2πs/L)

implicitly, until the characteristics first cross at
`t_b = L/(2πaD)` ([`burgers_breaktime`](@ref)). Newton from `u₀(s)`
converges in a handful of iterations for `t` comfortably below that.
"""
function burgers_exact(D, L::T, ubar::T, a::T, t::T) where {T}
    k = 2 * T(π) / L
    return function (s)
        u = ubar + a * sin(k * s)
        for _ in 1:50
            arg = k * (s - D * u * t)
            f = u - ubar - a * sin(arg)
            df = 1 + a * k * D * t * cos(arg)
            step = f / df
            u -= step
            abs(step) <= 4 * eps(T) * (abs(u) + eps(T)) && break
        end
        return u
    end
end

"""
Fill every owned cell with the **exact cell average** of the initial
sine — what a finite-volume scheme's stored numbers mean.

The average of `ū + a sin(k Σ_d x_d)` over a cube of side `h` is
analytic, `ū + a·((2/kh)·sin(kh/2))^D·sin(k Σ_d x_d)`, so no quadrature
is needed. But the damping factor depends on the *block's* own `h`, which
an `(x, v)` callback cannot see, so this is a host loop with one
`copyto!` rather than [`fill_by_coordinates!`](@ref) — which is also why
the adaptive drivers below, whose meshes change under them, initialize
from point samples instead ([`burgers_pointwise`](@ref)); they measure
conservation and shock tracking, where an `O(h²)` difference in the
initial data is beside the point.
"""
function fill_burgers_averages!(fs::FieldSet{T,D}, L::T, ubar::T, a::T) where {T,D}
    forest = fs.forest
    k = 2 * T(π) / L
    host = zeros(T, size(fs.work))
    for b in 1:nblocks(fs)
        h = spacing(T, forest, blockkey(fs, b))
        damp = (2 * sin(k * h / 2) / (k * h))^D
        for idx in CartesianIndices(ntuple(d -> (fs.G[d] + 1):(fs.G[d] + forest.N), D))
            s = sum(coordinates(T, fs, b, Tuple(idx)))
            host[Tuple(idx)..., 1, b] = ubar + a * damp * sin(k * s)
        end
    end
    copyto!(fs.work, host)
    return fs
end

"""
The initial sine as an `(x, v) -> value` callback — point samples, for
the adaptive drivers, which re-evaluate the initial data on every mesh
the initialization cycle produces.
"""
function burgers_pointwise(L::T, ubar::T, a::T) where {T}
    k = 2 * T(π) / L
    return (x, v) -> ubar + a * sin(k * sum(x))
end

# --- meshes and drivers --------------------------------------------------

"""
The M3 two-level hierarchy, as `wave_forest` builds it and for the same
reason: a `roots^D` periodic box with the middle sub-box refined once,
held fixed in physical space as `N` varies, so a convergence study really
does just shrink `h`. With `refined = false` the box is left uniform, as
a control.
"""
function burgers_forest(::Val{D}, N; roots=4, L=1.0, refined=true,
                        T::Type=Float64) where {D}
    L = T(L)
    forest = Forest(ntuple(_ -> roots, D); N=N, periodic=ntuple(_ -> true, D),
                    extents=ntuple(_ -> (zero(T), L), D))
    refined || return forest
    quarter, threequarters = L / 4, 3 * L / 4
    targets = filter(forest.leaves) do k
        ext = block_extent(forest, k)
        all(d -> quarter < (ext[d][1] + ext[d][2]) / 2 < threequarters, 1:D)
    end
    refine!(forest, targets)
    balance!(forest)
    return forest
end

# One fixed-step SSPRK33 solve of `nsteps` steps. SSPRK rather than RK4
# because a shock wants a strong-stability-preserving method; conservation
# holds for any Runge-Kutta scheme, since every stage's `du` already sums
# to zero.
function burgers_solve!(p::BurgersProblem{T}, u, t0::T, t1::T, nsteps::Int) where {T}
    prob = ODEProblem(burgers_rhs!, u, (t0, t1), p)
    sol = solve(prob, SSPRK33(); dt=(t1 - t0) / nsteps, adaptive=false,
                save_everystep=false)
    return sol.u[end]
end

# `dt` from the finest spacing and the fastest characteristic. The
# stability limit is `h / (D·max|u|)` — the characteristic speed is `u` in
# every dimension and an explicit dimensionally-split-in-space scheme adds
# them up — and the second factor of `D` here is plain headroom. `umax` is
# the *initial* maximum, which stays a bound: Burgers obeys a maximum
# principle. One global step for the whole hierarchy — there is no
# subcycling, ever.
burgers_dt(forest, cfl::T, umax::T, D) where {T} =
    cfl * minimum_spacing(T, forest) / (D * D * umax)

"""
Evolve the smooth sine to a fraction of the breaking time on the static
two-level mesh and return the volume-weighted L1 and L∞ errors against
the exact cell averages, plus the finest spacing.

This is the conservative family's half of the interface-order rule: the
scheme is second order, the fixup makes it conservative, and what is left
to limit the global rate is the order of the prolongation that fills the
state's ghosts at the coarse-fine interface.
"""
function burgers_errors(::Val{D}; N, ops, G=2, roots=4, L=1.0, ubar=1.0, amp=0.5,
                        cfl=0.4, fraction=0.5, refined=true, limiter=:none,
                        fixup=true, T::Type=Float64, backend=CPU()) where {D}
    L, ubar, amp = T(L), T(ubar), T(amp)
    forest = burgers_forest(Val(D), N; roots=roots, L=L, refined=refined, T=T)
    state = FieldSet{T}(forest, 1; G=G, backend=backend)
    p = BurgersProblem(state, ops; limiter=limiter, fixup=fixup)

    fill_burgers_averages!(state, L, ubar, amp)
    u = statevector(state)
    gather!(u, state)
    mass0 = total_mass(state)
    scale = burgers_mass_scale(state)

    t_end = T(fraction) * burgers_breaktime(D, L, amp)
    dt = burgers_dt(forest, T(cfl), ubar + amp, D)
    nsteps = ceil(Int, t_end / dt)
    u = burgers_solve!(p, u, zero(T), t_end, nsteps)
    scatter!(state, u)

    err = u .- burgers_reference(state, L, ubar, amp, t_end)
    return (l1=volume_weighted_norm(state, err; p=1),
            linf=volume_weighted_norm(state, err; p=Inf),
            drift=abs(total_mass(state) - mass0), scale=scale,
            h=minimum_spacing(T, forest), nsteps=nsteps, nblocks=nleaves(forest))
end

"""
The exact solution as a state vector of **cell averages**, through the
Gauss-Legendre oracle of `ghost_oracles.jl` — five points per dimension,
exact through degree 9, which is well past anything the scheme resolves.

Deliberately not the analytic average used for the initial data: after
`t = 0` the solution is implicit, so there is nothing to integrate in
closed form, and quadrature over the pointwise Newton solve is the
independent reference.
"""
function burgers_reference(fs::FieldSet{T,D}, L::T, ubar::T, a::T, t::T) where {T,D}
    forest = fs.forest
    exact = burgers_exact(D, L, ubar, a, t)
    N = forest.N
    host = zeros(T, statelength(fs))
    arr = reshape(host, ntuple(_ -> N, D)..., fs.nvars, nblocks(fs))
    for b in 1:nblocks(fs)
        h = spacing(T, forest, blockkey(fs, b))
        for idx in CartesianIndices(ntuple(_ -> N, D))
            x = coordinates(T, fs, b, ntuple(d -> Tuple(idx)[d] + fs.G[d], D))
            arr[Tuple(idx)..., 1, b] = cell_average(p -> exact(sum(p)), x, h)
        end
    end
    u = statevector(fs)
    copyto!(u, host)
    return u
end

"""
`u` reduced onto a uniform `M^D` grid of side `L` by exact volume
averaging — the common ground two runs at different resolutions can be
compared on at all. Every cell of every mesh here is a whole subdivision
of one target cell, provided `M` divides the coarsest resolution, which
the callers arrange.

An oracle in the same spirit as the others in this suite: it knows
positions and spacings and nothing about how the data got there.
"""
function reduce_to_grid(fs::FieldSet{T,D}, M::Int, L::T) where {T,D}
    forest = fs.forest
    H = L / M
    out = zeros(float(T), ntuple(_ -> M, D))
    work = Array(fs.work)
    for b in 1:nblocks(fs)
        h = spacing(T, forest, blockkey(fs, b))
        h <= H || error("reduce_to_grid needs cells no coarser than L/M")
        w = (h / H)^D
        for idx in CartesianIndices(ntuple(d -> (fs.G[d] + 1):(fs.G[d] + forest.N), D))
            x = coordinates(T, fs, b, Tuple(idx))
            cell = ntuple(d -> clamp(floor(Int, x[d] / H) + 1, 1, M), D)
            out[cell...] += w * work[Tuple(idx)..., 1, b]
        end
    end
    return out
end

"""Mean absolute difference of two reductions onto the same grid."""
l1_difference(a, b) = sum(abs, a .- b) / length(a)

# The refinement criterion: the undivided central difference of `u`,
# which is the shock indicator, evaluated through `firing_boxes` so the
# per-cell sweep and the min/max reduction run on the field set's own
# backend. What stays on the host is the verdict, which is the physics.
function burgers_fires(::Val{D}, thr::T) where {D,T}
    return function (work, idx, b, x)
        m = zero(T)
        for d in 1:D
            hi = Base.setindex(idx, idx[d] + 1, d)
            lo = Base.setindex(idx, idx[d] - 1, d)
            m = max(m, abs(work[hi..., 1, b] - work[lo..., 1, b]))
        end
        return m > thr
    end
end

"""
Evolve a sine that steepens into a shock, regridding every `chunk` of
time so the refined region follows it, and return what the conservation
claim is made of: the drift of the domain integral, the scale that drift
is measured against, and how well the refinement tracked the shock.

`fixup = false` is the negative control -- the same run with step (ii)
skipped. Everything else about it is identical, which is the point.

Regridding changes both the length and the meaning of the state vector,
so each chunk is a fresh `solve` and a fresh [`BurgersProblem`](@ref):
the flux sets are resized (`fs => nothing`, since the next right-hand
side overwrites them anyway) and both kinds of schedule are rebuilt.
"""
function track_shock(::Val{D}; N=8, roots=8, G=2, L=1.0, ubar=1.0, amp=0.5,
                     grid=roots * N,
                     ops=Operators(family=Conservative, prolongation=3,
                                   restriction=2),
                     t_end=0.6, chunk=0.1, cfl=0.4, maxlevel_wanted=1,
                     threshold=0.15, buffer=4, limiter=:minmod, fixup=true,
                     T::Type=Float64, backend=CPU()) where {D}
    L, ubar, amp = T(L), T(ubar), T(amp)
    t_end, chunk, cfl = T(t_end), T(chunk), T(cfl)
    forest = Forest(ntuple(_ -> roots, D); N=N, periodic=ntuple(_ -> true, D),
                    extents=ntuple(_ -> (zero(T), L), D))
    state = FieldSet{T}(forest, 1; G=G, backend=backend)

    fires = burgers_fires(Val(D), T(threshold))
    function flags_now(fs)
        boxes = firing_boxes(fires, fs)
        return map(1:nleaves(forest)) do b
            n, box = boxes[b]
            k = forest.leaves[b]
            n == 0 && return level(k) > 0 ? Coarsen : Keep
            return level(k) >= maxlevel_wanted ? (Keep, box) : (Refine, box)
        end
    end

    initial = burgers_pointwise(L, ubar, amp)
    fill_by_coordinates!(initial, state)
    adapt_to_initial_data!(state, ops; initial=initial, flags=flags_now,
                           buffer=buffer, maxpasses=8)

    p = BurgersProblem(state, ops; limiter=limiter, fixup=fixup)
    mass0 = total_mass(state)
    scale = burgers_mass_scale(state)
    drift = zero(float(T))
    nsteps = 0
    refined_fraction = float(T)[]
    t = zero(T)
    while t < t_end - 1000 * eps(t_end)
        stop = min(t + chunk, t_end)
        u = statevector(state)
        gather!(u, state)
        dt = burgers_dt(forest, cfl, ubar + amp, D)
        steps = max(1, ceil(Int, (stop - t) / dt))
        u = burgers_solve!(p, u, t, stop, steps)
        scatter!(state, u)
        nsteps += steps
        t = stop

        # The claim is about the whole run, not its endpoint: a leak that
        # reversed sign would otherwise hide.
        drift = max(drift, abs(total_mass(state) - mass0))

        fill_ghosts!(state, p.schedule)
        # Only once the criterion has produced a refined region at all:
        # the sine starts smooth enough that nothing fires, and a chunk
        # spent legitimately on a uniform mesh says nothing about whether
        # the mesh follows the shock.
        maxlevel(forest) > 0 && push!(refined_fraction, refined_share(state, fires))
        pairs = (state => p.schedule, ntuple(d -> p.fluxes[d] => nothing, D)...)
        if regrid!(forest, pairs; flags=flags_now(state), buffer=buffer)
            p = BurgersProblem(state, ops; limiter=limiter, fixup=fixup,
                               fluxes=p.fluxes)
        end
    end

    return (drift=drift, mass0=mass0, scale=scale, nsteps=nsteps,
            tracking=isempty(refined_fraction) ? zero(float(T)) :
                     minimum(refined_fraction),
            nblocks=nleaves(forest),
            maxlevel=maxlevel(forest), reduced=reduce_to_grid(state, grid, L),
            cells=nleaves(forest) * N^D, state=state, forest=forest)
end

"""
`Σ hᴰ |u|` over the domain — the scale a mass *drift* is roundoff against,
since `Σ hᴰ u` itself may be small through cancellation.
"""
function burgers_mass_scale(fs::FieldSet{T,D}) where {T,D}
    R = float(real(T))
    partials = block_mapreduce(abs, +, zero(R), fs; vars=1)
    for b in 1:nblocks(fs)
        partials[b] *= spacing(R, fs.forest, blockkey(fs, b))^D
    end
    return sum(partials)
end

# What fraction of the firing cells sit on blocks at the target level --
# the measure of whether the refined region is actually following the
# shock. Per block on the field set's backend, combined in block order.
function refined_share(fs::FieldSet{T,D}, fires) where {T,D}
    boxes = firing_boxes(fires, fs)
    total = 0
    inside = 0
    for b in 1:nblocks(fs)
        n = boxes[b][1]
        total += n
        level(blockkey(fs, b)) > 0 && (inside += n)
    end
    return total == 0 ? one(float(T)) : float(T)(inside) / total
end

"""
The same shock on a **uniform** mesh, as the reference an adaptive run is
judged against, and as the coarse control that says the refinement bought
anything. The final state comes back both as the field set and reduced
onto the `grid^D` comparison mesh (see [`reduce_to_grid`](@ref)).
"""
function uniform_shock(::Val{D}; roots, N, grid, G=2, L=1.0, ubar=1.0, amp=0.5,
                       ops=Operators(family=Conservative, prolongation=3,
                                     restriction=2),
                       t_end=0.6, cfl=0.4, limiter=:minmod, fixup=true,
                       T::Type=Float64, backend=CPU()) where {D}
    L, ubar, amp = T(L), T(ubar), T(amp)
    t_end, cfl = T(t_end), T(cfl)
    forest = Forest(ntuple(_ -> roots, D); N=N, periodic=ntuple(_ -> true, D),
                    extents=ntuple(_ -> (zero(T), L), D))
    state = FieldSet{T}(forest, 1; G=G, backend=backend)
    p = BurgersProblem(state, ops; limiter=limiter, fixup=fixup)
    # Point samples, as `track_shock` uses, so the two runs differ in
    # their meshes and in nothing else.
    fill_by_coordinates!(burgers_pointwise(L, ubar, amp), state)
    u = statevector(state)
    gather!(u, state)
    dt = burgers_dt(forest, cfl, ubar + amp, D)
    nsteps = ceil(Int, t_end / dt)
    u = burgers_solve!(p, u, zero(T), t_end, nsteps)
    scatter!(state, u)
    return (reduced=reduce_to_grid(state, grid, L), cells=nleaves(forest) * N^D,
            mass=total_mass(state), state=state)
end
