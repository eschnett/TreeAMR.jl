# The workload behind the M5 thread-count independence test.
#
# Run as a standalone script — `julia -t N --project=test
# test/thread_workload.jl` — it prints a digest of everything a full
# TreeAMR cycle produces: the state vector bit for bit, the leaf array,
# the schedule's shape, and the diagnostic reductions. Two runs at
# different thread counts must print the same lines, character for
# character. For everything but the `l2` and `mass` lines that is what
# `CODE.md` promises — every parallel loop in the package writes to its
# own slot. Those two are floating-point sums, promised to roundoff only
# since M8; they are exact today because the CPU fold combines its
# per-block partials in block order, and they are the lines to relax if
# that ever changes. See `thread_tests.jl`.
#
# It is a script rather than a testset because the thread count is a
# command-line argument to Julia and cannot be changed from inside a
# running session. Deliberately self-contained — no ODE package, just a
# hand-written RK4 for the wave cycles and an SSPRK3 for the conservative
# one — so that a subprocess starts in a couple of seconds.

using TreeAMR
using KernelAbstractions: @kernel, @index, @Const
using Printf: @sprintf
using SHA: sha256

@kernel function workload_rhs_kernel!(du, @Const(work), @Const(spacings),
                                      ::Val{D}, ::Val{G}) where {D,G}
    I = @index(Global, NTuple)
    b = I[D + 1]
    c = ntuple(d -> I[d] + G[d], Val(D))
    u0 = work[c..., 1, b]
    laplacian = zero(eltype(du))
    for d in 1:D
        up = Base.setindex(c, c[d] + 1, d)
        um = Base.setindex(c, c[d] - 1, d)
        laplacian += work[up..., 1, b] - 2 * u0 + work[um..., 1, b]
    end
    h = spacings[b]
    du[ntuple(d -> I[d], Val(D))..., 1, b] = work[c..., 2, b]
    du[ntuple(d -> I[d], Val(D))..., 2, b] = laplacian / (h * h)
end

function rhs!(du, u, fs, schedule, spacings, ::Val{D}, ::Val{G}, boundary) where {D,G}
    scatter!(fs, u)
    fill_ghosts!(fs, schedule; boundary=boundary)
    map_blocks!(workload_rhs_kernel!, fs, statearray(du, fs), fs.work, spacings,
                Val(D), Val(G))
    return du
end

# Plain fixed-step RK4, so the workload needs no ODE package.
function rk4!(u, fs, schedule, dt, nsteps, ::Val{D}, ::Val{G}, boundary) where {D,G}
    spacings = block_spacings(fs.forest)
    k1, k2, k3, k4, tmp = (similar(u) for _ in 1:5)
    for _ in 1:nsteps
        rhs!(k1, u, fs, schedule, spacings, Val(D), Val(G), boundary)
        @. tmp = u + (dt / 2) * k1
        rhs!(k2, tmp, fs, schedule, spacings, Val(D), Val(G), boundary)
        @. tmp = u + (dt / 2) * k2
        rhs!(k3, tmp, fs, schedule, spacings, Val(D), Val(G), boundary)
        @. tmp = u + dt * k3
        rhs!(k4, tmp, fs, schedule, spacings, Val(D), Val(G), boundary)
        @. u += (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    end
    return u
end

"""A Gaussian pulse travelling in +x, as an `(x, v) -> value` callback."""
function workload_pulse(D, L, x0, σ, t)
    return function (x, var)
        d = mod(x[1] - x0 - t + L / 2, L) - L / 2
        g = exp(-d^2 / (2σ^2))
        return var == 1 ? g : (d / σ^2) * g
    end
end

digest(bytes::AbstractVector{UInt8}) = bytes2hex(sha256(bytes))[1:32]
digest(u::Vector{Float64}) = digest(reinterpret(UInt8, u))
digest(s::AbstractString) = digest(codeunits(s))

"""
One full cycle — initial-data adaptation, evolution, a regrid with data
transfer, more evolution — reduced to a handful of printed lines.
"""
function workload(::Val{D}; roots, N, G, ops, periodic, σ, steps, buffer,
                  centering=cellcentered(D)) where {D}
    L = 1.0
    x0 = 0.35
    forest = Forest(ntuple(_ -> roots, D); N=N,
                    periodic=ntuple(_ -> periodic, D),
                    extents=ntuple(_ -> (0.0, L), D))
    fs = FieldSet(forest, 2; G=G, centering=centering)
    initial = workload_pulse(D, L, x0, σ, 0.0)
    boundary = periodic ? nothing : boundary_by_coordinates(initial)

    # Flags that report a box, so the buffer dilation runs too.
    function flag(b, k)
        fired = findall(x -> abs(x) > 0.05, interiorview(fs, b, 1))
        isempty(fired) && return level(k) > 0 ? Coarsen : Keep
        box = ntuple(d -> minimum(i -> i[d], fired):maximum(i -> i[d], fired), D)
        return level(k) >= 1 ? (Keep, box) : (Refine, box)
    end

    fill_by_coordinates!(initial, fs)
    schedule, passes, converged =
        adapt_to_initial_data!(fs, ops; initial=initial, flag=flag, buffer=buffer,
                               maxpasses=6, boundary=boundary)

    u = statevector(fs)
    gather!(u, fs)
    dt = 0.25 * minimum_spacing(forest)
    rk4!(u, fs, schedule, dt, steps, Val(D), Val(fs.G), boundary)

    # A regrid that actually moves data, then more evolution on the new mesh.
    scatter!(fs, u)
    fill_ghosts!(fs, schedule; boundary=boundary)
    changed = regrid!(forest, fs => schedule; flags=flag_blocks(flag, forest),
                      buffer=buffer, boundary=boundary)
    changed && (schedule = GhostSchedule(fs, ops))
    u = statevector(fs)
    gather!(u, fs)
    rk4!(u, fs, schedule, dt, steps, Val(D), Val(fs.G), boundary)

    tag = "D$(D)$(periodic ? "p" : "o")$(all(==(:cell), centering) ? "c" : "v")"
    println(tag, " passes ", passes, " ", converged, " changed ", changed)
    println(tag, " leaves ", nleaves(forest), " ", digest(string(forest.leaves)))
    println(tag, " schedule ", length(schedule.phase1), " ", schedule.levels, " ",
            length(schedule.boundaries))
    println(tag, " state ", digest(u))
    println(tag, " l2 ", @sprintf("%.17g", volume_weighted_norm(fs, u)))
    println(tag, " linf ", @sprintf("%.17g", volume_weighted_norm(fs, u; p=Inf)))
    println(tag, " mass ", @sprintf("%.17g", total_mass(fs, 1)))
    return nothing
end

const OPS4 = Operators(prolongation=4, restriction=4)
const OPS2 = Operators(prolongation=2, restriction=2)

workload(Val(1); roots=8, N=8, G=2, ops=OPS4, periodic=true, σ=0.04, steps=24, buffer=3)
workload(Val(2); roots=4, N=8, G=2, ops=OPS4, periodic=true, σ=0.05, steps=20, buffer=3)
workload(Val(2); roots=4, N=8, G=2, ops=OPS4, periodic=false, σ=0.05, steps=20, buffer=3)
workload(Val(3); roots=4, N=4, G=1, ops=OPS2, periodic=true, σ=0.03, steps=6, buffer=1)

# The same cycle on a staggered layout (M8a). It is not a rerun with a
# keyword changed: a vertex-like dimension adds the shared boundary
# plane to every exchange region, replaces the restriction stencils with
# injection, and narrows the prolongation window, so the parallel loops
# `run_phase!` deals out are differently shaped ones. `G = 1` is what
# order 4 needs along a stagger, which also keeps these cheap.
workload(Val(1); roots=8, N=8, G=1, ops=OPS4, periodic=true, σ=0.04, steps=24, buffer=3,
         centering=vertexcentered(1))
workload(Val(2); roots=4, N=8, G=1, ops=OPS4, periodic=false, σ=0.05, steps=12, buffer=3,
         centering=vertexcentered(2))

# --- the conservative cycle (M8b) ----------------------------------------
#
# Burgers' equation, which brings in everything the wave cycle above does
# not: a second field set per dimension with a different centering and no
# ghosts at all, a kernel launched over the *closed* range, and
# `restrict_interfaces!` — whose phases are dealt out to threads by the
# same `run_phase!` that fills ghosts, over target ranges one plane thick.
# If the interface fixup's slices were ever combined out of order, or if
# its schedule's neighbour walk merged its per-task lists by anything but
# block order, the mass line below would move with the thread count.
#
# Self-contained like the rest of this file: its own SSPRK3, no ODE
# package. `test/burgers.jl` is the readable version of the same scheme.

@kernel function workload_flux_kernel!(flux, @Const(work), ::Val{D}, ::Val{GU},
                                       ::Val{GF}, ::Val{d}) where {D,GU,GF,d}
    I = @index(Global, NTuple)                   # I[d] runs over 1:N+1
    b = I[D + 1]
    c = ntuple(e -> I[e] + GU[e], Val(D))
    m1 = Base.setindex(c, c[d] - 1, d)
    m2 = Base.setindex(c, c[d] - 2, d)
    p1 = Base.setindex(c, c[d] + 1, d)
    um2 = work[m2..., 1, b]
    um1 = work[m1..., 1, b]
    u0 = work[c..., 1, b]
    up1 = work[p1..., 1, b]
    uL = um1 + minmod(um1 - um2, u0 - um1) / 2
    uR = u0 - minmod(u0 - um1, up1 - u0) / 2
    a = max(abs(uL), abs(uR))
    flux[ntuple(e -> I[e] + GF[e], Val(D))..., 1, b] =
        (uL * uL / 2 + uR * uR / 2 - a * (uR - uL)) / 2
end

@inline function minmod(back, forw)
    z = zero(back)
    back * forw <= z && return z
    return abs(back) < abs(forw) ? back : forw
end

@kernel function workload_divergence_kernel!(du, fluxes, @Const(spacings),
                                             ::Val{D}, ::Val{GF}) where {D,GF}
    I = @index(Global, NTuple)
    b = I[D + 1]
    c = ntuple(e -> I[e] + GF[e], Val(D))
    acc = zero(eltype(du))
    for d in 1:D
        hi = Base.setindex(c, c[d] + 1, d)
        acc += fluxes[d][hi..., 1, b] - fluxes[d][c..., 1, b]
    end
    du[ntuple(e -> I[e], Val(D))..., 1, b] = -acc / spacings[b]
end

# The three steps: ghosts, fluxes with the fixup, divergence.
function burgers_rhs!(du, u, state, fluxes, schedule, ischeds, spacings,
                      ::Val{D}, ::Val{GU}, ::Val{GF}) where {D,GU,GF}
    scatter!(state, u)
    fill_ghosts!(state, schedule)
    ntuple(Val(D)) do d
        map_blocks!(workload_flux_kernel!, fluxes[d], fluxes[d].work, state.work,
                    Val(D), Val(GU), Val(GF), Val(d); closed=true)
        restrict_interfaces!(fluxes[d], ischeds[d])
        nothing
    end
    map_blocks!(workload_divergence_kernel!, state, statearray(du, state),
                map(f -> f.work, fluxes), spacings, Val(D), Val(GF))
    return du
end

# Plain fixed-step SSPRK33, so the workload needs no ODE package.
function ssprk33!(u, state, fluxes, schedule, ischeds, dt, nsteps,
                  ::Val{D}, ::Val{GU}, ::Val{GF}) where {D,GU,GF}
    spacings = block_spacings(state.forest)
    k, u1, u2 = (similar(u) for _ in 1:3)
    args = (state, fluxes, schedule, ischeds, spacings, Val(D), Val(GU), Val(GF))
    for _ in 1:nsteps
        burgers_rhs!(k, u, args...)
        @. u1 = u + dt * k
        burgers_rhs!(k, u1, args...)
        @. u2 = (3 * u + u1 + dt * k) / 4
        burgers_rhs!(k, u2, args...)
        @. u = (u + 2 * (u2 + dt * k)) / 3
    end
    return u
end

"""
An adapt / evolve / regrid / evolve cycle on the conservative scheme,
reduced to printed lines as the wave cycle above is. The mass line is the
one that tests the fixup: it is conserved to roundoff, so a thread
count that changed any of it would show up as a changed digit.
"""
function burgers_workload(::Val{D}; roots, N, G, ops, steps, buffer,
                          threshold, width) where {D}
    L = 1.0
    forest = Forest(ntuple(_ -> roots, D); N=N, periodic=ntuple(_ -> true, D),
                    extents=ntuple(_ -> (0.0, L), D))
    state = FieldSet(forest, 1; G=G)
    # A periodic profile that is *already* steep, so the criterion fires
    # from the first pass: a plain sine would have to be evolved past its
    # breaking time before anything refined, and a workload with no
    # coarse-fine face would exercise the fixup on an empty schedule.
    initial = (x, v) -> 1.0 + 0.5 * tanh(sin(2 * pi * sum(x) / L) / width)

    function fires(work, idx, b, x)
        m = 0.0
        for d in 1:D
            hi = Base.setindex(idx, idx[d] + 1, d)
            lo = Base.setindex(idx, idx[d] - 1, d)
            m = max(m, abs(work[hi..., 1, b] - work[lo..., 1, b]))
        end
        return m > threshold
    end
    flags_now(fs) = map(enumerate(firing_boxes(fires, fs))) do (b, (n, box))
        k = forest.leaves[b]
        n == 0 && return level(k) > 0 ? Coarsen : Keep
        return level(k) >= 1 ? (Keep, box) : (Refine, box)
    end

    fill_by_coordinates!(initial, state)
    schedule, passes, converged =
        adapt_to_initial_data!(state, ops; initial=initial, flags=flags_now,
                               buffer=buffer, maxpasses=6)
    fluxes = ntuple(d -> FieldSet(forest, 1; G=0, centering=facecentered(D, d)), D)
    ischeds = ntuple(d -> InterfaceSchedule(fluxes[d]), D)

    u = statevector(state)
    gather!(u, state)
    dt = 0.2 * minimum_spacing(forest) / (D * D * 1.5)
    valG = (Val(D), Val(state.G), Val(first(fluxes).G))
    ssprk33!(u, state, fluxes, schedule, ischeds, dt, steps, valG...)

    scatter!(state, u)
    fill_ghosts!(state, schedule)
    pairs = (state => schedule, ntuple(d -> fluxes[d] => nothing, D)...)
    changed = regrid!(forest, pairs; flags=flags_now(state), buffer=buffer)
    if changed
        schedule = GhostSchedule(state, ops)
        ischeds = ntuple(d -> InterfaceSchedule(fluxes[d]), D)
    end
    u = statevector(state)
    gather!(u, state)
    ssprk33!(u, state, fluxes, schedule, ischeds, dt, steps, valG...)

    tag = "B$D"
    println(tag, " passes ", passes, " ", converged, " changed ", changed)
    println(tag, " leaves ", nleaves(forest), " ", digest(string(forest.leaves)))
    println(tag, " ischeds ", join((string(s) for s in ischeds), " | "))
    println(tag, " state ", digest(u))
    println(tag, " l2 ", @sprintf("%.17g", volume_weighted_norm(state, u)))
    println(tag, " linf ", @sprintf("%.17g", volume_weighted_norm(state, u; p=Inf)))
    println(tag, " mass ", @sprintf("%.17g", total_mass(state, 1)))
    return nothing
end

const OPSC = Operators(family=Conservative, prolongation=3, restriction=2)

burgers_workload(Val(1); roots=8, N=8, G=2, ops=OPSC, steps=40, buffer=4,
                 threshold=0.15, width=0.2)
burgers_workload(Val(2); roots=4, N=8, G=2, ops=OPSC, steps=24, buffer=1,
                 threshold=0.6, width=0.08)
