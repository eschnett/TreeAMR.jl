# The scalar wave equation in 2nd-order form, as an application of the
# mesh. This lives in the tests, not in the package: TreeAMR supplies the
# mesh and its operations, never physics.
#
#     ∂ₜu = v
#     ∂ₜv = ∇²u                    (wave speed c = 1)
#
# On a periodic box of side L the standing sine mode
#
#     u(x,t) = cos(ωt) ∏ sin(2πm x_d / L),   ω = 2πm √D / L
#
# is an exact solution, which is what the convergence test measures
# against.
#
# Everything here is generic in the element type and in the backend, so
# that the same study runs on a device (M6): the mesh is generic in its
# float type, and the only thing a backend changes is where the storage
# is allocated. `T` and `backend` travel together through every entry
# point; the defaults are `Float64` on the CPU, which is what the M3
# convergence tests measure and what their recorded rates were taken
# with.

import KernelAbstractions
using KernelAbstractions: @kernel, @index, @Const, get_backend, CPU
using OrdinaryDiffEqLowOrderRK: RK4
using SciMLBase: ODEProblem, solve

@kernel function wave_rhs_kernel!(du, @Const(work), @Const(spacings),
                                  ::Val{D}, ::Val{G}) where {D,G}
    I = @index(Global, NTuple)                 # (i1..iD, block)
    b = I[D + 1]
    inner = ntuple(d -> I[d], Val(D))          # state-layout index
    c = ntuple(d -> I[d] + G[d], Val(D))       # working-array index

    u0 = work[c..., 1, b]
    laplacian = zero(eltype(du))
    for d in 1:D
        up = Base.setindex(c, c[d] + 1, d)
        um = Base.setindex(c, c[d] - 1, d)
        laplacian += work[up..., 1, b] - 2 * u0 + work[um..., 1, b]
    end
    h = spacings[b]

    du[inner..., 1, b] = work[c..., 2, b]
    du[inner..., 2, b] = laplacian / (h * h)
end

"""
Everything the right-hand side needs, built once. The application writes
`f!` itself and calls scatter -> fill_ghosts -> map_blocks explicitly, as
`CODE.md` specifies -- there is no `semidiscretize`-style wrapper.
"""
struct WaveProblem{T,D,G,F,S,V}
    fs::F
    schedule::S
    spacings::V                  # per block, wherever the kernel runs
    valD::Val{D}
    valG::Val{G}
end

# D and G are carried as Val parameters so the kernel specializes on
# them once, rather than rebuilding them at every RHS evaluation. The
# spacings are the only geometry the kernel needs, and they follow the
# field set onto its backend — an application-side instance of what
# `block_spacings` exists for.
function WaveProblem(fs::FieldSet{T,D}, schedule) where {T,D}
    G = fs.G
    spacings = to_backend(get_backend(fs.work), block_spacings(fs.forest, T))
    return WaveProblem{T,D,G,typeof(fs),typeof(schedule),typeof(spacings)}(
        fs, schedule, spacings, Val(D), Val(G))
end

# The mesh does this for its own metadata; an application has to do it
# for its own. Deliberately spelled out here rather than reaching into
# TreeAMR: this file stands for a downstream user of the public API.
to_backend(::CPU, a::AbstractArray) = a
function to_backend(backend, a::AbstractArray)
    dev = KernelAbstractions.allocate(backend, eltype(a), size(a))
    copyto!(dev, a)
    return dev
end

function wave_rhs!(du, u, p, t)
    scatter!(p.fs, u)
    fill_ghosts!(p.fs, p.schedule)
    map_blocks!(wave_rhs_kernel!, p.fs, statearray(du, p.fs), p.fs.work,
                p.spacings, p.valD, p.valG)
    return nothing
end

"""Angular frequency of the `m`-th sine mode on a box of side `L`."""
wave_omega(D, L::T, m) where {T} = 2 * T(π) * m * sqrt(T(D)) / L

"""
The exact solution, as a `(x, v) -> value` callback for a field set.

Everything is computed in `typeof(L)`: this closure is a kernel argument
and runs per cell, so a stray `Float64` literal here would need hardware
fp64 on a device — the exact failure the package's own "no floating
point literal in per-cell arithmetic" rule guards against.
"""
function wave_exact(D, L::T, m, t::T) where {T}
    ω = wave_omega(D, L, m)
    k = 2 * T(π) * m / L
    return function (x, var)
        shape = one(T)
        for d in 1:D
            shape *= sin(k * x[d])
        end
        return var == 1 ? cos(ω * t) * shape : -ω * sin(ω * t) * shape
    end
end

"""
A two-level hierarchy: a `roots^D` periodic box with the middle sub-box
refined once, held fixed in physical space as `N` varies so that a
convergence study really does just shrink `h`. With `refined=false` the
same box is left uniform, as a control.
"""
function wave_forest(::Val{D}, N; roots=4, L=1.0, refined=true,
                    T::Type=Float64) where {D}
    L = T(L)
    forest = Forest(ntuple(_ -> roots, D); N=N,
                    periodic=ntuple(_ -> true, D),
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

"""
Evolve the sine mode to `t_end` with fixed-step RK4 and return the
volume-weighted L2 and L∞ errors, plus the finest spacing.
"""
function wave_errors(::Val{D}; N, G=1, ops=Operators(prolongation=2, restriction=2),
                     roots=4, L=1.0, m=1,
                     cfl=0.25, periods=0.25, alg=RK4(), refined=true,
                     T::Type=Float64, backend=CPU()) where {D}
    forest = wave_forest(Val(D), N; roots=roots, L=L, refined=refined, T=T)
    L = T(L)
    fs = FieldSet{T}(forest, 2; G=G, backend=backend)
    problem = WaveProblem(fs, GhostSchedule(fs, ops))

    fill_by_coordinates!(wave_exact(D, L, m, zero(T)), fs)
    u0 = statevector(fs)
    gather!(u0, fs)

    h = minimum_spacing(forest)
    t_end = T(periods) * 2 * T(π) / wave_omega(D, L, m)
    dt = T(cfl) * h
    nsteps = ceil(Int, t_end / dt)
    dt = t_end / nsteps                          # land exactly on t_end

    prob = ODEProblem(wave_rhs!, u0, (zero(T), t_end), problem)
    sol = solve(prob, alg; dt=dt, adaptive=false, save_everystep=false)

    exact = FieldSet{T}(forest, 2; G=G, backend=backend)
    fill_by_coordinates!(wave_exact(D, L, m, t_end), exact)
    uexact = statevector(exact)
    gather!(uexact, exact)

    err = sol.u[end] .- uexact
    return (l2=volume_weighted_norm(fs, err),
            linf=volume_weighted_norm(fs, err; p=Inf),
            h=h, nsteps=nsteps, nblocks=nleaves(forest))
end

"""
A Gaussian pulse travelling in +x at the wave speed, exact for the 1D
wave equation and (with `σ ≪ L`) periodic to roundoff:

    u = G(d),  ∂ₜu = (d/σ²) G(d),   d = x - x₀ - t  (wrapped)

In more than one dimension it is a plane pulse, uniform in the
transverse directions, so `∇²u = ∂ₓ²u` and it stays exact.
"""
function pulse_exact(D, L::T, x0::T, σ::T, t::T) where {T}
    half = L / 2
    twoσ² = 2 * σ^2
    σ² = σ^2
    return function (x, var)
        d = mod(x[1] - x0 - t + half, L) - half
        g = exp(-d^2 / twoσ²)
        return var == 1 ? g : (d / σ²) * g
    end
end

"""
Evolve a travelling pulse, regridding every `chunk` of time so the
refined region follows it. Returns the worst error over the run and how
well the refinement tracked the pulse.

The flagging criterion is deliberately tight — `threshold` sits well up
the pulse rather than far down its tail — and it reports the bounding
box of the cells that fired with *every* flag, so the mesh can dilate
that box by `buffer` cells and refine ahead of the pulse, `CODE.md`'s
step 2. `buffer` counts cells at each block's own resolution. The block
that actually holds the pulse is at the target level and reports
`(Keep, box)`; because its box tracks the pulse across the block rather
than hugging the face the pulse entered through, a buffer of a few
cells is enough to keep an equal-level margin ahead of it.

Regridding changes both the length and the meaning of the state vector,
so each chunk is a fresh `solve`: stop, rebuild the schedule and the
state vector, restart — the pattern `CODE.md` prescribes for anything
beyond a one-step method.
"""
function track_pulse(::Val{D}; N=8, G=2, roots=8, L=1.0, σ=0.05, x0=0.25,
                     ops=Operators(prolongation=4, restriction=4),
                     t_end=0.5, chunk=0.05, cfl=0.25, maxlevel_wanted=2,
                     threshold=0.05, buffer=4,
                     T::Type=Float64, backend=CPU()) where {D}
    L, σ, x0 = T(L), T(σ), T(x0)
    t_end, chunk, cfl = T(t_end), T(chunk), T(cfl)
    forest = Forest(ntuple(_ -> roots, D); N=N, periodic=ntuple(_ -> true, D),
                    extents=ntuple(_ -> (zero(T), L), D))
    fs = FieldSet{T}(forest, 2; G=G, backend=backend)

    # Refine where the pulse actually is, judged from the current data,
    # and report the bounding box of the cells that fired — the min/max
    # reduction the mesh dilates by `buffer` cells. For this plane pulse
    # the box is a slab: narrow in x, the whole block transversally.
    #
    # The per-cell test and the min/max go through `firing_boxes`, so
    # they run on the field set's own backend; what is left on the host
    # is the *verdict*, which is the physics. That split is the point of
    # the device flagging path.
    thr = T(threshold)
    fires(work, idx, b, x) = abs(work[idx..., 1, b]) > thr
    function flags_now()
        boxes = firing_boxes(fires, fs)
        return map(1:nleaves(forest)) do b
            n, box = boxes[b]
            k = forest.leaves[b]
            n == 0 && return level(k) > 0 ? Coarsen : Keep
            # At the target level the block reports (Keep, box): it wants
            # no more refinement of itself, but it still asks for its own
            # level around the box, which is the margin that travels with
            # the pulse.
            level(k) >= maxlevel_wanted && return (Keep, box)
            return (Refine, box)
        end
    end

    fill_by_coordinates!(pulse_exact(D, L, x0, σ, zero(T)), fs)
    schedule, _, _ = adapt_to_initial_data!(fs, ops;
                                            initial=pulse_exact(D, L, x0, σ, zero(T)),
                                            flags=_ -> flags_now(), buffer=buffer,
                                            maxpasses=8)

    worst = zero(T)
    refined_fraction = T[]
    t = zero(T)
    while t < t_end - 1000 * eps(t_end)
        stop = min(t + chunk, t_end)
        problem = WaveProblem(fs, schedule)
        u = statevector(fs)
        gather!(u, fs)
        dt = cfl * minimum_spacing(forest)
        nsteps = max(1, ceil(Int, (stop - t) / dt))
        sol = solve(ODEProblem(wave_rhs!, u, (t, stop), problem), RK4();
                    dt=(stop - t) / nsteps, adaptive=false, save_everystep=false)
        scatter!(fs, sol.u[end])
        t = stop

        # Error against the exact travelling pulse.
        exact = FieldSet{T}(forest, 2; G=G, backend=backend)
        fill_by_coordinates!(pulse_exact(D, L, x0, σ, t), exact)
        ue = statevector(exact)
        gather!(ue, exact)
        worst = max(worst, volume_weighted_norm(fs, sol.u[end] .- ue; p=Inf))

        # How much of the pulse sits in refined blocks -- the measure of
        # whether the refined region is actually following it. Per-block
        # peaks, which is a reduction, so it goes through the mesh's own
        # `volume_weighted_norm`-shaped path rather than a host loop over
        # views: the same argument as for the flagging above.
        peaks = block_peaks(fs)
        inside = zero(T)
        total = zero(T)
        for b in 1:nblocks(fs)
            total = max(total, peaks[b])
            level(blockkey(fs, b)) > 0 && (inside = max(inside, peaks[b]))
        end
        push!(refined_fraction, total > 0 ? inside / total : zero(T))

        fill_ghosts!(fs, schedule)
        if regrid!(forest, fs => schedule; flags=flags_now(), buffer=buffer)
            schedule = GhostSchedule(fs, ops)
        end
    end

    return (worst=worst, tracking=minimum(refined_fraction),
            nblocks=nleaves(forest), maxlevel=maxlevel(forest))
end

"""
The same travelling pulse on a *uniform* mesh, as the reference the
adaptive run is judged against: matching the finest uniform mesh is what
"tracks the pulse without artifacts" has to mean.
"""
function uniform_pulse(::Val{D}; roots, N, G=2, L=1.0, σ=0.08, x0=0.25,
                       ops=Operators(prolongation=4, restriction=4),
                       t_end=0.5, cfl=0.25,
                       T::Type=Float64, backend=CPU()) where {D}
    L, σ, x0 = T(L), T(σ), T(x0)
    t_end, cfl = T(t_end), T(cfl)
    forest = Forest(ntuple(_ -> roots, D); N=N, periodic=ntuple(_ -> true, D),
                    extents=ntuple(_ -> (zero(T), L), D))
    fs = FieldSet{T}(forest, 2; G=G, backend=backend)
    schedule = GhostSchedule(fs, ops)
    fill_by_coordinates!(pulse_exact(D, L, x0, σ, zero(T)), fs)
    u = statevector(fs)
    gather!(u, fs)
    dt = cfl * minimum_spacing(forest)
    nsteps = ceil(Int, t_end / dt)
    sol = solve(ODEProblem(wave_rhs!, u, (zero(T), t_end), WaveProblem(fs, schedule)),
                RK4(); dt=t_end / nsteps, adaptive=false, save_everystep=false)
    exact = FieldSet{T}(forest, 2; G=G, backend=backend)
    fill_by_coordinates!(pulse_exact(D, L, x0, σ, t_end), exact)
    ue = statevector(exact)
    gather!(ue, exact)
    return (err=volume_weighted_norm(fs, sol.u[end] .- ue; p=Inf),
            cells=nleaves(forest) * N^D)
end

"""
The peak |u| of every block, as one launch on whatever backend the field
set lives on.

An application-side reduction, written the way the mesh writes its own:
one work item per block, each looping its own cells, so the answer does
not depend on the backend or on how the loop was split.
"""
@kernel function block_peak_kernel!(peaks, @Const(work), ::Val{D}, ::Val{G},
                                    ::Val{N}) where {D,G,N}
    b = @index(Global)
    m = zero(eltype(peaks))
    for c in CartesianIndices(ntuple(_ -> N, Val(D)))
        m = max(m, abs(work[ntuple(d -> Tuple(c)[d] + G[d], Val(D))..., 1, b]))
    end
    peaks[b] = m
end

function block_peaks(fs::FieldSet{T,D}) where {T,D}
    backend = get_backend(fs.work)
    peaks = KernelAbstractions.allocate(backend, T, (nblocks(fs),))
    block_peak_kernel!(backend)(peaks, fs.work, Val(D), Val(fs.G),
                                Val(fs.forest.N); ndrange=nblocks(fs))
    KernelAbstractions.synchronize(backend)
    return Array(peaks)
end

"""Least-squares convergence rate of `errs` against spacings `hs`."""
function convergence_rate(hs, errs)
    x = log.(hs)
    y = log.(errs)
    n = length(x)
    x̄, ȳ = sum(x) / n, sum(y) / n
    return sum((x .- x̄) .* (y .- ȳ)) / sum((x .- x̄) .^ 2)
end
