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

using KernelAbstractions: @kernel, @index, @Const
using OrdinaryDiffEqLowOrderRK: RK4
using SciMLBase: ODEProblem, solve

@kernel function wave_rhs_kernel!(du, @Const(work), @Const(spacings),
                                  ::Val{D}, ::Val{G}) where {D,G}
    I = @index(Global, NTuple)                 # (i1..iD, block)
    b = I[D + 1]
    inner = ntuple(d -> I[d], Val(D))          # state-layout index
    c = ntuple(d -> I[d] + G, Val(D))          # working-array index

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
struct WaveProblem{T,D,G,F,S}
    fs::F
    schedule::S
    spacings::Vector{T}
    valD::Val{D}
    valG::Val{G}
end

# D and G are carried as Val parameters so the kernel specializes on
# them once, rather than rebuilding them at every RHS evaluation.
function WaveProblem(fs::FieldSet{T,D}, schedule) where {T,D}
    G = fs.forest.G
    return WaveProblem{T,D,G,typeof(fs),typeof(schedule)}(
        fs, schedule, block_spacings(fs.forest, T), Val(D), Val(G))
end

function wave_rhs!(du, u, p, t)
    scatter!(p.fs, u)
    fill_ghosts!(p.fs, p.schedule)
    map_blocks!(wave_rhs_kernel!, p.fs, statearray(du, p.fs), p.fs.work,
                p.spacings, p.valD, p.valG)
    return nothing
end

"""Angular frequency of the `m`-th sine mode on a box of side `L`."""
wave_omega(D, L, m) = 2π * m * sqrt(D) / L

"""The exact solution, as a `(x, v) -> value` callback for a field set."""
function wave_exact(D, L, m, t)
    ω = wave_omega(D, L, m)
    return function (x, var)
        shape = prod(sin(2π * m * x[d] / L) for d in 1:D)
        return var == 1 ? cos(ω * t) * shape : -ω * sin(ω * t) * shape
    end
end

"""
A two-level hierarchy: a `roots^D` periodic box with the middle sub-box
refined once, held fixed in physical space as `N` varies so that a
convergence study really does just shrink `h`. With `refined=false` the
same box is left uniform, as a control.
"""
function wave_forest(::Val{D}, N, G; roots=4, L=1.0, refined=true) where {D}
    forest = Forest(ntuple(_ -> roots, D); N=N, G=G,
                    periodic=ntuple(_ -> true, D),
                    extents=ntuple(_ -> (0.0, L), D))
    refined || return forest
    targets = filter(forest.leaves) do k
        ext = block_extent(forest, k)
        all(d -> 0.25L < (ext[d][1] + ext[d][2]) / 2 < 0.75L, 1:D)
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
                     cfl=0.25, periods=0.25, alg=RK4(), refined=true) where {D}
    forest = wave_forest(Val(D), N, G; roots=roots, L=L, refined=refined)
    fs = FieldSet(forest, 2)
    problem = WaveProblem(fs, GhostSchedule(forest, ops))

    fill_by_coordinates!(wave_exact(D, L, m, 0.0), fs)
    u0 = statevector(fs)
    gather!(u0, fs)

    h = minimum_spacing(forest)
    t_end = periods * 2π / wave_omega(D, L, m)
    dt = cfl * h
    nsteps = ceil(Int, t_end / dt)
    dt = t_end / nsteps                          # land exactly on t_end

    prob = ODEProblem(wave_rhs!, u0, (0.0, t_end), problem)
    sol = solve(prob, alg; dt=dt, adaptive=false, save_everystep=false)

    exact = FieldSet(forest, 2)
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
function pulse_exact(D, L, x0, σ, t)
    return function (x, var)
        d = mod(x[1] - x0 - t + L / 2, L) - L / 2
        g = exp(-d^2 / (2σ^2))
        return var == 1 ? g : (d / σ^2) * g
    end
end

"""
Evolve a travelling pulse, regridding every `chunk` of time so the
refined region follows it. Returns the worst error over the run and how
well the refinement tracked the pulse.

The flagging criterion is deliberately tight — `threshold` sits well up
the pulse rather than far down its tail — and it reports the bounding
box of the cells that fired, so the mesh can dilate that box by
`buffer` cells and refine ahead of the pulse, `CODE.md`'s step 2.
`buffer` counts cells at each block's own resolution, and a buffer much
narrower than a block reaches nothing here: a block ahead of the pulse
is refined as soon as the criterion fires anywhere in it, so its box
hugs the face the pulse arrived through.

Regridding changes both the length and the meaning of the state vector,
so each chunk is a fresh `solve`: stop, rebuild the schedule and the
state vector, restart — the pattern `CODE.md` prescribes for anything
beyond a one-step method.
"""
function track_pulse(::Val{D}; N=8, G=2, roots=8, L=1.0, σ=0.05, x0=0.25,
                     ops=Operators(prolongation=4, restriction=4),
                     t_end=0.5, chunk=0.05, cfl=0.25, maxlevel_wanted=2,
                     threshold=0.05, buffer=4) where {D}
    forest = Forest(ntuple(_ -> roots, D); N=N, G=G, periodic=ntuple(_ -> true, D),
                    extents=ntuple(_ -> (0.0, L), D))
    fs = FieldSet(forest, 2)

    # Refine where the pulse actually is, judged from the current data,
    # and report the bounding box of the cells that fired — the min/max
    # reduction the mesh dilates by `buffer` cells. For this plane pulse
    # the box is a slab: narrow in x, the whole block transversally.
    function flag(b, k)
        fired = findall(x -> abs(x) > threshold, interiorview(fs, b, 1))
        isempty(fired) && return level(k) > 0 ? Coarsen : Keep
        level(k) >= maxlevel_wanted && return Keep
        box = ntuple(d -> minimum(i -> i[d], fired):maximum(i -> i[d], fired), D)
        return (Refine, box)
    end

    fill_by_coordinates!(pulse_exact(D, L, x0, σ, 0.0), fs)
    schedule, _, _ = adapt_to_initial_data!(fs, ops;
                                            initial=pulse_exact(D, L, x0, σ, 0.0),
                                            flag=flag, buffer=buffer, maxpasses=8)

    worst = 0.0
    refined_fraction = Float64[]
    t = 0.0
    while t < t_end - 1e-12
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
        exact = FieldSet(forest, 2)
        fill_by_coordinates!(pulse_exact(D, L, x0, σ, t), exact)
        ue = statevector(exact)
        gather!(ue, exact)
        worst = max(worst, volume_weighted_norm(fs, sol.u[end] .- ue; p=Inf))

        # How much of the pulse sits in refined blocks -- the measure of
        # whether the refined region is actually following it.
        inside = 0.0
        total = 0.0
        for b in 1:nblocks(fs)
            peak = maximum(abs, interiorview(fs, b, 1))
            total = max(total, peak)
            level(blockkey(fs, b)) > 0 && (inside = max(inside, peak))
        end
        push!(refined_fraction, total > 0 ? inside / total : 0.0)

        fill_ghosts!(fs, schedule)
        flags = flag_blocks(flag, forest)
        if regrid!(forest, fs, schedule; flags=flags, buffer=buffer)
            schedule = GhostSchedule(forest, ops)
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
                       t_end=0.5, cfl=0.25) where {D}
    forest = Forest(ntuple(_ -> roots, D); N=N, G=G, periodic=ntuple(_ -> true, D),
                    extents=ntuple(_ -> (0.0, L), D))
    fs = FieldSet(forest, 2)
    schedule = GhostSchedule(forest, ops)
    fill_by_coordinates!(pulse_exact(D, L, x0, σ, 0.0), fs)
    u = statevector(fs)
    gather!(u, fs)
    dt = cfl * minimum_spacing(forest)
    nsteps = ceil(Int, t_end / dt)
    sol = solve(ODEProblem(wave_rhs!, u, (0.0, t_end), WaveProblem(fs, schedule)), RK4();
                dt=t_end / nsteps, adaptive=false, save_everystep=false)
    exact = FieldSet(forest, 2)
    fill_by_coordinates!(pulse_exact(D, L, x0, σ, t_end), exact)
    ue = statevector(exact)
    gather!(ue, exact)
    return (err=volume_weighted_norm(fs, sol.u[end] .- ue; p=Inf),
            cells=nleaves(forest) * N^D)
end

"""Least-squares convergence rate of `errs` against spacings `hs`."""
function convergence_rate(hs, errs)
    x = log.(hs)
    y = log.(errs)
    n = length(x)
    x̄, ȳ = sum(x) / n, sum(y) / n
    return sum((x .- x̄) .* (y .- ȳ)) / sum((x .- x̄) .^ 2)
end
