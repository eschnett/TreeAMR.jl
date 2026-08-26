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

"""Least-squares convergence rate of `errs` against spacings `hs`."""
function convergence_rate(hs, errs)
    x = log.(hs)
    y = log.(errs)
    n = length(x)
    x̄, ȳ = sum(x) / n, sum(y) / n
    return sum((x .- x̄) .* (y .- ȳ)) / sum((x .- x̄) .^ 2)
end
