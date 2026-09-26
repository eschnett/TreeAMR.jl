# One time step, by integrator: OrdinaryDiffEq against IMEXRungeKutta,
# the latter with its stage arithmetic broadcast and by owner.
#
#     julia -t N --project=test bench/stepping.jl
#
# It runs in the test environment, which has both integrators. Prints the
# tab-separated lines of `bench/threads.jl`, so `bench/scan.sh` drives it
# with `TREEAMR_BENCH_SCRIPT=bench/stepping.jl TREEAMR_BENCH_PROJECT=test`.
#
# Why: OrdinaryDiffEq forms its stages with a serial broadcast, copies the
# state and scans it for finiteness every step, and allocates its buffers
# on the calling thread at every `solve` ("Open questions" in `CODE.md`,
# where that path is left as measured). IMEXRungeKutta does four
# combinations an RK4 step and nothing else, and by owner it gives each
# block's entries to the thread `map_blocks!` runs the block on. Each
# integrator is built once and stepped, the way a driver keeps one per
# mesh generation, so the per-`solve` allocation is not in these numbers;
# `solve_rk4_ode` is the `solve`-per-chunk pattern, four steps a call.
#
# Sizes come from the environment, as in `bench/threads.jl`:
#
#     TREEAMR_BENCH_D      dimension (default 3)
#     TREEAMR_BENCH_N      cells per block edge (default 16)
#     TREEAMR_BENCH_ROOTS  roots per edge (default 4)
#     TREEAMR_BENCH_REPS   timed repetitions (default 10)

using TreeAMR
using TreeAMR: threadchunks
using KernelAbstractions: @kernel, @index, @Const
import IMEXRungeKutta as IRK
import OrdinaryDiffEqLowOrderRK, OrdinaryDiffEqSSPRK
import SciMLBase
using Printf: @printf

const D = parse(Int, get(ENV, "TREEAMR_BENCH_D", "3"))
const N = parse(Int, get(ENV, "TREEAMR_BENCH_N", "16"))
const ROOTS = parse(Int, get(ENV, "TREEAMR_BENCH_ROOTS", "4"))
const REPS = parse(Int, get(ENV, "TREEAMR_BENCH_REPS", "10"))
const G = 2
const OPS = Operators(prolongation=4, restriction=4)

@kernel function bench_rhs_kernel!(du, @Const(work), @Const(spacings),
                                   ::Val{DD}, ::Val{GG}) where {DD,GG}
    I = @index(Global, NTuple)
    b = I[DD + 1]
    c = ntuple(d -> I[d] + GG[d], Val(DD))
    u0 = work[c..., 1, b]
    laplacian = zero(eltype(du))
    for d in 1:DD
        up = Base.setindex(c, c[d] + 1, d)
        um = Base.setindex(c, c[d] - 1, d)
        laplacian += work[up..., 1, b] - 2 * u0 + work[um..., 1, b]
    end
    h = spacings[b]
    du[ntuple(d -> I[d], Val(DD))..., 1, b] = work[c..., 2, b]
    du[ntuple(d -> I[d], Val(DD))..., 2, b] = laplacian / (h * h)
end

struct Problem{F,S,V}
    fs::F
    schedule::S
    spacings::V
end

function rhs!(du, u, p, t)
    scatter!(p.fs, u)
    fill_ghosts!(p.fs, p.schedule)
    map_blocks!(bench_rhs_kernel!, p.fs, statearray(du, p.fs), p.fs.work, p.spacings,
                Val(D), Val(p.fs.G))
    return nothing
end

# The ownership partition of the state vector, as `test/imex_tests.jl`
# defines and checks it.
function state_partition(fs)
    L = fs.forest.N^D * fs.nvars
    parts = UnitRange{Int}[(first(r) - 1) * L + 1:last(r) * L
                           for r in threadchunks(nblocks(fs))]
    append!(parts, fill(1:0, Threads.nthreads() - length(parts)))
    return parts
end

"""A two-level mesh: the middle eighth of the domain refined once."""
function build_forest()
    forest = Forest(ntuple(_ -> ROOTS, D); N=N, periodic=ntuple(_ -> true, D),
                    extents=ntuple(_ -> (0.0, 1.0), D))
    targets = filter(forest.leaves) do k
        ext = block_extent(forest, k)
        all(d -> 0.25 < (ext[d][1] + ext[d][2]) / 2 < 0.75, 1:D)
    end
    refine!(forest, targets)
    balance!(forest)
    return forest
end

function best(f, reps=REPS)
    f()
    t = Inf
    for _ in 1:reps
        t = min(t, @elapsed f())
    end
    return t
end

function main()
    forest = build_forest()
    fs = FieldSet(forest, 2; G=G)
    p = Problem(fs, GhostSchedule(fs, OPS), block_spacings(forest))
    fill_by_coordinates!((x, v) -> sin(2π * x[1]) + v, fs)
    u0 = statevector(fs)
    gather!(u0, fs)
    du = similar(u0)
    dt = 0.1 * minimum_spacing(forest)
    far = 1e6 * dt                               # never reached
    part = state_partition(fs)

    imex(tab, partition) =
        IRK.init(IRK.IMEXProblem(rhs!, nothing, u0, (0.0, far), p), tab; dt=dt,
                 partition=partition)
    ode(alg) = SciMLBase.init(SciMLBase.ODEProblem(rhs!, u0, (0.0, far), p), alg;
                              dt=dt, adaptive=false, save_everystep=false)
    rk4, ssp = OrdinaryDiffEqLowOrderRK.RK4(), OrdinaryDiffEqSSPRK.SSPRK33()
    solve4() = SciMLBase.solve(SciMLBase.ODEProblem(rhs!, u0, (0.0, 4dt), p), rk4;
                               dt=dt, adaptive=false, save_everystep=false)

    rows = Tuple{String,Float64}[]
    push!(rows, ("rhs", best(() -> rhs!(du, u0, p, 0.0))))
    for (name, integ, step) in (
            ("step_rk4_ode", ode(rk4), SciMLBase.step!),
            ("step_rk4_imex_broadcast", imex(IRK.RK4(), nothing), IRK.step!),
            ("step_rk4_imex_owner", imex(IRK.RK4(), part), IRK.step!),
            ("step_ssprk33_ode", ode(ssp), SciMLBase.step!),
            ("step_ssprk33_imex_broadcast", imex(IRK.SSPRK33(), nothing), IRK.step!),
            ("step_ssprk33_imex_owner", imex(IRK.SSPRK33(), part), IRK.step!))
        push!(rows, (name, best(() -> step(integ))))
    end
    push!(rows, ("solve_rk4_ode", best(solve4) / 4))

    @printf("threads=%d D=%d N=%d roots=%d blocks=%d entries=%d\n",
            Threads.nthreads(), D, N, ROOTS, nblocks(fs), length(u0))
    for (name, t) in rows
        @printf("%d\t%s\t%.6f\n", Threads.nthreads(), name, t)
    end
    t_rhs = rows[1][2]
    for (name, t) in rows[2:end]
        nrhs = occursin("rk4", name) ? 4 : 3
        @printf("# %s: %.1f ms over its %d RHS evaluations\n", name,
                1e3 * (t - nrhs * t_rhs), nrhs)
    end
    return nothing
end

main()
