# The workload behind the M5 thread-count independence test.
#
# Run as a standalone script — `julia -t N --project=test
# test/thread_workload.jl` — it prints a digest of everything a full
# TreeAMR cycle produces: the state vector bit for bit, the leaf array,
# the schedule's shape, and the diagnostic reductions. Two runs at
# different thread counts must print the same lines, character for
# character. That is stricter than `CODE.md`'s "matches serial to
# roundoff", and it is the property the implementation actually has:
# every parallel loop in the package writes to its own slot and every
# combination happens in a fixed order.
#
# It is a script rather than a testset because the thread count is a
# command-line argument to Julia and cannot be changed from inside a
# running session. Deliberately self-contained — no ODE package, just a
# hand-written RK4 — so that a subprocess starts in a couple of seconds.

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
