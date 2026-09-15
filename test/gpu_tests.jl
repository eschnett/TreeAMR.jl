# GPU support (M6).
#
# There is no GPU on CI, and there is no way to put one there. But
# nothing in this file is *about* a GPU: it is about the API that makes
# a device possible — an explicitly chosen backend, a schedule whose
# stencils live where the kernel reads them, a boundary hook expressed
# per cell, a flagging path that reduces blocks on the device, reductions
# that do not scalar-index. All of that runs on `CPU()` too, which is a
# KernelAbstractions backend like any other, and it is exercised here on
# every run. A device is then one more entry in `BACKENDS`.
#
# To add one, name it in the environment and make the package available
# in the active project:
#
#     julia --project=test -e 'using Pkg; Pkg.add("Metal")'
#     TREEAMR_TEST_BACKEND=metal julia --project=test test/runtests.jl
#
# `cuda` and `metal` are recognized. A backend that is named but not
# functional is reported and skipped rather than failing: the point is
# to make running on real hardware easy, not to make its absence an
# error. `bench/symmetry_gpu.sh` does exactly this on Symmetry's H200
# partition, which is where the Float64 CUDA numbers come from.

using KernelAbstractions: CPU, get_backend, supports_float64

# The device package is loaded at top level, in its own statement, so
# that everything after it is compiled in a world that can see it.
# `include` evaluates this file statement by statement at top level even
# though `runtests.jl` calls it from inside a testset, which is what
# makes a conditional `using` legal here.
const GPU_NAME = lowercase(get(ENV, "TREEAMR_TEST_BACKEND", ""))

if GPU_NAME == "cuda"
    using CUDA
elseif GPU_NAME == "metal"
    using Metal
elseif !isempty(GPU_NAME)
    error("TREEAMR_TEST_BACKEND must be \"cuda\" or \"metal\", got \"$GPU_NAME\"")
end

# Which backends to test, and in which element types. A device without
# hardware fp64 gets Float32 only — the mesh is generic in its float
# type precisely so that this is a configuration and not a port.
const BACKENDS = let out = Any[("CPU", CPU(), (Float64, Float32))]
    device = GPU_NAME == "cuda" ? (CUDA.functional() ? CUDABackend() : nothing) :
             GPU_NAME == "metal" ? (Metal.functional() ? MetalBackend() : nothing) :
             nothing
    if device === nothing
        isempty(GPU_NAME) ||
            @info "TREEAMR_TEST_BACKEND=$GPU_NAME is not functional here; CPU only"
    else
        types = supports_float64(device) ? (Float64, Float32) : (Float32,)
        @info "Also testing on $(nameof(typeof(device))) in $types"
        push!(out, (uppercase(GPU_NAME), device, types))
    end
    out
end

# Tolerances follow the type, as everywhere else in this suite: an
# absolute Float64 constant is meaningless at Float32.
gputol(::Type{T}, k=4096) where {T} = k * eps(T)

@testset "$bname: a field set is allocated where it is asked for: T=$T" for
        (bname, backend, types) in BACKENDS, T in types
    forest = Forest((2, 2); N=4, periodic=(true, true),
                    extents=ntuple(_ -> (zero(T), one(T)), 2))
    fs = FieldSet{T}(forest, 2; G=1, backend=backend)
    @test eltype(fs.work) === T
    @test typeof(get_backend(fs)) === typeof(backend)
    @test typeof(get_backend(statevector(fs))) === typeof(backend)
    # Fresh storage is zeroed through the kernel, not `fill!`.
    @test iszero(sum(Array(fs.work)))
end

@testset "$bname: a device rejects Float64 with a reason" for (bname, backend, _) in BACKENDS
    supports_float64(backend) && continue
    forest = Forest((2,); N=4, periodic=(true,))
    @test_throws "no hardware Float64" FieldSet{Float64}(forest, 1; G=1, backend=backend)
end

@testset "$bname: a schedule and a field set must agree on the backend" for
        (bname, backend, types) in BACKENDS
    bname == "CPU" && continue
    T = first(types)
    forest = Forest((2,); N=4, periodic=(true,),
                    extents=((zero(T), one(T)),))
    ops = Operators(prolongation=2, restriction=2)
    fs = FieldSet{T}(forest, 1; G=1, backend=backend)
    host = GhostSchedule(forest, ops; G=1, T=T)         # CPU stencils
    @test_throws "wrong memory" fill_ghosts!(fs, host)
end

@testset "$bname: ghost exchange is exact for polynomials: T=$T, D=$D" for
        (bname, backend, types) in BACKENDS, T in types, D in (1, 2)
    # The M2 claim, on whatever backend: a degree-(p-1) polynomial must
    # survive copy, restriction and prolongation exactly, including the
    # outer-boundary cells the cell-wise hook fills.
    p = 4
    ops = Operators(prolongation=p, restriction=p)
    forest = Forest(ntuple(_ -> 3, D); N=8, periodic=ntuple(_ -> false, D),
                    extents=ntuple(_ -> (zero(T), one(T)), D))
    refine!(forest, forest.leaves[1])
    balance!(forest)

    # Degree p-1 in each coordinate, which the operators reproduce
    # exactly. Note what the closure does *not* capture: a device kernel
    # argument has to be `isbits`, and a `Type` is not, so the element
    # type is taken from `x` rather than closed over. This is a real
    # sharp edge of writing callbacks for a device, and the reason it is
    # spelled out in the docstrings.
    poly = (x, v) -> begin
        acc = oftype(x[1], v)
        two = oftype(x[1], 2)
        for d in 1:D
            acc += x[d]^(p - 1) - two * x[d]
        end
        acc
    end

    G = 2
    fs = FieldSet{T}(forest, 2; G=G, backend=backend)
    schedule = GhostSchedule(fs, ops)
    fill_by_coordinates!(poly, fs)
    fill_ghosts!(fs, schedule; boundary=boundary_by_coordinates(poly))

    # Compare on the host against the analytic values, ghosts included.
    work = Array(fs.work)
    worst = zero(T)
    for b in 1:nblocks(fs), v in 1:2
        for idx in CartesianIndices(ntuple(_ -> forest.N + 2G, D))
            x = coordinates(T, fs, b, Tuple(idx))
            worst = max(worst, abs(work[Tuple(idx)..., v, b] - poly(x, v)))
        end
    end
    @test worst < gputol(T)
end

@testset "$bname: a staggered exchange runs on the device: T=$T, D=$D" for
        (bname, backend, types) in BACKENDS, T in types, D in (1, 2)
    # The same claim for a vertex-like layout (M8). Two things here are
    # device-specific rather than merely centering-specific: the transfer
    # kernel now carries one stencil width *per dimension*, so its
    # `Val{Ps}` must still be `isbits` and its `CartesianIndices(Ps)` loop
    # must still compile; and the boundary kernel forms a position from
    # the centering, which must agree with `coordinates` on the host.
    ops = Operators(prolongation=2, restriction=2)
    forest = Forest(ntuple(_ -> 3, D); N=8, periodic=ntuple(_ -> false, D),
                    extents=ntuple(_ -> (zero(T), one(T)), D))
    refine!(forest, forest.leaves[1])
    balance!(forest)

    # Linear in each coordinate, which order 2 reproduces exactly. As
    # above, the element type comes from `x` rather than being closed
    # over: a kernel argument has to be `isbits`.
    poly = (x, v) -> begin
        acc = oftype(x[1], v)
        for d in 1:D
            acc += (1 + oftype(x[1], d)) * x[d]
        end
        acc
    end

    for C in (vertexcentered(D), facecentered(D, D))
        c = staggers(C)
        G = 1
        fs = FieldSet{T}(forest, 2; G=G, centering=C, backend=backend)
        schedule = GhostSchedule(fs, ops)
        fill_by_coordinates!(poly, fs)
        fill_ghosts!(fs, schedule; boundary=boundary_by_coordinates(poly))

        work = Array(fs.work)
        worst = zero(T)
        for b in 1:nblocks(fs), v in 1:2
            for idx in CartesianIndices(ntuple(d -> forest.N + 2G + c[d], D))
                x = coordinates(T, fs, b, Tuple(idx))
                worst = max(worst, abs(work[Tuple(idx)..., v, b] - poly(x, v)))
            end
        end
        @test worst < gputol(T)
    end
end

@testset "$bname: the interface fixup runs on the device: T=$T, D=$D" for
        (bname, backend, types) in BACKENDS, T in types, D in (1, 2, 3)
    # M8b. The fixup adds no kernel of its own — it is the transfer
    # kernel over one-plane target ranges — so what has to hold on a
    # device is that the schedule's stencils were uploaded with it and
    # that the generic per-batch `run_phase!` drives them. Two claims: a
    # linear field is *invariant* under the restriction (injection and
    # the exact two-cell average both reproduce it), and the device lands
    # on the same numbers as the CPU for data with no structure at all.
    forest = Forest(ntuple(_ -> 3, D); N=8, periodic=ntuple(_ -> false, D),
                    extents=ntuple(_ -> (zero(T), one(T)), D))
    refine!(forest, forest.leaves[1])
    balance!(forest)

    C = facecentered(D, 1)
    c = staggers(C)
    # A flux carries no ghosts, so its stored range *is* its closed
    # range: what a block computes for itself, its own high face
    # included.
    closed = CartesianIndices(ntuple(d -> 1:(forest.N + c[d]), D))

    hostfs = FieldSet{T}(forest, 2; G=0, centering=C)
    for b in 1:nblocks(hostfs), v in 1:2, idx in closed
        x = coordinates(T, hostfs, b, Tuple(idx))
        hostfs.work[Tuple(idx)..., v, b] = sum(x) + T(v)
    end
    linear = copy(hostfs.work)
    restrict_interfaces!(hostfs, InterfaceSchedule(hostfs))
    @test maximum(abs, hostfs.work .- linear) < gputol(T)

    # The same schedule over structureless data, on both backends. What
    # the numbers are does not matter; that the two agree bit for bit
    # does, so the assertion does not depend on the RNG stream.
    noise = T.(rand(MersenneTwister(42), size(linear)...))
    copyto!(hostfs.work, noise)
    restrict_interfaces!(hostfs, InterfaceSchedule(hostfs))

    fs = FieldSet{T}(forest, 2; G=0, centering=C, backend=backend)
    copyto!(fs.work, noise)
    restrict_interfaces!(fs, InterfaceSchedule(fs))
    @test Array(fs.work) == hostfs.work
end

@testset "$bname: the conservative cycle conserves on the device: T=$T, D=$D" for
        (bname, backend, types) in BACKENDS, T in types, D in (1, 2)
    # M8b's acceptance claim, on whatever backend: the three-step
    # right-hand side of `burgers.jl` over the M3 two-level mesh keeps the
    # domain integral of `u` to within a few ulp, and the same run with
    # the fixup skipped does not.
    #
    # What is device-specific here is not the fixup — that has its own
    # test above — but the shape of the right-hand side around it: a
    # second field set per dimension with a different centering and no
    # ghosts, a kernel launched over the *closed* range, and a divergence
    # kernel whose flux argument is an `NTuple{D}` of device arrays,
    # which has to survive the adaptation of kernel arguments intact.
    ops = Operators(family=Conservative, prolongation=3, restriction=2)
    common = (; N=8, roots=4, ops=ops, fraction=0.5, limiter=:none, T=T,
              backend=backend)
    r = burgers_errors(Val(D); common...)
    control = burgers_errors(Val(D); common..., fixup=false)

    @test r.nblocks > 4^D                         # the mesh really is two-level
    @test isfinite(r.l1)
    @test r.drift <= 64 * eps(T) * r.scale
    @test control.drift > 100 * eps(T) * control.scale
end

@testset "$bname: the cell hook reproduces the host hook exactly: T=$T, D=$D" for
        (bname, backend, types) in BACKENDS, T in types, D in (1, 2)
    # `boundary_by_coordinates` used to be a host loop calling
    # `coordinates`; it is now a kernel forming the position from the
    # same origin and spacing. "Same expression" has to mean bit for
    # bit, or M5's thread-independence digests would have moved.
    ops = Operators(prolongation=2, restriction=2)
    forest = Forest(ntuple(_ -> 2, D); N=4, periodic=ntuple(_ -> false, D),
                    extents=ntuple(_ -> (zero(T), one(T)), D))
    f = (x, v) -> sum(x) * oftype(x[1], v) + one(x[1])

    fs = FieldSet{T}(forest, 2; G=2, backend=backend)
    fill_ghosts!(fs, GhostSchedule(fs, ops); boundary=boundary_by_coordinates(f))
    got = Array(fs.work)

    # The host formulation, spelled out here so the comparison is
    # against something independent of the implementation under test.
    hostfs = FieldSet{T}(forest, 2; G=2)
    want = zeros(T, size(got))
    schedule = GhostSchedule(hostfs, ops)
    for r in schedule.boundaries, v in 1:2, idx in r.region
        want[Tuple(idx)..., v, r.block] =
            f(coordinates(T, hostfs, Int(r.block), Tuple(idx)), v)
    end
    for r in schedule.boundaries, v in 1:2, idx in r.region
        @test got[Tuple(idx)..., v, r.block] === want[Tuple(idx)..., v, r.block]
    end
    @test !isempty(schedule.boundaries)
end

@testset "$bname: the region hook is refused on a device, with a way out" for
        (bname, backend, types) in BACKENDS
    bname == "CPU" && continue
    T = first(types)
    forest = Forest((2,); N=4, periodic=(false,),
                    extents=((zero(T), one(T)),))
    ops = Operators(prolongation=2, restriction=2)
    fs = FieldSet{T}(forest, 1; G=1, backend=backend)
    schedule = GhostSchedule(fs, ops)
    region_form = (fs, b, key, δ, region) -> nothing
    @test_throws "CellBoundary" fill_ghosts!(fs, schedule; boundary=region_form)
end

@testset "$bname: firing_boxes matches a host sweep: T=$T, D=$D" for
        (bname, backend, types) in BACKENDS, T in types, D in (1, 2)
    forest = Forest(ntuple(_ -> 3, D); N=8, periodic=ntuple(_ -> true, D),
                    extents=ntuple(_ -> (zero(T), one(T)), D))
    refine!(forest, forest.leaves[1])
    balance!(forest)
    G = 1
    fs = FieldSet{T}(forest, 1; G=G, backend=backend)
    # A blob, so that some blocks fire in part, some wholly, some not at
    # all — the three cases the box reduction has to get right.
    centre = ntuple(_ -> T(3) / 8, D)
    width = T(0.01)
    fill_by_coordinates!((x, v) -> exp(-sum((x .- centre) .^ 2) / width), fs)

    thr = T(0.25)
    fires(work, idx, b, x) = work[idx..., 1, b] > thr
    got = firing_boxes(fires, fs)

    # The oracle: the same predicate, on the host, over the interior.
    work = Array(fs.work)
    N = forest.N
    for b in 1:nblocks(fs)
        hits = [Tuple(c) for c in CartesianIndices(ntuple(_ -> N, D))
                if work[ntuple(d -> Tuple(c)[d] + G, D)..., 1, b] > thr]
        n, box = got[b]
        @test n == length(hits)
        if isempty(hits)
            @test all(isempty, box)
        else
            for d in 1:D
                @test box[d] == minimum(h -> h[d], hits):maximum(h -> h[d], hits)
            end
        end
    end
    @test any(g -> g[1] > 0, got)                 # the test is not vacuous
    @test any(g -> g[1] == 0, got)
end

@testset "$bname: regridding transfers conservatively: T=$T, D=$D" for
        (bname, backend, types) in BACKENDS, T in types, D in (1, 2)
    # M4's conservation claim, driven by the device flagging path end to
    # end: flag on the backend, regrid, and the volume integral must
    # survive with the conservative family.
    ops = Operators(prolongation=3, restriction=2, family=Conservative)
    forest = Forest(ntuple(_ -> 3, D); N=8, periodic=ntuple(_ -> true, D),
                    extents=ntuple(_ -> (zero(T), one(T)), D))
    fs = FieldSet{T}(forest, 1; G=2, backend=backend)
    schedule = GhostSchedule(fs, ops)
    centre = ntuple(_ -> T(1) / 2, D)
    width = T(0.05)
    fill_by_coordinates!((x, v) -> exp(-sum((x .- centre) .^ 2) / width), fs)
    fill_ghosts!(fs, schedule)

    before = total_mass(fs)
    thr = T(0.5)
    fires(work, idx, b, x) = work[idx..., 1, b] > thr
    flags = map(firing_boxes(fires, fs)) do (n, box)
        n == 0 ? Coarsen : (Refine, box)
    end
    @test regrid!(forest, fs => schedule; flags=flags)
    @test nleaves(forest) > 3^D                   # something really refined
    @test total_mass(fs) ≈ before rtol = gputol(T, 256)
end

@testset "$bname: the volume-weighted norm agrees with the CPU: T=$T, D=$D" for
        (bname, backend, types) in BACKENDS, T in types, D in (1, 2)
    # The device reduction is a per-block kernel; the CPU one is a host
    # `sum` over views. They cannot be bit-identical — the summation
    # orders differ — but they must agree to the precision's roundoff.
    forest = Forest(ntuple(_ -> 3, D); N=8, periodic=ntuple(_ -> true, D),
                    extents=ntuple(_ -> (zero(T), one(T)), D))
    refine!(forest, forest.leaves[1])
    balance!(forest)
    four = T(4)
    f = (x, v) -> sin(four * x[1]) + oftype(x[1], v)

    dev = FieldSet{T}(forest, 2; G=1, backend=backend)
    fill_by_coordinates!(f, dev)
    u = statevector(dev)
    gather!(u, dev)

    host = FieldSet{T}(forest, 2; G=1)
    fill_by_coordinates!(f, host)
    uh = statevector(host)
    gather!(uh, host)

    @test volume_weighted_norm(dev, u) ≈ volume_weighted_norm(host, uh) rtol = gputol(T)
    @test volume_weighted_norm(dev, u; p=Inf) ≈
          volume_weighted_norm(host, uh; p=Inf) rtol = gputol(T)
    @test total_mass(dev) ≈ total_mass(host) rtol = gputol(T, 65536)
end

@testset "$bname: block_mapreduce agrees with a host sweep: T=$T, D=$D" for
        (bname, backend, types) in BACKENDS, T in types, D in (1, 2)
    # The reduction an application builds its own diagnostics on. The
    # device forms the per-block values in one launch and the host in
    # threaded `mapreduce`s over views; only the association of `op`
    # differs, so `max` and an integer count must agree exactly and a
    # sum to the precision's roundoff.
    forest = Forest(ntuple(_ -> 3, D); N=8, periodic=ntuple(_ -> true, D),
                    extents=ntuple(_ -> (zero(T), one(T)), D))
    refine!(forest, forest.leaves[1])
    balance!(forest)
    four = T(4)
    f = (x, v) -> sin(four * x[1]) + oftype(x[1], v)

    dev = FieldSet{T}(forest, 2; G=2, backend=backend)
    fill_by_coordinates!(f, dev)
    host = FieldSet{T}(forest, 2; G=2)
    fill_by_coordinates!(f, host)

    half = T(1) / 2
    for vars in (1, 1:2)
        @test block_mapreduce(abs, max, zero(T), dev; vars=vars) ==
              block_mapreduce(abs, max, zero(T), host; vars=vars)
        @test block_mapreduce(x -> abs(x) > half, +, 0, dev; vars=vars) ==
              block_mapreduce(x -> abs(x) > half, +, 0, host; vars=vars)
        @test block_mapreduce(identity, +, zero(T), dev; vars=vars) ≈
              block_mapreduce(identity, +, zero(T), host; vars=vars) rtol = gputol(T)
    end

    # The state-vector form reads a different array with a different
    # ghost offset, so it is exercised separately.
    u = statevector(dev)
    gather!(u, dev)
    uh = statevector(host)
    gather!(uh, host)
    @test block_mapreduce(abs, max, zero(T), dev, u) ==
          block_mapreduce(abs, max, zero(T), host, uh)
end

# --- the acceptance test -------------------------------------------------

@testset "$bname: M3 convergence reproduced: T=$T, D=$D" for
        (bname, backend, types) in BACKENDS, T in types, D in (1, 2)
    # `CODE.md`'s M6 acceptance criterion: the M3 convergence result, on
    # a device. Order-4 operators with G = 2 over a two-level mesh give
    # 2nd order, by the interface-order rule. Cell-centered, spelled out:
    # this is the M3 study, and `wave_errors` is vertex-centered by
    # default from M8 on.
    #
    # The resolutions differ by precision, and deliberately. The error
    # measured here is a truncation error, which is the same number in
    # every precision, while the roundoff floor it has to clear moves:
    # in Float32, D = 1 at N = 64 is already at its floor (measured — the
    # rate over N = 16…128 collapses to 0.73), because the step count
    # grows with N. So Float32 is measured over the range where the
    # truncation error still dominates. This is the same caveat `CODE.md`
    # records for negative accuracy assertions, in its positive form.
    ops = Operators(prolongation=4, restriction=4)
    Ns = T === Float64 ? (8, 16, 32) : (D == 1 ? (16, 32) : (8, 16, 32))
    hs, l2 = T[], T[]
    for N in Ns
        r = wave_errors(Val(D); N=N, G=2, ops=ops, centering=cellcentered(D),
                        T=T, backend=backend)
        push!(hs, r.h)
        push!(l2, r.l2)
        @test isfinite(r.l2)
        @test r.nblocks > 2^D                     # refinement really happened
    end
    @test all(l2[i] > l2[i + 1] for i in 1:(length(l2) - 1))
    @test convergence_rate(Float64.(hs), Float64.(l2)) ≈ 2.0 atol = 0.15
end

@testset "$bname: vertex convergence reproduced: T=$T, D=$D" for
        (bname, backend, types) in BACKENDS, T in types, D in (1, 2)
    # M8a's acceptance criterion on a device: the same 2nd-order result
    # on a *staggered* layout, at G = 1. Three things here are
    # device-specific rather than merely centering-specific — the
    # per-dimension stencil widths the transfer kernel carries as a
    # `Val` of a tuple, the exchange filling a shared plane that the
    # state vector does not hold, and the regrid-free but
    # prolongation-heavy two-level mesh — and all three run through the
    # same launches as the cell-centered study above.
    ops = Operators(prolongation=4, restriction=4)
    Ns = T === Float64 ? (8, 16, 32) : (D == 1 ? (16, 32) : (8, 16, 32))
    hs, l2 = T[], T[]
    for N in Ns
        r = wave_errors(Val(D); N=N, G=1, ops=ops, centering=vertexcentered(D),
                        T=T, backend=backend)
        push!(hs, r.h)
        push!(l2, r.l2)
        @test isfinite(r.l2)
        @test r.nblocks > 2^D                     # refinement really happened
    end
    @test all(l2[i] > l2[i + 1] for i in 1:(length(l2) - 1))
    @test convergence_rate(Float64.(hs), Float64.(l2)) ≈ 2.0 atol = 0.15
end
