# The MPI+GPU acceptance check (step 8 of M7), run by hand: not part of
# `Pkg.test`, since the test environment has no device package and must
# not gain one (CI has no GPU). It runs `mpi_device_workload.jl` serially
# and under `mpiexec` at each rank count, every run on the same backend,
# and requires every line of a distributed run to be the serial run's,
# byte for byte, except the `sum` lines, which agree to the element
# type's roundoff. It also reads the `#` lines to check that the path it
# meant to test is the one that ran: through host mirrors on a device
# (every stage with messages staged), directly with
# `TREEAMR_TEST_DEVICEAWARE=1` or on the CPU (none staged); and that the
# regrids moved blocks both ways and coarsened siblings with different
# owners.
#
# In a scratch environment that develops this checkout and adds MPI,
# KernelAbstractions, SHA and the device package (Metal here, CUDA on
# Symmetry; see `bench/symmetry_mpi_gpu.sh`):
#
#     TREEAMR_TEST_BACKEND=metal julia --project=<env> test/mpi_device_tests.jl
#     TREEAMR_TEST_BACKEND=cuda TREEAMR_TEST_RANKS="2 4" \
#         julia --project=<env> test/mpi_device_tests.jl
#
# `TREEAMR_TEST_RANKS` lists the rank counts (default "2 3"). The
# launcher is `MPI.mpiexec()` of the environment's MPI binary.

using Test
using MPI: MPI

const WORKLOAD = joinpath(@__DIR__, "mpi_device_workload.jl")
const RANKCOUNTS = parse.(Int, split(get(ENV, "TREEAMR_TEST_RANKS", "2 3")))

function workload(n; timeout=1800)
    project = Base.active_project()
    julia = `$(Base.julia_cmd()) --threads=1 --project=$project $WORKLOAD`
    cmd = if n == 0
        julia
    else
        mpi = MPI.mpiexec()
        setenv(`$mpi -n $n $julia mpi`, mpi.env)
    end
    # The launcher's environment carries its library paths, which `setenv`
    # keeps; the backend, the type and the device-awareness setting are
    # added to it.
    cmd = addenv(cmd, [k => v for (k, v) in ENV if startswith(k, "TREEAMR_")]...)
    out, err = IOBuffer(), IOBuffer()
    start = time()
    proc = run(pipeline(cmd; stdout=out, stderr=err); wait=false)
    if timedwait(() -> process_exited(proc), timeout) !== :ok
        kill(proc)
        error("the device workload at -n $n did not finish in $timeout s:\n" *
              String(take!(err)))
    end
    success(proc) || error("the device workload at -n $n failed:\n" * String(take!(err)))
    @info "device workload" ranks = n seconds = round(time() - start; digits=1)
    return split(String(take!(out)), '\n'; keepempty=false)
end

function mismatches(ref, out, rtol)
    a = filter(!startswith("#"), ref)
    b = filter(!startswith("#"), out)
    length(a) == length(b) || return ["$(length(a)) lines against $(length(b))"]
    bad = String[]
    for (x, y) in zip(a, b)
        wx, wy = split(x), split(y)
        same = if length(wx) == 4 && wx[2] == "sum"
            wx[1:3] == wy[1:3] &&
                isapprox(parse(Float64, wx[4]), parse(Float64, wy[4]); rtol=rtol)
        else
            x == y
        end
        same || push!(bad, "$x  vs  $y")
    end
    return bad
end

# The numbers on the `#` lines whose words start with `words`.
numbers(lines, words...) =
    [parse.(Int, filter(w -> all(isdigit, w), split(l)[(length(words) + 2):end]))
     for l in lines if split(l)[2:(length(words) + 1)] == collect(words)]

@testset "A distributed device run prints what a serial device run prints" begin
    serial = workload(0)
    header = split(first(serial))
    @test header[1:3] == ["#", "ranks", "1"]
    backend, T, deviceaware = header[5], header[7], header[9] == "true"
    rtol = T == "Float32" ? 1e-5 : 1e-12
    @info "reference" backend T deviceaware lines = length(serial)
    @test count(l -> occursin(" state ", l), serial) >= 6
    @test "B2 conserved true" in serial
    @test "M conserved true" in serial
    for n in RANKCOUNTS
        out = workload(n)
        @test split(first(out))[1:3] == ["#", "ranks", string(n)]
        bad = mismatches(serial, out, rtol)
        isempty(bad) || foreach(l -> @info("mismatch at -n $n", l), bad)
        @test isempty(bad)

        # Every stage with messages went through host mirrors on a device
        # that is not handed to MPI directly, and none did otherwise.
        stages = [l for l in out if occursin(" stages messages ", l)]
        @test length(stages) == 5
        for l in stages
            w = split(l)
            messages, staged = parse(Int, w[end - 2]), parse(Int, w[end])
            @test messages > 0
            @test staged == (backend != "cpu" && !deviceaware ? messages : 0)
        end
        moved(name) = only(numbers(filter(startswith("# M"), out), name, "migrated"))
        @test moved("M4.refine")[1] > 0              # blocks moved up the ranks
        @test moved("M4.coarsen")[2] > 0             # and back down
        @test moved("M2.coarsen")[3] > 0             # siblings with different owners

        # The second regrid cycle took every buffer and mirror from the
        # pool, which kept the first cycle's; mirrors exactly where staged.
        for name in ("M4", "M2")
            pool = numbers(filter(startswith("# $name pool"), out), name, "pool")
            @test length(pool) == 2
            (r1, a1, m1), (r2, a2, m2) = pool
            @test (r1, r2) == (1, 2)
            @test a1 > 0 && a2 == a1 && m2 == m1
            @test (m1 > 0) == (backend != "cpu" && !deviceaware)
        end
    end
end
