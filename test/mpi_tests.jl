# Distributed meshes over MPI (M7, from step 3 on).
#
# The acceptance test of the exchange: `test/mpi_workload.jl` runs a set
# of distributed cases — the vertex- and cell-centered wave equation on
# three-level meshes in D = 1, 2, 3, Burgers' equation with the
# interface fixup, a reflecting box with odd and even variables, every
# centering through one fill and one interface restriction, and a rank
# with no blocks — and prints digests of the leaves, of the state and of
# the working arrays *with their ghosts*, gathered in block order, the
# exact reductions and the floating-point sums. Run serially in this
# process it is the reference; run under `mpiexec` at two and three
# ranks it must print the same lines, byte for byte, except the sums,
# which are promised to roundoff only ("Reductions" under "Distributed
# meshes" in CODE.md). A ghost that came from the wrong rank, the wrong
# slot or the wrong stage changes a digest.
#
# The ranks run one thread each: a CI runner has three or four cores,
# and MPICH polls while it waits, so two threads a rank at three ranks
# would oversubscribe it (CODE.md, "What an MPI test costs"). The
# launcher is `MPI.mpiexec()` of this process, so that the ranks load
# the MPI binary this process's preferences select — MPICH_jll by
# default, MPIABI_jll where a `LocalPreferences.toml` in the load path
# says so — and the project is the *active* one, as in `thread_tests.jl`,
# since under `Pkg.test` the tests run in a sandbox.

using MPI: MPI

const MPI_WORKLOAD = joinpath(@__DIR__, "mpi_workload.jl")

# The workload's lines, split into those that must agree byte for byte,
# the floating-point sums, and the rank-count-dependent `#` lines.
workload_lines(out) = split(out, '\n'; keepempty=false)

# Every line of `out` that does not match its counterpart in `ref`: the
# same line, or a `sum` line whose value agrees to `rtol`. `#` lines are
# not compared.
function workload_mismatches(ref, out; rtol=1e-12)
    a = filter(!startswith("#"), workload_lines(ref))
    b = filter(!startswith("#"), workload_lines(out))
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

# The serial reference, in this process: the script in a module of its
# own, printing into a buffer it finds there.
function serial_workload()
    m = Module(:MPIWorkloadSerial)
    io = IOBuffer()
    Core.eval(m, :(const WORKLOAD_IO = $io))
    Base.include(m, MPI_WORKLOAD)
    return String(take!(io))
end

# The workload under `mpiexec -n $n`, with a deadline: a rank that
# waited for a message never sent would otherwise hang the suite.
function mpi_workload(n; threads=1, timeout=900)
    mpi = MPI.mpiexec()
    project = Base.active_project()
    cmd = `$mpi -n $n $(Base.julia_cmd()) --threads=$threads --project=$project
           $MPI_WORKLOAD mpi`
    out, err = IOBuffer(), IOBuffer()
    proc = run(pipeline(setenv(cmd, mpi.env); stdout=out, stderr=err); wait=false)
    if timedwait(() -> process_exited(proc), timeout) !== :ok
        kill(proc)
        error("the MPI workload at -n $n did not finish in $timeout s; its stderr:\n" *
              String(take!(err)))
    end
    success(proc) || error("the MPI workload at -n $n failed; its stderr:\n" *
                           String(take!(err)))
    return String(take!(out))
end

@testset "A distributed run prints what a serial run prints, at 2 and 3 ranks" begin
    serial = serial_workload()
    lines = workload_lines(serial)
    @test first(lines) == "# ranks 1"
    @test count(l -> occursin(" state ", l), lines) >= 7
    @test count(l -> occursin(" filled ", l), lines) >= 13
    @test any(l -> startswith(l, "B2 conserved true"), lines)
    # Three levels wherever they were asked for.
    @test all(l -> split(l)[4] == "2", filter(l -> occursin(" leaves ", l) &&
                                                    !startswith(l, "W1p"), lines))
    # The comparison itself: a changed digest is caught, and so is a sum
    # that moved by more than roundoff — while the same sum to roundoff
    # is not.
    state = findfirst(l -> occursin(" state ", l), lines)
    flipped = replace(serial, lines[state] => lines[state][1:(end - 1)] *
                                              (lines[state][end] == '0' ? '1' : '0'))
    @test length(workload_mismatches(serial, flipped)) == 1
    mass = findfirst(l -> occursin(" sum mass ", l), lines)
    value = parse(Float64, split(lines[mass])[4])
    moved(x) = replace(serial, lines[mass] => join([split(lines[mass])[1:3]; string(x)],
                                                   " "))
    @test isempty(workload_mismatches(serial, moved(nextfloat(value, 3))))
    @test length(workload_mismatches(serial, moved(value * (1 + 1e-9)))) == 1

    for n in (2, 3)
        out = mpi_workload(n)
        @test first(workload_lines(out)) == "# ranks $n"
        bad = workload_mismatches(serial, out)
        isempty(bad) || foreach(l -> println("mismatch at -n $n: ", l), bad)
        @test isempty(bad)
        hashes = filter(startswith("#"), workload_lines(out))
        # The negative control: one received ghost rewritten from a
        # corrupted message buffer changes the gathered digest.
        @test "# W2v perturbed-ghost-changes-digest true" in hashes
        # A forest mutated on one rank only, a layout that differs on one
        # rank, and an argument one rank alone refuses: refused on every
        # rank, saying why.
        refused(what) = only(filter(startswith("# $what refused on"), hashes))
        @test startswith(refused("diverged"), "# diverged refused on $n of $n ranks: " *
                                              "the forest differs between ranks")
        @test occursin("rank(s) 1 hold a different one", refused("diverged"))
        @test occursin("collective", refused("diverged"))
        @test startswith(refused("layout"), "# layout refused on $n of $n ranks: " *
                                            "GhostSchedule was called for a different " *
                                            "layout on rank(s) 1")
        @test startswith(refused("partial"), "# partial refused on $n of $n ranks: " *
                                             "GhostSchedule was refused on rank(s) 1")
        @test "# one duplicate per communicator true" in hashes
        @test "# verbs agree on every rank true" in hashes
        @test startswith(refused("interface diverged"),
                         "# interface diverged refused on $n of $n ranks: the forest " *
                         "differs between ranks, so InterfaceSchedule is refused")
    end
end

@testset "An MPI communicator needs MPI initialized, and says so" begin
    # This process never initializes MPI — the ranks are subprocesses —
    # so a forest over `COMM_WORLD` here would duplicate a communicator
    # of a library that is not running.
    @test !MPI.Initialized()
    @test_throws "Call `MPI.Init()` before building the forest" Forest((2,); N=4,
                                                                       comm=MPI.COMM_WORLD)
end
