# Distributed meshes over MPI (M7, from step 3 on).
#
# The acceptance test of the exchange and the regrid: `test/mpi_workload.jl`
# runs a set of distributed cases — the vertex- and cell-centered wave
# equation on three-level meshes in D = 1, 2, 3, Burgers' equation with
# the interface fixup, a reflecting box with odd and even variables,
# every centering through one fill and one interface restriction, and a
# rank with no blocks; then (step 4) the tracked pulse and Burgers' shock
# through regrid cycles, the initial-data cycle from a single leaf, and
# regrids that move blocks between ranks both ways and coarsen siblings
# that had different owners, in three element types; then (step 5) point
# interpolation, each rank asking for its own slice of a global point
# list, and an outside point on one rank refused on all; a rank without
# blocks through every operation (added in 0.1.6); and (M12) a rotating
# quadrant: the wave in 2D and over a reflecting low face in 3D, a set
# that turns into itself and a `RotationPair` filled, regridded and
# interpolated beyond the seam, the pair's
# merged stages sharing their tags, and a single leaf at the axis that
# leaves every rank but one empty — and prints
# digests of the leaves, of the state and of the working arrays *with
# their ghosts*, gathered in block order, after every regrid, the exact
# reductions and the floating-point sums. Run serially in this
# process it is the reference; run under `mpiexec` at two and three
# ranks it must print the same lines, byte for byte, except the sums,
# which are promised to roundoff only ("Reductions" under "Distributed
# meshes" in CODE.md). A ghost that came from the wrong rank, the wrong
# slot or the wrong stage changes a digest.
#
# The ranks run one thread each: a CI runner has three or four cores,
# and MPICH polls while it waits, so two threads a rank at three ranks
# would oversubscribe it (HISTORY.md, "What an MPI test costs"). The
# launcher is `MPI.mpiexec()` of this process, so that the ranks load
# the MPI binary this process's preferences select — MPICH_jll by
# default, MPIABI_jll where a `LocalPreferences.toml` in the load path
# says so — and the project is the *active* one, as in `thread_tests.jl`,
# since under `Pkg.test` the tests run in a sandbox.
#
# The checkpoints, with their cross loads between rank counts, are
# TreeIOHDF5's, and its own MPI test runs them. Where the jobs start is
# `mpi_jobs.jl`'s: at the start of the suite where
# the machine has room for both jobs beside it. Otherwise both start
# here, beside the serial reference, where there is room for them (this
# file run on its own on a large machine), or one after the other, each
# after the run before it, where there is not (a CI runner). There the
# three ranks of about 2.4 GB each compiling beside the reference and
# the suite's own process outgrew a 7 GB macOS runner, and the job missed
# its 900 s deadline on Julia 1.13 (on `main` on 2026-10-04, and in the
# copy-kernel change's CI on 2026-10-06); one at a time costs the
# reference's minute or two of wall clock and keeps the three ranks from
# competing with it on one runner.

isdefined(@__MODULE__, :MPI_JOBS) || include("mpi_jobs.jl")

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

@testset "A distributed run prints what a serial run prints, at 2 and 3 ranks" begin
    early = take_mpi_jobs!()
    # Beside the serial reference only where there is room; on a small
    # machine the three ranks start once it is done (see the header).
    room = early !== nothing || concurrent_launches()
    three = early !== nothing ? early.three :
            room ? launch_workload(3) : nothing
    two = early !== nothing ? early.two :
          room ? launch_workload(2) : nothing
    serial = try
        serial_workload()
    catch
        foreach(j -> j === nothing || kill(j.proc), (three, two))
        rethrow()
    end
    three === nothing && (three = launch_workload(3))
    lines = workload_lines(serial)
    @test first(lines) == "# ranks 1"
    @test count(l -> occursin(" state ", l), lines) >= 7
    @test count(l -> occursin(" filled ", l), lines) >= 13
    @test any(l -> startswith(l, "B2 conserved true"), lines)
    # The regrid cycles (step 4) changed the mesh every time, Burgers'
    # conserved mass through them, and both initial-data cycles converged.
    @test count(l -> occursin(r"^\S+ regrid true ", l), lines) == 17
    @test "BR conserved true" in lines
    @test all(l -> split(l)[4] == "true", filter(startswith("A2 "), lines))
    @test count(l -> occursin(" unchanged false", l), lines) == 3
    # Point interpolation (step 5): every point answered, some flagged.
    @test count(startswith("I2"), lines) == 5
    @test all(l -> split(l)[3] == "301", filter(l -> occursin(" values ", l) &&
                                                     startswith(l, "I2"), lines))
    @test 0 < parse(Int, split(only(filter(startswith("I2.F64 excluded"), lines)))[3]) < 301
    words(prefix) = split(only(filter(startswith(prefix * " "), lines)))
    # Three levels wherever they were asked for.
    @test all(l -> split(l)[4] == "2", filter(l -> occursin(" leaves ", l) &&
                                                    !startswith(l, "W1p") &&
                                                    !startswith(l, "E2"), lines))
    # The ranks without blocks (E2): everything ran, and both initial-data
    # cycles refined and converged.
    @test all(l -> split(l)[5] == "true" && split(l)[6] == "5",
              filter(startswith("E2 adapted "), lines))
    # The rotating quadrant (M12): the turned ghosts hold `−0` in all
    # three sets, and three quarters of the points were turned back
    # across the seam.
    @test all(w -> parse(Int, w) > 0, words("Q2 negzero")[3:5])
    @test words("Q2 interpolated")[3:4] == ["257", "193"]
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

    outs = Dict{Int,String}()
    outs[3] = finish_workload(three; others=two === nothing ? () : (two,))
    two === nothing && (two = launch_workload(2))
    outs[2] = finish_workload(two)
    for n in (3, 2)
        out = outs[n]
        @test first(workload_lines(out)) == "# ranks $n"
        bad = workload_mismatches(serial, out)
        isempty(bad) || foreach(l -> println("mismatch at -n $n: ", l), bad)
        @test isempty(bad)
        hashes = filter(startswith("#"), workload_lines(out))
        refused(what) = only(filter(startswith("# $what refused on"), hashes))
        # E2's two leaves leave one of three ranks without a block, before
        # its regrids and after them.
        @test "# E2 empty ranks $(n == 3 ? 1 : 0)" in hashes
        @test "# E2 empty ranks after the regrids $(n == 3 ? 1 : 0)" in hashes
        # The rotating quadrant (M12): seam transfers crossed ranks, and the
        # single leaf left all ranks but one empty before its regrids and
        # after them, and none in between.
        @test parse(Int, split(only(filter(startswith("# Q2 rotated received"),
                                           hashes)))[end]) > 0
        @test "# QE2 empty ranks $(n - 1)" in hashes
        @test "# QE2 empty ranks after refining 0" in hashes
        @test "# QE2 empty ranks after coarsening $(n - 1)" in hashes
        # The negative control: one received ghost rewritten from a
        # corrupted message buffer changes the gathered digest.
        @test "# W2v perturbed-ghost-changes-digest true" in hashes
        # A forest mutated on one rank only, a layout that differs on one
        # rank, and an argument one rank alone refuses: refused on every
        # rank, saying why.
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
        @test "# device-aware shares the duplicate true" in hashes
        @test "# verbs agree on every rank true" in hashes
        @test startswith(refused("interface diverged"),
                         "# interface diverged refused on $n of $n ranks: the forest " *
                         "differs between ranks, so InterfaceSchedule is refused")
        # regrid!'s checks are agreed in the digest gather (step 4): a bad
        # flag box on rank 1, a flag vector of the wrong length on rank 0,
        # another `buffer` on rank 1, a forest refined on rank 1.
        @test startswith(refused("regrid box"), "# regrid box refused on $n of $n " *
                                                "ranks: regrid! was refused on rank(s) 1")
        @test startswith(refused("regrid length"), "# regrid length refused on $n of " *
                                                   "$n ranks: got ")
        @test occursin("one per local block", refused("regrid length"))
        @test startswith(refused("regrid buffer"), "# regrid buffer refused on $n of " *
                                                   "$n ranks: regrid! was called for a " *
                                                   "different layout on rank(s) 1")
        @test startswith(refused("regrid diverged"),
                         "# regrid diverged refused on $n of $n ranks: the forest differs " *
                         "between ranks, so regrid! is refused")
        @test "# regrid refusals left the forest alone true" in hashes
        # A point outside the domain on rank 1 only (step 5), refused on
        # every rank, rank 0 naming rank 1's point.
        @test startswith(refused("interpolate outside"),
                         "# interpolate outside refused on $n of $n ranks: interpolate is " *
                         "refused on every rank, this one (rank 0) included, since " *
                         "rank(s) 1 of $n passed a point outside the domain")
        @test occursin("On rank 1, point 2, (0.5, 9.0), is outside the domain",
                       refused("interpolate outside"))
        # M12: a point beyond the seam of a set that turns into its
        # partner, on rank 1 only, and such a set regridded alone on rank 1
        # where the others regrid the pair; refused on every rank.
        @test startswith(refused("rotating interpolate"),
                         "# rotating interpolate refused on $n of $n ranks: interpolate " *
                         "is refused on every rank, this one (rank 0) included, since " *
                         "rank(s) 1 of $n passed a point")
        @test occursin("On rank 1, point 2, (-0.5, 0.7), lies 1 quarter turn(s) away " *
                       "across the rotating seam", refused("rotating interpolate"))
        @test startswith(refused("rotating regrid"),
                         "# rotating regrid refused on $n of $n ranks: regrid! was " *
                         "refused on rank(s) 1 of $n")
        # What the regrid acceptance asks for actually happened at this
        # rank count: blocks moved up the ranks when the first ones were
        # refined and down again when they were coarsened, and a coarsened
        # block's children had more than one owner, in every element type.
        migrated(what) = parse.(Int, split(only(filter(startswith("# $what migrated"),
                                                     hashes)))[[4, 6, 8]])
        @test migrated("M4.refine")[1] > 0
        @test migrated("M4.coarsen")[2] > 0
        @test all(t -> migrated("$t.coarsen")[3] > 0, ("M2", "M32x2-2"))
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
