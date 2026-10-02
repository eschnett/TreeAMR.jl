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
# blocks through every operation (added after 0.1.5); then (steps 6
# and 6b) a run checkpointed after a regrid, with a part file per rank,
# per I/O group and per node, and continued from each, and the files of
# the earlier runs, written at other rank counts, loaded and continued —
# and prints
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
# would oversubscribe it (CODE.md, "What an MPI test costs"). The
# launcher is `MPI.mpiexec()` of this process, so that the ranks load
# the MPI binary this process's preferences select — MPICH_jll by
# default, MPIABI_jll where a `LocalPreferences.toml` in the load path
# says so — and the project is the *active* one, as in `thread_tests.jl`,
# since under `Pkg.test` the tests run in a sandbox.
#
# The checkpoints go to one directory for every run: a file saved at
# three ranks is loaded at two, at one (over `MPI.COMM_SELF` inside the
# two-rank run) and serially, and the serial file at three; each
# continuation must print the uninterrupted run's digests. A run that
# loads another's files waits for that run's marker
# (`TREEAMR_CHECKPOINT_FROM`), so the runs need not follow one another.
# Where they start is `mpi_jobs.jl`'s: at the start of the suite where
# the machine has room for both jobs beside it. Otherwise the three-rank
# job starts here, beside the serial reference, and the two-rank job
# beside it too where there is room (this file run on its own), or after
# it (a CI runner).

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
# own, printing into a buffer it finds there. It loads no other run's
# files.
function serial_workload(dir)
    m = Module(:MPIWorkloadSerial)
    io = IOBuffer()
    Core.eval(m, :(const WORKLOAD_IO = $io))
    withenv("TREEAMR_CHECKPOINT_DIR" => dir, "TREEAMR_CHECKPOINT_FROM" => "") do
        Base.include(m, MPI_WORKLOAD)
    end
    return String(take!(io)), m, io
end

# The two digests at the end of the `#` line that starts with `prefix`.
function digests_of(lines, prefix)
    hits = filter(startswith(prefix * " "), lines)
    length(hits) == 1 || return nothing
    return split(only(hits))[(end - 1):end]
end

@testset "A distributed run prints what a serial run prints, at 2 and 3 ranks" begin
    early = take_mpi_jobs!()
    dir = early === nothing ? mktempdir() : early.dir
    three = early === nothing ? launch_workload(3, dir; from=(1,)) : early.three
    two = early !== nothing ? early.two :
          concurrent_launches() ? launch_workload(2, dir; from=(1, 3)) : nothing
    serial, workload, io = try
        serial_workload(dir)
    catch
        foreach(j -> j === nothing || kill(j.proc), (three, two))
        rethrow()
    end
    lines = workload_lines(serial)
    @test first(lines) == "# ranks 1"
    @test count(l -> occursin(" state ", l), lines) >= 7
    @test count(l -> occursin(" filled ", l), lines) >= 13
    @test any(l -> startswith(l, "B2 conserved true"), lines)
    # The regrid cycles (step 4) changed the mesh every time, Burgers'
    # conserved mass through them, and both initial-data cycles converged.
    @test count(l -> occursin(r"^\S+ regrid true ", l), lines) == 14
    @test "BR conserved true" in lines
    @test all(l -> split(l)[4] == "true", filter(startswith("A2 "), lines))
    @test count(l -> occursin(" unchanged false", l), lines) == 3
    # Point interpolation (step 5): every point answered, some flagged.
    @test count(startswith("I2"), lines) == 5
    @test all(l -> split(l)[3] == "301", filter(l -> occursin(" values ", l) &&
                                                     startswith(l, "I2"), lines))
    @test 0 < parse(Int, split(only(filter(startswith("I2.F64 excluded"), lines)))[3]) < 301
    # Checkpoints (steps 6 and 6b): the restarted runs, from each of the
    # three files, load what was saved and end where the uninterrupted run
    # ends, on a mesh that changed after the save; and the version-1
    # fixtures load (their lines are compared with the distributed runs'
    # below, as every line is).
    words(prefix) = split(only(filter(startswith(prefix * " "), lines)))
    uninterrupted, saved = words("C uninterrupted")[3:4], words("C saved")[3:4]
    @test count(startswith("V1 "), lines) == 2
    for name in ("restarted", "restarted-filtered", "restarted-node")
        @test words("C $name loaded")[4:5] == saved
        @test words("C $name continued")[4:5] == uninterrupted
    end
    leaves = words("C mesh")
    @test leaves[3] != leaves[5]
    @test words("C1 loaded")[3] == words("C1 saved")[3]
    # Three levels wherever they were asked for.
    @test all(l -> split(l)[4] == "2", filter(l -> occursin(" leaves ", l) &&
                                                    !startswith(l, "W1p") &&
                                                    !startswith(l, "E2"), lines))
    # The ranks without blocks (E2): everything ran, the checkpoint came
    # back as saved, and both initial-data cycles refined and converged.
    @test words("E2 checkpoint")[3:4] == ["2", "true"]
    @test all(l -> split(l)[5] == "true" && split(l)[6] == "5",
              filter(startswith("E2 adapted "), lines))
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

    # A file another run wrote, at another rank count, loaded at `at` and
    # continued: it must load what was saved and end where the
    # uninterrupted run ends.
    function crossed(hashes, from, at)
        for suffix in ("", "-filtered", "-node")
            @test digests_of(hashes, "# C-n$from$suffix at $at loaded") == saved
            @test digests_of(hashes, "# C-n$from$suffix at $at continued") == uninterrupted
        end
        c1 = filter(startswith("# C1-n$from at $at loaded "), hashes)
        @test length(c1) == 1 && split(only(c1))[end] == words("C1 saved")[3]
    end

    outs = Dict{Int,String}()
    outs[3] = finish_workload(three; others=two === nothing ? () : (two,))
    two === nothing && (two = launch_workload(2, dir; from=(1, 3)))
    outs[2] = finish_workload(two)
    for n in (3, 2)
        out = outs[n]
        @test first(workload_lines(out)) == "# ranks $n"
        bad = workload_mismatches(serial, out)
        isempty(bad) || foreach(l -> println("mismatch at -n $n: ", l), bad)
        @test isempty(bad)
        hashes = filter(startswith("#"), workload_lines(out))
        refused(what) = only(filter(startswith("# $what refused on"), hashes))
        # The distributed checkpoints (steps 6 and 6b): the files record
        # the rank count and the number of parts — one a rank, two groups,
        # one node; at three ranks one rank holds none of the small
        # forest's two blocks, when saving and loading; the serial file
        # loads at three ranks, and the three-rank files at two and at one.
        @test "# C restarted nranks $n nparts $n" in hashes
        @test "# C restarted-filtered nranks $n nparts 2" in hashes
        @test "# C restarted-node nranks $n nparts 1" in hashes
        @test "# C1 empty ranks $(n == 3 ? 1 : 0)" in hashes
        # E2's two leaves leave one of three ranks without a block, before
        # its regrids and after them.
        @test "# E2 empty ranks $(n == 3 ? 1 : 0)" in hashes
        @test "# E2 empty ranks after the regrids $(n == 3 ? 1 : 0)" in hashes
        for from in (n == 3 ? (1,) : (1, 3)), at in (n, 1)
            crossed(hashes, from, at)
        end
        # Refused on every rank before the file is created: plain data
        # that differ between ranks, an argument only rank 1's checks
        # refuse, filters that differ on rank 1, and in the do-block a
        # write_plain of a value that differs; then a load whose
        # arguments differ on rank 1, and one of a file that is not
        # there, which rank 0 alone looks for.
        @test startswith(refused("checkpoint data"),
                         "# checkpoint data refused on $n of $n ranks: the plain data " *
                         "of save_checkpoint differ between ranks: rank(s) 1")
        @test startswith(refused("checkpoint partial"),
                         "# checkpoint partial refused on $n of $n ranks: " *
                         "save_checkpoint was refused on rank(s) 1")
        @test startswith(refused("checkpoint layout"),
                         "# checkpoint layout refused on $n of $n ranks: " *
                         "save_checkpoint was called for a different layout on rank(s) 1")
        @test startswith(refused("checkpoint write_plain"),
                         "# checkpoint write_plain refused on $n of $n ranks: the plain " *
                         "data of write_plain of /Refused/mine differ between ranks")
        @test "# checkpoint refusals left the file alone true" in hashes
        @test startswith(refused("load layout"),
                         "# load layout refused on $n of $n ranks: load_checkpoint was " *
                         "called for a different layout on rank(s) 1")
        @test startswith(refused("load missing"), "# load missing refused on $n of $n " *
                                                  "ranks: there is no checkpoint at")
        # Damage in the last part alone, refused by the checksums on every
        # rank; a part of another save, and a missing part, refused on
        # every rank before any data move (step 6b).
        @test startswith(refused("checkpoint damage"),
                         "# checkpoint damage refused on $n of $n ranks: the data of " *
                         "field set \"u\" do not match the checksums stored with them " *
                         "in 1 of its 16 blocks, the first being block 16")
        @test startswith(refused("checkpoint foreign part"),
                         "# checkpoint foreign part refused on $n of $n ranks:")
        @test occursin("belongs to another save", refused("checkpoint foreign part"))
        @test startswith(refused("checkpoint missing part"),
                         "# checkpoint missing part refused on $n of $n ranks:")
        @test occursin("is missing, so the checkpoint is refused",
                       refused("checkpoint missing part"))
        # Orphans removed, and nothing else; an I/O process that fails
        # mid-save takes the save down on every rank and leaves the
        # previous checkpoint as it was; and no file opened by two
        # processes, saving or loading.
        @test "# checkpoint orphans removed true" in hashes
        failed = only(filter(startswith("# checkpoint failed part on"), hashes))
        @test startswith(failed, "# checkpoint failed part on $n of $n ranks:")
        @test occursin("writing the parts", failed) || occursin("failed on purpose", failed)
        @test "# checkpoint failed part left the previous one true" in hashes
        @test "# checkpoint each file opened by one process true true" in hashes
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
    # The files of both distributed runs, loaded serially in this process.
    withenv("TREEAMR_CHECKPOINT_DIR" => dir, "TREEAMR_CHECKPOINT_FROM" => "2 3") do
        Base.invokelatest(workload.checkpoint_cross, "C")
    end
    hashes = workload_lines(String(take!(io)))
    for from in (2, 3)
        crossed(hashes, from, 1)
    end
    # The index's external links lead tools into the parts (step 6b).
    index = joinpath(dir, "C-n3.h5")
    @test HDF5.h5open(index) do file
        id = HDF5.read_attribute(file["TreeAMR.jl"], "save_id")
        all(0:2) do j
            HDF5.read_attribute(file["TreeAMR.jl/parts/$(lpad(j, 4, '0'))"], "save_id") ==
            id
        end
    end
end

@testset "An MPI communicator needs MPI initialized, and says so" begin
    # This process never initializes MPI — the ranks are subprocesses —
    # so a forest over `COMM_WORLD` here would duplicate a communicator
    # of a library that is not running.
    @test !MPI.Initialized()
    # The parallel-HDF5 extension of step 6 is gone (step 6b): with HDF5
    # and MPI both loaded, only their own extensions are.
    @test Base.get_extension(TreeAMR, :TreeAMRHDF5MPIExt) === nothing
    @test Base.get_extension(TreeAMR, :TreeAMRHDF5Ext) !== nothing
    @test_throws "Call `MPI.Init()` before building the forest" Forest((2,); N=4,
                                                                       comm=MPI.COMM_WORLD)
end
