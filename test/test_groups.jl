# The test files, and how the suite splits them over processes ("Parallel
# groups" in CODE.md). Every test file runs on its own after the helper
# files, so any split of them is a correct one: the weights below only
# balance it.

# The helper files every test process includes, in this order: the
# oracles and the two test applications first, then the definitions split
# off from test files (`helpers/`), which call nothing at include time.
const TEST_HELPERS = ["oracles.jl", "ghost_oracles.jl", "wave.jl", "burgers.jl",
                      "helpers/ghost.jl", "helpers/centering.jl", "helpers/reflect.jl",
                      "helpers/rotate.jl", "helpers/interpolate.jl",
                      "helpers/partition.jl", "helpers/exchange.jl",
                      "helpers/regrid_exchange.jl", "helpers/state.jl",
                      "helpers/walls.jl", "helpers/patch.jl"]

# Every test file in the order the serial suite runs them, with its
# seconds in that suite at one thread (an M3 Pro, 2026-11-05). The weights
# need refreshing only when the groups come out badly unbalanced. The MPI
# test is weighted by its two `mpiexec` jobs, which run beside its group
# for about a minute each, rather than by the 18 s of its own process:
# that makes it and the thread-independence test, whose two subprocesses
# are as heavy, the two heaviest entries, so that they start in different
# groups (`pack_groups` checks it).
const TEST_FILES = ["tree_tests" => 14, "ghost_tests" => 56, "centering_tests" => 32,
                    "element_tests" => 57, "reflect_tests" => 30,
                    "outflow_tests" => 11, "rotate_tests" => 50,
                    "halfturn_tests" => 35, "patch_tests" => 40,
                    "interpolate_tests" => 22,
                    "interface_tests" => 4, "partition_tests" => 6,
                    "exchange_tests" => 43, "regrid_exchange_tests" => 36,
                    "interpolate_exchange_tests" => 17, "allvariables_tests" => 6,
                    "state_tests" => 10, "fold_tests" => 8, "regrid_tests" => 14,
                    "bounded_tests" => 36, "staggered_conservative_tests" => 8,
                    "walls_tests" => 10, "relative_tests" => 6, "wave_tests" => 3,
                    "wave_cell_tests" => 2, "burgers_tests" => 15, "imex_tests" => 5,
                    "type_tests" => 15, "thread_tests" => 67, "mpi_tests" => 70,
                    "gpu_tests" => 47]

# How many processes the suite runs in: `TREEAMR_TEST_GROUPS` if set (1 is
# the serial suite), and otherwise by the rule `concurrent_launches` in
# `mpi_jobs.jl` applies to the MPI jobs, which run beside the groups: a
# machine with 8+ threads and 24+ GB splits the suite, anything smaller —
# a CI runner — runs it in one process, as it always has. Five is where
# the measured wall time stopped improving (CODE.md, "Tests").
function test_group_count(; cpus=Sys.CPU_THREADS, memory=Sys.total_memory(), env=ENV)
    choice = get(env, "TREEAMR_TEST_GROUPS", "")
    if !isempty(choice)
        n = tryparse(Int, choice)
        n !== nothing && n >= 1 ||
            throw(ArgumentError("TREEAMR_TEST_GROUPS must be a positive integer, the " *
                                "number of processes to run the suite in; got $choice"))
        return n
    end
    cpus >= 8 && memory >= 24 * 2^30 || return 1
    return min(5, cpus ÷ 2)
end

# The files split into `n` groups: heaviest first, each into the group
# with the least weight so far, ties to the lowest group; within a group
# the files keep the serial order. Deterministic, so a failure reproduces
# in the same group.
function pack_groups(n)
    n >= 1 || throw(ArgumentError("a suite runs in at least one group; got $n"))
    order = sortperm(last.(TEST_FILES); rev=true, alg=MergeSort)
    loads = zeros(Int, n)
    members = [Int[] for _ in 1:n]
    for i in order
        g = argmin(loads)
        push!(members[g], i)
        loads[g] += last(TEST_FILES[i])
    end
    groups = [first.(TEST_FILES[sort(m)]) for m in members if !isempty(m)]
    if length(groups) > 1 &&
       any(g -> "thread_tests" in g && "mpi_tests" in g, groups)
        error("the thread-independence and MPI tests landed in one group; their " *
              "subprocesses would then compete with each other rather than run beside " *
              "the rest — reweight TEST_FILES so that they are the two heaviest")
    end
    return groups
end

# The test files named on the command line, with or without `.jl` and a
# `test/` prefix, checked against the list so that a typo fails at once
# rather than after the helpers have loaded.
function selected_test_files(args)
    names = [replace(basename(a), r"\.jl$" => "") for a in args]
    known = first.(TEST_FILES)
    for n in names
        n in known ||
            throw(ArgumentError("$n is not a test file of this suite; the files are " *
                                join(known, ", ")))
    end
    return names
end

# Runs each group in a process of its own, all at once, and reports each
# as it finishes: its files, its wall time and its output, which goes to a
# file the child writes directly so that stdout and stderr interleave as
# on a terminal. A group passes if its process exits successfully, which
# a failing or erroring test prevents. The children get this process's
# thread counts — `Base.julia_cmd()` carries `--check-bounds` and
# `--code-coverage` but not the threads — and its project.
function run_test_groups(groups)
    project = Base.active_project()
    threads = "$(Threads.nthreads(:default)),$(Threads.nthreads(:interactive))"
    script = joinpath(@__DIR__, "runtests.jl")
    jobs = map(enumerate(groups)) do (i, files)
        path, io = mktemp()
        cmd = `$(Base.julia_cmd()) --threads=$threads --project=$project $script $files`
        proc = run(pipeline(cmd; stdout=io, stderr=io); wait=false)
        return (; i, files, proc, path, io, start=time())
    end
    atexit() do
        for j in jobs
            process_running(j.proc) && kill(j.proc)
        end
    end
    finished = Channel{Any}(length(jobs))
    for j in jobs
        errormonitor(@async (wait(j.proc); put!(finished, j)))
    end
    for _ in jobs
        j = take!(finished)
        close(j.io)
        seconds = round(Int, time() - j.start)
        println("\n── Group $(j.i) of $(length(jobs)), $seconds s: ", join(j.files, " "))
        print(read(j.path, String))
        flush(stdout)          # before the summary's stderr, under `Pkg.test`
        rm(j.path; force=true)
        @test success(j.proc)
    end
    return nothing
end
