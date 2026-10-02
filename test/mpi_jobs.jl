# The `mpiexec` jobs of the MPI test (M7), as a helper rather than a test:
# `runtests.jl` includes it before the first test, and `mpi_tests.jl`
# includes it if it was not.
#
# Each job runs `mpi_workload.jl` under `mpiexec` — three ranks, then two —
# and is compilation-bound: some 55 s of wall clock a job, against a few
# seconds of arithmetic (measured in step 9, "Suite cost" in CODE.md).
# Where the machine has room for both jobs beside the suite, they start
# here, at the start of the suite, and compile on otherwise idle cores
# while the rest of it runs; each then waits at its checkpoint loads for
# the serial reference that `mpi_tests.jl` writes (`TREEAMR_CHECKPOINT_FROM`
# in the workload), sleeping, not spinning in MPI. Elsewhere — a CI runner
# — nothing starts here, and `mpi_tests.jl` runs the three-rank job beside
# its serial reference and the two-rank job after it, as before step 9.

using MPI: MPI

const MPI_WORKLOAD = joinpath(@__DIR__, "mpi_workload.jl")

# Whether the two jobs may run at once and beside the suite: five
# single-threaded ranks, each compiling the workload, which measured
# about 2 GB resident a rank (2026-10-02), beside this process and the
# thread-independence test's subprocesses. A CI runner (3–4 cores, 7–16
# GB) does not qualify; `TREEAMR_TEST_MPI_CONCURRENT=0` or `1` decides
# it by hand.
function concurrent_launches()
    choice = get(ENV, "TREEAMR_TEST_MPI_CONCURRENT", "")
    choice == "1" && return true
    choice == "0" && return false
    return Sys.CPU_THREADS >= 8 && Sys.total_memory() >= 24 * 2^30
end

# The workload under `mpiexec -n $n`, started and not waited for, loading
# the files of the runs at the rank counts `from` once they are there,
# and failing if `timeout` seconds pass first.
function launch_workload(n, dir; from, threads=1, timeout=900)
    mpi = MPI.mpiexec()
    project = Base.active_project()
    cmd = `$mpi -n $n $(Base.julia_cmd()) --threads=$threads --project=$project
           $MPI_WORKLOAD mpi`
    cmd = addenv(setenv(cmd, mpi.env), "TREEAMR_CHECKPOINT_DIR" => dir,
                 "TREEAMR_CHECKPOINT_FROM" => join(from, " "),
                 "TREEAMR_CHECKPOINT_WAIT" => string(timeout))
    out, err = IOBuffer(), IOBuffer()
    proc = run(pipeline(cmd; stdout=out, stderr=err); wait=false)
    return (; n, proc, out, err, start=time(), timeout)
end

# Its output, with the deadline from its start: a rank that waited for a
# message never sent would otherwise hang the suite. A failed job takes
# the jobs in `others` down with it, since they may be waiting for its
# files.
function finish_workload(job; others=())
    (; n, proc, out, err, start, timeout) = job
    left = max(1.0, timeout - (time() - start))
    if timedwait(() -> process_exited(proc), left) !== :ok
        kill(proc)
        foreach(j -> kill(j.proc), others)
        error("the MPI workload at -n $n did not finish in $timeout s; its stderr:\n" *
              String(take!(err)))
    end
    if !success(proc)
        foreach(j -> kill(j.proc), others)
        error("the MPI workload at -n $n failed; its stderr:\n" * String(take!(err)))
    end
    return String(take!(out))
end

# The jobs started at the start of the suite, if any; `mpi_tests.jl` takes
# them. Their deadline covers the suite before `mpi_tests.jl` as well, and
# a suite that ends without taking them kills them.
const MPI_JOBS = Ref{Any}(nothing)

function start_mpi_jobs!()
    concurrent_launches() || return nothing
    dir = mktempdir()
    three = launch_workload(3, dir; from=(1,), timeout=3600)
    two = launch_workload(2, dir; from=(1, 3), timeout=3600)
    MPI_JOBS[] = (; dir, three, two)
    atexit() do
        for job in (three, two)
            process_running(job.proc) && kill(job.proc)
        end
    end
    return nothing
end

function take_mpi_jobs!()
    jobs = MPI_JOBS[]
    MPI_JOBS[] = nothing
    return jobs
end
