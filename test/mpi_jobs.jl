# The `mpiexec` jobs of the MPI test (M7), as a helper rather than a test:
# `runtests.jl` includes it before the first test, and `mpi_tests.jl`
# includes it if it was not.
#
# Each job runs `mpi_workload.jl` under `mpiexec` — three ranks, then two —
# and is compilation-bound: some 55 s of wall clock a job, against a few
# seconds of arithmetic (measured in M7 step 9, "Suite cost" in HISTORY.md).
# Where the machine has room for both jobs beside the suite, they start
# at the start of the process that runs `mpi_tests.jl` (the whole suite,
# or one group of it) and compile on otherwise idle cores while the rest
# of it runs. The jobs are independent of each other and
# of the serial reference. Elsewhere — a CI runner — nothing starts here,
# and `mpi_tests.jl` runs the serial reference, the three-rank job and
# the two-rank job one after the other, for the memory; on fewer than
# four CPU threads it skips the three-rank job (`three_ranks`).

using MPI: MPI

const MPI_WORKLOAD = joinpath(@__DIR__, "mpi_workload.jl")

# Whether the two jobs may run at once and beside the suite: five
# single-threaded ranks, each compiling the workload, which measured
# about 2 GB resident a rank with its heap-size hint (2026-11-16; 3 GB
# without), beside this process and the
# thread-independence test's subprocesses — and, where the suite runs in
# groups, beside the other groups' processes, which `test_group_count`
# allows only on a machine that passes this same test. A CI runner (3–4
# cores, 7–16 GB) does not qualify; `TREEAMR_TEST_MPI_CONCURRENT=0` or `1`
# decides it by hand.
function concurrent_launches()
    choice = get(ENV, "TREEAMR_TEST_MPI_CONCURRENT", "")
    choice == "1" && return true
    choice == "0" && return false
    return Sys.CPU_THREADS >= 8 && Sys.total_memory() >= 24 * 2^30
end

# Whether the three-rank job runs at all: only where the machine has a
# CPU thread beyond its three ranks. On CI's macOS runners (3 vCPUs of an
# M1) three ranks compiling at once got less than one core between them
# — 730 s against 222 s for two ranks, and past the 900 s deadline under
# coverage — while Ubuntu's 4 vCPUs ran them in about 240 s (2026-11-16).
# There the test compares the two-rank job alone, which Ubuntu's cells
# back with the three-rank one. `TREEAMR_TEST_MPI_THREE=0` or `1` decides
# it by hand.
function three_ranks()
    choice = get(ENV, "TREEAMR_TEST_MPI_THREE", "")
    choice == "1" && return true
    choice == "0" && return false
    return Sys.CPU_THREADS >= 4
end

# The workload under `mpiexec -n $n`, started and not waited for, and
# failing if `timeout` seconds pass first. Each rank gets a heap-size
# hint of 1 GB, which holds it to 2.0 GB resident in the same time,
# against 3.0 GB without (2026-11-16); nearly all of the rest is
# compiled code, which the hint cannot shrink (512 MB saves nothing more
# and costs 17 % in time). It was added for the macOS runners' 7 GB, but
# the timeouts there were short CPU, not memory (see `three_ranks`).
function launch_workload(n; threads=1, timeout=900)
    mpi = MPI.mpiexec()
    project = Base.active_project()
    cmd = `$mpi -n $n $(Base.julia_cmd()) --threads=$threads --heap-size-hint=1G
           --project=$project $MPI_WORKLOAD mpi`
    cmd = setenv(cmd, mpi.env)
    out, err = IOBuffer(), IOBuffer()
    proc = run(pipeline(cmd; stdout=out, stderr=err); wait=false)
    return (; n, proc, out, err, start=time(), timeout)
end

# The processes under `pid`, children first found by `pgrep -P`: `mpiexec`,
# MPICH's proxy and the ranks.
function descendants(pid)
    found = Int[]
    for line in eachline(ignorestatus(`pgrep -P $pid`))
        child = parse(Int, line)
        push!(found, child)
        append!(found, descendants(child))
    end
    return found
end

# What a job that ran out of time was doing, for the error: CPU, memory and
# age of each of its processes, which tells a hang (ranks spinning in MPI,
# or idle) from a slow run (ranks compiling), and each Julia rank's
# backtrace, which Julia writes to the job's stderr on SIGINFO (macOS, BSD)
# or SIGUSR1 (Linux), followed by a one-second profile.
function rank_state(proc)
    pids = descendants(getpid(proc))
    isempty(pids) && return "(none found)"
    table = read(ignorestatus(`ps -o pid,ppid,%cpu,rss,time,etime,command
                                  -p $(join(pids, ","))`), String)
    signal = Sys.isapple() || Sys.isbsd() ? "INFO" : "USR1"
    for line in split(table, '\n')[2:end]
        words = split(line)
        # By the binary's name, not its path: MPICH's `mpiexec` and proxy
        # live under `~/.julia`, and SIGUSR1 would end them.
        length(words) >= 7 && startswith(basename(words[7]), "julia") || continue
        run(ignorestatus(`kill -$signal $(words[1])`))
    end
    sleep(10)
    # The whole machine: whether the ranks were short of CPU, and to what.
    load = read(ignorestatus(`uptime`), String)
    bycpu = Sys.islinux() ? "--sort=-%cpu" : "-r"   # Linux's `-r` means running only
    top = join(first(split(read(ignorestatus(`ps -A -o pid,ppid,%cpu,rss,time,etime,command
                                                 $bycpu`), String), '\n'), 12), '\n')
    return table * "\nThe machine:\n" * load * top
end

# Its output, with the deadline from its start: a rank that waited for a
# message never sent would otherwise hang the suite. A failed job takes
# the jobs in `others` down with it, since the test fails anyway.
function finish_workload(job; others=())
    (; n, proc, out, err, start, timeout) = job
    left = max(1.0, timeout - (time() - start))
    if timedwait(() -> process_exited(proc), left) !== :ok
        state = rank_state(proc)
        kill(proc)
        foreach(j -> kill(j.proc), others)
        error("the MPI workload at -n $n did not finish in $timeout s.\n" *
              "Its processes before the kill:\n$state\n" *
              "Its output so far:\n" * String(take!(out)) *
              "\nIts stderr, with each rank's backtrace:\n" * String(take!(err)))
    end
    if !success(proc)
        foreach(j -> kill(j.proc), others)
        error("the MPI workload at -n $n failed; its stderr:\n" * String(take!(err)))
    end
    # How close it came to its deadline, in every CI cell's log.
    @info "The MPI workload at -n $n took $(round(Int, time() - start)) s of $timeout"
    return String(take!(out))
end

# The jobs started at the start of the suite, if any; `mpi_tests.jl` takes
# them. Their deadline covers the suite before `mpi_tests.jl` as well, and
# a suite that ends without taking them kills them.
const MPI_JOBS = Ref{Any}(nothing)

function start_mpi_jobs!()
    concurrent_launches() || return nothing
    three = three_ranks() ? launch_workload(3; timeout=3600) : nothing
    two = launch_workload(2; timeout=3600)
    MPI_JOBS[] = (; three, two)
    atexit() do
        for job in (three, two)
            job === nothing || process_running(job.proc) && kill(job.proc)
        end
    end
    return nothing
end

function take_mpi_jobs!()
    jobs = MPI_JOBS[]
    MPI_JOBS[] = nothing
    return jobs
end
