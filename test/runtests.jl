# The test suite. With no arguments it runs every test file, in one process
# or, where the machine has room, split over a few processes at once
# (`test_groups.jl`; "Parallel groups" in CODE.md). With test file names as
# arguments it runs only those, in this process:
#
#     julia --project=test test/runtests.jl exchange_tests regrid_tests
#
# which is also how each group's process is started.

using Test
using Random
using TreeAMR

include("test_groups.jl")
foreach(include, TEST_HELPERS)
include("mpi_jobs.jl")

const SELECTED = isempty(ARGS) ? first.(TEST_FILES) : selected_test_files(ARGS)
const NGROUPS = isempty(ARGS) ? test_group_count() : 1

@info "Running the tests on $(Threads.nthreads()) thread(s)" *
      (NGROUPS > 1 ? " in $NGROUPS processes" : "")

if NGROUPS > 1
    @testset "TreeAMR.jl" begin
        run_test_groups(pack_groups(NGROUPS))
    end
else
    # The MPI test's `mpiexec` jobs start now where the machine has room
    # for them beside the rest, and are collected by `mpi_tests.jl` (M7).
    "mpi_tests" in SELECTED && start_mpi_jobs!()
    @testset "TreeAMR.jl" begin
        for file in SELECTED
            include("$file.jl")
        end
    end
end
