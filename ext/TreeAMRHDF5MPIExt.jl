# Parallel checkpoints (step 6 of M7): the one MPI-specific call a
# checkpoint of a distributed forest needs, opening the shared file
# through HDF5's MPI-IO driver.
#
# That driver is HDF5.jl's own MPI extension — `h5open(path, mode, comm,
# info)` and its `MPIO` file access — which loads only with MPI, so the
# call lives here, in an extension triggered by HDF5 and MPI together.
# Everything else about a parallel checkpoint is plain HDF5 and the
# package's communicator verbs, and lives in `TreeAMRHDF5Ext` beside the
# serial code it shares: which rank writes what, the hyperslabs, the
# collective transfers, the agreement before the file is opened and the
# durability after it is closed ("Parallel checkpoints" in CODE.md).

module TreeAMRHDF5MPIExt

using HDF5: HDF5, h5open
using MPI: MPI
import TreeAMR: open_parallel_file

# Collective. `comm` is the forest's duplicate of the application's
# communicator; HDF5 duplicates it once more for the file and frees that
# copy when the file is closed, so nothing here outlives the call.
function open_parallel_file(comm::MPI.Comm, path::AbstractString, mode::AbstractString)
    HDF5.has_parallel() || throw(ArgumentError(
        "the HDF5 library HDF5.jl loaded is not a parallel build, and a checkpoint of a " *
        "distributed forest is one file that every rank writes through MPI-IO. The " *
        "stock HDF5_jll is a parallel build for the MPI binary MPIPreferences selects; " *
        "a system HDF5 set with `HDF5.API.set_libraries!` must be built with MPI too."))
    return h5open(path, mode, comm, MPI.Info())
end

end
