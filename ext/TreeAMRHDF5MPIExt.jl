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

# The MPI-IO hints every parallel checkpoint is opened with: no
# read-modify-write, ever. ROMIO (MPICH's MPI-IO, and the one MPICH_jll
# ships) knows no BeeGFS and treats it as a generic POSIX file system,
# whose writes it takes to be visible to every node at once. Under that
# assumption it turns a write into a read-modify-write of a wider range
# in two places: data sieving, for an independent write whose file view
# is noncontiguous — the chunks of a filtered dataset, of which HDF5 may
# place the first few in free space before the leaf columns — reads the
# whole extent, fills in its pieces and writes it all back; and
# collective buffering writes each aggregator's file domain whole, after
# reading it first if anything in it is not being written. Both write
# back bytes that other ranks wrote earlier, as they were when read. On
# BeeGFS, whose client buffers a node's writes until a flush, a write
# another node has made but not yet flushed reads as the zeros it
# replaced, and the write-back destroys it: on Symmetry's BeeGFS a rank's
# whole slab of the leaf coordinates came back zero (M7 step 6, in
# "Parallel checkpoints" in CODE.md, with the reproducers). Disabling
# both makes every rank write exactly its own bytes and nothing else,
# which no write-back caching can turn into a loss. The cost is nothing
# measurable here: a rank's blocks are one contiguous run of the file,
# or a few runs of whole chunks, so there is nothing for either to
# aggregate. Other MPI-IO implementations ignore the `romio_` keys, as
# the standard has them ignore any key they do not know.
const NO_READ_MODIFY_WRITE = (:romio_ds_write => "disable", :romio_cb_write => "disable")

# Collective. `comm` is the forest's duplicate of the application's
# communicator; HDF5 duplicates it once more for the file and frees that
# copy when the file is closed, so nothing here outlives the call.
function open_parallel_file(comm::MPI.Comm, path::AbstractString, mode::AbstractString)
    HDF5.has_parallel() || throw(ArgumentError(
        "the HDF5 library HDF5.jl loaded is not a parallel build, and a checkpoint of a " *
        "distributed forest is one file that every rank writes through MPI-IO. The " *
        "stock HDF5_jll is a parallel build for the MPI binary MPIPreferences selects; " *
        "a system HDF5 set with `HDF5.API.set_libraries!` must be built with MPI too."))
    return h5open(path, mode, comm, MPI.Info(NO_READ_MODIFY_WRITE...))
end

end
