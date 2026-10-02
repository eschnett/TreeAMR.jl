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
# read-modify-write in ROMIO, the MPI-IO of MPICH and of MPICH_jll. ROMIO
# knows no BeeGFS and drives it as a generic POSIX file system (its "UFS"
# driver), and turns a write into a read-modify-write of a wider range in
# two places: data sieving, for an independent write whose file view is
# noncontiguous — the chunks of a filtered dataset, of which HDF5 may
# place the first few in free space before the leaf columns — and
# collective buffering, which writes each aggregator's file domain whole,
# after reading it if anything in it is not being written. Data sieving
# holds an `fcntl` write lock over the extent while it does so; the
# collective path's read of holes takes none outside atomic mode (MPICH
# 5.0's `ad_write_str.c` and `ad_write_coll.c`). On generic POSIX that is
# safe, because a write that has returned is visible to every reader and
# a lock excludes every other locker. On Symmetry's BeeGFS neither holds:
# the client buffers a node's writes until a flush ("buffered" cache), so
# another node's read misses them, and `tuneUseGlobalFileLocks = false`
# makes `fcntl` locks local to a node. A read-modify-write then reads
# another node's completed but unflushed write as the zeros it replaced
# and writes them back, and a whole slab of a rank's leaf coordinates was
# lost that way (M7 step 6, "Parallel checkpoints" in CODE.md, with the
# reproducers; the write that was destroyed took no lock, so global
# locks alone would not have saved it). With both disabled every rank
# writes exactly its own bytes. The cost is in filtered saves of data
# that compress well: each chunk becomes its own write, and a blast-wave
# save with zstd(1) on two nodes ran at three quarters of its rate
# without them; unfiltered saves did not change. Open MPI's own MPI-IO, OMPIO, ignores the `romio_` keys, as
# the standard has it ignore any key it does not know, so this does not
# cover it, and it was not tested.
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
