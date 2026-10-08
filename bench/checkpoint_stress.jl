# The multi-node checkpoint corruption of M7 step 6, reproduced at every
# layer from TreeAMR down to POSIX (HISTORY.md, "Parallel checkpoints" under
# "Parallelism", which has the jobs and the numbers); and, since
# step 6b replaced the shared file, the regression job for the writer
# that replaced it, whose files each have one writer and one opener.
#
#     srun ... julia --project=<env> bench/checkpoint_stress.jl <mode> <iters> <dir>
#
# run by bench/symmetry_checkpoint_stress.sh (MPICH_jll, srun) and
# bench/symmetry_checkpoint_stress_hpcx.sh (HPC-X's Open MPI, mpiexec).
# Every rank owns the rows of its blocks, as `blockrange` splits the
# benchmark's 12648 leaves (`STRESS_ROOTS`, default 12, at `STRESS_N`
# cells a block edge, default 4), and writes them into one shared file
# under `dir`, `iters` times over. After each write two readers on
# different nodes, rank 0 and the last rank, read the file serially and
# print a `BAD` line for each run of rows that is not what was written,
# with the ranks and nodes that own it.
#
# The modes, from the top down:
#
# - `treeamr`, `treeamr-filt`: `save_checkpoint` of a field set over the
#   forest, unfiltered and with Shuffle + Deflate(1), with the part files
#   of step 6b, `STRESS_IO` (`node`, the default, `all` or a number)
#   choosing the I/O processes. The readers load each save serially,
#   one after the other, so that each file still has one opener at a
#   time, and check the leaves and the data; a load the checksums refuse
#   is a `BAD` line too. (Until step 6b these modes wrote the shared file
#   and read its datasets directly.)
# - `hdf5`: the leaf columns and a data column as contiguous datasets,
#   each rank its hyperslab, collectively, through HDF5.jl alone.
# - `sieve-ind`, `sieve-coll`: the checkpoint's pattern below HDF5. The
#   coords column (12 bytes a row, at the failing file's offset 94549)
#   by contiguous collective writes; then a write whose file view is a
#   small piece before the column and the rest after it on rank 0 — as
#   HDF5 placed a filtered dataset's first chunks in free space before
#   the leaf columns — and contiguous elsewhere, independent or
#   collective. With data sieving, or with collective buffering, ROMIO
#   reads the whole extent, fills in its pieces and writes it all back.
# - `mpiio-ind`, `mpiio-coll`, `posix`: adjacent, unaligned, contiguous
#   ranges from every rank, by `MPI_File_write_at`, `_write_at_all` and
#   `pwrite`: no read-modify-write anywhere.
# - `visible`: no MPI-IO at all. Every rank writes its slab through its
#   own descriptor (`write(2)`, no `fsync`), all meet at a barrier, and
#   each rank then reads the previous rank's slab through its own
#   descriptor; an `UNSEEN` line says a completed write was not visible
#   to another rank yet.
#
# `STRESS_SYNC=1` adds `fsync` or `MPI_File_sync` on every rank before
# the close (in `sieve-*`, MPI's sync-barrier-sync between the two
# writes; in `treeamr*`, `sync = true`). ROMIO's hints come from the
# file `ROMIO_HINTS` names, which the job script sets; the `treeamr`
# modes use no MPI-IO since step 6b, so the hints do not reach them.

using MPI
MPI.Init()
using TreeAMR
const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)
const P = MPI.Comm_size(comm)
mode = ARGS[1]; iters = parse(Int, ARGS[2]); dir = ARGS[3]
# HDF5 only where a mode needs it, so that the modes below it run under
# an MPI for which there is no parallel HDF5 at hand (HPC-X's).
if startswith(mode, "treeamr") || mode == "hdf5"
    @eval using HDF5
    @eval using HDF5.Filters: Shuffle, Deflate
    @eval const API = HDF5.API
end
ROOTS = parse(Int, get(ENV, "STRESS_ROOTS", "12"))
NB = parse(Int, get(ENV, "STRESS_N", "4"))
SYNC = get(ENV, "STRESS_SYNC", "0") == "1"
IOSET = let io = get(ENV, "STRESS_IO", "node")
    io in ("node", "all") ? Symbol(io) : parse(Int, io)
end
const D = 3

function build_forest(r₀; comm)
    forest = Forest(ntuple(_ -> ROOTS, D); N=NB, extents=ntuple(_ -> (-1.0, 1.0), D),
                    comm=comm)
    for lvl in 0:1
        targets = filter(forest.leaves) do k
            level(k) == lvl || return false
            ext = block_extent(forest, k)
            w = ext[1][2] - ext[1][1]
            near = sqrt(sum(d -> max(ext[d][1], 0.0, -ext[d][2])^2, 1:D))
            far = sqrt(sum(d -> max(abs(ext[d][1]), abs(ext[d][2]))^2, 1:D))
            return near <= r₀ + w / 2 && far >= r₀ - w / 2
        end
        refine!(forest, targets)
        balance!(forest)
    end
    return forest
end

names = [strip(String(copy(c))) for c in eachcol(reshape(MPI.Allgather(collect(codeunits(rpad(gethostname(), 32)[1:32])), comm), 32, :))]
forest = build_forest(0.5; comm=comm)
n = nleaves(forest)
counts = MPI.Allgather(length(blockrange(forest)), comm)
starts = cumsum([0; counts[1:end-1]])
rankof(b) = searchsortedlast(starts, b - 1) - 1
nodeof(r) = names[r + 1]
readers = (0, P - 1)
rank == 0 && println("# ranks $P on ", join(unique(names), ","), " mode $mode iters $iters ",
                     "sync=$SYNC io=$IOSET hints=", get(ENV, "ROMIO_HINTS", "-"), " ",
                     isfile(get(ENV, "ROMIO_HINTS", "")) ? replace(read(ENV["ROMIO_HINTS"], String), "\n" => "; ") : "",
                     " leaves $n readers on ", nodeof.(readers))
want_root = Int32[k.root for k in forest.leaves]
want_level = Int8[k.level for k in forest.leaves]
want_coords = reduce(hcat, [UInt32[k.coords...] for k in forest.leaves])
per = NB^D
pattern(b) = UInt32(0x1000_0000 + b)
value(b, i) = Float64(b) * 2^20 + i
const BASE = 94549                         # the failing file's coords offset
const BASE2 = BASE + 12 * n + 7            # and an unaligned data column after it

function runs(bad)
    r = UnitRange{Int}[]
    for i in bad
        if !isempty(r) && last(r[end]) == i - 1
            r[end] = first(r[end]):i
        else
            push!(r, i:i)
        end
    end
    return r
end

function report(it, name, bad, off, el)
    isempty(bad) && return 0
    for r in runs(bad)
        rs = rankof(first(r)):rankof(last(r))
        b0 = off + (first(r) - 1) * el
        b1 = off + last(r) * el
        full = [starts[q+1]+1:starts[q+1]+counts[q+1] for q in rs]
        println("BAD reader=$rank it=$it $name rows $r ranks $rs nodes ",
                join(unique(nodeof.(rs)), ","), " bytes [$b0,$b1) mod4096 ",
                "$(b0 % 4096)..$(b1 % 4096); slabs ", join(full, ","))
    end
    flush(stdout)
    return length(bad)
end

fs = FieldSet(forest, 1; G=1, centering=cellcentered(D))
u = statevector(fs)
U = reshape(u, per, :)
r = blockrange(forest)
for (j, b) in enumerate(r), i in 1:per
    U[i, j] = value(b, i)
end
cbuf = [pattern(b) for _ in 1:3, b in r]

path = joinpath(dir, "stress-$mode.h5")
nbad = 0
t0 = time()
info = MPI.Info()
for it in 1:iters
    if mode in ("treeamr", "treeamr-filt")
        filters = mode == "treeamr" ? () : (Shuffle(), Deflate(1))
        save_checkpoint(path, forest; fieldsets=("U" => (fs, u),), application="s" => 1,
                        filters=filters, sync=SYNC, io=IOSET)
    elseif mode == "hdf5"
        h5open(path, "w", comm, info) do f
            for (name, w, T) in (("root", 1, Int32), ("level", 1, Int8),
                                 ("coords", 3, UInt32))
                dset = create_dataset(f, name, T, (w, n); dxpl_mpio=:collective)
                dset[:, r] = [T(name == "coords" ? pattern(b) : b % 100) for _ in 1:w, b in r]
                close(dset)
            end
            dset = create_dataset(f, "data", Float64, (per, n); dxpl_mpio=:collective)
            dset[:, r] = U
            close(dset)
            SYNC && API.h5f_flush(f, API.H5F_SCOPE_LOCAL)
        end
    elseif mode in ("mpiio-ind", "mpiio-coll")
        rank == 0 && rm(path; force=true)
        MPI.Barrier(comm)
        fh = MPI.File.open(comm, path; write=true, create=true)
        w! = mode == "mpiio-ind" ? MPI.File.write_at : MPI.File.write_at_all
        skip = rank == parse(Int, get(ENV, "STRESS_SKIP", "-1"))   # the checker's control
        w!(fh, BASE + (first(r) - 1) * 12, skip ? similar(cbuf, 0) : cbuf)
        w!(fh, BASE2 + (first(r) - 1) * per * 8, U)
        SYNC && MPI.File.sync(fh)
        close(fh)
    elseif mode in ("sieve-coll", "sieve-ind")
        # The checkpoint's pattern below HDF5: the coords column by
        # contiguous collective writes, then a write whose file view is
        # noncontiguous on rank 0 — a small piece before the column, the
        # rest after it, as the chunk allocator placed a filtered dataset's
        # first chunks in a free gap before the leaf columns — and
        # contiguous elsewhere. With data sieving rank 0 reads, modifies
        # and writes back the whole extent, the column included.
        rank == 0 && rm(path; force=true)
        MPI.Barrier(comm)
        fh = MPI.File.open(comm, path; write=true, create=true)
        MPI.File.write_at_all(fh, BASE + (first(r) - 1) * 12, cbuf)
        if SYNC                  # sync-barrier-sync, MPI's consistency construct
            MPI.File.sync(fh); MPI.Barrier(comm); MPI.File.sync(fh)
        end
        bytes = reinterpret(UInt8, vec(U))
        L = length(bytes)
        off = BASE2 + 4176 + (first(r) - 1) * per * 8
        lens, offs = rank == 0 ? (Cint[1438, L - 1438], [6672, off + 1438]) : (Cint[L], [off])
        ft = MPI.Types.create_struct(lens, Int.(offs), [MPI.BYTE for _ in lens])
        MPI.Types.commit!(ft)
        MPI.File.set_view!(fh, 0, MPI.BYTE, ft)
        (mode == "sieve-coll" ? MPI.File.write_all : MPI.File.write)(fh, collect(bytes))
        SYNC && MPI.File.sync(fh)
        close(fh)
    elseif mode == "visible"
        # Visibility without RMW: every rank writes its slab through its
        # own descriptor (write(2), no fsync), all wait at a barrier, and
        # each rank then reads the previous rank's slab — on another node
        # at a node boundary — through its own descriptor, opened before.
        rank == 0 && (rm(path; force=true); touch(path))
        MPI.Barrier(comm)
        io = open(path, "r+")
        seek(io, BASE + (first(r) - 1) * 12)
        write(io, cbuf)
        flush(io)                                  # to the kernel, not to disk
        MPI.Barrier(comm)
        q = mod(rank - 1, P)
        lo, m = starts[q+1], counts[q+1]
        seek(io, BASE + lo * 12)
        raw = read(io, 12m)          # short if the file's end is not yet visible here
        short = 12m - length(raw)
        got = reinterpret(UInt32, append!(raw, zeros(UInt8, short)))
        missing_ = count(j -> got[3(j-1)+1] != pattern(lo + j), 1:m)
        short > 0 && println("SHORT it=$it reader=$rank ($(nodeof(rank))): $short bytes past the end")
        if missing_ > 0
            println("UNSEEN it=$it reader=$rank ($(nodeof(rank))) of rank $q ($(nodeof(q))): ",
                    "$missing_ of $m rows not yet visible")
        end
        MPI.Barrier(comm)
        close(io)
    elseif mode == "posix"
        rank == 0 && (rm(path; force=true); touch(path))
        MPI.Barrier(comm)
        open(path, "r+") do io
            seek(io, BASE + (first(r) - 1) * 12)
            write(io, cbuf)
            seek(io, BASE2 + (first(r) - 1) * per * 8)
            write(io, U)
            flush(io)
            SYNC && ccall(:fsync, Cint, (Cint,), fd(io))
        end
    end
    MPI.Barrier(comm)
    if mode in ("treeamr", "treeamr-filt")
        # A serial load on each reader in turn: the index and every part,
        # their checksums verified, then the leaves and the data compared
        # with what was written. Byte offsets mean nothing across parts,
        # so the `BAD` lines give 0.
        for reader in readers
            if rank == reader
                local bad = 0
                ck = try
                    load_checkpoint(path)
                catch err
                    err isa ArgumentError || rethrow()
                    println("BAD reader=$rank it=$it refused: ",
                            first(split(err.msg, ". The file was written")))
                    flush(stdout)
                    nothing
                end
                if ck === nothing
                    bad += 1
                else
                    keys_ = ck.forest.leaves
                    b = findall(i -> keys_[i] != forest.leaves[i], 1:n)
                    bad += report(it, "leaves", b, 0, 17)
                    got = reshape(ck.fieldsets["U"].state, per, n)
                    b = findall(j -> any(i -> got[i, j] != value(j, i), 1:per), 1:n)
                    bad += report(it, "data", b, 0, per * 8)
                end
                global nbad += bad
            end
            MPI.Barrier(comm)
        end
    elseif rank in readers
        local bad = 0
        if mode == "hdf5"
            h5open(path, "r") do f
                for (name, el) in (("root", 4), ("level", 1), ("coords", 12))
                    got = read(f[name])
                    b = findall(j -> any(x -> x != (name == "coords" ? pattern(j) : j % 100),
                                         got[:, j]), 1:n)
                    bad += report(it, name, b, API.h5d_get_offset(f[name]), el)
                end
                got = read(f["data"])
                b = findall(j -> any(i -> got[i, j] != value(j, i), 1:per), 1:n)
                bad += report(it, "data", b, API.h5d_get_offset(f["data"]), per * 8)
            end
        elseif mode == "visible"
            bytes = read(path)
            got = reshape(reinterpret(UInt32, bytes[BASE+1:BASE+12n]), 3, n)
            b = findall(j -> any(!=(pattern(j)), got[:, j]), 1:n)
            bad += report(it, "coords", b, BASE, 12)
        elseif startswith(mode, "sieve")
            bytes = read(path)
            got = reshape(reinterpret(UInt32, bytes[BASE+1:BASE+12n]), 3, n)
            b = findall(j -> any(!=(pattern(j)), got[:, j]), 1:n)
            bad += report(it, "coords", b, BASE, 12)
            # the data, rank 0's first piece at 6672
            want = reinterpret(UInt8, [value(j, i) for i in 1:per, j in 1:n])
            have = vcat(bytes[6672+1:6672+1438], bytes[BASE2+4176+1438+1:BASE2+4176+8*per*n])
            b = findall(j -> have[(j-1)*per*8+1:j*per*8] != want[(j-1)*per*8+1:j*per*8], 1:n)
            bad += report(it, "data", b, BASE2 + 4176, per * 8)
        else
            bytes = read(path)
            length(bytes) >= BASE2 + n * per * 8 ||
                println("SHORT reader=$rank it=$it file has $(length(bytes)) bytes")
            bytes = resize!(bytes, max(length(bytes), BASE2 + n * per * 8))
            got = reshape(reinterpret(UInt32, bytes[BASE+1:BASE+12n]), 3, n)
            b = findall(j -> any(!=(pattern(j)), got[:, j]), 1:n)
            bad += report(it, "coords", b, BASE, 12)
            got = reshape(reinterpret(Float64, bytes[BASE2+1:BASE2+8*per*n]), per, n)
            b = findall(j -> any(i -> got[i, j] != value(j, i), 1:per), 1:n)
            bad += report(it, "data", b, BASE2, per * 8)
        end
        global nbad += bad
    end
    MPI.Barrier(comm)
end
total = MPI.Allreduce(nbad, +, comm)
rank == 0 && println("# done mode $mode sync=$SYNC: $iters iterations, $total bad rows ",
                     "seen by the two readers, ", round(time() - t0; digits=1), " s")
MPI.Finalize()
