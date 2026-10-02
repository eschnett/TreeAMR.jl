# Checkpoint and restart through HDF5 (M9a; without parallel I/O, M7
# step 6b): the implementation of the functions declared, and documented,
# in `src/checkpoint.jl`.
#
# The file layout is format version 2 of `CODE.md`, "Checkpoint and
# restart", which is the specification this follows object for object,
# and the reader reads version 1 too:
#
#     path                         the index, written by rank 0
#       /TreeAMR.jl/               format, format_version, features,
#                                  application, save_id
#         provenance/              who wrote the file, and in which environment
#         forest/                  the brick, the extents, the leaves as columns
#         fieldsets/<name>/        the layout of each field set
#         parttable/               every part's file, block range, size, checksums
#         parts/<jjjj>             an external link to each part file, for tools;
#                                  with one part, the part itself
#       /<application>/            format_version (the application's own)
#         data                     the plain-data tree
#     path.<saveid>.<j>.h5         part j, written by its I/O process
#       /TreeAMR.jl/fieldsets/<name>/data, data_crc32c
#
# Two rules shape everything below. The file holds bits and a small
# documented vocabulary of names, never a Julia type's definition, so
# that a reader in another version, or in another language, can
# interpret it; and a load builds fresh objects through the package's
# own validating constructors, so that nothing read from a file is
# trusted before the forest and the field set have checked it.
#
# And over a distributed forest a third ("Checkpoints without parallel
# I/O" under "Distributed meshes" in CODE.md): every file has exactly one
# writer, and on reading exactly one opener, so HDF5 is only ever used
# serially and nothing depends on MPI-IO or on a file system's coherence
# between nodes. The data travel between the ranks as messages instead:
# to their group's I/O process when saving, and from a part's one reader
# to the blocks' owners when loading. A serial checkpoint is the
# degenerate case, one group of one rank, whose part lives inside the
# index, so it is still one file.
#
# Nothing here loops over cells on the host. The field data go to and
# from HDF5 a range of whole blocks at a time, straight from and into the
# state vector on the CPU — whose pages `statevector` has already placed
# by owner — and through one host buffer on a device. The per-leaf work,
# splitting keys into columns and building them back, runs through
# `threaded_foreach`, by owner.

module TreeAMRHDF5Ext

using HDF5: HDF5, API, h5open, create_group, create_dataset, dataspace, attributes,
            read_attribute, write_attribute, write_dataset, copy_object, create_external
using KernelAbstractions: CPU, Backend, get_backend
using CRC32c: crc32c
using TreeAMR
using TreeAMR: threaded_foreach, tohost, samebackend, Communicator, commrank, commsize,
               allgather, allgatherv, bcast, commnodes, isend, irecv, waitall,
               equalsplit, equalsplit_part, ForestDigest, digest_verdict, layouthash
import TreeAMR: save_checkpoint, load_checkpoint, write_plain, read_plain,
                checkpoint_environment

const GROUP = "TreeAMR.jl"
const FORMAT = "TreeAMR checkpoint"
const PART_FORMAT = "TreeAMR checkpoint part"
const FORMAT_VERSION = 2
# The versions this reader reads: version 1 is the single file of M9a and
# of step 6's shared-file writer, read as one part inline.
const READ_VERSIONS = (1, 2)
# Every feature listed in a file must be understood by its reader; one
# that may be ignored is simply not listed (Zarr v3's `must_understand`).
const FEATURES = ("brick",)

# --- element types -------------------------------------------------------

const NativeInteger = Union{Int8,UInt8,Int16,UInt16,Int32,UInt32,Int64,UInt64}
const NativeFloat = Union{Float16,Float32,Float64}
const NativeReal = Union{Bool,NativeInteger,NativeFloat}
const NativeNumber = Union{NativeReal,Complex{<:NativeReal}}

const NATIVE_REALS = (Bool, Int8, UInt8, Int16, UInt16, Int32, UInt32, Int64, UInt64,
                      Float16, Float32, Float64)
const NATIVE_TYPES = (NATIVE_REALS..., map(T -> Complex{T}, NATIVE_REALS)...)

# A type's name as a module that imports nothing but Base prints it:
# `Float64`, `ComplexF32`, `MultiFloats.MultiFloat{Float32, 2}`. Not
# `string(T)`, which qualifies a name or not according to what the
# *writer* happened to have imported into `Main` — `using MultiFloats`
# prints `MultiFloat{Float32, 2}`, `import MultiFloats` does not — so a
# file would record a different name for the same type depending on how
# the application was started, and the match on load would fail.
module Names end
typename(T::Type) = sprint(show, T; context=:module => Names)

const NATIVE_BY_NAME = Dict{String,Type}(typename(T) => T for T in NATIVE_TYPES)

# The HDF5 type of a native element type, as a fresh `Datatype` for the
# caller to close. A `Bool` is spelled `UInt8`, 0 or 1: HDF5.jl would
# write it as an HDF5 bitfield, which other readers handle poorly.
# `Float16` is the IEEE half type HDF5.jl does not predefine, built the
# way h5py builds it, so the bits are stored as they are.
h5type(::Type{Bool}) = HDF5.Datatype(API.h5t_copy(HDF5.hdf5_type_id(UInt8)))
h5type(::Type{T}) where {T<:Union{NativeInteger,Float32,Float64}} =
    HDF5.Datatype(API.h5t_copy(HDF5.hdf5_type_id(T)))
function h5type(::Type{Float16})
    id = API.h5t_copy(HDF5.hdf5_type_id(Float32))
    API.h5t_set_fields(id, 15, 10, 5, 0, 10)
    API.h5t_set_size(id, 2)
    API.h5t_set_ebias(id, 15)
    return HDF5.Datatype(id)
end
# A complex number is the compound `(r, i)`, HDF5.jl's and h5py's
# spelling, over the part type's own HDF5 type.
function h5type(::Type{Complex{T}}) where {T<:NativeReal}
    id = API.h5t_create(API.H5T_COMPOUND, 2 * sizeof(T))
    withtype(T) do part
        API.h5t_insert(id, "r", 0, part)
        API.h5t_insert(id, "i", sizeof(T), part)
    end
    return HDF5.Datatype(id)
end

function withtype(f, ::Type{T}) where {T}
    dt = h5type(T)
    try
        return f(dt)
    finally
        close(dt)
    end
end

# The one native type an `isbits` type is made of throughout, with no
# padding — `Float32` for MultiFloats' `Float32x2`, which is an
# `NTuple{2,Float32}` of limbs — and `nothing` for any other type. A
# `Bool` and a `Complex` are not limbs: a type made of them flattens to
# their parts, or is refused.
function limb_type(::Type{T}) where {T}
    T <: Union{NativeInteger,NativeFloat} && return T
    (isstructtype(T) && isconcretetype(T) && fieldcount(T) > 0) || return nothing
    F = nothing
    size = 0
    for i in 1:fieldcount(T)
        S = fieldtype(T, i)
        L = limb_type(S)
        (L === nothing || (F !== nothing && L !== F)) && return nothing
        F = L
        size += sizeof(S)
    end
    # All fields of one type, whose alignment is its size, and summing
    # to the whole: then there is no padding between or after them.
    return size == sizeof(T) ? F : nothing
end

function limbs_of(::Type{T}) where {T}
    isbitstype(T) || return nothing
    F = limb_type(T)
    F === nothing && return nothing
    return (F, sizeof(T) ÷ sizeof(F))
end

# How the elements of `T` are stored: `(F, limbs)`, the HDF5 element
# type and the leading dimension — `()` for a native type, `(n,)` for
# one made of `n` limbs of `F`. The limb dimension leads, so a value's
# limbs are adjacent in the file as they are in memory, and an array of
# `T` is written from its own memory with no reinterpretation.
function storage_of(::Type{T}, what) where {T}
    isconcretetype(T) && T <: NativeNumber && return (T, ())
    limbs = limbs_of(T)
    limbs === nothing && throw(ArgumentError(
        "$what has element type $T, which a checkpoint cannot store: it is neither an " *
        "HDF5 native type (Bool, the signed and unsigned integers, Float16, Float32, " *
        "Float64, or Complex of those) nor an isbits type made of one native type " *
        "throughout with no padding, which is stored as limbs (MultiFloats' Float32x2 " *
        "as two Float32). A file stores bits and a type's name, never its definition, " *
        "so no other type could be read back exactly."))
    return (limbs[1], (limbs[2],))
end

const ELTYPE_ATTRS = ("eltype", "limbtype", "nlimbs")
const GEOMETRY_ATTRS = ("geometry_type", "geometry_limbtype", "geometry_nlimbs")

function write_typeattrs(obj, ::Type{T}, names) where {T}
    F, limbs = storage_of(T, "a checkpoint")
    write_attribute(obj, names[1], typename(T))
    if !isempty(limbs)
        write_attribute(obj, names[2], typename(F))
        write_attribute(obj, names[3], Int64(limbs[1]))
    end
    return nothing
end

# The element type that the attributes `names` of `obj` describe, as
# `(T, F, limbs)` in the sense of `storage_of`. A native type is named
# by the file alone. A limb type is not: the reader cannot name a type
# without the package that defines it, so it must be among `types`,
# matched by name, and then its layout must be the file's.
function read_typeattrs(obj, names, types, what, context)
    name = read_attribute(obj, names[1])
    if !hasattr(obj, names[2])
        T = get(NATIVE_BY_NAME, name, nothing)
        T === nothing && throw(ArgumentError(
            "$what is stored in $name, which is neither an HDF5 native type nor " *
            "recorded with limbs, so this version cannot interpret it. " *
            writer_note(context)))
        return (T, T, ())
    end
    limbname = read_attribute(obj, names[2])
    n = Int(read_attribute(obj, names[3]))
    F = get(NATIVE_BY_NAME, limbname, nothing)
    (F isa Type && F <: Union{NativeInteger,NativeFloat} && n >= 1) ||
        throw(ArgumentError(
            "$what is stored as $n limbs of $limbname, which is not a native number " *
            "type this version reads limbs of. " * writer_note(context)))
    matches = [T for T in types if typename(T) == name]
    given = isempty(types) ? "no types were given" :
            "the types given were " * join(typename.(types), ", ")
    isempty(matches) && throw(ArgumentError(
        "$what is stored in $name, as $n $limbname limbs per value, and this reader " *
        "cannot name that type without the package that defines it: load the package " *
        "and pass the type in `types`, as in `load_checkpoint(path; types = (T,))` " *
        "($given). Types are matched by name; the limbs themselves are plain " *
        "$limbname, which is what a converter, or a reader in another language, sees. " *
        writer_note(context)))
    T = first(matches)
    limbs_of(T) == (F, n) || throw(ArgumentError(
        "the type given in `types` for $what, $T, does not match the file's: it has " *
        "sizeof $(sizeof(T)) and is $(describe_limbs(T)), where the file stores $n " *
        "$limbname limbs per value. A type of the same name with another layout would " *
        "read different numbers from the same bits. " * writer_note(context)))
    return (T, F, (n,))
end

function describe_limbs(::Type{T}) where {T}
    limbs = limbs_of(T)
    limbs === nothing && return "not made of limbs of one native type"
    return "$(limbs[2]) limbs of $(limbs[1])"
end

function check_types(types)
    types isa Type && (types = (types,))
    all(T -> T isa Type, types) || throw(ArgumentError(
        "`types` must be a collection of types, such as `(Float32x2,)`; got " *
        repr(types)))
    return Tuple(types)
end

# --- the ranks, and the test hooks (M7) ----------------------------------------
#
# The processes a save or a load runs on: the forest's communicator, or
# the one `load_checkpoint` is given, with this rank and the count read
# once. Serially it is rank 0 of 1, and every collective below returns at
# once.
struct Ranks
    comm::Communicator
    rank::Int
    size::Int
end

Ranks(comm::Communicator) = Ranks(comm, commrank(comm), commsize(comm))

# The message tags of a checkpoint's own messages, beside the exchange's
# (`schedule.jl`): a member's per-block checksums and its blocks, sent to
# its I/O process, and a part's blocks, sent by their reader to their
# owner. Within one tag, messages between two ranks match in the order
# they were posted, which both ends follow.
const TAG_HEAD = 60
const TAG_SAVE = 61
const TAG_LOAD = 62

# Three hooks for the tests, never set otherwise (M7 step 6b):
#
# - `OPEN_LOG`: when it holds a vector, every file this process opens —
#   an HDF5 file, or a file opened to flush it — is recorded in it by its
#   absolute path, so that the MPI test can check that no file of a save
#   or a load is opened by more than one process.
# - `FAIL_PART`: the number of the part whose I/O process fails on
#   purpose after its first write, so that the test can check what a
#   failed save leaves behind.
# - `MAX_MESSAGE`: the largest message, in bytes, that the gathering of a
#   save and the scattering of a load send, short of one block. 64 MiB,
#   well under the 1 GiB the design allows, so that the next message
#   arrives while the current one is written or read; small in the tests,
#   so that a rank's blocks travel in several messages.
const OPEN_LOG = Ref{Union{Nothing,Vector{String}}}(nothing)
const FAIL_PART = Ref(-1)
const MAX_MESSAGE = Ref(64 * 2^20)

function note_open(path)
    log = OPEN_LOG[]
    log === nothing || push!(log, abspath(path))
    return nothing
end

# Open the HDF5 file `path`, recording it.
function open_h5(path, mode)
    note_open(path)
    return h5open(path, mode)
end

# An HDF5 file in memory only (the core driver without a backing store):
# the image of an index that rank 0 broadcasts, and the scratch file in
# which the other ranks run a save's do-block. No file is created. Each
# gets a name of its own, since HDF5 tells open files apart by name.
const MEMORY_FILES = Threads.Atomic{Int}(0)

memory_file() = h5open("TreeAMR in-memory file $(Threads.atomic_add!(MEMORY_FILES, 1))",
                       "w"; driver=HDF5.Drivers.Core(; backing_store=false))

# --- agreement (M7) ------------------------------------------------------------
#
# Over a distributed forest every refusal is decided before anything is
# created, and agreed: one `allgather` of each rank's verdict and of a
# hash of what the ranks must pass alike, so that a refusal on some ranks
# is raised on all of them together, with the reason, rather than leaving
# the others waiting in a message that never comes. `ForestDigest` and
# `digest_verdict` are the schedule builds' (`forest.jl`); a load, which
# has no forest yet, gathers a digest with the forest's fields zeroed, as
# `interpolate` does. Beside it goes the hash of the plain data, which
# gets a refusal of its own.
function agreed(check, r::Ranks, what; forest=nothing)
    r.size == 1 && return first(check())
    checked, refusal = try
        check(), nothing
    catch err
        err isa ArgumentError || rethrow()
        nothing, err
    end
    layout, datahash = checked === nothing ? (UInt(0), UInt(0)) : checked[2:3]
    digest = forest === nothing ?
             ForestDigest(0, 0, UInt(0), UInt(0), layout, refusal !== nothing) :
             ForestDigest(forest, layout, refusal !== nothing)
    gathered = allgather(r.comm, (digest, datahash))
    digest_verdict(map(first, gathered), what, r.rank, refusal)
    differ = [q - 1 for q in 2:r.size if last(gathered[q]) != last(gathered[1])]
    isempty(differ) || throw(ArgumentError(
        "the plain data of $what differ between ranks: rank(s) $(join(differ, ", ")) " *
        "of $(r.size) pass other values than rank 0, so it is refused on every rank, " *
        "this one (rank $(r.rank)) included. A checkpoint is written by rank 0, with one " *
        "value of each item for every rank; a value that is per rank belongs in a field " *
        "set, or is gathered to every rank first."))
    return first(checked)
end

# The errors of a step that runs on some ranks only — an I/O process
# writing its part, a reader checking or reading its parts, rank 0
# writing and committing the index — agreed after it, so that every rank
# throws together. `err` is this rank's error, or `nothing`. A rank that
# failed throws its own; every other rank throws one quoting the first
# failing rank's, an `ArgumentError` if that was one (a refusal, such as
# a part from another save) and an `ErrorException` otherwise.
function agree_errors(r::Ranks, err, what)
    if r.size == 1
        err === nothing || throw(err)
        return nothing
    end
    text = err === nothing ? "" : err isa ArgumentError ? err.msg : sprint(showerror, err)
    bytes = collect(codeunits(text))
    flags = allgather(r.comm, (err !== nothing, err isa ArgumentError, length(bytes)))
    failed = [q - 1 for q in 1:r.size if flags[q][1]]
    isempty(failed) && return nothing
    texts = allgatherv(r.comm, bytes)
    err === nothing || throw(err)
    q = first(failed)
    offset = sum(last, flags[1:q]; init=0)
    quoted = String(texts[offset+1:offset+last(flags[q+1])])
    message = "$what failed on rank(s) $(join(failed, ", ")) of $(r.size), and so on " *
              "this one (rank $(r.rank)) too. On rank $q: " * quoted
    flags[q+1][2] ? throw(ArgumentError(message)) : error(message)
end

# --- checksums (M7) ----------------------------------------------------------
#
# Every array the forest and the field sets are restored from carries a
# CRC-32C, verified on load: the leaf list one over its three columns,
# each field set one per block, and the index one per part and field set
# over that part's per-block checksums ("Checksums" in the file layout
# under "Checkpoint and restart" in CODE.md). They were added after a
# checkpoint written by 32 ranks on four nodes of a cluster came back with
# one rank's leaf coordinates zeroed, a corruption the leaf list's own
# validation caught only because it broke the curve order; the same loss
# in a field set's data would have been restored silently. A checksum is
# over the bytes as the file stores them — little-endian, the limbs of a
# limb type in order — so a reader in another language can verify it. A
# version-1 file without them, written before they existed, is read
# without the check. HDF5's own Fletcher-32 is not used: among other
# reasons it accepts an all-zero chunk, trailer included, which is exactly
# the damage that was seen.

# The CRC-32C of `count` elements of the array `buf` from index `first`
# on, continuing `crc`.
function crc_of(buf::Array, first::Integer, count::Integer, crc::UInt32=UInt32(0))
    count == 0 && return crc
    GC.@preserve buf begin
        bytes = unsafe_wrap(Array, Ptr{UInt8}(pointer(buf, first)),
                            count * sizeof(eltype(buf)))
        return crc32c(bytes, crc)
    end
end

crc_of(buf::Array) = crc_of(buf, 1, length(buf))

# One CRC-32C per block of `buf` from element `start` on, which holds `m`
# blocks of `per` elements each in state-vector layout: the order of
# `data`, block `b` being the slice `data[…, b]`.
function block_checksums(buf::Array, m::Integer, per::Integer, start::Integer=1)
    sums = Vector{UInt32}(undef, m)
    threaded_foreach(m) do j
        sums[j] = crc_of(buf, start + (j - 1) * per, per)
    end
    return sums
end

# The leaf list's CRC-32C: over `root`, then `level`, then `coords`, each
# column whole.
leaves_checksum(roots, levels, coords) =
    crc_of(coords, 1, length(coords),
           crc_of(levels, 1, length(levels), crc_of(roots, 1, length(roots))))

# A checksum that fails is refused on every rank together: each reader
# has checked the blocks it read, so the verdicts are gathered. `bad` are
# the failing blocks, in the whole forest's numbering; the result is how
# many there are over all ranks and the first.
function damage(r::Ranks, bad)
    r.size == 1 && return (length(bad), isempty(bad) ? 0 : first(bad))
    gathered = allgather(r.comm, (length(bad), isempty(bad) ? 0 : first(bad)))
    total = sum(first, gathered)
    firsts = [last(g) for g in gathered if first(g) > 0]
    return (total, isempty(firsts) ? 0 : minimum(firsts))
end

# --- datasets --------------------------------------------------------------

hasattr(obj, name) = haskey(attributes(obj), name)

# The `type` attribute of a plain-data item, and any others beside it.
function tag!(obj, tag; attrs...)
    write_attribute(obj, "type", tag)
    for (key, value) in attrs
        write_attribute(obj, string(key), value)
    end
    return obj
end

# Create the dataset `name` of HDF5 element type `F` and dimensions
# `dims`, in Julia's order, and write `buf` into it whole unless `buf` is
# `nothing` (a dataset over blocks that is filled by `write_blocks`).
# `buf` may be an array of any isbits type whose bytes are `F`s, since
# HDF5 checks only the byte count: an array of a limb type is written from
# its own memory, with no copy (a `reinterpret` would cost one, because
# HDF5.jl copies a `ReinterpretArray` to take a pointer to it). A
# plain-data item passes its `tag`; the TreeAMR group's own datasets carry
# none, since the specification describes them. Returns the dataset open
# when `keep`, for the caller to fill and close.
function write_array(parent, name, ::Type{F}, dims::Dims, buf; chunk=nothing,
                     filters=(), tag=nothing, keep=false, attrs...) where {F}
    withtype(F) do dt
        space = dataspace(dims)
        props = isempty(filters) ? (;) : (; chunk=chunk, filters=filters)
        dset = try
            create_dataset(parent, name, dt, space; props...)
        finally
            close(space)
        end
        try
            buf === nothing || prod(dims) == 0 || HDF5.write_dataset(dset, dt, buf)
            tag === nothing || tag!(dset, tag; attrs...)
        catch
            close(dset)
            rethrow()
        end
        keep || close(dset)
        return keep ? dset : nothing
    end
end

# Write or read the blocks `rows` (of the dataset's last axis, counted
# from 1) of `dset`, whose dimensions are `dims` in Julia's order, from or
# into the memory at `ptr`, which holds exactly those blocks, in the HDF5
# element type `F`. HDF5's order is Julia's reversed, so the block axis
# comes first in the selection. Serial: one process holds the file.
function block_slab(transfer, dset, ::Type{F}, dims::Dims, rows::UnitRange{Int},
                    ptr::Ptr) where {F}
    isempty(rows) && return nothing
    withtype(F) do dt
        filespace = HDF5.dataspace(dset)
        memspace = HDF5.dataspace((dims[1:(end - 1)]..., length(rows)))
        try
            start = API.hsize_t[first(rows) - 1; zeros(Int, length(dims) - 1)]
            counts = API.hsize_t[length(rows); reverse(collect(dims[1:(end - 1)]))]
            API.h5s_select_hyperslab(filespace, API.H5S_SELECT_SET, start, C_NULL, counts,
                                     C_NULL)
            transfer(dset, dt, memspace, filespace, API.H5P_DEFAULT, ptr)
        finally
            close(memspace)
            close(filespace)
        end
    end
    return nothing
end

# A dataset is read only once its shape and its element type are the
# ones the layout says: a damaged file could otherwise have HDF5 convert
# its numbers silently, or overrun the buffer.
function check_dataset(dset, ::Type{F}, dims, what, note) where {F}
    size(dset) == dims || throw(ArgumentError(
        "$what has dimensions $(size(dset)) where the layout needs $dims: the file is " *
        "damaged, or was not written by TreeAMR. " * note))
    ft = HDF5.datatype(dset)
    ok = try
        withtype(F) do mt
            API.h5t_get_class(ft) == API.h5t_get_class(mt) && sizeof(ft) == sizeof(mt)
        end
    finally
        close(ft)
    end
    ok || throw(ArgumentError(
        "$what is not stored as $(typename(F)), which its attributes say it is: the " *
        "file is damaged, or was not written by TreeAMR. " * note))
    return nothing
end

function read_array!(buf, dset, ::Type{F}, dims, what, note) where {F}
    check_dataset(dset, F, dims, what, note)
    prod(dims) == 0 && return buf
    withtype(F) do dt
        HDF5.read_dataset(dset, dt, buf)
    end
    return buf
end

function read_array(dset, ::Type{T}, dims, what, note="") where {T}
    if T === Bool
        raw = read_array!(Array{UInt8}(undef, dims), dset, UInt8, dims, what, note)
        all(<=(0x01), raw) || throw(ArgumentError(
            "$what holds a Bool that is neither 0 nor 1: the file is damaged. " * note))
        return convert(Array{Bool}, raw)
    end
    return read_array!(Array{T}(undef, dims), dset, T, dims, what, note)
end

function read_scalar(dset, ::Type{T}, what) where {T}
    check_dataset(dset, T, (), what, "")
    buf = Ref{T === Bool ? UInt8 : T}()
    withtype(T) do dt
        API.h5d_read(dset, dt, API.H5S_ALL, API.H5S_ALL, API.H5P_DEFAULT, buf)
    end
    T === Bool || return buf[]
    buf[] <= 0x01 || throw(ArgumentError(
        "$what holds a Bool that is neither 0 nor 1: the file is damaged"))
    return buf[] == 0x01
end

# An HDF5 link name: nonempty, not `.`, and with no `/`, which separates
# the groups of a path.
function check_name(name::AbstractString, what)
    (isempty(name) || name == "." || occursin('/', name)) && throw(ArgumentError(
        "$what, $(repr(name)), cannot name an HDF5 object: a name is nonempty, is not " *
        "\".\", and contains no \"/\", which separates the groups of a path"))
    return nothing
end

# The forest's flag arrays are spelled `UInt8` 0 or 1, like every `Bool`
# in the file.
function read_flags(obj, name, dims, note)
    raw = read_attribute(obj, name)
    (raw isa AbstractArray{<:Integer} && size(raw) == dims &&
     all(x -> x == 0 || x == 1, raw)) || throw(ArgumentError(
        "the forest's $name, $(repr(raw)), is not a $(join(dims, "×")) array of 0 " *
        "and 1: the file is damaged. " * note))
    return raw
end

# Every attribute of `src`, written to `dst` as HDF5.jl reads it — which
# is how the reader reads it back.
function copy_attributes(src, dst)
    for name in keys(attributes(src))
        write_attribute(dst, name, read_attribute(src, name))
    end
    return nothing
end

# --- provenance ----------------------------------------------------------

# Who wrote the file: rank 0, whose values these are (`created`,
# `hostname` and `nthreads` can differ between ranks). M7 added two
# fields, `nranks` (decided 2026-10-01 with Erik) and `nparts`, the number
# of part files (step 6b); a file without them was written serially, and
# as one file.
const PROVENANCE = ("treeamr_version", "julia_version", "created", "hostname", "nthreads",
                    "nranks", "nparts", "project", "manifest")

function provenance_values(nranks, nparts)
    project, manifest = environment_texts()
    version = pkgversion(TreeAMR)
    return (version === nothing ? "" : string(version), string(VERSION), utc_now(),
            gethostname(), Int64(Threads.nthreads()), Int64(nranks), Int64(nparts),
            project, manifest)
end

function write_provenance(root, values)
    g = create_group(root, "provenance"; track_order=true)
    try
        for (name, value) in zip(PROVENANCE, values)
            write_dataset(g, name, value)
        end
    finally
        close(g)
    end
    return nothing
end

# The texts of the active project and of the manifest beside it, or ""
# for either that does not exist. Julia prefers a manifest named for its
# own minor version, and a `JuliaProject.toml` pairs with a
# `JuliaManifest`.
function environment_texts()
    project = Base.active_project()
    (project === nothing || !isfile(project)) && return ("", "")
    stem = basename(project) == "JuliaProject.toml" ? "JuliaManifest" : "Manifest"
    names = ("$stem-v$(VERSION.major).$(VERSION.minor).toml", "$stem.toml")
    manifest = findfirst(isfile, map(n -> joinpath(dirname(project), n), names))
    return (read(project, String),
            manifest === nothing ? "" : read(joinpath(dirname(project), names[manifest]),
                                             String))
end

# The time as ISO 8601 in UTC, without a Dates dependency: the civil date
# of a day count is Howard Hinnant's `civil_from_days`, exact over the
# proleptic Gregorian calendar.
function utc_now()
    days, secs = fldmod(floor(Int, time()), 86400)
    z = days + 719468
    era = fld(z, 146097)
    doe = z - era * 146097
    yoe = (doe - doe ÷ 1460 + doe ÷ 36524 - doe ÷ 146096) ÷ 365
    doy = doe - (365 * yoe + yoe ÷ 4 - yoe ÷ 100)
    mp = (5 * doy + 2) ÷ 153
    day = doy - (153 * mp + 2) ÷ 5 + 1
    month = mp < 10 ? mp + 3 : mp - 9
    year = yoe + 400 * era + (month <= 2)
    hour, rest = divrem(secs, 3600)
    minute, second = divrem(rest, 60)
    two(x) = lpad(x, 2, '0')
    return "$year-$(two(month))-$(two(day))T$(two(hour)):$(two(minute)):$(two(second))Z"
end

# Read leniently: provenance is what a refusal quotes, so it must be
# readable from a file that is refused for anything else.
function read_provenance(root)
    g = haskey(root, "provenance") ? root["provenance"] : nothing
    item(name, default) = (g !== nothing && haskey(g, name)) ? read(g, name) : default
    text(name) = (x = item(name, ""); x isa AbstractString ? String(x) : "")
    return (; treeamr_version=tryparse(VersionNumber, text("treeamr_version")),
            julia_version=tryparse(VersionNumber, text("julia_version")),
            created=text("created"), hostname=text("hostname"),
            nthreads=(n = item("nthreads", 0); n isa Integer ? Int(n) : 0),
            nranks=(n = item("nranks", 1); n isa Integer ? Int(n) : 1),
            nparts=(n = item("nparts", 1); n isa Integer ? Int(n) : 1),
            project=text("project"), manifest=text("manifest"))
end

function writer_note(context)
    p = context.provenance
    who = p.treeamr_version === nothing ? "an unknown version of TreeAMR" :
          "TreeAMR $(p.treeamr_version)"
    when = isempty(p.created) ? "" : " on $(p.created)"
    return "The file was written by $who$when; `checkpoint_environment(path, dir)` " *
           "writes the Project.toml and Manifest.toml it was written with into `dir`, " *
           "so that `julia --project=dir` recreates that environment."
end

# --- the forest ------------------------------------------------------------

function write_forest(root, forest::Forest{D,R}) where {D,R}
    g = create_group(root, "forest")
    try
        write_attribute(g, "D", Int64(D))
        write_attribute(g, "N", Int64(forest.N))
        write_attribute(g, "connectivity", "brick")
        write_attribute(g, "roots", Int64[forest.roots...])
        write_attribute(g, "periodic", UInt8[forest.periodic...])
        # (lo, hi) per dimension: a (2, D) array, which C sees as (D, 2).
        write_attribute(g, "reflecting",
                        UInt8[forest.reflecting[d][s] for s in 1:2, d in 1:D])
        write_typeattrs(g, R, GEOMETRY_ATTRS)
        F, limbs = storage_of(R, "the forest's geometry")
        extents = R[forest.extents[d][s] for s in 1:2, d in 1:D]
        write_array(g, "extents", F, (limbs..., 2, D), extents)
        # The leaves as columns, in curve order: block `b` of the forest is
        # row `b`, and the blocks of a part are one run of rows.
        n = nleaves(forest)
        roots = Vector{Int32}(undef, n)
        levels = Vector{Int8}(undef, n)
        coords = Matrix{UInt32}(undef, D, n)
        threaded_foreach(n) do i
            k = forest.leaves[i]
            roots[i] = k.root
            levels[i] = k.level
            for d in 1:D
                coords[d, i] = k.coords[d]
            end
        end
        write_attribute(g, "leaves_crc32c", leaves_checksum(roots, levels, coords))
        write_array(g, "root", Int32, (n,), roots)
        write_array(g, "level", Int8, (n,), levels)
        write_array(g, "coords", UInt32, (D, n), coords)
    finally
        close(g)
    end
    return nothing
end

function read_forest(g, types, context, comm, version)
    note = writer_note(context)
    connectivity = read_attribute(g, "connectivity")
    connectivity == "brick" || throw(ArgumentError(
        "the forest's connectivity is $(repr(connectivity)), and this version reads only " *
        "\"brick\". " * note))
    D = read_attribute(g, "D")
    (D isa Integer && D >= 1) || throw(ArgumentError(
        "the forest's dimension, D = $(repr(D)), is not a positive integer: the file is " *
        "damaged. " * note))
    return read_forest(g, Val(Int(D)), types, context, comm, version)
end

# Every rank reads every leaf, since the forest is replicated — over
# several ranks from its own copy of the index's image, so every rank
# reaches the same verdict on it without a message — and builds it over
# `comm`.
function read_forest(g, ::Val{D}, types, context, comm, version) where {D}
    note = writer_note(context)
    N = read_attribute(g, "N")
    roots = read_attribute(g, "roots")
    (roots isa AbstractVector{<:Integer} && length(roots) == D) || throw(ArgumentError(
        "the forest's roots, $(repr(roots)), are not $D integers: the file is damaged. " *
        note))
    periodic = read_flags(g, "periodic", (D,), note)
    reflecting = read_flags(g, "reflecting", (2, D), note)
    R, F, limbs = read_typeattrs(g, GEOMETRY_ATTRS, types, "the forest's geometry",
                                 context)
    extents = read_array!(Matrix{R}(undef, 2, D), g["extents"], F, (limbs..., 2, D),
                          "the forest's extents", note)
    rootset = g["root"]
    n = length(rootset)
    rootcol = read_array(rootset, Int32, (n,), "the leaves' roots", note)
    levels = read_array(g["level"], Int8, (n,), "the leaves' levels", note)
    coords = read_array(g["coords"], UInt32, (D, n), "the leaves' coordinates", note)
    # Required from version 2 on; a version-1 file from before M7 has none.
    version == 1 || hasattr(g, "leaves_crc32c") || throw(ArgumentError(
        "the leaf list has no checksum, which every file of format version 2 stores " *
        "with it: the file is damaged. " * note))
    if hasattr(g, "leaves_crc32c")
        stored = read_attribute(g, "leaves_crc32c")
        stored isa Integer && stored == leaves_checksum(rootcol, levels, coords) ||
            throw(ArgumentError(
                "the leaf list does not match the checksum stored with it (a CRC-32C over " *
                "its columns `root`, `level` and `coords`): the file was damaged while or " *
                "after it was written, and is refused rather than read into a wrong mesh. " *
                note))
    end
    # Each key is checked by its own constructor, the list as a whole by
    # the forest's.
    leaves = Vector{MortonKey{D}}(undef, n)
    threaded_foreach(n) do i
        leaves[i] = MortonKey{D}(rootcol[i], levels[i], ntuple(d -> coords[d, i], D))
    end
    return Forest{R}(ntuple(d -> Int(roots[d]), D); N=N,
                     periodic=ntuple(d -> periodic[d] == 1, D),
                     reflecting=ntuple(D) do d
                         (reflecting[1, d] == 1, reflecting[2, d] == 1)
                     end,
                     extents=ntuple(d -> (extents[1, d], extents[2, d]), D),
                     leaves=leaves, comm=comm)
end

# --- field sets ------------------------------------------------------------

const PARITY_NAMES = (EvenParity => "even", OddParity => "odd", NoParity => "none")

parity_name(p::Parity) =
    last(PARITY_NAMES[findfirst(q -> first(q) === p, PARITY_NAMES)])

function parity_of(name, note)
    i = findfirst(q -> last(q) == name, PARITY_NAMES)
    i === nothing && throw(ArgumentError(
        "a field set's parity is $(repr(name)), not \"even\", \"odd\" or \"none\": the " *
        "file is damaged. " * note))
    return first(PARITY_NAMES[i])
end

# A field set's layout, as the attributes of its group in the index; the
# data are in the parts.
function write_layout(parent, name, fs::FieldSet{T,D}) where {T,D}
    g = create_group(parent, name)
    try
        write_typeattrs(g, T, ELTYPE_ATTRS)
        write_attribute(g, "nvars", Int64(fs.nvars))
        write_attribute(g, "G", Int64[fs.G...])
        write_attribute(g, "centering", [string(c) for c in fs.centering])
        # (D, nvars), which C sees as (nvars, D).
        fs.parity === nothing ||
            write_attribute(g, "parity", [parity_name(p[d]) for d in 1:D, p in fs.parity])
        write_attribute(g, "range", "owned")
    finally
        close(g)
    end
    return nothing
end

# The layout of field set `name` from its group `g` in the index: the
# element type `T` and how it is stored (`F`, `limbs`), and the arguments
# of its constructor.
function read_layout(g, name, ::Val{D}, types, context) where {D}
    note = writer_note(context)
    what = "field set $(repr(name))"
    range = read_attribute(g, "range")
    range == "owned" || throw(ArgumentError(
        "$what stores the range $(repr(range)), and this version reads only \"owned\": " *
        "the owned points, from which everything else is rebuilt. " * note))
    T, F, limbs = read_typeattrs(g, ELTYPE_ATTRS, types, what, context)
    nvars = Int(read_attribute(g, "nvars"))
    G = Tuple(Int.(read_attribute(g, "G")))
    centering = Tuple(Symbol.(read_attribute(g, "centering")))
    parity = nothing
    if hasattr(g, "parity")
        names = read_attribute(g, "parity")
        size(names) == (D, nvars) || throw(ArgumentError(
            "$what has a parity table of size $(size(names)), not ($D, $nvars): the file " *
            "is damaged. " * note))
        parity = [ntuple(d -> parity_of(names[d, v], note), D) for v in 1:nvars]
    end
    return (; name=String(name), T, F, limbs, nvars, G, centering, parity)
end

# How a field set's blocks are stored: the dimensions of one block in the
# file, limbs first, and the number of elements of `T` a block holds in
# the state vector, which is a block's run of the data's last axis.
block_dims(N, ::Val{D}, limbs) where {D} = (limbs..., ntuple(_ -> N, D)...)
block_length(N, D, nvars) = N^D * nvars

# Validate the `fieldsets` keyword of `save_checkpoint` into a list of
# `(name, fs, u)`, `u` being `nothing` for the bare form, before a file
# is opened.
function collect_fieldsets(fieldsets, forest)
    sets = Tuple{String,FieldSet,Any}[]
    for item in fieldsets
        item isa Pair || throw(ArgumentError(
            "each entry of `fieldsets` is a pair, `name => (fs, u)` or `name => fs`; got " *
            "a $(typeof(item))"))
        name, value = item
        name isa AbstractString || throw(ArgumentError(
            "a field set's name is a string, the name of its group in the file; got " *
            repr(name)))
        check_name(name, "the field set name")
        any(s -> s[1] == name, sets) && throw(ArgumentError(
            "two field sets are named $(repr(name)): each is the group " *
            "fieldsets/$name of the file, so the names must differ"))
        fs, u = value isa FieldSet ? (value, nothing) :
                value isa Tuple{FieldSet,AbstractVector} ? value :
                throw(ArgumentError(
                    "field set $(repr(name)) must be given as `fs` or as `(fs, u)`, with " *
                    "`u` its state vector; got a $(typeof(value))"))
        fs.forest === forest || throw(ArgumentError(
            "field set $(repr(name)) is over another forest than the one being saved: a " *
            "checkpoint stores one leaf list, and block b of every field set in it is " *
            "leaf b of that list, as for `regrid!`"))
        what = "the state vector of field set $(repr(name))"
        if u !== nothing
            eltype(u) === eltype(fs.work) || throw(ArgumentError(
                "$what has element type $(eltype(u)), but the field set stores " *
                "$(eltype(fs.work)): `u` must be the set's own state vector"))
            length(u) == statelength(fs) || throw(ArgumentError(
                "$what has $(length(u)) entries, but the field set needs " *
                "$(statelength(fs)): `u` must be its state vector on the current mesh, " *
                "and a regrid changes that length"))
            samebackend(get_backend(u), get_backend(fs.work)) || throw(ArgumentError(
                "$what is on $(nameof(typeof(get_backend(u)))), but the field set is on " *
                "$(nameof(typeof(get_backend(fs.work)))): `u` must be its state vector, " *
                "which lives where its storage does"))
        end
        storage_of(eltype(fs.work), "field set $(repr(name))")
        push!(sets, (String(name), fs, u))
    end
    return sets
end

function check_application(application)
    application === nothing && throw(ArgumentError(
        "save_checkpoint has no default `application`: pass `application = name => " *
        "version`. The name is the application's own top-level group in the file, " *
        "beside TreeAMR's, and the version is the format version of what the " *
        "application stores there, which `load_checkpoint` returns for it to check. " *
        "Both belong to the application, which the mesh cannot know."))
    (application isa Pair && first(application) isa AbstractString &&
     last(application) isa Integer) || throw(ArgumentError(
        "`application` is `name => version`, a string and an integer, such as " *
        "`\"MyApp\" => 1`; got $(repr(application))"))
    name, version = application
    name == GROUP && throw(ArgumentError(
        "the application cannot be named \"$GROUP\": that is TreeAMR's own group in the " *
        "file. Each package owns one top-level group, so that the two layouts, and their " *
        "format versions, evolve independently."))
    check_name(name, "the application's name")
    return (String(name), Int64(version))
end

function check_io(io)
    (io === :node || io === :all || (io isa Integer && !(io isa Bool) && io >= 1)) ||
        throw(ArgumentError(
            "`io` is `:node` (one I/O process per shared-memory node, the default), " *
            "`:all` (every rank writes its own part) or a positive integer, the number " *
            "of I/O processes; got $(repr(io))"))
    return io
end

# --- I/O groups and parts (M7 step 6b) ---------------------------------------
#
# The ranks are split into `k` contiguous groups in rank order, by the
# equal-count split that partitions the blocks; the first rank of each
# group is its I/O process. The ranks own contiguous runs of the curve in
# rank order, so a group's blocks are one contiguous run of the curve,
# and its part one range of blocks. `:node` sets `k` to the number of
# shared-memory nodes and nothing else: the groups stay equal-count rank
# ranges even when a node's ranks are not contiguous, which costs
# messages across the network then and nothing else ("Checkpoints without
# parallel I/O" in CODE.md).
struct IOPlan
    k::Int                      # the number of groups, and of parts
    group::Int                  # this rank's group, from 0
    ranks::UnitRange{Int}       # the group's ranks, from 0; the first is its I/O process
end

function io_plan(r::Ranks, io)
    k = io === :node ? commnodes(r.comm) : io === :all ? r.size : min(Int(io), r.size)
    g = equalsplit_part(r.size, k, r.rank + 1)
    return IOPlan(k, g - 1, equalsplit(r.size, k, g) .- 1)
end

isio(plan::IOPlan, r::Ranks) = r.rank == first(plan.ranks)

# The blocks of rank `q` among `P`, and of group `g` among `k`, as global
# leaf indices from 1: `blockrange`'s arithmetic, for any rank.
rankblocks(n, P, q) = equalsplit(n, P, q + 1)

function groupblocks(n, P, k, g)
    ranks = equalsplit(P, k, g + 1)
    return first(rankblocks(n, P, first(ranks) - 1)):last(rankblocks(n, P, last(ranks) - 1))
end

# A range of blocks cut into pieces of whole blocks of at most
# `MAX_MESSAGE` bytes each, or of one block where a block is larger: how a
# member's blocks travel to its I/O process, and a part's blocks from
# their reader to their owner. Both ends cut a range the same way.
function pieces(blocks::UnitRange{Int}, blockbytes::Integer)
    per = max(1, MAX_MESSAGE[] ÷ max(1, Int(blockbytes)))
    return [b:min(b + per - 1, last(blocks)) for b in first(blocks):per:last(blocks)]
end

# The elements of blocks `rows` (from 1) of a state vector whose blocks
# hold `per` elements each.
elements(rows::UnitRange{Int}, per) = ((first(rows) - 1) * per + 1):(last(rows) * per)

# The part file of group `j`: beside the index, named after it and the
# save, so that a part from another save cannot be taken for one of this
# save's. The link to it in the index is named with four digits at least,
# so that the links list in order.
partname(path, saveid, j) = basename(path) * "." * saveid * "." * string(j) * ".h5"
linkname(j) = lpad(string(j), 4, '0')

# The save id and part number in `name` if it is exactly the name of a
# part of the index `base`, that is `base`, a dot, 32 lowercase hex
# digits, a dot, decimal digits and `.h5`; otherwise `nothing`. Only such
# a file is ever removed as an orphan.
function part_of(base::AbstractString, name::AbstractString)
    startswith(name, base * ".") || return nothing
    m = match(r"^([0-9a-f]{32})\.([0-9]+)\.h5$", name[(ncodeunits(base) + 2):end])
    return m === nothing ? nothing : (m[1], parse(Int, m[2]))
end

# Whether `name` has the form of some index's part, whatever the index
# is called: the only names a previous index may make a save remove.
is_partname(name::AbstractString) =
    !occursin('/', name) && occursin(r"\.[0-9a-f]{32}\.[0-9]+\.h5$", name)

# 128 random bits from the operating system, as 32 hex digits. Not
# `rand`: an application that seeds the global generator the same way in
# every run would draw the same id in a restarted run as in the run that
# wrote the checkpoint it restarts from, and its parts would overwrite
# that checkpoint's. `Libc.getrandom!` is what `RandomDevice` draws
# from.
function new_saveid()
    bytes = zeros(UInt8, 16)
    Base.Libc.getrandom!(bytes)
    return bytes2hex(bytes)
end

# Run `f` unless an error has happened already, and return the first
# error: how the I/O process and the readers keep going through their
# messages after a failure, so that no rank waits forever for one.
function attempt(f, err)
    err === nothing || return err
    try
        f()
        return nothing
    catch e
        return e
    end
end

# A field set's data for the save on this rank: its state vector on the
# host, `u` or the owned points gathered from `fs.work`, and one checksum
# per block.
function prepared((name, fs, u))
    if u === nothing
        u = statevector(fs)
        gather!(u, fs)
    end
    host = tohost(u)
    per = block_length(fs.forest.N, dimension(fs.forest), fs.nvars)
    return host, per, block_checksums(host, nblocks(fs), per)
end

dimension(::Forest{D}) where {D} = D

# A member's part of a save: for each field set its per-block checksums,
# after a status word (0, or 1 if it could not prepare its data, which its
# own error then says), and its blocks, to its I/O process. Returns this
# rank's error, or `nothing`.
function send_blocks(r::Ranks, plan::IOPlan, sets)
    io = first(plan.ranks)
    err = nothing
    for set in sets
        prep = nothing
        err = attempt(() -> (prep = prepared(set)), err)
        if prep === nothing
            status = UInt32[1]
            waitall(r.comm, [isend(r.comm, status, io, TAG_HEAD)])
            continue
        end
        host, per, sums = prep
        head = [UInt32(0); sums]
        requests = Any[isend(r.comm, head, io, TAG_HEAD)]
        for piece in pieces(1:length(sums), per * sizeof(eltype(host)))
            push!(requests, isend(r.comm, view(host, elements(piece, per)), io, TAG_SAVE))
        end
        waitall(r.comm, requests)
    end
    return err
end

# The I/O process's part of a save: its part group `pg` (`nothing` if it
# could not be created, `err` saying why), filled with each field set's
# data for the group's blocks — its own first, then each member's as it
# arrives, the next message posted before the current one is written —
# and their checksums. Every block received is checked against the
# checksum its member computed before sending it. After a failure it
# goes on receiving, without writing, so that no member waits forever.
# Returns the first error and, per field set, the CRC-32C of the part's
# `data_crc32c`.
function write_blocks!(pg, err, r::Ranks, plan::IOPlan, forest, sets, filters)
    n, P = nleaves(forest), r.size
    D = dimension(forest)
    nb = length(groupblocks(n, P, plan.k, plan.group))
    fg = nothing
    err = attempt(() -> (fg = create_group(pg, "fieldsets"; track_order=true)), err)
    summaries = UInt32[]
    for (s, set) in enumerate(sets)
        name, fs, _ = set
        T = eltype(fs.work)
        F, limbs = storage_of(T, "field set $(repr(name))")
        block = block_dims(forest.N, Val(D), limbs)
        dims = (block..., fs.nvars, nb)
        per = block_length(forest.N, D, fs.nvars)
        sums = zeros(UInt32, nb)
        g = dset = nothing
        err = attempt(err) do
            g = create_group(fg, name)
            # An empty part is stored contiguously: HDF5 refuses a chunk
            # larger than a fixed dataset of extent 0.
            dset = write_array(g, "data", F, dims, nothing; chunk=(block..., 1, 1),
                               filters=nb == 0 ? () : filters, keep=true)
        end
        # Its own blocks, from its own state vector.
        own = length(rankblocks(n, P, r.rank))
        prep = nothing
        err = attempt(() -> (prep = prepared(set)), err)
        # The members' checksums, which each sends first, and from them
        # the queue of every member's pieces in curve order.
        members = plan.ranks[2:end]
        heads = [Vector{UInt32}(undef, 1 + length(rankblocks(n, P, q))) for q in members]
        waitall(r.comm, [irecv(r.comm, heads[i], q, TAG_HEAD) for (i, q) in enumerate(members)])
        queue = Tuple{Int,Int,UnitRange{Int},Int}[]       # (rank, offset, piece, member)
        offset = own
        for (i, q) in enumerate(members)
            head = heads[i]
            m = length(head) - 1
            if head[1] != 0
                err === nothing && (err = ErrorException(
                    "rank $q could not prepare its blocks of field set $(repr(name)) for " *
                    "its I/O process, rank $(r.rank); rank $q's own error says why"))
            else
                sums[offset .+ (1:m)] = view(head, 2:(m + 1))
                for piece in pieces(1:m, per * sizeof(T))
                    push!(queue, (q, offset, piece, i))
                end
            end
            offset += m
        end
        # The next piece is always on its way while the current one is
        # written: the first while the I/O process writes its own blocks,
        # and each later one while the one before it is checked and
        # written, across the members' boundaries too.
        longest = maximum(e -> length(e[3]), queue; init=0) * per
        buffers = (Vector{T}(undef, longest), Vector{T}(undef, longest))
        post(j) = irecv(r.comm, view(buffers[mod1(j, 2)], 1:(length(queue[j][3]) * per)),
                        queue[j][1], TAG_SAVE)
        request = isempty(queue) ? nothing : post(1)
        if prep !== nothing
            host, _, mine = prep
            sums[1:own] = mine
            err = attempt(err) do
                GC.@preserve host block_slab(API.h5d_write, dset, F, dims, 1:own,
                                             pointer(host))
            end
            if err === nothing && FAIL_PART[] == plan.group
                err = ErrorException(
                    "the I/O process of part $(plan.group) failed on purpose after its " *
                    "first write (TreeAMRHDF5Ext.FAIL_PART, a test hook)")
            end
        end
        for (j, (q, offset, piece, i)) in enumerate(queue)
            waitall(r.comm, [request])
            j < length(queue) && (request = post(j + 1))
            err === nothing || continue
            buf = buffers[mod1(j, 2)]
            got = block_checksums(buf, length(piece), per)
            bad = findfirst(x -> got[x] != heads[i][1 + piece[x]], eachindex(got))
            if bad !== nothing
                b = first(groupblocks(n, P, plan.k, plan.group)) + offset + piece[bad] - 1
                err = ErrorException(
                    "block $b of field set $(repr(name)) arrived at its I/O process, " *
                    "rank $(r.rank), from rank $q damaged: its CRC-32C does not match " *
                    "the one rank $q computed before sending it, so it was not " *
                    "written, and the checkpoint is not saved")
                continue
            end
            err = attempt(err) do
                GC.@preserve buf block_slab(API.h5d_write, dset, F, dims, offset .+ piece,
                                            pointer(buf))
            end
        end
        err = attempt(() -> write_array(g, "data_crc32c", UInt32, (nb,), sums), err)
        push!(summaries, crc_of(sums))
        dset === nothing || close(dset)
        g === nothing || close(g)
    end
    fg === nothing || close(fg)
    return err, summaries
end

# The attributes that make a group a part (M7 step 6b): which save it
# belongs to, its number and its blocks.
function part_attributes!(pg, saveid, j, range)
    write_attribute(pg, "format", PART_FORMAT)
    write_attribute(pg, "format_version", Int64(FORMAT_VERSION))
    write_attribute(pg, "save_id", saveid)
    write_attribute(pg, "part", Int64(j))
    write_attribute(pg, "first_block", Int64(first(range)))
    write_attribute(pg, "last_block", Int64(last(range)))
    return nothing
end

# --- saving ------------------------------------------------------------------

save_checkpoint(path::AbstractString, forest::Forest; kwargs...) =
    save_checkpoint(nothing, path, forest; kwargs...)

function save_checkpoint(f, path::AbstractString, forest::Forest; fieldsets=nothing,
                         application=nothing, data=(;), filters=(), sync::Bool=true,
                         io=:node)
    r = Ranks(forest.comm)
    # Every refusal before anything is created, and over several ranks
    # agreed, so that a refusal is raised on all of them together.
    appname, appversion, sets, filters′, io′ = agreed(r, "save_checkpoint";
                                                      forest=forest) do
        c = check_save(forest, fieldsets, application, filters, io)
        appname, appversion, sets, filters′, io′ = c
        datahash = plain_hash("/$appname/data", data, UInt(0))
        layout = layouthash(String(path), appname, appversion, string(filters′), sync,
                            f === nothing, string(io′), map(set_layout, sets))
        return (c, layout, datahash)
    end
    write_checkpoint(f, r, String(path), forest, appname, appversion, sets, filters′,
                     data, sync, io′)
    return path
end

function check_save(forest, fieldsets, application, filters, io)
    fieldsets === nothing && throw(ArgumentError(
        "save_checkpoint has no default `fieldsets`: pass the field sets the " *
        "application evolves, as `name => (fs, u)` pairs, or `()` for none. Which sets " *
        "are state and which are scratch, rebuilt every step, is the application's to " *
        "say."))
    appname, appversion = check_application(application)
    sets = collect_fieldsets(fieldsets, forest)
    storage_of(floattype(forest), "the forest's geometry")
    filters isa HDF5.Filters.Filter && (filters = (filters,))
    return (appname, appversion, sets, filters, check_io(io))
end

# What the ranks must agree on about one field set: everything that
# shapes the objects created for it.
set_layout((name, fs, u)) =
    (name, typename(eltype(fs.work)), fs.nvars, fs.G, string.(fs.centering),
     fs.parity === nothing ? "" : string(map(p -> map(parity_name, p), fs.parity)),
     u === nothing)

# The steps of "Checkpoints without parallel I/O" in CODE.md: the parts,
# the index, the commit, the cleanup. Every rank runs this; each step
# that runs on some ranks only ends in an agreement, so that a failure
# anywhere is raised everywhere, and until the rename nothing the
# previous checkpoint needs has been touched.
function write_checkpoint(f, r::Ranks, path, forest, appname, appversion, sets, filters,
                          data, sync, io)
    plan = io_plan(r, io)
    # The save id, rank 0's, on every rank.
    saveid = String(bcast(r.comm, r.rank == 0 ? collect(codeunits(new_saveid())) : UInt8[],
                          0))
    dir = dirname(abspath(path))
    partial = path * ".partial"
    inline = plan.k == 1
    # Read before anything changes: the parts the checkpoint at `path`
    # names, which are removed once this one has replaced it.
    previous = r.rank == 0 ? previous_parts(path) : String[]
    index = nothing                       # rank 0's partial index, once created
    mypart = nothing                      # this I/O process's part file
    renamed = false
    try
        # 1. The parts. With one part it is a group of the index, which
        # rank 0, its I/O process, creates now.
        err = nothing
        report = UInt64[]
        if isio(plan, r)
            range = groupblocks(nleaves(forest), r.size, plan.k, plan.group)
            file = pg = nothing
            err = attempt(err) do
                if inline
                    note_open(partial)
                    index = h5open(partial, "w")
                    root = create_group(index, GROUP)
                    parts = create_group(root, "parts")
                    pg = create_group(parts, linkname(0))
                    close(parts)
                    close(root)
                else
                    mypart = joinpath(dir, partname(path, saveid, plan.group))
                    ispath(mypart) && error(
                        "the part file $(repr(mypart)) exists already, and a save never " *
                        "overwrites one: its name holds a save id drawn afresh for this " *
                        "save, so another process is writing to the same checkpoint")
                    file = open_h5(mypart, "w")
                    pg = create_group(file, GROUP)
                end
                part_attributes!(pg, saveid, plan.group, range)
            end
            err, sums = write_blocks!(pg, err, r, plan, forest, sets, filters)
            pg === nothing || close(pg)
            bytes = 0
            if !inline
                # Closed whatever happened, so that a failed part can be
                # removed; flushed and measured only if nothing failed.
                closing = attempt(() -> (file === nothing || close(file)), nothing)
                err = err === nothing ? closing : err
                err = attempt(err) do
                    if sync
                        flush_to_storage(mypart)
                        flush_to_storage(dir; directory=true)
                    end
                    bytes = filesize(mypart)
                end
            end
            report = UInt64[plan.group; bytes; sums]
        else
            err = send_blocks(r, plan, sets)
        end
        agree_errors(r, err, "save_checkpoint, writing the parts,")
        reports = reshape(allgatherv(r.comm, report), 2 + length(sets), plan.k)

        # 2. The index, on rank 0; the do-block on every rank.
        err = nothing
        scratch = nothing
        try
            err = attempt(err) do
                if r.rank == 0
                    if index === nothing
                        note_open(partial)
                        index = h5open(partial, "w")
                    end
                    write_index!(index, path, saveid, forest, sets, appname, reports, plan.k,
                                 r.size)
                    target = index
                else
                    scratch = memory_file()
                    target = scratch
                end
                app = create_group(target, appname; track_order=true)
                try
                    write_attribute(app, "format_version", appversion)
                    put_plain(app, "data", data)            # checked and agreed above
                    f === nothing || saving(() -> f(app), target, r)
                finally
                    close(app)
                end
            end
        finally
            scratch === nothing || close(scratch)
        end
        agree_errors(r, err, "save_checkpoint, writing the index,")

        # 3. The commit: the index in place of the previous one.
        err = nothing
        if r.rank == 0
            err = attempt(err) do
                close(index)
                index = nothing
                sync && flush_to_storage(partial)
                Base.Filesystem.rename(partial, path)
                renamed = true
                sync && flush_to_storage(dir; directory=true)
            end
        end
        renamed = r.size == 1 ? renamed : first(allgather(r.comm, renamed))
        agree_errors(r, err, "save_checkpoint, putting the index in place,")
    catch
        # Nothing is committed, unless the rename was: then the new
        # checkpoint is the one at `path`, and its parts stay.
        if !renamed
            mypart === nothing || rm(mypart; force=true)
            if r.rank == 0
                index === nothing || close(index)
                rm(partial; force=true)
            end
        end
        rethrow()
    end
    # 4. The cleanup, once the new checkpoint is in place.
    r.rank == 0 && remove_stale(path, saveid, previous)
    return nothing
end

# The files a save's do-block writes into: the index on rank 0, the
# scratch file on the others, so that `write_plain` in the block knows to
# agree its value across the ranks.
const SAVING = IdDict{HDF5.File,Ranks}()
const SAVING_LOCK = ReentrantLock()

function saving(f, file, r::Ranks)
    lock(() -> (SAVING[file] = r), SAVING_LOCK)
    try
        return f()
    finally
        lock(() -> delete!(SAVING, file), SAVING_LOCK)
    end
end

saving_ranks(obj) = lock(() -> get(SAVING, HDF5.file(obj), nothing), SAVING_LOCK)

# Everything of the index but the application's group: the format
# attributes, the provenance, the forest, the field sets' layouts, the
# part table and the links to the parts. `reports` holds a column per
# part, `(part, bytes, checksum per field set...)`, in part order.
function write_index!(file, path, saveid, forest, sets, appname, reports, k, nranks)
    root = haskey(file, GROUP) ? file[GROUP] : create_group(file, GROUP)
    try
        write_attribute(root, "format", FORMAT)
        write_attribute(root, "format_version", Int64(FORMAT_VERSION))
        write_attribute(root, "features", collect(String, FEATURES))
        write_attribute(root, "application", appname)
        write_attribute(root, "save_id", saveid)
        write_provenance(root, provenance_values(nranks, k))
        write_forest(root, forest)
        group = create_group(root, "fieldsets"; track_order=true)
        try
            for (name, fs, _) in sets
                write_layout(group, name, fs)
            end
        finally
            close(group)
        end
        n = nleaves(forest)
        ranges = [groupblocks(n, nranks, k, j) for j in 0:(k - 1)]
        files = k == 1 ? [""] : [partname(path, saveid, j) for j in 0:(k - 1)]
        table = create_group(root, "parttable")
        try
            write_dataset(table, "file", files)
            write_array(table, "first_block", Int64, (k,), Int64[first(x) for x in ranges])
            write_array(table, "last_block", Int64, (k,), Int64[last(x) for x in ranges])
            write_array(table, "bytes", Int64, (k,), Int64.(reports[2, :]))
            write_dataset(table, "fieldsets", String[name for (name, _, _) in sets])
            write_array(table, "data_crc32c", UInt32, (length(sets), k),
                        UInt32.(reports[3:end, :]))
        finally
            close(table)
        end
        if k > 1
            # For tools only: TreeAMR's reader never follows them.
            parts = create_group(root, "parts")
            try
                for j in 0:(k - 1)
                    create_external(parts, linkname(j), files[j + 1], "/" * GROUP)
                end
            finally
                close(parts)
            end
        end
    finally
        close(root)
    end
    return nothing
end

# The part files the checkpoint at `path` names, if it is a version-2
# index: read leniently, since a missing or damaged earlier checkpoint
# only means there is nothing of it to remove. Only names of the part
# form are returned, so that a damaged index cannot make a save remove
# any other file.
function previous_parts(path)
    isfile(path) && HDF5.ishdf5(path) || return String[]
    names = String[]
    try
        file = open_h5(path, "r")
        try
            key = GROUP * "/parttable/file"
            haskey(file, key) && append!(names, filter(is_partname, read(file[key])))
        finally
            close(file)
        end
    catch
    end
    return names
end

# Remove the parts the previous index named and every orphan: a file of
# the part form for this index (`part_of`) from a save other than this
# one. A failure to remove one is a warning, since the checkpoint is in
# place.
function remove_stale(path, saveid, previous)
    dir = dirname(abspath(path))
    stale = Set{String}(previous)
    for name in readdir(dir)
        part = part_of(basename(path), name)
        part === nothing || first(part) == saveid || push!(stale, name)
    end
    for name in stale
        part = part_of(basename(path), name)
        part !== nothing && first(part) == saveid && continue
        try
            rm(joinpath(dir, name); force=true)
        catch err
            @warn "the checkpoint $(repr(path)) is in place, but a part of an earlier " *
                  "one could not be removed" name exception = err
        end
    end
    return nothing
end

# Flush the file, or the directory, at `path` from the operating system's
# cache to stable storage. Closing a file hands its data to the page
# cache only, which survives the process and not a power loss. On macOS
# `fsync` hands the data to the drive without waiting for the drive's own
# cache, and `fcntl(F_FULLFSYNC)` waits (measured: 1 ms against 125 ms
# for 540 MB); a file system without it — some network and FUSE ones —
# falls back to `fsync`, as SQLite and libuv do. A directory that cannot
# be synced (`EINVAL` on some file systems) is accepted, since there is
# nothing further to ask of it. On Windows this does nothing.
const O_RDONLY = Cint(0)            # the same on Linux and macOS
const F_FULLFSYNC = Cint(51)        # macOS

function flush_to_storage(path::AbstractString; directory::Bool=false)
    Sys.iswindows() && return nothing
    directory || note_open(path)
    fd = ccall(:open, Cint, (Cstring, Cint), path, O_RDONLY)
    fd < 0 && systemerror("opening $(repr(path)) to flush it to stable storage")
    try
        full = Sys.isapple() && ccall(:fcntl, Cint, (Cint, Cint), fd, F_FULLFSYNC) == 0
        full || ccall(:fsync, Cint, (Cint,), fd) == 0 ||
            (directory && Libc.errno() == Libc.EINVAL) ||
            systemerror("flushing $(repr(path)) to stable storage")
    finally
        ccall(:close, Cint, (Cint,), fd)
    end
    return nothing
end

# --- loading ---------------------------------------------------------------

function check_file(path::AbstractString)
    isfile(path) || throw(ArgumentError(
        "there is no checkpoint at $(repr(path)): no such file"))
    HDF5.ishdf5(path) || throw(ArgumentError(
        "$(repr(path)) is not a TreeAMR checkpoint: it is not an HDF5 file at all"))
    return nothing
end

# The index at `path`, open, once it has been checked to be a TreeAMR
# checkpoint. Nothing about the format version is checked here, so that
# `checkpoint_environment` can read a file whose version `load_checkpoint`
# refuses.
function open_index(path::AbstractString)
    check_file(path)
    file = open_h5(path, "r")
    try
        check_checkpoint(file, path)
    catch
        close(file)
        rethrow()
    end
    return file
end

function check_checkpoint(file, path)
    haskey(file, GROUP) || throw(ArgumentError(
        "$(repr(path)) is not a TreeAMR checkpoint: it is an HDF5 file with no " *
        "/$GROUP group, which is where TreeAMR keeps everything it writes"))
    root = file[GROUP]
    format = hasattr(root, "format") ? read_attribute(root, "format") : nothing
    format == FORMAT || throw(ArgumentError(
        "$(repr(path)) is not a TreeAMR checkpoint: its /$GROUP group has format = " *
        "$(repr(format)), not $(repr(FORMAT))"))
    return nothing
end

context_of(root, path) = (; path=String(path), provenance=read_provenance(root))

# Over several ranks only rank 0 opens the index. It copies everything of
# it but the part data — the `/TreeAMR.jl` group's attributes, the
# provenance, the forest, the field sets' layout attributes, the part
# table, and the application's group — into an HDF5 file in memory and
# broadcasts its bytes, which every rank opens as a file image. So HDF5
# itself serializes the forest and the plain data, and every rank reads
# the same objects as a serial load would. Rank 0 keeps the index open
# for the part data inside it (one part, or a version-1 file). A refusal
# of the file itself is rank 0's, and is broadcast in place of the image.
function share_index(r::Ranks, path)
    real = nothing
    err = nothing
    payload = UInt8[]
    if r.rank == 0
        try
            real = open_index(path)
            payload = [0x00; index_image(real)]
        catch e
            real === nothing || close(real)
            real = nothing
            err = e
            payload = [e isa ArgumentError ? 0x01 : 0x02;
                       codeunits(e isa ArgumentError ? e.msg : sprint(showerror, e))]
        end
    end
    payload = bcast(r.comm, payload, 0)
    if payload[1] != 0x00
        err === nothing || throw(err)
        text = String(payload[2:end])
        payload[1] == 0x01 && throw(ArgumentError(text))
        error("rank 0 could not open the checkpoint index $(repr(path)): " * text)
    end
    meta = h5open(payload[2:end], "r")
    return real, meta
end

function index_image(file)
    root = file[GROUP]
    image = memory_file()
    try
        dst = create_group(image, GROUP)
        copy_attributes(root, dst)
        for name in keys(root)
            name == "parts" && continue              # the part data, or links to them
            if name == "fieldsets"
                sets = create_group(dst, name; track_order=true)
                for set in keys(root[name])
                    g = create_group(sets, set)
                    copy_attributes(root[name][set], g)
                    close(g)
                end
                close(sets)
            else
                copy_object(root, name, dst, name)
            end
        end
        close(dst)
        app = hasattr(root, "application") ? read_attribute(root, "application") : nothing
        app isa AbstractString && app != GROUP && haskey(file, app) &&
            copy_object(file, app, image, app)
        flush(image)
        return Vector{UInt8}(image)
    finally
        close(image)
    end
end

function check_format(root, context)
    note = writer_note(context)
    version = read_attribute(root, "format_version")
    version in READ_VERSIONS || throw(ArgumentError(
        "$(repr(context.path)) is in checkpoint format version $version, and this " *
        "version of TreeAMR reads format versions $(join(READ_VERSIONS, " and "))" *
        (version isa Integer && version > FORMAT_VERSION ?
         ", so it was written by a newer TreeAMR than this one. " :
         ", the only ones there have been. ") *
        "Compatibility is decided by the format version, not by the package version. " *
        note))
    features = read_attribute(root, "features")
    features isa AbstractString && (features = [features])
    unknown = [x for x in features if !(x in FEATURES)]
    isempty(unknown) || throw(ArgumentError(
        "$(repr(context.path)) uses the feature$(length(unknown) == 1 ? "" : "s") " *
        "$(join(repr.(unknown), ", ")), which this version of TreeAMR does not know. A " *
        "feature is listed only when a reader must understand it to read the file " *
        "correctly, so a file with an unknown one is refused rather than misread. " * note))
    return Int(version)
end

function select_fieldsets(group, fieldsets, context)
    available = collect(String, keys(group))
    fieldsets === nothing && return available
    wanted = fieldsets isa AbstractString ? [String(fieldsets)] :
             [String(name) for name in fieldsets]
    for name in wanted
        name in available || throw(ArgumentError(
            "$(repr(context.path)) holds no field set named $(repr(name)); it holds " *
            (isempty(available) ? "none" : join(repr.(available), ", "))))
    end
    return unique(wanted)
end

load_checkpoint(path::AbstractString; kwargs...) = load_checkpoint(nothing, path; kwargs...)

function load_checkpoint(f, path::AbstractString; backend::Backend=CPU(), types=(),
                         fieldsets=nothing, comm=nothing)
    c = communicator(comm)
    r = Ranks(c)
    # The arguments are agreed before the file is opened, and whether
    # it is a file at all is rank 0's to say.
    typelist = agreed(r, "load_checkpoint") do
        t = check_types(types)
        r.rank == 0 && check_file(path)
        layout = layouthash(String(path), map(typename, t), repr(fieldsets),
                            string(nameof(typeof(backend))), f === nothing)
        return (t, layout, UInt(0))
    end
    real = meta = nothing
    try
        if r.size == 1
            real = meta = open_index(path)
        else
            real, meta = share_index(r, path)
        end
        return load_from(f, meta, real, r, String(path), typelist, backend, fieldsets)
    finally
        meta === nothing || meta === real || close(meta)
        real === nothing || close(real)
    end
end

# A part, as the index describes it: its file (`""` for the part inside
# the index, and for a version-1 file, which is read as one part), its
# blocks, its size, and the checksum of each loaded field set's
# `data_crc32c` (`nothing` in a version-1 file).
struct PartEntry
    file::String
    blocks::UnitRange{Int}
    bytes::Int
    sums::Union{Nothing,Vector{UInt32}}
end

# The part table of the index `root`, for the field sets `names`, checked:
# the parts tile the blocks in order, and every file name is a plain one.
# A version-1 file is one inline part.
function read_parttable(root, version, n, names, context)
    version == 1 && return [PartEntry("", 1:n, 0, nothing)]
    note = writer_note(context)
    bad(what) = throw(ArgumentError(
        "the part table of $(repr(context.path)) is damaged: $what. " * note))
    haskey(root, "parttable") || bad("there is none")
    t = root["parttable"]
    for key in ("file", "first_block", "last_block", "bytes", "fieldsets", "data_crc32c")
        haskey(t, key) || bad("it has no `$key`")
    end
    files = read(t["file"])
    k = length(files)
    (files isa AbstractVector{<:AbstractString} && k >= 1) || bad("`file` lists no parts")
    firsts = read_array(t["first_block"], Int64, (k,), "the parts' first blocks", note)
    lasts = read_array(t["last_block"], Int64, (k,), "the parts' last blocks", note)
    bytes = read_array(t["bytes"], Int64, (k,), "the parts' sizes", note)
    setnames = read(t["fieldsets"])
    s = length(setnames)
    sums = read_array(t["data_crc32c"], UInt32, (s, k), "the parts' checksums", note)
    next = 1
    for p in 1:k
        (firsts[p] == next && lasts[p] >= firsts[p] - 1) ||
            bad("part $(p - 1) holds blocks $(firsts[p]) to $(lasts[p]), where the parts " *
                "must tile the blocks 1 to $n in order")
        next = lasts[p] + 1
    end
    next == n + 1 || bad("the parts hold $(next - 1) blocks, and the forest has $n")
    if k == 1
        files[1] == "" || bad("its one part is named $(repr(files[1])), not inline")
    else
        for (p, name) in enumerate(files)
            is_partname(name) && name == basename(name) ||
                bad("part $(p - 1) is named $(repr(name)), not a part file beside the " *
                    "index")
        end
    end
    columns = map(names) do name
        i = findfirst(==(name), setnames)
        i === nothing && bad("it has no checksums for field set $(repr(name))")
        return i
    end
    return [PartEntry(files[p], firsts[p]:lasts[p], bytes[p],
                      UInt32[sums[i, p] for i in columns]) for p in 1:k]
end

# The rank that reads each part: the owner of its first block under the
# new partition, moved on to the next rank while that one already reads
# `cld(k, P)` parts, so that no rank reads more than its share when that
# is possible. A function of the replicated part table alone, so the
# same on every rank.
function part_readers(table, n, P)
    k = length(table)
    share = cld(k, P)
    count = zeros(Int, P)
    readers = Vector{Int}(undef, k)
    previous = 0
    for p in 1:k
        q = max(previous, equalsplit_part(n, P, clamp(first(table[p].blocks), 1, n)) - 1)
        while count[q + 1] >= share && q < P - 1
            q += 1
        end
        readers[p] = q
        count[q + 1] += 1
        previous = q
    end
    return readers
end

# A part opened by its reader and checked against the index: its file
# (`nothing` when it is inside the index), and per field set its `data`
# dataset and its stored per-block checksums (`nothing` in a version-1
# file written before there were any).
struct OpenPart
    file::Union{Nothing,HDF5.File}
    datasets::Vector{HDF5.Dataset}
    sums::Vector{Union{Nothing,Vector{UInt32}}}
end

function open_part(entry::PartEntry, j, version, real, dir, saveid, layouts, forest,
                   context)
    note = writer_note(context)
    what = entry.file == "" ? "the part inside $(repr(context.path))" :
           "part $j of $(repr(context.path)), $(repr(entry.file)),"
    refuse(why) = throw(ArgumentError(
        "$what $why, so the checkpoint is refused rather than read. " * note))
    file = nothing
    if entry.file == ""
        real === nothing && error("the part inside the index is read by rank 0, which " *
                                  "opened the index; this rank did not")
        inside = version == 1 ? GROUP : GROUP * "/parts/" * linkname(0)
        haskey(real, inside) || refuse("is missing")
        g = real[inside]
    else
        full = joinpath(dir, entry.file)
        isfile(full) || refuse("is missing")
        filesize(full) == entry.bytes ||
            refuse("has $(filesize(full)) bytes where the index records $(entry.bytes): " *
                   "it was truncated, or replaced")
        HDF5.ishdf5(full) || refuse("is not an HDF5 file")
        file = open_h5(full, "r")
        haskey(file, GROUP) || (close(file); refuse("has no /$GROUP group"))
        g = file[GROUP]
    end
    try
        if version == 2
            attr(name) = hasattr(g, name) ? read_attribute(g, name) : nothing
            attr("format") == PART_FORMAT || refuse("is not a TreeAMR checkpoint part")
            attr("save_id") == saveid ||
                refuse("belongs to another save (its save id is $(repr(attr("save_id"))), " *
                       "the index's $(repr(saveid))): it is left from an earlier " *
                       "checkpoint, or from one that was not completed")
            (attr("part") == j && attr("first_block") == first(entry.blocks) &&
             attr("last_block") == last(entry.blocks)) ||
                refuse("is part $(attr("part")), of blocks $(attr("first_block")) to " *
                       "$(attr("last_block")), where the index names part $j, of blocks " *
                       "$(first(entry.blocks)) to $(last(entry.blocks))")
        end
        D = dimension(forest)
        nb = length(entry.blocks)
        datasets = HDF5.Dataset[]
        sums = Union{Nothing,Vector{UInt32}}[]
        for (s, layout) in enumerate(layouts)
            key = "fieldsets/$(layout.name)"
            haskey(g, key * "/data") || refuse("holds no data of field set " *
                                               repr(layout.name))
            dset = g[key * "/data"]
            push!(datasets, dset)
            dims = (block_dims(forest.N, Val(D), layout.limbs)..., layout.nvars, nb)
            check_dataset(dset, layout.F, dims, "the data of field set " *
                          "$(repr(layout.name)) in $what", note)
            if haskey(g, key * "/data_crc32c")
                stored = read_array(g[key * "/data_crc32c"], UInt32, (nb,),
                                    "the checksums of field set $(repr(layout.name))", note)
                entry.sums === nothing || crc_of(stored) == entry.sums[s] ||
                    refuse("holds checksums of field set $(repr(layout.name)) that do not " *
                           "match the index's checksum of them: it was damaged, or is " *
                           "from another save")
                push!(sums, stored)
            else
                version == 1 || refuse("has no checksums of field set " *
                                       repr(layout.name))
                push!(sums, nothing)
            end
        end
        return OpenPart(file, datasets, sums)
    catch
        file === nothing || close(file)
        rethrow()
    end
end

function load_from(f, meta, real, r::Ranks, path, types, backend, fieldsets)
    root = meta[GROUP]
    context = context_of(root, path)
    version = check_format(root, context)
    forest = read_forest(root["forest"], types, context, r.comm, version)
    D = dimension(forest)
    n = nleaves(forest)
    group = root["fieldsets"]
    names = select_fieldsets(group, fieldsets, context)
    layouts = [read_layout(group[name], name, Val(D), types, context) for name in names]
    # The constructor validates each layout against the forest, as it
    # would a caller's, and `statevector` places the pages by owner, which
    # is what decides their NUMA domain; the reads below only fill them.
    sets = map(layouts) do l
        fs = FieldSet{l.T}(forest, l.nvars; G=l.G, centering=l.centering, parity=l.parity,
                           backend=backend)
        return (; fieldset=fs, state=statevector(fs))
    end
    table = read_parttable(root, version, n, names, context)
    saveid = version == 2 ? read_attribute(root, "save_id") : ""
    readers = part_readers(table, n, r.size)
    mine = [p for p in eachindex(table) if readers[p] == r.rank]
    opened = Dict{Int,OpenPart}()
    try
        err = nothing
        for p in mine
            err = attempt(err) do
                opened[p] = open_part(table[p], p - 1, version, real,
                                      dirname(abspath(path)), saveid, layouts, forest,
                                      context)
            end
        end
        agree_errors(r, err, "load_checkpoint, opening the parts,")
        for (s, layout) in enumerate(layouts)
            read_blocks!(sets[s].state, r, s, layout, table, readers, opened, forest,
                         context)
        end
    finally
        for part in values(opened)
            part.file === nothing || close(part.file)
        end
    end
    for set in sets
        scatter!(set.fieldset, set.state)
    end
    appname = read_attribute(root, "application")
    haskey(meta, appname) || throw(ArgumentError(
        "$(repr(context.path)) names the application $(repr(appname)), but has no " *
        "/$appname group: the file is damaged. " * writer_note(context)))
    app = meta[appname]
    version′ = read_attribute(app, "format_version")
    data = read_plain(app, "data")
    result = f === nothing ? nothing : f(app)
    return (; forest=forest, fieldsets=Dict{String,Any}(zip(names, sets)),
            application=appname => version′, data=data, provenance=context.provenance,
            result=result)
end

# One field set's blocks into `u`: each owner posts its receives straight
# into its state vector, and each reader reads its parts' blocks for each
# owner by hyperslab, a piece at a time, checks each block's CRC-32C, and
# sends it, with at most two pieces in flight; a piece for itself it reads
# in place. A reader whose read fails goes on sending, so that no owner
# waits forever, and the failure and the damage are agreed afterwards.
function read_blocks!(u, r::Ranks, s, layout, table, readers, opened, forest, context)
    n, P = nleaves(forest), r.size
    T, F = layout.T, layout.F
    D = dimension(forest)
    per = block_length(forest.N, D, layout.nvars)
    bytes = per * sizeof(T)
    host = u isa Array ? u : Vector{T}(undef, length(u))
    mine = blockrange(forest)
    offset = first(mine) - 1
    requests = Any[]
    for p in eachindex(table)
        readers[p] == r.rank && continue
        for piece in pieces(intersect(table[p].blocks, mine), bytes)
            push!(requests, irecv(r.comm, view(host, elements(piece .- offset, per)),
                                  readers[p], TAG_LOAD))
        end
    end
    err = nothing
    bad = Int[]
    buffers = (Vector{T}(undef, 0), Vector{T}(undef, 0))
    pending = Any[nothing, nothing]
    turn = 1
    for p in eachindex(table)
        readers[p] == r.rank || continue
        entry, part = table[p], opened[p]
        dset, stored = part.datasets[s], part.sums[s]
        dims = (block_dims(forest.N, Val(D), layout.limbs)..., layout.nvars,
                length(entry.blocks))
        isempty(entry.blocks) && continue
        owner(b) = equalsplit_part(n, P, b) - 1
        owners = owner(first(entry.blocks)):owner(last(entry.blocks))
        for q in owners,
            piece in pieces(intersect(entry.blocks, rankblocks(n, P, q)), bytes)
            rows = piece .- (first(entry.blocks) - 1)
            if q == r.rank
                buf, start = host, (first(piece) - offset - 1) * per + 1
            else
                turn = 3 - turn
                pending[turn] === nothing || waitall(r.comm, [pending[turn]])
                pending[turn] = nothing
                buf, start = buffers[turn], 1
                length(buf) < length(piece) * per && resize!(buf, length(piece) * per)
            end
            err = attempt(err) do
                GC.@preserve buf block_slab(API.h5d_read, dset, F, dims, rows,
                                            pointer(buf, start))
                if stored !== nothing
                    got = block_checksums(buf, length(piece), per, start)
                    for (i, b) in enumerate(piece)
                        got[i] == stored[rows[i]] || push!(bad, b)
                    end
                end
            end
            q == r.rank ||
                (pending[turn] = isend(r.comm, view(buf, 1:(length(piece) * per)), q,
                                       TAG_LOAD))
        end
    end
    waitall(r.comm, [requests; filter(!isnothing, pending)])
    agree_errors(r, err, "load_checkpoint, reading the parts,")
    count, b = damage(r, sort!(bad))
    count == 0 || throw(ArgumentError(
        "the data of field set $(repr(layout.name)) do not match the checksums stored " *
        "with them in $count of its $n blocks, the first being block $b (a CRC-32C per " *
        "block): the file was damaged while or after it was written, and is refused " *
        "rather than restored into a wrong state. " * writer_note(context)))
    host === u || copyto!(u, host)
    return u
end

function checkpoint_environment(path::AbstractString, dir::AbstractString;
                                force::Bool=false)
    file = open_index(path)
    p = try
        read_provenance(file[GROUP])
    finally
        close(file)
    end
    files = Pair{String,String}[]
    isempty(p.project) || push!(files, joinpath(dir, "Project.toml") => p.project)
    isempty(p.manifest) || push!(files, joinpath(dir, "Manifest.toml") => p.manifest)
    isempty(files) && throw(ArgumentError(
        "$(repr(path)) stores neither a Project.toml nor a Manifest.toml: the process " *
        "that wrote it had no active project file, so its environment cannot be " *
        "recreated from the file. It was written by " *
        (p.treeamr_version === nothing ? "an unknown version of TreeAMR" :
         "TreeAMR $(p.treeamr_version)") *
        (p.julia_version === nothing ? "" : " under Julia $(p.julia_version)") * "."))
    if !force
        for (target, _) in files
            ispath(target) && throw(ArgumentError(
                "$(repr(target)) exists already, and checkpoint_environment does not " *
                "overwrite a file unless asked to: pass `force = true`"))
        end
    end
    mkpath(dir)
    for (target, text) in files
        write(target, text)
    end
    isempty(p.manifest) && @warn(
        "$(repr(path)) stores no Manifest.toml: the environment that wrote it had none " *
        "beside its Project.toml. Only the Project.toml was written, so " *
        "`Pkg.instantiate` there resolves versions afresh rather than reproducing the " *
        "writer's.")
    isempty(p.project) && @warn(
        "$(repr(path)) stores no Project.toml: the environment that wrote it had no " *
        "project file. Only the Manifest.toml was written.")
    return dir
end

# --- plain data ------------------------------------------------------------

function write_plain(parent::Union{HDF5.File,HDF5.Group}, name::AbstractString, value)
    # The value is walked before anything is written, so that what cannot
    # be stored is refused with nothing created. In the do-block of a save
    # over several ranks the value has to be the same on every rank, since
    # rank 0's is the one the index keeps, and the walk's hash is agreed
    # before anything is written.
    check() = begin
        check_name(name, "a plain-data item's name")
        haskey(parent, name) && throw(ArgumentError(
            "$(HDF5.name(parent)) already holds an item named $(repr(name)); each item " *
            "is written once"))
        datahash = plain_hash(itempath(parent, String(name)), value, UInt(0))
        (nothing, layouthash(HDF5.name(parent), String(name)), datahash)
    end
    r = saving_ranks(parent)
    r === nothing ? check() :
    agreed(check, r, "write_plain of $(itempath(parent, String(name)))")
    put_plain(parent, String(name), value)
    return nothing
end

itempath(parent, name) = rstrip(HDF5.name(parent), '/') * "/" * name

function put_plain(parent, name, x::NativeNumber)
    T = typeof(x)
    withtype(T) do dt
        space = HDF5.Dataspace(API.h5s_create(API.H5S_SCALAR))
        dset = try
            create_dataset(parent, name, dt, space)
        finally
            close(space)
        end
        try
            API.h5d_write(dset, dt, API.H5S_ALL, API.H5S_ALL, API.H5P_DEFAULT, Ref(x))
            tag!(dset, "number"; eltype=typename(T))
        finally
            close(dset)
        end
    end
    return nothing
end

put_plain(parent, name, x::Rational{I}) where {I<:NativeInteger} =
    write_array(parent, name, I, (2,), I[numerator(x), denominator(x)];
                tag="rational", eltype=typename(I))

put_plain(parent, name, x::AbstractString) = put_string(parent, name, String(x), "string")
put_plain(parent, name, x::Symbol) = put_string(parent, name, String(x), "symbol")
put_plain(parent, name, x::VersionNumber) = put_string(parent, name, string(x), "version")

function put_string(parent, name, value, tag)
    dset, dt = create_dataset(parent, name, value)
    try
        HDF5.write_dataset(dset, dt, value)
        tag!(dset, tag; (tag == "array" ? (; eltype="String") : (;))...)
    finally
        close(dset)
        close(dt)
    end
    return nothing
end

put_plain(parent, name, ::Nothing) = put_group(parent, name, "nothing", ())

function put_plain(parent, name, x::AbstractArray{T}) where {T<:NativeNumber}
    isconcretetype(T) || throw(not_plain(parent, name, x))
    buf = x isa Array ? x : collect(x)
    write_array(parent, name, T, size(buf), buf; tag="array", eltype=typename(T))
    return nothing
end

put_plain(parent, name, x::AbstractArray{<:AbstractString}) =
    put_string(parent, name, convert(Array{String}, x), "array")

function put_plain(parent, name, x::Tuple)
    T = isempty(x) ? Nothing : typeof(first(x))
    if T <: NativeNumber && all(y -> typeof(y) === T, x)
        write_array(parent, name, T, (length(x),), T[x...]; tag="tuple",
                    eltype=typename(T))
    else
        put_group(parent, name, "tuple", (string(i) => y for (i, y) in enumerate(x)))
    end
    return nothing
end

put_plain(parent, name, x::NamedTuple) = put_group(parent, name, "namedtuple", pairs(x))

function put_plain(parent, name, x::AbstractDict{K}) where {K}
    keytype = K <: AbstractString ? "String" : K === Symbol ? "Symbol" :
              throw(not_plain(parent, name, x))
    put_group(parent, name, "dict", pairs(x); keytype=keytype)
    return nothing
end

put_plain(parent, name, x) = throw(not_plain(parent, name, x))

# Groups keep the creation order of their items, which is how a
# NamedTuple keeps its field order.
function put_group(parent, name, tag, items; attrs...)
    g = create_group(parent, name; track_order=true)
    try
        tag!(g, tag; attrs...)
        for (key, value) in items
            key = string(key)
            check_name(key, "the plain-data item name in $(itempath(parent, name))")
            put_plain(g, key, value)
        end
    finally
        close(g)
    end
    return nothing
end

# The plain-data walk: what `put_plain` would write, as a hash, refusing
# what it would refuse, with the same messages, but before anything is
# written. A save checks its `data` this way before it creates the file,
# and over several ranks the hash is what the ranks agree on, since rank
# 0 writes every item, with its own value, for all of them. The hash follows the
# order of the writes — a Dict's items in its iteration order, which is
# their order in the file — and the bits of every number, so that `0.0`
# and `-0.0` differ, as they do in the file. Every hash is of integers
# and strings, which are the same in every process (as for the forest
# digest).
bitsof(x::Bool) = UInt8(x)
bitsof(x::NativeInteger) = x
bitsof(x::Float16) = reinterpret(UInt16, x)
bitsof(x::Float32) = reinterpret(UInt32, x)
bitsof(x::Float64) = reinterpret(UInt64, x)
bitsof(x::Complex) = (bitsof(real(x)), bitsof(imag(x)))

plain_hash(path, x::NativeNumber, h) =
    hash(("number", typename(typeof(x)), bitsof(x)), h)
plain_hash(path, x::Rational{I}, h) where {I<:NativeInteger} =
    hash(("rational", typename(I), bitsof(numerator(x)), bitsof(denominator(x))), h)
plain_hash(path, x::AbstractString, h) = hash(("string", no_nul(path, String(x))), h)
plain_hash(path, x::Symbol, h) = hash(("symbol", no_nul(path, String(x))), h)
plain_hash(path, x::VersionNumber, h) = hash(("version", string(x)), h)
plain_hash(path, ::Nothing, h) = hash("nothing", h)

function plain_hash(path, x::AbstractArray{T}, h) where {T<:NativeNumber}
    isconcretetype(T) || throw(not_plain_at(path, x))
    h = hash(("array", typename(T), size(x)), h)
    for y in x
        h = hash(bitsof(y), h)
    end
    return h
end

function plain_hash(path, x::AbstractArray{<:AbstractString}, h)
    h = hash(("array", "String", size(x)), h)
    for y in x
        h = hash(no_nul(path, String(y)), h)
    end
    return h
end

function plain_hash(path, x::Tuple, h)
    T = isempty(x) ? Nothing : typeof(first(x))
    if T <: NativeNumber && all(y -> typeof(y) === T, x)
        h = hash(("tuple", typename(T), length(x)), h)
        for y in x
            h = hash(bitsof(y), h)
        end
        return h
    end
    return group_hash(path, "tuple", (string(i) => y for (i, y) in enumerate(x)), h)
end

plain_hash(path, x::NamedTuple, h) =
    group_hash(path, "namedtuple", pairs(x), h)

function plain_hash(path, x::AbstractDict{K}, h) where {K}
    keytype = K <: AbstractString ? "String" : K === Symbol ? "Symbol" :
              throw(not_plain_at(path, x))
    return group_hash(path, "dict", pairs(x), hash(keytype, h))
end

plain_hash(path, x, h) = throw(not_plain_at(path, x))

# HDF5 stores a string as a C string, which ends at its first NUL, and
# HDF5.jl refuses one that holds a NUL; the walk refuses it first, with
# the item's path, before anything is written.
function no_nul(path, s::String)
    occursin('\0', s) && throw(ArgumentError(
        "$path holds a string with a NUL character, which HDF5 cannot store: it keeps a " *
        "string as a C string, which ends at its first NUL. Store such a string as an " *
        "array of UInt8."))
    return s
end

function group_hash(path, tag, items, h)
    h = hash(tag, h)
    for (key, value) in items
        key = string(key)
        check_name(key, "the plain-data item name in $path")
        h = plain_hash(path * "/" * key, value, hash(key, h))
    end
    return h
end

not_plain(parent, name, x) = not_plain_at(itempath(parent, name), x)

function not_plain_at(path, x)
    kind = x isa AbstractArray ? "an array of $(eltype(x)), and an array is plain data " *
                                 "only with a native number type or String as its " *
                                 "element type" :
           x isa AbstractDict ? "a Dict with keys of type $(keytype(x)), and a Dict is " *
                                "plain data only with String or Symbol keys" :
           "a $(typeof(x)), which is not plain data"
    return ArgumentError(
        "$path is $kind. Plain data are numbers of the HDF5 native types, Rationals " *
        "of them, strings, Symbols, VersionNumbers, nothing, arrays, tuples, " *
        "NamedTuples and Dicts of those (see `write_plain`). Convert anything else to " *
        "a NamedTuple of plain values — a struct field by field — since a type's name " *
        "in the file would tie the file to that type's definition.")
end

function read_plain(parent::Union{HDF5.File,HDF5.Group}, name::AbstractString)
    haskey(parent, name) || throw(ArgumentError(
        "$(HDF5.name(parent)) holds no item named $(repr(name))"))
    obj = parent[name]
    try
        return get_plain(obj)
    finally
        close(obj)
    end
end

function native_eltype(obj, path)
    name = read_attribute(obj, "eltype")
    T = get(NATIVE_BY_NAME, name, nothing)
    T === nothing && throw(ArgumentError(
        "$path has the element type $(repr(name)), which is not a native number type"))
    return T
end

function get_plain(obj)
    path = HDF5.name(obj)
    hasattr(obj, "type") || throw(ArgumentError(
        "$path has no `type` attribute, so it is not plain data written by `write_plain`"))
    tag = read_attribute(obj, "type")
    if obj isa HDF5.Group
        tag == "nothing" && return nothing
        if tag == "tuple"
            return Tuple(read_plain(obj, string(i)) for i in 1:length(keys(obj)))
        elseif tag == "namedtuple"
            names = keys(obj)
            return NamedTuple{Tuple(map(Symbol, names))}(
                Tuple(read_plain(obj, k) for k in names))
        elseif tag == "dict"
            keytype = read_attribute(obj, "keytype")
            K = keytype == "String" ? String : keytype == "Symbol" ? Symbol :
                throw(ArgumentError("$path is a dict with keytype $(repr(keytype)), not " *
                                    "\"String\" or \"Symbol\""))
            dict = Dict{K,Any}()
            for k in keys(obj)
                dict[K(k)] = read_plain(obj, k)
            end
            return dict
        end
    else
        if tag == "number"
            return read_scalar(obj, native_eltype(obj, path), path)
        elseif tag == "rational"
            I = native_eltype(obj, path)
            I <: NativeInteger || throw(ArgumentError(
                "$path is a rational of $I, which is not an integer type"))
            parts = read_array(obj, I, (2,), path)
            return Rational{I}(parts[1], parts[2])
        elseif tag == "string"
            return read(obj, String)
        elseif tag == "symbol"
            return Symbol(read(obj, String))
        elseif tag == "version"
            return VersionNumber(read(obj, String))
        elseif tag == "array"
            read_attribute(obj, "eltype") == "String" && return read(obj)
            return read_array(obj, native_eltype(obj, path), size(obj), path)
        elseif tag == "tuple"
            return Tuple(read_array(obj, native_eltype(obj, path), (length(obj),), path))
        end
    end
    throw(ArgumentError(
        "$path has the plain-data type $(repr(tag)), which this version does not read: " *
        "it was written by a newer version, or not by `write_plain`"))
end

end
