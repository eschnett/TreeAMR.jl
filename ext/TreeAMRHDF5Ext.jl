# Checkpoint and restart through HDF5 (M9a): the implementation of the
# functions declared, and documented, in `src/checkpoint.jl`.
#
# The file layout is format version 1 of `CODE.md`, "Checkpoint and
# restart", which is the specification this follows object for object:
#
#     /TreeAMR.jl/              format, format_version, features, application
#       provenance/             who wrote the file, and in which environment
#       forest/                 the brick, the extents, the leaves as columns
#       fieldsets/<name>/       the layout, and `data`, the owned points
#     /<application>/           format_version (the application's own)
#       data                    the plain-data tree
#
# Two rules shape everything below. The file holds bits and a small
# documented vocabulary of names, never a Julia type's definition, so
# that a reader in another version, or in another language, can
# interpret it; and a load builds fresh objects through the package's
# own validating constructors, so that nothing read from a file is
# trusted before the forest and the field set have checked it.
#
# Nothing here loops over cells on the host. The field data go to and
# from HDF5 as one whole-array transfer each, straight from and into the
# state vector on the CPU — whose pages `statevector` has already placed
# by owner — and through one host buffer on a device. The per-leaf work,
# splitting keys into columns and building them back, runs through
# `threaded_foreach`, by owner.
#
# Over a distributed forest (M7) the same code writes and reads the same
# file from every rank at once, through parallel HDF5: see "Serial and
# parallel access" below, which is the only place the two differ.

module TreeAMRHDF5Ext

using HDF5: HDF5, API, h5open, create_group, create_dataset, dataspace, attributes,
            read_attribute, write_attribute, write_dataset
using KernelAbstractions: CPU, Backend, get_backend
using TreeAMR
using TreeAMR: threaded_foreach, tohost, samebackend, Communicator, commrank, commsize,
               allgather, allgatherv, librarycomm, open_parallel_file, ForestDigest,
               digest_verdict, layouthash
import TreeAMR: save_checkpoint, load_checkpoint, write_plain, read_plain,
                checkpoint_environment

const GROUP = "TreeAMR.jl"
const FORMAT = "TreeAMR checkpoint"
const FORMAT_VERSION = 1
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

# --- serial and parallel access (M7) ---------------------------------------
#
# A checkpoint of a serial forest is the whole file in one process, and
# every HDF5 call below is the one M9a made: that is `Alone`. A
# checkpoint of a forest distributed over several ranks is the same file,
# written or read by all of them at once through parallel HDF5
# ("Parallel checkpoints" in CODE.md), and differs in three places:
#
# - the file is opened over the forest's communicator, by
#   `open_parallel_file`, whose method is `TreeAMRHDF5MPIExt`;
# - a dataset over the blocks — the leaf columns and each field set's
#   data — is written and read by last-axis hyperslab, each rank its own
#   `blockrange`, in one collective transfer, which a filtered dataset
#   requires; a rank without blocks takes part with an empty selection;
# - any other dataset, which every rank holds whole, is written once, by
#   rank 0, with an independent transfer, which is legal because parallel
#   HDF5 allocates an unfiltered dataset's storage when it is created
#   (`H5D__create` in libhdf5 2.2.0's `H5Dint.c`), and read by every rank.
#
# Everything else — creating a group, a dataset or an attribute, writing
# an attribute, closing the file — is made by every rank with the same
# arguments, the collective contract of parallel HDF5. The agreement
# before the file is opened is what makes the arguments the same.
abstract type Access end

struct Alone <: Access end

struct Ranked <: Access
    comm::Communicator
    rank::Int
    size::Int
    collective::HDF5.DatasetTransferProperties
end

access_for(comm::Communicator) =
    commsize(comm) == 1 ? Alone() :
    Ranked(comm, commrank(comm), commsize(comm),
           HDF5.DatasetTransferProperties(; dxpl_mpio=:collective))

release(::Alone) = nothing
release(a::Ranked) = (close(a.collective); nothing)

# The parallel files open in this process, so that `write_plain`, which is
# handed a group, knows whether its write is collective.
const OPEN_RANKED = IdDict{HDF5.File,Ranked}()
const OPEN_LOCK = ReentrantLock()

access_of(obj) = lock(() -> get(OPEN_RANKED, HDF5.file(obj), Alone()), OPEN_LOCK)

open_file(f, ::Alone, path, mode) = h5open(f, path, mode)

function open_file(f, a::Ranked, path, mode)
    file = open_parallel_file(librarycomm(a.comm), path, mode)
    lock(() -> (OPEN_RANKED[file] = a), OPEN_LOCK)
    try
        return f(file)
    finally
        lock(() -> delete!(OPEN_RANKED, file), OPEN_LOCK)
        close(file)                      # collective
    end
end

# A dataset that every rank holds whole: written by `write`, once.
write_whole(write, ::Alone) = write()
write_whole(write, a::Ranked) = (a.rank == 0 && write(); nothing)

# A dataset over the blocks, of dimensions `dims` in Julia's order with
# the blocks last: this rank's `count` blocks from `offset` on, from or
# into `buf`. Serially `buf` is the whole dataset, as in M9a.
write_slab(dset, dt, buf, dims, slab, ::Alone) = HDF5.write_dataset(dset, dt, buf)
read_slab(dset, dt, buf, dims, slab, ::Alone) = HDF5.read_dataset(dset, dt, buf)

write_slab(dset, dt, buf, dims, slab, a::Ranked) =
    transfer_slab(API.h5d_write, dset, dt, buf, dims, slab, a)
read_slab(dset, dt, buf, dims, slab, a::Ranked) =
    transfer_slab(API.h5d_read, dset, dt, buf, dims, slab, a)

function transfer_slab(transfer, dset, dt, buf, dims, (offset, count), a::Ranked)
    filespace = HDF5.dataspace(dset)
    memspace = HDF5.dataspace(count == 0 ? (1,) : (dims[1:(end - 1)]..., count))
    try
        if count == 0
            select_none(filespace)
            select_none(memspace)
        else
            # C's order is Julia's reversed: the block axis comes first.
            start = API.hsize_t[offset; zeros(Int, length(dims) - 1)]
            counts = API.hsize_t[count; reverse(collect(dims[1:(end - 1)]))]
            API.h5s_select_hyperslab(filespace, API.H5S_SELECT_SET, start, C_NULL, counts,
                                     C_NULL)
        end
        transfer(dset, dt, memspace, filespace, a.collective, buf)
    finally
        close(memspace)
        close(filespace)
    end
    return nothing
end

# HDF5.jl has no wrapper for `H5Sselect_none`.
function select_none(space)
    API.lock(API.liblock)
    status = try
        ccall((:H5Sselect_none, API.libhdf5), API.herr_t, (API.hid_t,), space)
    finally
        API.unlock(API.liblock)
    end
    status < 0 && error("H5Sselect_none failed")
    return nothing
end

# This rank's part of a dataset over the forest's leaves: `(offset,
# count)`, every leaf serially.
slab_of(forest::Forest) = (first(blockrange(forest)) - 1, length(blockrange(forest)))

# --- agreement before the file is opened (M7) ------------------------------
#
# Over a distributed forest every refusal is decided before the first
# collective HDF5 call, and agreed: one `allgather` of each rank's verdict
# and of a hash of what the ranks must pass alike, so that a refusal on
# some ranks is raised on all of them together, with the reason, rather
# than leaving the others waiting inside HDF5. `ForestDigest` and
# `digest_verdict` are the schedule builds' (`forest.jl`); a load, which
# has no forest yet, gathers a digest with the forest's fields zeroed, as
# `interpolate` does. Beside it goes the hash of the plain data, which
# gets a refusal of its own.
function agreed(check, a::Ranked, what; forest=nothing)
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
    gathered = allgather(a.comm, (digest, datahash))
    digest_verdict(map(first, gathered), what, a.rank, refusal)
    differ = [r - 1 for r in 2:a.size if last(gathered[r]) != last(gathered[1])]
    isempty(differ) || throw(ArgumentError(
        "the plain data of $what differ between ranks: rank(s) $(join(differ, ", ")) " *
        "of $(a.size) pass other values than rank 0, so it is refused on every rank, " *
        "this one (rank $(a.rank)) included. A checkpoint is one file, and each item " *
        "in it is written once, with one value, for every rank; a value that is per " *
        "rank belongs in a field set, or is gathered to every rank first."))
    return first(checked)
end

agreed(check, ::Alone, what; forest=nothing) = first(check())

# The texts and numbers of the provenance, every rank the same: rank 0's,
# broadcast through `allgatherv`, to which the others contribute nothing.
function agree_values(::Alone, values)
    return values
end

function agree_values(a::Ranked, values)
    io = IOBuffer()
    if a.rank == 0
        for v in values
            bytes = v isa AbstractString ? codeunits(String(v)) :
                    reinterpret(UInt8, [Int64(v)])
            write(io, Int64(length(bytes)), bytes)
        end
    end
    io = IOBuffer(allgatherv(a.comm, take!(io)))
    return map(values) do v
        bytes = read(io, read(io, Int64))
        v isa AbstractString ? String(bytes) : only(reinterpret(Int64, bytes))
    end
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
# `dims`, in Julia's order, and write `buf` into it whole. `buf` may be
# an array of any isbits type whose bytes are `F`s, since HDF5 checks
# only the byte count: an array of a limb type is written from its own
# memory, with no copy (a `reinterpret` would cost one, because HDF5.jl
# copies a `ReinterpretArray` to take a pointer to it). A plain-data item
# passes its `tag`; the TreeAMR group's own datasets carry none, since
# the specification describes them. A dataset over the blocks passes its
# `slab`, this rank's part, and `buf` holds only that part.
function write_array(parent, name, ::Type{F}, dims::Dims, buf; chunk=nothing,
                     filters=(), tag=nothing, slab=nothing, attrs...) where {F}
    access = access_of(parent)
    withtype(F) do dt
        space = dataspace(dims)
        props = isempty(filters) ? (;) : (; chunk=chunk, filters=filters)
        dset = try
            create_dataset(parent, name, dt, space; props...)
        finally
            close(space)
        end
        try
            if slab === nothing
                write_whole(() -> HDF5.write_dataset(dset, dt, buf), access)
            else
                write_slab(dset, dt, buf, dims, slab, access)
            end
            tag === nothing || tag!(dset, tag; attrs...)
        finally
            close(dset)
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

function read_array!(buf, dset, ::Type{F}, dims, what, note; slab=nothing) where {F}
    check_dataset(dset, F, dims, what, note)
    withtype(F) do dt
        if slab === nothing
            HDF5.read_dataset(dset, dt, buf)
        else
            read_slab(dset, dt, buf, dims, slab, access_of(dset))
        end
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

# --- provenance ----------------------------------------------------------

# Who wrote the file. Over several ranks these are rank 0's values,
# which every rank writes, since an object is created collectively with
# one value: `created`, `hostname` and `nthreads` can differ between
# ranks. `nranks` is the one field M7 added (decided 2026-10-01 with
# Erik); a file without it was written serially.
const PROVENANCE = ("treeamr_version", "julia_version", "created", "hostname", "nthreads",
                    "nranks", "project", "manifest")

function provenance_values(nranks)
    project, manifest = environment_texts()
    version = pkgversion(TreeAMR)
    return (version === nothing ? "" : string(version), string(VERSION), utc_now(),
            gethostname(), Int64(Threads.nthreads()), Int64(nranks), project, manifest)
end

function write_provenance(root, values)
    access = access_of(root)
    g = create_group(root, "provenance"; track_order=true)
    try
        for (name, value) in zip(PROVENANCE, values)
            put_value(access, g, name, value)
        end
    finally
        close(g)
    end
    return nothing
end

# A scalar dataset of a string or an Int64, created by every rank and
# written by one.
put_value(::Alone, g, name, value) = write_dataset(g, name, value)

function put_value(a::Ranked, g, name, value)
    dset, dt = create_dataset(g, name, value)
    try
        write_whole(() -> HDF5.write_dataset(dset, dt, value), a)
    finally
        close(dset)
        close(dt)
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
        # The leaves as columns, in curve order: block `b` of every field
        # set is row `b`, and any run of blocks is one hyperslab. Each rank
        # writes the rows of its own blocks; serially that is every row.
        n = nleaves(forest)
        slab = slab_of(forest)
        offset, m = slab
        roots = Vector{Int32}(undef, m)
        levels = Vector{Int8}(undef, m)
        coords = Matrix{UInt32}(undef, D, m)
        threaded_foreach(m) do i
            k = forest.leaves[offset + i]
            roots[i] = k.root
            levels[i] = k.level
            for d in 1:D
                coords[d, i] = k.coords[d]
            end
        end
        write_array(g, "root", Int32, (n,), roots; slab=slab)
        write_array(g, "level", Int8, (n,), levels; slab=slab)
        write_array(g, "coords", UInt32, (D, n), coords; slab=slab)
    finally
        close(g)
    end
    return nothing
end

function read_forest(g, types, context, comm)
    note = writer_note(context)
    connectivity = read_attribute(g, "connectivity")
    connectivity == "brick" || throw(ArgumentError(
        "the forest's connectivity is $(repr(connectivity)), and this version reads only " *
        "\"brick\". " * note))
    D = read_attribute(g, "D")
    (D isa Integer && D >= 1) || throw(ArgumentError(
        "the forest's dimension, D = $(repr(D)), is not a positive integer: the file is " *
        "damaged. " * note))
    return read_forest(g, Val(Int(D)), types, context, comm)
end

# Every rank reads every leaf, since the forest is replicated, and builds
# it over `comm`.
function read_forest(g, ::Val{D}, types, context, comm) where {D}
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

function write_fieldset(parent, name, fs::FieldSet{T,D}, u, filters) where {T,D}
    if u === nothing
        u = statevector(fs)
        gather!(u, fs)
    end
    F, limbs = storage_of(T, "field set $(repr(name))")
    N = fs.forest.N
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
        # One chunk per block and variable, when there is anything to
        # filter: reading a block then decompresses that block alone, and
        # over several ranks every chunk has one writer.
        block = (limbs..., ntuple(_ -> N, D)...)
        write_array(g, "data", F, (block..., fs.nvars, nleaves(fs.forest)), tohost(u);
                    chunk=(block..., 1, 1), filters=filters, slab=slab_of(fs.forest))
    finally
        close(g)
    end
    return nothing
end

function read_fieldset(g, name, forest::Forest{D}, types, backend, context) where {D}
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
    # The constructor validates the layout against the forest, as it
    # would a caller's, and `statevector` places the pages by owner, which
    # is what decides their NUMA domain; the read below only fills them.
    fs = FieldSet{T}(forest, nvars; G=G, centering=centering, parity=parity,
                     backend=backend)
    u = statevector(fs)
    # The whole dataset's dimensions, of which this rank reads its blocks.
    dims = (limbs..., ntuple(_ -> forest.N, D)..., nvars, nleaves(forest))
    slab = slab_of(forest)
    if u isa Array
        read_array!(u, g["data"], F, dims, "the data of $what", note; slab=slab)
    else
        host = Vector{T}(undef, length(u))
        read_array!(host, g["data"], F, dims, "the data of $what", note; slab=slab)
        copyto!(u, host)
    end
    scatter!(fs, u)
    return (; fieldset=fs, state=u)
end

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

# --- saving and loading ----------------------------------------------------

save_checkpoint(path::AbstractString, forest::Forest; kwargs...) =
    save_checkpoint(nothing, path, forest; kwargs...)

function save_checkpoint(f, path::AbstractString, forest::Forest; fieldsets=nothing,
                         application=nothing, data=(;), filters=(), sync::Bool=true)
    access = access_for(forest.comm)
    try
        # Every refusal before the file is created, and over several ranks
        # agreed, so that a refusal is raised on all of them together.
        checked = agreed(access, "save_checkpoint"; forest=forest) do
            c = check_save(forest, fieldsets, application, filters)
            appname, appversion, sets, filters′ = c
            datahash = plain_hash("/$appname/data", data, UInt(0), access isa Ranked)
            layout = layouthash(String(path), appname, appversion, string(filters′), sync,
                                f === nothing, map(set_layout, sets))
            return (c, layout, datahash)
        end
        write_checkpoint(f, access, path, forest, checked..., data, sync)
    finally
        release(access)
    end
    return path
end

function check_save(forest, fieldsets, application, filters)
    fieldsets === nothing && throw(ArgumentError(
        "save_checkpoint has no default `fieldsets`: pass the field sets the " *
        "application evolves, as `name => (fs, u)` pairs, or `()` for none. Which sets " *
        "are state and which are scratch, rebuilt every step, is the application's to " *
        "say."))
    appname, appversion = check_application(application)
    sets = collect_fieldsets(fieldsets, forest)
    storage_of(floattype(forest), "the forest's geometry")
    filters isa HDF5.Filters.Filter && (filters = (filters,))
    return (appname, appversion, sets, filters)
end

# What the ranks must agree on about one field set: everything that
# shapes the objects created for it.
set_layout((name, fs, u)) =
    (name, typename(eltype(fs.work)), fs.nvars, fs.G, string.(fs.centering),
     fs.parity === nothing ? "" : string(map(p -> map(parity_name, p), fs.parity)),
     u === nothing)

nranks(::Alone) = 1
nranks(a::Ranked) = a.size

function write_checkpoint(f, access, path, forest, appname, appversion, sets, filters, data,
                          sync)
    provenance = agree_values(access, provenance_values(nranks(access)))
    # Written beside `path` and renamed over it once complete. `rename`,
    # not `mv(…; force = true)`, which on Julia 1.11 removes `path` first
    # and leaves a moment with no checkpoint there at all; a rename on one
    # file system replaces the old file atomically.
    partial = path * ".partial"
    try
        open_file(access, partial, "w") do file
            root = create_group(file, GROUP)
            write_attribute(root, "format", FORMAT)
            write_attribute(root, "format_version", Int64(FORMAT_VERSION))
            write_attribute(root, "features", collect(String, FEATURES))
            write_attribute(root, "application", appname)
            write_provenance(root, provenance)
            write_forest(root, forest)
            group = create_group(root, "fieldsets"; track_order=true)
            for (name, fs, u) in sets
                write_fieldset(group, name, fs, u, filters)
            end
            app = create_group(file, appname; track_order=true)
            write_attribute(app, "format_version", appversion)
            put_plain(app, "data", data)            # checked and agreed above
            f === nothing || f(app)
            flush_ranks(access, file, sync)
            return nothing
        end
        publish(access, partial, path, sync)
    catch
        rm(partial; force=true)
        rethrow()
    end
    return nothing
end

# Over several ranks each rank's writes have to reach stable storage, not
# only rank 0's: on a parallel file system every client caches its own,
# and an `fsync` on one flushes none of the others'. A collective
# `H5Fflush` does it, since the MPI-IO driver's flush is `MPI_File_sync`
# on every rank (`H5FD__mpio_flush` in libhdf5 2.2.0's `H5FDmpio.c`,
# reached through `H5F__flush` with `closing = false`; a flush as the
# file closes skips the sync, which is why it is asked for here, before
# the close). ROMIO's `MPI_File_sync` is an `fsync` on each rank that
# wrote (`ADIOI_GEN_Flush`). Serially the file is flushed by
# `flush_to_storage` alone, after it is closed, as in M9a.
flush_ranks(::Alone, file, sync) = nothing
flush_ranks(::Ranked, file, sync) = (sync && API.h5f_flush(file, API.H5F_SCOPE_LOCAL);
                                     nothing)

# Put the closed partial file in place: the data first, then the rename,
# then the directory entry the rename wrote. A rename that reaches the
# disk before the data it points to would replace the previous checkpoint
# with a truncated file.
function publish(::Alone, partial, path, sync)
    sync && flush_to_storage(partial)
    Base.Filesystem.rename(partial, path)
    sync && flush_to_storage(dirname(abspath(path)); directory=true)
    return nothing
end

# Over several ranks rank 0 does it — its `flush_to_storage` is the full
# flush on macOS, where the ranks' `fsync` does not reach the drive — and
# the others wait for it, so that no rank returns before the checkpoint
# is in place, and every rank fails if rank 0 did.
function publish(a::Ranked, partial, path, sync)
    failure = nothing
    if a.rank == 0
        try
            publish(Alone(), partial, path, sync)
        catch err
            failure = err
        end
    end
    done = first(allgather(a.comm, failure === nothing))
    failure === nothing || throw(failure)
    done || error("the checkpoint $(repr(path)) was written by every rank but not put in " *
                  "place: rank 0, which flushes it and renames it over the old one, " *
                  "failed, and its error says why. The previous checkpoint is untouched.")
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

function check_file(path::AbstractString)
    isfile(path) || throw(ArgumentError(
        "there is no checkpoint at $(repr(path)): no such file"))
    HDF5.ishdf5(path) || throw(ArgumentError(
        "$(repr(path)) is not a TreeAMR checkpoint: it is not an HDF5 file at all"))
    return nothing
end

# Open `path` as a TreeAMR checkpoint and call `f(file, root, context)`,
# refusing a file that is not one. Nothing about the format version is
# checked here, so that `checkpoint_environment` can read a file whose
# version `load_checkpoint` refuses. A load over several ranks has checked
# the file already, on rank 0, before agreeing to open it.
function open_checkpoint(f, path::AbstractString, access::Access=Alone();
                         checked::Bool=false)
    checked || check_file(path)
    return open_file(access, path, "r") do file
        haskey(file, GROUP) || throw(ArgumentError(
            "$(repr(path)) is not a TreeAMR checkpoint: it is an HDF5 file with no " *
            "/$GROUP group, which is where TreeAMR keeps everything it writes"))
        root = file[GROUP]
        format = hasattr(root, "format") ? read_attribute(root, "format") : nothing
        format == FORMAT || throw(ArgumentError(
            "$(repr(path)) is not a TreeAMR checkpoint: its /$GROUP group has format = " *
            "$(repr(format)), not $(repr(FORMAT))"))
        context = (; path=String(path), provenance=read_provenance(root))
        return f(file, root, context)
    end
end

function check_format(root, context)
    note = writer_note(context)
    version = read_attribute(root, "format_version")
    version == FORMAT_VERSION || throw(ArgumentError(
        "$(repr(context.path)) is in checkpoint format version $version, and this " *
        "version of TreeAMR reads format version $FORMAT_VERSION" *
        (version isa Integer && version > FORMAT_VERSION ?
         ", so it was written by a newer TreeAMR than this one. " :
         ", which is the only one there has been. ") *
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
    return nothing
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
    access = access_for(c)
    try
        # The arguments are agreed before the file is opened, and whether
        # it is a file at all is rank 0's to say. What the file holds is
        # the same for every rank, so a refusal of its contents is raised
        # on every rank at the same point without a message.
        typelist = agreed(access, "load_checkpoint") do
            t = check_types(types)
            access isa Ranked && access.rank != 0 || check_file(path)
            layout = layouthash(String(path), map(typename, t), repr(fieldsets),
                                string(nameof(typeof(backend))), f === nothing)
            return (t, layout, UInt(0))
        end
        return open_checkpoint(path, access; checked=true) do file, root, context
            check_format(root, context)
            forest = read_forest(root["forest"], typelist, context, c)
            group = root["fieldsets"]
            sets = Dict{String,Any}()
            for name in select_fieldsets(group, fieldsets, context)
                sets[name] = read_fieldset(group[name], name, forest, typelist, backend,
                                           context)
            end
            appname = read_attribute(root, "application")
            haskey(file, appname) || throw(ArgumentError(
                "$(repr(context.path)) names the application $(repr(appname)), but has no " *
                "/$appname group: the file is damaged. " * writer_note(context)))
            app = file[appname]
            version = read_attribute(app, "format_version")
            data = read_plain(app, "data")
            result = f === nothing ? nothing : f(app)
            return (; forest=forest, fieldsets=sets, application=appname => version,
                    data=data, provenance=context.provenance, result=result)
        end
    finally
        release(access)
    end
end

function checkpoint_environment(path::AbstractString, dir::AbstractString;
                                force::Bool=false)
    p = open_checkpoint((file, root, context) -> context.provenance, path)
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
    access = access_of(parent)
    check() = begin
        check_name(name, "a plain-data item's name")
        haskey(parent, name) && throw(ArgumentError(
            "$(HDF5.name(parent)) already holds an item named $(repr(name)); each item " *
            "is written once"))
        datahash = access isa Ranked ?
                   plain_hash(itempath(parent, String(name)), value, UInt(0), true) :
                   UInt(0)
        (nothing, layouthash(HDF5.name(parent), String(name)), datahash)
    end
    # In a parallel checkpoint the write is collective, and the value has
    # to be the same on every rank, which is agreed before anything is
    # written; serially the value is checked as it is written, as in M9a.
    access isa Ranked ?
    agreed(check, access, "write_plain of $(itempath(parent, String(name)))") : check()
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
            write_whole(access_of(parent)) do
                API.h5d_write(dset, dt, API.H5S_ALL, API.H5S_ALL, API.H5P_DEFAULT, Ref(x))
            end
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
    access = access_of(parent)
    value isa AbstractArray && access isa Ranked &&
        return put_fixed_strings(parent, name, value)
    dset, dt = create_dataset(parent, name, value)
    try
        write_whole(() -> HDF5.write_dataset(dset, dt, value), access)
        tag!(dset, tag; (tag == "array" ? (; eltype="String") : (;))...)
    finally
        close(dset)
        close(dt)
    end
    return nothing
end

# Parallel HDF5 writes no variable-length data ("Parallel IO does not
# support writing VL or region reference datatypes yet", `H5D__write` in
# libhdf5 2.2.0), which is how HDF5.jl stores an array of strings. In a
# parallel file the array is fixed-length UTF-8 instead, every string
# NUL-padded to the longest, which `read` gives back as the same
# `Array{String}`: the same item to a reader, in another HDF5 string
# type. A string holding a NUL would lose it, so the plain-data walk
# refuses one beforehand.
function put_fixed_strings(parent, name, value)
    width = max(1, maximum(sizeof, value; init=0))
    buf = zeros(UInt8, width, length(value))
    for (j, x) in enumerate(value)
        copyto!(view(buf, 1:sizeof(x), j), codeunits(x))
    end
    id = API.h5t_copy(API.H5T_C_S1)
    API.h5t_set_size(id, width)
    API.h5t_set_strpad(id, API.H5T_STR_NULLPAD)
    API.h5t_set_cset(id, API.H5T_CSET_UTF8)
    dt = HDF5.Datatype(id)
    try
        space = dataspace(size(value))
        dset = try
            create_dataset(parent, name, dt, space)
        finally
            close(space)
        end
        try
            write_whole(() -> HDF5.write_dataset(dset, dt, buf), access_of(parent))
            tag!(dset, "array"; eltype="String")
        finally
            close(dset)
        end
    finally
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
# and over several ranks the hash is what the ranks agree on, since every
# item is created collectively, with one value. The hash follows the
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

plain_hash(path, x::NativeNumber, h, parallel) =
    hash(("number", typename(typeof(x)), bitsof(x)), h)
plain_hash(path, x::Rational{I}, h, parallel) where {I<:NativeInteger} =
    hash(("rational", typename(I), bitsof(numerator(x)), bitsof(denominator(x))), h)
plain_hash(path, x::AbstractString, h, parallel) = hash(("string", String(x)), h)
plain_hash(path, x::Symbol, h, parallel) = hash(("symbol", String(x)), h)
plain_hash(path, x::VersionNumber, h, parallel) = hash(("version", string(x)), h)
plain_hash(path, ::Nothing, h, parallel) = hash("nothing", h)

function plain_hash(path, x::AbstractArray{T}, h, parallel) where {T<:NativeNumber}
    isconcretetype(T) || throw(not_plain_at(path, x))
    h = hash(("array", typename(T), size(x)), h)
    for y in x
        h = hash(bitsof(y), h)
    end
    return h
end

function plain_hash(path, x::AbstractArray{<:AbstractString}, h, parallel)
    h = hash(("array", "String", size(x)), h)
    for y in x
        parallel && occursin('\0', y) && throw(ArgumentError(
            "$path holds a string with a NUL character, which a checkpoint written over " *
            "several ranks cannot store in an array: parallel HDF5 writes no " *
            "variable-length data, so there an array of strings is stored NUL-padded to " *
            "a fixed length, and the NUL would be lost. Store such a string as an array " *
            "of UInt8, or as a single string."))
        h = hash(String(y), h)
    end
    return h
end

function plain_hash(path, x::Tuple, h, parallel)
    T = isempty(x) ? Nothing : typeof(first(x))
    if T <: NativeNumber && all(y -> typeof(y) === T, x)
        h = hash(("tuple", typename(T), length(x)), h)
        for y in x
            h = hash(bitsof(y), h)
        end
        return h
    end
    return group_hash(path, "tuple", (string(i) => y for (i, y) in enumerate(x)), h,
                      parallel)
end

plain_hash(path, x::NamedTuple, h, parallel) =
    group_hash(path, "namedtuple", pairs(x), h, parallel)

function plain_hash(path, x::AbstractDict{K}, h, parallel) where {K}
    keytype = K <: AbstractString ? "String" : K === Symbol ? "Symbol" :
              throw(not_plain_at(path, x))
    return group_hash(path, "dict", pairs(x), hash(keytype, h), parallel)
end

plain_hash(path, x, h, parallel) = throw(not_plain_at(path, x))

function group_hash(path, tag, items, h, parallel)
    h = hash(tag, h)
    for (key, value) in items
        key = string(key)
        check_name(key, "the plain-data item name in $path")
        h = plain_hash(path * "/" * key, value, hash(key, h), parallel)
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
