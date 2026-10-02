# Checkpoint and restart (M9a).
#
# The functions are declared here, with their docstrings, and implemented
# in the package extension `TreeAMRHDF5Ext` (`ext/TreeAMRHDF5Ext.jl`),
# which Julia loads when HDF5 is loaded beside TreeAMR. HDF5 is a weak
# dependency rather than a hard one because TreeAMR's only hard
# dependency is KernelAbstractions, and an application that never
# checkpoints should not have to load HDF5 and its binary libraries.
# Without the extension these functions have no methods at all, and the
# error hint below turns the resulting `MethodError` into an instruction.
#
# The file format, and every decision behind it, is `CODE.md`,
# "Checkpoint and restart". This file holds no HDF5 code.

"""
    save_checkpoint(path, forest; fieldsets, application, data = (;), filters = (),
                    sync = true)
    save_checkpoint(f, path, forest; ...)        # do-block: f(app::HDF5.Group)

Write a checkpoint of `forest`, of the field sets over it that the
application evolves, and of the application's own plain data, to the
HDF5 file `path`, so that [`load_checkpoint`](@ref) restores them
exactly in a fresh process, on any thread count and any backend (M9a).
Implemented by the package extension `TreeAMRHDF5Ext`: load it with
`using HDF5`.

A checkpoint belongs at a chunk boundary, *after* the regrid: there a
fixed-step integrator holds nothing but `(t, u)`, so a restart that
restores both begins the next chunk with exactly what an uninterrupted
run began it with. Put `t`, the chunk index and any other run state in
`data`.

# Keywords

- `fieldsets` — the field sets to save, as `name => (fs, u)` or
  `name => fs` pairs, in any iterable: a tuple, a vector, a `Dict`.
  `name => (fs, u)` saves the state vector `u` and is the recommended
  form: after `solve` the working array holds whatever the last
  right-hand side scattered, which is a stage, not the solution.
  `name => fs` gathers the owned points from `fs.work` into a fresh
  state vector first, which is right for a set whose working array is
  current by construction — one filled by [`fill_by_coordinates!`](@ref),
  or one just moved by [`regrid!`](@ref). Every set must be over
  `forest` itself (`===`), as for `regrid!`, and `u` must be its state
  vector: of its element type, of [`statelength`](@ref)`(fs)` entries,
  on its backend. Scratch sets — fluxes, primitives — are left out;
  they are the application's to rebuild. Pass `()` to save none.
- `application` — `name => version`, **required**: the name of the
  application's own top-level group in the file, and the format version
  of what the application stores there. TreeAMR stores the version and
  returns it from `load_checkpoint`, and never interprets it; checking
  it is the application's business. The name cannot be `"TreeAMR.jl"`,
  which is TreeAMR's own group, and cannot contain `/`.
- `data` — the application's run state as plain data (see
  [`write_plain`](@ref)), written as the item `data` of its group. A
  NamedTuple is the natural form: it comes back as a NamedTuple, so
  keyword defaults on the application's side are how a field added in a
  later version gets its default.
- `filters` — HDF5.jl filter objects, such as `(HDF5.Filters.Shuffle(),
  HDF5.Filters.Deflate(1))`. With filters each field set is stored in
  one chunk per block and variable, so that reading a block decompresses
  that block and no other; without, contiguously. TreeAMR depends on no
  filter package.
- `sync` — whether to flush the file to stable storage before it
  replaces `path` (see "Atomicity"). On by default, since a checkpoint
  that a power loss can take with it is not one; `sync = false` is for
  files that need not survive the machine, such as a test's or a
  scratch file system's.

The do-block form calls `f` with the application's group open for
writing, after everything else is written, for datasets of the
application's own that are not plain data.

# What is saved

What cannot be recomputed, and nothing else: the forest's parameters
(`D`, `N`, the roots, `periodic`, `reflecting`, the extents bit for bit
in the geometry type) and its leaf list, in curve order; and for each
field set its element type, `nvars`, `G`, centering and parity, and its
**owned** points, in state-vector layout. Not saved: the ghosts, the
shared boundary planes of a vertex-like dimension and the derived wall
planes, all of which [`fill_ghosts!`](@ref) rebuilds from the owned
points with the application's own operators and hook; the forest's
[`generation`](@ref); schedules, operators and hooks. Beside the leaf
list and each field set go CRC-32C checksums, one over the leaf list and
one per block, which `load_checkpoint` verifies (added in M7, after a
parallel file came back damaged; a file without them loads unchecked).

An element type is stored as itself when it is an HDF5 native — `Bool`,
the signed and unsigned integers, `Float16`, `Float32`, `Float64`, and
`Complex` of those — and otherwise as *limbs* when it is an `isbits`
type made of one native type throughout with no padding: MultiFloats'
`Float32x2` is two `Float32`. The geometry type is treated the same way.
Anything else is refused, before the file is created.

# Atomicity

The file is written to `path * ".partial"` and renamed over `path` only
once it is complete. On any error, one thrown by the do-block included,
the partial file is removed and the error rethrown, so a failed or
interrupted write never destroys the previous checkpoint at `path`.

Closing a file only hands its data to the operating system, which
survives the process but not a power loss or a kernel crash — and after
one of those the rename, a separate update of the directory, can have
reached the disk before the data it points to, leaving `path` naming a
truncated file with the previous checkpoint already gone. So with
`sync = true` the partial file is flushed to stable storage before the
rename (`fsync`; on macOS `fcntl(F_FULLFSYNC)`, since `fsync` there does
not wait for the drive's own cache), and the directory after it, so
that the rename is durable too. On Windows `sync` does nothing.

# Over a distributed forest (M7)

Over a forest distributed over MPI ranks, `save_checkpoint` is
collective: every rank calls it with the same `path`, the same field
sets — each rank's own blocks of them — and the same keywords, `data`
included, and every rank writes its own blocks into the one shared file
through parallel HDF5 (the extension `TreeAMRHDF5MPIExt`, which loads
with HDF5 and MPI together). The file is the serial one: the same
layout and format version, the blocks in curve order, so it loads on
any rank count, and serially. Everything that can be refused is checked
on every rank before the file is created, and the verdict is agreed, so
a refusal — an argument or a value of `data` that differs between ranks
included — is raised on every rank together. The do-block runs on every
rank, and what it writes is collective: [`write_plain`](@ref) agrees on
its value across the ranks the same way, and a dataset the block writes
through HDF5 itself is created by every rank and must hold the same
values on each. An error inside a collective HDF5 call on some ranks
only cannot be recovered from, since the others wait in it; it is fatal
to the job, as in any MPI program. With `sync = true` every rank flushes
its writes to stable storage (a collective `H5Fflush`, which is an
`MPI_File_sync`) before the file is closed, and rank 0 then flushes the
file as in a serial save, renames it and flushes the directory, while
the others wait, so no rank returns before the checkpoint is in place.
The file is opened with MPI-IO hints that turn off ROMIO's two
read-modify-write paths for writes, data sieving and collective
buffering: on a parallel file system whose clients buffer writes, such
as BeeGFS, either can write back another rank's bytes as they were
before that rank's write arrived, destroying it.

Returns `path`. See `CODE.md`, "Checkpoint and restart" and "Parallel
checkpoints", for the file layout and the reasons behind it.

```julia
using HDF5
save_checkpoint("run.h5", forest; fieldsets = ("U" => (U, u),),
                application = "MyApp" => 1, data = (; t, chunk))
```
"""
function save_checkpoint end

"""
    load_checkpoint(path; backend = CPU(), types = (), fieldsets = nothing,
                    comm = nothing)
    load_checkpoint(f, path; ...)                # do-block: f(app::HDF5.Group)

Read a checkpoint written by [`save_checkpoint`](@ref) into fresh
objects. Implemented by the package extension `TreeAMRHDF5Ext`: load it
with `using HDF5`. Returns a NamedTuple with

- `forest` — a new [`Forest`](@ref), at [`generation`](@ref) 0, built
  through the validated `leaves` path, so a damaged or hand-edited leaf
  list is refused rather than turned into a mesh the schedule would get
  wrong;
- `fieldsets` — a `Dict{String,Any}` from each loaded name to
  `(; fieldset, state)`: a new [`FieldSet`](@ref) on `backend` and its
  state vector, allocated by [`statevector`](@ref), so first touched by
  the block owners, and read into. The owned points are scattered into
  the working array; its ghosts and shared planes are **not** filled,
  and are zero until the caller's `fill_ghosts!(fs, GhostSchedule(fs,
  ops); boundary)`, since the operators and the hook are the
  application's own;
- `application` — `name => version`, as the application saved it;
- `data` — the application's plain data, as [`read_plain`](@ref)
  returns it;
- `provenance` — `(; treeamr_version, julia_version, created, hostname,
  nthreads, nranks, project, manifest)`: who wrote the file, when (UTC),
  on how many threads (of rank 0) and ranks, and the texts of the
  writer's `Project.toml` and `Manifest.toml` (see
  [`checkpoint_environment`](@ref)). A file from before M7 has no
  `nranks`, which reads as 1;
- `result` — what the do-block returned, or `nothing`. The do-block is
  called with the application's group open, for whatever it wrote there
  beside `data`.

# Keywords

- `backend` — where the field sets are allocated. A checkpoint holds
  nothing that depends on the thread count or the backend, so it loads
  on any of them, exactly.
- `types` — the element types from packages that the reader cannot name
  on its own, such as `types = (Float32x2,)`. A type stored as limbs is
  recorded by name, and matched by name against this list; its size and
  its limbs are then checked against the file's. Native types need no
  entry.
- `fieldsets` — `nothing` for every field set in the file, or a
  collection of names for a subset.
- `comm` — the communicator to distribute the loaded forest over, as for
  [`Forest`](@ref): `nothing` for a serial forest, or an `MPI.Comm` such
  as `MPI.COMM_WORLD`. The call is then collective, with the same
  arguments on every rank: each rank reads the whole leaf list, builds
  the replicated forest over `comm` and reads only its own blocks of
  each field set. A file written on any number of ranks loads on any
  other, or serially, since it stores no partition.

# Refusals

A file this version cannot interpret is refused with an `ArgumentError`
that says why: one that is not a TreeAMR checkpoint at all, a
`format_version` other than the one this version reads (1), a feature
this version does not know (every listed feature must be understood), an
element type missing from `types` or one that does not match the file's
layout, and a field set name the file does not hold. A leaf list or a
block whose bytes do not match the checksum stored with them is refused
too: the file was damaged while or after it was written. Over several
ranks each rank checks its own blocks, and the verdict is agreed, so
the refusal comes on every rank. Compatibility is
decided by the format version and the features, never by package
versions. Each refusal of what a checkpoint holds names the TreeAMR
version that wrote it and points to [`checkpoint_environment`](@ref),
which recreates the environment that can read it.

No values are converted: a field set comes back in the type it was
saved in, since exactness is the point.

```julia
using HDF5
ck = load_checkpoint("run.h5")
forest = ck.forest
U, u = ck.fieldsets["U"].fieldset, ck.fieldsets["U"].state
schedule = GhostSchedule(U, ops)       # then fill_ghosts!, and continue from ck.data.t
```
"""
function load_checkpoint end

"""
    write_plain(parent, name, value)

Write `value` as the item `name` of the HDF5 group or file `parent`,
in the closed, documented set of plain data that a checkpoint stores
for an application; [`read_plain`](@ref) reads it back exactly.
Implemented by the package extension `TreeAMRHDF5Ext`: load it with
`using HDF5`. [`save_checkpoint`](@ref)'s `data` is written with it.

Every item is an HDF5 dataset or group carrying a `type` attribute from
this vocabulary:

| `type` | value | stored as |
|---|---|---|
| `"number"` | `Bool`, `Int8`–`Int64`, `UInt8`–`UInt64`, `Float16`, `Float32`, `Float64`, `Complex` of those | a scalar dataset of that type, named in `eltype`; a `Bool` as a `UInt8` 0 or 1, a `Complex` as the compound `(r, i)` |
| `"rational"` | `Rational` of a native integer type | a dataset `[numerator, denominator]` of that type, named in `eltype`, so the value is exact |
| `"string"`, `"symbol"`, `"version"` | `AbstractString`, `Symbol`, `VersionNumber` | a string dataset |
| `"nothing"` | `nothing` | an empty group |
| `"array"` | an array of native numbers, or of strings | a dataset of its shape, with `eltype` (`"String"` for strings) |
| `"tuple"` | a tuple | a one-dimensional dataset when it is nonempty and all its entries have one native number type; otherwise a group of the items `"1"`, `"2"`, … |
| `"namedtuple"` | a NamedTuple | a group of the items by field name, in field order |
| `"dict"` | an `AbstractDict` with `String` or `Symbol` keys | a group of the items by key, with `keytype` `"String"` or `"Symbol"` |

Groups track the creation order of their items, which is how a
NamedTuple keeps its field order. What comes back: a NamedTuple as a
NamedTuple, a Dict as a `Dict{String,Any}` or `Dict{Symbol,Any}`, an
array as an `Array` of its element type and shape (a range becomes a
`Vector`), a string as a `String`, everything else as itself.

Anything else — a struct, a closure, a `Char`, a `BigFloat`, a
MultiFloats scalar — is an `ArgumentError` naming its type. A struct is
the application's to convert, to a NamedTuple of plain values, because a
type name in the file would tie the file to that struct's definition,
and a file has to outlive the code that wrote it: two versions of a
package cannot be loaded into one process, so a converter from an old
layout can only read the old *file*. The same holds for JLD2 and
`Serialization`, which is why neither is used. An item name must be
nonempty, cannot be `"."`, and cannot contain `/`.

In the application's group of a checkpoint being saved over a
distributed forest (the do-block of [`save_checkpoint`](@ref)),
`write_plain` is collective: every rank passes the same `name` and the
same `value`, which is checked by a digest of what would be written,
gathered from every rank before anything is, and refused on every rank
if it differs. Parallel HDF5 cannot write variable-length data, so an
array of strings is stored there as fixed-length, NUL-padded UTF-8 —
read back as the same `Array{String}` — and a string in it that holds a
NUL character is refused.

See `CODE.md`, "Checkpoint and restart".
"""
function write_plain end

"""
    read_plain(parent, name)

Read the plain-data item `name` of the HDF5 group or file `parent`, as
written by [`write_plain`](@ref), which lists the types and what each
comes back as. Implemented by the package extension `TreeAMRHDF5Ext`:
load it with `using HDF5`. An item without a `type` attribute, or with
one outside the vocabulary, is refused.
"""
function read_plain end

"""
    checkpoint_environment(path, dir; force = false)

Write the `Project.toml` and `Manifest.toml` stored in the checkpoint
`path` into the directory `dir`, creating it, so that `julia
--project=dir` recreates the environment that wrote the file. Returns
`dir`. Implemented by the package extension `TreeAMRHDF5Ext`: load it
with `using HDF5`.

This is the answer to a refusal from [`load_checkpoint`](@ref): two
versions of TreeAMR cannot be loaded into one process, so a file in a
format this version does not read is read with the version that wrote
it. That is why this function reads nothing but the provenance, and
works on a file whose `format_version` this version would refuse.

The texts are those of the writer's active project and of the manifest
beside it (`Manifest-v1.x.toml` or `Manifest.toml`), as they were when
the file was written. An environment that had no manifest, or no
project file at all, stored an empty text; the file that is missing is
then not written, and a warning says so. A path in them — a `[sources]`
entry, a developed package — still names the writer's disk.

Existing files in `dir` are not overwritten unless `force = true`. It
is not collective: it reads the file serially, on whichever process
calls it.
"""
function checkpoint_environment end

# The parallel file of a checkpoint over a distributed forest (step 6 of
# M7): `path` opened in `mode` through HDF5's MPI-IO driver over the
# library communicator `comm` (`librarycomm`). Its method for an
# `MPI.Comm` is the extension `TreeAMRHDF5MPIExt`, which loads with HDF5
# and MPI together, since the MPI-IO driver is HDF5.jl's own MPI
# extension; everything else about a parallel checkpoint is in
# `TreeAMRHDF5Ext`. Collective.
function open_parallel_file end

open_parallel_file(comm, path, mode) = throw(ArgumentError(
    "a checkpoint of a distributed forest is written and read through parallel HDF5, " *
    "which needs HDF5.jl and MPI.jl both loaded (that loads TreeAMR's extension " *
    "TreeAMRHDF5MPIExt) and a communicator that is an MPI.Comm; got a " *
    "$(typeof(comm))"))

# The error hint. Without the extension the functions above have no
# methods, and the bare `MethodError` would say nothing about why; the
# hint says to load HDF5, and why it is optional. It stays silent once
# the extension is loaded, when a `MethodError` is a genuine one — wrong
# arguments — and loading HDF5 again would not help.
const CHECKPOINT_FUNCTIONS = (save_checkpoint, load_checkpoint, write_plain, read_plain,
                              checkpoint_environment)

function checkpoint_hint(io::IO, exc::MethodError, argtypes, kwargs)
    f = exc.f
    # A call with keywords fails in `Core.kwcall`, with the function second.
    f === Core.kwcall && length(exc.args) >= 2 && (f = exc.args[2])
    any(g -> f === g, CHECKPOINT_FUNCTIONS) || return nothing
    Base.get_extension(@__MODULE__, :TreeAMRHDF5Ext) === nothing || return nothing
    print(io, "\n`", nameof(f), "` is implemented by TreeAMR's HDF5 extension, which ",
          "loads with HDF5: add HDF5 to the environment and run `using HDF5`. HDF5 is ",
          "a weak dependency so that an application that never checkpoints does not ",
          "load HDF5 and its binary libraries.")
    return nothing
end

function __init__()
    Base.Experimental.register_error_hint(checkpoint_hint, MethodError)
    return nothing
end
