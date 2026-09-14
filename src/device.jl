# Device residency.
#
# M6 puts the leaf data on a GPU. Almost nothing in the per-cell path
# had to change for that — every kernel was written against
# KernelAbstractions from the start, and each one takes its backend from
# `get_backend(fs.work)`. What *did* have to change is everything a
# kernel reads that is not field data: the stencil weights and index
# vectors, the per-block geometry, the boundary region list. Those are
# built on the host (the weights in exact rational arithmetic, which a
# device could not do and should not have to) and must then live where
# the kernel that reads them runs.
#
# The rule this file encodes: **host-built metadata is uploaded once, at
# the point where it is already being rebuilt anyway** — schedule
# construction, regrid transfer setup — never per launch and never per
# RHS evaluation. That is the same reasoning that put the ghost exchange
# in a cached schedule rather than in `fill_ghosts!`.

"""
    todevice(backend, a::AbstractArray) -> AbstractArray

`a`, in memory the kernels of `backend` can read.

On the CPU backend this is `a` itself: there is nothing to move, and
copying would double the footprint of every schedule for nothing. On a
device it is a fresh allocation plus one copy.
"""
todevice(::CPU, a::AbstractArray) = a
function todevice(backend::Backend, a::AbstractArray)
    dev = allocate(backend, eltype(a), size(a))
    copyto!(dev, a)
    return dev
end

"""
    tohost(a::AbstractArray) -> Array

`a` as an ordinary host array, without copying when it already is one.

The mirror of [`todevice`](@ref TreeAMR.todevice), for the small
per-block results a device kernel produces that the host driver logic
then has to read: the flag boxes of [`firing_boxes`](@ref), the partials
of a reduction.
"""
tohost(a::Array) = a
tohost(a::AbstractArray) = Array(a)

# Whether `T` can be stored and computed in on `backend`. Checked where
# a field set is allocated rather than left to a compilation failure
# thousands of lines later: `Float64` on a device without hardware fp64
# is the single most likely way to misconfigure a GPU run, and the
# message should say so.
function check_floattype(::Type{T}, backend::Backend) where {T}
    T === Float64 && !supports_float64(backend) && throw(ArgumentError(
        "$(nameof(typeof(backend))) has no hardware Float64, so a Float64 field " *
        "set cannot run on it. Build the forest and the field set in Float32 — " *
        "the mesh is generic in its floating-point type and computes the geometry " *
        "and the interpolation weights in it from the first operation, so no fp64 " *
        "is needed anywhere on the per-cell path."))
    return nothing
end

# Two backends are "the same" when they are the same kind of backend.
# Comparing the instances would be wrong: `CPU()` and `CPU(static=true)`
# are different values that allocate identical arrays, and a device
# backend may or may not be a singleton.
samebackend(a::Backend, b::Backend) = typeof(a) === typeof(b)
