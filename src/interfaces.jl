# Interface restriction: the flux fixup at coarse-fine faces (M8b).
#
# With one global timestep, conservation needs only that the flux a
# coarse cell sees on a coarse-fine face equals the area-weighted sum of
# the fine-face fluxes there. That is a purely spatial condition, so a
# conservative right-hand side is three steps — compute fluxes, restrict
# them at coarse-fine faces, apply the divergence — and there are no flux
# registers and no time-accumulated corrections. See "Conservation at
# coarse-fine faces" in CODE.md for why that is enough.
#
# This is the one operation in the package that overwrites a block's own
# *closed-range* values rather than its ghosts, and doing so is its
# purpose: both sides of the face computed a value one step earlier, and
# the coarse side's is replaced. It reads nothing but closed-range values
# either, which is what lets a computed flux field have `G = 0`.
#
# Everything here is the ghost schedule's machinery over different target
# ranges: the transfers are the `:restrict` cases of the same neighbor
# walk, the stencils come from the same builder, and they are batched,
# sliced and replayed by `TransferGroup` / `run_phase!` on any backend.

# The plane the restriction writes in the face dimension: the block's
# owned low boundary plane on the low side, its shared high plane on the
# high side. Both are the first point of a range `target_range` already
# names — the block's own range and its high exchange region — so the
# single source of truth stays single.
function interface_plane(N::Int, G::Int, c::Int, δd::Int)
    f = δd == 1 ? first(target_range(N, G, c, 1, 0, false)) :
        first(target_range(N, G, c, 0, 0, false))
    return f:f
end

# One dimension of an interface transfer. In the face dimension it is the
# single plane above; tangentially it is the owned range (cell-like) or
# the closed range (vertex-like), split half-open between the finer
# neighbors with the top point going to the top one — which is
# `target_range`'s halved case with `closed`.
#
# The weights follow from the same builder the ghost restriction uses:
# injection in a vertex-like dimension (the coincident fine point), the
# exact two-cell average in a cell-like one, which is what the
# point-value restriction at order 2 already is. Their product over the
# `D - 1` tangential dimensions is the `1/2^(D-1)` of a face average when
# every tangential dimension is cell-like, as a flux's are.
interface_stencil(::Type{T}, N::Int, G::Int, c::Int, δd::Int, od::Int) where {T} =
    restrict_stencil_over(T, δd == 0 ? target_range(N, G, c, 0, od, true, true) :
                             interface_plane(N, G, c, δd),
                          N, G, c, δd, od, 2)

"""
    InterfaceSchedule{T,D,R}

The precomputed coarse-fine interface restriction for one field set,
replayed by [`restrict_interfaces!`](@ref).

    isched = InterfaceSchedule(flux)        # rebuilt when the tree changes
    restrict_interfaces!(flux, isched)

A **face** direction `±e_d` is admissible when dimension `d` is
vertex-like, so that the field has points lying *on* that boundary face;
a cell-centered field set has no admissible direction at all and is
refused here rather than silently given an empty schedule. Edge
directions are deliberately absent: the constrained-transport
consistency condition lives on the edges of coarse-fine **faces**,
including their boundary lines, and a finer neighbor across an edge alone
imposes nothing (see `CODE.md`).

For every block, every admissible direction, and every *finer* neighbor
across it, the block's boundary plane in `d` — its owned plane `G+1` on
the low side, its shared plane `G+N+1` on the high side — is overwritten
with the restriction of that neighbor's opposite boundary plane:
injection along vertex-like dimensions, the exact two-cell average along
cell-like ones. Tangentially the target is the block's owned range in a
cell-like dimension and its closed range in a vertex-like one, split
half-open among the finer neighbors.

There is **one phase per face dimension**, run in ascending order. Within
a phase every target point is written exactly once; a point on the line
where two coarse-fine faces of the same block meet is written in two
phases, by two same-level fine blocks that computed the same value there.

The schedule is tied to the forest's leaf array and to the field set's
layout as they were when it was built, and
[`restrict_interfaces!`](@ref) refuses a stale or mismatched one — as
[`GhostSchedule`](@ref) does, and for the same reason: every target range
in it is wrong for another mesh or another `G`.

Unlike a [`GhostSchedule`](@ref) this takes no [`Operators`](@ref). The
transfer is not interpolation: it is injection and the exact two-cell
average, fixed by the geometry, so there is no order to choose.
"""
struct InterfaceSchedule{T,D,R,BK<:Backend,GRP<:TransferGroup{T,D}}
    forest::Forest{D,R}
    generation::Int                              # forest generation it was built for
    G::NTuple{D,Int}                             # ghost width it was built for
    centering::NTuple{D,Symbol}                  # centering it was built for
    backend::BK
    dimensions::Vector{Int}                      # face dimension of each phase
    phases::Vector{Vector{GRP}}
end

isstale(s::InterfaceSchedule) = generation(s.forest) != s.generation

# The neighbor walk for one block: every coarse-fine face it is the
# *coarse* side of. A block whose neighbor across a face is coarser does
# nothing — that face is the coarser block's target — and a face at the
# domain boundary carries whatever flux the application computed there.
function interface_sources!(pairs::TransferPairs{D}, forest::Forest{D},
                            faces::Vector{Int}, b::Int) where {D}
    k = forest.leaves[b]
    for d in faces, s in (-1, 1)
        δ = ntuple(e -> e == d ? s : 0, D)
        nbrs = neighbor_keys(forest, k, δ)
        (isempty(nbrs) || level(first(nbrs)) <= level(k)) && continue
        for nbr in nbrs
            push!.(get!(pairs, GroupKey{D}(:restrict, δ, childoffset(nbr), 0),
                        (Int32[], Int32[])),
                   (Int32(b), Int32(find_leaf(forest, nbr))))
        end
    end
    return nothing
end

function InterfaceSchedule(fs::FieldSet{T,D}) where {T,D}
    forest = fs.forest
    N = forest.N
    ghosts = fs.G
    stags = staggers(fs)
    faces = filter(d -> stags[d] == 1, collect(1:D))
    isempty(faces) && throw(ArgumentError(
        "a cell-centered field set has no coarse-fine interface to restrict: the " *
        "fixup replaces the values lying *on* a block's boundary face, and a " *
        "cell-centered field has none — its values sit half a cell in from every " *
        "face. Build it over the field set that holds the face quantity, whose " *
        "centering is vertex-like in the face dimension (`facecentered($D, d)`), " *
        "not over the cell-centered state."))
    backend = get_backend(fs.work)
    nb = nleaves(forest)

    # Threaded over blocks and merged in chunk order, so the schedule is
    # a function of the tree alone — the same argument as in
    # `GhostSchedule`, and the same bit-identity obligation.
    chunks = threadchunks(nb)
    perpairs = [TransferPairs{D}() for _ in chunks]
    threaded_chunks(nb) do c, range
        for b in range
            interface_sources!(perpairs[c], forest, faces, b)
        end
    end
    pairs = TransferPairs{D}()
    for c in eachindex(chunks)
        merge_pairs!(pairs, perpairs[c])
    end

    GRP = grouptype(backend, T, Val(D))
    byface = Dict{Int,Vector{GRP}}()
    for (key, (targets, sources)) in pairs
        stencils = ntuple(d -> interface_stencil(T, N, ghosts[d], stags[d],
                                                 key.direction[d], key.offset[d]), D)
        d = findfirst(!=(0), key.direction)::Int
        push!(get!(byface, d, GRP[]),
              todevice(backend, TransferGroup{T,D}(:restrict, stencils, targets,
                                                   sources)))
    end

    dims = sort!(collect(keys(byface)))
    phases = [byface[d] for d in dims]
    return InterfaceSchedule{T,D,floattype(forest),typeof(backend),GRP}(
        forest, generation(forest), ghosts, fs.centering, backend, dims, phases)
end

function Base.show(io::IO, s::InterfaceSchedule{T,D}) where {T,D}
    n = sum(gs -> sum(ntransfers, gs; init=0), s.phases; init=0)
    print(io, "InterfaceSchedule{", T, ",", D, "}(", n, " restrictions over ",
          length(s.phases), " face dimension(s) ", Tuple(s.dimensions), ")")
end

"""
    restrict_interfaces!(fs::FieldSet, isched::InterfaceSchedule)

Overwrite every coarse-fine boundary plane of `fs` with the restriction
of the finer side, by replaying `isched`.

This is step (ii) of a conservative right-hand side — compute fluxes over
the closed range, restrict them here, then apply the divergence — and it
is what makes the two sides of a coarse-fine face agree on the
area-weighted flux, hence what makes the scheme conservative to roundoff
under a global timestep. See [`InterfaceSchedule`](@ref) for what is
transferred and `CODE.md` for why nothing more is needed.

The phases (one per face dimension) run in ascending order with a barrier
between them, exactly as [`fill_ghosts!`](@ref)'s do. It touches neither
ghosts nor interior values away from a coarse-fine face, so the field set
it runs over may have `G = 0`.
"""
function restrict_interfaces!(fs::FieldSet{T,D},
                              isched::InterfaceSchedule{T,D}) where {T,D}
    isched.forest === fs.forest || throw(ArgumentError(
        "interface schedule was built for a different forest than the field set"))
    isstale(isched) && throw(ArgumentError(
        "the forest changed since this interface schedule was built (generation " *
        "$(isched.generation) -> $(generation(isched.forest))); rebuild it"))
    nblocks(fs) == nleaves(isched.forest) || throw(ArgumentError(
        "field set has $(nblocks(fs)) blocks but the schedule's forest has " *
        "$(nleaves(isched.forest)) leaves; rebuild both"))
    fs.G == isched.G || throw(ArgumentError(
        "the field set has ghost width G=$(fs.G) but this interface schedule was " *
        "built for G=$(isched.G); every target plane in it is wrong for this " *
        "layout. Build one with `InterfaceSchedule(fs)`."))
    fs.centering == isched.centering || throw(ArgumentError(
        "the field set has centering $(fs.centering) but this interface schedule " *
        "was built for $(isched.centering); which directions are admissible, the " *
        "target planes and the one-dimensional stencils all follow from the " *
        "centering. Build one with `InterfaceSchedule(fs)`."))

    backend = get_backend(fs.work)
    samebackend(backend, isched.backend) || throw(ArgumentError(
        "the field set lives on $(nameof(typeof(backend))) but this interface " *
        "schedule was built for $(nameof(typeof(isched.backend))); its stencils " *
        "are in the wrong memory. Build it with `InterfaceSchedule(fs)` from a " *
        "field set on the backend you mean to run on."))

    for groups in isched.phases
        run_phase!(fs, groups, backend)
        synchronize(backend)
    end
    return fs
end

# As for `fill_ghosts!`: the transfer accumulates in `eltype(dest)` and
# reads the weights straight out of the stencils, so a mismatch would
# silently promote in the innermost loop.
restrict_interfaces!(::FieldSet{T}, ::InterfaceSchedule{S}) where {T,S} =
    throw(ArgumentError(
        "the field set stores $T but this interface schedule carries $S weights; " *
        "build the schedule from the field set itself, `InterfaceSchedule(fs)`"))
