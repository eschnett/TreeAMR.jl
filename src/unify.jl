# Unifying a set without its ghost fill: stage 0 alone (M22).
#
# A set with a vertex-like dimension owns both face planes of its blocks
# ("Closed blocks" in CODE.md), so a point on a block face has a copy on
# every block that holds it, and stage 0 of the fill gives every secondary
# copy its primary's value: a copy from a block of its own level, the
# set's restriction from a finer one (M22 rule 1). A flux computed on
# every block's own range is such a set without ghosts, and stage 0 is
# all it needs: at a coarse-fine face the coarse copy of a face value
# takes the restriction of the fine ones — injection along the stagger,
# the exact two-cell average across it — which is what conservation
# needs ("Conservation at coarse-fine faces" in CODE.md), and at a
# same-level face the two copies become one number. `UnifySchedule`
# holds that stage alone, with the restriction the geometry fixes.

"""
    UnifySchedule{T,D,R}

Stage 0 of a fill alone, for a field set with a vertex-like dimension,
replayed by [`unify!`](@ref): every secondary copy of a shared point
takes its primary's value (M22, "Closed blocks" in `CODE.md`).

    s = UnifySchedule(flux)                # rebuilt when the tree changes
    unify!(flux, s)

What a conservative right-hand side needs between computing its fluxes
and applying their divergence: at a coarse-fine face the coarse copy of
a face value becomes the restriction of the fine ones — injection along
the face's normal, the exact average over the `2^(D-1)` fine faces it
covers across it — so the flux a coarse cell sees there is the
area-weighted sum of the fine fluxes, and the scheme conserves to
roundoff under a global time step. At a same-level face the two blocks'
copies, which computed the same flux, become the same number, so
conservation there is exact; and on an edge-centered set every block
around an edge holds one value, a diagonal finer block's included, which
constrained transport needs for a face shared by two coarse blocks to
evolve as one.

Unlike a [`GhostSchedule`](@ref) it takes no [`Operators`](@ref): the
restriction is not interpolation, and there is no order to choose. A
`GhostSchedule` holds the same stage built with its operators'
restriction, and [`unify!`](@ref) takes either. The set needs no ghosts,
and nothing outside a block's own range is read or written.

A cell-centered set has no shared point and is refused, and so is an
element set (P > 0), whose conservative scheme needs mortar fluxes; a
forest with what a later step of M22 rebuilds is refused as
`GhostSchedule` refuses it. Building it is collective over a distributed
forest, and the schedule is tied to the forest's leaves and to the set's
layout as they were: `unify!` refuses a stale or mismatched one.
"""
struct UnifySchedule{T,D,R,BK<:Backend,GRP<:TransferGroup{T,D},ST<:ExchangeStage{GRP}}
    forest::Forest{D,R}
    generation::Int                              # forest generation it was built for
    G::NTuple{D,Int}                             # ghost width it was built for
    centering::NTuple{D,Symbol}                  # centering it was built for
    backend::BK
    unify::Vector{GRP}                           # stage 0's local groups
    stages::Vector{ST}                           # stage 0, one stage
    stateunify::Vector{ST}                       # stage 0 on a state, for `unify!`
    selfimage::Vector{GRP}                       # stage P (M22), local
    stateselfimage::Vector{GRP}                  # stage P on a state
end

isstale(s::UnifySchedule) = generation(s.forest) != s.generation

function UnifySchedule(fs::FieldSet{T,D}) where {T,D}
    forest = fs.forest
    N = forest.N
    ghosts = fs.G
    stags = staggers(fs)
    backend = get_backend(fs.work)
    # Collective over a distributed forest (M7), as `GhostSchedule` is.
    collective_checks(forest, "UnifySchedule") do
        # Before the stagger test, which an element set, cell-like in
        # every dimension, would fail with the wrong reason.
        fs.P == 0 || throw(ArgumentError(
            "an element-centered field set (P = $(fs.P)) has no unify schedule: across " *
            "a coarse-fine face a conservative element scheme needs mortar fluxes, the " *
            "coarse face's flux the integral of the fine faces' polynomials, not a " *
            "restriction of stored values onto one plane, and this schedule builds " *
            "none. The ghost fill's compatible restriction (`Operators(family = " *
            "ElementLocal)`) is what keeps a summation-by-parts scheme's energy " *
            "invariant across the face."))
        any(==(1), stags) || throw(ArgumentError(
            "a cell-centered field set has no shared point to unify: the fixup replaces " *
            "the values lying *on* a block's boundary face, and a cell-centered field " *
            "has none — its values sit half a cell in from every face. Build it over " *
            "the field set that holds the face quantity, whose centering is vertex-like " *
            "in the face dimension (`facecentered($D, d)`), not over the cell-centered " *
            "state."))
        check_closed_blocks(forest, fs.centering, fs.walls, "UnifySchedule")
        return nothing, layouthash(ghosts, fs.centering, T, nameof(typeof(backend)))
    end
    # This rank's blocks, in global leaf indices, as in `GhostSchedule`.
    owned = blockrange(forest)
    offset = first(owned) - 1
    nb = length(owned)

    # Threaded over blocks and merged in chunk order, so the schedule is
    # a function of the tree alone — the same argument as in
    # `GhostSchedule`, and the same bit-identity obligation.
    chunks = threadchunks(nb)
    perpairs = [TransferPairs{D}() for _ in chunks]
    threaded_chunks(nb) do c, range
        for b in range
            unify_sources!(perpairs[c], forest, stags, offset + b)
        end
    end
    pairs = TransferPairs{D}()
    for c in eachindex(chunks)
        merge_pairs!(pairs, perpairs[c])
    end
    # The three classes of a distributed forest (M7), as in `GhostSchedule`.
    received = split_received!(pairs, owned, nleaves(forest))
    sent = sent_transfers(forest, owned) do into, j
        unify_sources!(into, forest, stags, j)
    end

    # The restriction the geometry fixes: injection along a vertex-like
    # dimension, the exact two-cell average, order 2, along a cell-like one.
    build(key) = ntuple(d -> unify_stencil(T, N, ghosts[d], stags[d], key, d, 2, UNBOUNDED),
                        D)
    # Stage P's groups take the parity projection's column; every other is
    # a plain copy or restriction.
    factorcol(key) = any(==(Int8(2)), key.mirror) ? mirrorcolumn(key.mirror) : 0
    axesof(key) = group_axes(forest, key)
    bounded(key) = UNBOUNDED
    stageof(key) = UNIFY_TAG
    GRP = grouptype(backend, T, Val(D))
    unify, _, _ = local_groups(GRP, T, Val(D), backend, pairs, build, factorcol, axesof,
                               bounded, stageof)
    ST = stagetype(backend, T, Val(D))
    stages = build_stages(ST, T, backend, [UNIFY_TAG], tag -> unify, bystage(stageof, sent),
                          bystage(stageof, received), build, factorcol, forest;
                          required=[UNIFY_TAG], axesof=axesof, bounded=bounded)
    stateunify = state_unify_stages(ST, GRP, T, backend, forest, owned, ghosts, pairs, sent,
                                    received, build, factorcol, axesof, bounded, stageof)
    # Stage P (M22): a reflecting wall plane projected onto each variable's
    # parity, at the primaries, before stage 0; a flux normal to the wall
    # is odd there, and becomes zero.
    ppairs = TransferPairs{D}()
    for b in 1:nb
        selfimage_sources!(ppairs, forest, stags, offset + b, b, fs.parity)
    end
    selfimage = selfimage_groups(GRP, T, Val(D), backend, ppairs, build, factorcol, axesof)
    stateselfimage = selfimage_groups(GRP, T, Val(D), backend, ppairs, build, factorcol,
                                      axesof; ghosts=ghosts)
    return UnifySchedule{T,D,floattype(forest),typeof(backend),GRP,ST}(
        forest, generation(forest), ghosts, fs.centering, backend, unify, stages,
        stateunify, selfimage, stateselfimage)
end

function Base.show(io::IO, s::UnifySchedule{T,D}) where {T,D}
    bykind(kind) = sum(g -> g.kind === kind ? ntransfers(g) : 0, s.unify; init=0)
    print(io, "UnifySchedule{", T, ",", D, "}(", bykind(:copy), " copies, ",
          bykind(:restrict), " restrictions", messages_summary(s.stages), ")")
end
