# Ghost filling: replay a GhostSchedule.
#
# One KernelAbstractions kernel serves all three cases. Because every
# transfer is a tensor product of D one-dimensional stencils, a copy is
# just the width-1 case and needs no separate code path. Written against
# KA from the start so the CPU implementation is already the GPU one
# (M6); the only backend-specific step is `get_backend`.

using KernelAbstractions: @kernel, @index, @Const, get_backend, synchronize

# `dest` and `src` are the same array for ghost filling (targets are
# ghosts, sources interiors, so they never overlap) and different arrays
# when regridding transfers into freshly allocated storage. Neither is
# marked @Const, so the aliasing case stays well defined.
@kernel function transfer_kernel!(dest, src,
                                  @Const(targetblocks), @Const(sourceblocks),
                                  srcstarts, weights,
                                  targetfirst::NTuple{D,Int},
                                  boxlen::NTuple{D,Int},
                                  boxstride::NTuple{D,Int},
                                  ::Val{P}, ::Val{D}) where {P,D}
    cell, v, t = @index(Global, NTuple)

    tblock = targetblocks[t]
    sblock = sourceblocks[t]

    # Position within the target region, as a per-dimension offset.
    r = cell - 1
    off = ntuple(d -> (r ÷ boxstride[d]) % boxlen[d], Val(D))
    tidx = ntuple(d -> targetfirst[d] + off[d], Val(D))
    base = ntuple(d -> Int(srcstarts[d][off[d] + 1]), Val(D))

    # Tensor-product stencil: P^D contributions, the weight of each the
    # product of its D one-dimensional weights.
    acc = zero(eltype(dest))
    for m in 0:(P^D - 1)
        moff = ntuple(d -> (m ÷ P^(d - 1)) % P, Val(D))
        w = one(eltype(dest))
        for d in 1:D
            w *= weights[d][moff[d] + 1, off[d] + 1]
        end
        acc += w * src[ntuple(d -> base[d] + moff[d], Val(D))..., v, sblock]
    end
    dest[tidx..., v, tblock] = acc
end

function run_group!(dest, src, group::TransferGroup{T,D}, nvars::Integer,
                    backend) where {T,D}
    n = ntransfers(group)
    n == 0 && return nothing
    blen = boxsize(group)
    stride = ntuple(d -> prod(ntuple(e -> blen[e], d - 1)), D)
    tfirst = ntuple(d -> group.stencils[d].targetfirst, D)
    order = size(group.stencils[1].weights, 1)
    srcstarts = ntuple(d -> group.stencils[d].srcstart, D)
    weights = ntuple(d -> group.stencils[d].weights, D)

    kernel! = transfer_kernel!(backend)
    kernel!(dest, src, group.targetblocks, group.sourceblocks, srcstarts, weights,
            tfirst, blen, stride, Val(order), Val(D);
            ndrange=(prod(blen), nvars, n))
    return nothing
end

run_group!(fs::FieldSet{T,D}, group::TransferGroup{T,D}, backend) where {T,D} =
    run_group!(fs.work, fs.work, group, fs.nvars, backend)

"""
    fill_ghosts!(fs::FieldSet, schedule::GhostSchedule; boundary=nothing)

Fill every ghost cell of every block, by replaying `schedule`.

The three cases — same-level copy, restriction from finer neighbors,
prolongation from a coarser neighbor — run in the phases described in
[`GhostSchedule`](@ref): copies and restrictions together first, then
prolongations swept coarsest target first, then the physical boundary
hook. Each phase is an embarrassingly parallel loop over blocks with a
barrier between phases.

Periodic boundaries need nothing special; the tree wraps around, so they
are ordinary transfers.

`boundary` is called once per ghost region facing outside a non-periodic
domain, as

    boundary(fs, blockindex, key, δ, region)

with `key` the block's [`MortonKey`](@ref), `δ` the outward direction,
and `region` the `CartesianIndices` of the ghost cells in stored
coordinates. Use [`cell_center`](@ref) to get their positions. Passing
`nothing` leaves those ghosts untouched, which is what a fully periodic
domain wants.

!!! note "Where the boundary hook runs"
    The hook runs after the copies and restrictions but **before** the
    prolongation sweep, not after every inter-block phase. A block
    sitting against the domain edge has prolongation stencils that reach
    *tangentially* past that edge, into the coarse source's own outer
    ghosts; filling those last would feed unwritten memory into the
    interpolation. The hook may therefore read the block's interior
    (as reflecting and extrapolating conditions do) but not other
    blocks' ghosts.
"""
function fill_ghosts!(fs::FieldSet{T,D}, schedule::GhostSchedule{T,D};
                      boundary=nothing) where {T,D}
    schedule.forest === fs.forest || throw(ArgumentError(
        "schedule was built for a different forest than the field set"))
    isstale(schedule) && throw(ArgumentError(
        "the forest changed since this schedule was built (generation " *
        "$(schedule.generation) -> $(generation(schedule.forest))); rebuild it"))
    nblocks(fs) == nleaves(schedule.forest) || throw(ArgumentError(
        "field set has $(nblocks(fs)) blocks but the schedule's forest has " *
        "$(nleaves(schedule.forest)) leaves; rebuild both"))

    backend = get_backend(fs.work)

    # Phase 1: same-level copies and restrictions. Both read interiors
    # only, so they cannot race with each other.
    for group in schedule.phase1
        run_group!(fs, group, backend)
    end
    synchronize(backend)

    # Physical boundaries, before prolongation rather than after
    # everything: a block against the domain edge has prolongation
    # stencils that reach tangentially past that edge into its coarse
    # source's outer ghosts, so those must already hold data.
    if boundary !== nothing
        for region in schedule.boundaries
            boundary(fs, Int(region.block), fs.forest.leaves[region.block],
                     region.direction, region.region)
        end
    end

    # Phase 2: prolongations, coarsest targets first. A prolongation may
    # read its coarse source's ghosts, which the earlier sweeps filled.
    for groups in schedule.phase2
        for group in groups
            run_group!(fs, group, backend)
        end
        synchronize(backend)
    end
    return fs
end

"""
    boundary_by_coordinates(f)

A boundary hook that sets each outer ghost cell from `f(x, v)`, with `x`
the cell center and `v` the variable index — the same signature
[`fill_by_coordinates!`](@ref) takes.

Useful when the exact solution is known (manufactured solutions,
convergence tests); real applications supply their own hook to impose
outgoing, reflecting, or symmetry conditions.
"""
boundary_by_coordinates(f) =
    function (fs, b, key, δ, region)
        forest = fs.forest
        for v in 1:fs.nvars
            block = blockview(fs, b, v)
            for idx in region
                block[idx] = f(cell_center(forest, key, Tuple(idx)), v)
            end
        end
        return nothing
    end
