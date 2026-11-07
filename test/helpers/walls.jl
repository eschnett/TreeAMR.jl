# The definitions of `walls_tests.jl`, split off so that each test file runs
# on its own: every test process includes all of `helpers/` ("Parallel
# groups" in CODE.md).

# Point symmetric about the centre of a `faces_forest` box of side `L`,
# so even under every reflection of a dimension through the centre, and
# with no polynomial structure.
point_symmetric(D, L) = (x, v) -> prod(d -> cos(1.7 * (x[d] - L / 2)) +
                                            0.3 * (x[d] - L / 2)^2, 1:D) + v

# The worst difference between every stored value of `fs` and the value
# its mirror block holds at the mirrored index, under the point
# reflection `x → L − x` of the domain; and how many values were
# compared. A block's mirror is the leaf whose box is its own reflected,
# and stored index `i` maps to `n + 1 − i`, `n` the stored extent, since
# the reflection turns a block's low ghosts into the mirror's high ones.
function point_mirror_error(fs::FieldSet{T,D}, L) where {T,D}
    forest = fs.forest
    lower(k) = (level(k), ntuple(d -> block_extent(forest, k)[d][1], D))
    upper(k) = (level(k), ntuple(d -> L - block_extent(forest, k)[d][2], D))
    index = Dict(lower(blockkey(fs, b)) => b for b in 1:nblocks(fs))
    n = ntuple(d -> size(fs.work, d), D)
    worst, count = 0.0, 0
    for b in 1:nblocks(fs)
        m = get(index, upper(blockkey(fs, b)), nothing)
        m === nothing && return nothing
        for v in 1:fs.nvars, idx in CartesianIndices(n)
            j = ntuple(d -> n[d] + 1 - idx[d], D)
            worst = max(worst, abs(fs.work[Tuple(idx)..., v, b] - fs.work[j..., v, m]))
            count += 1
        end
    end
    return worst, count
end

# One kind for both faces of every dimension, so that the box is
# symmetric under its point reflection: the hook's, mirrored, clamped,
# or a mix of the three.
const WALL_BOXES = Dict(1 => [(:outer,), (:reflect_both,), (:outflow_both,)],
                        2 => [(:outer, :outer), (:reflect_both, :reflect_both),
                              (:outflow_both, :outflow_both), (:outflow_both, :reflect_both)],
                        3 => [(:outer, :reflect_both, :outflow_both)])

staggered_or_vertex(D) = filter(C -> any(==(:vertex), C), allcenterings(Val(D)))
