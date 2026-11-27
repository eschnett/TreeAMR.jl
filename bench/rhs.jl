# A right-hand side on closed blocks three ways: what computing only the
# primaries saves, and what `unify!(du)` costs to put the copies back.
#
#     julia -t T --project=test bench/rhs.jl                         # cpu
#     TREEAMR_BENCH_BACKEND=cuda julia --project=<env> bench/rhs.jl  # cuda
#     TREEAMR_BENCH_BACKEND=metal julia --project=<env> bench/rhs.jl # metal
#
# A vertex-centered set on closed blocks (M22) holds every copy of a
# shared point, `(N+1)^D` own points a block where half-open blocks held
# `N^D`, so a right-hand side that computes every own point computes each
# shared point once a copy: 3/16 more work at `N = 16`. The application
# can skip the secondaries instead and call `unify!(du)`, which gives each
# its primary's value ("Closed blocks" in `CODE.md`). This measures the
# three forms, with one kernel body:
#
#     a  every own point (`map_blocks!`), no unify
#     b  every own point, the body behind `isprimary` — the early return
#        of `isprimary`'s docstring, written as an `if` around the body —
#        then `unify!(du)`
#     c  the primaries only: a launch per shape over a flat list of
#        (block, box), each box a block's primary pieces (`primary_mask`'s
#        bits, one per piece, and `piece_range`), merged where two boxes
#        of a block share a face; then `unify!(du)`
#
# Before anything is timed the three are checked bit for bit: `du` at
# every primary is the same under a, b and c, and b and c are the same
# everywhere after their `unify!`. A failed check stops the run.
#
# The body is a stand-in for an application's: for each variable a
# fourth-order Laplacian (13 points) and a product with the next
# variable, read from the filled working array. The fill runs once,
# outside the timing. The mesh is `bench/copies.jl`'s uniform periodic
# one (default), or `bench/nested.jl`'s; `G = 3`, operators of order 6.
#
# Output: a header line, `#` lines with the boxes, and one line per form,
#
#     R  <mesh>  N  <form>  <best s>  <ns per own point>
#
# with `unify` timed alone too (the cost b and c add to their kernels),
# and `fill` for scale. "Own point" is every point of `statelength`, so
# the three forms share a denominator.
#
#     TREEAMR_BENCH_N        points per block edge (default 16)
#     TREEAMR_BENCH_MESH     uniform (default) or nested
#     TREEAMR_BENCH_VARS     variables (default 20)
#     TREEAMR_BENCH_REPS     timed repetitions (default 20)
#     TREEAMR_BENCH_T        Float64 (default where the backend has it) or Float32
#     TREEAMR_BENCH_BACKEND  cpu (default), cuda, or metal

using TreeAMR
using KernelAbstractions: @kernel, @index, @Const, get_backend, synchronize, CPU,
                          supports_float64
using Printf: @printf

const D = 3
const N = parse(Int, get(ENV, "TREEAMR_BENCH_N", "16"))
const MESH = Symbol(get(ENV, "TREEAMR_BENCH_MESH", "uniform"))
const NVARS = parse(Int, get(ENV, "TREEAMR_BENCH_VARS", "20"))
const REPS = parse(Int, get(ENV, "TREEAMR_BENCH_REPS", "20"))
const BNAME = lowercase(get(ENV, "TREEAMR_BENCH_BACKEND", "cpu"))
const G = 3

if BNAME == "cuda"
    using CUDA
elseif BNAME == "metal"
    using Metal
elseif BNAME != "cpu"
    error("TREEAMR_BENCH_BACKEND must be cpu, cuda, or metal; got \"$BNAME\"")
end

const BACKEND =
    BNAME == "cuda" ? (CUDA.functional() ? CUDABackend() :
                       error("CUDA is not functional here")) :
    BNAME == "metal" ? (Metal.functional() ? MetalBackend() :
                        error("Metal is not functional here")) :
    CPU()
const T = let want = get(ENV, "TREEAMR_BENCH_T", "")
    isempty(want) ? (supports_float64(BACKEND) ? Float64 : Float32) :
    want == "Float64" ? Float64 : want == "Float32" ? Float32 :
    error("TREEAMR_BENCH_T must be Float64 or Float32, got \"$want\"")
end

function build_forest()
    if MESH === :uniform
        return Forest{T}(ntuple(_ -> 8, D); N=N, periodic=ntuple(_ -> true, D),
                         extents=ntuple(_ -> (zero(T), one(T)), D))
    end
    MESH === :nested || error("TREEAMR_BENCH_MESH must be uniform or nested")
    forest = Forest(ntuple(_ -> 4, D); N=N, reflecting=ntuple(_ -> (true, false), D),
                    extents=ntuple(_ -> (zero(T), T(32)), D))
    for l in 1:3
        edge = T(32) / 2^l
        refine!(forest, filter(forest.leaves) do k
                    level(k) == l - 1 || return false
                    ext = block_extent(forest, k)
                    return all(d -> ext[d][2] <= edge, 1:D)
                end)
    end
    balance!(forest)
    return forest
end

# The body at own point `I` of block `b`: every variable's fourth-order
# Laplacian and its product with the next variable.
@inline function body!(du, work, spacings, I::NTuple{3,Int}, b::Int, nvars::Int)
    c = (I[1] + G, I[2] + G, I[3] + G)
    h = spacings[b]
    s = inv(12 * h * h)
    for v in 1:nvars
        u0 = work[c[1], c[2], c[3], v, b]
        lap = -90 * u0
        for d in 1:3
            p1 = Base.setindex(c, c[d] + 1, d)
            m1 = Base.setindex(c, c[d] - 1, d)
            p2 = Base.setindex(c, c[d] + 2, d)
            m2 = Base.setindex(c, c[d] - 2, d)
            lap += 16 * (work[p1..., v, b] + work[m1..., v, b]) -
                   (work[p2..., v, b] + work[m2..., v, b])
        end
        w = work[c[1], c[2], c[3], mod1(v + 1, nvars), b]
        du[I[1], I[2], I[3], v, b] = s * lap + u0 * w
    end
    return nothing
end

@kernel function rhs_all!(du, @Const(work), @Const(spacings), nvars)
    I = @index(Global, NTuple)
    body!(du, work, spacings, (Int(I[1]), Int(I[2]), Int(I[3])), Int(I[4]), nvars)
end

@kernel function rhs_masked!(du, @Const(work), @Const(spacings), mask, nvars)
    I = @index(Global, NTuple)
    if isprimary(mask, I[1], I[2], I[3], I[4])
        body!(du, work, spacings, (Int(I[1]), Int(I[2]), Int(I[3])), Int(I[4]), nvars)
    end
end

# Over the boxes of one shape: item `(i, j, k, n)` is point `(i, j, k)` of
# box `n`, whose block and first own point are `blocks[n]` and `starts[n]`.
@kernel function rhs_boxes!(du, @Const(work), @Const(spacings), @Const(blocks),
                            @Const(starts), nvars)
    I = @index(Global, NTuple)
    n = I[4]
    o = starts[n]
    body!(du, work, spacings, (Int(o[1]) + I[1] - 1, Int(o[2]) + I[2] - 1,
                               Int(o[3]) + I[3] - 1), Int(blocks[n]), nvars)
end

# A block's primary pieces as boxes of own indices, merged while two
# share a face: from the mask's bit per piece and each piece's range.
function primary_boxes(bits::UInt32, c::NTuple{D,Int})
    boxes = NTuple{D,UnitRange{Int}}[]
    for cls in TreeAMR.closed_pieces(c)
        (bits >> TreeAMR.piece_index(cls)) & 0x1 == 0x1 || continue
        push!(boxes, ntuple(d -> TreeAMR.piece_range(N, G, c[d],
                                                     TreeAMR.piece_span(cls[d])) .- G, D))
    end
    merged = true
    while merged
        merged = false
        for i in eachindex(boxes), j in eachindex(boxes)
            i == j && continue
            a, b = boxes[i], boxes[j]
            for d in 1:D
                all(e -> e == d || a[e] == b[e], 1:D) || continue
                last(a[d]) + 1 == first(b[d]) || continue
                boxes[i] = Base.setindex(a, first(a[d]):last(b[d]), d)
                deleteat!(boxes, j)
                merged = true
                break
            end
            merged && break
        end
    end
    return boxes
end

# Per shape, the (block, first point) of every box of that shape.
function box_lists(fs)
    c = staggers(fs)
    bits = Array(primary_mask(fs).pieces)
    byshape = Dict{NTuple{D,Int},Tuple{Vector{Int32},Vector{NTuple{D,Int32}}}}()
    for b in eachindex(bits), box in primary_boxes(bits[b], c)
        bl, st = get!(() -> (Int32[], NTuple{D,Int32}[]), byshape, map(length, box))
        push!(bl, b)
        push!(st, map(r -> Int32(first(r)), box))
    end
    return [(shape, TreeAMR.todevice(BACKEND, bl), TreeAMR.todevice(BACKEND, st))
            for (shape, (bl, st)) in sort!(collect(byshape); by=first)]
end

function best(f, reps=REPS)
    f()
    t = Inf
    for _ in 1:reps
        t = min(t, @elapsed f())
    end
    return t
end

function main()
    forest = build_forest()
    parity = [isodd(v) ? EvenParity :
              ntuple(d -> d == mod1(v ÷ 2, D) ? OddParity : EvenParity, D) for v in 1:NVARS]
    fs = MESH === :nested ?
         FieldSet{T}(forest, NVARS; G=G, centering=vertexcentered(D), parity=parity,
                     backend=BACKEND) :
         FieldSet{T}(forest, NVARS; G=G, centering=vertexcentered(D), backend=BACKEND)
    schedule = GhostSchedule(fs, Operators(prolongation=6, restriction=6))
    fill_by_coordinates!((x, v) -> sin(T(0.1) * x[1] + v) * cos(T(0.07) * x[2] - x[3] / 50),
                         fs)
    fill_ghosts!(fs, schedule)
    spacings = TreeAMR.todevice(BACKEND, block_spacings(forest, T))
    mask = primary_mask(fs)
    lists = box_lists(fs)
    nbox = sum(l -> length(l[2]), lists)
    npoints = sum(l -> prod(l[1]) * length(l[2]), lists)
    owned = statelength(fs) ÷ NVARS
    @printf("backend=%s T=%s threads=%d mesh=%s N=%d G=%d blocks=%d vars=%d\n",
            BNAME, T, Threads.nthreads(), MESH, N, G, nblocks(fs), NVARS)
    @printf("# %d own points, %d primaries (%.4f), in %d boxes of %d shapes\n", owned,
            npoints, npoints / owned, nbox, length(lists))

    dus = [statevector(fs) for _ in 1:3]
    A = [statearray(du, fs) for du in dus]
    run_a!() = map_blocks!(rhs_all!, fs, A[1], fs.work, spacings, NVARS)
    run_b!() = map_blocks!(rhs_masked!, fs, A[2], fs.work, spacings, mask, NVARS)
    function run_c!()
        for (shape, bl, st) in lists
            TreeAMR.launch_by_owner!(rhs_boxes!, BACKEND, A[3], fs.work, spacings, bl, st,
                                     NVARS; ndrange=(shape..., length(bl)))
        end
        synchronize(BACKEND)
    end

    # The check, before any timing: the primaries agree bit for bit, and so
    # do b and c everywhere once unified.
    fill!(dus[2], T(NaN))
    fill!(dus[3], T(NaN))
    run_a!()
    run_b!()
    run_c!()
    ha, hb, hc = Array(A[1]), Array(A[2]), Array(A[3])
    hmask = TreeAMR.PrimaryMask(Array(mask.pieces), mask.N, mask.c)
    for b in axes(ha, 5), I in CartesianIndices(size(ha)[1:D])
        isprimary(hmask, Tuple(I)..., b) || continue
        for v in 1:NVARS
            x = ha[I, v, b]
            reinterpret(UInt64, Float64(x)) == reinterpret(UInt64, Float64(hb[I, v, b])) ==
            reinterpret(UInt64, Float64(hc[I, v, b])) ||
                error("form b or c differs from a at primary $(Tuple(I)) of block $b, " *
                      "variable $v")
        end
    end
    unify!(dus[2], fs, schedule)
    unify!(dus[3], fs, schedule)
    reinterpret(UInt8, Array(dus[2])) == reinterpret(UInt8, Array(dus[3])) ||
        error("forms b and c differ after unify!(du)")
    any(isnan, Array(dus[2])) && error("unify!(du) left a secondary unwritten")
    println("# check: a, b and c agree bit for bit at every primary, b and c everywhere " *
            "after unify!")

    t_a = best(run_a!)
    t_b = best(run_b!)
    t_c = best(run_c!)
    t_u = best(() -> (unify!(dus[2], fs, schedule); synchronize(BACKEND)))
    t_f = best(() -> (fill_ghosts!(fs, schedule); synchronize(BACKEND)))
    for (name, t) in (("a", t_a), ("b", t_b), ("b+unify", t_b + t_u), ("c", t_c),
                      ("c+unify", t_c + t_u), ("unify", t_u), ("fill", t_f))
        @printf("R\t%s\t%d\t%s\t%.6f\t%.4f\n", MESH, N, name, t, 1e9 * t / owned)
    end
    return nothing
end

main()
