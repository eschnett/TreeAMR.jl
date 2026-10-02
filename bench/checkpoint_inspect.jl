# A checkpoint written by bench/checkpoint.jl, read serially and compared
# with the state it should hold: the leaf columns row by row, and the data
# block by block and variable by variable, each bad run reported with the
# rank that wrote it at `nranks` ranks and its byte range in the file.
# This is how the damaged file of job 567855 was found to have lost one
# rank's leaf coordinates and nothing else (M7 step 6; CODE.md, "Parallel
# checkpoints"); bench/checkpoint_layout.jl shows where the chunks lie.
#
#     julia --project=<env with HDF5 and the filter packages> \
#         bench/checkpoint_inspect.jl <file> <roots> <nranks> [blast|pulse]
using TreeAMR, HDF5, H5Zlz4, H5Zzstd, H5Zbitshuffle
const API = HDF5.API
path = ARGS[1]; ROOTS = parse(Int, ARGS[2]); P = parse(Int, ARGS[3])
which = length(ARGS) >= 4 ? ARGS[4] : "blast"
D = 3; N = 16; LEVELS = 2
function build_forest(r₀)
    forest = Forest(ntuple(_ -> ROOTS, D); N=N, extents=ntuple(_ -> (-1.0, 1.0), D))
    for lvl in 0:LEVELS-1
        targets = filter(forest.leaves) do k
            level(k) == lvl || return false
            ext = block_extent(forest, k)
            w = ext[1][2] - ext[1][1]
            near = sqrt(sum(d -> max(ext[d][1], 0.0, -ext[d][2])^2, 1:D))
            far = sqrt(sum(d -> max(abs(ext[d][1]), abs(ext[d][2]))^2, 1:D))
            return near <= r₀ + w / 2 && far >= r₀ - w / 2
        end
        refine!(forest, targets); balance!(forest)
    end
    return forest
end
function blast(forest, r₀)
    fs = FieldSet(forest, D + 2; G=2, centering=cellcentered(D))
    γ = 5 / 3; ρatm, Patm = 1.0, 1e-5
    state = function (x)
        r = sqrt(sum(abs2, x))
        r >= r₀ && return (ρatm, ntuple(_ -> 0.0, D)..., Patm / (γ - 1))
        s = r / r₀; ρ = ρatm * (0.05 + 3.95 * s^6); vr = 0.75 * s
        P = 0.3 + 0.2 * (1 - s^2)
        m = ntuple(d -> ρ * vr * x[d] / max(r, eps()), D)
        return (ρ, m..., P / (γ - 1) + ρ * vr^2 / 2)
    end
    fill_by_coordinates!(AllVariables(state), fs)
    return fs
end
function pulse(forest, r₀)
    fs = FieldSet(forest, 2; G=2, centering=vertexcentered(D)); σ = 0.1
    fill_by_coordinates!(fs) do x, v
        r = sqrt(sum(abs2, x)); g = exp(-(r - r₀)^2 / (2σ^2))
        return v == 1 ? g : (r - r₀) / σ^2 * g
    end
    return fs
end
# rank of block b (1-based) for P ranks, as blockrange splits them
n = 0
forest = build_forest(0.5); n = nleaves(forest)
counts = [div(n, P) + (r < rem(n, P) ? 1 : 0) for r in 0:P-1]
starts = cumsum([0; counts[1:end-1]])
rankof(b) = searchsortedlast(starts, b - 1) - 1
println("leaves $n, ranks $P, first blocks: ", starts[1:min(end, 40)])
runs(bad) = begin
    r = UnitRange{Int}[]
    for i in bad
        if !isempty(r) && last(r[end]) == i - 1; r[end] = first(r[end]):i; else push!(r, i:i); end
    end
    r
end
h5open(path, "r") do f
    g = f["TreeAMR.jl/forest"]
    want = Dict("root" => [Int32(k.root) for k in forest.leaves],
                "level" => [Int8(k.level) for k in forest.leaves],
                "coords" => reduce(hcat, [UInt32[k.coords...] for k in forest.leaves]))
    for name in ("root", "level", "coords")
        dset = g[name]; got = read(dset); off = API.h5d_get_offset(dset)
        el = sizeof(eltype(got)) * (name == "coords" ? D : 1)
        bad = name == "coords" ? findall(i -> got[:, i] != want[name][:, i], 1:n) :
              findall(i -> got[i] != want[name][i], 1:n)
        println("$name: file offset $off (mod 4096 = $(off % 4096)), $(el) B/row, ",
                length(bad), " bad rows")
        for r in runs(bad)
            b0 = off + (first(r) - 1) * el; b1 = off + last(r) * el
            println("   rows $r (ranks $(rankof(first(r)))..$(rankof(last(r)))), bytes [$b0, $b1)",
                    " page $(b0 ÷ 4096)..$((b1 - 1) ÷ 4096), b0 mod 4096 = $(b0 % 4096), b1 mod 4096 = $(b1 % 4096),",
                    " got first ", name == "coords" ? got[:, first(r)] : got[first(r)],
                    " zero: ", name == "coords" ? all(iszero, got[:, r]) : all(iszero, got[r]))
        end
        # rank boundaries' byte offsets
    end
    println("rank boundary byte offsets of coords (mod 4096): ",
            [(API.h5d_get_offset(g["coords"]) + s * 12) % 4096 for s in starts[2:min(end, 8)]])
    # data
    fs = which == "blast" ? blast(forest, 0.5) : pulse(forest, 0.5)
    u = statevector(fs); gather!(u, fs)
    nv = fs.nvars; per = length(u) ÷ (n * nv)
    U = reshape(u, per, nv, n)
    dset = f["TreeAMR.jl/fieldsets/U/data"]
    plist = API.h5d_get_create_plist(dset)
    chunked = API.h5p_get_layout(plist) == API.H5D_CHUNKED
    API.h5p_close(plist)
    println("data: size $(size(dset)), ", chunked ? "chunked (addresses: " *
            "bench/checkpoint_layout.jl)" :
            "contiguous at $(API.h5d_get_offset(dset)), $(per * nv * 8) B a block")
    bad = Tuple{Int,Int,String}[]
    for b in 1:n, v in 1:nv
        got = try
            vec(dset[:, :, :, v, b])
        catch err
            push!(bad, (b, v, "read error: " * first(sprint(showerror, err), 200))); continue
        end
        if reinterpret(UInt64, got) != reinterpret(UInt64, U[:, v, b])
            nz = count(iszero, got); nd = count(i -> got[i] !== U[i, v, b], eachindex(got))
            push!(bad, (b, v, "mismatch: $nd of $(length(got)) differ, $nz zero"))
        end
    end
    println("data: ", length(bad), " bad (block, var) of ", n * nv)
    for (b, v, why) in bad[1:min(end, 60)]
        println("   block $b var $v rank $(rankof(b)): $why")
    end
end
