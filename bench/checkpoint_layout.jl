# Where a checkpoint's raw data lie in the file: every contiguous
# dataset's address and size, and each chunked dataset's chunks, merged
# into runs in file order. This is what showed that HDF5 had placed the
# first chunks of a filtered dataset before the leaf columns, so that a
# rank's write of its chunks spanned them (M7 step 6; HISTORY.md, "Parallel
# checkpoints").
#
#     julia --project=<env with HDF5 and the filter packages> \
#         bench/checkpoint_layout.jl <file>
using HDF5, H5Zlz4, H5Zzstd, H5Zbitshuffle
const API = HDF5.API
path = ARGS[1]
items = Tuple{Int,Int,String}[]
function visit(g, prefix)
    for name in keys(g)
        obj = g[name]
        if obj isa HDF5.Group
            visit(obj, prefix * "/" * name)
        elseif obj isa HDF5.Dataset
            plist = API.h5d_get_create_plist(obj)
            layout = API.h5p_get_layout(plist)
            API.h5p_close(plist)
            if layout == API.H5D_CHUNKED
                space = API.h5d_get_space(obj)
                nchunks = Ref{API.hsize_t}()
                API.h5d_get_num_chunks(obj, space, nchunks)
                nd = length(size(obj))
                off = zeros(API.hsize_t, nd); mask = Ref{Cuint}(); addr = Ref{API.haddr_t}(); sz = Ref{API.hsize_t}()
                chunks = Tuple{Int,Int,Int}[]
                for i in 0:nchunks[]-1
                    API.h5d_get_chunk_info(obj, space, i, off, mask, addr, sz)
                    push!(chunks, (Int(addr[]), Int(sz[]), Int(off[1])))   # off[1] = block index (C order)
                end
                API.h5s_close(space)
                sort!(chunks)
                println("$prefix/$name: chunked, $(length(chunks)) chunks, addresses ",
                        "[$(chunks[1][1]), $(chunks[end][1] + chunks[end][2]))")
                # gaps between chunks, and the block order along the file
                gaps = [(chunks[i][1] - (chunks[i-1][1] + chunks[i-1][2])) for i in 2:length(chunks)]
                println("   gaps between consecutive chunks: ", count(!=(0), gaps), " nonzero, total ",
                        sum(gaps), " B, largest ", maximum(gaps))
                blocks = [c[3] for c in chunks]
                println("   blocks in file order are sorted: ", issorted(blocks),
                        "; first 20: ", blocks[1:min(end, 20)])
                for c in chunks
                    push!(items, (c[1], c[2], "$prefix/$name chunk block $(c[3])"))
                end
            else
                a = API.h5d_get_offset(obj)
                sz = API.h5d_get_storage_size(obj)
                println("$prefix/$name: $(layout == API.H5D_CONTIGUOUS ? "contiguous" : "compact"), address $(a == typemax(UInt64) ? "-" : Int(a)), $(Int(sz)) B")
                a != typemax(UInt64) && sz > 0 && push!(items, (Int(a), Int(sz), "$prefix/$name"))
            end
        end
        close(obj)
    end
end
h5open(path, "r") do f
    visit(f, "")
end
sort!(items)
println("raw data in file order (merged runs of chunks):")
i = 1
while i <= length(items)
    a, s, nm = items[i]; j = i
    base = split(nm, " chunk")[1]
    while j < length(items) && startswith(items[j+1][3], base * " chunk") && occursin(" chunk", nm)
        j += 1
    end
    e = items[j][1] + items[j][2]
    println("  [$a, $e) $(j > i ? "$base: chunks $(j - i + 1)" : nm)")
    global i = j + 1
end
println("file size ", filesize(path))
