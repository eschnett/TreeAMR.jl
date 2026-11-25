include("torus_noise_fn.jl")
kinked = Setup(0.8, 0.4, 0.3); harsh = Setup(0.6, 0.9, 0.6)
vars = [Variant("overlap o=0", :overlap, 0, false, 6), Variant("overlap o=0 m=4", :overlap, 0, false, 4),
        Variant("overlap o=2 m=4", :overlap, 2, false, 4)]
jobs = [(s, nm, v, e, n) for (s, nm) in ((kinked, "mild"), (harsh, "harsh")) for v in vars
        for e in (0.2, 0.5, 1.0) for n in (32, 64)]
res = Vector{Any}(undef, length(jobs))
Threads.@threads :greedy for k in eachindex(jobs)
    s, nm, v, e, n = jobs[k]
    res[k] = noise(s, v; eps=e, n, tend=200.0)
end
println("Torus, random data to t = 200: final E/E₀ or blow-up time (ε = 0.2, 0.5, 1.0)")
for (s, nm) in ((kinked, "mild"), (harsh, "harsh")), v in vars, n in (32, 64)
    @printf("  %-5s %-16s %3d", nm, v.name, n)
    for e in (0.2, 0.5, 1.0)
        r = res[findfirst(j -> j[2] == nm && j[3] === v && j[4] == e && j[5] == n, jobs)]
        isnan(r[3]) ? @printf("  %-9.1e", r[2]) : @printf("  t=%-7.1f", r[3])
    end
    println()
end
