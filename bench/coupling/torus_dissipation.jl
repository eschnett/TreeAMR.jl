# Follow-up: how much dissipation each coupling needs on the kinked torus,
# and whether a narrower ghost stencil changes it. Random data, n = 32 and
# 64, t ≤ 200; E/E₀ at the end, or the blow-up time.
include("torus_noise_fn.jl")
const kinked = Setup(0.8, 0.4, 0.3)
vars = [Variant("halfopen", :halfopen, 0, false, 6),
        Variant("halfopen+avg", :halfopen, 0, true, 6),
        Variant("halfopen m=4", :halfopen, 0, false, 4),
        Variant("overlap o=1", :overlap, 1, false, 6),
        Variant("overlap o=2", :overlap, 2, false, 6),
        Variant("overlap o=2 m=4", :overlap, 2, false, 4)]
epss = (0.02, 0.05, 0.1, 0.2, 0.4)
jobs = [(v, e, n) for v in vars for e in epss for n in (32, 64)]
res = Vector{Any}(undef, length(jobs))
Threads.@threads :greedy for k in eachindex(jobs)
    v, e, n = jobs[k]
    res[k] = noise(kinked, v; eps=e, n, tend=200.0)
end
println("Kinked torus, random data, t ≤ 200: final E/E₀, or the blow-up time")
@printf("%-16s %4s", "variant", "n")
foreach(e -> @printf("  ε=%-8.2f", e), epss); println()
for v in vars, n in (32, 64)
    @printf("%-16s %4d", v.name, n)
    for e in epss
        r = res[findfirst(j -> j[1] === v && j[2] == e && j[3] == n, jobs)]
        isnan(r[3]) ? @printf("  %-10.2e", r[2]) : @printf("  t=%-8.1f", r[3])
    end
    println()
end
