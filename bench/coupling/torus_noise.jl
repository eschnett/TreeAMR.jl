include("torus_noise_fn.jl")
const VARIANTS = [Variant("copy", :copy, 0, false, 6),
                  Variant("halfopen", :halfopen, 0, false, 6),
                  Variant("halfopen+avg", :halfopen, 0, true, 6),
                  Variant("overlap o=0", :overlap, 0, false, 6),
                  Variant("overlap o=1", :overlap, 1, false, 6),
                  Variant("overlap o=2", :overlap, 2, false, 6),
                  Variant("overlap o=3", :overlap, 3, false, 6)]
const smooth = Setup(1.0, 0.0, 0.0)
const kinked = Setup(0.8, 0.4, 0.3)
const wavy = Wavy(0.2, 0.3)

jobs = [(s, name, var, eps, n)
        for (s, name) in ((smooth, "flat"), (wavy, "wavy"), (kinked, "kinked"))
        for var in VARIANTS for eps in (0.0, 0.1) for n in (32, 64)
        if name == "kinked" || var.name in ("halfopen", "overlap o=2")]
res = Vector{Any}(undef, length(jobs))
Threads.@threads :greedy for k in eachindex(jobs)
    s, name, var, eps, n = jobs[k]
    res[k] = noise(s, var; eps, n, tend=200.0)
end
println("Random data, energy E(t)/E(0) over t ≤ 200")
println("torus   variant          ε    n   max E/E₀   final E/E₀  blow-up")
for (k, (s, name, var, eps, n)) in enumerate(jobs)
    @printf("%-7s %-14s %4.1f %4d  %9.3e  %9.3e   %s\n", name, var.name, eps, n,
            res[k][1], res[k][2], isnan(res[k][3]) ? "-" : @sprintf("t = %.1f", res[k][3]))
end
