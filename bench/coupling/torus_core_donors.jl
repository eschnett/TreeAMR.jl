include("torus_noise_fn.jl")
const kinked = Setup(0.8, 0.4, 0.3)
const wave = PlaneWave(kinked, 2, 1, 0.3)
vars = [Variant("overlap o=2 core", :overlapcore, 2, false, 6),
        Variant("overlap o=3 core", :overlapcore, 3, false, 6)]
epss = (0.05, 0.1, 0.2, 0.4)
jobs = [(v, e, n) for v in vars for e in epss for n in (32, 64)]
res = Vector{Any}(undef, length(jobs))
conv = Vector{Any}(undef, length(vars))
Threads.@threads :greedy for k in 1:length(jobs)+length(vars)
    if k <= length(jobs)
        v, e, n = jobs[k]
        res[k] = noise(kinked, v; eps=e, n, tend=200.0)
    else
        v = vars[k-length(jobs)]
        conv[k-length(jobs)] = map((16, 32, 64, 128)) do n
            sim = Sim(kinked, v, n)
            set_state!(sim, (x, y) -> exact(wave, x, y, 0.0))
            t = evolve!(sim, 2.0, 0.25)
            errors(sim, wave, t)[1]
        end
    end
end
for (v, c) in zip(vars, conv)
    @printf("%-18s L2 at t=2, n=16…128: %s  rates %s\n", v.name,
            join([@sprintf("%.2e", x) for x in c], " "),
            join([@sprintf("%.2f", log2(c[i] / c[i+1])) for i in 1:3], " "))
end
@printf("%-18s %4s", "variant", "n"); foreach(e -> @printf("  ε=%-8.2f", e), epss); println()
for v in vars, n in (32, 64)
    @printf("%-18s %4d", v.name, n)
    for e in epss
        r = res[findfirst(j -> j[1] === v && j[2] == e && j[3] == n, jobs)]
        isnan(r[3]) ? @printf("  %-10.2e", r[2]) : @printf("  t=%-8.1f", r[3])
    end
    println()
end
