include("torus.jl")
const VARIANTS = [Variant("copy", :copy, 0, false, 6),
                  Variant("halfopen", :halfopen, 0, false, 6),
                  Variant("halfopen+avg", :halfopen, 0, true, 6),
                  Variant("overlap o=0", :overlap, 0, false, 6),
                  Variant("overlap o=1", :overlap, 1, false, 6),
                  Variant("overlap o=2", :overlap, 2, false, 6),
                  Variant("overlap o=3", :overlap, 3, false, 6)]
const kinked = Setup(0.8, 0.4, 0.3)
const wave = PlaneWave(kinked, 2, 1, 0.3)
const CFL = 0.25

function convergence(var; eps=0.0, tend=2.0, ns=(16, 32, 64, 128))
    map(ns) do n
        sim = Sim(kinked, var, n; eps)
        set_state!(sim, (x, y) -> exact(wave, x, y, 0.0))
        t = evolve!(sim, tend, CFL)
        errors(sim, wave, t)
    end
end

conv = Vector{Any}(undef, length(VARIANTS))
Threads.@threads for k in eachindex(VARIANTS)
    conv[k] = convergence(VARIANTS[k])
end
println("Convergence, kinked torus, plane wave (2,1), t = 2, ε = 0, 6-point ghosts")
println("variant          n=16 L2   n=32      n=64      n=128     rates L2         | L∞ near glue rates")
for (k, var) in enumerate(VARIANTS)
    e = conv[k]
    l2 = [x[1] for x in e]; ln = [x[3] for x in e]
    r = [log2(l2[i] / l2[i+1]) for i in 1:3]
    rn = [log2(ln[i] / ln[i+1]) for i in 1:3]
    @printf("%-14s %9.2e %9.2e %9.2e %9.2e  %5.2f %5.2f %5.2f | %5.2f %5.2f %5.2f\n",
            var.name, l2..., r..., rn...)
end
flush(stdout)

