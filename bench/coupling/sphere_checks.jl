include("sphere.jl")
sols = orientations()
println("orientations with one owner per edge: ", length(sols))
println("their unowned corners: ", sort(unique([sort(s[2]) for s in sols])))
w = Harmonic([0.3, -0.5, 1.0], 3)
println("Laplacian of exact data, and the fill by kind (max errors)")
for n in (16, 32, 64)
    S = Sphere(n)
    @printf("  n = %3d: copies %d, ghosts %d, corner points %d;", n, length(S.copies),
            length(S.ghosts), length(S.corners))
    set_state!(S, X -> exact(w, X, 0.0))
    K = workspace(S)[2]
    e = 0.0
    for p in 1:6, i in 0:n-1, j in 0:n-1
        rhs_point!(K[p][1], K[p][2], S.P[p], i, j, 0.0, n / 2)
        X = pos(S.P[p].F, 2i / n - 1, 2j / n - 1)
        e = max(e, abs(K[p][2][ix(i), ix(j)] + w.ω^2 * exact(w, X, 0.0)[1]))
    end
    @printf(" Laplacian %.2e;", e)
    for (name, list) in (("copies", S.copies), ("corners", S.corners), ("ghosts", S.ghosts))
        fill_ghosts!(S)   # from exact owned data
        set_state!(S, X -> exact(w, X, 0.0))
        apply!(S, list)
        ee = maximum(abs(S.P[e.tp].u[ix(e.ti), ix(e.tj)] -
                         exact(w, pos(S.P[e.tp].F, 2e.ti / n - 1, 2e.tj / n - 1), 0.0)[1])
                     for e in list)
        @printf(" %s %.2e", name, ee)
    end
    println()
end
