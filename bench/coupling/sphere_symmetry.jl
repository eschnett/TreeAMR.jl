include("sphere.jl")
# Mirror asymmetry, as sphere_confirm.jl measures it (every owned point
# against the owner of its image), and restricted to points strictly
# inside a face, whose image is strictly inside a face too, so that no
# comparison is between two copies of a face point.
function asym(S, M; inner)
    n = S.n; d = 0.0; a = 0.0; F = [T.F for T in S.P]
    for p in 1:6, i in 0:n-1, j in 0:n-1
        inner && (i == 0 || j == 0) && continue
        X = pos(F[p], 2i / n - 1, 2j / n - 1)
        own = owner(F, n, M * X)
        own === nothing && continue
        q, a1, b1 = own
        inner && (a1 == 0 || b1 == 0) && continue
        d = max(d, abs(S.P[p].u[ix(i), ix(j)] - S.P[q].u[ix(a1), ix(b1)]))
        a = max(a, abs(S.P[p].u[ix(i), ix(j)]))
    end
    return d / a
end
mir = Matrix(Diagonal([-1.0, 1, 1]))
for (name, kw) in (("overlap o=0", (overlap=0,)), ("owned corners", (corner=:own,)),
                   ("owned corners + avg", (corner=:own, avg=true)))
    S = Sphere(32; kw..., eps=0.5)
    ws = Harmonic([1.0, 0, 0], 4)
    set_state!(S, X -> exact(ws, X, 0.0))
    evolve!(S, 10.0, 0.25)
    @printf("%-22s all owned points %.1e   strictly inside faces %.1e\n", name,
            asym(S, mir; inner=false), asym(S, mir; inner=true))
end
