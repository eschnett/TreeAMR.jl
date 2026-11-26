# A precompile workload (M22 step 9): the first fill in a fresh process
# compiled the schedule's build, the fill's stages and their kernels on
# the CPU, about 20 s before anything ran. PrecompileTools runs this when
# the package is precompiled, so that what it compiles is cached in the
# package image: a cell and a vertex set in 2D and 3D, a schedule, a fill,
# `unify!`, a gather and a norm. A separable choice: Erik's to take or
# drop (HISTORY.md, M22 step 9).
@setup_workload begin
    @compile_workload begin
        for D in (2, 3), centering in (cellcentered(D), vertexcentered(D))
            forest = Forest(ntuple(_ -> 2, D); N=8)
            refine!(forest, forest.leaves[1])
            balance!(forest)
            fs = FieldSet(forest, 2; G=2, centering=centering)
            fill!(fs.work, 1.0)
            sched = GhostSchedule(fs, Operators(prolongation=4, restriction=4))
            fill_ghosts!(fs, sched; boundary=boundary_by_coordinates((x, v) -> 1.0))
            u = statevector(fs)
            gather!(u, fs)
            any(==(1), staggers(fs)) && unify!(u, fs, sched)
            volume_weighted_norm(fs, u)
        end
    end
end
