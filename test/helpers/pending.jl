# Pending tests of M22, "Closed blocks" (PLAN.md). The milestone is built
# step by step on an integration branch, and what a later step rebuilds is
# refused until then; a test that needs it is pending, marked with the
# step that brings it back. A pending test shows in the summary as one
# Broken test under its own name, so that the count of what is still
# pending is visible in every run, and the branch merges when none is.

"""
    pending(step, name)

Record the test `name` as pending until M22 step `step`: one `@test_skip`
in a testset of that name, which the summary counts as Broken.
"""
function pending(step::Integer, name::AbstractString)
    @testset "$name — pending until M22 step $step" begin
        @test_skip false
    end
    return nothing
end

"""
    @pending step name begin … end

A testset that is pending until M22 step `step`: its body is not run (nor
compiled), and it is recorded as [`pending`](@ref)`(step, name)`. The
first line of a pending testset reads `@pending 3 "name" for D in …`,
or with a `begin` block, so that bringing it back is one edit.
"""
macro pending(step, name, body)
    # A testset over a loop, `@testset "… D=$D" for D in …`, is pending once
    # per iteration, so that its name can say which.
    if body isa Expr && body.head === :for
        return esc(Expr(:for, body.args[1], :(pending($step, $name))))
    end
    return :(pending($(esc(step)), $(esc(name))))
end

# Whether a centering, a field set or a layout has a vertex-like dimension:
# what M22 restricts during the milestone (vertex, face and edge sets).
vertexlike(C::Tuple) = any(==(:vertex), C)
vertexlike(fs::FieldSet) = vertexlike(fs.centering)
