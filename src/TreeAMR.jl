"""
    TreeAMR

A tree-based (octree-style) AMR discretization for Julia. Provides the
mesh, the storage, and the inter-grid operations for block-structured
adaptive mesh refinement — no physics.

See `CODE.md` in the package root for the full design document.
"""
module TreeAMR

# Tree core (M1)
export MortonKey, MAX_LEVEL, level, parentkey, childkeys, sortedchildkeys, isancestor
export Forest, nleaves, maxlevel, root_position, root_index, alldirections
export find_leaf, isleaf, neighbor_keys, refine!, coarsen!, balance!, isbalanced, generation
export root_spacing, spacing, minimum_spacing, block_origin, block_extent, cell_center,
       block_spacings
export FieldSet, nblocks, blockkey, blockview, interiorview, fill_by_coordinates!

# Ghost exchange and interpolation operators (M2)
export Operators, check_operators, GhostSchedule, isstale, fill_ghosts!,
       boundary_by_coordinates

# ODE coupling (M3)
export statelength, statevector, statearray, scatter!, gather!, map_blocks!,
       volume_weighted_norm

# Regridding (M4)
export RegridFlag, Refine, Coarsen, Keep, flag_blocks, complete_marks, regrid!,
       adapt_to_initial_data!, total_mass

include("morton.jl")
include("forest.jl")
include("geometry.jl")
include("storage.jl")
include("operators.jl")
include("schedule.jl")
include("ghosts.jl")
include("state.jl")
include("regrid.jl")

end
