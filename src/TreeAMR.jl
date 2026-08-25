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
export find_leaf, isleaf, neighbor_keys, refine!, coarsen!, balance!, isbalanced
export root_spacing, spacing, minimum_spacing, block_origin, block_extent, cell_center
export FieldSet, nblocks, blockkey, blockview, interiorview, fill_by_coordinates!

include("morton.jl")
include("forest.jl")
include("geometry.jl")
include("storage.jl")

end
