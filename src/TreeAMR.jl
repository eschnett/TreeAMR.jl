"""
    TreeAMR

A tree-based (octree-style) AMR discretization for Julia. Provides the
mesh, the storage, and the inter-grid operations for block-structured
adaptive mesh refinement — no physics.

See `CODE.md` in the package root for the full design document.
"""
module TreeAMR

using KernelAbstractions: @kernel, @index, @Const, get_backend, synchronize,
                          Backend, CPU, allocate, supports_float64
import KernelAbstractions

# Tree core (M1)
export MortonKey, MAX_LEVEL, level, parentkey, childkeys, sortedchildkeys, isancestor
export Forest, nleaves, maxlevel, root_position, root_index, alldirections, floattype
export find_leaf, isleaf, neighbor_keys, refine!, coarsen!, balance!, isbalanced, generation
export root_spacing, spacing, minimum_spacing, block_origin, block_extent,
       block_spacings, block_origins
export FieldSet, nblocks, blockkey, blockview, interiorview, closedview,
       coordinates, fill_by_coordinates!
export cellcentered, vertexcentered, facecentered, edgecentered, staggers

# Ghost exchange and interpolation operators (M2)
export Operators, OperatorFamily, PointValue, Conservative, check_operators,
       GhostSchedule, isstale, fill_ghosts!, boundary_by_coordinates, CellBoundary

# ODE coupling (M3)
export statelength, statevector, statearray, scatter!, gather!, map_blocks!,
       block_mapreduce, volume_weighted_norm

# Regridding (M4)
export RegridFlag, Refine, Coarsen, Keep, flag_blocks, buffered_flags, complete_marks,
       regrid!, adapt_to_initial_data!, total_mass

# GPU (M6)
export firing_boxes

include("threading.jl")
include("device.jl")
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
