# Ghost exchange and conservation

The inter-grid operators, the cached ghost-exchange schedule, the
physical-boundary hook, and the flux fixup at coarse-fine faces.

## Ghost exchange and operators

```@docs
Operators
OperatorFamily
ElementLocal
check_operators
GhostSchedule
isstale
fill_ghosts!
boundary_by_coordinates
CellBoundary
Relative
```

## Conservation at coarse-fine faces

```@docs
InterfaceSchedule
restrict_interfaces!
```
