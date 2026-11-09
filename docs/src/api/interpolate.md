# Point interpolation

The values of a field set, and their first and second derivatives, at
arbitrary points: a horizon finder's surface, a tracer, a sampled ray.
One batched launch per call, on the field set's backend; or a batch
located once by `locate` and evaluated at for several field sets.

```@docs
interpolate
interpolate!
locate
LocatedPoints
PatchPoint
locate_point
InterpolationBasis
Lagrange
ElementPolynomial
Region
Ellipsoid
multi_indices
```

## The atlas

An application's maps between physical space and its patches' logical
coordinates, recognised by these methods alone.

```@docs
TreeAMR.chart_locate
TreeAMR.chart_map
TreeAMR.chart_jacobian
TreeAMR.chart_hessian
```
