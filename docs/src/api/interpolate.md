# Point interpolation

The values of a field set, and their first derivatives, at arbitrary
points: a horizon finder's surface, a tracer, a sampled ray. One batched
launch per call, on the field set's backend.

```@docs
interpolate
interpolate!
locate_point
InterpolationBasis
Lagrange
Region
Ellipsoid
```
