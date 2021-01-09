"""Package for local path sampling."""

"""
CORE LOGIC
==========

INPUTS
------

- Global path
- Cubic approximations of track (left and right in form of coefficients)
- Ego pose (minimum x, y and yaw)
- Obstacle pose (minimum x, y and yaw)

PIPELINE
--------

- Find closest point to obstacle in global path
- Drop perpendicular to global path at the point, the line intersects
  track boundaries at p1 and p2.
- Foot of perpendicular from obstacle position to perpendicular is p0.

- For lateral splines
    - Sample points from line segment between p1 and p0 and p0 and p2
      with sufficient padding.
    - Compute cubic spline

- For longitudinal splines
    - Sample points to forward and backward of the closest point in
      global path with sufficient look-ahead indices
    - Compute cubic spline

"""

__author__ = [
    'Harish Iniyarajan',
    'Ashwin A Nayar'
]
