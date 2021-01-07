"""Package for local path sampling."""

# Core logic
#
# Inputs - Global path, Track limits and ego pose (minimum x, y and yaw), N
# Output - N feasible local path samples to for furthur processing
#
# Core logic:
# - Find index of closest point in global path to the ego vehicle position.
# - At some lookahead indices ahead, drop a perpendicular to global path.
# - Find intersection of perpendicular with track boundaries (say p1 and p2)
# - Sample N points between p1 and p2 - algorithmically or via random distribution sampling
# - Spline from ego position to sampled waypoint, based on constraints
#
# Potential areas for improvement:
# - Submodule: core.py
#   - Intersection of line and cubic:
#       - Yet to implement pipeline if numpy throws exception although chances are slim
#       - If by chance solution gives three real roots, handle them (again very low probability)
#
# - Submodule: global_path_handler.py
#   - Calculating slope of global path
#       - Current approach: Newton method
#       - Alternative: Spline through global path and find slope of tanget at point
#       - If points on global path are close enough this shouldn't be an issue.

__author__ = [
    'Harish Iniyarajan',
    'Ashwin A Nayar'
]
