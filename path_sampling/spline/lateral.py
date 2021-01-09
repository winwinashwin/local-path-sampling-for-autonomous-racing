import numpy as np

from ._base import _BaseSpG
from .._core import (
    intersection_line_cubic,
    parametrise_lineseg,
    linalg_solve,
    polyeval
)


class LateralSpG(_BaseSpG):

    def __init__(self, gp_handler):
        super(LateralSpG, self).__init__(gp_handler)

    def generate(self, n, ego_pose, road_poly_left, road_poly_right, look_ahead, padding):
        closest_pt_idx = self._gp_handler.get_closest_point(ego_pose)
        ppd_line = self._gp_handler.get_perpendicular(
            closest_pt_idx, look_ahead
        )

        p1 = intersection_line_cubic(ppd_line, road_poly_left)
        p2 = intersection_line_cubic(ppd_line, road_poly_right)

        parametric_pt = parametrise_lineseg(p1, p2, padding)

        for e in np.linspace(0, 1, n):
            p = parametric_pt(e)

            yield self._generate_spline(ego_pose, p)

    def _generate_spline(self, ego_pose, target):
        theta = ego_pose.yaw

        # Tranform to ego frame, such that ego vehicle is at origin and heading is
        # along the x axis.

        # Translation
        shift_x = target.x - ego_pose.x
        shift_y = target.y - ego_pose.y
        # Rotation
        tx = shift_x * np.cos(theta) - shift_y * np.sin(theta)
        ty = shift_x * np.sin(theta) + shift_y * np.cos(theta)

        # (tx, ty) -> New target with respect to ego frame

        # - To generate the cubic spline(say function F) between 2 points - origin(ego pose)
        # and target waypoint(tx, ty), we assume four conditions
        #
        # 1. F(0) = 0
        # 2. F(tx) = ty
        # 3. F'(0) = 0
        # 4. F'(tx) = 0
        #
        # - Conditions 3 and 4 makes sure that the paths have an 'S' shape and vehicle heading
        # remains the same. (It is for this reason we do a transform to ego frame)
        # - These four conditions gives us a matrix eqn on coefficients a, b, c, d, such that
        # F(x) = a * x**3 + b * x**2 + c * x + d

        a = np.array([
            [0, 0, 0, 1],
            [tx ** 3, tx ** 2, tx, 1],
            [0, 0, 1, 0],
            [3 * (tx ** 2), 2 * tx, 1, 0]
        ])
        b = np.array([0, ty, 0, 0])

        # Solve for coefficients
        coeffs = linalg_solve(a, b)[::-1]

        # Generate x coordinates in ego frame for spline
        xs = np.linspace(0, tx, 1000)
        # Calculate y for each corresponding x
        ys = np.array([polyeval(x, coeffs) for x in xs])

        # Tranform all points (x, y) in spline back to global frame
        for i, (x, y) in enumerate(zip(xs, ys)):
            xs[i] = x * np.cos(theta) + y * np.sin(theta) + ego_pose.x
            ys[i] = y * np.cos(theta) - x * np.sin(theta) + ego_pose.y

        return xs, ys
