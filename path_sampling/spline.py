"""Spline generation."""

import logging
from collections import deque
from typing import Generator, Tuple

import numpy as np

from ._core import (
    cubic_spline,
    intersection_line_cubic,
    parametrise_lineseg,
)
from .global_path_handler import GlobalPathHandler
from .types import PVector, Pose, RoadLinePolynom

_logger = logging.getLogger(__name__)


# Type aliases
PointGenerator = Generator[Tuple[np.ndarray, np.ndarray], None, None]


class SplineGenerator(object):
    """For generating splines."""

    def __init__(self, gp_handler: GlobalPathHandler, ego_pose: Pose, obs_pose: Pose,
                 road_poly_left: RoadLinePolynom, road_poly_right: RoadLinePolynom):
        """Constructor.

            Args:
                gp_handler (GlobalPathHandler): Handler for interacting with global path data
                ego_pose (Pose): Ego vehicle pose
                obs_pose (Pose): Pose of obstacle
                road_poly_left (RoadLinePolynom): Structure with cubic coeffs of track (left)
                road_poly_right (RoadLinePolynom): Structure with cubic coeffs of track (right)

            Returns:
                None

        """
        self._gp_handler = gp_handler
        self._ego_pose = ego_pose
        self._obs_pose = obs_pose

        self._coeffs_left = road_poly_left
        self._coeffs_right = road_poly_right

        # closest point to obstacle in global path
        self._cls_pt = self._gp_handler.get_closest_point(self._obs_pose)
        # Slope intercept data of perpendicular to global path at closest point
        self._ppd_line = self._gp_handler.get_perpendicular(self._cls_pt, 0)

        # Tuple containing points of intersection of perpendicular with track, left and right resp.
        self._lat_lims = (
            intersection_line_cubic(self._ppd_line, self._coeffs_left),
            intersection_line_cubic(self._ppd_line, self._coeffs_right)
        )

    def generate_lat(self, n: int, pts_per_spline: int, padding: float = 0.04, bias: float = 0.5) -> PointGenerator:
        """Generate lateral splines.

        Args:
            n (int): Number of splines to generate
            padding (float): A number between 0 and 0.25, denoting the extent of padding required
                             from either side
            bias (float): Percentage of splines to left of obstacle

        Returns:
            PointGenerator: A generator that generates x and y coordinates of spline.

        """
        padding = np.clip(padding, 0, 0.25)
        bias = np.clip(bias, 0, 1)

        n_left = int(n * bias)
        n_right = n - n_left

        p1, p2 = self._lat_lims
        m = self._ppd_line.m

        # e0 is the parameter value of the foot of perpendicular from obstacle positon to
        # perpendicular to global path at closest point to obstacle
        e0 = - ((p1.x - self._obs_pose.x) + m * (p1.y - self._obs_pose.y))
        e0 /= (p2.x - p1.x) + m * (p2.y - p1.y)

        # Pad both sides, near to track
        e1 = 0 + padding
        e2 = 1 - padding

        parametric_pt = parametrise_lineseg(p1, p2)

        epsilons = np.concatenate(
            (np.linspace(e1, e0 - padding, n_left), np.linspace(e0 + padding, e2, n_right)),
            axis=0
        )
        for e in epsilons:
            p = parametric_pt(e)
            pose = Pose(p.x, p.y, yaw=self._ego_pose.yaw)
            yield cubic_spline(self._ego_pose, pose, pts_per_spline)

    def generate_long(self, n: int, pts_per_spline: int, density: float = 1, bias: float = 0.5) -> PointGenerator:
        """Generate longitudinal splines.

        Args:
            n (int): Number of splines to generate
            density (int): Number of indices in the global path to skip between splines
            bias (float): Percentage of splines to forward of obstacle

        Returns:
            PointGenerator: A generator that generates x and y coordinates of spline.

        """
        n_fwd = int(n * bias)
        n_rev = n - n_fwd

        # Use deques for constant time append and pop
        pts = deque()

        for i in range(n_rev):
            idx = (self._cls_pt - i - 1) * density
            if idx < 0:
                msg = f'Not enough points; asked for {n_rev} backward paths, clipping to {len(pts)}'
                _logger.warning(msg)
                # if not enough points, generate rest of splines forward to obstacle
                n_fwd += 1
                continue
            x, y, *_ = self._gp_handler.global_path.loc[idx]
            slope = self._gp_handler.slopes[idx]
            pts.append((PVector(x, y), slope))

        for i in range(n_fwd):
            idx = (self._cls_pt + i + 1) * density
            try:
                x, y, *_ = self._gp_handler.global_path.loc[idx]
            except KeyError:
                msg = f'Not enough points, asked for {n} paths, clipping to {len(pts)}'
                _logger.warning(msg)
                break
            else:
                slope = self._gp_handler.slopes[idx]
                pts.append((PVector(x, y), slope))

        for p, slope in pts:
            pose = Pose(p.x, p.y, yaw=np.arctan(slope))
            yield cubic_spline(self._ego_pose, pose, pts_per_spline)
