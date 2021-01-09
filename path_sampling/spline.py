import numpy as np
from collections import deque
import logging
from typing import Generator, Tuple
from ._core import (
    cubic_spline,
    intersection_line_cubic,
    parametrise_lineseg,
)
from .types import PVector, Pose, RoadLinePolynom
from .global_path_handler import GlobalPathHandler

_logger = logging.getLogger(__name__)


class SplineGenerator(object):

    def __init__(self,
                 gp_handler: GlobalPathHandler,
                 ego_pose: Pose,
                 obs_pose: Pose,
                 road_poly_left: RoadLinePolynom,
                 road_poly_right: RoadLinePolynom
                 ):
        self._gp_handler = gp_handler
        self._ego_pose = ego_pose
        self._obs_pose = obs_pose

        self._coeffs_left = road_poly_left
        self._coeffs_right = road_poly_right

        self._cls_pt = self._gp_handler.get_closest_point(self._obs_pose)
        self._ppd_line = self._gp_handler.get_perpendicular(self._cls_pt, 0)

        self._lat_lims = (
            intersection_line_cubic(self._ppd_line, self._coeffs_left),
            intersection_line_cubic(self._ppd_line, self._coeffs_right)
        )

    def generate_lat(self, n: int, padding: float = 0.04, bias: float = 0.5) -> Generator[Tuple[np.ndarray, np.ndarray]]:
        padding = np.clip(padding, 0, 0.25)
        bias = np.clip(bias, 0, 1)

        n_left = int(n * bias)
        n_right = n - n_left

        p1, p2 = self._lat_lims
        m = self._ppd_line.m

        e0 = - ((p1.x - self._obs_pose.x) + m * (p1.y - self._obs_pose.y)) / \
            ((p2.x - p1.x) + m * (p2.y - p1.y))
        e1 = 0 + padding
        e2 = 1 - padding

        p0 = PVector(p1.x + e0 * (p2.x - p1.x), p1.y + e0 * (p2.y - p1.y))

        parametric_pt = parametrise_lineseg(p1, p2)

        epsilons = np.concatenate(
            (np.linspace(e1, e0 - padding, n_left), np.linspace(e0 + padding, e2, n_right)),
            axis=0
        )
        for e in epsilons:
            p = parametric_pt(e)
            pose = Pose(p.x, p.y, yaw=self._ego_pose.yaw)
            yield cubic_spline(self._ego_pose, pose)

    def generate_long(self, n: int, density: float = 1, bias: float = 0.5) -> Generator[Tuple[np.ndarray, np.ndarray]]:
        n_fwd = int(n * bias)
        n_rev = n - n_fwd

        pts = deque()

        for i in range(n_rev):
            idx = (self._cls_pt - i - 1) * density
            if idx < 0:
                n_fwd += 1
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
            yield cubic_spline(self._ego_pose, pose)
