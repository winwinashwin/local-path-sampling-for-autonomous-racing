"""Handle the global path."""

import logging
from collections import deque
from typing import Iterable, NoReturn

import numpy as np
import pandas as pd

from ._core import slope_of_segment
from .types import Line_SI, PVector, Pose

_logger = logging.getLogger(__name__)


class GlobalPathHandler(object):
    """Handler for interacting with the global path data."""

    def __init__(self):
        """Constructor."""
        self._gp_df = None
        self._n = None
        # using deques for constant time array access
        self._slopes = deque()
        self._loaded = False

    def load_from_csv(self, file: str) -> NoReturn:
        """Load global path from a csv file.

        Args:
            file (str): Path to csv file

        Returns:
            None

        """
        self._gp_df = pd.read_csv(file)
        self._loaded = True
        self._n = self._gp_df.shape[0]
        self._calculate_slopes()

        _logger.debug(f'Loaded global path from: {file}')

    @property
    def global_path(self) -> pd.core.frame.DataFrame:
        """Global path data as loaded from file."""
        return self._gp_df

    @property
    def slopes(self) -> Iterable[float]:
        """Slopes at each point in the global path."""
        return self._slopes

    def _calculate_slopes(self) -> NoReturn:
        """Calculate slopes at each point in global path.

        Args:
            None

        Returns:
            None

        """
        assert self._loaded

        for i in range(self._n - 1):
            x1, y1, *_ = self._gp_df.loc[i]
            x2, y2, *_ = self._gp_df.loc[i + 1]
            slope = slope_of_segment(PVector(x1, y1), PVector(x2, y2))
            self._slopes.append(slope)
        self._slopes.append(self._slopes[-1])

    def get_closest_point(self, ego_pose: Pose) -> int:
        """Find closet point to ego position in global path.

        Args:
            ego_pose (Pose): Ego vehicle pose

        Returns:
            int: index of the point in global path closest to ego position

        """
        assert self._loaded

        min_idx = 0
        min_dist = float('inf')

        for i in range(min_idx, self._n):
            gx, gy, *_ = self._gp_df.loc[i]
            dist = np.sqrt((ego_pose.x - gx)**2 + (ego_pose.y - gy)**2)

            if dist < min_dist:
                min_dist = dist
                min_idx = i
            if dist > min_dist:
                break
        return min_idx

    def get_perpendicular(self, closest_pt_idx: int, look_ahead: int = 10) -> Line_SI:
        """Find perpendicular to global path at a point.

        Args:
            closest_pt_idx (int): Index of point in global path closest to ego pose.
            look_ahead (int): Number of indices to look ahead

        Returns:
            Line_SI: Perpendicular and lookahead index to global path, in slope-intercept
                     form.

        """
        assert self._loaded

        idx = closest_pt_idx + look_ahead
        m = -1 / self._slopes[idx]
        c = self._gp_df.loc[idx][1] - m * self._gp_df.loc[idx][0]
        return Line_SI(m, c)
