"""Structs and Definitions used."""
from typing import NamedTuple


class Line_SI(NamedTuple):
    """Slope-intercept form of a line."""

    m: float
    c: float


class PVector(NamedTuple):
    """Point in space."""

    x: float
    y: float
    z: float = 0.0


class Pose(NamedTuple):
    """Ego vehicle position."""

    x: float
    y: float
    z: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0


class RoadLinePolynom(NamedTuple):
    """Sample structure according to Network.html.

    TODO: remove in production
    """

    c0: float
    c1: float
    c2: float
    c3: float
    lineId: int = 0
    curvatureRadius: float = 0.0
    estimatedCurvatureRadius: float = 0.0
