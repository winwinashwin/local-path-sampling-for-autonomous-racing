"""Core logic and implementations."""

import logging
from typing import Callable, Iterable, Tuple

import numpy as np

from .types import Line_SI, PVector, Pose
from .types import RoadLinePolynom  # TODO: Remove in production

_logger = logging.getLogger(__name__)

# Type aliases
ParametricLine = Callable[[float], PVector]


def slope_of_segment(point1: PVector, point2: PVector) -> float:
    """Find slope of line segment joining two points.

    Args:
        point1 (PVector): First point
        point2 (PVector): Second point

    Returns:
        float: slope of line joining point1 and point2

    """
    if point1.x == point2.x:
        slope = float('inf')
        _logger.warning('Slope calculation returning `inf`')
    else:
        slope = (point2.y - point1.y) / (point2.x - point1.x)
    return slope


def intersection_line_cubic(line: Line_SI, coeffs: RoadLinePolynom) -> PVector:
    """Find point of intersection of line and cubic.

    Args:
        line (Line_SI): Straight line in slope-intercept form
        coeffs (RoadLinePolynom): Structure holding cubic approximation
                                  of track boundaries.

    Returns:
        PVector: point of intersection

    """
    # Line is of the form y = m * x + c
    # Cubic is of the form y = c3 * x**3 + c2 * x**2 + c1 * x + c0
    #
    # To solve, find roots of cubic:
    # y = c3 * x**3 + c2 * x**2 + (c1 - m) * x + (c0 - c)
    fx = [coeffs.c3, coeffs.c2, coeffs.c1 - line.m, coeffs.c0 - line.c]
    roots = np.roots(fx)

    real_roots = np.where(np.isreal(roots))
    if len(real_roots) == 1:
        # One real root and two complex roots (conjugates)
        root = roots[real_roots[0][0]].real
    else:
        # TODO: add routine for handing three real roots
        _logger.warning('Got three real roots')
        root = roots[real_roots[0][0]].real

    return PVector(root, line.m * root + line.c)


def parametrise_lineseg(p1: PVector, p2: PVector, padding: float = 0.0) -> ParametricLine:
    """Parametrise all points in a line segment joining points p1 and p2.

    Args:
        p1 (PVector): First point
        p2 (PVector): Second point
        padding (float): Value between 0 and 0.5 denoting the extend of
                         padding required on either side

    Returns:
        Callable[[float], PVector]: A function that gives points based on the
                                    parameter provided.

    """
    # All points on the line segment joining p1 and p2 can be parametrised with a
    # parameter epsilon, such that as epsilon varies between 0 and 1, point p
    # varies between p1 and p2.

    # It is evident that paths generated with epsilon = 0 and epsilon = 1 are infeasible.
    # Hence we introduce some padding on both sides
    padding = np.clip(padding, 0, 0.25)
    e1, e2 = padding, 1 - padding
    p1_padded = PVector(p1.x + e1 * (p2.x - p1.x), p1.y + e1 * (p2.y - p1.y))
    p2_padded = PVector(p1.x + e2 * (p2.x - p1.x), p1.y + e2 * (p2.y - p1.y))

    def parametric_point(epsilon: float) -> PVector:
        epsilon = np.clip(epsilon, 0, 1)

        x = p1_padded.x + epsilon * (p2_padded.x - p1_padded.x)
        y = p1_padded.y + epsilon * (p2_padded.y - p1_padded.y)

        return PVector(x, y)

    # Return an inner function to generate points as needed
    return parametric_point


def polyeval(x: float, coeffs: Iterable[float]) -> float:
    """Evaluate a polynomial at x.

    Args:
        x (float): Point x to evaluate at
        coeffs (Iterable[float]): coefficients of the polynomial

    Returns:
        float: Value of polynomial at x.

    """
    y = 0.0
    for i, c in enumerate(coeffs):
        y += c * (x ** i)
    return y


def cubic_spline(pose1: Pose, pose2: Pose, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Cubic spline through two points.

    Args:
        pose1 (Pose): First point (with heading info, hence Pose and not PVector)
        pose2 (Pose): Second point (with heading info, hence Pose and not PVector)
        n (int): Number of points in spline

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray]: Tuple of x and y coordinates in the spline

    To generate cubic spline between two points (x1, y1) and (x2, y2) with heading
    theta1 and theta2 respectively, we use the following 4 conditions.

    To simplify compuation, we shift the origin to (x1, y1) and rotate the axes anticlockwise
    by theta1. Spline in new system is G and new target is (tx, ty).

    1. G(0) = 0
    2. G(tx) = ty
    3. G'(0) = 0
    4. G'(tx) = tan(theta2 - theta1)

    This gives us a linear matrix eqn AX = B with coefficient matrix as unknown

    """
    # Translation
    shift_x = pose2.x - pose1.x
    shift_y = pose2.y - pose1.y

    # Rotation
    tx = shift_x * np.cos(pose1.yaw) - shift_y * np.sin(pose1.yaw)
    ty = shift_x * np.sin(pose1.yaw) + shift_y * np.cos(pose1.yaw)

    a = np.array([
        [0, 0, 0, 1],
        [tx ** 3, tx ** 2, tx, 1],
        [0, 0, 1, 0],
        [3 * (tx ** 2), 2 * tx, 1, 0]
    ])
    b = np.array([0, ty, 0, np.tan(pose2.yaw - pose1.yaw)])

    # Solve for coefficients
    try:
        # Try to solve normally. If A is singular, proceed to find
        # pseudo inverse of A.
        #
        # np.linalg.solve(A, B) does not compute the inverse of A.
        # Instead it calls one of the gesv LAPACK routines, which
        # first factorizes A using LU decomposition, then solves
        # for x using forward and backward substitution
        #
        # Finding inverse(A) would incur yet more floating point
        # operations, and therefore slower performance and more
        # numerical error.

        coeffs = np.linalg.solve(a, b)

    except np.linalg.LinAlgError:
        msg = 'Matrix is singular, proceeding with pseudo-inverse instead'
        _logger.warning(msg)

        a_inv = np.linalg.pinv(a)
        coeffs = a_inv.dot(b)

    # Generate x coordinates in current frame
    xs = np.linspace(0, tx, n)
    # Calculate y for each corresponding x
    ys = np.array([polyeval(x, coeffs[::-1]) for x in xs])

    # Tranform all points (x, y) in spline back to global frame
    for i, (x, y) in enumerate(zip(xs, ys)):
        xs[i] = x * np.cos(pose1.yaw) + y * np.sin(pose1.yaw) + pose1.x
        ys[i] = y * np.cos(pose1.yaw) - x * np.sin(pose1.yaw) + pose1.y

    return xs, ys
