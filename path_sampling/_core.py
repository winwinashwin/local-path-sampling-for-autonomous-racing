"""Core logic and implementations."""

import logging
import numpy as np

from .types import Line_SI, PVector
from .types import RoadLinePolynom  # TODO: Remove in production

_logger = logging.getLogger(__name__)


def linalg_solve(a: np.ndarray, b: np.ndarray):
    """Solve linear matrix equation of the form AX = B.

    Args:
        a (numpy.ndarray): Matrix A
        b (numpy.ndarray): Matrix B

    Returns:
        numpy.ndarray: solution X

    """
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

        x = np.linalg.solve(a, b)

    except np.linalg.LinAlgError:
        msg = 'Matrix is singular, proceeding with pseudo-inverse instead'
        _logger.warning(msg)

        a_inv = np.linalg.pinv(a)
        x = a_inv.dot(b)
    finally:
        return x


def slope_of_segment(point1: PVector, point2: PVector):
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


def intersection_line_cubic(line: Line_SI, coeffs: RoadLinePolynom):
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
        # TODO: add logic for three real roots
        _logger.warning('Got three real roots')
        root = roots[real_roots[0][0]].real

    return PVector(root, line.m * root + line.c)


def parametrise_lineseg(p1: PVector, p2: PVector, padding: float = 0.0):
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
    padding = np.clip(padding, 0, 0.5)
    e1, e2 = padding, 1 - padding
    p1_padded = PVector(p1.x + e1 * (p2.x - p1.x), p1.y + e1 * (p2.y - p1.y))
    p2_padded = PVector(p1.x + e2 * (p2.x - p1.x), p1.y + e2 * (p2.y - p1.y))

    def parametric_point(epsilon):
        epsilon = np.clip(epsilon, 0, 1)

        x = p1_padded.x + epsilon * (p2_padded.x - p1_padded.x)
        y = p1_padded.y + epsilon * (p2_padded.y - p1_padded.y)

        return PVector(x, y)

    # Return an inner function to generate points as needed
    return parametric_point


def polyeval(x, coeffs):
    """Evaluate a polynomial at x.

    Args:
        x (float): Point x to evaluate at
        coeffs (numpy.ndarray): coefficients of the polynomial

    Returns:
        float: Value of polynomial at x.

    """
    y = 0.0
    for i, c in enumerate(coeffs):
        y += c * (x ** i)
    return y
