"""Core logic and implementations."""

import logging
import numpy as np

from .types import Line_SI, PVector, Pose
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
        msg = 'Matrix in singular, proceeding with pseudo-inverse instead'
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
        _logger.warning("Slope calculation returning `inf`")
    else:
        slope = (point2.y - point1.y) / (point2.x - point1.x)
    return slope


def intersection_line_cubic(line: Line_SI, coeffs: RoadLinePolynom):
    """Find point of intersection of line and cubic.

    Args:We can do a polyfit on global path and find slope

        line (Line_SI): Straight line in slope-intercept form
        coeffs (RoadLinePolynom): Structure holding cubic approximation
                                  of track boundaries.

    Returns:
        PVector: point of intersection

    """
    try:
        # Line is of the form y = m * x + c
        # Cubic is of the form y = c3 * x**3 + c2 * x**2 + c1 * x + c0
        #
        # To solve, find roots of cubic:
        # y = c3 * x**3 + c2 * x**2 + (c1 - m) * x + (c0 - c)
        xs = np.roots([
            coeffs.c3,
            coeffs.c2,
            coeffs.c1 - line.m,
            coeffs.c0 - line.c
        ])
    except Exception:
        msg = 'Error in computing intersection - Line and Cubic'
        _logger.critical(msg, exc_info=True)
        # TODO: add recovery routines/logic
        xs = np.empty()

    resx = 0.0
    # Ouf of obtained roots, one is a pure real number, rest two are
    # complex conjugates (or all three real roots, rarely in our case)

    # TODO: add logic for three real roots
    for x in xs:
        if x.imag == 0:
            resx = x.real
            break

    return PVector(resx, line.m * resx + line.c)


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
        y += c * x ** i
    return y


def gen_spline(ego_pose: Pose, target: PVector):
    """Generate coordinates of spline joining ego vehicle and target point.

    Args:
        ego_pose (Pose): Ego vehicle pose
        target (PVector): Target point to reach (w.r.t global frame)

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray]: Array of x and y points on spline
                                             w.r.t global frame

    """
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
