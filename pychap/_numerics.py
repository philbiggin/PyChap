"""
Small, dependency-free numerical building blocks (natural cubic spline
interpolation and a Nelder-Mead simplex optimiser), implemented with
nothing beyond NumPy.

pychap deliberately avoids a hard dependency on SciPy: the only genuine
external requirement for reading real trajectory files is MDAnalysis
(needed for GROMACS ``.xtc`` support), and it seemed unnecessary to add
SciPy on top of that just for cubic splines and a 2D optimisation that
are both easy to implement directly and easy to unit test in isolation.
"""

from __future__ import annotations

import numpy as np


class CubicSpline1D:
    """Natural cubic spline interpolation for a single scalar function y(x).

    Implements the standard textbook algorithm (e.g. Burden & Faires,
    "Numerical Analysis"): solves a tridiagonal system for the spline's
    second-derivative coefficients, subject to natural boundary
    conditions (zero second derivative at both endpoints).

    Parameters
    ----------
    x:
        Strictly increasing 1D array of node positions.
    y:
        1D array of values at each node, same length as ``x``.
    """

    def __init__(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if x.ndim != 1 or y.ndim != 1 or len(x) != len(y):
            raise ValueError("x and y must be 1D arrays of equal length")
        if len(x) < 2:
            raise ValueError("need at least 2 points")
        if np.any(np.diff(x) <= 0):
            raise ValueError("x must be strictly increasing")

        self.x = x
        self.y = y
        n = len(x) - 1
        h = np.diff(x)

        alpha = np.zeros(n + 1)
        for i in range(1, n):
            alpha[i] = (3.0 / h[i]) * (y[i + 1] - y[i]) - (3.0 / h[i - 1]) * (y[i] - y[i - 1])

        l = np.ones(n + 1)
        mu = np.zeros(n + 1)
        z = np.zeros(n + 1)
        for i in range(1, n):
            l[i] = 2.0 * (x[i + 1] - x[i - 1]) - h[i - 1] * mu[i - 1]
            mu[i] = h[i] / l[i]
            z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i]

        c = np.zeros(n + 1)
        b = np.zeros(n)
        d = np.zeros(n)
        for j in range(n - 1, -1, -1):
            c[j] = z[j] - mu[j] * c[j + 1]
            b[j] = (y[j + 1] - y[j]) / h[j] - h[j] * (c[j + 1] + 2.0 * c[j]) / 3.0
            d[j] = (c[j + 1] - c[j]) / (3.0 * h[j])

        self._h = h
        self._b = b
        self._c = c[:-1]
        self._d = d

    def _segment_index(self, t):
        # searchsorted gives, for each t, the index of the first node > t;
        # subtract 1 (clamped) to get the segment [x[i], x[i+1]] containing t.
        idx = np.searchsorted(self.x, t, side="right") - 1
        return np.clip(idx, 0, len(self.x) - 2)

    def __call__(self, t, derivative: int = 0):
        t_arr = np.atleast_1d(np.asarray(t, dtype=float))
        idx = self._segment_index(t_arr)
        dx = t_arr - self.x[idx]

        if derivative == 0:
            result = self.y[idx] + self._b[idx] * dx + self._c[idx] * dx**2 + self._d[idx] * dx**3
        elif derivative == 1:
            result = self._b[idx] + 2.0 * self._c[idx] * dx + 3.0 * self._d[idx] * dx**2
        else:
            raise ValueError("only derivative 0 or 1 is supported")

        if np.isscalar(t) or np.asarray(t).ndim == 0:
            return float(result[0])
        return result


def nelder_mead_2d(
    func,
    x0,
    initial_step: float = 1.0,
    max_iter: int = 200,
    x_tol: float = 1e-4,
    f_tol: float = 1e-5,
):
    """Minimise a 2-argument scalar function with a compact Nelder-Mead simplex.

    This is intentionally a minimal, dependency-free version of the
    classic Nelder-Mead algorithm, sufficient for the small, smooth-ish
    (if not everywhere differentiable) 2D optimisation used by
    :mod:`pychap.pathfinding` to locate the pore centre in a plane.

    Parameters
    ----------
    func:
        Callable taking a length-2 array-like and returning a scalar.
    x0:
        Initial guess, length-2 array-like.
    initial_step:
        Size of the initial simplex around ``x0``.
    max_iter:
        Maximum number of iterations.
    x_tol, f_tol:
        Convergence thresholds on simplex spread (in x and f).

    Returns
    -------
    best_x : ``np.ndarray`` of shape (2,)
    best_f : float
    """
    x0 = np.asarray(x0, dtype=float)
    assert x0.shape == (2,)

    # Build the initial simplex: x0 plus one perturbation per dimension.
    simplex = np.array(
        [
            x0,
            x0 + np.array([initial_step, 0.0]),
            x0 + np.array([0.0, initial_step]),
        ]
    )
    f_values = np.array([func(p) for p in simplex])

    alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5

    for _ in range(max_iter):
        order = np.argsort(f_values)
        simplex = simplex[order]
        f_values = f_values[order]

        if np.max(np.abs(simplex[1:] - simplex[0])) < x_tol and (
            np.max(f_values) - np.min(f_values) < f_tol
        ):
            break

        centroid = simplex[:-1].mean(axis=0)  # centroid of all but worst point

        # Reflection
        x_r = centroid + alpha * (centroid - simplex[-1])
        f_r = func(x_r)

        if f_values[0] <= f_r < f_values[-2]:
            simplex[-1], f_values[-1] = x_r, f_r
            continue

        if f_r < f_values[0]:
            # Expansion
            x_e = centroid + gamma * (x_r - centroid)
            f_e = func(x_e)
            if f_e < f_r:
                simplex[-1], f_values[-1] = x_e, f_e
            else:
                simplex[-1], f_values[-1] = x_r, f_r
            continue

        # Contraction
        x_c = centroid + rho * (simplex[-1] - centroid)
        f_c = func(x_c)
        if f_c < f_values[-1]:
            simplex[-1], f_values[-1] = x_c, f_c
            continue

        # Shrink
        for i in range(1, len(simplex)):
            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
            f_values[i] = func(simplex[i])

    order = np.argsort(f_values)
    return simplex[order[0]], float(f_values[order[0]])
