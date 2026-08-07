"""
Arc-length parameterised 3D path spline.

CHAP fits a smooth curve through a discrete set of pathway points and
then reports properties (radius, hydrophobicity, ...) as a function of
arc length ``s`` along that curve. This module provides a small,
dependency-free equivalent (natural cubic splines, one per Cartesian
coordinate, all sharing an arc-length parameterisation) built on top of
:class:`pychap._numerics.CubicSpline1D`.
"""

from __future__ import annotations

import numpy as np

from ._numerics import CubicSpline1D


class PathSpline:
    """A smooth 3D curve through an ordered sequence of points.

    Parameters
    ----------
    points:
        ``(N, 3)`` array of points, ordered along the path. Consecutive
        duplicate points are not permitted (they would produce a
        zero-length segment).
    """

    def __init__(self, points):
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points must be an (N, 3) array")
        if len(points) < 2:
            raise ValueError("need at least 2 points to build a path spline")

        seg_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        if np.any(seg_lengths <= 0):
            raise ValueError("path points must be strictly distinct")

        arc_length = np.concatenate([[0.0], np.cumsum(seg_lengths)])
        self._s = arc_length
        self._length = float(arc_length[-1])
        self._splines = [CubicSpline1D(arc_length, points[:, dim]) for dim in range(3)]

    @property
    def length(self) -> float:
        """Total arc length of the path."""
        return self._length

    @property
    def knot_arc_lengths(self) -> np.ndarray:
        """Arc-length coordinate of each original input point."""
        return self._s.copy()

    def __call__(self, s):
        """Evaluate the path position(s) at arc length coordinate(s) ``s``."""
        s_arr = np.clip(np.asarray(s, dtype=float), 0.0, self._length)
        scalar_input = s_arr.ndim == 0
        s_arr = np.atleast_1d(s_arr)
        coords = np.stack([spl(s_arr) for spl in self._splines], axis=-1)
        return coords[0] if scalar_input else coords

    def tangent(self, s):
        """Unit tangent vector(s) of the path at arc length coordinate(s) ``s``."""
        s_arr = np.clip(np.asarray(s, dtype=float), 0.0, self._length)
        scalar_input = s_arr.ndim == 0
        s_arr = np.atleast_1d(s_arr)
        d = np.stack([spl(s_arr, derivative=1) for spl in self._splines], axis=-1)
        norms = np.linalg.norm(d, axis=-1, keepdims=True)
        norms[norms == 0] = 1.0
        unit = d / norms
        return unit[0] if scalar_input else unit

    def resample(self, n: int = 200):
        """Return ``n`` evenly spaced (in arc length) samples along the path.

        Returns
        -------
        s : ``(n,)`` array of arc-length coordinates
        points : ``(n, 3)`` array of path positions
        """
        s = np.linspace(0.0, self._length, n)
        return s, self(s)
