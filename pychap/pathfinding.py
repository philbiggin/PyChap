"""
Pore centreline search and radius profile computation.

This is a simplified, Python-native reimplementation of CHAP's pathway
module. The original C++ tool uses a probabilistic search (informed by
simulated annealing) to find a permeation pathway that maximises the
minimum distance to the van der Waals surface, followed by an
optimisation-based refinement. Here we take a more direct approach that
is easier to reason about and test, while capturing the same physical
idea:

1. Choose an "axis" coordinate (e.g. the membrane normal, typically z).
   The pathway is assumed to be reasonably well described as a sequence
   of centre points, one per axis slice, similar to HOLE's algorithm.
2. At each axis slice, find the ``(u, v)`` position in the perpendicular
   plane that *maximises* the minimum distance to the van der Waals
   surface of nearby atoms -- i.e. the largest sphere that fits at that
   point without overlapping any atom. This is a small 2D optimisation
   problem solved with Nelder-Mead, warm-started from the previous
   slice's solution so the centreline varies smoothly.
3. A smooth 3D spline is fit through the resulting centreline points
   (see :mod:`pychap.spline`) and resampled at a finer, evenly-spaced
   (in arc length) resolution to produce the final pore radius profile.

Caveat vs. upstream CHAP: step 2 searches in planes perpendicular to the
chosen Cartesian axis rather than perpendicular to the local path
tangent. This is an approximation that is accurate for pores that are
roughly aligned with that axis and do not curve sharply -- true for the
large majority of ion channel pores, which are approximately normal to
the membrane -- but will be less accurate for pathways with strong
curvature. See README.md for more detail.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._numerics import nelder_mead_2d
from .spline import PathSpline


def _min_surface_distance(point: np.ndarray, positions: np.ndarray, radii: np.ndarray) -> float:
    """Minimum distance from ``point`` to the van der Waals surface of any atom."""
    d = np.linalg.norm(positions - point, axis=1) - radii
    return float(d.min())


def _neg_radius_objective(uv, axis_value, axis, positions, radii):
    point = np.empty(3)
    other = [i for i in range(3) if i != axis]
    point[axis] = axis_value
    point[other[0]] = uv[0]
    point[other[1]] = uv[1]
    return -_min_surface_distance(point, positions, radii)


@dataclass
class PoreProfile:
    """Result of a single-frame pore pathway / radius profile calculation."""

    s: np.ndarray                 #: resampled arc-length coordinate, shape (n_resample,)
    points: np.ndarray            #: resampled centreline points, shape (n_resample, 3)
    radius: np.ndarray            #: pore radius at each resampled point, shape (n_resample,)
    axis: int                     #: which Cartesian axis (0/1/2) was used as the pathway axis
    axis_values: np.ndarray       #: raw per-slice axis coordinate values used during search
    centreline: np.ndarray        #: raw (coarse) centreline points from the slice search
    centreline_radius: np.ndarray  #: raw (coarse) radius profile from the slice search
    length: float                 #: total arc length of the fitted path

    @property
    def min_radius(self) -> float:
        return float(np.min(self.radius))

    @property
    def min_radius_position(self) -> float:
        return float(self.s[np.argmin(self.radius)])

    @property
    def last_uv(self):
        """(u, v) offset of the final slice, useful as a warm start for the next frame."""
        other = [i for i in range(3) if i != self.axis]
        last = self.centreline[-1]
        return float(last[other[0]]), float(last[other[1]])


def find_centreline(
    positions: np.ndarray,
    radii: np.ndarray,
    axis: int = 2,
    n_slices: int = 50,
    axis_range=None,
    window: float = 15.0,
    seed_uv=(0.0, 0.0),
):
    """Find a coarse pore centreline by slicing along ``axis``.

    Parameters
    ----------
    positions:
        ``(N, 3)`` array of pore-lining atom coordinates (Angstrom).
    radii:
        ``(N,)`` array of van der Waals radii for those atoms (Angstrom).
    axis:
        Index (0, 1, or 2) of the Cartesian axis to march the search
        along -- typically 2 (z), i.e. the membrane normal.
    n_slices:
        Number of slices (search points) along the axis.
    axis_range:
        Optional ``(lo, hi)`` tuple restricting the axis range searched.
        Defaults to the full extent of ``positions`` along ``axis``.
    window:
        Only atoms within this distance (along ``axis``) of the current
        slice are considered in the local optimisation, for speed. Must
        be large enough to include the atoms that actually bound the
        pore at that slice.
    seed_uv:
        Initial guess for the pore centre in the plane perpendicular to
        ``axis``, used at the first slice (and refined slice-by-slice
        thereafter).

    Returns
    -------
    axis_values : ``(n_slices,)`` array
    centreline : ``(n_slices, 3)`` array
    radius_profile : ``(n_slices,)`` array
    """
    positions = np.asarray(positions, dtype=float)
    radii = np.asarray(radii, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must be an (N, 3) array")
    if len(positions) != len(radii):
        raise ValueError("positions and radii must have the same length")
    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1, or 2")
    if len(positions) == 0:
        raise ValueError("positions must not be empty")

    if axis_range is None:
        lo, hi = float(positions[:, axis].min()), float(positions[:, axis].max())
    else:
        lo, hi = axis_range

    axis_values = np.linspace(lo, hi, n_slices)
    other = [i for i in range(3) if i != axis]

    uv = np.array(seed_uv, dtype=float)
    centreline = np.zeros((n_slices, 3))
    radius_profile = np.zeros(n_slices)

    for i, s in enumerate(axis_values):
        mask = np.abs(positions[:, axis] - s) <= window
        if not np.any(mask):
            # Widen the search if the window happens to miss everything
            # (e.g. very sparse slice), rather than crashing.
            mask = np.ones(len(positions), dtype=bool)
        pos_local = positions[mask]
        rad_local = radii[mask]

        def objective(candidate_uv, _s=s, _pos=pos_local, _rad=rad_local):
            return _neg_radius_objective(candidate_uv, _s, axis, _pos, _rad)

        uv, neg_radius = nelder_mead_2d(objective, uv, initial_step=1.5, max_iter=200)
        point = np.empty(3)
        point[axis] = s
        point[other[0]] = uv[0]
        point[other[1]] = uv[1]
        centreline[i] = point
        radius_profile[i] = -neg_radius

    return axis_values, centreline, radius_profile


def compute_pore_profile(
    positions: np.ndarray,
    radii: np.ndarray,
    axis: int = 2,
    n_slices: int = 50,
    n_resample: int = 200,
    window: float = 15.0,
    seed_uv=(0.0, 0.0),
) -> PoreProfile:
    """Find the pore centreline and compute a smooth radius profile along it.

    This combines :func:`find_centreline` (coarse search) with a
    :class:`~pychap.spline.PathSpline` fit (smoothing/resampling), and
    recomputes the true minimum-surface-distance radius at each
    resampled point (rather than merely interpolating the coarse radius
    values), so the final profile reflects genuine van der Waals
    surface distances everywhere.
    """
    positions = np.asarray(positions, dtype=float)
    radii = np.asarray(radii, dtype=float)

    axis_values, centreline, coarse_radius = find_centreline(
        positions, radii, axis=axis, n_slices=n_slices, window=window, seed_uv=seed_uv
    )

    spline = PathSpline(centreline)
    s, points = spline.resample(n_resample)

    radius = np.empty(n_resample)
    for i, p in enumerate(points):
        window_mask = np.abs(positions[:, axis] - p[axis]) <= window
        if not np.any(window_mask):
            window_mask = np.ones(len(positions), dtype=bool)
        radius[i] = _min_surface_distance(p, positions[window_mask], radii[window_mask])

    return PoreProfile(
        s=s,
        points=points,
        radius=radius,
        axis=axis,
        axis_values=axis_values,
        centreline=centreline,
        centreline_radius=coarse_radius,
        length=spline.length,
    )
