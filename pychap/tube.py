"""
3D tube mesh construction around a pathway centreline.

Given a centreline (a sequence of 3D points with an associated
arc-length coordinate) and a radius at each point, this builds a ring of
vertices around each centreline point -- the same shape CHAP itself
renders as an OBJ surface mesh for VMD/PyMOL visualisation. This module
is shared by the (dependency-free, matplotlib-only) 3D plot in
``scripts/plot_pore_3d.py`` and by :mod:`pychap.obj_export`, which
writes the same geometry out as a Wavefront OBJ mesh for use in VMD.
"""

from __future__ import annotations

import numpy as np


def perpendicular_frames(tangents: np.ndarray):
    """Build a smoothly-varying (parallel-transported) orthonormal frame
    (e1, e2) perpendicular to each tangent vector, so that consecutive
    cross-sectional rings don't twist erratically around the tube."""
    n = len(tangents)
    e1 = np.zeros_like(tangents)

    ref = np.array([1.0, 0.0, 0.0]) if abs(tangents[0][0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    v = np.cross(tangents[0], ref)
    e1[0] = v / np.linalg.norm(v)

    for i in range(1, n):
        proj = e1[i - 1] - np.dot(e1[i - 1], tangents[i]) * tangents[i]
        norm = np.linalg.norm(proj)
        if norm < 1e-8:
            ref = np.array([1.0, 0.0, 0.0]) if abs(tangents[i][0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            proj = np.cross(tangents[i], ref)
            norm = np.linalg.norm(proj)
        e1[i] = proj / norm

    e2 = np.cross(tangents, e1)
    return e1, e2


def build_tube_mesh(points: np.ndarray, radius: np.ndarray, s: np.ndarray, n_theta: int = 24) -> np.ndarray:
    """Compute ring vertices around a centreline, sized by local radius.

    Parameters
    ----------
    points:
        ``(N, 3)`` centreline points.
    radius:
        ``(N,)`` radius at each point.
    s:
        ``(N,)`` arc-length coordinate at each point (used only to get a
        well-scaled tangent estimate via finite differences).
    n_theta:
        Number of vertices around each cross-sectional ring.

    Returns
    -------
    ``(N, n_theta, 3)`` array of ring vertices.
    """
    points = np.asarray(points, dtype=float)
    radius = np.asarray(radius, dtype=float)
    s = np.asarray(s, dtype=float)

    if len(points) < 2:
        raise ValueError("need at least 2 points to build a tube")

    tangents = np.gradient(points, s, axis=0)
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    tangents = tangents / norms

    e1, e2 = perpendicular_frames(tangents)

    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=True)
    cos_t = np.cos(theta)[None, :, None]
    sin_t = np.sin(theta)[None, :, None]

    rings = (
        points[:, None, :]
        + radius[:, None, None] * cos_t * e1[:, None, :]
        + radius[:, None, None] * sin_t * e2[:, None, :]
    )
    return rings  # shape (n_points, n_theta, 3)


def tube_faces(n_points: int, n_theta: int):
    """Yield ``(quad_index_i, [v00, v10, v11, v01])`` 0-based vertex index
    quads connecting consecutive rings of a tube mesh built by
    :func:`build_tube_mesh` (vertices flattened as ``i * n_theta + j``).
    Each quad should be split into two triangles by the caller."""
    for i in range(n_points - 1):
        for j in range(n_theta):
            j2 = (j + 1) % n_theta
            v00 = i * n_theta + j
            v01 = i * n_theta + j2
            v10 = (i + 1) * n_theta + j
            v11 = (i + 1) * n_theta + j2
            yield i, (v00, v10, v11, v01)
