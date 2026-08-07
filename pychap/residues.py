"""
Mapping residues onto the permeation pathway.

CHAP reports, for every pore-lining residue, its mean position along the
pathway (arc-length coordinate ``s``) and its mean radial distance from
the centreline (``rho``), plus the fraction of frames in which it counts
as "pore-facing" -- see the ``residueSummary`` section of CHAP's JSON
output and https://www.channotation.org/docs/plotting_python/, which
overlays these residues (coloured by hydrophobicity) on the radius
profile plot.

This module reimplements that projection: for a residue at position
``p``, we find the nearest point on the (already densely resampled)
pathway centreline, and report its arc-length coordinate and the
distance to it. A residue is considered "pore-facing" in a given frame
if that distance is below a cutoff (by default 12 Angstrom, loosely
representative of a generous vestibule radius) -- a simpler criterion
than upstream CHAP's solid-angle-based test, but capturing the same
basic idea: residues close to the pathway are pore-facing, residues far
from it are not.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def project_onto_path(positions: np.ndarray, path_points: np.ndarray, path_s: np.ndarray):
    """Project each position onto the nearest point of a discretised pathway.

    Parameters
    ----------
    positions:
        ``(R, 3)`` array of positions to project (e.g. residue centres).
    path_points:
        ``(M, 3)`` array of pathway centreline points.
    path_s:
        ``(M,)`` array of arc-length coordinates corresponding to ``path_points``.

    Returns
    -------
    s : ``(R,)`` array -- arc-length coordinate of the nearest pathway point
    rho : ``(R,)`` array -- Euclidean distance from each position to that point
    """
    positions = np.atleast_2d(np.asarray(positions, dtype=float))
    path_points = np.asarray(path_points, dtype=float)
    path_s = np.asarray(path_s, dtype=float)

    if len(path_points) == 0:
        raise ValueError("path_points must not be empty")

    diff = positions[:, None, :] - path_points[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    idx = np.argmin(dist, axis=1)
    return path_s[idx], dist[np.arange(len(positions)), idx]


@dataclass
class ResidueFrameProjection:
    """Per-frame projection of a set of residues onto the pathway."""

    s: np.ndarray     #: arc-length coordinate of each residue, shape (R,)
    rho: np.ndarray   #: radial distance of each residue from the centreline, shape (R,)
    pore_facing: np.ndarray  #: bool array, shape (R,)


def project_residues(res_positions: np.ndarray, profile, pore_facing_cutoff: float = 12.0) -> ResidueFrameProjection:
    """Project residue positions onto a single frame's :class:`~pychap.pathfinding.PoreProfile`."""
    s, rho = project_onto_path(res_positions, profile.points, profile.s)
    pore_facing = rho <= pore_facing_cutoff
    return ResidueFrameProjection(s=s, rho=rho, pore_facing=pore_facing)
