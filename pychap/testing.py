"""
Synthetic pore geometry generators.

These build simple, analytically-understood point-cloud "pores" (rings
of atoms arranged around a central axis) that are useful for unit
testing the pathfinding/radius-profile code without needing any real
molecular structure, and for building a self-contained example dataset.
They deliberately have no dependency on MDAnalysis, so they can be used
purely in-memory (as NumPy arrays) as well as to generate real
structure/trajectory files (see ``examples/generate_synthetic_pore.py``).
"""

from __future__ import annotations

import numpy as np


def ring_positions(radius: float, z: float, n_per_ring: int = 24) -> np.ndarray:
    """Coordinates of ``n_per_ring`` atoms evenly spaced on a ring of given radius at height z."""
    angles = np.linspace(0.0, 2.0 * np.pi, n_per_ring, endpoint=False)
    return np.stack(
        [radius * np.cos(angles), radius * np.sin(angles), np.full(n_per_ring, z)], axis=1
    )


def cylindrical_pore(
    radius: float = 8.0,
    n_rings: int = 21,
    n_per_ring: int = 24,
    z_min: float = -20.0,
    z_max: float = 20.0,
    vdw_radius: float = 1.0,
):
    """A straight cylindrical pore of constant radius.

    Useful for validating the pathfinding algorithm against an exact,
    analytically known answer: the resulting pore radius profile should
    equal ``radius - vdw_radius`` everywhere, and the centreline should
    lie exactly on the z axis.

    Returns
    -------
    positions : ``(N, 3)`` array
    radii : ``(N,)`` array of per-atom van der Waals radii
    """
    zs = np.linspace(z_min, z_max, n_rings)
    positions = np.concatenate([ring_positions(radius, z, n_per_ring) for z in zs], axis=0)
    radii = np.full(len(positions), vdw_radius)
    return positions, radii


def hourglass_pore(
    r_min: float = 3.0,
    r_max: float = 10.0,
    n_rings: int = 41,
    n_per_ring: int = 24,
    z_min: float = -20.0,
    z_max: float = 20.0,
    vdw_radius: float = 1.0,
):
    """An hourglass-shaped pore: narrowest at z=0, widest at z=z_min/z_max.

    The ring radius follows ``r(z) = r_min + (r_max - r_min) * (z / half_range)**2``,
    a smooth parabolic constriction centred at z=0 -- a simple stand-in
    for a channel's selectivity filter or gate.

    Returns
    -------
    positions : ``(N, 3)`` array
    radii : ``(N,)`` array of per-atom van der Waals radii
    ring_z : ``(n_rings,)`` array of the z coordinate of each ring
    ring_radius : ``(n_rings,)`` array of the (atom-centre) radius of each ring
    """
    zs = np.linspace(z_min, z_max, n_rings)
    half_range = (z_max - z_min) / 2.0
    ring_radius = r_min + (r_max - r_min) * (zs / half_range) ** 2
    positions = np.concatenate(
        [ring_positions(r, z, n_per_ring) for r, z in zip(ring_radius, zs)], axis=0
    )
    radii = np.full(len(positions), vdw_radius)
    return positions, radii, zs, ring_radius


def hourglass_pore_residues(ring_z: np.ndarray, ring_radius: np.ndarray, hydrophobic_half_width: float = 6.0):
    """Representative per-ring "residue" positions and names for the hourglass pore.

    Assigns a hydrophobic residue (leucine) to rings near the
    constriction (``|z| <= hydrophobic_half_width``) and a hydrophilic
    residue (aspartate) further away, mimicking a hydrophobic gate
    flanked by polar vestibules -- a common real-world channel motif --
    so that :func:`pychap.hydrophobicity.hydrophobicity_profile` has
    something structured to recover in tests/examples.

    Returns
    -------
    positions : ``(n_rings, 3)`` array (one point per ring, on the axis at that ring's z)
    resnames : list of str, length n_rings
    """
    positions = np.stack([np.zeros_like(ring_z), np.zeros_like(ring_z), ring_z], axis=1)
    resnames = [
        "LEU" if abs(z) <= hydrophobic_half_width else "ASP" for z in ring_z
    ]
    return positions, resnames
