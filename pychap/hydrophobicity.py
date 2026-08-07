"""
Residue hydrophobicity mapping onto the permeation pathway.

CHAP annotates its pore pathway with residue-level physicochemical
properties by attributing each residue's property value to nearby points
on the pathway. We reproduce that idea with a Gaussian-kernel weighted
average: a residue whose (representative) position is close to a given
pathway point contributes strongly to that point's value; distant
residues contribute little.

The default property is the Wimley-White whole-residue interfacial
hydrophobicity scale (kcal/mol; White & Wimley, Annu. Rev. Biophys.
Biomol. Struct. 1999), which is one of the scales supported by upstream
CHAP. More positive values are more favourable to a residue partitioning
into water (hydrophilic); more negative values indicate a more
hydrophobic residue.
"""

from __future__ import annotations

import numpy as np

#: Wimley-White whole-residue interfacial hydrophobicity scale (kcal/mol).
WIMLEY_WHITE_INTERFACE = {
    "ALA": 0.17,
    "ARG": 0.81,
    "ASN": 0.42,
    "ASP": 1.23,
    "CYS": -0.24,
    "GLN": 0.58,
    "GLU": 2.02,
    "GLY": 0.01,
    "HIS": 0.17,
    "HSD": 0.17,
    "HSE": 0.17,
    "HSP": 0.17,
    "ILE": -0.31,
    "LEU": -0.56,
    "LYS": 0.99,
    "MET": -0.23,
    "PHE": -1.13,
    "PRO": 0.45,
    "SER": 0.13,
    "THR": 0.14,
    "TRP": -1.85,
    "TYR": -0.94,
    "VAL": 0.07,
}

#: Neutral fallback for unrecognised residues (e.g. ligands, waters, ions).
DEFAULT_HYDROPHOBICITY = 0.0


def residue_hydrophobicity(resname: str, scale: dict | None = None) -> float:
    """Look up the hydrophobicity value for a residue name.

    Parameters
    ----------
    resname:
        Residue name (e.g. ``"LEU"``), case-insensitive.
    scale:
        Optional custom scale dictionary keyed by upper-case 3-letter
        residue name; defaults to :data:`WIMLEY_WHITE_INTERFACE`.
    """
    scale = scale or WIMLEY_WHITE_INTERFACE
    return scale.get(resname.strip().upper(), DEFAULT_HYDROPHOBICITY)


def hydrophobicity_profile(
    path_points: np.ndarray,
    residue_positions: np.ndarray,
    residue_values: np.ndarray,
    sigma: float = 5.0,
) -> np.ndarray:
    """Map residue-level hydrophobicity values onto pathway points.

    For each point on the pathway, computes a Gaussian-kernel weighted
    average of ``residue_values`` over all residues, weighted by their
    3D distance from that point (so nearby pore-lining residues dominate
    the value at each point).

    Parameters
    ----------
    path_points:
        ``(M, 3)`` array of pathway points.
    residue_positions:
        ``(R, 3)`` array of representative residue coordinates.
    residue_values:
        ``(R,)`` array of per-residue hydrophobicity values.
    sigma:
        Width (Angstrom) of the Gaussian distance kernel. Larger values
        smooth/broaden the influence of each residue along the pathway.

    Returns
    -------
    ``(M,)`` array of hydrophobicity values, one per pathway point.
    """
    path_points = np.atleast_2d(np.asarray(path_points, dtype=float))
    residue_positions = np.atleast_2d(np.asarray(residue_positions, dtype=float))
    residue_values = np.asarray(residue_values, dtype=float)

    if residue_positions.shape[0] != residue_values.shape[0]:
        raise ValueError("residue_positions and residue_values must have matching length")
    if residue_positions.shape[0] == 0:
        return np.full(path_points.shape[0], DEFAULT_HYDROPHOBICITY)

    diff = path_points[:, None, :] - residue_positions[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    weights = np.exp(-0.5 * (dist / sigma) ** 2)
    weight_sums = weights.sum(axis=1)
    safe_sums = np.where(weight_sums == 0, 1.0, weight_sums)
    values = (weights @ residue_values) / safe_sums
    values = np.where(weight_sums == 0, DEFAULT_HYDROPHOBICITY, values)
    return values
