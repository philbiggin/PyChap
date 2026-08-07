"""
Van der Waals radii lookup.

CHAP (and HOLE, on which its pathfinding is conceptually based) computes
pore radius as the distance from a probe point to the nearest atomic van
der Waals surface. We use the standard Bondi (1964) radii, extended with
a few common ions, and fall back to a carbon-like default for anything
unrecognised. Values are in Angstrom, matching the coordinate units
returned by MDAnalysis.
"""

from __future__ import annotations

BONDI_VDW_RADII = {
    "H": 1.20,
    "HE": 1.40,
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "F": 1.47,
    "NE": 1.54,
    "NA": 2.27,
    "MG": 1.73,
    "SI": 2.10,
    "P": 1.80,
    "S": 1.80,
    "CL": 1.75,
    "AR": 1.88,
    "K": 2.75,
    "CA": 2.31,
    "MN": 1.73,
    "FE": 1.72,
    "ZN": 1.39,
    "BR": 1.85,
    "I": 1.98,
}

#: Fallback radius (Angstrom) used when an element cannot be identified.
DEFAULT_VDW_RADIUS = 1.70


def guess_element_from_name(atom_name: str) -> str:
    """Best-effort guess of an element symbol from a PDB/GRO-style atom name.

    This mirrors the heuristic used by most MD tooling: strip leading
    digits, try the first two letters, then the first letter. Ambiguous
    two-letter overlaps with common atom naming (e.g. alpha-carbon "CA"
    vs. a calcium ion "CA") are resolved in favour of the single-letter
    guess where the two-letter symbol is a metal that is rare in protein
    atom names, since :func:`vdw_radius_for_name` is only meant as a
    fallback for when explicit element information is unavailable.
    """
    name = "".join(ch for ch in atom_name if ch.isalpha())
    if not name:
        return "C"

    # Protein/nucleic acid backbone and common sidechain atom names are
    # single-element (C, N, O, S, P, H) even when written with two
    # letters in some naming conventions (e.g. "CA", "CB", "OG1").
    # Only trust a two-letter symbol if it is not also a very common
    # biomolecular atom-naming prefix.
    two_letter_ambiguous = {"CA", "NA", "CD", "CE", "NE", "ND"}
    candidate2 = name[:2].upper()
    if candidate2 in BONDI_VDW_RADII and candidate2 not in two_letter_ambiguous:
        return candidate2

    candidate1 = name[0].upper()
    if candidate1 in BONDI_VDW_RADII:
        return candidate1

    if candidate2 in BONDI_VDW_RADII:
        return candidate2

    return "C"


def vdw_radius_for_element(element: str) -> float:
    """Look up the van der Waals radius for an element symbol."""
    return BONDI_VDW_RADII.get(element.strip().upper(), DEFAULT_VDW_RADIUS)


def vdw_radius_for_name(atom_name: str) -> float:
    """Look up the van der Waals radius from a raw atom name (fallback)."""
    return vdw_radius_for_element(guess_element_from_name(atom_name))
