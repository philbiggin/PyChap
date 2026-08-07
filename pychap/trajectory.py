"""
Trajectory reading via MDAnalysis.

This is the piece that lets pychap read "current" GROMACS trajectory
files directly: MDAnalysis natively supports ``.xtc`` (and ``.trr``)
trajectories paired with a topology in ``.gro``, ``.pdb``, or ``.tpr``
format (the latter requires the optional ``gsd``/``tpr`` parsing extras
that ship with MDAnalysis). This mirrors the ``-s``/``-f`` options of the
original CHAP command line tool.

All coordinates are returned in Angstrom (MDAnalysis' native unit),
which matches the van der Waals radii table in :mod:`pychap.elements`.
"""

from __future__ import annotations

import numpy as np

# MDAnalysis is imported lazily (only when a Trajectory is actually
# instantiated) so that the rest of pychap -- the pure-NumPy pathway,
# radius-profile, and hydrophobicity code -- can be imported and used
# (including in tests) even in environments where MDAnalysis is not
# installed. Real trajectory reading, of course, does need it.
try:
    import MDAnalysis as mda

    _HAS_MDANALYSIS = True
except ImportError:  # pragma: no cover - exercised only when MDAnalysis missing
    mda = None
    _HAS_MDANALYSIS = False

from .elements import vdw_radius_for_element, vdw_radius_for_name


class Trajectory:
    """A thin wrapper around an MDAnalysis Universe restricted to a selection.

    Parameters
    ----------
    topology:
        Path to a topology file (``.gro``, ``.pdb``, ``.tpr``, ``.psf``, ...).
    trajectory:
        Optional path (or list of paths) to a trajectory file (``.xtc``,
        ``.trr``, ``.dcd``, ...). If omitted, the topology file itself is
        treated as a single-frame "trajectory" (useful for a plain
        ``.gro``/``.pdb`` structure).
    selection:
        MDAnalysis atom selection string identifying the pore-lining
        atoms to consider, e.g. ``"protein"`` or
        ``"protein and resid 50-120"``.
    """

    def __init__(self, topology, trajectory=None, selection: str = "protein"):
        if not _HAS_MDANALYSIS:
            raise ImportError(
                "pychap.trajectory.Trajectory requires MDAnalysis to read "
                "structure/trajectory files. Install it with "
                "'pip install MDAnalysis' (or 'pip install -r requirements.txt')."
            )
        if trajectory is not None:
            self.universe = mda.Universe(str(topology), trajectory)
        else:
            self.universe = mda.Universe(str(topology))

        self.selection_str = selection
        self.atomgroup = self.universe.select_atoms(selection)
        if len(self.atomgroup) == 0:
            raise ValueError(f"selection {selection!r} matched no atoms")

    @property
    def n_frames(self) -> int:
        return len(self.universe.trajectory)

    @property
    def n_atoms(self) -> int:
        return len(self.atomgroup)

    def frame_positions(self, frame_index: int) -> np.ndarray:
        """Return a copy of the selected atoms' coordinates at a given frame."""
        self.universe.trajectory[frame_index]
        return self.atomgroup.positions.copy()

    def iter_frames(self, start: int = 0, stop: int | None = None, step: int = 1):
        """Iterate over ``(frame_index, positions)`` pairs."""
        stop = self.n_frames if stop is None else stop
        for i in range(start, stop, step):
            yield i, self.frame_positions(i)

    def vdw_radii(self) -> np.ndarray:
        """Per-atom van der Waals radii (Angstrom) for the current selection."""
        radii = np.empty(len(self.atomgroup))
        elements = None
        try:
            elements = self.atomgroup.elements
        except (mda.exceptions.NoDataError, AttributeError):
            elements = None

        for i, atom in enumerate(self.atomgroup):
            element = elements[i] if elements is not None else ""
            if element and element.strip():
                radii[i] = vdw_radius_for_element(element)
            else:
                radii[i] = vdw_radius_for_name(atom.name)
        return radii

    def residue_centers(self):
        """Per-residue representative coordinates, names, and resids.

        The representative position of each residue is the mean position
        of its atoms that are part of the active selection (i.e. this
        must be called after setting the desired trajectory frame, e.g.
        via :meth:`frame_positions`).

        Returns
        -------
        positions : ``(R, 3)`` array
        resnames : list of str, length R
        resids : list of int, length R
        """
        positions = []
        resnames = []
        resids = []
        for residue in self.atomgroup.residues:
            sel = residue.atoms.intersection(self.atomgroup)
            if len(sel) == 0:
                continue
            positions.append(sel.positions.mean(axis=0))
            resnames.append(residue.resname)
            resids.append(int(residue.resid))
        return np.array(positions), resnames, resids
