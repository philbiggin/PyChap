"""
Minimal, dependency-free reader for a single PDB structure.

Complements :mod:`pychap.pdb_export` (which writes annotated PDBs) with
a reader for the common case of analysing a single static structure --
CHAP's own "Running CHAP on Structures" mode, see
https://www.channotation.org/docs/annotation_example/ -- without
needing MDAnalysis. Real multi-frame trajectories (``.xtc``, ``.trr``,
...) still require MDAnalysis via :mod:`pychap.trajectory`; this module
only handles plain-text PDB files.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .elements import guess_element_from_name


@dataclass
class PDBStructure:
    """A parsed PDB structure: per-atom coordinates and identity."""

    positions: np.ndarray  #: (N, 3) array, Angstrom
    resnames: list          #: length N
    resids: list              #: length N (int)
    atom_names: list            #: length N
    elements: list                #: length N

    def __len__(self) -> int:
        return len(self.positions)

    def vdw_radii(self) -> np.ndarray:
        """Per-atom van der Waals radii, from element (preferred) or atom name."""
        from .elements import vdw_radius_for_element

        return np.array(
            [
                vdw_radius_for_element(el) if el else vdw_radius_for_element(guess_element_from_name(name))
                for el, name in zip(self.elements, self.atom_names)
            ]
        )

    def select(self, resnames_include=None, resnames_exclude=None) -> "PDBStructure":
        """Return a new PDBStructure containing only a subset of atoms,
        filtered by residue name.

        Parameters
        ----------
        resnames_include:
            If given, only atoms whose residue name is in this set are kept.
        resnames_exclude:
            If given, atoms whose residue name is in this set are dropped
            (applied after ``resnames_include``, if both are given).
        """
        n = len(self)
        mask = np.ones(n, dtype=bool)
        if resnames_include is not None:
            include = set(resnames_include)
            mask &= np.array([r in include for r in self.resnames])
        if resnames_exclude is not None:
            exclude = set(resnames_exclude)
            mask &= np.array([r not in exclude for r in self.resnames])

        idx = np.nonzero(mask)[0]
        return PDBStructure(
            positions=self.positions[idx],
            resnames=[self.resnames[i] for i in idx],
            resids=[self.resids[i] for i in idx],
            atom_names=[self.atom_names[i] for i in idx],
            elements=[self.elements[i] for i in idx],
        )

    def residue_centers(self):
        """Per-residue representative (mean) position, residue name, and
        residue id, grouped by ``resids`` in first-appearance order.

        Note: residues are grouped purely by numeric ``resid``. If a
        structure has multiple chains that reuse the same resid, they
        will be merged into a single "residue" here -- check for that
        (e.g. via chain-aware selection before calling this) if your
        structure has that property. The 4PIR example bundled with this
        package numbers all of its residues uniquely across its five
        subunits, so this isn't an issue for it.
        """
        order = []
        indices_by_resid = {}
        for i, resid in enumerate(self.resids):
            if resid not in indices_by_resid:
                indices_by_resid[resid] = []
                order.append(resid)
            indices_by_resid[resid].append(i)

        positions = []
        resnames = []
        resids_out = []
        for resid in order:
            idxs = indices_by_resid[resid]
            positions.append(self.positions[idxs].mean(axis=0))
            resnames.append(self.resnames[idxs[0]])
            resids_out.append(resid)

        return np.array(positions), resnames, resids_out


def read_pdb(path, model: int = 1) -> PDBStructure:
    """Parse ATOM/HETATM records from a PDB file.

    Parameters
    ----------
    path:
        Path to the PDB file.
    model:
        Which ``MODEL`` block to read, if the file has more than one
        (1-indexed, matching PDB's own ``MODEL`` numbering). Defaults to
        the first model. If the file has no ``MODEL``/``ENDMDL`` records
        at all (a plain single-structure PDB), this parameter has no
        effect and all records are read.

    Returns
    -------
    :class:`PDBStructure`
    """
    positions = []
    resnames = []
    resids = []
    atom_names = []
    elements = []

    current_model = 1
    in_target_model = True
    saw_model_record = False

    with open(path) as fh:
        for line in fh:
            if line.startswith("MODEL"):
                saw_model_record = True
                try:
                    current_model = int(line[10:14])
                except ValueError:
                    current_model += 1
                in_target_model = current_model == model
                continue

            if line.startswith("ENDMDL"):
                if saw_model_record and current_model >= model:
                    break
                continue

            if not in_target_model:
                continue

            if line.startswith(("ATOM", "HETATM")):
                name = line[12:16].strip()
                resname = line[17:20].strip()
                try:
                    resid = int(line[22:26])
                except ValueError:
                    continue
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                except ValueError:
                    continue

                element = line[76:78].strip() if len(line) >= 78 else ""

                positions.append((x, y, z))
                resnames.append(resname)
                resids.append(resid)
                atom_names.append(name)
                elements.append(element)

    if not positions:
        raise ValueError(f"no ATOM/HETATM records found in {path} (model={model})")

    return PDBStructure(
        positions=np.array(positions, dtype=float),
        resnames=resnames,
        resids=resids,
        atom_names=atom_names,
        elements=elements,
    )
