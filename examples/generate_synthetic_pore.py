#!/usr/bin/env python3
"""
Generate a small synthetic "ion channel" trajectory and write it out as a
genuine GROMACS-readable ``.gro`` topology + ``.xtc`` trajectory, using
MDAnalysis.

This exists to demonstrate (and let you test for yourself) that PyChap
reads real GROMACS trajectory files: the files this script produces are
read back by pychap.trajectory.Trajectory / pychap.analysis.PoreAnalysis
in exactly the way a genuine simulation's ``.xtc`` output would be.

The synthetic system is an "hourglass" pore -- a ring of pseudo-atoms
per residue, arranged in a tube that narrows to a constriction ("gate")
at its centre and flares out at both ends (like a simplified ion channel
vestibule + selectivity filter), which gently "breathes" (fluctuates)
from frame to frame. Residues near the constriction are named LEU
(hydrophobic); residues further away are named ASP (hydrophilic), so
that the hydrophobicity-mapping part of the pipeline has something
structured to recover.

Usage
-----
    python generate_synthetic_pore.py -o synthetic_pore_data -n-frames 10

Requires MDAnalysis: pip install MDAnalysis (or pip install -r requirements.txt)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pychap.testing import hourglass_pore  # noqa: E402

try:
    import MDAnalysis as mda
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "This script needs MDAnalysis to write .gro/.xtc files.\n"
        "Install it with: pip install MDAnalysis"
    ) from exc


def build_static_universe(ring_z, n_per_ring, positions0):
    """Build an MDAnalysis Universe with the right topology (names,
    residues, elements) for the synthetic pore, and an initial frame."""
    n_atoms = len(positions0)
    z_per_atom = np.repeat(ring_z, n_per_ring)

    # Hydrophobic "gate" residues (LEU) near the constriction, hydrophilic
    # vestibule residues (ASP) further away -- one atom per residue, as a
    # simple stand-in for e.g. a CA-only selection of a real pore lining.
    resnames = np.where(np.abs(z_per_atom) <= 6.0, "LEU", "ASP")
    names = np.full(n_atoms, "CA")
    resids = np.arange(1, n_atoms + 1)
    resindices = np.arange(n_atoms)  # one atom per residue

    universe = mda.Universe.empty(
        n_atoms,
        n_residues=n_atoms,
        atom_resindex=resindices,
        trajectory=True,
    )
    universe.add_TopologyAttr("name", names)
    universe.add_TopologyAttr("type", np.full(n_atoms, "C"))
    universe.add_TopologyAttr("resname", resnames)
    universe.add_TopologyAttr("resid", resids)
    universe.add_TopologyAttr("segid", ["SYNTH"])
    universe.atoms.positions = positions0

    return universe


def make_trajectory_frames(n_frames, seed=0, breathing_amplitude=0.5, r_min=3.0, **pore_kwargs):
    """Generate a short 'trajectory' of the hourglass pore with the
    constriction radius fluctuating slightly from frame to frame."""
    rng = np.random.default_rng(seed)
    frames = []
    for _ in range(n_frames):
        frame_r_min = max(0.5, r_min + breathing_amplitude * rng.normal())
        positions, _radii, _ring_z, _ring_radius = hourglass_pore(r_min=frame_r_min, **pore_kwargs)
        frames.append(positions)
    return frames


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-o", "--outdir", default="synthetic_pore_data")
    parser.add_argument("-n-frames", type=int, default=10)
    parser.add_argument("-n-rings", type=int, default=41)
    parser.add_argument("-n-per-ring", type=int, default=24)
    parser.add_argument("-seed", type=int, default=0)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    pore_kwargs = dict(
        r_max=10.0, n_rings=args.n_rings, n_per_ring=args.n_per_ring, z_min=-20.0, z_max=20.0, vdw_radius=1.0
    )

    _pos0, _radii0, ring_z, _ring_radius0 = hourglass_pore(r_min=3.0, **pore_kwargs)
    frames = make_trajectory_frames(args.n_frames, seed=args.seed, r_min=3.0, **pore_kwargs)

    universe = build_static_universe(ring_z, args.n_per_ring, frames[0])
    n_atoms = len(universe.atoms)

    box_size = 2.0 * pore_kwargs["r_max"] + 20.0
    universe.dimensions = [box_size, box_size, 60.0, 90.0, 90.0, 90.0]

    gro_path = outdir / "synthetic_pore.gro"
    xtc_path = outdir / "synthetic_pore.xtc"

    universe.atoms.positions = frames[0]
    universe.atoms.write(str(gro_path))

    with mda.Writer(str(xtc_path), n_atoms=n_atoms) as writer:
        for frame_positions in frames:
            universe.atoms.positions = frame_positions
            writer.write(universe.atoms)

    print(f"Wrote {gro_path}")
    print(f"Wrote {xtc_path} ({len(frames)} frames, {n_atoms} atoms)")
    print()
    print("Try it, e.g.:")
    print(
        f"  pychap -s {gro_path} -f {xtc_path} -sel all -o {outdir / 'pychap_output'}"
    )


if __name__ == "__main__":
    main()
