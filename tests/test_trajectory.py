"""Tests for pychap.trajectory.Trajectory.

The MDAnalysis-backed tests here are skipped automatically if MDAnalysis
is not installed (e.g. in network-restricted CI/sandbox environments);
they are written to run for real wherever MDAnalysis (and therefore real
GROMACS .xtc support) is available, which is the normal case for anyone
installing pychap per README.md.
"""

import os
import tempfile
import unittest

import numpy as np

from pychap.testing import cylindrical_pore

try:
    import MDAnalysis  # noqa: F401

    HAS_MDANALYSIS = True
except ImportError:
    HAS_MDANALYSIS = False

from pychap.trajectory import Trajectory


def write_gro(path, positions_angstrom, resnames, resids, atomnames, box_nm=(100.0, 100.0, 100.0)):
    """Write a minimal, valid GROMACS .gro file (plain text -- no MDAnalysis needed)."""
    positions_nm = np.asarray(positions_angstrom) / 10.0
    with open(path, "w") as fh:
        fh.write("pychap synthetic pore test structure\n")
        fh.write(f"{len(positions_nm)}\n")
        for i, (pos, resname, resid, atomname) in enumerate(
            zip(positions_nm, resnames, resids, atomnames), start=1
        ):
            fh.write(
                f"{resid % 100000:5d}{resname:<5.5s}{atomname:>5.5s}{i % 100000:5d}"
                f"{pos[0]:8.3f}{pos[1]:8.3f}{pos[2]:8.3f}\n"
            )
        fh.write(f"{box_nm[0]:10.5f}{box_nm[1]:10.5f}{box_nm[2]:10.5f}\n")


@unittest.skipUnless(HAS_MDANALYSIS, "MDAnalysis not installed in this environment")
class TestTrajectoryWithMDAnalysis(unittest.TestCase):
    def setUp(self):
        positions, _radii = cylindrical_pore(radius=8.0, n_rings=5, n_per_ring=6, z_min=-5, z_max=5)
        n = len(positions)
        resnames = ["ALA"] * n
        resids = list(range(1, n + 1))
        atomnames = ["CA"] * n
        self.tmpdir = tempfile.TemporaryDirectory()
        self.gro_path = os.path.join(self.tmpdir.name, "structure.gro")
        write_gro(self.gro_path, positions, resnames, resids, atomnames)
        self.n_atoms = n

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_loads_structure_and_selection(self):
        traj = Trajectory(self.gro_path, selection="all")
        self.assertEqual(traj.n_atoms, self.n_atoms)
        self.assertEqual(traj.n_frames, 1)

    def test_frame_positions_have_correct_shape(self):
        traj = Trajectory(self.gro_path, selection="all")
        pos = traj.frame_positions(0)
        self.assertEqual(pos.shape, (self.n_atoms, 3))

    def test_vdw_radii_have_correct_length_and_are_positive(self):
        traj = Trajectory(self.gro_path, selection="all")
        radii = traj.vdw_radii()
        self.assertEqual(len(radii), traj.n_atoms)
        self.assertTrue(np.all(radii > 0))

    def test_residue_centers_have_matching_lengths(self):
        traj = Trajectory(self.gro_path, selection="all")
        traj.frame_positions(0)
        positions, resnames, resids = traj.residue_centers()
        self.assertEqual(len(positions), len(resnames))
        self.assertEqual(len(positions), len(resids))

    def test_empty_selection_raises(self):
        with self.assertRaises(ValueError):
            Trajectory(self.gro_path, selection="resname ZZZ")


class TestTrajectoryImportGuard(unittest.TestCase):
    @unittest.skipIf(HAS_MDANALYSIS, "only relevant when MDAnalysis is unavailable")
    def test_raises_clear_error_without_mdanalysis(self):
        with self.assertRaises(ImportError):
            Trajectory("nonexistent.gro")


if __name__ == "__main__":
    unittest.main()
