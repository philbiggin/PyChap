"""Tests for pychap.pdb_export (minimal PDB writer)."""

import os
import tempfile
import unittest

import numpy as np

from pychap.pdb_export import write_minimal_pdb


class TestWriteMinimalPdb(unittest.TestCase):
    def setUp(self):
        self.positions = np.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        )
        self.resnames = ["ALA", "ALA", "GLY"]
        self.resids = [1, 1, 2]
        self.atom_names = ["N", "CA", "CA"]

    def test_writes_correct_number_of_atom_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.pdb")
            write_minimal_pdb(self.positions, self.resnames, self.resids, self.atom_names, path)
            with open(path) as fh:
                lines = fh.readlines()
            atom_lines = [l for l in lines if l.startswith("ATOM")]
            self.assertEqual(len(atom_lines), 3)
            self.assertTrue(lines[-1].startswith("END"))

    def test_coordinates_round_trip_at_3_decimal_precision(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.pdb")
            write_minimal_pdb(self.positions, self.resnames, self.resids, self.atom_names, path)
            with open(path) as fh:
                atom_lines = [l for l in fh if l.startswith("ATOM")]
            # PDB fixed columns: x=31-38, y=39-46, z=47-54 (1-indexed, inclusive)
            for line, expected in zip(atom_lines, self.positions):
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                np.testing.assert_allclose([x, y, z], expected, atol=1e-3)

    def test_b_factor_column_matches_input(self):
        b_factors = [1.23, -4.56, 7.89]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.pdb")
            write_minimal_pdb(
                self.positions, self.resnames, self.resids, self.atom_names, path, b_factors=b_factors
            )
            with open(path) as fh:
                atom_lines = [l for l in fh if l.startswith("ATOM")]
            # tempFactor column: 61-66 (1-indexed)
            for line, expected in zip(atom_lines, b_factors):
                temp = float(line[60:66])
                self.assertAlmostEqual(temp, expected, places=2)

    def test_occupancy_column_matches_input(self):
        occupancies = [1.0, 0.0, 1.0]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.pdb")
            write_minimal_pdb(
                self.positions,
                self.resnames,
                self.resids,
                self.atom_names,
                path,
                occupancies=occupancies,
            )
            with open(path) as fh:
                atom_lines = [l for l in fh if l.startswith("ATOM")]
            # occupancy column: 55-60 (1-indexed)
            for line, expected in zip(atom_lines, occupancies):
                occ = float(line[54:60])
                self.assertAlmostEqual(occ, expected, places=2)

    def test_default_occupancy_is_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.pdb")
            write_minimal_pdb(self.positions, self.resnames, self.resids, self.atom_names, path)
            with open(path) as fh:
                atom_lines = [l for l in fh if l.startswith("ATOM")]
            for line in atom_lines:
                self.assertAlmostEqual(float(line[54:60]), 1.0, places=2)

    def test_default_b_factor_is_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.pdb")
            write_minimal_pdb(self.positions, self.resnames, self.resids, self.atom_names, path)
            with open(path) as fh:
                atom_lines = [l for l in fh if l.startswith("ATOM")]
            for line in atom_lines:
                self.assertAlmostEqual(float(line[60:66]), 0.0, places=2)

    def test_resname_and_resid_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.pdb")
            write_minimal_pdb(self.positions, self.resnames, self.resids, self.atom_names, path)
            with open(path) as fh:
                atom_lines = [l for l in fh if l.startswith("ATOM")]
            self.assertIn("ALA", atom_lines[0])
            self.assertIn("GLY", atom_lines[2])

    def test_mismatched_lengths_raise(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.pdb")
            with self.assertRaises(ValueError):
                write_minimal_pdb(self.positions, ["ALA"], self.resids, self.atom_names, path)

    def test_mismatched_bfactor_length_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.pdb")
            with self.assertRaises(ValueError):
                write_minimal_pdb(
                    self.positions,
                    self.resnames,
                    self.resids,
                    self.atom_names,
                    path,
                    b_factors=[1.0, 2.0],
                )


if __name__ == "__main__":
    unittest.main()
