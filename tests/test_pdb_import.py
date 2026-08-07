"""Tests for pychap.pdb_import (dependency-free PDB reader)."""

import os
import tempfile
import unittest

import numpy as np

from pychap.pdb_import import read_pdb


SIMPLE_PDB = """\
REMARK   test structure
ATOM      1  N   ALA A   1       1.000   2.000   3.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.500   2.500   3.500  1.00  0.00           C
ATOM      3  C   ALA A   1       2.000   3.000   4.000  1.00  0.00           C
ATOM      4  N   GLY A   2       5.000   5.000   5.000  1.00  0.00           N
ATOM      5  CA  GLY A   2       5.500   5.500   5.500  1.00  0.00           C
HETATM    6  O   HOH W   1      10.000  10.000  10.000  1.00  0.00           O
END
"""

MULTI_MODEL_PDB = """\
MODEL        1
ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C
ENDMDL
MODEL        2
ATOM      1  CA  ALA A   1       9.000   9.000   9.000  1.00  0.00           C
ENDMDL
"""


class TestReadPdb(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmpdir.name, "test.pdb")
        with open(self.path, "w") as fh:
            fh.write(SIMPLE_PDB)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_reads_correct_number_of_atoms(self):
        structure = read_pdb(self.path)
        self.assertEqual(len(structure), 6)

    def test_positions_parsed_correctly(self):
        structure = read_pdb(self.path)
        np.testing.assert_allclose(structure.positions[0], [1.0, 2.0, 3.0])
        np.testing.assert_allclose(structure.positions[3], [5.0, 5.0, 5.0])

    def test_resnames_and_resids(self):
        structure = read_pdb(self.path)
        self.assertEqual(structure.resnames[:3], ["ALA", "ALA", "ALA"])
        self.assertEqual(structure.resids[:3], [1, 1, 1])
        self.assertEqual(structure.resnames[3], "GLY")
        self.assertEqual(structure.resids[3], 2)

    def test_hetatm_records_included(self):
        structure = read_pdb(self.path)
        self.assertIn("HOH", structure.resnames)

    def test_elements_parsed_from_columns_77_78(self):
        structure = read_pdb(self.path)
        self.assertEqual(structure.elements[0], "N")
        self.assertEqual(structure.elements[1], "C")

    def test_vdw_radii_length_and_values(self):
        structure = read_pdb(self.path)
        radii = structure.vdw_radii()
        self.assertEqual(len(radii), 6)
        self.assertAlmostEqual(radii[0], 1.55)  # N
        self.assertAlmostEqual(radii[1], 1.70)  # C

    def test_select_by_include(self):
        structure = read_pdb(self.path)
        subset = structure.select(resnames_include={"ALA"})
        self.assertEqual(len(subset), 3)
        self.assertTrue(all(r == "ALA" for r in subset.resnames))

    def test_select_by_exclude(self):
        structure = read_pdb(self.path)
        subset = structure.select(resnames_exclude={"HOH"})
        self.assertEqual(len(subset), 5)
        self.assertNotIn("HOH", subset.resnames)

    def test_residue_centers(self):
        structure = read_pdb(self.path)
        protein = structure.select(resnames_exclude={"HOH"})
        positions, resnames, resids = protein.residue_centers()
        self.assertEqual(len(positions), 2)  # ALA 1, GLY 2
        self.assertEqual(resnames, ["ALA", "GLY"])
        self.assertEqual(resids, [1, 2])
        np.testing.assert_allclose(positions[0], [1.5, 2.5, 3.5], atol=1e-6)

    def test_raises_on_file_with_no_atoms(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "empty.pdb")
            with open(path, "w") as fh:
                fh.write("REMARK nothing here\nEND\n")
            with self.assertRaises(ValueError):
                read_pdb(path)


class TestReadPdbMultiModel(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmpdir.name, "multi.pdb")
        with open(self.path, "w") as fh:
            fh.write(MULTI_MODEL_PDB)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_reads_first_model_by_default(self):
        structure = read_pdb(self.path)
        np.testing.assert_allclose(structure.positions[0], [0.0, 0.0, 0.0])

    def test_reads_specific_model(self):
        structure = read_pdb(self.path, model=2)
        np.testing.assert_allclose(structure.positions[0], [9.0, 9.0, 9.0])


if __name__ == "__main__":
    unittest.main()
