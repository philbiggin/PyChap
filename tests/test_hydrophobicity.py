"""Tests for pychap.hydrophobicity."""

import unittest

import numpy as np

from pychap.hydrophobicity import (
    DEFAULT_HYDROPHOBICITY,
    WIMLEY_WHITE_INTERFACE,
    hydrophobicity_profile,
    residue_hydrophobicity,
)


class TestResidueHydrophobicity(unittest.TestCase):
    def test_known_residue_values(self):
        self.assertAlmostEqual(residue_hydrophobicity("LEU"), -0.56)
        self.assertAlmostEqual(residue_hydrophobicity("ASP"), 1.23)

    def test_case_insensitive(self):
        self.assertEqual(residue_hydrophobicity("leu"), residue_hydrophobicity("LEU"))

    def test_unknown_residue_returns_default(self):
        self.assertEqual(residue_hydrophobicity("XYZ"), DEFAULT_HYDROPHOBICITY)

    def test_custom_scale(self):
        custom = {"FOO": 42.0}
        self.assertEqual(residue_hydrophobicity("FOO", scale=custom), 42.0)
        self.assertEqual(residue_hydrophobicity("LEU", scale=custom), DEFAULT_HYDROPHOBICITY)

    def test_scale_covers_all_20_standard_amino_acids(self):
        standard = {
            "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS",
            "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP",
            "TYR", "VAL",
        }
        self.assertTrue(standard.issubset(set(WIMLEY_WHITE_INTERFACE)))


class TestHydrophobicityProfile(unittest.TestCase):
    def test_point_colocated_with_single_residue_dominates(self):
        # One residue far away (weak influence) and one exactly at the query point.
        path_points = np.array([[0.0, 0.0, 0.0]])
        residue_positions = np.array([[0.0, 0.0, 0.0], [1000.0, 0.0, 0.0]])
        residue_values = np.array([-2.0, 5.0])
        result = hydrophobicity_profile(path_points, residue_positions, residue_values, sigma=5.0)
        self.assertAlmostEqual(result[0], -2.0, places=6)

    def test_symmetric_midpoint_averages_two_equal_weight_residues(self):
        path_points = np.array([[5.0, 0.0, 0.0]])
        residue_positions = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        residue_values = np.array([-1.0, 3.0])
        result = hydrophobicity_profile(path_points, residue_positions, residue_values, sigma=5.0)
        self.assertAlmostEqual(result[0], 1.0, places=6)  # average of -1 and 3

    def test_output_shape_matches_number_of_path_points(self):
        path_points = np.random.default_rng(0).normal(size=(15, 3))
        residue_positions = np.random.default_rng(1).normal(size=(6, 3))
        residue_values = np.arange(6, dtype=float)
        result = hydrophobicity_profile(path_points, residue_positions, residue_values, sigma=3.0)
        self.assertEqual(result.shape, (15,))

    def test_no_residues_returns_default_everywhere(self):
        path_points = np.zeros((4, 3))
        residue_positions = np.zeros((0, 3))
        residue_values = np.zeros(0)
        result = hydrophobicity_profile(path_points, residue_positions, residue_values)
        np.testing.assert_allclose(result, DEFAULT_HYDROPHOBICITY)

    def test_mismatched_lengths_raise(self):
        with self.assertRaises(ValueError):
            hydrophobicity_profile(
                np.zeros((3, 3)), np.zeros((2, 3)), np.zeros(3), sigma=5.0
            )


if __name__ == "__main__":
    unittest.main()
