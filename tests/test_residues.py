"""Tests for pychap.residues (residue-to-pathway projection) and
pychap.analysis.build_residue_summary."""

import unittest

import numpy as np

from pychap.analysis import build_residue_summary
from pychap.residues import project_onto_path, project_residues


class TestProjectOntoPath(unittest.TestCase):
    def setUp(self):
        # A straight path along z from 0 to 40, sampled every 1 A.
        self.path_s = np.linspace(0.0, 40.0, 41)
        self.path_points = np.stack(
            [np.zeros(41), np.zeros(41), self.path_s], axis=1
        )

    def test_point_on_axis_has_zero_rho(self):
        positions = np.array([[0.0, 0.0, 10.0]])
        s, rho = project_onto_path(positions, self.path_points, self.path_s)
        self.assertAlmostEqual(s[0], 10.0, places=6)
        self.assertAlmostEqual(rho[0], 0.0, places=6)

    def test_point_off_axis_has_correct_rho(self):
        positions = np.array([[3.0, 4.0, 20.0]])  # 5 A off-axis (3-4-5 triangle)
        s, rho = project_onto_path(positions, self.path_points, self.path_s)
        self.assertAlmostEqual(s[0], 20.0, places=6)
        self.assertAlmostEqual(rho[0], 5.0, places=6)

    def test_multiple_positions_vectorised(self):
        positions = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 40.0], [0.0, 5.0, 15.0]])
        s, rho = project_onto_path(positions, self.path_points, self.path_s)
        np.testing.assert_allclose(s, [0.0, 40.0, 15.0], atol=1e-6)
        np.testing.assert_allclose(rho, [0.0, 0.0, 5.0], atol=1e-6)

    def test_raises_on_empty_path(self):
        with self.assertRaises(ValueError):
            project_onto_path(np.zeros((1, 3)), np.zeros((0, 3)), np.zeros(0))


class _FakeProfile:
    """Minimal stand-in for pychap.pathfinding.PoreProfile, just exposing
    the .points and .s attributes that project_residues() needs."""

    def __init__(self, points, s):
        self.points = points
        self.s = s


class TestProjectResidues(unittest.TestCase):
    def test_pore_facing_flag_respects_cutoff(self):
        path_s = np.linspace(0.0, 40.0, 41)
        path_points = np.stack([np.zeros(41), np.zeros(41), path_s], axis=1)
        profile = _FakeProfile(points=path_points, s=path_s)

        res_positions = np.array(
            [
                [0.0, 0.0, 20.0],    # rho = 0, facing
                [10.0, 0.0, 20.0],   # rho = 10, facing (cutoff 12)
                [20.0, 0.0, 20.0],   # rho = 20, not facing
            ]
        )
        projection = project_residues(res_positions, profile, pore_facing_cutoff=12.0)
        np.testing.assert_array_equal(projection.pore_facing, [True, True, False])


class TestBuildResidueSummary(unittest.TestCase):
    def test_aggregates_mean_and_std_correctly(self):
        resids = [1, 2]
        resnames = ["LEU", "ASP"]
        # Two frames: residue 0 has s=[10, 12], residue 1 has s=[30, 30]
        s_frames = [np.array([10.0, 30.0]), np.array([12.0, 30.0])]
        rho_frames = [np.array([1.0, 5.0]), np.array([3.0, 5.0])]
        facing_frames = [np.array([True, False]), np.array([True, False])]

        summary = build_residue_summary(resids, resnames, s_frames, rho_frames, facing_frames)

        self.assertEqual(len(summary), 2)
        self.assertEqual(summary[0].resid, 1)
        self.assertEqual(summary[0].resname, "LEU")
        self.assertAlmostEqual(summary[0].s_mean, 11.0)
        self.assertAlmostEqual(summary[0].rho_mean, 2.0)
        self.assertAlmostEqual(summary[0].pore_facing_fraction, 1.0)
        self.assertAlmostEqual(summary[0].hydrophobicity, -0.56)  # LEU, Wimley-White

        self.assertEqual(summary[1].resname, "ASP")
        self.assertAlmostEqual(summary[1].s_mean, 30.0)
        self.assertAlmostEqual(summary[1].pore_facing_fraction, 0.0)
        self.assertAlmostEqual(summary[1].hydrophobicity, 1.23)

    def test_to_dict_has_expected_keys(self):
        summary = build_residue_summary(
            [1], ["LEU"], [np.array([5.0])], [np.array([1.0])], [np.array([True])]
        )
        d = summary[0].to_dict()
        for key in (
            "resid",
            "resname",
            "hydrophobicity_kcalmol",
            "s_mean_angstrom",
            "s_std_angstrom",
            "rho_mean_angstrom",
            "rho_std_angstrom",
            "pore_facing_fraction",
        ):
            self.assertIn(key, d)


if __name__ == "__main__":
    unittest.main()
