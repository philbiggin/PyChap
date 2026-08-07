"""Tests for pychap.pathfinding, validated against synthetic pores with known geometry."""

import unittest

import numpy as np

from pychap.pathfinding import compute_pore_profile, find_centreline
from pychap.testing import cylindrical_pore, hourglass_pore


class TestCylindricalPore(unittest.TestCase):
    """A straight cylinder has an exactly known radius profile: R - r_vdw everywhere,
    with the centreline exactly on the z axis. This is the cleanest possible
    validation of the pathfinding algorithm against ground truth."""

    def setUp(self):
        self.R = 8.0
        self.vdw = 1.0
        self.positions, self.radii = cylindrical_pore(
            radius=self.R, n_rings=21, n_per_ring=24, z_min=-20.0, z_max=20.0, vdw_radius=self.vdw
        )

    def test_radius_profile_matches_analytic_value(self):
        profile = compute_pore_profile(
            self.positions, self.radii, axis=2, n_slices=21, n_resample=60, window=6.0
        )
        expected = self.R - self.vdw
        # A little slack for the discretisation (24 atoms/ring) and the
        # Nelder-Mead convergence tolerance.
        np.testing.assert_allclose(profile.radius, expected, atol=0.1)

    def test_centreline_stays_on_axis(self):
        profile = compute_pore_profile(
            self.positions, self.radii, axis=2, n_slices=21, n_resample=60, window=6.0
        )
        np.testing.assert_allclose(profile.points[:, 0], 0.0, atol=1e-6)
        np.testing.assert_allclose(profile.points[:, 1], 0.0, atol=1e-6)

    def test_pathway_length_matches_axis_extent(self):
        profile = compute_pore_profile(
            self.positions, self.radii, axis=2, n_slices=21, n_resample=60, window=6.0
        )
        self.assertAlmostEqual(profile.length, 40.0, places=6)

    def test_min_radius_matches_overall_radius(self):
        profile = compute_pore_profile(
            self.positions, self.radii, axis=2, n_slices=21, n_resample=60, window=6.0
        )
        self.assertAlmostEqual(profile.min_radius, self.R - self.vdw, delta=0.1)


class TestHourglassPore(unittest.TestCase):
    """An hourglass pore has a well-defined constriction at z=0; the algorithm
    should locate both its position and its radius accurately."""

    def setUp(self):
        self.r_min = 3.0
        self.r_max = 10.0
        self.vdw = 1.0
        self.positions, self.radii, self.ring_z, self.ring_radius = hourglass_pore(
            r_min=self.r_min,
            r_max=self.r_max,
            n_rings=41,
            n_per_ring=24,
            z_min=-20.0,
            z_max=20.0,
            vdw_radius=self.vdw,
        )

    def test_finds_constriction_near_z_zero(self):
        # min_radius_position is reported in arc-length coordinate s
        # (0..length), so the z=0 constriction of this symmetric pore
        # should show up near the midpoint of the pathway, s ~ length/2.
        profile = compute_pore_profile(
            self.positions, self.radii, axis=2, n_slices=41, n_resample=120, window=3.0
        )
        self.assertLess(abs(profile.min_radius_position - profile.length / 2.0), 1.0)

    def test_constriction_radius_matches_analytic_value(self):
        profile = compute_pore_profile(
            self.positions, self.radii, axis=2, n_slices=41, n_resample=120, window=3.0
        )
        expected = self.r_min - self.vdw
        self.assertAlmostEqual(profile.min_radius, expected, delta=0.1)

    def test_radius_increases_away_from_constriction(self):
        # Sanity check on overall shape: points near the middle of the
        # profile should have a smaller radius than points near either end.
        profile = compute_pore_profile(
            self.positions, self.radii, axis=2, n_slices=41, n_resample=120, window=3.0
        )
        mid = len(profile.radius) // 2
        self.assertLess(profile.radius[mid], profile.radius[10])
        self.assertLess(profile.radius[mid], profile.radius[-10])


class TestFindCentreline(unittest.TestCase):
    def test_raises_on_mismatched_lengths(self):
        positions = np.zeros((5, 3))
        radii = np.zeros(4)
        with self.assertRaises(ValueError):
            find_centreline(positions, radii)

    def test_raises_on_empty_positions(self):
        with self.assertRaises(ValueError):
            find_centreline(np.zeros((0, 3)), np.zeros(0))

    def test_raises_on_invalid_axis(self):
        positions = np.zeros((5, 3))
        radii = np.ones(5)
        with self.assertRaises(ValueError):
            find_centreline(positions, radii, axis=3)


if __name__ == "__main__":
    unittest.main()
