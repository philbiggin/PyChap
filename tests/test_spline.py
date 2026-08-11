"""Tests for pychap.spline.PathSpline."""

import unittest

import numpy as np

from pychap.spline import PathSpline


class TestPathSpline(unittest.TestCase):
    def test_length_of_straight_line(self):
        points = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 10.0]])
        spline = PathSpline(points)
        self.assertAlmostEqual(spline.length, 10.0, places=8)

    def test_evaluates_endpoints_exactly(self):
        points = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [2.0, 0.0, 6.0]])
        spline = PathSpline(points)
        np.testing.assert_allclose(spline(0.0), points[0], atol=1e-8)
        np.testing.assert_allclose(spline(spline.length), points[-1], atol=1e-8)

    def test_resample_includes_endpoints_and_right_count(self):
        points = np.array([[0.0, 0.0, 0.0], [0.0, 3.0, 4.0], [0.0, 6.0, 8.0]])
        spline = PathSpline(points)
        s, pts = spline.resample(n=50)
        self.assertEqual(len(s), 50)
        self.assertEqual(pts.shape, (50, 3))
        np.testing.assert_allclose(pts[0], points[0], atol=1e-8)
        np.testing.assert_allclose(pts[-1], points[-1], atol=1e-6)

    def test_tangent_is_unit_length(self):
        rng = np.random.default_rng(42)
        points = np.cumsum(rng.normal(size=(10, 3)), axis=0)
        spline = PathSpline(points)
        s = np.linspace(0.0, spline.length, 25)
        tangents = spline.tangent(s)
        norms = np.linalg.norm(tangents, axis=-1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    def test_straight_line_tangent_matches_direction(self):
        points = np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]])  # length 5, direction (0.6, 0.8, 0)
        spline = PathSpline(points)
        tangent = spline.tangent(2.5)
        np.testing.assert_allclose(tangent, [0.6, 0.8, 0.0], atol=1e-6)

    def test_rejects_too_few_points(self):
        with self.assertRaises(ValueError):
            PathSpline(np.array([[0.0, 0.0, 0.0]]))

    def test_rejects_duplicate_points(self):
        with self.assertRaises(ValueError):
            PathSpline(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]))

    def test_rejects_all_points_collapsing_to_one_location(self):
        # Every point is a duplicate of the first -- nothing survives the
        # near-duplicate filter, so this should still raise, just with a
        # different (clearer) message than a bare "path points must be
        # strictly distinct".
        with self.assertRaises(ValueError):
            PathSpline(np.zeros((5, 3)))

    def test_tolerates_a_near_duplicate_point_deep_into_a_long_path(self):
        # Regression test for a real bug: a coarse centreline search can
        # occasionally produce a step so tiny (but not exactly zero) that,
        # once folded into a cumulative arc length that has already grown
        # large, floating-point rounding makes the *cumulative* coordinate
        # fail to increase -- even though the raw segment length was
        # technically positive and so passed an earlier, less careful
        # check. This used to surface as "x must be strictly increasing"
        # raised directly out of pychap._numerics.CubicSpline1D, deep
        # inside compute_pore_profile, rather than being handled here.
        # Build a path with a healthy cumulative arc length (~130 A) by
        # the time a point arrives that's only 1e-15 A from its
        # predecessor -- far below float64's resolution at that scale
        # (confirmed: 137.0 + 1e-15 == 137.0 in float64) -- and check this
        # does not raise, and that the resulting spline is still usable.
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 50.0],
                [0.0, 0.0, 100.0],
                [0.0, 0.0, 137.0],
                [0.0, 0.0, 137.0 + 1e-15],  # collapses onto the previous point
                [0.0, 0.0, 160.0],
            ]
        )
        spline = PathSpline(points)  # should not raise
        self.assertAlmostEqual(spline.length, 160.0, places=6)
        s, pts = spline.resample(n=40)
        self.assertEqual(len(s), 40)
        np.testing.assert_allclose(pts[0], points[0], atol=1e-6)
        np.testing.assert_allclose(pts[-1], points[-1], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
