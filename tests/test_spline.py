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


if __name__ == "__main__":
    unittest.main()
