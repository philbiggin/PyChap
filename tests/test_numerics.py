"""Tests for pychap._numerics (dependency-free spline + optimiser building blocks)."""

import unittest

import numpy as np

from pychap._numerics import CubicSpline1D, nelder_mead_2d


class TestCubicSpline1D(unittest.TestCase):
    def test_reproduces_linear_data_exactly(self):
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        y = 2.0 * x + 1.0
        spline = CubicSpline1D(x, y)
        query = np.linspace(0.0, 4.0, 41)
        np.testing.assert_allclose(spline(query), 2.0 * query + 1.0, atol=1e-10)

    def test_passes_through_nodes(self):
        x = np.array([0.0, 1.0, 2.5, 4.0, 7.0])
        y = np.array([0.0, 2.0, -1.0, 3.0, 1.0])
        spline = CubicSpline1D(x, y)
        for xi, yi in zip(x, y):
            self.assertAlmostEqual(spline(xi), yi, places=8)

    def test_derivative_of_linear_data_is_constant_slope(self):
        x = np.linspace(0, 10, 11)
        y = 3.0 * x - 5.0
        spline = CubicSpline1D(x, y)
        derivs = spline(np.linspace(0, 10, 21), derivative=1)
        np.testing.assert_allclose(derivs, 3.0, atol=1e-8)

    def test_scalar_input_returns_scalar(self):
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([0.0, 1.0, 0.0])
        spline = CubicSpline1D(x, y)
        result = spline(1.0)
        self.assertIsInstance(result, float)

    def test_rejects_non_increasing_x(self):
        with self.assertRaises(ValueError):
            CubicSpline1D(np.array([0.0, 1.0, 0.5]), np.array([0.0, 1.0, 2.0]))

    def test_rejects_mismatched_lengths(self):
        with self.assertRaises(ValueError):
            CubicSpline1D(np.array([0.0, 1.0]), np.array([0.0, 1.0, 2.0]))


class TestNelderMead2D(unittest.TestCase):
    def test_finds_minimum_of_simple_paraboloid(self):
        def f(p):
            x, y = p
            return (x - 3.0) ** 2 + (y + 2.0) ** 2 + 1.0

        best_x, best_f = nelder_mead_2d(f, x0=(0.0, 0.0), initial_step=1.0, max_iter=300)
        np.testing.assert_allclose(best_x, [3.0, -2.0], atol=1e-2)
        self.assertAlmostEqual(best_f, 1.0, places=2)

    def test_finds_minimum_from_offset_start(self):
        def f(p):
            x, y = p
            return (x + 5.0) ** 2 + (y - 7.0) ** 2

        best_x, best_f = nelder_mead_2d(f, x0=(10.0, 10.0), initial_step=2.0, max_iter=500)
        np.testing.assert_allclose(best_x, [-5.0, 7.0], atol=5e-2)
        self.assertLess(best_f, 1e-2)


if __name__ == "__main__":
    unittest.main()
