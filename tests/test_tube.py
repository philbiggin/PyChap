"""Tests for pychap.tube (3D tube mesh construction)."""

import unittest

import numpy as np

from pychap.tube import build_tube_mesh, perpendicular_frames, tube_faces


class TestPerpendicularFrames(unittest.TestCase):
    def test_frames_are_unit_and_perpendicular_to_tangent(self):
        tangents = np.array([[0.0, 0.0, 1.0]] * 10)
        e1, e2 = perpendicular_frames(tangents)
        np.testing.assert_allclose(np.linalg.norm(e1, axis=1), 1.0, atol=1e-8)
        np.testing.assert_allclose(np.linalg.norm(e2, axis=1), 1.0, atol=1e-8)
        np.testing.assert_allclose(np.sum(e1 * tangents, axis=1), 0.0, atol=1e-8)
        np.testing.assert_allclose(np.sum(e2 * tangents, axis=1), 0.0, atol=1e-8)

    def test_e1_e2_tangent_form_right_handed_basis(self):
        tangents = np.array([[0.0, 0.0, 1.0]] * 5)
        e1, e2 = perpendicular_frames(tangents)
        # e2 should equal tangent x e1 (as constructed) -- check orthogonality triple
        cross = np.cross(e1, e2)
        np.testing.assert_allclose(np.abs(np.sum(cross * tangents, axis=1)), 1.0, atol=1e-6)


class TestBuildTubeMesh(unittest.TestCase):
    def test_ring_vertices_at_correct_radius_for_straight_tube(self):
        s = np.linspace(0.0, 40.0, 21)
        points = np.stack([np.zeros(21), np.zeros(21), s], axis=1)
        radius = np.full(21, 5.0)

        rings = build_tube_mesh(points, radius, s, n_theta=16)
        self.assertEqual(rings.shape, (21, 16, 3))

        # every ring vertex should be exactly `radius` away from the
        # corresponding centreline point (in the xy plane, since the
        # tube axis is along z).
        dist_from_axis = np.linalg.norm(rings[:, :, :2] - points[:, None, :2], axis=-1)
        np.testing.assert_allclose(dist_from_axis, 5.0, atol=1e-6)

    def test_ring_z_matches_centreline_z_for_straight_tube(self):
        s = np.linspace(0.0, 40.0, 21)
        points = np.stack([np.zeros(21), np.zeros(21), s], axis=1)
        radius = np.full(21, 3.0)
        rings = build_tube_mesh(points, radius, s, n_theta=10)
        expected_z = np.broadcast_to(points[:, 2:3], rings[:, :, 2].shape)
        np.testing.assert_allclose(rings[:, :, 2], expected_z, atol=1e-6)

    def test_varying_radius_is_respected(self):
        s = np.linspace(0.0, 10.0, 11)
        points = np.stack([np.zeros(11), np.zeros(11), s], axis=1)
        radius = np.linspace(1.0, 5.0, 11)
        rings = build_tube_mesh(points, radius, s, n_theta=8)
        dist_from_axis = np.linalg.norm(rings[:, :, :2] - points[:, None, :2], axis=-1)
        for i in range(11):
            np.testing.assert_allclose(dist_from_axis[i], radius[i], atol=1e-6)

    def test_raises_on_too_few_points(self):
        with self.assertRaises(ValueError):
            build_tube_mesh(np.zeros((1, 3)), np.ones(1), np.zeros(1))


class TestTubeFaces(unittest.TestCase):
    def test_face_count_matches_expected(self):
        n_points, n_theta = 5, 6
        faces = list(tube_faces(n_points, n_theta))
        # one quad per (ring-segment, theta) pair
        self.assertEqual(len(faces), (n_points - 1) * n_theta)

    def test_vertex_indices_within_bounds(self):
        n_points, n_theta = 4, 5
        max_index = n_points * n_theta - 1
        for _i, quad in tube_faces(n_points, n_theta):
            for idx in quad:
                self.assertGreaterEqual(idx, 0)
                self.assertLessEqual(idx, max_index)

    def test_ring_index_wraps_around(self):
        # last theta index (n_theta - 1) should connect back to vertex 0 of the ring
        n_points, n_theta = 3, 4
        faces = list(tube_faces(n_points, n_theta))
        last_j_face = faces[n_theta - 1]  # i=0, j=n_theta-1
        _i, (v00, v10, v11, v01) = last_j_face
        self.assertEqual(v00, 0 * n_theta + (n_theta - 1))
        self.assertEqual(v01, 0 * n_theta + 0)  # wraps to j=0


if __name__ == "__main__":
    unittest.main()
