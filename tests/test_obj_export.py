"""Tests for pychap.obj_export (Wavefront OBJ + MTL pathway mesh export)."""

import os
import tempfile
import unittest

import numpy as np

from pychap.obj_export import export_pathway_obj


class TestExportPathwayObj(unittest.TestCase):
    def setUp(self):
        self.s = np.linspace(0.0, 40.0, 30)
        self.points = np.stack([np.zeros(30), np.zeros(30), self.s], axis=1)
        self.radius = 3.0 + np.sin(self.s / 40.0 * np.pi)
        self.color_values = np.cos(self.s / 40.0 * np.pi * 2)

    def test_writes_obj_and_mtl_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            obj_path = os.path.join(tmp, "pathway.obj")
            obj_out, mtl_out = export_pathway_obj(
                self.points, self.radius, self.s, self.color_values, obj_path, n_theta=12, n_color_bins=8
            )
            self.assertTrue(os.path.exists(obj_out))
            self.assertTrue(os.path.exists(mtl_out))

    def test_vertex_and_face_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            obj_path = os.path.join(tmp, "pathway.obj")
            n_theta = 10
            obj_out, _mtl_out = export_pathway_obj(
                self.points, self.radius, self.s, self.color_values, obj_path, n_theta=n_theta
            )
            with open(obj_out) as fh:
                lines = fh.readlines()
            n_vertices = sum(1 for l in lines if l.startswith("v "))
            n_normals = sum(1 for l in lines if l.startswith("vn "))
            n_faces = sum(1 for l in lines if l.startswith("f "))
            self.assertEqual(n_vertices, len(self.points) * n_theta)
            self.assertEqual(n_normals, len(self.points) * n_theta)
            self.assertEqual(n_faces, (len(self.points) - 1) * n_theta * 2)

    def test_vertex_normals_are_unit_length_and_radially_outward(self):
        with tempfile.TemporaryDirectory() as tmp:
            obj_path = os.path.join(tmp, "pathway.obj")
            n_theta = 12
            obj_out, _mtl_out = export_pathway_obj(
                self.points, self.radius, self.s, self.color_values, obj_path, n_theta=n_theta
            )
            with open(obj_out) as fh:
                lines = fh.readlines()
            vertices = [tuple(map(float, l.split()[1:4])) for l in lines if l.startswith("v ")]
            normals = [tuple(map(float, l.split()[1:4])) for l in lines if l.startswith("vn ")]
            self.assertEqual(len(vertices), len(normals))
            # Each ring vertex's normal should point from the corresponding
            # centreline point straight out to that vertex (this synthetic
            # pathway runs along z, so the centreline point for ring i is
            # simply self.points[i] // n_theta apart -- reconstruct that
            # mapping the same way the exporter orders vertices/normals.
            n_points = len(self.points)
            for i in range(n_points):
                centre = self.points[i]
                for j in range(n_theta):
                    idx = i * n_theta + j
                    vx, vy, vz = vertices[idx]
                    nx, ny, nz = normals[idx]
                    self.assertAlmostEqual(nx * nx + ny * ny + nz * nz, 1.0, places=3)
                    offset = np.array([vx, vy, vz]) - centre
                    offset_len = np.linalg.norm(offset)
                    if offset_len > 1e-6:
                        expected = offset / offset_len
                        np.testing.assert_allclose([nx, ny, nz], expected, atol=1e-3)

    def test_mtllib_reference_matches_mtl_filename(self):
        with tempfile.TemporaryDirectory() as tmp:
            obj_path = os.path.join(tmp, "pathway.obj")
            obj_out, mtl_out = export_pathway_obj(
                self.points, self.radius, self.s, self.color_values, obj_path, n_theta=8
            )
            with open(obj_out) as fh:
                content = fh.read()
            self.assertIn(f"mtllib {os.path.basename(mtl_out)}", content)

    def test_mtl_contains_expected_number_of_materials(self):
        with tempfile.TemporaryDirectory() as tmp:
            obj_path = os.path.join(tmp, "pathway.obj")
            n_bins = 16
            _obj_out, mtl_out = export_pathway_obj(
                self.points,
                self.radius,
                self.s,
                self.color_values,
                obj_path,
                n_theta=8,
                n_color_bins=n_bins,
            )
            with open(mtl_out) as fh:
                content = fh.read()
            n_materials = content.count("newmtl")
            self.assertEqual(n_materials, n_bins)

    def test_all_face_vertex_indices_are_valid_and_1_indexed(self):
        with tempfile.TemporaryDirectory() as tmp:
            obj_path = os.path.join(tmp, "pathway.obj")
            n_theta = 6
            obj_out, _mtl_out = export_pathway_obj(
                self.points, self.radius, self.s, self.color_values, obj_path, n_theta=n_theta
            )
            with open(obj_out) as fh:
                lines = fh.readlines()
            n_vertices = len(self.points) * n_theta
            n_face_lines = 0
            for line in lines:
                if line.startswith("f "):
                    n_face_lines += 1
                    for tok in line.split()[1:]:
                        # each token is "v_idx//vn_idx" (no texture coord)
                        v_part, sep, vn_part = tok.partition("//")
                        self.assertEqual(sep, "//", f"unexpected face token format: {tok!r}")
                        v_idx = int(v_part)
                        vn_idx = int(vn_part)
                        self.assertGreaterEqual(v_idx, 1)
                        self.assertLessEqual(v_idx, n_vertices)
                        # normal index intentionally matches vertex index --
                        # see the comment in obj_export.py's face writer.
                        self.assertEqual(vn_idx, v_idx)
            self.assertGreater(n_face_lines, 0)

    def test_raises_on_mismatched_lengths(self):
        with tempfile.TemporaryDirectory() as tmp:
            obj_path = os.path.join(tmp, "pathway.obj")
            with self.assertRaises(ValueError):
                export_pathway_obj(
                    self.points, self.radius[:-1], self.s, self.color_values, obj_path
                )

    def test_raises_on_too_few_points(self):
        with tempfile.TemporaryDirectory() as tmp:
            obj_path = os.path.join(tmp, "pathway.obj")
            with self.assertRaises(ValueError):
                export_pathway_obj(
                    self.points[:1], self.radius[:1], self.s[:1], self.color_values[:1], obj_path
                )


if __name__ == "__main__":
    unittest.main()
