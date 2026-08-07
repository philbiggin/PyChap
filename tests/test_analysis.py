"""Tests for pychap.analysis (trajectory-averaging and export), without needing
MDAnalysis or any file I/O -- FrameResult objects are constructed directly."""

import json
import os
import tempfile
import unittest

import numpy as np

from pychap.analysis import FrameResult, aggregate_frame_results


def make_frame(frame_index, length, radius_value, hydro_value, n=20):
    s = np.linspace(0.0, length, n)
    points = np.stack([np.zeros(n), np.zeros(n), s], axis=1)
    radius = np.full(n, radius_value)
    hydro = np.full(n, hydro_value)
    return FrameResult(
        frame=frame_index, s=s, points=points, radius=radius, hydrophobicity=hydro, length=length
    )


class TestAggregateFrameResults(unittest.TestCase):
    def test_single_frame_reproduces_its_own_values(self):
        frame = make_frame(0, length=40.0, radius_value=5.0, hydro_value=-0.5)
        result = aggregate_frame_results([frame], n_resample=20)
        np.testing.assert_allclose(result.radius_mean, 5.0, atol=1e-8)
        np.testing.assert_allclose(result.radius_std, 0.0, atol=1e-8)
        np.testing.assert_allclose(result.hydrophobicity_mean, -0.5, atol=1e-8)
        self.assertAlmostEqual(result.mean_length, 40.0)

    def test_averages_across_multiple_frames(self):
        frames = [
            make_frame(0, length=40.0, radius_value=4.0, hydro_value=1.0),
            make_frame(1, length=40.0, radius_value=6.0, hydro_value=-1.0),
        ]
        result = aggregate_frame_results(frames, n_resample=20)
        np.testing.assert_allclose(result.radius_mean, 5.0, atol=1e-8)
        np.testing.assert_allclose(result.radius_min, 4.0, atol=1e-8)
        np.testing.assert_allclose(result.radius_max, 6.0, atol=1e-8)
        np.testing.assert_allclose(result.hydrophobicity_mean, 0.0, atol=1e-8)

    def test_empty_frame_list_raises(self):
        with self.assertRaises(ValueError):
            aggregate_frame_results([], n_resample=20)

    def test_min_radius_and_position_helpers(self):
        n = 20
        s = np.linspace(0.0, 40.0, n)
        # radius dips to a minimum in the middle
        radius = 5.0 - 2.0 * np.exp(-((s - 20.0) ** 2) / 10.0)
        frame = FrameResult(
            frame=0,
            s=s,
            points=np.zeros((n, 3)),
            radius=radius,
            hydrophobicity=np.zeros(n),
            length=40.0,
        )
        result = aggregate_frame_results([frame], n_resample=n)
        self.assertAlmostEqual(result.min_radius_overall, radius.min(), places=6)
        self.assertGreater(result.min_radius_position, 15.0)
        self.assertLess(result.min_radius_position, 25.0)


class TestPoreAnalysisResultExport(unittest.TestCase):
    def setUp(self):
        frames = [
            make_frame(0, length=40.0, radius_value=5.0, hydro_value=0.2),
            make_frame(1, length=42.0, radius_value=5.5, hydro_value=-0.1),
        ]
        self.result = aggregate_frame_results(frames, n_resample=25)

    def test_to_dict_has_expected_keys(self):
        d = self.result.to_dict()
        for key in (
            "n_frames",
            "mean_length_angstrom",
            "min_radius_angstrom",
            "min_radius_position_angstrom",
            "s_grid_normalised",
            "radius_mean_angstrom",
            "radius_std_angstrom",
            "hydrophobicity_mean_kcalmol",
            "frames",
        ):
            self.assertIn(key, d)
        self.assertEqual(d["n_frames"], 2)
        self.assertEqual(len(d["frames"]), 2)

    def test_save_json_round_trips(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "result.json")
            self.result.save_json(path)
            with open(path) as fh:
                loaded = json.load(fh)
            self.assertEqual(loaded["n_frames"], 2)
            np.testing.assert_allclose(
                loaded["radius_mean_angstrom"], self.result.radius_mean.tolist()
            )

    def test_save_csv_has_header_and_correct_row_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "result.csv")
            self.result.save_csv(path)
            with open(path) as fh:
                lines = fh.readlines()
            self.assertIn("radius_mean_angstrom", lines[0])
            self.assertEqual(len(lines) - 1, len(self.result.s_grid))


if __name__ == "__main__":
    unittest.main()
