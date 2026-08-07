"""Tests for pychap.analysis (trajectory-averaging and export), without needing
MDAnalysis or any file I/O -- FrameResult objects are constructed directly.

The PoreAnalysis-with-MDAnalysis tests near the bottom are the exception:
they're skipped automatically if MDAnalysis isn't installed, same pattern
as tests/test_trajectory.py."""

import json
import os
import tempfile
import unittest

import numpy as np

from pychap.analysis import FrameResult, PoreAnalysis, aggregate_frame_results, plane_centroid_uv

try:
    import MDAnalysis  # noqa: F401

    HAS_MDANALYSIS = True
except ImportError:
    HAS_MDANALYSIS = False


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


class TestPlaneCentroidUV(unittest.TestCase):
    """plane_centroid_uv is what PoreAnalysis.run() now uses to auto-seed
    the pathway search instead of always starting at the origin -- see the
    off-centre-structure regression tests below for why that matters."""

    def test_centroid_of_symmetric_points_around_origin_is_zero(self):
        positions = np.array(
            [[1.0, 0.0, 5.0], [-1.0, 0.0, 5.0], [0.0, 1.0, -5.0], [0.0, -1.0, -5.0]]
        )
        u, v = plane_centroid_uv(positions, axis=2)
        self.assertAlmostEqual(u, 0.0, places=8)
        self.assertAlmostEqual(v, 0.0, places=8)

    def test_centroid_reflects_a_large_offset(self):
        # Same shape as above, but translated far from the origin in the
        # x/y plane -- exactly the "real structure in absolute simulation
        # box coordinates" scenario that broke the old seed_uv=(0, 0) default.
        offset = np.array([500.0, -300.0, 0.0])
        positions = (
            np.array([[1.0, 0.0, 5.0], [-1.0, 0.0, 5.0], [0.0, 1.0, -5.0], [0.0, -1.0, -5.0]])
            + offset
        )
        u, v = plane_centroid_uv(positions, axis=2)
        self.assertAlmostEqual(u, 500.0, places=8)
        self.assertAlmostEqual(v, -300.0, places=8)

    def test_respects_which_axis_is_excluded(self):
        positions = np.array([[10.0, 20.0, 30.0], [10.0, 20.0, 90.0]])
        # axis=0 (x) excluded -> centroid is over (y, z)
        u, v = plane_centroid_uv(positions, axis=0)
        self.assertAlmostEqual(u, 20.0, places=8)
        self.assertAlmostEqual(v, 60.0, places=8)


def _write_gro(path, positions_angstrom, resnames, resids, atomnames, box_nm=(200.0, 200.0, 200.0)):
    """Write a minimal, valid GROMACS .gro file (plain text -- no MDAnalysis needed
    to *write* one, only to *read* it back via pychap.trajectory.Trajectory)."""
    positions_nm = np.asarray(positions_angstrom) / 10.0
    with open(path, "w") as fh:
        fh.write("pychap off-centre synthetic pore test structure\n")
        fh.write(f"{len(positions_nm)}\n")
        for i, (pos, resname, resid, atomname) in enumerate(
            zip(positions_nm, resnames, resids, atomnames), start=1
        ):
            fh.write(
                f"{resid % 100000:5d}{resname:<5.5s}{atomname:>5.5s}{i % 100000:5d}"
                f"{pos[0]:8.3f}{pos[1]:8.3f}{pos[2]:8.3f}\n"
            )
        fh.write(f"{box_nm[0]:10.5f}{box_nm[1]:10.5f}{box_nm[2]:10.5f}\n")


@unittest.skipUnless(HAS_MDANALYSIS, "MDAnalysis not installed in this environment")
class TestPoreAnalysisAutoSeedsOffCentreStructure(unittest.TestCase):
    """Regression test for the bug reported after installing and running
    the `pychap` CLI on the real, off-centre 4PIR structure: PoreAnalysis
    used to always seed the pathway search at (0, 0), which is a fine
    default for the synthetic examples (built centred on the origin) but
    silently produces a nonsensical, wildly-inflated pore radius on any
    structure given in absolute simulation-box coordinates -- the search
    starts far outside the structure entirely, in empty space. This builds
    exactly that scenario (a cylindrical pore of known radius, translated
    far from the origin) and checks the *default* PoreAnalysis call (no
    manual seed_uv) still finds the right answer, plus that an explicit
    seed_uv override also still works.
    """

    def setUp(self):
        from pychap.testing import cylindrical_pore

        self.true_radius = 7.0  # 8.0 A pore radius - 1.0 A vdw radius
        positions, _radii = cylindrical_pore(
            radius=8.0, n_rings=21, n_per_ring=24, z_min=-20.0, z_max=20.0, vdw_radius=1.0
        )
        # translate far from the origin in the plane perpendicular to the
        # pore axis (z) -- like a real structure's absolute box coordinates
        self.offset = np.array([437.0, -612.0, 0.0])
        positions = positions + self.offset
        n = len(positions)
        resnames = ["ALA"] * n
        resids = list(range(1, n + 1))
        atomnames = ["CA"] * n

        self.tmpdir = tempfile.TemporaryDirectory()
        self.gro_path = os.path.join(self.tmpdir.name, "offcentre.gro")
        _write_gro(self.gro_path, positions, resnames, resids, atomnames)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_default_auto_seeding_finds_the_true_radius(self):
        analysis = PoreAnalysis(
            self.gro_path, selection="all", axis=2, n_slices=21, n_resample=50, window=5.0
        )
        result = analysis.run()
        self.assertAlmostEqual(result.min_radius_overall, self.true_radius, delta=1.0)

    def test_explicit_seed_uv_override_also_works(self):
        analysis = PoreAnalysis(
            self.gro_path,
            selection="all",
            axis=2,
            n_slices=21,
            n_resample=50,
            window=5.0,
            seed_uv=(float(self.offset[0]), float(self.offset[1])),
        )
        result = analysis.run()
        self.assertAlmostEqual(result.min_radius_overall, self.true_radius, delta=1.0)

    def test_bad_manual_seed_reproduces_the_original_bug(self):
        # Sanity check that this test scenario really does exercise the bug:
        # seeding at the origin (as PoreAnalysis unconditionally did before
        # this fix) on this off-centre structure should NOT find the true
        # radius -- confirming the auto-seeded tests above are actually
        # testing something real, not passing by coincidence.
        analysis = PoreAnalysis(
            self.gro_path,
            selection="all",
            axis=2,
            n_slices=21,
            n_resample=50,
            window=5.0,
            seed_uv=(0.0, 0.0),
        )
        result = analysis.run()
        self.assertGreater(result.min_radius_overall, self.true_radius + 10.0)


if __name__ == "__main__":
    unittest.main()
