"""
End-to-end integration test of the full analysis pipeline (pathfinding +
hydrophobicity + trajectory-averaging), run entirely in-memory against a
multi-"frame" synthetic hourglass pore. This does not require MDAnalysis
(the frames are plain NumPy arrays rather than a real trajectory file),
so it exercises the full scientific pipeline in any environment,
including ones where MDAnalysis cannot be installed.

Reading a *real* GROMACS .xtc file goes through the same
compute_pore_profile / hydrophobicity_profile / aggregate_frame_results
functions used here -- only the source of the per-frame coordinate
arrays differs (pychap.trajectory.Trajectory instead of an in-memory
array) -- see tests/test_trajectory.py and examples/run_example.py for
that path.
"""

import unittest

import numpy as np

from pychap.analysis import FrameResult, aggregate_frame_results
from pychap.hydrophobicity import hydrophobicity_profile, residue_hydrophobicity
from pychap.pathfinding import compute_pore_profile
from pychap.testing import hourglass_pore, hourglass_pore_residues


def run_synthetic_trajectory(n_frames=5, breathing_amplitude=0.3, seed=0):
    """Simulate a short 'trajectory' of a breathing hourglass pore and run
    the full pychap pipeline over it frame by frame, exactly as
    PoreAnalysis.run() would (but without needing MDAnalysis)."""
    rng = np.random.default_rng(seed)
    frame_results = []
    seed_uv = (0.0, 0.0)

    for frame_index in range(n_frames):
        # Small random "breathing" perturbation of the constriction radius,
        # mimicking thermal fluctuation of a channel gate across frames.
        r_min = 3.0 + breathing_amplitude * rng.normal()
        positions, radii, ring_z, ring_radius = hourglass_pore(
            r_min=r_min, r_max=10.0, n_rings=41, n_per_ring=24, z_min=-20.0, z_max=20.0, vdw_radius=1.0
        )
        res_positions, res_names = hourglass_pore_residues(ring_z, ring_radius)
        res_values = np.array([residue_hydrophobicity(name) for name in res_names])

        profile = compute_pore_profile(
            positions, radii, axis=2, n_slices=41, n_resample=80, window=3.0, seed_uv=seed_uv
        )
        seed_uv = profile.last_uv

        hydro = hydrophobicity_profile(profile.points, res_positions, res_values, sigma=6.0)

        frame_results.append(
            FrameResult(
                frame=frame_index,
                s=profile.s,
                points=profile.points,
                radius=profile.radius,
                hydrophobicity=hydro,
                length=profile.length,
            )
        )

    return aggregate_frame_results(frame_results, n_resample=80)


class TestFullPipelineOnSyntheticTrajectory(unittest.TestCase):
    def setUp(self):
        self.result = run_synthetic_trajectory(n_frames=5)

    def test_produces_one_frame_result_per_frame(self):
        self.assertEqual(len(self.result.frames), 5)

    def test_min_radius_is_near_constriction_value(self):
        # r_min averages to ~3.0 across frames, so the expected pore radius
        # at the constriction (with 1.0 A vdw radius) is ~2.0 A.
        self.assertAlmostEqual(self.result.min_radius_overall, 2.0, delta=0.3)

    def test_min_radius_position_is_near_the_middle_of_the_pathway(self):
        # The constriction is at z=0, i.e. the middle of the 40 A pathway.
        self.assertAlmostEqual(self.result.min_radius_position, self.result.mean_length / 2, delta=3.0)

    def test_hydrophobicity_profile_is_more_negative_near_constriction(self):
        # hourglass_pore_residues assigns LEU (hydrophobic, -0.56) near the
        # constriction and ASP (hydrophilic, +1.23) further away, so the
        # averaged hydrophobicity profile should dip in the middle.
        mid = len(self.result.hydrophobicity_mean) // 2
        edge_value = self.result.hydrophobicity_mean[5]
        mid_value = self.result.hydrophobicity_mean[mid]
        self.assertLess(mid_value, edge_value)

    def test_radius_profile_is_reasonably_reproducible_across_frames(self):
        # radius_std should be small relative to the breathing amplitude
        # scale, i.e. frames are broadly consistent with each other.
        self.assertTrue(np.all(self.result.radius_std < 2.0))

    def test_json_and_csv_export_do_not_raise(self):
        import json
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            json_path = os.path.join(tmp, "out.json")
            csv_path = os.path.join(tmp, "out.csv")
            self.result.save_json(json_path)
            self.result.save_csv(csv_path)
            self.assertTrue(os.path.getsize(json_path) > 0)
            self.assertTrue(os.path.getsize(csv_path) > 0)
            with open(json_path) as fh:
                data = json.load(fh)
            self.assertEqual(data["n_frames"], 5)


if __name__ == "__main__":
    unittest.main()
