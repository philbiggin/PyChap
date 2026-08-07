"""
High-level pore analysis orchestration.

:class:`PoreAnalysis` ties together trajectory reading
(:mod:`pychap.trajectory`), pathway/radius-profile computation
(:mod:`pychap.pathfinding`), and hydrophobicity mapping
(:mod:`pychap.hydrophobicity`) into a single entry point that mirrors
running the upstream ``chap`` executable on a trajectory: it processes
every requested frame and produces a time-averaged pore radius profile
and hydrophobicity profile, along with per-frame results.

Because the pore length and the exact number of resampled points can
differ slightly from frame to frame (the pathway is refit every frame),
frames are aggregated on a common *normalised* arc-length grid
(fraction of pathway length, 0 to 1) rather than on absolute arc length.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field, asdict

import numpy as np

from .hydrophobicity import hydrophobicity_profile, residue_hydrophobicity
from .pathfinding import compute_pore_profile
from .residues import project_residues
from .trajectory import Trajectory


@dataclass
class FrameResult:
    """Per-frame pathway/radius/hydrophobicity result."""

    frame: int
    s: np.ndarray
    points: np.ndarray
    radius: np.ndarray
    hydrophobicity: np.ndarray
    length: float

    @property
    def s_normalised(self) -> np.ndarray:
        if self.length <= 0:
            return np.zeros_like(self.s)
        return self.s / self.length


@dataclass
class ResidueSummary:
    """Trajectory-averaged summary of a single pore-lining residue's
    position relative to the permeation pathway, analogous to a row of
    CHAP's ``residueSummary`` JSON output.
    """

    resid: int
    resname: str
    hydrophobicity: float       #: constant per-residue value (kcal/mol, Wimley-White by default)
    s_mean: float                #: mean arc-length coordinate along the pathway (Angstrom)
    s_std: float
    rho_mean: float               #: mean radial distance from the pathway centreline (Angstrom)
    rho_std: float
    pore_facing_fraction: float    #: fraction of frames in which rho <= the pore-facing cutoff

    def to_dict(self) -> dict:
        return {
            "resid": self.resid,
            "resname": self.resname,
            "hydrophobicity_kcalmol": self.hydrophobicity,
            "s_mean_angstrom": self.s_mean,
            "s_std_angstrom": self.s_std,
            "rho_mean_angstrom": self.rho_mean,
            "rho_std_angstrom": self.rho_std,
            "pore_facing_fraction": self.pore_facing_fraction,
        }


@dataclass
class PoreAnalysisResult:
    """Trajectory-averaged pore analysis result."""

    frames: list  # list[FrameResult]
    s_grid: np.ndarray               # normalised arc length, 0..1
    mean_length: float                # mean absolute pathway length (Angstrom)
    radius_mean: np.ndarray
    radius_std: np.ndarray
    radius_min: np.ndarray
    radius_max: np.ndarray
    hydrophobicity_mean: np.ndarray
    points_mean: np.ndarray = None  # (n_resample, 3) time-averaged 3D centreline, Angstrom
    residue_summary: list = field(default_factory=list)  # list[ResidueSummary]

    @property
    def min_radius_overall(self) -> float:
        return float(np.min(self.radius_mean))

    @property
    def min_radius_position(self) -> float:
        """Position of the narrowest constriction, in absolute Angstrom along the mean-length pathway."""
        return float(self.s_grid[np.argmin(self.radius_mean)] * self.mean_length)

    def to_dict(self) -> dict:
        return {
            "n_frames": len(self.frames),
            "mean_length_angstrom": self.mean_length,
            "min_radius_angstrom": self.min_radius_overall,
            "min_radius_position_angstrom": self.min_radius_position,
            "s_grid_normalised": self.s_grid.tolist(),
            "radius_mean_angstrom": self.radius_mean.tolist(),
            "radius_std_angstrom": self.radius_std.tolist(),
            "radius_min_angstrom": self.radius_min.tolist(),
            "radius_max_angstrom": self.radius_max.tolist(),
            "hydrophobicity_mean_kcalmol": self.hydrophobicity_mean.tolist(),
            "points_mean_angstrom": self.points_mean.tolist() if self.points_mean is not None else None,
            "residue_summary": [r.to_dict() for r in self.residue_summary],
            "frames": [
                {
                    "frame": fr.frame,
                    "length_angstrom": fr.length,
                    "s_angstrom": fr.s.tolist(),
                    "points_angstrom": fr.points.tolist(),
                    "radius_angstrom": fr.radius.tolist(),
                    "hydrophobicity_kcalmol": fr.hydrophobicity.tolist(),
                }
                for fr in self.frames
            ],
        }

    def save_json(self, path):
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    def save_csv(self, path):
        """Write the trajectory-averaged profile (not per-frame data) as CSV."""
        with open(path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                [
                    "s_normalised",
                    "s_angstrom",
                    "radius_mean_angstrom",
                    "radius_std_angstrom",
                    "radius_min_angstrom",
                    "radius_max_angstrom",
                    "hydrophobicity_mean_kcalmol",
                ]
            )
            for i in range(len(self.s_grid)):
                writer.writerow(
                    [
                        self.s_grid[i],
                        self.s_grid[i] * self.mean_length,
                        self.radius_mean[i],
                        self.radius_std[i],
                        self.radius_min[i],
                        self.radius_max[i],
                        self.hydrophobicity_mean[i],
                    ]
                )

    def save_residue_summary_csv(self, path):
        """Write the per-residue pathway summary (s, rho, hydrophobicity,
        pore-facing fraction) as CSV -- the data used to overlay residues
        on the radius profile plot (see scripts/plot_pathway_profile.py)."""
        with open(path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                [
                    "resid",
                    "resname",
                    "hydrophobicity_kcalmol",
                    "s_mean_angstrom",
                    "s_std_angstrom",
                    "rho_mean_angstrom",
                    "rho_std_angstrom",
                    "pore_facing_fraction",
                ]
            )
            for r in self.residue_summary:
                writer.writerow(
                    [
                        r.resid,
                        r.resname,
                        r.hydrophobicity,
                        r.s_mean,
                        r.s_std,
                        r.rho_mean,
                        r.rho_std,
                        r.pore_facing_fraction,
                    ]
                )


def build_residue_summary(
    resids, resnames, s_frames, rho_frames, facing_frames, hydrophobicity_scale=None
) -> list:
    """Aggregate per-frame residue projections (see :mod:`pychap.residues`)
    into a list of trajectory-averaged :class:`ResidueSummary` objects.

    Parameters
    ----------
    resids, resnames:
        Length-``R`` sequences identifying each residue (assumed to be in
        the same order in every frame, which holds as long as the same
        static selection is used throughout the trajectory).
    s_frames, rho_frames, facing_frames:
        Length-``n_frames`` sequences of length-``R`` arrays (one entry
        per frame), as produced by :func:`pychap.residues.project_residues`.
    """
    s_stack = np.asarray(s_frames, dtype=float)
    rho_stack = np.asarray(rho_frames, dtype=float)
    facing_stack = np.asarray(facing_frames, dtype=float)

    summaries = []
    for i, (resid, resname) in enumerate(zip(resids, resnames)):
        summaries.append(
            ResidueSummary(
                resid=int(resid),
                resname=resname,
                hydrophobicity=residue_hydrophobicity(resname, hydrophobicity_scale),
                s_mean=float(s_stack[:, i].mean()),
                s_std=float(s_stack[:, i].std()),
                rho_mean=float(rho_stack[:, i].mean()),
                rho_std=float(rho_stack[:, i].std()),
                pore_facing_fraction=float(facing_stack[:, i].mean()),
            )
        )
    return summaries


class PoreAnalysis:
    """Run a CHAP-style pore pathway/radius-profile analysis over a trajectory.

    Parameters
    ----------
    topology:
        Path to a topology file (``.gro``, ``.pdb``, ``.tpr``, ...).
    trajectory:
        Optional path to a trajectory file (e.g. a GROMACS ``.xtc``).
        If omitted, only the single structure in ``topology`` is analysed.
    selection:
        MDAnalysis selection string for the pore-lining atoms to
        consider (e.g. ``"protein"``).
    axis:
        Cartesian axis (0=x, 1=y, 2=z) treated as the approximate pore /
        membrane-normal axis. Defaults to z.
    n_slices:
        Number of coarse slices used for the initial centreline search.
    n_resample:
        Number of points in the final, smoothly resampled radius profile.
    window:
        Distance (Angstrom) along ``axis`` within which atoms are
        considered "local" to a given slice during the search.
    hydrophobicity_sigma:
        Width (Angstrom) of the Gaussian kernel used to map residue
        hydrophobicity onto the pathway.
    hydrophobicity_scale:
        Optional custom hydrophobicity scale dict (defaults to
        Wimley-White interfacial scale).
    pore_facing_cutoff:
        Distance (Angstrom) from the pathway centreline within which a
        residue is considered "pore-facing" in a given frame (see
        :mod:`pychap.residues`).
    """

    def __init__(
        self,
        topology,
        trajectory=None,
        selection: str = "protein",
        axis: int = 2,
        n_slices: int = 50,
        n_resample: int = 200,
        window: float = 15.0,
        hydrophobicity_sigma: float = 5.0,
        hydrophobicity_scale: dict | None = None,
        pore_facing_cutoff: float = 12.0,
    ):
        self.traj = Trajectory(topology, trajectory, selection)
        self.axis = axis
        self.n_slices = n_slices
        self.n_resample = n_resample
        self.window = window
        self.hydrophobicity_sigma = hydrophobicity_sigma
        self.hydrophobicity_scale = hydrophobicity_scale
        self.pore_facing_cutoff = pore_facing_cutoff
        self._radii = self.traj.vdw_radii()

    def run(self, start: int = 0, stop: int | None = None, step: int = 1) -> PoreAnalysisResult:
        frame_results = []
        seed_uv = (0.0, 0.0)

        residue_ids = None
        residue_names = None
        residue_s_frames = []
        residue_rho_frames = []
        residue_facing_frames = []

        for frame_index, positions in self.traj.iter_frames(start, stop, step):
            profile = compute_pore_profile(
                positions,
                self._radii,
                axis=self.axis,
                n_slices=self.n_slices,
                n_resample=self.n_resample,
                window=self.window,
                seed_uv=seed_uv,
            )
            seed_uv = profile.last_uv  # warm-start next frame for a smoother trajectory

            res_positions, res_names, res_ids = self.traj.residue_centers()
            if residue_names is None:
                residue_names, residue_ids = res_names, res_ids
            res_values = np.array(
                [residue_hydrophobicity(name, self.hydrophobicity_scale) for name in res_names]
            )
            hydro = hydrophobicity_profile(
                profile.points, res_positions, res_values, sigma=self.hydrophobicity_sigma
            )

            projection = project_residues(res_positions, profile, pore_facing_cutoff=self.pore_facing_cutoff)
            residue_s_frames.append(projection.s)
            residue_rho_frames.append(projection.rho)
            residue_facing_frames.append(projection.pore_facing)

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

        result = aggregate_frame_results(frame_results, self.n_resample)
        if residue_names is not None:
            result.residue_summary = build_residue_summary(
                residue_ids,
                residue_names,
                residue_s_frames,
                residue_rho_frames,
                residue_facing_frames,
                self.hydrophobicity_scale,
            )
        return result


def aggregate_frame_results(frame_results, n_resample: int) -> PoreAnalysisResult:
    """Combine a list of per-frame :class:`FrameResult` objects into a
    trajectory-averaged :class:`PoreAnalysisResult`.

    Pulled out as a standalone function (rather than a method) so it can
    be exercised directly in tests without needing to construct a full
    :class:`PoreAnalysis` (and therefore without needing MDAnalysis or an
    actual trajectory file on disk).

    In addition to the scalar radius/hydrophobicity statistics, this also
    computes a time-averaged 3D centreline (``points_mean``) by
    interpolating each frame's (x, y, z) coordinates onto the common
    normalised arc-length grid and averaging. This assumes the pathway
    doesn't jump between very different geometric solutions from frame to
    frame -- true as long as ``PoreAnalysis.run()``'s warm-starting keeps
    consecutive frames' centrelines close to each other, which is the
    normal case for a channel that isn't undergoing a large conformational
    change during the trajectory.
    """
    if not frame_results:
        raise ValueError("no frames were analysed")

    s_grid = np.linspace(0.0, 1.0, n_resample)
    radius_stack = np.empty((len(frame_results), n_resample))
    hydro_stack = np.empty((len(frame_results), n_resample))
    points_stack = np.empty((len(frame_results), n_resample, 3))
    lengths = np.empty(len(frame_results))

    for i, fr in enumerate(frame_results):
        radius_stack[i] = np.interp(s_grid, fr.s_normalised, fr.radius)
        hydro_stack[i] = np.interp(s_grid, fr.s_normalised, fr.hydrophobicity)
        for dim in range(3):
            points_stack[i, :, dim] = np.interp(s_grid, fr.s_normalised, fr.points[:, dim])
        lengths[i] = fr.length

    return PoreAnalysisResult(
        frames=frame_results,
        s_grid=s_grid,
        mean_length=float(lengths.mean()),
        radius_mean=radius_stack.mean(axis=0),
        radius_std=radius_stack.std(axis=0),
        radius_min=radius_stack.min(axis=0),
        radius_max=radius_stack.max(axis=0),
        hydrophobicity_mean=hydro_stack.mean(axis=0),
        points_mean=points_stack.mean(axis=0),
    )
