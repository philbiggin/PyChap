#!/usr/bin/env python3
"""
Render a 3D "tube" visualisation of the pore permeation pathway, coloured
by local pore radius -- similar in spirit to CHAP's 3D pore visualisation
(which is normally viewed in VMD/PyMOL via a generated surface; this is a
lightweight, dependency-free (beyond matplotlib) approximation of that
using a ring-and-centreline wireframe).

Reads a pathway (points + radius) from a pychap results JSON file. By
default this plots the time-averaged ("mean") pathway -- the same
geometry used by scripts/export_visualisation.py for the VMD OBJ export
-- or a specific frame via --frame N.

Usage
-----
    python plot_pore_3d.py pore_profile.json -o pore_3d.png
    python plot_pore_3d.py pore_profile.json -o pore_3d.png --frame 2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3D projection)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pychap.tube import build_tube_mesh  # noqa: E402


def load_result(json_path):
    with open(json_path) as fh:
        return json.load(fh)


def _get_pathway(data: dict, frame_index):
    """Return (points, radius, s, label) for either the time-averaged
    pathway (frame_index is None / 'mean') or a specific frame."""
    if frame_index is None or frame_index == "mean":
        if "points_mean_angstrom" not in data:
            sys.exit(
                "Input JSON has no time-averaged pathway ('points_mean_angstrom'). "
                "Re-run the analysis with an up-to-date pychap, or pass --frame N "
                "to plot a single frame instead."
            )
        points = np.array(data["points_mean_angstrom"])
        radius = np.array(data["radius_mean_angstrom"])
        s = np.array(data["s_grid_normalised"]) * data["mean_length_angstrom"]
        return points, radius, s, "time-averaged"

    frame = data["frames"][frame_index]
    points = np.array(frame["points_angstrom"])
    radius = np.array(frame["radius_angstrom"])
    s = np.array(frame["s_angstrom"])
    return points, radius, s, f"frame {frame['frame']}"


def plot_pore_3d(data: dict, frame_index=None, title: str | None = None, n_theta: int = 20):
    points, radius, s, label = _get_pathway(data, frame_index)

    rings = build_tube_mesh(points, radius, s, n_theta=n_theta)

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")

    cmap = plt.get_cmap("viridis")
    r_min, r_max = radius.min(), radius.max()
    norm = plt.Normalize(vmin=r_min, vmax=r_max)

    # Draw a subset of cross-sectional rings, coloured by their radius.
    ring_stride = max(1, len(points) // 25)
    for i in range(0, len(points), ring_stride):
        ring = rings[i]
        ax.plot(ring[:, 0], ring[:, 1], ring[:, 2], color=cmap(norm(radius[i])), linewidth=1.0)

    # Longitudinal "cage" lines connecting corresponding ring vertices.
    for j in range(0, n_theta, max(1, n_theta // 8)):
        ax.plot(
            rings[:, j, 0], rings[:, j, 1], rings[:, j, 2], color="grey", alpha=0.35, linewidth=0.6
        )

    # Centreline itself, coloured by radius, drawn on top.
    for i in range(len(points) - 1):
        ax.plot(
            points[i : i + 2, 0],
            points[i : i + 2, 1],
            points[i : i + 2, 2],
            color=cmap(norm(radius[i])),
            linewidth=2.5,
        )

    mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    mappable.set_array(radius)
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label("Pore radius (Å)")

    ax.set_xlabel("x (Å)")
    ax.set_ylabel("y (Å)")
    ax.set_zlabel("z / pathway axis (Å)")
    ax.set_title(title or f"3D pore pathway ({label})")

    # Roughly equal aspect ratio so the tube isn't visually distorted.
    all_pts = rings.reshape(-1, 3)
    max_range = (all_pts.max(axis=0) - all_pts.min(axis=0)).max() / 2.0
    mid = all_pts.mean(axis=0)
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_json", help="pychap results JSON file (e.g. pore_profile.json)")
    parser.add_argument("-o", "--output", default="pore_3d.png")
    parser.add_argument(
        "--frame",
        default="mean",
        help="Which pathway to plot: 'mean' for the time-averaged pathway (default), or a frame index (e.g. 0).",
    )
    parser.add_argument("-t", "--title", default=None)
    parser.add_argument("--n-theta", type=int, default=20, help="Number of points around each ring")
    args = parser.parse_args()

    data = load_result(args.input_json)
    frame_index = None if args.frame == "mean" else int(args.frame)
    if frame_index is not None and ("frames" not in data or not data["frames"]):
        sys.exit("Input JSON has no per-frame pathway data (need 'frames' with 'points_angstrom').")

    fig = plot_pore_3d(data, frame_index=frame_index, title=args.title, n_theta=args.n_theta)
    fig.savefig(args.output, dpi=150)
    print(f"Wrote {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
