#!/usr/bin/env python3
"""
Plot a pore radius profile (radius vs. position along the permeation
pathway) from a pychap results JSON file, in the style of CHAP's
classic "radius vs. pore coordinate" plot.

Usage
-----
    python plot_radius_profile.py pore_profile.json -o radius_profile.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # safe for headless / scripted use
import matplotlib.pyplot as plt
import numpy as np


def load_result(json_path):
    with open(json_path) as fh:
        return json.load(fh)


def plot_radius_profile(data: dict, title: str | None = None):
    s_grid = np.array(data["s_grid_normalised"])
    mean_length = data["mean_length_angstrom"]
    # pychap's own s coordinate runs 0 -> mean_length (needed internally
    # for the spline/geometry code); shifting by half the pathway length
    # here re-centres the *plotted* x-axis on the pathway's midpoint --
    # matching upstream CHAP's own plots -- without touching the
    # underlying JSON/CSV data or anything else that consumes it.
    s_angstrom = s_grid * mean_length - mean_length / 2.0

    radius_mean = np.array(data["radius_mean_angstrom"])
    radius_min = np.array(data["radius_min_angstrom"])
    radius_max = np.array(data["radius_max_angstrom"])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.fill_between(
        s_angstrom, radius_min, radius_max, color="#4C72B0", alpha=0.25, label="min–max across frames"
    )
    ax.plot(s_angstrom, radius_mean, color="#4C72B0", linewidth=2, label="mean radius")

    min_idx = int(np.argmin(radius_mean))
    ax.scatter(
        [s_angstrom[min_idx]],
        [radius_mean[min_idx]],
        color="#C44E52",
        zorder=5,
        label=f"constriction ({radius_mean[min_idx]:.2f} Å)",
    )

    ax.set_xlabel("Pathway coordinate s (Å)")
    ax.set_ylabel("Pore radius (Å)")
    ax.set_title(title or "Pore radius profile")
    ax.axhline(0.0, color="grey", linewidth=0.5)
    ax.legend(loc="best", frameon=False)
    ax.set_xlim(s_angstrom.min(), s_angstrom.max())
    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_json", help="pychap results JSON file (e.g. pore_profile.json)")
    parser.add_argument("-o", "--output", default="radius_profile.png")
    parser.add_argument("-t", "--title", default=None)
    args = parser.parse_args()

    data = load_result(args.input_json)
    fig = plot_radius_profile(data, title=args.title)
    fig.savefig(args.output, dpi=150)
    print(f"Wrote {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
