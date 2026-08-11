#!/usr/bin/env python3
"""
Plot the hydrophobicity profile along the permeation pathway (mirrors
CHAP's hydrophobicity-annotated pore profile), from a pychap results
JSON file. Positive values (Wimley-White scale, kcal/mol) are more
hydrophilic; negative values are more hydrophobic.

Usage
-----
    python plot_hydrophobicity.py pore_profile.json -o hydrophobicity_profile.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_result(json_path):
    with open(json_path) as fh:
        return json.load(fh)


def plot_hydrophobicity_profile(data: dict, title: str | None = None):
    s_grid = np.array(data["s_grid_normalised"])
    mean_length = data["mean_length_angstrom"]
    # pychap's own s coordinate runs 0 -> mean_length (needed internally
    # for the spline/geometry code); shifting by half the pathway length
    # here re-centres the *plotted* x-axis on the pathway's midpoint --
    # matching upstream CHAP's own plots -- without touching the
    # underlying JSON/CSV data or anything else that consumes it.
    s_angstrom = s_grid * mean_length - mean_length / 2.0
    hydro = np.array(data["hydrophobicity_mean_kcalmol"])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.fill_between(
        s_angstrom, hydro, 0.0, where=(hydro < 0), color="#DD8452", alpha=0.6, interpolate=True,
        label="hydrophobic",
    )
    ax.fill_between(
        s_angstrom, hydro, 0.0, where=(hydro >= 0), color="#4C72B0", alpha=0.6, interpolate=True,
        label="hydrophilic",
    )
    ax.plot(s_angstrom, hydro, color="black", linewidth=1.0)
    ax.axhline(0.0, color="grey", linewidth=0.8)

    ax.set_xlabel("Pathway coordinate s (Å)")
    ax.set_ylabel("Hydrophobicity (kcal/mol, Wimley–White)")
    ax.set_title(title or "Hydrophobicity profile along permeation pathway")
    ax.legend(loc="best", frameon=False)
    ax.set_xlim(s_angstrom.min(), s_angstrom.max())
    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_json", help="pychap results JSON file (e.g. pore_profile.json)")
    parser.add_argument("-o", "--output", default="hydrophobicity_profile.png")
    parser.add_argument("-t", "--title", default=None)
    args = parser.parse_args()

    data = load_result(args.input_json)
    fig = plot_hydrophobicity_profile(data, title=args.title)
    fig.savefig(args.output, dpi=150)
    print(f"Wrote {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
