#!/usr/bin/env python3
"""
Reproduce CHAP's classic "time-averaged pathway profile" plot, as shown
at https://www.channotation.org/docs/plotting_python/:

  * the mean pore radius along the permeation pathway, drawn as a black
    line;
  * a grey shaded band showing the +/-1 standard deviation confidence
    interval around that mean;
  * a scatter of pore-facing residues, positioned at their mean (s, rho)
    -- pathway coordinate and radial distance from the centreline --
    and coloured by hydrophobicity using a diverging colormap
    ("BrBG_r", matching CHAP's own choice) with symmetric colour limits
    around zero, plus a colorbar.

This mirrors the structure of the example code on that page (which uses
`data["pathwayProfile"]["s"]`, `data["pathwayProfile"]["radiusMean"]`,
`data["residueSummary"][...]`, etc.) but reads pychap's own JSON schema
(`s_grid_normalised` / `radius_mean_angstrom` / `residue_summary`, as
written by PoreAnalysisResult.save_json()) rather than CHAP's.

Usage
-----
    python plot_pathway_profile.py pore_profile.json -o pathway_profile.png
    python plot_pathway_profile.py pore_profile.json -o pathway_profile.png --min-pore-facing-fraction 0.5
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


def plot_pathway_profile(
    data: dict,
    min_pore_facing_fraction: float = 0.5,
    show_residues: bool = True,
    title: str | None = None,
):
    s_grid = np.array(data["s_grid_normalised"])
    mean_length = data["mean_length_angstrom"]
    # pychap's own s coordinate runs 0 -> mean_length (needed internally
    # for the spline/geometry code); shifting by half the pathway length
    # here re-centres the *plotted* x-axis on the pathway's midpoint --
    # matching upstream CHAP's own plots -- without touching the
    # underlying JSON/CSV data or anything else that consumes it. Residue
    # s-positions (below) get the same shift so they stay aligned with
    # the profile line.
    s = s_grid * mean_length - mean_length / 2.0

    radius_mean = np.array(data["radius_mean_angstrom"])
    radius_std = np.array(data["radius_std_angstrom"])

    fig, ax = plt.subplots(figsize=(7.5, 5))

    # Mean radius as a black line, +/-1 SD as a grey shaded band --
    # matches the CHAP docs example almost verbatim.
    ax.plot(s, radius_mean, "k-", linewidth=1.5, label="mean radius", zorder=4)
    ax.fill_between(
        s,
        radius_mean - radius_std,
        radius_mean + radius_std,
        facecolor="#000000",
        alpha=0.2,
        label="±1 SD",
        zorder=1,
    )

    residues = data.get("residue_summary") or []
    if show_residues and residues:
        s_res = np.array([r["s_mean_angstrom"] for r in residues]) - mean_length / 2.0
        rho_res = np.array([r["rho_mean_angstrom"] for r in residues])
        hydro_res = np.array([r["hydrophobicity_kcalmol"] for r in residues])
        facing_fraction = np.array([r["pore_facing_fraction"] for r in residues])

        pore_facing = facing_fraction > min_pore_facing_fraction
        if np.any(pore_facing):
            hydro_facing = hydro_res[pore_facing]
            clim = float(np.max(np.abs(hydro_facing))) if len(hydro_facing) else 1.0
            clim = clim if clim > 0 else 1.0

            scatter = ax.scatter(
                s_res[pore_facing],
                rho_res[pore_facing],
                c=hydro_facing,
                marker="o",
                cmap="BrBG_r",
                vmin=-clim,
                vmax=clim,
                edgecolor="black",
                linewidth=0.3,
                zorder=5,
                label="pore-facing residues",
            )
            cbar = fig.colorbar(scatter, ax=ax)
            cbar.ax.set_ylabel("Hydrophobicity (kcal/mol, Wimley–White)")

    ax.set_xlabel("Pathway coordinate s (Å)")
    ax.set_ylabel("Pore radius / residue distance from centreline (Å)")
    ax.set_title(title or "Time-averaged pathway profile")
    ax.axhline(0.0, color="grey", linewidth=0.5, zorder=0)
    ax.set_xlim(s.min(), s.max())
    ax.legend(loc="upper center", frameon=False, ncol=3)
    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_json", help="pychap results JSON file (e.g. pore_profile.json)")
    parser.add_argument("-o", "--output", default="pathway_profile.png")
    parser.add_argument("-t", "--title", default=None)
    parser.add_argument(
        "--min-pore-facing-fraction",
        type=float,
        default=0.5,
        help="Only show residues that are pore-facing in more than this fraction of frames (default: 0.5).",
    )
    parser.add_argument(
        "--no-residues", action="store_true", help="Don't overlay the residue hydrophobicity scatter."
    )
    args = parser.parse_args()

    data = load_result(args.input_json)
    fig = plot_pathway_profile(
        data,
        min_pore_facing_fraction=args.min_pore_facing_fraction,
        show_residues=not args.no_residues,
        title=args.title,
    )
    fig.savefig(args.output, dpi=150)
    print(f"Wrote {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
