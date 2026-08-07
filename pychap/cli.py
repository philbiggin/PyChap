"""
Command line interface for pychap.

Loosely modelled on the original ``chap`` executable's options
(``-s`` structure/topology, ``-f`` trajectory, ``-sel-pathway`` atom
selection), so users familiar with upstream CHAP will find this
comfortable, even though the underlying algorithm is a simplified
Python reimplementation.

Example
-------
    pychap -s topol.gro -f traj.xtc -sel "protein" -o results \\
        -axis z -n-slices 50 -n-resample 200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .analysis import PoreAnalysis

_AXIS_MAP = {"x": 0, "y": 1, "z": 2}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pychap",
        description=(
            "Compute a CHAP-style pore permeation pathway and radius profile "
            "from a GROMACS (or other MDAnalysis-readable) trajectory."
        ),
    )
    parser.add_argument(
        "-s", "--structure", required=True, help="Topology/structure file (.gro, .pdb, .tpr, ...)"
    )
    parser.add_argument(
        "-f", "--trajectory", default=None, help="Trajectory file (.xtc, .trr, ...). Optional."
    )
    parser.add_argument(
        "-sel",
        "--selection",
        default="protein",
        help="MDAnalysis atom selection string for pore-lining atoms (default: 'protein').",
    )
    parser.add_argument(
        "-o", "--outdir", default="pychap_output", help="Output directory for results."
    )
    parser.add_argument(
        "-axis",
        choices=("x", "y", "z"),
        default="z",
        help="Cartesian axis treated as the pore/membrane-normal axis (default: z).",
    )
    parser.add_argument(
        "-n-slices", type=int, default=50, help="Number of coarse centreline search slices."
    )
    parser.add_argument(
        "-n-resample", type=int, default=200, help="Number of points in the final radius profile."
    )
    parser.add_argument(
        "-window",
        type=float,
        default=15.0,
        help="Distance (Angstrom) along the axis defining 'local' atoms per slice.",
    )
    parser.add_argument(
        "-hydrophobicity-sigma",
        type=float,
        default=5.0,
        help="Width (Angstrom) of the Gaussian kernel used for hydrophobicity mapping.",
    )
    parser.add_argument(
        "-pore-facing-cutoff",
        type=float,
        default=12.0,
        help="Distance (Angstrom) from the pathway centreline within which a residue "
        "counts as pore-facing (used for the residue summary / pathway profile plot).",
    )
    parser.add_argument(
        "-b", "--start", type=int, default=0, help="First frame to analyse (0-indexed)."
    )
    parser.add_argument("-e", "--stop", type=int, default=None, help="Last frame (exclusive).")
    parser.add_argument("-step", type=int, default=1, help="Frame stride.")
    parser.add_argument(
        "-seed-uv",
        type=float,
        nargs=2,
        default=None,
        metavar=("U", "V"),
        help="Starting point (u, v) for the pathway search, in the plane perpendicular "
        "to -axis. Default: computed automatically from the selected atoms' own centroid "
        "in that plane, which is usually right -- only pass this if that guess is poor "
        "(e.g. a strongly asymmetric structure).",
    )
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    analysis = PoreAnalysis(
        topology=args.structure,
        trajectory=args.trajectory,
        selection=args.selection,
        axis=_AXIS_MAP[args.axis],
        n_slices=args.n_slices,
        n_resample=args.n_resample,
        window=args.window,
        hydrophobicity_sigma=args.hydrophobicity_sigma,
        pore_facing_cutoff=args.pore_facing_cutoff,
        seed_uv=tuple(args.seed_uv) if args.seed_uv is not None else None,
    )

    print(
        f"Analysing {analysis.traj.n_atoms} atoms "
        f"(selection: {args.selection!r}) over "
        f"{analysis.traj.n_frames} frame(s)...",
        file=sys.stderr,
    )

    result = analysis.run(start=args.start, stop=args.stop, step=args.step)

    json_path = outdir / "pore_profile.json"
    csv_path = outdir / "pore_profile.csv"
    residue_csv_path = outdir / "residue_summary.csv"
    result.save_json(json_path)
    result.save_csv(csv_path)
    result.save_residue_summary_csv(residue_csv_path)

    print(f"Minimum pore radius: {result.min_radius_overall:.2f} Angstrom", file=sys.stderr)
    print(
        f"  at pathway position: {result.min_radius_position:.2f} Angstrom "
        f"(pathway length {result.mean_length:.2f} Angstrom)",
        file=sys.stderr,
    )
    print(f"Results written to {json_path}, {csv_path}, and {residue_csv_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
