#!/usr/bin/env python3
"""
Run PyChap on the real 4PIR trajectory (4pirtm.xtc / 4pirtm.tpr) --
upstream CHAP's own default trajectory example, see
https://www.channotation.org/docs/annotation_example/ -- mirroring
CHAP's "Running CHAP on Trajectories" mode:

    chap -f 4pirtm.xtc -s 4pirtm.tpr -sel-pathway 1 -sel-solvent 16

This uses pychap.analysis.PoreAnalysis, i.e. the real MDAnalysis-backed
trajectory-reading path -- unlike run_4pir_example.py (which reads just
the single 4pirtm.pdb structure with pychap's dependency-free PDB
parser), this script requires MDAnalysis to be installed:

    pip install MDAnalysis

Per CHAP's own documentation, the trajectory covers 10 frames / 10 ns
from a longer simulation -- enough to get a trajectory-averaged radius
profile with meaningful frame-to-frame statistics (radius_std,
radius_min/max), unlike the single-structure example.

IMPORTANT: this script could not actually be executed in the sandbox
PyChap was built in (no MDAnalysis, and no network access to install
it there -- see README.md's "how this was built and tested" section).
It follows the same pychap.analysis.PoreAnalysis API that is unit
tested elsewhere in this package, and the equivalent single-structure
analysis (run_4pir_example.py, using the bundled 4pirtm.pdb) has been
verified end-to-end against this exact system. Still, please treat
first use of *this* script as a check on your own machine.

The bundled example data (``examples/data/4pirtm.tpr`` / ``.xtc``) means
this script works with no arguments (once MDAnalysis is installed), and
writes into ``examples/4pir_output/`` -- the same directory
``run_4pir_example.py`` uses, with a ``p4pir_traj_`` filename prefix so
the two examples' outputs don't collide.

Usage
-----
    python run_4pir_trajectory_example.py                # uses the bundled example data
    python run_4pir_trajectory_example.py \\
        -s /path/to/4pirtm.tpr -f /path/to/4pirtm.xtc -o 4pir_output   # or point at your own copy

If MDAnalysis has trouble parsing the .tpr (TPR format support depends
on the GROMACS version it was written with -- this one is GROMACS 5.1),
you can substitute 4pirtm.pdb as the topology instead, since it has the
same atom count/order as the trajectory:

    python run_4pir_trajectory_example.py \\
        -s /path/to/4pirtm.pdb -f /path/to/4pirtm.xtc -o 4pir_output
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pychap.analysis import PoreAnalysis  # noqa: E402
from pychap.obj_export import export_pathway_obj  # noqa: E402

_AXIS_MAP = {"x": 0, "y": 1, "z": 2}

#: Bundled example data (see examples/data/README.md), used as the default
#: -s/-f/-o values so this script works out of the box with no arguments
#: (once MDAnalysis is installed).
_DEFAULT_STRUCTURE = Path(__file__).resolve().parent / "data" / "4pirtm.tpr"
_DEFAULT_TRAJECTORY = Path(__file__).resolve().parent / "data" / "4pirtm.xtc"
#: Same output directory as run_4pir_example.py, so all 4PIR example
#: results live in one place -- filenames are distinguished by the
#: "p4pir_traj_" prefix used below (vs. "p4pir_" for the structure-only
#: example), so nothing collides. Prefixed with "p" (not just "4pir_")
#: because PyMOL's `load` derives a default object name from a file's
#: basename, and PyMOL object/selection names can't start with a digit.
_DEFAULT_OUTDIR = Path(__file__).resolve().parent / "4pir_output"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "-s",
        "--structure",
        default=str(_DEFAULT_STRUCTURE),
        help=f"Path to 4pirtm.tpr, or 4pirtm.pdb (default: bundled example data, {_DEFAULT_STRUCTURE})",
    )
    parser.add_argument(
        "-f",
        "--trajectory",
        default=str(_DEFAULT_TRAJECTORY),
        help=f"Path to 4pirtm.xtc (default: bundled example data, {_DEFAULT_TRAJECTORY})",
    )
    parser.add_argument("-o", "--outdir", default=str(_DEFAULT_OUTDIR))
    parser.add_argument("-sel", "--selection", default="protein")
    parser.add_argument("-axis", choices=("x", "y", "z"), default="z")
    parser.add_argument("-n-slices", type=int, default=60)
    parser.add_argument("-n-resample", type=int, default=250)
    parser.add_argument("-window", type=float, default=15.0)
    parser.add_argument("-pore-facing-cutoff", type=float, default=12.0)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"Loading topology {args.structure} and trajectory {args.trajectory} ...")
    analysis = PoreAnalysis(
        topology=args.structure,
        trajectory=args.trajectory,
        selection=args.selection,
        axis=_AXIS_MAP[args.axis],
        n_slices=args.n_slices,
        n_resample=args.n_resample,
        window=args.window,
        pore_facing_cutoff=args.pore_facing_cutoff,
    )
    print(f"  {analysis.traj.n_atoms} atoms selected ({args.selection!r}), {analysis.traj.n_frames} frame(s)")

    print("Running pathway/radius-profile analysis frame by frame ...")
    result = analysis.run()
    print(f"  mean pathway length: {result.mean_length:.1f} A")
    print(f"  minimum radius: {result.min_radius_overall:.2f} A at s = {result.min_radius_position:.1f} A")
    constriction_idx = int(result.radius_mean.argmin())
    print(
        f"  radius std at constriction: {result.radius_std[constriction_idx]:.2f} A "
        f"(frame-to-frame fluctuation over {len(result.frames)} frames)"
    )

    json_path = outdir / "p4pir_traj_pore_profile.json"
    csv_path = outdir / "p4pir_traj_pore_profile.csv"
    residue_csv_path = outdir / "p4pir_traj_residue_summary.csv"
    result.save_json(json_path)
    result.save_csv(csv_path)
    result.save_residue_summary_csv(residue_csv_path)
    print(f"Wrote {json_path}, {csv_path}, {residue_csv_path}")

    s = result.s_grid * result.mean_length
    obj_path = outdir / "p4pir_traj_pathway.obj"
    mtl_path = outdir / "p4pir_traj_pathway.mtl"
    export_pathway_obj(
        result.points_mean,
        result.radius_mean,
        s,
        result.hydrophobicity_mean,
        obj_path,
        mtl_path,
        cmap_name="BrBG_r",
        symmetric_color_limits=True,
        value_label="hydrophobicity (kcal/mol)",
    )
    print(f"Wrote {obj_path}, {mtl_path}")
    print("For an annotated structure PDB (B-factor/occupancy) for VMD/PyMOL, run:")
    print(f"  python ../scripts/export_visualisation.py {json_path} -s {args.structure} -o {outdir}")

    print(f"\nDone in {time.time() - t0:.1f} s.")


if __name__ == "__main__":
    main()
