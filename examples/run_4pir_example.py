#!/usr/bin/env python3
"""
Run PyChap on the real 4PIR serotonin-receptor transmembrane domain
structure -- upstream CHAP's own default example, see
https://www.channotation.org/docs/annotation_example/ -- mirroring
CHAP's "Running CHAP on Structures" mode:

    chap -f 4pirtm.pdb -s 4pirtm.pdb -sel-pathway 1

4PIR is the transmembrane domain of a serotonin-gated ion channel
(5-HT3 receptor), a homopentameric Cys-loop receptor, embedded in a
POPC bilayer and solvated in NaCl solution. This script reads the
protein coordinates straight out of the PDB file (via
pychap.pdb_import, no MDAnalysis needed for a single static structure)
and runs the full pathway/radius-profile/hydrophobicity pipeline on the
protein atoms only -- the membrane and solvent aren't needed for that.

The example data (``examples/data/4pirtm.pdb``) is bundled with this
package, so this script works with no arguments at all -- see
``examples/data/README.md`` for what it is.

Usage
-----
    python run_4pir_example.py                              # uses the bundled example data
    python run_4pir_example.py -s /path/to/4pirtm.pdb -o out # or point at your own copy
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pychap.analysis import (  # noqa: E402
    FrameResult,
    aggregate_frame_results,
    build_residue_summary,
    plane_centroid_uv,
)
from pychap.hydrophobicity import (  # noqa: E402
    WIMLEY_WHITE_INTERFACE,
    hydrophobicity_profile,
    residue_hydrophobicity,
)
from pychap.obj_export import export_pathway_obj  # noqa: E402
from pychap.pathfinding import compute_pore_profile  # noqa: E402
from pychap.pdb_export import write_minimal_pdb  # noqa: E402
from pychap.pdb_import import read_pdb  # noqa: E402
from pychap.residues import project_residues  # noqa: E402

#: The 20 standard amino acids -- used to select "protein" atoms out of
#: the full solvated, membrane-embedded system (excludes POPC, SOL, ions).
PROTEIN_RESNAMES = set(WIMLEY_WHITE_INTERFACE.keys())

_AXIS_MAP = {"x": 0, "y": 1, "z": 2}

#: Bundled example data (see examples/data/README.md), used as the default
#: -s/-o values so this script works out of the box with no arguments.
_DEFAULT_STRUCTURE = Path(__file__).resolve().parent / "data" / "4pirtm.pdb"
_DEFAULT_OUTDIR = Path(__file__).resolve().parent / "4pir_output"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "-s",
        "--structure",
        default=str(_DEFAULT_STRUCTURE),
        help=f"Path to 4pirtm.pdb (default: bundled example data, {_DEFAULT_STRUCTURE})",
    )
    parser.add_argument("-o", "--outdir", default=str(_DEFAULT_OUTDIR))
    parser.add_argument("-axis", choices=("x", "y", "z"), default="z")
    parser.add_argument("-n-slices", type=int, default=60)
    parser.add_argument("-n-resample", type=int, default=250)
    parser.add_argument("-window", type=float, default=15.0)
    parser.add_argument("-pore-facing-cutoff", type=float, default=12.0)
    parser.add_argument("-hydrophobicity-sigma", type=float, default=5.0)
    args = parser.parse_args()
    axis = _AXIS_MAP[args.axis]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not Path(args.structure).exists():
        sys.exit(
            f"Structure file not found: {args.structure}\n"
            "If you deleted/moved the bundled example data, either restore "
            "examples/data/4pirtm.pdb or pass -s /path/to/4pirtm.pdb explicitly."
        )

    t0 = time.time()
    print(f"Reading {args.structure} ...")
    structure = read_pdb(args.structure)
    print(f"  {len(structure):,} total atoms (protein + POPC + solvent + ions)")

    protein = structure.select(resnames_include=PROTEIN_RESNAMES)
    print(f"  {len(protein):,} protein atoms selected (resname in the 20 standard amino acids)")

    radii = protein.vdw_radii()

    # A good starting guess for the pore centre in the plane perpendicular
    # to `axis` -- the protein centroid -- matters a lot here, since (unlike
    # the synthetic examples) this structure's coordinates are absolute
    # simulation-box coordinates, not centred on the pore axis. This is the
    # same automatic seeding pychap.analysis.PoreAnalysis now does by
    # default (see plane_centroid_uv) -- done explicitly here since this
    # script calls compute_pore_profile directly rather than going through
    # PoreAnalysis (no MDAnalysis needed for a single static structure).
    seed_uv = plane_centroid_uv(protein.positions, axis)
    print(f"  seeding pathway search at ({seed_uv[0]:.1f}, {seed_uv[1]:.1f}) A (protein centroid)")

    print(f"Computing permeation pathway and radius profile (axis={args.axis}) ...")
    profile = compute_pore_profile(
        protein.positions,
        radii,
        axis=axis,
        n_slices=args.n_slices,
        n_resample=args.n_resample,
        window=args.window,
        seed_uv=seed_uv,
    )
    print(f"  pathway length: {profile.length:.1f} A")
    print(f"  minimum radius: {profile.min_radius:.2f} A at s = {profile.min_radius_position:.1f} A")

    res_positions, res_names, res_ids = protein.residue_centers()
    res_values = np.array([residue_hydrophobicity(n) for n in res_names])
    hydro = hydrophobicity_profile(profile.points, res_positions, res_values, sigma=args.hydrophobicity_sigma)

    projection = project_residues(res_positions, profile, pore_facing_cutoff=args.pore_facing_cutoff)

    frame = FrameResult(
        frame=0,
        s=profile.s,
        points=profile.points,
        radius=profile.radius,
        hydrophobicity=hydro,
        length=profile.length,
    )
    result = aggregate_frame_results([frame], n_resample=args.n_resample)
    result.residue_summary = build_residue_summary(
        res_ids, res_names, [projection.s], [projection.rho], [projection.pore_facing]
    )

    json_path = outdir / "p4pir_pore_profile.json"
    csv_path = outdir / "p4pir_pore_profile.csv"
    residue_csv_path = outdir / "p4pir_residue_summary.csv"
    result.save_json(json_path)
    result.save_csv(csv_path)
    result.save_residue_summary_csv(residue_csv_path)
    print(f"Wrote {json_path}, {csv_path}, {residue_csv_path}")

    n_pore_facing = sum(1 for r in result.residue_summary if r.pore_facing_fraction > 0.5)
    print(f"  {n_pore_facing} of {len(result.residue_summary)} residues flagged as pore-facing "
          f"(within {args.pore_facing_cutoff:.0f} A of the pathway)")

    # --- annotated structure + pathway mesh, for VMD/PyMOL -------------
    pore_facing_resids = {r.resid for r in result.residue_summary if r.pore_facing_fraction > 0.5}
    by_resid_hydro = {r.resid: r.hydrophobicity for r in result.residue_summary}
    b_factors = np.array([by_resid_hydro.get(rid, 0.0) for rid in protein.resids])
    occupancies = np.array([1.0 if rid in pore_facing_resids else 0.0 for rid in protein.resids])

    pdb_out = outdir / "p4pir_annotated.pdb"
    write_minimal_pdb(
        protein.positions,
        protein.resnames,
        protein.resids,
        protein.atom_names,
        pdb_out,
        b_factors=b_factors,
        occupancies=occupancies,
        elements=protein.elements,
        title="PyChap annotated 4PIR structure (B-factor=hydrophobicity, occupancy=pore-facing flag)",
    )

    s = result.s_grid * result.mean_length
    obj_path = outdir / "p4pir_pathway.obj"
    mtl_path = outdir / "p4pir_pathway.mtl"
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
    print(f"Wrote {pdb_out}, {obj_path}, {mtl_path}")
    print(f"  VMD:   vmd -e ../scripts/vmd/visualise_pathway.tcl -args {pdb_out} {obj_path}")
    # PyMOL reads the same pathway.obj/.mtl mesh as the VMD line above
    # (parsed and converted to CGO by visualise_pathway.py, mirroring
    # upstream CHAP's own wobj.py -- see scripts/pymol/README.md).
    # "p4pir" (not "4pir") because PyMOL object/selection names can't
    # start with a digit -- see scripts/pymol/README.md.
    print(f"  PyMOL: pymol ../scripts/pymol/visualise_pathway.py -- {pdb_out} {obj_path} p4pir")

    print(f"\nDone in {time.time() - t0:.1f} s.")


if __name__ == "__main__":
    main()
