#!/usr/bin/env python3
"""
Produce VMD/PyMOL-ready visualisation files from a pychap results JSON:
a coloured pathway mesh (OBJ + MTL) and, if the original structure file
is provided, an annotated PDB with a chosen pathway property in the
B-factor column and a pore-facing flag in the occupancy column -- see
scripts/vmd/visualise_pathway.tcl and
https://www.channotation.org/docs/molecular_graphics_vmd/.

Usage
-----
    # pathway mesh only:
    python export_visualisation.py pore_profile.json -o vmd_output

    # pathway mesh + annotated structure (requires MDAnalysis):
    python export_visualisation.py pore_profile.json -s topol.gro -o vmd_output

    # then, in VMD:
    vmd -e ../scripts/vmd/visualise_pathway.tcl -args vmd_output/annotated.pdb vmd_output/pathway.obj

Note on residue matching: annotating a real structure works by matching
each atom's residue id (resid) against pychap's residue_summary. If
your structure has multiple chains that reuse the same numeric resid
(common for multi-subunit channels), atoms in *all* matching chains will
get the same annotation -- a simplification worth knowing about, since
upstream CHAP tracks residues by a unique internal index rather than by
resid alone.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pychap.obj_export import export_pathway_obj  # noqa: E402

_COLOR_BY_RESIDUE_KEY = {
    "hydrophobicity": "hydrophobicity_kcalmol",
    "rho": "rho_mean_angstrom",
}


def load_result(json_path):
    with open(json_path) as fh:
        return json.load(fh)


def export_pathway_mesh(data: dict, outdir: Path, color_by: str):
    if not data.get("points_mean_angstrom"):
        sys.exit(
            "Input JSON has no time-averaged pathway ('points_mean_angstrom'). "
            "Re-run the analysis with an up-to-date pychap (PoreAnalysis.run() "
            "computes this automatically)."
        )

    points = np.array(data["points_mean_angstrom"])
    radius = np.array(data["radius_mean_angstrom"])
    s = np.array(data["s_grid_normalised"]) * data["mean_length_angstrom"]

    if color_by == "radius":
        color_values = radius
        symmetric, cmap, label = False, "viridis", "pore radius (Angstrom)"
    else:
        color_values = np.array(data["hydrophobicity_mean_kcalmol"])
        symmetric, cmap, label = True, "BrBG_r", "hydrophobicity (kcal/mol)"

    obj_path = outdir / "pathway.obj"
    mtl_path = outdir / "pathway.mtl"
    export_pathway_obj(
        points,
        radius,
        s,
        color_values,
        obj_path,
        mtl_path,
        cmap_name=cmap,
        symmetric_color_limits=symmetric,
        value_label=label,
    )
    return obj_path, mtl_path


def export_annotated_structure(
    data: dict, structure_path, outdir: Path, color_by: str, pore_facing_threshold: float
):
    try:
        import MDAnalysis as mda
    except ImportError:
        sys.exit("Annotating a real structure file requires MDAnalysis: pip install MDAnalysis")

    residues = data.get("residue_summary") or []
    if not residues:
        sys.exit(
            "Input JSON has no 'residue_summary' data -- cannot annotate a structure. "
            "Re-run the analysis with an up-to-date pychap."
        )

    property_key = _COLOR_BY_RESIDUE_KEY.get(color_by, "hydrophobicity_kcalmol")
    by_resid = {int(r["resid"]): r for r in residues}

    universe = mda.Universe(str(structure_path))
    n_atoms = len(universe.atoms)

    b_values = np.zeros(n_atoms)
    occ_values = np.zeros(n_atoms)
    resids = universe.atoms.resids
    for idx, resid in enumerate(resids):
        r = by_resid.get(int(resid))
        if r is None:
            continue
        b_values[idx] = r[property_key]
        occ_values[idx] = 1.0 if r["pore_facing_fraction"] > pore_facing_threshold else 0.0

    for attr_name, default in (("tempfactors", 0.0), ("occupancies", 0.0)):
        try:
            getattr(universe.atoms, attr_name)
        except Exception:
            universe.add_TopologyAttr(attr_name, values=np.full(n_atoms, default))
    universe.atoms.tempfactors = b_values
    universe.atoms.occupancies = occ_values

    out_pdb = outdir / "annotated.pdb"
    universe.atoms.write(str(out_pdb))
    return out_pdb


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_json", help="pychap results JSON file (e.g. pore_profile.json)")
    parser.add_argument(
        "-s",
        "--structure",
        default=None,
        help="Original topology/structure file to annotate (e.g. .gro/.pdb/.tpr). Requires MDAnalysis.",
    )
    parser.add_argument("-o", "--outdir", default="vmd_output")
    parser.add_argument(
        "--color-by",
        choices=("hydrophobicity", "radius", "rho"),
        default="hydrophobicity",
        help="Property to colour the pathway mesh (and, if -s given, the structure B-factor) by.",
    )
    parser.add_argument(
        "--pore-facing-threshold",
        type=float,
        default=0.5,
        help="Fraction of frames a residue must be pore-facing in to be flagged (occupancy=1) in the annotated PDB.",
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    data = load_result(args.input_json)

    obj_path, mtl_path = export_pathway_mesh(data, outdir, args.color_by)
    print(f"Wrote {obj_path} and {mtl_path}")

    if args.structure:
        pdb_path = export_annotated_structure(
            data, args.structure, outdir, args.color_by, args.pore_facing_threshold
        )
        print(f"Wrote {pdb_path}")
        print()
        print("Try it in VMD:")
        print(f"  vmd -e scripts/vmd/visualise_pathway.tcl -args {pdb_path} {obj_path}")
    else:
        print()
        print("No -s/--structure given, so only the pathway mesh was exported.")
        print("Pass -s topol.gro (or similar) to also produce an annotated structure PDB.")


if __name__ == "__main__":
    main()
