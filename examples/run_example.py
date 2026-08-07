#!/usr/bin/env python3
"""
End-to-end example: run the full PyChap analysis pipeline on a synthetic
"breathing hourglass" pore and write out results (JSON + CSV) that the
plotting scripts in ../scripts can visualise.

This script runs in two modes:

1. In-memory mode (always runs, no extra dependencies beyond NumPy):
   builds the synthetic pore directly as NumPy arrays (via
   pychap.testing) and pushes them through compute_pore_profile /
   hydrophobicity_profile / aggregate_frame_results frame by frame --
   exactly the computation PoreAnalysis.run() performs, just without
   needing an actual trajectory file or MDAnalysis.

2. File mode (runs only if MDAnalysis is installed and
   synthetic_pore_data/synthetic_pore.{gro,xtc} exist -- see
   generate_synthetic_pore.py): reads those files with
   pychap.analysis.PoreAnalysis, i.e. exercises the real GROMACS
   trajectory-reading path.

Usage
-----
    python run_example.py -o example_output
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pychap.analysis import FrameResult, aggregate_frame_results, build_residue_summary  # noqa: E402
from pychap.hydrophobicity import hydrophobicity_profile, residue_hydrophobicity  # noqa: E402
from pychap.obj_export import export_pathway_obj  # noqa: E402
from pychap.pathfinding import compute_pore_profile  # noqa: E402
from pychap.pdb_export import write_minimal_pdb  # noqa: E402
from pychap.residues import project_residues  # noqa: E402
from pychap.testing import hourglass_pore, hourglass_pore_residues  # noqa: E402

#: Geometry shared between the analysis pipeline and the annotated
#: structure written for VMD (see export_visualisation_files()).
POREKWARGS = dict(r_max=10.0, n_rings=41, n_per_ring=24, z_min=-20.0, z_max=20.0, vdw_radius=1.0)


def run_in_memory_example(n_frames=10, seed=0, breathing_amplitude=0.5, pore_facing_cutoff=12.0):
    """Run the full pipeline on an in-memory synthetic hourglass pore,
    including the per-residue pathway projection used by
    scripts/plot_pathway_profile.py to reproduce CHAP's classic
    time-averaged pathway profile plot
    (https://www.channotation.org/docs/plotting_python/)."""
    rng = np.random.default_rng(seed)
    frame_results = []
    seed_uv = (0.0, 0.0)

    # The "residues" here are one per ring (see hourglass_pore_residues),
    # not one per atom -- a coarse but illustrative stand-in for a real
    # pore-lining residue list.
    residue_s_frames = []
    residue_rho_frames = []
    residue_facing_frames = []
    residue_names = None
    residue_ids = None

    pore_kwargs = POREKWARGS

    for frame_index in range(n_frames):
        r_min = max(0.5, 3.0 + breathing_amplitude * rng.normal())
        positions, radii, ring_z, ring_radius = hourglass_pore(r_min=r_min, **pore_kwargs)
        res_positions, res_names = hourglass_pore_residues(ring_z, ring_radius)
        if residue_names is None:
            residue_names = res_names
            residue_ids = list(range(1, len(res_names) + 1))
        res_values = np.array([residue_hydrophobicity(name) for name in res_names])

        profile = compute_pore_profile(
            positions, radii, axis=2, n_slices=41, n_resample=150, window=3.0, seed_uv=seed_uv
        )
        seed_uv = profile.last_uv

        hydro = hydrophobicity_profile(profile.points, res_positions, res_values, sigma=6.0)

        projection = project_residues(res_positions, profile, pore_facing_cutoff=pore_facing_cutoff)
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
        print(
            f"  frame {frame_index}: constriction radius = {profile.min_radius:.2f} A "
            f"at s = {profile.min_radius_position:.1f} A (pathway length {profile.length:.1f} A)"
        )

    result = aggregate_frame_results(frame_results, n_resample=150)
    result.residue_summary = build_residue_summary(
        residue_ids, residue_names, residue_s_frames, residue_rho_frames, residue_facing_frames
    )
    return result


def export_visualisation_files(result, outdir: Path):
    """Write a VMD-ready annotated PDB + coloured pathway OBJ/MTL mesh for
    the synthetic example, demonstrating the full visualisation workflow
    (see scripts/vmd/) without needing MDAnalysis or a real structure
    file -- the synthetic pore's own geometry is used directly.

    The "structure" written here is a representative snapshot (the
    r_min = 3.0 A hourglass geometry, i.e. the mean breathing amplitude)
    with one pseudo-atom per ring position; B-factors are set to each
    residue's mean hydrophobicity and occupancy flags pore-facing
    residues (fraction of frames > 0.5), matching the conventions
    documented in scripts/vmd/README.md.
    """
    positions, _radii, ring_z, _ring_radius = hourglass_pore(r_min=3.0, **POREKWARGS)
    n_per_ring = POREKWARGS["n_per_ring"]
    n_rings = POREKWARGS["n_rings"]

    resids_per_atom = np.repeat(np.arange(1, n_rings + 1), n_per_ring)
    resnames_per_ring = ["LEU" if abs(z) <= 6.0 else "ASP" for z in ring_z]
    resnames_per_atom = np.repeat(resnames_per_ring, n_per_ring)
    atom_names = ["CA"] * len(positions)

    by_resid = {r.resid: r for r in result.residue_summary}
    b_factors = np.array([by_resid[int(rid)].hydrophobicity for rid in resids_per_atom])
    occupancies = np.array(
        [1.0 if by_resid[int(rid)].pore_facing_fraction > 0.5 else 0.0 for rid in resids_per_atom]
    )

    pdb_path = outdir / "annotated.pdb"
    write_minimal_pdb(
        positions,
        resnames_per_atom,
        resids_per_atom,
        atom_names,
        pdb_path,
        b_factors=b_factors,
        occupancies=occupancies,
        title="PyChap synthetic pore example (B-factor=hydrophobicity, occupancy=pore-facing flag)",
    )

    s = result.s_grid * result.mean_length
    obj_path = outdir / "pathway.obj"
    mtl_path = outdir / "pathway.mtl"
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

    return pdb_path, obj_path, mtl_path


def try_file_based_example(outdir: Path, data_dir: Path):
    """If MDAnalysis is installed and synthetic trajectory files exist,
    also run the analysis via the real file-reading path (PoreAnalysis),
    to demonstrate reading a genuine GROMACS .gro/.xtc pair."""
    gro_path = data_dir / "synthetic_pore.gro"
    xtc_path = data_dir / "synthetic_pore.xtc"
    if not (gro_path.exists() and xtc_path.exists()):
        print(
            f"\n(Skipping file-based example: {gro_path} / {xtc_path} not found. "
            f"Run generate_synthetic_pore.py first if you want to exercise the "
            f"real GROMACS-file-reading path.)"
        )
        return None

    try:
        from pychap.analysis import PoreAnalysis
    except ImportError:
        print("\n(Skipping file-based example: MDAnalysis is not installed.)")
        return None

    print(f"\nRunning file-based example on {gro_path} / {xtc_path} ...")
    analysis = PoreAnalysis(
        topology=str(gro_path),
        trajectory=str(xtc_path),
        selection="all",
        axis=2,
        n_slices=41,
        n_resample=150,
        window=3.0,
    )
    result = analysis.run()
    json_path = outdir / "file_based_pore_profile.json"
    csv_path = outdir / "file_based_pore_profile.csv"
    residue_csv_path = outdir / "file_based_residue_summary.csv"
    result.save_json(json_path)
    result.save_csv(csv_path)
    result.save_residue_summary_csv(residue_csv_path)
    print(f"Wrote {json_path}, {csv_path}, and {residue_csv_path}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-o", "--outdir", default="example_output")
    parser.add_argument("-n-frames", type=int, default=10)
    parser.add_argument(
        "-data-dir",
        default="synthetic_pore_data",
        help="Directory to look for synthetic_pore.gro/.xtc (see generate_synthetic_pore.py)",
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Running in-memory example over {args.n_frames} synthetic frames...")
    result = run_in_memory_example(n_frames=args.n_frames)

    json_path = outdir / "pore_profile.json"
    csv_path = outdir / "pore_profile.csv"
    residue_csv_path = outdir / "residue_summary.csv"
    result.save_json(json_path)
    result.save_csv(csv_path)
    result.save_residue_summary_csv(residue_csv_path)

    print()
    print(f"Mean pathway length:        {result.mean_length:.2f} A")
    print(f"Minimum pore radius:        {result.min_radius_overall:.2f} A")
    print(f"  at pathway position:      {result.min_radius_position:.2f} A")
    print(f"Wrote {json_path}, {csv_path}, and {residue_csv_path}")

    pdb_path, obj_path, mtl_path = export_visualisation_files(result, outdir)
    print(f"Wrote {pdb_path}, {obj_path}, and {mtl_path}")
    print(f"  Try it in VMD: vmd -e ../scripts/vmd/visualise_pathway.tcl -args {pdb_path} {obj_path}")

    try_file_based_example(outdir, Path(args.data_dir))


if __name__ == "__main__":
    main()
