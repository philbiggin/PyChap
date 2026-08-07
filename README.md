# PyChap

A Python 3 reimplementation of the core analysis performed by **CHAP**
(the [Channel Annotation Package](https://github.com/channotation/chap)),
built on [MDAnalysis](https://www.mdanalysis.org/) so it reads real
GROMACS `.xtc` trajectories directly.

Phil Biggin with (significant!) help from Claude AI.   06/08/2026

Will probably merge into [channotation](https://github.com/channotation/chap) repo at some point.

CHAP upstream finds the permeation pathway through an ion channel or
other membrane pore in a molecular dynamics simulation, reports a pore
radius profile along that pathway, and annotates it with physicochemical
properties (e.g. residue hydrophobicity) of the pore-lining residues.
This package reimplements that core workflow — pathfinding, radius
profile, hydrophobicity mapping, trajectory-averaging — in pure Python,
and it ships with CHAP's own default worked example (PDB 4PIR) bundled
in, so it works out of the box with no data to find or download.

## Installation

```bash
cd PyChap
pip install -r requirements.txt      # numpy, matplotlib, MDAnalysis
pip install -e .                     # optional: installs the `pychap` CLI command
```

Requires Python >= 3.9. No compiler, no GROMACS installation, and no
Boost/LAPACKE are needed (unlike upstream CHAP) — MDAnalysis reads
`.xtc`/`.trr`/`.tpr`/`.gro`/`.pdb` etc. in pure Python + compiled-extension
form, without needing `libgromacs`. The single-structure example below
doesn't even need MDAnalysis.

## Examples — what to expect after installing

Three examples are bundled, in increasing order of what they need
installed. All three have already been run once and their output
committed to the repo (`examples/output/`, `examples/4pir_output/`), so
you can look at the results before running anything yourself.

### 1. Real 4PIR structure — no arguments, no MDAnalysis needed

`examples/data/` bundles upstream CHAP's own default example dataset —
`4pirtm.pdb`/`.xtc`/`.tpr`, the transmembrane domain of a serotonin-gated
ion channel (PDB [4PIR](https://www.rcsb.org/structure/4PIR), a
homopentameric Cys-loop receptor) embedded in a POPC bilayer, see
https://www.channotation.org/docs/annotation_example/. This is the
fastest way to see PyChap work on a real structure:

```bash
cd examples
python run_4pir_example.py
```

Expected console output (this is a real captured run, not illustrative):

```
Reading examples/data/4pirtm.pdb ...
  175,315 total atoms (protein + POPC + solvent + ions)
  12,113 protein atoms selected (resname in the 20 standard amino acids)
  seeding pathway search at (66.5, 63.7) A (protein centroid)
Computing permeation pathway and radius profile (axis=z) ...
  pathway length: 67.6 A
  minimum radius: 1.53 A at s = 42.9 A
Wrote examples/4pir_output/p4pir_pore_profile.json, examples/4pir_output/p4pir_pore_profile.csv, examples/4pir_output/p4pir_residue_summary.csv
  103 of 722 residues flagged as pore-facing (within 12 A of the pathway)
Wrote examples/4pir_output/p4pir_annotated.pdb, examples/4pir_output/p4pir_pathway.obj, examples/4pir_output/p4pir_pathway.mtl
  VMD:   vmd -e ../scripts/vmd/visualise_pathway.tcl -args examples/4pir_output/p4pir_annotated.pdb examples/4pir_output/p4pir_pathway.obj
  PyMOL: pymol ../scripts/pymol/visualise_pathway.py -- examples/4pir_output/p4pir_annotated.pdb examples/4pir_output/p4pir_pathway.obj p4pir

Done in 1.0 s.
```

(Output files are prefixed `p4pir_`, not `4pir_` — PyMOL's `load`
derives a default object name from a file's basename when you don't
give one explicitly, and PyMOL object/selection names can't start with
a digit, so a `4pir_`-prefixed filename would be unsafe to load
directly. See `examples/data/README.md` and `scripts/pymol/README.md`.)

This reads the structure with PyChap's dependency-free PDB parser
(`pychap.pdb_import`, no MDAnalysis needed for a single static
structure), selects the 12,113 protein atoms out of the full
175,315-atom system (excluding POPC/solvent/ions), and runs the full
pathway + radius profile + hydrophobicity + residue-summary pipeline —
in about a second. The result — a pathway ~68 Å long with a constriction
radius of ~1.5 Å partway along it — is consistent with a closed/
resting-state Cys-loop receptor gate, and there's a clear hydrophobic
band of pore-facing residues right at that constriction.

To see it as a plot, run the plotting scripts against the JSON this
just wrote (or just open the pre-generated copy already sitting in
`examples/4pir_output/pathway_profile.png`):

```bash
python ../scripts/plot_pathway_profile.py 4pir_output/p4pir_pore_profile.json -o 4pir_output/pathway_profile.png
```

which reproduces CHAP's own reference figure style — black mean-radius
line, grey ±1 SD band, pore-facing residues scattered at their `(s,
rho)` position and coloured by hydrophobicity on a diverging colour
scale, read directly from the `"residue_summary"` section of the input
JSON (no separate `--residue-csv` flag needed).

`examples/4pir_output/` (all pre-generated and checked into the repo)
contains: `p4pir_pore_profile.{json,csv}`, `p4pir_residue_summary.csv`,
`p4pir_annotated.pdb` + `p4pir_pathway.{obj,mtl}` (for VMD/PyMOL, see
below), and four plots — `radius_profile.png`, `hydrophobicity_profile.png`,
`pore_3d.png`, `pathway_profile.png`.

### 2. Real 4PIR trajectory — no arguments, needs MDAnalysis

The same dataset also includes a 10-frame/10 ns trajectory
(`4pirtm.xtc` + `4pirtm.tpr`), exercising the real MDAnalysis-backed
`pychap.analysis.PoreAnalysis` path (trajectory-averaged radius, with
per-point std/min/max across frames) rather than the single-structure
PDB reader used above:

```bash
pip install MDAnalysis     # if not already installed
python run_4pir_trajectory_example.py
```

This writes into `examples/4pir_output/` — the same directory as the
structure-only example above — with a `p4pir_traj_` filename prefix
(`p4pir_traj_pore_profile.json`, etc.), so all of the bundled 4PIR
example's outputs, structure and trajectory alike, live in one place.

**Caveat:** this script could not be executed in the sandbox this
package was built in (no MDAnalysis, no network access to install it —
see "How this was built and tested" below). It follows the same,
already-tested `pychap.analysis.PoreAnalysis` API as the structure-only
example above, and was syntax-checked, but please treat first use as a
check on your own machine. If MDAnalysis has trouble with the `.tpr`
(TPR format support depends on the GROMACS version it was written with —
this one is GROMACS 5.1), pass `-s data/4pirtm.pdb` instead; it has the
same atom count/order as the trajectory.

### 3. Synthetic "breathing hourglass" pore — zero real data, zero MDAnalysis

A fully synthetic pore (narrow hydrophobic gate at the centre, wide
hydrophilic vestibules at each end) that doesn't depend on any bundled
data file at all — useful as a self-contained sanity check or CI-style
fixture, and the quickest way to see the pipeline run before touching
any real files:

```bash
python run_example.py
```

Expected console output:

```
Running in-memory example over 10 synthetic frames...
  frame 0: constriction radius = 2.07 A at s = 19.9 A (pathway length 40.0 A)
  frame 1: constriction radius = 1.94 A at s = 19.9 A (pathway length 40.0 A)
  ...
  frame 9: constriction radius = 1.37 A at s = 19.9 A (pathway length 40.0 A)

Mean pathway length:        40.00 A
Minimum pore radius:        2.05 A
  at pathway position:      19.87 A
Wrote example_output/pore_profile.json, example_output/pore_profile.csv, and example_output/residue_summary.csv
Wrote example_output/annotated.pdb, example_output/pathway.obj, and example_output/pathway.mtl
  Try it in VMD: vmd -e ../scripts/vmd/visualise_pathway.tcl -args example_output/annotated.pdb example_output/pathway.obj
```

Optionally, generate a *real* `.gro`/`.xtc` pair first (via MDAnalysis)
to exercise the actual trajectory-file-reading path rather than the
in-memory one:

```bash
python generate_synthetic_pore.py -o synthetic_pore_data -n-frames 10
python run_example.py -o example_output --data-dir synthetic_pore_data
```

`examples/output/` (pre-generated, checked into the repo) contains:
`pore_profile.{json,csv}`, `residue_summary.csv`, `annotated.pdb` +
`pathway.{obj,mtl}`, and the same four plot types as the 4PIR example.
Note: in this synthetic example every "residue" sits exactly on the
pathway axis, so its `pathway_profile.png` shows the scatter clustered
near `rho = 0` — a real protein (like 4PIR above) shows a meaningful
spread instead, since side chains point in from the surrounding helices.

## Quick start (your own data)

### Command line

```bash
pychap -s topol.gro -f traj.xtc -sel "protein" -o results
```

or, without installing the package, `python -m pychap.cli -s ... -f ...`
after adding the repository root to `PYTHONPATH`.

Key options (`pychap -h` for the full list):

| Option | Meaning |
|---|---|
| `-s` | topology/structure file (`.gro`, `.pdb`, `.tpr`, ...) |
| `-f` | trajectory file (`.xtc`, `.trr`, ...); omit to analyse a single structure |
| `-sel` | MDAnalysis atom selection for pore-lining atoms, e.g. `"protein"` |
| `-axis` | `x`, `y`, or `z` — which axis is the approximate pore/membrane normal (default `z`) |
| `-pore-facing-cutoff` | distance (Å) from the centreline within which a residue counts as pore-facing (default 12) |
| `-o` | output directory (writes `pore_profile.json`, `pore_profile.csv`, `residue_summary.csv`) |

### Python API

```python
from pychap.analysis import PoreAnalysis

analysis = PoreAnalysis(
    topology="topol.gro",
    trajectory="traj.xtc",
    selection="protein",
    axis=2,          # z
    n_slices=50,      # coarse centreline search resolution
    n_resample=200,   # final radius-profile resolution
)
result = analysis.run()

print(f"Minimum pore radius: {result.min_radius_overall:.2f} A")
result.save_json("pore_profile.json")
result.save_csv("pore_profile.csv")
```

### Visualisation scripts

```bash
python scripts/plot_radius_profile.py results/pore_profile.json -o radius_profile.png
python scripts/plot_hydrophobicity.py results/pore_profile.json -o hydrophobicity_profile.png
python scripts/plot_pore_3d.py results/pore_profile.json -o pore_3d.png
python scripts/plot_pathway_profile.py results/pore_profile.json -o pathway_profile.png
```

`plot_pathway_profile.py` reproduces CHAP's own classic "time-averaged
pathway profile" figure, as documented at
[channotation.org/docs/plotting_python](https://www.channotation.org/docs/plotting_python/):
a black mean-radius line, a grey ±1 SD shaded band, and a scatter of
pore-facing residues positioned at their mean `(s, rho)` — pathway
position and radial distance from the centreline — coloured by
hydrophobicity on a diverging `BrBG_r` colormap with a colorbar, exactly
as in the CHAP docs example. Residues are drawn from the
`"residue_summary"` section of the results JSON; pass
`--min-pore-facing-fraction` to control how "pore-facing" a residue must
be (fraction of frames with radial distance below the cutoff) to be
shown, or `--no-residues` to plot just the radius profile.

## Molecular graphics (VMD and PyMOL)

Beyond the matplotlib plots, PyChap can export a coloured 3D pathway
mesh and an annotated structure for viewing in VMD or PyMOL — mirroring
CHAP's own workflows at
https://www.channotation.org/docs/molecular_graphics_vmd/ and
https://www.channotation.org/docs/molecular_graphics_pymol/:

```bash
python scripts/export_visualisation.py results/pore_profile.json \
    -s topol.gro -o results/vmd --color-by hydrophobicity

# VMD:
vmd -e scripts/vmd/visualise_pathway.tcl -args results/vmd/annotated.pdb results/vmd/pathway.obj

# PyMOL:
pymol -cq scripts/pymol/visualise_pathway.py -- results/vmd/annotated.pdb results/pathway.obj
```

`export_visualisation.py` writes:

- `pathway.obj` + `pathway.mtl` — the time-averaged pathway rendered as
  a tube mesh, coloured by hydrophobicity (or radius) with the colour
  baked in as OBJ materials (`pychap.obj_export`). Loaded directly by
  the VMD script; the PyMOL script reads the *same* mesh file and
  converts it to a native CGO surface (since stock PyMOL has no
  built-in OBJ mesh importer), mirroring the technique upstream CHAP's
  own `wobj.py` uses — see `scripts/pymol/README.md`.
- `annotated.pdb` — the original structure with the chosen property in
  the B-factor column and a pore-facing flag in the occupancy column
  (`pychap.pdb_export`), built by matching PyChap's per-residue
  `residue_summary` back onto the structure's atoms.

Both `examples/run_example.py` (synthetic pore) and
`examples/run_4pir_example.py` (real 4PIR structure) also produce
`annotated.pdb` / `pathway.obj` / `pathway.mtl` directly, without going
through `export_visualisation.py` — see `examples/output/` and
`examples/4pir_output/`. For the real structure:

```bash
vmd -e scripts/vmd/visualise_pathway.tcl -args examples/4pir_output/p4pir_annotated.pdb examples/4pir_output/p4pir_pathway.obj
pymol -cq scripts/pymol/visualise_pathway.py -- examples/4pir_output/p4pir_annotated.pdb examples/4pir_output/p4pir_pathway.obj p4pir
```

(Both scripts now read the same `pathway.obj`/`.mtl` mesh — the PyMOL
script converts it to a native CGO surface since stock PyMOL has no OBJ
importer, mirroring upstream CHAP's own `wobj.py` — and `p4pir`, not
`4pir`, because PyMOL object/selection names can't start with a digit;
see `scripts/pymol/README.md`.)

**Caveat:** the VMD Tcl scripts (`scripts/vmd/`) and PyMOL scripts
(`scripts/pymol/`) were written against each program's standard,
documented scripting API, but this package was built without either VMD
or PyMOL available to test against, so they haven't been visually
verified end-to-end (though the PyMOL scripts' pure-Python logic —
colour mapping and CGO mesh construction — *was* unit-verified by
stubbing out the `pymol` module and running the real functions against
the real 4PIR example data). See `scripts/vmd/README.md` and
`scripts/pymol/README.md` for details and usage notes.

## Relationship to upstream CHAP — please read before relying on this for research

Upstream CHAP is a mature, peer-reviewed (Klesse et al., *J. Mol. Biol.*
2019) C++ tool of roughly 800 commits, linked directly against
`libgromacs` and using Boost/LAPACKE for spline fitting and constrained
optimisation. **This is not a line-for-line port of that code**  Instead,
`pychap` reimplements the same *conceptual* algorithm from scratch and no dependency on `libgromacs` itself.

**What is included:**

- Pore centreline ("permeation pathway") search
- Pore radius profile along that pathway
- Trajectory-averaging (mean/min/max/std radius across frames)
- Residue hydrophobicity mapping onto the pathway (Wimley-White scale)
- Per-residue pathway projection (`pychap.residues`): each pore-lining
  residue's mean position along the pathway (`s`) and radial distance
  from the centreline (`rho`), plus the fraction of frames in which it
  counts as "pore-facing" — CHAP's `residueSummary` concept
- Reading real trajectories, including GROMACS `.xtc`, via MDAnalysis
- A command-line interface loosely modelled on the `chap` executable
- Visualisation scripts (radius profile, hydrophobicity profile, 3D
  tube, and CHAP's time-averaged pathway profile plot)
- A time-averaged 3D pathway centreline (`PoreAnalysisResult.points_mean`)
- A dependency-free PDB reader (`pychap.pdb_import`) for analysing a
  single real structure without needing MDAnalysis
- Molecular graphics export (`pychap.obj_export`, `pychap.pdb_export`,
  `scripts/export_visualisation.py`) and both VMD (`scripts/vmd/`) and
  PyMOL (`scripts/pymol/`) scripts to visualise the pathway alongside
  the structure, matching CHAP's own VMD/PyMOL workflows
- CHAP's own default example dataset (PDB 4PIR, a serotonin receptor
  channel) bundled in `examples/data/`, see "Examples" above

**What is deliberately *not* included** (out of scope for this port):

- Solvent/water density mapping through the pore
- CHAP's own JSON/HTML report format (PyChap has its own JSON/CSV
  output instead, see "Quick start" above)
- Permeation event detection / ion counting
- The pre-processing steps CHAP does via `libgromacs` (e.g. PBC
  make-whole, index-group handling) — you may need to pre-process
  trajectories (e.g. `gmx trjconv -pbc mol -center`) before analysis,
  exactly as you would for many other MDAnalysis-based tools.

### Algorithmic differences from upstream CHAP

1. **Pathfinding.** Upstream CHAP uses a probabilistic (simulated
   annealing-like) search followed by constrained optimisation. This
   port instead marches along a chosen Cartesian axis (default z, i.e.
   the membrane normal) and, at each slice, uses a small Nelder-Mead
   simplex search to find the `(x, y)` position that maximises the
   minimum distance to the van der Waals surface of nearby atoms — an
   approach conceptually close to the classic
   [HOLE](https://doi.org/10.1016/S0263-7855(97)00009-X) algorithm. A
   smooth cubic spline is then fit through the resulting centreline and
   resampled at high resolution.

   **Caveat:** the search plane at each slice is perpendicular to the
   fixed Cartesian axis, not to the local path tangent. This is accurate
   for pores that are roughly aligned with that axis and don't curve
   sharply (true of the large majority of ion channels, which run
   roughly normal to the membrane), but will be less accurate for
   strongly bent or tilted pathways.

2. **Reported radius is a true nearest-surface distance.** At each
   pathway point, the reported radius is the minimum distance to *any*
   nearby atom's van der Waals surface in full 3D — not just the atoms
   in the same cross-sectional slice. Near a flared pore mouth, this can
   mean the reported radius is smaller than the "local ring radius"
   would naively suggest, because the true nearest surface is an
   inward, narrower part of the pore. This is intentional and matches
   the physical definition of pore radius used by HOLE/CHAP-style tools
   — it's not a bug, but it can be surprising if you're expecting a
   purely local calculation.

3. **No SciPy dependency.** Cubic spline interpolation
   (`pychap._numerics.CubicSpline1D`) and the 2D optimiser
   (`pychap._numerics.nelder_mead_2d`) are implemented directly with
   NumPy rather than pulling in SciPy, since MDAnalysis was already a
   hard requirement for trajectory I/O and no other SciPy functionality
   was needed. Both are small, textbook implementations and are unit
   tested against known-analytic cases.

4. **Van der Waals radii** use the standard Bondi (1964) table, with a
   name-based element guess as a fallback when explicit element
   information isn't available in the topology (see `pychap/elements.py`).

5. **Hydrophobicity mapping** uses a Gaussian-kernel distance-weighted
   average of the Wimley-White whole-residue interfacial scale, rather
   than CHAP's exact mapping method.


## Validation

The pathfinding algorithm is checked against synthetic geometries with
an analytically known answer (`tests/test_pathfinding.py`):

- A straight cylindrical pore of radius `R` with atoms of van der Waals
  radius `r` gives a computed pore radius of `R - r` everywhere, and a
  centreline that stays exactly on the cylinder's axis (checked to
  numerical tolerance).
- An hourglass-shaped pore with an analytically placed constriction
  (narrowest point) is correctly located, both in position and in
  radius.

As a real-world sanity check (not a unit test, but reassuring), running
PyChap on the actual 4PIR structure (see "Examples" above) finds a
pathway ~68 Å long with a ~1.5 Å constriction, flanked by a hydrophobic
band of pore-facing residues right at that constriction and a spread of
residue radial distances from ~6-12 Å — a physically sensible
funnel-shaped vestibule narrowing to a gate, consistent with a
closed/resting-state Cys-loop receptor pore.

## Testing

```bash
cd PyChap
python -m unittest discover -s tests -v
```

105 tests, all passing (5 skipped when MDAnalysis isn't installed).
Standard library `unittest` is used (rather than pytest) so the test
suite has zero extra dependencies beyond what PyChap itself needs.
`tests/test_trajectory.py`'s MDAnalysis-backed tests are automatically
skipped if MDAnalysis isn't installed, and `tests/test_integration.py`
exercises the full pathfinding + hydrophobicity + trajectory-averaging
pipeline without needing MDAnalysis at all.

## Repository layout

```
pychap/
  pychap/
    _numerics.py       # dependency-free cubic spline + Nelder-Mead optimiser
    elements.py         # van der Waals radii lookup
    spline.py            # arc-length parameterised 3D path spline
    pathfinding.py        # pore centreline search + radius profile
    hydrophobicity.py      # residue hydrophobicity scale + pathway mapping
    residues.py              # per-residue pathway projection (s, rho, pore-facing)
    tube.py                    # 3D tube mesh construction (shared by plotting + OBJ export)
    obj_export.py                # coloured Wavefront OBJ/MTL pathway mesh export
    pdb_export.py                  # minimal annotated-PDB writer (B-factor/occupancy)
    pdb_import.py                    # dependency-free PDB reader (single structures)
    trajectory.py                      # MDAnalysis-backed structure/trajectory reader
    analysis.py                          # high-level orchestration + trajectory averaging
    testing.py                             # synthetic pore geometry generators
    cli.py                                   # command line interface
  tests/                                      # unittest-based test suite (105 tests)
  examples/
    data/                                        # bundled CHAP default example dataset
      4pirtm.pdb                                    # 175,315-atom structure snapshot (~14 MB)
      4pirtm.tpr                                     # GROMACS run-topology file (~9 MB)
      4pirtm.xtc                                      # 10-frame/10 ns trajectory (~7 MB)
      README.md                                        # what these files are, and which scripts use them
    generate_synthetic_pore.py                 # writes a real .gro/.xtc synthetic example
    run_example.py                              # synthetic pore: full pipeline, in-memory and/or from file
    run_4pir_example.py                          # real 4PIR structure (bundled data, no args needed)
    run_4pir_trajectory_example.py                # real 4PIR trajectory (bundled data, needs MDAnalysis)
    output/                                        # pre-generated synthetic-example results/plots
    4pir_output/                                    # pre-generated real-4PIR-example results/plots
  scripts/
    plot_radius_profile.py                           # radius vs. pathway position
    plot_hydrophobicity.py                            # hydrophobicity vs. pathway position
    plot_pore_3d.py                                    # 3D tube visualisation coloured by radius
    plot_pathway_profile.py                             # CHAP-style time-averaged pathway profile
    export_visualisation.py                              # writes annotated.pdb + pathway.obj/.mtl
    vmd/
      visualise_pathway.tcl                                # structure + coloured pathway mesh in VMD
      colour_by_property.tcl                                # structure-only B-factor colouring
      README.md
    pymol/
      visualise_pathway.py                                   # structure + CGO-tube pathway in PyMOL
      colour_by_property.py                                   # structure-only B-factor colouring
      README.md
  requirements.txt
  pyproject.toml
```

## License

GPL-3.0-or-later, matching upstream CHAP's license.

## Citing

If you use this reimplementation, please still cite the original CHAP
paper, since this package's design follows its methodology:

> G. Klesse, S. Rao, M. S. P. Sansom, and S. J. Tucker, "CHAP: A
> Versatile Tool for the Structural and Functional Annotation of Ion
> Channel Pores," *Journal of Molecular Biology*, 2019.
> https://doi.org/10.1016/j.jmb.2019.06.003
