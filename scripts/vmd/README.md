# VMD visualisation scripts

These Tcl scripts render PyChap results in [VMD](https://www.ks.uiuc.edu/Research/vmd/),
mirroring upstream CHAP's own visualisation workflow described at
https://www.channotation.org/docs/molecular_graphics_vmd/ and
https://www.channotation.org/docs/annotation_example/.

**Important caveat:** these scripts were written against the standard,
documented VMD Tcl scripting API, but this package was built in a
sandbox without VMD installed, so they could **not** be executed or
visually verified here. They should be a solid starting point, but
please treat first use as a check, and feel free to adjust
representations/materials/colours to taste.

## Files

- `visualise_pathway.tcl` -- loads an annotated structure PDB and a
  coloured pathway OBJ/MTL mesh together: protein as a cartoon, with
  pore-facing residues additionally shown as van der Waals spheres
  coloured by a property (from the PDB B-factor column), and the
  permeation pathway rendered as a coloured surface (pre-baked colours
  from the OBJ/MTL materials, see `pychap.obj_export`).
- `colour_by_property.tcl` -- simpler, structure-only variant: no
  pathway mesh needed, just colours the protein by whatever property is
  in the B-factor column and highlights pore-facing residues as
  licorice sticks.

## Generating the input files

Use `scripts/export_visualisation.py` (for a real structure/trajectory
you've already analysed with `pychap`) or just run
`examples/run_example.py`, which additionally writes an annotated PDB
and pathway OBJ/MTL for the synthetic example into its output directory
alongside the JSON/CSV/plots.

```bash
python scripts/export_visualisation.py results/pore_profile.json \
    -s topol.gro -o results/vmd --color-by hydrophobicity

vmd -e scripts/vmd/visualise_pathway.tcl -args results/vmd/annotated.pdb results/vmd/pathway.obj
```

## Conventions used

- **Occupancy column**: 1.0 for pore-facing atoms/residues, 0.0
  otherwise (a binary flag). Used for selections like `occupancy > 0.5`.
- **B-factor column**: the continuous property being visualised (e.g.
  hydrophobicity in kcal/mol, or radial distance from the pathway in
  Angstrom), broadcast from residue-level `pychap` data to every atom
  in that residue.
- **Pathway mesh colours**: baked into the OBJ file's referenced `.mtl`
  materials as a set of discrete colour bins approximating a colormap
  (`BrBG_r` by default, matching `scripts/plot_pathway_profile.py`), so
  the mesh displays correctly in VMD without any extra colouring
  commands.

VMD's built-in `color scale method BWR` (Blue-White-Red) is used as the
closest built-in diverging colour scale to `BrBG_r` for the Beta-coloured
atom representations; it isn't pixel-identical to the matplotlib scale
used elsewhere in this package, just a similarly-diverging alternative
available natively in VMD.
