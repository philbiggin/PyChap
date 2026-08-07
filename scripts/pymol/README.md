# PyMOL visualisation scripts

These scripts render PyChap results in [PyMOL](https://pymol.org/),
alongside the VMD scripts in `../vmd/` -- see upstream CHAP's own
equivalent workflow at
https://www.channotation.org/docs/molecular_graphics_pymol/.

**Testing status:** these use the standard, documented PyMOL Python API
(`pymol.cmd`, `pymol.cgo`). The pure Python logic (OBJ/MTL parsing,
CGO geometry construction) was unit-verified by stubbing out the
`pymol` module and running the real functions against real PyChap
output (the bundled 4PIR example, including regenerating
`p4pir_pathway.obj`/`.mtl` and checking vertex/normal/face counts,
normal unit-length, distinct colours read back from the `.mtl`, and
the exact CGO list length) -- so the *logic* is exercised and correct.
The *PyMOL-specific* calls (`cmd.load`, `cmd.spectrum`,
`cmd.load_cgo`, ...) could not be run against a real PyMOL
installation in the sandbox this package was built in (no PyMOL
there). Please treat first real use as a check.

## Files

- `visualise_pathway.py` -- loads an annotated structure PDB and draws
  the pathway as a single, continuous, smoothly-shaded triangulated
  surface, by reading the *same* `pathway.obj`/`pathway.mtl` mesh file
  the VMD script loads directly (rather than re-deriving pathway
  geometry from scratch). Standard PyMOL has no built-in Wavefront OBJ
  importer, so this is a small, Python-3-correct port of the
  parsing/CGO-building technique in upstream CHAP's own bundled
  `wobj.py` (`scripts/visualisation/PyMOL/wobj.py` in the CHAP repo):
  read `v`/`vn`/`f v//vn v//vn v//vn` records from the OBJ and
  `newmtl`/`Kd` records from the MTL, then build one CGO
  `BEGIN, TRIANGLES, ..., END` block with each vertex's exported
  outward normal (for smooth Gouraud shading) and its face's material
  colour. This is the *same technique upstream CHAP itself uses* to
  display its OBJ output in PyMOL, not a reinvention of it.

  This script went through several worse implementations first, worth
  knowing about if you're extending it or wondering why it looks the
  way it does:
  1. A chain of independent CGO `CYLINDER` primitives, one per pathway
     segment -- rendered fine, but looked visibly segmented/faceted at
     each joint (especially where the radius changes a lot between
     consecutive points).
  2. A from-scratch triangulated tube mesh rendered with the plain CGO
     `TRIANGLE` primitive (27 floats, self-contained per triangle).
     Geometrically correct, but invisible in PyMOL's normal interactive
     viewport: PyMOL's own C++ source (`layer1/CGO.h`/`CGO.cpp`)
     confirms `TRIANGLE` has no dedicated real-time-rendering
     conversion function (unlike `SPHERE`/`CYLINDER`/`CONE`, which each
     have one), matching a comment in that source: *"triangle *
     currently for ray-tracing only"*. Only visible via `cmd.ray()`.
  3. A chain of tapered CGO `CONE` segments (same family as `CYLINDER`,
     so reliably real-time rendered, and each segment is
     mathematically round rather than a polygonal approximation). This
     rendered, and fixed version 1's faceted joints, but still didn't
     read as a single continuous pore-lining *surface* -- more like a
     strand of beads.
  4. Reading PyChap's own already-tested `pathway.obj`/`.mtl` mesh
     (the same one the VMD script uses) and converting it to a
     `BEGIN, TRIANGLES, ..., END` CGO block with per-vertex normals and
     per-face colour -- exactly upstream CHAP's own `wobj.py` approach.
     `BEGIN`/`TRIANGLES`/`END` blocks *do* have a real-time rendering
     path in PyMOL (unlike bare `TRIANGLE`); this is confirmed by it
     being CHAP's own shipped technique. Getting vertex normals for
     this required adding `vn` output to `pychap.obj_export` (it
     previously wrote none). Note: upstream CHAP's `wobj.py` reads the
     `Ka` (ambient) MTL field for colour, but PyChap's own MTL writer
     puts the real colour in `Kd` (diffuse) and hardcodes `Ka` to
     black -- this script reads `Kd` accordingly. This version
     rendered as a genuine continuous surface for the first time, but
     had two remaining rough edges (fixed in version 5 below): the
     mesh's circumferential resolution was too low, and the surface
     rendered much darker than its actual assigned colours.
  5. **The current version**, refining version 4 rather than replacing
     it: (a) `pychap.obj_export.export_pathway_obj`'s default
     circumferential resolution (`n_theta`) went from 24 to 48 vertices
     per ring, and its colour-bin count (`n_color_bins`) from 32 to
     64 -- per-vertex normals make Gouraud shading smooth the *colour*
     across each triangle, but they can't hide a low-poly *silhouette*,
     so a 24-gon cross-section still visibly reads as faceted at the
     tube's outline no matter how good the shading is; doubling the
     vertex count makes that outline noticeably rounder. (b)
     `chap_visualise` now explicitly raises PyMOL's `ambient` and
     `direct` lighting settings (and flattens `specular`/`reflect`)
     specifically on the pathway CGO object. This was needed because
     PyMOL's *default* lighting (`ambient` ~0.14, `direct` ~0.45)
     noticeably under-lights a fully round surface like this tube: at
     any moment roughly half its circumference faces away from the
     camera-attached light and gets little more than the dim ambient
     term, so the object as a whole reads as very dark even though the
     underlying colours (verified directly against the real 4PIR mesh:
     mean luminance ≈0.69 across all 250 profile points, i.e. not
     inherently dark) aren't.
- `colour_by_property.py` -- simpler, structure-only variant: colours
  the protein by the B-factor property and shows pore-facing residues
  (occupancy > 0.5) as sticks.

## Usage

```bash
# command line (renders and exits)
pymol -cq scripts/pymol/visualise_pathway.py -- results/annotated.pdb results/pathway.obj
pymol -cq scripts/pymol/colour_by_property.py -- results/annotated.pdb

# or interactively inside PyMOL:
run scripts/pymol/visualise_pathway.py
chap_visualise results/annotated.pdb, results/pathway.obj

# for the bundled 4PIR example, both scripts also accept an optional
# trailing `name` argument, so you can load it alongside other objects
# without them colliding:
pymol -cq scripts/pymol/visualise_pathway.py -- examples/4pir_output/p4pir_annotated.pdb examples/4pir_output/p4pir_pathway.obj p4pir
```

`pathway.mtl` is found automatically via the `mtllib` line in
`pathway.obj` -- it must sit alongside it, which is how
`pychap.obj_export.export_pathway_obj` writes it by default.

**A note on naming:** PyMOL object/selection names may not *start* with
a digit (`load`, when not given an explicit object name, derives one
from the file's basename, and that fails/misbehaves for a leading
digit too). Two places this matters:

- **Filenames.** This repo's generated 4PIR example output files are
  prefixed `p4pir_` (`p4pir_annotated.pdb`, `p4pir_pathway.obj`, ...)
  rather than `4pir_`, specifically so they're safe to `load` directly
  in PyMOL without needing an explicit object name. (The original
  input data, `examples/data/4pirtm.pdb` etc., keeps its upstream name
  and *does* start with a digit -- if you load it directly, give it an
  explicit object name, e.g. `load data/4pirtm.pdb, structure`.)
- **The `name` argument.** Don't pass `name="4pir"` to
  `chap_visualise`/`chap_colour_by_property` -- both scripts detect a
  leading-digit (or otherwise unsafe) name and automatically rewrite it
  (e.g. `4pir` -> `p4pir`), printing a note when they do. Passing an
  already-safe name like `p4pir` up front avoids the rewrite and keeps
  object names predictable.

`annotated.pdb` and `pathway.obj`/`pathway.mtl` come from
`scripts/export_visualisation.py` (for a real structure) or
`examples/run_example.py` / `examples/run_4pir_example.py` (for the
bundled examples) -- see the main README.md.

## Conventions used

Same as the VMD scripts (`../vmd/README.md`):

- **Occupancy (`q` in PyMOL selection syntax)**: 1.0 for pore-facing
  atoms, 0.0 otherwise. Used for selections like `q > 0.5`.
- **B-factor (`b`)**: the continuous property being visualised (e.g.
  hydrophobicity in kcal/mol) for the *structure*'s pore-facing
  residues, coloured via PyMOL's built-in `blue_white_red` spectrum.
- **Pathway colours**: read directly from `pathway.mtl`'s discrete
  colour bins (written by `pychap.obj_export`, a small number of
  materials approximating the `BrBG_r` colormap) -- one flat colour
  per triangle, same as VMD's rendering of the same file.
- **Pathway smoothness**: `visualise_pathway.py` draws the pathway as
  a single triangulated CGO surface with per-vertex normals (smooth
  Gouraud-shaded lighting), not a chain of separate primitives -- so
  it reads as one continuous pore-lining surface rather than a
  sequence of joined segments.
