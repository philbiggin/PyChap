"""
visualise_pathway.py -- PyMOL script

Load a pychap-annotated structure alongside its permeation pathway
mesh (``pathway.obj`` + ``pathway.mtl``), rendered as a smooth,
per-face-coloured CGO surface -- PyMOL's counterpart to
scripts/vmd/visualise_pathway.tcl, following the same conventions as
upstream CHAP's own VMD/PyMOL visualisation workflow (see
https://www.channotation.org/docs/molecular_graphics_pymol/).

Unlike VMD, standard PyMOL has no built-in Wavefront OBJ mesh importer.
Upstream CHAP works around this with a bundled ``wobj.py`` module that
parses its own OBJ+MTL output and converts it into a PyMOL Compiled
Graphics Object (CGO) -- reading the *same* mesh file used for the VMD
rendering, not re-deriving pathway geometry from scratch. This script
does the same thing for pychap's OBJ/MTL output
(``pychap.obj_export.export_pathway_obj``): it's a small, Python
3-compatible, single-mesh-only port of that parsing/CGO-building
technique (``import_wobj``/``draw_wobj`` in upstream CHAP's ``wobj.py``),
built directly into this file so no extra module needs to be copied
around. Concretely: for each triangle it reads pychap's exported
per-vertex outward normal (``vn`` records, added specifically so this
approach would work -- see ``pychap.obj_export``) for smooth Gouraud
shading, and its material's diffuse colour (``Kd``, from the matching
``.mtl`` file) for a flat per-triangle colour -- built as a single CGO
``BEGIN, TRIANGLES, ..., END`` block, exactly as upstream CHAP's own
script does.

(This script went through several other implementations first, each
worse: independent CYLINDER primitives per pathway segment, which
looked visibly faceted; a from-scratch triangulated mesh rendered with
PyMOL's plain ``TRIANGLE`` CGO primitive, which turned out to have no
real-time rendering path at all in PyMOL -- confirmed in PyMOL's own
C++ source, which has a dedicated real-time conversion function for
``SPHERE``/``CYLINDER``/``CONE`` but not for bare ``TRIANGLE`` -- so it
was invisible outside of ray-tracing; and a chain of tapered CONE
primitives, which *did* render but only ever looks like a strand of
beads, not a continuous surface. Reading pychap's own already-tested
OBJ/MTL mesh -- the same technique upstream CHAP itself uses -- avoids
all of the above.

Two follow-up refinements once the mesh itself was rendering: (1) the
circumferential resolution of the tube mesh (``n_theta`` in
``pychap.obj_export.export_pathway_obj``) needed to go up from its
original 24 to 48 -- Gouraud shading (via the per-vertex normals above)
smooths *colour* across a triangle, but it can't hide a low-poly
*silhouette*, so a 24-gon cross-section still visibly reads as faceted
at the tube's outline even when perfectly lit. (2) PyMOL's default
lighting (``ambient`` ~0.14, ``direct`` ~0.45) noticeably under-lights
a fully round CGO surface -- roughly half the circumference faces away
from the camera-attached light at any moment and gets little more than
the dim ambient term, so the object as a whole reads as very dark even
though the assigned colours themselves aren't. ``chap_visualise`` now
raises ``ambient``/``direct`` and flattens ``specular``/``reflect``
specifically on the pathway object to compensate, without affecting the
structure's cartoon/spheres representations.)

Usage (command line)
---------------------
    pymol -cq visualise_pathway.py -- annotated.pdb pathway.obj [name]

    # pathway.mtl is found automatically via the "mtllib" line in
    # pathway.obj (must sit alongside it, which is how
    # pychap.obj_export.export_pathway_obj writes it by default)

Usage (inside an interactive PyMOL session)
---------------------------------------------
    run visualise_pathway.py
    chap_visualise annotated.pdb, pathway.obj

    # naming multiple sessions' objects distinctly, e.g. for the bundled
    # 4PIR example (examples/4pir_output/p4pir_annotated.pdb etc., already
    # named to avoid a leading digit -- see examples/data/README.md) --
    # note PyMOL object/selection names may not *start* with a digit
    # regardless, so a `name` argument like "4pir" is automatically
    # rewritten to "p4pir" (see _safe_pymol_name below); pass an
    # already-safe name to avoid the rewrite, e.g.:
    chap_visualise examples/4pir_output/p4pir_annotated.pdb, examples/4pir_output/p4pir_pathway.obj, p4pir

NOTE: written against the standard, documented PyMOL Python API
(``pymol.cmd``, ``pymol.cgo``) but this package was built in a sandbox
without PyMOL installed, so it could not be executed or visually
verified there. Please treat first use as a check.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from pymol import cmd
from pymol.cgo import BEGIN, COLOR, END, NORMAL, TRIANGLES, VERTEX


def _safe_pymol_name(name: str) -> str:
    """Make `name` safe to use as a PyMOL object/selection name.

    PyMOL's selection-language parser doesn't allow object/selection
    names to *start* with a digit (it conflicts with numeric literals in
    selection expressions, e.g. `resi 4`) -- so a name like "4pir"
    (a natural choice given this repo's "4pir_"-prefixed example output
    files) would make `cmd.select(...)`/`cmd.load(...)` calls built from
    it fail or behave unpredictably. This prepends a letter if needed,
    and replaces any other character that isn't alphanumeric/underscore.
    """
    safe = re.sub(r"[^0-9A-Za-z_]", "_", name)
    if not safe or safe[0].isdigit():
        safe = f"p{safe}"
    if safe != name:
        print(f"pychap: PyMOL object names can't start with a digit -- using '{safe}' instead of '{name}'.")
    return safe


# --- OBJ/MTL parsing -------------------------------------------------------
# A small, Python 3-correct, single-mesh-only port of the parsing logic in
# upstream CHAP's own scripts/visualisation/PyMOL/wobj.py -- reading
# vertices ("v"), vertex normals ("vn"), materials ("newmtl"/"Kd"), and
# faces ("f v//vn v//vn v//vn"), exactly the subset pychap.obj_export
# writes. Deliberately doesn't handle OBJ groups ("g") or texture
# coordinates, since pychap never writes either.

def _read_mtl_colors(mtl_path):
    """Parse `newmtl`/`Kd` records from an MTL file into
    {material_name: (r, g, b)}. pychap.obj_export writes the visible
    colour into `Kd` (diffuse); `Ka` (ambient) is always black there, so
    -- unlike upstream CHAP's wobj.py, which reads `Ka` -- this reads
    `Kd`, matching what pychap actually puts the colour in."""
    colors = {}
    current = None
    with open(mtl_path) as fh:
        for line in fh:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "newmtl":
                current = parts[1]
            elif parts[0] == "Kd" and current is not None:
                colors[current] = tuple(float(x) for x in parts[1:4])
    return colors


def _read_obj_mesh(obj_path):
    """Parse an OBJ file written by pychap.obj_export: `v`, `vn`,
    `usemtl`, `mtllib`, and `f v//vn v//vn v//vn` records only.

    Returns (vertices, normals, faces, mtl_path), where each face is
    ((v0, n0), (v1, n1), (v2, n2), material_name) with 0-based indices
    into `vertices`/`normals`, and `mtl_path` is resolved relative to
    `obj_path`'s directory (None if no `mtllib` line was found).
    """
    obj_path = Path(obj_path)
    vertices = []
    normals = []
    faces = []
    mtl_path = None
    current_material = None

    with open(obj_path) as fh:
        for line in fh:
            parts = line.split()
            if not parts or parts[0].startswith("#"):
                continue
            tag = parts[0]
            if tag == "mtllib":
                mtl_path = obj_path.parent / parts[1]
            elif tag == "v":
                vertices.append(tuple(float(x) for x in parts[1:4]))
            elif tag == "vn":
                normals.append(tuple(float(x) for x in parts[1:4]))
            elif tag == "usemtl":
                current_material = parts[1]
            elif tag == "f":
                corners = []
                for token in parts[1:4]:
                    v_part, _, vn_part = token.partition("//")
                    corners.append((int(v_part) - 1, int(vn_part) - 1))
                if len(corners) == 3:
                    faces.append((corners[0], corners[1], corners[2], current_material))

    return vertices, normals, faces, mtl_path


def _build_pathway_cgo(obj_path):
    """Build a CGO object list from a pychap pathway.obj/.mtl pair: a
    single ``BEGIN, TRIANGLES, ..., END`` block with each vertex's
    exported outward normal (for smooth Gouraud shading) and its face's
    material colour (flat per triangle, matching the discrete colour
    bins pychap.obj_export writes) -- the same technique upstream
    CHAP's own wobj.py uses to display its OBJ output in PyMOL.

    Returns (cgo_object_list, n_triangles).
    """
    vertices, normals, faces, mtl_path = _read_obj_mesh(obj_path)

    if not faces:
        print(f"pychap: no faces found in {obj_path} -- nothing to draw.")
        return [], 0
    if mtl_path is None or not mtl_path.exists():
        print(
            f"pychap: couldn't find the .mtl file referenced by {obj_path} "
            f"({mtl_path}) -- drawing the pathway in a flat default grey. "
            "Make sure pathway.mtl sits alongside pathway.obj."
        )
        colors = {}
    else:
        colors = _read_mtl_colors(mtl_path)

    default_color = (0.6, 0.6, 0.6)
    obj = [BEGIN, TRIANGLES]
    for (v0, n0), (v1, n1), (v2, n2), material in faces:
        colour = colors.get(material, default_color)
        for vi, ni in ((v0, n0), (v1, n1), (v2, n2)):
            vert = vertices[vi]
            norm = normals[ni]
            obj.extend([COLOR, colour[0], colour[1], colour[2]])
            obj.extend([NORMAL, norm[0], norm[1], norm[2]])
            obj.extend([VERTEX, vert[0], vert[1], vert[2]])
    obj.append(END)
    return obj, len(faces)


def chap_visualise(structure_pdb: str, obj_path: str, name: str = "chap"):
    """Load a pychap-annotated structure and its pathway mesh
    (``pathway.obj`` + ``pathway.mtl``) into the current PyMOL session.
    Registered as a PyMOL command (see bottom of file), so it can also
    be called interactively as:

        chap_visualise structure.pdb, pathway.obj
    """
    name = _safe_pymol_name(name)

    struct_name = f"{name}_structure"
    cmd.load(structure_pdb, struct_name)
    cmd.hide("everything", struct_name)
    cmd.show("cartoon", struct_name)
    cmd.color("grey80", struct_name)

    pore_facing_sel = f"{struct_name} and q > 0.5"
    cmd.select(f"{name}_pore_facing", pore_facing_sel)
    cmd.show("spheres", f"{name}_pore_facing")
    cmd.spectrum("b", "blue_white_red", f"{name}_pore_facing")

    pathway_obj, n_triangles = _build_pathway_cgo(obj_path)
    pathway_name = f"{name}_pathway"
    cmd.load_cgo(pathway_obj, pathway_name)
    cmd.set("cgo_transparency", 0.15, pathway_name)
    # Lights both faces of every triangle -- the mesh's winding isn't
    # guaranteed consistent with "outward" from every camera angle, and
    # without this, wrongly-facing triangles render dark/flat.
    cmd.set("two_sided_lighting", 1)
    cmd.set("backface_cull", 0)
    # PyMOL's default lighting (ambient ~0.14, direct ~0.45) noticeably
    # under-lights a fully round, curved CGO surface like this tube: the
    # half of the circumference facing away from the camera-attached
    # light gets little more than the ambient term, so the whole object
    # reads as "very dark" even though the assigned per-vertex colours
    # themselves aren't -- raising ambient/direct and flattening out
    # specular (which otherwise creates a bright glare band that makes
    # the rest look comparatively darker) keeps the shading that gives
    # the tube its rounded look while keeping the actual colours visible
    # all the way round. Scoped to this object only, so it doesn't affect
    # the cartoon/spheres representations of the structure.
    cmd.set("ambient", 0.45, pathway_name)
    cmd.set("direct", 0.65, pathway_name)
    cmd.set("reflect", 0.3, pathway_name)
    cmd.set("specular", 0, pathway_name)

    cmd.bg_color("white")
    cmd.set("ray_opaque_background", 0)
    cmd.orient(struct_name)

    print(
        f"pychap: loaded '{struct_name}' (pore-facing residues selected as "
        f"'{name}_pore_facing', coloured by B-factor) and '{pathway_name}' "
        f"({n_triangles} mesh triangles from {obj_path})."
    )


cmd.extend("chap_visualise", chap_visualise)


def _main():
    argv = sys.argv[1:]
    if len(argv) < 2:
        print("Usage: pymol -cq visualise_pathway.py -- annotated.pdb pathway.obj [name]")
        return
    structure_pdb = argv[0]
    obj_path = argv[1]
    name = argv[2] if len(argv) > 2 else "chap"
    chap_visualise(structure_pdb, obj_path, name=name)


# Auto-run only when invoked as `pymol -cq visualise_pathway.py -- <args>`,
# which is the only case where PyMOL populates sys.argv with real
# arguments for this script. If you `run` this file interactively inside
# an existing PyMOL session instead, sys.argv won't have these arguments
# -- call chap_visualise(...) directly instead (see the module docstring).
if len(sys.argv) > 2:
    _main()
