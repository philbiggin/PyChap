"""
colour_by_property.py -- PyMOL script

A simpler companion to visualise_pathway.py: loads just a pychap
annotated structure PDB (no pathway JSON needed) and colours the whole
protein by whatever property is stored in the B-factor column, with
pore-facing residues (occupancy > 0.5, see pychap.pdb_export)
additionally shown as sticks so they stand out.

Usage (command line)
---------------------
    pymol -cq colour_by_property.py -- annotated.pdb

Usage (inside an interactive PyMOL session)
---------------------------------------------
    run colour_by_property.py
    chap_colour_by_property annotated.pdb

    # naming multiple sessions' objects distinctly, e.g. for the bundled
    # 4PIR example -- note PyMOL object/selection names may not *start*
    # with a digit, so a name like "4pir" is automatically rewritten to
    # "p4pir" (see _safe_pymol_name below); pass an already-safe name to
    # avoid the rewrite, e.g.:
    chap_colour_by_property examples/4pir_output/p4pir_annotated.pdb, p4pir

NOTE: written against the standard, documented PyMOL Python API but
this package was built without PyMOL installed to test against --
please treat first use as a check.
"""

from __future__ import annotations

import re
import sys

from pymol import cmd


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


def chap_colour_by_property(structure_pdb: str, name: str = "chap"):
    name = _safe_pymol_name(name)
    cmd.load(structure_pdb, name)
    cmd.hide("everything", name)

    cmd.show("cartoon", name)
    cmd.set("cartoon_transparency", 0.5, name)
    cmd.spectrum("b", "blue_white_red", name)

    pore_facing_sel = f"{name} and q > 0.5"
    cmd.select(f"{name}_pore_facing", pore_facing_sel)
    cmd.show("sticks", f"{name}_pore_facing")
    cmd.spectrum("b", "blue_white_red", f"{name}_pore_facing")

    cmd.bg_color("white")
    cmd.orient(name)

    print(
        f"pychap: loaded '{name}', coloured by B-factor (blue_white_red); "
        f"pore-facing residues ('{name}_pore_facing', occupancy > 0.5) shown as sticks."
    )


cmd.extend("chap_colour_by_property", chap_colour_by_property)


def _main():
    argv = sys.argv[1:]
    if len(argv) < 1:
        print("Usage: pymol -cq colour_by_property.py -- annotated.pdb [name]")
        return
    name = argv[1] if len(argv) > 1 else "chap"
    chap_colour_by_property(argv[0], name=name)


# See visualise_pathway.py for why this guard checks argv length rather
# than just running unconditionally -- it only auto-runs when invoked as
# `pymol -cq colour_by_property.py -- <args>`.
if len(sys.argv) > 1:
    _main()
