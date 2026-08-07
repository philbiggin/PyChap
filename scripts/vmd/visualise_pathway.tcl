#
# visualise_pathway.tcl
#
# Load a pychap-annotated protein structure alongside its permeation
# pathway mesh in VMD, mirroring upstream CHAP's own visualisation
# workflow (see https://www.channotation.org/docs/molecular_graphics_vmd/
# and https://www.channotation.org/docs/annotation_example/):
#
#   The protein is shown as a cartoon, with pore-lining residues (as
#   flagged by pychap in the PDB occupancy column) additionally drawn
#   as van der Waals spheres coloured by a property stored in the
#   B-factor column (e.g. hydrophobicity or radial distance from the
#   pathway). The permeation pathway itself is loaded as a pre-coloured
#   OBJ/MTL surface mesh (see pychap.obj_export), produced by
#   scripts/export_visualisation.py.
#
# Usage (matches CHAP's own invocation style):
#
#   vmd -e visualise_pathway.tcl -args structure.pdb pathway.obj [title]
#
# where:
#   structure.pdb  -- a PDB file with pychap's B-factor/occupancy
#                      annotations (see scripts/export_visualisation.py
#                      or examples/run_example.py)
#   pathway.obj    -- the coloured pathway mesh (pathway.mtl must sit
#                      alongside it, as written by pychap.obj_export)
#   title          -- optional text label used only in a console message
#
# NOTE: this script was written against the standard, documented VMD
# Tcl scripting API but could not be tested against a real VMD
# installation in the environment this package was built in (no VMD
# available there). Please treat it as a solid starting point and check
# it renders as expected on your own machine -- and feel free to tweak
# representations/materials/colours to taste.
#

# --- parse arguments -------------------------------------------------

if { $argc < 2 } {
    puts "Usage: vmd -e visualise_pathway.tcl -args structure.pdb pathway.obj \[title\]"
    exit 1
}

set structure_pdb [lindex $argv 0]
set pathway_obj    [lindex $argv 1]
set title [expr {$argc > 2 ? [lindex $argv 2] : "pychap pathway"}]

puts "pychap: loading structure from $structure_pdb"
puts "pychap: loading pathway mesh from $pathway_obj"
puts "pychap: title: $title"

# --- load and represent the protein structure -------------------------

set structure_mol [mol new $structure_pdb waitfor all]
mol rename $structure_mol "pychap structure"

# Remove the default representation so we can define our own.
mol delrep 0 $structure_mol

# Representation 0: the whole protein as a cartoon/ribbon, coloured by
# chain (swap "Chain" for "ResType" or "Structure" to taste).
mol representation NewCartoon
mol color Chain
mol selection "protein"
mol material Opaque
mol addrep $structure_mol

# Representation 1: pore-lining residues (flagged via occupancy > 0.5
# by pychap's PDB export -- see pychap.pdb_export / occupancies
# argument) shown as van der Waals spheres, coloured by the continuous
# property stored in the B-factor column (e.g. hydrophobicity).
mol representation VDW 0.6 12.0
mol color Beta
mol selection "occupancy > 0.5"
mol material Opaque
mol addrep $structure_mol

# Use a diverging-ish built-in colour scale for the Beta colouring,
# closest available VMD equivalent to the BrBG_r scale used in
# pychap's own matplotlib plots (scripts/plot_pathway_profile.py).
color scale method BWR

# --- load and display the permeation pathway mesh ----------------------

# The OBJ/MTL pair already encodes its own per-face colours (baked in
# by pychap.obj_export as a set of discrete materials approximating a
# colormap), so no further colouring commands are needed here -- VMD's
# OBJ plugin will honour the referenced .mtl file automatically as long
# as it sits alongside the .obj file.
set pathway_mol [mol new $pathway_obj type {obj} waitfor all]
mol rename $pathway_mol "pychap pathway"

# --- view setup ----------------------------------------------------------

display projection Orthographic
display depthcue off
color Display Background white
axes location Off

mol top $structure_mol
display resetview

puts "pychap: done. Structure is molecule $structure_mol, pathway is molecule $pathway_mol."
puts "pychap: pore-lining residues are selected via 'occupancy > 0.5' and coloured by Beta."
