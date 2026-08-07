#
# colour_by_property.tcl
#
# A simpler companion to visualise_pathway.tcl: loads just a pychap
# annotated structure PDB (no pathway mesh needed) and colours the
# whole protein by whatever property is stored in the B-factor column,
# with pore-facing residues (occupancy > 0.5, see pychap.pdb_export)
# additionally drawn as licorice sticks so they stand out.
#
# Handy when you just want a quick look at a property (hydrophobicity,
# radial distance from the pathway, ...) without generating a pathway
# OBJ mesh first.
#
# Usage:
#
#   vmd -e colour_by_property.tcl -args structure.pdb
#
# NOTE: written against the standard VMD Tcl API but not tested against
# a real VMD installation in the environment this package was built in.
#

if { $argc < 1 } {
    puts "Usage: vmd -e colour_by_property.tcl -args structure.pdb"
    exit 1
}

set structure_pdb [lindex $argv 0]
puts "pychap: loading structure from $structure_pdb"

set mol_id [mol new $structure_pdb waitfor all]
mol rename $mol_id "pychap structure"
mol delrep 0 $mol_id

# Whole protein as a thin tube, coloured by the B-factor property.
mol representation Tube 0.3 12.0
mol color Beta
mol selection "protein"
mol material Opaque
mol addrep $mol_id

# Pore-facing residues as licorice sticks, same colouring.
mol representation Licorice 0.3 12.0 12.0
mol color Beta
mol selection "occupancy > 0.5"
mol material Opaque
mol addrep $mol_id

color scale method BWR

display projection Orthographic
color Display Background white
axes location Off
display resetview

puts "pychap: done. Coloured by Beta (property value); pore-facing residues shown as licorice."
