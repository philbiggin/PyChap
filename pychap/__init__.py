"""
PyChap
======

A Python 3 reimplementation of the core analysis performed by CHAP
(the Channel Annotation Package, https://github.com/channotation/chap).

CHAP upstream is a C++ tool linked directly against libgromacs that finds
the permeation pathway through an ion channel / pore, reports a pore
radius profile along that pathway, and annotates it with physicochemical
properties (e.g. hydrophobicity) of the pore-lining residues.

This package reimplements that core workflow in pure Python:

* :mod:`pychap.trajectory` -- reads topologies/trajectories (including
  GROMACS ``.xtc`` files) via MDAnalysis.
* :mod:`pychap.pathfinding` -- finds the pore centreline and computes the
  pore radius profile along it.
* :mod:`pychap.hydrophobicity` -- maps residue hydrophobicity onto the
  permeation pathway.
* :mod:`pychap.analysis` -- ties the above together into a single-frame
  or trajectory-averaged analysis, with JSON/CSV export.
* :mod:`pychap.cli` -- a command line interface loosely modelled on the
  original ``chap`` executable.

See the project ``README.md`` for a discussion of how this port differs
numerically from the original C++ implementation.
"""

from .analysis import PoreAnalysis, PoreAnalysisResult, FrameResult, ResidueSummary
from .pathfinding import compute_pore_profile, PoreProfile
from .residues import project_onto_path, project_residues
from .spline import PathSpline

__version__ = "0.1.0"

__all__ = [
    "PoreAnalysis",
    "PoreAnalysisResult",
    "FrameResult",
    "ResidueSummary",
    "compute_pore_profile",
    "PoreProfile",
    "project_onto_path",
    "project_residues",
    "PathSpline",
    "__version__",
]
