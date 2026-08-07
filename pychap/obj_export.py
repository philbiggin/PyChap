"""
Export the pore pathway as a coloured Wavefront OBJ + MTL mesh.

CHAP produces an ``output.obj`` / ``output.mtl`` pair representing the
permeation pathway as a tube surface, coloured by a scalar property
(e.g. solvent free energy), which is then loaded into VMD or PyMOL
alongside the annotated structure (see
https://www.channotation.org/docs/molecular_graphics_vmd/ and
scripts/vmd/visualise_pathway.tcl in this package).

This module reimplements that idea: given a pathway (points + radius,
typically the time-averaged pathway from
``PoreAnalysisResult.points_mean`` / ``radius_mean``) and a scalar value
at each point (e.g. mean hydrophobicity or mean radius itself), it
builds a tube mesh (:mod:`pychap.tube`) and writes it out with the
scalar value baked in as a small number of discrete materials (a
standard technique for colouring OBJ meshes, since the format has no
native per-vertex colour attribute that all viewers respect reliably --
discrete per-face materials are far more portable), plus a per-vertex
outward-normal (``vn``) record for every vertex so viewers can shade the
tube smoothly rather than faceting it at each ring -- this is also what
``scripts/pymol/visualise_pathway.py`` reads back out (alongside the
``Kd`` colour from the matching ``.mtl`` material) to rebuild the same
mesh as native PyMOL CGO geometry, mirroring upstream CHAP's own
``wobj.py`` approach (see
https://www.channotation.org/docs/molecular_graphics_pymol/).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .tube import build_tube_mesh, tube_faces


def _colour_bins(values: np.ndarray, n_bins: int, cmap_name: str, symmetric: bool):
    """Discretise `values` into n_bins colour bins using a matplotlib colormap.

    Returns (bin_index_per_value, bin_center_values, bin_rgb_colours).
    """
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    values = np.asarray(values, dtype=float)

    if symmetric:
        vmax = float(np.max(np.abs(values))) if values.size else 1.0
        vmax = vmax if vmax > 0 else 1.0
        vmin = -vmax
    else:
        vmin = float(np.min(values)) if values.size else 0.0
        vmax = float(np.max(values)) if values.size else 1.0
        if vmax <= vmin:
            vmax = vmin + 1.0

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap(cmap_name)

    bin_edges = np.linspace(vmin, vmax, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_colours = [cmap(norm(c))[:3] for c in bin_centers]

    bin_index = np.clip(np.digitize(values, bin_edges) - 1, 0, n_bins - 1)
    return bin_index, bin_centers, bin_colours


def export_pathway_obj(
    points: np.ndarray,
    radius: np.ndarray,
    s: np.ndarray,
    color_values: np.ndarray,
    obj_path,
    mtl_path=None,
    n_theta: int = 48,
    n_color_bins: int = 64,
    cmap_name: str = "BrBG_r",
    symmetric_color_limits: bool = True,
    value_label: str = "property",
):
    """Write a coloured tube mesh for the pathway as an OBJ + MTL file pair.

    Parameters
    ----------
    points, radius, s:
        Pathway centreline points ``(N, 3)``, radius ``(N,)``, and
        arc-length coordinate ``(N,)`` -- typically
        ``result.points_mean``, ``result.radius_mean``, and
        ``result.s_grid * result.mean_length`` for the time-averaged
        pathway, or a single frame's data.
    color_values:
        ``(N,)`` scalar value at each point used to colour the mesh
        (e.g. ``result.hydrophobicity_mean`` or ``result.radius_mean``).
    obj_path, mtl_path:
        Output paths. If ``mtl_path`` is omitted, it is derived from
        ``obj_path`` by replacing the extension with ``.mtl``.
    n_theta:
        Number of vertices around each cross-sectional ring.
    n_color_bins:
        Number of discrete colour bins (materials) used to approximate
        the continuous colormap -- higher looks smoother but produces a
        larger file.
    cmap_name:
        Matplotlib colormap name (default ``BrBG_r``, matching the
        diverging hydrophobicity colour scale used in
        scripts/plot_pathway_profile.py and by CHAP itself).
    symmetric_color_limits:
        If True (default, appropriate for a diverging quantity like
        hydrophobicity), colour limits are ``[-max(|value|), +max(|value|)]``.
        If False, limits are simply ``[min(value), max(value)]`` (more
        appropriate for a one-sided quantity like radius).
    value_label:
        Human-readable label for the coloured property, written into the
        file header comments only.

    Returns
    -------
    (obj_path, mtl_path) as ``pathlib.Path`` objects.
    """
    obj_path = Path(obj_path)
    mtl_path = Path(mtl_path) if mtl_path is not None else obj_path.with_suffix(".mtl")

    points = np.asarray(points, dtype=float)
    radius = np.asarray(radius, dtype=float)
    s = np.asarray(s, dtype=float)
    color_values = np.asarray(color_values, dtype=float)

    if not (len(points) == len(radius) == len(s) == len(color_values)):
        raise ValueError("points, radius, s, and color_values must all have the same length")
    if len(points) < 2:
        raise ValueError("need at least 2 pathway points to export a mesh")

    rings = build_tube_mesh(points, radius, s, n_theta=n_theta)
    n_points, n_theta_actual, _ = rings.shape

    # Outward-pointing unit normal at each ring vertex: by construction,
    # (rings[i, j] - points[i]) / radius[i] is exactly this direction
    # already (see pychap.tube.build_tube_mesh) -- re-normalising here
    # only guards against float error, it isn't doing real work.
    offsets = rings - points[:, None, :]
    offset_norms = np.linalg.norm(offsets, axis=2, keepdims=True)
    offset_norms[offset_norms == 0] = 1.0
    normals = offsets / offset_norms

    bin_index, bin_centers, bin_colours = _colour_bins(
        color_values, n_color_bins, cmap_name, symmetric_color_limits
    )

    with open(mtl_path, "w") as fh:
        fh.write(f"# pychap pathway material file, coloured by {value_label}\n")
        fh.write(f"# {n_color_bins} discrete bins approximating the '{cmap_name}' colormap\n")
        for k, colour in enumerate(bin_colours):
            fh.write(f"newmtl bin_{k}\n")
            fh.write(f"Kd {colour[0]:.4f} {colour[1]:.4f} {colour[2]:.4f}\n")
            fh.write("Ka 0.0 0.0 0.0\n")
            fh.write("Ks 0.15 0.15 0.15\n")
            fh.write("Ns 10.0\n")
            fh.write("d 1.0\n")
            fh.write("illum 2\n\n")

    with open(obj_path, "w") as fh:
        fh.write(f"# pychap pore pathway mesh, coloured by {value_label}\n")
        fh.write(f"# {n_points} rings x {n_theta_actual} vertices\n")
        fh.write(f"mtllib {mtl_path.name}\n")

        for i in range(n_points):
            for j in range(n_theta_actual):
                v = rings[i, j]
                fh.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")

        # Vertex normals, one per "v" line above and in the same order,
        # so normal index N corresponds to vertex index N throughout --
        # written separately (all "vn" lines after all "v" lines) purely
        # for readability; OBJ doesn't require any particular ordering.
        for i in range(n_points):
            for j in range(n_theta_actual):
                vn = normals[i, j]
                fh.write(f"vn {vn[0]:.4f} {vn[1]:.4f} {vn[2]:.4f}\n")

        current_bin = None
        for i, (v00, v10, v11, v01) in tube_faces(n_points, n_theta_actual):
            face_bin = int(round(0.5 * (bin_index[i] + bin_index[min(i + 1, n_points - 1)])))
            if face_bin != current_bin:
                fh.write(f"usemtl bin_{face_bin}\n")
                current_bin = face_bin
            # OBJ vertex/normal indices are 1-based; "v//vn" means
            # vertex index // (no texture coord) // normal index. Since
            # normal index N == vertex index N here, this looks
            # redundant but keeps the file format fully general/valid.
            fh.write(f"f {v00 + 1}//{v00 + 1} {v10 + 1}//{v10 + 1} {v11 + 1}//{v11 + 1}\n")
            fh.write(f"f {v00 + 1}//{v00 + 1} {v11 + 1}//{v11 + 1} {v01 + 1}//{v01 + 1}\n")

    return obj_path, mtl_path
