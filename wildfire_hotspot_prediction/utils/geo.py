"""
utils/geo.py
------------
Stateless geometry utilities shared across pipeline stages.
No domain knowledge — pure coordinate and Shapely operations.
"""

from __future__ import annotations

import numpy as np
from shapely.geometry import MultiPolygon, Polygon


def snap_grid_id(x: float, y: float, grid_res: float = 500.0) -> str:
    """Snap a single projected (x, y) point to the nearest grid cell centre.

    Args:
        x:        Projected x coordinate [m].
        y:        Projected y coordinate [m].
        grid_res: Grid cell size [m]. Defaults to 500.

    Returns:
        String grid cell ID, e.g. "531500_5898000".
    """
    sx = int(round(x / grid_res) * grid_res)
    sy = int(round(y / grid_res) * grid_res)
    return f"{sx}_{sy}"


def snap_grid_ids(xy: np.ndarray, grid_res: float = 500.0) -> np.ndarray:
    """Vectorised snap of projected (x, y) coordinates to grid cell IDs.

    Args:
        xy:       Array of shape (n, 2) with [x, y] columns [m].
        grid_res: Grid cell size [m]. Defaults to 500.

    Returns:
        1-D array of n strings, each of the form "<snapped_x>_<snapped_y>".
    """
    sx = (np.round(xy[:, 0] / grid_res) * grid_res).astype(int)
    sy = (np.round(xy[:, 1] / grid_res) * grid_res).astype(int)
    return np.array([f"{x}_{y}" for x, y in zip(sx, sy)])


def decode_grid_id(grid_id: str) -> tuple[float, float]:
    """Decode a grid cell ID string back to (x, y) coordinates.

    Args:
        grid_id: String of the form "<snapped_x>_<snapped_y>".

    Returns:
        Tuple (x, y) in projected CRS metres.
    """
    x, y = grid_id.split("_")
    return float(x), float(y)


def drop_interior_rings(geom):
    """Remove all interior rings (holes) from a Polygon or MultiPolygon.

    Args:
        geom: Shapely Polygon or MultiPolygon.

    Returns:
        Geometry with no interior rings.
    """
    if geom.geom_type == "Polygon":
        return Polygon(geom.exterior)
    if geom.geom_type == "MultiPolygon":
        return MultiPolygon([Polygon(p.exterior) for p in geom.geoms])
    return geom


def chaikin_smooth(geom, iterations: int = 4):
    """Apply Chaikin's Corner Cutting algorithm to smooth a polygon boundary.

    Each iteration replaces every edge P0→P1 with two points:
        Q = 0.75·P0 + 0.25·P1
        R = 0.25·P0 + 0.75·P1

    Args:
        geom:       Shapely Polygon or MultiPolygon.
        iterations: Number of smoothing iterations. Defaults to 4.

    Returns:
        Smoothed geometry.
    """
    def _smooth_ring(coords: np.ndarray) -> np.ndarray:
        pts = coords[:-1]   # drop closing duplicate
        for _ in range(iterations):
            n   = len(pts)
            p0  = pts
            p1  = np.roll(pts, -1, axis=0)
            new = np.empty((2 * n, 2), dtype=np.float64)
            new[0::2] = 0.75 * p0 + 0.25 * p1
            new[1::2] = 0.25 * p0 + 0.75 * p1
            pts = new
        return np.vstack([pts, pts[0]])   # re-close

    if geom.geom_type == "Polygon":
        return Polygon(_smooth_ring(np.array(geom.exterior.coords)))
    if geom.geom_type == "MultiPolygon":
        return MultiPolygon([
            Polygon(_smooth_ring(np.array(p.exterior.coords)))
            for p in geom.geoms
        ])
    return geom


def sample_boundary_with_normals(
    geom,
    spacing: float,
) -> tuple[np.ndarray, np.ndarray, list]:
    """Sample equidistant points and outward unit normals along polygon exterior rings.

    Args:
        geom:    Shapely Polygon or MultiPolygon.
        spacing: Approximate distance between sample points [m].

    Returns:
        Tuple of (pts_xy, outward_normals, ring_slices):
            pts_xy:          (N, 2) float64 projected coordinates.
            outward_normals: (N, 2) float64 unit outward normal vectors.
            ring_slices:     List of slice objects, one per ring.
    """
    rings = (
        [geom.exterior] if geom.geom_type == "Polygon"
        else [p.exterior for p in geom.geoms]
    )

    all_pts:     list[np.ndarray] = []
    all_normals: list[np.ndarray] = []
    all_slices:  list[slice]      = []
    offset = 0

    for ring in rings:
        coords = np.array(ring.coords)[:-1]    # drop closing duplicate
        n_verts = len(coords)

        # Edge vectors (wrap-around within this ring only)
        edges   = np.roll(coords, -1, axis=0) - coords
        lengths = np.linalg.norm(edges, axis=1)
        cum_len = np.concatenate([[0.0], np.cumsum(lengths)])
        total   = cum_len[-1]

        n_pts = max(int(total / spacing), 2)
        dists = np.linspace(0.0, total, n_pts, endpoint=False)

        pts = np.empty((n_pts, 2), dtype=np.float64)
        for k, d in enumerate(dists):
            i = min(np.searchsorted(cum_len, d, side="right") - 1, n_verts - 1)
            t = (d - cum_len[i]) / max(lengths[i], 1e-10)
            pts[k] = coords[i] + t * edges[i]

        # Outward unit normals: rotate edge 90° → perpendicular
        edge_at_pt = np.roll(pts, -1, axis=0) - pts
        perp = np.column_stack([-edge_at_pt[:, 1], edge_at_pt[:, 0]])
        norms = np.linalg.norm(perp, axis=1, keepdims=True)
        normals = perp / np.maximum(norms, 1e-10)

        # Flip inward normals
        centroid = coords.mean(axis=0)
        outward  = pts - centroid
        flip     = (normals * outward).sum(axis=1) < 0
        normals[flip] *= -1

        all_pts.append(pts)
        all_normals.append(normals)
        all_slices.append(slice(offset, offset + n_pts))
        offset += n_pts

    if not all_pts:
        return np.zeros((0, 2)), np.zeros((0, 2)), []

    return np.vstack(all_pts), np.vstack(all_normals), all_slices
