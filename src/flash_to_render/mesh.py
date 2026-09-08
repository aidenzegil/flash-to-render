"""Polygons -> extruded, closed triangle meshes.

Every :class:`~flash_to_render.trace.PolygonWithHoles` becomes a flat slab:
a front cap, a back cap and one quad wall per ring edge. Vertices are
duplicated per face group so caps and walls keep hard, flat normals in glTF;
:meth:`Mesh.edge_report` merges them by position again to check the result is
a closed surface (every edge shared by exactly two triangles).

Coordinate conventions
----------------------
Input polygons live in image space (x right, y **down**, pixels). Output
vertices live in a y-**up** frame centred on the crop, so y is flipped exactly
once here and nowhere else. Front cap = +z, back cap = -z.

Normals and winding
-------------------
Rings are oriented with shapely (exterior positive signed area, holes negative)
so the outward normal of an edge ``p0 -> p1`` is ``(dy, -dx)`` for both outers
and holes. Every triangle is then checked geometrically against its target
normal and flipped if it disagrees, so the mesh renders correctly even when the
viewer culls back faces.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import mapbox_earcut as earcut
import numpy as np
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.polygon import orient

from .trace import PolygonWithHoles

__all__ = ["Mesh", "extrude", "depth_for", "DEFAULT_DEPTH_FRAC"]

DEFAULT_DEPTH_FRAC = 0.035
"""Extrusion depth as a fraction of the piece's longest side. ~3-4% reads as a cast object; 6%+ reads as a slab."""


@dataclass
class Mesh:
    """Indexed triangle mesh with per-vertex normals. ``faces`` are ``(M, 3)`` uint32."""

    vertices: np.ndarray
    normals: np.ndarray
    faces: np.ndarray
    depth: float = 0.0

    @property
    def triangle_count(self) -> int:
        return int(len(self.faces))

    @property
    def vertex_count(self) -> int:
        return int(len(self.vertices))

    @property
    def is_empty(self) -> bool:
        return len(self.faces) == 0

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return self.vertices.min(axis=0), self.vertices.max(axis=0)

    def edge_report(self, tol: float = 1e-5) -> tuple[int, int]:
        """Merge vertices by position, then count ``(total_edges, edges_not_shared_by_exactly_two_faces)``.

        ``(n, 0)`` means the surface is closed ("manifold-ish").
        """
        if self.is_empty:
            return 0, 0
        keys = np.round(self.vertices / tol).astype(np.int64)
        _uniq, inverse = np.unique(keys, axis=0, return_inverse=True)
        f = inverse.reshape(-1)[self.faces]
        keep = (f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])
        f = f[keep]
        edges = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
        edges.sort(axis=1)
        _u, counts = np.unique(edges, axis=0, return_counts=True)
        return int(len(counts)), int(np.count_nonzero(counts != 2))

    def concatenate(self, other: "Mesh") -> "Mesh":
        return Mesh(
            vertices=np.concatenate([self.vertices, other.vertices]),
            normals=np.concatenate([self.normals, other.normals]),
            faces=np.concatenate([self.faces, other.faces + len(self.vertices)]).astype(np.uint32),
            depth=max(self.depth, other.depth),
        )


def depth_for(longest_side_px: float, depth_frac: float = DEFAULT_DEPTH_FRAC) -> float:
    """Extrusion depth (px) for a piece whose longest side is ``longest_side_px``."""
    return float(longest_side_px) * depth_frac


def _clean_polygons(polys: Sequence[PolygonWithHoles]) -> list[Polygon]:
    """Validate and orient every polygon; invalid ones are repaired with ``buffer(0)`` and may split."""
    out: list[Polygon] = []
    for p in polys:
        try:
            geom = Polygon(p.outer, p.holes)
            if not geom.is_valid:
                geom = geom.buffer(0)
        except Exception:
            continue
        parts = geom.geoms if isinstance(geom, MultiPolygon) else [geom]
        for part in parts:
            if part.is_empty or part.area <= 0 or not isinstance(part, Polygon):
                continue
            out.append(orient(part, sign=1.0))
    return out


def _rings(poly: Polygon) -> list[np.ndarray]:
    rings = [np.asarray(poly.exterior.coords, dtype=np.float64)[:-1]]
    rings += [np.asarray(r.coords, dtype=np.float64)[:-1] for r in poly.interiors]
    return [r for r in rings if len(r) >= 3]


def _flip_to_match(vertices: np.ndarray, faces: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return ``faces`` with any triangle whose geometric normal opposes ``target`` reversed."""
    if len(faces) == 0:
        return faces
    a, b, c = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    n = np.cross(b - a, c - a)
    wrong = np.einsum("ij,ij->i", n, target) < 0
    faces = faces.copy()
    faces[wrong] = faces[wrong][:, [0, 2, 1]]
    return faces


def extrude(
    polys: Sequence[PolygonWithHoles],
    depth: float,
    center: tuple[float, float] | None = None,
    scale: float = 1.0,
) -> Mesh:
    """Extrude nested polygons (image space, px) into a closed slab of thickness ``depth`` (px).

    ``center`` (px, image space) becomes the origin; ``scale`` multiplies every
    coordinate *before* it is stored, so quantising exporters never see values
    that round to zero.
    """
    cleaned = _clean_polygons(polys)
    if not cleaned:
        return Mesh(np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint32), depth * scale)

    if center is None:
        allpts = np.concatenate([np.asarray(p.exterior.coords) for p in cleaned])
        center = ((allpts[:, 0].min() + allpts[:, 0].max()) / 2, (allpts[:, 1].min() + allpts[:, 1].max()) / 2)
    cx, cy = center
    hz = depth / 2.0

    def to3d(xy: np.ndarray, z: float) -> np.ndarray:
        out = np.empty((len(xy), 3), dtype=np.float64)
        out[:, 0] = (xy[:, 0] - cx) * scale
        out[:, 1] = -(xy[:, 1] - cy) * scale  # the one and only y flip
        out[:, 2] = z * scale
        return out

    V: list[np.ndarray] = []
    N: list[np.ndarray] = []
    F: list[np.ndarray] = []
    base = 0

    def push(verts: np.ndarray, normals: np.ndarray, faces: np.ndarray) -> None:
        nonlocal base
        V.append(verts)
        N.append(normals)
        F.append(faces + base)
        base += len(verts)

    for poly in cleaned:
        rings = _rings(poly)
        if not rings:
            continue
        verts2d = np.concatenate(rings)
        ring_ends = np.cumsum([len(r) for r in rings]).astype(np.uint32)
        tris = np.asarray(earcut.triangulate_float64(verts2d, ring_ends), dtype=np.uint32).reshape(-1, 3)
        if len(tris) == 0:
            continue

        # caps
        front = to3d(verts2d, +hz)
        back = to3d(verts2d, -hz)
        up = np.tile(np.array([[0.0, 0.0, 1.0]]), (len(front), 1))
        push(front, up, _flip_to_match(front, tris, up[tris[:, 0]]))
        push(back, -up, _flip_to_match(back, tris, -up[tris[:, 0]]))

        # walls: one quad per ring edge, outward normal (dy, -dx) thanks to shapely's orientation
        for ring in rings:
            p0 = ring
            p1 = np.roll(ring, -1, axis=0)
            d = p1 - p0
            length = np.linalg.norm(d, axis=1)
            ok = length > 1e-9
            p0, p1, d, length = p0[ok], p1[ok], d[ok], length[ok]
            if len(p0) == 0:
                continue
            n_img = np.stack([d[:, 1], -d[:, 0]], axis=1) / length[:, None]
            n3 = np.stack([n_img[:, 0], -n_img[:, 1], np.zeros(len(n_img))], axis=1)

            v0 = to3d(p0, +hz)
            v1 = to3d(p1, +hz)
            v2 = to3d(p1, -hz)
            v3 = to3d(p0, -hz)
            quad_v = np.stack([v0, v1, v2, v3], axis=1).reshape(-1, 3)
            quad_n = np.repeat(n3, 4, axis=0)
            idx = np.arange(len(p0), dtype=np.uint32) * 4
            faces = np.concatenate(
                [np.stack([idx, idx + 1, idx + 2], axis=1), np.stack([idx, idx + 2, idx + 3], axis=1)]
            )
            targets = np.concatenate([n3, n3])
            push(quad_v, quad_n, _flip_to_match(quad_v, faces, targets))

    if not F:
        return Mesh(np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint32), depth * scale)

    return Mesh(
        vertices=np.concatenate(V).astype(np.float32),
        normals=np.concatenate(N).astype(np.float32),
        faces=np.concatenate(F).astype(np.uint32),
        depth=depth * scale,
    )
