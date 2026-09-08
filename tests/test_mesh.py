from __future__ import annotations

import numpy as np
from shapely.geometry import Point, Polygon
from shapely.prepared import prep

from flash_to_render.mesh import depth_for, extrude
from flash_to_render.trace import PolygonWithHoles, trace_ink

from conftest import donut_mask, island_in_hole_mask


def _square(cx, cy, r):
    return np.array([[cx - r, cy - r], [cx + r, cy - r], [cx + r, cy + r], [cx - r, cy + r]], dtype=float)


def test_extruded_square_is_closed_and_has_expected_counts():
    mesh = extrude([PolygonWithHoles(outer=_square(50, 50, 20))], depth=4.0)
    # 2 caps x 2 tris + 4 walls x 2 tris
    assert mesh.triangle_count == 12
    total, bad = mesh.edge_report()
    assert bad == 0 and total == 18  # 12 tris * 3 / 2

    lo, hi = mesh.bounds()
    assert np.allclose(lo[2], -2.0) and np.allclose(hi[2], 2.0)


def test_y_is_flipped_once_and_centred():
    # square centred on (60, 20) in image space, crop centre (50, 50): it should end up right (+x) and *up* (+y)
    mesh = extrude([PolygonWithHoles(outer=_square(60, 20, 5))], depth=1.0, center=(50, 50))
    c = mesh.vertices.mean(axis=0)
    assert c[0] > 0 and c[1] > 0


def test_scale_is_applied_before_storing():
    mesh = extrude([PolygonWithHoles(outer=_square(50, 50, 20))], depth=4.0, scale=0.001)
    lo, hi = mesh.bounds()
    assert np.allclose(hi[0] - lo[0], 0.04)
    assert np.allclose(mesh.depth, 0.004)


def test_cap_winding_matches_normals():
    mesh = extrude([PolygonWithHoles(outer=_square(50, 50, 20), holes=[_square(50, 50, 8)])], depth=3.0)
    V, F = mesh.vertices, mesh.faces
    geo = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    stored = mesh.normals[F[:, 0]]
    assert np.all(np.einsum("ij,ij->i", geo, stored) > 0)


def test_wall_normals_point_out_of_the_material():
    """Offset every wall's midpoint along its normal (back in image space) and check it leaves the solid."""
    for mask in (donut_mask(), island_in_hole_mask()):
        _p, polys, _f = trace_ink(mask)
        depth = depth_for(mask.shape[0])
        cx = cy = mask.shape[0] / 2
        mesh = extrude(polys, depth, center=(cx, cy))
        solid = prep(Polygon(polys[0].outer, polys[0].holes).buffer(0).union(
            Polygon(polys[1].outer).buffer(0) if len(polys) > 1 else Polygon()
        ))
        walls = np.abs(mesh.normals[:, 2]) < 0.5
        V = mesh.vertices[walls]
        N = mesh.normals[walls]
        # wall quads are 4 consecutive vertices; use the first of each
        for v, n in zip(V[::4], N[::4]):
            x_img, y_img = v[0] + cx, -v[1] + cy
            probe = Point(x_img + n[0] * 1.5, y_img - n[1] * 1.5)
            assert not solid.contains(probe), (v, n)


def test_traced_shapes_extrude_watertight():
    for mask in (donut_mask(), island_in_hole_mask()):
        _p, polys, _f = trace_ink(mask)
        mesh = extrude(polys, depth=6.0)
        total, bad = mesh.edge_report()
        assert total > 0 and bad == 0


def test_empty_input_gives_empty_mesh():
    mesh = extrude([], depth=1.0)
    assert mesh.is_empty and mesh.edge_report() == (0, 0)


def test_depth_fraction_reads_as_cast_object():
    assert 3.0 <= depth_for(100) <= 4.0
