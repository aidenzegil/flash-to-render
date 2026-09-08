"""Shaded work: grey line art, halftone dot fields, tone stamps and relief layers."""

from __future__ import annotations

import json

import numpy as np
import pytest
import trimesh

from flash_to_render.export import cutout_rgba_tone, write_glb
from flash_to_render.fidelity import count_holes
from flash_to_render.mesh import extrude
from flash_to_render.pipeline import PipelineOptions, process_pieces
from flash_to_render.segment import Piece, Region, crop_pieces, detect_boxes, load_gray
from flash_to_render.trace import TraceOptions, binarize, prepare, prepare_ink, route, trace_piece, _region_at, _upscale

from conftest import FIXTURES

OLD = {"fan_cat": (210, 17), "flaming_cat": (117, 4), "flower": (102, 3), "zenitsu": (59, 21)}
"""(outers, holes) these pieces traced to before this change (the blur path, user render 2c27c44989)."""


ANIME = {"fan_cat": (365, 473), "flaming_cat": (709, 707), "flower": (41, 932), "zenitsu": (1361, 56), "tanjiro": (499, 191), "kaigaku": (1127, 26), "banners": (824, 71)}
"""Top-left corners of the pieces the tests look at (the shipped regions file may be re-ordered by the user)."""


def by_corner(pieces, xy, tol=25):
    x, y = xy
    return min(pieces, key=lambda p: abs(p.x - x) + abs(p.y - y)) if min(abs(p.x - x) + abs(p.y - y) for p in pieces) <= tol else None


@pytest.fixture(scope="module")
def anime():
    regions = [Region.from_dict(r) for r in json.loads((FIXTURES / "payday-anime.regions.json").read_text())["regions"]]
    gray = load_gray(FIXTURES / "anime.webp")
    pieces = crop_pieces(gray, regions)
    found = {k: by_corner(pieces, xy) for k, xy in ANIME.items()}
    assert all(found.values()), found
    return found


def stipple(size: int = 200, dot: int = 3, pitch: int = 4, solid: bool = True) -> Piece:
    """A synthetic halftone piece: a 56% dot field (3x3 dots, pitch 4) next to a solid block and a light 25% tint."""
    img = np.full((size, size), 255, dtype=np.uint8)
    for y in range(20, size // 2, pitch):
        for x in range(20, size - 20, pitch):
            img[y : y + dot, x : x + dot] = 0
    if solid:
        img[size // 2 + 10 : size - 40, 20 : size // 2] = 0
    for y in range(size // 2 + 10, size - 40, pitch):  # 25% tint: 2x2 dots
        for x in range(size // 2 + 10, size - 20, pitch):
            img[y : y + 2, x : x + 2] = 0
    import cv2

    img = cv2.GaussianBlur(img, (0, 0), 0.7)  # a scan has antialiased dot edges
    ink = (img < 150).astype(np.uint8) * 255
    from flash_to_render.segment import grey_ratio, speckle_score

    n, spk = speckle_score(ink)
    region = np.ones_like(ink, dtype=bool)
    return Piece(0, 0, 0, size, size, img, ink, region, size, size, int((ink > 0).sum()), n, spk, grey_ratio(img, region))


def test_grey_line_art_is_routed_to_light_and_traces_far_fewer_pieces(anime):
    """Anime 14 (fan and cat), 21 (flaming cat), 27 (flower cluster), 07 (Zenitsu): grey outlines, not dots."""
    limits = {"fan_cat": 60, "flaming_cat": 45, "flower": 20, "zenitsu": 35}
    for idx, limit in limits.items():
        piece = anime[idx]
        assert piece.speckle > 3 and piece.grey_ratio > 0.6  # the old gate called these "halftone"
        assert route(piece, TraceOptions()) == "light"
        new = trace_piece(piece, TraceOptions())
        old_outers, old_holes = OLD[idx]
        assert new.tone_mode == "light" and not new.layers
        assert new.outer_count <= limit, (idx, new.outer_count)
        assert new.outer_count * 2 <= old_outers, (idx, new.outer_count, old_outers)
        assert new.fidelity.iou >= 0.78, (idx, new.fidelity.iou)  # against the light-line reference mask
        assert new.hole_count > old_holes


def test_line_art_masks_are_byte_for_byte_unchanged_by_the_tone_gate(anime):
    """Crisp pieces on the same sheet (and Leap of Faith) never see the light/tone paths."""
    regions = [Region.from_dict(r) for r in json.loads((FIXTURES / "payday-spider-verse.regions.json").read_text())["regions"]]
    leap = crop_pieces(load_gray(FIXTURES / "spiderverse.webp"), [regions[5]])[0]
    for piece in (anime["tanjiro"], anime["kaigaku"], anime["banners"], leap):
        opts = TraceOptions()
        assert route(piece, opts) == "line"
        auto, r, _ = prepare(piece, opts)
        plain = binarize(_upscale(piece.gray, 3), opts, 3)
        plain[~_region_at(piece, 3)] = 0
        assert r == "line" and np.array_equal(auto, plain)
        assert np.array_equal(prepare_ink(piece, TraceOptions(smooth="off"))[0], auto)


def test_dot_field_is_resolved_into_solid_regions_with_relief_layers(tmp_path):
    piece = stipple()
    opts = TraceOptions()
    assert route(piece, opts) == "tone"
    mask, r, spacing = prepare(piece, opts)
    assert r == "tone" and 3.0 <= spacing <= 5.0  # dots every 4 px
    res = trace_piece(piece, opts)
    assert res.tone_mode == "tone" and res.dot_spacing == spacing
    # the 56% field and the solid block become two solid shapes; the 25% tint is paper at 50% coverage
    assert res.outer_count <= 3, res.outer_count
    assert res.fidelity.iou >= 0.85
    line = trace_piece(piece, TraceOptions(smooth="off"))
    assert line.outer_count > 200  # what a plain trace of the stipple would produce
    k = res.scale
    assert mask[30 * k, 100 * k] > 0 and mask[110 * k, 60 * k] > 0 and mask[130 * k, 150 * k] == 0
    # relief: dark = solid only, mid = solid + 56% field
    names = [l.name for l in res.layers]
    assert names == ["mid", "dark"]
    mid, dark = res.layers
    assert 0 < mid.depth < 1 and dark.depth == 1.0
    assert sum(p.area for p in dark.polygons) < 0.6 * sum(p.area for p in mid.polygons)
    # both layers extrude to closed shells and share a GLB
    results = process_pieces([piece], PipelineOptions())
    (r0,) = results
    assert [n for n, _m in r0.layer_meshes] == ["mid", "dark"]
    for _n, m in r0.layer_meshes:
        assert m.edge_report()[1] == 0
    depths = {n: m.depth for n, m in r0.layer_meshes}
    assert depths["mid"] < depths["dark"]
    glb = write_glb(r0.layer_meshes, tmp_path / "s.glb", name="s")
    scene = trimesh.load(glb, force="scene")
    assert len(scene.geometry) == 2
    for g in scene.geometry.values():
        g = g.copy()
        g.merge_vertices()
        assert g.is_watertight
    # relief off: a single full-depth layer
    off = process_pieces([piece], PipelineOptions(trace=TraceOptions(relief=False)))[0]
    assert not off.layer_meshes and off.mesh.edge_report()[1] == 0


def test_tone_stamp_keeps_grey_and_stays_binary_for_line_art(anime):
    def mid_alpha_fraction(piece):
        alpha = cutout_rgba_tone(piece, 2, paper_level=250.0, ink_level=25.0)[..., 3] / 255.0
        inked = alpha > 0.1
        return ((alpha > 0.2) & (alpha < 0.8) & inked).sum() / max(1, inked.sum())

    shaded = mid_alpha_fraction(anime["fan_cat"])
    assert shaded >= 0.05, shaded
    crisp = mid_alpha_fraction(anime["tanjiro"])  # solid black brushwork
    assert crisp < shaded / 2, (crisp, shaded)
    # binary variant still available and really binary
    from flash_to_render.export import cutout_rgba_from_mask

    objects = load_gray(FIXTURES / "objects.webp")
    cassette = crop_pieces(objects, detect_boxes(objects))[6]
    mask, _r, _ = prepare(cassette, TraceOptions())
    alpha = cutout_rgba_from_mask(mask, 3, 2)[..., 3]
    assert ((alpha > 40) & (alpha < 215)).mean() < 0.06  # resampling antialiasing only
