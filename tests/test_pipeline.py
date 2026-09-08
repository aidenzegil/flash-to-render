"""End-to-end checks on the three real flash sheets in ``examples/``."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import trimesh
from PIL import Image
from shapely.geometry import Polygon

from flash_to_render.cli import main
from flash_to_render.pipeline import PipelineOptions, run
from flash_to_render.fidelity import rasterize
from flash_to_render.trace import card_likeness, is_cardlike

from conftest import FIXTURES, SHEETS

# piece counts with the default options, +/- tolerance (segmentation is heuristic)
EXPECTED = {"anime": (37, 5), "objects": (16, 2), "spiderverse": (20, 4)}


@pytest.mark.parametrize("name", SHEETS)
def test_piece_counts(sheet_results, name):
    expected, tol = EXPECTED[name]
    res = sheet_results[name]
    assert res.mode == "sheet"
    assert abs(len(res.pieces) - expected) <= tol, [p.piece.bbox for p in res.pieces]


@pytest.mark.parametrize("name", SHEETS)
def test_most_pieces_are_ok(sheet_results, name):
    res = sheet_results[name]
    ok = sum(1 for p in res.pieces if p.quality == "ok")
    assert ok / len(res.pieces) >= 0.6
    assert not any(p.quality == "empty" for p in res.pieces)
    assert not any(p.trace.flipped_polarity for p in res.pieces if p.trace)


@pytest.mark.parametrize("name", SHEETS)
def test_no_card_like_outer_rings(sheet_results, name):
    """The polarity bug: the design traced as a hole in a rectangular card."""
    for r in sheet_results[name].pieces:
        assert r.trace is not None
        k = r.trace.scale  # the traced mask is at k x source resolution, the polygons are in source px
        shape = (r.trace.ink.shape[0] / k, r.trace.ink.shape[1] / k)
        for poly in r.trace.polygons:
            outer = Polygon(poly.outer)
            ratio = outer.area / max(1.0, poly.bbox_area)
            if ratio >= 0.85:
                # only allowed when the mask really is ink where the ring is solid: a card has a sparse
                # drawing inside it, a rectangular design element (or frame with holes) is dense ink
                solid = rasterize([poly], r.trace.ink.shape, k) > 0
                density = (r.trace.ink[solid] > 0).mean() if solid.any() else 0.0
                assert density > 0.5, (name, r.piece.id, ratio, density)
                assert not is_cardlike([poly], shape), (name, r.piece.id)
        assert card_likeness(r.trace.polygons, shape) < 0.85, (name, r.piece.id)


@pytest.mark.parametrize("name", SHEETS)
def test_meshes_are_closed(sheet_results, name):
    for r in sheet_results[name].pieces:
        assert r.mesh is not None and not r.mesh.is_empty
        total, bad = r.mesh.edge_report()
        assert bad == 0, (name, r.piece.id, bad, total)


@pytest.mark.parametrize("name", SHEETS)
def test_outputs_exist_and_glbs_load(sheet_results, name):
    res = sheet_results[name]
    out = res.out_dir
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["mode"] == "sheet"
    assert manifest["sheet"] == {"width": res.sheet_size[0], "height": res.sheet_size[1]}
    assert (out / "segmentation.png").exists()
    assert len(manifest["pieces"]) == len(res.pieces)

    for entry in manifest["pieces"]:
        assert {"id", "bbox", "speckle", "triangles", "quality", "files"} <= entry.keys()
        assert entry["quality"] in ("ok", "tiny", "speckle", "empty")
        png = Image.open(out / entry["files"]["png"])
        k = entry["png_scale"]
        assert k == 2 and png.mode == "RGBA" and png.size == (entry["bbox"][2] * k, entry["bbox"][3] * k)
        rgb = np.asarray(png)[..., :3]
        assert rgb.min() == 255  # white, tintable
        assert (out / entry["files"]["svg"]).read_text().startswith("<svg")
        scene = trimesh.load(out / entry["files"]["glb"], force="scene")
        (geom,) = scene.geometry.values()
        assert len(geom.faces) == entry["triangles"]
        merged = geom.copy()
        merged.merge_vertices()
        assert merged.is_watertight, (name, entry["id"])

    scene = trimesh.load(out / manifest["scene"], force="scene")
    assert len(scene.graph.nodes_geometry) == sum(1 for e in manifest["pieces"] if "glb" in e["files"])
    # pieces are laid out on the sheet: the scene is about the sheet's size in metres (1 px = 1 mm)
    extent = scene.bounds[1] - scene.bounds[0]
    assert 0.5 * res.sheet_size[0] / 1000 < extent[0] <= res.sheet_size[0] / 1000
    assert 0.5 * res.sheet_size[1] / 1000 < extent[1] <= res.sheet_size[1] / 1000


def test_depth_is_a_cast_object_not_a_slab(sheet_results):
    for res in sheet_results.values():
        for r in res.pieces:
            frac = r.depth_px / r.piece.longest_side
            fine = r.trace.fidelity.thin_fraction > 0.3
            assert (0.02 <= frac <= 0.03) if fine else (0.03 <= frac <= 0.04)


def test_single_design_mode_auto_and_forced(tmp_path):
    """Crop one design out of a sheet and feed it back in: auto mode must not segment it."""
    res = run(FIXTURES / "objects.webp", tmp_path / "sheet", PipelineOptions(write_glb=False, write_svg=False), report=None)
    big = max(res.pieces, key=lambda r: r.piece.ink_area)
    # rebuild a standalone image with generous white padding
    crop = big.piece.gray
    canvas = np.full((crop.shape[0] + 80, crop.shape[1] + 80), 255, dtype=np.uint8)
    canvas[40:-40, 40:-40] = crop
    src = tmp_path / "one.png"
    Image.fromarray(canvas).save(src)

    auto = run(src, tmp_path / "auto", report=None)
    assert auto.mode == "single" and len(auto.pieces) == 1
    assert auto.pieces[0].mesh.edge_report()[1] == 0

    forced = run(src, tmp_path / "forced", PipelineOptions(mode="sheet", write_glb=False), report=None)
    assert forced.mode == "sheet"


def test_cli_convert_and_preview_arguments(tmp_path, capsys):
    out = tmp_path / "cli"
    assert main([str(FIXTURES / "objects.webp"), "-o", str(out), "--quiet", "--no-glb", "--no-svg"]) == 0
    assert (out / "manifest.json").exists() and not list(out.glob("*.glb"))
    assert main(["preview", str(tmp_path / "nowhere")]) == 2
    assert main([str(tmp_path / "missing.png")]) == 2


@pytest.mark.parametrize("name", ["anime", "spiderverse", "objects"])
def test_user_regions_render_nothing_empty_unless_truly_blank(name):
    """The hand-edited seed regions: every region with ink inside its outline yields a non-empty piece."""
    from flash_to_render.segment import Region, crop_pieces, ink_mask, load_gray

    entry = {"anime": "payday-anime", "spiderverse": "payday-spider-verse", "objects": "payday-objects"}[name]
    path = FIXTURES / f"{entry}.regions.json"
    if not path.exists():
        pytest.skip("no shipped regions for this sheet")
    regions = [Region.from_dict(r) for r in json.loads(path.read_text())["regions"]]
    gray = load_gray(FIXTURES / f"{name}.webp")
    ink = ink_mask(gray)
    pieces = crop_pieces(gray, regions)
    for reg, piece in zip(regions, pieces):
        box = reg.bbox().clamp(gray.shape[1], gray.shape[0])
        inside = int(ink[box.y : box.y + box.h, box.x : box.x + box.w][reg.mask(box)].sum() // 255)
        if inside > 50:
            assert piece.ink_area > 0, (name, piece.id, inside)
        # the two touching Hashira figures on the anime sheet both get their own side of the border
    if name == "anime":
        assert pieces[37].ink_area > 3000 and pieces[26].ink_area > 3000
        assert pieces[37].ink_area + pieces[26].ink_area >= 0.95 * (10927 + 5000)
