"""Fine-line fidelity on real pieces: "A Leap of Faith" (falling figure + caption), the web swing, the cassette."""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from flash_to_render.fidelity import count_holes, measure, rasterize
from flash_to_render.segment import Region, crop_pieces, detect_boxes, load_gray, speckle_score
from flash_to_render.trace import BEST_GRID, PolygonWithHoles, TraceOptions, estimate_triangles, prepare_ink, trace_piece

from conftest import FIXTURES

OLD = TraceOptions(upscale=1, binarize="global", turdsize=14, simplify_px=0.9, opttolerance=0.8, median=3, min_hole_area=0)
"""The pre-fidelity settings, kept here as the baseline the new path must beat."""


@pytest.fixture(scope="module")
def spider_pieces():
    regions = [Region.from_dict(r) for r in json.loads((FIXTURES / "payday-spider-verse.regions.json").read_text())["regions"]]
    gray = load_gray(FIXTURES / "spiderverse.webp")
    leap, swing = crop_pieces(gray, [regions[5], regions[3]])
    assert (leap.x, leap.y, leap.w, leap.h) == (58, 411, 151, 311)
    return {"leap": leap, "swing": swing}


@pytest.fixture(scope="module")
def cassette():
    gray = load_gray(FIXTURES / "objects.webp")
    piece = crop_pieces(gray, detect_boxes(gray))[6]
    assert (piece.w, piece.h) == (270, 298)
    return piece


def test_leap_of_faith_best_mode_keeps_lines_holes_and_caption(spider_pieces):
    leap = spider_pieces["leap"]
    old = trace_piece(leap, OLD)
    best = trace_piece(leap, TraceOptions(fidelity="best"))
    fid = best.fidelity
    assert fid.candidates == len(BEST_GRID) and fid.params["upscale"] in (3, 4)
    assert fid.iou >= 0.9
    assert fid.iou > old.fidelity.iou and fid.thin_recall > old.fidelity.thin_recall
    # interior holes: the cleaned source mask has ~45 enclosed paper regions at working resolution
    # (suit seams, gaps between limbs, letter counters); the old trace kept 12, the new one must keep 30+
    assert fid.holes_mask >= 40
    assert best.hole_count >= 30 and best.hole_count > 2 * old.hole_count
    assert fid.hole_ratio >= 0.8
    assert estimate_triangles(best.polygons) <= 20000

    # the caption "A LEAP OF FAITH" sits in the bottom ~10% of the crop: it must survive as separate letters
    k = best.scale
    caption = best.ink[int(0.88 * best.ink.shape[0]) :, :]
    n_letters, _ = speckle_score(caption)
    assert n_letters >= 10, n_letters
    raster = rasterize(best.polygons, best.ink.shape, k)[int(0.88 * best.ink.shape[0]) :, :]
    n_traced, _ = speckle_score(raster)
    assert n_traced >= 10, n_traced
    old_caption = rasterize(old.polygons, old.ink.shape, 1)[int(0.88 * old.ink.shape[0]) :, :]
    assert speckle_score(old_caption)[0] < n_traced  # the old trace collapsed the caption


def test_web_swing_and_cassette_improve_over_the_old_settings(spider_pieces, cassette):
    for piece in (spider_pieces["swing"], cassette):
        old = trace_piece(piece, OLD)
        new = trace_piece(piece, TraceOptions())  # fast mode is the default
        assert new.fidelity.iou >= 0.9 and new.fidelity.iou > old.fidelity.iou
        assert new.fidelity.thin_recall >= 0.65
        assert new.hole_count > old.hole_count
        # judged against the same hi-res reference mask, the new trace covers more of the thin ink
        ref = measure(new.ink, old.polygons, new.scale)
        assert new.fidelity.thin_recall > ref.thin_recall and new.fidelity.iou > ref.iou


def test_fidelity_score_is_exact_for_a_perfect_trace_and_penalises_a_bad_one():
    mask = np.zeros((120, 120), dtype=np.uint8)
    cv2.rectangle(mask, (10, 10), (109, 109), 255, -1)
    cv2.rectangle(mask, (40, 40), (79, 79), 0, -1)
    cv2.rectangle(mask, (55, 55), (64, 64), 255, -1)  # island inside the hole
    assert count_holes(mask) == 1
    square = lambda x0, y0, x1, y1: np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float)  # noqa: E731
    exact = [PolygonWithHoles(square(10, 10, 110, 110), [square(40, 40, 80, 80)]), PolygonWithHoles(square(55, 55, 65, 65))]
    f = measure(mask, exact, 1)
    assert f.iou > 0.95 and f.hole_ratio == 1.0 and f.holes_trace == 1 and f.score > 0.97  # fillPoly includes both edges
    # the raster keeps the island (largest-first painting)
    assert rasterize(exact, mask.shape, 1)[60, 60] == 255
    blob = [PolygonWithHoles(square(10, 10, 110, 110))]  # hole filled in
    g = measure(mask, blob, 1)
    assert g.hole_ratio == 0.0 and g.holes_trace == 0 and g.score < f.score


def test_prepare_ink_scales_with_upscale_and_never_blurs_line_art(spider_pieces):
    leap = spider_pieces["leap"]
    m1, s1 = prepare_ink(leap, TraceOptions(upscale=1))
    m3, s3 = prepare_ink(leap, TraceOptions(upscale=3))
    assert not s1 and not s3
    assert m3.shape == (leap.h * 3, leap.w * 3) and m1.shape == (leap.h, leap.w)
    # the adaptive path keeps thin interior lines that the plain threshold merges away: more enclosed paper
    plain, _ = prepare_ink(leap, TraceOptions(upscale=3, binarize="global", unsharp=0))
    assert count_holes(m3) > count_holes(plain)
