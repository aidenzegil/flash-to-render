from __future__ import annotations

import numpy as np

from flash_to_render.segment import (
    Region,
    SegmentOptions,
    crop_pieces,
    absorb_small,
    ink_mask,
    looks_like_single_design,
    reading_order,
    segment_sheet,
    speckle_score,
    whole_image_piece,
)
from flash_to_render.segment import _Component


def _sheet(boxes, size=(600, 800)):
    """White sheet with a filled black rectangle per (x, y, w, h)."""
    img = np.full(size, 255, dtype=np.uint8)
    for x, y, w, h in boxes:
        img[y : y + h, x : x + w] = 0
    return img


def test_segments_separate_designs_in_reading_order():
    boxes = [(400, 60, 120, 100), (60, 50, 100, 100), (60, 300, 150, 120), (450, 320, 100, 150)]
    pieces = segment_sheet(_sheet(boxes))
    assert len(pieces) == 4
    origins = [(p.x, p.y) for p in pieces]
    # top-left, top-right, bottom-left, bottom-right (allowing for padding)
    assert origins[0][0] < origins[1][0] and origins[0][1] < origins[2][1]
    assert origins[2][0] < origins[3][0]
    assert [p.id for p in pieces] == ["00", "01", "02", "03"]


def test_nearby_strokes_merge_into_one_piece():
    # two bars 6 px apart: one design with the default kernel, two with a tiny kernel and absorb off
    boxes = [(100, 100, 200, 20), (100, 126, 200, 20)]
    assert len(segment_sheet(_sheet(boxes))) == 1
    assert len(segment_sheet(_sheet(boxes), SegmentOptions(merge_kernel=3, absorb_max=0))) == 2


def test_noise_is_dropped():
    sheet = _sheet([(100, 100, 200, 200)])
    sheet[400, 400] = 0  # a single dark pixel
    pieces = segment_sheet(sheet)
    assert len(pieces) == 1


def test_crop_masks_out_neighbours_bleeding_into_bbox():
    # an L shape and a small square tucked into its corner: the square's bbox is inside the L's bbox
    sheet = _sheet([(100, 100, 300, 30), (100, 100, 30, 300), (300, 300, 40, 40)])
    pieces = segment_sheet(sheet, SegmentOptions(absorb_max=0))
    assert len(pieces) == 2
    big = max(pieces, key=lambda p: p.ink_area)
    small = min(pieces, key=lambda p: p.ink_area)
    assert small.ink_area == 40 * 40
    # the L's crop contains the square's bbox but none of its ink
    assert big.ink[300 - big.y + 5, 300 - big.x + 5] == 0


def test_absorb_small_keeps_caption_words_together():
    words = [_Component(100, 100, 40, 20, {1}, 500), _Component(150, 100, 60, 20, {2}, 700), _Component(220, 100, 50, 20, {3}, 600)]
    merged = absorb_small(words, max_side=80, gap=15)
    assert len(merged) == 1
    assert merged[0].labels == {1, 2, 3}
    assert (merged[0].x, merged[0].w) == (100, 170)
    # too far apart: nothing happens
    assert len(absorb_small(words, max_side=80, gap=5)) == 3


def test_reading_order_groups_rows_by_overlap():
    comps = [
        _Component(300, 10, 50, 50, {1}, 1),
        _Component(10, 30, 50, 50, {2}, 1),  # same row as the first (overlaps 30/50)
        _Component(10, 200, 50, 50, {3}, 1),
    ]
    assert [c.labels for c in reading_order(comps)] == [{2}, {1}, {3}]


def test_ink_mask_blur_and_close_reduce_speckle():
    # a 2x2 dot every 4 px: a 25% halftone tint
    halftone = np.full((200, 200), 255, dtype=np.uint8)
    for dy in (0, 1):
        for dx in (0, 1):
            halftone[10 + dy : 190 : 4, 10 + dx : 190 : 4] = 0
    n_raw, speck_raw = speckle_score(ink_mask(halftone, threshold=150))
    n_smooth, speck_smooth = speckle_score(ink_mask(halftone, threshold=210, blur=5, close=5))
    assert n_raw > 1000 and n_smooth == 1
    assert speck_smooth < speck_raw / 100


def test_single_image_detection_and_whole_image_piece():
    single = _sheet([(150, 100, 400, 300)])
    single[500, 700] = 0  # a stray speck must not change the verdict
    pieces = segment_sheet(single)
    assert looks_like_single_design(pieces)
    single[500, 700] = 255
    whole = whole_image_piece(single, pad=4)
    assert (whole.x, whole.y) == (146, 96)
    assert (whole.w, whole.h) == (408, 308)
    assert whole.ink_area == 400 * 300

    multi = _sheet([(50, 50, 100, 100), (400, 50, 100, 100), (50, 400, 100, 100)])
    assert not looks_like_single_design(segment_sheet(multi))


def test_polygon_region_excludes_neighbour_inside_its_bbox():
    """Two squares on a diagonal: a triangle around A has a bbox that also covers B."""
    sheet = _sheet([(100, 100, 100, 100), (210, 210, 100, 100)], size=(400, 400))
    triangle = Region([(90, 90), (320, 90), (90, 320)], "polygon", "A only")
    rect = Region.rect(90, 90, 230, 230, "both")
    (poly_piece,) = crop_pieces(sheet, [triangle])
    (rect_piece,) = crop_pieces(sheet, [rect])
    assert poly_piece.bbox == rect_piece.bbox == (90, 90, 230, 230)
    assert rect_piece.ink_area == 2 * 100 * 100
    assert poly_piece.ink_area == 100 * 100
    assert poly_piece.kind == "polygon" and poly_piece.id == "00-a-only"
    # everything outside the polygon is paper in the crop
    assert poly_piece.gray[250 - 90, 250 - 90] == 255 and poly_piece.ink[250 - 90, 250 - 90] == 0
    assert poly_piece.gray[150 - 90, 150 - 90] == 0
    # polygon serialisation round-trips and legacy rect dicts still load
    assert Region.from_dict(triangle.to_dict()).points == [(90.0, 90.0), (320.0, 90.0), (90.0, 320.0)]
    legacy = Region.from_dict({"x": 1, "y": 2, "w": 3, "h": 4, "name": "n"})
    assert legacy.kind == "rect" and legacy.name == "n"
    assert legacy.bbox().to_dict() == {"x": 1, "y": 2, "w": 3, "h": 4, "name": "n"}
