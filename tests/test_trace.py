from __future__ import annotations

import numpy as np
from shapely.geometry import Polygon

from flash_to_render.trace import TraceOptions, _trace_rings, card_likeness, is_cardlike, nest_rings, trace_ink

from conftest import disk_mask, donut_mask, island_in_hole_mask


def test_disk_traces_to_one_outer_no_holes():
    _path, polys, flipped = trace_ink(disk_mask())
    assert not flipped
    assert len(polys) == 1
    assert polys[0].holes == []
    # a disk covers pi/4 of its bbox: well under the "card" line, and close to the true area
    assert abs(polys[0].area - np.pi * 80**2) / (np.pi * 80**2) < 0.05


def test_donut_has_one_hole():
    _path, polys, _ = trace_ink(donut_mask())
    assert len(polys) == 1
    assert len(polys[0].holes) == 1


def test_island_inside_hole_becomes_new_outer():
    _path, polys, _ = trace_ink(island_in_hole_mask())
    assert len(polys) == 2
    by_area = sorted(polys, key=lambda p: -Polygon(p.outer).area)
    assert len(by_area[0].holes) == 1  # the ring
    assert by_area[1].holes == []  # the island
    assert Polygon(by_area[0].holes[0]).contains(Polygon(by_area[1].outer))


def test_nest_rings_by_depth():
    def square(c, r):
        return np.array([[c - r, c - r], [c + r, c - r], [c + r, c + r], [c - r, c + r]], dtype=float)

    rings = [square(50, 10), square(50, 40), square(50, 25), square(50, 5)]  # shuffled sizes
    polys = nest_rings(rings)
    assert len(polys) == 2
    assert sorted(len(p.holes) for p in polys) == [1, 1]


def test_wrong_polarity_produces_a_card_and_guard_flips_it():
    ink = donut_mask()
    # What happens if you hand potrace the ink mask directly (its `.invert()` makes it trace the paper).
    _path, wrong, = _trace_rings(ink, TraceOptions())[:2]
    assert is_cardlike(wrong, ink.shape)
    assert card_likeness(wrong, ink.shape) > 0.85

    # trace_ink expects an ink mask; give it the *paper* mask by mistake and the guard should recover.
    _path, polys, flipped = trace_ink(~ink)
    assert flipped
    assert not is_cardlike(polys, ink.shape)
    assert len(polys) == 1 and len(polys[0].holes) == 1


def test_filled_rectangle_is_left_alone():
    ink = np.zeros((120, 200), dtype=bool)
    ink[10:110, 10:190] = True
    _path, polys, flipped = trace_ink(ink)
    assert not flipped
    assert len(polys) == 1
    assert abs(polys[0].area - 100 * 180) / (100 * 180) < 0.03


def test_empty_mask_traces_to_nothing():
    _path, polys, flipped = trace_ink(np.zeros((50, 50), dtype=bool))
    assert polys == [] and not flipped
