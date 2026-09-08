"""Raster -> polygons with holes.

Uses ``potracer`` (a pure-Python port of potrace) to turn a binary ink mask into
Bezier outlines, samples those to polylines, simplifies them and nests the
resulting rings into outer boundaries and holes by containment.

Polarity, the bug that bit hardest
----------------------------------
``potracer.Bitmap(data)`` calls ``.invert()`` on whatever you hand it, so the
pixels that get traced as "black" are the ones that were **False / zero** in
``data``. In other words you pass a *paper* mask, not an ink mask. Pass an ink
mask and you get the design cut out of a rectangular card. :func:`trace_ink`
takes an ink mask (True = ink), does the inversion for you, and additionally
checks the result: if the biggest outline is a card the size of the crop it
flips polarity and traces again.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import cv2
import numpy as np
import potrace
from shapely import simplify as shp_simplify
from shapely.geometry import LinearRing, MultiPolygon, Polygon

from .segment import Piece, ink_mask, speckle_score

log = logging.getLogger(__name__)

__all__ = [
    "PolygonWithHoles",
    "TraceOptions",
    "TraceResult",
    "prepare_ink",
    "trace_ink",
    "nest_rings",
    "path_to_svg_d",
    "card_likeness",
    "is_cardlike",
    "trace_piece",
]


@dataclass
class PolygonWithHoles:
    """One solid region: an outer ring and zero or more hole rings, all ``(N, 2)`` float arrays in pixel space (y down)."""

    outer: np.ndarray
    holes: list[np.ndarray] = field(default_factory=list)

    def to_shapely(self) -> Polygon:
        poly = Polygon(self.outer, self.holes)
        if not poly.is_valid:
            poly = poly.buffer(0)
        return poly

    @property
    def area(self) -> float:
        return abs(self.to_shapely().area)

    @property
    def bbox_area(self) -> float:
        mn = self.outer.min(axis=0)
        mx = self.outer.max(axis=0)
        return float((mx[0] - mn[0]) * (mx[1] - mn[1]))


@dataclass
class TraceOptions:
    """Knobs for :func:`prepare_ink` and :func:`trace_ink`."""

    threshold: int = 150
    """Ink threshold used when a piece is re-thresholded for smoothing."""
    smooth: str = "auto"
    """``auto`` smooths only speckly pieces, ``on`` always, ``off`` never."""
    smooth_speckle: float = 3.0
    """In ``auto`` mode, pieces with a raw speckle score above this get the halftone treatment."""
    smooth_blur: int = 5
    """Gaussian kernel for the halftone treatment."""
    smooth_close: int = 3
    """Morphological close kernel for the halftone treatment."""
    smooth_threshold_boost: int = 15
    """Blurring lightens ink, so the threshold is raised by this much when smoothing."""
    median: int = 3
    """Median filter applied to every mask before tracing (0 = off). Kills 1-px jaggies."""
    turdsize: int = 14
    """potrace: drop ink blobs smaller than this many pixels."""
    alphamax: float = 1.0
    """potrace corner threshold (0 = all corners, 1.334 = all smooth)."""
    opttolerance: float = 0.8
    """potrace curve optimisation tolerance."""
    subdiv: int = 5
    """Line segments sampled per Bezier segment."""
    simplify_px: float = 0.9
    """Douglas-Peucker tolerance (px) applied to every polyline ring."""
    auto_polarity: bool = True
    """Re-trace with flipped polarity if the outline is a card the size of the crop."""


@dataclass
class TraceResult:
    polygons: list[PolygonWithHoles]
    svg_d: str
    ink: np.ndarray
    """The (possibly smoothed) mask that was actually traced, uint8 255 = ink."""
    smoothed: bool
    components: int
    speckle: float
    flipped_polarity: bool = False

    @property
    def outer_count(self) -> int:
        return len(self.polygons)

    @property
    def hole_count(self) -> int:
        return sum(len(p.holes) for p in self.polygons)


# --------------------------------------------------------------------------- #
# mask preparation
# --------------------------------------------------------------------------- #


def prepare_ink(piece: Piece, options: TraceOptions | None = None) -> tuple[np.ndarray, bool]:
    """Return ``(mask, smoothed)``: the uint8 ink mask to trace for ``piece``.

    Line art keeps its crisp threshold mask. Speckly (halftone / grey-shaded)
    pieces are re-thresholded from the greyscale crop with a blur and a
    morphological close so shading collapses into solid regions instead of
    thousands of dots.
    """
    opts = options or TraceOptions()
    want = opts.smooth == "on" or (opts.smooth == "auto" and piece.speckle > opts.smooth_speckle)
    if want:
        mask = ink_mask(
            piece.gray,
            threshold=min(255, opts.threshold + opts.smooth_threshold_boost),
            blur=opts.smooth_blur,
            close=opts.smooth_close,
        )
        mask[~piece.region] = 0
    else:
        mask = piece.ink.copy()
    if opts.median and opts.median > 1:
        mask = cv2.medianBlur(mask, opts.median | 1)
    return mask, want


# --------------------------------------------------------------------------- #
# potrace plumbing
# --------------------------------------------------------------------------- #


def _bezier_points(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, n: int) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n + 1)[1:]
    mt = 1.0 - t
    return (
        (mt**3)[:, None] * p0
        + (3 * mt**2 * t)[:, None] * p1
        + (3 * mt * t**2)[:, None] * p2
        + (t**3)[:, None] * p3
    )


def _curve_to_ring(curve, subdiv: int, simplify_px: float) -> np.ndarray | None:
    pts = [np.array([curve.start_point.x, curve.start_point.y], dtype=np.float64)]
    for seg in curve.segments:
        last = pts[-1]
        if seg.is_corner:
            pts.append(np.array([seg.c.x, seg.c.y], dtype=np.float64))
            pts.append(np.array([seg.end_point.x, seg.end_point.y], dtype=np.float64))
        else:
            c1 = np.array([seg.c1.x, seg.c1.y], dtype=np.float64)
            c2 = np.array([seg.c2.x, seg.c2.y], dtype=np.float64)
            end = np.array([seg.end_point.x, seg.end_point.y], dtype=np.float64)
            pts.extend(_bezier_points(last, c1, c2, end, subdiv))
    ring = np.array(pts)
    if len(ring) < 4:
        return None
    try:
        lr = LinearRing(ring)
        if simplify_px > 0:
            lr = shp_simplify(lr, simplify_px, preserve_topology=True)
        out = np.asarray(lr.coords)[:-1]
    except Exception:  # degenerate ring: keep the raw samples
        out = ring
    return out if len(out) >= 3 else None


def path_to_svg_d(path: potrace.Path) -> str:
    """Serialise a potrace path as an SVG ``d`` attribute (fill with ``evenodd``)."""
    parts: list[str] = []
    for curve in path:
        s = curve.start_point
        parts.append(f"M{s.x:.1f} {s.y:.1f}")
        for seg in curve:
            if seg.is_corner:
                c, e = seg.c, seg.end_point
                parts.append(f"L{c.x:.1f} {c.y:.1f}L{e.x:.1f} {e.y:.1f}")
            else:
                c1, c2, e = seg.c1, seg.c2, seg.end_point
                parts.append(f"C{c1.x:.1f} {c1.y:.1f} {c2.x:.1f} {c2.y:.1f} {e.x:.1f} {e.y:.1f}")
        parts.append("Z")
    return "".join(parts)


def _potrace(paper: np.ndarray, opts: TraceOptions) -> potrace.Path:
    """Run potrace on a *paper* mask (True = white). See the module docstring."""
    bitmap = potrace.Bitmap(np.ascontiguousarray(paper, dtype=bool))
    return bitmap.trace(
        turdsize=opts.turdsize,
        alphamax=opts.alphamax,
        opticurve=True,
        opttolerance=opts.opttolerance,
    )


# --------------------------------------------------------------------------- #
# nesting
# --------------------------------------------------------------------------- #


def nest_rings(rings: Iterable[np.ndarray]) -> list[PolygonWithHoles]:
    """Group closed rings into outers and holes by containment depth.

    Rings are sorted by area (largest first). Each ring's parent is the smallest
    ring that contains it. Even depth (0, 2, ...) = outer boundary, odd depth =
    hole of its parent. An island inside a hole therefore becomes a new outer,
    which is exactly what an extruder needs.
    """
    shp: list[tuple[Polygon, np.ndarray]] = []
    for r in rings:
        try:
            pg = Polygon(r)
            if not pg.is_valid:
                pg = pg.buffer(0)
            if pg.is_empty or pg.area <= 0:
                continue
            if isinstance(pg, MultiPolygon):
                pg = max(pg.geoms, key=lambda g: g.area)
            shp.append((pg, r))
        except Exception:
            continue
    shp.sort(key=lambda t: -t[0].area)

    parent = [-1] * len(shp)
    for i, (pg, _r) in enumerate(shp):
        probe = pg.representative_point()
        for j in range(i - 1, -1, -1):  # candidates are larger; first hit is the smallest container
            if shp[j][0].contains(probe):
                parent[i] = j
                break

    depth = [0] * len(shp)
    for i in range(len(shp)):
        depth[i] = 0 if parent[i] < 0 else depth[parent[i]] + 1

    polys: list[PolygonWithHoles] = []
    outer_index: dict[int, int] = {}
    for i, (_pg, r) in enumerate(shp):
        if depth[i] % 2 == 0:
            outer_index[i] = len(polys)
            polys.append(PolygonWithHoles(outer=np.asarray(r, dtype=np.float64)))
    for i, (_pg, r) in enumerate(shp):
        if depth[i] % 2 == 1:
            polys[outer_index[parent[i]]].holes.append(np.asarray(r, dtype=np.float64))
    return polys


def card_likeness(polys: Sequence[PolygonWithHoles], shape: tuple[int, int]) -> float:
    """``outer_area / crop_area`` for the largest outer ring.

    A correctly traced design is rarely above ~0.85 unless it really is a filled
    rectangle; a polarity-flipped trace produces one outer ring that *is* the crop.
    """
    if not polys:
        return 0.0
    h, w = shape[:2]
    biggest = max(polys, key=lambda p: Polygon(p.outer).area)
    return float(Polygon(biggest.outer).area / max(1.0, float(w * h)))


def is_cardlike(polys: Sequence[PolygonWithHoles], shape: tuple[int, int], area_frac: float = 0.85, span_frac: float = 0.9) -> bool:
    """True when the largest outer ring is essentially the crop rectangle itself."""
    if not polys:
        return False
    h, w = shape[:2]
    biggest = max(polys, key=lambda p: Polygon(p.outer).area)
    mn, mx = biggest.outer.min(axis=0), biggest.outer.max(axis=0)
    spans = (mx[0] - mn[0]) >= span_frac * w and (mx[1] - mn[1]) >= span_frac * h
    return spans and card_likeness([biggest], shape) >= area_frac


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #


def _trace_rings(paper: np.ndarray, opts: TraceOptions) -> tuple[potrace.Path, list[PolygonWithHoles]]:
    path = _potrace(paper, opts)
    rings = (_curve_to_ring(c, opts.subdiv, opts.simplify_px) for c in path.curves)
    return path, nest_rings(r for r in rings if r is not None)


def trace_ink(ink: np.ndarray, options: TraceOptions | None = None) -> tuple[potrace.Path, list[PolygonWithHoles], bool]:
    """Trace an **ink** mask (nonzero / True = ink) into nested polygons.

    Returns ``(potrace_path, polygons, flipped)``. ``flipped`` is True when the
    polarity guard had to invert the input: the trace came back as one ring the
    size of the crop (a "card"), which is what you get when a paper mask is
    handed to potrace as if it were ink. A design that genuinely is a filled
    rectangle is left alone because its flipped trace is empty or card-like too.
    """
    opts = options or TraceOptions()
    ink_b = np.asarray(ink) > 0
    shape = ink_b.shape

    path, polys = _trace_rings(~ink_b, opts)
    if not opts.auto_polarity or not is_cardlike(polys, shape):
        return path, polys, False

    alt_path, alt_polys = _trace_rings(ink_b, opts)
    if alt_polys and not is_cardlike(alt_polys, shape):
        log.warning("trace: outline was a card the size of the crop; input looked like a paper mask, flipped polarity")
        return alt_path, alt_polys, True
    return path, polys, False


def trace_piece(piece: Piece, options: TraceOptions | None = None) -> TraceResult:
    """Prepare and trace one :class:`Piece`."""
    opts = options or TraceOptions()
    mask, smoothed = prepare_ink(piece, opts)
    comps, speck = speckle_score(mask)
    path, polys, flipped = trace_ink(mask, opts)
    return TraceResult(
        polygons=polys,
        svg_d=path_to_svg_d(path),
        ink=mask,
        smoothed=smoothed,
        components=comps,
        speckle=speck,
        flipped_polarity=flipped,
    )
