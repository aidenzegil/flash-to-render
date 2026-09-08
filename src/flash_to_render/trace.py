"""Raster -> polygons with holes, with fine-line fidelity.

Uses ``potracer`` (a pure-Python port of potrace) to turn a binary ink mask into
Bezier outlines, samples those to polylines, simplifies them and nests the
resulting rings into outer boundaries and holes by containment.

Fine lines
----------
Tracing at source resolution loses 1-2 px seams, hairlines and small counters.
:func:`prepare_ink` therefore works on an **upscaled** crop (3-4x, bicubic),
sharpens it and binarises it edge-aware (a global threshold for solid ink OR-ed
with a Gaussian adaptive threshold that keeps thin strokes), and every
pixel-based parameter (turdsize, simplification tolerance, close kernel, hole
area) is scaled with it. Geometry comes back in source pixels. Genuinely
halftoned pieces (speckle-gated) keep the blur + close treatment instead; line
art is never blurred.

With ``fidelity="best"`` a small parameter grid is traced per piece and the
candidate with the best :mod:`fidelity` score under the triangle budget wins.

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

import itertools
import logging
from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
import potrace
from shapely import simplify as shp_simplify
from shapely.geometry import LinearRing, MultiPolygon, Polygon

from .fidelity import Fidelity, measure
from .segment import Piece, ink_mask, speckle_score

log = logging.getLogger(__name__)

__all__ = [
    "PolygonWithHoles",
    "TraceOptions",
    "TraceResult",
    "prepare_ink",
    "trace_ink",
    "trace_piece",
    "nest_rings",
    "path_to_svg_d",
    "card_likeness",
    "is_cardlike",
    "estimate_triangles",
    "FAST_GRID",
    "BEST_GRID",
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
    """Knobs for :func:`prepare_ink` and :func:`trace_ink`. Pixel units are *source* pixels unless noted."""

    fidelity: str = "fast"
    """``fast`` traces once with :data:`FAST_GRID`, ``best`` searches :data:`BEST_GRID` and keeps the best score."""
    upscale: int = 3
    """Working resolution multiplier for the crop before binarisation and tracing (1 = off)."""
    threshold: int = 150
    """Global ink threshold (0-255): darker than this is always ink."""
    binarize: str = "adaptive"
    """``adaptive`` adds a Gaussian adaptive threshold so thin strokes survive; ``global`` is the plain threshold."""
    adaptive_block: int = 25
    """Neighbourhood (source px, scaled with ``upscale``) for the adaptive threshold."""
    adaptive_c: float = 10.0
    """A pixel is ink when darker than its local Gaussian mean by this much."""
    adaptive_max: int = 215
    """Grey level above which the adaptive threshold may not mark ink (keeps paper texture out)."""
    unsharp: float = 1.0
    """Unsharp-mask amount applied before binarisation (0 = off)."""
    smooth: str = "auto"
    """``auto`` smooths only speckly pieces, ``on`` always, ``off`` never."""
    smooth_speckle: float = 3.0
    """In ``auto`` mode, pieces need a raw speckle score above this *and* ..."""
    smooth_grey: float = 0.6
    """... a mid-grey ratio above this to get the halftone treatment. Captions and hatching are speckly but
    black; shading is grey. See :attr:`Piece.grey_ratio`."""
    smooth_blur: int = 5
    """Gaussian kernel (source px) for the halftone treatment."""
    smooth_close: int = 3
    """Morphological close kernel (source px) for the halftone treatment."""
    smooth_threshold_boost: int = 15
    """Blurring lightens ink, so the threshold is raised by this much when smoothing."""
    median: int = 0
    """Median filter (source px) applied to line-art masks before tracing (0 = off)."""
    turdsize: int = 4
    """potrace: drop blobs (ink or paper) smaller than this many source px²."""
    min_hole_area: float = 1.5
    """Holes smaller than this (source px²) are dropped after nesting."""
    alphamax: float = 1.0
    """potrace corner threshold (0 = all corners, 1.334 = all smooth)."""
    opttolerance: float = 0.4
    """potrace curve optimisation tolerance (source px)."""
    subdiv: int = 6
    """Line segments sampled per Bezier segment."""
    simplify_px: float = 0.5
    """Douglas-Peucker tolerance (source px) applied to every polyline ring."""
    triangle_budget: int = 20000
    """``best`` mode prefers the highest score whose estimated triangle count stays under this."""
    auto_polarity: bool = True
    """Re-trace with flipped polarity if the outline is a card the size of the crop."""


FAST_GRID: tuple[dict[str, Any], ...] = ({},)
"""``fast``: the options as given, one trace."""

BEST_GRID: tuple[dict[str, Any], ...] = tuple(
    {"upscale": u, "turdsize": t, "opttolerance": o}
    for u, t, o in itertools.product((3, 4), (2, 4), (0.2, 0.5))
)
"""``best``: 8 candidates per piece (upscale x turdsize x curve tolerance)."""


@dataclass
class TraceResult:
    polygons: list[PolygonWithHoles]
    svg_d: str
    ink: np.ndarray
    """The cleaned mask that was actually traced (uint8, 255 = ink), at ``scale`` times source resolution."""
    scale: int
    smoothed: bool
    components: int
    speckle: float
    fidelity: Fidelity | None = None
    params: dict[str, Any] = field(default_factory=dict)
    flipped_polarity: bool = False

    @property
    def outer_count(self) -> int:
        return len(self.polygons)

    @property
    def hole_count(self) -> int:
        return sum(len(p.holes) for p in self.polygons)

    @property
    def vertex_count(self) -> int:
        return sum(len(p.outer) + sum(len(h) for h in p.holes) for p in self.polygons)


# --------------------------------------------------------------------------- #
# mask preparation
# --------------------------------------------------------------------------- #


def _odd(v: float, lo: int = 3) -> int:
    return max(lo, int(round(v)) | 1)


def _upscale(gray: np.ndarray, scale: int, interpolation: int = cv2.INTER_CUBIC) -> np.ndarray:
    if scale <= 1:
        return gray
    return cv2.resize(gray, None, fx=scale, fy=scale, interpolation=interpolation)


def binarize(gray: np.ndarray, opts: TraceOptions, scale: int) -> np.ndarray:
    """Edge-aware binarisation of an (upscaled) greyscale crop: uint8, 255 = ink."""
    src = gray
    if opts.unsharp > 0:
        blur = cv2.GaussianBlur(src, (0, 0), max(0.8, 0.6 * scale))
        src = cv2.addWeighted(src, 1.0 + opts.unsharp, blur, -opts.unsharp, 0)
    solid = src < opts.threshold
    if opts.binarize != "adaptive":
        return solid.astype(np.uint8) * 255
    block = _odd(opts.adaptive_block * scale)
    local = cv2.adaptiveThreshold(src, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, opts.adaptive_c) > 0
    # the adaptive test alone leaves the middle of solid areas as paper (local mean is black there)
    return (solid | (local & (src < opts.adaptive_max))).astype(np.uint8) * 255


def prepare_ink(piece: Piece, options: TraceOptions | None = None) -> tuple[np.ndarray, bool]:
    """Return ``(mask, smoothed)``: the uint8 ink mask to trace for ``piece`` at ``options.upscale`` x resolution.

    Line art is upscaled, sharpened and binarised edge-aware. Speckly
    (halftone / grey-shaded) pieces are instead upscaled, blurred, thresholded
    and closed so shading collapses into solid regions instead of thousands of
    dots. Either way ink outside the piece's region is removed.
    """
    opts = options or TraceOptions()
    scale = max(1, int(opts.upscale))
    gray = _upscale(piece.gray, scale)
    want = opts.smooth == "on" or (
        opts.smooth == "auto" and piece.speckle > opts.smooth_speckle and piece.grey_ratio > opts.smooth_grey
    )
    if want:
        mask = ink_mask(
            gray,
            threshold=min(255, opts.threshold + opts.smooth_threshold_boost),
            blur=_odd(opts.smooth_blur * scale),
            close=_odd(opts.smooth_close * scale),
        )
    else:
        mask = binarize(gray, opts, scale)
        if opts.median and opts.median > 1:
            mask = cv2.medianBlur(mask, _odd(opts.median * scale))
    region = piece.region if scale == 1 else _upscale(piece.region.astype(np.uint8), scale, cv2.INTER_NEAREST) > 0
    mask[~region] = 0
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


def path_to_svg_d(path: potrace.Path, scale: float = 1.0) -> str:
    """Serialise a potrace path as an SVG ``d`` attribute (fill with ``evenodd``), multiplying coordinates by ``scale``."""
    parts: list[str] = []
    f = lambda v: f"{v * scale:.2f}"  # noqa: E731
    for curve in path:
        s = curve.start_point
        parts.append(f"M{f(s.x)} {f(s.y)}")
        for seg in curve:
            if seg.is_corner:
                c, e = seg.c, seg.end_point
                parts.append(f"L{f(c.x)} {f(c.y)}L{f(e.x)} {f(e.y)}")
            else:
                c1, c2, e = seg.c1, seg.c2, seg.end_point
                parts.append(f"C{f(c1.x)} {f(c1.y)} {f(c2.x)} {f(c2.y)} {f(e.x)} {f(e.y)}")
        parts.append("Z")
    return "".join(parts)


def _potrace(paper: np.ndarray, opts: TraceOptions, scale: int) -> potrace.Path:
    """Run potrace on a *paper* mask (True = white) at ``scale`` x resolution. See the module docstring."""
    bitmap = potrace.Bitmap(np.ascontiguousarray(paper, dtype=bool))
    return bitmap.trace(
        turdsize=max(1, int(round(opts.turdsize * scale * scale))),
        alphamax=opts.alphamax,
        opticurve=True,
        opttolerance=opts.opttolerance * scale,
    )


# --------------------------------------------------------------------------- #
# nesting
# --------------------------------------------------------------------------- #


def nest_rings(rings: Iterable[np.ndarray], min_hole_area: float = 0.0) -> list[PolygonWithHoles]:
    """Group closed rings into outers and holes by containment depth.

    Rings are sorted by area (largest first). Each ring's parent is the smallest
    ring that contains it. Even depth (0, 2, ...) = outer boundary, odd depth =
    hole of its parent. An island inside a hole therefore becomes a new outer,
    which is exactly what an extruder needs. Holes (and the islands inside
    them) below ``min_hole_area`` are dropped.
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
    dropped = [False] * len(shp)
    for i in range(len(shp)):
        depth[i] = 0 if parent[i] < 0 else depth[parent[i]] + 1
        if parent[i] >= 0 and dropped[parent[i]]:
            dropped[i] = True
        elif depth[i] % 2 == 1 and shp[i][0].area < min_hole_area:
            dropped[i] = True

    polys: list[PolygonWithHoles] = []
    outer_index: dict[int, int] = {}
    for i, (_pg, r) in enumerate(shp):
        if depth[i] % 2 == 0 and not dropped[i]:
            outer_index[i] = len(polys)
            polys.append(PolygonWithHoles(outer=np.asarray(r, dtype=np.float64)))
    for i, (_pg, r) in enumerate(shp):
        if depth[i] % 2 == 1 and not dropped[i]:
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
    """True when the largest outer ring is essentially the crop rectangle itself (``shape`` in the polygons' units)."""
    if not polys:
        return False
    h, w = shape[:2]
    biggest = max(polys, key=lambda p: Polygon(p.outer).area)
    mn, mx = biggest.outer.min(axis=0), biggest.outer.max(axis=0)
    spans = (mx[0] - mn[0]) >= span_frac * w and (mx[1] - mn[1]) >= span_frac * h
    return spans and card_likeness([biggest], shape) >= area_frac


def estimate_triangles(polys: Sequence[PolygonWithHoles]) -> int:
    """Triangles the extruder will produce: ~2 per ring vertex for the caps plus 2 per edge for the walls."""
    n = sum(len(p.outer) + sum(len(h) for h in p.holes) for p in polys)
    return 4 * n


# --------------------------------------------------------------------------- #
# entry points
# --------------------------------------------------------------------------- #


def _trace_rings(paper: np.ndarray, opts: TraceOptions, scale: int) -> tuple[potrace.Path, list[PolygonWithHoles]]:
    """Trace a paper mask at ``scale`` x and return polygons in *source* pixels."""
    path = _potrace(paper, opts, scale)
    rings = (_curve_to_ring(c, opts.subdiv, opts.simplify_px * scale) for c in path.curves)
    polys = nest_rings((r for r in rings if r is not None), min_hole_area=opts.min_hole_area * scale * scale)
    if scale != 1:
        for p in polys:
            p.outer = p.outer / scale
            p.holes = [h / scale for h in p.holes]
    return path, polys


def trace_ink(
    ink: np.ndarray, options: TraceOptions | None = None, scale: int = 1
) -> tuple[potrace.Path, list[PolygonWithHoles], bool]:
    """Trace an **ink** mask (nonzero / True = ink, at ``scale`` x source resolution) into nested polygons.

    Returns ``(potrace_path, polygons, flipped)`` with polygons in source pixels
    (the path stays in mask pixels). ``flipped`` is True when the polarity guard
    had to invert the input: the trace came back as one ring the size of the
    crop (a "card"), which is what you get when a paper mask is handed to
    potrace as if it were ink. A design that genuinely is a filled rectangle is
    left alone because its flipped trace is empty or card-like too.
    """
    opts = options or TraceOptions()
    ink_b = np.asarray(ink) > 0
    shape = (ink_b.shape[0] / scale, ink_b.shape[1] / scale)

    path, polys = _trace_rings(~ink_b, opts, scale)
    if not opts.auto_polarity or not is_cardlike(polys, shape):
        return path, polys, False

    alt_path, alt_polys = _trace_rings(ink_b, opts, scale)
    if alt_polys and not is_cardlike(alt_polys, shape):
        log.warning("trace: outline was a card the size of the crop; input looked like a paper mask, flipped polarity")
        return alt_path, alt_polys, True
    return path, polys, False


def _trace_once(piece: Piece, opts: TraceOptions, params: dict[str, Any]) -> TraceResult:
    o = replace(opts, **params)
    scale = max(1, int(o.upscale))
    mask, smoothed = prepare_ink(piece, o)
    comps, speck = speckle_score(mask)
    path, polys, flipped = trace_ink(mask, o, scale)
    fid = measure(mask, polys, scale, {"upscale": scale, "turdsize": o.turdsize, "opttolerance": o.opttolerance, "binarize": o.binarize, "adaptive_block": o.adaptive_block})
    return TraceResult(
        polygons=polys,
        svg_d=path_to_svg_d(path, 1.0 / scale),
        ink=mask,
        scale=scale,
        smoothed=smoothed,
        components=comps,
        speckle=speck,
        fidelity=fid,
        params=fid.params,
        flipped_polarity=flipped,
    )


def trace_piece(piece: Piece, options: TraceOptions | None = None) -> TraceResult:
    """Prepare and trace one :class:`Piece`; in ``best`` mode try :data:`BEST_GRID` and keep the best score under budget."""
    opts = options or TraceOptions()
    grid = BEST_GRID if opts.fidelity == "best" else FAST_GRID
    results = [_trace_once(piece, opts, params) for params in grid]
    if len(results) == 1:
        return results[0]
    under = [r for r in results if estimate_triangles(r.polygons) <= opts.triangle_budget]
    pool = under or [min(results, key=lambda r: estimate_triangles(r.polygons))]
    best = max(pool, key=lambda r: r.fidelity.score if r.fidelity else 0.0)
    if best.fidelity is not None:
        best.fidelity.candidates = len(results)
    return best
