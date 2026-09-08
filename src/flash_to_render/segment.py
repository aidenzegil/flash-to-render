"""Sheet -> pieces.

A flash sheet is a white page with several independent designs on it. This module
finds those designs with plain OpenCV morphology:

1. threshold the greyscale sheet into an *ink* mask (dark pixels),
2. dilate the ink so the separate strokes of one design touch each other,
3. take the connected components of the dilated mask,
4. crop every component back out of the *undilated* ink mask.

The dilation kernel is the single most important knob: too small and a sparse
design splits into several pieces, too large and a caption merges into the
drawing above it. The defaults are derived from the sheet size and every knob is
exposed on :class:`SegmentOptions` (and the CLI).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from PIL import Image

__all__ = [
    "Box",
    "Region",
    "Piece",
    "SegmentOptions",
    "detect",
    "detect_boxes",
    "crop_pieces",
    "piece_id",
    "slugify",
    "load_gray",
    "ink_mask",
    "speckle_score",
    "segment_sheet",
    "absorb_small",
    "reading_order",
    "whole_image_piece",
    "looks_like_single_design",
    "draw_segmentation",
]


def load_gray(path: str | Path) -> np.ndarray:
    """Load any raster (PNG/JPG/WebP, with or without alpha) as an 8-bit greyscale array.

    Transparent pixels are composited over white so an already-cut-out design
    still reads as ink-on-paper.
    """
    img = Image.open(path)
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        bg.alpha_composite(img)
        img = bg
    return np.asarray(img.convert("L"), dtype=np.uint8)


def ink_mask(
    gray: np.ndarray,
    threshold: int = 150,
    blur: int = 0,
    close: int = 0,
) -> np.ndarray:
    """Return a uint8 mask where 255 = ink and 0 = paper.

    ``blur`` (odd Gaussian kernel size, 0 = off) and ``close`` (odd ellipse kernel
    for a morphological close, 0 = off) are the halftone tamers: grey-shaded
    designs otherwise threshold into thousands of speckles. Leave both off for
    crisp line art.
    """
    src = gray
    if blur and blur > 1:
        k = blur | 1
        src = cv2.GaussianBlur(src, (k, k), 0)
    _, mask = cv2.threshold(src, threshold, 255, cv2.THRESH_BINARY_INV)
    if close and close > 1:
        k = close | 1
        el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, el)
    return mask


def speckle_score(mask: np.ndarray) -> tuple[int, float]:
    """Connected-component count of an ink mask and the *speckle* metric.

    Speckle = ink components per 1000 ink pixels. Clean line art scores well under
    ~3; halftone shading scores 10-50 and traces into confetti.
    """
    binary = (mask > 0).astype(np.uint8)
    ink_px = int(binary.sum())
    if ink_px == 0:
        return 0, 0.0
    n = cv2.connectedComponents(binary, connectivity=8)[0] - 1
    return int(n), n / ink_px * 1000.0


@dataclass
class SegmentOptions:
    """Tunable knobs for :func:`segment_sheet`. ``None`` means "derive from image size"."""

    threshold: int = 150
    """Grey level below which a pixel is ink (0-255)."""
    margin: float = 0.015
    """Fraction of the shorter side to ignore along every edge (scanner shadows, card borders)."""
    merge_kernel: int | None = None
    """Odd dilation kernel (px) that fuses the strokes of one design. Default ~1% of the long side."""
    min_ink_area: int | None = None
    """Components with fewer ink pixels are dropped as noise. Default scales with sheet area."""
    min_size: int | None = None
    """Components narrower or shorter than this (px) are dropped. Default ~2.5% of the long side."""
    pad: int = 6
    """Padding (px) added around each crop."""
    absorb_max: int | None = None
    """Pieces whose longest side is below this (px) are *small* and get absorbed into a close neighbour
    (this is what keeps the words of a caption together). Default ~6% of the long side; 0 disables."""
    absorb_gap: int | None = None
    """Maximum bbox-to-bbox gap (px) for absorbing a small piece. Default 1.5x the merge kernel."""

    def resolved(self, shape: tuple[int, int]) -> "SegmentOptions":
        """Return a copy with every ``None`` replaced by a size-derived default."""
        h, w = shape[:2]
        long_side = max(h, w)
        merge = self.merge_kernel if self.merge_kernel is not None else max(5, int(round(long_side * 0.0095)))
        merge |= 1
        min_size = self.min_size if self.min_size is not None else max(12, int(long_side * 0.025))
        min_area = self.min_ink_area if self.min_ink_area is not None else max(100, int(h * w * 0.0003))
        absorb_max = self.absorb_max if self.absorb_max is not None else int(long_side * 0.06)
        absorb_gap = self.absorb_gap if self.absorb_gap is not None else int(merge * 1.5)
        return SegmentOptions(
            threshold=self.threshold,
            margin=self.margin,
            merge_kernel=merge,
            min_ink_area=min_area,
            min_size=min_size,
            pad=self.pad,
            absorb_max=absorb_max,
            absorb_gap=absorb_gap,
        )


@dataclass
class Piece:
    """One design cut out of a sheet.

    Coordinates are in the *original* sheet's pixel space (margin already added back).
    ``gray`` and ``ink`` are crops of identical shape; ``ink`` has neighbouring
    designs that bled into the bounding box already masked out.
    """

    index: int
    x: int
    y: int
    w: int
    h: int
    gray: np.ndarray
    ink: np.ndarray
    region: np.ndarray
    """Boolean mask of the dilated component: which pixels of the crop belong to this design."""
    sheet_w: int
    sheet_h: int
    ink_area: int = 0
    components: int = 0
    speckle: float = 0.0
    id: str = ""
    name: str = ""
    kind: str = "rect"
    """``rect`` or ``polygon``: the kind of region this piece was cut with."""
    extra: dict = field(default_factory=dict)

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)

    @property
    def longest_side(self) -> int:
        return max(self.w, self.h)


def _crop_margin(gray: np.ndarray, margin: float) -> tuple[np.ndarray, int]:
    h, w = gray.shape
    m = int(min(h, w) * margin)
    if m <= 0:
        return gray, 0
    return gray[m : h - m, m : w - m], m


@dataclass
class Box:
    """An axis-aligned box on the sheet (pixels, full-image coordinates) with an optional name."""

    x: int
    y: int
    w: int
    h: int
    name: str = ""

    def to_dict(self) -> dict:
        return {"x": int(self.x), "y": int(self.y), "w": int(self.w), "h": int(self.h), "name": self.name}

    @classmethod
    def from_dict(cls, d: dict) -> "Box":
        return cls(int(round(d["x"])), int(round(d["y"])), int(round(d["w"])), int(round(d["h"])), str(d.get("name", "") or ""))

    def clamp(self, width: int, height: int) -> "Box":
        x0, y0 = min(max(0, self.x), width - 1), min(max(0, self.y), height - 1)
        x1, y1 = min(width, self.x + self.w), min(height, self.y + self.h)
        return Box(x0, y0, max(1, x1 - x0), max(1, y1 - y0), self.name)

    @property
    def area(self) -> int:
        return self.w * self.h


@dataclass
class Region:
    """A design outline on the sheet: a rectangle or a freehand polygon, in image pixel coordinates.

    This is the shape stored in the library sidecar
    (``{"id", "name", "kind": "rect" | "polygon", "points": [[x, y], ...]}``).
    A rectangle is just a 4-point polygon; :meth:`bbox` and :meth:`mask` are
    what the cropper needs from either kind.
    """

    points: list[tuple[float, float]]
    kind: str = "polygon"
    name: str = ""
    id: str = ""

    @classmethod
    def rect(cls, x: float, y: float, w: float, h: float, name: str = "", id: str = "") -> "Region":
        return cls([(x, y), (x + w, y), (x + w, y + h), (x, y + h)], "rect", name, id)

    @classmethod
    def from_box(cls, box: "Box") -> "Region":
        return cls.rect(box.x, box.y, box.w, box.h, box.name)

    @classmethod
    def from_dict(cls, d: dict) -> "Region":
        if "points" in d:
            pts = [(float(p[0]), float(p[1])) for p in d["points"]]
            kind = d.get("kind") or ("rect" if len(pts) == 4 else "polygon")
            return cls(pts, kind, str(d.get("name", "") or ""), str(d.get("id", "") or ""))
        return cls.from_box(Box.from_dict(d))  # legacy rect-only shape

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "kind": self.kind,
            "points": [[round(float(x), 1), round(float(y), 1)] for x, y in self.points],
        }

    def bbox(self) -> Box:
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        x0, y0 = int(np.floor(min(xs))), int(np.floor(min(ys)))
        x1, y1 = int(np.ceil(max(xs))), int(np.ceil(max(ys)))
        return Box(x0, y0, max(1, x1 - x0), max(1, y1 - y0), self.name)

    def mask(self, box: "Box") -> np.ndarray:
        """Boolean mask of the region's interior, in the coordinates of ``box`` (normally its own clamped bbox)."""
        if self.kind == "rect":
            return np.ones((box.h, box.w), dtype=bool)
        m = np.zeros((box.h, box.w), dtype=np.uint8)
        pts = np.array([[x - box.x, y - box.y] for x, y in self.points], dtype=np.float32)
        cv2.fillPoly(m, [np.round(pts).astype(np.int32)], 255)
        return m > 0

    @property
    def area(self) -> float:
        xs = np.array([p[0] for p in self.points])
        ys = np.array([p[1] for p in self.points])
        return float(abs(np.dot(xs, np.roll(ys, -1)) - np.dot(ys, np.roll(xs, -1))) / 2)


def slugify(name: str) -> str:
    """Filename-safe version of a piece name (``"Skull & dagger"`` -> ``"skull-dagger"``)."""
    out = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return out[:48]


def piece_id(index: int, name: str = "") -> str:
    """``"03"`` for unnamed pieces, ``"03-skull"`` for named ones: sortable *and* readable."""
    slug = slugify(name) if name else ""
    return f"{index:02d}-{slug}" if slug else f"{index:02d}"


def _ink_components(gray: np.ndarray, opts: SegmentOptions) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(ink, labels, stats)``: the ink mask and the connected components of its dilation."""
    ink = ink_mask(gray, threshold=opts.threshold)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (opts.merge_kernel, opts.merge_kernel))
    merged = cv2.dilate(ink, k)
    _n, labels, stats, _ = cv2.connectedComponentsWithStats(merged, connectivity=8)
    return ink, labels, stats


def detect_boxes(gray: np.ndarray, options: SegmentOptions | None = None, mode: str = "sheet") -> list[Box]:
    """Find the designs on a sheet and return their boxes (padded, full-image coordinates, reading order).

    Pure: nothing is cropped. ``mode`` is ``sheet`` (always segment), ``single``
    (one box around all the ink) or ``auto`` (single when one component holds
    most of the ink). Feed the result, edited or not, to :func:`crop_pieces`.
    """
    return detect(gray, options, mode)[1]


def detect(gray: np.ndarray, options: SegmentOptions | None = None, mode: str = "auto") -> tuple[str, list[Box]]:
    """Like :func:`detect_boxes` but also returns the resolved mode (``sheet`` or ``single``)."""
    opts = (options or SegmentOptions()).resolved(gray.shape)
    full_h, full_w = gray.shape
    if mode == "single":
        return "single", [_ink_box(gray, opts.threshold, opts.pad)]

    inner, m = _crop_margin(gray, opts.margin)
    H, W = inner.shape
    ink, labels, stats = _ink_components(inner, opts)

    comps: list[_Component] = []
    for lab in range(1, len(stats)):
        x, y, w, h, _area = (int(v) for v in stats[lab])
        sub_ink = ink[y : y + h, x : x + w]
        sub_lab = labels[y : y + h, x : x + w]
        ink_area = int(np.count_nonzero(sub_ink[sub_lab == lab]))
        if ink_area == 0:
            continue
        comps.append(_Component(x, y, w, h, {lab}, ink_area))

    comps = absorb_small(comps, opts.absorb_max or 0, opts.absorb_gap or 0)
    comps = [c for c in comps if c.w >= opts.min_size and c.h >= opts.min_size and c.ink_area >= opts.min_ink_area]
    if mode == "auto" and _dominant(comps):
        return "single", [_ink_box(gray, opts.threshold, opts.pad)]
    comps = reading_order(comps)

    boxes: list[Box] = []
    for c in comps:
        pad = opts.pad
        x0, y0 = max(0, c.x - pad), max(0, c.y - pad)
        x1, y1 = min(W, c.x + c.w + pad), min(H, c.y + c.h + pad)
        boxes.append(Box(x0 + m, y0 + m, x1 - x0, y1 - y0))
    return "sheet", boxes


def _dominant(comps: Sequence["_Component"], fraction: float = 0.6) -> bool:
    if len(comps) <= 1:
        return True
    total = sum(c.ink_area for c in comps)
    return total == 0 or max(c.ink_area for c in comps) / total >= fraction


def _ink_box(gray: np.ndarray, threshold: int, pad: int) -> Box:
    """One box around every ink pixel in the image (single-design mode)."""
    ink = ink_mask(gray, threshold=threshold)
    ys, xs = np.nonzero(ink)
    h, w = gray.shape
    if len(xs) == 0:
        return Box(0, 0, w, h)
    x0, x1 = max(0, int(xs.min()) - pad), min(w, int(xs.max()) + 1 + pad)
    y0, y1 = max(0, int(ys.min()) - pad), min(h, int(ys.max()) + 1 + pad)
    return Box(x0, y0, x1 - x0, y1 - y0)


def crop_pieces(gray: np.ndarray, regions: Sequence[Box | Region], options: SegmentOptions | None = None) -> list[Piece]:
    """Cut a :class:`Piece` out of the sheet for every region (a :class:`Box` or a :class:`Region`).

    A polygon region is cropped to its bounding box and everything outside the
    polygon is set to paper before anything else happens, so ink from a
    neighbouring design inside the bbox is excluded.

    Which of the remaining ink belongs to the region? Every connected component
    of the dilated ink is owned by the *smallest* region that holds at least
    half of it, so a design tucked inside a neighbour's bbox goes to its own
    box and a lasso beats the auto box it overlaps. When no region holds half
    of a component, the user is splitting a merged design and every region
    keeps the pixels inside it. Anything else is a neighbour bleeding in and
    is masked.
    """
    opts = (options or SegmentOptions()).resolved(gray.shape)
    full_h, full_w = gray.shape
    ink, labels, stats = _ink_components(gray, opts)
    n_labels = len(stats)
    comp_px = np.bincount(labels.ravel(), minlength=n_labels).astype(np.float64)

    regs = [r if isinstance(r, Region) else Region.from_box(r) for r in regions]
    boxes = [r.bbox().clamp(full_w, full_h) for r in regs]
    masks = [r.mask(b) for r, b in zip(regs, boxes)]
    inside = np.zeros((len(boxes), n_labels), dtype=np.float64)
    for i, (box, mask) in enumerate(zip(boxes, masks)):
        sub = labels[box.y : box.y + box.h, box.x : box.x + box.w]
        inside[i] = np.bincount(sub[mask].ravel(), minlength=n_labels)
    frac = inside / np.maximum(comp_px, 1)[None, :]

    claims: list[set[int]] = [set() for _ in boxes]
    areas = np.array([m.sum() for m in masks]) if boxes else np.zeros(0)
    for lab in range(1, n_labels):
        if comp_px[lab] == 0 or not inside[:, lab].any():
            continue
        candidates = np.nonzero(frac[:, lab] >= 0.5)[0]
        if len(candidates):
            # the smallest region holding at least half of the component owns it: a design tucked
            # inside a neighbour's bbox goes to its own box, a lasso beats the auto box it overlaps
            owner = int(candidates[np.argmin(areas[candidates])])
            claims[owner].add(lab)
        else:
            for i in np.nonzero(inside[:, lab] > 0)[0]:
                claims[int(i)].add(lab)

    pieces: list[Piece] = []
    for idx, (reg, box, mask) in enumerate(zip(regs, boxes, masks)):
        x0, y0, x1, y1 = box.x, box.y, box.x + box.w, box.y + box.h
        sub_lab = labels[y0:y1, x0:x1]
        keep = sorted(claims[idx])
        region = (np.isin(sub_lab, keep) if keep else np.zeros(sub_lab.shape, dtype=bool)) & mask
        crop_ink = ink[y0:y1, x0:x1].copy()
        crop_ink[~region] = 0
        crop_gray = gray[y0:y1, x0:x1].copy()
        crop_gray[~region] = 255
        n_cc, speck = speckle_score(crop_ink)
        pieces.append(
            Piece(
                index=idx,
                x=x0,
                y=y0,
                w=box.w,
                h=box.h,
                gray=crop_gray,
                ink=crop_ink,
                region=region,
                sheet_w=full_w,
                sheet_h=full_h,
                ink_area=int(np.count_nonzero(crop_ink)),
                components=n_cc,
                speckle=speck,
                id=piece_id(idx, reg.name),
                name=reg.name,
                kind=reg.kind,
            )
        )
    return pieces


def segment_sheet(gray: np.ndarray, options: SegmentOptions | None = None) -> list[Piece]:
    """Detect and crop in one go: :func:`detect_boxes` (sheet mode) then :func:`crop_pieces`."""
    return crop_pieces(gray, detect_boxes(gray, options, mode="sheet"), options)


@dataclass
class _Component:
    x: int
    y: int
    w: int
    h: int
    labels: set[int]
    ink_area: int

    @property
    def longest_side(self) -> int:
        return max(self.w, self.h)

    def gap_to(self, other: "_Component") -> float:
        """Axis-aligned distance between two boxes (0 when they overlap)."""
        dx = max(0, max(self.x, other.x) - min(self.x + self.w, other.x + other.w))
        dy = max(0, max(self.y, other.y) - min(self.y + self.h, other.y + other.h))
        return float(np.hypot(dx, dy))

    def merged(self, other: "_Component") -> "_Component":
        x0, y0 = min(self.x, other.x), min(self.y, other.y)
        x1, y1 = max(self.x + self.w, other.x + other.w), max(self.y + self.h, other.y + other.h)
        return _Component(x0, y0, x1 - x0, y1 - y0, self.labels | other.labels, self.ink_area + other.ink_area)


def absorb_small(comps: list[_Component], max_side: int, gap: int) -> list[_Component]:
    """Merge every component smaller than ``max_side`` into its nearest neighbour within ``gap`` px.

    Dilation alone either splits the spaced words of a caption or glues the
    caption to the drawing above it; absorbing only *small* components lets the
    merge kernel stay tight while keeping "it's okay i'm spiderman" as one piece.
    """
    if max_side <= 0 or gap <= 0:
        return comps
    comps = list(comps)
    changed = True
    while changed and len(comps) > 1:
        changed = False
        comps.sort(key=lambda c: c.ink_area)  # smallest first so tiny bits join before big ones move
        for i, c in enumerate(comps):
            if c.longest_side >= max_side:
                continue
            best, best_gap = -1, float("inf")
            for j, o in enumerate(comps):
                if j == i:
                    continue
                g = c.gap_to(o)
                if g < best_gap:
                    best, best_gap = j, g
            if best >= 0 and best_gap <= gap:
                merged = c.merged(comps[best])
                comps = [o for k, o in enumerate(comps) if k not in (i, best)] + [merged]
                changed = True
                break
    return comps


def reading_order(comps: list[_Component]) -> list[_Component]:
    """Sort into rows (top to bottom) and left to right within a row.

    Two components share a row when their vertical extents overlap by at least
    half of the shorter one; each row is anchored by its first (topmost) member.
    """
    rows: list[list[_Component]] = []
    for c in sorted(comps, key=lambda c: (c.y, c.x)):
        for row in rows:
            a = row[0]
            overlap = min(a.y + a.h, c.y + c.h) - max(a.y, c.y)
            if overlap >= 0.5 * min(a.h, c.h):
                row.append(c)
                break
        else:
            rows.append([c])
    rows.sort(key=lambda r: r[0].y)
    return [c for row in rows for c in sorted(row, key=lambda c: c.x)]


def whole_image_piece(gray: np.ndarray, threshold: int = 150, pad: int = 0) -> Piece:
    """Treat the entire image as one design (single-image mode); the crop is tightened to the ink's bbox."""
    opts = SegmentOptions(threshold=threshold, pad=pad)
    return crop_pieces(gray, detect_boxes(gray, opts, mode="single"), opts)[0]


def looks_like_single_design(pieces: Sequence[Piece], dominant: float = 0.6) -> bool:
    """Heuristic for auto mode: one piece holds ``dominant`` of all ink, or there is only one."""
    if len(pieces) <= 1:
        return True
    total = sum(p.ink_area for p in pieces)
    if total == 0:
        return True
    return max(p.ink_area for p in pieces) / total >= dominant


def draw_segmentation(gray: np.ndarray, pieces: Sequence[Piece | Box]) -> np.ndarray:
    """Return a BGR copy of the sheet with every piece's box and id drawn on it (debug aid)."""
    out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    scale = max(0.4, min(gray.shape) / 1500)
    for p in pieces:
        cv2.rectangle(out, (p.x, p.y), (p.x + p.w, p.y + p.h), (0, 0, 255), 2)
        label = getattr(p, "id", None) or p.name or str(pieces.index(p))
        cv2.putText(out, label, (p.x + 3, p.y + int(18 * scale) + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, (0, 0, 255), 2)
    return out
