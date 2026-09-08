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

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from PIL import Image

__all__ = [
    "Piece",
    "SegmentOptions",
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


def segment_sheet(gray: np.ndarray, options: SegmentOptions | None = None) -> list[Piece]:
    """Split a greyscale sheet into :class:`Piece` crops in reading order."""
    opts = (options or SegmentOptions()).resolved(gray.shape)
    inner, m = _crop_margin(gray, opts.margin)
    H, W = inner.shape
    full_h, full_w = gray.shape

    ink = ink_mask(inner, threshold=opts.threshold)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (opts.merge_kernel, opts.merge_kernel))
    merged = cv2.dilate(ink, k)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(merged, connectivity=8)

    comps: list[_Component] = []
    for lab in range(1, n):
        x, y, w, h, _area = (int(v) for v in stats[lab])
        sub_ink = ink[y : y + h, x : x + w]
        sub_lab = labels[y : y + h, x : x + w]
        ink_area = int(np.count_nonzero(sub_ink[sub_lab == lab]))
        if ink_area == 0:
            continue
        comps.append(_Component(x, y, w, h, {lab}, ink_area))

    comps = absorb_small(comps, opts.absorb_max or 0, opts.absorb_gap or 0)
    comps = [c for c in comps if c.w >= opts.min_size and c.h >= opts.min_size and c.ink_area >= opts.min_ink_area]
    comps = reading_order(comps)

    pieces: list[Piece] = []
    for idx, c in enumerate(comps):
        pad = opts.pad
        x0, y0 = max(0, c.x - pad), max(0, c.y - pad)
        x1, y1 = min(W, c.x + c.w + pad), min(H, c.y + c.h + pad)
        sub_lab = labels[y0:y1, x0:x1]
        region = np.isin(sub_lab, list(c.labels)) if len(c.labels) > 1 else sub_lab == next(iter(c.labels))
        crop_ink = ink[y0:y1, x0:x1].copy()
        crop_ink[~region] = 0
        crop_gray = inner[y0:y1, x0:x1].copy()
        crop_gray[~region] = 255
        n_cc, speck = speckle_score(crop_ink)
        pieces.append(
            Piece(
                index=idx,
                x=x0 + m,
                y=y0 + m,
                w=x1 - x0,
                h=y1 - y0,
                gray=crop_gray,
                ink=crop_ink,
                region=region,
                sheet_w=full_w,
                sheet_h=full_h,
                ink_area=c.ink_area,
                components=n_cc,
                speckle=speck,
                id=f"{idx:02d}",
            )
        )
    return pieces


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
    """Treat the entire image as one design (single-image mode).

    The crop is tightened to the ink's bounding box (plus ``pad``) so a design on
    a large white canvas still gets a sensible extrusion depth.
    """
    ink = ink_mask(gray, threshold=threshold)
    ys, xs = np.nonzero(ink)
    h, w = gray.shape
    if len(xs) == 0:
        x0, y0, x1, y1 = 0, 0, w, h
    else:
        x0, x1 = max(0, int(xs.min()) - pad), min(w, int(xs.max()) + 1 + pad)
        y0, y1 = max(0, int(ys.min()) - pad), min(h, int(ys.max()) + 1 + pad)
    crop_ink = ink[y0:y1, x0:x1]
    comps, speck = speckle_score(crop_ink)
    return Piece(
        index=0,
        x=x0,
        y=y0,
        w=x1 - x0,
        h=y1 - y0,
        gray=gray[y0:y1, x0:x1].copy(),
        ink=crop_ink.copy(),
        region=np.ones(crop_ink.shape, dtype=bool),
        sheet_w=w,
        sheet_h=h,
        ink_area=int(np.count_nonzero(crop_ink)),
        components=comps,
        speckle=speck,
        id="00",
    )


def looks_like_single_design(pieces: Sequence[Piece], dominant: float = 0.6) -> bool:
    """Heuristic for auto mode: one piece holds ``dominant`` of all ink, or there is only one."""
    if len(pieces) <= 1:
        return True
    total = sum(p.ink_area for p in pieces)
    if total == 0:
        return True
    return max(p.ink_area for p in pieces) / total >= dominant


def draw_segmentation(gray: np.ndarray, pieces: Sequence[Piece]) -> np.ndarray:
    """Return a BGR copy of the sheet with every piece's box and id drawn on it (debug aid)."""
    out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    scale = max(0.4, min(gray.shape) / 1500)
    for p in pieces:
        cv2.rectangle(out, (p.x, p.y), (p.x + p.w, p.y + p.h), (0, 0, 255), 2)
        cv2.putText(out, p.id, (p.x + 3, p.y + int(18 * scale) + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, (0, 0, 255), 2)
    return out
