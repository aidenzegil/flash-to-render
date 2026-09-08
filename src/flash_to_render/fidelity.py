"""How faithfully did the trace reproduce the ink? A score you can optimise against.

The traced polygons are rasterised back at the (upscaled) crop resolution and
compared with the cleaned ink mask they were traced from:

- **IoU** of ink vs rasterised trace,
- **thin-feature recall**: the fraction of "thin" ink pixels (those an opening
  with a ~2 source-px kernel removes: seams, hairlines, small serifs) that the
  trace still covers,
- **hole ratio**: enclosed paper regions (counters, gaps between limbs) in the
  trace vs in the mask.

``score = 0.6 * iou + 0.25 * thin_recall + 0.15 * hole_ratio``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Sequence

import cv2
import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from .trace import PolygonWithHoles

__all__ = ["Fidelity", "rasterize", "count_holes", "measure"]


@dataclass
class Fidelity:
    iou: float
    thin_recall: float
    hole_ratio: float
    score: float
    holes_mask: int
    holes_trace: int
    thin_fraction: float
    """Share of the ink that is thin (used to pick a shallower extrusion for fine pieces)."""
    params: dict[str, Any] = field(default_factory=dict)
    candidates: int = 1

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for k in ("iou", "thin_recall", "hole_ratio", "score", "thin_fraction"):
            d[k] = round(d[k], 4)
        return d


def rasterize(polys: Sequence["PolygonWithHoles"], shape: tuple[int, int], scale: float = 1.0) -> np.ndarray:
    """Fill the polygons (source-px coordinates times ``scale``) into a uint8 mask of ``shape``."""
    out = np.zeros(shape[:2], dtype=np.uint8)
    # largest first: an island inside another polygon's hole is painted after that hole was cleared
    for p in sorted(polys, key=lambda p: -p.bbox_area):
        if len(p.outer) < 3:
            continue
        cv2.fillPoly(out, [np.round(np.asarray(p.outer) * scale).astype(np.int32)], 255)
        holes = [np.round(np.asarray(h) * scale).astype(np.int32) for h in p.holes if len(h) >= 3]
        if holes:
            cv2.fillPoly(out, holes, 0)
    return out


def count_holes(mask: np.ndarray) -> int:
    """Number of paper regions completely enclosed by ink."""
    paper = (mask == 0).astype(np.uint8)
    n, labels = cv2.connectedComponents(paper, connectivity=4)
    if n <= 1:
        return 0
    border = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    return int(n - 1 - len([b for b in border if b != 0]))


def measure(mask: np.ndarray, polys: Sequence["PolygonWithHoles"], scale: float, params: dict[str, Any] | None = None) -> Fidelity:
    """Score ``polys`` (source px) against ``mask`` (uint8, at ``scale`` times source resolution)."""
    ink = mask > 0
    raster = rasterize(polys, mask.shape, scale) > 0
    ink_px = int(ink.sum())
    if ink_px == 0:
        return Fidelity(1.0, 1.0, 1.0, 1.0, 0, 0, 0.0, params or {})
    inter = int((ink & raster).sum())
    union = int((ink | raster).sum())
    iou = inter / union if union else 1.0

    k = max(3, int(round(2 * scale)) | 1)
    el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, el) > 0
    thin = ink & ~opened
    thin_px = int(thin.sum())
    thin_recall = int((thin & raster).sum()) / thin_px if thin_px else 1.0

    hm, ht = count_holes(mask), count_holes(raster.astype(np.uint8) * 255)
    hole_ratio = 1.0 if hm == ht == 0 else min(hm, ht) / max(hm, ht)

    score = 0.6 * iou + 0.25 * thin_recall + 0.15 * hole_ratio
    return Fidelity(iou, thin_recall, hole_ratio, score, hm, ht, thin_px / ink_px, params or {})
