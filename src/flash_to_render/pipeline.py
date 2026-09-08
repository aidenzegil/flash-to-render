"""End-to-end: image file -> per-design PNG / SVG / GLB + manifest + combined scene."""

from __future__ import annotations

import logging
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, TextIO

import cv2
import numpy as np

from . import __version__
from .export import write_cutout_png, write_glb, write_manifest, write_scene_glb, write_svg
from .mesh import DEFAULT_DEPTH_FRAC, Mesh, depth_for, extrude
from .segment import (
    Piece,
    SegmentOptions,
    draw_segmentation,
    load_gray,
    looks_like_single_design,
    segment_sheet,
    whole_image_piece,
)
from .trace import TraceOptions, TraceResult, trace_piece

log = logging.getLogger(__name__)

__all__ = ["PipelineOptions", "PieceResult", "RunResult", "run", "process_pieces", "detect_mode"]

QUALITY_ORDER = ("ok", "tiny", "speckle", "empty")


@dataclass
class PipelineOptions:
    mode: str = "auto"
    """``auto`` decides sheet vs single design, ``sheet`` / ``single`` force it."""
    segment: SegmentOptions = field(default_factory=SegmentOptions)
    trace: TraceOptions = field(default_factory=TraceOptions)
    depth_frac: float = DEFAULT_DEPTH_FRAC
    """Extrusion depth as a fraction of each piece's longest side."""
    scale: float = 0.001
    """Output units per source pixel. glTF is in metres, so the default makes 1 px = 1 mm."""
    max_speckle: float = 8.0
    """Pieces whose (post-smoothing) speckle score exceeds this are flagged ``speckle``."""
    min_piece_px: int = 150
    """Pieces whose longest side is below this are flagged ``tiny``."""
    drop_flagged: bool = False
    """Skip writing assets for flagged pieces instead of just marking them."""
    debug: bool = False
    """Also write ``segmentation.png`` with every box drawn on the sheet."""
    write_png: bool = True
    write_svg: bool = True
    write_glb: bool = True


@dataclass
class PieceResult:
    piece: Piece
    trace: TraceResult | None
    mesh: Mesh | None
    depth_px: float
    quality: str
    warnings: list[str] = field(default_factory=list)
    files: dict[str, str] = field(default_factory=dict)

    @property
    def triangles(self) -> int:
        return self.mesh.triangle_count if self.mesh is not None else 0

    @property
    def vertices(self) -> int:
        return self.mesh.vertex_count if self.mesh is not None else 0

    def to_manifest(self) -> dict[str, Any]:
        p, t = self.piece, self.trace
        return {
            "id": p.id,
            "bbox": [p.x, p.y, p.w, p.h],
            "ink_area": p.ink_area,
            "components": t.components if t else p.components,
            "speckle": round(t.speckle if t else p.speckle, 3),
            "raw_speckle": round(p.speckle, 3),
            "smoothed": bool(t.smoothed) if t else False,
            "polarity_flipped": bool(t.flipped_polarity) if t else False,
            "outers": t.outer_count if t else 0,
            "holes": t.hole_count if t else 0,
            "vertices": self.vertices,
            "triangles": self.triangles,
            "depth_px": round(self.depth_px, 3),
            "quality": self.quality,
            "warnings": self.warnings,
            "files": self.files,
        }


@dataclass
class RunResult:
    source: Path
    out_dir: Path
    mode: str
    sheet_size: tuple[int, int]
    pieces: list[PieceResult]
    manifest_path: Path
    scene_path: Path | None

    @property
    def ok_pieces(self) -> list[PieceResult]:
        return [p for p in self.pieces if p.quality == "ok"]


def detect_mode(gray: np.ndarray, options: PipelineOptions) -> tuple[str, list[Piece]]:
    """Return ``(mode, pieces)``; segmentation is run at most once."""
    if options.mode == "single":
        return "single", [whole_image_piece(gray, threshold=options.segment.threshold, pad=options.segment.pad)]
    pieces = segment_sheet(gray, options.segment)
    if options.mode == "sheet":
        return "sheet", pieces
    if looks_like_single_design(pieces):
        return "single", [whole_image_piece(gray, threshold=options.segment.threshold, pad=options.segment.pad)]
    return "sheet", pieces


def _classify(piece: Piece, trace: TraceResult, mesh: Mesh, options: PipelineOptions) -> tuple[str, list[str]]:
    warnings: list[str] = []
    if trace.flipped_polarity:
        warnings.append("polarity flipped by guard")
    if trace.smoothed:
        warnings.append(f"halftone smoothing applied (raw speckle {piece.speckle:.1f})")
    if mesh.is_empty or not trace.polygons:
        return "empty", warnings + ["no polygons after tracing"]
    if trace.speckle > options.max_speckle:
        return "speckle", warnings + [f"speckle {trace.speckle:.1f} > {options.max_speckle}"]
    if piece.longest_side < options.min_piece_px:
        return "tiny", warnings + [f"longest side {piece.longest_side}px < {options.min_piece_px}px"]
    return "ok", warnings


def process_pieces(pieces: list[Piece], options: PipelineOptions) -> list[PieceResult]:
    """Trace and extrude every piece (no files written)."""
    results: list[PieceResult] = []
    for piece in pieces:
        depth_px = depth_for(piece.longest_side, options.depth_frac)
        trace = trace_piece(piece, options.trace)
        mesh = extrude(trace.polygons, depth_px, center=(piece.w / 2, piece.h / 2), scale=options.scale)
        quality, warnings = _classify(piece, trace, mesh, options)
        results.append(PieceResult(piece=piece, trace=trace, mesh=mesh, depth_px=depth_px, quality=quality, warnings=warnings))
    return results


def _sheet_levels(gray: np.ndarray) -> tuple[float, float]:
    """Grey values of paper and ink on this sheet, for the PNG alpha matte."""
    paper = float(np.percentile(gray, 60))
    ink = float(np.percentile(gray, 1))
    if paper - ink < 60:  # nearly blank image; fall back to sane constants
        paper, ink = 250.0, 40.0
    return paper, ink


def _report_line(r: PieceResult) -> str:
    p = r.piece
    smoothed = " smoothed" if r.trace and r.trace.smoothed else ""
    return (
        f"  {p.id}  {p.w:4d}x{p.h:<4d} at ({p.x},{p.y})  ink={p.ink_area:6d}  "
        f"speckle={r.trace.speckle if r.trace else p.speckle:5.2f}  outers={r.trace.outer_count if r.trace else 0:3d}  "
        f"holes={r.trace.hole_count if r.trace else 0:3d}  tris={r.triangles:6d}  {r.quality}{smoothed}"
    )


def run(
    input_path: str | Path,
    out_dir: str | Path,
    options: PipelineOptions | None = None,
    report: TextIO | None = sys.stderr,
) -> RunResult:
    """Run the whole pipeline on one image and write everything into ``out_dir``."""
    options = options or PipelineOptions()
    input_path = Path(input_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gray = load_gray(input_path)
    sheet_h, sheet_w = gray.shape
    mode, pieces = detect_mode(gray, options)
    if report:
        print(f"{input_path.name}: {sheet_w}x{sheet_h}, mode={mode}, {len(pieces)} piece(s)", file=report)
    if options.debug and mode == "sheet":
        cv2.imwrite(str(out_dir / "segmentation.png"), draw_segmentation(gray, pieces))

    paper, ink = _sheet_levels(gray)
    results = process_pieces(pieces, options)

    scene_entries: list[tuple[str, Mesh, tuple[float, float, float]]] = []
    for r in results:
        p = r.piece
        if report:
            print(_report_line(r), file=report)
        if options.drop_flagged and r.quality != "ok":
            continue
        if options.write_png:
            r.files["png"] = write_cutout_png(p, out_dir / f"{p.id}.png", paper_level=paper, ink_level=ink).name
        if r.trace and options.write_svg:
            r.files["svg"] = write_svg(out_dir / f"{p.id}.svg", r.trace.svg_d, p.w, p.h).name
        if r.mesh is not None and not r.mesh.is_empty and options.write_glb:
            r.files["glb"] = write_glb(r.mesh, out_dir / f"{p.id}.glb", name=p.id).name
            cx = (p.x + p.w / 2) - sheet_w / 2
            cy = (p.y + p.h / 2) - sheet_h / 2
            scene_entries.append((p.id, r.mesh, (cx * options.scale, -cy * options.scale, 0.0)))

    scene_path: Path | None = None
    if scene_entries and options.write_glb:
        scene_path = write_scene_glb(scene_entries, out_dir / "all.glb")

    manifest = {
        "generator": f"flash-to-render {__version__}",
        "source": input_path.name,
        "mode": mode,
        "sheet": {"width": sheet_w, "height": sheet_h},
        "units": "metres" if options.scale == 0.001 else "source px * scale",
        "scale": options.scale,
        "depth_frac": options.depth_frac,
        "scene": scene_path.name if scene_path else None,
        "options": {"segment": asdict(options.segment.resolved(gray.shape)), "trace": asdict(options.trace)},
        "pieces": [r.to_manifest() for r in results],
    }
    manifest_path = write_manifest(out_dir / "manifest.json", manifest)

    if report:
        counts = {q: sum(1 for r in results if r.quality == q) for q in QUALITY_ORDER}
        summary = ", ".join(f"{v} {k}" for k, v in counts.items() if v)
        print(f"  -> {out_dir}  ({summary})", file=report)

    return RunResult(
        source=input_path,
        out_dir=out_dir,
        mode=mode,
        sheet_size=(sheet_w, sheet_h),
        pieces=results,
        manifest_path=manifest_path,
        scene_path=scene_path,
    )
