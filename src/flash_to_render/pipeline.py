"""End-to-end: image file -> per-design PNG / SVG / GLB + manifest + combined scene."""

from __future__ import annotations

import logging
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence, TextIO

import cv2
import numpy as np

from . import __version__
from .export import write_cutout_png, write_glb, write_manifest, write_scene_glb, write_svg
from .mesh import DEFAULT_DEPTH_FRAC, Mesh, depth_for, extrude
from .segment import Box, Piece, Region, SegmentOptions, crop_pieces, detect, draw_segmentation, load_gray
from .trace import TraceOptions, TraceResult, trace_piece

log = logging.getLogger(__name__)

__all__ = ["PipelineOptions", "PieceResult", "RunResult", "run", "process_pieces", "detect_mode", "ProgressFn"]

ProgressFn = Callable[[int, int, str], None]
"""``progress(done, total, piece_id)`` callback, called after every piece is traced and meshed."""

QUALITY_ORDER = ("ok", "tiny", "speckle", "empty")


@dataclass
class PipelineOptions:
    mode: str = "auto"
    """``auto`` decides sheet vs single design, ``sheet`` / ``single`` force it."""
    segment: SegmentOptions = field(default_factory=SegmentOptions)
    trace: TraceOptions = field(default_factory=TraceOptions)
    depth_frac: float = DEFAULT_DEPTH_FRAC
    """Extrusion depth as a fraction of each piece's longest side."""
    fine_depth_frac: float = 0.025
    """Depth fraction used instead for fine pieces (thin features are more than ``fine_thin_fraction`` of the ink)."""
    fine_thin_fraction: float = 0.3
    png_scale: int = 2
    """Resolution of the ink cutout PNGs relative to the source crop."""
    stamp: str = "tone"
    """Ink cutout alpha: ``tone`` from the source grey (shading preserved), ``binary`` from the traced mask."""
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
    png_scale: int = 1
    layer_meshes: list[tuple[str, Mesh]] = field(default_factory=list)
    """Relief layers (name, mesh); empty for a single-layer piece."""

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
            "name": p.name,
            "kind": p.kind,
            "bbox": [p.x, p.y, p.w, p.h],
            "ink_area": p.ink_area,
            "components": t.components if t else p.components,
            "speckle": round(t.speckle if t else p.speckle, 3),
            "raw_speckle": round(p.speckle, 3),
            "grey_ratio": round(p.grey_ratio, 3),
            "smoothed": bool(t.smoothed) if t else False,
            "polarity_flipped": bool(t.flipped_polarity) if t else False,
            "outers": t.outer_count if t else 0,
            "holes": t.hole_count if t else 0,
            "vertices": self.vertices,
            "triangles": self.triangles,
            "depth_px": round(self.depth_px, 3),
            "png_scale": self.png_scale,
            "tone_mode": t.tone_mode if t else "line",
            "dot_spacing": round(t.dot_spacing, 2) if t else 0.0,
            "relief": bool(self.layer_meshes),
            "layers": [{"name": n, "triangles": m.triangle_count, "depth_px": round(m.depth / max(1e-9, self.mesh.depth) * self.depth_px, 3) if self.mesh and self.mesh.depth else 0} for n, m in self.layer_meshes],
            "fidelity": t.fidelity.to_dict() if t and t.fidelity else None,
            "trace_params": t.params if t else {},
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
    """Return ``(mode, pieces)`` using the automatic boxes; segmentation runs at most once."""
    mode, boxes = detect(gray, options.segment, options.mode)
    return mode, crop_pieces(gray, boxes, options.segment)


def _classify(piece: Piece, trace: TraceResult, mesh: Mesh, options: PipelineOptions) -> tuple[str, list[str]]:
    warnings: list[str] = []
    if trace.flipped_polarity:
        warnings.append("polarity flipped by guard")
    if trace.tone_mode == "tone":
        warnings.append(f"halftone: tone-resolved at dot spacing {trace.dot_spacing:.1f}px (raw speckle {piece.speckle:.1f})")
    elif trace.tone_mode == "light":
        warnings.append(f"grey line art: low-C adaptive threshold + gap close (raw speckle {piece.speckle:.1f})")
    if mesh.is_empty or not trace.polygons:
        return "empty", warnings + ["no polygons after tracing"]
    if trace.speckle > options.max_speckle:
        return "speckle", warnings + [f"speckle {trace.speckle:.1f} > {options.max_speckle}"]
    if piece.longest_side < options.min_piece_px:
        return "tiny", warnings + [f"longest side {piece.longest_side}px < {options.min_piece_px}px"]
    return "ok", warnings


def process_pieces(pieces: list[Piece], options: PipelineOptions, progress: ProgressFn | None = None) -> list[PieceResult]:
    """Trace and extrude every piece (no files written)."""
    results: list[PieceResult] = []
    for i, piece in enumerate(pieces):
        trace = trace_piece(piece, options.trace)
        fine = trace.fidelity is not None and trace.fidelity.thin_fraction > options.fine_thin_fraction
        depth_px = depth_for(piece.longest_side, options.fine_depth_frac if fine else options.depth_frac)
        center = (piece.w / 2, piece.h / 2)
        layer_meshes: list[tuple[str, Mesh]] = []
        if trace.layers:
            # relief: every layer shares the back plane at -depth/2
            for layer in trace.layers:
                d = depth_px * layer.depth
                m = extrude(layer.polygons, d, center=center, scale=options.scale, z_offset=-(depth_px - d) / 2)
                if not m.is_empty:
                    layer_meshes.append((layer.name, m))
        if layer_meshes:
            mesh = layer_meshes[0][1]
            for _n, m in layer_meshes[1:]:
                mesh = mesh.concatenate(m)
        else:
            mesh = extrude(trace.polygons, depth_px, center=center, scale=options.scale)
        quality, warnings = _classify(piece, trace, mesh, options)
        results.append(PieceResult(piece=piece, trace=trace, mesh=mesh, depth_px=depth_px, quality=quality, warnings=warnings, layer_meshes=layer_meshes))
        if progress:
            progress(i + 1, len(pieces), piece.id)
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
    fid = f"  fidelity={r.trace.fidelity.score:.3f}" if r.trace and r.trace.fidelity else ""
    return (
        f"  {p.id:<8s}{p.w:4d}x{p.h:<4d} at ({p.x},{p.y})  ink={p.ink_area:6d}  "
        f"speckle={r.trace.speckle if r.trace else p.speckle:5.2f}  outers={r.trace.outer_count if r.trace else 0:3d}  "
        f"holes={r.trace.hole_count if r.trace else 0:3d}  tris={r.triangles:6d}{fid}  {r.quality}{smoothed}"
    )


def run(
    input_path: str | Path,
    out_dir: str | Path,
    options: PipelineOptions | None = None,
    report: TextIO | None = sys.stderr,
    boxes: Sequence[Box | Region] | None = None,
    progress: ProgressFn | None = None,
) -> RunResult:
    """Run the whole pipeline on one image and write everything into ``out_dir``.

    Pass ``boxes`` (rectangles or freehand :class:`Region` polygons, e.g.
    reviewed in the browser) to skip auto-detection and render exactly those;
    the run's mode is then ``"boxes"``.
    """
    options = options or PipelineOptions()
    input_path = Path(input_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gray = load_gray(input_path)
    sheet_h, sheet_w = gray.shape
    if boxes is not None:
        mode, pieces = "boxes", crop_pieces(gray, boxes, options.segment)
    else:
        mode, pieces = detect_mode(gray, options)
    if report:
        print(f"{input_path.name}: {sheet_w}x{sheet_h}, mode={mode}, {len(pieces)} piece(s)", file=report)
    if options.debug and mode != "single":
        cv2.imwrite(str(out_dir / "segmentation.png"), draw_segmentation(gray, pieces))

    paper, ink = _sheet_levels(gray)
    results = process_pieces(pieces, options, progress)

    scene_entries: list[tuple[str, Mesh, tuple[float, float, float]]] = []
    for r in results:
        p = r.piece
        if report:
            print(_report_line(r), file=report)
        if options.drop_flagged and r.quality != "ok":
            continue
        if options.write_png:
            if r.trace is not None:
                r.png_scale = options.png_scale
                r.files["png"] = write_cutout_png(
                    p, out_dir / f"{p.id}.png", mask=r.trace.ink, mask_scale=r.trace.scale, out_scale=options.png_scale,
                    stamp=options.stamp, paper_level=paper, ink_level=ink,
                ).name
            else:
                r.png_scale = 1
                r.files["png"] = write_cutout_png(p, out_dir / f"{p.id}.png", paper_level=paper, ink_level=ink).name
        if r.trace and options.write_svg:
            r.files["svg"] = write_svg(out_dir / f"{p.id}.svg", r.trace.svg_d, p.w, p.h).name
        if r.mesh is not None and not r.mesh.is_empty and options.write_glb:
            r.files["glb"] = write_glb(r.layer_meshes or r.mesh, out_dir / f"{p.id}.glb", name=p.id).name
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
        "fidelity_mode": options.trace.fidelity,
        "stamp": options.stamp,
        "relief": options.trace.relief,
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
