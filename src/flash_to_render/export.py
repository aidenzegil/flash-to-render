"""Writers for the per-design outputs: PNG cutout, SVG, GLB and the combined scene."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import trimesh
from PIL import Image

from .mesh import Mesh
from .segment import Piece

__all__ = [
    "cutout_rgba",
    "cutout_rgba_from_mask",
    "cutout_rgba_tone",
    "write_cutout_png",
    "write_svg",
    "mesh_to_trimesh",
    "write_glb",
    "write_scene_glb",
    "write_manifest",
]

INK_COLOR = (28, 28, 30, 255)
"""Default glTF base colour for the extruded ink (near-black)."""


def cutout_rgba(piece: Piece, paper_level: float = 250.0, ink_level: float = 40.0) -> np.ndarray:
    """Alpha-matte the design: RGB is pure white, alpha is how inky each pixel is.

    Keeping RGB white lets a renderer tint the cutout by multiplying the colour.
    ``paper_level`` / ``ink_level`` are the grey values that map to alpha 0 / 255;
    use the sheet's own statistics for scans with a grey background.
    """
    gray = piece.gray.astype(np.float32)
    span = max(1.0, float(paper_level - ink_level))
    alpha = np.clip((paper_level - gray) / span, 0.0, 1.0)
    alpha[~piece.region] = 0.0
    rgba = np.empty((*gray.shape, 4), dtype=np.uint8)
    rgba[..., :3] = 255
    rgba[..., 3] = np.round(alpha * 255).astype(np.uint8)
    return rgba


def cutout_rgba_from_mask(mask: np.ndarray, mask_scale: int, out_scale: int = 2) -> np.ndarray:
    """Alpha-matte from the cleaned ink mask (``mask_scale`` x source), resampled to ``out_scale`` x source.

    Area resampling from the higher-resolution mask gives crisp, antialiased
    edges instead of the blur of a 1x threshold; RGB stays white so it tints.
    """
    h, w = mask.shape
    ow, oh = round(w * out_scale / mask_scale), round(h * out_scale / mask_scale)
    interp = cv2.INTER_AREA if out_scale <= mask_scale else cv2.INTER_CUBIC
    alpha = cv2.resize(mask, (max(1, ow), max(1, oh)), interpolation=interp)
    rgba = np.empty((*alpha.shape, 4), dtype=np.uint8)
    rgba[..., :3] = 255
    rgba[..., 3] = alpha
    return rgba


def cutout_rgba_tone(piece: Piece, out_scale: int = 2, paper_level: float = 250.0, ink_level: float = 40.0, unsharp: float = 0.5) -> np.ndarray:
    """Alpha from the source grey itself, at ``out_scale`` x: shading prints exactly as drawn.

    The crop is upscaled (bicubic), lightly unsharpened, mapped to alpha with
    ``clamp((paper - grey) / (paper - ink))`` and masked to the region outline.
    For crisp line art this is visually the binary cutout; for halftone it keeps the tone.
    """
    gray = piece.gray
    region = piece.region
    if out_scale > 1:
        gray = cv2.resize(gray, None, fx=out_scale, fy=out_scale, interpolation=cv2.INTER_CUBIC)
        region = cv2.resize(region.astype(np.uint8), None, fx=out_scale, fy=out_scale, interpolation=cv2.INTER_NEAREST) > 0
    if unsharp > 0:
        blur = cv2.GaussianBlur(gray, (0, 0), max(0.8, 0.5 * out_scale))
        gray = cv2.addWeighted(gray, 1.0 + unsharp, blur, -unsharp, 0)
    span = max(1.0, float(paper_level - ink_level))
    alpha = np.clip((paper_level - gray.astype(np.float32)) / span, 0.0, 1.0)
    alpha[~region] = 0.0
    rgba = np.empty((*alpha.shape, 4), dtype=np.uint8)
    rgba[..., :3] = 255
    rgba[..., 3] = np.round(alpha * 255).astype(np.uint8)
    return rgba


def write_cutout_png(
    piece: Piece,
    path: str | Path,
    mask: np.ndarray | None = None,
    mask_scale: int = 1,
    out_scale: int = 2,
    stamp: str = "tone",
    **levels: float,
) -> Path:
    """Write the ink cutout.

    ``stamp="tone"`` (default) takes alpha from the source grey (:func:`cutout_rgba_tone`);
    ``stamp="binary"`` uses the cleaned hi-res ``mask`` when given, else the raw crop.
    """
    path = Path(path)
    if stamp == "tone":
        rgba = cutout_rgba_tone(piece, out_scale, **levels)
    elif mask is not None:
        rgba = cutout_rgba_from_mask(mask, mask_scale, out_scale)
    else:
        rgba = cutout_rgba(piece, **levels)
    Image.fromarray(rgba, "RGBA").save(path, optimize=True)
    return path


def write_svg(path: str | Path, svg_d: str, width: int, height: int, fill: str = "#000") -> Path:
    """Write a single-path SVG (even-odd fill so holes stay holes)."""
    path = Path(path)
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
        f'<path d="{svg_d}" fill="{fill}" fill-rule="evenodd"/></svg>\n'
    )
    path.write_text(svg)
    return path


def mesh_to_trimesh(mesh: Mesh, color: Sequence[int] = INK_COLOR) -> trimesh.Trimesh:
    """Wrap a :class:`Mesh` in a ``trimesh.Trimesh`` with a flat PBR material (no re-processing)."""
    tm = trimesh.Trimesh(
        vertices=mesh.vertices.astype(np.float32),
        faces=mesh.faces.astype(np.int64),
        vertex_normals=mesh.normals.astype(np.float32),
        process=False,
    )
    material = trimesh.visual.material.PBRMaterial(
        baseColorFactor=[c / 255.0 for c in color],
        metallicFactor=0.0,
        roughnessFactor=0.65,
        doubleSided=False,
    )
    tm.visual = trimesh.visual.TextureVisuals(material=material)
    return tm


def write_glb(
    mesh: Mesh | Sequence[tuple[str, Mesh]],
    path: str | Path,
    name: str | None = None,
    color: Sequence[int] = INK_COLOR,
) -> Path:
    """Write one mesh, or several named layers (relief), as one GLB with a node per layer."""
    path = Path(path)
    base = name or path.stem
    layers = [(base, mesh)] if isinstance(mesh, Mesh) else [(f"{base}-{n}", m) for n, m in mesh]
    scene = trimesh.Scene()
    for node, m in layers:
        if m.is_empty:
            continue
        scene.add_geometry(mesh_to_trimesh(m, color), node_name=node, geom_name=node)
    path.write_bytes(scene.export(file_type="glb"))
    return path


def write_scene_glb(
    entries: Sequence[tuple[str, Mesh, tuple[float, float, float]]],
    path: str | Path,
    color: Sequence[int] = INK_COLOR,
) -> Path:
    """Write every ``(name, mesh, translation)`` as a node of one glTF scene."""
    path = Path(path)
    scene = trimesh.Scene()
    for name, mesh, (tx, ty, tz) in entries:
        if mesh.is_empty:
            continue
        transform = np.eye(4)
        transform[:3, 3] = (tx, ty, tz)
        scene.add_geometry(mesh_to_trimesh(mesh, color), node_name=name, geom_name=name, transform=transform)
    path.write_bytes(scene.export(file_type="glb"))
    return path


def write_manifest(path: str | Path, manifest: dict[str, Any]) -> Path:
    path = Path(path)
    path.write_text(json.dumps(manifest, indent=2, default=_json_default) + "\n")
    return path


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"not JSON serialisable: {type(obj)!r}")
