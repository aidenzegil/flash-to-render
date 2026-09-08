"""Local web app for reviewing segmentation boxes (``flash-to-render serve``).

The library lives on disk (``--library``, default ``~/.flash-to-render/library``):
one folder per image with the original file and ``meta.json`` (name, size,
current regions - rectangles or freehand polygons - and detection options). Renders go into ``<entry>/renders/<job>/``
and are served back through the same viewer page the CLI preview uses.

FastAPI is only imported here, so the CLI and the pipeline work without the
``server`` extra installed.
"""

from __future__ import annotations

import io
import json
import re
import shutil
import threading
import time
import uuid
import zipfile
from dataclasses import asdict, dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from PIL import Image
from pydantic import BaseModel

from . import __version__
from .pipeline import PipelineOptions, run
from .segment import Box, Region, SegmentOptions, detect, load_gray
from .trace import TraceOptions

__all__ = ["Library", "create_app", "serve", "SEEDS", "DEFAULT_LIBRARY"]

DEFAULT_LIBRARY = Path.home() / ".flash-to-render" / "library"
EXAMPLES = Path(__file__).resolve().parent.parent.parent / "examples"
SEEDS = (
    ("payday-spider-verse", "Payday · Spider-Verse", "spiderverse.webp"),
    ("payday-anime", "Payday · Anime", "anime.webp"),
    ("payday-objects", "Payday · Objects", "objects.webp"),
)
ALLOWED = {".png", ".jpg", ".jpeg", ".webp"}


# --------------------------------------------------------------------------- #
# library on disk
# --------------------------------------------------------------------------- #


@dataclass
class Entry:
    id: str
    name: str
    filename: str
    width: int
    height: int
    size: int
    added: float
    regions: list[dict] | None = None
    """``[{"id", "name", "kind": "rect" | "polygon", "points": [[x, y], ...]}, ...]`` in image pixels."""
    mode: str | None = None
    options: dict = field(default_factory=dict)

    @property
    def boxes(self) -> list[dict] | None:
        """Bounding boxes of the regions (legacy rect-only view)."""
        if self.regions is None:
            return None
        return [Region.from_dict(r).bbox().to_dict() for r in self.regions]

    def summary(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "filename": self.filename,
            "width": self.width,
            "height": self.height,
            "size": self.size,
            "added": self.added,
            "box_count": len(self.regions) if self.regions is not None else None,
            "mode": self.mode,
        }


class Library:
    """Folder of images + sidecar JSON. Thread-safe enough for one local user."""

    def __init__(self, root: Path, examples: Path = EXAMPLES, seed: bool = True):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        if seed:
            self.seed(examples)

    # -- paths
    def dir(self, entry_id: str) -> Path:
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", entry_id):
            raise KeyError(entry_id)
        return self.root / entry_id

    def meta_path(self, entry_id: str) -> Path:
        return self.dir(entry_id) / "meta.json"

    def image_path(self, entry: Entry) -> Path:
        return self.dir(entry.id) / entry.filename

    # -- crud
    def seed(self, examples: Path) -> None:
        for entry_id, name, filename in SEEDS:
            src = examples / filename
            if src.exists() and not self.meta_path(entry_id).exists():
                self.add(src.read_bytes(), filename, name=name, entry_id=entry_id)

    def list(self) -> list[Entry]:
        entries = []
        for meta in sorted(self.root.glob("*/meta.json")):
            try:
                entries.append(self._load(meta))
            except (OSError, ValueError, KeyError):
                continue
        entries.sort(key=lambda e: e.added)
        return entries

    def get(self, entry_id: str) -> Entry:
        try:
            return self._load(self.meta_path(entry_id))
        except (OSError, KeyError, ValueError):
            raise KeyError(entry_id) from None

    def add(self, data: bytes, filename: str, name: str = "", entry_id: str | None = None) -> Entry:
        ext = Path(filename).suffix.lower()
        if ext not in ALLOWED:
            raise ValueError(f"unsupported image type {ext or '(none)'}; use PNG, JPG or WebP")
        try:
            with Image.open(io.BytesIO(data)) as im:
                width, height = im.size
        except Exception as exc:  # pragma: no cover - PIL error text varies
            raise ValueError(f"not an image: {exc}") from exc
        entry_id = entry_id or uuid.uuid4().hex[:10]
        with self._lock:
            d = self.dir(entry_id)
            d.mkdir(parents=True, exist_ok=True)
            stored = f"image{ext}"
            (d / stored).write_bytes(data)
            entry = Entry(
                id=entry_id,
                name=name.strip() or Path(filename).stem,
                filename=stored,
                width=width,
                height=height,
                size=len(data),
                added=time.time(),
            )
            self._save(entry)
        return entry

    def save(self, entry: Entry) -> None:
        with self._lock:
            self._save(entry)

    def delete(self, entry_id: str) -> None:
        with self._lock:
            shutil.rmtree(self.dir(entry_id), ignore_errors=True)

    def _save(self, entry: Entry) -> None:
        self.meta_path(entry.id).write_text(json.dumps(asdict(entry), indent=2))

    def _load(self, meta: Path) -> Entry:
        d = json.loads(meta.read_text())
        if "regions" not in d and d.get("boxes") is not None:  # sidecars written before polygons existed
            d["regions"] = [_with_id(Region.from_box(Box.from_dict(b))).to_dict() for b in d["boxes"]]
        return Entry(**{k: d.get(k) for k in Entry.__dataclass_fields__ if k in d})


# --------------------------------------------------------------------------- #
# request models
# --------------------------------------------------------------------------- #


class BoxIn(BaseModel):
    x: float
    y: float
    w: float
    h: float
    name: str = ""


class DetectIn(BaseModel):
    merge_kernel: int | None = None
    min_area: int | None = None
    smooth: str = "auto"
    threshold: int = 150
    mode: str = "auto"


class RegionIn(BaseModel):
    points: list[list[float]]
    kind: str = "polygon"
    name: str = ""
    id: str = ""


class BoxesIn(BaseModel):
    regions: list[RegionIn] | None = None
    boxes: list[BoxIn] | None = None
    """Legacy rect-only payload; ``regions`` wins when both are given."""
    options: dict[str, Any] = {}


class RenderIn(BaseModel):
    regions: list[RegionIn] | None = None
    boxes: list[BoxIn] | None = None
    merge_kernel: int | None = None
    min_area: int | None = None
    smooth: str = "auto"
    threshold: int = 150
    depth: float = 0.035


def _segment_options(d: DetectIn | RenderIn) -> SegmentOptions:
    return SegmentOptions(threshold=d.threshold, merge_kernel=d.merge_kernel, min_ink_area=d.min_area)


def _regions_in(body: BoxesIn | RenderIn, width: int, height: int) -> list[Region] | None:
    """Regions from a request body (``regions`` or legacy ``boxes``), clamped to the image; ``None`` if absent."""
    if body.regions is not None:
        raw = [Region.from_dict(r.model_dump()) for r in body.regions]
    elif body.boxes is not None:
        raw = [Region.from_box(Box.from_dict(b.model_dump())) for b in body.boxes]
    else:
        return None
    out: list[Region] = []
    for i, r in enumerate(raw):
        if len(r.points) < 3:
            raise HTTPException(400, f"region {i} needs at least 3 points")
        pts = [(min(max(0.0, x), float(width)), min(max(0.0, y), float(height))) for x, y in r.points]
        if r.kind == "rect":
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            pts = [(min(xs), min(ys)), (max(xs), min(ys)), (max(xs), max(ys)), (min(xs), max(ys))]
        out.append(Region(pts, r.kind if r.kind in ("rect", "polygon") else "polygon", r.name.strip(), r.id or uuid.uuid4().hex[:6]))
    return out


def _regions_out(entry: Entry) -> dict[str, Any]:
    return {"regions": entry.regions, "boxes": entry.boxes, "mode": entry.mode, "options": entry.options, "width": entry.width, "height": entry.height}


# --------------------------------------------------------------------------- #
# jobs
# --------------------------------------------------------------------------- #


@dataclass
class Job:
    id: str
    entry_id: str
    out_dir: Path
    status: str = "queued"
    done: int = 0
    total: int = 0
    current: str = ""
    error: str | None = None
    pieces: list[dict] = field(default_factory=list)
    started: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "entry_id": self.entry_id,
            "status": self.status,
            "progress": {"done": self.done, "total": self.total, "current": self.current},
            "error": self.error,
            "pieces": self.pieces,
            "preview": f"/api/jobs/{self.id}/preview/",
            "download": f"/api/jobs/{self.id}/download.zip",
        }


# --------------------------------------------------------------------------- #
# app
# --------------------------------------------------------------------------- #


def _with_id(region: Region) -> Region:
    region.id = region.id or uuid.uuid4().hex[:6]
    return region


def _page(name: str) -> bytes:
    return resources.files(__package__).joinpath(name).read_bytes()


def create_app(library_dir: Path | str = DEFAULT_LIBRARY, seed: bool = True) -> FastAPI:
    library = Library(Path(library_dir), seed=seed)
    jobs: dict[str, Job] = {}
    app = FastAPI(title="flash-to-render", version=__version__)
    app.state.library = library
    app.state.jobs = jobs

    def entry_or_404(entry_id: str) -> Entry:
        try:
            return library.get(entry_id)
        except KeyError:
            raise HTTPException(404, f"no library entry {entry_id!r}") from None

    def job_or_404(job_id: str) -> Job:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, f"no job {job_id!r}")
        return job

    def ensure_boxes(entry: Entry, opts: DetectIn | None = None) -> Entry:
        """Run detection for an entry that has never been detected (or on demand) and persist it."""
        if entry.regions is not None and opts is None:
            return entry
        opts = opts or DetectIn()
        gray = load_gray(library.image_path(entry))
        mode, boxes = detect(gray, _segment_options(opts), opts.mode)
        entry.regions = [_with_id(Region.from_box(b)).to_dict() for b in boxes]
        entry.mode = mode
        entry.options = opts.model_dump()
        library.save(entry)
        return entry

    # -- pages
    @app.get("/", response_class=HTMLResponse)
    def index() -> Response:
        return HTMLResponse(_page("app.html"), headers={"Cache-Control": "no-store"})

    # -- library
    @app.get("/api/library")
    def list_library() -> list[dict]:
        return [e.summary() for e in library.list()]

    @app.post("/api/library/upload")
    async def upload(file: UploadFile = File(...), name: str = Form("")) -> dict:
        data = await file.read()
        try:
            entry = library.add(data, file.filename or "upload.png", name=name)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        entry = ensure_boxes(entry)
        return entry.summary()

    @app.get("/api/library/{entry_id}")
    def get_entry(entry_id: str) -> dict:
        return entry_or_404(entry_id).summary()

    @app.delete("/api/library/{entry_id}")
    def delete_entry(entry_id: str) -> dict:
        entry_or_404(entry_id)
        library.delete(entry_id)
        return {"ok": True}

    @app.get("/api/library/{entry_id}/image")
    def get_image(entry_id: str) -> FileResponse:
        entry = entry_or_404(entry_id)
        return FileResponse(library.image_path(entry), headers={"Cache-Control": "no-store"})

    @app.get("/api/library/{entry_id}/boxes")
    def get_boxes(entry_id: str) -> dict:
        """Current regions (rectangles and polygons) plus their bounding boxes."""
        return _regions_out(ensure_boxes(entry_or_404(entry_id)))

    @app.put("/api/library/{entry_id}/boxes")
    def put_boxes(entry_id: str, body: BoxesIn) -> dict:
        entry = entry_or_404(entry_id)
        regions = _regions_in(body, entry.width, entry.height)
        if regions is None:
            raise HTTPException(400, "send regions (or boxes)")
        entry.regions = [r.to_dict() for r in regions]
        entry.mode = "boxes"
        if body.options:
            entry.options = body.options
        library.save(entry)
        return _regions_out(entry)

    @app.post("/api/library/{entry_id}/detect")
    def redetect(entry_id: str, body: DetectIn | None = None) -> dict:
        """Re-run auto-segmentation with the given options. Returns the boxes without saving them."""
        entry = entry_or_404(entry_id)
        opts = body or DetectIn()
        gray = load_gray(library.image_path(entry))
        mode, boxes = detect(gray, _segment_options(opts), opts.mode)
        regions = [_with_id(Region.from_box(b)) for b in boxes]
        return {"regions": [r.to_dict() for r in regions], "boxes": [b.to_dict() for b in boxes], "mode": mode, "options": opts.model_dump()}

    # -- render jobs
    @app.post("/api/library/{entry_id}/render")
    def render(entry_id: str, body: RenderIn | None = None) -> dict:
        entry = entry_or_404(entry_id)
        body = body or RenderIn()
        regions = _regions_in(body, entry.width, entry.height)
        if regions is not None:
            entry.regions = [r.to_dict() for r in regions]
            entry.mode = "boxes"
            library.save(entry)
        else:
            entry = ensure_boxes(entry)
            regions = [Region.from_dict(r) for r in entry.regions or []]
        if not regions:
            raise HTTPException(400, "no regions to render")

        job_id = uuid.uuid4().hex[:10]
        out_dir = library.dir(entry.id) / "renders" / job_id
        job = Job(id=job_id, entry_id=entry.id, out_dir=out_dir, total=len(regions))
        jobs[job_id] = job
        options = PipelineOptions(
            segment=_segment_options(body),
            trace=TraceOptions(threshold=body.threshold, smooth=body.smooth),
            depth_frac=body.depth,
            debug=True,
        )

        def progress(done: int, total: int, current: str) -> None:
            job.done, job.total, job.current = done, total, current

        def work() -> None:
            job.status = "running"
            try:
                result = run(library.image_path(entry), out_dir, options, report=None, boxes=regions, progress=progress)
                job.pieces = [r.to_manifest() for r in result.pieces]
                job.status = "done"
            except Exception as exc:  # surface to the UI instead of dying silently
                job.error = f"{type(exc).__name__}: {exc}"
                job.status = "failed"

        threading.Thread(target=work, name=f"render-{job_id}", daemon=True).start()
        return {"job_id": job_id, **job.to_dict()}

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> dict:
        return job_or_404(job_id).to_dict()

    @app.get("/api/jobs/{job_id}/preview.glb")
    def job_scene(job_id: str) -> FileResponse:
        job = job_or_404(job_id)
        path = job.out_dir / "all.glb"
        if job.status != "done" or not path.exists():
            raise HTTPException(409, f"job is {job.status}")
        return FileResponse(path, media_type="model/gltf-binary")

    @app.get("/api/jobs/{job_id}/download.zip")
    def job_zip(job_id: str) -> Response:
        job = job_or_404(job_id)
        if job.status != "done":
            raise HTTPException(409, f"job is {job.status}")
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in sorted(job.out_dir.iterdir()):
                if f.is_file():
                    zf.write(f, f.name)
        entry_name = re.sub(r"[^a-z0-9]+", "-", library.get(job.entry_id).name.lower()).strip("-") or "render"
        headers = {"Content-Disposition": f'attachment; filename="{entry_name}-{job.id}.zip"'}
        return Response(buf.getvalue(), media_type="application/zip", headers=headers)

    @app.get("/api/jobs/{job_id}/preview/", response_class=HTMLResponse)
    def job_preview(job_id: str) -> Response:
        job_or_404(job_id)
        return HTMLResponse(_page("preview.html"), headers={"Cache-Control": "no-store"})

    @app.get("/api/jobs/{job_id}/preview/{filename}")
    def job_file(job_id: str, filename: str) -> FileResponse:
        job = job_or_404(job_id)
        if "/" in filename or filename.startswith("."):
            raise HTTPException(404)
        path = job.out_dir / filename
        if not path.is_file():
            raise HTTPException(404, filename)
        media = "model/gltf-binary" if path.suffix == ".glb" else None
        return FileResponse(path, media_type=media, headers={"Cache-Control": "no-store"})

    @app.get("/api/version")
    def version() -> dict:
        return {"version": __version__}

    @app.exception_handler(HTTPException)
    async def http_error(_request, exc: HTTPException) -> JSONResponse:
        return JSONResponse({"error": exc.detail}, status_code=exc.status_code)

    return app


def serve(library_dir: Path | str = DEFAULT_LIBRARY, port: int = 8766, open_browser: bool = True) -> int:
    import webbrowser

    import uvicorn

    app = create_app(library_dir)
    url = f"http://127.0.0.1:{port}/"
    print(f"flash-to-render serve: {url}  (library: {Path(library_dir)}; ctrl-c to stop)")
    if open_browser:
        threading.Timer(0.8, webbrowser.open, args=(url,)).start()
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
    return 0
