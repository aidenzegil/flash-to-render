"""Local web app for reviewing segmentation boxes (``flash-to-render serve``).

The library (see :mod:`flash_to_render.library`) lives on disk (``--library``,
default ``~/.flash-to-render/library``). Renders go into ``<entry>/renders/<job>/``
and are served back through the same viewer page the CLI preview uses.

FastAPI is only imported here, so the CLI and the pipeline work without the
``server`` extra installed.
"""

from __future__ import annotations

import io
import re
import threading
import time
import uuid
import zipfile
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from pydantic import BaseModel

from . import __version__
from .library import DEFAULT_LIBRARY, SEEDS, Entry, Library, _with_id
from .pipeline import PipelineOptions, run
from .segment import Box, Region, SegmentOptions, detect, load_gray
from .trace import TraceOptions

__all__ = ["Library", "create_app", "serve", "SEEDS", "DEFAULT_LIBRARY"]


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
    fidelity: str = "fast"
    relief: bool = True
    stamp: str = "tone"


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

    @app.get("/api/library/{entry_id}/regions.json")
    def export_regions(entry_id: str) -> Response:
        """The entry's regions as a downloadable file (same format as ``examples/<id>.regions.json``)."""
        entry = ensure_boxes(entry_or_404(entry_id))
        payload = {"entry": entry.id, "source": entry.filename, "width": entry.width, "height": entry.height, "regions": entry.regions}
        headers = {"Content-Disposition": f'attachment; filename="{entry.id}.regions.json"', "Cache-Control": "no-store"}
        return JSONResponse(payload, headers=headers)

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
            trace=TraceOptions(threshold=body.threshold, smooth=body.smooth, fidelity="best" if body.fidelity == "best" else "fast", relief=body.relief),
            depth_frac=body.depth,
            stamp="binary" if body.stamp == "binary" else "tone",
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
