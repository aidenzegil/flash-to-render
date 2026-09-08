"""The on-disk library used by the web app and the ``regions`` CLI (no FastAPI here).

One folder per image: the original file plus ``meta.json`` with the name, size,
current regions (rectangles or freehand polygons) and detection options.
"""

from __future__ import annotations

import io
import json
import re
import shutil
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from .segment import Box, Region

__all__ = ["Entry", "Library", "SEEDS", "DEFAULT_LIBRARY", "EXAMPLES", "read_regions_file", "write_regions_file"]

DEFAULT_LIBRARY = Path.home() / ".flash-to-render" / "library"
EXAMPLES = Path(__file__).resolve().parent.parent.parent / "examples"
SEEDS = (
    ("payday-spider-verse", "Payday · Spider-Verse", "spiderverse.webp"),
    ("payday-anime", "Payday · Anime", "anime.webp"),
    ("payday-objects", "Payday · Objects", "objects.webp"),
)
ALLOWED = {".png", ".jpg", ".jpeg", ".webp"}


def _with_id(region: Region) -> Region:
    region.id = region.id or uuid.uuid4().hex[:6]
    return region


def read_regions_file(path: Path) -> list[dict]:
    """Read a regions JSON file (``{"regions": [...]}`` or a bare list; legacy rect dicts accepted)."""
    data = json.loads(Path(path).read_text())
    raw = data["regions"] if isinstance(data, dict) else data
    return [_with_id(Region.from_dict(r)).to_dict() for r in raw]


def write_regions_file(path: Path, entry: "Entry") -> None:
    payload = {
        "entry": entry.id,
        "source": entry.filename,
        "width": entry.width,
        "height": entry.height,
        "regions": entry.regions or [],
    }
    Path(path).write_text(json.dumps(payload, indent=1) + "\n")


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
    def seed(self, examples: Path) -> list[Entry]:
        """Create the example entries that do not exist yet. Strictly idempotent: an existing entry
        (and its meta.json) is never touched. ``examples/<id>.regions.json`` supplies the initial regions."""
        created: list[Entry] = []
        for entry_id, name, filename in SEEDS:
            src = examples / filename
            if not src.exists() or self.meta_path(entry_id).exists():
                continue
            entry = self.add(src.read_bytes(), filename, name=name, entry_id=entry_id)
            regions_file = examples / f"{entry_id}.regions.json"
            if regions_file.exists():
                entry.regions = read_regions_file(regions_file)
                entry.mode = "boxes"
                self.save(entry)
            created.append(entry)
        return created

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


