"""Piece names: unique, descriptive fallbacks, and an optional Claude captioner.

Three guarantees, in order of importance:

1. **Uniqueness.** :func:`unique_names` makes any list of names unique
   deterministically (``"Kitsune Mask"``, ``"Kitsune Mask II"``, ``"Kitsune Mask III"`` ...).
   Every entry point that produces names runs through it: region PUT, detection,
   the labeler, the manifest writer and the editor's rename.
2. **A stable fallback** for unnamed pieces: ``"<Sheet> No. <n>"`` with ``n`` the
   1-based reading-order number (``"Anime No. 12"``), see :func:`fallback_names`.
3. **A pluggable labeler** (:func:`get_labeler`): ``caption`` sends each piece's
   stamp (downscaled, over white) to Claude and asks for a short Title Case name;
   ``none`` uses the fallback only. Nothing is ever sent unless
   ``ANTHROPIC_API_KEY`` is set *and* the labeler is invoked. Human-typed names
   are never overwritten unless ``force`` is given.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

import numpy as np
from PIL import Image

from .segment import Box, Piece, Region, piece_id, reading_positions

log = logging.getLogger(__name__)

__all__ = [
    "unique_names",
    "roman",
    "sheet_token",
    "grid_positions",
    "fallback_names",
    "clean_name",
    "Labeler",
    "NoneLabeler",
    "CaptionLabeler",
    "get_labeler",
    "stamp_bytes",
    "LabelReport",
    "label_pieces",
    "label_entry",
    "apply_names",
    "MODEL",
    "PROMPT",
]

MODEL = "claude-opus-5"
BATCH = 10
"""Pieces per request: several images in one message, one JSON array back."""

PROMPT = (
    "These are {n} tattoo flash designs, in order. Name each one in 2-4 words, Title Case, "
    "no trailing punctuation. Name the subject, not the style (\"Kitsune Mask\", not \"Bold Linework\"). "
    "If a design contains legible text, use that text as its name. "
    "Reply with only a JSON array of exactly {n} strings in the same order, nothing else."
)


# --------------------------------------------------------------------------- #
# uniqueness
# --------------------------------------------------------------------------- #


def roman(n: int) -> str:
    """1 -> I, 2 -> II ... (used for the disambiguating suffix)."""
    out, rest = "", n
    for value, sym in ((1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"), (50, "L"), (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")):
        while rest >= value:
            out += sym
            rest -= value
    return out


def unique_names(names: Sequence[str], reserved: Sequence[str] = ()) -> list[str]:
    """Return ``names`` with duplicates disambiguated deterministically; empty names pass through.

    The first occurrence keeps the name, later ones get ``" II"``, ``" III"`` ...
    (case-insensitive, whitespace-normalised). ``reserved`` names count as
    already taken. A name that already carries a numeral suffix is treated as
    a whole, so ``["Mask II", "Mask II"]`` becomes ``["Mask II", "Mask II II"]`` -
    never a collision.
    """
    taken = {_key(r) for r in reserved if r}
    out: list[str] = []
    for raw in names:
        name = " ".join((raw or "").split())
        if not name:
            out.append("")
            continue
        candidate, n = name, 1
        while _key(candidate) in taken:
            n += 1
            candidate = f"{name} {roman(n)}"
        taken.add(_key(candidate))
        out.append(candidate)
    return out


def _key(name: str) -> str:
    return " ".join(name.split()).casefold()


# --------------------------------------------------------------------------- #
# fallback names
# --------------------------------------------------------------------------- #


def sheet_token(name: str) -> str:
    """A short label for a sheet: the part after the last "·" / " - " separator, trimmed to a few words."""
    part = re.split(r"\s+[·\-–—:]\s+", name.strip())[-1] if name else ""
    part = re.sub(r"\.[A-Za-z0-9]{2,5}$", "", part)  # file extension
    words = [w for w in re.split(r"[\s_]+", part) if w][:3]
    return " ".join(w[:1].upper() + w[1:] for w in words) or "Sheet"


def grid_positions(boxes: Sequence[Box]) -> list[tuple[int, int]]:
    """``(row, col)`` (1-based) for each box in reading order (see :func:`segment.reading_positions`)."""
    return [(r, c) for _n, r, c in reading_positions(boxes)]


def fallback_names(boxes: Sequence[Box], sheet: str) -> list[str]:
    """``"<Sheet> No. <n>"`` with ``n`` the 1-based reading-order number (unique by construction)."""
    token = sheet_token(sheet)
    return unique_names([f"{token} No. {n}" for n, _r, _c in reading_positions(boxes)])


def clean_name(text: str) -> str:
    """Normalise a model-produced name: whitespace, quotes, trailing punctuation, at most 6 words."""
    s = " ".join(str(text).split()).strip().strip("\"'`")
    s = re.sub(r"[.,;:!?…]+$", "", s).strip()
    return " ".join(s.split()[:6])


# --------------------------------------------------------------------------- #
# labelers
# --------------------------------------------------------------------------- #


class Labeler(Protocol):
    name: str

    def label(self, stamps: Sequence[bytes]) -> list[str | None]:
        """One name per stamp (PNG bytes), ``None`` where the backend has nothing to offer."""


class NoneLabeler:
    name = "none"

    def label(self, stamps: Sequence[bytes]) -> list[str | None]:
        return [None] * len(stamps)


@dataclass
class CaptionLabeler:
    """Names pieces with Claude: batched image messages, JSON array back, one retry on malformed output."""

    client: Any = None
    """An ``anthropic.Anthropic``-compatible client; created lazily from the environment when omitted."""
    model: str = MODEL
    batch: int = BATCH
    requests: int = 0
    failures: int = 0
    name: str = field(default="caption", init=False)

    def _client(self) -> Any:
        if self.client is None:
            import anthropic  # optional extra: pip install "flash-to-render[label]"

            self.client = anthropic.Anthropic()
        return self.client

    def label(self, stamps: Sequence[bytes]) -> list[str | None]:
        out: list[str | None] = []
        for start in range(0, len(stamps), self.batch):
            chunk = stamps[start : start + self.batch]
            out.extend(self._label_batch(chunk))
        return out

    def _label_batch(self, stamps: Sequence[bytes]) -> list[str | None]:
        n = len(stamps)
        content: list[dict[str, Any]] = [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": base64.standard_b64encode(png).decode("ascii")}}
            for png in stamps
        ]
        content.append({"type": "text", "text": PROMPT.format(n=n)})
        messages: list[dict[str, Any]] = [{"role": "user", "content": content}]
        for attempt in range(2):
            try:
                self.requests += 1
                response = self._client().messages.create(
                    model=self.model,
                    max_tokens=1024,
                    output_config={"effort": "low"},
                    messages=messages,
                )
            except Exception as exc:  # network, auth, rate limit: the caller falls back
                self.failures += 1
                log.warning("label: request failed (%s); using fallback names for this batch", exc)
                return [None] * n
            text = "".join(getattr(b, "text", "") for b in getattr(response, "content", []) if getattr(b, "type", "") == "text")
            names = _parse_names(text, n)
            if names is not None:
                return names
            log.warning("label: malformed reply (attempt %d): %.120r", attempt + 1, text)
            messages = messages + [
                {"role": "assistant", "content": text or "(empty)"},
                {"role": "user", "content": f"That was not a JSON array of exactly {n} strings. Reply with only the JSON array."},
            ]
        self.failures += 1
        return [None] * n


def _parse_names(text: str, n: int) -> list[str] | None:
    m = re.search(r"\[.*\]", text or "", re.S)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, list) or len(data) != n or not all(isinstance(x, str) for x in data):
        return None
    names = [clean_name(x) for x in data]
    return [x if x else None for x in names]  # type: ignore[misc]


def get_labeler(backend: str = "auto", client: Any = None) -> tuple[Labeler, str | None]:
    """``(labeler, reason)``: ``reason`` says why the caption backend was not used, if it was not.

    ``auto`` picks ``caption`` when ``ANTHROPIC_API_KEY`` is set (or a client is
    injected) and ``none`` otherwise. Nothing is sent to the API unless the
    caption labeler is actually invoked.
    """
    if backend not in ("auto", "caption", "none"):
        raise ValueError(f"unknown label backend {backend!r}")
    have_key = bool(os.environ.get("ANTHROPIC_API_KEY")) or client is not None
    if backend == "none":
        return NoneLabeler(), None
    if not have_key:
        reason = "ANTHROPIC_API_KEY is not set; using descriptive fallback names (nothing was sent to the API)"
        if backend == "caption":
            log.warning("label: %s", reason)
        return NoneLabeler(), reason
    return CaptionLabeler(client=client), None


# --------------------------------------------------------------------------- #
# stamps + application
# --------------------------------------------------------------------------- #


def stamp_bytes(piece: Piece, max_px: int = 512) -> bytes:
    """The piece's tone stamp composited over white, downscaled to ``max_px`` on the long side, as PNG bytes."""
    from .export import cutout_rgba_tone

    rgba = cutout_rgba_tone(piece, out_scale=1)
    alpha = rgba[..., 3:4].astype(np.float32) / 255.0
    rgb = (255 * (1 - alpha)).astype(np.uint8).repeat(3, axis=2)  # white ink over white paper = black ink
    img = Image.fromarray(rgb, "RGB")
    if max(img.size) > max_px:
        s = max_px / max(img.size)
        img = img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


@dataclass
class LabelReport:
    backend: str
    named: int
    kept: int
    fallback: int
    requests: int = 0
    failures: int = 0
    reason: str | None = None
    names: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"backend": self.backend, "named": self.named, "kept": self.kept, "fallback": self.fallback, "requests": self.requests, "failures": self.failures, "reason": self.reason, "names": self.names}


def label_pieces(pieces: Sequence[Piece], sheet: str, backend: str = "auto", force: bool = False, client: Any = None) -> LabelReport:
    """Give every piece a unique name: keep human names (unless ``force``), ask the labeler for the rest, fall back to the grid name.

    Mutates ``piece.name`` / ``piece.id`` in place.
    """
    labeler, reason = get_labeler(backend, client)
    boxes = [Box(p.x, p.y, p.w, p.h) for p in pieces]
    fallbacks = fallback_names(boxes, sheet)
    todo = [i for i, p in enumerate(pieces) if force or not p.name.strip()]
    suggestions: dict[int, str] = {}
    if todo and not isinstance(labeler, NoneLabeler):
        got = labeler.label([stamp_bytes(pieces[i]) for i in todo])
        suggestions = {i: n for i, n in zip(todo, got) if n}
    names: list[str] = []
    fallback_count = 0
    for i, p in enumerate(pieces):
        if i in suggestions:
            names.append(suggestions[i])
        elif i in todo:
            names.append(fallbacks[i])
            fallback_count += 1
        else:
            names.append(p.name.strip())
    names = unique_names(names)
    fallback_idx = {i for i in todo if i not in suggestions}
    apply_names(pieces, names, fallback_idx)
    return LabelReport(
        backend=labeler.name,
        named=len(suggestions),
        kept=len(pieces) - len(todo),
        fallback=fallback_count,
        requests=getattr(labeler, "requests", 0),
        failures=getattr(labeler, "failures", 0),
        reason=reason,
        names=names,
    )


def apply_names(pieces: Sequence[Piece], names: Sequence[str], fallback_idx: set[int] | None = None) -> None:
    """Set ``piece.name`` for all; regenerate ``piece.id`` from the name except for grid fallbacks (ids stay index-based)."""
    fallback_idx = fallback_idx or set()
    for i, (p, n) in enumerate(zip(pieces, names)):
        p.name = n
        p.id = piece_id(p.number, "" if i in fallback_idx else n)


def label_entry(library: Any, entry_id: str, backend: str = "auto", force: bool = False, client: Any = None) -> LabelReport:
    """Name the regions of a library entry and persist them (the editor shows them, renders use them)."""
    from .segment import crop_pieces, load_gray

    entry = library.get(entry_id)
    regions = [Region.from_dict(r) for r in entry.regions or []]
    if not regions:
        return LabelReport(backend="none", named=0, kept=0, fallback=0, reason="entry has no regions")
    gray = load_gray(library.image_path(entry))
    pieces = crop_pieces(gray, regions)
    report = label_pieces(pieces, entry.name, backend=backend, force=force, client=client)
    for r, p in zip(regions, pieces):
        r.name = p.name
    entry.regions = [r.to_dict() for r in regions]
    library.save(entry)
    return report
