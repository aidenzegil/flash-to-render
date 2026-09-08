"""Piece naming: uniqueness everywhere, the grid fallback, and the caption labeler with a mocked client (no network)."""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from flash_to_render.label import (
    CaptionLabeler,
    NoneLabeler,
    _parse_names,
    clean_name,
    fallback_names,
    get_labeler,
    grid_positions,
    label_entry,
    label_pieces,
    sheet_token,
    stamp_bytes,
    unique_names,
)
from flash_to_render.library import Library
from flash_to_render.pipeline import PipelineOptions, run
from flash_to_render.segment import Box, Region, crop_pieces, load_gray
from flash_to_render.server import create_app

from conftest import FIXTURES


class FakeClient:
    """Stands in for anthropic.Anthropic: records every request, replies from a script."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=reply)], stop_reason="end_turn")


@pytest.fixture(autouse=True)
def no_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


def test_unique_names_is_deterministic_and_case_insensitive():
    assert unique_names(["Kitsune Mask", "kitsune mask", "Kitsune  Mask", "Dagger", ""]) == ["Kitsune Mask", "kitsune mask II", "Kitsune  Mask III".replace("  ", " "), "Dagger", ""]
    assert unique_names(["Mask", "Mask"], reserved=["Mask II"]) == ["Mask", "Mask III"]
    assert unique_names(["Mask II", "Mask II"]) == ["Mask II", "Mask II II"]
    assert unique_names([]) == []


def test_fallback_names_follow_reading_order():
    boxes = [Box(400, 10, 50, 50), Box(10, 30, 50, 50), Box(10, 200, 50, 50), Box(300, 205, 50, 50)]
    assert grid_positions(boxes) == [(1, 2), (1, 1), (2, 1), (2, 2)]
    assert fallback_names(boxes, "Payday · Anime") == ["Anime No. 2", "Anime No. 1", "Anime No. 3", "Anime No. 4"]
    assert sheet_token("spiderverse.webp") == "Spiderverse" and sheet_token("") == "Sheet"
    assert clean_name("  \"Kitsune Mask.\" ") == "Kitsune Mask" and clean_name("a b c d e f g h") == "a b c d e f"


def test_crop_pieces_and_region_put_dedupe_names(tmp_path):
    gray = load_gray(FIXTURES / "objects.webp")
    regs = [Region.rect(30, 30, 200, 180, "Skull"), Region.rect(480, 20, 300, 300, "Skull"), Region.rect(800, 30, 200, 300, "skull")]
    names = [p.name for p in crop_pieces(gray, regs)]
    assert names == ["Skull", "Skull II", "skull III"]
    assert [p.id for p in crop_pieces(gray, regs)] == ["01-skull", "02-skull-ii", "03-skull-iii"]

    with TestClient(create_app(tmp_path / "lib")) as c:
        body = {"regions": [{"kind": "rect", "name": "Mask", "points": [[10, 10], [200, 10], [200, 200], [10, 200]]},
                            {"kind": "rect", "name": "Mask", "points": [[300, 10], [500, 10], [500, 200], [300, 200]]}]}
        got = c.put("/api/library/payday-objects/boxes", json=body).json()["regions"]
        assert [r["name"] for r in got] == ["Mask", "Mask II"]
        job = c.post("/api/library/payday-objects/render", json={"regions": body["regions"]}).json()
        import time

        for _ in range(300):
            j = c.get(f"/api/jobs/{job['job_id']}").json()
            if j["status"] in ("done", "failed"):
                break
            time.sleep(0.2)
        assert j["status"] == "done" and [p["name"] for p in j["pieces"]] == ["Mask", "Mask II"]


def test_manifest_names_are_unique_and_unnamed_pieces_get_grid_fallback(tmp_path):
    res = run(FIXTURES / "objects.webp", tmp_path / "out", PipelineOptions(write_glb=False, write_svg=False, write_png=False), report=None)
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text())
    names = [p["name"] for p in manifest["pieces"]]
    assert all(names) and len(set(names)) == len(names)
    assert names == [f"Objects No. {i}" for i in range(1, len(names) + 1)]
    assert manifest["naming"]["backend"] == "none" and manifest["naming"]["fallback"] == len(names)
    assert [p["id"] for p in manifest["pieces"]][:3] == ["01", "02", "03"]  # fallbacks do not rename files
    assert [p["index"] for p in manifest["pieces"]] == list(range(1, len(names) + 1))


def test_get_labeler_never_sends_without_a_key():
    labeler, reason = get_labeler("auto")
    assert isinstance(labeler, NoneLabeler) and "ANTHROPIC_API_KEY" in reason
    labeler, reason = get_labeler("caption")
    assert isinstance(labeler, NoneLabeler) and reason
    labeler, reason = get_labeler("none")
    assert isinstance(labeler, NoneLabeler) and reason is None
    labeler, reason = get_labeler("auto", client=object())
    assert isinstance(labeler, CaptionLabeler) and reason is None
    with pytest.raises(ValueError):
        get_labeler("magic")


def _pieces(n=12):
    gray = load_gray(FIXTURES / "objects.webp")
    from flash_to_render.segment import detect_boxes

    boxes = detect_boxes(gray)[:n]
    return crop_pieces(gray, boxes)


def test_caption_labeler_batches_parses_retries_and_falls_back():
    pieces = _pieces(12)
    first = json.dumps([f"Design {i}" for i in range(10)])
    second_bad = "Sure! Here are the names: Sparkle, Heart"  # malformed
    second_ok = json.dumps(["Broken Chain", "Melting Disc."])
    fake = FakeClient([first, second_bad, second_ok])
    rep = label_pieces(pieces, "Payday · Objects", backend="auto", client=fake)
    assert rep.backend == "caption" and rep.named == 12 and rep.fallback == 0 and rep.requests == 3 and rep.failures == 0
    assert rep.names[:2] == ["Design 0", "Design 1"] and rep.names[-2:] == ["Broken Chain", "Melting Disc"]
    assert sorted(p.id for p in pieces)[-2:] == sorted(f"{p.number:02d}-{s}" for p, s in zip(pieces[-2:], ("broken-chain", "melting-disc")))
    # batching: 12 stamps -> a request of 10 images and one of 2; the retry carries the malformed reply back
    imgs = [sum(1 for c in call["messages"][0]["content"] if c["type"] == "image") for call in fake.calls]
    assert imgs == [10, 2, 2]
    assert fake.calls[0]["model"] == "claude-opus-5" and "JSON array of exactly 10" in fake.calls[0]["messages"][0]["content"][-1]["text"]
    assert len(fake.calls[2]["messages"]) == 3 and fake.calls[2]["messages"][1]["role"] == "assistant"
    src = fake.calls[0]["messages"][0]["content"][0]["source"]
    assert src["type"] == "base64" and src["media_type"] == "image/png"

    # two malformed replies in a row -> that batch falls back to grid names; an exception too
    pieces = _pieces(3)
    fake = FakeClient(["nope", "still nope"])
    rep = label_pieces(pieces, "Objects", backend="caption", client=fake)
    assert rep.named == 0 and rep.fallback == 3 and rep.failures == 1 and rep.names == ["Objects No. 1", "Objects No. 2", "Objects No. 3"]
    fake = FakeClient([RuntimeError("boom")])
    rep = label_pieces(_pieces(2), "Objects", backend="caption", client=fake)
    assert rep.fallback == 2 and rep.failures == 1
    # wrong length / non-strings are malformed
    assert _parse_names("[\"a\", \"b\"]", 3) is None and _parse_names("[1, 2]", 2) is None and _parse_names("[\"x\", \"\"]", 2) == ["x", None]


def test_labeler_keeps_human_names_and_dedupes_suggestions():
    pieces = _pieces(4)
    pieces[0].name = "Sparkle"
    pieces[1].name = "Vinyl"
    fake = FakeClient([json.dumps(["Sparkle", "Vinyl"])])  # the model reuses names a human already took
    rep = label_pieces(pieces, "Objects", backend="auto", client=fake)
    assert rep.kept == 2 and rep.named == 2
    assert rep.names == ["Sparkle", "Vinyl", "Sparkle II", "Vinyl II"]
    assert sum(1 for c in fake.calls[0]["messages"][0]["content"] if c["type"] == "image") == 2  # only the unnamed two were sent
    # force renames everything
    fake = FakeClient([json.dumps(["A", "B", "C", "D"])])
    rep = label_pieces(pieces, "Objects", backend="auto", force=True, client=fake)
    assert rep.names == ["A", "B", "C", "D"] and rep.kept == 0


def test_label_entry_persists_names_and_api_endpoint(tmp_path):
    lib = Library(tmp_path / "lib")
    entry = lib.get("payday-objects")
    entry.regions[0]["name"] = "Sparkle"
    lib.save(entry)
    n = len(entry.regions)
    fake = FakeClient([json.dumps([f"Piece {i}" for i in range(10)]), json.dumps([f"Piece {i}" for i in range(n - 11)])])
    rep = label_entry(lib, "payday-objects", backend="auto", client=fake)
    saved = lib.get("payday-objects").regions
    assert saved[0]["name"] == "Sparkle" and rep.kept == 1 and rep.named == n - 1
    assert len({r["name"] for r in saved}) == n

    # the API: no key -> fallback names, nothing sent, the reason is reported
    with TestClient(create_app(tmp_path / "lib")) as c:
        r = c.post("/api/library/payday-anime/label", json={"backend": "auto"})
        assert r.status_code == 200
        data = r.json()
        assert data["report"]["backend"] == "none" and "ANTHROPIC_API_KEY" in data["report"]["reason"]
        names = [x["name"] for x in data["regions"]]
        assert all(names) and len(set(names)) == len(names) and names[0].startswith("Anime ")
        assert c.post("/api/library/payday-anime/label", json={"backend": "magic"}).status_code == 400
        # a render after labeling carries the names into the manifest
        again = c.get("/api/library/payday-anime/boxes").json()["regions"]
        assert [x["name"] for x in again] == names


def test_stamp_bytes_is_a_small_png_over_white():
    piece = _pieces(1)[0]
    png = stamp_bytes(piece, max_px=128)
    img = Image.open(__import__("io").BytesIO(png))
    assert img.format == "PNG" and max(img.size) <= 128 and img.mode == "RGB"
    arr = np.asarray(img)
    assert arr.max() == 255 and arr.min() < 60  # white paper, dark ink


def test_numbering_is_reading_order_and_renumbers_on_add_delete_move():
    from flash_to_render.segment import reading_positions

    gray = load_gray(FIXTURES / "objects.webp")
    a, b, c = Region.rect(30, 30, 200, 180, "Sparkle"), Region.rect(480, 20, 300, 200), Region.rect(93, 400, 348, 300)
    # region list order is [a, b, c]; a and b share the top row, c is the second row: numbers 1, 2, 3
    pieces = crop_pieces(gray, [a, b, c])
    assert [p.number for p in pieces] == [1, 2, 3] and [p.id for p in pieces] == ["01-sparkle", "02", "03"]
    assert [(p.row, p.col) for p in pieces] == [(1, 1), (1, 2), (2, 1)]
    # delete b: c moves up to No. 2, the typed name stays with a
    pieces = crop_pieces(gray, [a, c])
    assert [p.id for p in pieces] == ["01-sparkle", "02"] and pieces[0].name == "Sparkle"
    # add a region left of a on the top row: everything after it shifts, the name still follows its region
    d = Region.rect(5, 40, 20, 100)
    pieces = crop_pieces(gray, [a, b, c, d])
    assert [p.number for p in pieces] == [2, 3, 4, 1] and pieces[0].id == "02-sparkle" and pieces[3].id == "01"
    # move b down next to c: reading order re-sorts, the list order does not change
    b_low = Region.rect(480, 420, 300, 200)
    pieces = crop_pieces(gray, [a, b_low, c])
    assert [p.number for p in pieces] == [1, 3, 2]
    # positions are a pure function of the boxes (stable across calls)
    boxes = [r.bbox() for r in (a, b, c, d)]
    assert reading_positions(boxes) == reading_positions(list(boxes))


def test_output_files_follow_the_index(tmp_path):
    gray_regions = [Region.rect(480, 20, 300, 300, "Vinyl"), Region.rect(30, 30, 200, 180)]  # list order is not reading order
    res = run(FIXTURES / "objects.webp", tmp_path / "out", PipelineOptions(write_svg=False), report=None, boxes=gray_regions)
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert [p["index"] for p in manifest["pieces"]] == [1, 2]
    assert [p["id"] for p in manifest["pieces"]] == ["01", "02-vinyl"]
    assert [p["name"] for p in manifest["pieces"]] == ["Objects No. 1", "Vinyl"]
    assert (tmp_path / "out" / "01.glb").exists() and (tmp_path / "out" / "02-vinyl.png").exists()
    assert [p["files"]["glb"] for p in manifest["pieces"]] == ["01.glb", "02-vinyl.glb"]
