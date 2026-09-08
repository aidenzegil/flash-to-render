"""Web API tests (FastAPI TestClient) against a temporary library seeded from ``examples/``."""

from __future__ import annotations

import io
import json
import time
import zipfile

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from flash_to_render.library import SEEDS, Library, read_regions_file
from flash_to_render.server import create_app

from conftest import FIXTURES
from test_pipeline import EXPECTED

SEED_FILE = {entry_id: filename for entry_id, _name, filename in SEEDS}
SEED_SHEET = {"payday-spider-verse": "spiderverse", "payday-anime": "anime", "payday-objects": "objects"}


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    app = create_app(tmp_path_factory.mktemp("library"))
    with TestClient(app) as c:
        yield c


def wait_for_job(client: TestClient, job_id: str, timeout: float = 120.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("done", "failed"):
            return job
        time.sleep(0.2)
    raise AssertionError("render job did not finish")


def test_index_serves_the_app(client):
    r = client.get("/")
    assert r.status_code == 200 and "<canvas" in r.text and "flash-to-render" in r.text


def test_library_lists_the_seeded_sheets(client):
    entries = client.get("/api/library").json()
    names = [e["name"] for e in entries]
    assert names == ["Payday · Spider-Verse", "Payday · Anime", "Payday · Objects"]
    for e in entries:
        assert e["width"] > 1000 and e["height"] > 1000 and e["size"] > 100_000
        img = client.get(f"/api/library/{e['id']}/image")
        assert img.status_code == 200 and len(img.content) == e["size"]


@pytest.mark.parametrize("entry_id", list(SEED_SHEET))
def test_detect_returns_boxes_within_pipeline_tolerance(client, entry_id):
    expected, tol = EXPECTED[SEED_SHEET[entry_id]]
    first = client.get(f"/api/library/{entry_id}/boxes").json()  # the hand-edited regions shipped with the repo
    assert len(first["regions"]) == len(first["boxes"]) and all(r["id"] for r in first["regions"])
    shipped = {"payday-spider-verse": 19, "payday-anime": 41, "payday-objects": 16}[entry_id]
    assert first["mode"] == "boxes" and len(first["regions"]) == shipped
    again = client.post(f"/api/library/{entry_id}/detect", json={"mode": "sheet"}).json()
    assert abs(len(again["boxes"]) - expected) <= tol
    assert all(r["kind"] == "rect" and len(r["points"]) == 4 and r["id"] for r in again["regions"])
    for b in again["boxes"]:
        assert 0 <= b["x"] < first["width"] and 0 <= b["y"] < first["height"] and b["w"] > 0 and b["h"] > 0
    # a bigger kernel merges more
    coarse = client.post(f"/api/library/{entry_id}/detect", json={"mode": "sheet", "merge_kernel": 41}).json()
    assert len(coarse["boxes"]) < len(first["boxes"])


def test_put_boxes_round_trips_and_clamps(client):
    entry_id = "payday-objects"
    boxes = [
        {"x": 10, "y": 20, "w": 300, "h": 200, "name": "skull"},
        {"x": 1500, "y": 1100, "w": 400, "h": 400, "name": ""},  # hangs off the sheet
    ]
    r = client.put(f"/api/library/{entry_id}/boxes", json={"boxes": boxes, "options": {"merge_kernel": 15}})
    assert r.status_code == 200
    saved = client.get(f"/api/library/{entry_id}/boxes").json()
    assert saved["mode"] == "boxes" and saved["options"] == {"merge_kernel": 15}
    assert saved["boxes"][0] == {"x": 10, "y": 20, "w": 300, "h": 200, "name": "skull"}
    assert saved["regions"][0]["kind"] == "rect" and saved["regions"][0]["points"][2] == [310, 220]
    assert saved["boxes"][1]["x"] + saved["boxes"][1]["w"] <= saved["width"]
    assert saved["boxes"][1]["y"] + saved["boxes"][1]["h"] <= saved["height"]
    # survives a fresh Library instance (it is on disk)
    fresh = create_app(client.app.state.library.root, seed=False)
    assert fresh.state.library.get(entry_id).regions[0]["name"] == "skull"

    # polygons round-trip too, keep their ids, and get one when missing
    poly = {"kind": "polygon", "name": "tri", "id": "abc123", "points": [[10, 10], [300, 10], [10, 300]]}
    r = client.put(f"/api/library/{entry_id}/boxes", json={"regions": [poly, {"kind": "polygon", "points": [[0, 0], [50, 0], [0, 50]]}]})
    assert r.status_code == 200
    got = client.get(f"/api/library/{entry_id}/boxes").json()["regions"]
    assert got[0]["id"] == "abc123" and got[0]["points"] == [[10, 10], [300, 10], [10, 300]]
    assert got[1]["id"] and got[1]["kind"] == "polygon"
    assert client.put(f"/api/library/{entry_id}/boxes", json={"regions": [{"kind": "polygon", "points": [[0, 0], [5, 5]]}]}).status_code == 400


def test_render_uses_edited_boxes_and_names(client):
    entry_id = "payday-objects"
    auto = client.post(f"/api/library/{entry_id}/detect", json={"mode": "sheet"}).json()["boxes"]
    # hand-edit: keep three of the auto boxes, rename two, and merge two neighbours into one wide box
    a, b, c, d = auto[0], auto[1], auto[2], auto[3]
    wide = {"x": min(c["x"], d["x"]), "y": min(c["y"], d["y"]), "name": "pair"}
    wide["w"] = max(c["x"] + c["w"], d["x"] + d["w"]) - wide["x"]
    wide["h"] = max(c["y"] + c["h"], d["y"] + d["h"]) - wide["y"]
    edited = [dict(a, name="Sparkle"), dict(b, name="Heart & Stripes"), wide]

    r = client.post(f"/api/library/{entry_id}/render", json={"boxes": edited})
    assert r.status_code == 200
    job = wait_for_job(client, r.json()["job_id"])
    assert job["status"] == "done", job["error"]
    assert job["progress"] == {"done": 3, "total": 3, "current": job["pieces"][-1]["id"]}

    ids = [p["id"] for p in job["pieces"]]
    assert ids == ["00-sparkle", "01-heart-stripes", "02-pair"]
    assert [p["name"] for p in job["pieces"]] == ["Sparkle", "Heart & Stripes", "pair"]
    assert all(p["files"]["glb"] == f"{p['id']}.glb" for p in job["pieces"])
    assert [p["bbox"] for p in job["pieces"]][2][:2] == [wide["x"], wide["y"]]

    # the edited boxes were persisted as the entry's current regions
    assert [bx["name"] for bx in client.get(f"/api/library/{entry_id}/boxes").json()["regions"]] == ["Sparkle", "Heart & Stripes", "pair"]
    assert all(p["kind"] == "rect" for p in job["pieces"])

    glb = client.get(f"/api/jobs/{job['id']}/preview.glb")
    assert glb.status_code == 200 and glb.content[:4] == b"glTF"
    manifest = client.get(f"/api/jobs/{job['id']}/preview/manifest.json").json()
    assert manifest["mode"] == "boxes" and len(manifest["pieces"]) == 3
    assert client.get(f"/api/jobs/{job['id']}/preview/").text.startswith("<!doctype html>")

    z = client.get(f"/api/jobs/{job['id']}/download.zip")
    assert z.status_code == 200 and z.headers["content-type"] == "application/zip"
    names = zipfile.ZipFile(io.BytesIO(z.content)).namelist()
    assert {"00-sparkle.glb", "01-heart-stripes.glb", "02-pair.glb", "all.glb", "manifest.json", "segmentation.png"} <= set(names)
    assert sum(n.endswith(".glb") for n in names) == 4


def test_upload_single_design_gets_one_box(client):
    canvas = np.full((400, 500), 255, dtype=np.uint8)
    canvas[100:300, 150:350] = 0
    canvas[150:250, 200:300] = 255  # a hole
    buf = io.BytesIO()
    Image.fromarray(canvas).save(buf, format="PNG")
    r = client.post("/api/library/upload", files={"file": ("square.png", buf.getvalue(), "image/png")}, data={"name": "My square"})
    assert r.status_code == 200, r.text
    entry = r.json()
    assert entry["name"] == "My square" and entry["mode"] == "single" and entry["box_count"] == 1
    boxes = client.get(f"/api/library/{entry['id']}/boxes").json()["boxes"]
    assert boxes[0]["x"] <= 150 and boxes[0]["y"] <= 100
    assert boxes[0]["x"] + boxes[0]["w"] >= 350 and boxes[0]["y"] + boxes[0]["h"] >= 300
    assert any(e["id"] == entry["id"] for e in client.get("/api/library").json())

    job = wait_for_job(client, client.post(f"/api/library/{entry['id']}/render", json={}).json()["job_id"])
    assert job["status"] == "done" and len(job["pieces"]) == 1 and job["pieces"][0]["holes"] == 1

    assert client.delete(f"/api/library/{entry['id']}").status_code == 200
    assert client.get(f"/api/library/{entry['id']}").status_code == 404


def test_bad_upload_and_missing_things(client):
    r = client.post("/api/library/upload", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 400 and "unsupported" in r.json()["error"]
    assert client.get("/api/library/nope/boxes").status_code == 404
    assert client.get("/api/jobs/nope").status_code == 404
    assert client.post("/api/library/payday-anime/render", json={"boxes": []}).status_code == 400


def test_polygon_region_render_excludes_neighbour(client):
    """A lasso around design A whose bbox also covers design B renders only A's ink."""
    sheet = np.full((400, 400), 255, dtype=np.uint8)
    sheet[100:200, 100:200] = 0
    sheet[210:310, 210:310] = 0
    buf = io.BytesIO()
    Image.fromarray(sheet).save(buf, format="PNG")
    entry = client.post("/api/library/upload", files={"file": ("two.png", buf.getvalue(), "image/png")}).json()
    tri = {"kind": "polygon", "name": "A only", "points": [[90, 90], [320, 90], [90, 320]]}
    rect = {"kind": "rect", "name": "both", "points": [[90, 90], [320, 90], [320, 320], [90, 320]]}

    def render(regions):
        r = client.post(f"/api/library/{entry['id']}/render", json={"regions": regions})
        assert r.status_code == 200, r.text
        job = wait_for_job(client, r.json()["job_id"])
        assert job["status"] == "done", job["error"]
        return job

    poly_job = render([tri])
    rect_job = render([rect])
    (poly,) = poly_job["pieces"]
    (both,) = rect_job["pieces"]
    assert poly["bbox"] == both["bbox"] == [90, 90, 230, 230]
    assert both["ink_area"] == 2 * 100 * 100
    assert poly["ink_area"] == 100 * 100
    assert poly["kind"] == "polygon" and poly["id"] == "00-a-only" and both["kind"] == "rect"
    assert poly["triangles"] < both["triangles"]
    # the cutout PNG is transparent where design B was
    png = Image.open(io.BytesIO(client.get(f"/api/jobs/{poly_job['id']}/preview/00-a-only.png").content))
    k = poly["png_scale"]
    assert png.size == (230 * k, 230 * k)
    alpha = np.asarray(png)[..., 3]
    assert alpha[(250 - 90) * k, (250 - 90) * k] == 0 and alpha[(150 - 90) * k, (150 - 90) * k] == 255
    # the polygon is what the entry now remembers
    saved = client.get(f"/api/library/{entry['id']}/boxes").json()["regions"]
    assert saved[0]["kind"] == "rect" and saved[0]["name"] == "both"  # last render wins
    client.delete(f"/api/library/{entry['id']}")


def test_seeding_is_idempotent_and_spider_verse_ships_with_regions(tmp_path):
    lib = Library(tmp_path / "lib")
    sv = lib.get("payday-spider-verse")
    assert sv.mode == "boxes" and len(sv.regions) == 19
    assert {r["kind"] for r in sv.regions} == {"rect", "polygon"}
    assert all(r["id"] for r in sv.regions)
    assert len(lib.get("payday-anime").regions) == 41  # the anime regions ship too

    # hand-edit, then seed again (a server restart): nothing is rewritten
    sv.regions = sv.regions[:3]
    sv.name = "edited"
    lib.save(sv)
    meta = lib.meta_path("payday-spider-verse")
    before = meta.read_text()
    stamp = meta.stat().st_mtime_ns
    (lib.root / "payday-anime" / "marker").write_text("x")
    assert Library(tmp_path / "lib").seed(FIXTURES) == []
    assert meta.read_text() == before and meta.stat().st_mtime_ns == stamp
    assert Library(tmp_path / "lib").get("payday-spider-verse").name == "edited"
    assert (lib.root / "payday-anime" / "marker").exists()

    # the app serves them and the export file round-trips through the CLI format
    with TestClient(create_app(tmp_path / "lib")) as c:
        got = c.get("/api/library/payday-spider-verse/boxes").json()
        assert len(got["regions"]) == 3
        exported = c.get("/api/library/payday-spider-verse/regions.json")
        assert exported.status_code == 200 and "attachment" in exported.headers["content-disposition"]
        assert len(exported.json()["regions"]) == 3


def test_regions_cli_export_import(tmp_path):
    from flash_to_render.cli import main

    lib_dir = tmp_path / "lib"
    Library(lib_dir)
    out = tmp_path / "sv.json"
    assert main(["regions", "export", "payday-spider-verse", str(out), "--library", str(lib_dir)]) == 0
    data = json.loads(out.read_text())
    assert len(data["regions"]) == 19 and data["entry"] == "payday-spider-verse"
    assert main(["regions", "import", "payday-objects", str(out), "--library", str(lib_dir)]) == 0
    assert len(Library(lib_dir, seed=False).get("payday-objects").regions) == 19
    assert len(read_regions_file(FIXTURES / "payday-spider-verse.regions.json")) == 19
    assert main(["regions", "export", "nope", str(out), "--library", str(lib_dir)]) == 2
