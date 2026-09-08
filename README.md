# flash-to-render

Turn tattoo flash (a single design or a whole flash sheet, as PNG / JPG / WebP)
into per-design 3D assets and clean ink cutouts:

- `NN.glb` - a closed, extruded mesh of the design (binary glTF, opens anywhere)
- `NN.png` - the ink cut out with alpha; RGB is pure white so you can tint it
- `NN.svg` - the traced vector outline
- `all.glb` - one scene with every piece laid out where it sat on the sheet
- `manifest.json` - bbox on the sheet, speckle score, triangle count and a quality flag per piece

Drop in a sheet, get back the pieces, look at them in a browser.

```
flash-to-render examples/spiderverse.webp -o out/spiderverse
flash-to-render preview out/spiderverse
```

## Install

Python 3.11+.

```
git clone https://github.com/aidenzegil/flash-to-render
cd flash-to-render
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"        # or: uv pip install -e ".[dev]"
```

Dependencies: opencv-python-headless, numpy, pillow, potracer (pure-Python
potrace), mapbox_earcut, shapely, trimesh. No compilers, no system potrace.
The optional `[server]` extra adds FastAPI, uvicorn and python-multipart for
the web app; `[dev]` includes it plus pytest.

## Usage

```
flash-to-render <input> -o <out dir> [options]
flash-to-render preview <out dir> [--port 8765]
```

Run it on the three example sheets in `examples/` (real flash, included as fixtures):

```
flash-to-render examples/anime.webp       -o out/anime --debug
flash-to-render examples/objects.webp     -o out/objects
flash-to-render examples/spiderverse.webp -o out/spiderverse
```

Every run prints a per-piece report:

```
objects.webp: 1600x1234, mode=sheet, 16 piece(s)
  00   183x167  at (32,35)     ink=  2080  speckle= 0.48  outers=  1  holes=  1  tris=   120  ok
  01   288x290  at (488,21)    ink= 28624  speckle= 0.10  outers=  2  holes= 27  tris= 2028  ok
  ...
  -> out/objects  (16 ok)
```

`--debug` also writes `segmentation.png`, the sheet with every piece's box and
id drawn on it, which is the fastest way to see whether the segmentation did
what you wanted.

A single design (not a sheet) is detected automatically and treated as one
piece; force it with `--single`, or force segmentation with `--sheet`.

`flash-to-render preview out/` serves a small page (Three.js from a CDN) with
the pieces in 3D on a neutral background, orbit controls, and a plain dropdown
to isolate one piece.

## Review boxes in the browser

```
pip install -e ".[server]"      # FastAPI + uvicorn, only needed for the web app
flash-to-render serve           # opens http://127.0.0.1:8766/
```

`serve` is a local one-page app for checking and fixing the segmentation
before rendering. Pick a sheet from the library (it starts with the three
example sheets) or upload a new image; the auto-detected boxes are drawn on
the full-size image in an editor:

- **Box** tool (`B`): drag a box to move it, drag its handles to resize, drag
  on empty canvas to add one
- **Lasso** tool (`L`): click-and-drag a freehand outline around a design; it
  closes on release. Polygons get vertex handles (drag to adjust, double-click
  an edge to insert a vertex, Delete on a selected vertex removes it). Ink from
  a neighbouring design inside the polygon's bounding box is excluded from
  the render.
- double-click a region to rename it (the name becomes part of the piece id
  and filenames, e.g. `03-skull.glb`)
- Delete / Backspace removes the selected region, arrow keys nudge it (shift =
  10 px), Esc deselects
- **Re-detect** re-runs the auto-segmentation with the settings row (merge
  kernel, min area, smoothing); **Save boxes** persists them; **Render** runs
  trace → mesh → export on the *edited* boxes, shows progress, then embeds the
  3D viewer and offers a zip of the outputs.

The library lives in `~/.flash-to-render/library/` (`--library DIR` to change
it): each entry is the original image plus a `meta.json` with the current
regions (`{"id", "name", "kind": "rect" | "polygon", "points": [[x, y], ...]}`
in image pixels), so edits survive restarts. A lone design opens with a single
box.

The same operations are available as a JSON API (`/api/library`,
`/api/library/{id}/boxes`, `/api/library/{id}/detect`,
`/api/library/{id}/render` → `/api/jobs/{id}`), see `server.py`.

### Output units

glTF is in metres; the default `--scale 0.001` makes one source pixel one
millimetre, so a 300 px design is a 30 cm object. The manifest records the
scale and every bbox in source pixels.

## How it works

1. **Segment** (`segment.py`) - `detect_boxes()` thresholds the sheet into an
   ink mask, dilates it so the strokes of one design touch, takes connected
   components and returns their boxes in reading order (small components close
   to a neighbour are absorbed so the words of a caption stay together).
   `crop_pieces()` then cuts a piece per box, auto-detected or hand-edited:
   each ink component goes to the box holding most of it, so neighbours that
   bleed into a box are masked away while several boxes can split one merged
   design.
2. **Trace** (`trace.py`) - potrace the crop into Bezier outlines, sample them
   to polylines and simplify. Speckly (halftone / grey-shaded) pieces get a
   blur + re-threshold + morphological close first; crisp line art is left
   untouched.
3. **Nest** - sort rings by area, find each ring's smallest containing ring,
   even depth = outer boundary, odd depth = hole. An island inside a hole
   becomes a new outer.
4. **Triangulate** (`mesh.py`) - `mapbox_earcut` on each outer with its holes.
5. **Extrude** - front cap, back cap and one quad per ring edge. Rings are
   oriented with shapely so wall normals point out of the material, y is
   flipped exactly once (image y-down to glTF y-up) and every triangle's
   winding is checked against its normal. Depth is 3.5% of the piece's longest
   side, which reads as a cast object rather than a slab.
6. **Export** (`export.py`) - GLB via trimesh, PNG cutout, SVG, manifest.

### Things that bit us (and are now tested)

- `potracer.Bitmap(data)` calls `.invert()` internally: the pixels it traces
  are the ones that were **False** in your array. Hand it an ink mask and you
  get the design cut out of a rectangular card. `trace_ink` takes an ink mask,
  inverts for you, and re-traces with flipped polarity if the biggest outline
  is a card the size of the crop.
- Halftone shading traces into thousands of specks. The *speckle* score
  (ink components per 1000 ink pixels) is in the manifest; pieces above
  `--max-speckle` are flagged so you can drop them.
- Quantising positions to int16 and then scaling in place truncates to zero.
  Scale first (`--scale` is applied before anything is stored).

## Tuning

| flag | default | what it does |
|---|---|---|
| `--merge-kernel N` | ~1% of the long side (15 px on a 1600 px sheet) | dilation that fuses one design's strokes; raise it if designs split, lower it if neighbours merge |
| `--min-area N` / `--min-size N` | scaled from the sheet | drop components with fewer ink pixels / smaller boxes |
| `--pad N` | 6 | padding around each crop |
| `--margin F` | 0.015 | fraction of the short side ignored at the sheet edges (scanner shadows, card borders) |
| `--threshold N` | 150 | grey level below which a pixel is ink |
| `--smooth auto\|on\|off` | auto | halftone pre-blur; `auto` applies it only to pieces with raw speckle > `--smooth-speckle` (3.0) |
| `--turdsize N` | 14 | potrace: ignore ink blobs smaller than this |
| `--simplify PX` | 0.9 | polyline simplification tolerance |
| `--depth F` | 0.035 | extrusion depth as a fraction of the longest side (6%+ looks like a slab) |
| `--scale F` | 0.001 | output units per pixel |
| `--max-speckle F` | 8 | flag pieces above this speckle score |
| `--min-piece N` | 150 | flag pieces whose longest side is below this |
| `--drop-flagged` | off | do not write assets for flagged pieces |

The same knobs are available from Python:

```python
from flash_to_render import run, PipelineOptions, SegmentOptions

result = run("sheet.webp", "out", PipelineOptions(segment=SegmentOptions(merge_kernel=21)))
for r in result.pieces:
    print(r.piece.id, r.piece.bbox, r.quality, r.triangles)
```

## Known limits

- **Halftone / grey shading.** Dot shading has no clean outline. The
  smoothing pass collapses most of it into solid regions but heavily shaded
  pieces still come out as confetti; they are flagged `speckle` in the manifest.
- **Tiny designs** (longest side under ~150 px) trace blobby and are flagged
  `tiny`. Scan at a higher resolution.
- **Captions and spaced words.** A caption right under a drawing merges into
  it; a phrase with wide word spacing can split into words. `--merge-kernel`
  moves that trade-off in either direction (e.g. `--merge-kernel 21` keeps
  "it's okay i'm spiderman" together on the example sheet at the cost of
  gluing a couple of captions to their drawings).
- Designs that touch each other on the sheet are one piece; there is no
  attempt to separate them.
- Everything is single-colour: the output is the ink silhouette, extruded.

## Development

```
pytest            # runs the full pipeline on the three example sheets + the web API
```

The suite checks piece counts (within a tolerance), that no piece traced as a
card (polarity guard), that every extrusion is a closed surface (each edge
shared by exactly two triangles), wall normals, the y flip, that every GLB
loads in trimesh and is watertight, and the web API end to end (seeded
library, detection, box round-trips, rendering hand-edited and renamed boxes).

## License

MIT. The example flash sheets in `examples/` are the artist's and are included
as test fixtures.
