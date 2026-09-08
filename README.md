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

## Fine lines

Thin linework is where a naive trace falls apart: 1-2 px seams and small
counters vanish and a caption turns into dots. The tracer therefore works
per piece at a higher resolution and measures itself:

1. **Upscale** the crop 3x (4x in `best` mode) with bicubic interpolation
   before anything is thresholded or traced; every pixel-based parameter
   (turdsize, simplification tolerance, close kernel, minimum hole area) is
   scaled with it and the geometry comes back in source pixels.
2. **Edge-aware binarisation**: unsharp mask, then a global threshold for solid
   ink OR-ed with a Gaussian adaptive threshold (`adaptiveThreshold`) so thin
   strokes and small counters survive. Line art is never blurred; only
   genuinely halftoned pieces get the blur + close treatment, gated on
   *both* a high speckle score and a high mid-grey ratio (a caption is
   speckly but black; shading is grey).
3. **Small things survive**: turdsize 4 source px², minimum hole area 1.5 px²,
   a 0.5 px simplification tolerance and a tighter potrace `opttolerance`.
4. **Fidelity score**: the traced polygons are rasterised back at the working
   resolution and compared with the cleaned ink mask - IoU, thin-feature recall
   (the share of pixels an opening removes that the trace still covers) and a
   hole-count ratio, combined as `0.6·IoU + 0.25·thin + 0.15·holes`. The score,
   its parts and the parameters used land in `manifest.json` per piece.
5. **`--fidelity best`** traces a small grid per piece (upscale 3/4 x turdsize
   2/4 x curve tolerance 0.2/0.5) and keeps the best score under
   `--triangle-budget` (20k triangles per piece by default). `fast` (the
   default) traces once at 3x. The editor has the same setting.
6. The ink PNG cutouts are rendered from the cleaned hi-res mask at 2x source
   resolution (`png_scale` in the manifest), so they are crisp rather than a
   blurry threshold of the raw crop.
7. Pieces whose ink is mostly thin (more than 30% removed by a 2 px opening)
   are extruded at 2.5% of their longest side instead of 3.5%, so hairlines do
   not read as slabs. Holes get walls on both sides like any other ring.

On the example sheets, "A Leap of Faith" goes from IoU 0.91 / 8 holes / a
caption reduced to three dots, to IoU 0.97 / 12 holes / every letter of the
caption; the cassette from IoU 0.81 to 0.90 with thin-feature recall 0.55 to
0.83. `tests/test_fidelity.py` pins these.

## Shaded and halftone work

Grey is not one thing, so the tracer looks twice before deciding what to do
with a piece that is speckly *and* grey (the old "halftone" gate):

- **Grey line art** (faint outlines, pencil-weight strokes - most of the
  shaded pieces on the anime sheet) is still line art. It gets the edge-aware
  path with a lower adaptive C and a ~1.7 px gap-close so faded strokes rejoin
  instead of breaking into confetti. Manifest: `tone_mode: "light"`. On the
  anime sheet this takes the fan-and-cat piece from 210 traced outers to 46,
  the flaming cat from 117 to 34, the flower cluster from 102 to 12.
- **Halftone dot fields** are recognised by their hi-res adaptive mask
  (hundreds of compact specks per unit of ink). Their grey is blurred over the
  estimated dot spacing (median nearest-neighbour distance between specks,
  fallback 2.5 px) into a coverage field, thresholded at 50% and opened/closed
  at the dot spacing, so a stipple becomes the solid shape the artist meant
  and hatched edges become soft outlines. Fidelity is scored against this
  tone-resolved mask. Manifest: `tone_mode: "tone"`, `dot_spacing`.
- **Relief** (`--relief on`, default): a dot-field piece is extruded as two
  closed layers sharing the back plane - coverage >= 70% at full depth and
  35-70% at 45% depth - written as two nodes of the piece's GLB, so shading
  reads as relief. `--relief off` gives one full-depth layer.
- **Stamps** (`--stamp tone`, default): the ink PNG's alpha comes from the
  source grey itself (`clamp((paper - grey) / (paper - ink))` on the 2x
  bicubic, lightly sharpened crop, masked to the region outline), so shading
  prints exactly as drawn. `--stamp binary` uses the traced mask instead.

Crisp black line art never touches any of this: its mask is byte-for-byte the
plain edge-aware binarisation, which the tests check.

## Naming pieces

Every piece gets a name, and within one render (and one library entry) no
two pieces share one:

- **Human names win.** A name typed in the editor (or sent in a region PUT)
  is never overwritten unless you pass `--force`.
- **Uniqueness is enforced wherever names enter** - region PUT, detection,
  the labeler, the editor's rename and the manifest writer. Duplicates are
  disambiguated deterministically: `Kitsune Mask`, `Kitsune Mask II`,
  `Kitsune Mask III` ...
- **Numbering** is 1-based reading order (rows top to bottom, left to
  right) derived from the *current* region list, never stored: add, delete or
  move a region and the others renumber, in the editor and in the output
  (`01.glb`, `02.png`, ...). The manifest carries `index`, `row` and `col`.
  An unnamed piece displays as `<Sheet> No. <index>` (`Anime No. 12`); a
  typed name stays with its region when numbers shift.
- **Labeler** (`label.py`): `flash-to-render label <library-id> [--backend
  caption|none] [--force]`, `--label auto|caption|none` on `convert`,
  `POST /api/library/{id}/label`, and the editor's **Suggest names** button.
  The `caption` backend sends each unnamed piece's stamp (downscaled to
  512 px, over white) to Claude (`claude-opus-5`), several images per request,
  and asks for a JSON array of 2-4 word Title Case names (legible text in the
  design becomes its name). Malformed replies are retried once; any failure
  falls back to the grid name. Names are written back into the library
  entry's regions, so the editor shows them and renders use them.

Install the optional SDK with `pip install -e ".[label]"` and set
`ANTHROPIC_API_KEY`. **Privacy:** images leave your machine only when that
key is set *and* the labeler is invoked (`label`, `--label`, the API endpoint
or the Suggest names button); with no key, `auto` silently uses the fallback
and says so in the report, and `--backend caption` warns and does the same.
Renders never call the API unless asked to.

## Tuning

| flag | default | what it does |
|---|---|---|
| `--merge-kernel N` | ~1% of the long side (15 px on a 1600 px sheet) | dilation that fuses one design's strokes; raise it if designs split, lower it if neighbours merge |
| `--min-area N` / `--min-size N` | scaled from the sheet | drop components with fewer ink pixels / smaller boxes |
| `--pad N` | 6 | padding around each crop |
| `--margin F` | 0.015 | fraction of the short side ignored at the sheet edges (scanner shadows, card borders) |
| `--threshold N` | 150 | grey level below which a pixel is ink |
| `--smooth auto\|on\|off` | auto | `auto` routes grey pieces to the light-line or tone path (see above); `on` forces tone, `off` forces line |
| `--relief on\|off` | on | two-layer relief for halftone pieces |
| `--stamp tone\|binary` | tone | ink PNG alpha from the source grey or from the traced mask |
| `--fidelity fast\|best` | fast | `best` searches a small grid per piece and keeps the best fidelity score under the triangle budget |
| `--upscale N` | 3 | working resolution per piece |
| `--binarize adaptive\|global` | adaptive | edge-aware binarisation vs plain threshold |
| `--triangle-budget N` | 20000 | `best` mode: prefer candidates under this many triangles |
| `--turdsize N` | 4 | potrace: ignore blobs smaller than this many source px² |
| `--simplify PX` | 0.5 | polyline simplification tolerance (source px) |
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
