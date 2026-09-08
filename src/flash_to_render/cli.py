"""Command line entry point.

    flash-to-render sheet.webp -o out/          # convert
    flash-to-render preview out/                # serve a 3D preview of a result folder
    flash-to-render serve                       # web UI to review boxes, render, preview
    flash-to-render regions export <id> <file>  # move hand-edited regions between machines
    flash-to-render regions import <id> <file>
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from . import __version__
from .mesh import DEFAULT_DEPTH_FRAC
from .pipeline import PipelineOptions, run
from .segment import SegmentOptions
from .trace import TraceOptions

SUBCOMMANDS = ("convert", "preview", "serve", "regions")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="flash-to-render",
        description="Turn tattoo flash (a single design or a whole sheet) into per-design 3D meshes and ink cutouts.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    c = sub.add_parser("convert", help="segment, trace and extrude an image (default command)")
    c.add_argument("input", type=Path, help="PNG / JPG / WebP flash sheet or single design")
    c.add_argument("-o", "--out", type=Path, default=None, help="output folder (default: <input stem>_out)")
    mode = c.add_mutually_exclusive_group()
    mode.add_argument("--single", action="store_const", const="single", dest="mode", help="treat the whole image as one design")
    mode.add_argument("--sheet", action="store_const", const="sheet", dest="mode", help="always segment into pieces")
    c.set_defaults(mode="auto")

    seg = c.add_argument_group("segmentation")
    seg.add_argument("--threshold", type=int, default=150, help="grey level below which a pixel is ink (default 150)")
    seg.add_argument("--merge-kernel", type=int, default=None, help="odd dilation kernel in px that fuses one design's strokes (default ~1.3%% of the long side)")
    seg.add_argument("--min-area", type=int, default=None, help="drop components with fewer ink pixels (default scales with sheet)")
    seg.add_argument("--min-size", type=int, default=None, help="drop components narrower/shorter than this many px")
    seg.add_argument("--pad", type=int, default=6, help="padding around each crop in px (default 6)")
    seg.add_argument("--margin", type=float, default=0.015, help="fraction of the short side to ignore at the sheet edges (default 0.015)")

    tr = c.add_argument_group("tracing")
    tr.add_argument("--fidelity", choices=("fast", "best"), default="fast", help="fast: one hi-res trace per piece; best: search a small parameter grid and keep the best fidelity score under the triangle budget")
    tr.add_argument("--upscale", type=int, default=3, help="work at this multiple of the source resolution per piece (default 3)")
    tr.add_argument("--binarize", choices=("adaptive", "global"), default="adaptive", help="adaptive keeps 1-2 px lines (default)")
    tr.add_argument("--triangle-budget", type=int, default=20000, help="best mode: prefer candidates under this many triangles per piece")
    tr.add_argument("--smooth", choices=("auto", "on", "off"), default="auto", help="halftone pre-blur: auto = only speckly pieces (default)")
    tr.add_argument("--smooth-speckle", type=float, default=3.0, help="auto mode: smooth pieces whose raw speckle exceeds this (default 3.0)")
    tr.add_argument("--turdsize", type=int, default=4, help="potrace: ignore blobs smaller than this many source px (default 4)")
    tr.add_argument("--simplify", type=float, default=0.5, help="polyline simplification tolerance in source px (default 0.5)")

    m = c.add_argument_group("mesh")
    m.add_argument("--depth", type=float, default=DEFAULT_DEPTH_FRAC, help=f"extrusion depth as a fraction of the longest side (default {DEFAULT_DEPTH_FRAC})")
    m.add_argument("--scale", type=float, default=0.001, help="output units per pixel (default 0.001: 1 px = 1 mm in glTF metres)")

    q = c.add_argument_group("quality")
    q.add_argument("--max-speckle", type=float, default=8.0, help="flag pieces whose speckle score exceeds this (default 8)")
    q.add_argument("--min-piece", type=int, default=150, help="flag pieces whose longest side is below this many px (default 150)")
    q.add_argument("--drop-flagged", action="store_true", help="do not write assets for flagged pieces")
    q.add_argument("--no-png", action="store_true")
    q.add_argument("--no-svg", action="store_true")
    q.add_argument("--no-glb", action="store_true")
    c.add_argument("--debug", action="store_true", help="also write segmentation.png with every box drawn")
    c.add_argument("-q", "--quiet", action="store_true", help="no per-piece report")

    p = sub.add_parser("preview", help="serve a local 3D preview of an output folder")
    p.add_argument("out_dir", type=Path, help="folder produced by `flash-to-render convert`")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-open", action="store_true", help="do not open a browser")

    s = sub.add_parser("serve", help="web app: upload or pick a sheet, review the boxes, render and preview")
    s.add_argument("--port", type=int, default=8766)
    s.add_argument("--library", type=Path, default=None, help="library folder (default ~/.flash-to-render/library)")
    s.add_argument("--no-open", action="store_true", help="do not open a browser")

    rg = sub.add_parser("regions", help="export or import a library entry's regions as JSON")
    rg.add_argument("action", choices=("export", "import"))
    rg.add_argument("entry_id", help="library entry id, e.g. payday-spider-verse")
    rg.add_argument("file", type=Path, help="JSON file to write (export) or read (import)")
    rg.add_argument("--library", type=Path, default=None, help="library folder (default ~/.flash-to-render/library)")
    return parser


def _options_from_args(a: argparse.Namespace) -> PipelineOptions:
    return PipelineOptions(
        mode=a.mode,
        segment=SegmentOptions(
            threshold=a.threshold,
            margin=a.margin,
            merge_kernel=a.merge_kernel,
            min_ink_area=a.min_area,
            min_size=a.min_size,
            pad=a.pad,
        ),
        trace=TraceOptions(
            threshold=a.threshold,
            smooth=a.smooth,
            smooth_speckle=a.smooth_speckle,
            turdsize=a.turdsize,
            simplify_px=a.simplify,
            fidelity=a.fidelity,
            upscale=a.upscale,
            binarize=a.binarize,
            triangle_budget=a.triangle_budget,
        ),
        depth_frac=a.depth,
        scale=a.scale,
        max_speckle=a.max_speckle,
        min_piece_px=a.min_piece,
        drop_flagged=a.drop_flagged,
        debug=a.debug,
        write_png=not a.no_png,
        write_svg=not a.no_svg,
        write_glb=not a.no_glb,
    )


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # `flash-to-render sheet.png -o out/` is shorthand for `convert`
    if argv and argv[0] not in SUBCOMMANDS and not argv[0].startswith("-"):
        argv.insert(0, "convert")
    if argv and argv[0] in ("-h", "--help", "--version"):
        pass
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    if args.command == "preview":
        from .preview import serve

        return serve(args.out_dir, port=args.port, open_browser=not args.no_open)

    if args.command == "serve":
        try:
            from .server import DEFAULT_LIBRARY, serve as serve_app
        except ImportError as exc:
            print(f"error: the web UI needs the server extra: pip install 'flash-to-render[server]' ({exc})", file=sys.stderr)
            return 2
        return serve_app(args.library or DEFAULT_LIBRARY, port=args.port, open_browser=not args.no_open)

    if args.command == "regions":
        from .library import DEFAULT_LIBRARY, Library, read_regions_file, write_regions_file

        lib = Library(args.library or DEFAULT_LIBRARY, seed=False)
        try:
            entry = lib.get(args.entry_id)
        except KeyError:
            print(f"error: no library entry {args.entry_id!r} in {lib.root}", file=sys.stderr)
            return 2
        if args.action == "export":
            write_regions_file(args.file, entry)
            print(f"wrote {len(entry.regions or [])} regions to {args.file}", file=sys.stderr)
        else:
            entry.regions = read_regions_file(args.file)
            entry.mode = "boxes"
            lib.save(entry)
            print(f"imported {len(entry.regions)} regions into {entry.id}", file=sys.stderr)
        return 0

    if not args.input.exists():
        print(f"error: {args.input} does not exist", file=sys.stderr)
        return 2
    out_dir = args.out or args.input.with_name(args.input.stem + "_out")
    result = run(args.input, out_dir, _options_from_args(args), report=None if args.quiet else sys.stderr)
    return 0 if result.pieces else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
