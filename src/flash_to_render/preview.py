"""Tiny local server for the 3D preview page (``flash-to-render preview out/``).

Serves the output folder as static files plus ``preview.html`` at ``/``. The
page loads Three.js from a CDN and reads ``manifest.json`` and the GLBs.
"""

from __future__ import annotations

import functools
import http.server
import sys
import threading
import webbrowser
from importlib import resources
from pathlib import Path

__all__ = ["serve", "preview_html"]


def preview_html() -> bytes:
    return resources.files(__package__).joinpath("preview.html").read_bytes()


class _Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 (stdlib name)
        if self.path.split("?", 1)[0] in ("/", "/index.html"):
            body = preview_html()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, fmt: str, *args) -> None:  # keep the terminal quiet
        return


def serve(out_dir: str | Path, port: int = 8765, open_browser: bool = True) -> int:
    out_dir = Path(out_dir)
    if not (out_dir / "manifest.json").exists():
        print(f"error: {out_dir} has no manifest.json (run `flash-to-render convert` first)", file=sys.stderr)
        return 2
    handler = functools.partial(_Handler, directory=str(out_dir))
    with http.server.ThreadingHTTPServer(("127.0.0.1", port), handler) as httpd:
        url = f"http://127.0.0.1:{httpd.server_address[1]}/"
        print(f"preview: {url}  (ctrl-c to stop)", file=sys.stderr)
        if open_browser:
            threading.Timer(0.3, webbrowser.open, args=(url,)).start()
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
    return 0
