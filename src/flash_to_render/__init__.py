"""flash-to-render: tattoo flash sheets -> per-design 3D meshes and ink cutouts."""

__version__ = "0.1.0"

from .mesh import Mesh, extrude  # noqa: E402
from .pipeline import PipelineOptions, RunResult, run  # noqa: E402
from .segment import Box, Piece, Region, SegmentOptions, crop_pieces, detect_boxes, segment_sheet  # noqa: E402
from .trace import PolygonWithHoles, TraceOptions, trace_ink  # noqa: E402

__all__ = [
    "__version__",
    "Mesh",
    "extrude",
    "PipelineOptions",
    "RunResult",
    "run",
    "Box",
    "Region",
    "Piece",
    "SegmentOptions",
    "crop_pieces",
    "detect_boxes",
    "segment_sheet",
    "PolygonWithHoles",
    "TraceOptions",
    "trace_ink",
]
