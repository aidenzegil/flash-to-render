from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from flash_to_render.pipeline import PipelineOptions, RunResult, run

FIXTURES = Path(__file__).resolve().parent.parent / "examples"
SHEETS = ("anime", "objects", "spiderverse")


@pytest.fixture(scope="session")
def sheet_results(tmp_path_factory: pytest.TempPathFactory) -> dict[str, RunResult]:
    """Run the full pipeline once per fixture sheet for the whole session."""
    out = {}
    for name in SHEETS:
        out[name] = run(FIXTURES / f"{name}.webp", tmp_path_factory.mktemp(name), PipelineOptions(debug=True), report=None)
    return out


def disk_mask(size: int = 200, radius: int = 80, center: tuple[int, int] | None = None) -> np.ndarray:
    """Boolean ink mask of a filled disk."""
    yy, xx = np.mgrid[:size, :size]
    cx, cy = center or (size // 2, size // 2)
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2


def donut_mask(size: int = 200, outer: int = 80, inner: int = 40) -> np.ndarray:
    return disk_mask(size, outer) & ~disk_mask(size, inner)


def island_in_hole_mask(size: int = 240) -> np.ndarray:
    """A ring with a smaller solid disk floating inside its hole."""
    return donut_mask(size, 100, 60) | disk_mask(size, 25)
