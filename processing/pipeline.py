from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from processing.edges import canny_edges
from processing.filters import FilterResult, apply_filter


@dataclass
class PipelineResult:
    name: str
    filter_result: FilterResult
    edges: np.ndarray
    filter_ms: float
    canny_ms: float

    @property
    def total_ms(self) -> float:
        return self.filter_ms + self.canny_ms


def run_pipeline(image_rgb: np.ndarray, name: str, filter_type: str, filter_params: dict, canny_params: dict) -> PipelineResult:
    result = apply_filter(image_rgb, filter_type, filter_params)
    edges, canny_ms = canny_edges(result.calculation, canny_params)
    return PipelineResult(
        name=name,
        filter_result=result,
        edges=edges,
        filter_ms=result.elapsed_ms,
        canny_ms=canny_ms,
    )
