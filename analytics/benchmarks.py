from __future__ import annotations

import pandas as pd

from processing.pipeline import PipelineResult


def timing_table(results: list[PipelineResult]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Pipeline": result.name,
                "Filter time (ms)": round(result.filter_ms, 3),
                "Canny time (ms)": round(result.canny_ms, 3),
                "Total time (ms)": round(result.total_ms, 3),
            }
            for result in results
        ]
    )
