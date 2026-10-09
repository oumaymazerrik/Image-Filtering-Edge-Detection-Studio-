from __future__ import annotations

import streamlit as st

from components.image_viewer import framed_image
from processing.pipeline import PipelineResult


def render_pipeline_columns(results: list[PipelineResult]) -> None:
    columns = st.columns(len(results))
    for column, result in zip(columns, results):
        with column:
            with st.container(border=True):
                st.subheader(result.name)
                st.caption(
                    f"Filter: {result.filter_ms:.2f} ms | Canny: {result.canny_ms:.2f} ms | Total: {result.total_ms:.2f} ms"
                )
                st.image(framed_image(result.filter_result.display, width=640, height=360), caption="Filtered image", use_container_width=True)
                st.image(framed_image(result.edges, width=640, height=360), caption="Canny edges", use_container_width=True)
