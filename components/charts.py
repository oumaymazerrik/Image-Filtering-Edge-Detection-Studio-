from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def line_histogram(frame: pd.DataFrame, title: str) -> None:
    fig = px.line(frame, x="Intensity", y="Count", color="Image", title=title)
    fig.update_layout(template="plotly_dark", height=330, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)


def timing_bar(frame: pd.DataFrame) -> None:
    melted = frame.melt(id_vars=["Pipeline"], value_vars=["Filter time (ms)", "Canny time (ms)"], var_name="Stage", value_name="Milliseconds")
    fig = px.bar(melted, x="Pipeline", y="Milliseconds", color="Stage", barmode="group", title="Execution Time")
    fig.update_layout(template="plotly_dark", height=340, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)


def kernel_heatmap(kernel: np.ndarray, title: str = "Kernel Heatmap") -> None:
    fig = px.imshow(kernel, text_auto=".4f", color_continuous_scale="Viridis", title=title, aspect="auto")
    fig.update_layout(template="plotly_dark", height=420, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)


def gaussian_surface(kernel: np.ndarray) -> None:
    fig = go.Figure(data=[go.Surface(z=kernel, colorscale="Viridis")])
    fig.update_layout(template="plotly_dark", title="Gaussian Weight Surface", height=430, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)
