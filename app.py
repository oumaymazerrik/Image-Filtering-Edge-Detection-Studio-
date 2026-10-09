from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from analytics.benchmarks import timing_table
from analytics.histograms import grayscale_histogram
from analytics.metrics import comparison_metrics
from components.charts import line_histogram, timing_bar
from components.comparison import render_pipeline_columns
from components.image_viewer import image_card
from components.sidebar import sidebar_controls
from processing.edges import absolute_difference, canny_edges, edge_overlay
from processing.filters import apply_filter
from processing.image_loader import list_asset_images, load_image_rgb, to_gray
from processing.noise import gaussian_noise, salt_pepper_noise, uniform_noise
from processing.pipeline import run_pipeline


ROOT = Path(__file__).parent
IMAGES_DIR = ROOT / "assets" / "images"


st.set_page_config(
    page_title="VisionLab Pro",
    page_icon="VL",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_css() -> None:
    css_path = ROOT / "styles" / "custom.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def session_defaults() -> None:
    st.session_state.setdefault("selected_asset", None)
    st.session_state.setdefault("reference_image", None)
    st.session_state.setdefault("reference_name", None)


@st.cache_data(show_spinner=False)
def cached_load(path: str) -> np.ndarray:
    return load_image_rgb(path)


@st.cache_data(show_spinner=False)
def cached_filter(image: np.ndarray, filter_type: str, params: dict):
    return apply_filter(image, filter_type, params)


@st.cache_data(show_spinner=False)
def cached_canny(source: np.ndarray, params: dict):
    return canny_edges(source, params)


def selection_thumbnail(image: np.ndarray, width: int = 520, height: int = 300) -> np.ndarray:
    canvas = np.full((height, width, 3), (9, 22, 38), dtype=np.uint8)
    src = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    h, w = src.shape[:2]
    scale = min(width / w, height / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = cv2.resize(src, (new_w, new_h), interpolation=cv2.INTER_AREA)
    x = (width - new_w) // 2
    y = (height - new_h) // 2
    canvas[y : y + new_h, x : x + new_w] = resized
    return canvas


def image_selector(asset_infos) -> tuple[str, np.ndarray]:
    st.subheader("Image Selection")
    choices = [(f"Image {index}", info.name, info) for index, info in enumerate(asset_infos[:3], start=1)]

    if len(choices) < 3:
        st.warning("Place exactly three images in assets/images to use VisionLab Pro.")
        st.stop()

    valid_keys = {name for _, name, _ in choices}
    if st.session_state.selected_asset not in valid_keys:
        st.session_state.selected_asset = choices[0][1]

    columns = st.columns(3)
    selected_name = ""
    selected_image = None
    for column, (label, name, info) in zip(columns, choices):
        key = name
        image = cached_load(info.path)
        selected = st.session_state.selected_asset == key
        with column:
            css_class = "image-select-card selected" if selected else "image-select-card"
            st.markdown(f'<div class="{css_class}">', unsafe_allow_html=True)
            st.image(selection_thumbnail(image), use_container_width=True)
            st.markdown(f"**{label}**")
            if st.button("Select", key=f"select_{key}", use_container_width=True):
                st.session_state.selected_asset = key
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
        if selected:
            selected_name = name
            selected_image = image

    if selected_image is None:
        label, name, info = choices[0]
        selected_name = name
        selected_image = cached_load(info.path)
    return selected_name, selected_image


def apply_optional_noise(image: np.ndarray, noise_params: dict, image_key: str) -> np.ndarray:
    if not noise_params.get("enabled"):
        return image
    seed = int(noise_params.get("seed", 42)) + int(noise_params.get("nonce", 0)) * 1009
    if noise_params["type"] == "Gaussian Noise":
        return gaussian_noise(image, noise_params.get("mean", 0.0), noise_params.get("sigma", 20.0), seed)
    if noise_params["type"] == "Salt & Pepper":
        return salt_pepper_noise(image, noise_params.get("density", 0.04), noise_params.get("salt_ratio", 0.5), seed)
    return uniform_noise(image, noise_params.get("minimum", -20), noise_params.get("maximum", 20), seed)


def noise_caption(noise_params: dict) -> str:
    if not noise_params.get("enabled"):
        return "No generated noise"
    if noise_params["type"] == "Gaussian Noise":
        return f"Gaussian Noise | Mean: {noise_params['mean']:.1f} | Sigma: {noise_params['sigma']:.1f}"
    if noise_params["type"] == "Salt & Pepper":
        return f"Salt & Pepper | Proportion: {noise_params['density'] * 100:.1f}%"
    return f"Uniform Noise | Min: {noise_params['minimum']} | Max: {noise_params['maximum']}"


def processing_studio(image_name, source_image, filter_type, filter_params, canny_params, noise_params, result, edges, canny_ms):
    gray = to_gray(source_image)
    overlay = edge_overlay(source_image, edges)
    diff = absolute_difference(source_image, result.display)

    st.subheader("Processing Studio")
    st.caption(
        f"{noise_caption(noise_params)}. Canny is calculated from the filtered result. "
        f"Filter time: {result.elapsed_ms:.2f} ms. Canny time: {canny_ms:.2f} ms."
    )
    row1 = st.columns(4)
    with row1[0]:
        image_card("A - Input Image", source_image, f"{image_name} | {source_image.shape[1]} x {source_image.shape[0]}")
    with row1[1]:
        image_card("B - Grayscale / Preprocessing", gray, "Grayscale conversion")
    with row1[2]:
        image_card("C - Filtered Image", result.display, filter_type)
    with row1[3]:
        image_card("D - Canny Edge Detection", edges, "White edges on black background")

    st.subheader("Advanced Visualization")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        image_card("Before / After - Filtered", result.display, "Filtered result")
    with col_b:
        image_card("Difference Map", diff, "Absolute difference")
    with col_c:
        image_card("Canny Overlay", overlay, "Contours superposes a l'image")


def comparison_tab(source_image, canny_params):
    st.subheader("Filter Comparison")
    st.caption("All pipelines use the same selected input image and shared Canny thresholds by default.")
    c1, c2, c3 = st.columns(3)
    with c1:
        g_sigma = st.slider("Gaussian sigma", 0.1, 10.0, 1.0, 0.1)
    with c2:
        st.caption("Median pipeline")
    with c3:
        b_sc = st.slider("Bilateral sigma color", 1, 200, 75)
        b_ss = st.slider("Bilateral sigma space", 1, 200, 75)

    clean_canny = {k: v for k, v in canny_params.items() if k not in {"source_mode", "overlay"}}
    results = [
        run_pipeline(source_image, "Gaussian + Canny", "Gaussian Filter", {"kernel_size": 5, "sigma_x": g_sigma, "sigma_y": g_sigma, "border_type": "BORDER_DEFAULT"}, clean_canny),
        run_pipeline(source_image, "Median + Canny", "Median Filter", {"kernel_size": 5}, clean_canny),
        run_pipeline(source_image, "Bilateral + Canny", "Bilateral Filter", {"diameter": 9, "sigma_color": b_sc, "sigma_space": b_ss}, clean_canny),
    ]
    render_pipeline_columns(results)
    table = timing_table(results)
    st.dataframe(table, use_container_width=True, hide_index=True)
    timing_bar(table)
    return results, table


def analytics_tab(source_image, result, edges, canny_ms, comparison_results) -> None:
    st.subheader("Analytics & Metrics")
    filtered_display = result.display if result.display.ndim == 3 else cv2.cvtColor(result.display, cv2.COLOR_GRAY2RGB)
    metrics = comparison_metrics(source_image, filtered_display)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("MSE Input vs Filtered", f"{metrics['MSE']:.4f}")
    m2.metric("PSNR Input vs Filtered", "inf" if metrics["PSNR"] == float("inf") else f"{metrics['PSNR']:.4f}")
    m3.metric("SSIM Input vs Filtered", f"{metrics['SSIM']:.4f}")
    m4.metric("Canny time", f"{canny_ms:.2f} ms")

    hist = pd.concat(
        [grayscale_histogram(source_image, "Input"), grayscale_histogram(filtered_display, "Filtered")],
        ignore_index=True,
    )
    line_histogram(hist, "Histogram Comparison")

    rows = [
        {
            "Pipeline": "Active filter + Canny",
            "MSE input-filtered": round(metrics["MSE"], 4),
            "PSNR input-filtered": "inf" if metrics["PSNR"] == float("inf") else round(metrics["PSNR"], 4),
            "SSIM input-filtered": round(metrics["SSIM"], 4),
            "Filter time (ms)": round(result.elapsed_ms, 3),
            "Canny time (ms)": round(canny_ms, 3),
            "Total time (ms)": round(result.elapsed_ms + canny_ms, 3),
        }
    ]
    if comparison_results:
        rows.extend(timing_table(comparison_results).to_dict("records"))
    table = pd.DataFrame(rows)
    st.dataframe(table, use_container_width=True, hide_index=True)
    timing_bar(table)


def main() -> None:
    load_css()
    session_defaults()
    asset_infos = list_asset_images(IMAGES_DIR)
    image_name, original_image = image_selector(asset_infos)
    image_key = image_name.replace(".", "_").replace(" ", "_")
    filter_type, filter_params, canny_params, noise_params = sidebar_controls(image_key)
    source_image = apply_optional_noise(original_image, noise_params, image_key)

    result = cached_filter(source_image, filter_type, filter_params)
    canny_source = result.calculation
    edges, canny_ms = cached_canny(canny_source, {k: v for k, v in canny_params.items() if k not in {"source_mode", "overlay"}})

    tabs = st.tabs(["Processing Studio", "Filter Comparison", "Analytics & Metrics"])
    comparison_results = []
    with tabs[0]:
        processing_studio(image_name, source_image, filter_type, filter_params, canny_params, noise_params, result, edges, canny_ms)
    with tabs[1]:
        comparison_results, _ = comparison_tab(source_image, canny_params)
    with tabs[2]:
        analytics_tab(source_image, result, edges, canny_ms, comparison_results)

if __name__ == "__main__":
    main()
