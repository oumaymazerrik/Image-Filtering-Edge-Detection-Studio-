from __future__ import annotations

import time

import cv2
import numpy as np

from processing.filters import normalize_for_display
from processing.image_loader import to_gray


def canny_edges(source: np.ndarray, params: dict) -> tuple[np.ndarray, float]:
    start = time.perf_counter()
    if source.ndim == 3:
        gray = to_gray(source.astype(np.uint8))
    else:
        gray = normalize_for_display(source)

    if bool(params.get("gaussian_preprocessing", False)):
        k = int(params.get("gaussian_kernel", 5))
        sigma = float(params.get("gaussian_sigma", 1.0))
        gray = cv2.GaussianBlur(gray, (k, k), sigma)

    low = int(params.get("threshold_low", 50))
    high = int(params.get("threshold_high", 150))
    if low > high:
        low = high
    edges = cv2.Canny(gray, low, high, L2gradient=bool(params.get("l2_gradient", True)))
    elapsed_ms = (time.perf_counter() - start) * 1000
    return edges, elapsed_ms


def edge_overlay(image_rgb: np.ndarray, edges: np.ndarray, color=(0, 255, 255), alpha: float = 0.85) -> np.ndarray:
    base = image_rgb.copy()
    overlay = base.copy()
    overlay[edges > 0] = color
    return cv2.addWeighted(overlay, alpha, base, 1 - alpha, 0)


def absolute_difference(input_rgb: np.ndarray, filtered_display: np.ndarray) -> np.ndarray:
    if filtered_display.ndim == 2:
        filtered_display = cv2.cvtColor(filtered_display, cv2.COLOR_GRAY2RGB)
    return cv2.absdiff(input_rgb.astype(np.uint8), filtered_display.astype(np.uint8))
