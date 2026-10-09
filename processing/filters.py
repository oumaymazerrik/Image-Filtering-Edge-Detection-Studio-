from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

from processing.image_loader import to_gray


BORDER_TYPES = {
    "BORDER_DEFAULT": cv2.BORDER_DEFAULT,
    "BORDER_REFLECT": cv2.BORDER_REFLECT,
    "BORDER_REPLICATE": cv2.BORDER_REPLICATE,
    "BORDER_CONSTANT": cv2.BORDER_CONSTANT,
}


@dataclass
class FilterResult:
    filtered: np.ndarray
    display: np.ndarray
    calculation: np.ndarray
    elapsed_ms: float
    details: dict


def normalize_for_display(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    return normalized.astype(np.uint8)


def signed_abs_uint8(image: np.ndarray) -> np.ndarray:
    return cv2.convertScaleAbs(image)


def apply_filter(image_rgb: np.ndarray, filter_type: str, params: dict) -> FilterResult:
    start = time.perf_counter()
    details: dict = {}

    if filter_type == "No Filter":
        filtered = image_rgb.copy()
        calculation = filtered
        display = filtered

    elif filter_type == "Mean Filter":
        k_w = int(params.get("kernel_width", 5))
        k_h = int(params.get("kernel_height", 5))
        border = BORDER_TYPES[params.get("border_type", "BORDER_DEFAULT")]
        filtered = cv2.blur(image_rgb, (k_w, k_h), borderType=border)
        calculation = filtered
        display = filtered
        details = {"kernel": mean_kernel(k_w, k_h)}

    elif filter_type == "Gaussian Filter":
        k = int(params.get("kernel_size", 5))
        sigma_x = float(params.get("sigma_x", 1.0))
        sigma_y = float(params.get("sigma_y", 1.0))
        border = BORDER_TYPES[params.get("border_type", "BORDER_DEFAULT")]
        filtered = cv2.GaussianBlur(image_rgb, (k, k), sigma_x, sigmaY=sigma_y, borderType=border)
        calculation = filtered
        display = filtered
        details = {"kernel": gaussian_kernel(k, sigma_x, sigma_y)}

    elif filter_type == "Median Filter":
        k = int(params.get("kernel_size", 5))
        filtered = cv2.medianBlur(image_rgb, k)
        calculation = filtered
        display = filtered
        details = {"median_example": median_example(k)}

    elif filter_type == "Bilateral Filter":
        d = int(params.get("diameter", 9))
        sigma_color = float(params.get("sigma_color", 75))
        sigma_space = float(params.get("sigma_space", 75))
        filtered = cv2.bilateralFilter(image_rgb, d, sigma_color, sigma_space)
        calculation = filtered
        display = filtered

    elif filter_type == "Sobel Filter":
        gray = to_gray(image_rgb)
        k = int(params.get("kernel_size", 3))
        scale = float(params.get("scale", 1.0))
        delta = float(params.get("delta", 0.0))
        mode = params.get("sobel_mode", "Sobel Magnitude")
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=k, scale=scale, delta=delta)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=k, scale=scale, delta=delta)
        if mode == "Sobel X":
            calculation = grad_x
        elif mode == "Sobel Y":
            calculation = grad_y
        else:
            calculation = cv2.magnitude(grad_x, grad_y)
        display = cv2.cvtColor(signed_abs_uint8(calculation), cv2.COLOR_GRAY2RGB)
        filtered = calculation
        details = {
            "grad_x": signed_abs_uint8(grad_x),
            "grad_y": signed_abs_uint8(grad_y),
            "kernel_x": sobel_kernel(k, "x"),
            "kernel_y": sobel_kernel(k, "y"),
        }

    elif filter_type == "Laplacian Filter":
        gray = to_gray(image_rgb)
        k = int(params.get("kernel_size", 3))
        scale = float(params.get("scale", 1.0))
        delta = float(params.get("delta", 0.0))
        pre_smooth = bool(params.get("pre_smooth", False))
        source = gray
        if pre_smooth:
            source = cv2.GaussianBlur(gray, (5, 5), 1.0)
        lap = cv2.Laplacian(source, cv2.CV_64F, ksize=k, scale=scale, delta=delta)
        calculation = lap
        display = cv2.cvtColor(signed_abs_uint8(lap), cv2.COLOR_GRAY2RGB)
        filtered = lap
        details = {"kernel": laplacian_kernel(k), "pre_smooth": pre_smooth}

    else:
        raise ValueError(f"Unknown filter: {filter_type}")

    elapsed_ms = (time.perf_counter() - start) * 1000
    return FilterResult(filtered=filtered, display=display, calculation=calculation, elapsed_ms=elapsed_ms, details=details)


def mean_kernel(width: int, height: int) -> np.ndarray:
    return np.ones((height, width), dtype=np.float64) / float(width * height)


def gaussian_kernel(size: int, sigma_x: float, sigma_y: float | None = None) -> np.ndarray:
    sigma_y = sigma_x if sigma_y in (None, 0) else sigma_y
    kx = cv2.getGaussianKernel(size, sigma_x)
    ky = cv2.getGaussianKernel(size, sigma_y)
    return ky @ kx.T


def median_example(size: int) -> dict:
    values = np.arange(size * size)
    shuffled = values[::-1]
    return {
        "neighborhood": shuffled.reshape(size, size),
        "sorted_values": np.sort(shuffled),
        "median": int(np.median(shuffled)),
    }


def sobel_kernel(size: int, axis: str) -> np.ndarray | None:
    if size == 1:
        return np.array([[-1, 0, 1]], dtype=np.float64) if axis == "x" else np.array([[-1], [0], [1]], dtype=np.float64)
    if size not in {3, 5, 7}:
        return None
    kx, ky = cv2.getDerivKernels(1 if axis == "x" else 0, 0 if axis == "x" else 1, size, normalize=False)
    return ky @ kx.T


def laplacian_kernel(size: int) -> np.ndarray | None:
    if size == 1:
        return np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
    if size == 3:
        return np.array([[2, 0, 2], [0, -8, 0], [2, 0, 2]], dtype=np.float64)
    return None
