from __future__ import annotations

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def align_reference(reference_rgb: np.ndarray, target_rgb: np.ndarray) -> np.ndarray:
    h, w = target_rgb.shape[:2]
    if reference_rgb.shape[:2] != (h, w):
        reference_rgb = cv2.resize(reference_rgb, (w, h), interpolation=cv2.INTER_AREA)
    if reference_rgb.ndim == 2:
        reference_rgb = cv2.cvtColor(reference_rgb, cv2.COLOR_GRAY2RGB)
    return reference_rgb


def mse(reference: np.ndarray, target: np.ndarray) -> float:
    ref = reference.astype(np.float64)
    tgt = target.astype(np.float64)
    return float(np.mean((ref - tgt) ** 2))


def quality_metrics(reference_rgb: np.ndarray | None, target_rgb: np.ndarray) -> dict[str, float | str]:
    if reference_rgb is None:
        return {"MSE": "Clean reference required", "PSNR": "Clean reference required", "SSIM": "Clean reference required"}
    reference_rgb = align_reference(reference_rgb, target_rgb)
    value_mse = mse(reference_rgb, target_rgb)
    if value_mse == 0:
        value_psnr = float("inf")
    else:
        value_psnr = float(peak_signal_noise_ratio(reference_rgb, target_rgb, data_range=255))
    value_ssim = float(structural_similarity(reference_rgb, target_rgb, channel_axis=2, data_range=255))
    return {"MSE": value_mse, "PSNR": value_psnr, "SSIM": value_ssim}


def comparison_metrics(source_rgb: np.ndarray, target_rgb: np.ndarray) -> dict[str, float]:
    target_rgb = align_reference(target_rgb, source_rgb)
    value_mse = mse(source_rgb, target_rgb)
    if value_mse == 0:
        value_psnr = float("inf")
    else:
        value_psnr = float(peak_signal_noise_ratio(source_rgb, target_rgb, data_range=255))
    value_ssim = float(structural_similarity(source_rgb, target_rgb, channel_axis=2, data_range=255))
    return {"MSE": value_mse, "PSNR": value_psnr, "SSIM": value_ssim}
