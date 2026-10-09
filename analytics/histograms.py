from __future__ import annotations

import cv2
import numpy as np
import pandas as pd


def grayscale_histogram(image_rgb: np.ndarray, label: str) -> pd.DataFrame:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY) if image_rgb.ndim == 3 else image_rgb
    hist = cv2.calcHist([gray.astype(np.uint8)], [0], None, [256], [0, 256]).flatten()
    return pd.DataFrame({"Intensity": np.arange(256), "Count": hist, "Image": label})
