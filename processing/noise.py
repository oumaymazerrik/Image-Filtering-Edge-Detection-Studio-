from __future__ import annotations

import numpy as np


def gaussian_noise(image: np.ndarray, mean: float, sigma: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.normal(mean, sigma, image.shape)
    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def salt_pepper_noise(
    image: np.ndarray, density: float, salt_ratio: float, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = image.copy()
    density = float(np.clip(density, 0.0, 1.0))
    salt_ratio = float(np.clip(salt_ratio, 0.0, 1.0))
    mask = rng.random(image.shape[:2])
    salt = mask < density * salt_ratio
    pepper = (mask >= density * salt_ratio) & (mask < density)
    out[salt] = 255
    out[pepper] = 0
    return out


def uniform_noise(image: np.ndarray, minimum: int, maximum: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    low = min(minimum, maximum)
    high = max(minimum, maximum)
    noise = rng.integers(low, high + 1, image.shape)
    return np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
