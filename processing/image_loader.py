from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class ImageInfo:
    name: str
    path: str
    width: int
    height: int
    channels: int

    @property
    def resolution(self) -> str:
        return f"{self.width} x {self.height}"

    @property
    def dimensions(self) -> str:
        return f"{self.width * self.height:,} px"


def list_asset_images(images_dir: Path) -> list[ImageInfo]:
    images_dir.mkdir(parents=True, exist_ok=True)
    infos: list[ImageInfo] = []
    for path in sorted(images_dir.iterdir()):
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS or not path.is_file():
            continue
        try:
            with Image.open(path) as img:
                width, height = img.size
                channels = len(img.getbands())
        except Exception:
            continue
        infos.append(
            ImageInfo(
                name=path.name,
                path=str(path),
                width=width,
                height=height,
                channels=channels,
            )
        )
    return infos


def load_image_rgb(path: str | Path) -> np.ndarray:
    image_bgr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image_bgr is None:
        raise ValueError(f"Unable to read image: {path}")
    return normalize_to_rgb(image_bgr)


def normalize_to_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    if image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    raise ValueError("Unsupported image format.")


def to_gray(image_rgb: np.ndarray) -> np.ndarray:
    if image_rgb.ndim == 2:
        return image_rgb
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)


def describe_array(name: str, image: np.ndarray) -> ImageInfo:
    h, w = image.shape[:2]
    channels = 1 if image.ndim == 2 else image.shape[2]
    return ImageInfo(name=name, path="", width=w, height=h, channels=channels)
