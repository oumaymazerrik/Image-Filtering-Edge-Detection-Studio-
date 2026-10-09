from __future__ import annotations

import cv2
import numpy as np
import streamlit as st


def framed_image(image, width: int = 720, height: int = 430) -> np.ndarray:
    canvas = np.full((height, width, 3), (8, 18, 32), dtype=np.uint8)
    source = np.asarray(image)
    if source.ndim == 2:
        source = cv2.cvtColor(source.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    else:
        source = source.astype(np.uint8)

    src_h, src_w = source.shape[:2]
    scale = min(width / src_w, height / src_h)
    target_w = max(1, int(src_w * scale))
    target_h = max(1, int(src_h * scale))
    resized = cv2.resize(source, (target_w, target_h), interpolation=cv2.INTER_AREA)
    x = (width - target_w) // 2
    y = (height - target_h) // 2
    canvas[y : y + target_h, x : x + target_w] = resized
    return canvas


def image_card(title: str, image, caption: str, filename: str | None = None, clamp_width: bool = True) -> None:
    with st.container(border=True):
        st.markdown(f"**{title}**")
        st.image(framed_image(image), caption=caption, use_container_width=clamp_width)
