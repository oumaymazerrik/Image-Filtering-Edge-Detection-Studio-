from __future__ import annotations

import io
import json

import cv2
import numpy as np
import pandas as pd


def image_to_png_bytes(image: np.ndarray) -> bytes:
    if image.ndim == 3:
        encoded_source = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2BGR)
    else:
        encoded_source = image.astype(np.uint8)
    ok, buffer = cv2.imencode(".png", encoded_source)
    if not ok:
        raise ValueError("Could not encode image as PNG.")
    return buffer.tobytes()


def json_bytes(payload: dict) -> bytes:
    return json.dumps(payload, indent=2, default=str).encode("utf-8")


def dataframe_csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8")


def dataframe_xlsx_bytes(frame: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        frame.to_excel(writer, index=False, sheet_name="results")
    return output.getvalue()
