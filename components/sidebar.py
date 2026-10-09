from __future__ import annotations

import streamlit as st

from processing.filters import BORDER_TYPES
from utils.validators import ensure_odd, valid_canny_thresholds


FILTER_TYPES = [
    "No Filter",
    "Mean Filter",
    "Gaussian Filter",
    "Median Filter",
    "Bilateral Filter",
    "Sobel Filter",
    "Laplacian Filter",
]


KERNEL_SIZE_OPTIONS = {
    "3 x 3": 3,
    "5 x 5": 5,
    "7 x 7": 7,
    "9 x 9": 9,
    "11 x 11": 11,
}


NOISE_GUIDE = {
    "Gaussian Noise": {
        "parameters": "Mean (mu), Sigma (sigma)",
        "role": "Mean controls the average noise value. Sigma controls the dispersion intensity.",
    },
    "Salt & Pepper": {
        "parameters": "Proportion",
        "role": "Percentage of pixels replaced by black or white values.",
    },
    "Uniform Noise": {
        "parameters": "Min / Max",
        "role": "Interval of random values added to the pixels.",
    },
}


def sidebar_controls(image_key: str) -> tuple[str, dict, dict, dict]:
    st.sidebar.header("Control Panel")
    state_key = f"params_{image_key}"
    if state_key not in st.session_state:
        st.session_state[state_key] = {}

    filter_type = st.sidebar.selectbox("Filter Type", FILTER_TYPES, key=f"filter_type_{image_key}")
    params = _filter_params(filter_type, image_key)
    canny = _canny_params(image_key)
    noise = _noise_params(image_key)
    return filter_type, params, canny, noise


def _filter_params(filter_type: str, key: str) -> dict:
    st.sidebar.subheader("Kernel Choices")
    params: dict = {}
    border_options = list(BORDER_TYPES.keys())
    kernel_labels = list(KERNEL_SIZE_OPTIONS.keys())

    if filter_type == "No Filter":
        pass

    elif filter_type == "Mean Filter":
        kernel_label = st.sidebar.selectbox("Kernel Size", kernel_labels, index=1, key=f"mean_k_{key}")
        params["kernel_width"] = KERNEL_SIZE_OPTIONS[kernel_label]
        params["kernel_height"] = KERNEL_SIZE_OPTIONS[kernel_label]
        params["border_type"] = st.sidebar.selectbox("Border Type", border_options, key=f"mean_border_{key}")

    elif filter_type == "Gaussian Filter":
        kernel_label = st.sidebar.selectbox("Kernel Size", kernel_labels, index=1, key=f"gauss_k_{key}")
        params["kernel_size"] = KERNEL_SIZE_OPTIONS[kernel_label]
        params["sigma_x"] = st.sidebar.slider("Sigma X", 0.1, 15.0, 1.0, 0.1, key=f"gauss_sx_{key}")
        params["sigma_y"] = st.sidebar.slider("Sigma Y", 0.1, 15.0, 1.0, 0.1, key=f"gauss_sy_{key}")
        params["border_type"] = st.sidebar.selectbox("Border Type", border_options, key=f"gauss_border_{key}")

    elif filter_type == "Median Filter":
        kernel_label = st.sidebar.selectbox("Kernel Size", kernel_labels, index=1, key=f"median_k_{key}")
        params["kernel_size"] = KERNEL_SIZE_OPTIONS[kernel_label]

    elif filter_type == "Bilateral Filter":
        kernel_label = st.sidebar.selectbox("Kernel Size", kernel_labels, index=3, key=f"bilat_d_{key}")
        params["diameter"] = KERNEL_SIZE_OPTIONS[kernel_label]
        params["sigma_color"] = st.sidebar.slider("Sigma Color", 1, 200, 75, 1, key=f"bilat_sc_{key}")
        params["sigma_space"] = st.sidebar.slider("Sigma Space", 1, 200, 75, 1, key=f"bilat_ss_{key}")

    elif filter_type == "Sobel Filter":
        params["sobel_mode"] = st.sidebar.radio(
            "Mode", ["Sobel X", "Sobel Y", "Sobel Magnitude"], index=2, horizontal=True, key=f"sobel_mode_{key}"
        )
        kernel_label = st.sidebar.selectbox("Kernel Size", kernel_labels, index=0, key=f"sobel_k_{key}")
        params["kernel_size"] = KERNEL_SIZE_OPTIONS[kernel_label]
        params["scale"] = st.sidebar.slider("Scale", 0.1, 10.0, 1.0, 0.1, key=f"sobel_scale_{key}")
        params["delta"] = st.sidebar.slider("Delta", -100.0, 100.0, 0.0, 1.0, key=f"sobel_delta_{key}")

    elif filter_type == "Laplacian Filter":
        kernel_label = st.sidebar.selectbox("Kernel Size", kernel_labels, index=0, key=f"lap_k_{key}")
        params["kernel_size"] = KERNEL_SIZE_OPTIONS[kernel_label]
        params["scale"] = st.sidebar.slider("Scale", 0.1, 10.0, 1.0, 0.1, key=f"lap_scale_{key}")
        params["delta"] = st.sidebar.slider("Delta", -100.0, 100.0, 0.0, 1.0, key=f"lap_delta_{key}")
        params["pre_smooth"] = st.sidebar.toggle("Gaussian pre-smoothing", value=False, key=f"lap_presmooth_{key}")

    for item in ("kernel_size", "gaussian_kernel"):
        if item in params:
            params[item] = ensure_odd(params[item], 1)
    return params


def _canny_params(key: str) -> dict:
    st.sidebar.subheader("Canny Edge Detection")
    low = st.sidebar.slider("Threshold Low", 0, 255, 50, 1, key=f"canny_low_{key}")
    high = st.sidebar.slider("Threshold High", 0, 255, 150, 1, key=f"canny_high_{key}")
    low, high = valid_canny_thresholds(low, high)
    l2_gradient = st.sidebar.toggle("L2 Gradient", value=True, key=f"canny_l2_{key}")
    return {
        "threshold_low": low,
        "threshold_high": high,
        "source_mode": "Filtered image",
        "overlay": True,
        "gaussian_preprocessing": False,
        "gaussian_kernel": 5,
        "gaussian_sigma": 1.0,
        "l2_gradient": l2_gradient,
    }


def _noise_params(key: str) -> dict:
    st.sidebar.subheader("Noise Generator")
    enabled = st.sidebar.toggle("Enable Noise Generation", value=False, key=f"noise_enabled_{key}")
    params = {"enabled": enabled}
    with st.sidebar.expander("Noise types and roles", expanded=False):
        st.markdown(
            """
            | Type de bruit | Parametres | Role |
            | --- | --- | --- |
            | Gaussian Noise | Mean (mu), Sigma (sigma) | Moyenne du bruit et intensite de dispersion |
            | Salt & Pepper | Proportion | Pourcentage de pixels remplaces par du noir ou du blanc |
            | Uniform Noise | Min / Max | Intervalle des valeurs aleatoires ajoutees aux pixels |
            """
        )
    if not enabled:
        st.sidebar.caption("Disabled by default because the asset images are already noisy.")
        return params

    params["type"] = st.sidebar.selectbox("Type de bruit", list(NOISE_GUIDE.keys()), key=f"noise_type_{key}")
    st.sidebar.caption(NOISE_GUIDE[params["type"]]["role"])
    params["seed"] = st.sidebar.number_input("Random Seed", min_value=0, max_value=999999, value=42, step=1, key=f"noise_seed_{key}")
    if st.sidebar.button("Regenerate Noise", use_container_width=True, key=f"noise_regen_{key}"):
        st.session_state[f"noise_nonce_{key}"] = st.session_state.get(f"noise_nonce_{key}", 0) + 1
    params["nonce"] = st.session_state.get(f"noise_nonce_{key}", 0)

    if params["type"] == "Gaussian Noise":
        params["mean"] = st.sidebar.slider("Mean (mu)", -50.0, 50.0, 0.0, 1.0, key=f"noise_mean_{key}")
        st.sidebar.caption("Moyenne du bruit ajoute aux pixels.")
        params["sigma"] = st.sidebar.slider("Sigma (sigma)", 0.0, 100.0, 20.0, 1.0, key=f"noise_sigma_{key}")
        st.sidebar.caption("Intensite de la dispersion du bruit.")
    elif params["type"] == "Salt & Pepper":
        proportion = st.sidebar.slider("Proportion (%)", 0.0, 100.0, 4.0, 0.5, key=f"noise_proportion_{key}")
        params["density"] = proportion / 100.0
        st.sidebar.caption("Pourcentage de pixels remplaces aleatoirement par noir ou blanc.")
        params["salt_ratio"] = st.sidebar.slider("White / Black Balance", 0.0, 1.0, 0.5, 0.05, key=f"noise_ratio_{key}")
        st.sidebar.caption("0.5 applique autant de pixels blancs que noirs.")
    else:
        params["minimum"] = st.sidebar.slider("Min Value", -100, 0, -20, 1, key=f"noise_min_{key}")
        params["maximum"] = st.sidebar.slider("Max Value", 0, 100, 20, 1, key=f"noise_max_{key}")
        st.sidebar.caption("Intervalle des valeurs aleatoires ajoutees aux pixels.")
    return params
