# VisionLab Pro

Advanced Image Filtering & Edge Detection Studio is a local Streamlit application for OpenCV-based image filtering, Canny edge detection, visual comparison, kernel inspection, and metrics.

## Features

- Automatic detection of images in `assets/images/`
- Fixed three-image selection from `assets/images/`
- Per-image filter and Canny controls
- Mean, Gaussian, Median, Bilateral, Sobel, and Laplacian filters using real OpenCV functions
- Canny edge detection from either the filtered output or original input
- Optional noise generator with Gaussian, Salt & Pepper, and Uniform controls
- Gaussian, Median, and Bilateral pipeline comparison
- Kernel explorer with matrices, heatmaps, and Gaussian 3D surface
- Difference maps, edge overlays, histograms, and timing tables
- Reference-only MSE, PSNR, and SSIM metrics

## Installation

Use Python 3.11 or another compatible Python 3 version.

```bash
cd visionlab-pro
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Launch

```bash
streamlit run app.py
```

The app opens in your browser. The three provided noisy images are already copied into `assets/images/` and are displayed as equal-size selection cards.

## Image Rules

The application never overwrites source files. Images loaded from `assets/images/` stay unchanged, and all filters operate on one of the three fixed input images. The optional noise generator is off by default and applies noise only in memory.

## Noise Generator

| Noise type | Parameters | Role |
| --- | --- | --- |
| Gaussian Noise | Mean, Sigma | Mean controls the average noise value. Sigma controls the dispersion intensity. |
| Salt & Pepper | Proportion | Percentage of pixels replaced by black or white. |
| Uniform Noise | Min / Max | Interval of random values added to the pixels. |

Noise uses a stable seed plus a manual regenerate action, so changing filters or Canny thresholds does not unexpectedly create a new random pattern.

## Filters

- **Mean Filter**: `cv2.blur(image, (kernel_width, kernel_height))`
- **Gaussian Filter**: `cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX, sigmaY=sigmaY)`
- **Median Filter**: `cv2.medianBlur(image, kernel_size)`
- **Bilateral Filter**: `cv2.bilateralFilter(image, d, sigmaColor, sigmaSpace)`
- **Sobel Filter**: `cv2.Sobel(image, cv2.CV_64F, dx, dy, ksize=kernel_size)`
- **Laplacian Filter**: `cv2.Laplacian(image, cv2.CV_64F, ksize=kernel_size)`
- **Canny**: `cv2.Canny(image, threshold1, threshold2, L2gradient=True)`

Sobel and Laplacian keep signed or floating-point calculation data internally, then convert a display-safe preview with OpenCV scaling.

## Metrics

MSE, PSNR, and SSIM are shown only when a clean reference is available in code. In the current interface, external file actions are disabled, so the app displays:

```text
Clean reference unavailable for MSE, PSNR and SSIM in this interface.
```

Execution time is measured with `time.perf_counter()` for filtering and Canny separately.
