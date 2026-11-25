"""
Illumination correction (pipeline step 6).

Image formation model (multiplicative):
    I(x, y) = L(x, y) * R(x, y)
      L = slowly varying illumination (vignetting, uneven flash)
      R = retinal reflectance (vessels, MA, exudates ... what we want)
If B(x, y) is an estimate of L, the corrected image is
    I_c = I / B * mean(I in FOV)          (divisive correction)
The scale factor keeps the corrected image in the original grey-level range.

B is estimated with a filter MUCH larger than any lesion, so lesions are
"invisible" to it:
  * median filter   - robust: ignores small dark/bright structures (vessels, MA, exudates)
  * Gaussian filter - smooth, but averages dark/bright structures into B

Two practical details:
  1. Pixels outside the FOV are black. A plain filter near the rim would mix them in
     and underestimate B -> bright rim artefact. We therefore use normalised
     convolution (masked mean) and fill outside-FOV pixels before median filtering.
  2. B is low-frequency, so it is estimated on a 1/scale down-sampled copy and
     up-sampled again. Only B is resampled; the green channel keeps full resolution.
"""
import cv2
import numpy as np

from src import config


def _odd(value: float) -> int:
    value = max(3, int(round(value)))
    return value if value % 2 == 1 else value + 1


def masked_gaussian(image: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    """Normalised convolution: Gaussian mean of `image` using only pixels where mask == 1."""
    image = image.astype(np.float32)
    mask = mask.astype(np.float32)
    numerator = cv2.GaussianBlur(image * mask, (0, 0), sigma)
    denominator = cv2.GaussianBlur(mask, (0, 0), sigma)
    fallback = float(image[mask > 0].mean())
    return np.where(denominator > 1e-4, numerator / np.maximum(denominator, 1e-4), fallback).astype(np.float32)


def estimate_background(
    green: np.ndarray,
    fov: np.ndarray,
    method: str = config.ILLUM_METHOD,
    size: int = config.ILLUM_SIZE,
    scale: int = config.ILLUM_SCALE,
) -> np.ndarray:
    """Return background (illumination) estimate B at full resolution, float32."""
    height, width = green.shape
    small_size = (width // scale, height // scale)
    green_small = cv2.resize(green, small_size, interpolation=cv2.INTER_AREA).astype(np.float32)
    fov_small = (cv2.resize(fov.astype(np.float32), small_size, interpolation=cv2.INTER_AREA) > 0.5).astype(np.float32)
    size_small = size / scale

    if method == "gaussian":
        # sigma = size / 4  ->  +-2 sigma spans the nominal kernel size
        background_small = masked_gaussian(green_small, fov_small, sigma=size_small / 4)
    elif method == "median":
        # Fill outside-FOV pixels with a smooth masked mean so the median never sees black
        fill = masked_gaussian(green_small, fov_small, sigma=max(size_small, 50))
        filled = np.where(fov_small > 0, green_small, fill)
        background_small = cv2.medianBlur(np.clip(filled, 0, 255).astype(np.uint8), _odd(size_small)).astype(np.float32)
    else:
        raise ValueError(f"Unknown method: {method}")

    return cv2.resize(background_small, (width, height), interpolation=cv2.INTER_LINEAR)


def correct_illumination(green: np.ndarray, fov: np.ndarray, background: np.ndarray) -> np.ndarray:
    """
    Divisive correction, float32 output.
    Outside the FOV the value is set to the FOV mean (neutral grey) instead of black,
    so later morphological filters do not respond to the FOV border.
    """
    inside = fov > 0
    mean_level = float(green[inside].mean())
    corrected = green.astype(np.float32) / np.maximum(background, 1.0) * mean_level
    corrected[~inside] = mean_level
    return np.clip(corrected, 0, 255)


def illumination_correct(green: np.ndarray, fov: np.ndarray, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    """Convenience wrapper: returns (corrected, background)."""
    background = estimate_background(green, fov, **kwargs)
    return correct_illumination(green, fov, background), background
