"""
Noise reduction (pipeline step 8).

The risk is specific to this project: MA are TINY (smallest GT lesion = 13 px area,
~4 px diameter). Any filter whose support is comparable to the lesion averages the
dark dot with its bright surroundings and makes it shallower (or deletes it).

Filters
  gaussian  : weighted local mean, sigma in px. Linear; blurs edges and small dots alike.
  median    : k x k median. Removes impulse noise and JPEG block corners, but deletes
              any dark dot occupying less than half of the window.
  bilateral : Gaussian in space x Gaussian in intensity. Pixels that differ a lot in grey
              level (an MA vs its background) are averaged less -> edge/dot preserving.
              sigma_color is in grey levels: it must be BELOW the MA contrast to protect it.
"""
import cv2
import numpy as np


def denoise(image: np.ndarray, method: str, param: float, param2: float = 0) -> np.ndarray:
    """Apply one filter to a uint8 single-channel image. Returns uint8."""
    if method == "none":
        return image.copy()
    if method == "gaussian":
        return cv2.GaussianBlur(image, (0, 0), sigmaX=float(param))
    if method == "median":
        return cv2.medianBlur(image, int(param))
    if method == "bilateral":
        # param = sigma_color (grey levels), param2 = sigma_space (px); d=-1 -> derived from sigma_space
        return cv2.bilateralFilter(image, d=-1, sigmaColor=float(param), sigmaSpace=float(param2))
    raise ValueError(f"Unknown method: {method}")
