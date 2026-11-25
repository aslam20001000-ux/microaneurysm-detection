"""
Field-of-View (FOV) mask extraction (pipeline step 4).

The fundus camera records a bright circular retina on a black background.
Every later step (CLAHE, black-hat, thresholding) must ignore the background,
otherwise the sharp retina/background edge produces false detections.

Algorithm:
  1. Red channel      - brightest channel, largest retina/background gap
  2. Global threshold - red > T  (background is <= 9 on all training images)
  3. Opening          - removes tiny bright specks / halo threads
  4. Largest component- keeps the retina only
  5. Hole filling     - dark structures inside the retina stay inside the FOV
  6. Erosion          - optional: trims the dark vignetted rim
"""
import cv2
import numpy as np

from src import config


def disk(radius: int) -> np.ndarray:
    """Circular (elliptical) structuring element of the given radius."""
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))


def largest_component(binary: np.ndarray) -> np.ndarray:
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n_labels <= 1:
        return np.zeros_like(binary)
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == biggest).astype(np.uint8)


def fill_holes(binary: np.ndarray) -> np.ndarray:
    """Fill background regions not connected to the image border."""
    padded = cv2.copyMakeBorder(binary, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    flood = padded.copy()
    mask = np.zeros((flood.shape[0] + 2, flood.shape[1] + 2), np.uint8)
    cv2.floodFill(flood, mask, (0, 0), 1)          # everything reachable from outside
    holes = (flood == 0)
    padded[holes] = 1
    return padded[1:-1, 1:-1]


def extract_fov_mask(
    image_bgr: np.ndarray,
    threshold: int = config.FOV_THRESHOLD,
    open_px: int = config.FOV_OPEN_PX,
    erode_px: int = config.FOV_ERODE_PX,
    return_steps: bool = False,
):
    """Return binary FOV mask (uint8 0/1). With return_steps=True also return intermediates."""
    red = image_bgr[:, :, 2]
    thresholded = (red > threshold).astype(np.uint8)
    opened = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, disk(open_px)) if open_px > 0 else thresholded
    main = largest_component(opened)
    filled = fill_holes(main)
    # Erosion: OpenCV pads with +inf for erode, so the clipped top/bottom image
    # borders are NOT eroded - only the real circular rim is trimmed.
    fov = cv2.erode(filled, disk(erode_px)) if erode_px > 0 else filled
    if return_steps:
        return fov, {"red": red, "thresholded": thresholded, "opened": opened, "filled": filled, "fov": fov}
    return fov
