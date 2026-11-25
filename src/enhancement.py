"""
Contrast enhancement (pipeline step 7): CLAHE.

CLAHE = Contrast Limited Adaptive Histogram Equalization
  1. The image is split into tiles (tileGridSize).
  2. In each tile a histogram is computed; bins higher than clipLimit x (mean bin height)
     are clipped and the excess is redistributed uniformly. This caps the slope of the
     mapping, i.e. the maximum contrast amplification -> limits noise amplification.
  3. Each tile gets its own histogram-equalisation mapping; pixel values are mapped with
     bilinear interpolation between the 4 nearest tile mappings (no visible tile seams).

Parameters
  clip  : higher -> stronger local contrast, but also stronger noise / JPEG-block amplification
  grid  : more tiles -> more local adaptation (smaller tiles), but tiles with few
          structures get over-stretched
"""
import cv2
import numpy as np

from src import config


def tile_grid(shape: tuple[int, int], tiles_across: int) -> tuple[int, int]:
    """OpenCV tileGridSize (tiles_x, tiles_y) giving approximately square tiles."""
    height, width = shape
    return tiles_across, max(1, int(round(tiles_across * height / width)))


def apply_clahe(
    image: np.ndarray,
    fov: np.ndarray,
    clip: float = config.CLAHE_CLIP,
    grid: int = config.CLAHE_GRID,
) -> np.ndarray:
    """
    CLAHE on a single-channel image (float or uint8, range 0-255). Returns uint8.
    Outside the FOV the input is set to the FOV mean before CLAHE (so the black
    background does not dominate border-tile histograms), and to the FOV median
    of the result afterwards (neutral value for later morphology).
    """
    inside = fov > 0
    work = np.clip(np.round(image), 0, 255).astype(np.uint8)
    work[~inside] = int(round(float(work[inside].mean())))
    clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=tile_grid(work.shape, grid))
    enhanced = clahe.apply(work)
    enhanced[~inside] = int(np.median(enhanced[inside]))
    return enhanced
