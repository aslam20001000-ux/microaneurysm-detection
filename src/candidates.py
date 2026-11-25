"""
Candidate generation for every ablation model (pipeline steps 9-13).

Every model produces a RESPONSE map S (high = "looks like an MA"), which is turned
into a robust z-score inside the FOV and thresholded the same way for all models
(adaptive thresholding, step 11). This keeps the comparison fair: only the
response changes between models.

  Model A  : S = 255 - green
  Model A+ : S = 255 - illumination-corrected green
  Model B  : S = 255 - CLAHE(corrected)
  Model B+ : S = 255 - Gaussian(CLAHE(corrected))                        = 255 - I
  Model C  : S = multi-scale black-hat of I
  Model D0 : S = line top-hat (oriented linear openings)  -> vessels suppressed
  Model D  : D0 candidates + feature-based false-positive removal (features.py);
             the disc black-hat of Model C is kept as a FEATURE.
  (Exp07: min(black-hat, line top-hat) was tested first; the line top-hat alone had
   2.5x higher training AUPR, because the weak black-hat peak limited the minimum.)

Black-hat (closing - image) is high on DARK structures narrower than the structuring
element. Granulometry property: for a disc, closing grows with the radius, so
black-hat at a larger radius is >= black-hat at a smaller radius everywhere.
Taking max over raw scales would simply return the largest scale -> each scale is
first normalised by its own robust spread (z-score), then the max is taken.
"""
import cv2
import numpy as np

from src import config
from src.denoise import denoise
from src.enhancement import apply_clahe
from src.fov import disk, extract_fov_mask
from src.illumination import illumination_correct
from src.preprocessing import green_channel

MODELS = ["A", "A+", "B", "B+", "C", "D"]


# ----------------------------------------------------------------------------- preprocessing
def preprocess(image_bgr: np.ndarray) -> dict:
    """All intermediate images needed by any model (computed once per image)."""
    fov = extract_fov_mask(image_bgr)
    green = green_channel(image_bgr)
    corrected, _ = illumination_correct(green, fov)
    clahe = apply_clahe(corrected, fov)
    smooth = denoise(clahe, config.DENOISE_METHOD, config.DENOISE_SIGMA)
    green_neutral = green.astype(np.float32)
    green_neutral[fov == 0] = green[fov > 0].mean()
    return {"fov": fov, "green": green_neutral, "corrected": corrected, "clahe": clahe, "smooth": smooth}


# ----------------------------------------------------------------------------- normalisation
def robust_z(score: np.ndarray, fov: np.ndarray, step: int = 7) -> np.ndarray:
    """
    z = (S - median) / (1.4826 * MAD) inside the FOV; 0 outside.
    Median/MAD (not mean/std) because the few lesion and vessel pixels must not
    inflate the estimate of "normal" background variation. Sub-sampled for speed.
    """
    values = score[fov > 0][::step].astype(np.float32)
    median = float(np.median(values))
    scale = float(np.median(np.abs(values - median))) * 1.4826
    # Sparse top-hat maps (e.g. black-hat r=3, line top-hat) are 0 in > 50 % of the FOV,
    # so their MAD collapses to 0 (or to a coarse integer). In that case the robust
    # estimate is meaningless and the standard deviation is used instead.
    std = float(values.std())
    if scale < 0.5 * std:
        scale = std
    z = (score.astype(np.float32) - median) / max(scale, 1e-6)
    z[fov == 0] = 0
    return z


# ----------------------------------------------------------------------------- morphology
def blackhat(image: np.ndarray, radius: int) -> np.ndarray:
    return cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, disk(radius))


def multiscale_blackhat(image: np.ndarray, fov: np.ndarray, radii) -> np.ndarray:
    """max over scales of the per-scale robust z-score of the black-hat."""
    return np.max([robust_z(blackhat(image, r), fov) for r in radii], axis=0)


def line_kernel(length: int, angle_deg: float) -> np.ndarray:
    """Binary line structuring element of a given length and orientation."""
    k = np.zeros((length, length), np.uint8)
    c = length // 2
    a = np.deg2rad(angle_deg)
    dx, dy = np.cos(a) * c, np.sin(a) * c
    cv2.line(k, (int(round(c - dx)), int(round(c + dy))), (int(round(c + dx)), int(round(c - dy))), 1, 1)
    return k


def line_tophat(image: np.ndarray, length: int, n_angles: int = 12) -> np.ndarray:
    """
    Vessel-suppressing top-hat on the inverted image J = 255 - I (dark -> bright):
        T = J - max_theta opening(J, line(length, theta))
    A vessel is long: in its own direction the line fits inside it, the opening keeps
    it, and the subtraction removes it. An MA is round and shorter than the line:
    no orientation fits, the opening removes it, and it survives in T.
    """
    inverted = 255 - image
    openings = [cv2.morphologyEx(inverted, cv2.MORPH_OPEN, line_kernel(length, a))
                for a in np.arange(0, 180, 180 / n_angles)]
    return cv2.subtract(inverted, np.max(openings, axis=0))


# ----------------------------------------------------------------------------- responses
def response(model: str, pre: dict, radii=None, line_length=None) -> np.ndarray:
    """Robust-z response map of a model (float32, 0 outside the FOV)."""
    fov = pre["fov"]
    radii = radii or config.BLACKHAT_RADII
    line_length = line_length or config.LINE_LENGTH
    if model == "A":
        return robust_z(255 - pre["green"], fov)
    if model == "A+":
        return robust_z(255 - pre["corrected"], fov)
    if model == "B":
        return robust_z(255 - pre["clahe"].astype(np.float32), fov)
    if model == "B+":
        return robust_z(255 - pre["smooth"].astype(np.float32), fov)
    if model == "C":
        return multiscale_blackhat(pre["smooth"], fov, radii)
    if model in ("D0", "D"):
        return robust_z(line_tophat(pre["smooth"], line_length), fov)
    raise ValueError(model)


# ----------------------------------------------------------------------------- candidates
def extract_candidates(z: np.ndarray, fov: np.ndarray, k: float, min_area: int = config.MIN_CANDIDATE_AREA):
    """
    Adaptive threshold (z > k) + connected-component analysis (8-connectivity).
    Components smaller than min_area px are discarded as noise (smallest GT MA = 13 px).
    Returns (label_image, stats, centroids) without the background row.
    """
    binary = ((z > k) & (fov > 0)).astype(np.uint8)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    keep = np.flatnonzero(stats[1:, cv2.CC_STAT_AREA] >= min_area) + 1
    if keep.size < n - 1:                       # relabel only if something was removed
        lut = np.zeros(n, np.int32)
        lut[keep] = np.arange(1, keep.size + 1)
        labels = lut[labels]
    return labels, stats[keep], centroids[keep]
