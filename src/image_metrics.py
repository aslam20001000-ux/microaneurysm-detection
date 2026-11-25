"""
Image-quality metrics used to TUNE preprocessing steps on the training set.
(GT masks are used only for measurement, never inside the detection algorithm.)
"""
import numpy as np
from sklearn.metrics import roc_auc_score


def local_contrast(image: np.ndarray, lesion: np.ndarray, ring: np.ndarray) -> dict:
    """
    Lesion vs local-background statistics.
      contrast = mu_ring - mu_lesion          (grey levels; MA are dark -> positive)
      weber    = contrast / mu_ring
      cnr      = contrast / sd_ring           (contrast in units of local background noise)
    """
    lesion_values = image[lesion].astype(np.float64)
    ring_values = image[ring].astype(np.float64)
    mu_l, mu_b, sd_b = lesion_values.mean(), ring_values.mean(), ring_values.std()
    contrast = mu_b - mu_l
    return {
        "contrast": contrast,
        "sd_background": sd_b,
        "weber": contrast / mu_b if mu_b > 0 else np.nan,
        "cnr": contrast / sd_b if sd_b > 0 else np.nan,
    }


def lesion_labels_and_rings(ma_mask: np.ndarray, fov: np.ndarray, inner_px: float = 3, outer_px: float = 10):
    """
    Per-lesion bookkeeping (computed once per image, independent of the filter):
      labels : every pixel gets the id of its NEAREST GT lesion (Voronoi partition)
      lesion : GT MA pixels
      ring   : pixels inner_px < d <= outer_px from the nearest lesion, inside the FOV
    Each ring pixel belongs to exactly one lesion, so each MA is compared with ITS OWN
    surroundings -> illumination differences between image regions do not enter the noise term.
    """
    import cv2
    distance, labels = cv2.distanceTransformWithLabels(
        (ma_mask == 0).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_CCOMP)
    lesion = ma_mask > 0
    ring = (distance > inner_px) & (distance <= outer_px) & (fov > 0)
    return labels, lesion, ring


def per_lesion_stats(image: np.ndarray, labels: np.ndarray, lesion: np.ndarray, ring: np.ndarray) -> dict:
    """Arrays (one entry per lesion id) of area, contrast (mu_ring - mu_lesion) and local CNR."""
    n = int(labels.max()) + 1
    img = image.astype(np.float64)
    l_lab, r_lab = labels[lesion], labels[ring]
    area = np.bincount(l_lab, minlength=n)
    l_mean = np.bincount(l_lab, weights=img[lesion], minlength=n) / np.maximum(area, 1)
    r_count = np.bincount(r_lab, minlength=n)
    r_mean = np.bincount(r_lab, weights=img[ring], minlength=n) / np.maximum(r_count, 1)
    r_sq = np.bincount(r_lab, weights=img[ring] ** 2, minlength=n) / np.maximum(r_count, 1)
    r_sd = np.sqrt(np.maximum(r_sq - r_mean ** 2, 1e-12))
    valid = (area > 0) & (r_count > 20)
    contrast = r_mean - l_mean
    return {"area": area[valid], "contrast": contrast[valid], "sd": r_sd[valid], "cnr": contrast[valid] / r_sd[valid]}


def global_auc(image: np.ndarray, lesion_idx: np.ndarray, background_idx: np.ndarray) -> float:
    """
    ROC-AUC of (-intensity) separating MA pixels from background pixels anywhere in the FOV:
    the best a single global threshold could do. 0.5 = chance, 1.0 = perfect.
    """
    flat = image.ravel()
    scores = np.concatenate([flat[lesion_idx], flat[background_idx]]).astype(np.float32)
    labels = np.concatenate([np.ones(lesion_idx.size), np.zeros(background_idx.size)])
    return float(roc_auc_score(labels, -scores))


def block_cv(image: np.ndarray, valid: np.ndarray, block: int = 128) -> float:
    """Illumination non-uniformity: CV of block means over blocks that are >= 90 % valid."""
    h = image.shape[0] // block * block
    w = image.shape[1] // block * block
    img = image[:h, :w].astype(np.float64)
    val = valid[:h, :w].astype(np.float64)
    sums = (img * val).reshape(h // block, block, w // block, block).sum(axis=(1, 3))
    counts = val.reshape(h // block, block, w // block, block).sum(axis=(1, 3))
    good = counts >= 0.9 * block * block
    means = sums[good] / counts[good]
    return float(means.std() / means.mean())


def jpeg_blockiness(image: np.ndarray, valid: np.ndarray) -> dict:
    """
    JPEG 8x8 block artefact strength (horizontal direction).
      border = mean |I(x+1) - I(x)| across 8-px block borders  (x % 8 == 7)
      inside = same, for all other column pairs
      ratio  = border / inside   (1.0 = no visible block grid)
    """
    image = image.astype(np.float32)
    diff = np.abs(np.diff(image, axis=1))
    pair_valid = valid[:, 1:] & valid[:, :-1]
    is_border = (np.arange(diff.shape[1]) % 8) == 7
    border = diff[:, is_border][pair_valid[:, is_border]].mean()
    inside = diff[:, ~is_border][pair_valid[:, ~is_border]].mean()
    return {"block_border": float(border), "block_inside": float(inside), "block_ratio": float(border / inside)}
