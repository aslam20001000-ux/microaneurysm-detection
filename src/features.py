"""
Candidate feature extraction (pipeline step 14).

One row per candidate (connected component). Features describe the three things
that separate a microaneurysm from its usual false positives:

  SHAPE      (MA are small and round; vessel pieces are elongated)
    area, perimeter, circularity = 4*pi*A / P^2, eccentricity and elongation from the
    second-order moments (covariance of the pixel coordinates), extent, bbox aspect
  RESPONSE   (how strongly the morphology filters reacted)
    peak/mean of the final z response, peak multi-scale black-hat z, peak line
    top-hat z, and their ratio (low ratio = the line opening "explained" the object,
    i.e. it is vessel-like)
  INTENSITY  (MA are dark red dots on a locally uniform background)
    mean grey level inside vs a 2-8 px ring around the candidate -> contrast and
    local CNR; mean of the original R, G, B channels (normalised by the FOV mean)
  POSITION
    distance to the FOV edge (rim artefacts)
All per-component statistics use np.bincount / scipy.ndimage on the label image,
so the cost is a few passes over the image regardless of the number of candidates.
"""
import cv2
import numpy as np
import pandas as pd
from scipy import ndimage

FEATURE_NAMES = [
    "area", "perimeter", "circularity", "eccentricity", "elongation", "extent", "bbox_aspect",
    "z_peak", "z_mean", "ms_peak", "line_peak", "line_ms_ratio",
    "mean_inside", "mean_ring", "contrast", "local_cnr",
    "red_norm", "green_norm", "blue_norm", "dist_fov_edge",
]


def _bincount(labels_flat, weights, n):
    return np.bincount(labels_flat, weights=weights, minlength=n + 1)[1:]


def candidate_features(labels: np.ndarray, stats: np.ndarray, centroids: np.ndarray, pre: dict,
                       image_bgr: np.ndarray, z: np.ndarray, ms: np.ndarray, line: np.ndarray) -> pd.DataFrame:
    n = int(labels.max())
    if n == 0:
        return pd.DataFrame(columns=FEATURE_NAMES)
    index = np.arange(1, n + 1)
    mask = labels > 0
    ys, xs = np.nonzero(mask)
    lab = labels[ys, xs]

    # ---------------------------------------------------------------- shape
    area = _bincount(lab, None, n).astype(np.float64)
    boundary = mask & (cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)) == 0)
    perimeter = _bincount(labels[boundary], None, n).astype(np.float64)
    circularity = 4 * np.pi * area / np.maximum(perimeter, 1) ** 2
    mx, my = _bincount(lab, xs, n) / area, _bincount(lab, ys, n) / area
    vxx = _bincount(lab, xs.astype(np.float64) ** 2, n) / area - mx ** 2
    vyy = _bincount(lab, ys.astype(np.float64) ** 2, n) / area - my ** 2
    vxy = _bincount(lab, xs.astype(np.float64) * ys, n) / area - mx * my
    root = np.sqrt(((vxx - vyy) / 2) ** 2 + vxy ** 2)
    l1, l2 = (vxx + vyy) / 2 + root, np.maximum((vxx + vyy) / 2 - root, 1e-6)
    eccentricity = np.sqrt(np.clip(1 - l2 / np.maximum(l1, 1e-6), 0, 1))
    elongation = np.sqrt(l1 / l2)
    w, h = stats[:, cv2.CC_STAT_WIDTH].astype(np.float64), stats[:, cv2.CC_STAT_HEIGHT].astype(np.float64)
    extent = area / (w * h)
    bbox_aspect = np.maximum(w, h) / np.minimum(w, h)

    # ---------------------------------------------------------------- response
    z_peak = ndimage.maximum(z, labels, index)
    z_mean = _bincount(lab, z[ys, xs], n) / area
    ms_peak = ndimage.maximum(ms, labels, index)
    line_peak = ndimage.maximum(line, labels, index)
    line_ms_ratio = line_peak / np.maximum(ms_peak, 1e-6)

    # ---------------------------------------------------------------- intensity (ring = 2-8 px)
    image = pre["smooth"].astype(np.float64)
    distance, nearest_cc = cv2.distanceTransformWithLabels((~mask).astype(np.uint8), cv2.DIST_L2, 5,
                                                           labelType=cv2.DIST_LABEL_CCOMP)
    lut = np.zeros(nearest_cc.max() + 1, np.int64)
    lut[nearest_cc[mask]] = labels[mask]                 # OpenCV component id -> our candidate id
    ring = (distance > 2) & (distance <= 8) & (pre["fov"] > 0)
    ring_lab = lut[nearest_cc[ring]]
    ring_count = np.maximum(_bincount(ring_lab, None, n), 1)
    mean_inside = _bincount(lab, image[ys, xs], n) / area
    mean_ring = _bincount(ring_lab, image[ring], n) / ring_count
    ring_sd = np.sqrt(np.maximum(_bincount(ring_lab, image[ring] ** 2, n) / ring_count - mean_ring ** 2, 1e-6))
    contrast = mean_ring - mean_inside
    local_cnr = contrast / ring_sd

    # ---------------------------------------------------------------- colour (original image)
    inside_fov = pre["fov"] > 0
    colour = {}
    for name, channel in zip(["blue_norm", "green_norm", "red_norm"], cv2.split(image_bgr)):
        ch = channel.astype(np.float64)
        colour[name] = (_bincount(lab, ch[ys, xs], n) / area) / max(ch[inside_fov].mean(), 1e-6)

    # ---------------------------------------------------------------- position
    fov_distance = cv2.distanceTransform(pre["fov"], cv2.DIST_L2, 5)
    cx = np.clip(np.round(centroids[:, 0]).astype(int), 0, labels.shape[1] - 1)
    cy = np.clip(np.round(centroids[:, 1]).astype(int), 0, labels.shape[0] - 1)
    dist_fov_edge = fov_distance[cy, cx]

    return pd.DataFrame({
        "area": area, "perimeter": perimeter, "circularity": circularity, "eccentricity": eccentricity,
        "elongation": elongation, "extent": extent, "bbox_aspect": bbox_aspect,
        "z_peak": z_peak, "z_mean": z_mean, "ms_peak": ms_peak, "line_peak": line_peak,
        "line_ms_ratio": line_ms_ratio, "mean_inside": mean_inside, "mean_ring": mean_ring,
        "contrast": contrast, "local_cnr": local_cnr, **colour, "dist_fov_edge": dist_fov_edge,
    })[FEATURE_NAMES]
