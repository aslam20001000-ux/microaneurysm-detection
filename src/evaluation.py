"""
Evaluation (pipeline steps 17-18). Defined BEFORE any detection results were seen.

Three levels
------------
1. Pixel level   : every FOV pixel is a sample (detection mask vs GT mask).
                   Sensitivity, Specificity, Precision, F1 (= Dice), IoU, plus
                   AUPR of the continuous response map (the official IDRiD
                   segmentation metric; robust to the ~0.1 % class imbalance).
                   Accuracy is NOT reported: predicting "no MA" everywhere already
                   gives > 99.8 % accuracy.
2. Lesion level  : every GT MA (8-connected component) is one lesion and every
                   detected connected component is one candidate.
                   Matching rule: a candidate is a TRUE POSITIVE if its centroid lies
                   inside a GT lesion or within MATCH_TOLERANCE px of it (the GT
                   outline is drawn slightly larger/smaller than the dark dot, and a
                   centroid can sit at the edge). A GT lesion is DETECTED if at least
                   one candidate matches it; extra candidates on the same lesion are
                   not counted as false positives.
                   -> Sensitivity = detected GT / all GT
                   -> Precision   = TP candidates / all candidates
                   -> FPI         = false-positive candidates per image
                   Centroid matching (instead of "any overlap") prevents one huge
                   blob (e.g. a whole vessel tree) from "detecting" every MA it touches.
3. Image level   : image is positive if >= 1 candidate. All 81 IDRiD segmentation
                   images contain MA, so only image-level sensitivity is defined here;
                   specificity needs images without MA (not available in this subset).

FROC: lesion sensitivity vs FPI when the detection threshold is swept.
CPM (Competition Performance Metric, Niemeijer et al., 2010): mean sensitivity at
FPI = 1/8, 1/4, 1/2, 1, 2, 4, 8.
"""
import cv2
import numpy as np

MATCH_TOLERANCE = 5           # px
CPM_FPI = np.array([1 / 8, 1 / 4, 1 / 2, 1, 2, 4, 8])


# ----------------------------------------------------------------------------- pixel level
def pixel_pr_counts(score: np.ndarray, gt: np.ndarray, fov: np.ndarray, bins: int = 2048,
                    value_range: tuple[float, float] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Histogram of the response for positive and negative FOV pixels (fast, exact up to binning).
    Summing these histograms over images gives a POOLED PR curve for a whole set.
    """
    inside = fov > 0
    values = score[inside].astype(np.float32)
    labels = gt[inside] > 0
    lo, hi = value_range if value_range else (float(values.min()), float(values.max()) + 1e-6)
    edges = np.linspace(lo, hi, bins + 1)
    pos, _ = np.histogram(values[labels], bins=edges)
    neg, _ = np.histogram(values[~labels], bins=edges)
    return pos, neg


def ap_from_counts(pos: np.ndarray, neg: np.ndarray) -> tuple[float, float]:
    """Average precision (AUPR) and ROC-AUC from per-bin positive/negative counts."""
    tp = np.cumsum(pos[::-1]).astype(np.float64)       # threshold sweeps from high to low score
    fp = np.cumsum(neg[::-1]).astype(np.float64)
    n_pos, n_neg = tp[-1], fp[-1]
    if n_pos == 0:
        return np.nan, np.nan
    recall = tp / n_pos
    precision = tp / np.maximum(tp + fp, 1)
    ap = float(np.sum(np.diff(np.concatenate([[0.0], recall])) * precision))
    fpr = fp / max(n_neg, 1)
    auc = float(np.trapezoid(np.concatenate([[0.0], recall]), np.concatenate([[0.0], fpr])))
    return ap, auc


def pixel_confusion(pred: np.ndarray, gt: np.ndarray, fov: np.ndarray) -> dict:
    inside = fov > 0
    p, g = (pred > 0) & inside, (gt > 0) & inside
    tp = int(np.count_nonzero(p & g))
    fp = int(np.count_nonzero(p & ~g))
    fn = int(np.count_nonzero(~p & g))
    tn = int(np.count_nonzero(inside)) - tp - fp - fn
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def pixel_metrics(c: dict) -> dict:
    tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
    sens = tp / max(tp + fn, 1)
    prec = tp / max(tp + fp, 1)
    return {
        "pixel_sensitivity": sens,
        "pixel_specificity": tn / max(tn + fp, 1),
        "pixel_precision": prec,
        "pixel_f1_dice": 2 * tp / max(2 * tp + fp + fn, 1),
        "pixel_iou": tp / max(tp + fp + fn, 1),
    }


# ----------------------------------------------------------------------------- lesion level
def gt_match_zones(gt: np.ndarray, tolerance: int = MATCH_TOLERANCE) -> tuple[np.ndarray, int]:
    """
    Label map: every pixel within `tolerance` px of a GT lesion carries that lesion's id
    (nearest lesion wins where zones touch). Returns (zone_labels, number_of_gt_lesions).
    """
    lesion = gt > 0
    if not lesion.any():
        return np.zeros(gt.shape, np.int32), 0
    # Each connected GT lesion gets one label; every other pixel gets the label of the
    # NEAREST lesion and its distance to it (a Voronoi partition of the image).
    distance, labels = cv2.distanceTransformWithLabels((~lesion).astype(np.uint8), cv2.DIST_L2, 5,
                                                       labelType=cv2.DIST_LABEL_CCOMP)
    zones = labels.copy()
    zones[distance > tolerance] = 0
    return zones, int(np.unique(labels[lesion]).size)


def match_candidates(centroids_xy: np.ndarray, zones: np.ndarray, n_gt: int) -> dict:
    """centroids_xy: (n, 2) array of candidate centroids. Returns lesion-level counts."""
    if len(centroids_xy) == 0:
        return {"n_gt": n_gt, "gt_detected": 0, "n_candidates": 0, "tp_candidates": 0, "fp_candidates": 0}
    xs = np.clip(np.round(centroids_xy[:, 0]).astype(int), 0, zones.shape[1] - 1)
    ys = np.clip(np.round(centroids_xy[:, 1]).astype(int), 0, zones.shape[0] - 1)
    hit = zones[ys, xs]
    tp = int(np.count_nonzero(hit))
    return {"n_gt": n_gt, "gt_detected": int(np.unique(hit[hit > 0]).size),
            "n_candidates": int(len(hit)), "tp_candidates": tp, "fp_candidates": int(len(hit) - tp)}


def lesion_metrics(counts: dict, n_images: int) -> dict:
    """Pooled lesion-level metrics from summed counts."""
    sens = counts["gt_detected"] / max(counts["n_gt"], 1)
    prec = counts["tp_candidates"] / max(counts["n_candidates"], 1)
    return {
        "lesion_sensitivity": sens,
        "lesion_precision": prec,
        "lesion_f1": 2 * sens * prec / max(sens + prec, 1e-12),
        "fp_per_image": counts["fp_candidates"] / max(n_images, 1),
    }


def froc_cpm(fpi: np.ndarray, sensitivity: np.ndarray) -> tuple[float, np.ndarray]:
    """
    CPM = mean sensitivity at the 7 reference FPI values (linear interpolation on the
    FROC curve). Below the lowest measured FPI sensitivity is taken as 0; above the
    highest measured FPI the last sensitivity is used.
    """
    order = np.argsort(fpi)
    x, y = np.asarray(fpi)[order], np.asarray(sensitivity)[order]
    y = np.maximum.accumulate(y)            # FROC must be non-decreasing in FPI
    values = np.interp(CPM_FPI, np.concatenate([[0.0], x]), np.concatenate([[0.0], y]))
    return float(values.mean()), values
