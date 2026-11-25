"""
Experiment 09 - FINAL evaluation on the held-out TEST set (27 images).

Run ONCE, after every parameter was frozen on the training set (config.py,
outputs/models/operating_points.json, outputs/models/fp_classifier.joblib).
No parameter is changed based on these results.

For every ablation model (A, A+, B, B+, C, D0, D):
  Lesion level : sensitivity, precision, F1, FPI at the frozen operating point,
                 FROC + CPM, 95 % bootstrap CI over images (2000 resamples)
  Pixel level  : sensitivity, specificity, precision, Dice (= F1), IoU of the detection
                 mask (pooled over images) + AUPR of the continuous response
  Image level  : sensitivity (all test images contain MA; specificity undefined)
  Time         : median processing time per image (full pipeline of that model)

Outputs: outputs/experiment_09/ (CSV, figures), outputs/detections/ (annotated images),
         outputs/masks/detections/ (binary masks of Model D)
Run:     python experiments/exp09_test_evaluation.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import matplotlib
import numpy as np
import pandas as pd

from src import config
from src.candidates import extract_candidates, line_tophat, multiscale_blackhat, robust_z
from src.dataset_loader import image_id_from_path, list_images, mask_path, read_image, read_mask
from src.denoise import denoise
from src.detection import MicroaneurysmDetector
from src.enhancement import apply_clahe
from src.evaluation import (CPM_FPI, ap_from_counts, froc_cpm, gt_match_zones, lesion_metrics, pixel_confusion,
                            pixel_metrics, pixel_pr_counts)
from src.features import FEATURE_NAMES, candidate_features
from src.fov import extract_fov_mask
from src.illumination import illumination_correct
from src.preprocessing import green_channel
from src.visualization import style_axes

OUTPUT_DIR = config.OUTPUTS_DIR / "experiment_09"
ANNOTATED_DIR = config.OUTPUTS_DIR / "detections"
MASK_DIR = config.OUTPUTS_DIR / "masks" / "detections"
MODELS = ["A", "A+", "B", "B+", "C", "D0", "D"]
K_GRID = [1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 8, 10, 12, 15, 20]
P_GRID = np.linspace(0, 1, 1001)       # uniform, so the low-FPI (high-probability) range is covered
Z_RANGE, BINS = (-10.0, 150.0), 8000
N_BOOT = 2000
RAMP = {"A": "#c6dbef", "A+": "#9ecae1", "B": "#6baed6", "B+": "#4292c6", "C": "#2171b5", "D0": "#084594",
        "D": "#d9822b"}
INK = "#222222"
TP_COLOR, FP_COLOR, FN_COLOR = (0, 255, 255), (255, 0, 255), (255, 255, 0)   # RGB: cyan, magenta, yellow


def timed(fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - t0


def per_threshold_counts(cand_scores, cand_gt, thresholds, n_gt):
    """Per-image counts for every threshold (score >= t kept)."""
    rows = np.zeros((len(thresholds), 4), np.int64)   # gt_detected, n_candidates, tp_candidates, fp_candidates
    for i, t in enumerate(thresholds):
        kept = cand_scores >= t
        tp = kept & (cand_gt > 0)
        rows[i] = [np.unique(cand_gt[tp]).size, kept.sum(), tp.sum(), (kept & (cand_gt == 0)).sum()]
    return rows


def aggregate(counts_list, n_gt_list, idx):
    c = np.sum([counts_list[i] for i in idx], axis=0)
    total_gt = int(np.sum([n_gt_list[i] for i in idx]))
    sens = c[..., 0] / max(total_gt, 1)
    fpi = c[..., 3] / len(idx)
    prec = c[..., 2] / np.maximum(c[..., 1], 1)
    return sens, fpi, prec


def annotate(image_bgr, centroids_kept, zones, n_gt, ma_mask, path, scale=0.5):
    """Downscaled RGB image with TP (cyan), FP (magenta) and missed GT lesions (yellow)."""
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    small = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    hit_ids = set()
    for x, y in centroids_kept:
        gid = zones[int(round(y)), int(round(x))]
        color = TP_COLOR if gid > 0 else FP_COLOR
        if gid > 0:
            hit_ids.add(int(gid))
        cv2.circle(small, (int(x * scale), int(y * scale)), 14, color, 2, cv2.LINE_AA)
    _, gt_labels, _, gt_centroids = cv2.connectedComponentsWithStats(ma_mask, connectivity=8)
    for j in range(1, len(gt_centroids)):
        ys, xs = np.nonzero(gt_labels == j)
        if int(zones[ys[0], xs[0]]) not in hit_ids:
            cx, cy = gt_centroids[j]
            cv2.rectangle(small, (int(cx * scale) - 12, int(cy * scale) - 12), (int(cx * scale) + 12, int(cy * scale) + 12),
                          FN_COLOR, 2, cv2.LINE_AA)
    cv2.imencode(".jpg", cv2.cvtColor(small, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tofile(str(path))
    return small


def main():
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    start = time.perf_counter()
    for d in (OUTPUT_DIR, ANNOTATED_DIR, MASK_DIR):
        d.mkdir(parents=True, exist_ok=True)
    detector = MicroaneurysmDetector()
    points = detector.points

    counts = {m: [] for m in MODELS}
    n_gt_list, conf = [], {m: dict(tp=0, fp=0, fn=0, tn=0) for m in MODELS}
    pr_hist = {m: [np.zeros(BINS), np.zeros(BINS)] for m in MODELS}
    image_pos = {m: 0 for m in MODELS}
    times = {m: [] for m in MODELS}
    op_counts = {m: [] for m in MODELS}
    per_image_rows = []

    images = list_images("test")
    for index, image_path in enumerate(images, start=1):
        image_id = image_id_from_path(image_path)
        print(f"  Processing {index}/{len(images)}: {image_id}", end="\r", flush=True)
        try:
            image_bgr = read_image(image_path)
            ma_mask = read_mask(mask_path("test", image_id, "MA"))
        except (FileNotFoundError, ValueError) as error:
            print(f"\n  SKIPPED {image_id}: {error}")
            continue
        zones, n_gt = gt_match_zones(ma_mask)
        n_gt_list.append(n_gt)

        # ---- staged preprocessing with timing (each model pays only for the stages it uses)
        fov, t_fov = timed(extract_fov_mask, image_bgr)
        green = green_channel(image_bgr)
        (corrected, _), t_ill = timed(illumination_correct, green, fov)
        clahe, t_cla = timed(apply_clahe, corrected, fov)
        smooth, t_den = timed(denoise, clahe, config.DENOISE_METHOD, config.DENOISE_SIGMA)
        green_n = green.astype(np.float32)
        green_n[fov == 0] = green[fov > 0].mean()
        ms, t_bh = timed(multiscale_blackhat, smooth, fov, config.BLACKHAT_RADII)
        line_raw, t_line = timed(line_tophat, smooth, config.LINE_LENGTH)
        line = robust_z(line_raw, fov)
        maps = {"A": robust_z(255 - green_n, fov), "A+": robust_z(255 - corrected, fov),
                "B": robust_z(255 - clahe.astype(np.float32), fov), "B+": robust_z(255 - smooth.astype(np.float32), fov),
                "C": ms, "D0": line}                       # D0 = line top-hat alone (Exp07)
        base_time = {"A": t_fov, "A+": t_fov + t_ill, "B": t_fov + t_ill + t_cla, "B+": t_fov + t_ill + t_cla + t_den}
        base_time["C"] = base_time["B+"] + t_bh
        base_time["D0"] = base_time["B+"] + t_line
        base_time["D"] = base_time["B+"] + t_line + t_bh      # D also needs the black-hat as a feature

        pre = {"fov": fov, "smooth": smooth}
        for m in MODELS:
            if m != "D":
                # FROC: sweep k
                rows = []
                for k in K_GRID:
                    _, _, cen = extract_candidates(maps[m], fov, k)
                    xs = np.clip(np.round(cen[:, 0]).astype(int), 0, fov.shape[1] - 1)
                    ys = np.clip(np.round(cen[:, 1]).astype(int), 0, fov.shape[0] - 1)
                    g = zones[ys, xs]
                    rows.append([np.unique(g[g > 0]).size, len(g), int((g > 0).sum()), int((g == 0).sum())])
                counts[m].append(np.array(rows))
                (labels, stats, cen), t_c = timed(extract_candidates, maps[m], fov, points[m]["k"])
                mask = (labels > 0).astype(np.uint8)
                kept_centroids = cen
                score_map = maps[m]
                times[m].append(base_time[m] + t_c)
            else:
                t0 = time.perf_counter()
                labels, stats, cen = extract_candidates(maps["D0"], fov, detector.k0)
                feats = candidate_features(labels, stats, cen, pre, image_bgr, maps["D0"], ms, line)
                prob = detector.classifier.predict_proba(feats[FEATURE_NAMES].to_numpy())[:, 1] if len(feats) else np.zeros(0)
                keep = prob >= points["D"]["threshold"]
                times["D"].append(base_time["D"] + time.perf_counter() - t0)
                xs = np.clip(np.round(cen[:, 0]).astype(int), 0, fov.shape[1] - 1)
                ys = np.clip(np.round(cen[:, 1]).astype(int), 0, fov.shape[0] - 1)
                g = zones[ys, xs]
                counts["D"].append(per_threshold_counts(prob, g, P_GRID, n_gt))
                lut = np.zeros(len(stats) + 1, np.float32)
                lut[1:] = prob
                score_map = lut[labels]                      # probability painted on candidate pixels
                lut_keep = np.zeros(len(stats) + 1, np.uint8)
                lut_keep[1:][keep] = 1
                mask = lut_keep[labels]
                kept_centroids = cen[keep]
                cv2.imencode(".png", mask * 255)[1].tofile(str(MASK_DIR / f"{image_id}_MA_detected.png"))
                annotate(image_bgr, kept_centroids, zones, n_gt, ma_mask, ANNOTATED_DIR / f"{image_id}_model_D.jpg")

            # operating-point counts (lesion), pixel confusion, AUPR, image level
            xs = np.clip(np.round(kept_centroids[:, 0]).astype(int), 0, fov.shape[1] - 1)
            ys = np.clip(np.round(kept_centroids[:, 1]).astype(int), 0, fov.shape[0] - 1)
            g = zones[ys, xs]
            op = np.array([np.unique(g[g > 0]).size, len(g), int((g > 0).sum()), int((g == 0).sum())])
            op_counts[m].append(op)
            c = pixel_confusion(mask, ma_mask, fov)
            for key in conf[m]:
                conf[m][key] += c[key]
            rng_ = (0.0, 1.0) if m == "D" else Z_RANGE
            pos, neg = pixel_pr_counts(np.clip(score_map, *rng_), ma_mask, fov, BINS, rng_)
            pr_hist[m][0] += pos
            pr_hist[m][1] += neg
            image_pos[m] += int(len(g) > 0)
            per_image_rows.append({"image_id": image_id, "model": m, "n_gt": n_gt, "gt_detected": op[0],
                                   "candidates": op[1], "fp": op[3], **pixel_metrics(c), "time_s": times[m][-1]})
    print(f"  Done: {len(images)} test images." + " " * 20)
    n_images = len(n_gt_list)

    # ------------------------------------------------------------------ aggregate + bootstrap
    rng = np.random.default_rng(0)
    boot_idx = [rng.integers(0, n_images, n_images) for _ in range(N_BOOT)]
    summary_rows, froc_rows = [], []
    for m in MODELS:
        all_idx = np.arange(n_images)
        sens, fpi, prec = aggregate(counts[m], n_gt_list, all_idx)
        cpm, at_points = froc_cpm(fpi, sens)
        grid = K_GRID if m != "D" else P_GRID
        for t, s, f, p in zip(grid, sens, fpi, prec):
            froc_rows.append({"model": m, "threshold": t, "lesion_sensitivity": s, "fp_per_image": f, "lesion_precision": p})
        op_sum = np.sum(op_counts[m], axis=0)
        lm = lesion_metrics(dict(n_gt=sum(n_gt_list), gt_detected=op_sum[0], n_candidates=op_sum[1],
                                 tp_candidates=op_sum[2], fp_candidates=op_sum[3]), n_images)
        boot = {"cpm": [], "lesion_sensitivity": [], "lesion_f1": [], "fp_per_image": []}
        for idx in boot_idx:
            s_b, f_b, _ = aggregate(counts[m], n_gt_list, idx)
            boot["cpm"].append(froc_cpm(f_b, s_b)[0])
            o = np.sum([op_counts[m][i] for i in idx], axis=0)
            lb = lesion_metrics(dict(n_gt=int(np.sum([n_gt_list[i] for i in idx])), gt_detected=o[0], n_candidates=o[1],
                                     tp_candidates=o[2], fp_candidates=o[3]), len(idx))
            for key in ("lesion_sensitivity", "lesion_f1", "fp_per_image"):
                boot[key].append(lb[key])
        ci = {f"{k}_ci": f"[{np.percentile(v, 2.5):.3f}, {np.percentile(v, 97.5):.3f}]" for k, v in boot.items()}
        ap, auc = ap_from_counts(*pr_hist[m])
        summary_rows.append({"model": m, "cpm": cpm, **{f"sens@{f:g}FPI": v for f, v in zip(CPM_FPI, at_points)},
                             **lm, **pixel_metrics(conf[m]), "pixel_aupr": ap, "pixel_auc": auc,
                             "image_sensitivity": image_pos[m] / n_images,
                             "time_median_s": float(np.median(times[m])), **ci})
    summary = pd.DataFrame(summary_rows).set_index("model")
    summary.round(5).to_csv(OUTPUT_DIR / "ablation_test_summary.csv")
    pd.DataFrame(froc_rows).round(5).to_csv(OUTPUT_DIR / "froc_test.csv", index=False)
    pd.DataFrame(per_image_rows).round(5).to_csv(OUTPUT_DIR / "per_image_test.csv", index=False)

    report(summary, pd.DataFrame(froc_rows), n_images, sum(n_gt_list))
    print(f"Elapsed: {time.perf_counter() - start:.1f} s")


def report(summary: pd.DataFrame, froc: pd.DataFrame, n_images: int, total_gt: int):
    """Figures + console table from the saved results (no recomputation)."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9.5, 6))
    for m in MODELS:
        part = froc[(froc["model"] == m) & (froc["fp_per_image"] > 0)].sort_values("fp_per_image")
        ax.plot(part["fp_per_image"], part["lesion_sensitivity"], color=RAMP[m], lw=2.5 if m == "D" else 2,
                marker=None if m == "D" else "o", ms=4, label=f"{m}  (CPM {summary.loc[m, 'cpm']:.3f})")
    ax.legend(frameon=False, fontsize=9, loc="upper left", title="Model", title_fontsize=9)   # labels overlapped at top right
    for x in [1 / 8, 1, 8]:
        ax.axvline(x, color="#bbbbbb", lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel("False positives per image (log scale)")
    ax.set_ylabel("Lesion-level sensitivity")
    ax.set_title(f"FROC on the TEST set ({n_images} images, {total_gt} MA)", fontsize=11, loc="left")
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig_froc_test.png", dpi=150)

    panels = [("cpm", "CPM (FROC)", False), ("lesion_f1", "Lesion-level F1", False),
              ("fp_per_image", "False positives / image (log)", True), ("pixel_aupr", "Pixel AUPR", False)]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.4))
    for ax, (col, title, log) in zip(axes, panels):
        vals = summary[col].to_numpy()
        ax.bar(MODELS, vals, color=[RAMP[m] for m in MODELS], width=0.65)
        if f"{col}_ci" in summary:
            lo_hi = np.array([[float(v) for v in s.strip("[]").split(",")] for s in summary[f"{col}_ci"]])
            # CI bounds are rounded to 3 decimals and percentile-based, so a point estimate can sit just
            # outside its interval (e.g. 0.0002 vs [0.000, 0.000]) -> clip the bar lengths at 0
            ax.errorbar(range(len(MODELS)), vals,
                        yerr=[np.clip(vals - lo_hi[:, 0], 0, None), np.clip(lo_hi[:, 1] - vals, 0, None)], fmt="none",
                        ecolor=INK, capsize=3, lw=1)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:,.0f}" if v >= 100 else f"{v:.3g}", ha="center", va="bottom", fontsize=8, color=INK)
        if log:
            ax.set_yscale("log")
        ax.set_title(title, fontsize=11, loc="left")
        style_axes(ax)
        ax.grid(axis="x", visible=False)
    fig.suptitle("Ablation study on the TEST set (error bars: 95 % bootstrap CI over images)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig_ablation_test.png", dpi=150)

    # ------------------------------------------------------------------ report
    pd.set_option("display.width", 200)
    print(f"\n=== TEST set: {n_images} images, {total_gt} GT MA ===")
    print(f"{'model':<6}{'CPM':>7}{'sens':>7}{'prec':>7}{'F1':>7}{'FPI':>9}{'pxSens':>8}{'pxSpec':>9}{'Dice':>7}"
          f"{'IoU':>7}{'AUPR':>8}{'imgSens':>8}{'time s':>8}")
    for m, r in summary.iterrows():
        print(f"{m:<6}{r['cpm']:>7.3f}{r['lesion_sensitivity']:>7.3f}{r['lesion_precision']:>7.3f}{r['lesion_f1']:>7.3f}"
              f"{r['fp_per_image']:>9.1f}{r['pixel_sensitivity']:>8.3f}{r['pixel_specificity']:>9.4f}"
              f"{r['pixel_f1_dice']:>7.3f}{r['pixel_iou']:>7.3f}{r['pixel_aupr']:>8.4f}{r['image_sensitivity']:>8.2f}"
              f"{r['time_median_s']:>8.2f}")
    print("\n95 % bootstrap CI:")
    for m, r in summary.iterrows():
        print(f"  {m:<4} CPM {r['cpm_ci']}  sens {r['lesion_sensitivity_ci']}  F1 {r['lesion_f1_ci']}  FPI {r['fp_per_image_ci']}")
    print(f"Outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    if "--plots-only" in sys.argv:
        # Redraw figures from the saved CSV files without re-running the evaluation
        matplotlib.use("Agg")
        saved = pd.read_csv(OUTPUT_DIR / "ablation_test_summary.csv", index_col=0)
        per_image = pd.read_csv(OUTPUT_DIR / "per_image_test.csv")
        gt_per_image = per_image.drop_duplicates("image_id")["n_gt"]
        report(saved, pd.read_csv(OUTPUT_DIR / "froc_test.csv"), len(gt_per_image), int(gt_per_image.sum()))
    else:
        main()
