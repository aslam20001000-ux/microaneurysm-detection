"""
Experiment 07 - Blood Vessel Suppression (training set only).

Problem: the multi-scale black-hat (Model C) responds to EVERY dark structure narrower
than the disc - including vessel segments, which are the main source of false positives.

Method (oriented linear openings, after Zana & Klein 2001 / Spencer et al. 1996):
  J = 255 - I                          (vessels and MA become bright)
  V = max over 12 orientations of opening(J, line of length L)
      -> keeps anything that CONTAINS a straight segment of length L (vessels, background)
      -> removes round objects shorter than L (MA)
  T = J - V                            (line top-hat: MA survive, vessels are removed)
  Model D response = min( z(multi-scale black-hat), z(T) )
      -> a pixel must look like a small dark blob AND must not be part of a linear structure

Parameter: L must exceed the MA diameter (largest GT MA ~ 2051 px area -> ~51 px
diameter; median ~18 px) but stay shorter than straight vessel segments.

Metric: pooled pixel AUPR (all / small / large MA), as in Exp06.
Selection rule (fixed BEFORE running): highest pooled AUPR (all MA).

Outputs: outputs/experiment_07/
Run:     python experiments/exp07_vessels.py
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
from src.candidates import line_tophat, multiscale_blackhat, preprocess, robust_z
from src.dataset_loader import image_id_from_path, list_images, mask_path, read_image, read_mask
from src.evaluation import ap_from_counts, pixel_pr_counts
from src.visualization import densest_ma_window, style_axes

OUTPUT_DIR = config.OUTPUTS_DIR / "experiment_07"
LENGTHS = [21, 31, 41, 61, 81]
Z_RANGE, BINS = (-10.0, 150.0), 8000
SMALL, LARGE = 148, 377
INK, MUTED, BLUE, ORANGE = "#222222", "#888888", "#2a6fbb", "#d9822b"


def main():
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    start = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    names = ["C: multi-scale BH"] + [f"D: min(C, line L={L})" for L in LENGTHS] + [f"line top-hat only L={L}" for L in (41,)]
    pooled = {n: {g: [np.zeros(BINS), np.zeros(BINS)] for g in ("all", "small", "large")} for n in names}
    sample = None

    images = list_images("train")
    for index, image_path in enumerate(images, start=1):
        image_id = image_id_from_path(image_path)
        print(f"  Processing {index}/{len(images)}: {image_id}", end="\r", flush=True)
        try:
            image_bgr = read_image(image_path)
            ma_mask = read_mask(mask_path("train", image_id, "MA"))
        except (FileNotFoundError, ValueError) as error:
            print(f"\n  SKIPPED {image_id}: {error}")
            continue
        pre = preprocess(image_bgr)
        fov = pre["fov"]
        ms = multiscale_blackhat(pre["smooth"], fov, config.BLACKHAT_RADII)
        lines = {L: robust_z(line_tophat(pre["smooth"], L), fov) for L in LENGTHS}
        maps = {"C: multi-scale BH": ms}
        maps.update({f"D: min(C, line L={L})": np.minimum(ms, lines[L]) for L in LENGTHS})
        maps["line top-hat only L=41"] = lines[41]

        n, labels, stats, _ = cv2.connectedComponentsWithStats(ma_mask, connectivity=8)
        ids, area = np.arange(1, n), stats[1:, cv2.CC_STAT_AREA]
        small, large = np.isin(labels, ids[area <= SMALL]), np.isin(labels, ids[area > LARGE])
        inside, lesion = fov > 0, ma_mask > 0
        groups = {"all": (lesion, inside), "small": (small, inside & ~(lesion & ~small)),
                  "large": (large, inside & ~(lesion & ~large))}
        for name, z in maps.items():
            for g, (gt, region) in groups.items():
                pos, neg = pixel_pr_counts(np.clip(z, *Z_RANGE), gt, region.astype(np.uint8), BINS, Z_RANGE)
                pooled[name][g][0] += pos
                pooled[name][g][1] += neg
        if image_id == "IDRiD_01":
            sample = (pre, ms, lines, ma_mask)
    print(f"  Done: {len(images)} training images." + " " * 20)

    rows = []
    for name in names:
        ap_all, auc_all = ap_from_counts(*pooled[name]["all"])
        rows.append({"config": name, "ap_all": ap_all, "auc_all": auc_all,
                     "ap_small": ap_from_counts(*pooled[name]["small"])[0],
                     "ap_large": ap_from_counts(*pooled[name]["large"])[0]})
    summary = pd.DataFrame(rows).set_index("config")
    summary.round(5).to_csv(OUTPUT_DIR / "vessel_summary_train.csv")
    d_rows = summary.loc[[f"D: min(C, line L={L})" for L in LENGTHS]]
    chosen = d_rows["ap_all"].idxmax()
    chosen_length = int(chosen.split("L=")[1].rstrip(")"))

    # ---------------------------------------------------------------- figure 1: AUPR vs L
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for col, label, color, marker in [("ap_all", "all MA", "#222222", "o"), ("ap_small", f"small MA (<= {SMALL} px)", BLUE, "o"),
                                      ("ap_large", f"large MA (> {LARGE} px)", ORANGE, "s")]:
        values = d_rows[col].to_numpy()
        ax.plot(LENGTHS, values, color=color, marker=marker, lw=2, ms=7, markeredgecolor="white")
        ax.axhline(summary.loc["C: multi-scale BH", col], color=color, lw=1, ls="--", alpha=0.7)
        ax.annotate(label, (LENGTHS[-1], values[-1]), xytext=(6, 0), textcoords="offset points", fontsize=9, va="center")
    ax.annotate("dashed = Model C (no vessel suppression)", (LENGTHS[0], ax.get_ylim()[1]), xytext=(0, -12),
                textcoords="offset points", fontsize=8.5, color=MUTED)
    ax.set_xticks(LENGTHS)
    ax.set_xlim(LENGTHS[0] - 4, LENGTHS[-1] + 32)
    ax.set_xlabel("Line structuring-element length L (px)")
    ax.set_ylabel("Pooled pixel AUPR")
    ax.set_title(f"Vessel suppression: AUPR vs line length (chosen L = {chosen_length})", fontsize=11, loc="left")
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig_vessel_aupr_vs_length.png", dpi=150)

    # ---------------------------------------------------------------- figure 2: maps
    pre, ms, lines, ma_mask = sample
    size = 400
    y0, x0 = densest_ma_window(ma_mask, size)
    crop = (slice(y0, y0 + size), slice(x0, x0 + size))
    contours, _ = cv2.findContours(ma_mask[crop].copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    panels = [("Model B+ image", pre["smooth"][crop], "gray", None),
              ("C: multi-scale black-hat (z)", ms[crop], "magma", (0, 8)),
              (f"line top-hat L={chosen_length} (z)", lines[chosen_length][crop], "magma", (0, 8)),
              (f"D: min of both (z)", np.minimum(ms, lines[chosen_length])[crop], "magma", (0, 8))]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.9))
    for ax, (title, img, cmap, lim) in zip(axes, panels):
        ax.imshow(img, cmap=cmap, vmin=None if lim is None else lim[0], vmax=None if lim is None else lim[1],
                  interpolation="nearest")
        for c in contours:
            pts = c[:, 0, :]
            ax.plot(np.append(pts[:, 0], pts[0, 0]), np.append(pts[:, 1], pts[0, 1]), color="#39d353", lw=0.8)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    fig.suptitle("IDRiD_01: vessel suppression (GT MA outlined in green)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig_vessel_maps_IDRiD_01.png", dpi=125)

    print(f"\n=== Pooled pixel AUPR, training set (black-hat radii {config.BLACKHAT_RADII}) ===")
    print(f"{'config':<26}{'AUPR all':>10}{'AUC all':>9}{'AUPR small':>12}{'AUPR large':>12}")
    for name, r in summary.iterrows():
        mark = "  <== chosen" if name == chosen else ""
        print(f"{name:<26}{r['ap_all']:>10.4f}{r['auc_all']:>9.4f}{r['ap_small']:>12.4f}{r['ap_large']:>12.4f}{mark}")
    print(f"Chosen line length: {chosen_length}\nOutputs: {OUTPUT_DIR}\nElapsed: {time.perf_counter() - start:.1f} s")


if __name__ == "__main__":
    main()
