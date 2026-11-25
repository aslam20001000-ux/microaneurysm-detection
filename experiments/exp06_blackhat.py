"""
Experiment 06 - Multi-Scale Black-hat Transform (training set only).

Input: Model-B+ image I (green -> illumination -> CLAHE -> Gaussian 1.5).
Compared responses (all robust-z normalised inside the FOV):
  darkness            : 255 - I                         (= Model B+ response)
  BH r                : black-hat with a disc of radius r (single scale)
  MS {r...} (z-max)   : max over scales of per-scale z-scored black-hat (proposed)
  MS {r...} raw max   : max over scales WITHOUT normalisation (shows the granulometry
                        effect: equals the largest scale)

Metric: pixel-level AUPR (average precision), the official IDRiD segmentation metric,
pooled over all training pixels; also split by lesion size:
  small MA: area <= 148 px (1st quartile)   large MA: area > 377 px (4th quartile)
(for the size-specific AUPR, pixels of the other MA are excluded from the evaluation).

Selection rule (fixed BEFORE running): highest pooled AUPR (all MA).

Outputs: outputs/experiment_06/
Run:     python experiments/exp06_blackhat.py
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import matplotlib
import numpy as np
import pandas as pd

from src import config
from src.candidates import blackhat, preprocess, robust_z
from src.dataset_loader import image_id_from_path, list_images, mask_path, read_image, read_mask
from src.evaluation import ap_from_counts, pixel_pr_counts
from src.visualization import densest_ma_window, style_axes

OUTPUT_DIR = config.OUTPUTS_DIR / "experiment_06"
RADII = [3, 5, 8, 12, 16, 24]
MULTI = [(3, 5, 8), (3, 5, 8, 12), (3, 5, 8, 12, 16), (5, 8, 12, 16, 24), (3, 5, 8, 12, 16, 24)]
Z_RANGE = (-10.0, 150.0)
BINS = 8000
SMALL, LARGE = 148, 377
INK, MUTED = "#222222", "#888888"
BLUE, ORANGE, PURPLE = "#2a6fbb", "#d9822b", "#8e5bc4"


def config_names():
    names = ["darkness"] + [f"BH r={r}" for r in RADII] + [f"MS {set(m)}".replace(" ", "") for m in MULTI]
    return names + ["MS{3,5,8,12} raw max"]


def responses(pre: dict) -> dict:
    fov, image = pre["fov"], pre["smooth"]
    bh = {r: blackhat(image, r) for r in RADII}
    bz = {r: robust_z(bh[r], fov) for r in RADII}
    out = {"darkness": robust_z(255 - image.astype(np.float32), fov)}
    out.update({f"BH r={r}": bz[r] for r in RADII})
    for m in MULTI:
        out[f"MS {set(m)}".replace(" ", "")] = np.max([bz[r] for r in m], axis=0)
    out["MS{3,5,8,12} raw max"] = robust_z(np.max([bh[r] for r in (3, 5, 8, 12)], axis=0), fov)
    return out


def plot_ap(summary: pd.DataFrame, chosen: str, path: Path):
    import matplotlib.pyplot as plt
    order = summary.sort_values("ap_all")
    fig, ax = plt.subplots(figsize=(9, 6))
    y = np.arange(len(order))
    colors = [ORANGE if n == chosen else (MUTED if n == "darkness" else BLUE) for n in order.index]
    ax.barh(y, order["ap_all"], color=colors, height=0.6)
    for yi, v in zip(y, order["ap_all"]):
        ax.text(v + 0.003, yi, f"{v:.3f}", va="center", fontsize=9, color=INK)
    ax.set_yticks(y, order.index, fontsize=9)
    ax.set_xlabel("Pooled pixel AUPR (training set, all MA)")
    ax.set_title(f"Black-hat configurations (orange = chosen: {chosen}; grey = no black-hat)", fontsize=11, loc="left")
    style_axes(ax)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def plot_scale_selectivity(summary: pd.DataFrame, path: Path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for col, name, color, marker in [("ap_small", f"small MA (<= {SMALL} px)", BLUE, "o"),
                                     ("ap_large", f"large MA (> {LARGE} px)", ORANGE, "s")]:
        vals = [summary.loc[f"BH r={r}", col] for r in RADII]
        ax.plot(RADII, vals, color=color, marker=marker, lw=2, ms=7, markeredgecolor="white", label=name)
        ax.annotate(name, (RADII[-1], vals[-1]), xytext=(6, 0), textcoords="offset points", fontsize=9,
                    va="center", color=INK)
    ax.set_xticks(RADII)
    ax.set_xlim(RADII[0] - 1, RADII[-1] + 9)
    ax.set_xlabel("Black-hat disc radius r (px)")
    ax.set_ylabel("Pooled pixel AUPR")
    ax.set_title("Scale selectivity: each lesion size prefers a different radius", fontsize=11, loc="left")
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def plot_maps(pre, maps, ma_mask, chosen, image_id, path):
    import matplotlib.pyplot as plt
    crop_size = 320
    y0, x0 = densest_ma_window(ma_mask, crop_size)
    crop = (slice(y0, y0 + crop_size), slice(x0, x0 + crop_size))
    contours, _ = cv2.findContours(ma_mask[crop].copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    panels = [("Model B+ image", pre["smooth"][crop], "gray", None),
              ("darkness z", maps["darkness"][crop], "magma", (0, 8)),
              ("BH r=3 z", maps["BH r=3"][crop], "magma", (0, 8)),
              ("BH r=12 z", maps["BH r=12"][crop], "magma", (0, 8)),
              (f"{chosen} z", maps[chosen][crop], "magma", (0, 8))]
    fig, axes = plt.subplots(1, 5, figsize=(21, 4.6))
    for ax, (title, img, cmap, lim) in zip(axes, panels):
        ax.imshow(img, cmap=cmap, vmin=None if lim is None else lim[0], vmax=None if lim is None else lim[1],
                  interpolation="nearest")
        for c in contours:
            pts = c[:, 0, :]
            ax.plot(np.append(pts[:, 0], pts[0, 0]), np.append(pts[:, 1], pts[0, 1]), color="#39d353", lw=0.8)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    fig.suptitle(f"{image_id}: response maps (z-score, 0-8), GT MA outlined in green", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=125)


def main():
    parser = argparse.ArgumentParser(description="Experiment 06: multi-scale black-hat")
    parser.add_argument("--sample", nargs="+", default=["IDRiD_01"])
    args = parser.parse_args()
    matplotlib.use("Agg")

    start = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    names = config_names()
    pooled = {n: {g: [np.zeros(BINS), np.zeros(BINS)] for g in ("all", "small", "large")} for n in names}
    image_rows = []

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
        n, labels, stats, _ = cv2.connectedComponentsWithStats(ma_mask, connectivity=8)
        area = stats[:, cv2.CC_STAT_AREA]
        lesion_ids = np.arange(1, n)                      # label 0 = background
        small = np.isin(labels, lesion_ids[area[1:] <= SMALL])
        large = np.isin(labels, lesion_ids[area[1:] > LARGE])
        inside, lesion = fov > 0, ma_mask > 0
        groups = {"all": (lesion, inside),
                  "small": (small, inside & ~(lesion & ~small)),   # other MA excluded from evaluation
                  "large": (large, inside & ~(lesion & ~large))}

        maps = responses(pre)
        for name, z in maps.items():
            for g, (gt, region) in groups.items():
                pos, neg = pixel_pr_counts(np.clip(z, *Z_RANGE), gt, region.astype(np.uint8), BINS, Z_RANGE)
                pooled[name][g][0] += pos
                pooled[name][g][1] += neg
                if g == "all":
                    image_rows.append({"image_id": image_id, "config": name, "ap": ap_from_counts(pos, neg)[0]})
        if image_id in args.sample:
            sample = (image_id, pre, maps, ma_mask)
    print(f"  Done: {len(images)} training images." + " " * 20)

    rows = []
    for name in names:
        ap_all, auc_all = ap_from_counts(*pooled[name]["all"])
        rows.append({"config": name, "ap_all": ap_all, "auc_all": auc_all,
                     "ap_small": ap_from_counts(*pooled[name]["small"])[0],
                     "ap_large": ap_from_counts(*pooled[name]["large"])[0]})
    summary = pd.DataFrame(rows).set_index("config")
    per_image = pd.DataFrame(image_rows)
    summary["ap_median_per_image"] = per_image.groupby("config")["ap"].median()
    summary.round(5).to_csv(OUTPUT_DIR / "blackhat_summary_train.csv")
    per_image.round(5).to_csv(OUTPUT_DIR / "blackhat_per_image_train.csv", index=False)

    chosen = summary["ap_all"].idxmax()
    plot_ap(summary, chosen, OUTPUT_DIR / "fig_blackhat_aupr.png")
    plot_scale_selectivity(summary, OUTPUT_DIR / "fig_blackhat_scale_selectivity.png")
    image_id, pre, maps, ma_mask = sample
    plot_maps(pre, maps, ma_mask, chosen, image_id, OUTPUT_DIR / f"fig_blackhat_maps_{image_id}.png")

    print(f"\n=== Pooled pixel AUPR, training set ===")
    print(f"{'config':<26}{'AUPR all':>10}{'AUC all':>9}{'AUPR small':>12}{'AUPR large':>12}{'median/img':>12}")
    for name, r in summary.iterrows():
        mark = "  <== chosen" if name == chosen else ""
        print(f"{name:<26}{r['ap_all']:>10.4f}{r['auc_all']:>9.4f}{r['ap_small']:>12.4f}{r['ap_large']:>12.4f}"
              f"{r['ap_median_per_image']:>12.4f}{mark}")
    print(f"Outputs: {OUTPUT_DIR}\nElapsed: {time.perf_counter() - start:.1f} s")


if __name__ == "__main__":
    main()
