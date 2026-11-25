"""
Experiment 04 - CLAHE Contrast Enhancement (training set only).

Inputs compared (to support the ablation-design decision):
  green     : raw green channel (FOV-masked)
  corrected : green after illumination correction (Exp03: gaussian 151, divisive)
For each input: no CLAHE  +  clipLimit in CLIPS  x  tiles-across in GRIDS.

Metrics per image (relative ones are vs. "no CLAHE" of the same input and image):
  global_auc     : MA vs background separability by one global threshold (higher = better)
  contrast_gain  : (mu_ring - mu_MA) after / before  -> how much deeper the MA dip becomes
  cnr_ratio      : CNR after / before -> did lesions grow MORE than local noise? (>= 1 good)
  block_ratio    : JPEG 8x8 border/inside gradient ratio (1 = no block grid visible)

Selection rule (fixed BEFORE running), per input:
  highest median global_auc among configs with median cnr_ratio >= 0.95.

Outputs: outputs/experiment_04/
Run:     python experiments/exp04_clahe.py  [--show] [--sample IDRiD_01 ...]
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
from src.dataset_loader import image_id_from_path, list_images, mask_path, read_image, read_mask
from src.enhancement import apply_clahe, tile_grid
from src.fov import disk, extract_fov_mask
from src.illumination import illumination_correct
from src.image_metrics import global_auc, jpeg_blockiness, local_contrast
from src.preprocessing import green_channel, lesion_background_ring
from src.visualization import densest_ma_window, style_axes

OUTPUT_DIR = config.OUTPUTS_DIR / "experiment_04"
CLIPS = [1.0, 2.0, 3.0, 4.0]
GRIDS = [8, 16, 32]
INPUTS = ["green", "corrected"]
MIN_CNR_RATIO = 0.95
BACKGROUND_SAMPLES = 100_000
CROP = 256
INK, MUTED = "#222222", "#888888"


def configs():
    yield ("none", 0.0, 0)
    for clip in CLIPS:
        for grid in GRIDS:
            yield ("clahe", clip, grid)


# ----------------------------------------------------------------------------- figures
def plot_heatmaps(summary: pd.DataFrame, baselines: dict, chosen: dict, path: Path):
    import matplotlib.pyplot as plt
    metrics = [("global_auc", "Global AUC", "{:.3f}"),
               ("cnr_ratio", "CNR ratio (after/before)", "{:.3f}"),
               ("contrast_gain", "MA contrast gain (x)", "{:.2f}")]
    fig, axes = plt.subplots(len(INPUTS), len(metrics), figsize=(14, 7.6))
    for r, source in enumerate(INPUTS):
        for c, (metric, title, fmt) in enumerate(metrics):
            ax = axes[r, c]
            grid_values = np.array([[summary.loc[(source, "clahe", clip, g), metric] for g in GRIDS] for clip in CLIPS])
            image = ax.imshow(grid_values, cmap="Blues", aspect="auto")
            lo, hi = grid_values.min(), grid_values.max()
            for i, clip in enumerate(CLIPS):
                for j, g in enumerate(GRIDS):
                    value = grid_values[i, j]
                    dark = (value - lo) / (hi - lo + 1e-12) > 0.6
                    ax.text(j, i, fmt.format(value), ha="center", va="center", fontsize=10,
                            color="white" if dark else INK)
                    if chosen[source] == ("clahe", clip, g):
                        ax.add_patch(matplotlib.patches.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec="#d9822b", lw=3))
            ax.set_xticks(range(len(GRIDS)), [f"{g} ({4288 // g} px)" for g in GRIDS])
            ax.set_yticks(range(len(CLIPS)), [f"{c:g}" for c in CLIPS])
            ax.set_xlabel("Tiles across (tile width)")
            if c == 0:
                ax.set_ylabel(f"{source.upper()} input\nclipLimit")
            note = f"no CLAHE = {baselines[source][metric]:.3f}" if metric == "global_auc" else "no CLAHE = 1"
            ax.set_title(f"{title}\n({note})", fontsize=10.5)
    fig.suptitle("CLAHE parameter sweep, training-set medians (orange box = chosen by rule)", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def plot_steps(base, fov, ma_mask, chosen_cfg, image_id, path):
    import matplotlib.pyplot as plt
    _, clip, grid = chosen_cfg
    variants = [("Illum.-corrected (no CLAHE)", np.clip(np.round(base), 0, 255).astype(np.uint8), MUTED),
                (f"CLAHE clip {clip:g}, {grid} tiles (chosen)", apply_clahe(base, fov, clip, grid), "#2a6fbb"),
                ("CLAHE clip 4, 32 tiles (strongest)", apply_clahe(base, fov, 4.0, 32), "#d9822b")]
    inside = fov > 0
    y0, x0 = densest_ma_window(ma_mask, CROP)
    crop = (slice(y0, y0 + CROP), slice(x0, x0 + CROP))
    contours, _ = cv2.findContours(ma_mask[crop].copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # MA closest to the crop centre -> horizontal intensity profile through it
    n, _, stats, centroids = cv2.connectedComponentsWithStats(ma_mask[crop], connectivity=8)
    k = 1 + int(np.argmin(np.hypot(*(centroids[1:] - CROP / 2).T)))
    py, px = int(round(centroids[k][1])), int(round(centroids[k][0]))
    x_range = np.arange(max(px - 40, 0), min(px + 41, CROP))

    fig, axes = plt.subplots(2, 4, figsize=(19, 9))
    for col, (title, img, color) in enumerate(variants):
        ax = axes[0, col]
        ax.imshow(np.where(inside, img, np.nan), cmap="gray", vmin=0, vmax=255)
        ax.add_patch(matplotlib.patches.Rectangle((x0, y0), CROP, CROP, fill=False, ec="yellow", lw=1.2))
        ax.set_title(title, fontsize=11)
        ax.axis("off")
        ax = axes[1, col]
        ax.imshow(img[crop], cmap="gray", vmin=0, vmax=255, interpolation="nearest")  # TRUE grey levels, no stretch
        for c in contours:
            pts = c[:, 0, :]
            ax.plot(np.append(pts[:, 0], pts[0, 0]), np.append(pts[:, 1], pts[0, 1]), color="yellow", lw=0.7)
        ax.axhline(py, color=color, lw=0.8, ls="--", xmin=x_range[0] / CROP, xmax=x_range[-1] / CROP)
        ax.set_title("Zoom, true grey levels (0-255), GT MA outlined", fontsize=9.5)
        ax.axis("off")
        axes[0, 3].hist(img[inside].ravel(), bins=128, range=(0, 255), histtype="step", lw=1.6, color=color,
                        density=True, label=title.split(" (")[0])
        axes[1, 3].plot(x_range - px, img[crop][py, x_range].astype(float), color=color, lw=1.6, label=title.split(" (")[0])
    axes[0, 3].set_title("Grey-level histogram inside FOV", fontsize=10.5)
    axes[0, 3].set_xlabel("Grey level")
    axes[0, 3].legend(frameon=False, fontsize=8)
    style_axes(axes[0, 3])
    axes[1, 3].axvspan(-stats[k, cv2.CC_STAT_WIDTH] / 2, stats[k, cv2.CC_STAT_WIDTH] / 2, color="#f2e600", alpha=0.25, lw=0)
    axes[1, 3].set_title("Profile through the central MA (yellow band = GT width)", fontsize=10.5)
    axes[1, 3].set_xlabel("Offset from MA centre (px)")
    axes[1, 3].set_ylabel("Grey level")
    style_axes(axes[1, 3])
    fig.suptitle(f"{image_id}: CLAHE on the illumination-corrected green channel", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=120)


# ----------------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(description="Experiment 04: CLAHE")
    parser.add_argument("--sample", nargs="+", default=["IDRiD_01"])
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    start = time.perf_counter()
    (OUTPUT_DIR / "enhanced").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    rows, samples = [], {}

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

        green = green_channel(image_bgr)
        fov = extract_fov_mask(image_bgr)
        inside = fov > 0
        corrected, _ = illumination_correct(green, fov)
        green_neutral = green.astype(np.float32)
        green_neutral[~inside] = green[inside].mean()   # same neutral outside-FOV handling for both inputs
        sources = {"green": green_neutral, "corrected": corrected}

        lesion = ma_mask.astype(bool) & inside
        ring = lesion_background_ring(ma_mask).astype(bool) & inside
        near_ma = cv2.dilate(ma_mask, disk(10)) > 0
        pool = np.flatnonzero((inside & ~near_ma).ravel())
        background_idx = rng.choice(pool, size=min(BACKGROUND_SAMPLES, pool.size), replace=False)
        lesion_idx = np.flatnonzero(lesion.ravel())
        valid = cv2.erode(fov, disk(10)).astype(bool)   # blockiness measured away from the rim

        for source, base in sources.items():
            reference = None
            for kind, clip, grid in configs():
                t0 = time.perf_counter()
                image = np.clip(np.round(base), 0, 255).astype(np.uint8) if kind == "none" else apply_clahe(base, fov, clip, grid)
                elapsed = time.perf_counter() - t0
                lc = local_contrast(image, lesion, ring)
                row = {"image_id": image_id, "input": source, "kind": kind, "clip": clip, "grid": grid,
                       **lc, **jpeg_blockiness(image, valid),
                       "global_auc": global_auc(image, lesion_idx, background_idx), "time_s": elapsed}
                if kind == "none":
                    reference = row
                row["contrast_gain"] = row["contrast"] / reference["contrast"]
                row["cnr_ratio"] = row["cnr"] / reference["cnr"]
                rows.append(row)

        if image_id in args.sample:
            samples[image_id] = (corrected, fov, ma_mask)
    print(f"  Done: {len(images)} training images." + " " * 20)

    table = pd.DataFrame(rows)
    table.round(5).to_csv(OUTPUT_DIR / "clahe_metrics_train.csv", index=False)
    cols = ["global_auc", "contrast_gain", "cnr_ratio", "block_ratio", "contrast", "cnr", "time_s"]
    summary = table.groupby(["input", "kind", "clip", "grid"])[cols].median()
    summary.round(5).to_csv(OUTPUT_DIR / "clahe_summary_train.csv")

    chosen, baselines = {}, {}
    for source in INPUTS:
        baselines[source] = summary.loc[(source, "none", 0.0, 0)]
        candidates = summary.loc[source].drop(index="none", level=0)
        candidates = candidates[candidates["cnr_ratio"] >= MIN_CNR_RATIO]
        chosen[source] = candidates["global_auc"].idxmax()

    plot_heatmaps(summary, baselines, chosen, OUTPUT_DIR / "fig_clahe_heatmaps.png")
    for image_id, (corrected, fov, ma_mask) in samples.items():
        plot_steps(corrected, fov, ma_mask, chosen["corrected"], image_id, OUTPUT_DIR / f"fig_clahe_steps_{image_id}.png")
        _, clip, grid = chosen["corrected"]
        out = apply_clahe(corrected, fov, clip, grid)
        cv2.imencode(".png", out)[1].tofile(str(OUTPUT_DIR / "enhanced" / f"{image_id}_clahe_c{clip:g}_g{grid}.png"))

    pd.set_option("display.width", 160)
    for source in INPUTS:
        print(f"\n=== Input: {source.upper()}  (medians over {table['image_id'].nunique()} training images) ===")
        print(f"{'config':<22}{'AUC':>8}{'contrast x':>12}{'CNR ratio':>11}{'block ratio':>13}{'time s':>8}")
        for key, r in summary.loc[source].iterrows():
            kind, clip, grid = key
            label = "no CLAHE" if kind == "none" else f"clip {clip:g}, {grid} tiles"
            mark = "  <== chosen" if key == chosen[source] else ""
            print(f"{label:<22}{r['global_auc']:>8.4f}{r['contrast_gain']:>12.2f}{r['cnr_ratio']:>11.3f}"
                  f"{r['block_ratio']:>13.3f}{r['time_s']:>8.3f}{mark}")
    for source in INPUTS:
        kind, clip, grid = chosen[source]
        tx, ty = tile_grid((2848, 4288), grid)
        print(f"Chosen for {source}: clip {clip:g}, {grid} tiles -> tileGridSize ({tx}, {ty}) ~ {4288 // tx}x{2848 // ty} px tiles")
    print(f"Outputs: {OUTPUT_DIR}\nElapsed: {time.perf_counter() - start:.1f} s")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
