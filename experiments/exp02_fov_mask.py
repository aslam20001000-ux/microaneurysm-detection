"""
Experiment 02 - Field of View (FOV) Mask Extraction.

IDRiD provides NO ground-truth FOV masks, so the FOV is validated indirectly
(training set only):
  A. Threshold sweep  : FOV area vs threshold T -> a flat plateau means T is not critical.
  B. Otsu comparison  : data-driven threshold vs our fixed T.
  C. Lesion coverage  : % of GT MA pixels inside the FOV (must be 100 %).
  D. Shape check      : solidity = area / convex-hull area (~1 for a clean disc).
  E. Rim analysis     : relative green intensity vs distance to the FOV edge, and how many
                        GT MA lie within that distance -> chooses the erosion radius.
  F. Visual check     : step-by-step figure + contact sheet of all 54 training masks.

Outputs: outputs/experiment_02/  and  outputs/masks/fov/<id>_fov.png (train only)
Run:     python experiments/exp02_fov_mask.py  [--show] [--sample IDRiD_01 ...]
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
from scipy import ndimage

from src import config
from src.dataset_loader import image_id_from_path, list_images, mask_path, read_image, read_mask
from src.fov import extract_fov_mask

OUTPUT_DIR = config.OUTPUTS_DIR / "experiment_02"
MASK_DIR = config.OUTPUTS_DIR / "masks" / "fov"
SWEEP_THRESHOLDS = [5, 10, 15, 20, 30, 40, 60, 80]
RIM_MAX_PX = 200        # analyse the first 200 px inside the rim
REFERENCE_BAND = (150, 200)  # "interior" band used to normalise the rim profile
INK, MUTED, ACCENT = "#222222", "#888888", "#1e8449"


def rim_profile(green: np.ndarray, distance: np.ndarray) -> np.ndarray:
    """Mean green intensity for each 1-px distance bin, normalised by the interior band."""
    inside = (distance > 0) & (distance < RIM_MAX_PX)
    bins = distance[inside].astype(int)
    sums = np.bincount(bins, weights=green[inside], minlength=RIM_MAX_PX)
    counts = np.bincount(bins, minlength=RIM_MAX_PX)
    profile = np.divide(sums, counts, out=np.full(RIM_MAX_PX, np.nan), where=counts > 0)
    reference = np.nanmean(profile[REFERENCE_BAND[0]:REFERENCE_BAND[1]])
    return profile / reference


def lesion_min_distance(ma_mask: np.ndarray, distance: np.ndarray) -> np.ndarray:
    """Smallest distance-to-rim of every GT MA lesion (one value per lesion)."""
    labels, n = ndimage.label(ma_mask, structure=np.ones((3, 3)))
    if n == 0:
        return np.array([])
    return np.asarray(ndimage.minimum(distance, labels, index=np.arange(1, n + 1)))


def solidity(mask: np.ndarray) -> float:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    return float(cv2.contourArea(contour) / cv2.contourArea(cv2.convexHull(contour)))


# ----------------------------------------------------------------------------- figures
def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#e5e5e5", lw=0.6)
    ax.set_axisbelow(True)


def plot_steps(image_bgr, steps, image_id, path):
    import matplotlib.pyplot as plt
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    overlay = rgb.copy()
    contours, _ = cv2.findContours(steps["fov"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(overlay, contours, -1, (255, 255, 0), 12)
    panels = [
        ("1. Red channel", steps["red"], "gray"),
        (f"2. Threshold  red > {config.FOV_THRESHOLD}", steps["thresholded"], "gray"),
        (f"3. Opening r={config.FOV_OPEN_PX} + largest comp.", steps["opened"], "gray"),
        ("4. Holes filled", steps["filled"], "gray"),
        (f"5. Final FOV (erode r={config.FOV_ERODE_PX})", steps["fov"], "gray"),
        ("6. FOV contour on image", overlay, None),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.2))
    for ax, (title, img, cmap) in zip(axes.flat, panels):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    fig.suptitle(f"{image_id}: FOV extraction steps", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def plot_sweep(sweep: pd.DataFrame, otsu: np.ndarray, path):
    import matplotlib.pyplot as plt
    stats = sweep.groupby("threshold")["area_pct"].agg(["mean", "min", "max"])
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.fill_between(stats.index, stats["min"], stats["max"], color=ACCENT, alpha=0.18, lw=0, label="min-max over images")
    ax.plot(stats.index, stats["mean"], color=ACCENT, lw=2, marker="o", ms=6, label="mean")
    ax.axvline(config.FOV_THRESHOLD, color=INK, lw=1, ls="--")
    ax.annotate(f"chosen T = {config.FOV_THRESHOLD}", (config.FOV_THRESHOLD + 1.5, stats["max"].max() - 0.3), fontsize=9, color=INK)
    ax.axvspan(otsu.min(), otsu.max(), color=MUTED, alpha=0.15, lw=0)
    ax.annotate(f"Otsu range {otsu.min():.0f}-{otsu.max():.0f}", (otsu.min() + 1, stats["min"].min() + 0.2), fontsize=9, color=MUTED)
    ax.set_xlabel("Red-channel threshold T")
    ax.set_ylabel("FOV area (% of image)")
    ax.set_title(f"FOV area vs threshold (training set, n = {sweep['image_id'].nunique()})", fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def plot_rim(profiles: np.ndarray, d: np.ndarray, ma_distances: np.ndarray, n_ma: int, path):
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 7), sharex=True)
    ax1.fill_between(d, np.nanpercentile(profiles, 10, axis=0), np.nanpercentile(profiles, 90, axis=0),
                     color=ACCENT, alpha=0.18, lw=0, label="10th-90th percentile")
    ax1.plot(d, np.nanmedian(profiles, axis=0), color=ACCENT, lw=2, label="median")
    ax1.axhline(1.0, color=MUTED, lw=0.8)
    for ax in (ax1, ax2):
        ax.axvline(config.FOV_ERODE_PX, color=INK, lw=1, ls="--")
    ax1.annotate(f"erosion r = {config.FOV_ERODE_PX} px", (config.FOV_ERODE_PX + 3, 0.3), fontsize=9, color=INK)
    ax1.set_ylabel("Green intensity / interior level")
    ax1.set_title("A. Rim darkening: green level vs distance from FOV edge", fontsize=11, loc="left")
    ax1.legend(frameon=False, fontsize=9)
    style(ax1)
    cumulative = [(ma_distances <= x).sum() for x in d]
    ax2.plot(d, cumulative, color=INK, lw=2)
    ax2.set_ylabel(f"GT MA within distance (of {n_ma})")
    ax2.set_xlabel("Distance from FOV edge (px)")
    ax2.set_title("B. Cost of erosion: lesions lost if rim of this width is removed", fontsize=11, loc="left")
    style(ax2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def plot_contact_sheet(thumbs, path):
    import matplotlib.pyplot as plt
    cols = 9
    rows = int(np.ceil(len(thumbs) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 1.6))
    for ax in axes.flat:
        ax.axis("off")
    for ax, (image_id, thumb) in zip(axes.flat, thumbs):
        ax.imshow(thumb)
        ax.set_title(image_id.replace("IDRiD_", ""), fontsize=8)
    fig.suptitle("FOV contour (yellow) on every training image", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


# ----------------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(description="Experiment 02: FOV mask extraction")
    parser.add_argument("--sample", nargs="+", default=["IDRiD_01"])
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    start = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MASK_DIR.mkdir(parents=True, exist_ok=True)

    sweep_rows, metric_rows, profiles, ma_distances, thumbs, otsu_values = [], [], [], [], [], []
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
        red = image_bgr[:, :, 2]

        # A. threshold sweep (raw threshold only, to isolate the effect of T)
        for t in SWEEP_THRESHOLDS:
            sweep_rows.append({"image_id": image_id, "threshold": t, "area_pct": 100 * np.mean(red > t)})
        # B. Otsu
        otsu_t, _ = cv2.threshold(red, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        otsu_values.append(otsu_t)

        # Full pipeline with the configured parameters
        fov, steps = extract_fov_mask(image_bgr, return_steps=True)
        n_raw_components = cv2.connectedComponents(steps["thresholded"], connectivity=8)[0] - 1

        # E. rim analysis on the un-eroded mask
        distance = cv2.distanceTransform(steps["filled"], cv2.DIST_L2, 5)
        profiles.append(rim_profile(image_bgr[:, :, 1].astype(np.float64), distance))
        ma_distances.append(lesion_min_distance(ma_mask, distance))

        metric_rows.append({
            "image_id": image_id,
            "otsu_threshold": otsu_t,
            "raw_components": n_raw_components,
            "fov_area_pct": 100 * fov.mean(),
            "solidity": solidity(fov),
            "ma_pixels_inside_fov_pct": 100 * (ma_mask & fov).sum() / max(ma_mask.sum(), 1),
            "otsu_vs_fixed_area_diff_pct": 100 * (np.mean(red > otsu_t) - np.mean(red > config.FOV_THRESHOLD)),
        })

        cv2.imencode(".png", fov * 255)[1].tofile(str(MASK_DIR / f"{image_id}_fov.png"))

        # F. thumbnail (display only; full-resolution data untouched)
        small = cv2.resize(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), None, fx=1 / 16, fy=1 / 16, interpolation=cv2.INTER_AREA)
        small_fov = cv2.resize(fov, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours(small_fov, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(small, contours, -1, (255, 255, 0), 1)
        thumbs.append((image_id, small))

        if image_id in args.sample:
            plot_steps(image_bgr, steps, image_id, OUTPUT_DIR / f"fig_fov_steps_{image_id}.png")
    print(f"  Done: {len(images)} training images." + " " * 20)

    sweep = pd.DataFrame(sweep_rows)
    metrics = pd.DataFrame(metric_rows)
    # Bin 0 is always empty (every FOV pixel is >= 1 px from the edge) -> drop it
    profiles = np.vstack(profiles)[:, 1:]
    distances = np.arange(1, RIM_MAX_PX)
    all_ma_dist = np.concatenate(ma_distances)
    otsu_values = np.array(otsu_values)

    metrics.round(4).to_csv(OUTPUT_DIR / "fov_metrics_train.csv", index=False)
    sweep.round(4).to_csv(OUTPUT_DIR / "threshold_sweep_train.csv", index=False)
    rim_table = pd.DataFrame({
        "distance_px": distances,
        "green_relative_median": np.nanmedian(profiles, axis=0),
        "green_relative_p10": np.nanpercentile(profiles, 10, axis=0),
        "ma_within_distance": [(all_ma_dist <= x).sum() for x in distances],
    }).set_index("distance_px")
    rim_table.round(4).to_csv(OUTPUT_DIR / "rim_profile_train.csv")

    plot_sweep(sweep, otsu_values, OUTPUT_DIR / "fig_threshold_sweep.png")
    plot_rim(profiles, distances, all_ma_dist, all_ma_dist.size, OUTPUT_DIR / "fig_rim_profile.png")
    plot_contact_sheet(thumbs, OUTPUT_DIR / "fig_fov_contact_sheet.png")

    # ------------------------------------------------------------------ report
    print("\n=== A. Threshold sweep: FOV area % (mean [min-max]) ===")
    for t, g in sweep.groupby("threshold")["area_pct"]:
        print(f"  T={t:>3}: {g.mean():6.2f} [{g.min():.2f}-{g.max():.2f}]")
    print(f"\n=== B. Otsu threshold: median {np.median(otsu_values):.0f} [range {otsu_values.min():.0f}-{otsu_values.max():.0f}]")
    print(f"  Area lost by Otsu vs T={config.FOV_THRESHOLD}: median {metrics['otsu_vs_fixed_area_diff_pct'].median():.2f} "
          f"% of image (worst {metrics['otsu_vs_fixed_area_diff_pct'].min():.2f})")
    print(f"\n=== C/D. Final FOV (T={config.FOV_THRESHOLD}, open={config.FOV_OPEN_PX}, erode={config.FOV_ERODE_PX}) ===")
    print(f"  FOV area %          : median {metrics['fov_area_pct'].median():.2f} [{metrics['fov_area_pct'].min():.2f}-{metrics['fov_area_pct'].max():.2f}]")
    print(f"  Solidity            : min {metrics['solidity'].min():.4f}, median {metrics['solidity'].median():.4f}")
    print(f"  Raw components (T)  : median {metrics['raw_components'].median():.0f}, max {metrics['raw_components'].max()}")
    print(f"  MA pixels inside FOV: min {metrics['ma_pixels_inside_fov_pct'].min():.2f} %  "
          f"(images < 100 %: {(metrics['ma_pixels_inside_fov_pct'] < 100).sum()})")
    print("\n=== E. Rim profile (median green / interior) and MA lost by erosion ===")
    for d in [1, 2, 3, 4, 5, 6, 7, 10, 20, 50, 100, 150]:
        r = rim_table.loc[d]
        print(f"  d={d:>3} px: green {r['green_relative_median']:.3f} (p10 {r['green_relative_p10']:.3f}) | "
              f"MA within d: {int(r['ma_within_distance'])} / {all_ma_dist.size}")
    print(f"\nOutputs: {OUTPUT_DIR}\nMasks  : {MASK_DIR}\nElapsed: {time.perf_counter() - start:.1f} s")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
