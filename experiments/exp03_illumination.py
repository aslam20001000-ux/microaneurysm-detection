"""
Experiment 03 - Illumination Correction (training set only).

Compared configurations
  none                       : green channel as is (baseline)
  median   size in SIZES     : median-filter background, divisive correction
  gaussian size in SIZES     : masked-Gaussian background (sigma = size/4), divisive correction

Metrics per image (GT MA masks are used ONLY to measure, never inside the algorithm):
  1. block_cv        : illumination uniformity. FOV split into 128x128 blocks (MA excluded);
                       CV = std(block means) / mean(block means).        lower  = flatter
  2. weber_retention : MA local Weber contrast after / before correction.
                       ~1 = lesions preserved; < 1 = background filter "ate" the lesions
  3. global_auc      : ROC-AUC of (-intensity) separating MA pixels from background pixels
                       anywhere in the FOV. This is what a single GLOBAL threshold
                       (Model A) can achieve at best.                    higher = better

Selection rule (fixed BEFORE looking at the results):
  choose the configuration with the highest median global_auc among those whose
  median weber_retention >= 0.95.

Outputs: outputs/experiment_03/
Run:     python experiments/exp03_illumination.py  [--show] [--sample IDRiD_01 ...]
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
from sklearn.metrics import roc_auc_score

from src import config
from src.dataset_loader import image_id_from_path, list_images, mask_path, read_image, read_mask
from src.fov import disk, extract_fov_mask
from src.illumination import correct_illumination, estimate_background
from src.preprocessing import green_channel, lesion_background_ring
from src.visualization import densest_ma_window, style_axes

OUTPUT_DIR = config.OUTPUTS_DIR / "experiment_03"
SIZES = [51, 101, 151, 201, 301]
CONFIGS = [("none", 0)] + [("median", s) for s in SIZES] + [("gaussian", s) for s in SIZES]
BLOCK = 128
BACKGROUND_SAMPLES = 100_000
MIN_RETENTION = 0.95
COLORS = {"median": "#2a6fbb", "gaussian": "#d9822b"}   # validated: CVD dE 25, normal dE 32
MARKERS = {"median": "o", "gaussian": "s"}
INK, MUTED = "#222222", "#888888"
CROP = 256


# ----------------------------------------------------------------------------- metrics
def block_cv(image: np.ndarray, valid: np.ndarray) -> float:
    """CV of 128x128 block means over blocks that are >= 90 % valid."""
    h = image.shape[0] // BLOCK * BLOCK
    w = image.shape[1] // BLOCK * BLOCK
    img = image[:h, :w].astype(np.float64)
    val = valid[:h, :w].astype(np.float64)
    sums = (img * val).reshape(h // BLOCK, BLOCK, w // BLOCK, BLOCK).sum(axis=(1, 3))
    counts = val.reshape(h // BLOCK, BLOCK, w // BLOCK, BLOCK).sum(axis=(1, 3))
    good = counts >= 0.9 * BLOCK * BLOCK
    means = sums[good] / counts[good]
    return float(means.std() / means.mean())


def weber(image: np.ndarray, lesion: np.ndarray, ring: np.ndarray) -> float:
    mu_l, mu_b = image[lesion].mean(), image[ring].mean()
    return float((mu_b - mu_l) / mu_b)


def global_auc(image: np.ndarray, lesion_idx: np.ndarray, background_idx: np.ndarray) -> float:
    flat = image.ravel()
    scores = np.concatenate([flat[lesion_idx], flat[background_idx]]).astype(np.float32)
    labels = np.concatenate([np.ones(lesion_idx.size), np.zeros(background_idx.size)])
    return float(roc_auc_score(labels, -scores))   # MA are dark -> lower intensity = more likely MA


# ----------------------------------------------------------------------------- figures
def plot_sweep(summary: pd.DataFrame, baseline: pd.Series, chosen: tuple, path: Path):
    import matplotlib.pyplot as plt
    panels = [
        ("block_cv", "A. Illumination non-uniformity (CV)", "lower is better"),
        ("weber_retention", "B. MA contrast retention", "1 = lesions preserved"),
        ("global_auc", "C. Global MA separability (AUC)", "higher is better"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, (metric, title, note) in zip(axes, panels):
        for method in ["median", "gaussian"]:
            part = summary.loc[method]
            ax.plot(part.index, part[metric], color=COLORS[method], marker=MARKERS[method], ms=7, lw=2,
                    markeredgecolor="white", markeredgewidth=1.5, label=method.capitalize())
            ax.annotate(method.capitalize(), (part.index[-1] + 8, part[metric].iloc[-1]),
                        fontsize=9, color=INK, va="center")
        ax.axhline(baseline[metric], color=MUTED, lw=1.2, ls="--")
        ax.annotate("no correction", (SIZES[0], baseline[metric]), xytext=(0, 4), textcoords="offset points",
                    fontsize=8, color=MUTED)
        if metric == "weber_retention":
            ax.axhline(MIN_RETENTION, color=INK, lw=0.8, ls=":")
            ax.annotate(f"constraint {MIN_RETENTION}", (SIZES[-1], MIN_RETENTION), xytext=(-70, -12),
                        textcoords="offset points", fontsize=8, color=INK)
        chosen_value = summary.loc[chosen, metric]
        ax.scatter([chosen[1]], [chosen_value], s=180, facecolors="none", edgecolors=INK, lw=1.5, zorder=4)
        ax.set_title(f"{title}\n({note})", fontsize=10.5, loc="left")
        ax.set_xlabel("Background kernel size (px, full resolution)")
        ax.set_xticks(SIZES)
        ax.set_xlim(SIZES[0] - 20, SIZES[-1] + 70)
        style_axes(ax)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle(f"Illumination-correction sweep, training set median over images (circle = chosen: {chosen[0]} {chosen[1]})",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def plot_steps(green, fov, background, corrected, ma_mask, image_id, label, path):
    import matplotlib.pyplot as plt
    inside = fov > 0
    green_view = np.where(inside, green, np.nan)
    y0, x0 = densest_ma_window(ma_mask, CROP)
    crop = (slice(y0, y0 + CROP), slice(x0, x0 + CROP))
    contours, _ = cv2.findContours(ma_mask[crop].copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    fig, axes = plt.subplots(2, 3, figsize=(16, 8.6))
    vmin, vmax = np.percentile(green[inside], [1, 99])
    for ax, img, title in [(axes[0, 0], green_view, "1. Green channel (FOV)"),
                           (axes[0, 1], np.where(inside, background, np.nan), f"2. Background B ({label})"),
                           (axes[0, 2], np.where(inside, corrected, np.nan), "3. Corrected  I / B x mean")]:
        lo, hi = (vmin, vmax) if "Background" not in title else np.nanpercentile(img, [1, 99])
        ax.imshow(img, cmap="gray", vmin=lo, vmax=hi)
        ax.add_patch(matplotlib.patches.Rectangle((x0, y0), CROP, CROP, fill=False, ec="yellow", lw=1.2))
        ax.axhline(green.shape[0] // 2, color="#2a6fbb", lw=0.8, ls="--")
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    for ax, img, title in [(axes[1, 0], green, "Zoom: before"), (axes[1, 1], corrected, "Zoom: after")]:
        patch = img[crop].astype(np.float32)
        lo, hi = np.percentile(patch, [1, 99])
        ax.imshow(patch, cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
        for c in contours:
            pts = c[:, 0, :]
            ax.plot(np.append(pts[:, 0], pts[0, 0]), np.append(pts[:, 1], pts[0, 1]), color="yellow", lw=0.7)
        ax.set_title(f"{title} (GT MA outlined, display stretch 1-99 %)", fontsize=9.5)
        ax.axis("off")

    ax = axes[1, 2]
    row = green.shape[0] // 2
    cols = np.where(inside[row])[0]
    kernel = np.ones(25) / 25   # 25-px moving average: display only, to make the trend readable
    for values, name, color in [(green[row, cols].astype(float), "before", MUTED),
                                (corrected[row, cols].astype(float), "after", "#2a6fbb")]:
        # mode="valid": no zero padding, so the curve ends are not artificially pulled down
        ax.plot(cols[12:-12], np.convolve(values, kernel, mode="valid"), color=color, lw=1.6, label=name)
    ax.set_title("Intensity along the dashed centre row (25-px smoothed)", fontsize=10)
    ax.set_xlabel("Column (px)")
    ax.set_ylabel("Green level")
    ax.legend(frameon=False, fontsize=9)
    style_axes(ax)
    fig.suptitle(f"{image_id}: illumination correction ({label})", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def plot_background_sizes(green, fov, image_id, path):
    import matplotlib.pyplot as plt
    inside = fov > 0
    show_sizes = [51, 151, 301]
    fig, axes = plt.subplots(2, 3, figsize=(14, 6.6))
    for r, method in enumerate(["median", "gaussian"]):
        for c, size in enumerate(show_sizes):
            b = estimate_background(green, fov, method=method, size=size)
            ax = axes[r, c]
            ax.imshow(np.where(inside, b, np.nan), cmap="gray", vmin=np.percentile(b[inside], 1),
                      vmax=np.percentile(b[inside], 99))
            ax.set_title(f"{method.capitalize()}, size {size} px", fontsize=11)
            ax.axis("off")
    fig.suptitle(f"{image_id}: background estimate B vs kernel size  (vessels visible in B = kernel too small)", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


# ----------------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(description="Experiment 03: illumination correction")
    parser.add_argument("--sample", nargs="+", default=["IDRiD_01"])
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    start = time.perf_counter()
    (OUTPUT_DIR / "corrected").mkdir(parents=True, exist_ok=True)
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
        lesion = ma_mask.astype(bool) & (fov > 0)
        ring = lesion_background_ring(ma_mask).astype(bool) & (fov > 0)
        near_ma = cv2.dilate(ma_mask, disk(10)) > 0
        background_pool = np.flatnonzero(((fov > 0) & ~near_ma).ravel())
        background_idx = rng.choice(background_pool, size=min(BACKGROUND_SAMPLES, background_pool.size), replace=False)
        lesion_idx = np.flatnonzero(lesion.ravel())
        valid_blocks = (fov > 0) & ~near_ma
        weber_before = weber(green.astype(np.float32), lesion, ring)

        for method, size in CONFIGS:
            t0 = time.perf_counter()
            if method == "none":
                image = green.astype(np.float32)
                background = None
            else:
                background = estimate_background(green, fov, method=method, size=size)
                image = correct_illumination(green, fov, background)
            elapsed = time.perf_counter() - t0
            rows.append({
                "image_id": image_id, "method": method, "size": size,
                "block_cv": block_cv(image, valid_blocks),
                "weber": weber(image, lesion, ring),
                "weber_retention": weber(image, lesion, ring) / weber_before,
                "global_auc": global_auc(image, lesion_idx, background_idx),
                "time_s": elapsed,
            })
        if image_id in args.sample:
            samples[image_id] = (green, fov, ma_mask)
    print(f"  Done: {len(images)} training images." + " " * 20)

    table = pd.DataFrame(rows)
    table.round(5).to_csv(OUTPUT_DIR / "illumination_metrics_train.csv", index=False)
    metrics = ["block_cv", "weber_retention", "global_auc", "time_s"]
    summary = table.groupby(["method", "size"])[metrics].median()
    q25 = table.groupby(["method", "size"])[metrics].quantile(0.25)
    q75 = table.groupby(["method", "size"])[metrics].quantile(0.75)
    summary.round(5).to_csv(OUTPUT_DIR / "illumination_summary_train.csv")

    # Selection rule (defined in the docstring before running)
    candidates = summary.drop(index="none", level=0)
    candidates = candidates[candidates["weber_retention"] >= MIN_RETENTION]
    chosen = candidates["global_auc"].idxmax()
    baseline = summary.loc[("none", 0)]

    # How many images improve in AUC with the chosen config vs none?
    pivot = table.pivot_table(index="image_id", columns=["method", "size"], values="global_auc")
    improved = int((pivot[chosen] > pivot[("none", 0)]).sum())

    plot_sweep(summary.drop(index="none", level=0), baseline, chosen, OUTPUT_DIR / "fig_illumination_sweep.png")
    for image_id, (green, fov, ma_mask) in samples.items():
        background = estimate_background(green, fov, method=chosen[0], size=chosen[1])
        corrected = correct_illumination(green, fov, background)
        cv2.imencode(".png", corrected.round().astype(np.uint8))[1].tofile(
            str(OUTPUT_DIR / "corrected" / f"{image_id}_illum_{chosen[0]}{chosen[1]}.png"))
        plot_steps(green, fov, background, corrected, ma_mask, image_id, f"{chosen[0]} {chosen[1]} px",
                   OUTPUT_DIR / f"fig_illumination_steps_{image_id}.png")
        plot_background_sizes(green, fov, image_id, OUTPUT_DIR / f"fig_background_sizes_{image_id}.png")

    print("\n=== Median [IQR] over 54 training images ===")
    print(f"{'config':<16}{'block_cv':>24}{'weber_retention':>24}{'global_auc':>24}{'time_s':>9}")
    for key in summary.index:
        label = "none" if key[0] == "none" else f"{key[0]} {key[1]}"
        cells = [f"{summary.loc[key, m]:.4f} [{q25.loc[key, m]:.3f}-{q75.loc[key, m]:.3f}]" for m in metrics[:3]]
        mark = "  <== chosen" if key == chosen else ""
        print(f"{label:<16}" + "".join(f"{c:>24}" for c in cells) + f"{summary.loc[key, 'time_s']:>9.3f}{mark}")
    print(f"\nChosen (max AUC with retention >= {MIN_RETENTION}): {chosen[0]} {chosen[1]} px")
    print(f"Images whose global AUC improved vs no correction: {improved} / {pivot.shape[0]}")
    print(f"Outputs: {OUTPUT_DIR}\nElapsed: {time.perf_counter() - start:.1f} s")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
