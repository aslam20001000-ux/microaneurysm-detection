"""
Experiment 05 - Noise Reduction (training set only).

Input: Model-B image = green -> illumination correction (gaussian 151) -> CLAHE (clip 1, 8 tiles).

Filters compared: none, Gaussian (sigma), Median (k), Bilateral (sigma_color, sigma_space).

Per-lesion metrics (each MA vs ITS OWN 3-10 px ring, so regional illumination
differences do not inflate the noise term):
  cnr                  : median over lesions of (mu_ring - mu_MA) / sd_ring        higher = better
  contrast_retention   : median MA contrast after / before filtering (all lesions)
  small_retention      : same, only for SMALL MA (area <= 25th percentile = 148 px)
Image-level metrics:
  block_ratio          : JPEG 8x8 border/inside gradient (1 = block grid gone)
  global_auc           : MA vs background separability for one global threshold

Selection rule (fixed BEFORE running):
  highest median cnr among configs with median small_retention >= 0.90
  (a filter may smooth noise, but must keep >= 90 % of the depth of the smallest MA).

Outputs: outputs/experiment_05/
Run:     python experiments/exp05_denoise.py  [--show] [--sample IDRiD_01 ...]
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
from src.denoise import denoise
from src.enhancement import apply_clahe
from src.fov import disk, extract_fov_mask
from src.illumination import illumination_correct
from src.image_metrics import global_auc, jpeg_blockiness, lesion_labels_and_rings, per_lesion_stats
from src.preprocessing import green_channel
from src.visualization import densest_ma_window, style_axes

OUTPUT_DIR = config.OUTPUTS_DIR / "experiment_05"
SMALL_AREA = 148          # 25th percentile of GT MA area on the training set (px)
SIZE_BINS = [0, 148, 235, 377, np.inf]   # training-set area quartiles
SIZE_LABELS = ["<=148", "149-235", "236-377", ">377"]
MIN_SMALL_RETENTION = 0.90
BACKGROUND_SAMPLES = 100_000
CROP = 256
CONFIGS = [("none", 0, 0),
           ("gaussian", 0.7, 0), ("gaussian", 1.0, 0), ("gaussian", 1.5, 0), ("gaussian", 2.0, 0),
           ("median", 3, 0), ("median", 5, 0), ("median", 7, 0),
           ("bilateral", 5, 2), ("bilateral", 10, 2), ("bilateral", 20, 2), ("bilateral", 10, 4)]
COLORS = {"gaussian": "#2a6fbb", "median": "#d9822b", "bilateral": "#8e5bc4", "none": "#888888"}
MARKERS = {"gaussian": "o", "median": "s", "bilateral": "D", "none": "X"}
INK, MUTED = "#222222", "#888888"


def label(cfg) -> str:
    method, p1, p2 = cfg
    return {"none": "none", "gaussian": f"G s={p1:g}", "median": f"M k={p1:g}",
            "bilateral": f"B sc={p1:g},ss={p2:g}"}[method]


# ----------------------------------------------------------------------------- figures
def plot_tradeoff(summary: pd.DataFrame, chosen, path: Path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 6))
    for method in ["none", "gaussian", "median", "bilateral"]:
        part = summary[summary["method"] == method]
        ax.scatter(part["small_retention"], part["cnr"], s=90, color=COLORS[method], marker=MARKERS[method],
                   edgecolors="white", linewidths=1.5, zorder=3, label=method.capitalize())
        for _, r in part.iterrows():
            ax.annotate(r["label"], (r["small_retention"], r["cnr"]), xytext=(6, 4), textcoords="offset points",
                        fontsize=8, color=INK)
    c = summary[summary["label"] == label(chosen)].iloc[0]
    ax.scatter([c["small_retention"]], [c["cnr"]], s=320, facecolors="none", edgecolors=INK, lw=1.5, zorder=4)
    ax.axvline(MIN_SMALL_RETENTION, color=INK, lw=0.9, ls=":")
    ax.annotate(f"constraint: small-MA retention >= {MIN_SMALL_RETENTION}", (MIN_SMALL_RETENTION, ax.get_ylim()[0]),
                xytext=(5, 8), textcoords="offset points", fontsize=8.5, color=INK)
    ax.set_xlabel("Small-MA contrast retention (after / before)   ->  lesions preserved")
    ax.set_ylabel("Per-lesion CNR (median)   ->  lesions stand out of noise")
    ax.set_title("Noise reduction trade-off, training-set medians (circle = chosen)", fontsize=11.5, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def plot_by_size(lesions: pd.DataFrame, path: Path):
    import matplotlib.pyplot as plt
    families = ["gaussian", "median", "bilateral"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    x = np.arange(len(SIZE_LABELS))
    for ax, family in zip(axes, families):
        cfgs = [c for c in CONFIGS if c[0] == family]
        shades = np.linspace(0.35, 1.0, len(cfgs))
        for cfg, alpha in zip(cfgs, shades):
            part = lesions[lesions["label"] == label(cfg)]
            ratio = part.groupby("size_bin", observed=False).apply(
                lambda g: g["contrast_after"].median() / g["contrast_before"].median(), include_groups=False)
            ax.plot(x, ratio.values, color=COLORS[family], alpha=alpha, lw=2, marker=MARKERS[family], ms=6,
                    markeredgecolor="white")
            ax.annotate(label(cfg), (x[-1], ratio.values[-1]), xytext=(6, 0), textcoords="offset points",
                        fontsize=8, va="center", color=INK)
        ax.axhline(1.0, color=MUTED, lw=1, ls="--")
        ax.axhline(MIN_SMALL_RETENTION, color=INK, lw=0.8, ls=":")
        ax.set_xticks(x, SIZE_LABELS)
        ax.set_xlim(-0.3, len(x) + 0.4)
        ax.set_xlabel("GT MA area (px), training-set quartiles")
        ax.set_title(family.capitalize(), fontsize=11, loc="left")
        style_axes(ax)
    axes[0].set_ylabel("MA contrast retention (after / before)")
    fig.suptitle("Which lesions does each filter damage? (smaller MA = left)", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def plot_steps(base, fov, ma_mask, chosen, image_id, path):
    import matplotlib.pyplot as plt
    variants = [("none", 0, 0), chosen, ("gaussian", 2.0, 0), ("median", 7, 0)]
    colors = [MUTED, COLORS[chosen[0]], COLORS["gaussian"], COLORS["median"]]
    y0, x0 = densest_ma_window(ma_mask, CROP)
    crop = (slice(y0, y0 + CROP), slice(x0, x0 + CROP))
    contours, _ = cv2.findContours(ma_mask[crop].copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    _, _, stats, centroids = cv2.connectedComponentsWithStats(ma_mask[crop], connectivity=8)
    k = 1 + int(np.argmin(np.hypot(*(centroids[1:] - CROP / 2).T)))
    py, px = int(round(centroids[k][1])), int(round(centroids[k][0]))
    xr = np.arange(max(px - 40, 0), min(px + 41, CROP))
    patch_ref = base[crop]
    lo, hi = np.percentile(patch_ref, [0.5, 99.5])   # SAME display window for all panels

    fig, axes = plt.subplots(1, 5, figsize=(22, 4.9), gridspec_kw={"width_ratios": [1, 1, 1, 1, 1.25]})
    for ax, cfg, color in zip(axes[:4], variants, colors):
        img = denoise(base, *cfg)
        ax.imshow(img[crop], cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
        for c in contours:
            pts = c[:, 0, :]
            ax.plot(np.append(pts[:, 0], pts[0, 0]), np.append(pts[:, 1], pts[0, 1]), color="yellow", lw=0.7)
        title = label(cfg) + ("  (chosen)" if cfg == chosen else "")
        ax.set_title(title, fontsize=11)
        ax.axis("off")
        style = "--" if (cfg[0] == chosen[0] and cfg != chosen) else "-"   # same family as chosen -> dashed
        axes[4].plot(xr - px, img[crop][py, xr].astype(float), color=color, lw=1.7, ls=style, label=title)
    axes[4].axvspan(-stats[k, cv2.CC_STAT_WIDTH] / 2, stats[k, cv2.CC_STAT_WIDTH] / 2, color="#f2e600", alpha=0.25, lw=0)
    axes[4].set_title("Profile through the central MA (yellow = GT width)", fontsize=10.5)
    axes[4].set_xlabel("Offset from MA centre (px)")
    axes[4].set_ylabel("Grey level")
    axes[4].legend(frameon=False, fontsize=8.5)
    style_axes(axes[4])
    fig.suptitle(f"{image_id}: noise reduction on the Model-B image (same display window in all zooms)", fontsize=12.5)
    fig.tight_layout()
    fig.savefig(path, dpi=125)


# ----------------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(description="Experiment 05: noise reduction")
    parser.add_argument("--sample", nargs="+", default=["IDRiD_01"])
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    start = time.perf_counter()
    (OUTPUT_DIR / "denoised").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    image_rows, lesion_rows, samples = [], [], {}

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

        fov = extract_fov_mask(image_bgr)
        inside = fov > 0
        corrected, _ = illumination_correct(green_channel(image_bgr), fov)
        base = apply_clahe(corrected, fov)                       # Model-B image (uint8)

        labels, lesion, ring = lesion_labels_and_rings(ma_mask, fov)
        near_ma = cv2.dilate(ma_mask, disk(10)) > 0
        pool = np.flatnonzero((inside & ~near_ma).ravel())
        background_idx = rng.choice(pool, size=min(BACKGROUND_SAMPLES, pool.size), replace=False)
        lesion_idx = np.flatnonzero((lesion & inside).ravel())
        valid = cv2.erode(fov, disk(10)).astype(bool)

        before = per_lesion_stats(base, labels, lesion, ring)
        for cfg in CONFIGS:
            t0 = time.perf_counter()
            image = denoise(base, *cfg)
            elapsed = time.perf_counter() - t0
            after = per_lesion_stats(image, labels, lesion, ring)
            small = before["area"] <= SMALL_AREA
            image_rows.append({
                "image_id": image_id, "method": cfg[0], "p1": cfg[1], "p2": cfg[2], "label": label(cfg),
                "cnr": np.median(after["cnr"]),
                "cnr_small": np.median(after["cnr"][small]) if small.any() else np.nan,
                "contrast_retention": np.median(after["contrast"]) / np.median(before["contrast"]),
                "small_retention": (np.median(after["contrast"][small]) / np.median(before["contrast"][small])
                                    if small.any() else np.nan),
                **jpeg_blockiness(image, valid),
                "global_auc": global_auc(image, lesion_idx, background_idx),
                "time_s": elapsed,
            })
            lesion_rows.append(pd.DataFrame({"image_id": image_id, "label": label(cfg), "area": before["area"],
                                             "contrast_before": before["contrast"], "contrast_after": after["contrast"]}))
        if image_id in args.sample:
            samples[image_id] = (base, fov, ma_mask)
    print(f"  Done: {len(images)} training images." + " " * 20)

    table = pd.DataFrame(image_rows)
    table.round(5).to_csv(OUTPUT_DIR / "denoise_metrics_train.csv", index=False)
    lesions = pd.concat(lesion_rows, ignore_index=True)
    lesions["size_bin"] = pd.cut(lesions["area"], SIZE_BINS, labels=SIZE_LABELS)

    metric_cols = ["cnr", "cnr_small", "contrast_retention", "small_retention", "block_ratio", "global_auc", "time_s"]
    summary = table.groupby(["method", "p1", "p2", "label"], sort=False)[metric_cols].median().reset_index()
    summary.round(5).to_csv(OUTPUT_DIR / "denoise_summary_train.csv", index=False)

    candidates = summary[(summary["method"] != "none") & (summary["small_retention"] >= MIN_SMALL_RETENTION)]
    best = candidates.loc[candidates["cnr"].idxmax()]
    chosen = (best["method"], best["p1"], best["p2"])
    chosen = next(c for c in CONFIGS if label(c) == best["label"])

    # Paired check: in how many images does the chosen filter beat "none" on CNR?
    pivot = table.pivot_table(index="image_id", columns="label", values="cnr")
    wins = int((pivot[label(chosen)] > pivot["none"]).sum())

    plot_tradeoff(summary, chosen, OUTPUT_DIR / "fig_denoise_tradeoff.png")
    plot_by_size(lesions, OUTPUT_DIR / "fig_denoise_by_size.png")
    for image_id, (base, fov, ma_mask) in samples.items():
        plot_steps(base, fov, ma_mask, chosen, image_id, OUTPUT_DIR / f"fig_denoise_steps_{image_id}.png")
        out = denoise(base, *chosen)
        cv2.imencode(".png", out)[1].tofile(str(OUTPUT_DIR / "denoised" / f"{image_id}_{chosen[0]}.png"))

    print(f"\n=== Medians over {table['image_id'].nunique()} training images "
          f"({len(lesions) // len(CONFIGS)} GT MA, small = area <= {SMALL_AREA} px) ===")
    print(f"{'filter':<20}{'CNR':>7}{'CNR small':>11}{'retention':>11}{'small ret.':>12}{'block':>8}{'AUC':>8}{'time s':>8}")
    for _, r in summary.iterrows():
        mark = "  <== chosen" if r["label"] == label(chosen) else ""
        print(f"{r['label']:<20}{r['cnr']:>7.3f}{r['cnr_small']:>11.3f}{r['contrast_retention']:>11.3f}"
              f"{r['small_retention']:>12.3f}{r['block_ratio']:>8.3f}{r['global_auc']:>8.4f}{r['time_s']:>8.3f}{mark}")
    print(f"\nChosen: {label(chosen)}  | CNR higher than no filter in {wins} / {pivot.shape[0]} images")
    print(f"Outputs: {OUTPUT_DIR}\nElapsed: {time.perf_counter() - start:.1f} s")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
