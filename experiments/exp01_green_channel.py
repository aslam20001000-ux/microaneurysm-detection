"""
Experiment 01 - Image Acquisition & Green Channel Extraction.

Question: which RGB channel shows microaneurysms (MA) with the best contrast?
We answer it with measurements on the TRAINING set only (test set untouched).

For every training image and every channel we measure, using the GT MA mask:
  mu_L  = mean intensity of MA pixels
  mu_B  = mean intensity of the local background ring (3-10 px around each MA)
  sd_B  = standard deviation of that ring
  Contrast        C   = mu_B - mu_L              (MA are darker -> positive)
  Weber contrast  Cw  = (mu_B - mu_L) / mu_B
  Contrast-to-Noise CNR = (mu_B - mu_L) / sd_B   (how far MA stand out of background variation)
We also measure, inside a rough FOV, the fraction of saturated (255) and
near-black (<=10) pixels: clipped pixels carry no lesion information.

Outputs (outputs/experiment_01/):
  channel_metrics_train.csv      per image x channel
  channel_summary_train.csv      median / IQR per channel
  green/<id>_green.png           lossless green channel of the sample image(s)
  fig_channels_<id>.png          full view + zoomed MA region per channel, GT contours
  fig_cnr_boxplot.png            CNR distribution per channel over 54 training images

Run:
  python experiments/exp01_green_channel.py                 # save figures only
  python experiments/exp01_green_channel.py --show          # also open figure windows
  python experiments/exp01_green_channel.py --sample IDRiD_05 IDRiD_20
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
from src.preprocessing import green_channel, lesion_background_ring, rough_fov_mask, split_rgb
from src.visualization import densest_ma_window

OUTPUT_DIR = config.OUTPUTS_DIR / "experiment_01"
CHANNEL_COLORS = {"red": "#c0392b", "green": "#1e8449", "blue": "#2e86c1"}  # identity = the channel itself
CROP_SIZE = 256  # px; MA median diameter is ~18 px, so a 256 px window shows several in context


# ----------------------------------------------------------------------------- metrics
def channel_metrics(image_bgr: np.ndarray, ma_mask: np.ndarray) -> list[dict]:
    ring = lesion_background_ring(ma_mask)
    fov = rough_fov_mask(image_bgr).astype(bool)
    lesion = ma_mask.astype(bool)
    ring = ring.astype(bool)

    rows = []
    for name, channel in split_rgb(image_bgr).items():
        lesion_values = channel[lesion].astype(np.float64)
        ring_values = channel[ring].astype(np.float64)
        fov_values = channel[fov]
        mu_l, mu_b, sd_b = lesion_values.mean(), ring_values.mean(), ring_values.std()
        rows.append({
            "channel": name,
            "mean_lesion": mu_l,
            "mean_background": mu_b,
            "std_background": sd_b,
            "contrast": mu_b - mu_l,
            "weber_contrast": (mu_b - mu_l) / mu_b if mu_b > 0 else np.nan,
            "cnr": (mu_b - mu_l) / sd_b if sd_b > 0 else np.nan,
            "fov_saturated_pct": 100 * np.mean(fov_values >= 255),
            "fov_near_black_pct": 100 * np.mean(fov_values <= 10),
        })
    return rows


# ----------------------------------------------------------------------------- figures
def plot_channels(image_bgr: np.ndarray, ma_mask: np.ndarray, image_id: str, metrics: list[dict], path: Path):
    import matplotlib.pyplot as plt

    channels = split_rgb(image_bgr)
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    y0, x0 = densest_ma_window(ma_mask, CROP_SIZE)
    crop = (slice(y0, y0 + CROP_SIZE), slice(x0, x0 + CROP_SIZE))
    contours, _ = cv2.findContours(ma_mask[crop].copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnr = {row["channel"]: row["cnr"] for row in metrics}

    fig, axes = plt.subplots(2, 4, figsize=(16, 8.4))
    panels = [("Original (RGB)", rgb, None)] + [(f"{n.capitalize()} channel", c, "gray") for n, c in channels.items()]
    for col, (title, img, cmap) in enumerate(panels):
        # Row 1: full image (display only is downsampled by matplotlib; data is untouched)
        ax = axes[0, col]
        ax.imshow(img, cmap=cmap, vmin=0, vmax=255)
        ax.add_patch(matplotlib.patches.Rectangle((x0, y0), CROP_SIZE, CROP_SIZE, fill=False, ec="yellow", lw=1.5))
        name = title.split()[0].lower()
        ax.set_title(title + (f"\nCNR = {cnr[name]:.2f}" if name in cnr else "\n"), fontsize=11)
        ax.axis("off")
        # Row 2: zoom at native resolution with GT MA contours
        ax = axes[1, col]
        ax.imshow(img[crop], cmap=cmap, vmin=0, vmax=255, interpolation="nearest")
        for contour in contours:
            pts = contour[:, 0, :]
            ax.plot(np.append(pts[:, 0], pts[0, 0]), np.append(pts[:, 1], pts[0, 1]), color="yellow", lw=0.8)
        ax.set_title(f"Zoom {CROP_SIZE}x{CROP_SIZE} px, GT MA outlined", fontsize=9, color="#555")
        ax.axis("off")
    fig.suptitle(f"{image_id}: RGB channel comparison for microaneurysm visibility", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    return fig


def plot_cnr_boxplot(table: pd.DataFrame, path: Path):
    import matplotlib.pyplot as plt

    order = ["red", "green", "blue"]
    data = [table.loc[table["channel"] == c, "cnr"].dropna().to_numpy() for c in order]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    box = ax.boxplot(data, patch_artist=True, widths=0.5, showfliers=False, medianprops={"color": "black", "lw": 2})
    for patch, c in zip(box["boxes"], order):
        patch.set_facecolor(CHANNEL_COLORS[c])
        patch.set_alpha(0.55)
        patch.set_edgecolor(CHANNEL_COLORS[c])
    for i, values in enumerate(data, start=1):
        jitter = np.random.default_rng(0).uniform(-0.12, 0.12, values.size)
        ax.scatter(np.full(values.size, i) + jitter, values, s=10, color="#333", alpha=0.5, zorder=3)
        ax.annotate(f"median {np.median(values):.2f}", (i + 0.28, np.median(values)), fontsize=9, va="center", color="#333")
    ax.set_xticks([1, 2, 3], [c.capitalize() for c in order])
    ax.axhline(0, color="#999", lw=0.8)
    ax.set_ylabel("CNR  (mu_background - mu_MA) / sd_background")
    ax.set_title(f"MA contrast-to-noise ratio per channel (training set, n = {data[0].size} images)", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e5e5e5", lw=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    return fig


# ----------------------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 01: green channel extraction")
    parser.add_argument("--sample", nargs="+", default=["IDRiD_01"], help="training image id(s) for figures")
    parser.add_argument("--show", action="store_true", help="open matplotlib windows")
    args = parser.parse_args()
    if not args.show:
        matplotlib.use("Agg")  # file output only, no GUI windows
    import matplotlib.pyplot as plt

    start = time.perf_counter()
    (OUTPUT_DIR / "green").mkdir(parents=True, exist_ok=True)

    rows = []
    images = list_images("train")  # tuning/analysis on TRAIN only
    for index, image_path in enumerate(images, start=1):
        image_id = image_id_from_path(image_path)
        print(f"  Processing {index}/{len(images)}: {image_id}", end="\r", flush=True)
        try:
            image_bgr = read_image(image_path)
            ma_mask = read_mask(mask_path("train", image_id, "MA"))
        except (FileNotFoundError, ValueError) as error:
            print(f"\n  SKIPPED {image_id}: {error}")
            continue

        metrics = channel_metrics(image_bgr, ma_mask)
        rows += [{"image_id": image_id, **m} for m in metrics]

        if image_id in args.sample:
            # Lossless PNG so later experiments can reuse it without JPEG artefacts
            out_png = OUTPUT_DIR / "green" / f"{image_id}_green.png"
            cv2.imencode(".png", green_channel(image_bgr))[1].tofile(str(out_png))
            plot_channels(image_bgr, ma_mask, image_id, metrics, OUTPUT_DIR / f"fig_channels_{image_id}.png")
    print(f"  Done: {len(images)} training images." + " " * 20)

    table = pd.DataFrame(rows)
    table.round(4).to_csv(OUTPUT_DIR / "channel_metrics_train.csv", index=False)

    metric_cols = ["contrast", "weber_contrast", "cnr", "fov_saturated_pct", "fov_near_black_pct"]
    summary = table.groupby("channel")[metric_cols].describe(percentiles=[0.25, 0.5, 0.75])
    summary = summary.loc[:, (slice(None), ["25%", "50%", "75%"])].reindex(["red", "green", "blue"])
    summary.round(4).to_csv(OUTPUT_DIR / "channel_summary_train.csv")

    # How often is green the best channel per image?
    best = table.loc[table.groupby("image_id")["cnr"].idxmax(), "channel"].value_counts()

    plot_cnr_boxplot(table, OUTPUT_DIR / "fig_cnr_boxplot.png")

    pd.set_option("display.width", 140)
    print("\n=== Median [IQR] over training images ===")
    for col in metric_cols:
        cells = [f"{summary.loc[c, (col, '50%')]:8.3f} [{summary.loc[c, (col, '25%')]:.3f}-{summary.loc[c, (col, '75%')]:.3f}]"
                 for c in summary.index]
        print(f"{col:<20}" + "  ".join(f"{c}: {v}" for c, v in zip(summary.index, cells)))
    print(f"\nChannel with highest CNR per image: {best.to_dict()}")
    print(f"Outputs saved in: {OUTPUT_DIR}")
    print(f"Elapsed: {time.perf_counter() - start:.1f} s")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
