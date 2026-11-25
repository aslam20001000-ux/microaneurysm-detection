"""
Figures for the IEEE-style report, sized for one IEEE column (3.5 in wide), serif 8 pt.
Every graph is drawn from the CSV files written by the experiments (no typed numbers),
every image from the dataset / pipeline (training images only, except the annotated test example).

Run:  python docs/make_report_figures.py
Out:  docs/report_figures/*.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src.candidates import line_tophat, multiscale_blackhat, preprocess, robust_z
from src.dataset_loader import list_images, mask_path, read_image, read_mask
from src.illumination import illumination_correct
from src.visualization import densest_ma_window

OUT = config.PROJECT_ROOT / "docs" / "report_figures"
OUT.mkdir(parents=True, exist_ok=True)
EXP = config.OUTPUTS_DIR
COL = 3.5                      # IEEE column width (in)
ACCENT, GREY, DARK = "#c8502a", "#9aa5b1", "#2e3a4f"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"], "font.size": 8,
                     "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "legend.fontsize": 7, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.6, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})


def grid(ax):
    ax.grid(color="#e3e6ea", lw=0.5)
    ax.set_axisbelow(True)


def contour_on(ax, mask, color="#ffd400", lw=0.8):
    cs, _ = cv2.findContours(mask.astype(np.uint8).copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for c in cs:
        p = c[:, 0, :]
        ax.plot(np.append(p[:, 0], p[0, 0]), np.append(p[:, 1], p[0, 1]), color=color, lw=lw)


# ---------------------------------------------------------------- Fig. 1 pipeline (block diagram)
def fig_pipeline():
    steps = [("Green channel + FOV mask", "A"), ("Illumination correction", "A+"), ("CLAHE", "B"),
             ("Gaussian denoising (σ = 1.5)", "B+"), ("Black-hat, disc r = 16", "C"),
             ("Line top-hat, L = 41, 12 angles", "D0"), ("Adaptive threshold z > k + components", ""),
             ("20 features + random forest", "D")]
    fig, ax = plt.subplots(figsize=(COL, 2.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(steps) * 1.0 + 0.2)
    ax.axis("off")
    for i, (name, tag) in enumerate(steps):
        y = len(steps) - i - 0.4
        final = tag == "D"
        ax.add_patch(matplotlib.patches.FancyBboxPatch((0.4, y - 0.32), 6.8, 0.64, boxstyle="round,pad=0.02,rounding_size=0.12",
                                                       fc=ACCENT if final else "#eef1f5", ec=ACCENT if final else "#8a96a3", lw=0.6))
        ax.text(3.8, y, name, ha="center", va="center", color="white" if final else DARK, fontsize=7.5)
        if tag:
            ax.text(8.0, y, f"Model {tag}", ha="left", va="center", color=ACCENT if final else DARK, fontsize=7.5,
                    fontweight="bold" if final else "normal")
            ax.plot([7.2, 7.85], [y, y], color="#8a96a3", lw=0.5, ls=":")
        if i < len(steps) - 1:
            ax.annotate("", xy=(3.8, y - 0.36 - 0.28), xytext=(3.8, y - 0.36),
                        arrowprops=dict(arrowstyle="-|>", color="#8a96a3", lw=0.6, mutation_scale=6))
    fig.savefig(OUT / "fig01_pipeline.png")
    plt.close(fig)


# ---------------------------------------------------------------- Fig. 2 MA example (training image)
def fig_ma_example(image, ma):
    y0, x0 = densest_ma_window(ma, 300)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(1, 2, figsize=(COL, 1.55), gridspec_kw={"width_ratios": [1.5, 1]})
    ax[0].imshow(rgb)
    ax[0].add_patch(matplotlib.patches.Rectangle((x0, y0), 300, 300, fill=False, ec="#ffd400", lw=0.9))
    ax[0].set_title("(a) full image")
    ax[1].imshow(rgb[y0:y0 + 300, x0:x0 + 300], interpolation="nearest")
    contour_on(ax[1], ma[y0:y0 + 300, x0:x0 + 300])
    ax[1].set_title("(b) 300 × 300 px zoom")
    for a in ax:
        a.axis("off")
    fig.savefig(OUT / "fig02_ma_example.png")
    plt.close(fig)


# ---------------------------------------------------------------- Fig. 3 channel CNR
def fig_channels():
    t = pd.read_csv(EXP / "experiment_01" / "channel_metrics_train.csv")
    order = ["red", "green", "blue"]
    data = [t.loc[t.channel == c, "cnr"].to_numpy() for c in order]
    fig, ax = plt.subplots(figsize=(COL, 1.7))
    bp = ax.boxplot(data, widths=0.5, patch_artist=True, showfliers=False, medianprops=dict(color="black", lw=1))
    for patch, c in zip(bp["boxes"], order):
        patch.set_facecolor(ACCENT if c == "green" else "#dfe4ea")
        patch.set_edgecolor("#55606d")
        patch.set_linewidth(0.6)
    for i, v in enumerate(data, 1):
        ax.scatter(np.full(v.size, i) + np.random.default_rng(0).uniform(-0.12, 0.12, v.size), v, s=3, c="#333", alpha=0.45, lw=0)
        ax.text(i + 0.3, np.median(v), f"{np.median(v):.2f}", va="center", fontsize=7)
    ax.set_xticks([1, 2, 3], ["Red", "Green", "Blue"])
    ax.set_ylabel("MA contrast-to-noise ratio")
    grid(ax)
    fig.savefig(OUT / "fig03_channel_cnr.png")
    plt.close(fig)


# ---------------------------------------------------------------- Fig. 4 illumination (images + sweep)
def fig_illumination(image, fov):
    green = cv2.split(image)[1]
    corrected, _ = illumination_correct(green, fov)
    f = fov > 0
    fig, ax = plt.subplots(1, 2, figsize=(COL, 1.6))
    for a, img, t in [(ax[0], green.astype(float), "(a) green channel"), (ax[1], corrected, "(b) after correction")]:
        v = np.where(f, img, np.nan)
        lo, hi = np.nanpercentile(v, [1, 99])
        a.imshow(v, cmap="gray", vmin=lo, vmax=hi)
        a.set_title(t)
        a.axis("off")
    fig.savefig(OUT / "fig04a_illumination_images.png")
    plt.close(fig)

    s = pd.read_csv(EXP / "experiment_03" / "illumination_summary_train.csv")
    base = s[s.method == "none"].iloc[0]
    fig, ax = plt.subplots(1, 2, figsize=(COL, 1.6))
    for m, mk, col in [("median", "o", "#55606d"), ("gaussian", "s", ACCENT)]:
        p = s[s.method == m]
        ax[0].plot(p["size"], p["global_auc"], marker=mk, ms=3, lw=1, color=col, label=m.capitalize())
        ax[1].plot(p["size"], p["weber_retention"], marker=mk, ms=3, lw=1, color=col)
    ax[0].axhline(base.global_auc, color="#8a96a3", lw=0.7, ls="--")
    ax[0].text(55, base.global_auc + 0.006, "no correction", fontsize=6.5, color="#55606d")
    ax[1].axhline(0.95, color="black", lw=0.6, ls=":")
    ax[1].text(150, 0.935, "constraint 0.95", fontsize=6.5)
    ax[0].plot([151], [s[(s.method == "gaussian") & (s["size"] == 151)].global_auc], marker="o", ms=7, mfc="none", mec="black", mew=0.7)
    ax[0].set_ylabel("Global AUC")
    ax[1].set_ylabel("MA contrast retention")
    for a in ax:
        a.set_xlabel("Kernel size (px)")
        a.set_xticks([51, 151, 301])
        grid(a)
    ax[0].legend(frameon=False, loc="center right")
    fig.tight_layout(w_pad=1.0)
    fig.savefig(OUT / "fig04b_illumination_sweep.png")
    plt.close(fig)


# ---------------------------------------------------------------- Fig. 5 denoising trade-off
def fig_denoise():
    s = pd.read_csv(EXP / "experiment_05" / "denoise_summary_train.csv")
    style = {"none": ("X", "#55606d"), "gaussian": ("o", ACCENT), "median": ("s", "#55606d"), "bilateral": ("D", "#9aa5b1")}
    fig, ax = plt.subplots(figsize=(COL, 2.1))
    for m, (mk, col) in style.items():
        p = s[s.method == m]
        ax.scatter(p.small_retention, p.cnr, marker=mk, s=16, c=col, label=m.capitalize(), lw=0)
    for _, r in s.iterrows():
        lab = r["label"].replace("G s=", "σ=").replace("M k=", "k=").replace("B sc=", "bil ")
        off = {"bil 10,ss=2": (-34, -7), "σ=0.7": (4, -3), "bil 5,ss=2": (-44, -8), "k=3": (3, 3), "σ=1": (3, 4)}.get(lab, (3, 2))
        ax.annotate(lab, (r.small_retention, r.cnr), xytext=off, textcoords="offset points", fontsize=5.8)
    ch = s[s.label == "G s=1.5"].iloc[0]
    ax.scatter([ch.small_retention], [ch.cnr], s=70, facecolors="none", edgecolors="black", lw=0.7)
    ax.axvline(0.90, color="black", lw=0.6, ls=":")
    ax.set_xlabel("Small-MA contrast retention")
    ax.set_ylabel("Per-lesion CNR")
    ax.legend(frameon=False, loc="lower left", ncol=2)
    grid(ax)
    fig.savefig(OUT / "fig05_denoise_tradeoff.png")
    plt.close(fig)


# ---------------------------------------------------------------- Fig. 6 vessel suppression maps
def fig_vessels(pre, ma):
    ms = multiscale_blackhat(pre["smooth"], pre["fov"], config.BLACKHAT_RADII)
    ln = robust_z(line_tophat(pre["smooth"], config.LINE_LENGTH), pre["fov"])
    y0, x0 = densest_ma_window(ma, 400)
    c = (slice(y0, y0 + 400), slice(x0, x0 + 400))
    fig, ax = plt.subplots(1, 3, figsize=(COL, 1.35))
    for a, img, t, cm, lim in [(ax[0], pre["smooth"][c], "(a) B+ image", "gray", None),
                               (ax[1], ms[c], "(b) black-hat (C)", "magma", (0, 8)),
                               (ax[2], ln[c], "(c) line top-hat (D0)", "magma", (0, 8))]:
        a.imshow(img, cmap=cm, vmin=None if lim is None else lim[0], vmax=None if lim is None else lim[1], interpolation="nearest")
        contour_on(a, ma[c], "#39d353", 0.6)
        a.set_title(t, fontsize=7)
        a.axis("off")
    fig.savefig(OUT / "fig06_vessel_suppression.png")
    plt.close(fig)


# ---------------------------------------------------------------- Fig. 7 feature importance
def fig_features():
    imp = pd.read_csv(EXP / "experiment_08" / "feature_importances.csv", index_col=0).iloc[:, 0].sort_values().tail(10)
    fig, ax = plt.subplots(figsize=(COL, 2.0))
    ax.barh(imp.index.str.replace("_", " "), imp.values, color=[ACCENT if i >= len(imp) - 3 else "#b9c2cc" for i in range(len(imp))], height=0.65)
    for i, v in enumerate(imp.values):
        ax.text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=6.5)
    ax.set_xlabel("Impurity-based importance")
    ax.set_xlim(0, imp.max() * 1.18)
    grid(ax)
    ax.grid(axis="y", visible=False)
    fig.savefig(OUT / "fig07_feature_importance.png")
    plt.close(fig)


# ---------------------------------------------------------------- Fig. 8 FROC test + Fig. 9 FPI bars
def fig_results():
    froc = pd.read_csv(EXP / "experiment_09" / "froc_test.csv")
    summ = pd.read_csv(EXP / "experiment_09" / "ablation_test_summary.csv", index_col=0)
    models = ["A", "A+", "B", "B+", "C", "D0", "D"]
    greys = {"A": "#d4d9df", "A+": "#c3cad2", "B": "#b1b9c3", "B+": "#9aa5b1", "C": "#7d8996", "D0": "#2e3a4f"}
    fig, ax = plt.subplots(figsize=(COL, 2.4))
    ax.axvspan(1 / 8, 8, color="#f1f3f5", lw=0)
    for m in models:
        p = froc[(froc.model == m) & (froc.fp_per_image > 0)].sort_values("fp_per_image")
        ax.plot(p.fp_per_image, p.lesion_sensitivity, color=ACCENT if m == "D" else greys[m], lw=1.4 if m == "D" else 0.9,
                label=f"{m} (CPM {summ.loc[m, 'cpm']:.3f})")
    op = summ.loc["D"]
    ax.plot([op.fp_per_image], [op.lesion_sensitivity], "o", color=ACCENT, ms=4)
    ax.annotate(f"operating point\n{op.lesion_sensitivity:.3f} at {op.fp_per_image:.1f} FPI", (op.fp_per_image, op.lesion_sensitivity),
                xytext=(-82, -4), textcoords="offset points", fontsize=6.5,
                arrowprops=dict(arrowstyle="-", lw=0.5, color="#55606d"))
    ax.text(1, 1.06, "CPM range", ha="center", fontsize=6.5, color="#55606d")
    ax.set_xscale("log")
    ax.set_xlabel("False positives per image")
    ax.set_ylabel("Lesion-level sensitivity")
    ax.set_ylim(0, 1.12)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4, fontsize=6, columnspacing=1.0, handlelength=1.5)
    grid(ax)
    fig.savefig(OUT / "fig08_froc_test.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(COL, 1.9))
    vals = summ.loc[models, "fp_per_image"].to_numpy()
    lo_hi = np.array([[float(v) for v in x.strip("[]").split(",")] for x in summ.loc[models, "fp_per_image_ci"]])
    ax.bar(models, vals, color=[ACCENT if m == "D" else greys[m] for m in models], width=0.65,
           yerr=[np.clip(vals - lo_hi[:, 0], 0, None), np.clip(lo_hi[:, 1] - vals, 0, None)], error_kw=dict(lw=0.6, capsize=2))
    for i, v in enumerate(vals):
        ax.text(i, v * 1.35, f"{v:,.0f}" if v >= 100 else f"{v:.1f}", ha="center", fontsize=6.5)
    ax.set_yscale("log")
    ax.set_ylim(5, 20000)
    ax.set_ylabel("False positives per image")
    ax.set_xlabel("Ablation model")
    grid(ax)
    ax.grid(axis="x", visible=False)
    fig.savefig(OUT / "fig09_fpi_ablation.png")
    plt.close(fig)


# ---------------------------------------------------------------- Fig. 10 annotated test example
def fig_annotated():
    ann = cv2.imdecode(np.fromfile(str(EXP / "detections" / "IDRiD_55_model_D.jpg"), np.uint8), cv2.IMREAD_COLOR)
    crop = cv2.cvtColor(ann[150:1250, 450:1700], cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(COL, COL * 1100 / 1250))
    ax.imshow(crop)
    ax.axis("off")
    fig.savefig(OUT / "fig10_annotated_test.png")
    plt.close(fig)


def main():
    train01 = [p for p in list_images("train") if p.stem == "IDRiD_01"][0]
    image = read_image(train01)
    ma = read_mask(mask_path("train", "IDRiD_01", "MA"))
    pre = preprocess(image)
    fig_pipeline()
    fig_ma_example(image, ma)
    fig_channels()
    fig_illumination(image, pre["fov"])
    fig_denoise()
    fig_vessels(pre, ma)
    fig_features()
    fig_results()
    fig_annotated()
    print("\n".join(sorted(p.name for p in OUT.glob("*.png"))))


if __name__ == "__main__":
    main()
