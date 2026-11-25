"""
Experiment 08 - Thresholding, Connected Components, Features and False-Positive
Removal; tuning of every ablation model on the TRAINING set.

Models (each adds ONE technique):
  A    green darkness                  -> z > k -> components
  A+   + illumination correction
  B    + CLAHE
  B+   + Gaussian denoising
  C    + black-hat (morphology)
  D0   vessel-suppressing morphology: line top-hat replaces the disc black-hat
       (Exp07: alone it beat min(black-hat, line top-hat))   [intermediate, reported]
  D    + feature-based false-positive removal (classifier on candidate features)

Operating point of A ... D0: threshold k on the z response, chosen to maximise the
pooled lesion-level F1 on the training set. FROC = sweep of k.

Model D: candidates = D0 components at k0 = K0_D (high-sensitivity pool); a classifier
trained on their features scores every candidate. To avoid evaluating on the images the
classifier was trained on, scores come from 5-fold GROUPED cross-validation (all
candidates of one image are in the same fold). FROC = sweep of the probability
threshold; operating point = max lesion F1 (out-of-fold).
Two classifiers are compared (fixed before running): a depth-4 decision tree
(fully interpretable rules) and a random forest; the higher out-of-fold CPM wins.

Outputs: outputs/experiment_08/, outputs/models/ (classifier + operating points)
Run:     python experiments/exp08_detection_train.py
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.tree import DecisionTreeClassifier, export_text

from src import config
from src.candidates import (extract_candidates, line_tophat, multiscale_blackhat, preprocess, response,
                            robust_z)
from src.dataset_loader import image_id_from_path, list_images, mask_path, read_image, read_mask
from src.evaluation import froc_cpm, gt_match_zones, lesion_metrics, match_candidates
from src.features import FEATURE_NAMES, candidate_features
from src.visualization import style_axes

OUTPUT_DIR = config.OUTPUTS_DIR / "experiment_08"
MODEL_DIR = config.OUTPUTS_DIR / "models"
K_GRID = [1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 8, 10, 12, 15, 20]
K0_D = 3.0                 # candidate pool: ~95 % of MA kept, ~10k candidates / image
NEG_FRACTION = 0.25        # training only: keep all positives + 25 % of negatives (class imbalance, speed)
SWEEP_MODELS = ["A", "A+", "B", "B+", "C", "D0"]
RAMP = {"A": "#c6dbef", "A+": "#9ecae1", "B": "#6baed6", "B+": "#4292c6", "C": "#2171b5", "D0": "#084594"}
FINAL_COLOR, INK = "#d9822b", "#222222"


def classifiers():
    return {
        "decision_tree_d4": DecisionTreeClassifier(max_depth=4, min_samples_leaf=50, class_weight="balanced",
                                                   random_state=0),
        "random_forest": RandomForestClassifier(n_estimators=200, min_samples_leaf=5, max_features="sqrt",
                                                class_weight="balanced_subsample", n_jobs=-1, random_state=0),
    }


def froc_from_scores(pool: pd.DataFrame, scores: np.ndarray, total_gt: int, n_images: int) -> pd.DataFrame:
    """Lesion-level FROC when keeping candidates with score >= t, for many t."""
    # Quantiles alone cluster near 0 (most candidates have tiny probabilities) and leave the
    # high-score / low-FPI range empty -> add a uniform grid with step 0.001.
    thresholds = np.unique(np.concatenate([np.quantile(scores, np.linspace(0, 1, 400)), np.linspace(0, 1, 1001)]))
    is_tp = pool["gt_id"].to_numpy() > 0
    # one integer key per (image, GT lesion): counts each GT lesion once however many candidates hit it
    keys = pd.factorize(pool["image_id"])[0].astype(np.int64) * 100_000 + pool["gt_id"].to_numpy()
    rows = []
    for t in thresholds:
        kept = scores >= t
        detected = np.unique(keys[kept & is_tp]).size
        counts = {"n_gt": total_gt, "gt_detected": detected, "n_candidates": int(kept.sum()),
                  "tp_candidates": int((kept & is_tp).sum()), "fp_candidates": int((kept & ~is_tp).sum())}
        rows.append({"threshold": t, **lesion_metrics(counts, n_images)})
    return pd.DataFrame(rows)


def main():
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    start = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    keys = ["n_gt", "gt_detected", "n_candidates", "tp_candidates", "fp_candidates"]
    counts = {m: {k: dict.fromkeys(keys, 0) for k in K_GRID} for m in SWEEP_MODELS}
    pool_rows, timing = [], {m: [] for m in SWEEP_MODELS}
    total_gt = 0

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
        zones, n_gt = gt_match_zones(ma_mask)
        total_gt += n_gt

        maps = {}
        for m in ["A", "A+", "B", "B+"]:
            t0 = time.perf_counter()
            maps[m] = response(m, pre)
            timing[m].append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        ms = multiscale_blackhat(pre["smooth"], fov, config.BLACKHAT_RADII)
        timing["C"].append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        line = robust_z(line_tophat(pre["smooth"], config.LINE_LENGTH), fov)
        timing["D0"].append(time.perf_counter() - t0)
        maps["C"], maps["D0"] = ms, line      # D0 = line top-hat alone (Exp07)

        for m in SWEEP_MODELS:
            for k in K_GRID:
                _, _, centroids = extract_candidates(maps[m], fov, k)
                c = match_candidates(centroids, zones, n_gt)
                for key in keys:
                    counts[m][k][key] += c[key]

        # Candidate pool for the false-positive classifier (Model D)
        labels, stats, centroids = extract_candidates(maps["D0"], fov, K0_D)
        feats = candidate_features(labels, stats, centroids, pre, image_bgr, maps["D0"], ms, line)
        xs = np.clip(np.round(centroids[:, 0]).astype(int), 0, fov.shape[1] - 1)
        ys = np.clip(np.round(centroids[:, 1]).astype(int), 0, fov.shape[0] - 1)
        feats.insert(0, "gt_id", zones[ys, xs])
        feats.insert(0, "image_id", image_id)
        pool_rows.append(feats)
    print(f"  Done: {len(images)} training images." + " " * 20)
    n_images = len(images)

    # ------------------------------------------------------------------ A ... D0: k sweep
    froc_rows, operating = [], {}
    for m in SWEEP_MODELS:
        for k in K_GRID:
            froc_rows.append({"model": m, "k": k, **lesion_metrics(counts[m][k], n_images)})
    froc = pd.DataFrame(froc_rows)
    froc.round(5).to_csv(OUTPUT_DIR / "froc_threshold_sweep_train.csv", index=False)
    summary_rows = []
    for m in SWEEP_MODELS:
        part = froc[froc["model"] == m]
        best = part.loc[part["lesion_f1"].idxmax()]
        cpm, _ = froc_cpm(part["fp_per_image"].to_numpy(), part["lesion_sensitivity"].to_numpy())
        operating[m] = {"k": float(best["k"])}
        summary_rows.append({"model": m, "cpm": cpm, "k": best["k"], **best[["lesion_sensitivity", "lesion_precision",
                                                                             "lesion_f1", "fp_per_image"]].to_dict(),
                             "response_time_s": float(np.median(timing[m]))})

    # ------------------------------------------------------------------ D: classifier
    pool = pd.concat(pool_rows, ignore_index=True)
    joblib.dump((pool, total_gt, n_images, froc, counts), OUTPUT_DIR / "candidate_pool_train.joblib")  # for re-analysis
    X, y, groups = pool[FEATURE_NAMES].to_numpy(), (pool["gt_id"] > 0).to_numpy(), pool["image_id"].to_numpy()
    sampler = np.random.default_rng(0)

    def subsample(idx):
        """All positives + NEG_FRACTION of negatives (training folds only)."""
        neg = idx[~y[idx]]
        return np.concatenate([idx[y[idx]], sampler.choice(neg, int(len(neg) * NEG_FRACTION), replace=False)])
    pool_ceiling = pool.loc[y, ["image_id", "gt_id"]].drop_duplicates().shape[0] / total_gt
    cv_results, oof_scores = {}, {}
    for name, clf in classifiers().items():
        oof = np.zeros(len(pool))
        for train_idx, test_idx in GroupKFold(n_splits=5).split(X, y, groups):
            fit_idx = subsample(train_idx)
            clf.fit(X[fit_idx], y[fit_idx])
            oof[test_idx] = clf.predict_proba(X[test_idx])[:, 1]
        curve = froc_from_scores(pool, oof, total_gt, n_images)
        cpm, at_points = froc_cpm(curve["fp_per_image"].to_numpy(), curve["lesion_sensitivity"].to_numpy())
        cv_results[name] = (cpm, at_points, curve)
        oof_scores[name] = oof
        curve.round(5).to_csv(OUTPUT_DIR / f"froc_D_{name}_oof_train.csv", index=False)
    chosen_clf = max(cv_results, key=lambda n: cv_results[n][0])
    cpm_d, at_points_d, curve_d = cv_results[chosen_clf]
    best_d = curve_d.loc[curve_d["lesion_f1"].idxmax()]
    operating["D"] = {"k0": K0_D, "classifier": chosen_clf, "threshold": float(best_d["threshold"])}
    summary_rows.append({"model": "D", "cpm": cpm_d, "k": K0_D, **best_d[["lesion_sensitivity", "lesion_precision",
                                                                         "lesion_f1", "fp_per_image"]].to_dict(),
                         "response_time_s": float(np.median(timing["D0"]))})

    fit_idx = subsample(np.arange(len(y)))
    final_clf = classifiers()[chosen_clf].fit(X[fit_idx], y[fit_idx])
    joblib.dump({"model": final_clf, "features": FEATURE_NAMES, "k0": K0_D}, MODEL_DIR / "fp_classifier.joblib")
    importances = pd.Series(final_clf.feature_importances_, index=FEATURE_NAMES).sort_values()
    importances.round(5).to_csv(OUTPUT_DIR / "feature_importances.csv")
    if chosen_clf.startswith("decision_tree"):
        (OUTPUT_DIR / "decision_tree_rules.txt").write_text(export_text(final_clf, feature_names=FEATURE_NAMES))
    with open(MODEL_DIR / "operating_points.json", "w") as f:
        json.dump({"models": operating, "blackhat_radii": list(config.BLACKHAT_RADII),
                   "line_length": config.LINE_LENGTH, "min_candidate_area": config.MIN_CANDIDATE_AREA}, f, indent=2)

    summary = pd.DataFrame(summary_rows).set_index("model")
    summary.round(5).to_csv(OUTPUT_DIR / "ablation_train_summary.csv")

    # ------------------------------------------------------------------ figures
    fig, ax = plt.subplots(figsize=(9.5, 6))
    for m in SWEEP_MODELS:
        part = froc[froc["model"] == m].sort_values("fp_per_image")
        ax.plot(part["fp_per_image"], part["lesion_sensitivity"], color=RAMP[m], lw=2, marker="o", ms=4)
        last = part.iloc[-1]
        ax.annotate(m, (last["fp_per_image"], last["lesion_sensitivity"]), xytext=(5, 0), textcoords="offset points",
                    fontsize=9, color=INK, va="center")
    curve = curve_d[curve_d["fp_per_image"] > 0].sort_values("fp_per_image")   # log axis: drop FPI = 0
    ax.plot(curve["fp_per_image"], curve["lesion_sensitivity"], color=FINAL_COLOR, lw=2.5)
    ax.annotate(f"D ({chosen_clf}, out-of-fold)", (curve["fp_per_image"].iloc[len(curve) // 3],
                                                    curve["lesion_sensitivity"].iloc[len(curve) // 3]),
                xytext=(8, -14), textcoords="offset points", fontsize=9, color=INK)
    for x in [1 / 8, 1, 8]:
        ax.axvline(x, color="#bbbbbb", lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel("False positives per image (log scale)")
    ax.set_ylabel("Lesion-level sensitivity")
    ax.set_title("FROC on the training set (dotted: FPI = 1/8, 1, 8)", fontsize=11, loc="left")
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig_froc_train.png", dpi=150)

    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.barh(importances.index, importances.values, color="#2a6fbb", height=0.6)
    ax.set_xlabel("Impurity-based importance")
    ax.set_title(f"Feature importance ({chosen_clf}, trained on all training candidates)", fontsize=10.5, loc="left")
    style_axes(ax)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig_feature_importance.png", dpi=150)

    # ------------------------------------------------------------------ report
    print(f"\n=== Training set: {total_gt} GT MA in {n_images} images ===")
    print(f"{'model':<6}{'CPM':>7}{'k':>6}{'sens':>8}{'prec':>8}{'F1':>8}{'FPI':>10}{'resp. s':>9}")
    for m, r in summary.iterrows():
        print(f"{m:<6}{r['cpm']:>7.3f}{r['k']:>6.1f}{r['lesion_sensitivity']:>8.3f}{r['lesion_precision']:>8.3f}"
              f"{r['lesion_f1']:>8.3f}{r['fp_per_image']:>10.1f}{r['response_time_s']:>9.2f}")
    print(f"\nCandidate pool (D0, k0={K0_D}): {len(pool)} candidates, {int(y.sum())} TP; "
          f"pool sensitivity ceiling = {pool_ceiling:.3f}")
    for name, (cpm, at_points, _) in cv_results.items():
        print(f"  {name:<18} out-of-fold CPM {cpm:.3f}  sens@FPI[1/8..8] {np.round(at_points, 3).tolist()}")
    print(f"Chosen classifier: {chosen_clf}, threshold {operating['D']['threshold']:.4f}")
    print(f"Top features: {importances.sort_values(ascending=False).head(6).round(3).to_dict()}")
    print(f"Outputs: {OUTPUT_DIR}\nElapsed: {time.perf_counter() - start:.1f} s")


if __name__ == "__main__":
    main()
