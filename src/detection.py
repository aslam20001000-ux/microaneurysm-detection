"""
Final microaneurysm detector (pipeline steps 15-16), used by the test evaluation and the GUI.

All parameters are FROZEN: they come from src/config.py and from the files written by
experiments/exp08_detection_train.py (outputs/models/operating_points.json and
outputs/models/fp_classifier.joblib). Nothing here looks at test data.
"""
import json
import time

import joblib
import numpy as np

from src import config
from src.candidates import extract_candidates, line_tophat, multiscale_blackhat, preprocess, response, robust_z
from src.features import FEATURE_NAMES, candidate_features

MODEL_DIR = config.OUTPUTS_DIR / "models"


class MicroaneurysmDetector:
    def __init__(self):
        points_file = MODEL_DIR / "operating_points.json"
        classifier_file = MODEL_DIR / "fp_classifier.joblib"
        if not points_file.is_file() or not classifier_file.is_file():
            raise FileNotFoundError("Run experiments/exp08_detection_train.py first (trained parameters missing).")
        with open(points_file) as f:
            self.points = json.load(f)["models"]
        bundle = joblib.load(classifier_file)
        self.classifier, self.k0 = bundle["model"], bundle["k0"]

    def run(self, image_bgr: np.ndarray, model: str = "D") -> dict:
        """
        Detect MA candidates with one ablation model.
        Returns: mask (uint8 0/1), candidates (DataFrame with centroid, area, score),
                 response z map, preprocessing images, processing time (s).
        """
        start = time.perf_counter()
        pre = preprocess(image_bgr)
        fov = pre["fov"]
        if model in ("A", "A+", "B", "B+", "C"):
            z = response(model, pre)
            labels, stats, centroids = extract_candidates(z, fov, self.points[model]["k"])
            scores = np.ones(len(stats))
            keep = np.ones(len(stats), bool)
        elif model in ("D0", "D"):
            ms = multiscale_blackhat(pre["smooth"], fov, config.BLACKHAT_RADII)
            line = robust_z(line_tophat(pre["smooth"], config.LINE_LENGTH), fov)
            z = line                        # D0 response = line top-hat (Exp07); black-hat only as feature
            if model == "D0":
                labels, stats, centroids = extract_candidates(z, fov, self.points["D0"]["k"])
                scores, keep = np.ones(len(stats)), np.ones(len(stats), bool)
            else:
                labels, stats, centroids = extract_candidates(z, fov, self.k0)
                feats = candidate_features(labels, stats, centroids, pre, image_bgr, z, ms, line)
                scores = (self.classifier.predict_proba(feats[FEATURE_NAMES].to_numpy())[:, 1]
                          if len(feats) else np.zeros(0))
                keep = scores >= self.points["D"]["threshold"]
        else:
            raise ValueError(f"Unknown model {model}")

        lut = np.zeros(len(stats) + 1, np.uint8)
        lut[1:][keep] = 1
        mask = lut[labels]
        import pandas as pd
        candidates = pd.DataFrame({"x": centroids[keep, 0], "y": centroids[keep, 1],
                                   "area": stats[keep, 4], "score": scores[keep]})
        return {"mask": mask, "candidates": candidates, "z": z, "pre": pre,
                "all_scores": scores, "labels": labels, "centroids": centroids,
                "time_s": time.perf_counter() - start}
