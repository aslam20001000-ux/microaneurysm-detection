"""
Central configuration: all paths live here, nothing is hard-coded elsewhere.

Dataset location, first match wins:
  1. environment variable IDRID_ROOT
  2. first line of the file dataset_path.txt in the project root (not committed to git)
  3. <project>/dataset/IDRID
so the code works on another machine without editing source files.
"""
import os
from pathlib import Path

# Project root = folder that contains this "src" package
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _dataset_root() -> Path:
    if os.environ.get("IDRID_ROOT"):
        return Path(os.environ["IDRID_ROOT"])
    local_file = PROJECT_ROOT / "dataset_path.txt"
    if local_file.is_file():
        line = local_file.read_text(encoding="utf-8").strip().splitlines()
        if line:
            return Path(line[0].strip())
    return PROJECT_ROOT / "dataset" / "IDRID"


# Root of the extracted IDRiD dataset (read-only: we never modify it)
DATASET_ROOT = _dataset_root()

# Kaggle extraction kept the URL-encoded "%20" in the outer folder name,
# so the real folder is literally named "A.%20Segmentation".
SEGMENTATION_ROOT = DATASET_ROOT / "A.%20Segmentation" / "A. Segmentation"
IMAGES_DIR = SEGMENTATION_ROOT / "1. Original Images"
GROUNDTRUTH_DIR = SEGMENTATION_ROOT / "2. All Segmentation Groundtruths"

SPLITS = {
    "train": "a. Training Set",
    "test": "b. Testing Set",
}

# Lesion sub-folder name -> filename suffix used by IDRiD
LESIONS = {
    "MA": "1. Microaneurysms",
    "HE": "2. Haemorrhages",
    "EX": "3. Hard Exudates",
    "SE": "4. Soft Exudates",
    "OD": "5. Optic Disc",
}

# ---------------------------------------------------------------- FOV parameters
# Tuned on the TRAINING set in experiments/exp02_fov_mask.py (see its outputs).
FOV_THRESHOLD = 30   # red-channel level; background <= 9, area plateau 15-60; 30 rejects halo glow
FOV_OPEN_PX = 5      # opening radius: removes thin JPEG/halo specks attached to the rim
FOV_ERODE_PX = 4     # removes the 1-4 px partial-volume edge band; first GT MA is at 7 px

# ------------------------------------------------- Illumination correction
# Tuned on the TRAINING set in experiments/exp03_illumination.py
ILLUM_METHOD = "gaussian" # "median" or "gaussian"; chosen by the Exp03 selection rule
ILLUM_SIZE = 151          # kernel size in full-res px (sigma = size/4); median 101 is an equal runner-up
ILLUM_SCALE = 4           # background is estimated at 1/4 resolution (it is low-frequency);
                          # the image itself is NEVER resized

# ------------------------------------------------------------------ CLAHE
# Tuned on the TRAINING set in experiments/exp04_clahe.py
CLAHE_CLIP = 1.0          # clip limit; chosen by Exp04 rule on the illumination-corrected input
CLAHE_GRID = 8            # tiles across the width -> tileGridSize (8, 5), ~536 px tiles

# --------------------------------------------------------- Noise reduction
# Tuned on the TRAINING set in experiments/exp05_denoise.py
DENOISE_METHOD = "gaussian"
DENOISE_SIGMA = 1.5       # px; keeps 92 % of small-MA depth, removes the JPEG 8x8 block grid

# ------------------------------------------------ Candidate generation
BLACKHAT_RADII = (16,)           # Exp06: single r=16 beat every multi-scale combination (pixel AUPR)
LINE_LENGTH = 41                 # Exp07: best pixel AUPR; line top-hat alone = Model D0 response
MIN_CANDIDATE_AREA = 5           # px; components smaller than this are noise (smallest GT MA = 13 px)

# Output folders
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
INSPECTION_DIR = OUTPUTS_DIR / "inspection"
METRICS_DIR = OUTPUTS_DIR / "metrics"
