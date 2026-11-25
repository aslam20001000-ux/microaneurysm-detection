"""
Dataset Inspection (Pipeline steps 1-3).

Checks:
  * counts of images and MA masks per split
  * every image has exactly one MA mask, and no orphan masks exist
  * image and mask have identical height/width
  * mask values are truly binary
  * MA statistics per image: lesion pixels, number of lesions (connected components),
    smallest/largest lesion size -> tells us how tiny the targets are

Output: outputs/inspection/dataset_inspection.csv
Run:    python scripts/inspect_dataset.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # allow "import src"

import cv2
import numpy as np
import pandas as pd

from src import config
from src.dataset_loader import image_id_from_path, list_images, mask_path, read_image, read_mask


def inspect_split(split: str) -> list[dict]:
    images = list_images(split)
    ma_folder = config.GROUNDTRUTH_DIR / config.SPLITS[split] / config.LESIONS["MA"]
    mask_ids = {image_id_from_path(p) for p in ma_folder.glob("*.tif")}
    image_ids = {image_id_from_path(p) for p in images}

    print(f"\n=== {split.upper()} ===")
    print(f"Images: {len(images)} | MA masks: {len(mask_ids)}")
    orphan_masks = sorted(mask_ids - image_ids)
    if orphan_masks:
        print(f"  WARNING masks without image: {orphan_masks}")

    rows = []
    for index, image_path in enumerate(images, start=1):
        image_id = image_id_from_path(image_path)
        # Progress line (overwritten in place) so the user knows the script is working
        print(f"  Inspecting {index}/{len(images)}: {image_path.name}", end="\r", flush=True)
        row = {"split": split, "image_id": image_id, "image_file": image_path.name}
        try:
            image = read_image(image_path)
            row["height"], row["width"] = image.shape[:2]

            ma_path = mask_path(split, image_id, "MA")
            row["has_ma_mask"] = ma_path.is_file()
            if row["has_ma_mask"]:
                raw = read_image(ma_path, cv2.IMREAD_UNCHANGED)
                row["mask_dtype"] = str(raw.dtype)
                row["mask_channels"] = 1 if raw.ndim == 2 else raw.shape[2]
                # bincount is much faster than np.unique for uint8 (no sorting of 12M pixels)
                present_values = np.flatnonzero(np.bincount(raw.ravel(), minlength=256))
                row["mask_unique_values"] = str(present_values.tolist()[:6])

                mask = read_mask(ma_path)
                row["size_match"] = mask.shape[:2] == image.shape[:2]
                row["ma_pixels"] = int(mask.sum())
                row["ma_pixel_ratio_%"] = round(100 * mask.mean(), 5)

                # 8-connectivity: each connected blob = one annotated microaneurysm
                n_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
                areas = stats[1:, cv2.CC_STAT_AREA]  # skip background label 0
                row["ma_count"] = int(n_labels - 1)
                row["ma_area_min"] = int(areas.min()) if areas.size else 0
                row["ma_area_median"] = float(np.median(areas)) if areas.size else 0
                row["ma_area_max"] = int(areas.max()) if areas.size else 0
            row["status"] = "OK"
        except (FileNotFoundError, ValueError) as error:
            row["status"] = f"ERROR: {error}"
        rows.append(row)
    print(f"  Done: {len(rows)} images inspected." + " " * 20)
    return rows


def main() -> None:
    start_time = time.perf_counter()
    print(f"Dataset root: {config.DATASET_ROOT}")
    if not config.SEGMENTATION_ROOT.is_dir():
        sys.exit(f"Segmentation folder not found: {config.SEGMENTATION_ROOT}")

    rows = inspect_split("train") + inspect_split("test")
    table = pd.DataFrame(rows)

    config.INSPECTION_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = config.INSPECTION_DIR / "dataset_inspection.csv"
    table.to_csv(csv_path, index=False)

    print("\n=== SUMMARY ===")
    print(f"Read errors      : {(table['status'] != 'OK').sum()}")
    print(f"Missing MA masks : {(~table['has_ma_mask']).sum()}")
    print(f"Size mismatches  : {(~table['size_match'].fillna(False).astype(bool)).sum()}")
    print(f"Image sizes      : {table.groupby(['height', 'width']).size().to_dict()}")
    print(f"Mask dtypes      : {table['mask_dtype'].unique().tolist()}, "
          f"channels: {table['mask_channels'].unique().tolist()}, "
          f"values: {table['mask_unique_values'].unique().tolist()}")
    summary = table.groupby("split").agg(
        images=("image_id", "count"),
        total_MA=("ma_count", "sum"),
        MA_per_image_mean=("ma_count", "mean"),
        MA_per_image_min=("ma_count", "min"),
        MA_per_image_max=("ma_count", "max"),
        lesion_area_min=("ma_area_min", "min"),
        lesion_area_median=("ma_area_median", "median"),
        lesion_area_max=("ma_area_max", "max"),
        MA_pixel_ratio_mean_pct=("ma_pixel_ratio_%", "mean"),
    )
    print(summary.round(3).T.to_string())
    print(f"\nSaved: {csv_path}")
    print(f"Elapsed: {time.perf_counter() - start_time:.1f} s")


if __name__ == "__main__":
    main()
