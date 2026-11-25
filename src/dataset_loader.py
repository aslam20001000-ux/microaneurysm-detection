"""
Dataset access helpers: locate image/mask pairs and read them safely.
"""
from pathlib import Path

import cv2
import numpy as np

from src import config


def image_id_from_path(path: Path) -> str:
    """'IDRiD_01.jpg' -> 'IDRiD_01', 'IDRiD_01_MA.tif' -> 'IDRiD_01'."""
    parts = path.stem.split("_")
    return "_".join(parts[:2])


def list_images(split: str) -> list[Path]:
    """Return sorted list of original fundus images for 'train' or 'test'."""
    folder = config.IMAGES_DIR / config.SPLITS[split]
    if not folder.is_dir():
        raise FileNotFoundError(f"Image folder not found: {folder}")
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif"})


def mask_path(split: str, image_id: str, lesion: str = "MA") -> Path:
    """Expected ground-truth mask path (may not exist for some lesions)."""
    folder = config.GROUNDTRUTH_DIR / config.SPLITS[split] / config.LESIONS[lesion]
    return folder / f"{image_id}_{lesion}.tif"


def read_image(path: Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    """
    Read an image robustly on Windows.
    np.fromfile + cv2.imdecode works with any Unicode path, unlike cv2.imread.
    Raises an error instead of silently returning None.
    """
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    buffer = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(buffer, flags)
    if image is None:
        raise ValueError(f"OpenCV could not decode: {path}")
    return image


def read_mask(path: Path) -> np.ndarray:
    """
    Read a ground-truth mask and return it as binary uint8 (0 or 1).
    IDRiD masks may be stored as RGB/indexed TIFF, so any non-zero channel = lesion.
    """
    raw = read_image(path, cv2.IMREAD_UNCHANGED)
    if raw.ndim == 3:
        raw = raw.max(axis=2)
    return (raw > 0).astype(np.uint8)
