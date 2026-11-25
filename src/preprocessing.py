"""
Preprocessing helpers (pipeline steps 2 and 5: acquisition + channel extraction).
"""
import cv2
import numpy as np


def split_rgb(image_bgr: np.ndarray) -> dict[str, np.ndarray]:
    """
    OpenCV loads images in BGR order. Return the channels with explicit names
    so the rest of the code never has to remember the index order.
    """
    blue, green, red = cv2.split(image_bgr)
    return {"red": red, "green": green, "blue": blue}


def green_channel(image_bgr: np.ndarray) -> np.ndarray:
    """Green channel (index 1 in BGR) as a 2-D uint8 array."""
    return image_bgr[:, :, 1].copy()


def rough_fov_mask(image_bgr: np.ndarray, threshold: int = 20) -> np.ndarray:
    """
    TEMPORARY field-of-view estimate: pixels whose red value exceeds a small
    threshold (the black border around the retina is ~0 in every channel).
    A proper FOV extraction is pipeline step 4 (Experiment 02).
    """
    return (image_bgr[:, :, 2] > threshold).astype(np.uint8)


def lesion_background_ring(lesion_mask: np.ndarray, inner_px: int = 3, outer_px: int = 10) -> np.ndarray:
    """
    Local background around lesions: a ring between `inner_px` and `outer_px`
    pixels away from every lesion boundary.
      * inner gap skips the blurred lesion edge
      * outer limit keeps the background *local* (illumination varies across the retina)
    Lesion pixels themselves are always excluded.
    """
    def disk(radius: int) -> np.ndarray:
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))

    outer = cv2.dilate(lesion_mask, disk(outer_px))
    inner = cv2.dilate(lesion_mask, disk(inner_px))
    return ((outer > 0) & (inner == 0)).astype(np.uint8)
