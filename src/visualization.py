"""
Shared plotting helpers.
"""
import cv2
import numpy as np


def densest_ma_window(ma_mask: np.ndarray, size: int) -> tuple[int, int]:
    """
    Top-left corner of the size x size window containing the most MA *lesions*.
    Lesion centroids are counted (not pixels) so one large blob cannot dominate.
    """
    _, _, _, centroids = cv2.connectedComponentsWithStats(ma_mask, connectivity=8)
    centroid_map = np.zeros(ma_mask.shape, np.float32)
    cx_all, cy_all = np.round(centroids[1:]).astype(int).T
    centroid_map[cy_all, cx_all] = 1.0
    counts = cv2.boxFilter(centroid_map, -1, (size, size), normalize=False)
    cy, cx = np.unravel_index(np.argmax(counts), counts.shape)
    y0 = int(np.clip(cy - size // 2, 0, ma_mask.shape[0] - size))
    x0 = int(np.clip(cx - size // 2, 0, ma_mask.shape[1] - size))
    return y0, x0


def style_axes(ax):
    """Recessive grid and spines."""
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#e5e5e5", lw=0.6)
    ax.set_axisbelow(True)
