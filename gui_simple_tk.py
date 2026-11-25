"""
Simple desktop GUI (Tkinter) for the final Model D detector.

Run:  python gui.py
Requires the trained files from experiments/exp08_detection_train.py.

Shows: original image, enhanced image (Model B+ preprocessing), detection mask,
and the original image with suspected microaneurysms circled; reports the number of
suspected MA and the processing time; exports an annotated PNG and a CSV of candidates.

This is a research prototype: it reports SUSPECTED lesions only and gives NO diagnosis.
"""
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from src.dataset_loader import read_image
from src.detection import MicroaneurysmDetector

PANEL_SIZE = (430, 300)
DISCLAIMER = ("Research prototype - highlights SUSPECTED microaneurysms only. "
              "Not a medical device; not a diagnosis. Results must be reviewed by an ophthalmologist.")


def to_photo(image: np.ndarray) -> ImageTk.PhotoImage:
    """numpy (gray or RGB) -> Tk image, downscaled for display only."""
    pil = Image.fromarray(image)
    pil.thumbnail(PANEL_SIZE, Image.LANCZOS)
    return ImageTk.PhotoImage(pil)


def annotate(image_bgr: np.ndarray, candidates) -> np.ndarray:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    for _, c in candidates.iterrows():
        cv2.circle(rgb, (int(c["x"]), int(c["y"])), 28, (0, 255, 255), 5, cv2.LINE_AA)
    return rgb


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Microaneurysm Detection - research prototype")
        self.geometry("920x780")
        self.detector = None
        self.result = None
        self.image_path = None
        self.photos = {}

        bar = ttk.Frame(self, padding=8)
        bar.pack(fill="x")
        ttk.Button(bar, text="Upload fundus image", command=self.load).pack(side="left")
        self.export_button = ttk.Button(bar, text="Export results", command=self.export, state="disabled")
        self.export_button.pack(side="left", padx=8)
        self.status = ttk.Label(bar, text="Load an image to start.")
        self.status.pack(side="left", padx=12)

        grid = ttk.Frame(self, padding=8)
        grid.pack(fill="both", expand=True)
        self.panels = {}
        for i, name in enumerate(["Original", "Enhanced (illumination + CLAHE + denoise)",
                                  "Detection mask", "Suspected microaneurysms"]):
            frame = ttk.LabelFrame(grid, text=name, padding=4)
            frame.grid(row=i // 2, column=i % 2, padx=6, pady=6, sticky="nsew")
            label = ttk.Label(frame)
            label.pack()
            self.panels[name] = label
        ttk.Label(self, text=DISCLAIMER, foreground="#a33", wraplength=880, padding=8).pack(fill="x")

    def load(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png *.tif *.tiff")])
        if not path:
            return
        self.image_path = Path(path)
        self.status.config(text="Processing ... (about 10-20 s on a 12-megapixel image)")
        self.export_button.config(state="disabled")
        threading.Thread(target=self.process, daemon=True).start()

    def process(self):
        try:
            if self.detector is None:
                self.detector = MicroaneurysmDetector()
            image_bgr = read_image(self.image_path)
            result = self.detector.run(image_bgr, model="D")
            result["image_bgr"] = image_bgr
            self.result = result
            self.after(0, self.show)
        except Exception as error:   # show any failure to the user instead of crashing silently
            self.after(0, lambda: messagebox.showerror("Error", str(error)))
            self.after(0, lambda: self.status.config(text="Failed."))

    def show(self):
        r = self.result
        images = {
            "Original": cv2.cvtColor(r["image_bgr"], cv2.COLOR_BGR2RGB),
            "Enhanced (illumination + CLAHE + denoise)": r["pre"]["smooth"],
            # dilate only for DISPLAY so that tiny detections remain visible after downscaling
            "Detection mask": cv2.dilate(r["mask"] * 255, np.ones((15, 15), np.uint8)),
            "Suspected microaneurysms": annotate(r["image_bgr"], r["candidates"]),
        }
        for name, img in images.items():
            self.photos[name] = to_photo(img)
            self.panels[name].config(image=self.photos[name])
        self.status.config(text=f"{len(r['candidates'])} suspected MA  |  processing time {r['time_s']:.1f} s")
        self.export_button.config(state="normal")

    def export(self):
        folder = filedialog.askdirectory(title="Choose export folder")
        if not folder:
            return
        stem = self.image_path.stem
        out = Path(folder)
        annotated = cv2.cvtColor(annotate(self.result["image_bgr"], self.result["candidates"]), cv2.COLOR_RGB2BGR)
        cv2.imencode(".png", annotated)[1].tofile(str(out / f"{stem}_suspected_MA.png"))
        cv2.imencode(".png", self.result["mask"] * 255)[1].tofile(str(out / f"{stem}_MA_mask.png"))
        self.result["candidates"].round(3).to_csv(out / f"{stem}_candidates.csv", index=False)
        messagebox.showinfo("Exported", f"Saved annotated image, mask and CSV to:\n{out}")


if __name__ == "__main__":
    App().mainloop()
