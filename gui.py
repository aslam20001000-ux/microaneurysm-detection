"""
Microaneurysm Detection Studio - desktop application (PySide6 / Qt).

Run:  python gui.py

Features
  * open one fundus image, or a whole folder (batch mode with CSV summary)
  * choose the ablation model (A ... D) and run it in a background thread
  * zoom (mouse wheel) and pan (drag) at full 12-megapixel resolution
  * sensitivity slider for Model D: re-filters the classifier scores instantly
  * pipeline gallery: every intermediate stage can be shown in the main viewer
  * table of suspected MA: click a row to fly to that lesion
  * if the image belongs to IDRiD and its ground-truth mask exists, live TP / FP / missed counts
  * export: annotated PNG, binary mask, candidates CSV and a text report

Research prototype: it marks SUSPECTED lesions only and gives no diagnosis.
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PySide6.QtCore import QPointF, QRectF, QSize, Qt, QThread, Signal
from PySide6.QtGui import QAction, QBrush, QColor, QFont, QIcon, QImage, QKeySequence, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox, QComboBox, QFileDialog, QFrame,
                               QGraphicsEllipseItem, QGraphicsPixmapItem, QGraphicsRectItem, QGraphicsScene,
                               QGraphicsView, QHBoxLayout, QHeaderView, QLabel, QListWidget, QListWidgetItem,
                               QMainWindow, QMessageBox, QProgressBar, QPushButton, QSlider, QSplitter, QTableWidget,
                               QTableWidgetItem, QVBoxLayout, QWidget)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src import config                                     # noqa: E402
from src.dataset_loader import read_image, read_mask        # noqa: E402
from src.detection import MicroaneurysmDetector             # noqa: E402
from src.evaluation import gt_match_zones                   # noqa: E402

# ----------------------------------------------------------------------------- look
BG, PANEL, CARD, BORDER = "#0f1420", "#161d2b", "#1e2738", "#2b3548"
TEXT, MUTED, ACCENT, ACCENT2 = "#e6e9ef", "#8d97a8", "#e0663a", "#f0a33b"
TP_COL, FP_COL, FN_COL, CAND_COL = QColor(0, 214, 214), QColor(236, 72, 236), QColor(250, 210, 0), QColor(224, 102, 58)
MODELS = [("D", "D — Full system (recommended)"), ("D0", "D0 — Vessel suppression"),
          ("C", "C — Black-hat"), ("B+", "B+ — Denoised threshold"), ("A", "A — Green-channel threshold")]
STYLE = f"""
QMainWindow, QWidget {{ background: {BG}; color: {TEXT}; font-family: 'Segoe UI'; font-size: 10pt; }}
QLabel, QCheckBox {{ background: transparent; }}
QFrame#panel {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 10px; }}
QFrame#card {{ background: {CARD}; border-radius: 10px; }}
QLabel#h1 {{ font-size: 17pt; font-weight: 600; }}
QLabel#sub {{ color: {MUTED}; font-size: 9pt; }}
QLabel#section {{ color: {MUTED}; font-size: 8.5pt; font-weight: 600; letter-spacing: 1px; }}
QLabel#value {{ font-size: 22pt; font-weight: 600; color: {TEXT}; }}
QLabel#accentvalue {{ font-size: 22pt; font-weight: 600; color: {ACCENT}; }}
QLabel#caption {{ color: {MUTED}; font-size: 8.5pt; }}
QPushButton {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 8px; padding: 8px 12px; }}
QPushButton:hover {{ border-color: {ACCENT}; }}
QPushButton:disabled {{ color: #556070; }}
QPushButton#primary {{ background: {ACCENT}; border: none; color: white; font-weight: 600; }}
QPushButton#primary:hover {{ background: #ef7a4e; }}
QComboBox {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 8px; padding: 6px 8px; }}
QComboBox QAbstractItemView {{ background: {CARD}; selection-background-color: {ACCENT}; }}
QSlider::groove:horizontal {{ height: 6px; background: {BORDER}; border-radius: 3px; }}
QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 3px; }}
QSlider::handle:horizontal {{ background: white; width: 16px; margin: -6px 0; border-radius: 8px; }}
QCheckBox::indicator {{ width: 16px; height: 16px; }}
QProgressBar {{ background: {CARD}; border: none; border-radius: 4px; height: 8px; text-align: center; color: transparent; }}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 4px; }}
QTableWidget {{ background: {PANEL}; gridline-color: {BORDER}; border: none; selection-background-color: #3a2a24; }}
QHeaderView::section {{ background: {CARD}; color: {MUTED}; border: none; padding: 4px; font-weight: 600; }}
QListWidget {{ background: {PANEL}; border: none; }}
QListWidget::item {{ color: {MUTED}; border-radius: 6px; padding: 4px; }}
QListWidget::item:selected {{ background: {CARD}; color: {TEXT}; border: 1px solid {ACCENT}; }}
QGraphicsView {{ background: #05070b; border: 1px solid {BORDER}; border-radius: 10px; }}
QStatusBar {{ color: {MUTED}; }}
"""


def to_qimage(arr: np.ndarray) -> QImage:
    """numpy uint8 (gray or RGB) -> independent QImage copy."""
    arr = np.ascontiguousarray(arr)
    if arr.ndim == 2:
        return QImage(arr.data, arr.shape[1], arr.shape[0], arr.strides[0], QImage.Format_Grayscale8).copy()
    return QImage(arr.data, arr.shape[1], arr.shape[0], arr.strides[0], QImage.Format_RGB888).copy()


def heat(z: np.ndarray, fov: np.ndarray) -> np.ndarray:
    """z-score response -> colour heat map (RGB) for display."""
    scaled = np.clip(z / 8.0 * 255, 0, 255).astype(np.uint8)
    rgb = cv2.cvtColor(cv2.applyColorMap(scaled, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)
    rgb[fov == 0] = 0
    return rgb


def find_ground_truth(image_path: Path):
    """IDRiD naming IDRiD_xx.jpg -> look for the MA mask in the training and testing folders."""
    stem = image_path.stem
    if not stem.startswith("IDRiD_"):
        return None
    for split in config.SPLITS.values():
        p = config.GROUNDTRUTH_DIR / split / config.LESIONS["MA"] / f"{stem}_MA.tif"
        if p.is_file():
            return p
    return None


# ----------------------------------------------------------------------------- image viewer
class ImageView(QGraphicsView):
    def __init__(self):
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.pixmap_item = QGraphicsPixmapItem()
        self.pixmap_item.setTransformationMode(Qt.SmoothTransformation)
        self.scene().addItem(self.pixmap_item)
        self.overlays = []
        self.placeholder = self.scene().addText("Open a fundus image to start  (Ctrl+O)", QFont("Segoe UI", 16))
        self.placeholder.setDefaultTextColor(QColor(MUTED))

    def set_image(self, arr: np.ndarray, keep_view: bool = False):
        self.placeholder.setVisible(False)
        self.pixmap_item.setPixmap(QPixmap.fromImage(to_qimage(arr)))
        self.scene().setSceneRect(QRectF(0, 0, arr.shape[1], arr.shape[0]))
        if not keep_view:
            self.fit()

    def set_marks(self, marks):
        """marks: list of (x, y, QColor, kind) - kind 'circle' or 'square'."""
        for item in self.overlays:
            self.scene().removeItem(item)
        self.overlays = []
        for x, y, colour, kind in marks:
            r = 26
            pen = QPen(colour, 2.2)
            pen.setCosmetic(True)
            item = (QGraphicsEllipseItem if kind == "circle" else QGraphicsRectItem)(x - r, y - r, 2 * r, 2 * r)
            item.setPen(pen)
            item.setBrush(QBrush(Qt.NoBrush))
            self.scene().addItem(item)
            self.overlays.append(item)

    def fit(self):
        if not self.pixmap_item.pixmap().isNull():
            self.fitInView(self.pixmap_item, Qt.KeepAspectRatio)

    def zoom_to(self, x: float, y: float, size: float = 260):
        self.fitInView(QRectF(x - size / 2, y - size / 2, size, size), Qt.KeepAspectRatio)
        self.centerOn(QPointF(x, y))

    def wheelEvent(self, event):
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        self.scale(factor, factor)


# ----------------------------------------------------------------------------- background workers
class AnalyzeWorker(QThread):
    done = Signal(dict)
    failed = Signal(str)

    def __init__(self, detector_holder, image, model):
        super().__init__()
        self.holder, self.image, self.model = detector_holder, image, model

    def run(self):
        try:
            if self.holder.get("detector") is None:
                self.holder["detector"] = MicroaneurysmDetector()
            result = self.holder["detector"].run(self.image, self.model)
            labels = result["labels"]
            result["areas"] = np.bincount(labels.ravel(), minlength=len(result["centroids"]) + 1)[1:]
            self.done.emit(result)
        except Exception as error:  # report every failure to the UI
            self.failed.emit(str(error))


class BatchWorker(QThread):
    progress = Signal(int, int, str)
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, detector_holder, paths, threshold):
        super().__init__()
        self.holder, self.paths, self.threshold = detector_holder, paths, threshold

    def run(self):
        try:
            if self.holder.get("detector") is None:
                self.holder["detector"] = MicroaneurysmDetector()
            rows = []
            for i, p in enumerate(self.paths, 1):
                self.progress.emit(i, len(self.paths), p.name)
                r = self.holder["detector"].run(read_image(p), "D")
                rows.append({"image": p.name, "suspected_MA": int((r["all_scores"] >= self.threshold).sum()),
                             "time_s": round(r["time_s"], 2)})
            self.done.emit(pd.DataFrame(rows))
        except Exception as error:
            self.failed.emit(str(error))


# ----------------------------------------------------------------------------- small widgets
def card(title: str, accent: bool = False):
    frame = QFrame(objectName="card")
    lay = QVBoxLayout(frame)
    lay.setContentsMargins(14, 10, 14, 10)
    lay.setSpacing(0)
    value = QLabel("—", objectName="accentvalue" if accent else "value")
    cap = QLabel(title, objectName="caption")
    lay.addWidget(value)
    lay.addWidget(cap)
    return frame, value


def section(text: str):
    return QLabel(text.upper(), objectName="section")


# ----------------------------------------------------------------------------- main window
class Studio(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Microaneurysm Detection Studio")
        self.resize(1560, 940)
        self.holder = {"detector": None}
        self.image = self.path = self.result = self.gt = None
        self.stages = {}
        self.current_stage = "Detections"
        self._build()
        self._shortcuts()
        self.statusBar().showMessage("Research prototype — highlights SUSPECTED microaneurysms only; not a diagnosis.")

    # ------------------------------------------------------------------ layout
    def _build(self):
        root = QWidget()
        outer = QVBoxLayout(root)
        outer.setContentsMargins(14, 12, 14, 8)

        header = QHBoxLayout()
        titles = QVBoxLayout()
        titles.addWidget(QLabel("Microaneurysm Detection Studio", objectName="h1"))
        titles.addWidget(QLabel("Early diabetic-retinopathy screening aid · classical image processing · CPU only", objectName="sub"))
        header.addLayout(titles)
        header.addStretch()
        self.file_label = QLabel("No image loaded", objectName="sub")
        header.addWidget(self.file_label)
        outer.addLayout(header)

        split = QSplitter(Qt.Horizontal)
        self.view = ImageView()                          # created first: the side panels connect to it
        split.addWidget(self._left_panel())
        center = QWidget()
        cl = QVBoxLayout(center)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.addWidget(self.view, 1)
        self.gallery = QListWidget()
        self.gallery.setViewMode(QListWidget.IconMode)
        self.gallery.setFlow(QListWidget.LeftToRight)
        self.gallery.setWrapping(False)
        self.gallery.setIconSize(QSize(150, 100))
        self.gallery.setFixedHeight(142)
        self.gallery.setMovement(QListWidget.Static)
        self.gallery.itemClicked.connect(lambda it: self.show_stage(it.text()))
        cl.addWidget(self.gallery)
        split.addWidget(center)
        split.addWidget(self._right_panel())
        split.setSizes([300, 900, 360])
        outer.addWidget(split, 1)
        self.setCentralWidget(root)

    def _left_panel(self):
        panel = QFrame(objectName="panel")
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.setSpacing(10)
        lay.addWidget(section("Image"))
        open_btn = QPushButton("Open fundus image…")
        open_btn.clicked.connect(self.open_image)
        lay.addWidget(open_btn)
        batch_btn = QPushButton("Batch: analyse a folder…")
        batch_btn.clicked.connect(self.batch)
        lay.addWidget(batch_btn)

        lay.addSpacing(6)
        lay.addWidget(section("Model"))
        self.model_box = QComboBox()
        for key, name in MODELS:
            self.model_box.addItem(name, key)
        lay.addWidget(self.model_box)
        self.run_btn = QPushButton("Analyse", objectName="primary")
        self.run_btn.setEnabled(False)
        self.run_btn.clicked.connect(self.analyse)
        lay.addWidget(self.run_btn)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        lay.addWidget(self.progress)

        lay.addSpacing(6)
        lay.addWidget(section("Sensitivity (Model D)"))
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(30, 99)
        self.slider.setValue(84)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self.refresh)
        lay.addWidget(self.slider)
        self.slider_label = QLabel("Classifier threshold 0.84 (frozen operating point)", objectName="caption")
        self.slider_label.setWordWrap(True)
        lay.addWidget(self.slider_label)
        reset = QPushButton("Reset to operating point")
        reset.clicked.connect(lambda: self.slider.setValue(84))
        lay.addWidget(reset)

        lay.addSpacing(6)
        lay.addWidget(section("Overlays"))
        self.show_marks = QCheckBox("Suspected microaneurysms")
        self.show_marks.setChecked(True)
        self.show_gt = QCheckBox("Ground truth (if available)")
        self.show_gt.setChecked(True)
        for cb in (self.show_marks, self.show_gt):
            cb.toggled.connect(self.refresh)
            lay.addWidget(cb)
        legend = QLabel(
            f"<span style='color:{CAND_COL.name()}'>○</span> suspected MA &nbsp; "
            f"<span style='color:{TP_COL.name()}'>○</span> correct &nbsp; "
            f"<span style='color:{FP_COL.name()}'>○</span> false alarm &nbsp; "
            f"<span style='color:{FN_COL.name()}'>□</span> missed", objectName="caption")
        legend.setWordWrap(True)
        lay.addWidget(legend)

        lay.addStretch()
        self.export_btn = QPushButton("Export results…")
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self.export)
        lay.addWidget(self.export_btn)
        fit = QPushButton("Fit image  (F)")
        fit.clicked.connect(self.view.fit)
        lay.addWidget(fit)
        return panel

    def _right_panel(self):
        panel = QFrame(objectName="panel")
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.setSpacing(10)
        lay.addWidget(section("Summary"))
        row1 = QHBoxLayout()
        c1, self.v_count = card("suspected MA", accent=True)
        c2, self.v_time = card("processing time")
        row1.addWidget(c1)
        row1.addWidget(c2)
        lay.addLayout(row1)
        row2 = QHBoxLayout()
        c3, self.v_score = card("mean confidence")
        c4, self.v_model = card("model")
        row2.addWidget(c3)
        row2.addWidget(c4)
        lay.addLayout(row2)

        lay.addWidget(section("Against ground truth"))
        self.gt_label = QLabel("Load an IDRiD image to compare with its annotation.", objectName="caption")
        self.gt_label.setWordWrap(True)
        self.gt_label.setTextFormat(Qt.RichText)
        lay.addWidget(self.gt_label)

        lay.addWidget(section("Suspected lesions  (click to zoom)"))
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["#", "x", "y", "score"])
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.cellClicked.connect(self.zoom_to_row)
        lay.addWidget(self.table, 1)
        return panel

    def _shortcuts(self):
        for key, fn in [(QKeySequence.Open, self.open_image), ("F", self.view.fit), ("Ctrl+R", self.analyse),
                        ("Ctrl+E", self.export)]:
            act = QAction(self)
            act.setShortcut(QKeySequence(key))
            act.triggered.connect(fn)
            self.addAction(act)

    # ------------------------------------------------------------------ actions
    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open fundus image", str(config.IMAGES_DIR) if config.IMAGES_DIR.exists() else "",
                                              "Images (*.jpg *.jpeg *.png *.tif *.tiff)")
        if path:
            self.load_path(Path(path))

    def load_path(self, path: Path):
        try:
            self.image = read_image(Path(path))
        except (FileNotFoundError, ValueError) as error:
            QMessageBox.critical(self, "Cannot open image", str(error))
            return
        self.path = Path(path)
        self.result = None
        gt_path = find_ground_truth(self.path)
        self.gt = None
        if gt_path is not None:
            mask = read_mask(gt_path)
            zones, n_gt = gt_match_zones(mask)
            _, labels, _, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
            self.gt = {"zones": zones, "n": n_gt, "centroids": centroids[1:],
                       "ids": np.array([zones[int(round(y)), int(round(x))] for x, y in centroids[1:]])}
        rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.stages = {"Original": rgb}
        self.view.set_image(rgb)
        self.view.set_marks([])
        self._fill_gallery()
        h, w = self.image.shape[:2]
        self.file_label.setText(f"{self.path.name}  ·  {w} × {h} px" + ("  ·  ground truth found" if self.gt else ""))
        self.run_btn.setEnabled(True)
        self.export_btn.setEnabled(False)
        self.table.setRowCount(0)
        for v in (self.v_count, self.v_time, self.v_score, self.v_model):
            v.setText("—")
        self.gt_label.setText(f"Ground truth: {self.gt['n']} annotated MA. Run the analysis to compare." if self.gt
                              else "No ground-truth mask for this image.")
        self.analyse()

    def analyse(self):
        if self.image is None or (hasattr(self, "worker") and self.worker.isRunning()):
            return
        model = self.model_box.currentData()
        self.run_btn.setEnabled(False)
        self.progress.setRange(0, 0)                     # busy animation
        self.statusBar().showMessage(f"Analysing with model {model} … (first run loads the classifier)")
        self.worker = AnalyzeWorker(self.holder, self.image, model)
        self.worker.done.connect(lambda r: self.on_result(r, model))
        self.worker.failed.connect(self.on_error)
        self.worker.start()

    def on_error(self, message):
        self.progress.setRange(0, 1)
        self.run_btn.setEnabled(True)
        QMessageBox.critical(self, "Analysis failed", message)

    def on_result(self, result, model):
        self.progress.setRange(0, 1)
        self.progress.setValue(1)
        self.run_btn.setEnabled(True)
        self.export_btn.setEnabled(True)
        self.result, self.result["model"] = result, model
        self.slider.setEnabled(model == "D")
        pre = result["pre"]
        self.stages = {
            "Original": cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB),
            "Green channel": np.clip(pre["green"], 0, 255).astype(np.uint8),
            "Illumination corrected": np.clip(pre["corrected"], 0, 255).astype(np.uint8),
            "CLAHE": pre["clahe"],
            "Denoised (B+)": pre["smooth"],
            f"Response ({model})": heat(result["z"], pre["fov"]),
            "Detections": None,                          # filled by refresh()
        }
        self.v_time.setText(f"{result['time_s']:.1f} s")
        self.v_model.setText(model)
        self.statusBar().showMessage(f"Done in {result['time_s']:.1f} s. Research prototype — suspected lesions only, not a diagnosis.")
        self.refresh()
        self._fill_gallery()
        self.show_stage("Detections")

    def kept(self):
        r = self.result
        if r is None:
            return np.zeros(0, bool)
        t = self.slider.value() / 100 if r["model"] == "D" else 0.0
        return r["all_scores"] >= t

    def refresh(self):
        if self.result is None:
            return
        r = self.result
        t = self.slider.value() / 100
        self.slider_label.setText(f"Classifier threshold {t:.2f}" + ("  (frozen operating point)" if self.slider.value() == 84
                                  else "  — lower = more sensitive, more false alarms"))
        keep = self.kept()
        cen, scores = r["centroids"][keep], r["all_scores"][keep]
        self.v_count.setText(str(int(keep.sum())))
        self.v_score.setText(f"{scores.mean():.2f}" if r["model"] == "D" and len(scores) else "—")

        # detection overlay image (mask tinted on the original)
        lut = np.zeros(len(keep) + 1, np.uint8)
        lut[1:][keep] = 1
        mask = lut[r["labels"]]
        r["mask"] = mask
        overlay = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB).copy()
        # outline (not fill) so the lesion itself stays visible when zoomed in
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (255, 235, 59), 2, cv2.LINE_AA)
        self.stages["Detections"] = overlay

        # marks: TP / FP / missed when ground truth exists
        marks, hit = [], set()
        for x, y in cen:
            colour = CAND_COL
            if self.gt is not None:
                gid = self.gt["zones"][int(round(y)), int(round(x))]
                colour = TP_COL if gid > 0 else FP_COL
                if gid > 0:
                    hit.add(int(gid))
            if self.show_marks.isChecked():
                marks.append((x, y, colour, "circle"))
        if self.gt is not None:
            missed = [(x, y) for (x, y), gid in zip(self.gt["centroids"], self.gt["ids"]) if int(gid) not in hit]
            if self.show_gt.isChecked():
                marks += [(x, y, FN_COL, "square") for x, y in missed]
            tp = sum(1 for x, y in cen if self.gt["zones"][int(round(y)), int(round(x))] > 0)
            fp = len(cen) - tp
            sens = len(hit) / max(self.gt["n"], 1)
            self.gt_label.setText(
                f"<b style='color:{TP_COL.name()}'>{len(hit)}</b> of {self.gt['n']} annotated MA found "
                f"(sensitivity <b>{sens:.0%}</b>)<br><b style='color:{FP_COL.name()}'>{fp}</b> false alarms &nbsp;·&nbsp; "
                f"<b style='color:{FN_COL.name()}'>{len(missed)}</b> missed")
        self.marks = marks
        self.view.set_marks(marks if self.current_stage in ("Original", "Detections") else [])
        if self.current_stage == "Detections":
            self.view.set_image(self.stages["Detections"], keep_view=True)

        order = np.argsort(-scores)
        self.table.setRowCount(len(order))
        for row, i in enumerate(order):
            for col, val in enumerate([row + 1, int(cen[i][0]), int(cen[i][1]), f"{scores[i]:.2f}" if r["model"] == "D" else "—"]):
                item = QTableWidgetItem(str(val))
                item.setTextAlignment(Qt.AlignCenter)
                item.setData(Qt.UserRole, (float(cen[i][0]), float(cen[i][1])))
                self.table.setItem(row, col, item)
        self._fill_gallery()

    def show_stage(self, name):
        if name not in self.stages or self.stages[name] is None:
            return
        self.current_stage = name
        self.view.set_image(self.stages[name], keep_view=True)
        self.view.set_marks(self.marks if name in ("Original", "Detections") and self.result is not None else [])
        for i in range(self.gallery.count()):
            if self.gallery.item(i).text() == name:
                self.gallery.setCurrentRow(i)

    def _fill_gallery(self):
        current = self.current_stage
        self.gallery.clear()
        for name, arr in self.stages.items():
            if arr is None:
                continue
            thumb = cv2.resize(arr, (150, int(150 * arr.shape[0] / arr.shape[1])), interpolation=cv2.INTER_AREA)
            item = QListWidgetItem(QIcon(QPixmap.fromImage(to_qimage(thumb))), name)
            self.gallery.addItem(item)
            if name == current:
                self.gallery.setCurrentItem(item)

    def zoom_to_row(self, row, _col):
        x, y = self.table.item(row, 0).data(Qt.UserRole)
        if self.current_stage not in ("Original", "Detections"):
            self.show_stage("Original")
        self.view.zoom_to(x, y)

    def export(self):
        if self.result is None:
            return
        folder = QFileDialog.getExistingDirectory(self, "Export results to folder")
        if not folder:
            return
        out, stem, r = Path(folder), self.path.stem, self.result
        keep = self.kept()
        annotated = self.image.copy()
        for x, y in r["centroids"][keep]:
            cv2.circle(annotated, (int(x), int(y)), 28, (58, 102, 224), 5, cv2.LINE_AA)
        cv2.imencode(".png", annotated)[1].tofile(str(out / f"{stem}_suspected_MA.png"))
        cv2.imencode(".png", r["mask"] * 255)[1].tofile(str(out / f"{stem}_MA_mask.png"))
        pd.DataFrame({"x": r["centroids"][keep, 0].round(1), "y": r["centroids"][keep, 1].round(1),
                      "area_px": r["areas"][keep], "score": r["all_scores"][keep].round(3)}).to_csv(out / f"{stem}_candidates.csv", index=False)
        report = [f"Image: {self.path.name}", f"Model: {r['model']}", f"Classifier threshold: {self.slider.value() / 100:.2f}",
                  f"Suspected microaneurysms: {int(keep.sum())}", f"Processing time: {r['time_s']:.1f} s",
                  f"Exported: {time.strftime('%Y-%m-%d %H:%M')}", "",
                  "Research prototype - suspected lesions only. Not a diagnosis; results must be reviewed by an ophthalmologist."]
        (out / f"{stem}_report.txt").write_text("\n".join(report), encoding="utf-8")
        self.statusBar().showMessage(f"Exported 4 files to {out}")

    def batch(self):
        folder = QFileDialog.getExistingDirectory(self, "Choose a folder of fundus images")
        if not folder:
            return
        paths = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"})
        if not paths:
            QMessageBox.information(self, "Batch", "No images found in that folder.")
            return
        self.progress.setRange(0, len(paths))
        self.bworker = BatchWorker(self.holder, paths, self.slider.value() / 100)
        self.bworker.progress.connect(lambda i, n, name: (self.progress.setValue(i), self.statusBar().showMessage(f"Batch {i}/{n}: {name}")))
        self.bworker.failed.connect(self.on_error)
        self.bworker.done.connect(lambda df: self.batch_done(df, Path(folder)))
        self.bworker.start()

    def batch_done(self, df, folder):
        target = folder / "MA_batch_summary.csv"
        try:
            df.to_csv(target, index=False)
        except OSError:                                  # read-only folder -> save next to the project outputs
            target = config.OUTPUTS_DIR / "MA_batch_summary.csv"
            df.to_csv(target, index=False)
        QMessageBox.information(self, "Batch finished",
                                f"{len(df)} images analysed with Model D.\nMedian time {df.time_s.median():.1f} s per image.\nSummary saved to:\n{target}")


def main():
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)
    window = Studio()
    window.show()
    if len(sys.argv) > 1:                                 # optional: python gui.py path/to/image.jpg
        window.load_path(Path(sys.argv[1]))
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
