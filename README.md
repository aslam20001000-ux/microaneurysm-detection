# Early Detection of Diabetic Retinopathy — Microaneurysm Detection with Traditional Image Processing

University project (Biomedical Engineering, Digital Image Processing).
A low-cost, CPU-only pipeline that localises **microaneurysms (MA)** in colour fundus images of the
**IDRiD** dataset, evaluated with an **ablation study** (each model adds one technique).

> Research prototype — **not** a certified medical device and not a diagnostic tool.

## 1. Environment (Windows, VS Code)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install -r requirements.txt
python scripts\check_environment.py
```

Dataset: IDRiD (Kaggle / IEEE DataPort, CC-BY 4.0) is **not** included in this repository; download it
separately. Tell the code where it is in one of three ways (first match wins):

1. environment variable `IDRID_ROOT`
2. a file `dataset_path.txt` in the project root whose first line is the path to the `IDRID` folder
   (ignored by git, so each machine keeps its own)
3. put the dataset in `dataset/IDRID/` inside the project

The original images are never modified.

## 2. Project structure

```text
src/
  config.py            all paths and frozen parameters (with the experiment that set each one)
  dataset_loader.py    robust image / mask reading (Unicode-safe on Windows)
  preprocessing.py     channel extraction
  fov.py               field-of-view mask                         (step 4)
  illumination.py      illumination correction                    (step 6)
  enhancement.py       CLAHE                                      (step 7)
  denoise.py           noise reduction                            (step 8)
  candidates.py        responses of models A ... D, thresholding, connected components (steps 9-13)
  features.py          candidate features                         (step 14)
  detection.py         final frozen detector                      (steps 15-16)
  evaluation.py        pixel / lesion / image-level metrics, FROC, CPM (steps 17-18)
  image_metrics.py     CNR, AUC, blockiness used for tuning
  visualization.py     shared plotting helpers
scripts/               environment check, dataset inspection
experiments/           exp01 ... exp09 (one script per pipeline stage, training set only, except exp09)
outputs/               all generated figures, CSV files, masks, trained classifier
docs/                  research report and presentation material
```

## 3. Reproducing the results (in order)

| Script | Stage | Data | Approx. time |
|---|---|---|---|
| `scripts\inspect_dataset.py` | dataset inspection | train + test | 0.5 min |
| `experiments\exp01_green_channel.py` | channel selection | train | 0.5 min |
| `experiments\exp02_fov_mask.py` | FOV mask | train | 3 min |
| `experiments\exp03_illumination.py` | illumination correction | train | 6 min |
| `experiments\exp04_clahe.py` | CLAHE | train | 13 min |
| `experiments\exp05_denoise.py` | noise reduction | train | 6 min |
| `experiments\exp06_blackhat.py` | black-hat radius | train | 10 min |
| `experiments\exp07_vessels.py` | vessel suppression | train | 7 min |
| `experiments\exp08_detection_train.py` | thresholds + FP classifier | train | ~25 min |
| `experiments\exp09_test_evaluation.py` | **final evaluation** | **test (once)** | ~10 min |

Every tuning experiment states its **selection rule before running** and uses only the 54 training
images. The 27 test images are used once, by `exp09`, with all parameters frozen.

## 4. Ablation models

| Model | Adds | Response that is thresholded |
|---|---|---|
| A | green channel + adaptive threshold | 255 − green |
| A+ | illumination correction | 255 − corrected |
| B | CLAHE | 255 − CLAHE |
| B+ | Gaussian denoising | 255 − denoised |
| C | morphological black-hat | black-hat (disc r = 16) |
| D0 | vessel suppression | line top-hat (12 orientations, L = 41) |
| D | feature-based false-positive removal | classifier probability on D0 candidates |

## 5. Final results (test set, 27 images, 1085 MA, run once)

| Model | CPM | Lesion sensitivity | Precision | FP / image | Pixel AUPR | Time / image |
|---|---|---|---|---|---|---|
| A | 0.000 | 0.149 | 0.005 | 3213 | 0.002 | 0.24 s |
| B+ | 0.000 | 0.501 | 0.025 | 903 | 0.008 | 0.65 s |
| C | 0.001 | 0.322 | 0.024 | 568 | 0.010 | 0.97 s |
| D0 | 0.038 | 0.503 | 0.280 | 60 | 0.169 | 0.94 s |
| **D** | **0.228** (95 % CI 0.195–0.267) | **0.547** | **0.589** | **15.3** | **0.326** | **5.46 s** |

Full table: `outputs/experiment_09/ablation_test_summary.csv`. Annotated test images (cyan = correct,
magenta = false alarm, yellow square = missed MA): `outputs/detections/`.

## 6. Deliverables

- Research report: Claude Doc (exportable to Word/PDF), see link shared in the project chat
- Presentation: `docs/Microaneurysm_Detection_Presentation.pptx` (speaker notes in Arabic); rebuild with
  `node docs/build_presentation.js` (needs `pptxgenjs`)
- Desktop application: `python gui.py` (PySide6 "Microaneurysm Detection Studio": model choice, zoom/pan, sensitivity slider, pipeline stages, ground-truth comparison, export, batch). The earlier Tkinter version is kept as `gui_simple_tk.py`

## 7. References (selection)

- Porwal, P., et al. (2018). Indian Diabetic Retinopathy Image Dataset (IDRiD). *Data, 3*(3), 25. https://doi.org/10.3390/data3030025
- Niemeijer, M., et al. (2010). Retinopathy Online Challenge. *IEEE TMI, 29*(1), 185–195. https://doi.org/10.1109/TMI.2009.2033909
