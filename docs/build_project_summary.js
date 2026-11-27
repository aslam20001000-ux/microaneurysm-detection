// Two-page project summary for the supervisor (English). Needs docx@8.5.0 on NODE_PATH.
const fs = require("fs");
const path = require("path");
const { Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, WidthType, AlignmentType,
  BorderStyle, ShadingType } = require("docx");

const ROOT = path.resolve(__dirname, "..");
const OUT = path.join(ROOT, "docs/Project_Summary_for_Supervisor.docx");
const FIG = (f) => path.join(ROOT, "docs/report_figures", f);
const IN = 1440, FONT = "Calibri", ACCENT = "C8502A", DARK = "1B2433";

const runs = (t, o = {}) => t.split(/(\*\*[^*]+\*\*)/g).filter(Boolean).map((s) => s.startsWith("**")
  ? new TextRun({ text: s.slice(2, -2), bold: true, font: FONT, ...o }) : new TextRun({ text: s, font: FONT, ...o }));
const P = (t) => new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 100 }, children: runs(t, { size: 21 }) });
const H = (t) => new Paragraph({ spacing: { before: 180, after: 80 }, keepNext: true, children: runs(t, { size: 25, bold: true, color: ACCENT }) });
const B = (t) => new Paragraph({ numbering: { reference: "b", level: 0 }, alignment: AlignmentType.JUSTIFIED, spacing: { after: 50 }, children: runs(t, { size: 21 }) });
function img(file, widthIn, caption) {
  const b = fs.readFileSync(FIG(file)); const w = b.readUInt32BE(16), h = b.readUInt32BE(20);
  return [new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, children: [new ImageRun({ data: b, transformation: { width: widthIn * 96, height: Math.round(widthIn * 96 * h / w) } })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 120 }, children: runs(caption, { size: 17, italics: true, color: "5A6573" }) })];
}
const LINE = { style: BorderStyle.SINGLE, size: 4, color: "C9D0D8" };
function table(head, rows, widths, hi) {
  const cell = (t, i, isHead, isHi) => new TableCell({ width: { size: widths[i] * IN, type: WidthType.DXA },
    shading: isHead ? { type: ShadingType.CLEAR, fill: DARK, color: "auto" } : isHi ? { type: ShadingType.CLEAR, fill: "F4E3D7", color: "auto" } : undefined,
    borders: { top: LINE, bottom: LINE, left: LINE, right: LINE }, margins: { top: 40, bottom: 40, left: 80, right: 80 },
    children: [new Paragraph({ alignment: i ? AlignmentType.CENTER : AlignmentType.LEFT, children: runs(t, { size: 18, bold: isHead || isHi, color: isHead ? "FFFFFF" : DARK }) })] });
  return [new Table({ width: { size: widths.reduce((a, b) => a + b, 0) * IN, type: WidthType.DXA }, columnWidths: widths.map((w) => w * IN),
    rows: [new TableRow({ tableHeader: true, children: head.map((h, i) => cell(h, i, true, false)) }),
      ...rows.map((r, ri) => new TableRow({ children: r.map((c, i) => cell(c, i, false, ri === hi)) }))] }), new Paragraph({ spacing: { after: 80 }, children: [] })];
}

const c = [];
const add = (...x) => x.flat().forEach((e) => c.push(e));
add(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 }, children: runs("Early Detection of Diabetic Retinopathy: Microaneurysm Detection Using Traditional Digital Image Processing", { size: 30, bold: true, color: DARK }) }));
add(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 160 }, children: runs("Project summary · Digital Image Processing · Biomedical Engineering · [Student Name] · Supervisor: [Dr. Name]", { size: 19, color: "5A6573" }) }));

add(H("1. Aim"));
add(P("To build a low-cost, CPU-only system that marks suspected **microaneurysms (MA)**, the earliest visible sign of diabetic retinopathy, in colour fundus images, and to measure what each classical image-processing step contributes through an **ablation study**. The system is a research prototype that assists graders; it does not diagnose."));

add(H("2. Data and protocol"));
add(P("IDRiD segmentation subset (Porwal et al., 2018): 54 training and 27 test images of 4288 × 2848 px with pixel-level MA masks (2412 and 1085 lesions). Every parameter was chosen on the training images by a selection rule fixed before each experiment; the test images were processed **once**, after all parameters were frozen."));

add(H("3. Method"));
add(img("fig01_pipeline.png", 3.3, "Pipeline and ablation models; each model adds one technique to the previous one."));
add(B("**Preprocessing:** green channel, field-of-view mask, divisive illumination correction (Gaussian background, 151 px), CLAHE (clip 1.0), Gaussian denoising (σ = 1.5)."));
add(B("**Candidates:** morphological black-hat (disc r = 16) and vessel suppression by an oriented line top-hat (length 41 px, 12 orientations), robust z-score thresholding and connected components."));
add(B("**False-positive removal:** random forest on 20 candidate features (shape, response, intensity, colour, position)."));
add(B("**Evaluation:** lesion-level sensitivity, precision, false positives per image (FPI), FROC and CPM; pixel-level Dice and AUPR; 95 % bootstrap confidence intervals. Accuracy is not used because MA cover only 0.10 % of pixels."));

add(H("4. Results on the test set (27 images, run once)"));
add(table(["Model", "Adds", "CPM", "Sensitivity", "Precision", "FP / image", "Time"], [
  ["A", "Green + threshold", "0.000", "0.149", "0.005", "3213", "0.24 s"],
  ["A+", "Illumination correction", "0.000", "0.749", "0.015", "4544", "0.44 s"],
  ["B", "CLAHE", "0.000", "0.715", "0.015", "4367", "0.64 s"],
  ["B+", "Denoising", "0.000", "0.501", "0.025", "903", "0.65 s"],
  ["C", "Black-hat", "0.001", "0.322", "0.024", "568", "0.97 s"],
  ["D0", "Vessel suppression", "0.038", "0.503", "0.280", "60", "0.94 s"],
  ["D", "Random forest", "0.228", "0.547", "0.589", "15.3", "5.46 s"],
], [0.55, 1.65, 0.6, 0.85, 0.8, 0.85, 0.65], 6));
add(P("The full system (Model D) reached a **CPM of 0.228** (95 % CI 0.195–0.267), found **54.7 % of MA** with precision 58.9 % at **15.3 false positives per image** (3213 for the plain threshold), with pixel AUPR 0.326, in 5.5 s per 12-megapixel image on a CPU. Training (cross-validated) and test CPM were 0.194 and 0.228, so the parameters did not overfit."));
add(img("fig08_froc_test.png", 3.6, "Lesion-level FROC on the test set; Model D dominates at every operating point."));

add(H("5. Key findings"));
add(B("**Vessel suppression** (line top-hat) and **feature-based filtering** gave almost all of the gain: false positives per image fell from 568 to 60 and then to 15."));
add(B("**Illumination correction** made a single threshold usable across the retina (training AUC 0.68 → 0.84); **CLAHE added almost nothing** after it (+0.003)."));
add(B("A **multi-scale black-hat did not beat a single 16 px disc**: for a disc, the largest scale already contains the smaller ones."));
add(B("On the same test images, leading deep-learning entries of the IDRiD challenge reached pixel AUPR ≈ 0.5 (Porwal et al., 2020); our CPU-only method reaches 0.33 (indicative comparison)."));

add(H("6. Limitations and next steps"));
add(P("Small single-source dataset; no MA-free images, so image-level specificity was not measured; ground-truth outlines larger than the dark core lower pixel Dice; small haemorrhages count as false positives; pixel sizes are tuned for this camera resolution. Next: specificity on IDRiD images graded “no DR”, features against haemorrhages, a second camera, and comparison with a small CNN."));

add(H("7. Deliverables"));
add(P("Python source code and nine experiments; desktop application (PySide6) with model selection, sensitivity control and ground-truth comparison; IEEE-style report; presentation; all results and figures. Repository: github.com/aslam20001000-ux/microaneurysm-detection"));

const doc = new Document({ styles: { default: { document: { run: { font: FONT, size: 21 } } } },
  numbering: { config: [{ reference: "b", levels: [{ level: 0, format: "bullet", text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 360, hanging: 220 } } } }] }] },
  sections: [{ properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 0.8 * IN, bottom: 0.8 * IN, left: 0.9 * IN, right: 0.9 * IN } } }, children: c }] });
Packer.toBuffer(doc).then((b) => { fs.writeFileSync(OUT, b); console.log("written", OUT); });
