// IEEE conference-style report (two columns, Times New Roman 10 pt) built with docx-js.
// Run (from the folder that has node_modules with docx@8.5.0):  node build_ieee_report.js
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, WidthType, AlignmentType,
  SectionType, HeadingLevel, BorderStyle, TabStopType, VerticalAlign, Math: OMath, MathRun, MathFraction,
  MathSubScript, MathSuperScript, ExternalHyperlink,
} = require("docx");

const ROOT = path.resolve(__dirname, "..");
const FIG = (f) => path.join(ROOT, "docs/report_figures", f);
const OUT = path.join(ROOT, "docs/Microaneurysm_Detection_IEEE_Report.docx");

const FONT = "Times New Roman";
const IN = 1440;                         // twips per inch
const COL_W = 3.5 * IN;                  // one IEEE column
const PAGE = { size: { width: 12240, height: 15840 }, margin: { top: 0.75 * IN, bottom: 1 * IN, left: 0.625 * IN, right: 0.625 * IN } };

// ------------------------------------------------------------------ text helpers
// Inline markup: *italic*, **bold**. Everything else is plain.
function runs(text, base = {}) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*)/g;
  let last = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), ...base }));
    const t = m[0];
    if (t.startsWith("**")) out.push(new TextRun({ text: t.slice(2, -2), bold: true, ...base }));
    else out.push(new TextRun({ text: t.slice(1, -1), italics: true, ...base }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), ...base }));
  return out;
}
const P = (text, opt = {}) => new Paragraph({ children: runs(text), alignment: AlignmentType.JUSTIFIED, indent: { firstLine: 0.17 * IN }, ...opt });
const P0 = (text, opt = {}) => P(text, { indent: { firstLine: 0 }, ...opt });
const H1 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_1, alignment: AlignmentType.CENTER, children: [new TextRun({ text, smallCaps: true })] });
const H2 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun({ text, italics: true })] });
const bullet = (text) => new Paragraph({ children: runs(text), alignment: AlignmentType.JUSTIFIED, indent: { left: 0.2 * IN, hanging: 0.12 * IN },
  numbering: { reference: "dots", level: 0 } });

// ------------------------------------------------------------------ figures
function pngSize(file) {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}
let figNo = 0;
function figure(file, caption, widthIn = 3.5) {
  figNo += 1;
  const { w, h } = pngSize(FIG(file));
  const px = widthIn * 96;
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120 }, keepNext: true,
      children: [new ImageRun({ data: fs.readFileSync(FIG(file)), transformation: { width: px, height: Math.round(px * h / w) } })] }),
    new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { before: 60, after: 160 },
      children: [new TextRun({ text: `Fig. ${figNo}. `, size: 16 }), ...runs(caption, { size: 16 })] }),
  ];
}

// ------------------------------------------------------------------ tables (IEEE: rules only at top, under header, bottom)
const NONE = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
const RULE = { style: BorderStyle.SINGLE, size: 6, color: "000000" };
function table(roman, title, header, rows, widthsIn, boldRow = -1) {
  const widths = widthsIn.map((w) => Math.round(w * IN));
  const total = widths.reduce((a, b) => a + b, 0);
  const cell = (text, i, r, isHeader, isLast) => new TableCell({
    width: { size: widths[i], type: WidthType.DXA },
    margins: { top: 30, bottom: 30, left: 50, right: 50 },
    verticalAlign: VerticalAlign.CENTER,
    borders: { left: NONE, right: NONE, top: isHeader ? RULE : NONE, bottom: isHeader || isLast ? RULE : NONE },
    children: [new Paragraph({ alignment: i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER,
      children: runs(String(text), { size: 14, bold: isHeader || r === boldRow }) })],
  });
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 160 }, keepNext: true, children: [new TextRun({ text: `TABLE ${roman}`, size: 16 })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 }, keepNext: true, children: [new TextRun({ text: title, size: 16, smallCaps: true })] }),
    new Table({
      width: { size: total, type: WidthType.DXA }, columnWidths: widths, alignment: AlignmentType.CENTER,
      borders: { top: NONE, bottom: NONE, left: NONE, right: NONE, insideHorizontal: NONE, insideVertical: NONE },
      rows: [new TableRow({ tableHeader: true, children: header.map((h, i) => cell(h, i, -1, true, false)) }),
        ...rows.map((r, ri) => new TableRow({ children: r.map((c, i) => cell(c, i, ri, false, ri === rows.length - 1)) }))],
    }),
    new Paragraph({ spacing: { after: 120 }, children: [] }),
  ];
}

// ------------------------------------------------------------------ equations: centred math, number at the right margin
let eqNo = 0;
const R = (t) => new MathRun(t);
const sub = (b, s) => new MathSubScript({ children: [R(b)], subScript: [R(s)] });
const frac = (n, d) => new MathFraction({ numerator: n, denominator: d });
function eq(children) {
  eqNo += 1;
  return new Paragraph({
    tabStops: [{ type: TabStopType.CENTER, position: COL_W / 2 }, { type: TabStopType.RIGHT, position: COL_W }],
    spacing: { before: 80, after: 80 },
    children: [new TextRun("\t"), new OMath({ children }), new TextRun(`\t(${eqNo})`)],
  });
}

// ================================================================== content
const titleBlock = [
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 },
    children: [new TextRun({ text: "Early Detection of Diabetic Retinopathy: A Step-by-Step Ablation of Classical Image Processing for Microaneurysm Detection on IDRiD", size: 48 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "[Student Name]", size: 22 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Department of Biomedical Engineering, [Faculty / University], [City], Egypt", size: 20, italics: true })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 160 }, children: [new TextRun({ text: "[email address] \u2014 Course: Digital Image Processing \u2014 Supervisor: [Dr. Name]", size: 20 })] }),
];

const body = [];
const add = (...xs) => xs.flat().forEach((x) => body.push(x));

// ---- Abstract + index terms
add(new Paragraph({ alignment: AlignmentType.JUSTIFIED, children: [
  new TextRun({ text: "Abstract\u2014", bold: true, italics: true, size: 18 }),
  new TextRun({ text: "Microaneurysms (MA) are the earliest visible sign of diabetic retinopathy (DR). We built a CPU-only pipeline of classical image-processing steps that marks suspected MA in colour fundus images, and measured what each step contributes in an ablation study of seven models. Steps: green channel, field-of-view mask, divisive illumination correction, CLAHE, Gaussian denoising, morphological black-hat, vessel suppression by an oriented line top-hat, adaptive thresholding, connected components, and a random forest on 20 candidate features. All parameters were chosen on the 54 IDRiD training images by selection rules fixed before each experiment; the 27 test images (1085 MA) were processed once. On the test set the full model reached a competition performance metric (CPM) of 0.228 (95 % CI 0.195\u20130.267), a lesion-level sensitivity of 0.547 with precision 0.589 at 15.3 false positives per image, and a pixel AUPR of 0.326, in 5.5 s per 12-megapixel image. A plain green-channel threshold produced 3213 false positives per image. Vessel suppression and feature-based filtering gave most of the gain, CLAHE added little after illumination correction, and a multi-scale black-hat did not outperform a single 16-pixel disc.", bold: true, size: 18 }),
] }));
add(new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { before: 100, after: 120 }, children: [
  new TextRun({ text: "Index Terms\u2014", bold: true, italics: true, size: 18 }),
  new TextRun({ text: "diabetic retinopathy, microaneurysm detection, mathematical morphology, illumination correction, CLAHE, ablation study, FROC, IDRiD.", bold: true, size: 18 }),
] }));

// ---- I. Introduction (the six required axes as subsections)
add(H1("I. Introduction"));
add(H2("A. Importance"));
add(P("About 536.6 million adults (10.5 %) had diabetes in 2021, a number projected to reach 783.2 million by 2045 [1]. The pooled prevalence of DR among people with diabetes is 22.27 %, and adults with DR are projected to increase from 103.12 million in 2020 to 160.50 million in 2045 [2]. Early DR has no symptoms and is found only by retinal screening. MA, small saccular dilations of retinal capillaries seen as dark-red dots, are the first lesion of the international DR severity scale [3]. Screening produces more images than ophthalmologists can grade; a tool that marks suspected MA can prioritise images for review."));
add(H2("B. Background"));
add(P("The green channel is used because haemoglobin absorbs green light strongly. Uneven illumination is removed by estimating a smooth background [4], and contrast-limited adaptive histogram equalisation (CLAHE) raises local contrast while capping noise amplification [5]. MA are small, dark and round, so morphological operators are the classical candidate generators: a top-hat with oriented linear structuring elements removes vessels while keeping round lesions [6], oriented linear openings model vessels [7], and diameter closing with an automatic threshold extracts candidates [8]. Local contrast normalisation with local vessel detection [9], feature-based classification of candidates [10], and ensembles of candidate extractors [11] reduce false positives. The Retinopathy Online Challenge defined lesion-level FROC evaluation and the CPM [12]. The IDRiD dataset [13] and challenge [14] provide pixel-level masks at 4288 \u00d7 2848 px. Deep learning grades referable DR accurately but was developed on 128,175 labelled images [15], far more than the 54 annotated training images available here."));
add(H2("C. Problem Statement"));
add(P("We measured each difficulty on IDRiD. MA cover 0.10 % of image pixels; the smallest has an area of 13 px (about 4 px across) and the median 235 px. In the green channel MA are on average only 7.6 grey levels darker than their surroundings. Grey level varied from about 45 to 185 along a single image row, and JPEG 8 \u00d7 8 block edges were 2.6 times stronger than the texture inside a block. Vessels, vessel crossings and small haemorrhages are dark like MA. A single threshold therefore produces thousands of false positives per image, and any filter that removes noise risks flattening the smallest lesions."));
add(H2("D. Research Gap"));
add(P("Combining illumination normalisation, morphological candidate extraction, vessel suppression and feature classification is established [6], [8], [9], [11]; we do not claim the combination as new. Classical detectors were mostly evaluated on other images [12], and IDRiD results are dominated by deep networks ranked by pixel AUPR [14]. In the sources we reviewed we did not find a controlled ablation of classical steps on full-resolution IDRiD images that reports, for each added step, lesion-level sensitivity, false positives per image (FPI), CPM and processing time with all parameters fixed before one test run. The literature search was targeted, not systematic, so the gap is stated as \u201cnot found in the reviewed sources\u201d."));
add(H2("E. Evidence and Objectives"));
add(P("Green-channel analysis, background normalisation and morphological candidate extraction are standard in MA detection [6], [8]; vessel removal and local normalisation reduced false detections [9]; and supervised classification of candidate features separates MA from look-alikes [10], [11]. Our objectives were: (i) a reproducible Python pipeline that processes 12-megapixel images at full resolution on a CPU; (ii) every parameter chosen on training data by a rule stated before each experiment; (iii) an ablation measuring what each step adds in lesion sensitivity, FPI and CPM; (iv) a single, frozen test-set evaluation with confidence intervals and time; and (v) annotated outputs and a simple interface that mark suspected MA without stating a diagnosis."));
add(H2("F. Context and Aim"));
add(P("The Middle East and North Africa region had the highest age-adjusted diabetes prevalence of all IDF regions in 2021, 18.1 % [1]. In a population-based survey of people aged 50 and over in Sohag, Egypt, 20.9 % had diabetes, DR prevalence was 17.9 %, and up to 85.3 % had never had an eye examination [16]. In Alexandria and the north-west Delta, DR was found in 32.49 % of 10,811 gradable patients with diabetes [17]. Where fundus cameras are available but specialists and GPU hardware are scarce, a transparent method running on an ordinary PC fits the constraint. Our aim is a research prototype that assists graders; it does not diagnose."));

// ---- II. Materials and methods
add(H1("II. Materials and Methods"));
add(H2("A. Dataset and Protocol"));
add(P("We used the IDRiD segmentation subset [13]: 81 JPEG images of 4288 \u00d7 2848 px (54 training, 27 test) with pixel-level MA masks. Inspection found no unreadable file, no missing mask and no size mismatch; the training set contains 2412 MA and the test set 1085 (8-connected components). Every image contains MA, so image-level specificity cannot be measured on this subset. Each tuning experiment used only training images and stated its selection rule before running; the test images were processed once after all parameters were frozen. Fig. 1 summarises the pipeline and Fig. 2 shows the scale of the problem."));
add(figure("fig01_pipeline.png", "Processing pipeline and ablation models. Each model adds one technique to the previous one; all models share the field-of-view mask, the adaptive threshold and the connected-component analysis."));
add(figure("fig02_ma_example.png", "Scale of the problem (training image IDRiD_01). (a) Full image with the zoom region. (b) 300 \u00d7 300 px zoom with ground-truth MA outlined; the lesions are a few pixels across."));
add(H2("B. Preprocessing"));
add(P("A field-of-view (FOV) mask is obtained from the red channel (threshold 30, opening of radius 5 px, largest component, hole filling, erosion of 4 px). The green channel is then corrected for illumination with the multiplicative model"));
add(eq([R("I=L\u00b7R")]));
add(P0("where *L* is the slowly varying illumination and *R* the retinal reflectance. With a background estimate *B* computed as a masked Gaussian mean (kernel 151 px, \u03c3 = 37.75 px, estimated at quarter resolution because it is low-frequency), the corrected image is"));
add(eq([sub("I", "c"), R("="), frac([R("I")], [R("B")]), R("\u00b7"), sub("\u03bc", "FOV")]));
add(P0("Pixels outside the FOV are set to the FOV mean so that later morphology does not respond to the border. CLAHE [5] is applied with clip limit 1.0 and 8 \u00d7 5 tiles, followed by a Gaussian filter of \u03c3 = 1.5 px."));
add(H2("C. Candidate Generation"));
add(P("Every model thresholds a response map *S* after converting it to a robust z-score inside the FOV,"));
add(eq([R("z="), frac([R("S\u2212med(S)")], [R("1.4826\u00b7MAD(S)")])]));
add(P0("with the standard deviation used instead when the MAD collapses to zero, which happens for sparse top-hat maps. Models A to B+ use the darkness 255 \u2212 *I*. Model C uses the black-hat with a disc of radius *r* = 16 px,"));
add(eq([R("BH="), sub("\u03c6", "r"), R("(I)\u2212I")]));
add(P0("where \u03c6 is a morphological closing. Model D0 suppresses vessels with a line top-hat on the inverted image *J* = 255 \u2212 *I* [6], [7]:"));
add(eq([R("T=J\u2212"), new MathSubScript({ children: [R("max")], subScript: [R("\u03b8")] }), new MathSubScript({ children: [R("\u03b3")], subScript: [R("L,\u03b8")] }), R("(J)")]));
add(P0("where \u03b3 is an opening with a line of length *L* = 41 px at 12 orientations. A vessel contains a straight segment of length *L* in its own direction, survives the opening and is subtracted; a round MA fits no orientation and remains in *T*."));
add(H2("D. Thresholding, Components and Features"));
add(P("Pixels with *z* > *k* form candidates by 8-connected component analysis; components under 5 px are removed. The threshold *k* of each model was chosen to maximise lesion-level F1 on the training set. For Model D, D0 components at *k* = 3 (a pool holding 97.4 % of training MA) are described by 20 features: shape (area, perimeter, circularity, eccentricity, elongation, extent, bounding-box aspect), response (peak and mean *z*, peak black-hat, peak line top-hat and their ratio), intensity (inside, a 2\u20138 px ring, contrast, local contrast-to-noise ratio), colour (R, G, B normalised to the FOV mean) and distance to the FOV edge. A random forest (200 trees, balanced class weights, all positives and 25 % of negatives per training fold) scores each candidate; candidates with probability \u2265 0.84 are kept."));
add(H2("E. Evaluation"));
add(P("The primary evaluation is at lesion level: a candidate is a true positive if its centroid lies inside a ground-truth MA or within 5 px of it, and a lesion is detected if at least one candidate matches it. We report sensitivity, precision, F1, FPI, the FROC curve and the CPM, the mean sensitivity at 1/8, 1/4, 1/2, 1, 2, 4 and 8 FPI [12]. Pixel-level sensitivity, specificity, Dice, IoU and AUPR are reported inside the FOV; accuracy is not, because predicting \u201cno MA\u201d everywhere already exceeds 99.8 %. Image-level sensitivity is defined, specificity is not. 95 % confidence intervals come from 2000 bootstrap resamples of test images. Times are medians on a Windows 10 PC with Python 3.12 and OpenCV 5.1."));

// ---- III. Results
add(H1("III. Experiments and Results"));
add(H2("A. Design Experiments on the Training Set"));
add(P("Table I lists each decision with its evidence. The green channel gave the highest MA contrast-to-noise ratio in 53 of 54 images (Fig. 3). Illumination correction raised the global MA-versus-background AUC from 0.682 to 0.844 and equalised brightness across the retina (Figs. 4 and 5). Gaussian denoising removed the JPEG block grid while keeping 92 % of the depth of the smallest MA (Fig. 6). The line top-hat removed vessel responses that the black-hat kept (Fig. 7), and area, peak response and perimeter were the most important classifier features (Fig. 8)."));
add(figure("fig03_channel_cnr.png", "MA contrast-to-noise ratio per colour channel over the 54 training images (median printed). Green was best in 53 of 54 images."));
add(figure("fig04a_illumination_images.png", "Illumination correction on IDRiD_01: (a) green channel with dark periphery and bright optic disc; (b) corrected image."));
add(figure("fig04b_illumination_sweep.png", "Choice of the background estimator (training medians). Left: global AUC; right: retained MA contrast. Small kernels erase lesions, large ones miss the illumination; Gaussian 151 px (circled) had the highest AUC with \u2265 95 % contrast kept."));
add(figure("fig05_denoise_tradeoff.png", "Noise-reduction trade-off: per-lesion CNR against the contrast kept by the smallest quartile of MA (area \u2264 148 px). Filters left of the dotted line were rejected; Gaussian \u03c3 = 1.5 (circled) was chosen."));
add(figure("fig06_vessel_suppression.png", "Vessel suppression on a 400 \u00d7 400 px region: (a) preprocessed image, (b) black-hat response, high along vessels, (c) line top-hat response, where vessels vanish and the outlined MA remain."));
add(table("I", "Parameter Selection on the Training Set (54 Images)", ["Step", "Choice", "Evidence"], [
  ["Channel", "green", "CNR 0.67 vs 0.28 (R), 0.17 (B)"],
  ["FOV mask", "red > 30, erode 4", "area flat for T = 15\u201360; 100 % MA inside"],
  ["Illumination", "Gaussian 151 px", "AUC 0.682 \u2192 0.844 (53/54 images)"],
  ["CLAHE", "clip 1, 8 \u00d7 5", "AUC +0.003 (27/54 images)"],
  ["Denoising", "Gaussian \u03c3 = 1.5", "CNR 1.69 \u2192 2.58; blocks 2.45 \u2192 0.99"],
  ["Black-hat", "disc r = 16", "AUPR 0.011; multi-scale \u2264 0.006"],
  ["Vessels", "line top-hat L = 41", "AUPR 0.197 vs 0.079 combined"],
  ["Classifier", "random forest", "CPM 0.194 vs 0.015 (tree)"],
], [0.75, 1.1, 1.65]));
add(figure("fig07_feature_importance.png", "Ten most important candidate features of the random forest (impurity-based importance, trained on all training candidates)."));
add(H2("B. Ablation on the Test Set"));
add(P("Table II and Figs. 9 and 10 give the single test run. False positives per image fell from 3213 (A) to 15.3 (D). Illumination correction raised sensitivity at the F1 operating point from 0.149 to 0.749, but the threshold that maximised F1 also admitted more candidates. Denoising reduced FPI from 4367 to 903. The black-hat did not improve lesion-level results because it also responds to vessels. Vessel suppression lowered FPI from 568 to 60, and the classifier lowered it to 15.3 while raising CPM from 0.038 to 0.228. Model D reached 0.067, 0.087, 0.129, 0.194, 0.282, 0.366 and 0.471 sensitivity at 1/8 to 8 FPI. Its 95 % intervals were 0.489\u20130.604 for sensitivity, 0.525\u20130.602 for F1 and 12.5\u201318.7 for FPI. Training (out-of-fold) and test CPM were 0.194 and 0.228, so the frozen parameters did not overfit."));
add(table("II", "Ablation Results on the Test Set (27 Images, 1085 MA)", ["Model", "CPM", "Sens.", "Prec.", "F1", "FPI", "Dice", "AUPR", "Time (s)"], [
  ["A", "0.000", "0.149", "0.005", "0.009", "3213", "0.005", "0.002", "0.24"],
  ["A+", "0.000", "0.749", "0.015", "0.029", "4544", "0.017", "0.007", "0.44"],
  ["B", "0.000", "0.715", "0.015", "0.029", "4367", "0.020", "0.007", "0.64"],
  ["B+", "0.000", "0.501", "0.025", "0.047", "903", "0.018", "0.008", "0.65"],
  ["C", "0.001", "0.322", "0.024", "0.045", "568", "0.024", "0.010", "0.97"],
  ["D0", "0.038", "0.503", "0.280", "0.360", "59.8", "0.202", "0.169", "0.94"],
  ["D", "0.228", "0.547", "0.589", "0.567", "15.3", "0.448", "0.326", "5.46"],
], [0.42, 0.38, 0.37, 0.37, 0.37, 0.4, 0.37, 0.4, 0.42], 6));
add(figure("fig08_froc_test.png", "Lesion-level FROC on the test set. The grey band marks the CPM range (1/8 to 8 false positives per image); only the vessel-suppressed models reach it, and Model D dominates at every operating point."));
add(figure("fig09_fpi_ablation.png", "False positives per image at each model\u2019s operating point (log scale, 95 % bootstrap intervals). The largest reductions come from vessel suppression (C \u2192 D0) and feature-based filtering (D0 \u2192 D)."));
add(H2("C. Qualitative Result"));
add(P("Fig. 11 shows Model D on an unseen test image. Many missed lesions lie near hard exudates or in dark regions; many false alarms are small dark spots that resemble MA, including small haemorrhages, which count as false positives because only MA are scored."));
add(figure("fig10_annotated_test.png", "Model D on test image IDRiD_55 (crop): cyan circles are correct detections, magenta circles false alarms, yellow squares ground-truth MA that were missed."));

// ---- IV. Discussion
add(H1("IV. Discussion"));
add(H2("A. Contribution of Each Step"));
add(P("Two steps did most of the work: vessel suppression and feature-based filtering. CLAHE added almost nothing once illumination was corrected; applied to the raw green channel it alone reached an AUC of 0.837, which shows that both steps equalise brightness locally and are largely redundant. Denoising mattered mainly because the IDRiD images carry JPEG block artefacts of a size comparable to MA contrast."));
add(H2("B. Negative Results"));
add(P("A multi-scale black-hat, taken as the maximum of per-scale z-scores, never beat a single 16 px disc: for a disc the closing grows with the radius, so the largest scale already contains the smaller ones, and normalising each scale separately amplified fine-scale noise. The pixel-wise minimum of black-hat and line top-hat (training AUPR 0.079) was worse than the line top-hat alone (0.197). A depth-4 decision tree reached a CPM of only 0.015 against 0.194 for the random forest, so interpretability through a few rules cost most of the gain."));
add(H2("C. Comparison with Published Results"));
add(P("On the same 27 test images, leading IDRiD challenge entries for MA segmentation, all deep networks, reached a pixel AUPR of about 0.5 [14]; Model D reaches 0.326 on a CPU without deep learning. The comparison is indicative only, because our output is a set of scored candidate blobs rather than a pixel segmentation optimised for that metric. CPM values from the ROC benchmark [12] are not directly comparable because images, resolution and annotations differ."));
add(H2("D. Limitations"));
add(bullet("The data are small (54 + 27 images) and come from one dataset and one annotation protocol; the confidence intervals reflect this."));
add(bullet("No image is free of MA, so image-level specificity was not measured; the IDRiD disease-grading subset contains images graded as no DR."));
add(bullet("Ground-truth outlines are often larger than the visible dark core, which lowers pixel Dice and IoU for any method."));
add(bullet("Sizes in pixels (disc 16, line 41, Gaussian 151) were tuned for 4288 \u00d7 2848 images and must be rescaled for other cameras."));
add(bullet("Two implementation errors (a collapsed robust scale and a FROC threshold grid that skipped low FPI) were found and fixed on the training set before the test run."));

// ---- V. Conclusion
add(H1("V. Conclusion"));
add(P("A transparent, CPU-only pipeline found 54.7 % of microaneurysms on unseen IDRiD images at 15.3 false positives per image (CPM 0.228) in about 5.5 s per 12-megapixel image. The ablation shows that vessel suppression and feature-based false-positive removal account for almost all of the gain. Future work will measure specificity on images without DR, add features that separate MA from small haemorrhages and vessel crossings, test a second camera with resolution-scaled parameters, and compare with a small convolutional network under the same lesion-level protocol."));

// ---- References (IEEE numeric, in order of first citation)
add(H1("References"));
const refs = [
  "H. Sun et al., \u201cIDF Diabetes Atlas: Global, regional and country-level diabetes prevalence estimates for 2021 and projections for 2045,\u201d *Diabetes Res. Clin. Pract.*, vol. 183, Art. no. 109119, 2022, doi: 10.1016/j.diabres.2021.109119.",
  "Z. L. Teo et al., \u201cGlobal prevalence of diabetic retinopathy and projection of burden through 2045: Systematic review and meta-analysis,\u201d *Ophthalmology*, vol. 128, no. 11, pp. 1580\u20131591, 2021, doi: 10.1016/j.ophtha.2021.04.027.",
  "C. P. Wilkinson et al., \u201cProposed international clinical diabetic retinopathy and diabetic macular edema disease severity scales,\u201d *Ophthalmology*, vol. 110, no. 9, pp. 1677\u20131682, 2003, doi: 10.1016/S0161-6420(03)00475-5.",
  "M. Foracchia, E. Grisan, and A. Ruggeri, \u201cLuminosity and contrast normalization in retinal images,\u201d *Med. Image Anal.*, vol. 9, no. 3, pp. 179\u2013190, 2005, doi: 10.1016/j.media.2004.07.001.",
  "K. Zuiderveld, \u201cContrast limited adaptive histogram equalization,\u201d in *Graphics Gems IV*, P. S. Heckbert, Ed. San Diego, CA, USA: Academic Press, 1994, pp. 474\u2013485.",
  "T. Spencer, J. A. Olson, K. C. McHardy, P. F. Sharp, and J. V. Forrester, \u201cAn image-processing strategy for the segmentation and quantification of microaneurysms in fluorescein angiograms of the ocular fundus,\u201d *Comput. Biomed. Res.*, vol. 29, no. 4, pp. 284\u2013302, 1996.",
  "F. Zana and J.-C. Klein, \u201cSegmentation of vessel-like patterns using mathematical morphology and curvature evaluation,\u201d *IEEE Trans. Image Process.*, vol. 10, no. 7, pp. 1010\u20131019, 2001, doi: 10.1109/83.931095.",
  "T. Walter, P. Massin, A. Erginay, R. Ordonez, C. Jeulin, and J.-C. Klein, \u201cAutomatic detection of microaneurysms in color fundus images,\u201d *Med. Image Anal.*, vol. 11, no. 6, pp. 555\u2013566, 2007, doi: 10.1016/j.media.2007.05.001.",
  "A. D. Fleming, S. Philip, K. A. Goatman, J. A. Olson, and P. F. Sharp, \u201cAutomated microaneurysm detection using local contrast normalization and local vessel detection,\u201d *IEEE Trans. Med. Imag.*, vol. 25, no. 9, pp. 1223\u20131232, 2006, doi: 10.1109/TMI.2006.879953.",
  "M. Niemeijer, B. van Ginneken, J. Staal, M. S. A. Suttorp-Schulten, and M. D. Abr\u00e0moff, \u201cAutomatic detection of red lesions in digital color fundus photographs,\u201d *IEEE Trans. Med. Imag.*, vol. 24, no. 5, pp. 584\u2013592, 2005, doi: 10.1109/TMI.2005.843738.",
  "B. Antal and A. Hajdu, \u201cAn ensemble-based system for microaneurysm detection and diabetic retinopathy grading,\u201d *IEEE Trans. Biomed. Eng.*, vol. 59, no. 6, pp. 1720\u20131726, 2012, doi: 10.1109/TBME.2012.2193126.",
  "M. Niemeijer et al., \u201cRetinopathy online challenge: Automatic detection of microaneurysms in digital color fundus photographs,\u201d *IEEE Trans. Med. Imag.*, vol. 29, no. 1, pp. 185\u2013195, 2010, doi: 10.1109/TMI.2009.2033909.",
  "P. Porwal et al., \u201cIndian Diabetic Retinopathy Image Dataset (IDRiD): A database for diabetic retinopathy screening research,\u201d *Data*, vol. 3, no. 3, Art. no. 25, 2018, doi: 10.3390/data3030025.",
  "P. Porwal et al., \u201cIDRiD: Diabetic retinopathy \u2013 segmentation and grading challenge,\u201d *Med. Image Anal.*, vol. 59, Art. no. 101561, 2020, doi: 10.1016/j.media.2019.101561.",
  "V. Gulshan et al., \u201cDevelopment and validation of a deep learning algorithm for detection of diabetic retinopathy in retinal fundus photographs,\u201d *JAMA*, vol. 316, no. 22, pp. 2402\u20132410, 2016, doi: 10.1001/jama.2016.17216.",
  "H. AlSawahli et al., \u201cPopulation-based cross-sectional prevalence survey of diabetes and diabetic retinopathy in Sohag, Egypt,\u201d *BMJ Open*, vol. 11, no. 6, Art. no. e047757, 2021, doi: 10.1136/bmjopen-2020-047757.",
  "A. Elmassry, I. S. H. Ahmed, N. Adly, and M. Torki, \u201cPrevalence of diabetic retinopathy in patients with diabetes in Alexandria and North-West Delta, Egypt,\u201d *Int. Ophthalmol.*, vol. 43, no. 8, pp. 2883\u20132895, 2023, doi: 10.1007/s10792-023-02692-4.",
];
refs.forEach((r, i) => add(new Paragraph({ alignment: AlignmentType.JUSTIFIED, indent: { left: 0.3 * IN, hanging: 0.3 * IN }, spacing: { after: 40 },
  tabStops: [{ type: TabStopType.LEFT, position: 0.3 * IN }],
  children: [new TextRun({ text: `[${i + 1}]\t`, size: 16 }), ...runs(r, { size: 16 })] })));

// ================================================================== document
const doc = new Document({
  creator: "Biomedical Engineering student", title: "Microaneurysm Detection - IEEE report",
  styles: {
    default: {
      document: { run: { font: FONT, size: 20 }, paragraph: { spacing: { line: 240, after: 0 } } },
      heading1: { run: { font: FONT, size: 20, bold: false, color: "000000" }, paragraph: { spacing: { before: 200, after: 100 }, keepNext: true } },
      heading2: { run: { font: FONT, size: 20, italics: true, color: "000000" }, paragraph: { spacing: { before: 120, after: 60 }, keepNext: true } },
    },
  },
  numbering: { config: [{ reference: "dots", levels: [{ level: 0, format: "bullet", text: "\u2022", alignment: AlignmentType.LEFT }] }] },
  sections: [
    { properties: { page: PAGE }, children: titleBlock },
    { properties: { type: SectionType.CONTINUOUS, page: PAGE, column: { count: 2, space: 0.25 * IN } }, children: body },
  ],
});

Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(OUT, buf); console.log("written", OUT, buf.length, "bytes; figures", figNo, "equations", eqNo); });
