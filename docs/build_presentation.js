// Builds the project presentation (structured deck: theme + layouts + sections).
const path = require("path");
const fs = require("fs");
const pptxgen = require("pptxgenjs");
// apply_theme.js comes from the pptx skill used to build the deck; set APPLY_THEME_JS to its path.
const { applyTheme } = require(process.env.APPLY_THEME_JS || "./apply_theme.js");

const ROOT = path.resolve(__dirname, "..");
const A = (p) => path.join(ROOT, p);
const OUT = A("docs/Microaneurysm_Detection_Presentation.pptx");

const THEME = {
  name: "Fundus",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "1B2433", lt1: "FFFFFF", dk2: "2E3A4F", lt2: "EEF1F5",
    accent1: "C8502A", accent2: "E39B2D", accent3: "2A7F8F", accent4: "6B7A8F",
    accent5: "A9B6C6", accent6: "F4E3D7", hlink: "2A7F8F", folHlink: "6B7A8F",
  },
};
const HEX = THEME.colors;

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5 in
pres.title = "Early Detection of Diabetic Retinopathy: Microaneurysm Detection";
pres.author = "Biomedical Engineering student";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;
const W = 13.333;

// ------------------------------------------------------------------ layouts
pres.defineSlideMaster({
  title: "DARK", background: { color: C.text1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.7, y: 2.0, w: 7.4, h: 1.9, fontSize: 40, bold: true, color: C.background1, valign: "bottom" }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.7, y: 4.05, w: 7.4, h: 1.9, fontSize: 18, color: C.accent5, valign: "top" }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT", background: { color: C.background1 },
  margin: [0.5, 0.6, 0.6, 0.6],
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.35, w: W - 1.2, h: 0.95, fontSize: 30, bold: true, color: C.text1, valign: "middle" }, text: "" } },
  ],
  slideNumber: { x: W - 1.0, y: 7.0, w: 0.5, h: 0.3, fontSize: 10, color: C.accent4 },
});

// ------------------------------------------------------------------ helpers
const card = (s, x, y, w, h, fill, name) =>
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.12, fill: { color: fill }, line: { color: fill }, objectName: name });
const text = (s, t, opt) => s.addText(t, { isTextBox: true, margin: 0, fontSize: 15, color: C.text2, valign: "top", ...opt });
const source = (s, t) => text(s, t, { x: 0.6, y: 6.95, w: 11.2, h: 0.3, fontSize: 10, color: C.accent4, objectName: "source" });
const stat = (s, x, y, w, big, label, colour, name) => {
  card(s, x, y, w, 2.3, C.background2, name + "-card");
  text(s, big, { x: x + 0.3, y: y + 0.25, w: w - 0.6, h: 1.0, fontSize: 44, bold: true, color: colour, fontFace: THEME.headFontFace, objectName: name + "-value" });
  text(s, label, { x: x + 0.3, y: y + 1.3, w: w - 0.6, h: 0.9, fontSize: 15, color: C.text2, objectName: name + "-label" });
};
const content = (section, title) => {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: section });
  s.addText(title, { placeholder: "title" });
  return s;
};

// ================================================================== 1 Title
pres.addSection({ title: "Introduction" });
let s = pres.addSlide({ masterName: "DARK", sectionTitle: "Introduction" });
s.addText("Early Detection of Diabetic Retinopathy", { placeholder: "title" });
s.addText([
  { text: "Microaneurysm detection with traditional digital image processing", options: { breakLine: true } },
  { text: "Biomedical Engineering \u00b7 Digital Image Processing \u00b7 IDRiD dataset", options: { fontSize: 14, breakLine: true } },
  { text: "Research prototype \u2014 not a diagnostic device", options: { fontSize: 14, italic: true, color: C.accent2 } },
], { placeholder: "body" });
s.addImage({ path: A("docs/presentation_assets/fundus_box.png"), x: 8.55, y: 1.55, w: 4.2, h: 2.79, objectName: "fundus" });
text(s, "Yellow box: region zoomed on slide 3 (IDRiD_01, training set)", { x: 8.55, y: 4.45, w: 4.2, h: 0.4, fontSize: 11, color: C.accent5 });
s.addNotes("السلام عليكم. المشروع عن الكشف المبكر عن اعتلال الشبكية السكري، وبالتحديد اكتشاف الـ Microaneurysms، ودي أول علامة بتظهر في صور قاع العين. استخدمنا Traditional Digital Image Processing من غير Deep Learning، على داتا IDRiD. وده نموذج بحثي مساعد، مش جهاز تشخيص.");

// ================================================================== 2 Importance
s = content("Introduction", "Diabetic retinopathy is common and silent until late");
stat(s, 0.6, 1.6, 3.85, "536.6 M", "adults with diabetes worldwide in 2021, rising to 783.2 M by 2045", C.accent1, "stat-diabetes");
stat(s, 4.74, 1.6, 3.85, "22 %", "of people with diabetes have retinopathy (pooled 22.27 %)", C.accent1, "stat-dr");
stat(s, 8.88, 1.6, 3.85, "85 %", "of people aged 50+ in Sohag, Egypt, had never had an eye examination", C.accent3, "stat-egypt");
card(s, 0.6, 4.3, 12.13, 1.75, C.accent6, "message");
text(s, [
  { text: "Microaneurysms (MA) are the earliest visible sign. ", options: { bold: true, color: C.text1 } },
  { text: "Early DR has no symptoms, so it is found only by screening fundus photographs. A tool that marks suspected MA can help graders where ophthalmologists are few." },
], { x: 0.95, y: 4.55, w: 11.4, h: 1.3, fontSize: 17 });
source(s, "Sources: Sun et al. (2022), IDF Diabetes Atlas; Teo et al. (2021), Ophthalmology; AlSawahli et al. (2021), BMJ Open.");
s.addNotes("المرض منتشر جدا: أكتر من 500 مليون عندهم سكر، وحوالي واحد من كل خمسة عنده اعتلال شبكية. وفي سوهاج 85% من اللي فوق الخمسين عمرهم ما كشفوا على عينهم. المرض في الأول مالوش أعراض، فالحل هو الـ screening، وده اللي المشروع بيساعد فيه.");

// ================================================================== 3 Problem
s = content("Introduction", "A microaneurysm is a few dark pixels in a 12-megapixel image");
s.addImage({ path: A("docs/presentation_assets/ma_zoom.png"), x: 0.6, y: 1.5, w: 4.9, h: 4.9, objectName: "ma-zoom" });
text(s, "300 \u00d7 300 px region, ground-truth MA outlined in yellow", { x: 0.6, y: 6.45, w: 4.9, h: 0.35, fontSize: 11, color: C.accent4 });
const problems = [
  ["0.10 %", "of image pixels are MA: accuracy is meaningless"],
  ["13 px", "smallest MA area (about 4 px across); median 235 px"],
  ["7.6", "grey levels: average MA contrast in the green channel"],
  ["45\u2013185", "grey-level range along one row: uneven illumination"],
  ["2.6\u00d7", "JPEG 8\u00d78 block edges stronger than normal texture"],
];
problems.forEach(([v, l], i) => {
  const y = 1.55 + i * 1.02;
  text(s, v, { x: 6.0, y, w: 2.2, h: 0.8, fontSize: 30, bold: true, color: C.accent1, fontFace: THEME.headFontFace, valign: "middle", objectName: `problem-${i}-value` });
  text(s, l, { x: 8.3, y, w: 4.4, h: 0.8, fontSize: 16, valign: "middle", objectName: `problem-${i}-label` });
});
source(s, "All numbers measured on the IDRiD segmentation subset in this project (dataset inspection, Experiments 1\u20135).");
s.addNotes("الـ MA صغيرة جدا: أصغر واحدة 13 بكسل بس في صورة 12 ميجا. وهي أقل من واحد على ألف من البكسلات، علشان كده الـ Accuracy ملهاش معنى. وكمان الفرق بينها وبين الخلفية حوالي 7 درجات رمادي بس، والإضاءة مش منتظمة، وفيه مربعات JPEG في الصور.");

// ================================================================== 4 Gap + aim
s = content("Introduction", "The gap: no step-by-step evidence on IDRiD at lesion level");
const gapCards = [
  ["Already known", "Illumination correction, morphology, vessel suppression and feature classifiers are established (Spencer 1996; Fleming 2006; Walter 2007; Antal & Hajdu 2012). We do not claim the combination is new.", C.background2],
  ["Not found in reviewed sources", "A controlled ablation on IDRiD's 4288 \u00d7 2848 images reporting, for every added step, lesion sensitivity, false positives per image, CPM and time.", C.background2],
  ["Aim", "Build a transparent, CPU-only pipeline that marks suspected MA, and measure what each step contributes \u2014 with all parameters fixed on training data before one test run.", C.accent6],
];
gapCards.forEach(([h, b, f], i) => {
  const x = 0.6 + i * 4.14;
  card(s, x, 1.6, 3.85, 4.6, f, `gap-${i}`);
  text(s, h, { x: x + 0.3, y: 1.85, w: 3.25, h: 0.5, fontSize: 20, bold: true, color: i === 2 ? C.accent1 : C.text1, fontFace: THEME.headFontFace });
  text(s, b, { x: x + 0.3, y: 2.5, w: 3.25, h: 3.5, fontSize: 16 });
});
source(s, "Literature search was targeted, not systematic: the gap means \u201cnot found in the reviewed sources\u201d.");
s.addNotes("الفكرة نفسها مش جديدة، وده مهم نقوله بأمانة. الجديد إننا قسنا مساهمة كل خطوة لوحدها على IDRiD بالدقة الكاملة وبتقييم على مستوى الآفة. والهدف إننا نعمل نظام بسيط على CPU ونقيس كل خطوة بتضيف إيه.");

// ================================================================== 5 Data
pres.addSection({ title: "Method" });
s = content("Method", "Data: tune on 54 images, test once on 27");
stat(s, 0.6, 1.6, 2.85, "81", "IDRiD images with pixel-level MA masks", C.accent1, "stat-images");
stat(s, 3.7, 1.6, 2.85, "12 MP", "4288 \u00d7 2848 px, processed at full resolution", C.accent1, "stat-res");
stat(s, 6.8, 1.6, 2.85, "2412", "MA in the 54 training images", C.accent3, "stat-train");
stat(s, 9.9, 1.6, 2.85, "1085", "MA in the 27 test images", C.accent3, "stat-test");
const steps = ["Inspect data: 0 read errors, 0 missing masks", "Fix a selection rule before each experiment", "Tune on training images only", "Freeze every parameter", "Run the test set once"];
steps.forEach((t, i) => {
  const x = 0.6 + i * 2.48;
  s.addShape(pres.shapes.OVAL, { x: x, y: 4.45, w: 0.55, h: 0.55, fill: { color: C.accent1 }, line: { color: C.accent1 }, objectName: `step-${i}-dot` });
  text(s, String(i + 1), { x: x, y: 4.45, w: 0.55, h: 0.55, fontSize: 16, bold: true, color: C.background1, align: "center", valign: "middle" });
  text(s, t, { x: x, y: 5.15, w: 2.2, h: 1.1, fontSize: 15 });
});
source(s, "Dataset: Porwal et al. (2018), IDRiD, Data 3(3), 25. Every image contains MA, so image-level specificity cannot be measured on this subset.");
s.addNotes("استخدمنا جزء الـ Segmentation من IDRiD: 54 صورة تدريب و27 اختبار. كل الـ parameters اتظبطت على التدريب بس، وكل تجربة قاعدة الاختيار فيها اتكتبت قبل ما نشغلها. والاختبار اتشغل مرة واحدة في الآخر.");

// ================================================================== 6 Pipeline
s = content("Method", "Seven models: each one adds exactly one technique");
const models = [
  ["A", "Green channel + adaptive threshold"], ["A+", "Illumination correction"], ["B", "CLAHE"],
  ["B+", "Gaussian denoising"], ["C", "Black-hat (morphology)"], ["D0", "Vessel suppression (line top-hat)"], ["D", "Feature-based FP removal"],
];
models.forEach(([m, d], i) => {
  const x = 0.6 + i * 1.75;
  const last = i === models.length - 1;
  card(s, x, 1.75, 1.55, 2.55, last ? C.accent1 : C.background2, `model-${i}`);
  text(s, m, { x: x, y: 1.95, w: 1.55, h: 0.7, fontSize: 28, bold: true, align: "center", color: last ? C.background1 : C.accent1, fontFace: THEME.headFontFace });
  text(s, d, { x: x + 0.12, y: 2.75, w: 1.31, h: 1.45, fontSize: 14, align: "center", color: last ? C.background1 : C.text2 });
  if (!last) s.addShape(pres.shapes.RIGHT_TRIANGLE, { x: x + 1.58, y: 2.9, w: 0.14, h: 0.25, rotate: 90, fill: { color: C.accent4 }, line: { color: C.accent4 }, objectName: `arrow-${i}` });
});
card(s, 0.6, 4.75, 12.13, 1.15, C.background2, "shared");
text(s, [
  { text: "Shared by all models  ", options: { bold: true, color: C.text1 } },
  { text: "field-of-view mask \u00b7 robust z-score of the response \u00b7 adaptive threshold z > k \u00b7 connected components (8-connectivity, < 5 px removed) \u00b7 the same lesion-level evaluation" },
], { x: 0.9, y: 4.98, w: 11.5, h: 0.8, fontSize: 16 });
s.addNotes("ده قلب الـ Ablation Study: سبع موديلات، كل واحد بيزود تقنية واحدة بس على اللي قبله. ده بيخلينا نعرف كل خطوة ساهمت بإيه. وكلهم بيستخدموا نفس الـ threshold ونفس طريقة التقييم علشان المقارنة تبقى عادلة.");

// ================================================================== 7 Preprocessing evidence
s = content("Method", "Illumination correction made one threshold work everywhere");
s.addImage({ path: A("docs/presentation_assets/illumination.png"), x: 0.6, y: 1.5, w: 7.6, h: 2.79, objectName: "illumination" });
const pre = [
  ["0.68 \u2192 0.84", "MA-vs-background AUC after illumination correction (53 of 54 images better)"],
  ["+0.003", "AUC added by CLAHE once illumination was corrected"],
  ["1.69 \u2192 2.58", "per-lesion contrast-to-noise after Gaussian \u03c3 = 1.5; JPEG blocks removed"],
];
pre.forEach(([v, l], i) => {
  const y = 1.55 + i * 1.6;
  text(s, v, { x: 8.6, y, w: 4.1, h: 0.65, fontSize: 28, bold: true, color: C.accent1, fontFace: THEME.headFontFace });
  text(s, l, { x: 8.6, y: y + 0.65, w: 4.1, h: 0.85, fontSize: 15 });
});
text(s, "Divisive correction I / B \u00d7 mean, background B = masked Gaussian (151 px). Every choice was made by a rule on the training set.", { x: 0.6, y: 4.6, w: 7.6, h: 0.9, fontSize: 15 });
source(s, "Training set, Experiments 3\u20135. CLAHE and illumination correction both equalise brightness locally, so they are largely redundant.");
s.addNotes("تصحيح الإضاءة كان أهم خطوة في الـ preprocessing: خلى threshold واحد ينفع على الشبكية كلها. الـ CLAHE بعده تقريبا ما ضافش حاجة، ودي نتيجة مهمة. والـ Gaussian شال مربعات الـ JPEG وحافظ على 92% من عمق أصغر الآفات.");

// ================================================================== 8 Vessel suppression
s = content("Method", "The line top-hat removes vessels that the black-hat keeps");
s.addImage({ path: A("docs/presentation_assets/vessel_suppression.png"), x: 0.6, y: 1.45, w: 12.13, h: 4.08, objectName: "vessels" });
text(s, [
  { text: "Pixel AUPR (training): ", options: { bold: true, color: C.text1 } },
  { text: "black-hat 0.011 \u2192 line top-hat 0.197. " },
  { text: "Why it works: ", options: { bold: true, color: C.text1 } },
  { text: "a vessel contains a 41 px straight segment in its own direction, so the opening keeps it and the subtraction removes it; a round MA fits no orientation and survives." },
], { x: 0.6, y: 5.8, w: 12.13, h: 1.0, fontSize: 15 });
s.addNotes("الـ Black-hat بيطلع أي حاجة غامقة ضيقة، فبيطلع الأوعية الدموية كمان. الحل كان Line top-hat: خط طوله 41 بكسل في 12 اتجاه. الوعاء الدموي بيحتوي على خط مستقيم فبيتشال، لكن الـ MA مدورة فبتفضل. شوف الصورة: الأوعية اختفت والـ MA اللي في الدايرة الخضرا لسه ظاهرة.");

// ================================================================== 9 Classifier
s = content("Method", "A random forest on 20 features removes most false alarms");
const imp = fs.readFileSync(A("outputs/experiment_08/feature_importances.csv"), "utf8").trim().split(/\r?\n/).slice(1)
  .map((r) => r.split(",")).map(([n, v]) => [n, parseFloat(v)]).sort((a, b) => b[1] - a[1]).slice(0, 8).reverse();
s.addChart(pres.charts.BAR, [{ name: "Importance", labels: imp.map((r) => r[0]), values: imp.map((r) => +r[1].toFixed(3)) }], {
  x: 0.6, y: 1.5, w: 6.6, h: 5.0, barDir: "bar", chartColors: [HEX.accent1],
  showTitle: true, title: "Top 8 features (impurity importance)", titleFontSize: 14, titleColor: HEX.dk1, titleFontFace: "+mn-lt",
  showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 11, dataLabelColor: HEX.dk2, dataLabelFontFace: "+mn-lt", dataLabelFormatCode: "0.000",
  catAxisLabelColor: HEX.dk2, catAxisLabelFontSize: 12, catAxisLabelFontFace: "+mn-lt",
  valAxisHidden: true, valGridLine: { style: "none" }, catGridLine: { style: "none" }, showLegend: false, barGapWidthPct: 60,
  objectName: "importance-chart",
});
const clf = [
  ["Shape", "area, perimeter, circularity, eccentricity, elongation"],
  ["Response", "peak and mean z, black-hat and line top-hat peaks"],
  ["Intensity and colour", "inside vs 2\u20138 px ring, local contrast-to-noise, R G B"],
  ["Result", "out-of-fold CPM 0.194 (random forest) vs 0.015 (depth-4 tree)"],
];
clf.forEach(([h, b], i) => {
  const y = 1.6 + i * 1.22;
  text(s, h, { x: 7.7, y, w: 5.0, h: 0.45, fontSize: 18, bold: true, color: i === 3 ? C.accent1 : C.text1, fontFace: THEME.headFontFace });
  text(s, b, { x: 7.7, y: y + 0.47, w: 5.0, h: 0.7, fontSize: 15 });
});
source(s, "Candidates: line top-hat components at z > 3 (97.4 % of training MA in the pool). Scores from 5-fold cross-validation grouped by image.");
s.addNotes("بعد كده كل candidate بنحسبله 20 خاصية: شكله، قوة الاستجابة، الإضاءة واللون. والـ Random Forest بيقرر هل ده MA ولا لأ. أهم خاصية كانت المساحة. وجربنا Decision Tree بسيطة بس كانت أضعف بكتير.");

// ================================================================== 10 Evaluation
s = content("Method", "Evaluation at three levels; accuracy is not used");
const levels = [
  ["Lesion level (primary)", "A detection is correct if its centroid lies inside a ground-truth MA or within 5 px. Sensitivity, precision, F1, false positives per image, FROC and CPM (mean sensitivity at 1/8 \u2026 8 FP per image)."],
  ["Pixel level", "Sensitivity, specificity, Dice, IoU and AUPR inside the field of view. Accuracy is excluded: predicting \u201cno MA\u201d everywhere already scores above 99.8 %."],
  ["Image level", "Sensitivity only. Every IDRiD segmentation image contains MA, so specificity needs healthy images from another subset."],
];
levels.forEach(([h, b], i) => {
  const x = 0.6 + i * 4.14;
  card(s, x, 1.6, 3.85, 4.4, i === 0 ? C.accent6 : C.background2, `level-${i}`);
  text(s, h, { x: x + 0.3, y: 1.85, w: 3.25, h: 0.55, fontSize: 20, bold: true, color: i === 0 ? C.accent1 : C.text1, fontFace: THEME.headFontFace });
  text(s, b, { x: x + 0.3, y: 2.55, w: 3.25, h: 3.3, fontSize: 16 });
});
source(s, "FROC and CPM: Niemeijer et al. (2010), Retinopathy Online Challenge. 95 % confidence intervals by bootstrap over test images.");
s.addNotes("التقييم الأساسي على مستوى الآفة، يعني كل MA بتتحسب لوحدها. وبنستخدم FROC وCPM زي مسابقة Retinopathy Online Challenge. ومستخدمناش Accuracy لأن نظام بيقول مفيش MA خالص هياخد أكتر من 99.8%.");

// ================================================================== 11 Ablation chart
pres.addSection({ title: "Results" });
s = content("Results", "False positives fell from 3,213 to 15 per image");
const fpi = [["A", 3213], ["A+", 4544], ["B", 4367], ["B+", 903], ["C", 568], ["D0", 60], ["D", 15.3]];
s.addChart(pres.charts.BAR, [{ name: "False positives per image", labels: fpi.map((r) => r[0]), values: fpi.map((r) => r[1]) }], {
  x: 0.6, y: 1.5, w: 7.9, h: 5.2, barDir: "col",
  chartColors: [HEX.accent5, HEX.accent5, HEX.accent5, HEX.accent5, HEX.accent5, HEX.accent4, HEX.accent1],
  showTitle: true, title: "False positives per image, test set (log scale)", titleFontSize: 14, titleColor: HEX.dk1, titleFontFace: "+mn-lt",
  showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 12, dataLabelColor: HEX.dk1, dataLabelFontFace: "+mn-lt", dataLabelFormatCode: "General",
  valAxisLogScaleBase: 10, valAxisMinVal: 10, valAxisMaxVal: 10000, valAxisLabelColor: HEX.accent4, valAxisLabelFontSize: 11, valAxisLabelFontFace: "+mn-lt",
  catAxisLabelColor: HEX.dk2, catAxisLabelFontSize: 14, catAxisLabelFontFace: "+mn-lt",
  valGridLine: { color: "DDE2E8", size: 0.5 }, catGridLine: { style: "none" }, showLegend: false, barGapWidthPct: 50,
  objectName: "fpi-chart",
});
const res = [
  ["568 \u2192 60", "vessel suppression (C \u2192 D0)"],
  ["60 \u2192 15", "feature-based filtering (D0 \u2192 D)"],
  ["0.038 \u2192 0.228", "CPM gained by the classifier"],
];
res.forEach(([v, l], i) => {
  const y = 1.7 + i * 1.55;
  text(s, v, { x: 8.9, y, w: 3.8, h: 0.7, fontSize: 30, bold: true, color: C.accent1, fontFace: THEME.headFontFace });
  text(s, l, { x: 8.9, y: y + 0.7, w: 3.8, h: 0.6, fontSize: 16 });
});
source(s, "IDRiD test set, 27 images, 1085 MA; each model at the threshold that maximised lesion F1 on the training set.");
s.addNotes("دي أهم نتيجة: الـ False positives نزلت من أكتر من 3000 في الصورة لـ 15 بس. أكبر خطوتين: Vessel suppression نزلها من 568 لـ 60، والـ classifier نزلها لـ 15. لاحظ إن تصحيح الإضاءة زود العدد شوية لأنه زود الـ sensitivity جامد عند الـ threshold اللي اتختار.");

// ================================================================== 12 Final result
s = content("Results", "Final system finds 55 % of MA at 15 false positives/image");
s.addImage({ path: A("outputs/experiment_09/fig_froc_test.png"), x: 0.6, y: 1.45, w: 7.35, h: 4.64, objectName: "froc" });
const fin = [["0.228", "CPM (95 % CI 0.195\u20130.267)"], ["54.7 %", "lesion sensitivity, precision 58.9 %"], ["0.33", "pixel AUPR; Dice 0.45"], ["5.5 s", "per 12-MP image on a CPU"]];
fin.forEach(([v, l], i) => {
  const y = 1.5 + i * 1.18;
  card(s, 8.3, y, 4.43, 1.0, i === 0 ? C.accent6 : C.background2, `final-${i}`);
  text(s, v, { x: 8.5, y: y + 0.12, w: 1.65, h: 0.75, fontSize: 26, bold: true, color: C.accent1, fontFace: THEME.headFontFace, valign: "middle" });
  text(s, l, { x: 10.2, y: y + 0.12, w: 2.4, h: 0.75, fontSize: 14, valign: "middle" });
});
text(s, "Training (out-of-fold) CPM 0.194 vs test 0.228: no sign of overfitting.", { x: 0.6, y: 6.25, w: 7.35, h: 0.5, fontSize: 15 });
source(s, "Leading IDRiD challenge entries for MA (deep networks) reached pixel AUPR of about 0.5 on the same test images (Porwal et al., 2020); indicative comparison only.");
s.addNotes("النتيجة النهائية على صور الاختبار اللي النظام عمره ما شافها: بيلاقي حوالي 55% من الـ MA مع 15 إنذار كاذب في الصورة، وبياخد 5 ثواني ونص على CPU. ونتيجة التدريب قريبة من الاختبار، يعني مفيش overfitting. للمقارنة: أحسن موديلات Deep Learning في المسابقة جابت AUPR حوالي 0.5، وإحنا 0.33 من غير GPU.");

// ================================================================== 13 Example
s = content("Results", "What the detector marks on an unseen test image");
s.addImage({ path: A("docs/presentation_assets/annotated_55.png"), x: 0.6, y: 1.45, w: 7.6, h: 5.32, objectName: "annotated" });
const legend = [["00C8C8", "Cyan circle", "correct detection (true positive)"], ["E040E0", "Magenta circle", "false alarm"], ["E0C000", "Yellow square", "ground-truth MA that was missed"]];
legend.forEach(([col, h, b], i) => {
  const y = 1.7 + i * 1.15;
  s.addShape(pres.shapes.OVAL, { x: 8.7, y: y + 0.08, w: 0.45, h: 0.45, fill: { color: "FFFFFF" }, line: { color: col, width: 3 }, objectName: `legend-${i}` });
  text(s, h, { x: 9.35, y, w: 3.4, h: 0.4, fontSize: 17, bold: true, color: C.text1 });
  text(s, b, { x: 9.35, y: y + 0.42, w: 3.4, h: 0.5, fontSize: 15 });
});
text(s, "Many misses sit near hard exudates and dark patches; many false alarms are small dark spots that resemble MA.", { x: 8.7, y: 5.25, w: 4.0, h: 1.3, fontSize: 15 });
source(s, "IDRiD_55 (test set), Model D at its frozen operating point; crop of the downscaled annotated output.");
s.addNotes("ده مثال من صور الاختبار. الدواير السماوي إصابات صح، البنفسجي إنذار كاذب، والمربعات الصفرا MA النظام ماشافهاش. أغلب اللي اتفوت جنب الـ exudates أو في مناطق غامقة.");

// ================================================================== 14 Negative results
pres.addSection({ title: "Discussion" });
s = content("Discussion", "Three things that did not help, and why");
const neg = [
  ["Multi-scale black-hat", "Never beat a single 16 px disc (AUPR \u2264 0.006 vs 0.011). Closing grows with the disc, so the largest scale already contains the smaller ones."],
  ["CLAHE after illumination correction", "AUC +0.003, better in only 27 of 54 images. Both steps equalise brightness locally."],
  ["Combining black-hat with line top-hat", "The pixel-wise minimum (AUPR 0.079) was worse than the line top-hat alone (0.197): the weak black-hat peak limited it."],
];
neg.forEach(([h, b], i) => {
  const y = 1.55 + i * 1.7;
  s.addShape(pres.shapes.OVAL, { x: 0.6, y: y + 0.1, w: 0.6, h: 0.6, fill: { color: C.accent4 }, line: { color: C.accent4 }, objectName: `neg-${i}-dot` });
  text(s, "\u2715", { x: 0.6, y: y + 0.1, w: 0.6, h: 0.6, fontSize: 18, bold: true, color: C.background1, align: "center", valign: "middle" });
  text(s, h, { x: 1.45, y, w: 11.2, h: 0.5, fontSize: 20, bold: true, color: C.text1, fontFace: THEME.headFontFace });
  text(s, b, { x: 1.45, y: y + 0.55, w: 11.2, h: 0.9, fontSize: 16 });
});
source(s, "All on the training set (Experiments 4, 6, 7); decisions were made before the test run.");
s.addNotes("مهم نعرض النتايج السلبية بأمانة. الـ Multi-scale ما نفعش لأن الـ disc الكبير بيغطي الصغير. والـ CLAHE بعد تصحيح الإضاءة كان زيادة. ودمج الـ black-hat مع الـ line top-hat كان أسوأ من الـ line top-hat لوحده.");

// ================================================================== 15 Limitations / next steps
s = content("Discussion", "Limitations and next steps");
card(s, 0.6, 1.55, 5.9, 5.0, C.background2, "limits");
text(s, "Limitations", { x: 0.9, y: 1.75, w: 5.3, h: 0.5, fontSize: 20, bold: true, color: C.text1, fontFace: THEME.headFontFace });
text(s, [
  { text: "54 + 27 images from one dataset and one annotation protocol", options: { bullet: true, breakLine: true } },
  { text: "No healthy images: image-level specificity not measured", options: { bullet: true, breakLine: true } },
  { text: "Ground-truth outlines larger than the dark core lower pixel Dice", options: { bullet: true, breakLine: true } },
  { text: "Small haemorrhages are counted as false positives", options: { bullet: true, breakLine: true } },
  { text: "Pixel sizes tuned for 4288 \u00d7 2848 images", options: { bullet: true } },
], { x: 0.9, y: 2.4, w: 5.3, h: 3.9, fontSize: 16, paraSpaceAfter: 8 });
card(s, 6.83, 1.55, 5.9, 5.0, C.accent6, "next");
text(s, "Next steps", { x: 7.13, y: 1.75, w: 5.3, h: 0.5, fontSize: 20, bold: true, color: C.accent1, fontFace: THEME.headFontFace });
text(s, [
  { text: "Measure specificity on IDRiD images graded \u201cno DR\u201d", options: { bullet: true, breakLine: true } },
  { text: "Add features against haemorrhages and vessel crossings", options: { bullet: true, breakLine: true } },
  { text: "Test on a second camera, rescaling parameters by resolution", options: { bullet: true, breakLine: true } },
  { text: "Compare with a small CNN under the same protocol", options: { bullet: true, breakLine: true } },
  { text: "Evaluate the desktop interface with graders", options: { bullet: true } },
], { x: 7.13, y: 2.4, w: 5.3, h: 3.9, fontSize: 16, paraSpaceAfter: 8 });
s.addNotes("أهم القيود: الداتا صغيرة ومن مصدر واحد، ومفيش صور سليمة علشان نحسب الـ specificity. والخطوات الجاية: نجرب على صور سليمة من IDRiD، ونضيف خصائص تفرق بين MA والنزيف الصغير، ونجرب على كاميرا تانية.");

// ================================================================== 16 Conclusion
pres.addSection({ title: "Conclusion" });
s = pres.addSlide({ masterName: "DARK", sectionTitle: "Conclusion" });
s.addText("Classical processing marks half of all MA on a standard PC", { placeholder: "title" });
s.addText([
  { text: "CPM 0.228 \u00b7 54.7 % sensitivity at 15 false positives per image \u00b7 5.5 s per image", options: { breakLine: true } },
  { text: "Vessel suppression and feature-based filtering gave most of the gain", options: { breakLine: true } },
  { text: "Thank you \u2014 questions welcome", options: { bold: true, color: C.accent2 } },
], { placeholder: "body" });
s.addImage({ path: A("docs/presentation_assets/ma_zoom.png"), x: 8.75, y: 1.6, w: 3.9, h: 3.9, objectName: "closing-image" });
s.addNotes("الخلاصة: معالجة صور تقليدية على كمبيوتر عادي قدرت تعلّم حوالي نص الـ MA بعدد إنذارات كاذبة معقول. وأهم خطوتين كانوا إزالة الأوعية الدموية والتصنيف بالخصائص. شكرا، ومستعد للأسئلة.");

// ------------------------------------------------------------------ write + theme
(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("written", OUT);
})();
