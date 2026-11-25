// Arabic project summary (RTL) for the student to explain the project to the supervisor.
// Run with NODE_PATH pointing to node_modules that contain docx@8.5.0.
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, WidthType, AlignmentType,
  BorderStyle, ShadingType, VerticalAlign, PageBreak,
} = require("docx");

const ROOT = path.resolve(__dirname, "..");
const FIG = (f) => path.join(ROOT, "docs/report_figures", f);
const OUT = path.join(ROOT, "docs/Project_Summary_Arabic.docx");
const IN = 1440;
const TEXT_W = 6.27 * IN;                        // A4 width minus 1" margins
const ACCENT = "C8502A", DARK = "1B2433", MUTED = "5A6573", TINT = "F4E3D7", LIGHT = "EEF1F5";
const FONT = { ascii: "Calibri", hAnsi: "Calibri", cs: "Arial" };

// *bold* markup inside Arabic text. Each piece is split further into Arabic and non-Arabic
// segments: only Arabic segments are marked right-to-left, so English words, model names
// ("A+") and numbers with units ("0.24 s") keep their own left-to-right order.
const ARABIC = /([؀-ۿ]+(?:[\s،؟؛]+[؀-ۿ]+)*)/g;
function runs(text, o = {}) {
  const out = [];
  text.split(/(\*[^*]+\*)/g).filter(Boolean).forEach((piece) => {
    const bold = piece.startsWith("*");
    const body = bold ? piece.slice(1, -1) : piece;
    body.split(ARABIC).filter(Boolean).forEach((seg) => {
      const isArabic = /[؀-ۿ]/.test(seg);
      out.push(new TextRun({ text: seg, bold: bold || undefined, rightToLeft: isArabic, font: FONT, ...o }));
    });
  });
  return out;
}
const P = (t, o = {}) => new Paragraph({ bidirectional: true, alignment: AlignmentType.BOTH, spacing: { after: 120, line: 300 }, children: runs(t, { size: 24, sizeComplexScript: 24 }), ...o });
const H1 = (t) => new Paragraph({ bidirectional: true, spacing: { before: 360, after: 160 }, keepNext: true,
  children: runs(t, { size: 34, sizeComplexScript: 34, bold: true, color: ACCENT }) });
const H2 = (t) => new Paragraph({ bidirectional: true, spacing: { before: 200, after: 100 }, keepNext: true,
  children: runs(t, { size: 27, sizeComplexScript: 27, bold: true, color: DARK }) });
const B = (t) => new Paragraph({ bidirectional: true, alignment: AlignmentType.BOTH, spacing: { after: 80, line: 290 },
  numbering: { reference: "bul", level: 0 }, children: runs(t, { size: 24, sizeComplexScript: 24 }) });
const N = (t) => new Paragraph({ bidirectional: true, alignment: AlignmentType.BOTH, spacing: { after: 80, line: 290 },
  numbering: { reference: "num", level: 0 }, children: runs(t, { size: 24, sizeComplexScript: 24 }) });

function pngSize(f) { const b = fs.readFileSync(f); return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) }; }
function img(file, caption, widthIn = 4.6) {
  const { w, h } = pngSize(FIG(file));
  const px = widthIn * 96;
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120 }, keepNext: true,
      children: [new ImageRun({ data: fs.readFileSync(FIG(file)), transformation: { width: px, height: Math.round(px * h / w) } })] }),
    new Paragraph({ bidirectional: true, alignment: AlignmentType.CENTER, spacing: { after: 200 },
      children: runs(caption, { size: 20, sizeComplexScript: 20, color: MUTED, italics: true }) }),
  ];
}

const BORDER = { style: BorderStyle.SINGLE, size: 4, color: "C9D0D8" };
function table(header, rows, widthsIn, highlightRow = -1) {
  const widths = widthsIn.map((w) => Math.round(w * IN));
  const cell = (t, i, header, hi) => new TableCell({
    width: { size: widths[i], type: WidthType.DXA }, verticalAlign: VerticalAlign.CENTER,
    margins: { top: 60, bottom: 60, left: 90, right: 90 },
    shading: header ? { type: ShadingType.CLEAR, fill: DARK, color: "auto" } : hi ? { type: ShadingType.CLEAR, fill: TINT, color: "auto" } : undefined,
    borders: { top: BORDER, bottom: BORDER, left: BORDER, right: BORDER },
    children: [new Paragraph({ bidirectional: true, alignment: AlignmentType.CENTER,
      children: runs(String(t), { size: 20, sizeComplexScript: 20, bold: header || hi, color: header ? "FFFFFF" : DARK }) })],
  });
  return [new Table({
    visuallyRightToLeft: true, width: { size: widths.reduce((a, b) => a + b, 0), type: WidthType.DXA }, columnWidths: widths,
    rows: [new TableRow({ tableHeader: true, children: header.map((h, i) => cell(h, i, true, false)) }),
      ...rows.map((r, ri) => new TableRow({ children: r.map((c, i) => cell(c, i, false, ri === highlightRow)) }))],
  }), new Paragraph({ spacing: { after: 160 }, children: [] })];
}

// Highlighted box (one-cell table)
function box(lines, fill = TINT) {
  return [new Table({
    visuallyRightToLeft: true, width: { size: TEXT_W, type: WidthType.DXA }, columnWidths: [TEXT_W],
    rows: [new TableRow({ children: [new TableCell({
      width: { size: TEXT_W, type: WidthType.DXA }, shading: { type: ShadingType.CLEAR, fill, color: "auto" },
      margins: { top: 140, bottom: 140, left: 200, right: 200 },
      borders: { top: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" }, bottom: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" },
        left: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" }, right: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" } },
      children: lines.map((l) => new Paragraph({ bidirectional: true, alignment: AlignmentType.BOTH, spacing: { after: 80, line: 300 },
        children: runs(l, { size: 24, sizeComplexScript: 24 }) })),
    })] })],
  }), new Paragraph({ spacing: { after: 160 }, children: [] })];
}

// Q&A item
const Q = (q, a) => [
  new Paragraph({ bidirectional: true, spacing: { before: 160, after: 60 }, keepNext: true, children: runs("س: " + q, { size: 24, sizeComplexScript: 24, bold: true, color: ACCENT }) }),
  new Paragraph({ bidirectional: true, alignment: AlignmentType.BOTH, spacing: { after: 100, line: 300 }, children: runs("ج: " + a, { size: 24, sizeComplexScript: 24 }) }),
];

// ================================================================== content
const c = [];
const add = (...x) => x.flat().forEach((e) => c.push(e));

add(new Paragraph({ bidirectional: true, alignment: AlignmentType.CENTER, spacing: { after: 120 },
  children: runs("ملخص المشروع ودليل الشرح للدكتور", { size: 44, sizeComplexScript: 44, bold: true, color: DARK }) }));
add(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 80 },
  children: [new TextRun({ text: "Early Detection of Diabetic Retinopathy: Microaneurysm Detection Using Traditional Digital Image Processing", size: 24, italics: true, color: MUTED, font: FONT })] }));
add(new Paragraph({ bidirectional: true, alignment: AlignmentType.CENTER, spacing: { after: 300 },
  children: runs("Biomedical Engineering · Digital Image Processing · IDRiD Dataset · Python", { size: 20, sizeComplexScript: 20, color: MUTED }) }));

// ---------------------------------------------------------------- 1
add(H1("1. المشروع في 30 ثانية"));
add(box([
  "*الفكرة:* عملنا نظام بالـ Python بيدوّر على الـ *Microaneurysms (MA)* في صور قاع العين، ودي أول علامة بتظهر في *اعتلال الشبكية السكري (Diabetic Retinopathy)*.",
  "*الطريقة:* Traditional Digital Image Processing فقط (من غير Deep Learning)، وبيشتغل على كمبيوتر عادي من غير GPU.",
  "*الإضافة البحثية:* عملنا *Ablation Study*: سبع موديلات، كل واحد بيزود تقنية واحدة، علشان نقيس كل خطوة بتساهم بإيه بالظبط.",
  "*أهم نتيجة:* على صور اختبار النظام عمره ما شافها، لقى *54.7 %* من الـ MA، مع *15 إنذار كاذب* بس في الصورة (بعد ما كانوا *3213*)، في *5.5 ثانية* للصورة.",
]));

// ---------------------------------------------------------------- 2
add(H1("2. فكرة المشروع والمشكلة"));
add(H2("ليه المشروع مهم؟"));
add(B("حوالي *536.6 مليون* بالغ عندهم سكر في العالم سنة 2021، ومتوقع يوصلوا *783.2 مليون* سنة 2045 (IDF Diabetes Atlas)."));
add(B("حوالي *22 %* من مرضى السكر عندهم اعتلال شبكية (Teo et al., 2021)."));
add(B("في مصر: في سوهاج *85 %* من اللي فوق 50 سنة عمرهم ما كشفوا على عينهم (AlSawahli et al., 2021)، وفي الإسكندرية والدلتا *32.5 %* من مرضى السكر عندهم اعتلال شبكية (Elmassry et al., 2023)."));
add(B("المرض في الأول *مالوش أعراض*، فالحل الوحيد هو الـ *Screening* بصور قاع العين، والصور كتير جدًا على عدد الأطباء المتاح."));
add(H2("ليه اكتشاف الـ MA صعب؟ (أرقام قسناها بنفسنا على الداتا)"));
add(table(["التحدي", "الرقم اللي قسناه"], [
  ["حجم الإصابة صغير جدًا", "أصغر MA = 13 pixel في صورة 12 Megapixel"],
  ["عدم توازن الـ classes", "الـ MA = 0.10 % بس من الـ pixels"],
  ["التباين ضعيف", "الـ MA أغمق من اللي حواليها بـ 7.6 درجة رمادي بس"],
  ["الإضاءة مش منتظمة", "الإضاءة بتتغير من 45 لـ 185 على نفس السطر"],
  ["جودة الصورة", "مربعات JPEG 8×8 أقوى 2.6 مرة من النسيج الطبيعي"],
  ["حاجات شبه الـ MA", "الأوعية الدموية والنزيف الصغير غامقين زيها"],
], [2.4, 3.8]));
add(img("fig02_ma_example.png", "شكل 1: صورة من الداتا (يمين: الصورة كاملة، شمال: تكبير 300×300 والـ MA متعلم عليها بالأصفر). الإصابة كام pixel بس."));

// ---------------------------------------------------------------- 3
add(H1("3. الداتا والأدوات"));
add(H2("الداتا: IDRiD (Indian Diabetic Retinopathy Image Dataset)"));
add(table(["البند", "القيمة"], [
  ["عدد الصور", "81 صورة فيها Ground Truth على مستوى الـ pixel"],
  ["التقسيم", "54 تدريب (Training) + 27 اختبار (Testing)"],
  ["عدد الـ MA", "2412 في التدريب + 1085 في الاختبار"],
  ["مقاس الصورة", "4288 × 2848 pixel (اشتغلنا على المقاس الكامل من غير تصغير)"],
  ["المصدر", "Porwal et al. (2018) من Kaggle"],
], [2.0, 4.2]));
add(H2("الأدوات"));
add(B("*اللغة والبيئة:* Python 3.12 على Windows، وVS Code، وVirtual Environment (venv)."));
add(B("*المكتبات:* OpenCV (معالجة الصور والـ morphology)، NumPy، SciPy، scikit-image، scikit-learn (الـ Random Forest والمقاييس)، pandas (الجداول وملفات CSV)، Matplotlib (الرسومات)."));
add(B("*الواجهة:* Tkinter (برنامج بسيط ترفع فيه صورة ويعلّم على الإصابات)."));

// ---------------------------------------------------------------- 4
add(H1("4. اشتغلنا إزاي؟ (الـ Pipeline)"));
add(P("الصورة بتعدي على الخطوات دي بالترتيب. كل خطوة اخترنا الـ parameters بتاعتها بتجربة على صور التدريب بس، وكل تجربة كان ليها *قاعدة اختيار مكتوبة قبل ما نشغلها*، علشان منختارش اللي يعجبنا بعد ما نشوف النتيجة."));
add(img("fig01_pipeline.png", "شكل 2: الـ Pipeline والموديلات السبعة. كل Model بيضيف تقنية واحدة على اللي قبله.", 4.2));
add(table(["الخطوة", "بتعمل إيه؟", "اخترنا إيه وليه؟"], [
  ["Green channel", "ناخد القناة الخضرا بس", "أعلى تباين للـ MA في 53 من 54 صورة، لأن الهيموجلوبين بيمتص الأخضر"],
  ["FOV mask", "نحدد دايرة الشبكية ونتجاهل الخلفية السودا", "Threshold 30 على القناة الحمرا، و100 % من الـ MA جوه الـ mask"],
  ["Illumination correction", "نشيل عدم انتظام الإضاءة", "نقسم على خلفية Gaussian حجمها 151 pixel؛ الـ AUC طلع من 0.68 لـ 0.84"],
  ["CLAHE", "نزود التباين المحلي", "Clip 1.0؛ زوّد حاجة بسيطة جدًا بعد تصحيح الإضاءة"],
  ["Gaussian denoising", "نشيل الـ noise ومربعات الـ JPEG", "σ = 1.5؛ حافظ على 92 % من عمق أصغر الإصابات"],
  ["Black-hat", "يطلّع الحاجات الغامقة الصغيرة", "Disc نص قطره 16 pixel؛ بس بيطلّع الأوعية كمان"],
  ["Line top-hat", "يشيل الأوعية الدموية ويسيب الـ MA", "خط طوله 41 pixel في 12 اتجاه؛ أكبر تحسن في المشروع"],
  ["Threshold + CCA", "نحول الاستجابة لـ candidates", "Adaptive threshold على robust z-score + Connected Components"],
  ["Random Forest", "يفرّق الـ MA الحقيقية عن الإنذارات الكاذبة", "20 خاصية (شكل، شدة، لون، استجابة)؛ أحسن بكتير من Decision Tree"],
], [1.45, 2.05, 2.7]));
add(img("fig06_vessel_suppression.png", "شكل 3: إزالة الأوعية الدموية. (a) الصورة بعد التحسين، (b) الـ Black-hat بيطلّع الأوعية، (c) الـ Line top-hat الأوعية اختفت والـ MA (المتعلم عليها بالأخضر) لسه موجودة.", 5.6));

// ---------------------------------------------------------------- 5
add(H1("5. تصميم التجارب وطريقة التقييم"));
add(H2("الـ Ablation Study"));
add(P("عملنا 7 موديلات، كل واحد بيزود حاجة واحدة بس على اللي قبله، كده نقدر نقول كل خطوة ساهمت بإيه بالظبط:"));
add(B("النموذج *A*: نقطة البداية، القناة الخضرا مع Threshold."));
add(B("النموذج *A+*: بيضيف تصحيح الإضاءة (Illumination correction)."));
add(B("النموذج *B*: بيضيف CLAHE."));
add(B("النموذج *B+*: بيضيف إزالة الـ noise (Gaussian denoising)."));
add(B("النموذج *C*: بيضيف الـ Black-hat (Morphology)."));
add(B("النموذج *D0*: بيضيف إزالة الأوعية الدموية (Line top-hat)."));
add(B("النموذج *D*: بيضيف الـ Random Forest على الخصائص، وده النظام النهائي."));
add(H2("إزاي ضمنا إن النتيجة أمينة؟"));
add(B("ضبطنا كل حاجة على *صور التدريب بس* (54 صورة)."));
add(B("صور الاختبار (27 صورة) *اتشغلت مرة واحدة بس* في الآخر، بعد ما كل الـ parameters اتثبتت."));
add(B("حسبنا *95 % Confidence Intervals* بالـ Bootstrap علشان نعرف النتيجة ثابتة قد إيه."));
add(H2("المقاييس"));
add(B("*Lesion-level (الأساسي):* كل MA بتتحسب لوحدها. الاكتشاف صح لو مركزه جوه الإصابة أو على بعد 5 pixel منها. بنحسب Sensitivity وPrecision وF1 وعدد الإنذارات الكاذبة في الصورة (FPI)."));
add(B("*FROC و CPM:* منحنى بيوضح الـ Sensitivity عند كل عدد إنذارات كاذبة، والـ CPM هو متوسط الـ Sensitivity عند 1/8 لحد 8 إنذارات في الصورة (مقياس مسابقة Retinopathy Online Challenge)."));
add(B("*Pixel-level:* Dice وIoU وAUPR (المقياس الرسمي لمسابقة IDRiD)."));
add(B("*ليه مش Accuracy؟* لأن الـ MA أقل من 0.1 % من الصورة، فنظام بيقول \"مفيش MA خالص\" هياخد Accuracy أكتر من 99.8 % وهو فاشل."));

// ---------------------------------------------------------------- 6
add(H1("6. أفضل النتايج"));
add(box([
  "*النظام النهائي (Model D) على 27 صورة اختبار فيها 1085 MA:*",
  "• CPM = *0.228* (95 % CI: 0.195 – 0.267)",
  "• Sensitivity = *54.7 %* و Precision = *58.9 %* و F1 = *0.567*",
  "• عدد الإنذارات الكاذبة = *15.3 في الصورة* (كان 3213 في أبسط موديل)",
  "• Pixel AUPR = *0.326* و Dice = *0.448*",
  "• الوقت = *5.5 ثانية* للصورة الواحدة (12 Megapixel) على CPU عادي",
  "• *مفيش Overfitting:* الـ CPM على التدريب 0.194 وعلى الاختبار 0.228",
]));
add(table(["Model", "CPM", "Sensitivity", "Precision", "إنذارات كاذبة/صورة", "الوقت"], [
  ["A", "0.000", "0.149", "0.005", "3213", "0.24 s"],
  ["A+", "0.000", "0.749", "0.015", "4544", "0.44 s"],
  ["B", "0.000", "0.715", "0.015", "4367", "0.64 s"],
  ["B+", "0.000", "0.501", "0.025", "903", "0.65 s"],
  ["C", "0.001", "0.322", "0.024", "568", "0.97 s"],
  ["D0", "0.038", "0.503", "0.280", "60", "0.94 s"],
  ["D", "0.228", "0.547", "0.589", "15.3", "5.46 s"],
], [0.7, 0.8, 1.0, 1.0, 1.6, 0.9], 6));
add(img("fig10_fpi_ablation.png".replace("10", "09"), "شكل 4: عدد الإنذارات الكاذبة في الصورة لكل Model (scale لوغاريتمي). أكبر نزول جه من إزالة الأوعية (C → D0) والـ Random Forest (D0 → D)."));
add(img("fig08_froc_test.png", "شكل 5: منحنى FROC على صور الاختبار. الخط البرتقالي (Model D) أعلى من كل الموديلات عند أي عدد إنذارات."));
add(img("fig10_annotated_test.png", "شكل 6: النظام على صورة اختبار. سماوي = اكتشاف صح، بنفسجي = إنذار كاذب، مربع أصفر = MA النظام ماشافهاش.", 4.4));

// ---------------------------------------------------------------- 7
add(H1("7. البحث طلع بإيه؟ (الاستنتاجات)"));
add(N("*إزالة الأوعية الدموية هي أهم خطوة:* الـ Line top-hat نزّل الإنذارات الكاذبة من 568 لـ 60 في الصورة، وزوّد الـ AUPR حوالي 17 مرة على التدريب (من 0.011 لـ 0.197)."));
add(N("*الـ Random Forest على الخصائص هو تاني أهم خطوة:* نزّل الإنذارات من 60 لـ 15، وطلّع الـ CPM من 0.038 لـ 0.228. وأهم الخصائص كانت المساحة (area) وقوة الاستجابة والمحيط."));
add(N("*تصحيح الإضاءة خلّى threshold واحد ينفع على الشبكية كلها:* الـ AUC طلع من 0.68 لـ 0.84، والتحسن ظهر في 53 من 54 صورة."));
add(N("*الـ CLAHE بعد تصحيح الإضاءة ما ضافش تقريبًا حاجة* (AUC زاد 0.003 بس). الاتنين بيعملوا نفس الشغلانة: بيساووا الإضاءة محليًا. ودي نتيجة مهمة لأن ناس كتير بتستخدم الاتنين مع بعض."));
add(N("*الـ Multi-scale Black-hat ما نفعش:* scale واحد (r = 16) كان أحسن من أي تركيبة. السبب رياضي (granulometry): الـ disc الكبير بيغطي كل اللي الصغير بيلاقيه."));
add(N("*دمج الـ Black-hat مع الـ Line top-hat كان أسوأ* من الـ Line top-hat لوحده (AUPR 0.079 مقابل 0.197)."));
add(N("*الـ Denoising مهم مخصوص في الداتا دي* لأن صور IDRiD فيها مربعات JPEG تأثيرها قريب من تباين الـ MA نفسها."));
add(H2("المقارنة بالأبحاث"));
add(P("على نفس صور الاختبار، أحسن فرق في مسابقة IDRiD (كلها Deep Learning) جابت Pixel AUPR حوالي *0.5*. إحنا جبنا *0.326* من غير Deep Learning ومن غير GPU. المقارنة دي استرشادية بس، لأن نظامنا مش معمول أصلًا للـ pixel segmentation."));
add(H2("القيود (لازم تقولها بنفسك قبل ما الدكتور يسأل)"));
add(B("الداتا صغيرة (81 صورة) ومن مصدر واحد."));
add(B("كل الصور فيها MA، فمقدرناش نحسب الـ Specificity على مستوى الصورة."));
add(B("الـ Ground Truth مرسومة أكبر من النقطة الغامقة، وده بيقلل الـ Dice."));
add(B("النزيف الصغير بيتحسب إنذار كاذب لأننا بنقيّم الـ MA بس."));
add(B("الأحجام بالـ pixel متظبطة على صور 4288 × 2848، ولو الكاميرا اتغيرت لازم تتظبط تاني."));

// ---------------------------------------------------------------- 8
add(H1("8. أسئلة متوقعة من الدكتور وإجاباتها"));
add(Q("ليه مستخدمتوش Deep Learning؟", "المشروع في مادة Digital Image Processing والهدف كان نفهم ونقيس كل تقنية تقليدية. كمان الداتا فيها 54 صورة تدريب بس، والـ Deep Learning محتاج آلاف الصور (بحث Google اتدرب على 128 ألف صورة) ومحتاج GPU. نظامنا بيشتغل على أي كمبيوتر وكل خطوة فيه مفهومة."));
add(Q("ليه القناة الخضرا؟", "الهيموجلوبين بيمتص الضوء الأخضر، فالـ MA والأوعية بيبانوا غامقين. وقسناها: الـ Contrast-to-Noise بتاع الأخضر كان 0.67 مقابل 0.28 للأحمر و0.17 للأزرق، وكان الأحسن في 53 من 54 صورة."));
add(Q("ليه مستخدمتوش Accuracy؟", "لأن الـ MA أقل من 0.1 % من الصورة. نظام فاشل بيقول \"مفيش MA\" هياخد 99.8 % Accuracy. علشان كده استخدمنا مقاييس على مستوى الإصابة زي FROC وCPM."));
add(Q("إيه هو الـ CPM؟", "متوسط الـ Sensitivity عند 7 نقط: 1/8 و1/4 و1/2 و1 و2 و4 و8 إنذار كاذب في الصورة. ده المقياس اللي اتعمل في مسابقة Retinopathy Online Challenge علشان يقارن أنظمة الـ MA."));
add(Q("إزاي ضمنتوا إنكم مغشتوش من صور الاختبار؟", "كل الـ parameters اتظبطت على صور التدريب بس، وكل تجربة كان ليها قاعدة اختيار مكتوبة قبل التشغيل. صور الاختبار اتشغلت مرة واحدة بس في الآخر. والدليل إن نتيجة التدريب (0.194) قريبة من الاختبار (0.228)."));
add(Q("الـ Random Forest مش ده Machine Learning؟", "أيوه، بس ده Machine Learning تقليدي مش Deep Learning. هو بيشتغل على خصائص إحنا حسبناها بإيدينا بالـ Image Processing (المساحة، الاستدارة، التباين، اللون). وده الأسلوب المعروف في الأبحاث الكلاسيكية للـ MA (Niemeijer 2005، Walter 2007، Antal & Hajdu 2012)."));
add(Q("ليه تصحيح الإضاءة زوّد الإنذارات الكاذبة في الجدول؟", "لأنه خلّى النظام يشوف إصابات كتير أكتر (الـ Sensitivity طلعت من 15 % لـ 75 %)، والـ threshold اللي اتختار علشان أحسن F1 دخّل candidates أكتر. الإنذارات الكاذبة اتحلت بعد كده بإزالة الأوعية والـ Random Forest."));
add(Q("ليه الـ Multi-scale مانفعش مع إنه في عنوان الفكرة؟", "جربناه بأكتر من طريقة وقسناه. الـ Black-hat بـ disc كبير بيلاقي كل اللي الـ disc الصغير بيلاقيه (خاصية الـ granulometry)، والـ normalization لكل scale كبّر الـ noise في الـ scales الصغيرة. فاخترنا scale واحد مناسب لحجم الـ MA. والنتيجة السلبية دي في حد ذاتها نتيجة بحثية."));
add(Q("النتيجة دي كويسة؟", "بالنسبة لنظام تقليدي على CPU: النظام بيلاقي حوالي نص الإصابات مع 15 إنذار كاذب في الصورة، والطبيب يقدر يراجعهم بسرعة. أحسن أنظمة Deep Learning في المسابقة أحسن مننا (AUPR 0.5 مقابل 0.33)، بس محتاجة GPU وداتا كبيرة. ونظامنا مساعد للطبيب مش بيشخّص."));
add(Q("إيه الإضافة البحثية بتاعتكم؟", "مش بنقول إن دمج التقنيات دي جديد. الإضافة إننا قسنا مساهمة كل خطوة لوحدها على IDRiD بالدقة الكاملة بتقييم على مستوى الإصابة، وطلعنا بنتايج واضحة: أهم خطوتين هما إزالة الأوعية والـ classifier، والـ CLAHE زيادة بعد تصحيح الإضاءة، والـ multi-scale مش ضروري."));
add(Q("لو هتكملوا هتعملوا إيه؟", "نحسب الـ Specificity على صور سليمة من IDRiD، ونضيف خصائص تفرّق بين الـ MA والنزيف الصغير، ونجرب على كاميرا تانية، ونقارن بـ CNN صغير بنفس طريقة التقييم."));

// ---------------------------------------------------------------- 9
add(H1("9. ملفات المشروع"));
add(table(["الملف", "فيه إيه"], [
  ["src/", "كود الـ pipeline كله مقسم modules"],
  ["experiments/exp01 ... exp09", "التجارب التسعة بالترتيب (كل تجربة لخطوة)"],
  ["outputs/", "كل النتايج والرسومات وملفات CSV"],
  ["outputs/detections/", "صور الاختبار متعلم عليها"],
  ["gui.py", "الواجهة (python gui.py)"],
  ["docs/Microaneurysm_Detection_IEEE_Report.docx", "التقرير البحثي بصيغة IEEE"],
  ["docs/Microaneurysm_Detection_Presentation.pptx", "العرض التقديمي (فيه notes بالعربي)"],
], [3.1, 3.1]));

// ================================================================== document
const doc = new Document({
  creator: "Project", title: "Project summary (Arabic)",
  styles: { default: { document: { run: { font: FONT, size: 24, sizeComplexScript: 24 } } } },
  numbering: { config: [
    { reference: "bul", levels: [{ level: 0, format: "bullet", text: "\u2022", alignment: AlignmentType.RIGHT, style: { paragraph: { indent: { right: 360, hanging: 260 } } } }] },
    { reference: "num", levels: [{ level: 0, format: "decimal", text: "%1.", alignment: AlignmentType.RIGHT, style: { paragraph: { indent: { right: 400, hanging: 300 } } } }] },
  ] },
  sections: [{ properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: IN, bottom: IN, left: IN, right: IN } } }, children: c }],
});
Packer.toBuffer(doc).then((b) => { fs.writeFileSync(OUT, b); console.log("written", OUT, b.length); });
