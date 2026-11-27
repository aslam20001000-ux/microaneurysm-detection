// Arabic two-page project summary for the supervisor (RTL, English terms inline). Needs docx@8.5.0 on NODE_PATH.
const fs = require("fs");
const path = require("path");
const { Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, WidthType, AlignmentType,
  BorderStyle, ShadingType } = require("docx");

const ROOT = path.resolve(__dirname, "..");
const OUT = path.join(ROOT, "docs/Project_Summary_for_Supervisor_AR.docx");
const FIG = (f) => path.join(ROOT, "docs/report_figures", f);
const IN = 1440, ACCENT = "C8502A", DARK = "1B2433";
const FONT = { ascii: "Calibri", hAnsi: "Calibri", cs: "Arial" };

// Only Arabic segments are right-to-left; English words and numbers keep their own order.
const ARABIC = /([؀-ۿ]+(?:[\s،؟؛]+[؀-ۿ]+)*)/g;
function runs(text, o = {}) {
  const out = [];
  text.split(/(\*[^*]+\*)/g).filter(Boolean).forEach((piece) => {
    const bold = piece.startsWith("*");
    (bold ? piece.slice(1, -1) : piece).split(ARABIC).filter(Boolean).forEach((seg) =>
      out.push(new TextRun({ text: seg, bold: bold || undefined, rightToLeft: /[؀-ۿ]/.test(seg), font: FONT, ...o })));
  });
  return out;
}
const sz = (n) => ({ size: n, sizeComplexScript: n });
const P = (t) => new Paragraph({ bidirectional: true, alignment: AlignmentType.BOTH, spacing: { after: 90, line: 280 }, children: runs(t, sz(22)) });
const H = (t) => new Paragraph({ bidirectional: true, spacing: { before: 170, after: 70 }, keepNext: true, children: runs(t, { ...sz(26), bold: true, color: ACCENT }) });
const B = (t) => new Paragraph({ bidirectional: true, alignment: AlignmentType.BOTH, numbering: { reference: "b", level: 0 }, spacing: { after: 50, line: 270 }, children: runs(t, sz(22)) });
const N = (t) => new Paragraph({ bidirectional: true, alignment: AlignmentType.BOTH, numbering: { reference: "n", level: 0 }, spacing: { after: 50, line: 270 }, children: runs(t, sz(22)) });
function img(file, widthIn, caption) {
  const b = fs.readFileSync(FIG(file)); const w = b.readUInt32BE(16), h = b.readUInt32BE(20);
  return [new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, children: [new ImageRun({ data: b, transformation: { width: widthIn * 96, height: Math.round(widthIn * 96 * h / w) } })] }),
    new Paragraph({ bidirectional: true, alignment: AlignmentType.CENTER, spacing: { after: 110 }, children: runs(caption, { ...sz(18), italics: true, color: "5A6573" }) })];
}
const LINE = { style: BorderStyle.SINGLE, size: 4, color: "C9D0D8" };
function table(head, rows, widths, hi) {
  const cell = (t, i, isHead, isHi) => new TableCell({ width: { size: widths[i] * IN, type: WidthType.DXA },
    shading: isHead ? { type: ShadingType.CLEAR, fill: DARK, color: "auto" } : isHi ? { type: ShadingType.CLEAR, fill: "F4E3D7", color: "auto" } : undefined,
    borders: { top: LINE, bottom: LINE, left: LINE, right: LINE }, margins: { top: 40, bottom: 40, left: 80, right: 80 },
    children: [new Paragraph({ bidirectional: true, alignment: AlignmentType.CENTER, children: runs(t, { ...sz(19), bold: isHead || isHi, color: isHead ? "FFFFFF" : DARK }) })] });
  return [new Table({ visuallyRightToLeft: true, width: { size: widths.reduce((a, b) => a + b, 0) * IN, type: WidthType.DXA }, columnWidths: widths.map((w) => w * IN),
    rows: [new TableRow({ tableHeader: true, children: head.map((h, i) => cell(h, i, true, false)) }),
      ...rows.map((r, ri) => new TableRow({ children: r.map((c, i) => cell(c, i, false, ri === hi)) }))] }), new Paragraph({ spacing: { after: 70 }, children: [] })];
}

const c = [];
const add = (...x) => x.flat().forEach((e) => c.push(e));
add(new Paragraph({ bidirectional: true, alignment: AlignmentType.CENTER, spacing: { after: 50 }, children: runs("الكشف المبكر عن اعتلال الشبكية السكري: اكتشاف الـ Microaneurysms باستخدام Digital Image Processing", { ...sz(30), bold: true, color: DARK }) }));
add(new Paragraph({ bidirectional: true, alignment: AlignmentType.CENTER, spacing: { after: 150 }, children: runs("ملخص المشروع · مادة Digital Image Processing · هندسة طبية حيوية · [اسم الطالب] · إشراف: [اسم الدكتور]", { ...sz(19), color: "5A6573" }) }));

add(H("1. فكرة المشروع"));
add(P("المشروع نظام بلغة Python يستقبل صورة قاع العين (Fundus image) ويحدد أماكن الـ *Microaneurysms (MA)*، وهي أول علامة تظهر في اعتلال الشبكية السكري (Diabetic Retinopathy). النظام يعتمد على Traditional Digital Image Processing فقط، بدون Deep Learning وبدون GPU، ويعمل على أي كمبيوتر عادي. الهدف هو مساعدة الطبيب في الفرز السريع، وليس التشخيص."));

add(H("2. الداتا وطريقة العمل"));
add(P("استخدمنا داتا *IDRiD*: عدد 54 صورة تدريب (Training) و27 صورة اختبار (Testing)، مقاس كل صورة 4288 × 2848 pixel، وفيها 2412 و1085 MA على الترتيب. كل الـ parameters تم اختيارها على صور التدريب فقط بقاعدة اختيار محددة قبل كل تجربة، وصور الاختبار تم تشغيلها *مرة واحدة فقط* في النهاية."));

add(H("3. الـ Flow: الصورة بتمشي إزاي في النظام"));
add(img("fig01_pipeline.png", 3.2, "مراحل النظام والموديلات السبعة؛ كل Model بيضيف تقنية واحدة على اللي قبله."));
add(N("*Green channel:* بناخد القناة الخضرا لأن الهيموجلوبين بيمتص الضوء الأخضر، فالـ MA بتظهر غامقة وواضحة."));
add(N("*FOV mask:* بنحدد دايرة الشبكية ونتجاهل الخلفية السودا حواليها."));
add(N("*Illumination correction:* بنصلّح عدم انتظام الإضاءة بقسمة الصورة على خلفية Gaussian حجمها 151 pixel."));
add(N("*CLAHE:* بنزود التباين المحلي (clip limit = 1.0)."));
add(N("*Gaussian denoising:* بنشيل الـ noise ومربعات الـ JPEG بفلتر σ = 1.5، مع الحفاظ على الإصابات الصغيرة."));
add(N("*Black-hat:* عملية Morphology بتطلّع الحاجات الغامقة الصغيرة (disc نصف قطره 16 pixel)."));
add(N("*Vessel suppression (Line top-hat):* بنشيل الأوعية الدموية بخط طوله 41 pixel في 12 اتجاه؛ الوعاء طويل فبيتشال، والـ MA مدورة فبتفضل."));
add(N("*Adaptive threshold + Connected Components:* بنحوّل الاستجابة لمجموعة candidates."));
add(N("*Random Forest:* بنحسب 20 خاصية لكل candidate (الشكل، الشدة، اللون، المكان)، والـ classifier بيفرّق الـ MA الحقيقية عن الإنذارات الكاذبة."));
add(P("*التقييم:* على مستوى الإصابة (Lesion-level) باستخدام Sensitivity وPrecision وعدد الإنذارات الكاذبة في الصورة (FPI) وFROC وCPM، ومعاهم Dice وAUPR على مستوى الـ pixel. لم نستخدم Accuracy لأن الـ MA تمثل 0.10 % فقط من الصورة."));

add(H("4. النتايج على صور الاختبار"));
add(table(["Model", "بيضيف إيه", "CPM", "Sensitivity", "Precision", "FP / صورة", "الوقت"], [
  ["A", "Green + threshold", "0.000", "0.149", "0.005", "3213", "0.24 s"],
  ["A+", "Illumination correction", "0.000", "0.749", "0.015", "4544", "0.44 s"],
  ["B", "CLAHE", "0.000", "0.715", "0.015", "4367", "0.64 s"],
  ["B+", "Denoising", "0.000", "0.501", "0.025", "903", "0.65 s"],
  ["C", "Black-hat", "0.001", "0.322", "0.024", "568", "0.97 s"],
  ["D0", "Vessel suppression", "0.038", "0.503", "0.280", "60", "0.94 s"],
  ["D", "Random Forest", "0.228", "0.547", "0.589", "15.3", "5.46 s"],
], [0.55, 1.65, 0.6, 0.85, 0.8, 0.85, 0.65], 6));
add(P("النظام النهائي (Model D) وصل لـ *CPM = 0.228* (95 % CI: 0.195 – 0.267)، واكتشف *54.7 %* من الـ MA بـ Precision = 58.9 %، والإنذارات الكاذبة نزلت لـ *15.3 في الصورة* بدل 3213 في أبسط موديل، والصورة الواحدة بتاخد 5.5 ثانية على CPU. ونتيجة التدريب (0.194) قريبة من الاختبار (0.228)، يعني مفيش Overfitting."));
add(img("fig08_froc_test.png", 3.5, "منحنى FROC على صور الاختبار؛ Model D (البرتقالي) أعلى من كل الموديلات."));

add(H("5. أهم الاستنتاجات"));
add(B("أهم خطوتين هما *إزالة الأوعية الدموية* و*الـ Random Forest*: الإنذارات الكاذبة نزلت من 568 لـ 60 ثم لـ 15 في الصورة."));
add(B("*تصحيح الإضاءة* خلّى threshold واحد يشتغل على الشبكية كلها (AUC من 0.68 لـ 0.84)، و*الـ CLAHE بعده تقريبًا ماضافش حاجة* (+0.003)."));
add(B("*الـ Multi-scale Black-hat ماكانش أحسن من scale واحد* (r = 16)، لأن الـ disc الكبير بيغطي اللي الصغير بيلاقيه."));
add(B("أحسن موديلات Deep Learning في مسابقة IDRiD وصلت لـ AUPR حوالي 0.5، ونظامنا وصل لـ 0.33 من غير GPU (مقارنة استرشادية)."));

add(H("6. القيود والخطوات القادمة"));
add(P("الداتا صغيرة ومن مصدر واحد، وكل الصور فيها MA فلم نحسب الـ Specificity على مستوى الصورة، والنزيف الصغير بيتحسب إنذار كاذب. الخطوات القادمة: تجربة صور سليمة، إضافة خصائص تفرّق بين الـ MA والنزيف، تجربة كاميرا تانية، والمقارنة مع CNN صغير."));

add(H("7. المخرجات"));
add(P("كود Python مع 9 تجارب، وواجهة احترافية (PySide6) لاختيار الـ Model والتحكم في الـ sensitivity والمقارنة مع الـ Ground Truth، وتقرير بصيغة IEEE، وعرض تقديمي. الكود على GitHub: github.com/aslam20001000-ux/microaneurysm-detection"));

const doc = new Document({ styles: { default: { document: { run: { font: FONT, ...sz(22) } } } },
  numbering: { config: [
    { reference: "b", levels: [{ level: 0, format: "bullet", text: "•", alignment: AlignmentType.RIGHT, style: { paragraph: { indent: { right: 360, hanging: 240 } } } }] },
    { reference: "n", levels: [{ level: 0, format: "decimal", text: "%1.", alignment: AlignmentType.RIGHT, style: { paragraph: { indent: { right: 400, hanging: 300 } } } }] }] },
  sections: [{ properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 0.8 * IN, bottom: 0.8 * IN, left: 0.9 * IN, right: 0.9 * IN } } }, children: c }] });
Packer.toBuffer(doc).then((b) => { fs.writeFileSync(OUT, b); console.log("written", OUT); });
