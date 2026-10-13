/* CrossVMAF site — interactions: i18n, oscilloscope, terminal, reveals. */
(() => {
  "use strict";

  const $ = (s, c = document) => c.querySelector(s);
  const $$ = (s, c = document) => Array.from(c.querySelectorAll(s));
  const REDUCED = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const FINE = window.matchMedia("(pointer: fine)").matches;

  /* ================= i18n ================= */

  const I18N = {
    en: {
      "meta.title": "CrossVMAF — Video quality, measured on the GPU",
      "nav.features": "Features", "nav.pipeline": "Pipeline", "nav.studio": "Studio",
      "nav.cli": "CLI", "nav.api": "API", "nav.download": "Download",
      "hero.kicker": "v0.1.2 · early development · open source",
      "hero.title": "Video quality, measured on the GPU.",
      "hero.sub": "CrossVMAF is Vulkan-accelerated VMAF assessment for Windows and Linux — an independent C++20 implementation of the VMAF 1.0.16 feature and model pipeline, with a desktop studio, a CLI, and a streaming C++ API.",
      "cta.download": "Download for Windows",
      "scope.title": "PER-FRAME VMAF · MEAN POOLING", "scope.sim": "SIMULATED",
      "scope.score": "SCORE", "scope.frames": "FRAMES", "scope.band": "BAND",
      "stats.models": "BUNDLED MODELS", "stats.backends": "DECODE BACKENDS",
      "stats.platforms": "PLATFORMS", "stats.codecs": "CODECS",
      "what.kicker": "OVERVIEW",
      "what.title": "Two videos in. One score out.",
      "what.lede": "CrossVMAF compares a reference video with a distorted video. Feature evaluation runs on a Vulkan GPU; final model prediction runs on the CPU. Every frame is scored, and pooling condenses the sequence into a single 0–100 result.",
      "what.s1t": "Reference + distorted",
      "what.s1d": "Two inputs with matching decoded dimensions, pixel format, bit depth, and frame count. No resizing, no color conversion — compared exactly as decoded.",
      "what.s2t": "Frames paired, features on GPU",
      "what.s2d": "Decoded frames pair in display order with timestamp checks. The VMAF 1.0.16 feature set is extracted in Vulkan compute on any vendor's GPU.",
      "what.s3t": "A pooled 0–100 score",
      "what.s3d": "Per-frame scores pool by mean, min, or harmonic mean. Export JSON or CSV from the CLI — or an A4 PDF report from Studio.",
      "what.legend": "QUALITY BANDS",
      "band.excellent": "EXCELLENT", "band.good": "GOOD", "band.fair": "FAIR", "band.poor": "POOR",
      "feat.kicker": "HIGHLIGHTS",
      "feat.title": "Built like an instrument.",
      "feat.c1t": "Vulkan compute",
      "feat.c1d": "Feature evaluation runs on any Vulkan 1.1+ GPU — Intel, NVIDIA, AMD, or others. Cross-vendor by design, not by port.",
      "feat.c2t": "Five decode paths",
      "feat.c2d": "QSV, NVDEC, AMF, Vulkan Video, and VA-API — picked automatically per GPU vendor, with a software fallback always behind them.",
      "feat.c3t": "Automatic model selection",
      "feat.c3d": "Eight bundled VMAF 1.0.16 models chosen from the reference's decoded dimensions and frame rate — 4K, 1080p, and HFR variants, with CLI overrides.",
      "feat.c4t": "CPU verification",
      "feat.c4d": "<code>--verify-cpu</code> decodes again and runs an independent CPU reference pass, comparing it against the Vulkan results frame by frame.",
      "feat.c5t": "Pooling your way",
      "feat.c5d": "Mean by default. Min and harmonic mean when the worst frames matter more than the average ones.",
      "feat.c6t": "Reports &amp; exports",
      "feat.c6d": "JSON and CSV from the CLI. Studio adds A4 customer PDF reports — pooled score, quality curve, distribution, review segments — with fonts embedded.",
      "pipe.kicker": "PIPELINE",
      "pipe.title": "GPU where it pays, CPU where it counts.",
      "pipe.n1": "DECODE", "pipe.n1s": "auto backend",
      "pipe.n1tip": "Per GPU vendor: Intel QSV · NVIDIA NVDEC · AMD AMF (Windows) / VA-API (Linux) → Vulkan Video → software.",
      "pipe.n2": "FEATURES", "pipe.n2s": "vulkan compute",
      "pipe.n2tip": "The VMAF 1.0.16 feature set is evaluated in compute shaders on the selected Vulkan device.",
      "pipe.n3": "TEMPORAL", "pipe.n3s": "across frames",
      "pipe.n3tip": "Motion and temporal features aggregate across frames on the GPU.",
      "pipe.n4": "MODEL", "pipe.n4s": "svm on cpu",
      "pipe.n4tip": "The bundled model matched to your input predicts per-frame scores on the CPU.",
      "pipe.n5": "SCORE", "pipe.n5s": "mean · min · harmonic",
      "pipe.n5tip": "Pooling condenses the sequence into a single 0–100 score, recorded in every report.",
      "studio.kicker": "STUDIO",
      "studio.title": "A desktop lab for your queue.",
      "studio.lede": "Studio wraps the engine in a desktop workflow: select inputs, queue evaluations, compare saved runs, and review reports — in English or 简体中文, light or dark.",
      "studio.p1t": "Queue with confidence",
      "studio.p1d": "Settings are captured the moment a task is enqueued; the queue pauses on failure by default instead of burning through the batch.",
      "studio.p2t": "Compare runs",
      "studio.p2d": "Overlay quality curves from saved evaluations in four color-blind-safe series colors.",
      "studio.p3t": "Reports that travel",
      "studio.p3d": "Export JSON or CSV for analysis, or generate an A4 customer PDF — pooled score, curve, distribution, review segments — with fonts embedded. No browser or printer driver needed.",
      "studio.chart": "QUALITY CURVE",
      "chip.done": "DONE", "chip.run": "RUNNING", "chip.queue": "QUEUED",
      "cli.kicker": "CLI",
      "cli.title": "Scriptable to the last flag.",
      "cli.sim": "ILLUSTRATIVE OUTPUT",
      "cli.o1": "Reference video · required", "cli.o2": "Distorted video · required",
      "cli.o3": "Force a bundled model · default auto", "cli.o4": "Vulkan compute device",
      "cli.o5": "Sequence pooling · default mean", "cli.o6": "Output format · default json",
      "cli.o7": "Independent CPU verification pass", "cli.o8": "Pin vulkan · qsv · nvdec · vaapi · amf",
      "api.kicker": "C++ API",
      "api.title": "Stream results as frames land.",
      "api.lede": "Embed CrossVMAF in your own pipeline. Open an EvaluationStream on the two paths; next() returns ordered batches, and finish() drains the remaining GPU and temporal results.",
      "api.p1": "Ordered batches",
      "api.p1d": "Results arrive in display order as the GPU pipeline produces them — no waiting for the whole sequence.",
      "api.p2": "Full control",
      "api.p2d": "Cancellation, metadata, frame-count hints, and execution statistics through the public streaming API.",
      "api.copy": "COPY", "api.copied": "COPIED",
      "dl.kicker": "GET CROSSVMAF",
      "dl.title": "Start measuring.",
      "dl.noMac": "macOS — NOT SUPPORTED", "dl.noArm": "ARM — NOT SUPPORTED",
      "dl.panelT": "Grab the release, or build it yourself.",
      "dl.panelD": "The Windows x64 package ships the CLI, Studio, bundled models, and license notices. Building from source needs CMake, Ninja, LLVM/Clang, Python, and Rust for Studio.",
      "dl.releases": "Releases — Windows x64", "dl.build": "Build from source",
      "dl.note": "REQUIRES MSVC X64 RUNTIME + VULKAN-CAPABLE GPU DRIVER · KEEP BUNDLED DLLS AND licenses/ TOGETHER",
      "f.disclaimer": "CrossVMAF is an independent C++20 implementation of the VMAF 1.0.16 pipeline. It does not modify or link against Netflix libvmaf, and is not an official Netflix product.",
      "f.license": "LICENSE",
      "f.licenseD": "BSD-2-Clause-Patent. Dependencies keep their own terms: FFmpeg LGPL-2.1-or-later, dav1d BSD-2-Clause, Slint under separate terms.",
      "f.links": "LINKS", "f.issues": "Issues", "f.releases": "Releases", "f.models": "Model provenance",
      "f.made": "BSD-2-CLAUSE-PATENT · NOT A NETFLIX PRODUCT"
    },
    zh: {
      "meta.title": "CrossVMAF — 在 GPU 上度量视频质量",
      "nav.features": "功能", "nav.pipeline": "流水线", "nav.studio": "Studio",
      "nav.cli": "CLI", "nav.api": "API", "nav.download": "下载",
      "hero.kicker": "v0.1.2 · 早期开发 · 开源",
      "hero.title": "在 GPU 上，度量视频质量。",
      "hero.sub": "CrossVMAF 是 Vulkan 加速的 VMAF 视频质量评估工具，面向 Windows 与 Linux——以独立 C++20 实现 VMAF 1.0.16 特征与模型流水线，提供桌面 Studio、命令行工具与流式 C++ API。",
      "cta.download": "下载 Windows 版",
      "scope.title": "逐帧 VMAF · 均值池化", "scope.sim": "示意",
      "scope.score": "总分", "scope.frames": "帧数", "scope.band": "评级",
      "stats.models": "内置模型", "stats.backends": "解码后端",
      "stats.platforms": "平台", "stats.codecs": "编解码器",
      "what.kicker": "概览",
      "what.title": "两个视频进，一个分数出。",
      "what.lede": "CrossVMAF 将参考视频与劣化视频进行对比：特征评估在 Vulkan GPU 上运行，最终模型预测在 CPU 上完成。逐帧打分，再经池化聚合为 0–100 的单一结果。",
      "what.s1t": "参考片 × 劣化片",
      "what.s1d": "两个输入需具备一致的解码尺寸、像素格式、位深与帧数。不缩放、不做色彩转换——按解码原样逐帧对比。",
      "what.s2t": "帧配对，特征上 GPU",
      "what.s2d": "解码帧按显示顺序配对并校验时间戳。VMAF 1.0.16 特征集在任意厂商的 Vulkan 计算设备上提取。",
      "what.s3t": "池化为 0–100 分",
      "what.s3d": "逐帧分数按均值、最小值或调和均值池化。CLI 导出 JSON / CSV，Studio 还能生成 A4 PDF 报告。",
      "what.legend": "质量分带",
      "band.excellent": "优秀", "band.good": "良好", "band.fair": "一般", "band.poor": "较差",
      "feat.kicker": "亮点",
      "feat.title": "像仪器一样打造。",
      "feat.c1t": "Vulkan 计算",
      "feat.c1d": "特征评估运行于任意 Vulkan 1.1+ GPU——Intel、NVIDIA、AMD 或其他厂商。天生跨厂商，而非移植适配。",
      "feat.c2t": "五路硬件解码",
      "feat.c2d": "QSV、NVDEC、AMF、Vulkan Video、VA-API——按 GPU 厂商自动选择，软解始终兜底。",
      "feat.c3t": "模型自动选择",
      "feat.c3d": "内置八个 VMAF 1.0.16 模型，按参考片的解码尺寸与帧率自动匹配——覆盖 4K、1080p 与高帧率变体，亦可用 CLI 覆盖。",
      "feat.c4t": "CPU 独立验证",
      "feat.c4d": "<code>--verify-cpu</code> 重新解码并运行独立的 CPU 参考流程，与 Vulkan 结果逐帧比对。",
      "feat.c5t": "池化方式任选",
      "feat.c5d": "默认均值。当最差帧比平均水平更重要时，选用最小值或调和均值。",
      "feat.c6t": "报告与导出",
      "feat.c6d": "CLI 输出 JSON 与 CSV。Studio 可生成 A4 客户 PDF 报告——总分、质量曲线、分布与复查片段，字体内嵌。",
      "pipe.kicker": "流水线",
      "pipe.title": "GPU 算特征，CPU 定分数。",
      "pipe.n1": "解码", "pipe.n1s": "自动后端",
      "pipe.n1tip": "按 GPU 厂商：Intel QSV · NVIDIA NVDEC · AMD AMF（Windows）/ VA-API（Linux）→ Vulkan Video → 软解。",
      "pipe.n2": "特征提取", "pipe.n2s": "vulkan 计算",
      "pipe.n2tip": "VMAF 1.0.16 特征集在所选 Vulkan 设备上以计算着色器求值。",
      "pipe.n3": "时域", "pipe.n3s": "跨帧聚合",
      "pipe.n3tip": "运动与时域特征在 GPU 上跨帧聚合。",
      "pipe.n4": "模型", "pipe.n4s": "CPU 运行 SVM",
      "pipe.n4tip": "与输入匹配的内置模型在 CPU 上预测逐帧分数。",
      "pipe.n5": "分数", "pipe.n5s": "均值 · 最小值 · 调和",
      "pipe.n5tip": "池化将整个序列聚合为 0–100 的单一分数，并写入每份报告。",
      "studio.kicker": "STUDIO",
      "studio.title": "桌面端的评测实验室。",
      "studio.lede": "Studio 将引擎包装为桌面工作流：选择输入、排队评测、对比历史结果、审阅报告——界面支持 English 与简体中文、明暗双主题。",
      "studio.p1t": "安心排队",
      "studio.p1d": "任务入队瞬间即捕获设置；默认失败即暂停队列，不会带病跑完整个批次。",
      "studio.p2t": "对比多次评测",
      "studio.p2d": "以四种色盲友好的序列颜色，叠加对比已保存评估的质量曲线。",
      "studio.p3t": "可携带的报告",
      "studio.p3d": "导出 JSON 或 CSV 用于分析，或生成 A4 客户 PDF——总分、曲线、分布、复查片段，字体内嵌。无需浏览器或打印机驱动。",
      "studio.chart": "质量曲线",
      "chip.done": "完成", "chip.run": "进行中", "chip.queue": "排队中",
      "cli.kicker": "CLI",
      "cli.title": "每个开关都可脚本化。",
      "cli.sim": "示意输出",
      "cli.o1": "参考视频 · 必填", "cli.o2": "劣化视频 · 必填",
      "cli.o3": "强制指定内置模型 · 默认自动", "cli.o4": "Vulkan 计算设备",
      "cli.o5": "序列池化 · 默认均值", "cli.o6": "输出格式 · 默认 json",
      "cli.o7": "独立 CPU 验证流程", "cli.o8": "固定后端：vulkan · qsv · nvdec · vaapi · amf",
      "api.kicker": "C++ API",
      "api.title": "帧到即得的流式结果。",
      "api.lede": "把 CrossVMAF 嵌入你自己的流水线。EvaluationStream 打开两个输入路径；next() 返回有序批次，finish() 排空剩余的 GPU 与时域结果。",
      "api.p1": "有序批次",
      "api.p1d": "结果按显示顺序随 GPU 流水线产出即返回——不必等待整个序列跑完。",
      "api.p2": "完全掌控",
      "api.p2d": "通过公开流式 API 提供取消、元数据、帧数提示与执行统计。",
      "api.copy": "复制", "api.copied": "已复制",
      "dl.kicker": "获取 CROSSVMAF",
      "dl.title": "开始度量。",
      "dl.noMac": "macOS — 暂不支持", "dl.noArm": "ARM — 暂不支持",
      "dl.panelT": "下载发布包，或自行构建。",
      "dl.panelD": "Windows x64 包内含 CLI、Studio、内置模型与许可声明。从源码构建需要 CMake、Ninja、LLVM/Clang、Python，Studio 另需 Rust。",
      "dl.releases": "Releases — Windows x64", "dl.build": "从源码构建",
      "dl.note": "需要 MSVC X64 运行库与支持 VULKAN 的 GPU 驱动 · 请保持自带 DLL 与 licenses/ 目录完整",
      "f.disclaimer": "CrossVMAF 是 VMAF 1.0.16 流水线的独立 C++20 实现，未修改或链接 Netflix libvmaf，亦非 Netflix 官方产品。",
      "f.license": "许可证",
      "f.licenseD": "BSD-2-Clause-Patent。依赖保留各自条款：FFmpeg LGPL-2.1-or-later、dav1d BSD-2-Clause、Slint 另有条款。",
      "f.links": "链接", "f.issues": "问题反馈", "f.releases": "发布页", "f.models": "模型溯源",
      "f.made": "BSD-2-CLAUSE-PATENT · 非 NETFLIX 官方产品"
    }
  };

  let lang = "en";
  try {
    const saved = localStorage.getItem("cvm-lang");
    lang = saved === "zh" || saved === "en" ? saved
      : ((navigator.language || "").toLowerCase().startsWith("zh") ? "zh" : "en");
  } catch (e) { /* private mode: keep default */ }

  let relabelBand = () => {};

  function applyLang(l) {
    lang = l;
    try { localStorage.setItem("cvm-lang", l); } catch (e) {}
    document.documentElement.lang = l === "zh" ? "zh-CN" : "en";
    const dict = I18N[l];
    $$("[data-i18n]").forEach((el) => {
      const v = dict[el.getAttribute("data-i18n")];
      if (v != null) el.innerHTML = v;
    });
    document.title = dict["meta.title"];
    $$(".lang button").forEach((b) => b.classList.toggle("on", b.dataset.lang === l));
    relabelBand();
  }

  $$(".lang button").forEach((b) => b.addEventListener("click", () => {
    if (b.dataset.lang !== lang) applyLang(b.dataset.lang);
  }));

  /* ================= nav ================= */

  const nav = $("#nav");
  const onScroll = () => nav.classList.toggle("scrolled", window.scrollY > 10);
  window.addEventListener("scroll", onScroll, { passive: true });
  onScroll();

  /* ================= reveal ================= */

  const revealIO = new IntersectionObserver((entries) => {
    for (const e of entries) {
      if (e.isIntersecting) { e.target.classList.add("in"); revealIO.unobserve(e.target); }
    }
  }, { threshold: 0.12, rootMargin: "0px 0px -6% 0px" });
  $$(".reveal").forEach((el) => revealIO.observe(el));

  /* ================= count-up ================= */

  const easeOut = (t) => 1 - Math.pow(1 - t, 3);
  const countIO = new IntersectionObserver((entries) => {
    for (const e of entries) {
      if (!e.isIntersecting) continue;
      countIO.unobserve(e.target);
      const target = Number(e.target.dataset.count || 0);
      if (REDUCED) { e.target.textContent = String(target); continue; }
      const t0 = performance.now();
      const tick = (now) => {
        const p = Math.min(1, (now - t0) / 900);
        e.target.textContent = String(Math.round(easeOut(p) * target));
        if (p < 1) requestAnimationFrame(tick);
      };
      requestAnimationFrame(tick);
    }
  }, { threshold: 0.6 });
  $$(".stat-num").forEach((el) => countIO.observe(el));

  /* ================= oscilloscope ================= */

  const BANDS = [
    { key: "band.excellent", min: 93, color: "#4EA97C" },
    { key: "band.good", min: 80, color: "#9CBE5A" },
    { key: "band.fair", min: 60, color: "#D9A03F" },
    { key: "band.poor", min: -1, color: "#C9615F" }
  ];
  const bandOf = (v) => BANDS.find((b) => v >= b.min);

  function initScope() {
    const cv = $("#scope");
    if (!cv) return;
    const ctx = cv.getContext("2d");
    const N = 240;

    // Deterministic plausible per-frame series: smooth drift, one deep dip, one mild dip.
    let seed = 20261013;
    const rnd = () => (seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296;
    const data = new Float32Array(N);
    for (let i = 0; i < N; i++) {
      let v = 93.6 + 3.1 * Math.sin(i * 0.045) + 1.6 * Math.sin(i * 0.131 + 1.3) + (rnd() - 0.5) * 2.1;
      v -= 17 * Math.exp(-(((i - 162) / 8.5) ** 2));
      v -= 7 * Math.exp(-(((i - 88) / 6) ** 2));
      data[i] = Math.min(99.1, Math.max(42, v));
    }
    const totalMean = data.reduce((a, b) => a + b, 0) / N;

    const scoreEl = $("#r-score"), framesEl = $("#r-frames"), bandEl = $("#r-band");
    let lastBand = bandOf(totalMean);
    relabelBand = () => {
      if (bandEl.textContent !== "—") {
        bandEl.textContent = I18N[lang][lastBand.key];
        bandEl.style.color = lastBand.color;
        scoreEl.style.color = lastBand.color;
      }
    };

    let W = 0, H = 0, dpr = 1;
    const PAD = { l: 34, r: 12, t: 12, b: 10 };
    const y = (v) => PAD.t + (1 - v / 100) * (H - PAD.t - PAD.b);
    const x = (i) => PAD.l + (i / (N - 1)) * (W - PAD.l - PAD.r);

    function drawFrame(upto, alpha, showMean) {
      if (!W || !H) return;
      ctx.save();
      ctx.clearRect(0, 0, W, H);
      ctx.globalAlpha = alpha;

      // quality-band zones
      const zones = [
        [93, 100, "#4EA97C0F"], [80, 93, "#9CBE5A0C"],
        [60, 80, "#D9A03F0D"], [0, 60, "#C9615F0F"]
      ];
      for (const [lo, hi, c] of zones) {
        ctx.fillStyle = c;
        ctx.fillRect(PAD.l, y(hi), W - PAD.l - PAD.r, y(lo) - y(hi));
      }

      // grid + labels
      ctx.strokeStyle = "#FFFFFF0E";
      ctx.fillStyle = "#6E7178";
      ctx.font = "9px 'JetBrains Mono', monospace";
      ctx.textAlign = "right";
      ctx.lineWidth = 1;
      for (const g of [0, 25, 50, 75, 100]) {
        ctx.beginPath();
        ctx.moveTo(PAD.l, y(g) + 0.5);
        ctx.lineTo(W - PAD.r, y(g) + 0.5);
        ctx.stroke();
        if (g % 50 === 0) ctx.fillText(String(g), PAD.l - 7, y(g) + 3);
      }

      // mean line once fully revealed
      if (showMean) {
        ctx.strokeStyle = "#6FA8DC55";
        ctx.setLineDash([5, 5]);
        ctx.beginPath();
        ctx.moveTo(PAD.l, y(totalMean));
        ctx.lineTo(W - PAD.r, y(totalMean));
        ctx.stroke();
        ctx.setLineDash([]);
      }

      // the curve
      const k = Math.max(2, upto);
      ctx.strokeStyle = "#6FA8DC";
      ctx.lineWidth = 2;
      ctx.lineJoin = "round";
      ctx.lineCap = "round";
      ctx.shadowColor = "#6FA8DC";
      ctx.shadowBlur = 5;
      ctx.beginPath();
      ctx.moveTo(x(0), y(data[0]));
      for (let i = 1; i < k; i++) ctx.lineTo(x(i), y(data[i]));
      ctx.stroke();
      ctx.shadowBlur = 0;

      // scan head
      if (upto < N) {
        const hi = upto - 1;
        const hx = x(hi), hy = y(data[hi]);
        const band = bandOf(data[hi]);
        ctx.strokeStyle = "#6FA8DC52";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(hx + 0.5, PAD.t);
        ctx.lineTo(hx + 0.5, H - PAD.b);
        ctx.stroke();
        ctx.fillStyle = band.color;
        ctx.shadowColor = band.color;
        ctx.shadowBlur = 10;
        ctx.beginPath();
        ctx.arc(hx, hy, 3.4, 0, Math.PI * 2);
        ctx.fill();
        ctx.shadowBlur = 0;
      }
      ctx.restore();
    }

    function resize() {
      const box = cv.parentElement.getBoundingClientRect();
      dpr = Math.min(2.5, window.devicePixelRatio || 1);
      W = Math.max(1, Math.round(box.width));
      H = Math.max(1, Math.round(box.height));
      cv.width = Math.round(W * dpr);
      cv.height = Math.round(H * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }
    new ResizeObserver(() => { resize(); }).observe(cv.parentElement);
    resize();

    function setReadouts(k) {
      let sum = 0;
      for (let i = 0; i < k; i++) sum += data[i];
      const mean = sum / k;
      lastBand = bandOf(mean);
      scoreEl.textContent = mean.toFixed(1);
      scoreEl.style.color = lastBand.color;
      framesEl.textContent = String(k);
      bandEl.textContent = I18N[lang][lastBand.key];
      bandEl.style.color = lastBand.color;
    }

    if (REDUCED) {
      drawFrame(N, 1, true);
      setReadouts(N);
      return;
    }

    // Sweep loop: reveal → hold → fade → restart.
    const REVEAL_MS = 5200, HOLD_MS = 3000, FADE_MS = 700;
    let phase = "reveal", t0 = performance.now(), lastK = 0;

    function loop(now) {
      const t = now - t0;
      if (phase === "reveal") {
        const p = Math.min(1, t / REVEAL_MS);
        const k = Math.max(2, Math.round(p * N));
        if (k !== lastK) { lastK = k; setReadouts(k); }
        drawFrame(k, 1, false);
        if (p >= 1) { phase = "hold"; t0 = now; }
      } else if (phase === "hold") {
        drawFrame(N, 1, true);
        if (t >= HOLD_MS) { phase = "fade"; t0 = now; }
      } else {
        const p = Math.min(1, t / FADE_MS);
        drawFrame(N, 1 - p, true);
        if (p >= 1) {
          phase = "reveal"; t0 = now; lastK = 0;
          scoreEl.textContent = "—"; framesEl.textContent = "0"; bandEl.textContent = "—";
        }
      }
      requestAnimationFrame(loop);
    }
    requestAnimationFrame(loop);
  }
  initScope();

  /* ================= terminal ================= */

  function initTerm() {
    const term = $("#term");
    if (!term) return;
    const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
    const CMD = "crossvmaf --reference ref.mp4 --distorted dist.mp4 --output result.json";

    const span = (cls, text) => {
      const s = document.createElement("span");
      if (cls) s.className = cls;
      s.textContent = text;
      return s;
    };
    const line = (...nodes) => {
      const d = document.createElement("div");
      nodes.forEach((n) => d.appendChild(n));
      term.appendChild(d);
      return d;
    };
    const cursor = () => span("t-cur", "");

    async function run() {
      term.textContent = "";
      if (REDUCED) {
        line(span("t-p", "$ "), span("", CMD));
        line(span("t-dim", "decode   auto → hardware · transfer gpu"));
        line(span("t-dim", "compute  vulkan · model vmaf_v1.0.16_3d0h_2160 (auto)"));
        line(span("t-dim", "frames   1241/1241 "), span("t-key", "▓".repeat(14) + " done"));
        line(span("t-pn", "{"));
        line(span("t-pn", "  "), span("t-key", '"pool"'), span("t-pn", ": "), span("t-str", '"mean"'), span("t-pn", ","));
        line(span("t-pn", "  "), span("t-key", '"vmaf"'), span("t-pn", ": "), span("t-num", "96.42"));
        line(span("t-pn", "}"));
        line(span("t-ok", "→ wrote result.json"));
        line(span("t-p", "$ "), cursor());
        return;
      }
      // typed command
      const cmdLine = line(span("t-p", "$ "));
      const cur = cursor();
      cmdLine.appendChild(cur);
      const txt = document.createTextNode("");
      cmdLine.insertBefore(txt, cur);
      for (const ch of CMD) {
        txt.textContent += ch;
        await sleep(16 + Math.random() * 30);
      }
      cur.remove();
      await sleep(380);
      line(span("t-dim", "decode   auto → hardware · transfer gpu"));
      await sleep(300);
      line(span("t-dim", "compute  vulkan · model vmaf_v1.0.16_3d0h_2160 (auto)"));
      await sleep(420);
      // progress
      const prog = line(span("t-dim", "frames   0/1241 "));
      const bar = span("t-key", "");
      prog.appendChild(bar);
      const FR = 1241, STEPS = 26;
      for (let s = 1; s <= STEPS; s++) {
        const n = Math.round((s / STEPS) * FR);
        prog.firstChild.textContent = "frames   " + n + "/" + FR + " ";
        bar.textContent = "▓".repeat(Math.round((s / STEPS) * 14)) + "░".repeat(14 - Math.round((s / STEPS) * 14));
        await sleep(34);
      }
      prog.firstChild.textContent = "frames   1241/1241 ";
      bar.textContent = "▓".repeat(14) + " done";
      await sleep(300);
      line(span("t-pn", "{"));
      await sleep(130);
      line(span("t-pn", "  "), span("t-key", '"pool"'), span("t-pn", ": "), span("t-str", '"mean"'), span("t-pn", ","));
      await sleep(130);
      line(span("t-pn", "  "), span("t-key", '"vmaf"'), span("t-pn", ": "), span("t-num", "96.42"));
      await sleep(130);
      line(span("t-pn", "}"));
      await sleep(220);
      line(span("t-ok", "→ wrote result.json"));
      await sleep(260);
      line(span("t-p", "$ "), cursor());
      await sleep(4600);
      run();
    }
    run();
  }
  initTerm();

  /* ================= copy button ================= */

  const copyBtn = $("#copy-btn");
  if (copyBtn) {
    const codeEl = $("#code");
    const label = copyBtn.querySelector("span");
    copyBtn.addEventListener("click", async () => {
      const text = codeEl.innerText;
      let ok = false;
      try { await navigator.clipboard.writeText(text); ok = true; }
      catch (e) {
        const ta = document.createElement("textarea");
        ta.value = text;
        ta.style.position = "fixed";
        ta.style.opacity = "0";
        document.body.appendChild(ta);
        ta.select();
        try { ok = document.execCommand("copy"); } catch (e2) {}
        ta.remove();
      }
      if (ok) {
        copyBtn.classList.add("copied");
        label.textContent = I18N[lang]["api.copied"];
        setTimeout(() => {
          copyBtn.classList.remove("copied");
          label.textContent = I18N[lang]["api.copy"];
        }, 1600);
      }
    });
  }

  /* ================= magnetic buttons ================= */

  if (FINE && !REDUCED) {
    $$(".magnetic").forEach((el) => {
      el.addEventListener("pointermove", (ev) => {
        const r = el.getBoundingClientRect();
        const dx = (ev.clientX - (r.left + r.width / 2)) / (r.width / 2);
        const dy = (ev.clientY - (r.top + r.height / 2)) / (r.height / 2);
        el.style.setProperty("--mx", (dx * 4).toFixed(1) + "px");
        el.style.setProperty("--my", (dy * 3).toFixed(1) + "px");
      });
      el.addEventListener("pointerleave", () => {
        el.style.setProperty("--mx", "0px");
        el.style.setProperty("--my", "0px");
      });
    });
  }

  /* ================= misc ================= */

  $("#year").textContent = String(new Date().getFullYear());

  applyLang(lang);
})();
