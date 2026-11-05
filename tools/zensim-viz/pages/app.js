import { VizArchive } from "./archive.js";
import { ARCHIVE_URL, SOURCE_COMMIT } from "./config.js";
const $ = (s) => document.querySelector(s),
  fmt = (n) => Number(n).toPrecision(5),
  esc = (s) =>
    String(s).replace(
      /[&<>"']/g,
      (c) =>
        ({
          "&": "&amp;",
          "<": "&lt;",
          ">": "&gt;",
          '"': "&quot;",
          "'": "&#39;",
        })[c],
    );
const worker = new Worker("./worker.js", { type: "module" }),
  pending = new Map();
let serial = 0;
const call = (op, ...args) =>
  new Promise((resolve, reject) => {
    const id = ++serial;
    pending.set(id, { resolve, reject });
    worker.postMessage({ id, op, args });
  });
const ready = new Promise((resolve, reject) => {
  worker.onmessage = ({ data: d }) => {
    if (d.ready) {
      resolve(d);
      return;
    }
    const p = pending.get(d.id);
    if (p) {
      pending.delete(d.id);
      d.error ? p.reject(Error(d.error)) : p.resolve(d.value);
    }
  };
  worker.onerror = (e) => reject(Error(e.message));
});
const state = {
  view: "atlas",
  input: 0,
  bank: "sdr",
  pins: [],
  pair: null,
  curve: null,
  map: null,
  generation: 0,
  concernPage: 0,
  unit: null,
  allConcerns: false,
  bankGeneration: 0,
  icePage: 0,
  lensMode: "input",
};
let archive,
  weights,
  audit,
  functions,
  references,
  bank,
  grids,
  modelGrids,
  rows = [],
  moduleInfo;
function status(s) {
  $("#status").textContent = s;
}
function link(i, label) {
  return `<a href="#input=${i}&view=${state.view}&bank=${state.bank}" data-input="${i}">${esc(label ?? weights.inputs[i].name)}</a>`;
}
function chosen() {
  return weights.inputs[state.input];
}
function gridFor(id) {
  const i = bank.slots.indexOf(id);
  if (i < 0) throw Error("Feature absent from bank");
  return grids.slice(i * 21, (i + 1) * 21);
}
function rebuildTree() {
  const q = $("#search").value.toLowerCase();
  rows = [];
  let f = "",
    s = "",
    scale = -1;
  const order = new Map();
  for (const i of weights.inputs)
    if (!order.has(i.family + "/" + i.signal))
      order.set(i.family + "/" + i.signal, order.size);
  const sorted = weights.inputs
    .slice()
    .sort(
      (a, b) =>
        a.family.localeCompare(b.family) ||
        order.get(a.family + "/" + a.signal) -
          order.get(b.family + "/" + b.signal) ||
        a.scale - b.scale ||
        "xyb".indexOf(a.channel) - "xyb".indexOf(b.channel) ||
        a.pooling.localeCompare(b.pooling) ||
        Number(a.factor != null) - Number(b.factor != null) ||
        a.index - b.index,
    );
  for (const i of sorted) {
    if (
      q &&
      !`${i.name} ${i.family} ${i.signal} s${i.scale} ${i.channel} input${i.index} f${i.feature} ${i.factor == null ? "raw" : "fragility f" + i.factor}`
        .toLowerCase()
        .includes(q)
    )
      continue;
    if (f !== i.family) {
      rows.push({ label: i.family, level: 0 });
      f = i.family;
      s = "";
      scale = -1;
    }
    if (s !== i.signal) {
      rows.push({ label: i.signal, level: 1 });
      s = i.signal;
      scale = -1;
    }
    if (scale !== i.scale) {
      rows.push({ label: `Scale ${i.scale}`, level: 2 });
      scale = i.scale;
    }
    rows.push({ input: i, level: 3 });
  }
  $("#feature-count").textContent =
    `${weights.inputs.length} explained model inputs · ${bank.slots.length} raw slots`;
  $("#spacer").style.height = `${rows.length * 32}px`;
  paintTree();
}
function paintTree() {
  const start = Math.max(0, Math.floor($("#tree").scrollTop / 32) - 3),
    end = Math.min(
      rows.length,
      start + Math.ceil($("#tree").clientHeight / 32) + 6,
    );
  $("#rows").style.transform = `translateY(${start * 32}px)`;
  $("#rows").innerHTML = rows
    .slice(start, end)
    .map((r) =>
      r.input
        ? `<div role="treeitem" aria-level="4" class="tree-row ${r.input.index === state.input ? "selected" : ""}" style="padding-left:20px">${link(r.input.index, `${r.input.channel}${r.input.factor == null ? "" : " × fragility"} · ${r.input.pooling}`)}<span class="chip" title="Bank p50 of this declared input before scaling">p50 ${fmt(modelGrids[r.input.index * 21 + 10])}</span></div>`
        : `<div role="treeitem" aria-level="${r.level + 1}" class="tree-row group" style="padding-left:${r.level * 8}px">${esc(r.label)}</div>`,
    )
    .join("");
}
const DETECTS = {
  "basic/ssim":
    "Local mean mismatch and variance-normalized direct error in XYB.",
  "basic/edge_art":
    "Excess distorted high-frequency magnitude, retained on the positive side of the edge ratio.",
  "basic/edge_det":
    "Lost high-frequency magnitude, retained on the negative side of the edge ratio.",
  "basic/mse": "Squared XYB sample error pooled over this scale and channel.",
  "basic/var_loss": "Global high-frequency energy loss as a ratio of moments.",
  "basic/tex_loss": "Global loss of mean absolute high-frequency magnitude.",
  "basic/contrast_inc":
    "A saturating ratio of global high-frequency energy gain.",
  "v2/ssim":
    "A bounded structural-error detector. Mean and central-moment pools summarize its distribution.",
  "v2/art":
    "Bounded high-frequency magnitude error where the distorted magnitude exceeds the reference.",
  "v2/det":
    "Bounded high-frequency magnitude error where the distorted magnitude is smaller.",
  "v2/mse": "Squared XYB sample error passed through a saturating denominator.",
  "v2/gms": "Normalized squared difference between local gradient magnitudes.",
  "v2/blockiness": "Excess step energy on the eight-pixel lattice.",
  "v2/ringing":
    "Residual error gated by reference activity and gradient response.",
  "v2/banding":
    "Distorted gradient excess weighted toward low reference gradients.",
  "v2/edge_width_change":
    "A global comparison of adjacent-scale gradient ratios.",
};
function describes(i) {
  const direct = DETECTS[`${i.family}/${i.signal}`];
  if (direct) return direct;
  const v = i.signal;
  if (v.startsWith("masked_") || v.startsWith("iw_"))
    return `Reference-activity-weighted pooling of the ${v.split("_").slice(1).join("_")} detector. ${v.startsWith("masked_") ? "Low-activity locations receive more weight." : "High-activity locations receive more weight; this is an activity proxy."}`;
  if (v.endsWith("_soft_peak"))
    return "A magnitude-weighted mean of the local detector, emphasizing larger local errors. The formula gives its weighting and denominator guard.";
  if (v.startsWith("pjnd_transducer"))
    return "A saturating absolute XYB sample error whose denominator depends on reference activity. The k in the formula controls that dependence; the name alone is not a measured visibility threshold.";
  if (v.startsWith("ssim_dev"))
    return `Spatial dispersion of bounded structural error, using ${v.endsWith("2") ? "standard deviation" : "the fourth central moment root"}.`;
  if (v.startsWith("hf_"))
    return v === "hf_gain"
      ? "Normalized excess distorted high-frequency energy."
      : v === "hf_loss"
        ? "Normalized loss of reference high-frequency energy."
        : "Normalized loss of absolute high-frequency magnitude.";
  throw Error("Unexplained registry signal " + i.family + "/" + v);
}
function unitLink(unit, input = state.input) {
  return `<a data-unit="${unit}" href="#input=${input}&view=function&unit=${unit}&bank=${state.bank}">h${unit}</a>`;
}
function explainReason(text, layer = 0) {
  return esc(text)
    .replace(/\binput (\d+)\b/g, (all, id) =>
      layer === 1
        ? unitLink(Number(id))
        : weights.inputs[Number(id)]
          ? link(Number(id), weights.inputs[Number(id)].short_name)
          : all,
    )
    .replace(/\bunit (\d+)\b/g, (all, id) =>
      layer === 1 ? "scalar output " + id : unitLink(Number(id)),
    );
}
function flagLocation(f) {
  if (f.layer === 1 && f.from != null)
    return `Output layer · ${unitLink(f.from)} → scalar output`;
  if (f.layer === 0 && f.from != null)
    return `Encoder · ${link(f.from)}${f.to == null ? "" : ` → ${unitLink(f.to, f.from)}`}`;
  if (f.layer === 0 && f.to != null) return `Hidden ${unitLink(f.to)}`;
  return esc(f.location);
}
function revealFeature() {
  let row = rows.findIndex((r) => r.input?.index === state.input);
  if (row < 0 && $("#search").value) {
    $("#search").value = "";
    rebuildTree();
    row = rows.findIndex((r) => r.input?.index === state.input);
  }
  if (row >= 0) {
    const y = row * 32,
      t = $("#tree");
    if (y < t.scrollTop || y + 32 > t.scrollTop + t.clientHeight)
      t.scrollTop = Math.max(0, y - t.clientHeight / 3);
  }
  paintTree();
}
function reference() {
  const i = chosen(),
    ref = references[`${i.family}/${i.signal}`],
    key = `${i.family}/${i.signal}`;
  $("#feature-title").textContent = i.name;
  const formula = ref?.formula ?? "\\text{See the registered implementation}";
  try {
    if (!globalThis.katex) throw Error("formula renderer unavailable");
    katex.render(formula, $("#formula"), {
      throwOnError: false,
      displayMode: true,
    });
  } catch (e) {
    $("#formula").textContent = formula;
  }
  $("#feature-detail").innerHTML =
    `<p><code>${esc(i.family)} / ${esc(i.signal)} / s${i.scale} / ${esc(i.channel)}</code></p><p>${esc(describes(i))}<br>Pool: ${esc(i.pooling)}. Registry slot f${i.feature}${i.factor == null ? "" : ` × fragility f${i.factor}`}.</p><p>${i.factor == null ? "Direct raw detector input." : "The bake multiplies damage by its paired fragility feature before scaling. A raw-feature intervention recomputes this join."}</p><p>${ref?.code ? `<a href="${esc(typeof ref.code === "string" ? ref.code : (ref.code.url ?? "https://github.com/imazen/zensimx"))}">Registered implementation</a>` : ""}</p>`;
  if (i.factor != null) {
    const product = document.createElement("div"),
      factor = document.createElement("div");
    $("#feature-detail").append(product, factor);
    if (globalThis.katex) {
      katex.render(references.fragility.formula, product, {
        throwOnError: false,
        displayMode: true,
      });
      katex.render(bank.fragility_formula, factor, {
        throwOnError: false,
        displayMode: true,
      });
    } else {
      product.textContent = references.fragility.formula;
      factor.textContent = bank.fragility_formula;
    }
    $("#feature-detail").insertAdjacentHTML(
      "beforeend",
      `<p>Fragility uses the mean reference gradient magnitude at this scale/channel. It is a global reference-only factor in [0,1].</p>`,
    );
  }
  if (ref?.inspiration?.length)
    $("#feature-detail").innerHTML +=
      `<h3>Related methods</h3>` +
      ref.inspiration
        .map(
          (c) =>
            `<p><a href="${esc(c.url)}">${esc(c.author_year)} · ${esc(c.title)}</a></p>`,
        )
        .join("");
  if (state.summary)
    $("#feature-detail").innerHTML +=
      `<p>Current raw f${i.feature}: <code>${fmt(state.summary.features[i.feature])}</code>${i.factor == null ? "" : ` · fragility f${i.factor}: <code>${fmt(state.summary.features[i.factor])}</code>`}</p>`;
  const grid = gridFor(i.feature),
    modelGrid = modelGrids.slice(i.index * 21, (i.index + 1) * 21);
  $("#standing").innerHTML =
    `<h3>Standing in ${state.bank.toUpperCase()} TRAIN</h3><p>${bank.population_rows.toLocaleString()} population rows · ${bank.rows} observed sample rows</p><p>Declared input dᵢ: p0 ${fmt(modelGrid[0])} · p50 ${fmt(modelGrid[10])} · p100 ${fmt(modelGrid[20])}</p><p>Raw detector axis: p0 ${fmt(grid[0])} · p50 ${fmt(grid[10])} · p100 ${fmt(grid[20])}</p><p>Encoder row L1 ${fmt(weights.input_l1[i.index])}; ${weights.nonzero_counts[i.index]} nonzero incoming weights</p><p>Row median ${fmt(weights.layers[0].stats.rows[i.index][0])}; MAD ${fmt(weights.layers[0].stats.rows[i.index][1])}. ${weights.layers[0].stats.rows[i.index][1] === 0 ? "Robust z is undefined when MAD is zero." : "Native robust z uses the recorded median and MAD."}</p><p>Scaler mean ${fmt(weights.scaler_mean[i.index])}; scale ${fmt(weights.scaler_scale[i.index])}</p><h3>Largest stored connections</h3><p>${weights.per_input_columns[
      i.index
    ]
      .slice(0, 16)
      .map(
        (h) =>
          `${unitLink(h)} <code>${fmt(weights.layers[0].weights[i.index * 128 + h])}</code>`,
      )
      .join(" · ")}</p>`;
  const flags = audit.flags.filter(
    (f) =>
      state.allConcerns ||
      (f.layer === 0 &&
        f.from === i.index &&
        (state.unit == null || f.to == null || f.to === state.unit)) ||
      (state.unit != null &&
        ((f.layer === 0 && f.from == null && f.to === state.unit) ||
          (f.layer === 1 && f.from === state.unit))),
  );
  const rank = { critical: 0, serious: 1, warning: 2, note: 3 };
  flags.sort((a, b) => rank[a.severity] - rank[b.severity]);
  state.concernPage = Math.min(
    state.concernPage,
    Math.max(0, Math.ceil(flags.length / 50) - 1),
  );
  const start = state.concernPage * 50;
  const connection =
    state.unit == null
      ? ""
      : `<p>Selected connection → h${state.unit}: <code>${weights.layers[0].weights[i.index * 128 + state.unit]}</code></p>`;
  $("#concerns").innerHTML =
    connection +
    `<h3>SDR TRAIN audit · ${flags.length} concerns</h3><button data-all-concerns>${state.allConcerns ? "Selected feature" : "All model concerns"}</button><p>${audit.flags.length} native flags retained; observational audit statistics use SDR TRAIN.</p><button data-concern-page="${Math.max(0, state.concernPage - 1)}">Previous 50</button> <button data-concern-page="${Math.min(Math.max(0, Math.ceil(flags.length / 50) - 1), state.concernPage + 1)}">Next 50</button>` +
    flags
      .slice(start, start + 50)
      .map(
        (f) =>
          `<p data-flag-layer="${f.layer}" data-flag-from="${f.from}" data-flag-to="${f.to}" class="severity-${esc(f.severity)}"><strong>${esc(f.severity)} · ${esc(f.kind)}</strong><br>${explainReason(f.reason, f.layer)}<br><code title="${esc(f.location)}">${flagLocation(f)} · ${fmt(f.value)}</code><br>Statistic ${f.statistic == null ? "undefined" : fmt(f.statistic)} · threshold ${fmt(f.threshold)}</p>`,
      )
      .join("");
  if (state.view === "lens" && state.curve?.features.includes(i.feature)) {
    const rises = state.curve.rise_flags,
      notes = rises.filter((f) => f.severity === "note").length,
      visible = rises
        .filter((f) => f.severity !== "note")
        .sort((a, b) => rank[a.severity] - rank[b.severity]);
    state.icePage = Math.min(
      state.icePage,
      Math.max(0, Math.ceil(visible.length / 50) - 1),
    );
    $("#concerns").innerHTML +=
      `<h3>ICE rises</h3><p>${notes} floating-point rises (notes); ${rises.length - notes} larger rises. Every witness is retained in the current curve.</p><button data-ice-page="${Math.max(0, state.icePage - 1)}">Previous 50 rises</button><button data-ice-page="${Math.min(Math.max(0, Math.ceil(visible.length / 50) - 1), state.icePage + 1)}">Next 50 rises</button>` +
      visible
        .slice(state.icePage * 50, state.icePage * 50 + 50)
        .map(
          (f) =>
            `<p class="severity-${f.severity}">${f.severity} · population row ${bank.row_indices[f.row]}, grid ${f.point} · Δ ${fmt(f.delta)}</p>`,
        )
        .join("");
  }
}
function fitCanvas() {
  const c = $("#chart"),
    e = $("#exhibit"),
    r = e.classList.contains("matrix")
      ? 1
      : Math.min(e.clientWidth / c.width, e.clientHeight / c.height);
  c.style.width = `${c.width * r}px`;
  c.style.height = `${c.height * r}px`;
}
new ResizeObserver(fitCanvas).observe($("#exhibit"));
function canvas(w, h) {
  $("#exhibit").classList.remove("matrix");
  const c = $("#chart");
  c.width = w;
  c.height = h;
  c.hidden = false;
  $("#table").hidden = true;
  $("#empty").hidden = true;
  fitCanvas();
  return c.getContext("2d");
}
function css(name) {
  return getComputedStyle(document.documentElement)
    .getPropertyValue(name)
    .trim();
}
function rgb(name) {
  const h = css(name).slice(1);
  return [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16));
}
function ramp(t, colors) {
  const x = Math.max(0, Math.min(1, t)) * (colors.length - 1),
    a = Math.floor(x),
    b = Math.min(colors.length - 1, a + 1);
  return colors[a].map((v, i) => Math.round(v + (colors[b][i] - v) * (x - a)));
}

function drawMap() {
  if (!state.pair) {
    $("#chart").hidden = true;
    $("#table").hidden = true;
    $("#empty").hidden = false;
    return;
  }
  const m = state.map;
  if (!m) return;
  if (!m.values.length) {
    $("#chart").hidden = true;
    $("#table").hidden = true;
    $("#empty").hidden = false;
    $("#empty").textContent = m.role;
    status(m.role);
    return;
  }
  const { w, h, reference } = state.pair,
    ctx = canvas(w, h),
    base = new ImageData(new Uint8ClampedArray(reference), w, h);
  if (!$("#map-only").checked) {
    for (let i = 0; i < base.data.length; i += 4) {
      const g = Math.round(
        0.2126 * base.data[i] +
          0.7152 * base.data[i + 1] +
          0.0722 * base.data[i + 2],
      );
      base.data[i] = base.data[i + 1] = base.data[i + 2] = g;
    }
    ctx.putImageData(base, 0, 0);
  } else ctx.clearRect(0, 0, w, h);
  const c = document.createElement("canvas");
  c.width = m.width;
  c.height = m.height;
  const cctx = c.getContext("2d"),
    pixels = cctx.createImageData(m.width, m.height);
  let range = m.percentile_range;
  if ($("#range").value === "own") range = m.range;
  if ($("#range").value === "bank")
    range = bank.ranges_p1_p99[bank.slots.indexOf(chosen().feature)];
  const [lo, hi] = range,
    floor = Number($("#alpha").value),
    colors = Array.from({ length: 7 }, (_, i) => rgb("--ramp-seq-" + i));
  for (let k = 0; k < m.values.length; k++) {
    const v = m.values[k],
      t = hi === lo ? 0 : Math.max(0, Math.min(1, (v - lo) / (hi - lo)));
    pixels.data.set(
      [
        ...ramp(t, colors),
        $("#map-only").checked
          ? 255
          : v === 0
            ? 0
            : Math.round(255 * (floor + (1 - floor) * t)),
      ],
      k * 4,
    );
  }
  cctx.putImageData(pixels, 0, 0);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(c, 0, 0, w, h);
  if (chosen().family === "basic") {
    let busy = false;
    $("#chart").onmousemove = async (e) => {
      if (busy || state.view !== "atlas") return;
      busy = true;
      try {
        const r = e.currentTarget.getBoundingClientRect(),
          x = Math.min(
            m.width - 1,
            Math.max(0, Math.floor(((e.clientX - r.left) * m.width) / r.width)),
          ),
          y = Math.min(
            m.height - 1,
            Math.max(
              0,
              Math.floor(((e.clientY - r.top) * m.height) / r.height),
            ),
          );
        const terms = await call("basic", chosen().feature, x, y);
        let pane = $("#cursor-formula");
        if (!pane) {
          pane = document.createElement("div");
          pane.id = "cursor-formula";
          $("#exhibit").append(pane);
        }
        pane.style.left = `${Math.max(8, Math.min(innerWidth - 300, e.clientX + 16))}px`;
        pane.style.top = `${Math.max(64, Math.min(innerHeight - 550, e.clientY + 16))}px`;
        pane.innerHTML = `<h3>Formula at (${x}, ${y})</h3><p>Native f32 operands · retained d₁ replay is bit-identical</p><table>${terms.names.map((name, i) => `<tr><td>${name}</td><td>${fmt(terms.values[i])}</td></tr>`).join("")}</table>`;
      } catch (e) {
        status(String(e));
      } finally {
        busy = false;
      }
    };
  }

  status(
    `${m.role} · grid ${m.width} × ${m.height} · range ${fmt(lo)} … ${fmt(hi)} · ${$("#map-only").checked ? "opaque map-only" : "exact zero is transparent"}`,
  );
}
function paths(ctx, xs, ys, color, width = 1) {
  ctx.beginPath();
  xs.forEach((x, j) => (j ? ctx.lineTo(x, ys[j]) : ctx.moveTo(x, ys[j])));
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.stroke();
}
function drawCurve() {
  const d = state.curve;
  if (!d) return;
  const ctx = canvas(960, 640);
  ctx.fillStyle = css("--surface-0");
  ctx.fillRect(0, 0, 960, 640);
  const centered = $("#centered").checked;
  // Extents only affect chart geometry; all scores, bands and flags come from Wasm.
  const ys = centered ? d.centered_ice : d.ice;
  let lo = Infinity,
    hi = -Infinity;
  for (const row of ys)
    for (const y of row) {
      lo = Math.min(lo, y);
      hi = Math.max(hi, y);
    }
  if (lo === hi) {
    lo -= 1;
    hi += 1;
  }
  const xx = d.x.map((_, j) => 70 + (j * 840) / (d.x.length - 1)),
    yy = (r) => r.map((v) => 570 - ((v - lo) / (hi - lo)) * 510);
  for (let i = 0; i < ys.length; i += 5)
    paths(ctx, xx, yy(ys[i]), css("--ink-3") + "55");
  const mean = centered ? d.centered_mean : d.mean,
    low = yy(centered ? d.centered_p25 : d.p25),
    high = yy(centered ? d.centered_p75 : d.p75);
  ctx.beginPath();
  xx.forEach((x, i) => (i ? ctx.lineTo(x, low[i]) : ctx.moveTo(x, low[i])));
  for (let i = xx.length - 1; i >= 0; i--) ctx.lineTo(xx[i], high[i]);
  ctx.closePath();
  ctx.fillStyle = css("--series-1") + "33";
  ctx.fill();
  paths(ctx, xx, yy(mean), css("--series-1"), 3);
  ctx.fillStyle = css("--ink-2");
  ctx.font = "14px system-ui";
  ctx.fillText(
    `${centered ? "Change from p0" : "Served score"} · 200 ICE traces retained, 40 drawn`,
    70,
    28,
  );
  ctx.fillText(fmt(hi), 12, 65);
  ctx.fillText(fmt(lo), 12, 575);
  ctx.fillText(
    `${d.family_strength ? "Strength" : "p0"} ${fmt(d.x[0])}`,
    70,
    615,
  );
  ctx.fillText(
    `${d.family_strength ? "Strength" : "p100"} ${fmt(d.x.at(-1))}`,
    760,
    615,
  );
  const counts = {};
  for (const f of d.rise_flags)
    counts[f.severity] = (counts[f.severity] ?? 0) + 1;
  status(
    `Native forward · ${d.rows} × ${d.x.length} scores · ${d.family_strength ? "joint strength intervention" : "raw f" + d.raw_feature + " intervention"}, products recomputed · rises ${JSON.stringify(counts)}`,
  );
}
function hingePlot(pair) {
  const points = pair.slice.points,
    keys =
      state.functionMode === "network"
        ? ["network"]
        : state.functionMode === "served"
          ? ["served"]
          : ["branch_a", "branch_b", "term_sum"];
  let lo = Infinity,
    hi = -Infinity;
  for (const p of points)
    for (const k of keys) {
      lo = Math.min(lo, p[k]);
      hi = Math.max(hi, p[k]);
    }
  if (lo === hi) hi = lo + 1;
  const x = (v) =>
      60 + (680 * (v - points[0].x)) / (points.at(-1).x - points[0].x || 1),
    y = (v) => 220 - (180 * (v - lo)) / (hi - lo);
  return `<svg class="function-plot" viewBox="0 0 800 270" role="img" aria-label="Native pair hinge and basis slice" style="width:100%;max-height:240px"><path d="M60 30V220H750" fill="none" stroke="var(--ink-3)"/>${keys.map((k, i) => `<polyline points="${points.map((p) => `${x(p.x)},${y(p[k])}`).join(" ")}" fill="none" stroke="var(--series-${i + 1})" stroke-width="2"/>`).join("")}<path d="M${x(pair.slice.threshold)} 30V220" stroke="var(--ink-2)" stroke-dasharray="4 4"/><g fill="var(--ink-2)" font-size="12"><text x="60" y="255">${fmt(points[0].x)}</text><text x="690" y="255">${fmt(points.at(-1).x)}</text><text x="${x(pair.slice.threshold) + 5}" y="25">hinge ${fmt(pair.slice.threshold)}</text><text x="5" y="45">${fmt(hi)}</text><text x="5" y="220">${fmt(lo)}</text></g></svg><p>Raw axis ${link(pair.slice.input)} · ${keys.length === 3 ? "cA / cB / paired contribution" : "Complete " + keys[0]}<br><small>Mathematical slice; other raw slots and fragility are zero. Not PDP.</small></p>`;
}
function drawFunction() {
  $("#chart").hidden = true;
  $("#empty").hidden = true;
  $("#table").hidden = false;
  const bg = functions.linear_background,
    table = (heads, body) =>
      `<table><thead><tr>${heads.map((h) => `<th>${h}</th>`).join("")}</tr></thead><tbody>${body}</tbody></table>`;
  const pairs = functions.pairs
    .map(
      (p) =>
        `<h3>Pair h${p.units.join(" / h")} · c ≈ ${fmt(p.coefficient)} · hinge A = B</h3>${hingePlot(p)}<details><summary>Both coefficient planes, ${p.coefficients.length} explained inputs</summary><p>A = Σ aᵢzᵢ; B = Σ bᵢzᵢ + ${fmt(p.bias)}.</p>${table(["Explained feature", "a", "b", "b − a"], p.coefficients.map((c) => `<tr><td>${link(c.input)}</td><td>${fmt(c.a)}</td><td>${fmt(c.b)}</td><td>${fmt(c.delta)}</td></tr>`).join(""))}<p>${esc(p.slice.definition)}</p></details>`,
    )
    .join("");
  $("#table").innerHTML =
    `<h2>The approximating function</h2><p class="mono">Head p(z) ≈ ${fmt(functions.output_bias)} + L(z) + Σ c max(A(z), B(z))</p>${pairs}<details><summary>Domain, preprocessing and memorization limits</summary><p>${esc(functions.preprocessing)}</p><p>${functions.pairs.length} max pair(s); ${bg.units.length} always-active background units; ${functions.other_hinge_count} other nonzero hinges. At most ${functions.global_affine_forms_upper_bound} affine forms and ${functions.certified_line_segments_upper_bound} pieces along a line on the certified domain, before calibration.</p><p>${esc(functions.memorization)}</p><p>${esc(functions.algebra_scope)}</p><p>${esc(functions.global_bound_scope)}</p><p>${esc(functions.certified_line_bound_scope)}</p></details><details><summary>Complete affine background L(z)</summary><p>L(z) ≈ ${fmt(bg.bias)} + Σ ℓᵢzᵢ. ${esc(bg.proof)}</p>${table(["Explained feature", "Coefficient"], bg.coefficients.map((c, i) => `<tr><td>${link(i)}</td><td>${fmt(c)}</td></tr>`).join(""))}</details><details><summary>Native output calibration · ${functions.output.spline?.x.length ?? 0} spline knots</summary><p>Tanh pin scale: ${esc(functions.output.tanh_pin_scale ?? "absent")}. Codec calibration: ${functions.output.codec_calibration_present}. ${esc(functions.output.scope)}</p>${table(["Network", "Calibrated", "Derivative"], (functions.output.spline?.x ?? []).map((x, j) => `<tr><td>${fmt(x)}</td><td>${fmt(functions.output.spline.y[j])}</td><td>${fmt(functions.output.spline.derivatives[j])}</td></tr>`).join(""))}</details>`;
  if (state.unit != null) {
    const u = weights.capacity.units[state.unit],
      incoming = weights.top_inputs_by_unit[state.unit];
    $("#table").insertAdjacentHTML(
      "beforeend",
      `<h3>Selected h${state.unit} · ${esc(u.status)}</h3><p>Output coefficient ${fmt(u.output_coefficient)} · bias ${fmt(u.bias)} · ${u.encoder_nonzero} nonzero encoder weights.</p><details><summary>Every explained incoming weight</summary>${table(["Explained feature", "Stored weight"], incoming.map((i) => `<tr><td>${link(i)}</td><td>${fmt(weights.layers[0].weights[i * 128 + state.unit])}</td></tr>`).join(""))}</details>`,
    );
  }
  status(
    "Native head and calibration; structural bounds do not establish absence of memorization.",
  );
}

function drawWeights() {
  const layer = weights.layers[0],
    at = weights.row_order.indexOf(state.input),
    start = Math.max(0, at - 16),
    inputs = weights.row_order
      .slice(start, start + 40)
      .map((i) => weights.inputs[i]),
    columns = weights.column_order,
    cell = 16,
    header = 64,
    ctx = canvas(128 * cell, inputs.length * cell + header);
  $("#exhibit").classList.add("matrix");
  fitCanvas();
  const labels = document.createElement("div");
  labels.id = "matrix-labels";
  labels.innerHTML =
    '<div style="height:64px">Explained feature rows</div>' +
    inputs
      .map(
        (i) =>
          `<div title="${esc(i.name)}">${link(i.index, `${i.pooling}${i.factor == null ? "" : " ×b"} · ${i.short_name}`)}</div>`,
      )
      .join("");
  $("#exhibit").insertBefore(labels, $("#chart"));
  const cellFlags = new Map();
  for (const flag of audit.flags)
    if (flag.layer === 0 && flag.from != null && flag.to != null) {
      const key = `${flag.from}:${flag.to}`;
      if (!cellFlags.has(key)) cellFlags.set(key, []);
      cellFlags.get(key).push(flag);
    }

  const magnitude = layer.abs_max;
  if (!(magnitude > 0)) throw Error("Missing native weight extent");
  const negative = [
      "--ramp-div-0",
      "--ramp-div-n1",
      "--ramp-div-n2",
      "--ramp-div-n3",
    ].map(rgb),
    positive = [
      "--ramp-div-0",
      "--ramp-div-p1",
      "--ramp-div-p2",
      "--ramp-div-p3",
    ].map(rgb);
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  ctx.fillStyle = css("--ink-2");
  ctx.font = "11px monospace";
  ctx.fillText(
    "Native energy order · stored unit IDs · 14 px cells + 2 px hover targets",
    4,
    12,
  );
  for (let p = 0; p < 128; p++) {
    ctx.save();
    ctx.translate(p * cell + 11, 58);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText("h" + columns[p], 0, 0);
    ctx.restore();
  }
  for (let r = 0; r < inputs.length; r++)
    for (let p = 0; p < 128; p++) {
      const h = columns[p];
      const v = layer.weights[inputs[r].index * 128 + h],
        t = Math.log1p(Math.abs(v) * 1e6) / Math.log1p(magnitude * 1e6);
      ctx.fillStyle =
        v === 0
          ? css("--ramp-div-0")
          : `rgb(${ramp(t, v < 0 ? negative : positive).join(",")})`;
      ctx.fillRect(p * cell, r * cell + header, 14, 14);
      if (cellFlags.has(`${inputs[r].index}:${h}`)) {
        ctx.fillStyle = css("--ink-2");
        ctx.fillRect(p * cell + 10, r * cell + header, 3, 3);
      }
    }
  ctx.strokeStyle = css("--ink");
  ctx.strokeRect(0, (at - start) * cell + header, 128 * cell, cell);
  $("#chart").onclick = (e) => {
    const rect = e.currentTarget.getBoundingClientRect(),
      x = ((e.clientX - rect.left) * ctx.canvas.width) / rect.width,
      y = ((e.clientY - rect.top) * ctx.canvas.height) / rect.height;
    const h = columns[Math.floor(x / cell)],
      r = Math.floor((y - header) / cell);
    if (inputs[r] && h != null)
      location.hash = `input=${inputs[r].index}&unit=${h}&view=weights&bank=${state.bank}`;
  };
  status(
    `${audit.weight_count} audited weights · all ${audit.flags.length} measured flags retained · selected input ${chosen().name}`,
  );
  $("#chart").onmousemove = (e) => {
    const rect = e.currentTarget.getBoundingClientRect(),
      x = ((e.clientX - rect.left) * ctx.canvas.width) / rect.width,
      y = ((e.clientY - rect.top) * ctx.canvas.height) / rect.height;
    const h = columns[Math.floor(x / cell)],
      r = Math.floor((y - header) / cell);
    if (inputs[r] && h != null)
      e.currentTarget.title = `${inputs[r].name} → h${h}: ${layer.weights[inputs[r].index * 128 + h]}\nRow z: ${layer.stats.row_z[inputs[r].index * 128 + h] ?? "undefined"} · column z: ${layer.stats.column_z[inputs[r].index * 128 + h] ?? "undefined"} · threshold ${layer.stats.modified_z_threshold}\n${(cellFlags.get(`${inputs[r].index}:${h}`) ?? []).map((f) => `${f.kind}: ${f.reason}`).join("\n")}`;
  };
}
async function render() {
  const generation = ++state.generation;
  $("#matrix-labels")?.remove();
  reference();
  revealFeature();
  $("#stage-title").textContent =
    state.view === "function"
      ? "A · max pairs, planes and capacity"
      : `${chosen().name} · ${state.view}`;
  for (const a of document.querySelectorAll("#capacity a")) {
    const h = new URLSearchParams(a.hash.slice(1)).get("unit");
    a.href = `#input=${state.input}&view=function&unit=${h}&bank=${state.bank}`;
  }
  for (const b of $("#tabs").children)
    b.setAttribute("aria-pressed", b.dataset.view === state.view);
  $("#chart").onmousemove = null;
  $("#chart").onclick = null;
  $("#chart").onmouseleave = () => $("#cursor-formula")?.remove();
  $("#cursor-formula")?.remove();
  $("#center-label").hidden = state.view !== "lens";
  $("#lens-label").hidden = state.view !== "lens";
  $("#download").hidden = state.view !== "lens" || !state.curve;
  $("#function-label").hidden = state.view !== "function";
  $("#unit-label").hidden = !["weights", "function"].includes(state.view);
  $("#capacity").hidden = !["weights", "function"].includes(state.view);
  $("#unit").value = state.unit ?? "";
  for (const control of document.querySelectorAll("[data-atlas]"))
    control.hidden = state.view !== "atlas";
  $(".legend").textContent =
    state.view === "weights"
      ? "− blue · zero gray · + red · log"
      : state.view === "function"
        ? state.functionMode && state.functionMode !== "pair"
          ? "Complete native " + state.functionMode
          : "cA · cB · paired contribution"
        : state.view === "lens"
          ? "40 ICE · IQR · PDP"
          : "Low ▰▰▰ High";
  try {
    if (state.view === "atlas") {
      if (state.pair) {
        status("Evaluating detector in Wasm…");
        const map = await call("map", chosen().feature);
        if (generation !== state.generation) return;
        state.map = map;
      }
      drawMap();
    } else if (state.view === "lens") await renderLens(generation);
    else if (state.view === "function") drawFunction();
    else drawWeights();
  } catch (e) {
    status(String(e));
    throw e;
  }
}
function lensFeatures() {
  return [
    ...new Set(
      weights.inputs
        .filter(
          (i) =>
            i.family === chosen().family &&
            (state.lensMode !== "signal" || i.signal === chosen().signal),
        )
        .map((i) => i.feature),
    ),
  ];
}
async function renderLens(generation) {
  const mode = state.lensMode,
    feature = chosen().feature,
    features = ["signal", "family"].includes(mode) ? lensFeatures() : [feature];
  if (mode === "quantiles") {
    $("#chart").hidden = true;
    $("#empty").hidden = true;
    $("#table").hidden = false;
    const raw = gridFor(feature),
      derived = modelGrids.slice(state.input * 21, (state.input + 1) * 21);
    $("#table").innerHTML =
      "<h2>Population quantiles before scaling</h2><p>Full admitted TRAIN population; raw detector and declared model input include the native fragility join.</p><table><tr><th>Percentile</th><th>Raw detector</th><th>Declared input</th></tr>" +
      Array.from(
        raw,
        (v, j) =>
          `<tr><td>p${j * 5}</td><td>${fmt(v)}</td><td>${fmt(derived[j])}</td></tr>`,
      ).join("") +
      "</table>";
    status("Native full-population quantiles; no human ratings.");
    return;
  }
  if (mode === "diagnostics") {
    status("Computing identity ablation and empirical separation…");
    const d = await call("diagnostics", feature, Uint32Array.from(features));
    if (generation !== state.generation) return;
    $("#chart").hidden = true;
    $("#empty").hidden = true;
    $("#table").hidden = false;
    $("#table").innerHTML =
      `<h2>Raw damage ablation</h2><p>${esc(d.definition)}</p><p>200 observed rows · mean absolute served-score change ${fmt(d.mean_abs_delta)} · median ${fmt(d.median_delta)} · p10 ${fmt(d.p10)} · p90 ${fmt(d.p90)}</p><h3>Served-score separation</h3><table><tr><th>Raw range</th><th>Rows</th><th>Score p10</th><th>Median</th><th>p90</th></tr>${d.separation.map((b) => `<tr><td>${fmt(b.low)} … ${fmt(b.high)}</td><td>${b.count}</td><td>${fmt(b.p10)}</td><td>${fmt(b.median)}</td><td>${fmt(b.p90)}</td></tr>`).join("")}</table>`;
    status(
      "Native bake scores, not human ratings; overlapping effects are not additive.",
    );
    return;
  }
  status("Computing 200 native ICE curves…");
  const curve = await (["signal", "family"].includes(mode)
    ? call("family", Uint32Array.from(features))
    : call("curve", feature, gridFor(feature)));
  if (generation !== state.generation) return;
  state.curve = curve;
  reference();
  $("#download").hidden = false;
  if (mode === "table") {
    $("#chart").hidden = true;
    $("#empty").hidden = true;
    $("#table").hidden = false;
    $("#table").innerHTML =
      "<h2>Computed curve</h2><p>200 native ICE traces and every rise witness are available in Download computed curve.</p><table><tr><th>Raw value</th><th>PDP</th><th>p25</th><th>p75</th><th>Centered PDP</th></tr>" +
      curve.x
        .map(
          (x, j) =>
            `<tr><td>${fmt(x)}</td><td>${fmt(curve.mean[j])}</td><td>${fmt(curve.p25[j])}</td><td>${fmt(curve.p75[j])}</td><td>${fmt(curve.centered_mean[j])}</td></tr>`,
        )
        .join("") +
      "</table>";
    status("Exact native computed values; displayed decimals are rounded.");
  } else drawCurve();
}
function hash() {
  const h = new URLSearchParams(location.hash.slice(1)),
    i = Number(h.get("input") ?? 0);
  if (state.input !== i) {
    state.concernPage = 0;
    state.allConcerns = false;
  }
  state.input = Number.isInteger(i) && weights.inputs[i] ? i : 0;
  const unit = h.get("unit");
  state.unit =
    unit != null && Number(unit) >= 0 && Number(unit) < 128
      ? Number(unit)
      : null;
  state.view = ["atlas", "lens", "function", "weights"].includes(h.get("view"))
    ? h.get("view")
    : "atlas";
  return h;
}
async function setBank(name) {
  const generation = ++state.bankGeneration;
  ++state.generation;
  status("Loading compact bank from the archive…");
  const next = await archive.json(`${name}/bank.json`);
  if (next.exporter_metric_source_sha256 !== moduleInfo.metricSourceHash)
    throw Error("Bank export and module metric sources differ");
  const [b, derived, s] = await Promise.all(
    ["quantiles.f32le", "model-quantiles.f32le", "sample.f32le"].map((file) =>
      archive.bytes(`${name}/${file}`),
    ),
  );
  if (generation !== state.bankGeneration) return false;
  await call(
    "bank",
    Uint32Array.from(next.slots),
    new Float32Array(s.buffer, s.byteOffset, s.byteLength / 4),
  );
  if (generation !== state.bankGeneration) return false;
  bank = next;
  grids = new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4);
  modelGrids = new Float32Array(
    derived.buffer,
    derived.byteOffset,
    derived.byteLength / 4,
  );
  state.bank = name;
  state.curve = null;
  rebuildTree();
  return true;
}
async function pair(p) {
  status("Extracting browser-local pair in Wasm…");
  const start = performance.now(),
    summary = await call("pair", p.reference, p.distorted, p.w, p.h, p.w * 4);
  if (
    summary.bake_sha256 !==
    archive.catalog.entries.find((e) => e.name === "A.bin")?.sha256
  )
    throw Error("Module and archive bake identities differ");
  state.pair = p;
  $("#score").textContent =
    `A ${fmt(summary.score)}${p.kind === "synthetic" ? " · synthetic demo" : ""}`;
  state.summary = summary;
  state.pairMs = performance.now() - start;
  await render();
}
async function decode(file) {
  const image = await createImageBitmap(file, {
      imageOrientation: "from-image",
    }),
    c = document.createElement("canvas");
  c.width = image.width;
  c.height = image.height;
  c.getContext("2d").drawImage(image, 0, 0);
  const data = c.getContext("2d").getImageData(0, 0, c.width, c.height).data;
  image.close();
  return { w: c.width, h: c.height, data };
}
async function readPair() {
  const a = $("#reference-file").files[0],
    b = $("#distorted-file").files[0];
  if (!a || !b) return;
  const [r, d] = await Promise.all([decode(a), decode(b)]);
  if (r.w !== d.w || r.h !== d.h) throw Error("Image dimensions must match");
  await pair({ w: r.w, h: r.h, reference: r.data, distorted: d.data });
}
$("#search").oninput = rebuildTree;
$("#tree").onscroll = paintTree;
$("#tree").onkeydown = (e) => {
  if (e.key !== "ArrowDown" && e.key !== "ArrowUp") return;
  e.preventDefault();
  const entries = rows.filter((r) => r.input).map((r) => r.input.index),
    at = entries.indexOf(state.input),
    next =
      entries[
        Math.max(
          0,
          Math.min(entries.length - 1, at + (e.key === "ArrowDown" ? 1 : -1)),
        )
      ];
  if (next != null)
    location.hash = `input=${next}&view=${state.view}&bank=${state.bank}`;
};
document.addEventListener("click", (e) => {
  const ice = e.target.closest("[data-ice-page]");
  if (ice) {
    state.icePage = Number(ice.dataset.icePage);
    reference();
    return;
  }
  if (e.target.closest("[data-all-concerns]")) {
    state.allConcerns = !state.allConcerns;
    state.concernPage = 0;
    reference();
    return;
  }
  const page = e.target.closest("[data-concern-page]");
  if (page) {
    state.concernPage = Number(page.dataset.concernPage);
    reference();
    return;
  }
  const a = e.target.closest("[data-input]");
  if (a) {
    e.preventDefault();
    location.hash = `input=${a.dataset.input}&view=${state.view}&bank=${state.bank}`;
  }
});
$("#tabs").onclick = (e) => {
  const v = e.target.dataset.view;
  if (v) location.hash = `input=${state.input}&view=${v}&bank=${state.bank}`;
};
$("#theme").onclick = () => {
  const light = document.documentElement.dataset.theme !== "light";
  document.documentElement.dataset.theme = light ? "light" : "dark";
  $("#theme").textContent = light ? "Dark" : "Light";
  render();
};
$("#bank").onchange = async () => {
  if (!(await setBank($("#bank").value))) return;
  location.hash = `input=${state.input}&view=${state.view}&bank=${state.bank}`;
  await render();
};
for (const s of ["#range", "#alpha", "#map-only"])
  $(s).oninput = () => {
    if (state.view === "atlas") drawMap();
  };
$("#centered").onchange = drawCurve;
$("#function-mode").onchange = () => {
  state.functionMode = $("#function-mode").value;
  $(".legend").textContent =
    state.functionMode === "pair"
      ? "cA · cB · paired contribution"
      : "Complete native " + state.functionMode;
  drawFunction();
};
$("#lens-mode").onchange = () => {
  state.lensMode = $("#lens-mode").value;
  render();
};
$("#unit").onchange = () => {
  location.hash =
    `input=${state.input}&view=${state.view}&bank=${state.bank}` +
    ($("#unit").value === "" ? "" : `&unit=${$("#unit").value}`);
};
$("#download").onclick = () => {
  if (!state.curve) return;
  const a = document.createElement("a"),
    url = URL.createObjectURL(
      new Blob([JSON.stringify(state.curve)], { type: "application/json" }),
    );
  a.href = url;
  a.download = `A-${state.bank}-f${chosen().feature}-computed.json`;
  a.click();
  URL.revokeObjectURL(url);
};
$("#pins").textContent = "Enter to pin";
$("#pin").onclick = () => {
  if (state.pins.includes(state.input))
    state.pins = state.pins.filter((i) => i !== state.input);
  else state.pins.push(state.input);
  $("#pins").innerHTML =
    state.pins.map((i) => link(i, weights.inputs[i].short_name)).join("") ||
    "Enter to pin";
};
document.addEventListener("keydown", (e) => {
  if (["INPUT", "SELECT", "TEXTAREA", "BUTTON", "A"].includes(e.target.tagName))
    return;
  if (e.key === "/") {
    e.preventDefault();
    $("#search").focus();
  }
  if (e.key === "Enter") {
    e.preventDefault();
    $("#pin").click();
  }
});
for (const s of ["#reference-file", "#distorted-file"])
  $(s).onchange = () => readPair().catch((e) => status(String(e)));
function syntheticPair() {
  const w = 256,
    h = 192,
    r = new Uint8Array(w * h * 4),
    d = r.slice();
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      const v = ((x * 7 + y * 13) % 200) + 20,
        k = (y * w + x) * 4;
      r.set([v, Math.floor(v / 2) + 40, 255 - v, 255], k);
      d.set([Math.max(0, v - 8), Math.floor(v / 2) + 42, 255 - v, 255], k);
    }
  return { w, h, reference: r, distorted: d, kind: "synthetic" };
}
$("#synthetic").onclick = async () => {
  $("#reference-file").value = "";
  $("#distorted-file").value = "";
  await pair(syntheticPair());
};

try {
  const info = await ready;
  moduleInfo = info;
  if (info.threads !== 1 || JSON.stringify(info.metrics) !== '["A"]')
    throw Error("Expected the single-thread A module only");
  archive = await VizArchive.open(ARCHIVE_URL, (b) => call("digest", b));
  if (
    info.bakeSha !==
    archive.catalog.entries.find((e) => e.name === "A.bin")?.sha256
  )
    throw Error("Wrong embedded A bake");
  if (archive.catalog.source_commit !== SOURCE_COMMIT)
    throw Error("Archive and public source identities differ");
  [weights, audit, functions, references] = await Promise.all(
    [
      "weights.json.gz",
      "audit.json.gz",
      "function.json.gz",
      "reference.json",
    ].map((n) => archive.json(n)),
  );
  $("#unit").innerHTML =
    '<option value="">All units</option>' +
    weights.capacity.units
      .map(
        (u) =>
          `<option value="${u.unit}">h${u.unit} · ${esc(u.status)}</option>`,
      )
      .join("");
  $("#capacity").innerHTML = weights.capacity.units
    .map(
      (u) =>
        `<a href="#input=${state.input}&view=function&unit=${u.unit}&bank=${state.bank}" title="h${u.unit}: ${esc(u.status)}" style="background:var(--${u.status === "dead" ? "surface-2" : u.status === "residual" ? "series-2" : "series-1"})"></a>`,
    )
    .join("");
  for (const i of weights.inputs) {
    if (!references[`${i.family}/${i.signal}`]?.formula || !describes(i))
      throw Error("Incomplete feature reference");
  }
  $("#identity").textContent =
    `A d5e2a56d · source ${SOURCE_COMMIT.slice(0, 12)} · 1 thread`;
  const h = hash();
  await setBank(h.get("bank") === "hdr" ? "hdr" : "sdr");
  $("#bank").value = state.bank;
  window.viz = {
    state,
    call,
    archive,
    weights,
    audit,
    functions,
    render,
    pair,
    gridFor,
  };
  window.onhashchange = async () => {
    const h = hash();
    if (h.get("bank") && h.get("bank") !== state.bank) {
      if (!(await setBank(h.get("bank")))) return;
      $("#bank").value = state.bank;
    }
    await render();
  };
  await render();
  if (state.view === "atlas") await $("#synthetic").onclick();
  window.vizReady = true;
} catch (e) {
  status(String(e));
  console.error(e);
  window.vizError = String(e);
}
