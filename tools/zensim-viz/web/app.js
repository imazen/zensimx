import { formatNumber } from "./format.mjs";
import { WeightView } from "./weight-view.js";
import { buildFrame } from "./frame.js";
import { Studio } from "./studio.js";
buildFrame();
const $ = (id) => document.getElementById(id);
const state = {
  fullPrecision: false,
  manifest: null,
  pair: null,
  feature: null,
  pins: [],
  maps: new Map(),
  bake: null,
  audit: null,
  weights: null,
  lens: null,
  compare: null,
  input: 0,
  unit: 0,
  filters: {
    scale: "all",
    channel: "all",
    fragility: "damage",
    pooling: "all",
  },
  view: "atlas",
  pan: [0, 0],
};
function refreshSubject() {
  if (state.view === "atlas" && state.pair) {
    const p = state.pair,
      label =
        state.manifest?.bakes.find((b) => b.bake_sha256 === p.bake_sha256)
          ?.label || p.bake_sha256.slice(0, 12);
    $("subject").textContent =
      `${p._label || p.name} · ${p.width}×${p.height} · ${label}`;
  } else if (state.view === "lens")
    $("subject").textContent = state.bake
      ? `${state.bake.label} · ${state.audit?.bake_sha256?.slice(0, 12) || ""}`
      : "No exported bank evidence";
}
const fmt = formatNumber;
const full = (n) => (n == null ? "undefined" : String(n));
const el = (tag, text, cls) => {
  const e = document.createElement(tag);
  if (text != null) e.textContent = text;
  if (cls) e.className = cls;
  return e;
};
const css = (v) =>
  getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const svgns = "http://www.w3.org/2000/svg";
function svg(tag, attrs = {}, text) {
  const e = document.createElementNS(svgns, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (text != null) e.textContent = text;
  return e;
}
const documents = new Map();
async function get(path) {
  if (!documents.has(path))
    documents.set(
      path,
      (async () => {
        const r = await fetch(path);
        if (!r.ok) throw Error(`${r.status}: ${path}`);
        return r.json();
      })().catch((e) => {
        documents.delete(path);
        throw e;
      }),
    );
  return documents.get(path);
}
function status(message) {
  $("status").textContent = message;
}
function options(select, items, value) {
  select.replaceChildren(
    ...items.map(([v, label]) => {
      const o = el("option", label);
      o.value = v;
      return o;
    }),
  );
  if (value != null) select.value = value;
}
function tooltip(e, text) {
  const t = $("tooltip");
  t.textContent = text;
  t.hidden = false;
  const r = e.target.getBoundingClientRect();
  t.style.left = `${Math.max(8, Math.min(window.innerWidth - 420, e.clientX || r.right + 8))}px`;
  t.style.top = `${Math.min(window.innerHeight - t.offsetHeight - 12, (e.clientY || r.top) + 16)}px`;
}
function hideTooltip() {
  $("tooltip").hidden = true;
}
function tip(node, text) {
  node._tip = text;
  node.addEventListener("pointermove", (e) =>
    tooltip(e, typeof text === "function" ? text() : text),
  );
  node.addEventListener("focus", (e) =>
    tooltip(e, typeof text === "function" ? text() : text),
  );
  node.addEventListener("pointerleave", hideTooltip);
  node.addEventListener("blur", hideTooltip);
}
function table(headers, rows, onSelect) {
  const t = el("table"),
    head = el("thead"),
    tr = el("tr");
  headers.forEach((h) => tr.append(el("th", h)));
  head.append(tr);
  t.append(head);
  const body = el("tbody");
  rows.forEach((row, i) => {
    const r = el("tr");
    row.forEach((v) => {
      const c = el("td");
      const raw = String(v ?? "");
      if (
        !(v instanceof Node) &&
        /^[+-]?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?$/i.test(raw)
      ) {
        c.dataset.rawNumber = raw;
        c.textContent = state.fullPrecision ? raw : fmt(Number(raw));
        c.title = raw;
      } else c.append(v instanceof Node ? v : document.createTextNode(raw));
      r.append(c);
    });
    if (onSelect) {
      r.tabIndex = 0;
      r.onclick = () => onSelect(i);
      r.onkeydown = (e) => {
        if (e.key === "Enter") onSelect(i);
      };
    }
    body.append(r);
  });
  t.append(body);
  return t;
}
$("full-precision").onchange = () => {
  state.fullPrecision = $("full-precision").checked;
  document.querySelectorAll("td[data-raw-text]").forEach((cell) => {
    cell.textContent = state.fullPrecision
      ? cell.dataset.rawText
      : cell.dataset.compactText;
  });
  document
    .querySelectorAll("td[data-raw-number]")
    .forEach(
      (c) =>
        (c.textContent = state.fullPrecision
          ? c.dataset.rawNumber
          : fmt(Number(c.dataset.rawNumber))),
    );
};
function empty(node, message) {
  node.replaceChildren(el("div", message, "empty"));
}
function severity(f) {
  const icons = { critical: "⚠", serious: "⚠", warning: "●", note: "○" };
  const chip = el("span", `${icons[f.severity] || "○"} ${f.severity}`, "chip");
  chip.dataset.severity = f.severity;
  return chip;
}
function ramp(t, diverging = false) {
  let names;
  if (diverging) {
    names = [
      "--ramp-div-n3",
      "--ramp-div-n2",
      "--ramp-div-n1",
      "--ramp-div-0",
      "--ramp-div-p1",
      "--ramp-div-p2",
      "--ramp-div-p3",
    ];
  } else names = Array.from({ length: 7 }, (_, i) => `--ramp-seq-${i}`);
  const x = Math.max(0, Math.min(1, t)) * 6,
    i = Math.min(5, Math.floor(x)),
    f = x - i;
  const rgb = (n) =>
    css(n)
      .match(/[a-f0-9]{2}/gi)
      .map((v) => parseInt(v, 16));
  const a = rgb(names[i]),
    b = rgb(names[i + 1]);
  return a.map((v, k) => Math.round(v + (b[k] - v) * f));
}
function colour(t, div = false) {
  return `rgb(${ramp(t, div).join(",")})`;
}
function view(v) {
  state.view = v;
  refreshSubject();
  document.body.dataset.instrument = v;
  studio.renderTree();
  if (state.manifest) status(state.manifest.note || "");
  $("atlas").hidden = v !== "atlas";
  $("lens").hidden = v !== "lens";
  document
    .querySelectorAll("[data-view]")
    .forEach((b) =>
      b.setAttribute("aria-pressed", String(b.dataset.view === v)),
    );
}
document
  .querySelectorAll("[data-view]")
  .forEach((b) => (b.onclick = () => view(b.dataset.view)));
$("theme").onclick = () => {
  const light = document.documentElement.dataset.theme !== "light";
  document.documentElement.dataset.theme = light ? "light" : "dark";
  mapPalette = null;
  $("theme").textContent = light ? "Dark theme" : "Light theme";
  if (state.pair) {
    renderMatrix();
    preview(state.feature);
    renderPins();
  }
  if (state.weights) renderLens();
};
$("atlas-table-toggle").onclick = () => {
  $("atlas-table").hidden = !$("atlas-table").hidden;
};
document.querySelectorAll("[data-table]").forEach(
  (b) =>
    (b.onclick = () => {
      $(b.dataset.table).hidden = !$(b.dataset.table).hidden;
    }),
);
function segments(name, items) {
  const root = el("div", null, "seg");
  root.setAttribute("aria-label", name);
  for (const v of items) {
    const b = el("button", String(v));
    b.setAttribute("aria-pressed", String(state.filters[name] === String(v)));
    b.onclick = () => {
      state.filters[name] = String(v);
      renderFilters();
      renderMatrix();
    };
    root.append(b);
  }
  return root;
}
function renderFilters() {
  $("fragility-twin").checked = state.filters.fragility === "fragility";
}
function visible() {
  return (state.pair?.features || []).filter(
    (f) =>
      Object.entries(state.filters).every(
        ([key, v]) =>
          v === "all" ||
          (key === "fragility"
            ? f.fragility
              ? "fragility"
              : "damage"
            : String(f[key])) === v,
      ) &&
      (!$("family-lane").checked ||
        $("family").value === (f.analysis_family || f.family)),
  );
}
function percentile(f) {
  const bank = state.manifest.banks.find((b) => b.id === $("bank").value),
    sha = bank?.sha256 || bank?.id;
  const p =
    f.percentiles?.[sha] ??
    (sha === state.pair.bank_sha256 || state.manifest.banks.length === 1
      ? f.percentile
      : null);
  return p == null ? null : p > 1 ? p / 100 : p;
}
function renderMatrix() {
  studio.renderTree();
}
async function loadMap(f) {
  if (!f?.map_file) return null;
  const url = new URL(f.map_file, state.pair._url).href;
  if (!state.maps.has(url))
    state.maps.set(
      url,
      (async () => {
        const r = await fetch(url);
        if (!r.ok) throw Error(`Missing map: ${url}`);
        const image = await createImageBitmap(await r.blob());
        const c = document.createElement("canvas");
        c.width = image.width;
        c.height = image.height;
        const ctx = c.getContext("2d", { willReadFrequently: true });
        ctx.drawImage(image, 0, 0);
        return {
          width: c.width,
          height: c.height,
          pixels: ctx.getImageData(0, 0, c.width, c.height).data,
        };
      })(),
    );
  return state.maps.get(url);
}
let mapPalette;
async function paint(canvas, f, range, alphaByValue = false) {
  const map = await loadMap(f);
  if (!map) {
    canvas.width = 1;
    canvas.height = 1;
    canvas.getContext("2d").clearRect(0, 0, 1, 1);
    return;
  }
  canvas.width = map.width;
  canvas.height = map.height;
  const ctx = canvas.getContext("2d"),
    data = ctx.createImageData(map.width, map.height),
    own = f.map_range,
    [lo, hi] = range || own,
    palette = (mapPalette ||= Array.from({ length: 256 }, (_, i) =>
      ramp(i / 255),
    ));
  for (let i = 0; i < map.pixels.length; i += 4) {
    const v = own[0] + (map.pixels[i] / 255) * (own[1] - own[0]),
      t = hi === lo ? 0 : (v - lo) / (hi - lo),
      rgb = palette[Math.max(0, Math.min(255, Math.round(t * 255)))];
    data.data[i] = rgb[0];
    data.data[i + 1] = rgb[1];
    data.data[i + 2] = rgb[2];
    data.data[i + 3] = alphaByValue
      ? Math.round(Math.max(0, Math.min(1, t)) * 255)
      : 255;
  }
  ctx.putImageData(data, 0, 0);
}
async function renderFamilyLane() {
  const root = $("family-maps");
  root.hidden = !$("family-lane").checked;
  if (root.hidden || !state.feature) {
    root.replaceChildren();
    return;
  }
  const selected = state.feature,
    signature = (f) => f.name.replace(/_s\d+_/, "_s*_");
  const maps = visible().filter(
    (f) =>
      f.family === selected.family &&
      f.channel === selected.channel &&
      signature(f) === signature(selected),
  );
  const available = maps.filter((f) => f.map_file),
    range = available.length
      ? [
          Math.min(...available.map((f) => f.map_range[0])),
          Math.max(...available.map((f) => f.map_range[1])),
        ]
      : [0, 0];
  root.replaceChildren(
    el(
      "p",
      `${selected.name} · scales · shared ${fmt(range[0])} … ${fmt(range[1])}`,
      "ink-2",
    ),
  );
  const grid = el("div", null, "family-map-grid");
  root.append(grid);
  for (const f of maps) {
    const figure = el("figure"),
      canvas = el("canvas");
    figure.append(canvas, el("figcaption", `s${f.scale} · ${fmt(f.value)}`));
    grid.append(figure);
    await paint(canvas, f, range);
  }
}
let previewGeneration = 0;
async function preview(f) {
  const started = performance.now();
  const generation = ++previewGeneration;
  if (!f) return;
  state.feature = f;
  studio.reference(f);
  renderFamilyLane().catch((e) => status(e.message));
  $("feature-detail").textContent =
    `f${f.id} ${f.name} · value ${fmt(f.value)} · ${percentile(f) == null ? "percentile unavailable" : `p${fmt(100 * percentile(f))} in ${$("bank").selectedOptions[0]?.textContent}`}`;
  $("feature-detail").title = JSON.stringify(f, null, 2);
  $("matrix")
    .querySelectorAll(".cell")
    .forEach((b) => (b.dataset.hover = String(b._feature === f)));
  if (!f.map_file) {
    $("overlay").width = 1;
    $("overlay").dataset.map = f.key ?? String(f.id);
    $("map-legend").textContent =
      `${f.map_status || "Global statistic; no spatial map"} · pooled ${full(f.value)}`;
    return;
  }
  const painted = document.createElement("canvas");
  await paint(painted, f, undefined, !$("map-only").checked);
  if (state.feature !== f || generation !== previewGeneration) return;
  const overlay = $("overlay");
  overlay.width = painted.width;
  overlay.height = painted.height;
  overlay.getContext("2d").drawImage(painted, 0, 0);
  overlay.dataset.map = f.key ?? String(f.id);
  overlay.dataset.latency = String(performance.now() - started);
  $("map-legend").replaceChildren(
    el("span", fmt(f.map_range[0])),
    el("span", null, "ramp"),
    el("span", fmt(f.map_range[1])),
    el(
      "span",
      `Own range · pooled ${fmt(f.value)} · ${f.pooling} · pooling grid ${(f.pooling_grid || []).join("×")}`,
    ),
  );
}
function pin(f) {
  if (!f) return;
  const index = state.pins.indexOf(f);
  if (index >= 0) state.pins.splice(index, 1);
  else {
    if (state.pins.length === 4) state.pins.shift();
    state.pins.push(f);
  }
  studio.hold = state.pins.includes(f) ? f : null;
  preview(f);
  studio.reference(f, true);
  renderPins();
  renderMatrix();
}
let pinsGeneration = 0;
async function renderPins() {
  const generation = ++pinsGeneration,
    pins = [...state.pins];
  const maps = pins.filter((f) => f.map_file),
    range = maps.length
      ? [
          Math.min(...maps.map((f) => f.map_range[0])),
          Math.max(...maps.map((f) => f.map_range[1])),
        ]
      : [0, 0];
  $("pins").replaceChildren();
  if (!pins.length)
    $("pins").append(el("div", "pin a feature: Enter", "empty-pins"));
  for (const f of pins) {
    if (generation !== pinsGeneration) return;
    const fig = el("figure"),
      c = el("canvas"),
      caption = el("figcaption", `f${f.id} · ${fmt(f.value)}`),
      remove = el("button", "×");
    remove.setAttribute("aria-label", `Unpin f${f.id}`);
    remove.onclick = () => pin(f);
    caption.append(remove);
    c.tabIndex = 0;
    c.setAttribute("role", "button");
    c.setAttribute("aria-label", `Show pinned f${f.id}`);
    c.onclick = () => {
      studio.hold = f;
      preview(f);
      studio.reference(f, true);
    };
    c.onkeydown = (e) => {
      if (e.key === "Enter" || e.key === " ") c.click();
    };
    fig.append(
      c,
      caption,
      el(
        "small",
        f.map_file
          ? `Shared ${fmt(range[0])} … ${fmt(range[1])}`
          : f.map_status || "Global statistic",
      ),
    );
    $("pins").append(fig);
    await paint(c, f, range);
  }
}
let imageMode = "distorted";
function showImage() {
  if (!state.pair) return;
  const file = state.pair[imageMode] || state.pair.images?.[imageMode];
  $("pair-image").src = new URL(file, state.pair._url).href;
  $("pair-image").alt = `${imageMode}: ${state.pair.name}`;
  document
    .querySelectorAll("[data-mode]")
    .forEach((b) =>
      b.setAttribute("aria-pressed", String(b.dataset.mode === imageMode)),
    );
}
document.querySelectorAll("[data-mode]").forEach(
  (b) =>
    (b.onclick = () => {
      imageMode = b.dataset.mode;
      showImage();
    }),
);
$("flip").onclick = () => {
  imageMode = imageMode === "reference" ? "distorted" : "reference";
  showImage();
};
$("zoom").oninput = () =>
  document.documentElement.style.setProperty("--zoom", $("zoom").value);
$("alpha").oninput = () =>
  document.documentElement.style.setProperty(
    "--overlay-alpha",
    $("alpha").value,
  );
$("map-only").onchange = () => {
  preview(state.feature).catch((e) => status(e.message));
  $("pair-image").style.visibility = $("map-only").checked
    ? "hidden"
    : "visible";
  $("overlay").style.opacity = $("map-only").checked ? "1" : "";
};
function pan() {
  document.documentElement.style.setProperty("--pan-x", state.pan[0] + "px");
  document.documentElement.style.setProperty("--pan-y", state.pan[1] + "px");
}
$("reset-pan").onclick = () => {
  state.pan = [0, 0];
  $("zoom").value = 1;
  document.documentElement.style.setProperty("--zoom", 1);
  pan();
};
let drag = null;
$("stage").onpointerdown = (e) => {
  drag = [e.clientX - state.pan[0], e.clientY - state.pan[1]];
  $("stage").setPointerCapture(e.pointerId);
};
$("stage").onpointermove = (e) => {
  if (drag) {
    state.pan = [e.clientX - drag[0], e.clientY - drag[1]];
    pan();
  }
};
$("stage").onpointerup = () => (drag = null);
for (const id of ["colour", "family-lane", "family"])
  $(id).onchange = renderMatrix;
let pairGeneration = 0;
async function selectPair() {
  const generation = ++pairGeneration;
  const p = state.manifest.pairs.find((p) => p.id === $("pair").value);
  if (!p) return;
  const url = new URL(p.meta, location.href).href;
  const pair = await get(url);
  if (generation !== pairGeneration) return;
  pair._url = url;
  pair._label = p.label;
  state.pair = pair;
  state.pins = [];
  studio.hold = null;
  state.feature = null;
  options(
    $("family"),
    [...new Set(pair.features.map((f) => f.analysis_family || f.family))].map(
      (v) => [v, v],
    ),
  );
  refreshSubject();
  renderFilters();
  renderMatrix();
  showImage();
  preview(visible()[0]);
  renderPins();
}
$("pair").onchange = () => selectPair().catch((e) => status(e.message));
$("bank").onchange = () => {
  if (state.pair) {
    renderMatrix();
    preview(state.feature);
  }
  selectBake().catch((e) => status(e.message));
};
// Lens charts and data loaders are appended below.
function chartFrame(xmin, xmax, ymin, ymax, xlabel, ylabel, xticks) {
  const W = 800,
    H = 300,
    L = 64,
    R = 28,
    T = 20,
    B = 60;
  const node = svg("svg", {
    viewBox: `0 0 ${W} ${H}`,
    class: "chart",
    role: "img",
    "aria-label": `${ylabel} by ${xlabel}`,
  });
  if (xmin === xmax) xmax = xmin + 1;
  if (ymin === ymax) ymax = ymin + 1;
  const x = (v) => L + ((v - xmin) / (xmax - xmin)) * (W - L - R),
    y = (v) => H - B - ((v - ymin) / (ymax - ymin)) * (H - T - B);
  const grid = svg("g", { class: "grid" });
  for (let i = 0; i <= 4; i++) {
    const v = ymin + ((ymax - ymin) * i) / 4;
    grid.append(
      svg("line", { x1: L, x2: W - R, y1: y(v), y2: y(v) }),
      svg("text", { x: L - 12, y: y(v) + 4, "text-anchor": "end" }, fmt(v)),
    );
  }
  node.append(grid);
  const ticks =
    xticks ||
    Array.from({ length: 5 }, (_, i) => {
      const v = xmin + ((xmax - xmin) * i) / 4;
      return [v, fmt(v)];
    });
  for (const [v, label] of ticks)
    node.append(
      svg("text", { x: x(v), y: H - B + 20, "text-anchor": "middle" }, label),
    );
  node.append(
    svg("text", { x: L, y: 12 }, ylabel),
    svg(
      "text",
      { x: (W + L - R) / 2, y: H - 8, "text-anchor": "middle" },
      xlabel,
    ),
  );
  return { node, x, y, W, H, L, R, T, B };
}
function linePath(xs, ys, x, y) {
  return xs.map((v, i) => `${i ? "L" : "M"}${x(v)} ${y(ys[i])}`).join(" ");
}
function addCrosshair(c, xs, describe) {
  const cross = svg("line", {
      class: "crosshair",
      y1: c.T,
      y2: c.H - c.B,
      visibility: "hidden",
    }),
    hit = svg("rect", {
      x: c.L,
      y: c.T,
      width: c.W - c.L - c.R,
      height: c.H - c.T - c.B,
      class: "hit",
      tabindex: 0,
      "aria-label": "Use left and right arrow keys to inspect chart values",
    });
  c.node.append(cross, hit);
  let selected = 0;
  const show = (e, i) => {
    selected = i;
    cross.setAttribute("visibility", "visible");
    cross.setAttribute("x1", c.x(xs[i]));
    cross.setAttribute("x2", c.x(xs[i]));
    tooltip(e, describe(i));
  };
  hit.addEventListener("pointermove", (e) => {
    const p =
      ((e.clientX - c.node.getBoundingClientRect().left) /
        c.node.getBoundingClientRect().width) *
      c.W;
    let best = 0;
    xs.forEach((v, i) => {
      if (Math.abs(c.x(v) - p) < Math.abs(c.x(xs[best]) - p)) best = i;
    });
    show(e, best);
  });
  hit.addEventListener("focus", (e) => show(e, selected));
  hit.addEventListener("keydown", (e) => {
    if (["ArrowLeft", "ArrowRight"].includes(e.key)) {
      e.preventDefault();
      show(
        e,
        Math.max(
          0,
          Math.min(xs.length - 1, selected + (e.key === "ArrowRight" ? 1 : -1)),
        ),
      );
    }
  });
  for (const event of ["pointerleave", "blur"])
    hit.addEventListener(event, () => {
      cross.setAttribute("visibility", "hidden");
      hideTooltip();
    });
}
function renderPdp() {
  const d = state.lens,
    other = state.compare;
  if (!d) {
    empty($("pdp-chart"), "No curve exported for this input.");
    $("pdp-table").replaceChildren();
    $("pdp-legend").replaceChildren();
    $("pdp-note").textContent = "";
    $("pdp-title").textContent = "";
    return;
  }
  const p = d.pdp,
    op = other?.pdp;
  $("center-ice").disabled = !p.centered_ice?.length;
  if ($("center-ice").disabled) $("center-ice").checked = false;
  const centered = $("center-ice").checked,
    ice = centered ? p.centered_ice : p.ice,
    mean = centered ? p.centered_mean : p.mean,
    low = centered ? p.centered_p25 : p.p25,
    high = centered ? p.centered_p75 : p.p75,
    otherMean = op ? (centered ? op.centered_mean : op.mean) : null;
  const values = [
      ...mean,
      ...(low || []),
      ...(high || []),
      ...(otherMean || []),
    ],
    tickIndices = [0],
    span = p.x.at(-1) - p.x[0];
  if (span > 0) {
    for (const i of [5, 10, 15])
      if (
        ((p.x[i] - p.x[tickIndices.at(-1)]) / span) * 708 >= 90 &&
        ((p.x.at(-1) - p.x[i]) / span) * 708 >= 90
      )
        tickIndices.push(i);
    tickIndices.push(20);
  }
  const c = chartFrame(
    Math.min(...p.x),
    Math.max(...p.x),
    Math.min(...values),
    Math.max(...values),
    p.axis_label || `f${p.feature} · feature units`,
    centered ? "Δ served score from each row’s first value" : "Served score",
    p.percentiles.length
      ? tickIndices.map((i) => [p.x[i], fmt(p.x[i])])
      : undefined,
  );
  if (p.percentiles.length)
    for (const i of tickIndices)
      c.node.append(
        svg(
          "text",
          { x: c.x(p.x[i]), y: c.H - c.B + 36, "text-anchor": "middle" },
          `p${fmt(p.percentiles[i] * 100)}`,
        ),
      );
  const definitions = svg("defs"),
    clip = svg("clipPath", { id: "pdp-plot-clip" });
  clip.append(
    svg("rect", {
      x: c.L,
      y: c.T,
      width: c.W - c.L - c.R,
      height: c.H - c.B - c.T,
    }),
  );
  definitions.append(clip);
  c.node.append(definitions);
  const plot = svg("g", { "clip-path": "url(#pdp-plot-clip)" });
  if (low?.length)
    plot.append(
      svg("path", {
        d:
          linePath(p.x, low, c.x, c.y) +
          p.x
            .toReversed()
            .map((v, i) => `L${c.x(v)} ${c.y(high[high.length - 1 - i])}`)
            .join(" ") +
          "Z",
        class: "iqr-band",
      }),
    );
  const shown = Math.min(40, ice.length);
  for (let k = 0; k < shown; k++) {
    const row = Math.floor((k * (ice.length - 1)) / Math.max(1, shown - 1));
    plot.append(
      svg("path", {
        d: linePath(p.x, ice[row], c.x, c.y),
        class: "ice",
        "data-bank-row": p.row_indices[row],
      }),
    );
  }
  plot.append(
    svg("path", { d: linePath(p.x, mean, c.x, c.y), class: "line mean" }),
  );
  if (otherMean?.length)
    plot.append(
      svg("path", {
        d: linePath(op.x, otherMean, c.x, c.y),
        class: "line compare",
      }),
    );
  c.node.append(
    plot,
    svg(
      "text",
      { x: c.x(p.x.at(-1)) - 4, y: c.y(mean.at(-1)) - 8, "text-anchor": "end" },
      `${state.bake.label} ${fmt(mean.at(-1))}`,
    ),
  );
  addCrosshair(
    c,
    p.x,
    (i) =>
      `${full(p.x[i])} · f${p.feature}${p.percentiles.length ? ` (p${fmt(p.percentiles[i] * 100)})` : ""}\n${full(mean[i])} · ${state.bake.label} mean${low?.length ? `\n${full(low[i])} … ${full(high[i])} · interquartile range` : ""}\n${ice.length} recorded ICE rows · ${shown} drawn${otherMean?.length ? `\n${full(otherMean[i])} · ${$("compare").selectedOptions[0].textContent} mean` : ""}`,
  );
  $("pdp-chart").replaceChildren(c.node);
  $("pdp-title").textContent = p.family
    ? `${p.family} family`
    : `f${p.feature}`;
  const legend = el("span");
  legend.append(
    el("span", null, "key"),
    document.createTextNode(`${state.bake.label} mean · solid`),
  );
  const band = el("span");
  band.append(
    el("span", null, "key band-key"),
    document.createTextNode("Interquartile band · all 200 rows"),
  );
  $("pdp-legend").replaceChildren(
    legend,
    ...(low?.length ? [band] : []),
    el("span", `${shown} of ${ice.length} ICE traces`),
  );
  if (otherMean?.length) {
    const label = el("span");
    label.append(
      el("span", null, "key compare"),
      document.createTextNode(
        `${$("compare").selectedOptions[0].textContent} mean · dashed`,
      ),
    );
    $("pdp-legend").append(label);
  }
  $("pdp-note").textContent =
    `${p.monotonicity_violations.length} strict floating-point rises in the original served scores · ${p.ice.length} recorded rows · ${p.family ? "0–2 × observed family damage" : "p1–p99"}${centered ? " · centered on each row’s first point" : ""}. Every native curve and flag remains in the numeric table/export.`;
  $("pdp-table").replaceChildren(
    table(
      [
        "Feature value",
        "Bank percentile",
        `${state.bake.label} ${centered ? "centered " : ""}mean`,
        ...(otherMean?.length ? ["Comparison mean"] : []),
        "p25",
        "p75",
        ...ice.map(
          (_, i) => `${centered ? "Δ " : ""}ICE row ${p.row_indices[i]}`,
        ),
      ],
      p.x.map((v, i) => [
        full(v),
        p.percentiles.length ? full(p.percentiles[i]) : "multiplier",
        full(mean[i]),
        ...(otherMean?.length ? [full(otherMean[i])] : []),
        full(low?.[i]),
        full(high?.[i]),
        ...ice.map((row) => full(row[i])),
      ]),
    ),
  );
}
$("center-ice").onchange = renderPdp;
function rangePanel(bins, human = false) {
  const values = bins.flatMap((b) =>
      human ? b.human.slice(1) : [b.p10, b.p90],
    ),
    c = chartFrame(
      0,
      9.5,
      Math.min(...values),
      Math.max(...values),
      "Quantile bin · ties kept together",
      human ? "Human score · native units" : "Served score",
      Array.from({ length: 10 }, (_, i) => [i, String(i + 1)]),
    );
  for (const b of bins) {
    const med = human ? b.human[0] : b.median,
      low = human ? b.human[1] : b.p10,
      high = human ? b.human[2] : b.p90,
      x = c.x(b.bin + (human ? 0.5 : 0)),
      g = svg("g", { tabindex: 0 });
    g.append(
      svg("line", {
        x1: x,
        x2: x,
        y1: c.y(low),
        y2: c.y(high),
        class: human ? "whisker human" : "whisker",
      }),
    );
    for (const v of [low, high])
      g.append(
        svg("line", {
          x1: x - 6,
          x2: x + 6,
          y1: c.y(v),
          y2: c.y(v),
          class: human ? "whisker human" : "whisker",
        }),
      );
    g.append(
      human
        ? svg("rect", {
            x: x - 4,
            y: c.y(med) - 4,
            width: 8,
            height: 8,
            fill: "var(--series-2)",
          })
        : svg("circle", {
            cx: x,
            cy: c.y(med),
            r: 4,
            class: "marker",
            fill: "var(--series-1)",
          }),
      svg("rect", {
        x: x - 12,
        y: c.T,
        width: 24,
        height: c.H - c.T - c.B,
        class: "hit",
      }),
    );
    tip(
      g,
      `${full(med)} · ${human ? "human" : "served"} median
p10 ${full(low)} · p90 ${full(high)}
${state.lens.separation_axis}: ${full(b.low)} … ${full(b.high)}
${b.count} observed rows${
        b.human
          ? `
Human median ${full(b.human[0])} · p10 ${full(b.human[1])} · p90 ${full(b.human[2])}`
          : ""
      }`,
    );
    c.node.append(g);
  }
  return c.node;
}
function renderRange() {
  const bins = state.lens?.separation;
  if (!bins?.length) {
    empty($("range-chart"), "Range separation has not been exported.");
    $("range-table").replaceChildren();
    return;
  }
  $("range-chart").replaceChildren(
    el("p", state.lens.separation_axis, "ink-2"),
    rangePanel(bins),
  );
  const human = bins.filter((b) => b.human);
  if (human.length) {
    $("range-chart").prepend(
      el(
        "p",
        "● served score · ■ human score (separate axes, native units)",
        "ink-2",
      ),
    );
    $("range-chart").append(rangePanel(human, true));
  }
  $("range-table").replaceChildren(
    table(
      [
        "Bin",
        "Range low",
        "Range high",
        "Rows",
        "Score median",
        "Score p10",
        "Score p90",
        "Human median / p10 / p90",
      ],
      bins.map((b) => [
        b.bin + 1,
        full(b.low),
        full(b.high),
        b.count,
        full(b.median),
        full(b.p10),
        full(b.p90),
        b.human?.map(full).join(" / ") || "not present",
      ]),
    ),
  );
}
function renderFamily() {
  const rows = state.lens?.attribution;
  if (!rows?.length) {
    empty($("family-chart"), "Family ablations have not been exported.");
    return;
  }
  const W = 800,
    L = 260,
    R = 64,
    T = 12,
    gap = 30,
    H = T + rows.length * gap + 40,
    max = Math.max(...rows.map((r) => r.mean_abs_delta), 1e-30),
    node = svg("svg", {
      viewBox: `0 0 ${W} ${H}`,
      class: "chart",
      role: "img",
      "aria-label": "Mean absolute served score change under family ablation",
    });
  for (let i = 0; i < rows.length; i++) {
    const r = rows[i],
      y = T + i * gap,
      len = ((W - L - R) * r.mean_abs_delta) / max,
      g = svg("g", { tabindex: 0 });
    node.append(
      svg("text", { x: L - 12, y: y + 15, "text-anchor": "end" }, r.label),
    );
    const d = `M${L} ${y} H${L + Math.max(0, len - 4)} Q${L + len} ${y} ${L + len} ${y + 4} V${y + 20} Q${L + len} ${y + 24} ${L + Math.max(0, len - 4)} ${y + 24} H${L} Z`;
    g.append(
      svg("path", { d, class: "bar" }),
      svg("rect", {
        x: L,
        y,
        width: Math.max(24, len),
        height: 24,
        class: "hit",
      }),
    );
    node.append(
      g,
      svg("text", { x: L + len + 8, y: y + 16 }, fmt(r.mean_abs_delta)),
    );
    tip(
      g,
      `${full(r.mean_abs_delta)} · mean |Δscore|\n${r.label}\nMedian Δ ${full(r.median_delta)} · p10 ${full(r.p10)} · p90 ${full(r.p90)}`,
    );
  }
  node.append(
    svg("text", { x: L, y: H - 4 }, "Mean |Δ served score| · zero baseline"),
  );
  $("family-chart").replaceChildren(node);
  $("attribution-note").textContent = state.lens.attribution_definition;
  $("family-table").replaceChildren(
    table(
      [
        "Family / input",
        "Feature IDs",
        "Mean |Δscore|",
        "Median Δ",
        "p10",
        "p90",
      ],
      rows.map((r) => [
        r.label,
        r.features.join(", "),
        full(r.mean_abs_delta),
        full(r.median_delta),
        full(r.p10),
        full(r.p90),
      ]),
    ),
  );
}
let concernsPage = 1;
function selectConcern(f) {
  if (!state.weights) return;
  if (f.feature != null)
    state.input = Math.max(
      0,
      state.weights.inputs.findIndex((i) => i.feature === f.feature),
    );
  if (f.layer === 0) {
    state.input = f.from ?? state.input;
    state.unit = f.to ?? state.unit;
  } else if (f.layer === 1) state.unit = f.from ?? state.unit;
  $("input").value = state.input;
  $("lens-family").value = f.family_intervention || "";
  renderWeights();
  loadLens().catch((e) => status(e.message));
}
function renderConcerns() {
  if (!state.audit) return;
  const a = state.audit,
    structural = new Set([
      "sign",
      "scaler_sign",
      "certificate",
      "dead-input",
      "dead_input",
      "constant_input",
      "dead_hidden",
      "saturated_hidden",
      "duplicate_input",
      "drift",
    ]);
  const allFlags = [...a.flags, ...(state.bake.monotonicity_flags || [])].map(
    (f) => ({
      ...f,
      severity: structural.has(f.kind) ? "critical" : f.severity,
    }),
  );
  const flags = allFlags.filter(
    (f) =>
      (!$("severity").value || f.severity === $("severity").value) &&
      (!$("kind").value || f.kind === $("kind").value),
  );
  const groups = new Map(),
    priorities = { critical: 0, serious: 1, warning: 2, note: 3 };
  const magnitude = (f) => Math.abs(f.statistic ?? f.value);
  for (const f of flags) {
    const input =
      f.intervention ??
      (f.layer === 0 && f.from != null
        ? `input ${f.from}`
        : f.layer === 1 && f.from != null
          ? `hidden ${f.from} → output`
          : f.to != null
            ? `hidden ${f.to}`
            : "model");
    const key = JSON.stringify([input, f.kind, f.reason]);
    if (!groups.has(key))
      groups.set(key, {
        input,
        kind: f.kind,
        reason: f.reason,
        severity: f.severity,
        flags: [],
        worst: f,
      });
    const group = groups.get(key);
    group.flags.push(f);
    if (priorities[f.severity] < priorities[group.severity])
      group.severity = f.severity;
    if (magnitude(f) > magnitude(group.worst)) group.worst = f;
  }
  const rows = [...groups.values()].sort(
    (a, b) =>
      (a.kind === "dead-input" ? -1 : 0) - (b.kind === "dead-input" ? -1 : 0) ||
      priorities[a.severity] - priorities[b.severity] ||
      magnitude(b.worst) - magnitude(a.worst),
  );
  if ($("sort").value === "magnitude")
    rows.sort((a, b) => magnitude(b.worst) - magnitude(a.worst));
  const pages = Math.max(1, Math.ceil(rows.length / 50));
  concernsPage = Math.min(concernsPage, pages);
  const selected = rows.slice((concernsPage - 1) * 50, concernsPage * 50);
  $("concern-title").textContent =
    `Concerns · ${rows.length.toLocaleString()} groups / ${flags.length.toLocaleString()} measured flags`;
  $("audit-summary").textContent =
    `${a.weight_count.toLocaleString()} weights · ${a.sampled_rows.toLocaleString()} recorded rows of ${a.bank_rows.toLocaleString()} · bank ${a.bank_sha256.slice(0, 12)} · paired constraints ${a.constraints?.max_pairs?.length ?? "unavailable"}. MAD notes are diagnostic; grouped counts retain every measured flag.`;
  const t = table(
    [
      "Severity",
      "Kind",
      "Input / unit",
      "Count",
      "Worst magnitude",
      "Worst witness",
      "Reason",
    ],
    selected.map((g) => [
      severity(g),
      g.kind,
      g.input,
      g.flags.length,
      full(magnitude(g.worst)),
      g.worst.location,
      g.reason,
    ]),
    (i) => selectConcern(selected[i].worst),
  );
  const pager = el("div", null, "controls"),
    previous = el("button", "Previous 50"),
    next = el("button", "Next 50"),
    label = el(
      "span",
      `Page ${concernsPage} of ${pages} · ${rows.length} groups`,
      "ink-2",
    );
  previous.disabled = concernsPage === 1;
  next.disabled = concernsPage === pages;
  previous.onclick = () => {
    concernsPage--;
    renderConcerns();
  };
  next.onclick = () => {
    concernsPage++;
    renderConcerns();
  };
  pager.append(previous, label, next);
  $("concern-table").replaceChildren(t);
  $("concern-pager").replaceChildren(pager);
}
function renderArchitecture() {
  refreshSubject();
  const m = state.weights,
    l = m.layers[0],
    node = $("architecture");
  node.replaceChildren();
  const boxes = [
    [12, `${l.in_dim} inputs`],
    [125, `${l.out_dim} hidden`],
    [238, "output stage"],
    [350, "dial"],
  ];
  for (let i = 0; i < boxes.length; i++) {
    const [x, label] = boxes[i];
    node.append(
      svg("rect", {
        x,
        y: 20,
        width: 82,
        height: 32,
        rx: 4,
        class: "selected",
      }),
      svg("text", { x: x + 41, y: 40, "text-anchor": "middle" }, label),
    );
    if (i < 3)
      node.append(
        svg("path", {
          d: `M${x + 84} 36 H${boxes[i + 1][0] - 4}`,
          class: "selected",
        }),
      );
  }
  node.append(
    svg(
      "text",
      { x: 12, y: 78 },
      `${m.inputs[state.input]?.name || state.input} → hidden ${state.unit} → served score`,
    ),
  );
}
function weightStatisticsText(i, j) {
  const l = state.weights.layers[0],
    k = i * l.out_dim + j,
    stats = l.stats;
  const nonzero =
    state.weights.nonzero_counts?.[i] ??
    l.weights.slice(i * l.out_dim, (i + 1) * l.out_dim).filter((v) => v !== 0)
      .length;
  if (nonzero === 0) return `all ${l.out_dim} weights are zero: dead input`;
  let text =
    `${nonzero} of ${l.out_dim} weights are nonzero. ` +
    (stats
      ? `Row z ${stats.row_z[k] == null ? "undefined (MAD=0)" : full(stats.row_z[k])} · column z ${stats.column_z[k] == null ? "undefined (MAD=0)" : full(stats.column_z[k])}
Row median/MAD ${stats.rows[i].map(full).join(" / ")}
Column median/MAD ${stats.columns[j].map(full).join(" / ")}`
      : "Per-weight z statistics are not present in this bundle.");
  const pair = state.audit.constraints?.max_pairs?.find((p) => p.includes(j));
  if (pair)
    text += `
Coupled max pair h${pair[0]} / h${pair[1]}`;
  if (state.weights.layers[1].weights[j] === 0)
    text += `\nOutgoing weight is exactly zero`;
  return text;
}
const weightView = new WeightView({
  state,
  $,
  css,
  svg,
  el,
  table,
  tip,
  tooltip,
  hideTooltip,
  fmt,
  full,
  describe: weightStatisticsText,
  select: (input, unit) => {
    state.input = input;
    state.unit = unit;
    studio.showUnit = true;
    studio.reference(state.feature, true);
    $("input").value = input;
    renderWeights();
    loadLens().catch((e) => status(e.message));
  },
});
function renderWeights() {
  if (!state.weights) return;
  weightView.render();
  renderArchitecture();
}
function renderInputRanking() {
  const rows = state.lens?.input_attribution;
  if (!rows?.length) {
    empty(
      $("input-ranking-chart"),
      "Individual input ablations have not been exported.",
    );
    $("input-ranking-table").replaceChildren();
    return;
  }
  const shown = rows.slice(0, 20),
    max = Math.max(...shown.map((r) => r.mean_abs_delta), 1e-30),
    W = 800,
    L = 280,
    R = 80,
    H = shown.length * 28 + 32,
    node = svg("svg", {
      viewBox: `0 0 ${W} ${H}`,
      class: "chart",
      role: "img",
      "aria-label": "Top 20 individual raw damage input ablations",
    });
  for (let i = 0; i < shown.length; i++) {
    const r = shown[i],
      y = i * 28,
      len = ((W - L - R) * r.mean_abs_delta) / max,
      g = svg("g", { tabindex: 0 });
    node.append(
      svg("text", { x: L - 12, y: y + 16, "text-anchor": "end" }, r.label),
    );
    g.append(
      svg("path", {
        d: `M${L} ${y} H${L + Math.max(0, len - 4)} Q${L + len} ${y} ${L + len} ${y + 4} V${y + 20} Q${L + len} ${y + 24} ${L + Math.max(0, len - 4)} ${y + 24} H${L} Z`,
        class: "bar",
      }),
      svg("rect", {
        x: L,
        y,
        width: Math.max(24, len),
        height: 24,
        class: "hit",
      }),
    );
    tip(
      g,
      `${full(r.mean_abs_delta)} · mean |Δscore|
${r.label}
Median Δ ${full(r.median_delta)} · p10 ${full(r.p10)} · p90 ${full(r.p90)}`,
    );
    g.onclick = () => {
      const input = state.weights.inputs.find(
        (i) => i.feature === r.features[0] && i.factor == null,
      );
      if (input) {
        state.input = input.index;
        $("input").value = input.index;
        $("lens-family").value = "";
        renderWeights();
        loadLens().catch((e) => status(e.message));
      }
    };
    g.onkeydown = (e) => {
      if (e.key === "Enter") g.onclick();
    };
    node.append(
      g,
      svg("text", { x: L + len + 8, y: y + 16 }, fmt(r.mean_abs_delta)),
    );
  }
  node.append(
    svg("text", { x: L, y: H - 4 }, "Mean |Δ served score| · zero baseline"),
  );
  $("input-ranking-chart").replaceChildren(node);
  $("input-ranking-table").replaceChildren(
    table(
      ["Input", "Mean |Δscore|", "Median Δ", "p10", "p90"],
      rows.map((r) => [
        r.label,
        full(r.mean_abs_delta),
        full(r.median_delta),
        full(r.p10),
        full(r.p90),
      ]),
    ),
  );
}
function renderHidden() {
  $("hidden-units").replaceChildren(
    ...state.audit.hidden_units.map((u) => {
      const b = el("button", String(u.unit));
      tip(
        b,
        `h${u.unit}\n${fmt(100 * u.activation_rate)}% positive\nMean |activation| ${full(u.mean_abs_activation)}\nRange ${full(u.min)} … ${full(u.max)}\nTop inputs: ${u.top_inputs.map(([i, v]) => `${i}: ${full(v)}`).join(", ")}`,
      );
      b.onclick = () => {
        state.unit = u.unit;
        renderWeights();
      };
      return b;
    }),
  );
  const details = el("details"),
    summary = el("summary", "Hidden-unit values · table");
  details.append(
    summary,
    table(
      [
        "Unit",
        "Positive fraction",
        "Mean |activation|",
        "Min",
        "Max",
        "Outgoing weight",
        "Top incoming inputs",
      ],
      state.audit.hidden_units.map((u) => [
        u.unit,
        full(u.activation_rate),
        full(u.mean_abs_activation),
        full(u.min),
        full(u.max),
        full(state.weights.layers[1]?.weights[u.unit]),
        u.top_inputs.map(([i, w]) => `${i}: ${full(w)}`).join(", "),
      ]),
      (j) => {
        state.unit = j;
        renderWeights();
      },
    ),
  );
  $("hidden-units").append(details);
}
function renderLens() {
  renderConcerns();
  renderWeights();
  renderHidden();
  renderPdp();
  renderRange();
  renderFamily();
  renderInputRanking();
}
let lensGeneration = 0;
function bankScopedBake(base) {
  if (!base) return null;
  return base.banks
    ? base.banks[$("bank").value]
      ? { ...base, ...base.banks[$("bank").value] }
      : null
    : base;
}
async function lensDocument(bake, path) {
  if (!path) return null;
  const [summary, curve] = await Promise.all([
    bake.summary ? get(bake.summary) : Promise.resolve({}),
    get(path),
  ]);
  return { ...summary, ...curve };
}
async function loadLens() {
  if (!state.bake || !state.weights) return;
  const generation = ++lensGeneration,
    feature = state.weights.inputs[state.input].feature,
    family = $("lens-family").value;
  const path = family
    ? state.bake.family_lenses?.[family]
    : state.bake.lenses?.[`f${feature}`];
  const comparison = bankScopedBake(
    state.manifest.bakes.find((b) => b.id === $("compare").value),
  );
  const cp = family
    ? comparison?.family_lenses?.[family]
    : comparison?.lenses?.[`f${feature}`];
  const comparisonPath = state.bake.comparisons?.[comparison?.id];
  document.querySelector(".diagnostics").classList.add("loading");
  const [primary, secondary, comparisonData] = await Promise.all([
    lensDocument(state.bake, path),
    lensDocument(comparison, cp),
    comparisonPath ? get(comparisonPath) : Promise.resolve(null),
  ]);
  if (generation !== lensGeneration) return;
  state.lens = primary;
  state.compare = secondary;
  $("comparison-panel").hidden = !comparisonData;
  if (comparisonData) {
    $("comparison-detail").textContent = comparisonData.weight_drift_caveat;
    const drift = comparisonData.weight_drift.find(
      (d) => d.primary_input === state.input,
    );
    $("comparison-table").replaceChildren(
      table(
        ["Kind", "Primary count", "Comparison count", "Delta"],
        Object.keys(comparisonData.primary_counts).map((k) => [
          k,
          comparisonData.primary_counts[k],
          comparisonData.secondary_counts[k],
          comparisonData.audit_count_delta[k],
        ]),
      ),
      table(
        [
          "Input",
          "Primary L2 norm",
          "Comparison L2 norm",
          "Norm delta",
          "Stored-index L2 drift",
        ],
        drift
          ? [
              [
                state.input,
                full(drift.primary_l2),
                full(drift.secondary_l2),
                full(drift.l2_norm_delta),
                full(drift.indexwise_l2_delta),
              ],
            ]
          : [],
      ),
    );
  }
  renderPdp();
  renderRange();
  renderFamily();
  renderInputRanking();
  studio.renderTree();
  studio.reference(studio.hold || state.feature, true);
  document.querySelector(".diagnostics").classList.remove("loading");
}
let bakeGeneration = 0;
async function selectBake() {
  const generation = ++bakeGeneration;
  ++lensGeneration;
  state.weights = null;
  $("input").disabled = $("lens-family").disabled = true;
  document.querySelector(".diagnostics").classList.add("loading");
  const bake = bankScopedBake(
    state.manifest?.bakes.find((b) => b.id === $("bake").value),
  );
  if (!bake) {
    lensGeneration++;
    state.bake =
      state.audit =
      state.weights =
      state.lens =
      state.compare =
        null;
    for (const id of [
      "concern-table",
      "pdp-chart",
      "range-chart",
      "family-chart",
      "hidden-units",
    ]) {
      const node = $(id);
      if (node) empty(node, "No evidence exported for this bake and bank.");
    }
    $("weight-matrix")
      .getContext("2d")
      .clearRect(0, 0, $("weight-matrix").width, $("weight-matrix").height);
    const zoom = $("weight-zoom");
    zoom.getContext("2d").clearRect(0, 0, zoom.width, zoom.height);
    for (const id of ["weight-table", "unit-input-strip", "input-unit-strip"])
      empty($(id), "No evidence for this bake and bank.");
    $("center-ice").disabled = true;
    $("architecture").replaceChildren();
    $("weight-detail").textContent = "No evidence for this bake and bank.";
    $("weight-legend").replaceChildren();
    $("comparison-panel").hidden = true;
    renderInputRanking();
    $("pdp-legend").replaceChildren();
    $("pdp-note").textContent = "";
    $("audit-summary").textContent = "No bank audit exported.";
    $("concern-pager").replaceChildren();
    document.querySelector(".diagnostics").classList.remove("loading");
    status("This bake has no exported evidence for the selected bank.");
    return;
  }
  const [audit, weights] = await Promise.all([
    get(bake.audit),
    get(bake.weights),
  ]);
  if (generation !== bakeGeneration) return;
  state.bake = bake;
  state.audit = audit;
  state.weights = weights;
  $("input").disabled = $("lens-family").disabled = false;
  state.input = 0;
  state.unit = 0;
  options(
    $("input"),
    state.weights.inputs.map((i) => [i.index, `${i.index} · ${i.name}`]),
  );
  options($("lens-family"), [
    ["", "Selected input"],
    ...[
      ...new Set(
        state.weights.inputs.flatMap((i) => [
          i.family,
          i.analysis_family || i.family,
        ]),
      ),
    ].map((f) => [f, f]),
  ]);
  options($("kind"), [
    ["", "All"],
    ["monotonicity", `monotonicity (${bake.monotonicity_flag_count || 0})`],
    ...Object.keys(state.audit.counts).map((k) => [
      k,
      `${k} (${state.audit.counts[k]})`,
    ]),
  ]);
  concernsPage = 1;
  renderConcerns();
  renderWeights();
  renderHidden();
  await loadLens();
}
$("bake").onchange = () => selectBake().catch((e) => status(e.message));
$("lens-family").onchange = () => {
  renderWeights();
  loadLens().catch((e) => status(e.message));
};
$("compare").onchange = () => loadLens().catch((e) => status(e.message));
$("input").onchange = () => {
  state.input = Number($("input").value);
  $("lens-family").value = "";
  renderWeights();
  loadLens().catch((e) => status(e.message));
};
for (const id of ["severity", "kind", "sort"])
  $(id).onchange = () => {
    concernsPage = 1;
    renderConcerns();
  };
document.addEventListener("keydown", (e) => {
  if (["INPUT", "SELECT", "TEXTAREA"].includes(e.target.tagName)) return;
  if (state.view === "atlas") {
    if (e.key === "f") $("flip").click();
    else if (e.key === "Backspace") {
      e.preventDefault();
      state.pins.pop();
      renderPins();
      renderMatrix();
    } else if (/[1-5]/.test(e.key) && e.key.length === 1) {
      const scale = Number(e.key) - 1;
      if (!state.pair?.features.some((f) => f.scale === scale)) return;
      state.filters.scale = String(scale);
      renderFilters();
      renderMatrix();
    } else if (["d", "g"].includes(e.key)) {
      state.filters.fragility = e.key === "d" ? "damage" : "fragility";
      renderFilters();
      renderMatrix();
    }
  }
});
async function boot() {
  try {
    [state.manifest, studio.references] = await Promise.all([
      get("data/manifest.json"),
      get("data/reference.json"),
    ]);
    options(
      $("bank"),
      (state.manifest.banks || []).map((b) => [b.id, b.label]),
    );
    options(
      $("pair"),
      state.manifest.pairs.map((p) => [p.id, p.label]),
    );
    options(
      $("bake"),
      state.manifest.bakes.map((b) => [b.id, b.label]),
    );
    options($("compare"), [
      ["", "None"],
      ...state.manifest.bakes.map((b) => [b.id, b.label]),
    ]);
    await Promise.all([selectPair(), selectBake()]);
    studio.ready();
    status(state.manifest.note || "");
    if (!state.manifest.pairs.length && state.manifest.bakes.length)
      view("lens");
    if (!state.manifest.pairs.length)
      empty($("stage"), "No atlas pair has been exported yet.");
    if (!state.manifest.bakes.length)
      empty($("concern-table"), "No bake audit has been exported yet.");
  } catch (e) {
    status(
      `Evidence unavailable: ${e.message}. Export bundles and create web/data/manifest.json.`,
    );
    empty($("stage"), "Atlas data has not been exported.");
    empty($("matrix"), "Waiting for measured feature maps.");
    empty($("concern-table"), "Waiting for the Rust weight audit.");
  }
}
const studio = new Studio({
  state,
  $,
  el,
  fmt,
  full,
  percentile,
  colour,
  preview,
  pin,
  loadMap,
  get,
  renderWeights,
  loadLens,
  status,
});
boot();
