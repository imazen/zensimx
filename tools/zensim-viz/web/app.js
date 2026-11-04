import { formatNumber } from "./format.mjs";
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
  const features = state.pair?.features || [];
  $("filters").replaceChildren(
    ...["scale", "channel", "fragility", "pooling"].map((name) => {
      const values =
        name === "fragility"
          ? ["damage", "fragility"]
          : [
              "all",
              ...new Set(
                features
                  .map((f) => String(f[name]))
                  .filter((v) => v !== "undefined"),
              ),
            ];
      const block = el("label", name + " ");
      block.append(segments(name, values));
      return block;
    }),
  );
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
      (!$("family-lane").checked || $("family").value === f.family),
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
  const fs = visible(),
    root = $("matrix");
  root.replaceChildren();
  const header = el("div", null, "matrix-row matrix-header");
  ["family × scale", "X", "Y", "B", "global / chroma"].forEach((s) =>
    header.append(el("span", s, "ink-2")),
  );
  root.append(header);
  const max = Math.max(
      ...state.pair.features.map((f) => Math.abs(f.value)),
      1e-30,
    ),
    signed = $("colour").value === "signed";
  $("matrix-range").textContent = signed
    ? `${fmt(-max)} … 0 … ${fmt(max)}`
    : "p0 … p100";
  $("matrix-ramp").style.background = signed
    ? "linear-gradient(to right,var(--ramp-div-n3),var(--ramp-div-0),var(--ramp-div-p3))"
    : "";
  const groups = new Map();
  const active = new Set(fs);
  const positions = state.pair.features.filter(
    (f) =>
      (f.fragility ? "fragility" : "damage") === state.filters.fragility &&
      (state.filters.scale === "all" ||
        String(f.scale) === state.filters.scale) &&
      (!$("family-lane").checked || f.family === $("family").value),
  );
  for (const f of positions) {
    const key = `${f.family} · s${f.scale}`;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(f);
  }
  for (const [label, items] of groups) {
    if (!items.some((f) => active.has(f))) continue;
    const row = el("div", null, "matrix-row");
    row.append(el("span", label, "row-label"));
    for (const channel of ["x", "y", "b", "s"]) {
      const group = el("div", null, "channel");
      for (const f of items.filter(
        (f) =>
          (["x", "y", "b"].includes(f.channel) ? f.channel : "s") === channel,
      )) {
        const b = el("button", null, "cell" + (!f.map_file ? " hollow" : ""));
        b.dataset.filtered = String(!active.has(f));
        b.style.visibility = active.has(f) ? "visible" : "hidden";
        b.tabIndex = active.has(f) ? 0 : -1;
        b.dataset.feature = f.key ?? f.id;
        b.dataset.pinned = String(
          state.pins.some((p) => p.key === f.key && p.id === f.id),
        );
        b.dataset.hover = String(f === state.feature);
        b.style.background = signed
          ? colour(0.5 + (0.5 * f.value) / max, true)
          : percentile(f) == null
            ? "transparent"
            : colour(percentile(f));
        b.setAttribute(
          "aria-label",
          `f${f.id} ${f.name}, value ${full(f.value)}, percentile ${full(percentile(f))}${f.map_file ? "" : ", no spatial map"}`,
        );
        b.addEventListener("pointerenter", () => preview(f));
        b.addEventListener("focus", () => preview(f));
        b.onclick = () => pin(f);
        b.onkeydown = matrixKey;
        b._feature = f;
        group.append(b);
      }
      row.append(group);
    }
    root.append(row);
  } // Preload the visible maps; pointer previews then use decoded bitmaps.
  fs.filter((f) => f.map_file).forEach((f) =>
    loadMap(f).catch((e) => status(e.message)),
  );
  $("atlas-table").replaceChildren(
    table(
      [
        "ID",
        "Name",
        "Scale",
        "Channel",
        "Pooling",
        "Value",
        "Percentile",
        "Map",
      ],
      fs.map((f) => [
        `f${f.id}`,
        f.name,
        f.scale,
        f.channel,
        f.pooling,
        full(f.value),
        full(percentile(f)),
        f.map_file || f.map_status || "global",
      ]),
      (i) => {
        preview(fs[i]);
        pin(fs[i]);
      },
    ),
  );
  if (!fs.includes(state.feature)) preview(fs[0]);
  else renderFamilyLane().catch((e) => status(e.message));
}
function matrixKey(e) {
  if (e.key === "Enter") {
    pin(e.currentTarget._feature);
    return;
  }
  if (!e.key.startsWith("Arrow")) return;
  e.preventDefault();
  const cells = [
      ...$("matrix").querySelectorAll(".cell:not([data-filtered=true])"),
    ],
    r = e.currentTarget.getBoundingClientRect(),
    cx = r.x + r.width / 2,
    cy = r.y + r.height / 2;
  const dir = {
    ArrowRight: [1, 0],
    ArrowLeft: [-1, 0],
    ArrowDown: [0, 1],
    ArrowUp: [0, -1],
  }[e.key];
  const candidates = cells.filter((c) => {
    const q = c.getBoundingClientRect();
    return (
      (q.x + r.width / 2 - cx) * dir[0] + (q.y + r.height / 2 - cy) * dir[1] > 1
    );
  });
  candidates.sort((a, b) => {
    const distance = (c) => {
      const q = c.getBoundingClientRect(),
        dx = q.x + r.width / 2 - cx,
        dy = q.y + r.height / 2 - cy;
      return Math.hypot(dx, dy) + Math.abs(dx * dir[1] - dy * dir[0]) * 4;
    };
    return distance(a) - distance(b);
  });
  candidates[0]?.focus();
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
async function paint(canvas, f, range) {
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
    data.data[i + 3] = 255;
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
async function preview(f) {
  if (!f) return;
  state.feature = f;
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
  await paint(painted, f);
  if (state.feature !== f) return;
  const overlay = $("overlay");
  overlay.width = painted.width;
  overlay.height = painted.height;
  overlay.getContext("2d").drawImage(painted, 0, 0);
  overlay.dataset.map = f.key ?? String(f.id);
  $("map-legend").replaceChildren(
    el("span", fmt(f.map_range[0])),
    el("span", null, "ramp"),
    el("span", fmt(f.map_range[1])),
    el("span", `Own range · pooled ${fmt(f.value)} · ${f.pooling}`),
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
  renderPins();
  renderMatrix();
}
async function renderPins() {
  const maps = state.pins.filter((f) => f.map_file),
    range = maps.length
      ? [
          Math.min(...maps.map((f) => f.map_range[0])),
          Math.max(...maps.map((f) => f.map_range[1])),
        ]
      : [0, 0];
  $("pins").replaceChildren();
  for (const f of state.pins) {
    const fig = el("figure"),
      c = el("canvas"),
      caption = el("figcaption", `f${f.id} · ${fmt(f.value)}`),
      remove = el("button", "×");
    remove.setAttribute("aria-label", `Unpin f${f.id}`);
    remove.onclick = () => pin(f);
    caption.append(remove);
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
async function selectPair() {
  const p = state.manifest.pairs.find((p) => p.id === $("pair").value);
  if (!p) return;
  const url = new URL(p.meta, location.href).href;
  const pair = await get(url);
  pair._url = url;
  state.pair = pair;
  state.pins = [];
  state.feature = null;
  options(
    $("family"),
    [...new Set(pair.features.map((f) => f.family))].map((v) => [v, v]),
  );
  $("subject").textContent = `${p.label} · ${pair.width}×${pair.height}`;
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
    empty(
      $("pdp-chart"),
      "No curve exported for this input. Export mlp-lens for its raw feature id.",
    );
    $("pdp-table").replaceChildren();
    $("pdp-legend").replaceChildren();
    $("pdp-note").textContent = "";
    $("pdp-title").textContent = "";
    return;
  }
  const p = d.pdp,
    op = other?.pdp;
  const all = [...p.ice.flat(), ...(op?.mean || [])];
  const c = chartFrame(
    Math.min(...p.x),
    Math.max(...p.x),
    Math.min(...all),
    Math.max(...all),
    p.axis_label || `f${p.feature} · feature units`,
    "Served score",
    p.percentiles.length
      ? [0, 5, 10, 15, 20].map((i) => [p.x[i], fmt(p.x[i])])
      : undefined,
  );
  if (p.percentiles.length)
    for (const i of [0, 5, 10, 15, 20])
      c.node.append(
        svg(
          "text",
          { x: c.x(p.x[i]), y: c.H - c.B + 36, "text-anchor": "middle" },
          `p${fmt(p.percentiles[i] * 100)}`,
        ),
      );
  p.ice.forEach((row) =>
    c.node.append(
      svg("path", { d: linePath(p.x, row, c.x, c.y), class: "ice" }),
    ),
  );
  c.node.append(
    svg("path", { d: linePath(p.x, p.mean, c.x, c.y), class: "line mean" }),
  );
  if (op)
    c.node.append(
      svg("path", {
        d: linePath(op.x, op.mean, c.x, c.y),
        class: "line compare",
      }),
    );
  c.node.append(
    svg(
      "text",
      {
        x: c.x(p.x.at(-1)) - 4,
        y: c.y(p.mean.at(-1)) - 8,
        "text-anchor": "end",
      },
      `${state.bake.label} ${fmt(p.mean.at(-1))}`,
    ),
  );
  if (op)
    c.node.append(
      svg(
        "text",
        { x: c.L + 4, y: c.T + 12 },
        `${$("compare").selectedOptions[0].textContent} · dashed`,
      ),
    );
  addCrosshair(
    c,
    p.x,
    (i) =>
      `${full(p.x[i])} · f${p.feature}${p.percentiles.length ? ` (p${fmt(p.percentiles[i] * 100)})` : ""}\n${full(p.mean[i])} · ${state.bake.label} mean\n${p.ice.length} observed ICE rows${op ? `\n${full(op.mean[i])} · ${$("compare").selectedOptions[0].textContent} mean` : ""}`,
  );
  $("pdp-chart").replaceChildren(c.node);
  $("pdp-title").textContent = p.family
    ? `${p.family} family`
    : `f${p.feature}`;
  const legend = el("span"),
    key = el("span", null, "key");
  legend.append(
    key,
    document.createTextNode(state.bake.label + " mean · solid"),
  );
  const ice = el("span", "ICE · 20% opacity");
  $("pdp-legend").replaceChildren(legend, ice);
  if (op) {
    const label = el("span");
    label.append(
      el("span", null, "key compare"),
      document.createTextNode(
        $("compare").selectedOptions[0].textContent + " mean · dashed",
      ),
    );
    $("pdp-legend").append(label);
  }
  $("pdp-note").textContent =
    `${p.ice.length} observed rows · ${p.family ? "0–2 × observed family damage" : "p1–p99"} · ${p.monotonicity_violations.length} strict floating-point monotonicity rises. Raw damage intervention recomputes its fragility products.`;
  $("pdp-table").replaceChildren(
    table(
      [
        "Feature value",
        "Bank percentile",
        `${state.bake.label} mean`,
        ...(op ? ["Comparison mean"] : []),
        ...p.ice.map(
          (_, i) => `ICE row ${(p.row_indices || d.row_indices)[i]}`,
        ),
      ],
      p.x.map((v, i) => [
        full(v),
        p.percentiles.length ? full(p.percentiles[i]) : "not a percentile axis",
        full(p.mean[i]),
        ...(op ? [full(op.mean[i])] : []),
        ...p.ice.map((row) => full(row[i])),
      ]),
    ),
  );
}
function rangePanel(bins, human = false) {
  const values = bins.flatMap((b) =>
      human ? b.human.slice(1) : [b.p10, b.p90],
    ),
    c = chartFrame(
      0,
      9,
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
      x = c.x(b.bin),
      g = svg("g", { tabindex: 0 });
    g.append(
      svg("line", {
        x1: x,
        x2: x,
        y1: c.y(low),
        y2: c.y(high),
        class: "whisker",
      }),
    );
    for (const v of [low, high])
      g.append(
        svg("line", {
          x1: x - 6,
          x2: x + 6,
          y1: c.y(v),
          y2: c.y(v),
          class: "whisker",
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
function renderConcerns() {
  if (!state.audit) return;
  const a = state.audit;
  const allFlags = [...a.flags, ...(state.bake.monotonicity_flags || [])];
  let flags = allFlags.filter(
    (f) =>
      (!$("severity").value || f.severity === $("severity").value) &&
      (!$("kind").value || f.kind === $("kind").value),
  );
  const priorities = { critical: 0, serious: 1, warning: 2, note: 3 };
  flags.sort(
    (a, b) =>
      priorities[a.severity] - priorities[b.severity] ||
      Math.abs(b.statistic ?? b.value) - Math.abs(a.statistic ?? a.value),
  );
  if ($("sort").value === "magnitude")
    flags = flags.toSorted(
      (a, b) =>
        Math.abs(b.statistic ?? b.value) - Math.abs(a.statistic ?? a.value),
    );
  $("concern-title").textContent =
    `Concerns (${flags.length} of ${allFlags.length})`;
  $("audit-summary").textContent =
    `${a.weight_count.toLocaleString()} weights · ${a.sampled_rows} recorded rows of ${a.bank_rows.toLocaleString()} · bank ${a.bank_sha256.slice(0, 12)} · paired constraints ${a.constraints?.max_pairs?.length ?? "unavailable"}. MAD outliers are diagnostics.`;
  const selected = flags.slice(0, concernsPage * 100);
  const t = table(
    [
      "Severity",
      "Kind",
      "Location",
      "Value",
      "Statistic / threshold",
      "Reason",
    ],
    selected.map((f) => [
      severity(f),
      f.kind,
      f.location,
      fmt(f.value),
      `${fmt(f.statistic)} / ${fmt(f.threshold)}`,
      f.reason,
    ]),
    (i) => {
      const f = selected[i];
      if (f.feature != null)
        state.input = Math.max(
          0,
          state.weights.inputs.findIndex((i) => i.feature === f.feature),
        );
      if (f.layer === 0) {
        state.input = f.from ?? state.input;
        state.unit = f.to ?? state.unit;
      } else if (f.layer === 1) {
        state.unit = f.from ?? state.unit;
      }
      if (f.layer !== 0)
        status(
          `Selected ${f.location}; this matrix shows layer 0. Full output weights are in the layer table.`,
        );
      $("input").value = state.input;
      renderWeights();
      loadLens().catch((e) => status(e.message));
    },
  );
  $("concern-table").replaceChildren(t);
  if (selected.length < flags.length) {
    const more = el(
      "button",
      `Show next 100 · ${flags.length - selected.length} remaining`,
    );
    more.onclick = () => {
      concernsPage++;
      renderConcerns();
    };
    $("concern-table").append(more);
  }
}
function renderArchitecture() {
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
  let text = stats
    ? `Row z ${stats.row_z[k] == null ? "undefined (MAD=0)" : full(stats.row_z[k])} · column z ${stats.column_z[k] == null ? "undefined (MAD=0)" : full(stats.column_z[k])}
Row median/MAD ${stats.rows[i].map(full).join(" / ")}
Column median/MAD ${stats.columns[j].map(full).join(" / ")}`
    : "Per-weight z statistics are not present in this bundle.";
  const pair = state.audit.constraints?.max_pairs?.find((p) => p.includes(j));
  if (pair)
    text += `
Coupled max pair h${pair[0]} / h${pair[1]}`;
  if (state.weights.layers[1].weights[j] === 0)
    text += `\nOutgoing weight is exactly zero`;
  return text;
}
function renderWeights() {
  if (!state.weights) return;
  const l = state.weights.layers[0],
    c = $("weight-matrix"),
    ctx = c.getContext("2d"),
    scale = 3;
  c.width = l.out_dim * scale;
  c.height = l.in_dim * scale;
  const neutral = css("--ramp-div-0"),
    dim = Number(css("--dim"));
  const palette = Array.from({ length: 256 }, (_, i) => colour(i / 255, true));
  const max = l.weights.reduce((m, v) => Math.max(m, Math.abs(v)), 0),
    flagged = new Set(
      state.audit.flags
        .filter((f) => f.layer === 0 && f.from != null && f.to != null)
        .map((f) => `${f.from},${f.to}`),
    );
  for (let i = 0; i < l.in_dim; i++)
    for (let j = 0; j < l.out_dim; j++) {
      ctx.globalAlpha =
        ($("lens-family").value
          ? state.weights.inputs[i].family === $("lens-family").value
          : i === state.input) || j === state.unit
          ? 1
          : dim;
      ctx.fillStyle =
        l.weights[i * l.out_dim + j] === 0
          ? neutral
          : palette[
              Math.max(
                0,
                Math.min(
                  255,
                  Math.round(
                    (0.5 + (0.5 * l.weights[i * l.out_dim + j]) / (max || 1)) *
                      255,
                  ),
                ),
              )
            ];
      ctx.fillRect(j * scale, i * scale, scale, scale);
    }
  ctx.globalAlpha = 1;
  ctx.fillStyle = css("--ink");
  for (const key of flagged) {
    const [i, j] = key.split(",").map(Number);
    ctx.fillRect(j * scale, i * scale, 1, 1);
  }
  ctx.strokeStyle = css("--ink");
  ctx.strokeRect(state.unit * scale, state.input * scale, scale, scale);
  const value = l.weights[state.input * l.out_dim + state.unit];
  const flags = state.audit.flags.filter(
    (f) => f.layer === 0 && f.from === state.input && f.to === state.unit,
  );
  $("weight-detail").textContent =
    `input ${state.input} ${state.weights.inputs[state.input]?.name} → h${state.unit}: ${full(value)}\n${weightStatisticsText(state.input, state.unit)}${flags.length ? " · ⚠ " + flags.map((f) => f.kind).join(", ") : ""}`;
  $("weight-legend").replaceChildren(
    el("span", fmt(-max)),
    el("span", null, "ramp"),
    el("span", fmt(max)),
    el(
      "span",
      "⚠ flagged cells have ink markers; details in the concern table",
    ),
  );
  $("weight-legend").querySelector(".ramp").style.background =
    "linear-gradient(to right,var(--ramp-div-n3),var(--ramp-div-0),var(--ramp-div-p3))";
  $("weight-table").replaceChildren(
    table(
      ["Hidden unit", "Weight", "Flags"],
      Array.from({ length: l.out_dim }, (_, j) => [
        j,
        full(l.weights[state.input * l.out_dim + j]),
        state.audit.flags
          .filter((f) => f.layer === 0 && f.from === state.input && f.to === j)
          .map((f) => f.kind)
          .join(", "),
      ]),
      (j) => {
        state.unit = j;
        renderWeights();
      },
    ),
  );
  const wrap = $("weight-wrap"),
    position = (state.input / l.in_dim) * c.clientHeight;
  if (
    position < wrap.scrollTop ||
    position > wrap.scrollTop + wrap.clientHeight - 8
  )
    wrap.scrollTop = Math.max(0, position - wrap.clientHeight / 2);
  renderArchitecture();
}
$("weight-matrix").onpointermove = (e) => {
  if (!state.weights) return;
  const r = e.currentTarget.getBoundingClientRect(),
    l = state.weights.layers[0],
    i = Math.min(
      l.in_dim - 1,
      Math.floor(((e.clientY - r.top) / r.height) * l.in_dim),
    ),
    j = Math.min(
      l.out_dim - 1,
      Math.floor(((e.clientX - r.left) / r.width) * l.out_dim),
    );
  tooltip(
    e,
    `${full(l.weights[i * l.out_dim + j])} · weight\ninput ${i} (${state.weights.inputs[i]?.name}) → hidden ${j}\n${weightStatisticsText(i, j)}\n${state.audit.flags
      .filter((f) => f.layer === 0 && f.from === i && f.to === j)
      .map((f) => `${f.severity} · ${f.kind}: ${f.reason}`)
      .join("\n")}`,
  );
};
$("weight-matrix").onpointerleave = hideTooltip;
$("weight-matrix").onclick = (e) => {
  const r = e.currentTarget.getBoundingClientRect(),
    l = state.weights.layers[0];
  state.input = Math.min(
    l.in_dim - 1,
    Math.floor(((e.clientY - r.top) / r.height) * l.in_dim),
  );
  state.unit = Math.min(
    l.out_dim - 1,
    Math.floor(((e.clientX - r.left) / r.width) * l.out_dim),
  );
  $("input").value = state.input;
  renderWeights();
  loadLens().catch((e) => status(e.message));
};
$("weight-matrix").onkeydown = (e) => {
  if (!state.weights) return;
  const l = state.weights.layers[0];
  if (e.key.startsWith("Arrow")) {
    e.preventDefault();
    state.input = Math.max(
      0,
      Math.min(
        l.in_dim - 1,
        state.input +
          (e.key === "ArrowDown" ? 1 : e.key === "ArrowUp" ? -1 : 0),
      ),
    );
    state.unit = Math.max(
      0,
      Math.min(
        l.out_dim - 1,
        state.unit +
          (e.key === "ArrowRight" ? 1 : e.key === "ArrowLeft" ? -1 : 0),
      ),
    );
    $("input").value = state.input;
    renderWeights();
    loadLens().catch((e) => status(e.message));
  }
};
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
    L = 100,
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
      svg(
        "text",
        { x: L - 12, y: y + 16, "text-anchor": "end" },
        `f${r.features[0]}`,
      ),
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
  document.querySelector(".diagnostics").classList.remove("loading");
}
async function selectBake() {
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
    status("This bake has no exported evidence for the selected bank.");
    return;
  }
  state.bake = bake;
  [state.audit, state.weights] = await Promise.all([
    get(bake.audit),
    get(bake.weights),
  ]);
  state.input = 0;
  state.unit = 0;
  options(
    $("input"),
    state.weights.inputs.map((i) => [i.index, `${i.index} · ${i.name}`]),
  );
  options($("lens-family"), [
    ["", "Selected input"],
    ...[...new Set(state.weights.inputs.map((i) => i.family))].map((f) => [
      f,
      f,
    ]),
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
    state.manifest = await get("data/manifest.json");
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
boot();
