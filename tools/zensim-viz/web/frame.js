// Preserve the evidence widgets while placing them in a single viewport.
export function buildFrame() {
  const retained = new Map(
    [...document.querySelectorAll("[id]")].map((n) => [n.id, n]),
  );
  const $ = (id) => retained.get(id);
  const node = (tag, cls, text) => {
    const n = document.createElement(tag);
    n.className = cls || "";
    if (text) n.textContent = text;
    return n;
  };
  const old = document.querySelector("header");
  const header = node("header", "topbar");
  header.append(
    node("strong", "app-name", "zensim-viz"),
    $("bake").closest("label"),
    $("pair").closest("label"),
    $("bank").closest("label"),
    old.querySelector("nav"),
    $("theme"),
  );
  const bankWrapper = node("span", "bank-picker");
  $("bank").replaceWith(bankWrapper);
  bankWrapper.append($("bank"), node("span", "bank-selected"));
  const subject = $("subject");
  subject.hidden = true;
  const workspace = node("div", "workspace");
  workspace.id = "workspace";
  const sidebar = node("aside", "feature-sidebar");
  sidebar.setAttribute("aria-label", "Features");
  sidebar.innerHTML =
    '<div class="sidebar-title">FEATURES <span id="tree-count" class="ink-3"></span></div><label class="search-label"><input id="feature-search" type="search" placeholder="Search id, signal or family…" aria-label="Search features" /><kbd>/</kbd></label><label class="twin"><input id="fragility-twin" type="checkbox" /> Fragility twin</label>';
  const matrix = $("matrix");
  matrix.setAttribute("role", "tree");
  matrix.setAttribute("aria-label", "Feature tree");
  const chipLegend = node("div", "tree-chip-legend");
  chipLegend.id = "tree-chip-legend";
  chipLegend.append(node("span", "", "p0"));
  const miniRamp = node("span", "mini-ramp");
  for (const step of [0, 1, 3, 5, 6]) {
    const mark = node("i");
    mark.style.background = `var(--ramp-seq-${step})`;
    miniRamp.append(mark);
  }
  chipLegend.append(
    miniRamp,
    node("span", "", "p100"),
    node("span", "", "chip = bank percentile of this pair's value"),
  );
  chipLegend.title = "x · y · b: chip = bank percentile of this pair's value";
  sidebar.append(chipLegend, matrix);
  const center = node("div", "center-stage");
  center.id = "center-stage";
  const atlas = $("atlas"),
    lens = $("lens"),
    stage = $("stage");
  stage.tabIndex = 0;
  stage.setAttribute(
    "aria-label",
    "Image stage: arrows pan, wheel zooms, double click resets",
  );
  atlas.replaceChildren(stage);
  atlas.className = "atlas-stage";
  const concerns = document.querySelector(".concerns");
  const concernFilters = node("div", "controls");
  for (const id of ["severity", "kind", "sort", "compare"])
    concernFilters.append($(id).closest("label"));
  concerns.prepend(concernFilters);
  concerns.dataset.exhibit = "Concerns";
  concerns.append($("comparison-panel"));
  const instrumentNote = node("p", "ink-3");
  instrumentNote.id = "instrument-note";
  concerns.append(instrumentNote);
  const network = document.querySelector(".network");
  network.dataset.exhibit = "Weights";
  const pathDetails = node("details");
  pathDetails.id = "network-path";
  pathDetails.append(node("summary", "", "Model path"), $("architecture"));
  network.querySelector("h2").remove();
  const capacity = node("div", "capacity-strip");
  capacity.id = "capacity-strip";
  capacity.setAttribute("aria-label", "Hidden-unit capacity");
  const weightAudit = node("details");
  weightAudit.id = "weight-readout";
  const weightSummary = node("summary", "", "Selected weight · detailed audit");
  weightSummary.id = "weight-readout-summary";
  weightAudit.append(weightSummary, $("weight-detail"));
  const weightControls = node("div", "weights-controls");
  weightControls.append(pathDetails, $("column-order").closest("label"));
  const weightPrimary = node("div", "weights-primary");
  weightPrimary.id = "weights-primary";
  network.querySelector("h2").remove();
  weightPrimary.append(
    capacity,
    weightControls,
    weightAudit,
    $("weight-wrap"),
    $("weight-legend"),
  );
  network.prepend(weightPrimary);
  const figures = [...document.querySelectorAll(".diagnostics > figure")];
  figures[0].dataset.exhibit = "Dependence";
  figures[1].dataset.exhibit = "Ranges";
  const attribution = node("section", "attribution-pane");
  attribution.dataset.exhibit = "Attribution";
  attribution.append(figures[2], figures[3]);
  const tabs = node("nav", "exhibit-tabs seg");
  tabs.setAttribute("aria-label", "Lens exhibit");
  for (const name of [
    "Concerns",
    "Weights",
    "Dependence",
    "Ranges",
    "Attribution",
  ]) {
    const b = node("button", "", name);
    b.dataset.exhibitTab = name;
    b.setAttribute("aria-pressed", String(name === "Dependence"));
    tabs.append(b);
  }
  const exhibits = node("div", "exhibits diagnostics");
  exhibits.append(concerns, network, figures[0], figures[1], attribution);
  for (const e of exhibits.children)
    e.hidden = e.dataset.exhibit !== "Dependence";
  lens.replaceChildren(tabs, exhibits);
  center.append(atlas, lens, $("status"));
  const reference = node("aside", "reference-panel");
  reference.id = "reference-panel";
  reference.setAttribute("aria-label", "Feature reference");
  reference.innerHTML =
    '<div class="sidebar-title">REFERENCE <button id="release-hold" hidden>Release pin</button></div><div id="reference-body"><p class="ink-2">Hover a feature to read its formula and background.</p></div><details class="reference-options"><summary>Display and model controls</summary></details>';
  const options = reference.querySelector("details");
  for (const id of ["input", "lens-family", "full-precision", "colour"])
    options.append($(id).closest("label"));
  const compatibility = node("div");
  compatibility.hidden = true;
  for (const id of [
    "subject",
    "filters",
    "family-lane",
    "family",
    "atlas-table-toggle",
    "atlas-table",
    "family-maps",
    "matrix-range",
    "matrix-ramp",
    "feature-detail",
  ])
    compatibility.append($(id));
  const gutter = (side) => {
    const g = node("div", "gutter");
    g.dataset.side = side;
    g.tabIndex = 0;
    g.setAttribute("role", "separator");
    g.setAttribute("aria-orientation", "vertical");
    g.setAttribute("aria-label", `Resize ${side} panel`);
    return g;
  };
  workspace.append(sidebar, gutter("left"), center, gutter("right"), reference);
  const bottom = node("footer", "bottom-strip");
  bottom.id = "bottom-strip";
  const modes = $("image-mode");
  modes.setAttribute("aria-label", "Stage image");
  const only = node("button", "", "Map");
  only.id = "map-mode-button";
  modes.append(only);
  const ranges = node("div", "seg");
  ranges.id = "range-mode";
  ranges.setAttribute("aria-label", "Map range");
  for (const [mode, label] of [
    ["robust", "p2–p98"],
    ["own", "own"],
    ["bank", "bank"],
  ]) {
    const button = node("button", "", label);
    button.dataset.range = mode;
    button.setAttribute("aria-pressed", String(mode === "robust"));
    ranges.append(button);
  }
  const controls = node("div", "stage-controls");
  for (const id of ["zoom", "alpha"]) controls.append($(id).closest("label"));
  controls.append($("reset-pan"));
  compatibility.append($("map-only").closest("label"), $("flip"));
  bottom.append(
    $("pins"),
    $("map-legend"),
    ranges,
    modes,
    controls,
    node(
      "span",
      "keyboard-hints",
      "/ search · arrows move · Enter pin · wheel zoom",
    ),
  );
  const tooltip = $("tooltip");
  document.body.replaceChildren(
    header,
    workspace,
    bottom,
    compatibility,
    tooltip,
  );
}
