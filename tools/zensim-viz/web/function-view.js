// Plot native samples and render stored coefficient planes; no browser scorer.
export class FunctionView {
  constructor(api) {
    Object.assign(this, api);
    this.selection = null;
    this.mode = "term";
  }
  chooseUnit(unit) {
    const d = this.state.function;
    if (!d) return;
    const pair = d.pairs.find((p) => p.units.includes(unit));
    if (!pair && !d.ordinary_terms.some((u) => u.unit === unit)) {
      this.state.unit = unit;
      this.renderWeights();
      document.querySelector('[data-exhibit-tab="Weights"]').click();
      return;
    }
    this.selection = pair ? `pair:${pair.index}` : `unit:${unit}`;
    this.state.unit = unit;
    this.renderWeights();
    document.querySelector('[data-exhibit-tab="Function"]').click();
    this.render();
  }
  render() {
    const { state, $, el, full, fmt, featureLink, table } = this,
      root = $("function-exhibit"),
      d = state.function;
    if (state.view !== "lens" || root.hidden) return;
    const signature = () =>
      `${state.bake?.id}/${$("bank").value}/${this.selection}/${this.mode}/${$("center-stage").clientWidth}/${$("center-stage").clientHeight}`;
    if (
      this.document === d &&
      this.signature === signature() &&
      root.children.length
    )
      return;
    root.replaceChildren();
    if (!d) {
      root.append(
        el(
          "p",
          "Native function explanation has not been exported for this bake and bank.",
          "empty",
        ),
      );
      return;
    }
    root.dataset.bake = state.bake.id;
    const summary = el("div", null, "function-summary");
    summary.append(
      el("h2", `${state.bake.label} · the approximating function`),
    );
    const equation = el("div", null, "function-equation");
    const tex = `p(z)\\approx ${full(d.output_bias)}${d.pairs.map((p) => `${p.coefficient < 0 ? "-" : "+"}${fmt(Math.abs(p.coefficient))}\\max(A_{${p.index + 1}},B_{${p.index + 1}})`).join("")}${d.ordinary_terms.length ? "+R(z)" : ""}`;
    equation.title =
      "Displayed output coefficients are rounded; exact stored coefficients are below. R(z) contains every remaining nonzero weighted ReLU, without a magnitude cutoff.";
    if (window.katex)
      window.katex.render(tex, equation, {
        throwOnError: false,
        displayMode: true,
      });
    else equation.textContent = tex;
    summary.append(
      equation,
      el(
        "p",
        `${d.pairs.length} certified max ${d.pairs.length === 1 ? "pair" : "pairs"} · ${d.ordinary_terms.length} other nonzero terms retained · ${d.capacity.counts.dead} structurally dead · ${d.zero_output_nondead_units?.length ?? "unexported"} other units with exactly zero output weight`,
      ),
    );
    const capacity = el(
      "p",
      `Before calibration: ≤ ${d.global_affine_forms_upper_bound ?? "unexported"} affine formulas globally; ≤ ${d.certified_line_segments_upper_bound ?? d.normalized_line_segments_upper_bound} pieces per straight line in the certified standardized-input domain. Upper bounds, not measured occupied regions.`,
      "ink-2",
    );
    capacity.title = d.certified_line_bound_scope;
    summary.append(capacity);
    const memo = el("details", null, "memorization-note");
    memo.append(
      el("summary", "Memorization: not established by this inspection"),
      el("p", d.memorization),
      el("p", d.line_bound_scope),
      el("p", d.certified_line_bound_scope),
      el(
        "p",
        `${d.global_affine_forms_upper_bound ?? "unexported"} affine formulas at most over the whole certified domain, before calibration (${d.binary_switches_upper_bound ?? "unexported"} binary switches).`,
      ),
      el("p", d.global_bound_scope),
    );
    summary.append(memo);
    root.append(summary);
    const controls = el("div", null, "function-controls"),
      select = el("select");
    select.id = "function-term";
    select.setAttribute("aria-label", "Pair or individual hinge term");
    for (const p of d.pairs) {
      const o = el(
        "option",
        `Max pair h${p.units.join(" / h")} · c ${fmt(p.coefficient)}`,
      );
      o.value = `pair:${p.index}`;
      select.append(o);
    }
    for (const u of d.ordinary_terms) {
      const o = el(
        "option",
        `h${u.unit} · ${u.status} · c ${fmt(u.coefficient)}`,
      );
      o.value = `unit:${u.unit}`;
      select.append(o);
    }
    if (![...select.options].some((o) => o.value === this.selection))
      this.selection = select.options[0]?.value;
    select.value = this.selection;
    const label = el("label", "Inspect ");
    label.append(select);
    controls.append(label);
    const modes = el("div", null, "seg");
    for (const [mode, title] of [
      ["term", "Selected term"],
      ["network", "Full network"],
      ["served", "Served score"],
    ]) {
      const b = el("button", title);
      b.dataset.functionMode = mode;
      b.setAttribute("aria-pressed", String(mode === this.mode));
      b.onclick = () => {
        this.mode = mode;
        this.render();
      };
      modes.append(b);
    }
    controls.append(modes);
    root.append(controls);
    const [kind, id] = this.selection.split(":"),
      term =
        kind === "pair"
          ? d.pairs[Number(id)]
          : d.ordinary_terms.find((u) => u.unit === Number(id));
    select.onchange = () => {
      this.selection = select.value;
      const [kind, id] = this.selection.split(":");
      state.unit =
        kind === "pair" ? d.pairs[Number(id)].units.at(-1) : Number(id);
      this.renderWeights();
      this.render();
    };
    root.dataset.term = this.selection;
    const formula = el("p", null, "term-formula");
    formula.textContent =
      kind === "pair"
        ? `${full(term.coefficient)} × [ReLU(A) + ReLU(B − A)] = ${full(term.coefficient)} × max(A, B), where A = w${term.units[0]}·z and B = (w${term.units[0]} + w${term.units[1]})·z ${term.bias < 0 ? "−" : "+"} ${full(Math.abs(term.bias))}.`
        : `${full(term.coefficient)} × ReLU(w${term.unit}·z ${term.bias < 0 ? "−" : "+"} ${full(Math.abs(term.bias))}) · ${term.status}, retained exactly.`;
    formula.title = d.algebra_scope;
    if (kind === "pair" && term.coefficient < 0)
      formula.append(
        document.createTextNode(
          " Since c < 0, this is min(cA, cB), the lower envelope in the plot.",
        ),
      );
    root.append(formula);
    for (const unit of kind === "pair" ? term.units : [term.unit])
      formula.append(document.createTextNode(" "), this.unitLink(unit));
    const axis = el("p", "Basis slice along ", "function-axis");
    axis.append(
      featureLink(term.slice.input),
      document.createTextNode(
        ` · switch at ${fmt(term.slice.threshold)} raw feature units · all other raw slots, including fragility, are zero.`,
      ),
    );
    root.append(axis);
    this.plot(root, term, kind);
    root.append(el("p", term.slice.definition, "function-footnote"));
    const coefficients = el("details");
    coefficients.id = "function-coefficients";
    coefficients.append(
      el(
        "summary",
        kind === "pair"
          ? "A, B and switch plane · every linked feature coefficient"
          : "Hinge plane · every linked feature coefficient",
      ),
    );
    const query = el("input");
    query.type = "search";
    query.placeholder = "Find feature name, id, family, channel or fragility…";
    query.setAttribute("aria-label", "Find coefficient feature");
    const wrap = el("div", null, "function-coefficient-table");
    const rows =
      kind === "pair"
        ? term.coefficients.map((r) => [r.input, r.a, r.b, r.delta])
        : (term.input_order || state.weights.inputs.map((i) => i.index)).map(
            (index) => [
              index,
              state.weights.layers[0].weights[
                index * state.weights.layers[0].out_dim + term.unit
              ],
            ],
          );
    const render = () => {
      const q = query.value.toLowerCase();
      wrap.replaceChildren(
        table(
          kind === "pair"
            ? [
                "Explained feature",
                "A coefficient on z",
                "B coefficient on z",
                "Switch B − A",
                "Input scale",
              ]
            : ["Explained feature", "Coefficient on z", "Input scale"],
          rows
            .filter((r) =>
              `${state.weights.inputs[r[0]].short_name} ${state.weights.inputs[r[0]].feature} ${r[0]}`
                .toLowerCase()
                .includes(q),
            )
            .map(([i, ...v]) => [
              featureLink(i),
              ...v.map(full),
              full(state.weights.scaler_scale[i]),
            ]),
        ),
      );
    };
    query.oninput = render;
    coefficients.append(query, wrap);
    coefficients.ontoggle = () => {
      if (coefficients.open && !wrap.childNodes.length) render();
    };
    root.append(coefficients);
    const preprocessing = el("details");
    preprocessing.append(
      el("summary", "Preprocessing, all terms and output calibration"),
      el("p", d.preprocessing),
      el("p", d.algebra_scope),
      el("p", d.output.scope),
      el(
        "p",
        `Complete network: p(z) = ${full(d.output_bias)} + Σ cⱼ ReLU(wⱼ·z + bⱼ), j ∈ {${d.included_units.map((j) => `h${j}`).join(", ")}}. Displayed pair coefficients are rounded; all exact terms are retained.`,
      ),
    );
    preprocessing.append(
      table(
        ["Stage", "Stored value"],
        [
          ["Network bias", full(d.output_bias)],
          ["Retained units", d.included_units.map((j) => `h${j}`).join(", ")],
          [
            "Tanh pin scale",
            d.output.tanh_pin_scale == null
              ? "absent"
              : full(d.output.tanh_pin_scale),
          ],
          [
            "Calibration spline",
            d.output.spline
              ? `${d.output.spline.x.length} native PCHIP knots; no-codec serving curve plotted above`
              : "absent",
          ],
          [
            "Codec calibration",
            d.output.codec_calibration_present
              ? "present; basis plot uses no codec hint"
              : "absent",
          ],
        ],
      ),
    );
    if (d.output.spline)
      preprocessing.append(
        table(
          ["Network x", "Calibrated y", "Derivative"],
          d.output.spline.x.map((x, i) => [
            full(x),
            full(d.output.spline.y[i]),
            full(d.output.spline.derivatives[i]),
          ]),
        ),
      );
    root.append(preprocessing);
    this.document = d;
    this.signature = signature();
  }
  plot(root, term, kind) {
    const { svg, fmt, full } = this,
      points = term.slice.points,
      W = Math.max(420, this.$("center-stage").clientWidth - 48),
      H = Math.max(320, this.$("center-stage").clientHeight - 280),
      L = 76,
      R = 24,
      T = 24,
      B = 52;
    const series =
      this.mode === "term"
        ? kind === "pair"
          ? [
              ["branch_a", "c × A", "--series-2", true],
              ["branch_b", "c × B", "--series-3", true],
              ["term_sum", "Native c × max(A, B)", "--series-1", false],
            ]
          : [["term_sum", "Native hinge contribution", "--series-1", false]]
        : [
            [
              this.mode,
              this.mode === "network"
                ? "Complete native network"
                : "Complete native served score",
              "--series-1",
              false,
            ],
          ];
    const values = points.flatMap((p) => series.map(([key]) => p[key])),
      lo = Math.min(...values),
      hi = Math.max(...values),
      xmin = points[0].x,
      xmax = points.at(-1).x;
    const X = (x) => L + ((x - xmin) / (xmax - xmin || 1)) * (W - L - R),
      Y = (y) => T + ((hi - y) / (hi - lo || 1)) * (H - T - B);
    const chart = svg("svg", {
      viewBox: `0 0 ${W} ${H}`,
      class: "function-chart",
      role: "img",
      "aria-label": `Native ${this.mode} basis slice`,
    });
    chart.dataset.points = points.length;
    chart.dataset.nativeReplay = term.slice.native_layer_replay_bit_identity;
    chart.onpointermove = (e) => {
      const rect = chart.getBoundingClientRect(),
        x = ((e.clientX - rect.left) / rect.width) * W;
      const point =
        points[
          Math.max(
            0,
            Math.min(
              points.length - 1,
              Math.round(((x - L) / (W - L - R)) * (points.length - 1)),
            ),
          )
        ];
      this.tooltip(
        e,
        `Native basis sample\nRaw x ${full(point.x)}\nHidden activations ${point.activations.map(full).join(" · ")}\nSelected term ${full(point.term_sum)}\nComplete network ${full(point.network)}\nServed score ${full(point.served)}`,
      );
    };
    chart.onpointerleave = this.hideTooltip;
    chart.append(
      svg(
        "text",
        { x: L, y: 14 },
        this.mode === "served"
          ? "Served score · native calibration"
          : this.mode === "network"
            ? "Complete network · before calibration"
            : "Selected term contribution · before calibration",
      ),
    );
    for (let i = 0; i < 5; i++) {
      const y = lo + ((hi - lo) * i) / 4;
      chart.append(
        svg("line", { x1: L, x2: W - R, y1: Y(y), y2: Y(y), class: "grid" }),
        svg("text", { x: L - 8, y: Y(y) + 4, "text-anchor": "end" }, fmt(y)),
      );
    }
    for (const x of [xmin, term.slice.threshold, xmax])
      chart.append(
        svg(
          "text",
          { x: X(x), y: H - B + 20, "text-anchor": "middle" },
          fmt(x),
        ),
      );
    chart.append(
      svg("line", {
        x1: X(term.slice.threshold),
        x2: X(term.slice.threshold),
        y1: T,
        y2: H - B,
        class: "function-switch",
      }),
      svg(
        "text",
        { x: X(term.slice.threshold) + 6, y: T + 12 },
        "hinge switch",
      ),
      svg("text", { x: L, y: H - 4 }, "Raw feature units · native basis slice"),
    );
    const legend = this.el("div", null, "legend");
    for (const [key, label, color, dashed] of series) {
      chart.append(
        svg("path", {
          d: points
            .map((p, i) => `${i ? "L" : "M"}${X(p.x)} ${Y(p[key])}`)
            .join(" "),
          fill: "none",
          stroke: `var(${color})`,
          "stroke-width": dashed ? 1.5 : 2.5,
          "stroke-dasharray": dashed ? "5 4" : "none",
        }),
      );
      const text = this.el("span", `${dashed ? "┄" : "━"} ${label}`);
      text.style.color = `var(${color})`;
      legend.append(text);
    }
    root.append(legend, chart);
    const numeric = this.el("details");
    numeric.id = "function-values";
    numeric.append(
      this.el("summary", "Native curve values · full precision"),
      this.table(
        [
          "Raw x",
          "Hidden activations",
          "Term contribution",
          "Network",
          "Served",
        ],
        points.map((p) => [
          full(p.x),
          p.activations.map(full).join(" · "),
          full(p.term_sum),
          full(p.network),
          full(p.served),
        ]),
      ),
    );
    root.append(numeric);
  }
}
