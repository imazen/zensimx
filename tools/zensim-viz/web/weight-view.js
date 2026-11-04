// Display orders are native-exported; selection always uses stored neuron IDs.
export class WeightView {
  constructor(api) {
    Object.assign(this, api);
    for (const [id, zoom] of [
      ["weight-matrix", false],
      ["weight-zoom", true],
    ]) {
      const canvas = this.$(id);
      canvas.onpointermove = (e) => {
        const hit = this.hit(e, zoom);
        if (!hit) return;
        const [i, j] = hit,
          l = this.state.weights.layers[0];
        this.tooltip(
          e,
          `${this.full(l.weights[i * l.out_dim + j])} · weight\ninput ${i} · ${this.state.weights.inputs[i].name} → hidden ${j}\n${this.describe(i, j)}`,
        );
      };
      canvas.onpointerleave = this.hideTooltip;
      canvas.onclick = (e) => {
        const hit = this.hit(e, zoom);
        if (hit) this.select(...hit);
      };
      canvas.onkeydown = (e) => {
        if (!this.state.weights || !e.key.startsWith("Arrow")) return;
        e.preventDefault();
        let r = this.rows.indexOf(this.state.input),
          c = this.cols.indexOf(this.state.unit);
        r = Math.max(
          0,
          Math.min(
            this.rows.length - 1,
            r + (e.key === "ArrowDown" ? 1 : e.key === "ArrowUp" ? -1 : 0),
          ),
        );
        c = Math.max(
          0,
          Math.min(
            this.cols.length - 1,
            c + (e.key === "ArrowRight" ? 1 : e.key === "ArrowLeft" ? -1 : 0),
          ),
        );
        this.select(this.rows[r], this.cols[c]);
      };
    }
    this.$("column-order").onchange = () => this.render();
  }
  hit(e, zoom) {
    if (!this.state.weights || !this.rows) return null;
    const r = e.currentTarget.getBoundingClientRect();
    let x = ((e.clientX - r.left) / r.width) * e.currentTarget.width,
      y = ((e.clientY - r.top) / r.height) * e.currentTarget.height;
    if (zoom) {
      x = Math.floor((x - 64) / 16);
      y = Math.floor((y - 24) / 16);
      if (
        x < 0 ||
        y < 0 ||
        x >= this.windowCols.length ||
        y >= this.windowRows.length
      )
        return null;
      return [this.windowRows[y], this.windowCols[x]];
    }
    return [
      this.rows[Math.max(0, Math.min(this.rows.length - 1, Math.floor(y)))],
      this.cols[Math.max(0, Math.min(this.cols.length - 1, Math.floor(x)))],
    ];
  }
  render() {
    const { state, $, css, fmt, full, el, table } = this,
      m = state.weights;
    if (!m) return;
    this.renderCapacity();
    const l = m.layers[0];
    this.rows = m.row_order || Array.from({ length: l.in_dim }, (_, i) => i);
    this.cols =
      ($("column-order").value === "selected"
        ? m.per_input_columns?.[state.input]
        : m.column_order) || Array.from({ length: l.out_dim }, (_, i) => i);
    const rp = this.rows.indexOf(state.input),
      cp = this.cols.indexOf(state.unit);
    const startR = Math.max(0, Math.min(this.rows.length - 24, rp - 12)),
      startC = Math.max(0, Math.min(this.cols.length - 16, cp - 8));
    this.windowRows = this.rows.slice(startR, startR + 24);
    this.windowCols = this.cols.slice(startC, startC + 16);
    const maximum =
      l.weights.reduce((a, b) => Math.max(a, Math.abs(b)), 0) || 1;
    const stops = [
      "--ramp-div-n3",
      "--ramp-div-n2",
      "--ramp-div-n1",
      "--ramp-div-0",
      "--ramp-div-p1",
      "--ramp-div-p2",
      "--ramp-div-p3",
    ].map((n) =>
      css(n)
        .match(/[a-f0-9]{2}/gi)
        .map((v) => parseInt(v, 16)),
    );
    const rgb = (v) => {
      if (v === 0) return stops[3];
      const t = Math.max(0, Math.min(6, (0.5 + (0.5 * v) / maximum) * 6)),
        i = Math.min(5, Math.floor(t)),
        f = t - i;
      return stops[i].map((n, k) => Math.round(n + (stops[i + 1][k] - n) * f));
    };
    const isGlobal = (i) => {
      const input = m.inputs[i],
        feature = state.pair?.features.find(
          (f) =>
            f.id === input.feature && f.fragility === (input.factor != null),
        );
      return feature
        ? !feature.map_file && feature.map_status?.startsWith("global")
        : input.pooling === "ratio";
    };
    const family = $("lens-family").value;
    const selectedRow = (i) =>
      family
        ? m.inputs[i].family === family ||
          m.inputs[i].analysis_family === family
        : i === state.input;
    const mini = $("weight-matrix");
    mini.width = l.out_dim;
    mini.height = l.in_dim;
    const ctx = mini.getContext("2d"),
      data = ctx.createImageData(mini.width, mini.height);
    for (let r = 0; r < this.rows.length; r++)
      for (let c = 0; c < this.cols.length; c++) {
        const i = this.rows[r],
          j = this.cols[c],
          k = (r * l.out_dim + c) * 4,
          color = rgb(l.weights[i * l.out_dim + j]);
        data.data[k] = color[0];
        data.data[k + 1] = color[1];
        data.data[k + 2] = color[2];
        data.data[k + 3] = selectedRow(i) || j === state.unit ? 255 : 102;
      }
    ctx.putImageData(data, 0, 0);
    ctx.strokeStyle = css("--ink");
    ctx.lineWidth = 1;
    ctx.strokeRect(0.5, rp + 0.5, l.out_dim - 1, 1);
    ctx.strokeRect(
      startC + 0.5,
      startR + 0.5,
      this.windowCols.length - 1,
      this.windowRows.length - 1,
    );
    const scroll = $("weight-minimap-scroll");
    if (
      rp < scroll.scrollTop ||
      rp > scroll.scrollTop + scroll.clientHeight - 24
    )
      scroll.scrollTop = Math.max(0, rp - scroll.clientHeight / 2);
    const zoom = $("weight-zoom");
    zoom.dataset.rows = this.windowRows.join(",");
    zoom.dataset.columns = this.windowCols.join(",");
    zoom.width = 64 + this.windowCols.length * 16;
    zoom.height = 24 + this.windowRows.length * 16;
    const z = zoom.getContext("2d"),
      flags = new Set(
        state.audit.flags
          .filter((f) => f.layer === 0 && f.from != null && f.to != null)
          .map((f) => `${f.from},${f.to}`),
      );
    z.font = "12px system-ui";
    z.textBaseline = "middle";
    this.windowCols.forEach((j, c) => {
      z.fillStyle = css("--ink-2");
      if (
        j === state.unit ||
        (c % 2 === 0 && Math.abs(c - this.windowCols.indexOf(state.unit)) > 1)
      )
        z.fillText(String(j), 64 + c * 16, 10);
    });
    this.windowRows.forEach((i, r) => {
      z.globalAlpha = 1;
      z.fillStyle = css("--ink-2");
      z.fillText(`in ${i}`, 0, 24 + r * 16 + 7);
      if (isGlobal(i)) z.fillText("◇", 48, 24 + r * 16 + 7);
      this.windowCols.forEach((j, c) => {
        z.globalAlpha = selectedRow(i) || j === state.unit ? 1 : 0.4;
        z.fillStyle = `rgb(${rgb(l.weights[i * l.out_dim + j]).join(",")})`;
        z.fillRect(64 + c * 16, 24 + r * 16, 14, 14);
        if (flags.has(`${i},${j}`)) {
          z.globalAlpha = 1;
          z.fillStyle = css("--ink");
          z.beginPath();
          z.moveTo(64 + c * 16, 24 + r * 16);
          z.lineTo(68 + c * 16, 24 + r * 16);
          z.lineTo(64 + c * 16, 28 + r * 16);
          z.fill();
        }
      });
    });
    z.globalAlpha = 1;
    z.strokeStyle = css("--ink");
    z.lineWidth = 2;
    const vr = this.windowRows.indexOf(state.input),
      vc = this.windowCols.indexOf(state.unit);
    if (vr >= 0 && vc >= 0) z.strokeRect(64 + vc * 16, 24 + vr * 16, 14, 14);
    const nonzero = m.nonzero_counts[state.input];
    $("weight-readout-summary").textContent =
      `input ${state.input} → h${state.unit} · weight ${full(l.weights[state.input * l.out_dim + state.unit])} · ${nonzero === 0 ? `all ${l.out_dim} weights zero: dead input` : `${nonzero}/${l.out_dim} nonzero`} · detailed audit`;
    $("weight-detail").textContent =
      `input ${state.input} · ${m.inputs[state.input].short_name || m.inputs[state.input].name} → h${state.unit}: ${full(l.weights[state.input * l.out_dim + state.unit])}\n${this.describe(state.input, state.unit)}`;
    $("weight-legend").textContent =
      `${fmt(-maximum)} … 0 … ${fmt(maximum)} · 1:1 minimap / selected 24×16 window · ink corner = flagged weight`;
    const globalLabel = el("span", " · ◇ global");
    globalLabel.id = "global-weight-label";
    globalLabel.title =
      "◇ marks a globally pooled input with no spatial detector map.";
    $("weight-legend").append(globalLabel);
    $("weight-table").replaceChildren(
      table(
        ["Hidden unit", "Weight", "Row z", "Column z", "Flags"],
        Array.from({ length: l.out_dim }, (_, j) => [
          j,
          full(l.weights[state.input * l.out_dim + j]),
          full(l.stats?.row_z[state.input * l.out_dim + j]),
          full(l.stats?.column_z[state.input * l.out_dim + j]),
          state.audit.flags
            .filter(
              (f) => f.layer === 0 && f.from === state.input && f.to === j,
            )
            .map((f) => f.kind)
            .join(", "),
        ]),
        (j) => this.select(state.input, j),
      ),
    );
    this.strip(
      $("unit-input-strip"),
      m.top_inputs_by_unit?.[state.unit] || this.rows.slice(0, 32),
      "input",
    );
    this.strip(
      $("input-unit-strip"),
      (m.per_input_columns?.[state.input] || this.cols).slice(0, 16),
      "unit",
    );
  }
  renderCapacity() {
    const root = this.$("capacity-strip"),
      capacity = this.state.weights?.capacity;
    if (!capacity) {
      root.textContent = "Native unit classification has not been exported.";
      return;
    }
    const active = root.contains(document.activeElement),
      counts = capacity.counts;
    root.dataset.bake = this.state.bake.id;
    root.title = capacity.rule;
    root.replaceChildren(
      this.el(
        "div",
        `${counts.live} live · ${counts.residual} residual · ${counts.dead} dead`,
        "capacity-counts",
      ),
    );
    const cells = this.el("div", null, "capacity-cells");
    cells.setAttribute("role", "group");
    cells.setAttribute("aria-label", "One cell per stored hidden unit");
    for (const unit of capacity.units) {
      const b = this.el("button");
      b.dataset.unit = unit.unit;
      b.dataset.status = unit.status;
      b.tabIndex = unit.unit === this.state.unit ? 0 : -1;
      b.setAttribute("aria-label", `h${unit.unit} ${unit.status}`);
      b.setAttribute("aria-pressed", String(unit.unit === this.state.unit));
      b.title = `h${unit.unit} · ${unit.status}\nOutput coefficient ${this.full(unit.output_coefficient)} · bias ${this.full(unit.bias)}\n${capacity.rule}`;
      b.onclick = () => this.select(this.state.input, unit.unit);
      b.onkeydown = (e) => {
        let next = unit.unit;
        if (e.key === "ArrowRight") next++;
        else if (e.key === "ArrowLeft") next--;
        else if (e.key === "Home") next = 0;
        else if (e.key === "End") next = capacity.units.length - 1;
        else return;
        e.preventDefault();
        this.select(
          this.state.input,
          Math.max(0, Math.min(capacity.units.length - 1, next)),
        );
      };
      cells.append(b);
    }
    root.append(cells);
    if (active)
      root
        .querySelector(`[data-unit="${this.state.unit}"]`)
        ?.focus({ preventScroll: true });
  }
  strip(root, order, type) {
    const { state, svg, fmt, full, tip } = this,
      m = state.weights,
      l = m.layers[0],
      values = order.map(
        (id) =>
          l.weights[
            (type === "input" ? id : state.input) * l.out_dim +
              (type === "unit" ? id : state.unit)
          ],
      ),
      max = Math.max(...values.map(Math.abs), 1e-30),
      W = 520,
      C = 350,
      R = 90;
    const node = svg("svg", {
      viewBox: `0 0 ${W} ${order.length * 26 + 26}`,
      role: "img",
      "aria-label":
        type === "input"
          ? "Top 32 inputs for selected unit"
          : "Top 16 units for selected input",
      class: "weight-strip",
    });
    node.append(
      svg("line", {
        x1: C,
        x2: C,
        y1: 0,
        y2: order.length * 26,
        class: "zero-line",
      }),
    );
    order.forEach((id, k) => {
      const v = values[k],
        length = (Math.abs(v) / max) * R,
        y = k * 26,
        g = svg("g", { tabindex: 0 });
      const name =
        type === "input"
          ? `in ${id} · ${m.inputs[id].short_name || m.inputs[id].name}`
          : `hidden ${id}`;
      g.append(
        svg("text", { x: 0, y: y + 15 }, name),
        svg("rect", {
          x: v < 0 ? C - length : C,
          y: y + 3,
          width: length,
          height: 20,
          rx: 2,
          fill: v < 0 ? "var(--ramp-div-n3)" : "var(--ramp-div-p3)",
        }),
        svg("text", { x: W - 4, y: y + 15, "text-anchor": "end" }, fmt(v)),
        svg("rect", { x: 0, y, width: W, height: 26, class: "hit" }),
      );
      tip(g, `${full(v)} · signed weight\n${name}`);
      g.onclick = () =>
        this.select(
          type === "input" ? id : state.input,
          type === "unit" ? id : state.unit,
        );
      g.onkeydown = (e) => {
        if (e.key === "Enter") g.onclick();
      };
      node.append(g);
    });
    root.replaceChildren(node);
  }
}
