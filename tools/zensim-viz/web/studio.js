// The studio joins stored IDs, native evidence and paper references. It does
// not calculate metric values or model statistics.
export class Studio {
  constructor(api) {
    Object.assign(this, api);
    this.closed = new Set();
    this.hold = null;
    this.rows = [];
    this.query = "";
    this.$("feature-search").oninput = (e) => {
      this.query = e.target.value.trim().toLowerCase();
      this.$("matrix").scrollTop = 0;
      this.renderTree();
    };
    this.$("fragility-twin").onchange = (e) => {
      this.state.filters.fragility = e.target.checked ? "fragility" : "damage";
      this.renderTree();
    };
    this.$("matrix").onscroll = () => this.drawRows();
    this.$("release-hold").onclick = () => {
      this.hold = null;
      this.reference(this.state.feature, true);
    };
    for (const tab of document.querySelectorAll("[data-exhibit-tab]"))
      tab.onclick = () => {
        this.exhibit = tab.dataset.exhibitTab;
        document
          .querySelectorAll("[data-exhibit-tab]")
          .forEach((b) => b.setAttribute("aria-pressed", String(b === tab)));
        document
          .querySelectorAll("[data-exhibit]")
          .forEach((p) => (p.hidden = p.dataset.exhibit !== this.exhibit));
        this.updateLegend();
      };
    this.exhibit = "Dependence";
    this.$("stage").addEventListener(
      "wheel",
      (e) => {
        e.preventDefault();
        const zoom = this.$("zoom");
        zoom.value = Math.max(
          1,
          Math.min(8, Number(zoom.value) * (e.deltaY < 0 ? 1.12 : 1 / 1.12)),
        );
        zoom.dispatchEvent(new Event("input"));
      },
      { passive: false },
    );
    this.$("stage").ondblclick = () => this.$("reset-pan").click();
    this.$("map-mode-button").onclick = () => {
      const c = this.$("map-only");
      c.checked = !c.checked;
      c.dispatchEvent(new Event("change"));
      this.$("map-mode-button").setAttribute("aria-pressed", String(c.checked));
    };
    for (const b of document.querySelectorAll("[data-mode]"))
      b.addEventListener("click", () => {
        const c = this.$("map-only");
        c.checked = false;
        c.dispatchEvent(new Event("change"));
        this.$("map-mode-button").setAttribute("aria-pressed", "false");
      });
    document.addEventListener("keydown", (e) => {
      if (
        e.key === "/" &&
        !["INPUT", "SELECT", "TEXTAREA"].includes(e.target.tagName)
      ) {
        e.preventDefault();
        this.$("feature-search").focus();
      }
    });
    const fit = () => {
      const img = this.$("pair-image"),
        box = this.$("stage");
      if (!img.naturalWidth || !box.clientWidth) return;
      const ratio = Math.min(
        box.clientWidth / img.naturalWidth,
        box.clientHeight / img.naturalHeight,
      );
      document.documentElement.style.setProperty(
        "--image-w",
        `${img.naturalWidth * ratio}px`,
      );
      document.documentElement.style.setProperty(
        "--image-h",
        `${img.naturalHeight * ratio}px`,
      );
    };
    this.$("pair-image").addEventListener("load", fit);
    new ResizeObserver(() => {
      fit();
      this.drawRows();
    }).observe(this.$("center-stage"));
    for (const gutter of document.querySelectorAll(".gutter")) {
      const resize = (delta) => {
        const root = this.$("workspace"),
          left = gutter.dataset.side === "left",
          key = left ? "--left" : "--right",
          other = parseFloat(
            getComputedStyle(root).getPropertyValue(
              left ? "--right" : "--left",
            ),
          );
        const cap = Math.min(
          480,
          innerWidth * (innerWidth >= 1500 ? 0.45 : 0.6) - 12 - other,
        );
        const value = Math.max(
          240,
          Math.min(
            cap,
            parseFloat(getComputedStyle(root).getPropertyValue(key)) + delta,
          ),
        );
        root.style.setProperty(key, `${value}px`);
        gutter.setAttribute("aria-valuenow", String(value));
        gutter.setAttribute("aria-valuemin", "240");
        gutter.setAttribute("aria-valuemax", String(cap));
      };
      let prior = null;
      gutter.onpointerdown = (e) => {
        prior = e.clientX;
        gutter.setPointerCapture(e.pointerId);
      };
      gutter.onpointermove = (e) => {
        if (prior == null) return;
        resize((e.clientX - prior) * (gutter.dataset.side === "left" ? 1 : -1));
        prior = e.clientX;
      };
      gutter.onpointerup = gutter.onlostpointercapture = () => (prior = null);
      gutter.onkeydown = (e) => {
        if (["ArrowLeft", "ArrowRight"].includes(e.key)) {
          e.preventDefault();
          resize(
            (e.key === "ArrowRight" ? 16 : -16) *
              (gutter.dataset.side === "left" ? 1 : -1),
          );
        }
      };
    }
    window.addEventListener("resize", () => {
      const root = this.$("workspace");
      root.style.removeProperty("--left");
      root.style.removeProperty("--right");
    });
    this.$("stage").addEventListener("pointermove", (e) => this.demoAt(e));
  }
  ready() {
    this.renderTree();
    this.reference(this.state.feature, true);
    // This is also the complete source-ID join used by the browser acceptance.
    window.zensimStudio = {
      catalog: () =>
        this.state.pair.features.map((f) => ({
          key: f.key,
          id: f.id,
          reference: `${f.family}/${f.signal}`,
          fragility: f.fragility,
        })),
      references: () => this.references,
      select: (key) =>
        this.choose(this.state.pair.features.find((f) => f.key === key)),
      preload: (family) => this.preload(family),
      state: () => ({
        input: this.state.input,
        unit: this.state.unit,
        feature: this.state.feature?.key,
        rows: this.rows.length,
      }),
      demo: (x, y) => this.demoAt({ clientX: x, clientY: y }),
    };
    document.body.dataset.instrument = this.state.view;
    this.preload("basic");
  }
  async preload(family) {
    await Promise.all(
      this.state.pair.features
        .filter((f) => f.family === family && !f.fragility)
        .map((f) => this.loadMap(f)),
    );
  }
  choose(f, force = false) {
    if (!f) return;
    this.showUnit = false;
    if (this.hold && !force) return;
    if (f === this.state.feature && this.state.view !== "lens") return;
    this.preview(f).catch((e) => this.status(e.message));
    if (this.state.view === "lens" && this.state.weights) {
      const input = this.state.weights.inputs.find(
        (i) => i.feature === f.id && Boolean(i.factor != null) === f.fragility,
      );
      if (input) {
        if (
          this.state.input === input.index &&
          this.state.lens &&
          !this.$("lens-family").value
        )
          return;
        this.state.input = input.index;
        this.$("input").value = input.index;
        this.$("lens-family").value = "";
        this.renderWeights();
        this.loadLens().catch((e) => this.status(e.message));
      } else {
        this.$("pdp-title").textContent = `f${f.id} · not consumed`;
        this.$("pdp-chart").replaceChildren(
          this.el(
            "p",
            "This input is not consumed by the selected bake. Its raw intervention is available in the exported bank curves.",
            "empty",
          ),
        );
      }
    }
  }
  renderTree() {
    const fs = (this.state.pair?.features || []).filter(
      (f) =>
        f.fragility === (this.state.filters.fragility === "fragility") &&
        ["scale", "channel", "pooling"].every(
          (k) =>
            this.state.filters[k] === "all" ||
            String(f[k]) === this.state.filters[k],
        ),
    );
    const search = this.query,
      match = (f) =>
        !search ||
        `${f.id} f${f.id} ${f.key} ${f.name} ${f.family} ${f.signal}`
          .toLowerCase()
          .includes(search);
    const groups = new Map();
    for (const f of fs.filter(match)) {
      if (!groups.has(f.family)) groups.set(f.family, new Map());
      const signals = groups.get(f.family);
      if (!signals.has(f.signal)) signals.set(f.signal, []);
      signals.get(f.signal).push(f);
    }
    const rows = [];
    const summary = (features) =>
      Math.max(0, ...features.map((f) => this.percentile(f) || 0));
    for (const [family, signals] of groups) {
      const all = [...signals.values()].flat();
      rows.push({
        kind: "family",
        label: family,
        key: family,
        p: summary(all),
      });
      if (!search && this.closed.has(family)) continue;
      for (const [signal, features] of signals) {
        const key = `${family}/${signal}`;
        rows.push({ kind: "signal", label: signal, key, p: summary(features) });
        if (!search && this.closed.has(key)) continue;
        const scales = new Map();
        for (const f of features) {
          const k = `${f.scale}·${f.pooling}`;
          if (!scales.has(k)) scales.set(k, []);
          scales.get(k).push(f);
        }
        for (const [k, items] of scales)
          rows.push({
            kind: "leaf",
            label: `s${k.replace("·", " · ")}`,
            features: items,
            key: `${key}/${k}`,
          });
      }
    }
    this.rows = rows;
    this.$("tree-count").textContent = `${fs.filter(match).length} inputs`;
    const spacer = this.el("div", null, "tree-spacer");
    spacer.style.height = `${rows.length * 32}px`;
    this.$("matrix").replaceChildren(spacer);
    this.drawRows();
  }
  drawRows() {
    const root = this.$("matrix"),
      spacer = root.firstElementChild;
    if (!spacer?.classList.contains("tree-spacer")) return;
    const start = Math.max(0, Math.floor(root.scrollTop / 32) - 3),
      end = Math.min(
        this.rows.length,
        Math.ceil((root.scrollTop + root.clientHeight) / 32) + 3,
      );
    const active = document.activeElement?.dataset.feature;
    spacer.replaceChildren();
    for (let index = start; index < end; index++) {
      const row = this.rows[index],
        n = this.el("div", null, `tree-row ${row.kind}`);
      n.style.top = `${index * 32}px`;
      n.dataset.row = String(index);
      if (row.kind !== "leaf") {
        const b = this.el("button", null, "tree-header"),
          open = this.query || !this.closed.has(row.key);
        b.setAttribute("role", "treeitem");
        b.setAttribute("aria-level", row.kind === "family" ? "1" : "2");
        b.setAttribute("aria-expanded", String(Boolean(open)));
        b.append(this.el("span", open ? "▾" : "▸"), this.el("span", row.label));
        const chip = this.el("i", null, "percentile-chip");
        chip.style.background = this.colour(row.p);
        chip.title = `Max child p${this.fmt(row.p * 100)}`;
        b.append(chip);
        b.onclick = () => {
          this.closed.has(row.key)
            ? this.closed.delete(row.key)
            : this.closed.add(row.key);
          this.renderTree();
        };
        b.onkeydown = (e) => this.navigate(e, index);
        n.append(b);
      } else {
        n.append(this.el("span", row.label, "scale-label"));
        for (const ch of ["x", "y", "b", "xy"]) {
          const f = row.features.find((f) => f.channel === ch);
          if (!f) continue;
          const b = this.el(
            "button",
            null,
            `cell${f.map_file ? "" : " global-cell"}`,
          );
          b._feature = f;
          b.dataset.feature = f.key;
          b.dataset.id = f.id;
          b.dataset.pinned = String(this.state.pins.includes(f));
          b.dataset.hover = String(this.state.feature === f);
          b.setAttribute("role", "treeitem");
          b.setAttribute("aria-level", "3");
          b.setAttribute(
            "aria-label",
            `f${f.id} ${f.name} ${ch} · ${this.fmt(f.value)} · p${this.fmt(100 * (this.percentile(f) || 0))}`,
          );
          const pixel = this.el("i", null, "pixel percentile-chip");
          const bank = this.state.manifest.banks.find(
              (b) => b.id === this.$("bank").value,
            ),
            range = f.bank_ranges?.[bank?.sha256];
          const divergent =
            this.$("colour").value === "signed" &&
            range?.[0] < 0 &&
            range?.[1] > 0;
          b.dataset.diverging = String(Boolean(divergent));
          pixel.style.background = divergent
            ? this.colour(
                (f.value / Math.max(Math.abs(range[0]), Math.abs(range[1])) +
                  1) /
                  2,
                true,
              )
            : this.colour(this.percentile(f) || 0);
          b.append(pixel, this.el("span", ch));
          if (this.state.view === "lens") {
            const attrs = this.state.lens?.input_attribution || [],
              a = attrs.find((a) => a.features.includes(f.id)),
              max = attrs[0]?.mean_abs_delta;
            if (a && max) {
              const mark = this.el("i", null, "ablation-mark");
              mark.style.width = `${(34 * a.mean_abs_delta) / max}px`;
              b.append(mark);
            }
          }
          b.title =
            b.getAttribute("aria-label") +
            (f.map_file ? "" : " · global statistic");
          b.onpointerenter = () => this.choose(f);
          b.onfocus = () => this.choose(f);
          b.onclick = () => {
            this.choose(f);
            this.pin(f);
          };
          b.onkeydown = (e) => {
            if (e.key === "Enter" || e.key === " ") {
              e.preventDefault();
              this.pin(f);
            } else this.navigate(e, index, ch);
          };
          n.append(b);
        }
      }
      spacer.append(n);
    }
    if (active) {
      const b = spacer.querySelector(`[data-feature="${active}"]`);
      if (b && document.activeElement === document.body)
        b.focus({ preventScroll: true });
    }
  }
  navigate(e, index, ch = "x") {
    if (!e.key.startsWith("Arrow")) return;
    e.preventDefault();
    if (
      ["ArrowLeft", "ArrowRight"].includes(e.key) &&
      this.rows[index].kind !== "leaf"
    ) {
      const row = this.rows[index];
      if (e.key === "ArrowLeft") this.closed.add(row.key);
      else this.closed.delete(row.key);
      this.renderTree();
      return;
    }
    const horizontal = ["ArrowLeft", "ArrowRight"].includes(e.key);
    let target = index;
    if (!horizontal)
      target = Math.max(
        0,
        Math.min(
          this.rows.length - 1,
          index + (e.key === "ArrowDown" ? 1 : -1),
        ),
      );
    const root = this.$("matrix");
    if (target * 32 < root.scrollTop) root.scrollTop = target * 32;
    if ((target + 1) * 32 > root.scrollTop + root.clientHeight)
      root.scrollTop = (target + 1) * 32 - root.clientHeight;
    this.drawRows();
    const n = root.querySelector(`[data-row="${target}"]`),
      buttons = [...n.querySelectorAll("button")];
    let focus = buttons[0];
    if (horizontal) {
      const current = buttons.findIndex((b) => b._feature?.channel === ch);
      focus =
        buttons[
          Math.max(
            0,
            Math.min(
              buttons.length - 1,
              current + (e.key === "ArrowRight" ? 1 : -1),
            ),
          )
        ];
    } else focus = buttons.find((b) => b._feature?.channel === ch) || focus;
    focus?.focus({ preventScroll: true });
  }
  reference(f, force = false) {
    if (!f || !this.references) return;
    if (this.hold && !force) f = this.hold;
    this.$("release-hold").hidden = !this.hold;
    const r = this.references.entries[`${f.family}/${f.signal}`],
      root = this.$("reference-body");
    if (!r) return;
    root.dataset.feature = f.key;
    root.replaceChildren();
    root.append(
      this.el("div", `f${f.id} · ${f.short_name || f.name}`, "reference-id"),
      this.el(
        "div",
        `${f.family} · ${f.signal} · s${f.scale} · ${f.channel.toUpperCase()} · ${f.pooling}${f.fragility ? " · fragility product" : ""}`,
        "reference-meta",
      ),
      this.el(
        "div",
        `${this.fmt(f.value)} pooled · ${this.percentile(f) == null ? "percentile unavailable" : `p${this.fmt(this.percentile(f) * 100)}`}`,
        "reference-value",
      ),
    );
    root.append(this.el("h2", "Formula"));
    const formula = this.el("div", null, "formula");
    root.append(formula);
    const tex = f.fragility
      ? this.references.entries.fragility.formula
      : r.formula;
    if (tex && window.katex) {
      try {
        window.katex.render(tex, formula, {
          throwOnError: true,
          displayMode: true,
          trust: false,
        });
      } catch (e) {
        formula.textContent = tex;
        this.status(`Formula render failed: ${e.message}`);
      }
    } else
      formula.textContent =
        tex || "Formula not transcribed in the paper draft.";
    root.append(
      this.el(
        "div",
        tex
          ? "Code-derived equation · revision 7. Paper coverage is identified below."
          : "No equation available; see code below.",
        "formula-note",
      ),
    );
    root.append(
      this.el("h2", "What it detects"),
      this.el("p", r.detects),
      this.el("h2", "Background and inspiration"),
      this.el("p", r.background),
    );
    if (r.notation) root.append(this.el("p", r.notation, "formula-note"));
    if (!r.paper_text)
      root.append(this.el("p", "no paper text yet", "reference-missing"));
    if (f.fragility)
      root.append(
        this.el(
          "p",
          this.references.entries.fragility.detects +
            " · no paper text yet for the product.",
          "reference-missing",
        ),
      );
    const citations = this.el("ul");
    for (const c of r.inspiration) {
      const li = this.el("li"),
        a = this.el("a", `${c.author_year} · ${c.title}`);
      a.href = c.url;
      a.target = "_blank";
      a.rel = "noopener";
      li.append(a);
      citations.append(li);
    }
    root.append(citations);
    const paper = this.el("a", "open in the paper");
    paper.href = r.section;
    paper.target = "_blank";
    paper.rel = "noopener";
    root.append(paper, document.createTextNode(" · "));
    const code = this.el("a", "code");
    code.href = r.code;
    code.target = "_blank";
    code.rel = "noopener";
    root.append(code);
    root.append(this.el("h2", "At the cursor"));
    const demo = this.el(
      "div",
      f.family === "basic" && f.id % 13 < 10
        ? "Move over the image to inspect its sampled native window."
        : "terms not exported for this family",
      "demo",
    );
    demo.id = "formula-demo";
    root.append(demo);
    root.append(
      this.el(
        "h2",
        this.showUnit ? `Hidden unit h${this.state.unit}` : "In this model",
      ),
    );
    const w = this.state.weights,
      i = w?.inputs.find(
        (i) => i.feature === f.id && Boolean(i.factor != null) === f.fragility,
      );
    if (!w || !i)
      root.append(
        this.el(
          "p",
          "Not consumed by the selected bake, or no bank evidence exported.",
        ),
      );
    else {
      const attrs = this.state.lens?.input_attribution || [],
        rank = attrs.findIndex((a) => a.features.includes(f.id));
      root.append(
        this.el(
          "p",
          `Input ${i.index} · total |weight| ${w.input_l1 ? this.fmt(w.input_l1[i.index]) : "not exported"} · ${w.nonzero_counts[i.index]} of ${w.layers[0].out_dim} units read it.`,
        ),
      );
      root.append(
        this.el(
          "p",
          rank < 0
            ? "Ablation ranking loading…"
            : `Ablation rank ${rank + 1} of ${attrs.length} · mean |score change| ${this.fmt(attrs[rank].mean_abs_delta)}. Raw-feature intervention also changes its fragility twin.`,
        ),
      );
      root.append(
        this.el(
          "p",
          `Strongest readers: ${
            (w.per_input_columns[i.index] || [])
              .filter(
                (j) =>
                  w.layers[0].weights[i.index * w.layers[0].out_dim + j] !== 0,
              )
              .slice(0, 8)
              .map((j) => `h${j}`)
              .join(", ") || "none"
          }.`,
        ),
      );
      if (this.showUnit) {
        const unit = this.state.audit?.hidden_units.find(
          (u) => u.unit === this.state.unit || u.index === this.state.unit,
        );
        if (unit) {
          root.append(
            this.el(
              "p",
              `${this.fmt(unit.activation_rate * 100)}% positive · mean |activation| ${this.fmt(unit.mean_abs_activation)} · range ${this.fmt(unit.min)} … ${this.fmt(unit.max)} · outgoing weight ${this.fmt(w.layers[1].weights[this.state.unit])}.`,
            ),
          );
          root.append(
            this.el(
              "p",
              `Top inputs: ${unit.top_inputs
                .slice(0, 8)
                .map(
                  ([idx]) =>
                    `f${w.inputs[idx].feature} ${w.inputs[idx].short_name}`,
                )
                .join("; ")}.`,
            ),
          );
        }
      }
      const flags = [
        ...(this.state.audit?.flags || []),
        ...(this.state.bake?.monotonicity_flags || []),
      ].filter(
        (flag) =>
          flag.feature === f.id ||
          (flag.layer === 0 && flag.from === i.index) ||
          (this.showUnit &&
            ((flag.layer === 0 && flag.to === this.state.unit) ||
              (flag.layer === 1 && flag.from === this.state.unit))),
      );
      root.dataset.flagCount = String(flags.length);
      const kinds = new Map();
      for (const flag of flags) {
        const key = `${flag.severity} ${flag.kind}`;
        kinds.set(key, (kinds.get(key) || 0) + 1);
      }
      for (const [key, count] of kinds) {
        const icon =
          key.startsWith("critical") || key.startsWith("serious")
            ? "⚠"
            : key.startsWith("note")
              ? "○"
              : "●";
        const chip = this.el("span", `${icon} ${key} · ${count}`, "chip");
        chip.dataset.severity = key.split(" ")[0];
        root.append(chip);
      }
      if (!flags.length)
        root.append(this.el("p", "No measured flag on this feature.", "ink-3"));
    }
    if (f.terms_file)
      this.termPlanes(new URL(f.terms_file, this.state.pair._url).href).catch(
        (e) => this.status(e.message),
      );
    const numbers = this.el("details");
    numbers.append(this.el("summary", "Stored values · full precision"));
    numbers.append(
      this.table(
        ["Field", "Value"],
        [
          ["Pooled slot", f.pooled_value],
          ["Model input", f.model_input_value],
          ["Bank percentile", this.percentile(f)],
          ...(f.fragility
            ? [
                ["Fragility slot", f.fragility_slot],
                ["Fragility value", f.fragility_value],
              ]
            : []),
        ],
      ),
    );
    root.append(numbers);
    this.updateLegend();
  }
  termPlanes(url) {
    this.termCache ||= new Map();
    if (!this.termCache.has(url))
      this.termCache.set(
        url,
        (async () => {
          const header = await this.get(url),
            response = await fetch(new URL(header.data_file, url));
          if (!response.ok)
            throw Error(`Missing native terms: ${response.status}`);
          const bytes = await response.arrayBuffer(),
            count = header.width * header.height;
          if (bytes.byteLength !== count * header.planes.length * 4)
            throw Error("Native term plane length differs");
          const view = new DataView(bytes),
            planes = {};
          header.planes.forEach((name, p) => {
            const values = new Float32Array(count);
            for (let i = 0; i < count; i++)
              values[i] = view.getFloat32((p * count + i) * 4, true);
            planes[name] = values;
          });
          return { ...header, planes };
        })(),
      );
    return this.termCache.get(url);
  }
  updateLegend() {
    if (this.state.view === "lens")
      this.$("map-legend").textContent =
        `${this.exhibit} · ${this.state.bake?.label || "No evidence"} · input ${this.state.input} · h${this.state.unit} · ${this.state.lens?.pdp?.ice?.length || 0} native curves · IQR p25–p75`;
  }
  async demoAt(e) {
    const f = this.hold || this.state.feature,
      root = this.$("formula-demo");
    if (!f || !root || this.state.view !== "atlas") return;
    if (!f.terms_file) {
      root.textContent = "terms not exported for this family";
      return;
    }
    const rect = this.$("pair-image").getBoundingClientRect(),
      u = (e.clientX - rect.left) / rect.width,
      v = (e.clientY - rect.top) / rect.height;
    if (u < 0 || v < 0 || u >= 1 || v >= 1) return;
    const key = f.key,
      terms = await this.termPlanes(
        new URL(f.terms_file, this.state.pair._url).href,
      );
    if (
      (this.hold || this.state.feature)?.key !== key ||
      root !== this.$("formula-demo")
    )
      return;
    const x = Math.min(
        terms.width - 1,
        Math.floor((u * terms.native_grid[0]) / terms.step),
      ),
      y = Math.min(
        terms.height - 1,
        Math.floor((v * terms.native_grid[1]) / terms.step),
      ),
      i = y * terms.width + x;
    const result =
      f.id % 13 < 3
        ? "d1"
        : f.id % 13 < 6
          ? "artifact"
          : f.id % 13 < 9
            ? "detail"
            : "mse";
    root.dataset.index = i;
    root.dataset.result = String(terms.planes[result][i]);
    root.dataset.feature = key;
    root.textContent = `μr ${this.fmt(terms.planes.mu_r[i])} · μd ${this.fmt(terms.planes.mu_d[i])} · V ${this.fmt(terms.planes.variance_sum[i])} · Eδ² ${this.fmt(terms.planes.error_moment[i])} → ${result} ${this.fmt(terms.planes[result][i])} at (${x * terms.step * 2 ** f.scale}, ${y * terms.step * 2 ** f.scale}) · s${f.scale}, sampled native window${f.fragility ? " · local detector before global fragility product" : ""}`;
  }
}
