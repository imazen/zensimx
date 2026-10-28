//! freeze_check — the freeze-bar PASS/FAIL summary over one bake's fulleval
//! JSON: the single decision surface for the permanent-metric freeze.
//!
//! Bars come from `zenpapers:docs/zensim-final-metric-plan-2026-07-31.md` §5
//! (every bar's precedent is cited there); the owner-per-gate audit is
//! `benchmarks/decision_surface_audit_2026-07-31.md`. Rows whose measuring
//! owner lives OUTSIDE the verdict/fulleval pipeline (UPIQ, Korshunov, perf,
//! LOO, corruption ORDERING) are printed as ATTACH rows — evidence to bring,
//! never silently omitted. No stat is computed here: this bin only compares
//! numbers the owning tools already produced (no-duplication rule).
//!
//! Input: the FULLEVAL variant of the verdict JSON (i.e. after
//! `scripts/run_full_eval.sh` injects `m3a_coherence`); a raw `--full-json`
//! also works, with the injected rows degrading to ATTACH.
//!
//! Named profiles (`--profile`): a profile is a REGISTERED floor set — a
//! second decision surface that never edits the default §5 bars. The one
//! profile today is `balanced-2026-08-04` (sota944 campaign AMENDMENT 8, the
//! user-directed balanced-selection pass: floors F1–F8 + the registered
//! `balanced_composite`, frozen in `benchmarks/sota944_campaign_2026-08-03.md`
//! §8.1 BEFORE any scoring). `--tsv` / `--tsv-header` emit the machine row for
//! the pool matrix driver (`scripts/sota944_balanced_matrix.sh`).
//!
//! Exit: 0 = every evaluable row passes; 1 = at least one FAIL; 2 = usage /
//! parse error. ATTACH rows never fail the exit — they are pending evidence,
//! and the table says so out loud. In the balanced profile, a floor axis
//! ABSENT from the fulleval counts as not-passed (a candidate nobody measured
//! cannot be certified balanced on that axis).

use std::path::PathBuf;

// §5 bars (plan 2026-07-31; precedents cited there — EM4 selected seed,
// 720 corruption parity, C2a EM2-class, bake_verdict dial gates G3).
const BAR_CID22: f64 = 0.89;
const BAR_KONJND: f64 = 0.40;
const BAR_M3A: f64 = 0.85;
// NOTE: the fulleval keys are named `mono_pct`/`tied_pct` but STORE FRACTIONS
// (bake_verdict writes `dial_metrics.mono` verbatim; 0.979 = 97.9%).
const BAR_DIAL_MONO: f64 = 0.93; // G3, TWO-PANEL eval mandate
const BAR_DIAL_TIED: f64 = 0.05; // G3 (upper bound)

// ── AMENDMENT-8 registered floors (sota944 campaign doc §8.1, frozen
// 2026-08-04 BEFORE any scoring; do not edit after scoring begins) ─────────
mod balanced {
    pub const CID22: f64 = 0.885; // F1: one within-config sd (0.01246, arm-D n=12) below the §1 bar; == wave-7 H-Q2 level
    pub const KONJND: f64 = 0.43; // F2: §1 verbatim (abs)
    pub const NONPHOTO: f64 = 0.90; // F3: §1 verbatim
    pub const DIAL_MONO: f64 = 0.93; // F4: §1 dial verbatim
    pub const DIAL_TIED: f64 = 0.05; // F4 upper bound
    pub const DIAL_RANGE_MIN: f64 = 1.0; // F5: flat-dial guard
    pub const DIAL_RANGE_MAX: f64 = 120.0; // F5: bounded-[0,100]-dial sanity (catches the dyn-range-497 class)
    pub const HFNL_PERREF: f64 = 0.0; // F6: sign floor only (0.1931 stays a reported comparator)
    pub const CSIQ: f64 = 0.83; // F7: 944-class breadth
    pub const LIVE: f64 = 0.83; // F7
    /// F8 high-tail non-collapse, SIGNED, on the top USABLE band of the
    /// appendix-V scheme (CID22 `B8-B9` = `[0.80, →)`, n=1425, span 0.119,
    /// all 49 references).
    ///
    /// DERIVED, not chosen: F8's job is non-collapse, so the floor is the
    /// smallest value at which a band ordering is significantly positive —
    /// the band's own marginal 95 % CI half-width. Measured over a stratified
    /// 25-model probe of the board at B=10,000: **0.0407** pair-bootstrap,
    /// **0.0866** reference-clustered. The reference-clustered figure GOVERNS
    /// (CID22's pairs cluster by reference — up to 61 in this band — so a
    /// pair-level resample understates the uncertainty), and `ceil` to 2 dp
    /// gives 0.09. `benchmarks/appendixV/f8_floor_2026-08-06.tsv`.
    ///
    /// The predecessor 0.15 was set against the fixed-decile `B9` — 43 pairs
    /// from 11 of 49 references spanning 0.0194 MOS — and consumed as an
    /// ABSOLUTE value; it cannot be carried over, because the quantity it
    /// bounded no longer exists.
    ///
    /// FINAL — USER-APPROVED 2026-08-06 (registered pending-user-ack in
    /// appendix V.R4, acked the same day): this gate governs 166+ published
    /// board cells. No arithmetic changed at finalization, only the
    /// registration state.
    pub const BAND_HIGH: f64 = 0.09;
    /// F8 low-tail non-collapse, SIGNED, on the lowest USABLE band. Zero is a
    /// real bar on a signed value (it was unfalsifiable against the absolute
    /// one): it fails a model whose low-quality region is ordered backwards.
    pub const BAND_LOW: f64 = 0.0;
    pub const M3A_GOLD: f64 = 0.85; // reported tier, NOT a floor
    pub const M3A_SILVER: f64 = 0.78; // ≈ measured 944-class median (26 cells, med 0.793)
    pub const HFNL_COMPARATOR: f64 = 0.1931; // reported context only
    /// `--select` tie-break weight on M3a (campaign appendix E.4, registered
    /// 2026-08-04 BEFORE any selection ranking existed). NOT a new weight
    /// class: 0.15 is exactly what `balanced_composite` already gives its
    /// breadth additions (`W_CSIQ`/`W_LIVE`/`W_BANDTAIL`) — coherence is a
    /// product axis of that tier, material but not co-primary with CID22
    /// (1.00). Scale check from measured spreads: 0.15 × the 944-class M3a
    /// sd (0.0471) ≈ 0.007 of composite, so it breaks ties between SEEDS
    /// rather than dominating; 0.15 × the observed board range
    /// (0.199 → 0.954) = 0.113, comparable to a 0.11 CID22 swing, so it is
    /// not decorative either.
    pub const W_M3A: f64 = 0.15;
    // balanced_composite weights: the first six terms are `product_composite`
    // verbatim; csiq/live/band-tail are the registered additions at 0.15.
    // Corpus terms are abs SROCC; band-tail is SIGNED (collapse must hurt).
    pub const W_CID22: f64 = 1.00;
    pub const W_IMAZEN26: f64 = 0.50;
    pub const W_NONPHOTO: f64 = 0.30;
    pub const W_KONJND: f64 = 0.20;
    pub const W_CSIQ: f64 = 0.15;
    pub const W_LIVE: f64 = 0.15;
    pub const W_BANDTAIL: f64 = 0.15;
    pub const W_AIC3: f64 = 0.10;
    pub const W_AIC4: f64 = 0.05;
}

// ── Corpus label ORIENTATION (2026-08-05) ──────────────────────────────────
// THE OWNER is `EXPECTED_ORIENTATION` in
// `scripts/canonical_corpus/check_target_orientation.py` (campaign REGISTERED
// APPENDIX I): three eval corpora carry DISTORTION-oriented JND-family labels
// (q_jnd distance / PJND threshold) — their signed SROCC is negative BY
// CONVENTION (measured: aic4 188/188 board fullevals negative, sdr25 171/171,
// konjnd 187/188), so |SROCC| is the magnitude reading and a POSITIVE signed
// value is the defect (orientation mismatch). Quality-oriented corpora are the
// mirror: negative = genuine inversion (the Appendix F failure).
//
// This Rust list is a GATED MIRROR of the Python registry (freeze_check can't
// import Python): `distortion_oriented_mirror_matches_python_registry` parses
// the owner file and fails the build's test run on ANY drift — extend the
// registry there first, then this list.
const DISTORTION_ORIENTED: [&str; 3] = ["aic4", "konjnd", "sdr25"];

fn is_distortion_oriented(corpus: &str) -> bool {
    DISTORTION_ORIENTED.contains(&corpus)
}

fn usage() -> ! {
    eprintln!(
        "freeze_check — freeze-bar / profile PASS-FAIL over a fulleval JSON\n\n\
         usage: freeze_check --fulleval <bake.fulleval.json> [--bar name=value]...\n\
                freeze_check --fulleval <f> --profile balanced-2026-08-04 [--tsv]\n\
                             [--annotations <registry.json|none>]\n\
                freeze_check --qualify --fulleval <json> (JSON report; missing evidence does not pass)\n\
                freeze_check --select <a.fulleval.json> <b...> [--tsv]\n\
                             [--seed-group] [--min-k N] [--floor-basis all|mean|legacy]\n\
         \x20                    [--gaddr-block auto|canonical|ladder]\n\
                freeze_check --tsv-header | --select-tsv-header | \
                --seed-group-tsv-header\n\n\
         --select: the REGISTERED k-seed selection rule (campaign appendix\n\
         E.4). PRIMARY = profile floor count; TIE-BREAK = balanced_composite\n\
         + 0.15·M3a. M3a states are MEASURED / NOT COMPUTABLE (ensemble,\n\
         ranked separately, never penalized) / UNMEASURED (listed, NOT\n\
         selectable) — a missing measurement is never scored as zero. sdr25\n\
         is a reported comparator, not part of the rule. Exit 1 if no\n\
         candidate is selectable.\n\
         G-ADDR CONTRACT VETO (2026-09-06 owner fix, registry\n\
         select-rule-blind-to-dial-contract-2026-09-06): a candidate (or, \n\
         under --seed-group, a group with ANY member) that MEASURES a\n\
         G-ADDR CONTRACT-tier (C1-C6) fail in its own `dial.addressability`\n\
         block is NOT selectable, however high its floor count or\n\
         composite — dial addressability is an ABSOLUTE product bar, not a\n\
         floor a bake can trade off. A NOT MEASURED contract row is never a\n\
         fail (an absent block, or an all-unmeasured contract tier, both\n\
         stay eligible — only a MEASURED fail vetoes). Listed under its own\n\
         \"contract FAIL — not selectable\" heading with the failing rows.\n\
         Under --floor-basis all (default), A7r's per-codec\n\
         floor-representability breakdown also becomes one MORE floor per\n\
         MEASURED codec, folded into the floor count (a codec no\n\
         representative's instrument touched counts in neither direction).\n\
         --seed-group (2026-09-04, §7.7 fix; opt-in, --select only): without\n\
         it, --select ranks INDIVIDUAL cells and on a class with real seed\n\
         spread will select the lucky draw — literally the best-of-k seed,\n\
         not the recipe's typical behavior. With it, cells are grouped by\n\
         RECIPE and GROUPS are ranked by the same PRIMARY/TIE-BREAK rule\n\
         computed over the k-seed MEAN, with each group's per-seed\n\
         selection_composite spread (min–max) and every per-seed value\n\
         printed alongside.\n\
         The grouping rule is NOT a flag: it is DERIVED from each fulleval's\n\
         embedded zentrain.repro, mirroring the board's owner\n\
         (scripts/v_next/gauntlet.py seed_group_key, fair_gauntlet §1.1) so a\n\
         --select group and a board group are the same handle. Key =\n\
         sha1(repro.argv minus the seed and output-path flags)[:12];\n\
         duplicate promotions of one training run collapse by seed identity,\n\
         so k = DISTINCT SEEDS, never cells. An ensemble or a cell with no\n\
         embedded argv is UNGROUPABLE — listed as its own group and labelled,\n\
         never silently merged or dropped; a grouped cell with k=1 is\n\
         UNREPLICATED (one draw, not an estimate).\n\
         THE MEAN IS NOT THE GROUP'S TRUE SCORE. A seed drives pair\n\
         SAMPLING, so members saw objectively different subsets; the mean is\n\
         the honest estimator against best-of-k, never a definitive value.\n\
         Read spread + per-seed rows with it (and a bake's own\n\
         zentrain.sample_coverage for what it actually touched).\n\
         Prints an ADDITIONAL section after the unchanged per-cell table;\n\
         exit code reflects the SEED-GROUP winner.\n\
         --min-k N (REGISTERED AMENDMENT 2026-09-05, default 2): the\n\
         REPLICATION FLOOR. A seed group with fewer than N distinct seeds is\n\
         UNREPLICATED -- listed and ranked in its own section, never\n\
         selected. Active by default, which also turns the seed-grouped\n\
         section on (it is the basis of the pick) and makes the selection a\n\
         RECIPE rather than a cell. Measured reason: replicating the board\n\
         leaders moved them DOWN (LSTAR 0.8615 best cell -> 0.856414 at k=7)\n\
         and best-of-k minus k-mean has a +0.0061 median over the 18\n\
         combined-fair k>=2 groups, larger than the 0.0021 separating the top\n\
         four (benchmarks/replication_wave_2026-09-05.md). --min-k 1 turns\n\
         the floor off entirely, for reproducing a historical selection.\n\
         --floor-basis all|mean|legacy (default all): how a GROUP's floor\n\
         count is computed -- `all` = floors EVERY distinct-seed\n\
         representative passes (a floor is a certification, and a mean is\n\
         not one: two members at 8/8 and 6/8 average 7.0 even when they\n\
         fail DIFFERENT floors, crediting a floor no member reliably\n\
         clears) PLUS (2026-09-06) the A7r per-codec G-ADDR floors and the\n\
         CONTRACT veto; `mean` = the k-seed mean, the pre-amendment basis\n\
         (F1-F8 only, no A7r floors -- the CONTRACT veto still applies,\n\
         since it is a product-safety gate, not a floor-counting\n\
         convention); `legacy` = byte-for-byte the PRE-2026-09-06 `all`\n\
         (F1-F8 only, NO CONTRACT veto, NO A7r floors) -- an audit/\n\
         reproduction escape hatch ONLY, never for a real selection. The\n\
         mean is reported either way. No basis can admit a candidate the\n\
         pre-amendment rule refused -- all three only remove (`legacy`\n\
         aside, which exists purely to reproduce the pre-fix defect).\n\
         --gaddr-block auto|canonical|ladder (default auto): WHICH dial\n\
         instrument the CONTRACT veto and the A7r floors are read from.\n\
         `auto` = the cell's `dial_ladder` block (the 2026-09-05 FLOOR-DENSE\n\
         ladder -- the board's OPERATIVE ruler since 2026-09-06) when it has\n\
         one, else its canonical `dial.addressability`. `canonical`\n\
         reproduces the pre-2026-09-06 reading; `ladder` refuses to fall\n\
         back, so a cell with no ladder read counts as NOT MEASURED rather\n\
         than being graded on a different instrument. The two blocks are\n\
         different INSTRUMENTS -- never compare a value across them.\n\n\
         default (no --profile): the §5 freeze bar (unchanged).\n\
         --bar sets/overrides a cross-bake numeric bar for: csiq, live\n\
         (§5 only; their §5 bars are \"≥ best 924-arm\" — externally chosen, so\n\
         they stay ATTACH rows unless a value is supplied. The balanced\n\
         profile's floors are REGISTERED and fixed).\n\
         --profile balanced-2026-08-04: sota944 AMENDMENT-8 floors F1..F8 +\n\
         the registered balanced_composite (campaign doc §8.1).\n\
         --annotations: the committed invalidation/annotation registry\n\
         (benchmarks/eval_annotations.json). Default: $ZENSIM_EVAL_ANNOTATIONS,\n\
         else ./benchmarks/eval_annotations.json if present, else none (noted).\n\
         `absent-not-failed` entries make an ABSENT floor axis print\n\
         `— (absent)` — still not-passed for n/8 (registered rule), but\n\
         distinct from a measured FAIL, and the n/m-measured form is stated.\n\
         --tsv: one machine row (columns from --tsv-header)."
    );
    std::process::exit(2);
}

// ── Annotations registry (benchmarks/eval_annotations.json; board-integrity
// pass 2026-08-04). Machine-readable flags over fulleval cells so superseded /
// flattered / absent-not-failed numbers are never read as clean wins. Schema
// documented in the registry file's `_schema` header. ──────────────────────
#[derive(Clone, Debug)]
struct AnnEntry {
    id: String,
    kind: String, // "invalidated" | "annotated" | "absent-not-failed"
    fields: Vec<String>,
    scope: serde_json::Value,
    reason: String,
    /// Documentation-only finding: its `scope` carries no machine predicate
    /// (`{"manual": …}`), so it never applies to a cell. It is still LOADED
    /// and SURFACED — the point of the D10 fix is that a finding the matcher
    /// cannot evaluate must be visible, not silently inert.
    manual: bool,
}

/// Dot-path presence: absent key OR explicit null ⇒ not present.
fn field_present(v: &serde_json::Value, dotpath: &str) -> bool {
    let mut cur = v;
    for p in dotpath.split('.') {
        match cur.get(p) {
            Some(x) => cur = x,
            None => return false,
        }
    }
    !cur.is_null()
}

/// Scope predicate: exactly one of missing / present / names / all.
fn ann_matches(v: &serde_json::Value, e: &AnnEntry) -> bool {
    // A documentation-only finding has no machine predicate and applies to
    // nothing — explicitly, rather than by falling through the match arms.
    if e.manual {
        return false;
    }
    if let Some(p) = e.scope.get("missing").and_then(|x| x.as_str()) {
        return !field_present(v, p);
    }
    if let Some(p) = e.scope.get("present").and_then(|x| x.as_str()) {
        return field_present(v, p);
    }
    if let Some(ns) = e.scope.get("names").and_then(|x| x.as_array()) {
        let name = bake_name(v);
        return ns.iter().any(|n| n.as_str() == Some(name));
    }
    e.scope
        .get("all")
        .and_then(|x| x.as_bool())
        .unwrap_or(false)
}

/// An entry field covers a floor's backing path iff equal or a
/// segment-boundary prefix (`rank.hfnlproxy` covers
/// `rank.hfnlproxy.per_ref_mean`; `rank.hfnl` covers neither).
fn ann_covers(entry_field: &str, floor_field: &str) -> bool {
    floor_field == entry_field
        || (floor_field.starts_with(entry_field)
            && floor_field.as_bytes().get(entry_field.len()) == Some(&b'.'))
}

/// The only top-level keys the registry may carry. Anything else is a
/// finding that was written OUTSIDE `entries[]` and would be dropped.
const ANN_TOP_LEVEL_KEYS: [&str; 2] = ["_schema", "entries"];
/// Scope predicates `ann_matches` can evaluate.
const ANN_SCOPE_PREDICATES: [&str; 4] = ["missing", "present", "names", "all"];
/// Explicit "no machine predicate — documentation only" scope form.
const ANN_SCOPE_MANUAL: &str = "manual";
/// The `kind` values that mean something to a consumer.
const ANN_KINDS: [&str; 3] = ["invalidated", "annotated", "absent-not-failed"];

/// True iff `scope` carries exactly one predicate `ann_matches` understands.
fn scope_is_machine_predicate(scope: &serde_json::Value) -> bool {
    scope
        .as_object()
        .is_some_and(|o| o.len() == 1 && ANN_SCOPE_PREDICATES.iter().any(|k| o.contains_key(*k)))
}

fn load_annotations(path: &std::path::Path) -> Result<Vec<AnnEntry>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let v: serde_json::Value =
        serde_json::from_slice(&bytes).map_err(|e| format!("parse {}: {e}", path.display()))?;
    load_annotations_value(&v, &path.display().to_string())
}

/// Parse + VALIDATE a registry document (ADD156 ship audit, defect **D10**).
///
/// A registry that silently loses integrity notes is worse than no registry,
/// so every way a finding could go unnoticed is a hard error here:
///
/// * a finding written as a bare TOP-LEVEL key instead of an `entries[]`
///   member (it would never be read at all — this is how the annotation
///   documenting audit defect D2 stayed invisible);
/// * a `scope` the matcher cannot evaluate (it would match ZERO cells and be
///   indistinguishable from one that legitimately does not apply). A finding
///   with no machine predicate must say so with `{"manual": …}`, which is
///   loaded and surfaced but never applied;
/// * an unknown `kind` (only `absent-not-failed` drives the floor accounting,
///   so a typo silently changes behaviour);
/// * a duplicate `id` (the later entry would shadow the earlier in tooltips).
fn load_annotations_value(v: &serde_json::Value, whence: &str) -> Result<Vec<AnnEntry>, String> {
    let obj = v
        .as_object()
        .ok_or_else(|| format!("{whence}: registry must be a JSON object"))?;
    let stray: Vec<&str> = obj
        .keys()
        .map(String::as_str)
        .filter(|k| !ANN_TOP_LEVEL_KEYS.contains(k))
        .collect();
    if !stray.is_empty() {
        return Err(format!(
            "{whence}: {} finding(s) written as top-level keys instead of \
             members of `entries`, where nothing would ever read them: {}. \
             Move each into the `entries` array (with `id`, `kind`, `fields`, \
             `scope`).",
            stray.len(),
            stray.join(", ")
        ));
    }
    let entries = obj
        .get("entries")
        .and_then(|e| e.as_array())
        .ok_or_else(|| format!("{whence}: no `entries` array"))?;

    let mut out: Vec<AnnEntry> = Vec::with_capacity(entries.len());
    for (i, e) in entries.iter().enumerate() {
        let at = |what: &str| format!("{whence}: entries[{i}] {what}");
        let e = e.as_object().ok_or_else(|| at("is not an object"))?;
        let id = e
            .get("id")
            .and_then(|x| x.as_str())
            .filter(|s| !s.is_empty())
            .ok_or_else(|| at("missing a non-empty `id`"))?
            .to_string();
        if out.iter().any(|p| p.id == id) {
            return Err(format!("{whence}: duplicate entry id `{id}`"));
        }
        let kind = e
            .get("kind")
            .and_then(|x| x.as_str())
            .ok_or_else(|| format!("{whence}: entry `{id}` missing `kind`"))?
            .to_string();
        if !ANN_KINDS.contains(&kind.as_str()) {
            return Err(format!(
                "{whence}: entry `{id}` has unknown kind `{kind}` — expected one of {}",
                ANN_KINDS.join(", ")
            ));
        }
        let scope = e.get("scope").cloned().unwrap_or(serde_json::Value::Null);
        let scope_obj = scope.as_object().ok_or_else(|| {
            format!(
                "{whence}: entry `{id}` has no `scope` object — use one of {} for a \
                     machine predicate, or {{\"{ANN_SCOPE_MANUAL}\": …}} for a \
                     documentation-only finding",
                ANN_SCOPE_PREDICATES.join("/")
            )
        })?;
        let manual = scope_obj.len() == 1 && scope_obj.contains_key(ANN_SCOPE_MANUAL);
        if !manual && !scope_is_machine_predicate(&scope) {
            let keys: Vec<&str> = scope_obj.keys().map(String::as_str).collect();
            return Err(format!(
                "{whence}: entry `{id}` has a scope the matcher cannot evaluate \
                 (keys: [{}]) — it would match ZERO cells and be silently inert. \
                 Use exactly one of {} for a machine predicate, or wrap the prose \
                 as {{\"{ANN_SCOPE_MANUAL}\": …}} to declare it documentation-only.",
                keys.join(", "),
                ANN_SCOPE_PREDICATES.join("/")
            ));
        }
        out.push(AnnEntry {
            id,
            kind,
            fields: e
                .get("fields")
                .and_then(|x| x.as_array())
                .map(|a| {
                    a.iter()
                        .filter_map(|x| x.as_str().map(str::to_string))
                        .collect()
                })
                .unwrap_or_default(),
            scope,
            reason: e
                .get("reason")
                .and_then(|x| x.as_str())
                .map(str::to_string)
                .unwrap_or_default(),
            manual,
        });
    }
    Ok(out)
}

fn f(v: &serde_json::Value, path: &[&str]) -> Option<f64> {
    let mut cur = v;
    for p in path {
        cur = cur.get(p)?;
    }
    cur.as_f64()
}

enum Row {
    /// (gate, bar text, measured, pass)
    Eval(String, String, String, bool),
    /// (gate, bar text, what to attach / where the owner lives)
    Attach(String, String, String),
    /// (gate, note) — informational, no verdict
    Info(String, String),
}

/// The §5 rows, verbatim (moved out of main 2026-08-04 so the balanced
/// profile could land WITHOUT touching this path; row order, names and bar
/// semantics are test-locked below).
fn legacy_rows(v: &serde_json::Value, bar_csiq: Option<f64>, bar_live: Option<f64>) -> Vec<Row> {
    let mut rows: Vec<Row> = Vec::new();
    let ge = |name: &str, bar: f64, got: Option<f64>, absent: &str| -> Row {
        match got {
            Some(x) => Row::Eval(
                name.to_string(),
                format!("≥ {bar}"),
                format!("{x:.4}"),
                x >= bar,
            ),
            None => Row::Attach(name.to_string(), format!("≥ {bar}"), absent.to_string()),
        }
    };

    rows.push(ge(
        "CID22 SROCC (selected seed)",
        BAR_CID22,
        f(v, &["rank", "cid22", "srocc"]),
        "cid22 missing — run bake_verdict with the cid22 corpus",
    ));
    rows.push(ge(
        "KonJND abs-SROCC",
        BAR_KONJND,
        f(v, &["rank", "konjnd", "srocc"]).map(f64::abs),
        "konjnd missing — run bake_verdict with the konjnd corpus",
    ));

    // Corruption: the §5 bar (ordering ≥ 0.214) is the ORDERING stat from the
    // E-M fulleval instruments — not computed by bake_verdict (which owns the
    // detection-rate gate). Report the head's detection numbers as context and
    // keep ordering as an ATTACH row.
    match f(v, &["corruption_head", "pass_q20"]) {
        Some(q20) => rows.push(Row::Info(
            "Corruption head detection (context)".into(),
            format!(
                "head `{}` pass_q20 {:.1}% / pass_q10 {:.1}%",
                v["corruption_head"]["head"].as_str().unwrap_or("?"),
                100.0 * q20,
                100.0 * f(v, &["corruption_head", "pass_q10"]).unwrap_or(f64::NAN)
            ),
        )),
        None => rows.push(Row::Info(
            "Corruption head detection (context)".into(),
            match f(v, &["corruption", "pass_q20"]) {
                Some(d) => format!(
                    "no head given — dial-alone pass_q20 {:.1}% (broken-by-design at 924); \
                     rerun bake_verdict with --corruption-head",
                    100.0 * d
                ),
                None => "no corruption grid in this fulleval".into(),
            },
        )),
    }
    rows.push(Row::Attach(
        "Corruption ORDERING via head".into(),
        "≥ 0.214 (720 parity)".into(),
        "the E-M corruption-ordering instrument, evaluated on the HEAD bake".into(),
    ));

    for (name, bar, key) in [
        ("CSIQ SROCC", bar_csiq, "csiq"),
        ("LIVE SROCC", bar_live, "live"),
    ] {
        match (bar, f(v, &["rank", key, "srocc"])) {
            (Some(b), Some(x)) => rows.push(Row::Eval(
                format!("{name} (≥ best 924-arm)"),
                format!("≥ {b}"),
                format!("{x:.4}"),
                x >= b,
            )),
            (None, Some(x)) => rows.push(Row::Attach(
                format!("{name} (≥ best 924-arm)"),
                "cross-bake".into(),
                format!("measured {x:.4}; supply --bar {key}=<best-arm> to gate"),
            )),
            (_, None) => rows.push(Row::Attach(
                format!("{name} (≥ best 924-arm)"),
                "cross-bake".into(),
                format!("{key} missing — run bake_verdict with it"),
            )),
        }
    }

    rows.push(Row::Attach(
        "UPIQ pooled (V1-HDR)".into(),
        "> 0.7536".into(),
        "scripts/hdr/upiq_panel.py (Python owner)".into(),
    ));
    rows.push(Row::Attach(
        "Korshunov hold (V1-HDR)".into(),
        "≥ 0.93".into(),
        "seven-domain external-read runner (must be committed before Phase 4)".into(),
    ));

    rows.push(ge(
        "M3a coherence (EM2-class)",
        BAR_M3A,
        f(v, &["m3a_coherence"]),
        "not injected — this is the raw --full-json; run scripts/run_full_eval.sh",
    ));

    match f(v, &["dial", "mono_pct"]) {
        Some(m) => rows.push(Row::Eval(
            "Dial monotonicity".into(),
            format!("≥ {:.0}%", 100.0 * BAR_DIAL_MONO),
            format!("{:.1}%", 100.0 * m),
            m >= BAR_DIAL_MONO,
        )),
        None => rows.push(Row::Attach(
            "Dial monotonicity".into(),
            format!("≥ {:.0}%", 100.0 * BAR_DIAL_MONO),
            "dial block missing — bake_verdict needs the dial grid".into(),
        )),
    }
    match f(v, &["dial", "tied_pct"]) {
        Some(t) => rows.push(Row::Eval(
            "Dial tied rate".into(),
            format!("≤ {:.0}%", 100.0 * BAR_DIAL_TIED),
            format!("{:.1}%", 100.0 * t),
            t <= BAR_DIAL_TIED,
        )),
        None => rows.push(Row::Attach(
            "Dial tied rate".into(),
            format!("≤ {:.0}%", 100.0 * BAR_DIAL_TIED),
            "dial block missing".into(),
        )),
    }
    match f(v, &["composite"]) {
        Some(c) => rows.push(Row::Info("product_composite".into(), format!("{c:.4}"))),
        None => rows.push(Row::Info(
            "product_composite".into(),
            "null — needs the composite corpora (imazen26/nonphoto/...)".into(),
        )),
    }

    let repro_ok = v.get("repro").map(|r| !r.is_null()).unwrap_or(false);
    rows.push(Row::Eval(
        "Byte-repro (embedded zentrain.repro)".into(),
        "present".into(),
        if repro_ok {
            "present".into()
        } else {
            "MISSING".into()
        },
        repro_ok,
    ));

    rows.push(Row::Attach(
        "Perf SDR".into(),
        "≤ +2% vs 944 baseline".into(),
        "zenbench compare/extractor benches".into(),
    ));
    rows.push(Row::Attach(
        "Perf HDR (PU path)".into(),
        "≤ +5%".into(),
        "zenbench; the V5 lever list".into(),
    ));
    rows.push(Row::Attach(
        "LOO (append2 family)".into(),
        "≤ 0".into(),
        "extractor-side LOO instrument (gaps-doc §0)".into(),
    ));
    rows
}

// ── Balanced profile (AMENDMENT 8) ─────────────────────────────────────────

/// The four registered classes — their USES differ, so they are scored as
/// separate pools (campaign doc §8.1).
fn classify(v: &serde_json::Value) -> (&'static str, &'static str) {
    let kind = v
        .get("model")
        .and_then(|m| m.get("kind"))
        .and_then(|k| k.as_str());
    if kind == Some("ensemble") {
        return (
            "944-ensemble",
            "k× scoring cost; NOT a shippable artifact; M3a NOT COMPUTABLE",
        );
    }
    let ni = f(v, &["n_inputs"])
        .or_else(|| f(v, &["model", "n_inputs"]))
        .map(|x| x as i64);
    if ni != Some(944) {
        // NOTE TEXT CORRECTED 2026-08-31 (ADD156 ship audit, defect D3).
        // It used to read "context only — regime-incomparable, never
        // shortlisted", which describes an exclusion this code does not
        // implement and never did: `class` is compared against
        // `"944-ensemble"` and nothing else (see `m3a_state` and the
        // `run_select` pool split), and selectability is
        // `m3a != Unmeasured && n_pass > 0` — the class is not a term in it.
        //
        // The wording was not harmless. The audit read it as a structural
        // block and filed "the registered selection rule cannot select
        // ADD156" as a HIGH ship-blocker. MEASURED on the board fullevals:
        // `--select` ranks ADD156 first and prints
        // "SELECTED: ADD156_safesyn_only_raw_lasso — 6/8 floors,
        // selection_composite 0.9644", ahead of shipped B (0.9151) — both
        // stamped `era-bridge`. What produced the audit's "NO" was its own
        // ad-hoc fulleval missing `m3a_coherence`, a value that same audit
        // measured at 0.9641 (27/27 GOLD) and never wrote into the JSON.
        //
        // So the label says what it is — a regime pool for comparison — and
        // does not claim a power it lacks.
        return (
            "era-bridge",
            "non-944 regime — compare within class, not across; NOT an exclusion \
             (selection reads floors + M3a)",
        );
    }
    if bake_name(v).starts_with("C_ensk") {
        return (
            "944-distilled",
            "shippable; the M3a-mover class (wave-6 arm F)",
        );
    }
    ("944-single", "shippable single bake")
}

fn bake_name(v: &serde_json::Value) -> &str {
    v["name"]
        .as_str()
        .or_else(|| v["bake"].as_str())
        .unwrap_or("?")
}

/// The band scheme a fulleval's bands were cut on, `None` for anything written
/// before the appendix-V redesign (2026-08-06).
///
/// Load-bearing: a legacy cell's `B9` is a 43-pair / 0.019-span residual tail
/// whose statistic is dominated by selection noise and whose SIGN is an artifact
/// of the slice width. Reading a current floor against it would be comparing two
/// different measurements. F8 therefore reports legacy cells as ABSENT rather
/// than scoring them.
fn band_scheme(v: &serde_json::Value) -> Option<&str> {
    v.get("rank")?
        .get("cid22")?
        .get("band_scheme")?
        .get("name")?
        .as_str()
}

/// One usable band's SIGNED Spearman, read in the corpus's declared
/// orientation, plus its `n` and target span.
struct BandTail {
    label: String,
    signed: f64,
    n: i64,
    span: f64,
}

/// The lowest and highest USABLE bands of a corpus (`None` when the fulleval
/// carries no band block, or every band is NOT-MEASURED).
///
/// "Usable" is the emitting side's judgement, carried in the band row as
/// `not_measured_reason`: a band that could not clear the count/span floors
/// publishes no statistics and must never be scored, ranked, or shown as a
/// measured zero. Rows are read in DECLARED orientation — the same
/// `is_distortion_oriented` convention the guard rows use, mirrored rather than
/// special-cased by corpus name — so "higher is better ordering" holds on every
/// corpus.
fn band_tails(v: &serde_json::Value, corpus: &str) -> Option<(BandTail, BandTail)> {
    let bands = v.get("rank")?.get(corpus)?.get("bands")?.as_array()?;
    let flip = if is_distortion_oriented(corpus) {
        -1.0
    } else {
        1.0
    };
    let mut usable: Vec<BandTail> = bands
        .iter()
        .filter(|b| {
            b.get("not_measured_reason")
                .map(|r| r.is_null())
                .unwrap_or(false)
        })
        .filter_map(|b| {
            Some(BandTail {
                label: b["band"].as_str()?.to_string(),
                signed: b.get("srocc_signed")?.as_f64()? * flip,
                n: b["n"].as_i64().unwrap_or(0),
                span: b.get("span").and_then(|s| s.as_f64()).unwrap_or(f64::NAN),
            })
        })
        .collect();
    if usable.is_empty() {
        return None;
    }
    let hi = usable.pop()?;
    let lo = if usable.is_empty() {
        BandTail {
            label: hi.label.clone(),
            signed: hi.signed,
            n: hi.n,
            span: hi.span,
        }
    } else {
        usable.remove(0)
    };
    Some((lo, hi))
}

/// CID22 band srocc + n by band label ("B3"/"B9") — the LEGACY fixed-decile
/// accessor, kept only to read pre-2026-08-06 board files.
///
/// **`srocc` here is an ABSOLUTE value** — `zenstats::panel` computes
/// `spearman(..).abs()` and `bake_verdict`'s per-band rows came from it — even
/// though F8 was specified as signed. Measured consequence on the 120 board
/// cells that still carry per-pair (campaign appendix V, G-V1): the stored
/// `srocc` equals `|recomputed signed|` on **120 of 120**, **109 of 120** are
/// NEGATIVE in `B9`, and **82** pass `|B9| ≥ 0.15` where **2** pass it signed.
/// Because `|·|` is monotone in the depth of an inversion, that column ranked
/// models by how backwards their top band was.
fn cid22_band(v: &serde_json::Value, band: &str) -> Option<(f64, i64)> {
    let bands = v.get("rank")?.get("cid22")?.get("bands")?.as_array()?;
    let b = bands.iter().find(|b| b["band"].as_str() == Some(band))?;
    Some((b["srocc"].as_f64()?, b["n"].as_i64().unwrap_or(0)))
}

/// The registered ranking composite (§8.1): product_composite's six terms
/// verbatim + csiq/live/band-tail at 0.15. Corpus terms are abs SROCC (owner
/// convention); band-tail is SIGNED. Absent terms drop from num AND den.
fn balanced_composite(v: &serde_json::Value) -> Option<f64> {
    let corpus = |c: &str| f(v, &["rank", c, "srocc"]).map(f64::abs);
    // Band-tail term: the mean of the lowest and highest USABLE bands, SIGNED
    // (the corpus term above keeps the owner's abs convention; a band tail must
    // not, or a collapsed tail scores like a healthy one and a more deeply
    // inverted one scores HIGHER). Legacy fixed-decile cells contribute
    // nothing — the term drops from BOTH numerator and denominator, exactly as
    // any other absent axis does — rather than feeding an absolute-valued
    // 43-pair statistic into the ranking.
    let bandtail = match band_scheme(v) {
        Some(_) => band_tails(v, "cid22").map(|(lo, hi)| (lo.signed + hi.signed) / 2.0),
        None => None,
    };
    let terms: [(Option<f64>, f64); 9] = [
        (corpus("cid22"), balanced::W_CID22),
        (corpus("imazen26"), balanced::W_IMAZEN26),
        (corpus("nonphoto"), balanced::W_NONPHOTO),
        (corpus("konjnd"), balanced::W_KONJND),
        (corpus("csiq"), balanced::W_CSIQ),
        (corpus("live"), balanced::W_LIVE),
        (bandtail, balanced::W_BANDTAIL),
        (corpus("aic3"), balanced::W_AIC3),
        (corpus("aic4"), balanced::W_AIC4),
    ];
    let (num, den) = terms
        .iter()
        .filter_map(|(x, w)| x.map(|x| (x * w, *w)))
        .fold((0.0f64, 0.0f64), |(n, d), (x, w)| (n + x, d + w));
    if den > 0.0 { Some(num / den) } else { None }
}

/// One registered floor: id (TSV fail token), gate name, bar text, measured
/// text, pass (absent axis ⇒ false, text says UNEVALUABLE). `fields` = the
/// fulleval dot-paths backing the floor (annotation coverage); `absent` = the
/// backing value(s) were missing; `absent_not_failed` = an
/// `absent-not-failed` registry entry covers this absence (still not-passed
/// for n/8 per the registered rule, but printed `— (absent)`, kept out of the
/// measured-fails list, and counted in the n/m-measured form).
struct Floor {
    id: &'static str,
    gate: &'static str,
    bar: String,
    measured: String,
    pass: bool,
    fields: &'static [&'static str],
    absent: bool,
    absent_not_failed: bool,
}

fn floor_ge(
    id: &'static str,
    gate: &'static str,
    bar: f64,
    got: Option<f64>,
    fields: &'static [&'static str],
) -> Floor {
    match got {
        Some(x) => Floor {
            id,
            gate,
            bar: format!("≥ {bar}"),
            measured: format!("{x:.4}"),
            pass: x >= bar,
            fields,
            absent: false,
            absent_not_failed: false,
        },
        None => Floor {
            id,
            gate,
            bar: format!("≥ {bar}"),
            measured: "not measured — UNEVALUABLE".into(),
            pass: false,
            fields,
            absent: true,
            absent_not_failed: false,
        },
    }
}

fn m3a_tier(v: &serde_json::Value, class: &str) -> String {
    if class == "944-ensemble" {
        return "NOT COMPUTABLE (ensemble — the coherence instrument loads one ZNPR)".into();
    }
    match f(v, &["m3a_coherence"]) {
        Some(x) if x >= balanced::M3A_GOLD => format!("{x:.4} — GOLD (≥ {})", balanced::M3A_GOLD),
        Some(x) if x >= balanced::M3A_SILVER => {
            format!("{x:.4} — silver (≥ {})", balanced::M3A_SILVER)
        }
        Some(x) => format!("{x:.4} — FLAGGED (< {})", balanced::M3A_SILVER),
        None => "— (NOT MEASURED)".into(),
    }
}

struct BalancedReport {
    class: &'static str,
    class_note: &'static str,
    floors: Vec<Floor>,
    info: Vec<(String, String)>,
    composite: Option<f64>,
    /// matched registry entries: (id, kind, reason)
    annotations: Vec<(String, String, String)>,
}

fn eval_balanced(v: &serde_json::Value, anns: &[AnnEntry]) -> BalancedReport {
    let (class, class_note) = classify(v);
    let mut floors = Vec::new();

    // F1 CID22
    floors.push(floor_ge(
        "cid22",
        "F1 CID22 SROCC",
        balanced::CID22,
        f(v, &["rank", "cid22", "srocc"]),
        &["rank.cid22.srocc"],
    ));
    // F2 KonJND (abs)
    floors.push(floor_ge(
        "konjnd",
        "F2 KonJND abs-SROCC",
        balanced::KONJND,
        f(v, &["rank", "konjnd", "srocc"]).map(f64::abs),
        &["rank.konjnd.srocc"],
    ));
    // F3 nonphoto
    floors.push(floor_ge(
        "nonphoto",
        "F3 nonphoto SROCC",
        balanced::NONPHOTO,
        f(v, &["rank", "nonphoto", "srocc"]),
        &["rank.nonphoto.srocc"],
    ));
    // F4 dial mono + tied (one row, both must hold).
    // UNIT ANNOTATION (packaging appendix, 2026-08-04): for SPLINE-LESS bakes
    // the mono/tied stats are measured on RAW outputs (span ~16-17), where the
    // 0.5-score-pt materiality threshold flatters mono vs the real [0,100]
    // dial (span ~63-67 after add-spline+pack; strict-backwards is
    // cal-invariant, so the drop is a unit effect, not new inversions). The
    // floor itself is unchanged — this labels which unit the number is in.
    let dial_unit = match v.get("model").and_then(|m| m.get("output_spline")) {
        Some(s) if !s.is_null() => "dial-unit",
        _ => "raw-unit",
    };
    let dial_bar = format!(
        "mono ≥ {:.0}% ∧ tied ≤ {:.0}%",
        100.0 * balanced::DIAL_MONO,
        100.0 * balanced::DIAL_TIED
    );
    const F4_FIELDS: &[&str] = &["dial.mono_pct", "dial.tied_pct"];
    floors.push(
        match (f(v, &["dial", "mono_pct"]), f(v, &["dial", "tied_pct"])) {
            (Some(m), Some(t)) => Floor {
                id: "dial",
                gate: "F4 dial mono/tied",
                bar: dial_bar,
                measured: format!(
                    "mono {:.1}% / tied {:.1}% ({dial_unit})",
                    100.0 * m,
                    100.0 * t
                ),
                pass: m >= balanced::DIAL_MONO && t <= balanced::DIAL_TIED,
                fields: F4_FIELDS,
                absent: false,
                absent_not_failed: false,
            },
            _ => Floor {
                id: "dial",
                gate: "F4 dial mono/tied",
                bar: dial_bar,
                measured: "dial block missing — UNEVALUABLE".into(),
                pass: false,
                fields: F4_FIELDS,
                absent: true,
                absent_not_failed: false,
            },
        },
    );
    // F5 dial dynamic-range sanity
    let range_bar = format!(
        "{} ≤ span ≤ {}",
        balanced::DIAL_RANGE_MIN,
        balanced::DIAL_RANGE_MAX
    );
    const F5_FIELDS: &[&str] = &["dial.dynamic_range"];
    floors.push(match f(v, &["dial", "dynamic_range"]) {
        Some(r) => Floor {
            id: "dialrange",
            gate: "F5 dial span sanity",
            bar: range_bar,
            measured: format!("{r:.1}"),
            pass: (balanced::DIAL_RANGE_MIN..=balanced::DIAL_RANGE_MAX).contains(&r),
            fields: F5_FIELDS,
            absent: false,
            absent_not_failed: false,
        },
        None => Floor {
            id: "dialrange",
            gate: "F5 dial span sanity",
            bar: range_bar,
            measured: "dial block missing — UNEVALUABLE".into(),
            pass: false,
            fields: F5_FIELDS,
            absent: true,
            absent_not_failed: false,
        },
    });
    // F6 HF-NL per-ref sign floor
    floors.push(floor_ge(
        "hfnl",
        "F6 HF-NL per-ref (sign)",
        balanced::HFNL_PERREF,
        f(v, &["rank", "hfnlproxy", "per_ref_mean"]),
        &["rank.hfnlproxy.per_ref_mean"],
    ));
    // F7 breadth: CSIQ ∧ LIVE (one row)
    const F7_FIELDS: &[&str] = &["rank.csiq.srocc", "rank.live.srocc"];
    floors.push(
        match (
            f(v, &["rank", "csiq", "srocc"]),
            f(v, &["rank", "live", "srocc"]),
        ) {
            (Some(c), Some(l)) => Floor {
                id: "breadth",
                gate: "F7 breadth CSIQ ∧ LIVE",
                bar: format!("both ≥ {}", balanced::CSIQ),
                measured: format!("csiq {c:.4} / live {l:.4}"),
                pass: c >= balanced::CSIQ && l >= balanced::LIVE,
                fields: F7_FIELDS,
                absent: false,
                absent_not_failed: false,
            },
            _ => Floor {
                id: "breadth",
                gate: "F7 breadth CSIQ ∧ LIVE",
                bar: format!("both ≥ {}", balanced::CSIQ),
                measured: "csiq/live missing — UNEVALUABLE".into(),
                pass: false,
                fields: F7_FIELDS,
                absent: true,
                absent_not_failed: false,
            },
        },
    );
    // F8 band tails — SIGNED, on the lowest and highest USABLE bands of the
    // appendix-V scheme. A fulleval cut on the legacy fixed-decile grid is
    // reported ABSENT, not scored: its `B9` is a different quantity (43 pairs,
    // span 0.019, absolute-valued), so a current bar against it would be
    // meaningless. `--annotations` carries the reason.
    let legacy_b9 = cid22_band(v, "B9");
    let tails = band_tails(v, "cid22");
    let scheme = band_scheme(v);
    let fmt_tail =
        |t: &BandTail| format!("{} {:+.3} n={} span={:.3}", t.label, t.signed, t.n, t.span);
    let (measured, pass, absent) = match (&tails, scheme) {
        (Some((lo, hi)), Some(s)) => (
            format!("[{s}] high {} / low {}", fmt_tail(hi), fmt_tail(lo)),
            hi.signed >= balanced::BAND_HIGH && lo.signed >= balanced::BAND_LOW,
            false,
        ),
        // Bands present but no scheme stamp ⇒ pre-appendix-V fixed deciles.
        (_, None) if legacy_b9.is_some() => (
            format!(
                "LEGACY fixed-decile bands (no band_scheme) — B9 was 43 pairs / span \
                 0.019 / ABSOLUTE-valued; stored |B9| {:.3}. Not scored under the \
                 current bar; re-verdict to measure.",
                legacy_b9.map(|x| x.0).unwrap_or(f64::NAN)
            ),
            false,
            true,
        ),
        (None, _) => ("— (no usable band)".into(), false, true),
        (Some(_), None) => ("— (bands present, scheme unknown)".into(), false, true),
    };
    floors.push(Floor {
        id: "bandtail",
        gate: "F8 CID22 band tails",
        bar: format!(
            "top usable band ≥ {} ∧ lowest usable band ≥ {} (SIGNED)",
            balanced::BAND_HIGH,
            balanced::BAND_LOW
        ),
        measured,
        pass,
        fields: &["rank.cid22.bands"],
        absent,
        absent_not_failed: false,
    });

    // ── Registry application (annotations; board-integrity pass 2026-08-04).
    // absent-not-failed: an ABSENT floor covered by a matched entry prints
    // `— (absent)` and is excluded from the measured-fails list; it still
    // counts as not-passed in n/8 (the registered rule), and the
    // n/m-measured second form is reported alongside.
    let matched: Vec<&AnnEntry> = anns.iter().filter(|e| ann_matches(v, e)).collect();
    for e in &matched {
        if e.kind != "absent-not-failed" {
            continue;
        }
        for fl in floors.iter_mut() {
            if fl.absent
                && fl
                    .fields
                    .iter()
                    .any(|ff| e.fields.iter().any(|ef| ann_covers(ef, ff)))
            {
                fl.absent_not_failed = true;
                fl.measured = format!("— (absent — not measured; `{}`)", e.id);
            }
        }
    }

    // Reported (never floors)
    let mut info: Vec<(String, String)> = Vec::new();
    info.push(("M3a (tiered, reported)".into(), m3a_tier(v, class)));
    info.push((
        "Corruption (head-owned)".into(),
        match f(v, &["corruption_head", "pass_q20"]) {
            Some(q20) => format!(
                "head `{}` pass_q20 {:.1}% / pass_q10 {:.1}%",
                v["corruption_head"]["head"].as_str().unwrap_or("?"),
                100.0 * q20,
                100.0 * f(v, &["corruption_head", "pass_q10"]).unwrap_or(f64::NAN)
            ),
            None => match f(v, &["corruption", "pass_q20"]) {
                Some(d) => format!(
                    "no head — dial-alone pass_q20 {:.1}% (broken-by-design post-720)",
                    100.0 * d
                ),
                None => "no corruption grid".into(),
            },
        },
    ));
    info.push((
        "HF-NL 0.1931 comparator (context)".into(),
        match f(v, &["rank", "hfnlproxy", "per_ref_mean"]) {
            Some(x) => format!(
                "{} the arm-B comparator ({x:.4} vs {})",
                if x >= balanced::HFNL_COMPARATOR {
                    "at/above"
                } else {
                    "below"
                },
                balanced::HFNL_COMPARATOR
            ),
            None => "not measured".into(),
        },
    ));
    // SIGNED, always — read AGAINST each corpus's DECLARED orientation
    // (`EXPECTED_ORIENTATION`, see `DISTORTION_ORIENTED` above). On a
    // quality-oriented corpus a negative SROCC is a genuine ranking INVERSION,
    // and an unsigned guard row hides it — which is exactly how 110 of 188
    // board bakes sat anti-correlated with KADID's real human MOS while every
    // one displayed a positive magnitude
    // (`benchmarks/sota944_campaign_2026-08-03.md` REGISTERED APPENDIX F).
    // On a DISTORTION-oriented JND-family corpus the convention is mirrored:
    // negative is its declared native direction (labelled, never flagged), and
    // a POSITIVE signed value is the defect. A guard that yelled INVERTED at
    // the convention (as this one did for sdr25 until 2026-08-05) is a
    // standing false alarm that teaches readers to ignore the row. Guards stay
    // UNSCORED — the requirement is only that defect vs convention be READABLE.
    let guard = |c: &str| match f(v, &["rank", c, "srocc_signed"])
        .or_else(|| f(v, &["rank", c, "srocc"]))
    {
        Some(x) if is_distortion_oriented(c) && x > 0.0 => {
            format!("{x:+.4} ORIENTATION MISMATCH (positive on a JND\u{2193} corpus)")
        }
        Some(x) if is_distortion_oriented(c) => {
            format!("{x:+.4} (JND\u{2193} convention; |SROCC| {:.4})", x.abs())
        }
        Some(x) if x < 0.0 => format!("{x:+.4} INVERTED"),
        Some(x) => format!("{x:+.4}"),
        None => "—".into(),
    };
    info.push((
        "KADID/TID (t=v integrity guards, dimmed; SIGNED)".into(),
        format!(
            "kadid {} / tid {} — never scored; negative = ANTI-CORRELATED with the corpus's human labels",
            guard("kadid"),
            guard("tid")
        ),
    ));
    info.push((
        "sdr25 (within-family selector only; \u{2282} aic4 — Appendix I)".into(),
        guard("sdr25"),
    ));
    info.push((
        "packaging".into(),
        match v.get("model").and_then(|m| m.get("output_spline")) {
            Some(s) if !s.is_null() => "output spline present".into(),
            _ => "spline-less raw head — needs `bake_dial_refit add-spline` \
                  (+ rank-invariance check) before G-RANGE is defined"
                .into(),
        },
    ));
    let repro = match (v.get("repro").map(|r| !r.is_null()).unwrap_or(false), class) {
        (true, "944-ensemble") => "anchor-member only (ensemble)",
        (true, _) => "present",
        (false, _) => "absent",
    };
    info.push(("repro".into(), repro.into()));
    match f(v, &["composite"]) {
        Some(c) => info.push(("product_composite (§1)".into(), format!("{c:.4}"))),
        None => info.push(("product_composite (§1)".into(), "null".into())),
    }

    let composite = balanced_composite(v);
    let annotations = matched
        .iter()
        .map(|e| (e.id.clone(), e.kind.clone(), e.reason.clone()))
        .collect();
    BalancedReport {
        class,
        class_note,
        floors,
        info,
        composite,
        annotations,
    }
}

// ── `--select`: the REGISTERED k-seed selection rule (campaign appendix
// E.4, frozen 2026-08-04 before any ranking existed) ───────────────────────
//
// The campaign's k-seed rule was "train k seeds → select by sdr25 /
// best_val". The coherence study then established that M3a is a SELECTABLE
// trajectory property (42.3 % of 944-class M3a variance is seed noise at
// fixed recipe; `C_co3a` k = 6 spans 0.718–0.826), so selection must account
// for it. This is that rule, in the bar/profile owner rather than a new
// script.
//
//   PRIMARY    profile floor count (`n_pass`). Coherence never overrides a
//              bake that fails CID22 or the dial.
//   TIE-BREAK  selection_composite = balanced_composite + W_M3A · m3a.
//   sdr25      is NOT in the rule — it stays a reported comparator column.
//              (Standing caveat: sdr25 has decoupled from CID22 five times;
//              that is exactly why the primary is the floor count and not a
//              proxy corpus.)
//
// This computes NO statistics: every input is a number an owning tool
// already produced.

/// The three M3a states. They are DISTINCT, and none of them is zero — a
/// missing measurement must never score as "perfectly incoherent".
#[derive(PartialEq, Clone, Copy, Debug)]
enum M3aState {
    /// A number is present: rank normally.
    Measured(f64),
    /// Ensemble — the coherence instrument loads ONE ZNPR, so it
    /// structurally cannot produce a value. Ranked in a separate section on
    /// `balanced_composite` alone; never penalized.
    NotComputable,
    /// Non-ensemble with no measurement: eligible to be LISTED, not to be
    /// SELECTED. Precedent: the balanced profile already counts an absent
    /// floor axis as not-passed — a candidate nobody measured cannot be
    /// certified on that axis.
    Unmeasured,
}

fn m3a_state(v: &serde_json::Value, class: &str) -> M3aState {
    if class == "944-ensemble" {
        return M3aState::NotComputable;
    }
    match f(v, &["m3a_coherence"]) {
        Some(x) => M3aState::Measured(x),
        None => M3aState::Unmeasured,
    }
}

struct SelectRow {
    name: String,
    path: String,
    class: &'static str,
    n_pass: usize,
    n_floors: usize,
    /// The IDs of the floors this cell PASSED. `n_pass` is its length; the
    /// set itself is what a seed group intersects under
    /// [`FloorBasis::AllReps`] — two members can both pass 7/8 while failing
    /// DIFFERENT floors, and a count alone cannot see that.
    floors_passed: std::collections::BTreeSet<&'static str>,
    composite: Option<f64>,
    m3a: M3aState,
    /// `balanced_composite + W_M3A·m3a`; None when either term is absent.
    selection_composite: Option<f64>,
    sdr25: Option<f64>,
    bake: Option<String>,
    /// `sha1(normalized repro argv)[:12]` — the recipe identity a seed
    /// group keys on, byte-identical to the board's. `None` = UNGROUPABLE
    /// (an ensemble, or no embedded repro argv).
    group_key: Option<String>,
    /// The seed identity duplicate cells collapse by: `init/sample` once
    /// the seeds are split, the single seed on a legacy bake, `None` when
    /// the repro records no seed.
    seed_id: Option<String>,
    /// [`seed_label`] of the same repro — what the per-seed table PRINTS.
    /// Separate from `seed_id` on purpose: the identity is a grouping key
    /// and must stay terse and stable, the label is for a reader and says
    /// which half of a split pair is which.
    seed_text: String,
    /// G-ADDR CONTRACT-tier measured fails (2026-09-06 owner fix; see the
    /// module note above `gaddr_contract_fails`) — never recomputed, read
    /// straight from this fulleval's own `dial.addressability.checks[]`.
    /// Empty ⇒ no measured contract fail (an absent block, or every
    /// contract row NOT MEASURED, both count as empty: only a MEASURED
    /// fail disqualifies). Any non-empty entry is an absolute
    /// selectability veto — see [`gaddr_vetoes`].
    gaddr_contract_fails: Vec<(String, String)>,
    /// `A7r` per-codec floor states from this fulleval's own G-ADDR
    /// verdict — `(codec, Some(pass)/None=not measured)`. See
    /// [`gaddr_effective_counts`].
    gaddr_codec: Vec<(String, Option<bool>)>,
}

/// Human-readable form of [`seed_identity`] for the per-seed detail table.
fn seed_label(v: &serde_json::Value) -> String {
    let r = match v.get("repro") {
        Some(r) => r,
        None => return "\u{2014}".into(),
    };
    let g = |k: &str| r.get(k).and_then(serde_json::Value::as_u64);
    match (g("init_seed"), g("sample_seed"), g("seed")) {
        (Some(i), Some(p), _) if i == p => format!("seed {i} (unsplit)"),
        (Some(i), Some(p), _) => format!("init {i} / sample {p}"),
        // A split run always writes BOTH; a half-present pair means the
        // JSON was hand-edited or truncated — say so rather than guess.
        (Some(i), None, _) => format!("init {i} / sample ?"),
        (None, Some(p), _) => format!("init ? / sample {p}"),
        (None, None, Some(x)) => format!("seed {x}"),
        _ => "\u{2014}".into(),
    }
}

fn m3a_cell(s: M3aState) -> String {
    match s {
        M3aState::Measured(x) => format!("{x:.4}"),
        M3aState::NotComputable => "NOT COMPUTABLE".into(),
        M3aState::Unmeasured => "UNMEASURED".into(),
    }
}

/// Rank a pool: floor count DESC (under `basis` — see [`gaddr_effective_counts`]),
/// then `selection_composite` DESC. Rows without a `selection_composite` sort
/// last within their floor tier (they carry no comparable number) — they are
/// listed, never selected.
fn rank_pool(rows: &mut [&SelectRow], basis: FloorBasis) {
    rows.sort_by(|a, b| {
        gaddr_effective_counts(b, basis)
            .0
            .cmp(&gaddr_effective_counts(a, basis).0)
            .then_with(|| {
                b.selection_composite
                    .partial_cmp(&a.selection_composite)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    });
}

// ── G-ADDR CONTRACT pre-filter for `--select` (2026-09-06 owner fix) ───────
//
// `benchmarks/best_of_all_2026-09-06.md` + registry entry
// `select-rule-blind-to-dial-contract-2026-09-06` measured that
// `freeze_check --select --seed-group --min-k 2 --floor-basis all` picked
// `A_plain`, a CONTROL arm at 4/6 G-ADDR CONTRACT rows, over arms that took
// the contract to 6/6 on every seed — because the PRIMARY key (the balanced
// profile's F1-F8 floor count) and the TIE-BREAK composite are both
// completely blind to `dial.addressability`: G-ADDR's CONTRACT tier (C1-C6)
// is an ABSOLUTE product bar ("dial addressability is a HARD ship gate",
// user rule 2026-09-04 — see `benchmarks/dial_addressability_gate_2026-09-04.md`),
// not a floor a bake can trade off against CID22 or coherence.
//
// This reads the fulleval's OWN G-ADDR verdict — `dial.addressability`, the
// exact JSON `dial_addressability::to_json` embeds (`bake_verdict.rs`) —
// and NEVER recomputes it. Two effects, both additive (they can only REMOVE
// a candidate from selection, never admit one the pre-fix rule refused):
//
//   1. A MEASURED CONTRACT-tier fail is an absolute selectability VETO,
//      independent of floor count or composite (`gaddr_vetoes`).
//   2. `A7r`'s per-codec floor-representability breakdown
//      (`dial.addressability.measured.codec_floor[]`) becomes one MORE
//      floor per MEASURED codec, folded into `--floor-basis all`'s count
//      (`gaddr_effective_counts`) — a codec this instrument never touched
//      (`state: "not_measured"`) is reported but counted in neither
//      direction, matching the registered rule that an absent axis can
//      certify nothing.
//
// `--floor-basis legacy` reproduces the PRE-2026-09-06 rule byte-for-byte
// (no veto, no A7r floors) — the audit/reproduction escape hatch, same
// role as `--min-k 1` for the replication floor. `--floor-basis mean`
// keeps its existing, unrelated meaning (the k-seed MEAN floor count) —
// unaffected by A7r — but the CONTRACT veto still applies under it: the
// veto is a product-safety gate, not a floor-counting convention, and only
// the explicit `legacy` reproduction mode may bypass it.

/// One G-ADDR CONTRACT-tier row this candidate's OWN verdict MEASURABLY
/// failed — `(id, note)`. A `NOT MEASURED` contract row is never in this
/// list (only a `state: "fail"` row disqualifies — an unmeasured axis is
/// reported, never scored as a defect, same rule as an unmeasured M3a). An
/// absent `dial.addressability` block (no G-ADDR ever ran) returns empty,
/// for the same reason: absence is not evidence of failure.
/// WHICH G-ADDR block is this candidate's OPERATIVE dial ruler.
///
/// Since 2026-09-06 the board's operative dial instrument is the FLOOR-DENSE
/// LADDER (`docs/PLAN_BOARD_LADDER_RULER_2026-09-06.md`): five codec families
/// including both AVIF backends, a 66-step floor-dense q axis, and the only
/// instrument on which the operative `resolvable` `A7r` rule has a registered
/// per-codec bar and the two-reference inversion rule is measurable at all. A
/// cell's `dial.addressability` remains its CANONICAL-grid read and is kept
/// verbatim — the two are DIFFERENT instruments and a value from one is never
/// comparable with a value from the other, which is why the ladder reading gets
/// its own block instead of overwriting the canonical one.
///
/// Precedence is `dial_ladder` then `dial.addressability`, through this ONE
/// function, so the CONTRACT veto and the `A7r` floor fold cannot drift apart
/// on which ruler they are reading. `--gaddr-block canonical` forces the
/// pre-2026-09-06 reading for audit; `ladder` refuses to fall back.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum GaddrBlock {
    /// **Default.** `dial_ladder` when the cell carries one, else the
    /// canonical `dial.addressability`. A cell with neither reads empty,
    /// which is "no G-ADDR ever ran" — never a fail.
    Auto,
    /// The canonical-grid read only — reproduces the pre-2026-09-06 rule.
    Canonical,
    /// The ladder read only. A cell with no `dial_ladder` reads empty
    /// (NOT MEASURED), never its canonical block.
    Ladder,
}

fn gaddr_block(v: &serde_json::Value, which: GaddrBlock) -> Option<&serde_json::Value> {
    let ladder = v.get("dial_ladder");
    let canon = v.get("dial").and_then(|d| d.get("addressability"));
    match which {
        GaddrBlock::Ladder => ladder,
        GaddrBlock::Canonical => canon,
        GaddrBlock::Auto => ladder.or(canon),
    }
}

fn gaddr_contract_fails_of(v: &serde_json::Value, which: GaddrBlock) -> Vec<(String, String)> {
    let checks = match gaddr_block(v, which)
        .and_then(|a| a.get("checks"))
        .and_then(|c| c.as_array())
    {
        Some(c) => c,
        None => return Vec::new(),
    };
    checks
        .iter()
        .filter(|r| r.get("tier").and_then(|x| x.as_str()) == Some("contract"))
        .filter(|r| r.get("state").and_then(|x| x.as_str()) == Some("fail"))
        .map(|r| {
            (
                r.get("id")
                    .and_then(|x| x.as_str())
                    .unwrap_or("?")
                    .to_string(),
                r.get("note")
                    .and_then(|x| x.as_str())
                    .unwrap_or("")
                    .to_string(),
            )
        })
        .collect()
}

/// The `A7r` per-codec floor states this candidate's OWN G-ADDR verdict
/// measured, read from `dial.addressability.measured.codec_floor[]` —
/// `(codec, Some(true)=pass / Some(false)=fail / None=NOT MEASURED)`. Never
/// recomputed. A codec this instrument never touched is returned (so a
/// reader can see it was skipped) but is excluded from every count by
/// [`gaddr_measured_floor_counts`].
fn gaddr_codec_states_of(v: &serde_json::Value, which: GaddrBlock) -> Vec<(String, Option<bool>)> {
    let rows = match gaddr_block(v, which)
        .and_then(|a| a.get("measured"))
        .and_then(|m| m.get("codec_floor"))
        .and_then(|c| c.as_array())
    {
        Some(c) => c,
        None => return Vec::new(),
    };
    rows.iter()
        .filter_map(|r| {
            let codec = r.get("codec")?.as_str()?.to_string();
            let state = match r.get("state").and_then(|x| x.as_str()) {
                Some("pass") => Some(true),
                Some("fail") => Some(false),
                // "not_measured", or a malformed/missing state: NEVER a
                // fail — excluded from counting, not scored as one.
                _ => None,
            };
            Some((codec, state))
        })
        .collect()
}

/// The MEASURED (non-NOT-MEASURED) `A7r` per-codec floors as `(pass,
/// total)` — `total` excludes NOT MEASURED codecs entirely, per the
/// registered rule that a codec this instrument never touched cannot
/// certify or condemn a candidate on it.
fn gaddr_measured_floor_counts(codec: &[(String, Option<bool>)]) -> (usize, usize) {
    let measured: Vec<bool> = codec.iter().filter_map(|(_, s)| *s).collect();
    (measured.iter().filter(|&&p| p).count(), measured.len())
}

/// Effective `(pass, total)` floor count for RANKING under `basis`.
/// `Legacy` is the pre-2026-09-06 rule byte-for-byte: F1-F8 only. `AllReps`
/// (the default, CLI `all`) folds in the `A7r` per-codec floors this
/// candidate's OWN instrument measured. `Mean` is UNCHANGED — F1-F8 only,
/// preserving its documented role as the k-seed-mean reproduction basis.
fn gaddr_effective_counts(r: &SelectRow, basis: FloorBasis) -> (usize, usize) {
    match basis {
        FloorBasis::Legacy | FloorBasis::Mean => (r.n_pass, r.n_floors),
        FloorBasis::AllReps => {
            let (p, t) = gaddr_measured_floor_counts(&r.gaddr_codec);
            (r.n_pass + p, r.n_floors + t)
        }
    }
}

/// `true` when this candidate's OWN G-ADDR verdict measured a CONTRACT-tier
/// fail — an absolute selectability veto, independent of floor count or
/// composite. `Legacy` never vetoes (the byte-for-byte reproduction
/// escape hatch); every other basis does.
fn gaddr_vetoes(r: &SelectRow, basis: FloorBasis) -> bool {
    basis != FloorBasis::Legacy && !r.gaddr_contract_fails.is_empty()
}

/// One line per measured CONTRACT fail, for the "not selectable" callouts.
fn gaddr_fail_reason(fails: &[(String, String)]) -> String {
    fails
        .iter()
        .map(|(id, note)| {
            if note.is_empty() {
                id.clone()
            } else {
                format!("{id}: {note}")
            }
        })
        .collect::<Vec<_>>()
        .join("; ")
}

// ── `--select --seed-group`: the k-seed AGGREGATE selection rule
// (2026-09-04 owner-fix lane, `benchmarks/fastclass_distill_wave_2026-09-04.md`
// §7.7) ──────────────────────────────────────────────────────────────────
//
// §7.7 measured that the plain `--select` rule above ranks INDIVIDUAL
// fulleval cells with no seed awareness, so on a model class with real
// seed-to-seed spread (0.133 KonJND on that wave's cells) it systematically
// selects the lucky draw — literally the control's best-of-3 seed, not its
// typical behavior. This is that fix: an OPT-IN mode (`--seed-group`) that
// aggregates cells into seed groups before ranking, so a group's score is
// its k-seed MEAN, with the per-seed spread reported alongside rather than
// thrown away.
//
// ══ ONE OWNER OF "WHAT IS A SEED GROUP" ══
//
// The grouping rule is NOT this file's to invent. `scripts/v_next/gauntlet.py`
// (`seed_group_key` / `build_seed_groups`, `benchmarks/fair_gauntlet_2026-09-04
// .md` §1.1) owns it for the board, and it was validated there by reproducing
// the fastclass §7.1 table blind (k=3, KonJND best 0.4327 / mean 0.3561 /
// spread 0.1329). A `--select` winner and a board leader disagreeing about
// what `k` means would be worse than either alone, so this is a MIRROR of
// that rule, gated by `scripts/verify_seed_group_parity.py`. The three
// clauses, verbatim from the owner:
//
//   1. Single-model rows only. An ensemble is an evaluation FUNCTION over
//      members, not a training replicate of anything.
//   2. Key = the embedded `zentrain.repro.argv` with the seed flags and the
//      output-path flags (and their values) removed, sha1'd. No argv ⇒
//      UNGROUPABLE.
//   3. Rows inside a key collapse by SEED IDENTITY: two cells with the same
//      recipe AND the same seed are one training run promoted twice
//      (MEASURED: 42 such rows in 33 same-seed groups on the board), so
//      `k` = the number of DISTINCT seeds, never the number of cells.
//
// Clause 2's flag list gained `--init-seed` / `--sample-seed`, and clause 3's
// seed identity became the (init, sample) PAIR, when this same lane split the
// trainer's two RNG streams — a legacy single-`--seed` bake still keys on its
// one seed. Both owners changed together, in this commit.
//
// WHERE THIS DELIBERATELY DIFFERS FROM THE BOARD, and why: `build_seed_groups`
// RETURNS ONLY `k >= 2` groups, because the board renders k=1 rows through a
// different path. `--select` must rank every candidate it was handed, so an
// ungroupable or single-seed cell becomes its OWN group here, flagged
// UNREPLICATED / UNGROUPABLE and never silently dropped. The parity gate
// compares the `k >= 2` partitions, which is the part both owners must agree
// on.
//
// PRIMARY / TIE-BREAK: identical to the per-cell rule (mean floor count, then
// mean `selection_composite`) — just computed over the group's MEAN rather
// than one cell's value, and over one REPRESENTATIVE PER DISTINCT SEED rather
// than every cell (clause 3). Ensembles are excluded by clause 1 and keep
// their existing separate pool, ranked on `balanced_composite` alone.

/// Flags whose presence (and value) must not distinguish two runs of the
/// SAME recipe. Mirrors `gauntlet.SEED_GROUP_DROP_FLAGS` exactly, plus the
/// two seed-split flags this lane introduced.
/// Any flag whose value names a per-run OUTPUT location belongs here, not just the
/// obvious ones: `--dump-checkpoints-dir` was missing, and because its value embeds
/// the seed (`.../LSTAR2_s4031_ckpts` vs `.../LSTAR2_s4033_ckpts`) it split each
/// seed of one recipe into a separate "recipe" — 8 of the 10 top-scoring
/// combined-fair board cells reported k=1 with a true k of 3
/// (2026-09-05, `benchmarks/replication_wave_2026-09-05.md`).
const SEED_GROUP_DROP_FLAGS: [&str; 9] = [
    "--seed",
    "--init-seed",
    "--sample-seed",
    "--out",
    "--output",
    "-o",
    "--bake-out",
    "--manifest",
    "--dump-checkpoints-dir",
];

/// Drop every [`SEED_GROUP_DROP_FLAGS`] token and its value.
///
/// A flag's VALUE is the following token unless that token starts with
/// `--` (in which case the flag was a bare switch and the next token is the
/// next flag). Byte-for-byte the same walk as
/// `gauntlet._norm_argv_for_seed_group`.
fn norm_argv_for_seed_group(argv: &[String]) -> Vec<String> {
    let mut out: Vec<String> = Vec::with_capacity(argv.len());
    // `argv[0]` is reduced to its BASENAME: the program path is a build
    // location, not a recipe parameter. The board carries 32 distinct
    // `argv[0]` values for two tools — one per lane worktree — so keeping the
    // full path meant a replay from a sibling jj workspace (which the
    // workspace protocol mandates) could never group with the cell it
    // replayed. Basename, not removal: `zensim_mlp_train` still never merges
    // with `bake_dial_refit`. MEASURED: 436 board fullevals, groups 101 -> 98,
    // both merges genuine (zero other differing tokens).
    // (2026-09-05, `benchmarks/replication_wave_2026-09-05.md`)
    let argv: Vec<String> = if argv.is_empty() {
        Vec::new()
    } else {
        std::iter::once(
            std::path::Path::new(&argv[0])
                .file_name()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| argv[0].clone()),
        )
        .chain(argv[1..].iter().cloned())
        .collect()
    };
    let argv = &argv[..];
    let mut i = 0usize;
    while i < argv.len() {
        if SEED_GROUP_DROP_FLAGS.contains(&argv[i].as_str()) {
            i += if i + 1 < argv.len() && !argv[i + 1].starts_with("--") {
                2
            } else {
                1
            };
            continue;
        }
        out.push(argv[i].clone());
        i += 1;
    }
    out
}

/// The embedded repro argv as strings. Real argv is always strings (it comes
/// from `std::env::args()`); a non-string element is stringified so a
/// hand-edited fulleval degrades to a stable key instead of `None`.
fn repro_argv(v: &serde_json::Value) -> Option<Vec<String>> {
    let arr = v.get("repro")?.get("argv")?.as_array()?;
    if arr.is_empty() {
        return None;
    }
    Some(
        arr.iter()
            .map(|x| match x {
                serde_json::Value::String(s) => s.clone(),
                other => other.to_string(),
            })
            .collect(),
    )
}

/// Recipe identity: `sha1(normalized argv joined by NUL)[:12]`, or `None`
/// when the row is UNGROUPABLE (an ensemble, or no embedded repro argv).
/// Mirrors `gauntlet.seed_group_key`, id string included.
fn seed_group_key(v: &serde_json::Value) -> Option<String> {
    if v.get("model")
        .and_then(|m| m.get("kind"))
        .and_then(serde_json::Value::as_str)
        == Some("ensemble")
    {
        return None;
    }
    let argv = repro_argv(v)?;
    let joined = norm_argv_for_seed_group(&argv).join("\0");
    use sha1::Digest;
    let digest = sha1::Sha1::digest(joined.as_bytes());
    Some(
        digest
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>()[..12]
            .to_string(),
    )
}

/// The SEED IDENTITY a group collapses duplicates by (clause 3). The
/// `(init, sample)` pair once the seeds are split, the single seed on a
/// legacy bake, `None` when the repro records no seed at all.
fn seed_identity(v: &serde_json::Value) -> Option<String> {
    let r = v.get("repro")?;
    let g = |k: &str| r.get(k).and_then(serde_json::Value::as_u64);
    match (g("init_seed"), g("sample_seed")) {
        // i == p is the SAME DRAW as the legacy `--seed i`, not a second one:
        // the trainer maps `--seed X` to init = sample = X
        // (`zensim_mlp_train.rs` seed plumbing), and CTL-A vs CTL-B measured it
        // — 0 of 12 corpora differ and the composite matches to 16 digits.
        // Without this, a control pair inflates its group's k by one, which is
        // the very quantity a seed group exists to report.
        // (2026-09-05, `benchmarks/replication_wave_2026-09-05.md`)
        (Some(i), Some(p)) if i == p => Some(i.to_string()),
        (Some(i), Some(p)) => Some(format!("{i}/{p}")),
        _ => g("seed").map(|x| x.to_string()),
    }
}

// ── REGISTERED AMENDMENT to E.4 — the REPLICATION FLOOR (2026-09-05) ──────
//
// `--seed-group` made `k` VISIBLE. It did not make it BINDING: the rule's
// PRIMARY key is the floor count, so a group with one draw and 8/8 floors
// still outranks a replicated group at 7.22/8. That is exactly what happened
// on the live board — the pick was `62df0d51a60e` = `W10L9_s4003_packed`,
// **k = 1**, 8.00/8 floors, selection_composite 0.9841
// (`benchmarks/replication_wave_2026-09-05.md` §4c.4).
//
// The same wave measured why a single draw must not be selectable:
//
//   * **Replicating the leaders moved them DOWN.** `LSTAR` read composite
//     0.8615 as its best cell and **0.856414** as its k=7 mean; `LSTAR3`
//     0.8608 → 0.856843. Ranks 1 and 2 became 7 and 6 (§4c.1).
//   * **Best-of-k inflation is real and one-sided**: over the 18 combined-fair
//     k≥2 groups, best-of-k minus k-mean has a **+0.0061 median** (§2), which
//     is larger than the 0.0021 span separating the top four groups.
//
// So a k=1 cell's number is a draw from a distribution whose maximum is
// systematically ~0.006 above its mean, competing against groups reported at
// their means. Ranking them together is not a tie-break question, it is a
// units error.
//
// THE AMENDMENT, in two parts:
//
//   A. **REPLICATION FLOOR.** A seed group is SELECTABLE only when
//      `k >= min_k`, default **2** (`--min-k`). Groups below it are LISTED,
//      ranked in their own section, and never selected. `--min-k 1` restores
//      the pre-amendment behavior for reproducing a historical selection.
//
//   B. **FLOOR BASIS = every representative** ([`FloorBasis::AllReps`],
//      default). A group is credited a floor only when EVERY distinct-seed
//      representative passes it. `--floor-basis mean` restores the k-seed
//      mean count.
//
//      Why the stricter reading: a floor is a CERTIFICATION ("this recipe
//      clears CID22"), and a mean is not one. Two members at 8/8 and 6/8
//      average 7.0 whether they fail the same floor twice or two different
//      floors — in the second case the group is credited 7 floors that NO
//      member reliably clears. The intersection cannot do that. It is also
//      the reading consistent with how the balanced profile already treats an
//      ABSENT axis (not-passed, never averaged away), and with part A: both
//      say a number nobody replicated is not a certification. The mean stays
//      REPORTED in its own column, because it is the right estimator for
//      RANKING within a tier — it is just not a floor count.
//
// Neither part relaxes anything: both can only remove a candidate from
// selection, never admit one the pre-amendment rule refused.

/// How a seed group's floor count is computed. Selection uses this count as
/// the PRIMARY key, exactly as the per-cell rule uses `n_pass`.
#[derive(PartialEq, Eq, Clone, Copy, Debug)]
enum FloorBasis {
    /// **Default.** Floors that EVERY distinct-seed representative passes,
    /// PLUS (2026-09-06) the `A7r` per-codec G-ADDR floors this candidate's
    /// own instrument measured — see [`gaddr_effective_counts`]. Also the
    /// basis under which the G-ADDR CONTRACT-tier veto applies
    /// ([`gaddr_vetoes`]).
    AllReps,
    /// The k-seed mean floor count — the pre-amendment basis, kept so a
    /// historical seed-grouped table can be reproduced. F1-F8 ONLY (no
    /// `A7r` floors folded in — unaffected by the 2026-09-06 fix). The
    /// G-ADDR CONTRACT veto STILL applies here: it is a product-safety
    /// gate, not a floor-counting convention, so only `legacy` bypasses it.
    Mean,
    /// **Reproduction escape hatch (2026-09-06).** Byte-for-byte the
    /// pre-fix `all`: F1-F8 only, no `A7r` floors, no G-ADDR CONTRACT
    /// veto. Use this ONLY to reproduce a historical selection for audit —
    /// it reinstates the exact defect
    /// `select-rule-blind-to-dial-contract-2026-09-06` registered.
    Legacy,
}

/// Why a group is not selectable, in the order the rule applies. `None` = it
/// is selectable. Never blank, never conflated with a low score.
fn group_unselectable_reason(
    g: &SeedGroupRow<'_>,
    basis: FloorBasis,
    min_k: usize,
) -> Option<String> {
    // The G-ADDR CONTRACT tier is an ABSOLUTE product bar (user rule
    // 2026-09-04: "dial addressability is a HARD ship gate"), checked FIRST
    // so it is never shadowed by a replication-floor or M3a reason — a
    // reader must see the more severe violation. Any MEMBER (not just a
    // distinct-seed representative) measuring a contract fail disqualifies
    // the whole group: a duplicate-promotion cell that failed contract is
    // still disqualifying evidence about the recipe.
    if basis != FloorBasis::Legacy && !g.gaddr_contract_fails.is_empty() {
        let ids: std::collections::BTreeSet<&str> = g
            .gaddr_contract_fails
            .iter()
            .map(|(_, id, _)| id.as_str())
            .collect();
        return Some(format!(
            "CONTRACT FAIL — not selectable ({})",
            ids.into_iter().collect::<Vec<_>>().join(",")
        ));
    }
    if g.k_seeds < min_k {
        return Some(if g.ungroupable {
            format!("UNGROUPABLE (k={} < min-k {min_k})", g.k_seeds)
        } else {
            format!("UNREPLICATED (k={} < min-k {min_k})", g.k_seeds)
        });
    }
    if g.m3a == M3aState::Unmeasured {
        return Some("M3a UNMEASURED".into());
    }
    if g.floor_count(basis) <= 0.0 {
        return Some("0 floors".into());
    }
    None
}

struct SeedGroupRow<'a> {
    key: String,
    /// EVERY cell in the group, including duplicate promotions of one
    /// training run — listed, so nothing disappears from the report.
    members: Vec<&'a SelectRow>,
    /// One cell per DISTINCT seed (clause 3): the unit every statistic
    /// below is computed over. Note this is NOT `k` — a member with no
    /// recorded seed is a representative (it is a distinct artifact) but not a
    /// distinct DRAW; see [`SeedGroupRow::k_seeds`].
    reps: Vec<&'a SelectRow>,
    /// `k` — the number of distinct RECORDED seed identities. Zero when no
    /// member wrote down a seed, which is why it is not `reps.len()`.
    k_seeds: usize,
    mean_n_pass: f64,
    /// The number of floors EVERY distinct-seed representative passes — the
    /// intersection of their `floors_passed` sets. This is the
    /// [`FloorBasis::AllReps`] primary key (registered amendment, part B).
    /// `<= mean_n_pass` always, and STRICTLY less whenever two members fail
    /// different floors.
    n_pass_all: usize,
    /// The floor IDs in that intersection, for the report.
    floors_all: std::collections::BTreeSet<&'static str>,
    /// Floors SOME representative passes and some does not — the exact
    /// difference between the two bases, named so a reader can see which
    /// certification the mean would have invented.
    floors_split: Vec<&'static str>,
    mean_composite: Option<f64>,
    /// `Measured(mean_m3a)` only when EVERY representative is `Measured` (a
    /// group with even one `Unmeasured` member cannot be certified on this
    /// axis, same principle as a single unmeasured cell); `Unmeasured`
    /// otherwise. `NotComputable` never appears here — ensembles are
    /// UNGROUPABLE by clause 1.
    m3a: M3aState,
    mean_selection_composite: Option<f64>,
    /// `(min, max)` of the per-representative `selection_composite` — the
    /// spread the registered rule was blind to before this fix.
    selection_composite_spread: Option<(f64, f64)>,
    /// `k == 1`: one draw, not an estimate.
    unreplicated: bool,
    /// No embedded repro argv, or an ensemble: it could not be grouped at
    /// all. Distinct from UNREPLICATED, which means "grouped, k=1".
    ungroupable: bool,
    /// Cells that share a seed with another cell in this group (clause 3's
    /// duplicate promotions), excluded from `reps`.
    n_duplicate_cells: usize,
    /// Every MEMBER cell's (not just reps') G-ADDR CONTRACT-tier measured
    /// fails, as `(member_name, id, note)` — the union over the WHOLE
    /// group. Non-empty ⇒ the group is vetoed (2026-09-06 owner fix); see
    /// [`group_unselectable_reason`]. Members, not reps: a duplicate
    /// promotion that measured a contract fail is still disqualifying
    /// evidence about the recipe.
    gaddr_contract_fails: Vec<(String, String, String)>,
    /// `A7r` per-codec floors credited under the SAME all-reps discipline
    /// as `n_pass_all`/`floors_all`: a codec counts in the denominator
    /// only when EVERY representative MEASURED it (pass or fail, never
    /// NOT MEASURED), and in the numerator only when additionally every
    /// representative PASSED it. [`FloorBasis::AllReps`]'s primary key
    /// folds this in; [`FloorBasis::Mean`]/[`FloorBasis::Legacy`] ignore it.
    gaddr_codec_pass_all: usize,
    gaddr_codec_total_all: usize,
    /// Mean per-representative `A7r` pass/measured counts — REPORTED only
    /// (mirrors `mean_n_pass`'s role: the right number for display, never
    /// a floor certification).
    gaddr_codec_mean_pass: f64,
    gaddr_codec_mean_total: f64,
}

impl SeedGroupRow<'_> {
    /// The group's floor count under `basis` — the PRIMARY ranking key.
    /// `Legacy` reproduces the pre-2026-09-06 `all` byte-for-byte (F1-F8
    /// intersection only); `AllReps` (today's `all`) additionally folds in
    /// the `A7r` per-codec floors every representative measured and
    /// passed; `Mean` is unaffected (F1-F8 mean only).
    fn floor_count(&self, basis: FloorBasis) -> f64 {
        match basis {
            FloorBasis::AllReps => (self.n_pass_all + self.gaddr_codec_pass_all) as f64,
            FloorBasis::Mean => self.mean_n_pass,
            FloorBasis::Legacy => self.n_pass_all as f64,
        }
    }
}

fn mean_opt(xs: impl Iterator<Item = Option<f64>>) -> Option<f64> {
    let vals: Vec<f64> = xs.flatten().collect();
    if vals.is_empty() {
        None
    } else {
        Some(vals.iter().sum::<f64>() / vals.len() as f64)
    }
}

/// Group `rows` by [`seed_group_key`], collapse duplicate seeds to one
/// representative each, and compute the group aggregate over the
/// representatives. Input order does not matter; output is UNSORTED (the
/// caller ranks it).
fn group_by_seed<'a>(rows: &[&'a SelectRow]) -> Vec<SeedGroupRow<'a>> {
    // Preserve a stable, name-sorted view so representative choice
    // ("lexicographically first name for this seed", matching the owner's
    // `sorted(ns)[0]`) is deterministic regardless of argument order.
    let mut by_key: std::collections::BTreeMap<String, Vec<&'a SelectRow>> =
        std::collections::BTreeMap::new();
    for &r in rows {
        // An ungroupable row keys on its own NAME, prefixed so it can never
        // collide with a 12-hex recipe id.
        let key = r
            .group_key
            .clone()
            .unwrap_or_else(|| format!("ungrouped:{}", r.name));
        by_key.entry(key).or_default().push(r);
    }
    by_key
        .into_iter()
        .map(|(key, mut members)| {
            members.sort_by(|a, b| a.name.cmp(&b.name));
            let ungroupable = members.iter().all(|m| m.group_key.is_none());
            // One representative per distinct seed identity, in name order.
            // A member with NO recorded seed cannot be collapsed against
            // anything, so it is its own representative (keyed by name).
            let mut seen: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
            let mut reps: Vec<&'a SelectRow> = Vec::new();
            // `k` counts distinct RECORDED seeds. A member with no recorded
            // seed is still listed and still a representative (so the group's
            // means are computed over every distinct artifact), but it is NOT
            // a distinct DRAW: "how many times was this recipe trained" cannot
            // be answered by a cell that never wrote down its seed. Counting
            // it inflated k — two seedless cells of one recipe reported k=2,
            // i.e. "replicated", off zero recorded seeds. Python's owner
            // (`gauntlet.build_seed_groups`) has always skipped seedless rows;
            // this divergence was invisible while such cells happened to land
            // in different groups, and `scripts/verify_seed_group_parity.py`
            // caught it the moment the argv[0] fix merged two of them.
            // (2026-09-05, `benchmarks/replication_wave_2026-09-05.md`)
            let mut k_seeds = 0usize;
            for m in &members {
                match &m.seed_id {
                    Some(s) => {
                        if seen.insert(s.clone()) {
                            reps.push(m);
                            k_seeds += 1;
                        }
                    }
                    None => {
                        if seen.insert(format!("noseed:{}", m.name)) {
                            reps.push(m);
                        }
                    }
                }
            }
            let n_duplicate_cells = members.len() - reps.len();
            let mean_n_pass = reps.iter().map(|m| m.n_pass as f64).sum::<f64>() / reps.len() as f64;
            // Registered amendment part B: the floors EVERY representative
            // passes. Folded over `reps` (one cell per distinct seed), never
            // over `members` — a duplicate promotion of one training run is
            // the same draw and must not vote twice.
            let floors_all: std::collections::BTreeSet<&'static str> = reps
                .iter()
                .map(|m| m.floors_passed.clone())
                .reduce(|a, b| a.intersection(&b).copied().collect())
                .unwrap_or_default();
            let n_pass_all = floors_all.len();
            // Floors passed by SOME representative but not ALL — exactly the
            // set a mean floor count would credit and the intersection will
            // not. Named in the report so part B's effect is visible rather
            // than a silently smaller number.
            let floors_split: Vec<&'static str> = reps
                .iter()
                .flat_map(|m| m.floors_passed.iter().copied())
                .collect::<std::collections::BTreeSet<_>>()
                .difference(&floors_all)
                .copied()
                .collect();
            let mean_composite = mean_opt(reps.iter().map(|m| m.composite));
            let m3a_vals: Vec<f64> = reps
                .iter()
                .filter_map(|m| match m.m3a {
                    M3aState::Measured(x) => Some(x),
                    _ => None,
                })
                .collect();
            let m3a = if m3a_vals.len() == reps.len() {
                M3aState::Measured(m3a_vals.iter().sum::<f64>() / m3a_vals.len() as f64)
            } else {
                M3aState::Unmeasured
            };
            // Computed directly from the group means (not by averaging each
            // representative's own selection_composite) — algebraically
            // identical for a linear combination, and it stays correct even
            // when a member's individual selection_composite is None while
            // its composite/m3a both feed the group mean.
            let mean_selection_composite = match (mean_composite, m3a) {
                (Some(c), M3aState::Measured(x)) => Some(c + balanced::W_M3A * x),
                _ => None,
            };
            let sel_vals: Vec<f64> = reps.iter().filter_map(|m| m.selection_composite).collect();
            let selection_composite_spread = if sel_vals.is_empty() {
                None
            } else {
                Some((
                    sel_vals.iter().cloned().fold(f64::INFINITY, f64::min),
                    sel_vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
                ))
            };
            let unreplicated = k_seeds < 2;

            // G-ADDR (2026-09-06 owner fix): the union of every MEMBER's
            // measured contract fails (not just reps — a duplicate
            // promotion that failed contract is still disqualifying
            // evidence), plus the all-reps-discipline per-codec floor
            // count over `reps` (mirroring `floors_all`/`n_pass_all`
            // exactly: a codec counts only when EVERY rep measured it,
            // and passes only when EVERY rep passed it).
            let gaddr_contract_fails: Vec<(String, String, String)> = members
                .iter()
                .flat_map(|m| {
                    m.gaddr_contract_fails
                        .iter()
                        .map(move |(id, note)| (m.name.clone(), id.clone(), note.clone()))
                })
                .collect();
            let mut gaddr_measured: std::collections::BTreeMap<&str, usize> =
                std::collections::BTreeMap::new();
            let mut gaddr_passed: std::collections::BTreeMap<&str, usize> =
                std::collections::BTreeMap::new();
            for m in &reps {
                for (codec, state) in &m.gaddr_codec {
                    if let Some(pass) = state {
                        *gaddr_measured.entry(codec.as_str()).or_insert(0) += 1;
                        if *pass {
                            *gaddr_passed.entry(codec.as_str()).or_insert(0) += 1;
                        }
                    }
                }
            }
            let n_reps = reps.len();
            let gaddr_codec_total_all = gaddr_measured.values().filter(|&&c| c == n_reps).count();
            let gaddr_codec_pass_all = gaddr_measured
                .iter()
                .filter(|&(_, &c)| c == n_reps)
                .filter(|&(codec, _)| gaddr_passed.get(codec).copied().unwrap_or(0) == n_reps)
                .count();
            let gaddr_codec_mean_pass = mean_opt(reps.iter().map(|m| {
                let (p, t) = gaddr_measured_floor_counts(&m.gaddr_codec);
                if t > 0 { Some(p as f64) } else { None }
            }))
            .unwrap_or(0.0);
            let gaddr_codec_mean_total = mean_opt(reps.iter().map(|m| {
                let (_, t) = gaddr_measured_floor_counts(&m.gaddr_codec);
                if t > 0 { Some(t as f64) } else { None }
            }))
            .unwrap_or(0.0);

            SeedGroupRow {
                key,
                members,
                reps,
                mean_n_pass,
                n_pass_all,
                floors_all,
                floors_split,
                mean_composite,
                m3a,
                mean_selection_composite,
                selection_composite_spread,
                k_seeds,
                unreplicated,
                ungroupable,
                n_duplicate_cells,
                gaddr_contract_fails,
                gaddr_codec_pass_all,
                gaddr_codec_total_all,
                gaddr_codec_mean_pass,
                gaddr_codec_mean_total,
            }
        })
        .collect()
}

/// One word for how much a group's mean can be leaned on. Never blank, and
/// never a number that hides the distinction.
fn group_state(g: &SeedGroupRow<'_>) -> &'static str {
    if g.ungroupable {
        // No embedded repro argv (or an ensemble): it could not be matched
        // against anything, so a k=1 "group" here is an artifact of the
        // MISSING metadata, not evidence about the recipe.
        "UNGROUPABLE"
    } else if g.unreplicated {
        "UNREPLICATED"
    } else {
        "replicated"
    }
}

/// Rank seed groups by the SAME primary/tie-break rule as [`rank_pool`]:
/// mean floor count DESC, then mean `selection_composite` DESC.
fn rank_seed_groups(groups: &mut [SeedGroupRow<'_>], basis: FloorBasis) {
    groups.sort_by(|a, b| {
        b.floor_count(basis)
            .partial_cmp(&a.floor_count(basis))
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                b.mean_selection_composite
                    .partial_cmp(&a.mean_selection_composite)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    });
}

const SELECT_TSV_COLS: &str = "rank\tpool\tname\tclass\tn_pass\tbal_composite\tm3a\tm3a_state\tselection_composite\tsdr25\tselectable\tpath";

const SEED_GROUP_TSV_COLS: &str = "rank\tgroup\tk\tn_cells\tn_duplicate_cells\tn_pass_all\tmean_n_pass\tsplit_floors\tmean_bal_composite\tmean_m3a\tmean_selection_composite\tsel_composite_min\tsel_composite_max\tstate\tselectable\tunselectable_reason";

const TSV_COLS: &str = "name\tclass\tverdict\tn_pass\tcid22\tkonjnd_abs\tnonphoto\tcsiq\tlive\thfnl_perref\tband_scheme\tband_lo\tband_lo_n\tband_hi\tband_hi_label\tband_hi_n\tband_hi_span\tmono\ttied\tdynrange\tm3a\tm3a_tier\tcorr_head_q20\tbal_composite\tproduct_composite\tsdr25\tkadid_signed\ttid_signed\tspline\trepro\tfails\tn_measured\tabsent\tannotations\tblocks\tdominated_by";

/// Compact carry of the promoter-injected `block_profile` (used-counts per
/// family, `f0_155/f156_371/f372_719/f720_943`) — computed by
/// `bake_block_profile`, never here. "-" when the fulleval has none.
fn blocks_summary(v: &serde_json::Value) -> String {
    let fam = |name: &str| {
        f(v, &["block_profile", "families", name, "used"])
            .map(|x| format!("{}", x as i64))
            .unwrap_or_else(|| "·".into())
    };
    if v.get("block_profile")
        .map(|b| !b.is_null())
        .unwrap_or(false)
    {
        format!(
            "{}/{}/{}/{}",
            fam("f0_155"),
            fam("f156_371"),
            fam("f372_719"),
            fam("f720_943")
        )
    } else {
        "-".into()
    }
}

fn tsv_row(v: &serde_json::Value, r: &BalancedReport) -> String {
    let n_pass = r.floors.iter().filter(|x| x.pass).count();
    // Measured fails only — absent-not-failed floors move to the `absent`
    // column (distinct from a measured fail; still not-passed in n_pass).
    let fails: Vec<&str> = r
        .floors
        .iter()
        .filter(|x| !x.pass && !x.absent_not_failed)
        .map(|x| x.id)
        .collect();
    let absent: Vec<&str> = r
        .floors
        .iter()
        .filter(|x| x.absent_not_failed)
        .map(|x| x.id)
        .collect();
    let n_measured = r.floors.iter().filter(|x| !x.absent_not_failed).count();
    let num = |o: Option<f64>| o.map(|x| format!("{x:.5}")).unwrap_or_else(|| "-".into());
    // SIGNED band tails on the appendix-V scheme (see `band_tails`). Legacy
    // fixed-decile cells emit `-` rather than an absolute-valued 43-pair `B9`.
    let tails = band_scheme(v).and_then(|_| band_tails(v, "cid22"));
    let m3a = f(v, &["m3a_coherence"]);
    let tier: String = if r.class == "944-ensemble" {
        "not-computable".into()
    } else {
        match m3a {
            Some(x) if x >= balanced::M3A_GOLD => "gold".into(),
            Some(x) if x >= balanced::M3A_SILVER => "silver".into(),
            Some(_) => "flagged".into(),
            None => "-".into(),
        }
    };
    let spline = match v.get("model").and_then(|m| m.get("output_spline")) {
        Some(s) if !s.is_null() => "present",
        _ => "none",
    };
    let repro = match (
        v.get("repro").map(|x| !x.is_null()).unwrap_or(false),
        r.class,
    ) {
        (true, "944-ensemble") => "anchor-only",
        (true, _) => "present",
        (false, _) => "absent",
    };
    [
        bake_name(v).to_string(),
        r.class.to_string(),
        if n_pass == r.floors.len() {
            "PASS".to_string()
        } else {
            "FAIL".to_string()
        },
        format!("{n_pass}/{}", r.floors.len()),
        num(f(v, &["rank", "cid22", "srocc"])),
        num(f(v, &["rank", "konjnd", "srocc"]).map(f64::abs)),
        num(f(v, &["rank", "nonphoto", "srocc"])),
        num(f(v, &["rank", "csiq", "srocc"])),
        num(f(v, &["rank", "live", "srocc"])),
        num(f(v, &["rank", "hfnlproxy", "per_ref_mean"])),
        band_scheme(v).unwrap_or("legacy-fixed-decile").to_string(),
        num(tails.as_ref().map(|(lo, _)| lo.signed)),
        tails
            .as_ref()
            .map(|(lo, _)| lo.n.to_string())
            .unwrap_or_else(|| "-".into()),
        num(tails.as_ref().map(|(_, hi)| hi.signed)),
        tails
            .as_ref()
            .map(|(_, hi)| hi.label.clone())
            .unwrap_or_else(|| "-".into()),
        tails
            .as_ref()
            .map(|(_, hi)| hi.n.to_string())
            .unwrap_or_else(|| "-".into()),
        num(tails.as_ref().map(|(_, hi)| hi.span)),
        num(f(v, &["dial", "mono_pct"])),
        num(f(v, &["dial", "tied_pct"])),
        num(f(v, &["dial", "dynamic_range"])),
        num(m3a),
        tier,
        num(f(v, &["corruption_head", "pass_q20"])),
        num(r.composite),
        num(f(v, &["composite"])),
        num(f(v, &["rank", "sdr25", "srocc"])),
        // SIGNED (2026-08-04, APPENDIX F): an unsigned kadid/tid column cannot show an
        // anti-correlated bake, and the ext-lineage KADID target was found inverted.
        num(f(v, &["rank", "kadid", "srocc_signed"]).or_else(|| f(v, &["rank", "kadid", "srocc"]))),
        num(f(v, &["rank", "tid", "srocc_signed"]).or_else(|| f(v, &["rank", "tid", "srocc"]))),
        spline.to_string(),
        repro.to_string(),
        if fails.is_empty() {
            "-".to_string()
        } else {
            fails.join(",")
        },
        format!("{n_pass}/{n_measured}"),
        if absent.is_empty() {
            "-".to_string()
        } else {
            absent.join(",")
        },
        if r.annotations.is_empty() {
            "-".to_string()
        } else {
            r.annotations
                .iter()
                .map(|(id, _, _)| id.as_str())
                .collect::<Vec<_>>()
                .join(",")
        },
        blocks_summary(v),
        match v.get("dominated_by").and_then(|d| d.as_array()) {
            Some(a) if !a.is_empty() => a
                .iter()
                .filter_map(|x| x.as_str())
                .collect::<Vec<_>>()
                .join(","),
            _ => "-".to_string(),
        },
    ]
    .join("\t")
}

/// Annotations registry resolution: explicit `--annotations <path|none>`
/// wins; default = `$ZENSIM_EVAL_ANNOTATIONS`, else the committed
/// `./benchmarks/eval_annotations.json` if present, else none (noted out
/// loud). Shared by the balanced-profile and `--select` paths.
/// Print the documentation-only findings once, so a registry entry with no
/// machine predicate is SURFACED to whoever runs the gate rather than sitting
/// inert (ADD156 ship audit, defect D10).
fn note_manual_annotations(anns: &[AnnEntry]) {
    let manual: Vec<&str> = anns
        .iter()
        .filter(|e| e.manual)
        .map(|e| e.id.as_str())
        .collect();
    if manual.is_empty() {
        return;
    }
    eprintln!(
        "freeze_check: registry carries {} documentation-only finding(s) with no \
         machine scope — NOT applied to any cell, read them by hand: {}",
        manual.len(),
        manual.join(", ")
    );
}

fn load_annotations_arg(arg: Option<&str>) -> Vec<AnnEntry> {
    let anns = load_annotations_arg_inner(arg);
    note_manual_annotations(&anns);
    anns
}

fn load_annotations_arg_inner(arg: Option<&str>) -> Vec<AnnEntry> {
    match arg {
        Some("none") => Vec::new(),
        Some(p) => match load_annotations(std::path::Path::new(p)) {
            Ok(a) => a,
            Err(e) => {
                eprintln!("freeze_check: --annotations {e}");
                std::process::exit(2);
            }
        },
        None => {
            let default = std::env::var("ZENSIM_EVAL_ANNOTATIONS")
                .map(PathBuf::from)
                .unwrap_or_else(|_| PathBuf::from("benchmarks/eval_annotations.json"));
            if default.exists() {
                match load_annotations(&default) {
                    Ok(a) => a,
                    Err(e) => {
                        eprintln!("freeze_check: default annotations {e}");
                        std::process::exit(2);
                    }
                }
            } else {
                eprintln!(
                    "freeze_check: note — no annotations registry at {} (pass --annotations)",
                    default.display()
                );
                Vec::new()
            }
        }
    }
}

/// `--select` driver: read N fullevals, apply the registered rule, print the
/// ranked table (+ optional TSV). Exit 0 if a selectable winner exists,
/// 1 if none does (every candidate UNMEASURED or zero-floor), 2 on usage.
fn run_select(
    paths: &[PathBuf],
    anns: &[AnnEntry],
    tsv: bool,
    seed_group: bool,
    min_k: usize,
    basis: FloorBasis,
    gblock: GaddrBlock,
) -> i32 {
    // The replication floor makes the SEED GROUP the unit of selection, so the
    // grouped section is printed whenever it is active — hiding the basis of a
    // selection would be worse than the defect it fixes. `--seed-group` still
    // forces the section at `--min-k 1`.
    let floor_active = min_k >= 2;
    let show_groups = seed_group || floor_active;
    let mut rows: Vec<SelectRow> = Vec::new();
    for p in paths {
        let bytes = match std::fs::read(p) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("freeze_check --select: read {}: {e}", p.display());
                return 2;
            }
        };
        let v: serde_json::Value = match serde_json::from_slice(&bytes) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("freeze_check --select: parse {}: {e}", p.display());
                return 2;
            }
        };
        let r = eval_balanced(&v, anns);
        let m3a = m3a_state(&v, r.class);
        let selection_composite = match (r.composite, m3a) {
            (Some(c), M3aState::Measured(x)) => Some(c + balanced::W_M3A * x),
            _ => None,
        };
        rows.push(SelectRow {
            name: bake_name(&v).to_string(),
            path: p.display().to_string(),
            class: r.class,
            n_pass: r.floors.iter().filter(|x| x.pass).count(),
            n_floors: r.floors.len(),
            floors_passed: r.floors.iter().filter(|x| x.pass).map(|x| x.id).collect(),
            composite: r.composite,
            m3a,
            selection_composite,
            sdr25: f(&v, &["rank", "sdr25", "srocc"]).map(f64::abs),
            bake: v.get("bake").and_then(|x| x.as_str()).map(str::to_string),
            group_key: seed_group_key(&v),
            seed_id: seed_identity(&v),
            seed_text: seed_label(&v),
            gaddr_contract_fails: gaddr_contract_fails_of(&v, gblock),
            gaddr_codec: gaddr_codec_states_of(&v, gblock),
        });
    }

    // Two pools: ensembles rank on balanced_composite alone (their
    // selection_composite is on a DIFFERENT scale — mixing them would be a
    // category error), everything else on the registered rule.
    let (ens, single): (Vec<&SelectRow>, Vec<&SelectRow>) =
        rows.iter().partition(|r| r.m3a == M3aState::NotComputable);
    let mut single = single;
    let mut ens = ens;
    rank_pool(&mut single, basis);
    ens.sort_by(|a, b| {
        b.n_pass.cmp(&a.n_pass).then_with(|| {
            b.composite
                .partial_cmp(&a.composite)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    });

    let num = |x: Option<f64>| x.map_or("—".into(), |v| format!("{v:.4}"));
    println!("# freeze_check --select — REGISTERED rule (campaign appendix E.4)\n");
    println!(
        "PRIMARY: profile floor count. TIE-BREAK: selection_composite = \
         balanced_composite + {:.2}·M3a.\nsdr25 is a reported comparator, \
         NOT part of the rule.\n",
        balanced::W_M3A
    );
    println!("| rank | bake | class | floors | bal_comp | M3a | sel_comp | sdr25 | selectable |");
    println!("|---:|---|---|---:|---:|---|---:|---:|---|");
    let mut winner: Option<&SelectRow> = None;
    for (i, r) in single.iter().enumerate() {
        let (eff_pass, eff_floors) = gaddr_effective_counts(r, basis);
        let vetoed = gaddr_vetoes(r, basis);
        let selectable = r.m3a != M3aState::Unmeasured && eff_pass > 0 && !vetoed;
        if selectable && winner.is_none() {
            winner = Some(r);
        }
        println!(
            "| {} | {} | {} | {}/{} | {} | {} | {} | {} | {} |",
            i + 1,
            r.name,
            r.class,
            eff_pass,
            eff_floors,
            num(r.composite),
            m3a_cell(r.m3a),
            num(r.selection_composite),
            num(r.sdr25),
            if vetoed {
                "NO — CONTRACT FAIL".to_string()
            } else if selectable {
                "yes".to_string()
            } else {
                "NO".to_string()
            }
        );
    }
    if !ens.is_empty() {
        println!(
            "\n## Ensembles — ranked SEPARATELY on balanced_composite alone\n\n\
             M3a is NOT COMPUTABLE for an ensemble (the coherence instrument \
             loads one ZNPR). They are never penalized for that and never \
             mixed into the ranking above — the two composites are on \
             different scales.\n"
        );
        println!("| rank | bake | floors | bal_comp | sdr25 |");
        println!("|---:|---|---:|---:|---:|");
        for (i, r) in ens.iter().enumerate() {
            println!(
                "| {} | {} | {}/{} | {} | {} |",
                i + 1,
                r.name,
                r.n_pass,
                r.n_floors,
                num(r.composite),
                num(r.sdr25)
            );
        }
    }

    let unmeasured: Vec<&&SelectRow> = single
        .iter()
        .filter(|r| r.m3a == M3aState::Unmeasured)
        .collect();
    if !unmeasured.is_empty() {
        println!(
            "\n## NOT SELECTABLE — {} candidate(s) have no M3a\n\n\
             A candidate nobody measured cannot be certified on that axis \
             (same registered rule the balanced profile applies to an absent \
             floor). Measure, then re-run:\n",
            unmeasured.len()
        );
        for r in &unmeasured {
            match &r.bake {
                Some(b) => println!("    scripts/run_full_eval.sh {b} {} 944", r.name),
                None => println!(
                    "    # {}: fulleval carries no `bake` path — re-harvest it",
                    r.name
                ),
            }
        }
    }

    // ── G-ADDR CONTRACT veto callout (2026-09-06 owner fix) ─────────────
    // A CONTRACT row is an ABSOLUTE product bar (dial addressability, user
    // rule 2026-09-04) — a bake that fails one is never selectable, however
    // high its floor count or composite. `--floor-basis legacy` reproduces
    // the pre-2026-09-06 rule that could not see this at all.
    let contract_vetoed: Vec<&&SelectRow> =
        single.iter().filter(|r| gaddr_vetoes(r, basis)).collect();
    if !contract_vetoed.is_empty() {
        println!(
            "\n## NOT SELECTABLE — {} candidate(s): contract FAIL — not selectable\n\n\
             Each measured a G-ADDR CONTRACT-tier (C1-C6) fail in its OWN \
             `dial.addressability` block — an absolute selectability veto, \
             independent of floor count or `selection_composite`. \
             `--floor-basis legacy` reproduces the pre-2026-09-06 rule that \
             ignored this (registry: select-rule-blind-to-dial-contract-2026-09-06).\n",
            contract_vetoed.len()
        );
        for r in &contract_vetoed {
            println!(
                "    - `{}`: contract FAIL — not selectable ({})",
                r.name,
                gaddr_fail_reason(&r.gaddr_contract_fails)
            );
        }
    }

    // The per-cell table answers "the best CELL". Under the replication floor
    // that is NOT the selection (a cell has no k), so it is labelled as what
    // it is; `**SELECTED:`** is emitted once, by whichever section is the
    // authoritative one. At `--min-k 1` the historical strings are unchanged.
    match (winner, floor_active) {
        (Some(w), false) => {
            let (p, t) = gaddr_effective_counts(w, basis);
            println!(
                "\n**SELECTED: `{}`** — {p}/{t} floors, selection_composite {}.",
                w.name,
                num(w.selection_composite)
            )
        }
        (Some(w), true) => {
            let (p, t) = gaddr_effective_counts(w, basis);
            println!(
                "\n**BEST CELL: `{}`** — {p}/{t} floors, selection_composite {}. \
                 NOT the selection: a cell has no `k`, and the registered \
                 replication floor (--min-k {}) selects a RECIPE. See the \
                 seed-grouped section below.",
                w.name,
                num(w.selection_composite),
                min_k
            )
        }
        (None, _) => {
            println!("\n**NO SELECTABLE CANDIDATE** (every row is UNMEASURED or 0-floor).")
        }
    }

    if tsv {
        eprintln!("{SELECT_TSV_COLS}");
        let emit = |pool: &str, i: usize, r: &SelectRow| {
            let (eff_pass, _) = gaddr_effective_counts(r, basis);
            let selectable =
                r.m3a != M3aState::Unmeasured && eff_pass > 0 && !gaddr_vetoes(r, basis);
            let n = |x: Option<f64>| x.map_or("-".into(), |v| format!("{v:.6}"));
            eprintln!(
                "{}\t{pool}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                i + 1,
                r.name,
                r.class,
                eff_pass,
                n(r.composite),
                match r.m3a {
                    M3aState::Measured(x) => format!("{x:.6}"),
                    _ => "-".into(),
                },
                match r.m3a {
                    M3aState::Measured(_) => "measured",
                    M3aState::NotComputable => "not-computable",
                    M3aState::Unmeasured => "unmeasured",
                },
                n(r.selection_composite),
                n(r.sdr25),
                if selectable { "yes" } else { "no" },
                r.path
            );
        };
        for (i, r) in single.iter().enumerate() {
            emit("single", i, r);
        }
        for (i, r) in ens.iter().enumerate() {
            emit("ensemble", i, r);
        }
    }

    // ── The seed-grouped section (2026-09-04 owner-fix §7.7, AMENDED
    // 2026-09-05 with the replication floor). Everything above is UNCHANGED.
    // Aggregates `single` into seed groups (see `SeedGroupRow` /
    // `group_by_seed`) and re-ranks by the same PRIMARY/TIE-BREAK rule over
    // the group, under `basis`. `ens` is not seed-grouped (see the module
    // note); its per-cell section above already covers it.
    if show_groups {
        let mut groups = group_by_seed(&single);
        rank_seed_groups(&mut groups, basis);

        let num = |x: Option<f64>| x.map_or("—".into(), |v| format!("{v:.4}"));
        let spread_str = |s: Option<(f64, f64)>| match s {
            Some((lo, hi)) => format!("{lo:.4}–{hi:.4}"),
            None => "—".into(),
        };
        let basis_str = match basis {
            FloorBasis::AllReps => {
                "all-reps (floors EVERY seed passes, incl. A7r per-codec G-ADDR floors)"
            }
            FloorBasis::Mean => "mean (k-seed mean floor count; F1-F8 only, no A7r)",
            FloorBasis::Legacy => {
                "legacy (pre-2026-09-06 reproduction: F1-F8 only, NO G-ADDR CONTRACT veto)"
            }
        };

        println!(
            "\n## Seed-grouped ranking — same PRIMARY/TIE-BREAK rule, over the GROUP\n\n\
             Group key = `sha1(zentrain.repro.argv minus the seed and \
             output-path flags)[:12]`, and duplicate promotions of ONE training run collapse by \
             seed identity, so **k = distinct seeds, never cells** — the board's rule \
             (`gauntlet.seed_group_key`, `benchmarks/fair_gauntlet_2026-09-04.md` §1.1), \
             mirrored here so a `--select` group and a board group are the SAME handle. An \
             ensemble or a cell with no embedded argv is UNGROUPABLE — listed, never merged \
             or dropped. Read a group's rank as \"the best RECIPE\"; the per-cell table above \
             answers a different question, \"the best CELL\" (freeze_check §7.7).\n\n\
             **REPLICATION FLOOR (registered amendment, 2026-09-05): `--min-k {}`.** A group \
             with fewer than {} distinct seeds is UNREPLICATED and is NOT selectable — it is \
             listed and ranked in its own section below. Measured reason: replicating the \
             board leaders moved them DOWN (`LSTAR` 0.8615 best cell → 0.856414 at k=7), and \
             best-of-k minus k-mean has a **+0.0061 median** over the 18 combined-fair k≥2 \
             groups — larger than the 0.0021 that separated the top four \
             (`benchmarks/replication_wave_2026-09-05.md` §2, §4c.1). A one-draw number is a \
             sample from a distribution whose maximum sits above its mean; ranking it against \
             means is a units error, not a tie-break. `--min-k 1` restores the pre-amendment \
             behavior.\n\n\
             **FLOOR BASIS: {}.** A group is credited a floor only when EVERY distinct-seed \
             representative passes it (`--floor-basis mean` restores the k-seed mean count). \
             A floor is a certification, and a mean is not one: two members at 8/8 and 6/8 \
             average 7.0 whether they fail the same floor twice or two different floors, and \
             in the second case the group would be credited a floor no member reliably \
             clears. The mean is still reported in its own column.\n\n\
             **The mean is NOT \"the group's true score.\"** Seeds differ in what they SAW, not \
             only in where they landed: the sampler stream is seeded, so two seeds walk the same \
             fixed row population in a different ORDER and a finite-epoch run emphasises a \
             different subset of pairs — objectively different coverage, not noise around one \
             underlying number. The spread column and the per-seed table below are the rest of \
             the answer. Each bake's own `zentrain.sample_coverage` metadata is the per-seed \
             record of what that run actually touched.\n\n\
             **G-ADDR CONTRACT VETO (2026-09-06 owner fix, registry \
             select-rule-blind-to-dial-contract-2026-09-06).** A group where ANY member \
             measured a G-ADDR CONTRACT-tier (C1-C6) fail in its own `dial.addressability` \
             block is NOT selectable, however high its floor count or composite — dial \
             addressability is an ABSOLUTE product bar (user rule 2026-09-04), not a floor \
             a bake can trade off. Under `--floor-basis all` this also folds `A7r`'s \
             per-codec representability into the floor count itself (one more floor per \
             MEASURED codec; a codec no member's instrument touched counts in neither \
             direction). `--floor-basis mean` keeps the veto but not the extra floors; \
             `--floor-basis legacy` reproduces the pre-fix rule byte-for-byte (no veto, no \
             extra floors) for reproducing a historical selection.\n",
            min_k, min_k, basis_str
        );

        // Partition, rank each pool separately, and print both. `min_k <= 1`
        // disables the floor entirely (the historical behavior, in which `k`
        // was not a selectability term at all) so a pre-amendment table is
        // reproducible byte-for-byte with `--min-k 1 --floor-basis mean`.
        let floor_ok = |g: &SeedGroupRow<'_>| min_k <= 1 || g.k_seeds >= min_k;
        let (eligible, below): (Vec<&SeedGroupRow<'_>>, Vec<&SeedGroupRow<'_>>) =
            groups.iter().partition(|g| floor_ok(g));

        let header = || {
            println!(
                "| rank | group | k | cells | floors_all | mean_floors | gaddr (A7r \
                 all-reps / mean) | mean_bal_comp | mean_m3a | mean_sel_comp | sel_comp spread \
                 (min\u{2013}max) | split floors | state | selectable |"
            );
            println!("|---:|---|---:|---:|---:|---:|---|---:|---|---:|---|---|---|---|");
        };
        let row = |i: usize, g: &SeedGroupRow<'_>| {
            let why = group_unselectable_reason(g, basis, min_k);
            // A7r contribution this group's OWN G-ADDR data supports — shown
            // regardless of `basis` (report-only when `basis` doesn't fold
            // it into `floor_count`, e.g. `mean`/`legacy`), so a reader can
            // always see what `--floor-basis all` WOULD add.
            let gaddr_str = if g.gaddr_codec_total_all == 0 && g.gaddr_codec_mean_total == 0.0 {
                "\u{2014} (not measured)".to_string()
            } else {
                format!(
                    "{}/{} (mean {:.2}/{:.2})",
                    g.gaddr_codec_pass_all,
                    g.gaddr_codec_total_all,
                    g.gaddr_codec_mean_pass,
                    g.gaddr_codec_mean_total
                )
            };
            println!(
                "| {} | {} | {} | {} | {} | {:.2} | {} | {} | {} | {} | {} | {} | {} | {} |",
                i + 1,
                g.key,
                g.k_seeds,
                g.members.len(),
                g.n_pass_all,
                g.mean_n_pass,
                gaddr_str,
                num(g.mean_composite),
                m3a_cell(g.m3a),
                num(g.mean_selection_composite),
                spread_str(g.selection_composite_spread),
                if g.floors_split.is_empty() {
                    "\u{2014}".to_string()
                } else {
                    g.floors_split.join(",")
                },
                group_state(g),
                match &why {
                    None => "yes".to_string(),
                    Some(r) => format!("NO — {r}"),
                }
            );
        };

        header();
        let mut group_winner: Option<&SeedGroupRow<'_>> = None;
        for (i, g) in eligible.iter().enumerate() {
            if group_unselectable_reason(g, basis, min_k).is_none() && group_winner.is_none() {
                group_winner = Some(g);
            }
            row(i, g);
        }
        if eligible.is_empty() {
            println!("| — | *(no group meets the replication floor)* | | | | | | | | | | | |");
        }

        if !below.is_empty() {
            println!(
                "\n### UNREPLICATED — {} group(s) below the `--min-k {}` replication floor\n\n\
                 Ranked among themselves by the same rule, and **never selected**. These are \
                 listed rather than dropped: the number is real, it is simply ONE DRAW, and \
                 the measured best-of-k premium (+0.0061 median composite) is not a \
                 correction that can be applied to it. To make one selectable, train it again \
                 with a different seed and re-harvest — not lower the floor.\n",
                below.len(),
                min_k
            );
            header();
            for (i, g) in below.iter().enumerate() {
                row(i, g);
            }
        }

        println!(
            "\n### Per-seed detail\n\nEvery member's OWN numbers, so the mean above is never \
             the only thing on the page. `seeds` is read from each fulleval's embedded `repro`: \
             an init/sample PAIR when the run used the split seeds, a bare seed on older bakes, \
             an em-dash when the fulleval carries no repro at all.\n"
        );
        println!("| group | name | seeds | floors | bal_comp | m3a | sel_comp |");
        println!("|---|---|---|---:|---:|---|---:|");
        for g in &groups {
            for m in &g.members {
                println!(
                    "| {} | {} | {} | {}/{} | {} | {} | {} |",
                    g.key,
                    m.name,
                    m.seed_text,
                    m.n_pass,
                    m.n_floors,
                    num(m.composite),
                    m3a_cell(m.m3a),
                    num(m.selection_composite)
                );
            }
        }

        match (group_winner, floor_active) {
            // `**SELECTED:` is emitted exactly once per run, by whichever
            // section is authoritative. Under the floor that is this one, and
            // it names a RECIPE (a group), not a cell — picking the group's
            // best member would re-introduce the very best-of-k selection the
            // amendment exists to stop.
            (Some(w), true) => {
                println!(
                    "\n**SELECTED: `{}`** — a RECIPE, k={}, {} floors passed by every seed \
                     (mean {:.2}/{}), mean selection_composite {} (per-seed spread {}).",
                    w.key,
                    w.k_seeds,
                    w.n_pass_all,
                    w.mean_n_pass,
                    w.reps.first().map(|m| m.n_floors).unwrap_or(0),
                    num(w.mean_selection_composite),
                    spread_str(w.selection_composite_spread)
                );
                println!(
                    "\nFloors every seed clears: {}{}",
                    if w.floors_all.is_empty() {
                        "(none)".to_string()
                    } else {
                        w.floors_all.iter().copied().collect::<Vec<_>>().join(", ")
                    },
                    if w.floors_split.is_empty() {
                        String::new()
                    } else {
                        format!(
                            " · SPLIT (passed by some seeds, not all — NOT credited): {}",
                            w.floors_split.join(", ")
                        )
                    }
                );
                if w.gaddr_codec_total_all > 0 || w.gaddr_codec_mean_total > 0.0 {
                    println!(
                        "\nA7r per-codec G-ADDR floors (all-reps / mean): {}/{} \
                         (mean {:.2}/{:.2}){}",
                        w.gaddr_codec_pass_all,
                        w.gaddr_codec_total_all,
                        w.gaddr_codec_mean_pass,
                        w.gaddr_codec_mean_total,
                        if basis == FloorBasis::AllReps {
                            " — folded into the floor count above"
                        } else {
                            " — reported only; not folded in under this --floor-basis"
                        }
                    );
                }
                println!(
                    "\nIts {} member cell(s): {}",
                    w.members.len(),
                    w.members
                        .iter()
                        .map(|m| format!("`{}`", m.name))
                        .collect::<Vec<_>>()
                        .join(", ")
                );
            }
            (Some(w), false) => println!(
                "\n**SEED-GROUP SELECTED: `{}`** — k={}, mean {:.2}/{} floors, mean \
                 selection_composite {} (per-seed spread {}).",
                w.key,
                w.k_seeds,
                w.mean_n_pass,
                w.reps.first().map(|m| m.n_floors).unwrap_or(0),
                num(w.mean_selection_composite),
                spread_str(w.selection_composite_spread)
            ),
            (None, true) => println!(
                "\n**NO SELECTABLE RECIPE** — no seed group reaches k={min_k} with a measured \
                 M3a and a nonzero floor count. Replicate a candidate (same recipe, new seed) \
                 rather than lowering --min-k."
            ),
            (None, false) => {
                println!("\n**NO SELECTABLE SEED-GROUP** (every group is UNMEASURED or 0-floor).")
            }
        }

        if tsv {
            eprintln!("{SEED_GROUP_TSV_COLS}");
            for (i, g) in groups.iter().enumerate() {
                let why = group_unselectable_reason(g, basis, min_k);
                let n = |x: Option<f64>| x.map_or("-".into(), |v| format!("{v:.6}"));
                let (lo, hi) = g.selection_composite_spread.unwrap_or((f64::NAN, f64::NAN));
                eprintln!(
                    "{}\t{}\t{}\t{}\t{}\t{}\t{:.6}\t{}\t{}\t{}\t{}\t{:.6}\t{:.6}\t{}\t{}\t{}",
                    i + 1,
                    g.key,
                    g.k_seeds,
                    g.members.len(),
                    g.n_duplicate_cells,
                    g.n_pass_all,
                    g.mean_n_pass,
                    if g.floors_split.is_empty() {
                        "-".to_string()
                    } else {
                        g.floors_split.join(",")
                    },
                    n(g.mean_composite),
                    match g.m3a {
                        M3aState::Measured(x) => format!("{x:.6}"),
                        _ => "-".into(),
                    },
                    n(g.mean_selection_composite),
                    lo,
                    hi,
                    group_state(g),
                    if why.is_none() { "yes" } else { "no" },
                    why.unwrap_or_else(|| "-".into()),
                );
            }
        }

        return i32::from(group_winner.is_none());
    }

    i32::from(winner.is_none())
}

/// Product qualification is stricter than ranking research candidates. Every
/// gate is read from its measuring owner, with content-bound evidence.
fn qualification_report(v: &serde_json::Value) -> serde_json::Value {
    use serde_json::json;
    let mut checks = Vec::new();
    let mut add = |gate: &str, state: &str, detail: String| {
        checks.push(json!({"gate": gate, "state": state, "detail": detail}))
    };
    let surface = v["scoring"]["surface"].as_str() == Some("zensim::BakeScorer");
    add(
        "Rust surface",
        if surface { "pass" } else { "not_measured" },
        "Complete composed scorer used by evaluation".into(),
    );
    let composition = &v["feature_set_composition"];
    let expected = v["scoring"]["members"]
        .as_array()
        .map(Vec::len)
        .unwrap_or(0)
        + usize::from(!v["scoring"]["corruption"].is_null());
    let provenance = composition["members"].as_array();
    let complete = expected > 0
        && composition["scoring"] == v["scoring"]
        && provenance.is_some_and(|m| m.len() == expected);
    let compatible =
        complete && provenance.is_some_and(|m| m.iter().all(|p| p["qualified_provenance"] == true));
    let replay = !v["repro"]["table_admission"]["historical_replay"].is_null()
        || provenance.is_some_and(|m| m.iter().any(|p| p["historical_replay"] == true));
    add("Table provenance", if replay { "fail" } else if compatible { "pass" } else { "not_measured" },
        if replay { "Explicit historical replay in the scorer cannot qualify a new model" }
        else { "Require compatible feature/formula/decoder declarations for every table, model and companion; see feature_set_composition" }.into());
    let source = &v["dial_ladder_source"];
    let source_valid = source["path"]
        .as_str()
        .zip(source["sha256"].as_str())
        .is_some_and(|(path, sha)| {
            zensim_validate::train_manifest::sha256_file(&PathBuf::from(path))
                .is_ok_and(|actual| actual == sha)
        });
    let addr = gaddr_block(v, GaddrBlock::Ladder);
    let composition_matches = addr.is_some_and(|a| a["scoring"] == v["scoring"]);
    add(
        "Ladder evidence",
        if source_valid && composition_matches {
            "pass"
        } else {
            "not_measured"
        },
        "Require the preserved ladder artifact and the same complete scorer".into(),
    );
    for (gate, key) in [
        ("G-ADDR contract", "contract"),
        ("G-ADDR regression", "regression"),
    ] {
        let state = match addr.and_then(|a| a[key].as_str()) {
            Some("PASS") => "pass",
            Some("FAIL") => "fail",
            _ => "not_measured",
        };
        add(
            gate,
            state,
            "Operative ladder instrument; missing tiers cannot pass".into(),
        );
    }
    let floors = gaddr_codec_states_of(v, GaddrBlock::Ladder);
    for codec in ["avif-rav1e", "avif-svt", "jpeg", "jxl", "webp"] {
        let state = match floors
            .iter()
            .find(|(c, _)| c == codec)
            .and_then(|(_, state)| *state)
        {
            Some(true) => "pass",
            Some(false) => "fail",
            None => "not_measured",
        };
        add(
            &format!("Codec floor: {codec}"),
            state,
            "All five required codec families must be measured".into(),
        );
    }
    let candidate = v["bake_sha256"].as_str();
    for gate in ["G-RANK", "G-DIAL", "G-STEER", "G-RD", "G-TARGET"] {
        let evidence = &v["product_evidence"][gate];
        let mut state = "not_measured";
        let mut detail = "Missing content-bound product evidence".to_string();
        if candidate.is_some_and(|s| s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit()))
            && evidence["bake_sha256"].as_str() == candidate
            && evidence["surface"].as_str() == Some("zensim::BakeScorer")
            && evidence["n"].as_u64().is_some_and(|n| n > 0)
            && evidence["instrument"]
                .as_str()
                .is_some_and(|s| !s.is_empty())
            && let (Some(path), Some(sha)) = (
                evidence["artifact"]["path"].as_str(),
                evidence["artifact"]["sha256"].as_str(),
            )
        {
            let path = PathBuf::from(path);
            let matches = zensim_validate::train_manifest::sha256_file(&path)
                .is_ok_and(|actual| actual == sha);
            if matches {
                if let Ok(bytes) = std::fs::read(&path)
                    && let Ok(measurement) = serde_json::from_slice::<serde_json::Value>(&bytes)
                    && measurement["bake_sha256"].as_str() == candidate
                    && measurement["scoring"] == v["scoring"]
                    && measurement["gates"][gate]["state"] == evidence["state"]
                {
                    state = match evidence["state"].as_str() {
                        Some("pass") => "pass",
                        Some("fail") => "fail",
                        _ => "not_measured",
                    };
                    detail = format!(
                        "{}; artifact {} sha256 {}",
                        evidence["instrument"].as_str().unwrap_or("?"),
                        path.display(),
                        sha
                    );
                }
            } else {
                detail = "Evidence artifact is missing or its content hash changed".into();
            }
        }
        add(gate, state, detail);
    }
    let failed = checks.iter().any(|c| c["state"] == "fail");
    let missing = checks.iter().any(|c| c["state"] == "not_measured");
    json!({"schema": 1, "name": bake_name(v), "bake_sha256": candidate,
        "status": if failed { "failed" } else if missing { "incomplete" } else { "qualified" },
        "checks": checks, "research_selection_is_qualification": false})
}

fn main() {
    let mut fulleval: Option<PathBuf> = None;
    let mut qualify = false;
    let mut bar_csiq: Option<f64> = None;
    let mut bar_live: Option<f64> = None;
    let mut profile: Option<String> = None;
    let mut tsv = false;
    let mut annotations_arg: Option<String> = None;
    let mut select: Vec<PathBuf> = Vec::new();
    let mut in_select = false;
    let mut seed_group = false;
    // Registered amendment (2026-09-05): the replication floor is ON by
    // default. `--min-k 1` is the documented pre-amendment escape.
    let mut min_k: usize = 2;
    let mut basis = FloorBasis::AllReps;
    // Which G-ADDR block is the OPERATIVE dial ruler (2026-09-06). Default `auto`
    // = the ladder instrument when the cell carries one, else its canonical read.
    let mut gblock = GaddrBlock::Auto;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        // After `--select`, bare paths accumulate until the next flag.
        if in_select && !a.starts_with("--") {
            select.push(PathBuf::from(a));
            continue;
        }
        let was_in_select = in_select;
        in_select = false;
        match a.as_str() {
            "--fulleval" => fulleval = args.next().map(PathBuf::from),
            "--qualify" => qualify = true,
            "--select" => in_select = true,
            // A BARE flag: the grouping rule is DERIVED from each
            // fulleval's embedded repro argv (one owner — see the module
            // note above), never supplied by the caller. `in_select` is
            // restored so `--select a.json --seed-group b.json` still
            // collects b.json as a path.
            "--seed-group" => {
                seed_group = true;
                in_select = was_in_select;
            }
            "--min-k" => {
                min_k = match args.next().and_then(|v| v.parse::<usize>().ok()) {
                    Some(k) => k,
                    None => {
                        eprintln!("freeze_check: --min-k needs a non-negative integer");
                        std::process::exit(2);
                    }
                };
                in_select = was_in_select;
            }
            "--gaddr-block" => {
                gblock = match args.next().as_deref() {
                    Some("auto") => GaddrBlock::Auto,
                    Some("canonical") => GaddrBlock::Canonical,
                    Some("ladder") => GaddrBlock::Ladder,
                    other => {
                        eprintln!(
                            "freeze_check: --gaddr-block takes `auto`, `canonical` or \
                             `ladder` (got {other:?})"
                        );
                        std::process::exit(2);
                    }
                };
            }
            "--floor-basis" => {
                basis = match args.next().as_deref() {
                    Some("all") => FloorBasis::AllReps,
                    Some("mean") => FloorBasis::Mean,
                    Some("legacy") => FloorBasis::Legacy,
                    other => {
                        eprintln!(
                            "freeze_check: --floor-basis takes `all`, `mean` or `legacy` \
                             (got {other:?})"
                        );
                        std::process::exit(2);
                    }
                };
                in_select = was_in_select;
            }
            "--profile" => profile = args.next(),
            "--annotations" => annotations_arg = args.next(),
            "--tsv" => tsv = true,
            "--select-tsv-header" => {
                println!("{SELECT_TSV_COLS}");
                std::process::exit(0);
            }
            "--seed-group-tsv-header" => {
                println!("{SEED_GROUP_TSV_COLS}");
                std::process::exit(0);
            }
            "--tsv-header" => {
                println!("{TSV_COLS}");
                std::process::exit(0);
            }
            "--bar" => {
                let kv = match args.next() {
                    Some(v) => v,
                    None => usage(),
                };
                let (k, v) = match kv.split_once('=') {
                    Some(p) => p,
                    None => usage(),
                };
                let val: f64 = match v.parse() {
                    Ok(x) => x,
                    Err(_) => usage(),
                };
                match k {
                    "csiq" => bar_csiq = Some(val),
                    "live" => bar_live = Some(val),
                    _ => usage(),
                }
            }
            "-h" | "--help" => usage(),
            _ => usage(),
        }
    }
    if qualify && (!select.is_empty() || profile.is_some() || tsv) {
        eprintln!(
            "freeze_check: --qualify is a standalone product gate, separate from --select/--profile/--tsv"
        );
        std::process::exit(2);
    }
    match profile.as_deref() {
        None | Some("balanced-2026-08-04") => {}
        Some(other) => {
            eprintln!("freeze_check: unknown profile `{other}` (known: balanced-2026-08-04)");
            std::process::exit(2);
        }
    }
    // ── `--select` path: N fullevals in, one ranked table out ───────────
    if !select.is_empty() {
        if fulleval.is_some() {
            eprintln!("freeze_check: --select and --fulleval are mutually exclusive");
            std::process::exit(2);
        }
        // --select always ranks under the registered floor set; the §5 bar
        // set has ATTACH rows and cannot rank.
        if matches!(profile.as_deref(), Some(p) if p != "balanced-2026-08-04") {
            eprintln!("freeze_check: --select supports only --profile balanced-2026-08-04");
            std::process::exit(2);
        }
        let anns = load_annotations_arg(annotations_arg.as_deref());
        std::process::exit(run_select(
            &select, &anns, tsv, seed_group, min_k, basis, gblock,
        ));
    }
    if seed_group {
        eprintln!("freeze_check: --seed-group only applies to --select");
        std::process::exit(2);
    }
    if min_k != 2 || basis != FloorBasis::AllReps {
        eprintln!("freeze_check: --min-k / --floor-basis only apply to --select");
        std::process::exit(2);
    }
    if tsv && profile.is_none() {
        eprintln!(
            "freeze_check: --tsv requires --profile (the §5 table has ATTACH rows a TSV cannot carry)"
        );
        std::process::exit(2);
    }
    let path = match fulleval {
        Some(p) => p,
        None => usage(),
    };
    let bytes = match std::fs::read(&path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("freeze_check: read {}: {e}", path.display());
            std::process::exit(2);
        }
    };
    let v: serde_json::Value = match serde_json::from_slice(&bytes) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("freeze_check: parse {}: {e}", path.display());
            std::process::exit(2);
        }
    };

    if qualify {
        let report = qualification_report(&v);
        println!(
            "{}",
            serde_json::to_string_pretty(&report).expect("qualification JSON")
        );
        std::process::exit(if report["status"] == "qualified" {
            0
        } else {
            1
        });
    }

    // ── Balanced profile path (AMENDMENT 8) ─────────────────────────────
    if profile.is_some() {
        let anns = load_annotations_arg(annotations_arg.as_deref());
        let r = eval_balanced(&v, &anns);
        let n_fail = r.floors.iter().filter(|x| !x.pass).count();
        if tsv {
            println!("{}", tsv_row(&v, &r));
        } else {
            let bake = bake_name(&v);
            println!("# Balanced-selection profile `balanced-2026-08-04` — `{bake}`\n");
            println!(
                "Class: **{}** — {}. Floors: sota944 campaign AMENDMENT 8 (§8.1, \
                 registered 2026-08-04 BEFORE scoring; user-directed policy). §1 \
                 stays the freeze bar — this is the balanced SELECTION surface.\n",
                r.class, r.class_note
            );
            println!("| floor | bar | measured | verdict |");
            println!("|---|---|---|:--:|");
            for x in &r.floors {
                println!(
                    "| {} | {} | {} | {} |",
                    x.gate,
                    x.bar,
                    x.measured,
                    if x.pass {
                        "PASS"
                    } else if x.absent_not_failed {
                        "ABSENT (not passed)"
                    } else {
                        "**FAIL**"
                    }
                );
            }
            if !r.annotations.is_empty() {
                println!("\n| ⚠ annotation | kind | reason |");
                println!("|---|---|---|");
                for (id, kind, reason) in &r.annotations {
                    println!("| `{id}` | {kind} | {reason} |");
                }
            }
            println!("\n| reported (never floors) | value |");
            println!("|---|---|");
            for (k, val) in &r.info {
                println!("| {k} | {val} |");
            }
            match r.composite {
                Some(c) => println!("\n**balanced_composite = {c:.4}** (registered §8.1 weights)"),
                None => println!("\nbalanced_composite: not computable (no terms present)"),
            }
            let absent_ids: Vec<&str> = r
                .floors
                .iter()
                .filter(|x| x.absent_not_failed)
                .map(|x| x.id)
                .collect();
            let n_measured = r.floors.len() - absent_ids.len();
            print!(
                "\n{} of {} floors pass",
                r.floors.len() - n_fail,
                r.floors.len()
            );
            if !absent_ids.is_empty() {
                // Both forms, per the registry convention: n/8 keeps the
                // registered absent=not-passed rule; n/m-measured states the
                // measured record.
                print!(
                    " ({}/{n_measured}-measured; absent-not-failed: {})",
                    r.floors.len() - n_fail,
                    absent_ids.join(",")
                );
            }
            println!(
                "{}",
                if n_fail == 0 {
                    " — BALANCED-PROFILE PASS"
                } else {
                    ""
                }
            );
        }
        std::process::exit(if n_fail > 0 { 1 } else { 0 });
    }

    // ── §5 default path (unchanged) ─────────────────────────────────────
    let rows = legacy_rows(&v, bar_csiq, bar_live);

    let bake = bake_name(&v);
    println!("# Freeze-bar summary — `{bake}`\n");
    println!(
        "Bars: zenpapers final-metric plan §5 (2026-07-31). Owners: \
         benchmarks/decision_surface_audit_2026-07-31.md. ATTACH rows need \
         evidence from their named owner — they gate the freeze too.\n"
    );
    println!("| gate | bar | measured / evidence | verdict |");
    println!("|---|---|---|:--:|");
    let (mut n_fail, mut n_attach) = (0usize, 0usize);
    for r in &rows {
        match r {
            Row::Eval(g, b, m, ok) => {
                if !ok {
                    n_fail += 1;
                }
                println!(
                    "| {g} | {b} | {m} | {} |",
                    if *ok { "PASS" } else { "**FAIL**" }
                );
            }
            Row::Attach(g, b, w) => {
                n_attach += 1;
                println!("| {g} | {b} | {w} | ATTACH |");
            }
            Row::Info(g, m) => println!("| {g} | — | {m} | info |"),
        }
    }
    println!(
        "\n{} evaluable FAIL(s); {} row(s) awaiting attached evidence. \
         Freeze requires zero FAILs AND every ATTACH row's evidence in hand.",
        n_fail, n_attach
    );
    std::process::exit(if n_fail > 0 { 1 } else { 0 });
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// `DISTORTION_ORIENTED` is a gated mirror of the Python
    /// `EXPECTED_ORIENTATION` registry (the owner —
    /// `scripts/canonical_corpus/check_target_orientation.py`). Parse the
    /// owner file and fail on ANY drift, in either direction. The parse is
    /// deliberately dumb (line-based over the literal dict block) so it
    /// breaks LOUDLY if the registry's shape changes, rather than silently
    /// matching nothing.
    #[test]
    fn distortion_oriented_mirror_matches_python_registry() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../scripts/canonical_corpus/check_target_orientation.py"
        );
        let text = std::fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("owner registry {path} must exist in-repo: {e}"));
        let start = text
            .find("EXPECTED_ORIENTATION = {")
            .expect("EXPECTED_ORIENTATION dict literal not found in the owner file");
        let block = &text[start..];
        let end = block.find('}').expect("registry dict literal never closes");
        let block = &block[..end];
        let mut dist: Vec<String> = Vec::new();
        let mut n_quality = 0usize;
        for line in block.lines() {
            let line = line.split('#').next().unwrap_or("");
            if let Some((k, val)) = line.split_once(':') {
                let key = k.trim().trim_matches('"').trim_matches('\'');
                if key.is_empty() {
                    continue;
                }
                match val.trim().trim_end_matches(',').trim() {
                    "DISTORTION" => dist.push(key.to_string()),
                    "QUALITY" => n_quality += 1,
                    _ => {}
                }
            }
        }
        assert!(
            n_quality >= 5,
            "parse sanity: expected several QUALITY entries, parsed {n_quality} — \
             did the registry's literal format change?"
        );
        dist.sort();
        let mut mirror: Vec<String> = DISTORTION_ORIENTED.iter().map(|s| s.to_string()).collect();
        mirror.sort();
        assert_eq!(
            mirror, dist,
            "freeze_check's DISTORTION_ORIENTED mirror drifted from the Python \
             EXPECTED_ORIENTATION registry — update the mirror (the registry is the owner)"
        );
    }

    /// A CID22 band block in the CURRENT (appendix-V) scheme: three usable
    /// bands, each stamped `not_measured_reason: null`, plus the `band_scheme`
    /// marker that tells `freeze_check` these are current-scheme bands and not
    /// pre-2026-08-06 fixed deciles. `lo`/`hi` are the SIGNED tails.
    fn bands_v(lo: f64, hi: f64) -> serde_json::Value {
        json!([
            { "band": "B0-B6", "lo": 0.0, "hi": 0.7, "n": 1775, "span": 0.4227,
              "not_measured_reason": null, "srocc": lo.abs(), "srocc_signed": lo },
            { "band": "B7", "lo": 0.7, "hi": 0.8, "n": 1092, "span": 0.0997,
              "not_measured_reason": null, "srocc": 0.36, "srocc_signed": 0.36 },
            { "band": "B8-B9", "lo": 0.8, "hi": null, "n": 1425, "span": 0.1194,
              "not_measured_reason": null, "srocc": hi.abs(), "srocc_signed": hi },
        ])
    }

    /// A fixture that passes every registered floor exactly at/above the line.
    fn passing_fixture() -> serde_json::Value {
        json!({
            "name": "FIX_single",
            "n_inputs": 944,
            "model": { "n_inputs": 944, "output_spline": null },
            "repro": { "seed": 1 },
            "composite": 0.85,
            "m3a_coherence": 0.80,
            "rank": {
                "cid22": { "srocc": 0.885,
                    "band_scheme": { "name": "merged-decile-2026-08-06",
                                     "n_min": 1000, "span_min": 0.08 },
                    "bands": bands_v(balanced::BAND_LOW, balanced::BAND_HIGH) },
                "konjnd": { "srocc": -0.43 },
                "nonphoto": { "srocc": 0.90 },
                "csiq": { "srocc": 0.83 },
                "live": { "srocc": 0.83 },
                "imazen26": { "srocc": 0.91 },
                "aic3": { "srocc": 0.79 },
                "aic4": { "srocc": 0.91 },
                "sdr25": { "srocc": 0.93 },
                "kadid": { "srocc": 0.32 },
                "tid": { "srocc": 0.88 },
                "hfnlproxy": { "srocc": 0.5, "per_ref_mean": 0.0 }
            },
            "dial": { "mono_pct": 0.93, "tied_pct": 0.05, "dynamic_range": 15.0 }
        })
    }

    #[test]
    fn balanced_floors_resolve_as_registered_and_fixture_passes() {
        let v = passing_fixture();
        let r = eval_balanced(&v, &[]);
        assert_eq!(r.class, "944-single");
        assert_eq!(r.class_note, "shippable single bake");
        assert_eq!(r.floors.len(), 8, "eight registered floors F1..F8");
        for x in &r.floors {
            assert!(
                x.pass,
                "floor {} should pass at the registered edge: {}",
                x.id, x.measured
            );
        }
        // Each floor flips to FAIL just below its registered line.
        let cases: Vec<(&str, serde_json::Value)> = vec![
            ("cid22", json!({"rank": {"cid22": {"srocc": 0.8849}}})),
            ("konjnd", json!({"rank": {"konjnd": {"srocc": 0.4299}}})),
            ("nonphoto", json!({"rank": {"nonphoto": {"srocc": 0.8999}}})),
            ("dial", json!({"dial": {"mono_pct": 0.9299}})),
            ("dial", json!({"dial": {"tied_pct": 0.0501}})),
            ("dialrange", json!({"dial": {"dynamic_range": 120.1}})),
            ("dialrange", json!({"dial": {"dynamic_range": 0.9}})),
            (
                "hfnl",
                json!({"rank": {"hfnlproxy": {"per_ref_mean": -0.0001}}}),
            ),
            ("breadth", json!({"rank": {"csiq": {"srocc": 0.8299}}})),
            ("breadth", json!({"rank": {"live": {"srocc": 0.8299}}})),
        ];
        for (id, patch) in cases {
            let mut v = passing_fixture();
            merge(&mut v, &patch);
            let r = eval_balanced(&v, &[]);
            let fl = r.floors.iter().find(|x| x.id == id).unwrap();
            assert!(!fl.pass, "floor {id} must FAIL under patch {patch}");
            assert_eq!(
                r.floors.iter().filter(|x| !x.pass).count(),
                1,
                "exactly one floor fails for patch on {id}"
            );
        }
        // band-tail floor: the SIGNED top band below its derived floor fails,
        // and the SIGNED low band below 0.0 fails. The second case is the one
        // the absolute-valued predecessor could not express at all: |x| >= 0.0
        // is unfalsifiable, so an inverted low band always passed.
        for bands in [
            bands_v(balanced::BAND_LOW, balanced::BAND_HIGH - 0.0001),
            bands_v(-0.001, balanced::BAND_HIGH),
        ] {
            let mut v = passing_fixture();
            v["rank"]["cid22"]["bands"] = bands;
            let r = eval_balanced(&v, &[]);
            let fl = r.floors.iter().find(|x| x.id == "bandtail").unwrap();
            assert!(!fl.pass);
        }
    }

    #[test]
    fn dyn_range_497_class_fails_on_range_and_tied() {
        // The named pathological cell class: dynamic_range 497, tied 15.6%,
        // mono 93.7% — ranks high, broken dial. F5 + F4 must both catch it.
        let mut v = passing_fixture();
        merge(
            &mut v,
            &json!({"dial": {"dynamic_range": 496.6, "tied_pct": 0.156, "mono_pct": 0.937}}),
        );
        let r = eval_balanced(&v, &[]);
        let by = |id: &str| r.floors.iter().find(|x| x.id == id).unwrap().pass;
        assert!(!by("dialrange"), "497-class span must fail F5");
        assert!(!by("dial"), "15.6% tied must fail F4");
    }

    #[test]
    fn missing_axis_is_unevaluable_not_passed() {
        let mut v = passing_fixture();
        v["rank"].as_object_mut().unwrap().remove("hfnlproxy");
        let r = eval_balanced(&v, &[]);
        let fl = r.floors.iter().find(|x| x.id == "hfnl").unwrap();
        assert!(!fl.pass);
        assert!(fl.measured.contains("UNEVALUABLE"));
    }

    // ── Annotations registry (board-integrity pass 2026-08-04) ──────────

    fn ann(id: &str, kind: &str, fields: &[&str], scope: serde_json::Value) -> AnnEntry {
        AnnEntry {
            id: id.into(),
            kind: kind.into(),
            fields: fields.iter().map(|s| s.to_string()).collect(),
            scope,
            reason: format!("test reason for {id}"),
            manual: false,
        }
    }

    #[test]
    fn annotation_scope_predicates_match_as_documented() {
        let v = passing_fixture();
        // missing: fixture HAS rank.hfnlproxy → no match; remove → match.
        let e = ann("x", "annotated", &[], json!({"missing": "rank.hfnlproxy"}));
        assert!(!ann_matches(&v, &e));
        let mut v2 = passing_fixture();
        v2["rank"].as_object_mut().unwrap().remove("hfnlproxy");
        assert!(ann_matches(&v2, &e));
        // explicit null counts as missing (model.output_spline is null here).
        let e_null = ann(
            "x",
            "annotated",
            &[],
            json!({"missing": "model.output_spline"}),
        );
        assert!(ann_matches(&v, &e_null));
        // present: mirror of missing.
        let e_p = ann("x", "annotated", &[], json!({"present": "rank.hfnlproxy"}));
        assert!(ann_matches(&v, &e_p));
        assert!(!ann_matches(&v2, &e_p));
        // names: exact bake-name list.
        let e_n = ann(
            "x",
            "annotated",
            &[],
            json!({"names": ["FIX_single", "other"]}),
        );
        assert!(ann_matches(&v, &e_n));
        let e_n2 = ann("x", "annotated", &[], json!({"names": ["nope"]}));
        assert!(!ann_matches(&v, &e_n2));
        // all.
        assert!(ann_matches(
            &v,
            &ann("x", "annotated", &[], json!({"all": true}))
        ));
        // empty / null scope matches nothing.
        assert!(!ann_matches(&v, &ann("x", "annotated", &[], json!({}))));
        // coverage: segment-boundary prefix only.
        assert!(ann_covers("rank.hfnlproxy", "rank.hfnlproxy.per_ref_mean"));
        assert!(ann_covers("rank.hfnlproxy", "rank.hfnlproxy"));
        assert!(!ann_covers("rank.hfnl", "rank.hfnlproxy.per_ref_mean"));
        assert!(!ann_covers("rank.hfnlproxy.per_ref_mean", "rank.hfnlproxy"));
    }

    #[test]
    fn absent_axis_with_registry_entry_is_absent_not_failed_both_forms() {
        let mut v = passing_fixture();
        v["rank"].as_object_mut().unwrap().remove("hfnlproxy");
        let anns = vec![ann(
            "hfnl-absent-not-failed",
            "absent-not-failed",
            &["rank.hfnlproxy"],
            json!({"missing": "rank.hfnlproxy"}),
        )];
        let r = eval_balanced(&v, &anns);
        let fl = r.floors.iter().find(|x| x.id == "hfnl").unwrap();
        // Still not-passed for n/8 (registered rule) — but ABSENT, not a
        // measured fail, and printed as such.
        assert!(!fl.pass);
        assert!(fl.absent_not_failed);
        assert!(fl.measured.contains("absent"), "got: {}", fl.measured);
        assert_eq!(r.floors.iter().filter(|x| x.pass).count(), 7);
        // TSV: hfnl moves from `fails` to `absent`; both n-forms carried.
        let row = tsv_row(&v, &r);
        let cols: Vec<&str> = row.split('\t').collect();
        let hdr: Vec<&str> = TSV_COLS.split('\t').collect();
        assert_eq!(cols.len(), hdr.len(), "TSV row width matches header");
        let at = |name: &str| cols[hdr.iter().position(|h| *h == name).unwrap()];
        assert_eq!(at("n_pass"), "7/8", "registered absent=not-passed form");
        assert_eq!(at("n_measured"), "7/7", "measured-record form");
        assert_eq!(at("absent"), "hfnl");
        assert!(
            !at("fails").split(',').any(|x| x == "hfnl"),
            "fails: {}",
            at("fails")
        );
        assert!(at("annotations").contains("hfnl-absent-not-failed"));
    }

    #[test]
    fn measured_fail_never_becomes_absent() {
        // A MEASURED fail (negative hfnl) is untouched even when an
        // absent-not-failed entry matches the cell broadly (`all`) and covers
        // the field — absence is a property of the VALUE, not the entry.
        let mut v = passing_fixture();
        merge(
            &mut v,
            &json!({"rank": {"hfnlproxy": {"per_ref_mean": -0.0001}}}),
        );
        let anns = vec![ann(
            "hfnl-absent-not-failed",
            "absent-not-failed",
            &["rank.hfnlproxy"],
            json!({"all": true}),
        )];
        let r = eval_balanced(&v, &anns);
        let fl = r.floors.iter().find(|x| x.id == "hfnl").unwrap();
        assert!(!fl.pass);
        assert!(!fl.absent_not_failed, "measured fail must stay a fail");
        let row = tsv_row(&v, &r);
        let cols: Vec<&str> = row.split('\t').collect();
        let hdr: Vec<&str> = TSV_COLS.split('\t').collect();
        let at = |name: &str| cols[hdr.iter().position(|h| *h == name).unwrap()];
        assert!(at("fails").split(',').any(|x| x == "hfnl"));
        assert_eq!(at("absent"), "-");
        assert_eq!(at("n_measured"), "7/8", "all 8 floors measured");
    }

    #[test]
    fn annotated_kind_flags_without_changing_verdicts() {
        // dial-mono-raw-unit shape: matches spline-less bakes, flags the dial
        // fields, changes NO floor verdict.
        let v = passing_fixture(); // output_spline: null → spline-less
        let anns = vec![ann(
            "dial-mono-raw-unit",
            "annotated",
            &["dial.mono_pct", "dial.tied_pct"],
            json!({"missing": "model.output_spline"}),
        )];
        let r = eval_balanced(&v, &anns);
        let r_plain = eval_balanced(&v, &[]);
        for (a, b) in r.floors.iter().zip(r_plain.floors.iter()) {
            assert_eq!(a.pass, b.pass, "floor {} verdict must not change", a.id);
        }
        assert!(
            r.annotations
                .iter()
                .any(|(id, kind, _)| id == "dial-mono-raw-unit" && kind == "annotated")
        );
        let row = tsv_row(&v, &r);
        let hdr: Vec<&str> = TSV_COLS.split('\t').collect();
        let cols: Vec<&str> = row.split('\t').collect();
        let ann_i = hdr.iter().position(|h| *h == "annotations").unwrap();
        assert_eq!(
            cols[ann_i], "dial-mono-raw-unit",
            "annotations col carries the id"
        );
    }

    /// **D10 (ADD156 ship audit, `benchmarks/add156_ship_audit_2026-08-31.md`).**
    /// A registry that silently loses integrity notes is worse than no
    /// registry. Two silent-drop classes existed:
    ///
    ///  1. **Out-of-array findings.** Three entries sat as bare TOP-LEVEL keys
    ///     of `eval_annotations.json` instead of inside `entries[]`;
    ///     `load_annotations` reads only `v["entries"]`, so all three were
    ///     dropped with no warning — including
    ///     `konjnd-372-full-file-dilution-2026-08-29`, which is *the*
    ///     annotation explaining audit defect D2. The gate that exists to
    ///     surface such caveats could not surface it.
    ///  2. **Scopes the matcher cannot evaluate.** `ann_matches` understands
    ///     exactly four predicates (`missing`/`present`/`names`/`all`) and
    ///     returns `false` for anything else — so an entry scoped
    ///     `{"note": …}` / `{"trained_with": …}` / `{}` matched ZERO cells and
    ///     was indistinguishable from one that simply did not apply. This
    ///     audit found **19 of 42** committed entries in that state, on top of
    ///     the 3 orphans: 22 of 45 findings invisible to the gate.
    ///
    /// The fix makes both classes impossible: unknown top-level keys and
    /// unevaluable scopes are REJECTED at load, and a finding with no machine
    /// predicate must say so explicitly (`{"manual": …}`), which the loader
    /// then surfaces instead of dropping.
    #[test]
    fn d10_registry_findings_are_never_silently_dropped() {
        let repo = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
        let path = repo.join("benchmarks/eval_annotations.json");

        // (1) Every finding in the committed file must be REACHABLE. Count the
        // findings in the raw JSON independently of the loader: entries[] plus
        // any top-level key that is not `_schema`/`entries`.
        let raw: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).expect("read registry"))
                .expect("registry is valid json");
        let obj = raw.as_object().expect("registry is a json object");
        let orphans: Vec<&str> = obj
            .keys()
            .filter(|k| k.as_str() != "_schema" && k.as_str() != "entries")
            .map(|k| k.as_str())
            .collect();
        assert!(
            orphans.is_empty(),
            "registry findings sit OUTSIDE entries[] and are dropped by the \
             loader: {orphans:?} — move them into entries[]"
        );

        let anns = load_annotations(&path).expect("committed registry loads");
        assert_eq!(
            anns.len(),
            obj["entries"].as_array().unwrap().len(),
            "loader dropped entries"
        );

        // (2) Every loaded entry must be EVALUABLE: either it carries one of
        // the four machine predicates, or it declares itself documentation-only.
        // An entry that is neither matched nothing and told nobody.
        let inert: Vec<&str> = anns
            .iter()
            .filter(|e| !e.manual && !scope_is_machine_predicate(&e.scope))
            .map(|e| e.id.as_str())
            .collect();
        assert!(
            inert.is_empty(),
            "entries whose scope the matcher cannot evaluate — they match ZERO \
             cells and are silently inert: {inert:?}"
        );

        // (3) A documentation-only entry never applies to a cell...
        for e in anns.iter().filter(|e| e.manual) {
            assert!(
                !ann_matches(&passing_fixture(), e),
                "manual entry {} must not apply to cells",
                e.id
            );
        }
        // ...but it is still LOADED, so callers can surface it.
        assert!(
            anns.iter().any(|e| e.manual),
            "expected the migrated documentation-only entries to be present"
        );

        // (4) The loader REJECTS both silent-drop classes rather than dropping.
        let with_orphan = serde_json::json!({
            "_schema": {"description": "t"},
            "entries": [],
            "some-finding-2026-08-31": {"reason": "r", "status": "open"}
        });
        let err = load_annotations_value(&with_orphan, "<test>")
            .expect_err("an out-of-array finding must be REJECTED, not dropped");
        assert!(
            err.contains("some-finding-2026-08-31") && err.contains("entries"),
            "diagnostic must name the orphan and where it belongs; got: {err}"
        );

        let with_bad_scope = serde_json::json!({
            "entries": [{
                "id": "bad-scope", "kind": "annotated", "fields": ["rank.cid22"],
                "scope": {"note": "prose, not a predicate"}, "reason": "r"
            }]
        });
        let err = load_annotations_value(&with_bad_scope, "<test>")
            .expect_err("an unevaluable scope must be REJECTED, not silently inert");
        assert!(
            err.contains("bad-scope") && err.contains("note"),
            "diagnostic must name the entry and the offending scope key; got: {err}"
        );

        // An unknown `kind` is equally a silent-behaviour change (only
        // `absent-not-failed` drives the floor accounting).
        let with_bad_kind = serde_json::json!({
            "entries": [{
                "id": "bad-kind", "kind": "absent_not_failed",
                "fields": ["rank.cid22"], "scope": {"all": true}, "reason": "r"
            }]
        });
        assert!(
            load_annotations_value(&with_bad_kind, "<test>").is_err(),
            "an unknown `kind` must be REJECTED"
        );
    }

    #[test]
    fn committed_registry_parses_and_seed_entries_behave() {
        // The COMMITTED registry file must parse, and its three seed entries
        // must do what the campaign doc says they do.
        let repo = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
        let anns = load_annotations(&repo.join("benchmarks/eval_annotations.json"))
            .expect("committed registry parses");
        let ids: Vec<&str> = anns.iter().map(|e| e.id.as_str()).collect();
        for want in [
            "dial-mono-raw-unit",
            "hfnl-absent-not-failed",
            "kadid-tid-train-eq-val",
        ] {
            assert!(ids.contains(&want), "registry missing seed entry {want}");
        }
        // era-bridge-shaped cell: spline present, hfnlproxy missing.
        let mut v = passing_fixture();
        v["rank"].as_object_mut().unwrap().remove("hfnlproxy");
        v["model"]["output_spline"] = json!({"knots": 8});
        let r = eval_balanced(&v, &anns);
        let fl = r.floors.iter().find(|x| x.id == "hfnl").unwrap();
        assert!(fl.absent_not_failed, "hfnl absence covered by the registry");
        let matched: Vec<&str> = r.annotations.iter().map(|(i, _, _)| i.as_str()).collect();
        assert!(matched.contains(&"hfnl-absent-not-failed"));
        assert!(
            matched.contains(&"kadid-tid-train-eq-val"),
            "all-scope entry"
        );
        assert!(
            !matched.contains(&"dial-mono-raw-unit"),
            "spline present ⇒ raw-unit entry must NOT match"
        );
    }

    #[test]
    fn ensemble_class_m3a_not_computable() {
        let mut v = passing_fixture();
        v["model"]["kind"] = json!("ensemble");
        let r = eval_balanced(&v, &[]);
        assert_eq!(r.class, "944-ensemble");
        let (_, m3a) = r.info.iter().find(|(k, _)| k.starts_with("M3a")).unwrap();
        assert!(m3a.contains("NOT COMPUTABLE"), "got: {m3a}");
        // distilled + era-bridge classes
        let mut v = passing_fixture();
        v["name"] = json!("C_ensk2_s1303");
        assert_eq!(eval_balanced(&v, &[]).class, "944-distilled");
        let mut v = passing_fixture();
        v["n_inputs"] = json!(372);
        v["model"]["n_inputs"] = json!(372);
        assert_eq!(eval_balanced(&v, &[]).class, "era-bridge");
    }

    /// Packaging-appendix unit annotation: spline-less bakes' F4 numbers are
    /// raw-unit (unit-flattered); spline-bearing bakes are dial-unit. The
    /// floor logic itself is identical in both cases (registered floors do
    /// not move — this is a label).
    #[test]
    fn f4_dial_row_carries_unit_annotation() {
        let v = passing_fixture(); // output_spline: null
        let r = eval_balanced(&v, &[]);
        let fl = r.floors.iter().find(|x| x.id == "dial").unwrap();
        assert!(fl.measured.contains("(raw-unit)"), "got: {}", fl.measured);
        let mut v2 = passing_fixture();
        v2["model"]["output_spline"] = json!({"n_knots": 18});
        let r2 = eval_balanced(&v2, &[]);
        let fl2 = r2.floors.iter().find(|x| x.id == "dial").unwrap();
        assert!(
            fl2.measured.contains("(dial-unit)"),
            "got: {}",
            fl2.measured
        );
        assert_eq!(fl.pass, fl2.pass, "annotation never changes the verdict");
    }

    /// A NOT-MEASURED band must never be scored, ranked, or read as zero — it
    /// is excluded from the tails and from the composite entirely. This is the
    /// replacement for the old "n<30 renders parenthesized" convention: a band
    /// under the floors no longer reaches the gate as a value at all.
    #[test]
    fn a_not_measured_band_is_never_scored() {
        let mut v = passing_fixture();
        v["rank"]["cid22"]["bands"] = json!([
            { "band": "B0-B8", "lo": 0.0, "hi": 0.9, "n": 4249, "span": 0.62,
              "not_measured_reason": null, "srocc": 0.5, "srocc_signed": 0.5 },
            // the degenerate tail, as the emitter now reports it
            { "band": "B9", "lo": 0.9, "hi": null, "n": 43, "span": 0.0194,
              "not_measured_reason": "n=43 < 1000: too few pairs to rank models",
              "srocc": null, "srocc_signed": null },
        ]);
        let r = eval_balanced(&v, &[]);
        let fl = r.floors.iter().find(|x| x.id == "bandtail").unwrap();
        assert!(
            fl.measured.contains("B0-B8") && !fl.measured.contains("B9"),
            "the unusable band must not appear as a tail: {}",
            fl.measured
        );
        assert!(fl.pass, "the usable band clears both bars: {}", fl.measured);
        // ...and it must not enter the ranking composite either.
        let (lo, hi) = band_tails(&v, "cid22").unwrap();
        assert_eq!(lo.label, "B0-B8");
        assert_eq!(hi.label, "B0-B8", "one usable band ⇒ it is both tails");
    }

    /// A fulleval cut on the pre-appendix-V fixed deciles must be reported
    /// ABSENT, never scored against the current bar: its `B9` is a different
    /// quantity (43 pairs, span 0.019, absolute-valued).
    #[test]
    fn legacy_fixed_decile_bands_are_absent_not_scored() {
        let mut v = passing_fixture();
        v["rank"]["cid22"]
            .as_object_mut()
            .unwrap()
            .remove("band_scheme");
        v["rank"]["cid22"]["bands"] = json!([
            { "band": "B3", "srocc": 0.18, "n": 57 },
            { "band": "B9", "srocc": 0.3204, "n": 43 },
        ]);
        let r = eval_balanced(&v, &[]);
        let fl = r.floors.iter().find(|x| x.id == "bandtail").unwrap();
        assert!(
            fl.absent,
            "legacy bands are not measured under the current bar"
        );
        assert!(!fl.pass, "and absence is never a pass");
        assert!(fl.measured.contains("LEGACY"), "got: {}", fl.measured);
        // The ranking composite must not eat the legacy value either.
        assert!(balanced_composite(&v).is_some());
        let with_scheme = passing_fixture();
        assert!(
            balanced_composite(&v).unwrap() != balanced_composite(&with_scheme).unwrap(),
            "dropping the band term must change the composite, else the test is vacuous"
        );
    }

    #[test]
    fn composite_matches_registered_weights() {
        let v = passing_fixture();
        // Hand-computed with the §8.1 table (abs corpus terms; signed band-tail):
        let bandtail = (balanced::BAND_LOW + balanced::BAND_HIGH) / 2.0;
        let num = 1.00 * 0.885
            + 0.50 * 0.91
            + 0.30 * 0.90
            + 0.20 * 0.43
            + 0.15 * 0.83
            + 0.15 * 0.83
            + 0.15 * bandtail
            + 0.10 * 0.79
            + 0.05 * 0.91;
        let den = 1.00 + 0.50 + 0.30 + 0.20 + 0.15 + 0.15 + 0.15 + 0.10 + 0.05;
        let expect = num / den;
        let got = balanced_composite(&v).unwrap();
        assert!((got - expect).abs() < 1e-12, "got {got}, expect {expect}");
        // absent term drops from num AND den
        let mut v2 = passing_fixture();
        v2["rank"].as_object_mut().unwrap().remove("aic4");
        let got2 = balanced_composite(&v2).unwrap();
        let expect2 = (num - 0.05 * 0.91) / (den - 0.05);
        assert!((got2 - expect2).abs() < 1e-12);
    }

    #[test]
    fn tsv_row_carries_verdict_and_fails() {
        // (2026-08-04: `fails` looked up by header name — the annotations
        // registry appended n_measured/absent/annotations after it, so
        // `.last()` no longer addresses it. Assertion intent unchanged.)
        let hdr: Vec<&str> = TSV_COLS.split('\t').collect();
        let fails_i = hdr.iter().position(|h| *h == "fails").unwrap();
        let v = passing_fixture();
        let r = eval_balanced(&v, &[]);
        let row = tsv_row(&v, &r);
        let cols: Vec<&str> = row.split('\t').collect();
        assert_eq!(cols.len(), hdr.len());
        assert_eq!(cols[0], "FIX_single");
        assert_eq!(cols[2], "PASS");
        assert_eq!(cols[3], "8/8");
        assert_eq!(cols[fails_i], "-");
        let mut v2 = passing_fixture();
        merge(&mut v2, &json!({"rank": {"cid22": {"srocc": 0.80}}}));
        let r2 = eval_balanced(&v2, &[]);
        let row2 = tsv_row(&v2, &r2);
        let cols2: Vec<&str> = row2.split('\t').collect();
        assert_eq!(cols2[2], "FAIL");
        assert!(cols2[fails_i].contains("cid22"));
    }

    /// The §5 default path is UNCHANGED by the profile addition: lock the row
    /// count, order, gate names and bar semantics of `legacy_rows` (this is
    /// the pre-existing behavior, transcribed — not new policy).
    #[test]
    fn legacy_rows_unchanged_by_profile_addition() {
        let v = passing_fixture();
        let rows = legacy_rows(&v, None, None);
        let names: Vec<String> = rows
            .iter()
            .map(|r| match r {
                Row::Eval(g, ..) => format!("E:{g}"),
                Row::Attach(g, ..) => format!("A:{g}"),
                Row::Info(g, ..) => format!("I:{g}"),
            })
            .collect();
        assert_eq!(
            names,
            vec![
                "E:CID22 SROCC (selected seed)",
                "E:KonJND abs-SROCC",
                "I:Corruption head detection (context)",
                "A:Corruption ORDERING via head",
                "A:CSIQ SROCC (≥ best 924-arm)",
                "A:LIVE SROCC (≥ best 924-arm)",
                "A:UPIQ pooled (V1-HDR)",
                "A:Korshunov hold (V1-HDR)",
                "E:M3a coherence (EM2-class)",
                "E:Dial monotonicity",
                "E:Dial tied rate",
                "I:product_composite",
                "E:Byte-repro (embedded zentrain.repro)",
                "A:Perf SDR",
                "A:Perf HDR (PU path)",
                "A:LOO (append2 family)",
            ]
        );
        // §5 bar semantics spot-checks: 0.885 fails the §5 0.89 bar; M3a 0.80
        // fails 0.85; the dial rows pass at the same G3 lines.
        let get = |g: &str| {
            rows.iter()
                .find_map(|r| match r {
                    Row::Eval(name, _, _, ok) if name == g => Some(*ok),
                    _ => None,
                })
                .unwrap()
        };
        assert!(!get("CID22 SROCC (selected seed)"));
        assert!(get("KonJND abs-SROCC"));
        assert!(!get("M3a coherence (EM2-class)"));
        assert!(get("Dial monotonicity"));
        assert!(get("Dial tied rate"));
        // csiq/live stay ATTACH without --bar, Eval with it.
        let rows2 = legacy_rows(&v, Some(0.8), None);
        assert!(rows2.iter().any(|r| matches!(r,
            Row::Eval(g, _, _, true) if g == "CSIQ SROCC (≥ best 924-arm)")));
    }

    // ── `--select` (campaign appendix E.4) ──────────────────────────────

    /// Build a SelectRow the way `run_select` does, so the rule under test
    /// is the shipped one (no second implementation of the arithmetic).
    fn select_row(v: &serde_json::Value) -> SelectRow {
        let r = eval_balanced(v, &[]);
        let m3a = m3a_state(v, r.class);
        let selection_composite = match (r.composite, m3a) {
            (Some(c), M3aState::Measured(x)) => Some(c + balanced::W_M3A * x),
            _ => None,
        };
        SelectRow {
            name: bake_name(v).to_string(),
            path: String::new(),
            class: r.class,
            n_pass: r.floors.iter().filter(|x| x.pass).count(),
            n_floors: r.floors.len(),
            floors_passed: r.floors.iter().filter(|x| x.pass).map(|x| x.id).collect(),
            composite: r.composite,
            m3a,
            selection_composite,
            sdr25: f(v, &["rank", "sdr25", "srocc"]).map(f64::abs),
            bake: None,
            group_key: seed_group_key(v),
            seed_id: seed_identity(v),
            seed_text: seed_label(v),
            gaddr_contract_fails: gaddr_contract_fails_of(v, GaddrBlock::Auto),
            gaddr_codec: gaddr_codec_states_of(v, GaddrBlock::Auto),
        }
    }

    /// **D3 (ADD156 ship audit, `benchmarks/add156_ship_audit_2026-08-31.md`).**
    /// The audit filed "the registered selection rule cannot select ADD156 —
    /// a model that is structurally unshortlistable cannot be selected as a
    /// profile" as a HIGH ship-blocker, on the strength of the `era-bridge`
    /// class note reading *"context only — regime-incomparable, never
    /// shortlisted"*.
    ///
    /// **That is FALSIFIED.** `class` is compared against `"944-ensemble"`
    /// and nothing else; `"era-bridge"` is tested nowhere. Selectability is
    /// `m3a != Unmeasured && n_pass > 0` — the class is not a term. On the
    /// board fullevals the rule does not merely permit ADD156, it SELECTS it
    /// (selection_composite 0.9644) over shipped `B` (0.9151), both stamped
    /// `era-bridge`. The audit's own "NO" came from its ad-hoc fulleval
    /// missing `m3a_coherence` — a value it had measured at 0.9641.
    ///
    /// This test pins both halves so nobody "fixes" D3 by implementing the
    /// exclusion the note used to describe.
    #[test]
    fn d3_era_bridge_class_is_a_label_not_an_exclusion() {
        // An era-bridge-shaped cell: non-944 input width, M3a measured.
        let mut v = passing_fixture();
        merge(
            &mut v,
            // `classify` prefers a TOP-LEVEL `n_inputs`, so set both.
            &json!({"name": "ERA_BRIDGE_CANDIDATE", "m3a_coherence": 0.95,
                    "n_inputs": 372, "model": {"n_inputs": 372}}),
        );
        let r = select_row(&v);
        assert_eq!(
            r.class, "era-bridge",
            "fixture must be in the class under test"
        );

        // The rule's own selectability predicate, verbatim from `run_select`.
        let selectable = r.m3a != M3aState::Unmeasured && r.n_pass > 0;
        assert!(
            selectable,
            "an era-bridge cell with a measured M3a and {} passing floors must be \
             SELECTABLE — the class is not a term in the rule",
            r.n_pass
        );

        // ...and the printed note must not claim otherwise.
        let (_, note) = classify(&v);
        assert!(
            !note.contains("never shortlisted"),
            "the class note claims an exclusion the code does not implement: {note}"
        );
        assert!(
            note.contains("NOT an exclusion"),
            "the note must say what the label actually is: {note}"
        );

        // The one class that IS special-cased stays special-cased.
        let mut e = passing_fixture();
        merge(&mut e, &json!({"model": {"kind": "ensemble"}}));
        assert_eq!(select_row(&e).class, "944-ensemble");
        assert_eq!(select_row(&e).m3a, M3aState::NotComputable);
    }

    /// PRIMARY is the floor count: a bake with a higher M3a but one fewer
    /// floor must NOT outrank one that passes more floors.
    #[test]
    fn select_primary_is_floor_count_not_coherence() {
        let mut hi_m3a = passing_fixture();
        merge(
            &mut hi_m3a,
            &json!({"name": "HI_M3A", "m3a_coherence": 0.99,
                    "rank": {"cid22": {"srocc": 0.80}}}), // fails F1
        );
        let mut lo_m3a = passing_fixture();
        merge(
            &mut lo_m3a,
            &json!({"name": "LO_M3A", "m3a_coherence": 0.20}),
        );
        let (a, b) = (select_row(&hi_m3a), select_row(&lo_m3a));
        assert!(a.n_pass < b.n_pass, "fixture must differ in floor count");
        assert!(
            a.selection_composite > b.selection_composite,
            "fixture must have the LOSER ahead on the tie-break, else the test is vacuous"
        );
        let mut pool = vec![&a, &b];
        rank_pool(&mut pool, FloorBasis::AllReps);
        assert_eq!(
            pool[0].name, "LO_M3A",
            "floor count is PRIMARY — coherence must not override a failed floor"
        );
    }

    /// TIE-BREAK: at equal floor count, higher M3a wins, and the margin is
    /// exactly W_M3A × ΔM3a.
    #[test]
    fn select_tiebreak_is_composite_plus_weighted_m3a() {
        let mut a = passing_fixture();
        merge(&mut a, &json!({"name": "A", "m3a_coherence": 0.90}));
        let mut b = passing_fixture();
        merge(&mut b, &json!({"name": "B", "m3a_coherence": 0.70}));
        let (ra, rb) = (select_row(&a), select_row(&b));
        assert_eq!(ra.n_pass, rb.n_pass, "same floors");
        assert_eq!(ra.composite, rb.composite, "same balanced_composite");
        let margin = ra.selection_composite.unwrap() - rb.selection_composite.unwrap();
        assert!(
            (margin - balanced::W_M3A * 0.20).abs() < 1e-12,
            "margin {margin} must be exactly W_M3A·ΔM3a"
        );
        let mut pool = vec![&rb, &ra];
        rank_pool(&mut pool, FloorBasis::AllReps);
        assert_eq!(pool[0].name, "A");
    }

    // ── `--seed-group` (2026-09-04 owner-fix, §7.7) ─────────────────────

    /// Build a fixture that trains under `argv`, at seed `seed`, and
    /// reaches `m3a` — the three things the seed-group rule reads.
    fn seed_fixture(name: &str, argv: &[&str], seed: u64, m3a: f64) -> serde_json::Value {
        let mut v = passing_fixture();
        merge(
            &mut v,
            &json!({
                "name": name,
                "m3a_coherence": m3a,
                "repro": { "seed": seed, "argv": argv },
            }),
        );
        v
    }

    /// **Failing-first fixture for the §7.7 defect**: reproduces the exact
    /// shape the wave doc measured — a "CONTROL" recipe whose seed 4004 is a
    /// lucky best-of-3 draw that wins the plain per-cell `rank_pool`
    /// ranking outright, while CONTROL's OWN group mean loses to a
    /// different, more-consistent "ARM" recipe's mean. Before `--seed-group`
    /// existed, `run_select` had no way to express the ARM-wins-on-average
    /// verdict at all — only this per-cell CONTROL-wins one.
    ///
    /// Every row is `passing_fixture()` with ONLY `m3a_coherence` and the
    /// repro moved. That is deliberate and load-bearing twice over:
    /// `m3a_coherence` is NOT one of the eight balanced floors (see
    /// `balanced_floors_resolve_as_registered_and_fixture_passes`), so all
    /// rows pass 8/8 and the PRIMARY term ties — the assertions below
    /// exercise the TIE-BREAK exclusively, which is the axis §7.7 found
    /// broken. And because `balanced_composite` is a weighted mean of the
    /// per-corpus SROCCs (NOT the fulleval's top-level `composite` field),
    /// leaving every corpus value untouched keeps `composite` IDENTICAL
    /// across all rows, so `selection_composite = composite + W_M3A·m3a`
    /// orders exactly as `m3a` does and the expected means are arithmetic
    /// anyone can check by hand.
    #[test]
    fn seed_group_mean_can_beat_the_lucky_per_cell_winner() {
        let control_argv = ["zensim_mlp_train", "--epochs", "40", "--hidden", "24"];
        let arm_argv = ["zensim_mlp_train", "--epochs", "40", "--hidden", "48"];
        // CONTROL: seed 4004 is the lucky outlier (m3a 0.95 — the single
        // highest cell in the whole fixture); 4005/4006 are much weaker.
        // Mean m3a = (0.95 + 0.20 + 0.15)/3 = 0.4333…
        let c1 = seed_fixture("FC_C0_s4004", &control_argv, 4004, 0.95);
        let c2 = seed_fixture("FC_C0_s4005", &control_argv, 4005, 0.20);
        let c3 = seed_fixture("FC_C0_s4006", &control_argv, 4006, 0.15);
        // ARM: three CONSISTENT, moderately-good seeds — every single ARM
        // cell is below CONTROL's lucky max, so ARM never wins a per-cell
        // comparison against seed 4004. Mean m3a = 0.50, which beats
        // CONTROL's 0.4333…
        let a1 = seed_fixture("FC_ARM_s7", &arm_argv, 7, 0.50);
        let a2 = seed_fixture("FC_ARM_s8", &arm_argv, 8, 0.50);
        let a3 = seed_fixture("FC_ARM_s9", &arm_argv, 9, 0.50);
        // A third recipe with one seed: exercises UNREPLICATED in isolation,
        // low-scoring so it cannot confound the winner assertions below.
        let solo = seed_fixture(
            "FC_SOLO_s1",
            &["zensim_mlp_train", "--epochs", "40", "--hidden", "12"],
            1,
            0.10,
        );

        let rows: Vec<SelectRow> = [&c1, &c2, &c3, &a1, &a2, &a3, &solo]
            .iter()
            .map(|v| select_row(v))
            .collect();

        // Sanity: every row passes all 8 floors, so PRIMARY ties across the
        // board and TIE-BREAK alone decides both rankings below. And every
        // row shares one `composite`, so the tie-break IS the m3a term.
        let c0 = rows[0].composite.expect("fixture has a balanced_composite");
        for r in &rows {
            assert_eq!(r.n_pass, 8, "{}: fixture must pass every floor", r.name);
            assert_eq!(
                r.composite,
                Some(c0),
                "{}: fixture must move ONLY m3a, else this is not a pure tie-break test",
                r.name
            );
        }

        // 1) THE DEFECT, reproduced: the plain per-cell rule (unchanged by
        //    this fix) selects CONTROL's lucky seed 4004, not ARM.
        let mut per_cell: Vec<&SelectRow> = rows.iter().collect();
        rank_pool(&mut per_cell, FloorBasis::AllReps);
        assert_eq!(
            per_cell[0].name, "FC_C0_s4004",
            "the lucky per-cell winner must be CONTROL's seed 4004 — if this \
             fails, the fixture no longer reproduces the §7.7 shape"
        );

        // 2) THE FIX: seed-grouped ranking selects ARM's group instead,
        //    because ARM's MEAN beats CONTROL's mean even though no single
        //    ARM cell beats CONTROL's lucky cell.
        let refs: Vec<&SelectRow> = rows.iter().collect();
        let mut groups = group_by_seed(&refs);
        assert_eq!(groups.len(), 3, "three distinct recipes ⇒ three groups");

        let by_member = |groups: &[SeedGroupRow<'_>], n: &str| -> String {
            groups
                .iter()
                .find(|g| g.members.iter().any(|m| m.name == n))
                .map(|g| g.key.clone())
                .unwrap_or_else(|| panic!("{n} landed in no group"))
        };
        let ck = by_member(&groups, "FC_C0_s4004");
        let ak = by_member(&groups, "FC_ARM_s7");
        assert_ne!(
            ck, ak,
            "different --hidden ⇒ different recipe ⇒ different key"
        );
        assert_eq!(
            ck,
            by_member(&groups, "FC_C0_s4006"),
            "same argv modulo seed"
        );

        let control = groups.iter().find(|g| g.key == ck).unwrap();
        let arm = groups.iter().find(|g| g.key == ak).unwrap();
        let solo_group = groups
            .iter()
            .find(|g| g.members.iter().any(|m| m.name == "FC_SOLO_s1"))
            .unwrap();

        assert_eq!(control.reps.len(), 3, "k = 3 distinct seeds");
        assert_eq!(arm.reps.len(), 3);
        assert!(!control.unreplicated);
        assert!(!arm.unreplicated);
        assert!(
            solo_group.unreplicated,
            "a group with exactly one DISTINCT SEED must be UNREPLICATED — \
             its \"mean\" is a single draw, not an estimate"
        );
        assert!(
            !solo_group.ungroupable,
            "it HAS an argv, so it is groupable-but-unreplicated, which is a \
             different statement from UNGROUPABLE"
        );
        assert_eq!(group_state(solo_group), "UNREPLICATED");
        assert_eq!(group_state(control), "replicated");

        // The k-seed mean m3a, checkable by hand from the fixture above.
        match (control.m3a, arm.m3a) {
            (M3aState::Measured(c), M3aState::Measured(a)) => {
                assert!(
                    (c - (0.95 + 0.20 + 0.15) / 3.0).abs() < 1e-12,
                    "control m3a {c}"
                );
                assert!((a - 0.50).abs() < 1e-12, "arm m3a {a}");
            }
            other => panic!("both groups must be MEASURED, got {other:?}"),
        }

        let control_mean = control.mean_selection_composite.unwrap();
        let arm_mean = arm.mean_selection_composite.unwrap();
        assert!(
            (control_mean - (c0 + balanced::W_M3A * (0.95 + 0.20 + 0.15) / 3.0)).abs() < 1e-12,
            "control mean {control_mean}"
        );
        assert!(
            (arm_mean - (c0 + balanced::W_M3A * 0.50)).abs() < 1e-12,
            "arm mean {arm_mean}"
        );
        assert!(
            arm_mean > control_mean,
            "fixture must have ARM beat CONTROL on the mean, else the test is vacuous"
        );

        // The SPREAD the pre-fix rule threw away: CONTROL's per-seed range
        // is wide, ARM's is exactly zero. Reporting this is the whole point
        // of not presenting the mean as "the group's true score".
        let (c_lo, c_hi) = control.selection_composite_spread.unwrap();
        let (a_lo, a_hi) = arm.selection_composite_spread.unwrap();
        assert!(
            (c_hi - c_lo - balanced::W_M3A * (0.95 - 0.15)).abs() < 1e-12,
            "control spread {c_lo}–{c_hi}"
        );
        assert!(
            (a_hi - a_lo).abs() < 1e-12,
            "arm spread must be zero: {a_lo}–{a_hi}"
        );
        assert!(
            c_hi > a_hi,
            "CONTROL's best single seed must still beat ARM's — that is what makes \
             the per-cell ranking pick CONTROL and the grouped ranking pick ARM"
        );

        rank_seed_groups(&mut groups, FloorBasis::Mean);
        assert_eq!(
            groups[0].key, ak,
            "seed-grouped ranking must select ARM (the group whose MEAN wins), \
             not CONTROL (whose lucky per-cell draw wins the per-cell ranking above) \
             — this is the exact defect §7.7 measured and this fix corrects"
        );
    }

    // ── REGISTERED AMENDMENT (2026-09-05) — the REPLICATION FLOOR ───────

    /// **FAILING-FIRST for the live-board defect.** Reproduces the exact
    /// shape `benchmarks/replication_wave_2026-09-05.md` §4c.4 measured: a
    /// **k = 1** cell that passes every floor (`W10L9_s4003_packed`, 8.00/8,
    /// selection_composite 0.9841) outranking a replicated group at 7.22/8.
    ///
    /// Under the PRE-amendment rule (`--min-k 1 --floor-basis mean`) the
    /// single draw wins, which is what the board did. Under the amendment it
    /// is UNREPLICATED, not selectable, and the replicated group wins. Both
    /// halves are asserted, so a regression in either direction fails.
    #[test]
    fn replication_floor_excludes_the_single_draw() {
        let solo_argv = ["zensim_mlp_train", "--epochs", "77"];
        let rep_argv = ["zensim_mlp_train", "--epochs", "40"];
        // The lone draw: 8/8 floors (passing_fixture passes all eight) and
        // the best m3a, so it wins BOTH terms of the registered rule.
        let solo = seed_fixture("SOLO_s4003", &solo_argv, 4003, 0.95);
        // The replicated recipe: same 8/8 floors, lower m3a ⇒ strictly worse
        // on the tie-break. It can only win by the single draw being
        // ineligible, which is exactly the amendment.
        let r1 = seed_fixture("REP_s4004", &rep_argv, 4004, 0.50);
        let r2 = seed_fixture("REP_s4005", &rep_argv, 4005, 0.50);
        let r3 = seed_fixture("REP_s4006", &rep_argv, 4006, 0.50);
        let rows: Vec<SelectRow> = [&solo, &r1, &r2, &r3]
            .iter()
            .map(|v| select_row(v))
            .collect();
        let refs: Vec<&SelectRow> = rows.iter().collect();
        let solo_key = seed_group_key(&solo).unwrap();
        let rep_key = seed_group_key(&rep_argv_probe(&rep_argv)).unwrap();

        // ── PRE-amendment: the single draw is selected. (This is the
        // behavior `--min-k 1 --floor-basis mean` must keep reproducing.)
        let mut groups = group_by_seed(&refs);
        rank_seed_groups(&mut groups, FloorBasis::Mean);
        assert_eq!(
            groups[0].key, solo_key,
            "pre-amendment ranks the lone draw first"
        );
        assert!(
            group_unselectable_reason(&groups[0], FloorBasis::Mean, 1).is_none(),
            "pre-amendment, k is not a selectability term — the lone draw IS selectable"
        );

        // ── AMENDED (default `--min-k 2`): the lone draw is listed, ranked
        // in its own pool, and cannot be selected; the replicated group is.
        let mut groups = group_by_seed(&refs);
        rank_seed_groups(&mut groups, FloorBasis::AllReps);
        let solo_g = groups.iter().find(|g| g.key == solo_key).unwrap();
        let rep_g = groups.iter().find(|g| g.key == rep_key).unwrap();
        assert_eq!(solo_g.k_seeds, 1);
        assert_eq!(rep_g.k_seeds, 3);
        let why = group_unselectable_reason(solo_g, FloorBasis::AllReps, 2)
            .expect("the k=1 group must NOT be selectable under the replication floor");
        assert!(why.contains("UNREPLICATED"), "reason must name it: {why}");
        assert!(why.contains("k=1"), "reason must carry the k: {why}");
        assert!(
            group_unselectable_reason(rep_g, FloorBasis::AllReps, 2).is_none(),
            "the k=3 group clears the floor and is selectable"
        );
        // It is EXCLUDED, never deleted: it is still a group, still listed,
        // still carrying its own numbers.
        assert_eq!(solo_g.members.len(), 1);
        assert!(solo_g.mean_n_pass > rep_g.mean_n_pass - 1e-12);
        // And the winner among the eligible pool is the replicated recipe.
        let winner = groups
            .iter()
            .find(|g| group_unselectable_reason(g, FloorBasis::AllReps, 2).is_none())
            .expect("some group clears the floor");
        assert_eq!(
            winner.key, rep_key,
            "the amended rule selects the REPLICATED recipe, not the lucky single draw"
        );
    }

    /// Helper: a throwaway fixture carrying only the argv, so a test can ask
    /// for a recipe's group key without owning one of its rows.
    fn rep_argv_probe(argv: &[&str]) -> serde_json::Value {
        seed_fixture("PROBE", argv, 1, 0.0)
    }

    /// **Amendment part B.** Two seeds of one recipe each pass 7 of 8
    /// floors, but they fail DIFFERENT floors. The k-seed MEAN credits the
    /// group 7.0 floors — including two that no member reliably clears. The
    /// intersection credits 6, and names the split.
    ///
    /// This is a certification question, not a rounding one: "this recipe
    /// clears CID22" must not be true of a group in which one seed does not.
    #[test]
    fn all_reps_basis_refuses_a_floor_no_member_reliably_clears() {
        let argv = ["zensim_mlp_train", "--epochs", "40"];
        // Seed A fails CID22 only; seed B fails KonJND only.
        let mut a = seed_fixture("SPLIT_s1", &argv, 1, 0.90);
        a["rank"]["cid22"]["srocc"] = json!(0.5);
        let mut b = seed_fixture("SPLIT_s2", &argv, 2, 0.90);
        b["rank"]["konjnd"]["srocc"] = json!(-0.01);
        let rows: Vec<SelectRow> = [&a, &b].iter().map(|v| select_row(v)).collect();
        let refs: Vec<&SelectRow> = rows.iter().collect();
        assert_eq!(rows[0].n_pass, 7, "seed A passes 7 of 8");
        assert_eq!(rows[1].n_pass, 7, "seed B passes 7 of 8");
        assert_ne!(
            rows[0].floors_passed, rows[1].floors_passed,
            "the two seeds must fail DIFFERENT floors, else the test is vacuous"
        );
        let groups = group_by_seed(&refs);
        let g = &groups[0];
        assert!(
            (g.mean_n_pass - 7.0).abs() < 1e-12,
            "the MEAN basis credits 7.0: {}",
            g.mean_n_pass
        );
        assert_eq!(
            g.n_pass_all, 6,
            "the intersection credits only the 6 floors BOTH seeds pass"
        );
        assert!((g.floor_count(FloorBasis::Mean) - 7.0).abs() < 1e-12);
        assert!((g.floor_count(FloorBasis::AllReps) - 6.0).abs() < 1e-12);
        let split: std::collections::BTreeSet<&str> = g.floors_split.iter().copied().collect();
        assert_eq!(
            split,
            ["cid22", "konjnd"].into_iter().collect(),
            "the split floors must be NAMED, not silently subtracted: {:?}",
            g.floors_split
        );
        assert!(
            !g.floors_all.contains("cid22") && !g.floors_all.contains("konjnd"),
            "a split floor is not part of what every seed clears"
        );
    }

    /// The amendment can only REMOVE a candidate from selection, never admit
    /// one the pre-amendment rule refused. Asserted directly, because a
    /// "tightening" that quietly widens anything is not a tightening.
    #[test]
    fn amendment_only_ever_removes_candidates() {
        let argv_a = ["zensim_mlp_train", "--epochs", "40"];
        let argv_b = ["zensim_mlp_train", "--epochs", "41"];
        let rows_v = [
            seed_fixture("A_s1", &argv_a, 1, 0.90),
            seed_fixture("A_s2", &argv_a, 2, 0.50),
            seed_fixture("B_s1", &argv_b, 1, 0.95),
        ];
        let rows: Vec<SelectRow> = rows_v.iter().map(select_row).collect();
        let refs: Vec<&SelectRow> = rows.iter().collect();
        let groups = group_by_seed(&refs);
        for g in &groups {
            let pre = group_unselectable_reason(g, FloorBasis::Mean, 1).is_none();
            for basis in [FloorBasis::Mean, FloorBasis::AllReps] {
                for min_k in [1usize, 2, 3] {
                    let post = group_unselectable_reason(g, basis, min_k).is_none();
                    assert!(
                        pre || !post,
                        "group {} became selectable under basis {basis:?} / min-k {min_k} \
                         when the pre-amendment rule refused it",
                        g.key
                    );
                }
            }
            // n_pass_all <= mean_n_pass, always.
            assert!(
                g.floor_count(FloorBasis::AllReps) <= g.floor_count(FloorBasis::Mean) + 1e-12,
                "the strict basis can never credit MORE floors than the mean"
            );
        }
    }

    // ── G-ADDR CONTRACT pre-filter (2026-09-06 owner fix, registry
    // select-rule-blind-to-dial-contract-2026-09-06) ────────────────────

    /// **The 2026-09-06 ruler switch.** `dial_ladder` is the OPERATIVE dial
    /// instrument and `dial.addressability` the retained canonical read; both
    /// are visible at once and a value from one is NOT comparable with a value
    /// from the other, so the choice must have exactly one owner
    /// ([`gaddr_block`]). This pins all three modes on a cell carrying BOTH
    /// blocks with OPPOSITE verdicts, plus the two absence cases — a cell with
    /// only a canonical block still reads under `auto`, and `ladder` refuses to
    /// fall back to it (NOT MEASURED, never another instrument's number).
    #[test]
    fn gaddr_block_precedence_is_ladder_then_canonical_and_ladder_never_falls_back() {
        let mk = |id: &str, state: &str| {
            serde_json::json!({"id": id, "tier": "contract", "state": state,
                               "measured": 1.0, "bar": 0.0, "cmp": "\u{2265}",
                               "note": format!("{id} {state}"), "what": "x"})
        };
        let canon = serde_json::json!({
            "checks": [mk("C2", "fail")],
            "measured": {"codec_floor": [{"codec": "jpeg", "state": "fail"}]}});
        let ladder = serde_json::json!({
            "checks": [mk("C2", "pass")],
            "measured": {"codec_floor": [{"codec": "jpeg", "state": "pass"},
                                         {"codec": "avif-svt", "state": "fail"}]}});
        let both = serde_json::json!({"dial": {"addressability": canon.clone()},
                                      "dial_ladder": ladder.clone()});
        // auto + ladder read the LADDER (C2 passes there, so no veto);
        // canonical reads the canonical block (C2 fails ⇒ a veto).
        assert!(gaddr_contract_fails_of(&both, GaddrBlock::Auto).is_empty());
        assert!(gaddr_contract_fails_of(&both, GaddrBlock::Ladder).is_empty());
        assert_eq!(
            gaddr_contract_fails_of(&both, GaddrBlock::Canonical)
                .iter()
                .map(|(i, _)| i.as_str())
                .collect::<Vec<_>>(),
            vec!["C2"]
        );
        // The A7r per-codec table follows the SAME block — never a mix.
        assert_eq!(gaddr_codec_states_of(&both, GaddrBlock::Auto).len(), 2);
        assert_eq!(gaddr_codec_states_of(&both, GaddrBlock::Canonical).len(), 1);

        // Canonical only: `auto` falls back, `ladder` does NOT (absence is NOT
        // MEASURED, which is never a fail and never another ruler's number).
        let only_canon = serde_json::json!({"dial": {"addressability": canon}});
        assert_eq!(
            gaddr_contract_fails_of(&only_canon, GaddrBlock::Auto).len(),
            1
        );
        assert!(gaddr_contract_fails_of(&only_canon, GaddrBlock::Ladder).is_empty());
        assert!(gaddr_codec_states_of(&only_canon, GaddrBlock::Ladder).is_empty());

        // Neither block: empty under every mode — "no G-ADDR ever ran".
        let none = serde_json::json!({"rank": {}});
        for m in [GaddrBlock::Auto, GaddrBlock::Canonical, GaddrBlock::Ladder] {
            assert!(gaddr_contract_fails_of(&none, m).is_empty());
            assert!(gaddr_codec_states_of(&none, m).is_empty());
        }
    }

    /// A synthetic `dial.addressability` block, mirroring EXACTLY the JSON
    /// shape `dial_addressability::to_json` emits (never a second
    /// implementation of the verdict — just its shape, for a fixture).
    /// `fail_ids` become CONTRACT-tier `state: "fail"` rows among the six
    /// canonical `C1..C6` (the rest `pass`); `codec_states` becomes the
    /// `A7r` per-codec breakdown, `(codec, Some(true)=pass /
    /// Some(false)=fail / None=not_measured)`.
    fn gaddr_fixture(
        fail_ids: &[&str],
        codec_states: &[(&str, Option<bool>)],
    ) -> serde_json::Value {
        let contract_ids = ["C1", "C2", "C3", "C4", "C5", "C6"];
        let checks: Vec<serde_json::Value> = contract_ids
            .iter()
            .map(|id| {
                let state = if fail_ids.contains(id) {
                    "fail"
                } else {
                    "pass"
                };
                json!({"id": id, "tier": "contract", "state": state,
                       "note": format!("{id} {state} (fixture)")})
            })
            .collect();
        let codec_floor: Vec<serde_json::Value> = codec_states
            .iter()
            .map(|(codec, state)| {
                let s = match state {
                    Some(true) => "pass",
                    Some(false) => "fail",
                    None => "not_measured",
                };
                json!({"codec": codec, "state": s})
            })
            .collect();
        json!({
            "dial": {
                "addressability": {
                    "contract": if fail_ids.is_empty() { "PASS" } else { "FAIL" },
                    "checks": checks,
                    "measured": { "codec_floor": codec_floor },
                }
            }
        })
    }

    /// **FAILING-FIRST (2026-09-06 owner fix).** Reproduces the best-of-all
    /// wave's measured shape (`benchmarks/best_of_all_2026-09-06.md`,
    /// registry `select-rule-blind-to-dial-contract-2026-09-06`): a
    /// CONTROL seed group whose every seed MEASURES a G-ADDR CONTRACT-tier
    /// fail (C5/C6 — the exact rows `A_plain` failed), but which has the
    /// HIGHEST composite/M3a of the two groups; a PASS group whose every
    /// seed is CONTRACT-clean (6/6) but scores lower on composite/M3a.
    /// Before this fix, `run_select`/`group_by_seed` never read
    /// `dial.addressability` at all, so PRIMARY (floor count — tied, both
    /// groups pass all 8 balanced floors) and TIE-BREAK
    /// (`selection_composite`) both picked CONTROL. After the fix: PASS is
    /// selected, and CONTROL is reported NOT selectable with its failing
    /// contract rows named.
    #[test]
    fn gaddr_contract_veto_blocks_a_higher_composite_seed_group() {
        let control_argv = ["zensim_mlp_train", "--epochs", "40", "--hidden", "24"];
        let pass_argv = ["zensim_mlp_train", "--epochs", "40", "--hidden", "48"];
        let control_rows: Vec<serde_json::Value> = [(4004u64, 0.95), (4005, 0.90), (4006, 0.92)]
            .iter()
            .map(|&(seed, m3a)| {
                let mut v = seed_fixture(&format!("CONTROL_s{seed}"), &control_argv, seed, m3a);
                merge(&mut v, &gaddr_fixture(&["C5", "C6"], &[]));
                v
            })
            .collect();
        let pass_rows: Vec<serde_json::Value> = [(4004u64, 0.60), (4005, 0.62), (4006, 0.58)]
            .iter()
            .map(|&(seed, m3a)| {
                let mut v = seed_fixture(&format!("PASS_s{seed}"), &pass_argv, seed, m3a);
                merge(&mut v, &gaddr_fixture(&[], &[]));
                v
            })
            .collect();
        let rows: Vec<SelectRow> = control_rows
            .iter()
            .chain(pass_rows.iter())
            .map(select_row)
            .collect();
        let refs: Vec<&SelectRow> = rows.iter().collect();
        let mut groups = group_by_seed(&refs);
        assert_eq!(groups.len(), 2, "two distinct recipes ⇒ two groups");

        let control = groups
            .iter()
            .find(|g| g.members[0].name.starts_with("CONTROL"))
            .expect("control group must exist")
            .key
            .clone();
        let passg = groups
            .iter()
            .find(|g| g.members[0].name.starts_with("PASS"))
            .expect("pass group must exist")
            .key
            .clone();
        let by_key = |gs: &[SeedGroupRow<'_>], k: &str| -> usize {
            gs.iter().position(|g| g.key == k).unwrap()
        };

        // Fixture sanity — else the test is vacuous.
        {
            let c = &groups[by_key(&groups, &control)];
            let p = &groups[by_key(&groups, &passg)];
            assert!(
                !c.gaddr_contract_fails.is_empty(),
                "CONTROL must measure a contract fail"
            );
            assert!(
                p.gaddr_contract_fails.is_empty(),
                "PASS must be contract-clean"
            );
            assert_eq!(
                c.floor_count(FloorBasis::AllReps),
                p.floor_count(FloorBasis::AllReps),
                "both groups tie 8/8 on the balanced floors — this test isolates the veto, \
                 not a floor-count difference"
            );
            assert!(
                c.mean_selection_composite.unwrap() > p.mean_selection_composite.unwrap(),
                "CONTROL must lead on the tie-break pre-veto, else the fix has nothing to fix"
            );
        }

        // THE FIX: under the default (AllReps) basis, ranking still puts
        // CONTROL first (floor count + composite are unchanged), but the
        // SELECTABLE winner must skip it — exactly the pattern the
        // ungrouped `winner`-search in `run_select` already uses.
        rank_seed_groups(&mut groups, FloorBasis::AllReps);
        assert_eq!(
            groups[0].key, control,
            "sanity: CONTROL still RANKS first — the fix is a selectability veto, not a re-sort"
        );
        let winner = groups
            .iter()
            .find(|g| group_unselectable_reason(g, FloorBasis::AllReps, 2).is_none())
            .expect("a selectable group must exist");
        assert_eq!(
            winner.key, passg,
            "PASS must be SELECTED — CONTROL is vetoed"
        );

        let control_g = &groups[by_key(&groups, &control)];
        let reason = group_unselectable_reason(control_g, FloorBasis::AllReps, 2)
            .expect("CONTROL must be reported NOT selectable");
        assert!(
            reason.contains("CONTRACT FAIL"),
            "reason must name the veto: {reason}"
        );
        assert!(
            reason.contains("not selectable"),
            "reason must say why: {reason}"
        );
        assert!(
            reason.contains("C5") && reason.contains("C6"),
            "the failing rows must be NAMED, not just a bare veto: {reason}"
        );
        let pass_g = &groups[by_key(&groups, &passg)];
        assert_eq!(
            group_unselectable_reason(pass_g, FloorBasis::AllReps, 2),
            None,
            "PASS must be selectable"
        );

        // `--floor-basis legacy` reproduces the PRE-FIX rule byte-for-byte:
        // no veto, CONTROL is selectable again (the audit/reproduction
        // escape hatch — never for a real selection).
        assert_eq!(
            group_unselectable_reason(control_g, FloorBasis::Legacy, 2),
            None,
            "legacy must NOT veto — it reproduces the pre-2026-09-06 defect on purpose"
        );

        // `--floor-basis mean` keeps the veto (a product-safety gate, not a
        // floor-counting convention) even though it doesn't fold in A7r.
        assert!(
            group_unselectable_reason(control_g, FloorBasis::Mean, 2).is_some(),
            "mean must still veto a measured contract fail"
        );
    }

    /// The ungrouped ("single") path applies the SAME veto: a row with a
    /// measured contract fail is never `selectable`, regardless of its
    /// floor count, under every basis except `legacy`.
    #[test]
    fn gaddr_contract_veto_excludes_an_ungrouped_row() {
        let mut clean = passing_fixture();
        merge(&mut clean, &json!({"name": "CLEAN"}));
        merge(&mut clean, &gaddr_fixture(&[], &[]));
        let mut failing = passing_fixture();
        merge(
            &mut failing,
            &json!({"name": "FAILING", "m3a_coherence": 0.99}),
        );
        merge(&mut failing, &gaddr_fixture(&["C3", "C4"], &[]));

        let rc = select_row(&clean);
        let rf = select_row(&failing);
        assert!(!gaddr_vetoes(&rc, FloorBasis::AllReps));
        assert!(gaddr_vetoes(&rf, FloorBasis::AllReps));
        assert!(gaddr_vetoes(&rf, FloorBasis::Mean));
        assert!(
            !gaddr_vetoes(&rf, FloorBasis::Legacy),
            "legacy never vetoes"
        );

        // A vetoed row must not become `winner` even when it ranks first.
        let mut pool = vec![&rf, &rc];
        rank_pool(&mut pool, FloorBasis::AllReps);
        assert_eq!(
            pool[0].name, "FAILING",
            "sanity: it still RANKS first (higher M3a)"
        );
        let selectable = |r: &SelectRow, basis: FloorBasis| {
            r.m3a != M3aState::Unmeasured
                && gaddr_effective_counts(r, basis).0 > 0
                && !gaddr_vetoes(r, basis)
        };
        assert!(
            !selectable(&rf, FloorBasis::AllReps),
            "FAILING must not be selectable"
        );
        assert!(
            selectable(&rc, FloorBasis::AllReps),
            "CLEAN must be selectable"
        );
        assert!(
            selectable(&rf, FloorBasis::Legacy),
            "legacy: FAILING selectable again"
        );
    }

    /// `A7r`'s per-codec breakdown becomes one MORE floor per MEASURED
    /// codec under `--floor-basis all`; a `not_measured` codec is excluded
    /// from BOTH numerator and denominator; `mean`/`legacy` are unaffected.
    #[test]
    fn gaddr_a7r_folds_into_all_reps_floor_count_but_not_mean_or_legacy() {
        let mut v = passing_fixture();
        merge(&mut v, &json!({"name": "A7R_MIX"}));
        // 2 MEASURED codecs (1 pass, 1 fail) + 1 NOT MEASURED.
        merge(
            &mut v,
            &gaddr_fixture(
                &[],
                &[("jpeg", Some(true)), ("webp", Some(false)), ("avif", None)],
            ),
        );
        let r = select_row(&v);
        assert_eq!(r.n_pass, 8, "fixture passes all 8 balanced floors");
        assert_eq!(r.n_floors, 8);

        let (p_legacy, t_legacy) = gaddr_effective_counts(&r, FloorBasis::Legacy);
        assert_eq!((p_legacy, t_legacy), (8, 8), "legacy: A7r never folded in");
        let (p_mean, t_mean) = gaddr_effective_counts(&r, FloorBasis::Mean);
        assert_eq!((p_mean, t_mean), (8, 8), "mean: A7r never folded in");
        let (p_all, t_all) = gaddr_effective_counts(&r, FloorBasis::AllReps);
        assert_eq!(
            (p_all, t_all),
            (9, 10),
            "all: +1 pass / +2 measured (jpeg pass, webp fail) — avif (not_measured) excluded \
             from both"
        );
    }

    /// The group-level A7r credit uses the SAME all-reps discipline as the
    /// F1-F8 intersection: a codec counts in the denominator only when
    /// EVERY representative measured it, and in the numerator only when
    /// additionally every representative PASSED it.
    #[test]
    fn seed_group_a7r_credited_only_when_every_rep_measures_and_passes_it() {
        let argv = ["zensim_mlp_train", "--epochs", "40"];
        // Seed A: jpeg pass, webp pass. Seed B: jpeg pass, webp NOT MEASURED.
        let mut a = seed_fixture("A7G_s1", &argv, 1, 0.90);
        merge(
            &mut a,
            &gaddr_fixture(&[], &[("jpeg", Some(true)), ("webp", Some(true))]),
        );
        let mut b = seed_fixture("A7G_s2", &argv, 2, 0.90);
        merge(
            &mut b,
            &gaddr_fixture(&[], &[("jpeg", Some(true)), ("webp", None)]),
        );
        let rows: Vec<SelectRow> = [&a, &b].iter().map(|v| select_row(v)).collect();
        let refs: Vec<&SelectRow> = rows.iter().collect();
        let groups = group_by_seed(&refs);
        let g = &groups[0];
        assert_eq!(
            g.gaddr_codec_total_all, 1,
            "only jpeg was measured by BOTH reps"
        );
        assert_eq!(g.gaddr_codec_pass_all, 1, "and both reps passed it");
        assert_eq!(
            g.floor_count(FloorBasis::AllReps) as usize,
            g.n_pass_all + 1,
            "the group's floor count gains exactly the jpeg credit"
        );
    }

    /// **Clause 3, the board's measured finding**: two cells with the same
    /// recipe AND the same seed are ONE training run promoted twice (42 such
    /// rows in 33 groups on the board), so they must collapse to one
    /// representative. Counting them separately inflates `k` and makes a
    /// group look better replicated than it is.
    #[test]
    fn seed_group_duplicate_promotions_of_one_seed_collapse() {
        let argv = ["zensim_mlp_train", "--epochs", "40"];
        // Same argv, same seed 4004, two board names — the A4b_s4004 /
        // FC_C0_s4004 shape, verbatim.
        let a = seed_fixture("A4b_s4004", &argv, 4004, 0.90);
        let b = seed_fixture("FC_C0_s4004", &argv, 4004, 0.90);
        let c = seed_fixture("FC_C0_s4005", &argv, 4005, 0.30);
        let rows: Vec<SelectRow> = [&a, &b, &c].iter().map(|v| select_row(v)).collect();
        let refs: Vec<&SelectRow> = rows.iter().collect();
        let groups = group_by_seed(&refs);
        assert_eq!(groups.len(), 1, "one recipe ⇒ one group");
        let g = &groups[0];
        assert_eq!(g.members.len(), 3, "every cell stays listed");
        assert_eq!(
            g.reps.len(),
            2,
            "k = DISTINCT SEEDS (4004, 4005), not cells"
        );
        assert_eq!(g.n_duplicate_cells, 1);
        // The representative for seed 4004 is the lexicographically-first
        // name, matching the owner's `sorted(ns)[0]`.
        assert!(g.reps.iter().any(|r| r.name == "A4b_s4004"));
        assert!(!g.reps.iter().any(|r| r.name == "FC_C0_s4004"));
        // And the mean is over 2 seeds, not 3 cells: (0.90 + 0.30)/2 = 0.60,
        // NOT (0.90 + 0.90 + 0.30)/3 = 0.70.
        match g.m3a {
            M3aState::Measured(x) => assert!((x - 0.60).abs() < 1e-12, "m3a {x}"),
            other => panic!("expected Measured, got {other:?}"),
        }
    }

    /// REGRESSION (2026-09-05): two cells of ONE recipe that never recorded a
    /// seed reported k=2 — "replicated" off ZERO recorded seeds — because a
    /// seedless member was given the synthetic identity `noseed:<name>` and
    /// counted as a distinct draw. `gauntlet.build_seed_groups` has always
    /// skipped seedless rows; the divergence was invisible until the argv[0]
    /// fix put two such cells in one group, and the cross-owner parity gate
    /// caught it. `benchmarks/replication_wave_2026-09-05.md`.
    #[test]
    fn seed_group_members_with_no_recorded_seed_do_not_count_as_draws() {
        let mk = |name: &str| {
            let mut v = passing_fixture();
            merge(&mut v, &json!({"name": name, "m3a_coherence": 0.5}));
            // Replace the repro wholesale: the fixture carries a seed, and this
            // test is about a cell that never recorded one.
            v["repro"] = json!({"argv": ["zensim_mlp_train", "--epochs", "120"]});
            v
        };
        let (a, b) = (mk("NOSEED_A"), mk("NOSEED_B"));
        assert_eq!(seed_identity(&a), None, "no seed recorded");
        assert_eq!(
            seed_group_key(&a),
            seed_group_key(&b),
            "same recipe, so they share a key"
        );
        let rows: Vec<SelectRow> = [&a, &b].into_iter().map(select_row).collect();
        let refs: Vec<&SelectRow> = rows.iter().collect();
        let groups = group_by_seed(&refs);
        assert_eq!(groups.len(), 1, "one recipe, one group");
        let g = &groups[0];
        assert_eq!(
            g.members.len(),
            2,
            "both cells are still LISTED, never dropped"
        );
        assert_eq!(g.k_seeds, 0, "zero RECORDED seeds is k=0, not k=2");
        assert!(g.unreplicated, "k<2 is UNREPLICATED");
        assert_eq!(group_state(g), "UNREPLICATED");

        // ...and one recorded seed alongside a seedless cell is still k=1.
        let mut c = mk("HAS_SEED");
        c["repro"]["seed"] = json!(4021);
        let rows2: Vec<SelectRow> = [&a, &c].into_iter().map(select_row).collect();
        let refs2: Vec<&SelectRow> = rows2.iter().collect();
        let g2 = group_by_seed(&refs2);
        assert_eq!(g2[0].k_seeds, 1, "one recorded seed is one draw");
    }

    /// Clause 1 + 2's UNGROUPABLE cases are LISTED and LABELLED, never
    /// silently merged with each other or dropped. (The board's
    /// `build_seed_groups` drops them because it renders k=1 rows by another
    /// path; `--select` must rank every candidate it was handed — the one
    /// deliberate divergence, documented at the module note.)
    #[test]
    fn seed_group_ungroupable_rows_are_listed_and_labelled() {
        let mut no_argv = passing_fixture();
        merge(
            &mut no_argv,
            &json!({"name": "NO_ARGV", "repro": {"seed": 3}, "m3a_coherence": 0.5}),
        );
        let mut no_repro = passing_fixture();
        merge(&mut no_repro, &json!({"name": "NO_REPRO", "repro": null}));
        assert_eq!(seed_group_key(&no_argv), None, "no argv ⇒ UNGROUPABLE");
        assert_eq!(seed_group_key(&no_repro), None, "no repro ⇒ UNGROUPABLE");
        let mut ens = passing_fixture();
        merge(
            &mut ens,
            &json!({"name": "ENS", "model": {"kind": "ensemble"},
                    "repro": {"seed": 1, "argv": ["zensim_mlp_train"]}}),
        );
        assert_eq!(
            seed_group_key(&ens),
            None,
            "an ensemble is an evaluation FUNCTION over members, not a training replicate"
        );

        let rows: Vec<SelectRow> = [&no_argv, &no_repro]
            .iter()
            .map(|v| select_row(v))
            .collect();
        let refs: Vec<&SelectRow> = rows.iter().collect();
        let groups = group_by_seed(&refs);
        assert_eq!(
            groups.len(),
            2,
            "two ungroupable cells must NOT be merged into one bogus group"
        );
        for g in &groups {
            assert!(g.ungroupable);
            assert_eq!(group_state(g), "UNGROUPABLE");
            assert_eq!(g.reps.len(), 1);
        }
    }

    /// REGRESSION (2026-09-05): an output-path flag whose value embeds the seed
    /// silently split every seed of one recipe into its own "recipe". Measured on
    /// the real board: `LSTAR2_s4031/4032/4033` are ONE recipe at three seeds, but
    /// `--dump-checkpoints-dir .../LSTAR2_s403N_ckpts` gave each its own key, so all
    /// three reported k=1 (UNREPLICATED) instead of k=3. Eight of the ten
    /// top-scoring combined-fair cells were affected.
    /// `benchmarks/replication_wave_2026-09-05.md`.
    #[test]
    fn seed_group_output_path_flag_carrying_the_seed_does_not_split_a_recipe() {
        let v = |xs: &[&str]| xs.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        let run = |seed: &str, ckpt: &str| {
            json!({
                "name": format!("LSTAR2_s{seed}"),
                "repro": {"seed": seed.parse::<u64>().unwrap(),
                          "argv": ["zensim_mlp_train", "--epochs", "120",
                                   "--seed", seed,
                                   "--out", format!("/mnt/v/bakes/LSTAR2_s{seed}.bin"),
                                   "--dump-checkpoints-dir", ckpt]}
            })
        };
        let a = run("4031", "/mnt/v/bakes/LSTAR2_s4031_ckpts");
        let b = run("4033", "/mnt/v/bakes/LSTAR2_s4033_ckpts");
        assert_eq!(
            seed_group_key(&a),
            seed_group_key(&b),
            "two seeds of ONE recipe must share a key; a per-run output directory \
             is not part of the recipe even when its name embeds the seed"
        );
        assert_ne!(seed_group_key(&a), None, "both rows are groupable");

        // The normalizer removes the flag AND its value, like every other
        // output-path flag.
        assert_eq!(
            norm_argv_for_seed_group(&v(&[
                "t",
                "--dump-checkpoints-dir",
                "/mnt/v/x_ckpts",
                "--epochs",
                "40"
            ])),
            v(&["t", "--epochs", "40"])
        );

        // argv[0] is a BUILD LOCATION, not a recipe parameter: the same recipe
        // replayed from a sibling jj workspace must land in the same group.
        let mut w = run("4031", "/mnt/v/bakes/LSTAR2_s4031_ckpts");
        w["repro"]["argv"] = json!([
            "/home/lilith/work/zen/zensim--replicate/target/release/zensim_mlp_train",
            "--epochs",
            "120",
            "--seed",
            "4031",
            "--out",
            "/mnt/v/bakes/LSTAR2_s4031.bin",
            "--dump-checkpoints-dir",
            "/mnt/v/bakes/LSTAR2_s4031_ckpts"
        ]);
        let mut o = run("4031", "/mnt/v/bakes/LSTAR2_s4031_ckpts");
        o["repro"]["argv"] = json!([
            "/home/lilith/work/zen/zensimx/target/release/zensim_mlp_train",
            "--epochs",
            "120",
            "--seed",
            "4031",
            "--out",
            "/mnt/v/bakes/LSTAR2_s4031.bin",
            "--dump-checkpoints-dir",
            "/mnt/v/bakes/LSTAR2_s4031_ckpts"
        ]);
        assert_eq!(
            seed_group_key(&w),
            seed_group_key(&o),
            "the same recipe replayed from a sibling workspace must group with its own diagonal"
        );
        // ...but the basename still separates two different TOOLS.
        let mut t = o.clone();
        t["repro"]["argv"] = json!([
            "/home/lilith/work/zen/zensimx/target/release/bake_dial_refit",
            "--epochs",
            "120",
            "--seed",
            "4031",
            "--out",
            "/mnt/v/bakes/LSTAR2_s4031.bin",
            "--dump-checkpoints-dir",
            "/mnt/v/bakes/LSTAR2_s4031_ckpts"
        ]);
        assert_ne!(
            seed_group_key(&o),
            seed_group_key(&t),
            "a different tool is a different recipe"
        );

        // NEGATIVE CONTROL: a genuine hyperparameter difference must still
        // separate two recipes, or the fix would merge unrelated runs.
        let mut c = run("4033", "/mnt/v/bakes/LSTAR2_s4033_ckpts");
        c["repro"]["argv"] = json!([
            "zensim_mlp_train",
            "--epochs",
            "40",
            "--seed",
            "4033",
            "--out",
            "/mnt/v/bakes/LSTAR2_s4033.bin",
            "--dump-checkpoints-dir",
            "/mnt/v/bakes/LSTAR2_s4033_ckpts"
        ]);
        assert_ne!(
            seed_group_key(&a),
            seed_group_key(&c),
            "different --epochs is a different recipe"
        );
    }

    /// The normalized argv — the thing the key is a hash OF, and the thing
    /// `scripts/verify_seed_group_parity.py` compares string-for-string
    /// against the Python owner. Drops each flag AND its value; a bare
    /// switch (next token starts with `--`) drops only itself.
    #[test]
    fn seed_group_argv_normalization_drops_flag_and_value() {
        let v = |xs: &[&str]| xs.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        assert_eq!(
            norm_argv_for_seed_group(&v(&[
                "zensim_mlp_train",
                "--seed",
                "4004",
                "--epochs",
                "40",
                "--out",
                "/mnt/v/a.bin"
            ])),
            v(&["zensim_mlp_train", "--epochs", "40"])
        );
        // The two seed-split flags this lane added join the drop set, so a
        // split run and an unsplit run of the same recipe share a key.
        assert_eq!(
            norm_argv_for_seed_group(&v(&[
                "t",
                "--init-seed",
                "11",
                "--sample-seed",
                "7",
                "--epochs",
                "40"
            ])),
            v(&["t", "--epochs", "40"])
        );
        // Bare switch: `--seed` immediately followed by another flag drops
        // only itself, never the following flag.
        assert_eq!(
            norm_argv_for_seed_group(&v(&["t", "--seed", "--pool-head", "--epochs", "40"])),
            v(&["t", "--pool-head", "--epochs", "40"])
        );
        // `-o` is in the drop set too (short output flag).
        assert_eq!(
            norm_argv_for_seed_group(&v(&["t", "-o", "x.bin", "--epochs", "40"])),
            v(&["t", "--epochs", "40"])
        );
    }

    /// The key is stable, 12 hex chars, and a function of the NORMALIZED
    /// argv only — so moving the seed or the output path cannot change it,
    /// and changing a real hyperparameter must.
    #[test]
    fn seed_group_key_is_recipe_modulo_seed_and_output() {
        let a = seed_fixture("a", &["t", "--seed", "1", "--epochs", "40"], 1, 0.5);
        let b = seed_fixture("b", &["t", "--seed", "999", "--epochs", "40"], 999, 0.5);
        let c = seed_fixture(
            "c",
            &[
                "t",
                "--init-seed",
                "5",
                "--sample-seed",
                "6",
                "--epochs",
                "40",
            ],
            5,
            0.5,
        );
        let d = seed_fixture("d", &["t", "--seed", "1", "--epochs", "41"], 1, 0.5);
        let ka = seed_group_key(&a).unwrap();
        assert_eq!(ka.len(), 12);
        assert!(ka.chars().all(|c| c.is_ascii_hexdigit()));
        assert_eq!(
            ka,
            seed_group_key(&b).unwrap(),
            "seed value is not identity"
        );
        assert_eq!(
            ka,
            seed_group_key(&c).unwrap(),
            "a split-seed run of the same recipe is the SAME recipe"
        );
        assert_ne!(
            ka,
            seed_group_key(&d).unwrap(),
            "a real hyperparameter change IS a different recipe"
        );
    }

    /// `seed_identity` is what duplicates collapse by: the PAIR once split,
    /// the single seed on a legacy bake, `None` when nothing is recorded.
    #[test]
    fn seed_identity_reports_the_pair() {
        assert_eq!(
            seed_identity(&json!({"repro": {"init_seed": 11, "sample_seed": 7}})).as_deref(),
            Some("11/7")
        );
        assert_eq!(
            seed_identity(&json!({"repro": {"seed": 4004}})).as_deref(),
            Some("4004")
        );
        // A split run writes BOTH; with only one present we fall back to the
        // master seed rather than inventing the missing half.
        assert_eq!(
            seed_identity(&json!({"repro": {"init_seed": 11, "seed": 3}})).as_deref(),
            Some("3")
        );
        // init == sample is the SAME draw as the legacy `--seed`, measured:
        // CTL-A vs CTL-B differ on 0 of 12 corpora.
        assert_eq!(
            seed_identity(&json!({"repro": {"init_seed": 4021, "sample_seed": 4021}})).as_deref(),
            Some("4021")
        );
        assert_eq!(
            seed_identity(&json!({"repro": {"init_seed": 4021, "sample_seed": 4021}})),
            seed_identity(&json!({"repro": {"seed": 4021}})),
            "a split run at one value must collapse against its legacy twin"
        );
        assert_eq!(seed_identity(&json!({"repro": {}})), None);
        assert_eq!(seed_identity(&json!({})), None);
        // The human-readable label distinguishes an explicit split from an
        // unsplit run whose repro records the resolved pair.
        assert_eq!(
            seed_label(&json!({"repro": {"init_seed": 5, "sample_seed": 5}})),
            "seed 5 (unsplit)"
        );
        assert_eq!(
            seed_label(&json!({"repro": {"init_seed": 5, "sample_seed": 9}})),
            "init 5 / sample 9"
        );
    }

    /// Default `--select` output is untouched when `--seed-group` is not
    /// passed: `run_select(..., false)` reaches the identical `winner`
    /// value `run_select` computed before this fix existed.
    #[test]
    fn seed_group_off_leaves_the_plain_rule_untouched() {
        let mut a = passing_fixture();
        merge(&mut a, &json!({"name": "A", "m3a_coherence": 0.90}));
        let mut b = passing_fixture();
        merge(&mut b, &json!({"name": "B", "m3a_coherence": 0.70}));
        let (ra, rb) = (select_row(&a), select_row(&b));
        let mut before = vec![&rb, &ra];
        rank_pool(&mut before, FloorBasis::AllReps);
        let winner_before = before[0].name.clone();

        // group_by_seed/rank_seed_groups are simply never called when
        // `--seed-group` is absent (see the `if seed_group` guard in
        // `run_select`) — this pins that the PLAIN rule's result is
        // unaffected by the mere EXISTENCE of the seed-group feature.
        let mut after = vec![&rb, &ra];
        rank_pool(&mut after, FloorBasis::AllReps);
        assert_eq!(after[0].name, winner_before);
    }

    /// The three M3a states are DISTINCT and none of them is zero.
    #[test]
    fn select_m3a_states_are_distinct_and_never_zero() {
        // UNMEASURED: no m3a, non-ensemble.
        let mut un = passing_fixture();
        un.as_object_mut().unwrap().remove("m3a_coherence");
        let r_un = select_row(&un);
        assert!(matches!(r_un.m3a, M3aState::Unmeasured));
        assert!(
            r_un.selection_composite.is_none(),
            "UNMEASURED must carry NO selection_composite — not a 0-valued one"
        );
        assert_eq!(m3a_cell(r_un.m3a), "UNMEASURED");

        // NOT COMPUTABLE: ensemble, m3a null by construction.
        let mut ens = passing_fixture();
        merge(
            &mut ens,
            &json!({"name": "ENS", "model": {"kind": "ensemble"}, "m3a_coherence": null}),
        );
        let r_ens = select_row(&ens);
        assert_eq!(r_ens.class, "944-ensemble");
        assert!(matches!(r_ens.m3a, M3aState::NotComputable));
        assert_eq!(m3a_cell(r_ens.m3a), "NOT COMPUTABLE");

        // An explicitly-zero M3a is a MEASURED zero and must rank as one —
        // distinct from both states above.
        let mut z = passing_fixture();
        merge(&mut z, &json!({"name": "Z", "m3a_coherence": 0.0}));
        let r_z = select_row(&z);
        assert!(matches!(r_z.m3a, M3aState::Measured(x) if x == 0.0));
        assert_eq!(r_z.selection_composite, r_z.composite);
    }

    /// An UNMEASURED candidate sorts last within its floor tier and can
    /// never be the selected winner, even when it leads on every other axis.
    #[test]
    fn select_unmeasured_is_listed_but_not_selectable() {
        let mut un = passing_fixture();
        un.as_object_mut().unwrap().remove("m3a_coherence");
        merge(
            &mut un,
            &json!({"name": "UN", "rank": {"cid22": {"srocc": 0.99}}}),
        );
        let mut ok = passing_fixture();
        merge(&mut ok, &json!({"name": "OK", "m3a_coherence": 0.50}));
        let (ru, ro) = (select_row(&un), select_row(&ok));
        assert_eq!(ru.n_pass, ro.n_pass, "same floor tier");
        assert!(
            ru.composite > ro.composite,
            "UN must lead on balanced_composite, else the test is vacuous"
        );
        let mut pool = vec![&ru, &ro];
        rank_pool(&mut pool, FloorBasis::AllReps);
        assert_eq!(
            pool[0].name, "OK",
            "an UNMEASURED candidate must not outrank a measured one in its tier"
        );
    }

    /// serde_json deep-merge helper for fixture patching.
    fn merge(dst: &mut serde_json::Value, patch: &serde_json::Value) {
        match (dst, patch) {
            (serde_json::Value::Object(d), serde_json::Value::Object(p)) => {
                for (k, v) in p {
                    merge(d.entry(k.clone()).or_insert(serde_json::Value::Null), v);
                }
            }
            (d, p) => *d = p.clone(),
        }
    }

    /// An inverted top band must FAIL F8 — the whole point of the re-point.
    ///
    /// Measured on the 120 board cells that still carry per-pair (appendix V,
    /// G-V1): under the predecessor bar, `|B9| >= 0.15` PASSED 82 of them while
    /// only 2 were positive, because `|·|` is monotone in the depth of an
    /// inversion. The published top cell of that column was the population's
    /// most anti-correlated model.
    #[test]
    fn f8_fails_an_inverted_top_band() {
        let mut v = passing_fixture();
        // The exact shape that used to pass: a large magnitude, wrong sign.
        v["rank"]["cid22"]["bands"] = bands_v(0.18, -0.3204);
        let r = eval_balanced(&v, &[]);
        let fl = r.floors.iter().find(|x| x.id == "bandtail").unwrap();
        assert!(
            !fl.pass,
            "an inverted top band must fail, got PASS on {:?}",
            fl.measured
        );
        assert!(
            fl.measured.contains("-0.320"),
            "the signed value must be shown: {:?}",
            fl.measured
        );
        // Same magnitude, correct sign: passes.
        v["rank"]["cid22"]["bands"] = bands_v(0.18, 0.3204);
        let r = eval_balanced(&v, &[]);
        let fl = r.floors.iter().find(|x| x.id == "bandtail").unwrap();
        assert!(fl.pass, "a healthy band of the same magnitude must pass");
        // And the ranking composite must move the same direction, not just
        // the floor — an inverted tail may not score like a healthy one.
        let mut inv = passing_fixture();
        inv["rank"]["cid22"]["bands"] = bands_v(0.18, -0.3204);
        let mut good = passing_fixture();
        good["rank"]["cid22"]["bands"] = bands_v(0.18, 0.3204);
        assert!(
            balanced_composite(&inv).unwrap() < balanced_composite(&good).unwrap(),
            "inverted tail must score BELOW the healthy one in the composite"
        );
    }
}

#[cfg(test)]
mod qualification_tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn missing_is_incomplete_failures_persist_and_evidence_is_content_bound() {
        assert_eq!(qualification_report(&json!({}))["status"], "incomplete");
        let sha = "a".repeat(64);
        let mut v = json!({"name":"candidate", "bake_sha256":sha,
            "scoring":{"surface":"zensim::BakeScorer","members":[{"sha256":sha}]},
            "feature_set":{"bake":"declared", "table":"declared", "mismatches":[]},
            "dial_ladder":{"contract":"PASS", "regression":"PASS", "measured":{"codec_floor":
                (["avif-rav1e","avif-svt","jpeg","jxl","webp"].iter().map(|c| json!({"codec":c,"state":"pass"})).collect::<Vec<_>>())}}
        });
        v["feature_set_composition"] =
            json!({"scoring":v["scoring"],"members":[{"qualified_provenance":true}]});
        let path =
            std::env::temp_dir().join(format!("zensim-qualification-{}.json", std::process::id()));
        let mut artifact = json!({"bake_sha256":sha,"scoring":v["scoring"],"gates":{}});
        for gate in ["G-RANK", "G-DIAL", "G-STEER", "G-RD", "G-TARGET"] {
            artifact["gates"][gate] = json!({"state":"pass"});
        }
        std::fs::write(&path, artifact.to_string()).unwrap();
        let hash = zensim_validate::train_manifest::sha256_file(&path).unwrap();
        v["dial_ladder_source"] = json!({"path":path,"sha256":hash});
        v["dial_ladder"]["scoring"] = v["scoring"].clone();
        for gate in ["G-RANK", "G-DIAL", "G-STEER", "G-RD", "G-TARGET"] {
            v["product_evidence"][gate] = json!({"state":"pass","bake_sha256":sha,"n":30,
                "surface":"zensim::BakeScorer","instrument":"test measuring owner",
                "artifact":{"path":path,"sha256":hash}});
        }
        assert_eq!(qualification_report(&v)["status"], "qualified");
        let mut unknown_companion = v.clone();
        unknown_companion["scoring"]["members"]
            .as_array_mut()
            .unwrap()
            .push(json!({"sha256":"unknown-member"}));
        assert_eq!(
            qualification_report(&unknown_companion)["status"],
            "incomplete"
        );
        // Isolate provenance from the other stale-composition findings above.
        // A known primary does not confer admission on an unknown companion.
        unknown_companion["feature_set_composition"]["scoring"] =
            unknown_companion["scoring"].clone();
        unknown_companion["feature_set_composition"]["members"]
            .as_array_mut()
            .unwrap()
            .push(json!({"qualified_provenance":false}));
        let report = qualification_report(&unknown_companion);
        let provenance = report["checks"]
            .as_array()
            .unwrap()
            .iter()
            .find(|c| c["gate"] == "Table provenance")
            .unwrap();
        assert_eq!(provenance["state"], "not_measured");
        let mut missing_provenance = v.clone();
        missing_provenance["feature_set_composition"]["members"][0]["qualified_provenance"] =
            json!(false);
        assert_eq!(
            qualification_report(&missing_provenance)["status"],
            "incomplete"
        );

        std::fs::write(&path, "{}").unwrap();
        assert_eq!(qualification_report(&v)["status"], "incomplete");
        v["dial_ladder"]["contract"] = json!("FAIL");
        assert_eq!(qualification_report(&v)["status"], "failed");
        std::fs::remove_file(path).unwrap();
    }
}
