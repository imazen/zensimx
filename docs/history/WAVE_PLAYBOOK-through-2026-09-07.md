> Historical snapshot archived September 7, 2026. Current workflow: [WAVE_PLAYBOOK](../WAVE_PLAYBOOK.md). Later rulings and current code supersede these dated commands and statuses. Exact original bytes remain in Git at `9879bbfd53b8`; relative links here are adjusted.

# WAVE_PLAYBOOK — the skeleton every R&D wave uses

**Read this before launching any wave, arm, or sweep that runs detached compute.**

**User ruling 2026-09-07:** before introducing/evaluating a new model, make its
entire inference path executable and servable in Rust through a zensim surface
API. Evaluation uses that API, including heads, corruption gates and splines.
Python can invent the method; a private evaluator is not model admission.
B/C/D may be replaced (no consumers are calibrated to them). Consult
`../zenpapers` and later dated summaries for fundamental ideas. Prioritize
feature ablations only when they remove actual compute or settle a named
scientific failure. See [the authorized cleanup plan](../PLAN_CRUFT_PURGE_2026-09-06.md).

Every rule here is priced. The costs come from
[`benchmarks/rnd_cycle_audit_2026-08-04.md`](../../benchmarks/rnd_cycle_audit_2026-08-04.md),
which measured the 2026-08-03/04 campaign: 34.3 h of wall-clock, 11 waves,
**14.80 h of whole-session idle of which 6.77 h was dead** — nothing computing,
or finished work nobody had looked at — and **$395.24 (13.9 % of a $2,837
session) burned re-creating prompt cache that idle waiting had expired.**

The compute was fine. The orchestration around it was the whole loss.

---

## The skeleton

```
1. PRE-REGISTER      arms, seeds, gates, decision rule -> commit BEFORE launching
                     ... INCLUDING scripts/endgame_<wave>.sh (committed, idempotent)
2. BUILD ONCE        build the binaries; export ZL_* pointers for every consumer
3. LAUNCH DETACHED   setsid nohup <driver>            (one lane per executor)
4. HARVEST INLINE    setsid nohup scripts/harvest_bakes.sh --glob ... --count N
5. ONE TERMINAL      setsid nohup scripts/await_artifacts.sh --glob ... --count N \
       + ENDGAME         --then 'scripts/endgame_<wave>.sh'
                     ... then Monitor exactly ONE file: <heartbeat>.endgame.done
6. SELECT            freeze_check --select <every fulleval> [--tsv]
                     (the endgame script runs this — see below. The
                      REPLICATION FLOOR is on by default: k>=2 or not
                      selectable, so plan k>=2 seeds per arm.)
7. REVIEW FOREGROUND read the endgame's tables + doc DRAFT; judge; finalize
8. PUSH + VERIFY     scripts/safe_push.sh -r <reviewed-revision>
                     scripts/verify_push.sh <sha>   # verify the reviewed commit
9. CLOSE OUT        record artifacts and remaining decisions; preserve work
```

**Updated 2026-09-07 to match the September 5–6 push rule in `CLAUDE.md`.**
The old direct bookmark-move sequence could discard other lanes' commits.
Use `safe_push.sh`; if it refuses, reconcile ancestry rather than overriding
the refusal. Follow the workspace `AGENTS.md` for checkout cleanup: do not
create worktrees, and rescue modified/untracked files before any removal.
This playbook is not authorization to push or delete unrelated work.

`scripts/verify_push.sh` (appendix W, C4) is the required form of step 8's
verification: it fetches, ancestry-tests, and prints ONE line
(`VERIFY-PUSH OK <full-sha> is-ancestor-of origin/main checked=<ts>`) that a
sub-agent's report must paste verbatim and a supervisor re-runs — it works
from secondary jj workspaces too, where bare `git` silently fails. "The agent
said it pushed" is not evidence; this line is. For process waits, source
`scripts/lib/proc.sh` (C7) instead of hand-rolling pgrep — `pgrep -f`
self-matches your own wrapper shell and has burned multi-hour waits twice.

Steps 4 and 5 make a late wake-up **free** and a dead waiter **visible**. The
`--then` endgame (added 2026-08-05) makes the wake-up itself **optional**: the
detached driver executes the endgame, so a lost notification costs review
latency only — never recompute, never a stalled wave.

---

## Step 4 — harvest inline. Never batch the eval at the end.

```bash
setsid nohup scripts/harvest_bakes.sh \
    --glob '/mnt/v/output/zensim/bakes/sota944/bakes/C_w8_s*.bin' \
    --count 12 --regime 944 \
    --heartbeat ~/tmp/wave8/harvest --timeout 21600 >/dev/null 2>&1 &
```

Each bake is verdicted + fullevaled the moment it lands. By the time anyone
looks, the results exist. **This is what makes a missed wake-up cost zero.**

> **Measured:** wave 6 arm F did exactly this with an uncommitted
> `~/tmp/wave6/process.sh`. `ALL SIX PROCESSED 03:08:40Z`; results commit
> `05:20:40Z`. The daemon was perfect and the wave still lost **125.6 min** —
> because nothing woke the agent. Harvest-inline bounds the damage; it does not
> by itself remove it. You need step 5 too.

> **Measured:** the coherence wave's driver *had* an inline auto-eval that
> exited 2 on a missing corpus, **nine times**, into a log nobody read. 3 h
> 24 min of training produced zero verdicts; 21 were re-run by hand
> (804 s). `harvest_bakes.sh` writes `<bake>.HARVEST_FAILED`, appends to
> `<heartbeat>.failures`, and exits **6** — you cannot not notice.

Check a harvest is healthy without reading a log:

```bash
cat ~/tmp/wave8/harvest.status     # rewritten every poll — a stale mtime = dead
ls  /mnt/v/.../bakes/*.HARVEST_FAILED 2>/dev/null   # empty = clean
```

## Step 5 — one terminal condition, and it is a FILE

```bash
setsid nohup scripts/await_artifacts.sh \
    --glob '/mnt/v/output/zensim/reports/fulleval/C_w8_s*.fulleval.json' \
    --count 12 --label 'wave-8 fullevals' \
    --then 'scripts/endgame_w8.sh' \
    --heartbeat ~/tmp/wave8/await --timeout 14400 >/dev/null 2>&1 &
```

Then arm **one** `Monitor` on `~/tmp/wave8/await.endgame.done` (the chain's
true terminal — artifacts *and* endgame) and go do other work.

`await_artifacts.sh` guarantees the sentinel is written on **every** exit path —
COMPLETE, TIMEOUT (rc 3), or SIGNAL (rc 5) — via an `EXIT` trap, and it sleeps
as `sleep & wait` so a signal is honoured immediately rather than after the
poll interval. SIGKILL cannot be trapped; that case shows up as a
`<heartbeat>.status` whose mtime stops advancing, which is still evidence.

**Watch the terminal artifact, not the last one.** Await the *fullevals*, not
the bakes — the bake landing is the middle of the pipeline, and an agent woken
by it still has 35 s × N of eval to run serially in the foreground.

---

## Anti-patterns, each with what it cost

| anti-pattern | measured cost | do this instead |
|---|---|---|
| Hand-rolled `while sleep` / `tail -f` waiter that exits without a trace | the day's two worst events: **125.6 min** + **80.6 min** dead | `await_artifacts.sh` — sentinel on every exit path |
| Endgame armed as an **agent wake** (Monitor/notification → agent runs tables) | 2026-08-05: FOUR orphaned lanes in one day, each stalled on a human nudge — wakes don't survive a host restart | `await_artifacts.sh --then 'scripts/endgame_<wave>.sh'` — the driver runs it |
| Re-messaging a closed agent lane ("did you finish?") | each message = ≥1 more wake+stop+tick; ~10 residual ticks observed on one lane | read its sentinels/artifacts from disk; let queues drain |
| `Monitor` armed on `tail -f <log>` | loses the file on rotation/truncate, then waits forever | `Monitor` a file that appears **exactly once** (`<hb>.done`) |
| Batching verdicts+fullevals at the end of the wave | a late wake-up costs its own delay **plus** the whole eval chain | `harvest_bakes.sh` inline |
| A post-bake hook whose failure is non-fatal | **13.4 min** recompute + a 3 h 24 min lane silently voided | fail loud: marker file + failures file + nonzero exit |
| Agent parks and polls on short intervals | polls are cheap, but the **wake-up** after >5 min re-creates the whole prefix: **$395.24, 13.9 %** of session cost | arm one Monitor, then work on something else |
| Supervisor idle-waits on a delegated wave | its context ages while a subagent's does too — **two** prefixes re-charged per event | supervisor does independent work; artifacts are the channel |
| Static split of cells across lanes | local box idle **57.5 min** while lianli held the critical path (coherence wave) | rebalance: give the slower lane fewer cells, or pull work back |
| No work queued while an agent thinks | **1.46 h** of the day with nothing running *and* nobody awake | launch the next arm before writing up the last one |
| A second `cargo` on a shared `CARGO_TARGET_DIR` | **31.8 s** blocked on the build lock, per invocation | per-agent target dirs + `ZL_*` binary handoff (below) |

---

## Builds: do not build if you only consume

Measured: total `cargo` wall-clock across every agent, all day, was **23.0 min**
(91 invocations; a cold `bake_verdict` build is 72 s / 221 crates). A shared
target dir is **not** worth it — see the audit §5. Per-agent dirs stay.

But an agent that only *runs* binaries should not build at all. Every driver
honours env pointers:

```bash
export ZL_BV=/home/lilith/work/zen/zensimx/target/release/bake_verdict
export ZL_TRAIN=/home/lilith/work/zen/zensimx/target/release/zensim_mlp_train
# or, to reuse a whole tree:
export CARGO_TARGET_DIR=$HOME/tmp/zensimw8-target
```

`sota944_verdict.sh`, `run_full_eval.sh`, `harvest_bakes.sh`,
`wave6_distill_seed.sh` and `wave7_armH_seed.sh` all resolve binaries through
`ZL_*` → `CARGO_TARGET_DIR` → `$REPO_ROOT/target`.

Target dirs are cheap in time and **expensive in disk**: 28 dirs, 113.6 GB, on
a root filesystem at 95 %. Delete yours when the wave closes.

---

## Step 6 — SELECT on rank + dial + COHERENCE, with the owner

```bash
freeze_check --select /mnt/v/output/zensim/reports/fulleval/C_w8_s*.fulleval.json
```

The wave's decision is not "eyeball the scoreboard". It is a **registered
rule**, implemented in `freeze_check` (the bar/profile owner — do NOT write a
ranking script):

- **PRIMARY: profile floor count.** Coherence never overrides a bake that
  fails CID22 or the dial.
- **TIE-BREAK: `balanced_composite + 0.15 · M3a`.** Registered in the sota944
  campaign doc appendix E.4. 0.15 is the same weight class
  `balanced_composite` already gives csiq/live/band-tail.
- **G-ADDR CONTRACT PRE-FILTER (2026-09-06 owner fix, `benchmarks/select_gaddr_prefilter_2026-09-06.md`,
  registry `select-rule-blind-to-dial-contract-2026-09-06`).** PRIMARY and TIE-BREAK are BOTH
  blind to `dial.addressability` — the best-of-all wave measured this pick `A_plain`
  (G-ADDR CONTRACT 4/6, fails C5+C6) over five arms at 6/6, because a floor count and a
  composite have no way to see a contract failure. Fixed at the owner: any candidate (or,
  under `--seed-group`, any GROUP with one member) that MEASURES a G-ADDR CONTRACT-tier
  (C1-C6) fail is now an absolute selectability VETO, independent of floor count or
  composite — listed under its own "contract FAIL — not selectable" heading, never silently
  ranked in. `--floor-basis all` (default) also folds `A7r`'s per-codec floor-representability
  into the floor count itself (one more floor per MEASURED codec — a `not_measured` codec
  counts in neither direction). `--floor-basis legacy` reproduces the PRE-2026-09-06 rule
  byte-for-byte (no veto, no A7r floors) — an audit/reproduction escape hatch ONLY, never for
  a real selection; `--floor-basis mean` keeps the veto (a product-safety gate, not a
  floor-counting convention) but not the A7r extension. **Dial addressability is a HARD ship
  gate (user rule 2026-09-04) — a `--select` pick that fails it was never a valid selection,
  regardless of how it scored on rank + dial + coherence.**
- **Why M3a is in the rule at all:** the coherence study (n = 50,
  pre-registered) measured that **42.3 % of 944-class M3a variance is seed
  noise at fixed recipe** — at fixed data, recipe and width, `C_co3a` k = 6
  spans **0.7367–0.8786** (0.142, on corrected post-`299ccc8c` values; the
  study's own 0.718–0.826 was measured before the append2 coverage fix and
  understates the spread). Coherence is a *selectable trajectory property*, so
  a k-seed wave that ignores it leaves ~0.14 of M3a on the table for free.
- **`sdr25` is NOT the primary.** It is a reported comparator column. It has
  decoupled from CID22 five times; that is exactly why the primary is the
  floor count.
- **Three M3a states, none of them zero.** `MEASURED` ranks normally;
  `NOT COMPUTABLE` (ensemble — the instrument loads one ZNPR) ranks in a
  separate section and is never penalized; `UNMEASURED` is listed but **not
  selectable**, and the tool prints the command to fix it.

### The REPLICATION FLOOR — `--min-k 2`, on by default (registered 2026-09-05)

**Plan k ≥ 2 seeds per arm, or the arm cannot be selected.** A seed group with
fewer than `--min-k` (default **2**) distinct seeds is UNREPLICATED: listed and
ranked in its own section, with the reason printed, and **never selected**.

- **Why (measured, `benchmarks/replication_wave_2026-09-05.md`).** Replicating
  the board leaders moved them DOWN — `LSTAR` 0.8615 as its best cell,
  **0.856414** as its k=7 mean; ranks 1 → 7. Best-of-k minus k-mean has a
  **+0.0061 median** over the 18 combined-fair k ≥ 2 groups, larger than the
  0.0021 that separated the top four. A one-draw number competes at its
  maximum against groups reported at their means.
- **The floor makes the SEED GROUP the unit of selection.** `--select` prints
  the grouped section whenever the floor is active (the basis of a pick is
  never hidden) and `**SELECTED:**` names a **RECIPE**, not a cell — naming the
  group's best member would be the very best-of-k pick the floor removes. The
  per-cell table is unchanged and now says `**BEST CELL:**`.
- **Floor basis `all` (default): floors EVERY seed passes, PLUS (2026-09-06) the
  `A7r` per-codec G-ADDR floors and the CONTRACT veto.** A group is credited a
  floor only when every distinct-seed representative clears it, and the report
  names the `split floors` (passed by some seeds, not all). A floor is a
  certification and a mean is not one — two members at 8/8 and 6/8 average
  7.0 even when they fail *different* floors. `--floor-basis mean` restores the
  k-seed mean count (F1-F8 only — the CONTRACT veto still applies); the mean
  is reported either way.
- **Reproducing a historical selection:** `--min-k 1 --floor-basis mean` is the
  pre-amendment rule exactly (verified: same 34 rows, same order, same values,
  same winner on the 113 combined-fair cells). `--floor-basis legacy`
  additionally reproduces the pre-2026-09-06 G-ADDR-blind rule byte-for-byte
  (no CONTRACT veto, no A7r floors) — an audit escape hatch, never for a real
  selection.
- **The remedy for an UNREPLICATED front-runner is another seed, not a lower
  floor.** Re-run the same recipe with a new seed and re-harvest.

Registration + the measured before/after: campaign appendix **E.4 AMENDMENT**.

M3a comes free with harvest (`run_full_eval.sh` computes it), so by the time
step 5 says COMPLETE the selection data already exists. If it does not,
harvest drops a `.NO_M3A` marker next to the bake and counts it in the
terminal line — read that before selecting.

---

## The restart-proof endgame — the DRIVER runs it, never a wake (2026-08-05)

**Incident of record (2026-08-05, four orphans in one day):** the featsub,
wave-11, hygiene2, and HDR lanes each finished their compute while no agent
wake arrived. Every one recovered **losslessly** — per-bake harvest + `.done`
sentinels had all state on disk — but every one needed a **manual supervisor
nudge** to run its endgame, because the thing that was supposed to wake an
agent was gone. Agent wake-chains live inside the Claude Code host process;
the compute does not.

**What actually survives a Claude Code host restart (verified Aug 2026):**

| mechanism | survives restart? | evidence |
|---|---|---|
| `setsid` OS process + sentinel files | **yes** | OS-owned; indifferent to the CLI ([durable-execution surveys](https://www.inngest.com/blog/durable-execution-key-to-harnessing-ai-agents) all land here: state on disk + idempotent reconciler) |
| `settings.json` hooks | **yes** (re-evaluated each session; `SessionStart` fires with `source: resume`) | [hooks docs](https://code.claude.com/docs/en/hooks.md) — but **no hook event fires on background-task completion**, so hooks cannot replace the driver |
| cloud Routines / scheduled agents | yes, but each trigger is a **fresh session** — no cross-invocation state, min 1 h cadence | [routines docs](https://code.claude.com/docs/en/routines.md) |
| in-session `/loop` scheduled task | partial (restored on `--resume` if ≤ 7 d) | [scheduled-tasks docs](https://code.claude.com/docs/en/scheduled-tasks.md) |
| background Bash / `Monitor` watches | **no — "never restored on resume"** | [scheduled-tasks docs, Limitations](https://code.claude.com/docs/en/scheduled-tasks.md#limitations) |
| subagent completion notifications | **no**, and buggy even in-session (zombie "running" tasks, duplicate ticks) | [#65925](https://github.com/anthropics/claude-code/issues/65925), [#58637](https://github.com/anthropics/claude-code/issues/58637), [#47930](https://github.com/anthropics/claude-code/issues/47930) |

Conclusion, and the registered design: **the "what happens next" logic must
live in the detached OS layer, not in any agent's pending wake.** The wave's
endgame is a committed script, executed by the driver the moment the terminal
condition is met:

```bash
scripts/await_artifacts.sh ... --then 'scripts/endgame_w8.sh'
```

- `--then` runs on COMPLETE (add `--then-always` to also draft a partial wave
  on TIMEOUT; a deliberate SIGNAL kill never triggers it).
- The driver writes `<heartbeat>.endgame.done` on **every** endgame exit path
  (COMPLETE / FAILED rc / SIGNAL) — the same no-silent-death contract as the
  watch sentinel. Endgame stdout+stderr land in `<heartbeat>.then.log`; a
  failed endgame makes the await exit 7.
- An agent that never wakes costs nothing but review latency. An agent that
  wakes late (or a human running `claude -r` tomorrow) finds tables + doc
  draft already on disk.

**The per-wave `endgame_<wave>.sh` contract** — committed BEFORE launch (it is
part of pre-registration; an uncommitted endgame is the wave-6 `process.sh`
failure class and `--then` warns on `~/tmp` paths), **idempotent** (re-runs
must be safe — the driver may re-run it), and it does the *bounded, judgment-
free* tail only:

```bash
#!/usr/bin/env bash
# endgame_w8.sh — wave-8 endgame: runs IN THE DRIVER on chain completion.
set -euo pipefail
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
W=~/tmp/wave8; FE=/mnt/v/output/zensim/reports/fulleval
FC=${ZL_FC:-${CARGO_TARGET_DIR:-$REPO_ROOT/target}/release/freeze_check}
# 1. selection table (the registered rule, run by the owner)
"$FC" --select "$FE"/C_w8_s*.fulleval.json --tsv > "$W/select.tsv"
# 2. doc-append DRAFT (never touches the campaign doc itself)
{ echo "## WAVE 8 RESULTS (DRAFT $(date -u +%FT%TZ) — review before folding in)";
  column -t -s$'\t' "$W/select.tsv"; } > "$W/doc_append.draft.md"
# 3. anything else bounded: gates, per-arm medians, tarball of evidence
```

The reviewer (agent or human) then does only judgment + landing: read the
draft, fold into the campaign doc, commit tables under `benchmarks/`, push,
verify, clean up. **The driver never commits or pushes** — landing needs
review by construction.

**Why not a "native" mechanism instead (evaluated 2026-08-05):** hooks have no
background-completion event to fire on; Routines are fresh-session cloud runs
that can't see the box's `/mnt/v` state mid-wave; `/loop` dies with the
session ≥ half the time we care about. The one native piece that IS worth
using is `SessionStart` (it fires on `claude -r` resume): wire
`scripts/wave_resume_probe.sh` as a SessionStart hook and every resumed
session opens already knowing what finished, failed, is live, or died
silently — no human nudge needed to *orient*. User-side snippet (optional, in
`~/.claude/settings.json`; the probe also works run by hand):

```json
"SessionStart": [{ "matcher": "*", "hooks": [{ "type": "command",
  "command": "bash /home/lilith/work/zen/zensimx/scripts/wave_resume_probe.sh" }] }]
```

## Residual notification ticks from closed lanes — determination (2026-08-05)

Observed: a closed agent lane re-woken ~10× by empty queue drains. Local
investigation found **no repairable state** — no stale `/loop` cron entries,
no persisted Monitor state (Monitors are in-process only), and the on-disk
`~/.claude/tasks/*` files are the TaskList board, not a notification queue.
The mechanism is **harness-internal**: a task-notification fires **every**
time a background lane stops with no live children — not once per lifetime —
and anything delivered into a finished lane (a `SendMessage` nudge, a queued
message drain, a resume) runs it again, so it stops again, so it ticks again.
This is the documented bug class:
[#65925](https://github.com/anthropics/claude-code/issues/65925) (zombie
"running" tasks; `TaskStop` → "No task found"),
[#47930](https://github.com/anthropics/claude-code/issues/47930) (idle-
notification loops burning 13–22 % of lead-session input tokens),
[#58637](https://github.com/anthropics/claude-code/issues/58637).

Mitigation (behavioral — there is nothing to fix locally):

1. **Never message a lane after its terminal report.** Each message to a
   closed lane buys ≥ 1 more wake+stop+tick. If you need its artifacts, read
   them from disk — files are the channel.
2. **Treat every tick as level-triggered, not edge-triggered.** A tick means
   "go look at the sentinels", never "new event happened". Idempotent
   handling (read `.done` / `.endgame.done`, act only on state you haven't
   acted on) makes duplicate ticks free.
3. **Let queues drain.** After a wave closes, expect a few residual ticks;
   answer them with a sentinel glance and silence, not with replies into the
   dead lane (which re-arm the loop).

## Review in the foreground

Once `<heartbeat>.endgame.done` says COMPLETE, everything left is judgment:
read the draft + tables (already computed), decide, fold into the doc,
commit, push, verify, clean up. **Do not arm another waiter for a bounded
review** — run it inline. The audit's `A8 balanced-selection` pass shows what
an unbounded endgame costs: a 15-minute tail on a pass whose compute was done.

Terminal checklist:

```bash
just lint-scripts                              # any script you added
scripts/safe_push.sh -r <reviewed-revision>
scripts/verify_push.sh <sha>                   # paste the VERIFY-PUSH OK line
# Record artifact locations and follow the workspace's preservation rules.
```

---

## The 30-second self-check before you go quiet

1. Is something computing **right now**? If not, why not — launch it.
2. Will the results be **fully evaluated** when they land, or will I have to
   run the eval myself after I wake?
3. Is there **exactly one** file whose appearance means "done", and is
   something watching it that leaves evidence if it dies?
4. If no wake ever arrives, will the endgame still have run? It must: the
   driver owns it (`--then`), not a notification. A wake you *need* is a
   single point of failure that a host restart deletes.
5. Am I about to wait, or about to **work on something else**? Waiting attached
   is what costs $395/day.
