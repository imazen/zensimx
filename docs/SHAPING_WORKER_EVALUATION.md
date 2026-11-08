# Evaluation after primary worker collection

`shaping_evaluation_worker.py` coordinates evaluation of newly collected primary
neural fits on their original approved worker. The coordinator reads metadata,
transfers requests and imports reports; native prediction and re-export run on
the worker under an explicit resource cap. This path currently covers inner and
fresh-inner fits, not rescue or outer-stage transport.

A request requires a successful collected fit, the approved reduced registration,
the original RUN/RESULT and training-log hashes, and every scheduled diagnostic
and final checkpoint hash. Existing coordinator evaluation artifacts prevent a
new request. The worker verifies the same inputs and uses the existing scientific
evaluation and supplemental-representation owners, restricted to that cell.
Undefined convergence remains an explicit failure; no rank or seed is imputed.

Leave one compute slot and at least seven GiB available before admitting the
three-GiB evaluation job. Keep export and import on the reserved study filesystem
with the thirty-GiB reserve. Package complete report trees and explicit terminal
diagnostics with a bounded, checksummed manifest. Import through verified staging
and publish whole trees without overwriting existing evidence. Preserve failed
or unresolved operations for reconciliation; only a capacity wait can retry
automatically, since it has not begun evaluation.

A full host is probed once per coordinator sweep. Its other pending cells stay
queued without repeated request uploads or remote capacity processes. Capacity
is checked afresh on the next sweep; failed or unresolved evaluations remain
separate and never become automatic retries.

Worker capsules carry the unchanged reduced policy as `REDUCED_PROTOCOL.md`,
analogous to the original `PROTOCOL.md`. Its hash and the installed source's
policy must both match the immutable registration. The original policy path and
all scientific registrations remain unchanged.

`just shaping-evaluation-worker-check` checks transport, cell isolation, failure
preservation and registration guards with synthetic artifacts. Those tests do
not establish a successful native worker evaluation; record the first real
import and its scientific gates separately after a new fit completes.

## Complete inputs on newly admitted workers

A training capsule contains TRAIN views. Evaluation also needs the existing
registered validation views, probes, frozen population/source arrays and the
prepared group metadata. Supply this closure before starting an evaluator on a
new worker. `shaping_evaluation_inputs.py register` builds a bounded 256-MiB
manifest from the six registered inner partitions. It checks their exact TEST
membership and existing file hashes; no target value is parsed by transport.

Run `preflight` on an idle evaluator, copy only missing files, then `verify`.
Incomplete view directories are preserved as `.bak` siblings. Complete different
files are refused rather than overwritten. Native trainers continue throughout.
The November 8 repair verified 113 support files on all six eligible workers;
outer and final-holdout views were excluded. Its original failed requests and
incomplete cache directories remain preserved.

Corrected inner scoring selects exactly one explicit binary directory matching
the run/result BUILD hash and registered corrected activation binding. The
legacy runtime is not a fallback for a corrected bake. This dispatch repair
changes neither native scoring arithmetic nor scientific gates or training.
