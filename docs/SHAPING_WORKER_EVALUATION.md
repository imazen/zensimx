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

Worker capsules carry the unchanged reduced policy as `REDUCED_PROTOCOL.md`,
analogous to the original `PROTOCOL.md`. Its hash and the installed source's
policy must both match the immutable registration. The original policy path and
all scientific registrations remain unchanged.

`just shaping-evaluation-worker-check` checks transport, cell isolation, failure
preservation and registration guards with synthetic artifacts. Those tests do
not establish a successful native worker evaluation; record the first real
import and its scientific gates separately after a new fit completes.
