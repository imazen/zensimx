# Compact orchestration for frozen worker requests

`shaping_compact_worker.py` keeps each immutable request, scientific source,
native owner and command. It executes the original `verify` and `reproduce`
functions in a short-lived 512-MiB process. The long-lived queue retains only
standard-library modules and the original CPU/SSD admission functions. At most
two preflights run per host, with an additional half-GiB admission margin above
the machine's four-GiB launch floor. Native fits remain concurrent under the
existing CPU ceilings, warmup allowances, memory caps and snapshot reservations.

This addresses replicated data-library residency in waiting Python queues.
It does not change the training binary, inputs, optimizer, seeds or exposure.
The wrapper records its own source hash separately from the unchanged frozen
worker hash. Preflight must reproduce the frozen resolution before a fit can
claim CPU. The original request verifier runs again after a capacity wait.

An existing queue can hand over only if every member is unopened, it has no
active child or CPU claim, and its live command, original status and request
hashes match. The short original admission locks protect this transaction.
The idle queue stops after its handoff receipt is written; original manifests,
requests and status remain preserved. A queue containing any active, failed,
partial or completed fit refuses this handoff. No native fit is interrupted.

Run `wave --qualify` first with the original full `--source`, `--wave` manifest
and a fresh `--output`. It executes the real original admissions without
claiming a scientific fit or launching its native command. `handoff` records an
explicit idle retirement; ordinary `wave` requires that exact receipt. Startup
uses an exclusive status-file creation and never replays an existing compact
wave. `--study` provides the same explicit observer scope as other controllers.

Use the resource wrapper for qualification and execution. The light parent can
be measured in a 128-MiB scope; preflights retain their separate 512-MiB scopes,
and corrected native controllers keep their original 512-MiB scopes. Native
trainers retain their two-GiB scopes. A usable systemd user manager is required
for these separate caps; do not substitute an uncapped fallback.

Fixture checks cover subprocess gate equality/refusal, attempt preservation,
request drift, real idle-process retirement and active-child refusal. They are
orchestration checks, not native scientific qualification or a measured memory
improvement. Record real qualification, wrapper accounting and deployment
evidence separately before claiming either.
