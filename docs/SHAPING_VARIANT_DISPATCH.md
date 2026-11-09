# Reduced variant dispatch

`shaping_variant_pipeline.py` waits for a complete inner-only outer admission
and for the outer pipeline to submit that fold's higher-priority fits. It then
submits the existing reduced variant cells: 16 human-only, 32 source-fraction
and 16 population-shuffle fits per fold. It never selects on outer responses.

Each stage uses the existing neural/control admission, transport, CPU/RAM/SSD
reservations and independent output owners. Placement spreads paired seeds
across the approved compatible pool. The next diagnostic stage waits until
every higher-priority fit has acquired admission; it need not wait for those
fits to finish. Human-only work therefore enters the queue before source curves
and shuffled-label controls. Missing paired cells block submission.

Requests, canonical assignments and launch receipts precede submission. An
uncertain or failed submission stops the pipeline with its evidence preserved;
the pipeline never retries a scientific fit. Existing terminal collectors
collect neural outputs, while the pipeline collects native control results.

The prepared memberships, label permutations, TRAIN-only shaping banks and
their independent audits are reused unchanged. The original scope policy is
[SHAPING_REDUCED_SCOPE.md](SHAPING_REDUCED_SCOPE.md); this dispatcher changes
execution placement and ordering, not that policy or scientific exposure.

Outer, extension and variant placement samples each worker's live native jobs,
shared claims, and unclaimed waiting controllers before assigning a new wave.
It distributes new work relative to that existing backlog and approved CPU
capacity, avoiding queues on busy workers while other workers are empty. Low
memory or SSD headroom excludes a worker from that placement wave. These are
scheduling hints; each fit still passes the original CPU/RAM/SSD admission.
Already registered or active conditions are never moved by this dispatcher.

The progress viewer reports approved cells separately from preserved original
fits. Once each 64-cell rescue/outer shortlist is frozen, its IDs enter the
approved counters; older nonshortlisted rescues remain visible as preserved
evidence and do not advance the 144-cell rescue progress bar. Unresolved
assignment counts reflect the remaining unfrozen folds, while the original
scope registration remains unchanged.

`shaping_bootstrap_pipeline.py` separately queues the existing primary source
bootstrap and all 24 reduced matched comparisons. Each analysis waits for all
three outer admissions and its complete paired model/report evidence. It then
uses the existing bootstrap admission and transport owners on an approved
worker, with shared CPU/RAM/SSD admission. Registered source draws, seeds,
replicate counts and comparison definitions remain unchanged. Analyses can run
independently as their inputs finish; their collected results receive complete
destination checksum readback. No model is trained by this queue.
