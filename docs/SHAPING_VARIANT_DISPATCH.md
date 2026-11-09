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
