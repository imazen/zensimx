# Inference cost from reduced-study outer models

The approved reduced study omits full-data retuning and refits. Measure the actual
48 outer finalist models instead: six selected roles, three source-disjoint
folds, three seeds per stochastic role and one additive model per fold.

`shaping_cost_finalists.py` prepares three native instrument workloads, each
containing sixteen role/seed arms. Freeze forty timing rows from each fold's
TRAIN membership before reading prediction tables: eight rows per admitted
population, using the existing deterministic row selector. No common set of rows
belongs to all three outer training sets, so the workloads have separate row
identities. Report their timings separately; do not pool cross-fold rounds as
though they were paired on identical inputs. Record each role's actual family.

Require the complete outer admission, scheduled final models, evaluated dense
representations, prediction/model checksums and all paired seeds. Export parity
must match the representation evaluated for quality. Instrument preflight uses
the recorded native scores for the selected TRAIN rows. A missing fit or changed
artifact prevents workload admission. This preparation does not measure timing.

Use the existing qualified native cost instrument and its zenbench admission on
an owner-approved compute host. Preserve exact preflight/postflight checks,
thirty-two clean paired timing rounds, resource receipts, binary identity and
the distinction between vector scoring and image extraction. This changes the
model source from omitted full-data refits to actual outer finalists; it does
not weaken the timing gates in the original inference-cost protocol.
