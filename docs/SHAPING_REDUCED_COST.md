# Inference cost from reduced-study outer models

Image extraction can run before the outer models finish. Every finalist uses
the same frozen raw table read set, `tree_study.RAW_IDS`, including the mandatory
tree control. `shaping_extract_bench.py` verifies all six already qualified
workloads against that read set and the original plan before timing. It measures
the prescribed three RGB8 pairs at 256, 1024 and 4096 pixels per side with one
and four threads, using an enforced uniform CPU class and zenbench's existing
paired-round gates. The complete input/binary/launcher closure is hash bound.
This independent extraction measurement admits no finalist model and supplies no
vector-scoring or end-to-end timing claim. The vector workload still requires
all actual outer finalists below.

The launcher records its own live PID through `ZENBENCH_LAUNCHER_PIDS` so the
existing resource monitor can separate the benchmark's work from foreign CPU
load. All other process, temperature, memory and round gates retain their
original settings. Keep incomplete timing attempts; a retry uses a fresh
registration/output and names only its still-needed cases.

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

`shaping_cost_pipeline.py` waits for the same complete primary outer evidence
as the source-bootstrap owner. It seals all 48 actual models, evaluated dense
representations, selected TRAIN rows and native converter into the existing
bounded worker transport. A worker runs the existing workload preparer; no model
is trained. Preparation transport checks every fold, row selection, schedule,
model and input pin before timing admission.

`shaping_vector_bench.py` then times all sixteen arms in each fold through the
already qualified native instrument and verified CPU boundary. It registers the
actual launcher PID with zenbench, preserves the 32/64 paired-round gates, and
collects each model's fresh-process maximum RSS through the existing GNU-time
owner. Models remain shared in the prepared workload. A busy training host keeps
the benchmark waiting without holding a CPU claim. Native work has a four-GiB
memory cap and the existing 30-GiB SSD floor.

The coordinator collects terminal outputs only after complete worker inventory
agreement and archives both prepared models and timing/memory evidence with
full durable checksum readback. Failure preserves the original attempt; neither
fits nor timings replay implicitly. Observation failures re-poll the same owner.
These tools do not establish that the still-outstanding finalist measurements
have executed.
