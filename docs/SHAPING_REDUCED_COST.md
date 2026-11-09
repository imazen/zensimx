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

Finalist preparation also records each timed model's byte size and native tree
or layer structure through the existing qualified inventory executable. Neural
models receive their original final-phase parameter accounting, including tied
weights, fixed intercepts, frozen shapers and learned temperatures, through
`shaping_parameter_count`. Their observed training time and run-heavy receipts
come from the existing phase-resource reader. Additive models report the actual
fitted basis-coefficient count and nonzero coefficients. Tree/node counts remain
separate from scalar training-parameter counts. These are structural and observed
resource reports; none measures effective capacity or isolated training speed.
Preparation and terminal collection bind these reports to all sixteen models
per fold, the exact timed representation and the sealed original training inputs.
Missing reports, changed counts or unbound evidence refuse collection.

The integrated reader has passed its transport/count-binding checks. Its native
qualification uses the existing eight fixed inner specimens and previously
qualified parameter counts; no fit or timing is replayed. The new preparation
source must pass that check before replacing the waiting deployed owner.

The native check passed on i134: all eight structural inventories matched their
qualified references, and all six neural counts matched their earlier native
parameter reports. The reader also parsed each neural specimen's saved training
resource receipts. Run-heavy reported `done rc=0 1s | peak-RSS 0.04GiB |
min-avail 25151MiB | peak-load 6.57`. Input hashes were checked before transfer
and again after execution. The qualification is under
`finalist-complexity-native-30148477-20261109/qualification` in the study's
artifact root. The separate twenty metadata checks include complete 48-model
transport, count/role/input corruption refusals and unchanged ordinary-fit caps;
they do not constitute native scoring qualification.

The `71eeaf3897cb` cost owner now waits for the complete outer matrix, with
the same i270 placement, CPU boundary, native timing instrument and 32/64 gates.
Its predecessor was verified waiting and childless before replacement; no
preparation, fit or timing was interrupted. The complete accounting checks also
passed with the actual i270 interpreter. The deployment and qualification closure
passed durable checksum readback for 127 files and 83,998,205 bytes under
`/mnt/tower/output/zensim/shaping-completion-20261107/finalist-complexity-deployment-archive-20261109`.
The archive includes fixed specimen inputs, prebuilt tools, readable reports,
compressed source plus overlays and the verified owner identity. Actual finalist
counts and inference observations remain pending the registered outer fits.

The frozen `b6784ba1a692` owner is deployed and verified live while waiting for
the complete outer matrix. Preparation and timing target i270; existing training
owners retain their previous sources. Admission/transport checks passed on
r7900x and again with the explicit i270 worker interpreter. They cover the
48-model preparation/transport path, changed row/model/input refusals, original
worker caps, reproduced gates and CPU-claim release after failed timing. The
fixture bytes in the transport tests are not scoring qualification.

The deployment closure passed complete durable readback for 24 files and
53,106,501 bytes under
`/mnt/tower/output/zensim/shaping-completion-20261107/cost-pipeline-deployment-archive-20261109`.
It includes the native tools, preserved setup refusal, successful checks, source
base plus overlays, launch identity and the verified waiting owner. No actual
finalist timing or fresh finalist RSS observation is claimed by this deployment.
