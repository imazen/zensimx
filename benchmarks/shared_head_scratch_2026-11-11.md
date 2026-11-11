# Shared-head backward scratch allocation

The plain shared ReLU/SmoothHinge head allocated its pre-activation gradient
buffer on every backward call. It now reuses a zeroed buffer owned by the head.
The field is skipped by serialization; decoding initializes it empty. Arithmetic,
update order and checkpoint payloads are unchanged by this storage change.

A fixed-work allocation probe uses a 128-to-64 head and 8,192 backward calls.
The same compiled function is run with scratch discarded before each call
(the previous allocation policy) or retained. This is an allocation comparison,
not a separate historical binary comparison or a wall-clock speed benchmark.

| Policy | Allocations below the fixed-work probe |
|---|---:|
| Fresh scratch each backward | 16,404 |
| Retained scratch | 8,213 |

Heaptrack measured 8,191 fewer allocations: the retained buffer allocates on
its first call and is reused afterwards. Setup and the returned gradient vector
remain in both counts. No end-to-end training speedup is claimed. The separate
analytic research head is unchanged.

The focused checks cover stale scratch, decoding, identical serialized updates,
finite-difference gradients, fused/unfused updates, identity, monotonicity and
bake/runtime agreement. All nine focused tests and strict trainer clippy pass.

Command: cargo test -p zensim-validate --lib mlp_train::shared_depth::tests
-- --test-threads=1. Run the scratch_allocations_fresh and
scratch_allocations_reused tests separately under heaptrack. Captures and
flamegraph allocation stacks are in the external shaping-completion artifact
bundle, study-resume-20261111/scratch-allocation/RESULT.json.

Profiling wrapper: run-heavy: done rc=0 1s | peak-RSS 0.22GiB.
Focused test wrapper: run-heavy: done rc=0 31s | peak-RSS 1.60GiB.

This candidate is not activated in the frozen research study. The registered
study continues with its existing qualified native binaries.
