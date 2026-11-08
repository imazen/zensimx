# Fixed external models and learned-curve consistency, November 8

All 34 registered fixed models have completed both external datasets: 68 native
dataset reports and 102 population scatterplots. These are exposed development
reads at setting 0 / inner-0-0, outside selection. They are not an outer-study
result, a tuned-family comparison, or untouched final-holdout evidence.

| Family | Seeds | CSIQ SROCC | LIVE clean10 SROCC |
|---|---:|---:|---:|
| extra_trees | 3 | 0.959952007 | 0.944171274 |
| product_joint_identity | 3 | 0.953634972 | 0.962848500 |
| product_pretrained_frozen | 3 | 0.955030838 | 0.963536747 |
| product_pretrained_joint | 3 | 0.955781186 | 0.960380900 |
| relu | 3 | 0.942568924 | 0.962992454 |
| separate_joint_identity | 3 | 0.955052823 | 0.956458630 |
| separate_pretrained_frozen | 3 | 0.955549680 | 0.954859796 |
| separate_pretrained_joint | 3 | 0.956150880 | 0.954339461 |
| shaped_additive | 1 | 0.935884161 | 0.941939043 |
| silu_diagnostic | 3 | 0.893702690 | 0.797161167 |
| silu_monotone | 3 | 0.939651057 | 0.963339835 |
| softplus | 3 | 0.940742445 | 0.963018513 |

Values are arithmetic means of native signed SROCC over the listed paired
seeds. The additive model has one deterministic fit. CSIQ includes 866 rows;
LIVE clean10 includes the registered 267 rows from ten non-overlapping sources.
Each individual model retains its registered source-bootstrap interval; this
table does not provide a paired family-difference interval.

Product shaping improves the displayed CSIQ means over the unshaped ReLU
control while retaining similar LIVE clean10 results. Extra Trees have the
highest displayed CSIQ mean and a lower LIVE clean10 mean. These observations
support continuing the registered source-fold and rescue comparisons; they do
not establish a universal winner.

## Input-curve consistency

The metadata comparison covers all 144 trained identity-initialized joint
models: 72 product and 72 separate, each with four settings, six inner partitions
and three seeds. It reuses existing native inspections, with zero fitting or
prediction replays. Four of 984 product channels remain nonzero in every fit;
none of the separate channels do. The four product channels are logical columns
585, 586, 603 and 1912. Their normalized curve spreads remain substantial;
nonzero response is not evidence of stable predictive importance.

The comparison excludes downstream head weights. Shape/head compensation and
failed or collapsed heads remain visible in this full-grid metadata analysis.
It establishes no universal fixed per-feature bank and no benefit from
initializing another recipe with these pooled-fold curves.

## Retained evidence

External publication INDEX SHA-256: `947e2dc409d910cbb13d3e340fd954b9122bbfee02b35d6bce8fc2d366f4e706`.

Artifacts on the coordinator under `/mnt/v/output/zensim/shaping-completion-20261107/`:

- `external-corrected-joint-archive-20261108`: 202 artifacts, 91,181,833 bytes,
  cold destination SHA-256 readback and executable metadata verified. It includes
  the final six models, 12 native dataset reports and a prepared code-free replay.
- `feature-curve-stability-all-settings-20261108`: 168 artifacts, 55,800,495 bytes,
  cold destination SHA-256 readback and executable metadata verified. It includes
  all 144 input inspections and 11 comparison panels.

Both mirrors are under the matching `/mnt/tower/output/zensim/` study root.
The earlier model and control archives remain separate and preserved. Replay
closures were prepared, not rerun solely for this result note.
