# Fixed external shaping validation

The owner requested more datasets for the frozen bakes. This exploratory comparison
uses the already exposed CSIQ and LIVE release 2 evaluation populations, with
cached features from the exact Rev7 extractor used by this study. No new pixels,
feature formulas, training data or calibration are introduced.

The model list is fixed before parsing these external responses: every family
at setting 0 in inner-0-0, including all paired seeds, gives 34 conditions.
Corrected pretrained models wait for their registered final report. Each exact
bake and its qualified final report are bound before scoring. No external result
may influence the registered tuning, rescue selection or final shortlist.

Report CSIQ (866 rows), LIVE full (779 rows, descriptive content overlap), and
LIVE's existing fixed clean10 subset (267 rows, 10 sources) separately. Preserve
all scalar predictions and source identities; report signed SROCC, Kendall,
PLCC and scatters. Use paired whole-source bootstrap uncertainty (1,000 draws,
seed 20261108). No new label mapping is fitted to these responses.

Registration artifact: `/mnt/v/output/zensim/shaping-completion-20261107/external-validation-20261108/REGISTRATION.json`

SHA-256: `61c15270990222fed25f24237664525a9cab1dfd65d822dbdb55ecebabfab7c1`.

This extends descriptive evaluation only. The registered core remains 1,461
training fits. Untouched final holdouts remain outside this comparison.

Completed model families may arrive in separate immutable binding batches, all
using the same original registration bytes. `shaping_external_publish.py` merges
only their native scalar reports. It checks model/report bindings, prediction
hashes, population denominators and finite scatter coverage, and refuses duplicate
dataset results or a changed registration. Earlier reports without an inline
driver hash require their existing verified archive inventory. Each published
result and scatter retains a source hash and relative hyperlink. Publication
does not rerun native statistics, fit a calibration or select a model.
