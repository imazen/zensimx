# R12: non-negative linear heads against adopted A (2026-11-04)

The nine registered fits are frozen. Report-only assessments follow this fit record;
no candidate is selected or adopted here.

Registration: [E36 R12 record](e36_subset_test_2026-11-01.md#r12-registration-2026-11-04-before-any-fit-or-new-read-linear-formulas-over-as-inputs),
commit `9088eacf`; exposure: [DATA_SPLITS R12](../docs/DATA_SPLITS.md#exposure-registration--2026-11-04-r12-linear-heads-against-the-adopted-mix-a).
The development-unit correction was committed as `284ccfb7` before L3.

L1 permits all 984 inputs. L2 fixes all but the top eight audit families to zero:
144 damage features and their 144 fragility products remain available. Both direct
and product coefficients are non-negative. Three random seed indices per arm.
L3 continues each L1 seed through the mix's AIC-2026 public_train ×256 recipe with
rehearsal and fixed epoch 59. No public_test or confirmation-set read is authorized.
No AIC-2026 result statistics are recorded here.

All base fits use the mix's exact admitted root, group targets, weights and sampling
through `v2_confirm_fit.py` / `v2_lodo_mlp.py`: TID2013 + MDID MOS, BAPPS, SafeSyn,
CID22 TRAIN teacher, coverage and chroma-teacher legs. Only the head architecture
and L2's fixed zero-weight mask differ. The existing N + mono path with H1 and zero
hidden bias gives an identity ReLU on non-negative damage inputs: raw = 100 minus
a non-negative weighted sum. The common output stage and TRAIN-fitted PCHIP dial
remain nonlinear. Adam optimizes the existing loss; these are three fitted solutions,
not a certified global linear optimum. L1/L2 stop at epoch 119, L3 at epoch 59.
Canonical densify, f16 quantization with protected last layer, no input pruning,
then CID22 TRAIN calibration with E33 identity knot/tail factor 2.

The historical dataset authority pins H128. R12 has an explicit H1 authorization
in the Python/native admission owners (`b901766a`), with the registration identity
and sparse-mask digest bound before optimization. All old population, seed, budget,
continuation and data-role checks remain. Seven Python admission tests and twelve
native admission tests pass, including unregistered-H1 and changed-mask refusals.

## Training-side feasibility decision

The registered teacher-development rows are SafeSyn (38,757) and CID22 (3,785).
Stored targets already use score points (`--target-scale 1`). The invalid initial
100× conversion is retained under `dev-unit-error/`; it does not qualify L3.
Corrected pooled MAE, seeds 0 / 1 / 2:

| Model | SafeSyn | CID22 | Row-pooled |
|---|---:|---:|---:|
| A (research F) | 3.393523 / 3.398859 / 3.568893 | 1.500572 / 1.586669 / 1.601302 | 3.225105 / 3.237627 / 3.393835 |
| L1 | 2.956284 / 2.712661 / 3.163496 | 1.370633 / 1.314540 / 1.437120 | 2.815207 / 2.588269 / 3.009899 |

L1 seed-mean pooled error 2.804458 versus A 3.285522: delta −0.481064 score
points, within the registered +1 point allowance. L3 therefore ran. This is
training-side teacher evidence, not independent human generalization.

## Frozen fits

Full hashes, epochs and native certificates: [fit receipt](r12_linear_fits_2026-11-04.json).
All nine packed bakes are certified monotone with zero ramps and zero max pairs.

| Arm | Seed | Packed SHA-256 | Bytes |
|---|---:|---|---:|
| L1 | 0 | `6a9312a478d727195e5cb25ab72c2f3e04ee0f8e8b6aeebbc9f0d0aaa843a2b9` | 21814 |
| L1 | 1 | `985d2b234254f4a9c691cee9cc0cef6b4aec805b46ee966e297798eed649ce08` | 21881 |
| L1 | 2 | `2c54e0991eba3173c647ea285d7d031e2b54e7b27733ff1aebec14a35c29683e` | 21782 |
| L2 | 0 | `51a3ff42cbf48ddadbddaafe0df9b730871798af8f73d24827a88abfa2096dd5` | 20930 |
| L2 | 1 | `9d6a9bcb36a5aa844d4561303d580842680f09c170e0194a1508ec9b9b6c89bc` | 20933 |
| L2 | 2 | `012545710c8edfb074fb28f65b06d883091ca76b9e8eecf5d68bbdaa4fe9185d` | 20914 |
| L3 | 0 | `7230b6046d6f9df6345e5904eb79af98cd8986de3df30f14d3a8a8e17578c0ac` | 23957 |
| L3 | 1 | `3984473bbbc46cbc6f283ed1f205365b6a761733f0df34629b5760f2cd8a7bf1` | 23668 |
| L3 | 2 | `62ec6a8ee8f04417a75bae8f4005afaf9baabdbc2791153ec477819132eddaf9` | 24023 |

Artifact root, checked on dev: `/var/tmp/byc-linear-20261104/`. Model files live under
`fits/L1-s*/refit/production-f16.bin`, `fits/L2-s*/refit/production-f16.bin` and
`fits/L3-s*/pack_e059/refit/production-f16.bin`. `SOURCE_PINS.json` binds source files,
trainer/packer binaries and the immutable root. `MODELS.json` freezes the assessment
inventory. All logs and the refused pre-fit/unit-error attempts remain retained.

Execution through the shared heavy lock and `run-heavy --mem 24G`: six base fits
536 s, peak-RSS 2.00 GiB, min available 35,772 MiB, peak load 3.68; three
continuations 344 s, peak-RSS 1.96 GiB, min available 36,502 MiB, peak load 2.41.
These are execution receipts, not inference timing comparisons.
