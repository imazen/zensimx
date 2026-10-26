# Pointer: candidate bakes kept for a later AIC2026 evaluation (2026-10-25)

Owner, 2026-10-25: "do keep canidate bakes around for later eval on aic2026". Every candidate bake is copied, with its
SHA-256, into one registry by `scripts/byconstruction/candidate_registry.py` (re-run it after adding a candidate; existing
entries are verified, never overwritten).

- **dev:** `/mnt/v/output/zensim/candidates-2026-10-25/` (`CANDIDATES.tsv`, `CANDIDATES.json`, `SHA256SUMS`, one
  directory per candidate). Checked on dev 2026-10-25.
- **tower:** `/mnt/tower/output/zensim/candidates-2026-10-25/` (copied 2026-10-25, `sha256sum -c` passed; Mlic rows added
  2026-10-26, 18 bakes, `sha256sum -c` passed).

AIC2026 rules still apply to any evaluation: public_train is the study's training split (Tga trained on it);
public_test has one sanctioned frozen read, with the exposure recorded (`docs/DATA_SPLITS.md`).

| Candidate | Formula revision | Data / recipe | SHA-256 | Note |
|---|---|---|---|---|
| B-rev5 | 5 | today's default B (diff+frag-820) | `89cb4861e2651ff8…` | shipped profile B |
| B7 | 7 | B's weights stamped revision 7 | `5fe0a96230badfa4…` | byte-identical weights to B-rev5 |
| M-rev5-s0 | 5 | M recipe mono:thr:mp32, v2e29 root (rev5) | `6291eeb5a7835655…` |  |
| M64-rev5-s0 | 5 | M64 recipe mono:thr:mp64, rev5 | `0ff53fdcc34405cb…` |  |
| T-rev5-s0 | 5 | T recipe mono:thr, rev5 | `8c40f24c688af660…` |  |
| Tga-rev6 | 6 | Tga (E21 + AIC2026 public_train leg), served form | `5ece9e45e3e48841…` | trained on AIC2026 public_train: in-sample there |
| M7-s0 | 7 | M recipe, rev7raw root | `bc471d215bc9813a…` |  |
| M7-s1 | 7 | M recipe, rev7raw root | `a332b4e6dc1b64ab…` |  |
| M7-s2 | 7 | M recipe, rev7raw root | `0b9fdb8b3301870d…` |  |
| Mdata-s0 | 7 | M recipe + 11 new datasets' TRAIN legs (rev7data root) | `9fe1aa47a8273055…` | trains on VCL@FER, PieAPP, FG-IQA (validate-only licences) |
| Mdata-s1 | 7 | M recipe + 11 new datasets' TRAIN legs (rev7data root) | `268aee6bcea06100…` | trains on VCL@FER, PieAPP, FG-IQA (validate-only licences) |
| Mdata-s2 | 7 | M recipe + 11 new datasets' TRAIN legs (rev7data root) | `692c9507d5e3167d…` | trains on VCL@FER, PieAPP, FG-IQA (validate-only licences) |
| Mdata64-s0 | 7 | M64 shape on the Mdata data (rev7data root) | `257194e446beefd4…` | trains on VCL@FER, PieAPP, FG-IQA (validate-only licences) |
| Mdata64-s1 | 7 | M64 shape on the Mdata data (rev7data root) | `3562cc06ec9a995f…` | trains on VCL@FER, PieAPP, FG-IQA (validate-only licences) |
| Mdata64-s2 | 7 | M64 shape on the Mdata data (rev7data root) | `cb0d32a278e0ddee…` | trains on VCL@FER, PieAPP, FG-IQA (validate-only licences) |
| Mlic-s0 | 7 | M recipe on the Mlic root (Mdata's legs without VCL@FER, PieAPP, FG-IQA) | `7612c2e46154fe9c…` | licence-clean under the 2026-10-25 policy |
| Mlic-s1 | 7 | M recipe on the Mlic root (Mdata's legs without VCL@FER, PieAPP, FG-IQA) | `5618cd8b2206428a…` | licence-clean under the 2026-10-25 policy |
| Mlic-s2 | 7 | M recipe on the Mlic root (Mdata's legs without VCL@FER, PieAPP, FG-IQA) | `4ecd413abdb70004…` | licence-clean under the 2026-10-25 policy |
| Mlic64-s0 | 7 | M64 shape on the Mlic root (licence-clean data) | `7a05deec8b630d9d…` | lost the registered shape rule to Mlic on E21 |
| Mlic64-s1 | 7 | M64 shape on the Mlic root (licence-clean data) | `2f4fbabcb3403410…` | lost the registered shape rule to Mlic on E21 |
| Mlic64-s2 | 7 | M64 shape on the Mlic root (licence-clean data) | `85532fb18a974660…` | lost the registered shape rule to Mlic on E21 |
