# Frozen seed-0 adjudication preread

Analysis only, on base `dcf7e3552b06e1bc0b36520f945d77d62d60295b`.
No fit, calibration, threshold change, protected payload read or push.
The production composition remains D1 seed 0 alone, SHA-256
`f803b74c4252952f337abdc0234c2930839d45dddc32ae9b8b5296d6c840f400`.

## Quiet-window gate

Do not build, test, benchmark or execute numerical replays until
`/home/lilith/tmp/zensim-paper/rev4/SPEEDQ_DONE.md` changes from the old
stopped report, SHA-256
`3d5cd77dc2e5f5054e40b5acf2e022615aa39b094902fe4547292752b7f80c5e`.
Its initial modification time is 2026-10-07 11:20:59.106636025 -0600.
Record the replacement marker's bytes, hash and status before proceeding.
This preread uses source and program/provenance metadata only.

## Fixed packets, controls and bars

Reuse the already admitted PRODQUAL-B engineering population, at
`/mnt/v/output/zensim/prodqual-b-2026-10-07/`, with its preread registrations
and hashes. The instruments are the explicitly named standard, ladder,
identity, broad, owner, jpeg8 and m3 packets under
`/var/tmp/shippath7/verified/instruments/`. Do not scan adjacent datasets.
Do not load TRAIN calibration anchors or any human opinion labels.

C5 uses [97.5,100]; ties <=0.05; M2 >=0.99; M3f >=0.70.
Keep existing epsilon, material inversion and codec-saturation definitions.
Record original and diagnostic classifications separately.

Compare the same 38 identity references using D1 seeds 0, 1 and 2,
the research Rev5 by_v2fy full-s0 model and serving B. Seeds 1 and 2 are
controls, not members of the frozen production composition. Research model:
`/home/lilith/tmp/rev5bakes/byv2fy-full-s0.bin`, SHA-256
`690b270936b1b361d50573b9572e6c34f8edfe73fb21ec197dbcc9705ae7a314`.
Serving B is the registered September 6 bake, SHA-256
`a96a5a66cd16282ded84580c37207b403334f1af069c924d9883346d23941276`.
Serving B must run in its own Rev1 process, extracting its own canonical
features from the same pixels; never treat Rev5 cached rows as Rev1 inputs.

## Planned evidence

1. Replay cached C5 scores and native canonical identity extraction. Compare
   every structurally consumed ID and f64 bit, including all 420 production
   inputs. Record nonzero/reference-only inputs, raw feature inference,
   identity-aware feature inference and pixel identity separately. Native
   identity shortcuts cannot waive the raw C5 contract.
   The cache manifest records an f32 digest: compare full native f64 bits
   and the canonical f32 storage round trip separately. Expected storage
   rounding is not proof of stale data; any residual disagreement is reported.
2. Dump exact standard and ladder cell scores through bake_verdict's existing
   `ZENSIM_DIAL_PRED_OUT` owner. Identify every tied adjacent rung by key and
   compare consumed features, uncomputed slots, and existing image hashes.
   The source comparator applies subtraction to NaN slots, making equal
   missing slots unequal: this is a hypothesis to test, not a measured cause.
3. Reconstruct or locate the declared dense intermediate, SHA-256
   `e6995b1073de6022c31e012a104a50a1710488d4e520ee7fd016c649d924c4c4`,
   via the existing densify owner without calibration payloads. Compare dense
   and packed complete scores as requested, and their uncalibrated outputs
   through `BakeScorer::without_output_calibration` to isolate network effects.
   Packing also refits the TRAIN spline; complete-score differences alone
   cannot identify f16 rounding as their cause.
4. Replay all seven reported steering failures using the registered pixel
   hashes, block sizes, environment and canonical diffmap owner. Compare dense
   and packed score changes, maps and finite refinements. Explain B's explicit
   masked/IW refusal from its consumed IDs and steering support check.
5. Reconcile coverage against original registrations. The original owner12
   definition is four level-03 pairs times three individual seeds at block32.
   The Rev5 JPEG runner uses eight pairs at block8; the gate map's JPEG8x8
   notation must not silently become eight different block geometries.
   Distinguish historical three-seed comparisons from the frozen single seed.

The stale zero-vector annotation in bake_verdict is not accepted as evidence:
the current registry explicitly computes v2 PJND_FRAGILITY as ReferenceOnly.
Whether the admitted identity cache matches those canonical values remains
unmeasured at registration.

Use existing Rust inference/extraction owners, no replacement numerical
scorer. Any helper is private to the audit example or an instrument-only
owner overlay. Persist exact commands, hashes, scores, per-cell causes and
raw reports in a small evidence archive; mirror and verify before cleaning
only this lane's Cargo target. Commit on a private review bookmark, never
move shared main. Write ADJUDICATE_DONE.md after all other work.
