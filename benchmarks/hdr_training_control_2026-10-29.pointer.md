# HDR training replay control, 2026-10-29

The existing Mlic KADID seed0 control retrained all120 epochs using the frozen
root/binaries and existing fx2-study owner. Predictions match as bytes; the
model matches after the native owner strips only `zentrain.repro`.

Local (checked on dev): `/var/tmp/byc-hdr-20261029/control-v1/`.
Receipt/log/input pins mirrored and compared byte for byte under
`/mnt/tower/output/zensim/byc-hdr-20261029/control-v1/`.
The full fit/checkpoint directory is retained locally; its complete mirror
is pending. No R2 mirror created. No candidate comparison has run.

- Prediction SHA256: `0cae0f6fcc1ca3f83108df4c3fd400751a6321842a56a0e942ecba09ab4bd8fa`.
- Model without repro metadata: `0e03eaf02fdf140291b8423fe55de19d857b9af6a2b3ecb80ae439b331b57678`.
- Owner SHA256: `1572ffa02ce5e009cd4fbb1b74fa135dfb7872443aeea49f2198ad28cc8b7607`.
- Frozen root: `a7fd4af0c2bb5a8981be636e5c1cb1a51db57496b7c60d7bf6da01d71dd16952`.

Command: shared `heavy.lock`, run-heavy16G/8jobs, existing
`fx2_study.sh control <run-directory>`; the owner fixes v3/one-thread arithmetic.
run-heavy: `done rc=0 1771s | peak-RSS 1.91GiB | min-avail 39984MiB | peak-load 5.44`.
This is training replay bookkeeping, not a runtime benchmark.
