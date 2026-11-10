# Codec consumer SIMD compatibility — 2026-11-10

The workspace retains its SIMD minimums and admits archmage/magetypes 0.9.31
stable at the upper bound. Research pins and the normal lock remain unchanged;
frozen study manifests, model inputs and numerical expectations are untouched.

An isolated source export qualified the current codec consumer graph. Its
resolved archmage, archmage-macros and magetypes versions were 0.9.31. Existing
v1 feature-golden tests (five controls) and the pinned serving-profile test pass
on both the normal 0.9.30 graph and this 0.9.31 graph. Stable library compilation
passes; normal-graph strict library Clippy passes. Legacy SIMD aliases emit
deprecation notices on 0.9.31; the upper bound excludes future arithmetic changes.
This gate covers metric arithmetic and serving values.

The consumer graph uses zencodec `1fb997bb`, zenpixels `3286169c`, zenflate
`6832cd2e`, zenpng runtime `dceae9c1`, zenresize `01e66aac`, ultrahdr `aadf1021`,
linear-srgb `c5b83e3e` and garb `7356e8f1`. Its JPEG dev fixture override uses
the consumer's current checkout. These overrides apply only to the isolated
qualification export, whose lock does not replace the research lock.

Reproduce using an isolated source export and a Cargo config carrying the
consumer patch table (including its JPEG fixture override):

```sh
just codec-stable-simd-check "$metric_export/Cargo.toml" "$consumer_config"
```

The library/model/test sources are unchanged from `4e9912f1`; the compatibility
change is the two manifest upper bounds. No training, evaluation, model selection
or performance campaign was run.
