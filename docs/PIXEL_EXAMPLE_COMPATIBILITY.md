# Pixel example compatibility, 2026-11-06

At `089093ef`, all four requested pixel consumers compile with the repository's
actual pins: zenpixels/zenpixels-convert `fa836abf` and zencodec `1816b4d0`.
The consumers are `e34_pixels`, `verify_bitstream_decode`, `m3_fixture_gen`, and
`gmsbank_decode_dump` (which includes the shared scoring input code).

Run `just pixel-example-check`. It compiles only; it does not run extraction,
training, evaluation or benchmarks. The separate bench manifest ignores its
lockfile, so the recipe generates that lockfile before the locked check and
enables `verify-decode` explicitly. The optional C++ FFI benchmark objects are
absent in the isolated checkout; their build script reports that fact, while
all four requested Rust examples finish checking successfully.

The free orientation and ICC helpers still exist at those pins. The replacement
associated methods belong to a later zenpixels revision and cannot be substituted
without updating the dependency cohort. A trial substitution with the existing
pins fails with missing associated methods. Therefore no compatibility-only
source rename is appropriate at the current pins. Move the examples together
with the pixel/core cohort rather than changing names independently.

Qualification: resource wrapper exit 0, 39 seconds, peak RSS 0.88 GiB, minimum
available 8531 MiB, peak load 10.00. This is compiler resource use, not runtime
or model performance evidence.
