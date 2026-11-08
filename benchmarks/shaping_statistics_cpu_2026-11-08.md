# Dashboard CPU observation

The dashboard publisher on `dev` used 3,114.80 ms task-clock during a
60.095268528-second observation window: 5.18% of one CPU. This covers
`shaping_progress.py`, not native numerical evaluation or the long neutral
qualification. Its source export was `ecf198b81f32`; the repository at capture
was `f50cb45f78c686ecafc0a04ba5a7083ccd5c73a4`.

Commands, through `run-heavy --mem 1G/512M --jobs 1`:

```text
sudo perf record -p 4136752 -F 99 -g -o perf.data -- sleep 60
sudo perf stat --timeout 60000 --no-big-num -e task-clock,cycles,instructions,context-switches -p 4136752
```

The task-clock window recorded 16,961,459,056 cycles, 53,385,718,484
instructions and 2,780 context switches. The overlapping sampling window
recorded 328 samples with zero lost; SHA-256 accounted for 27.98% of samples.
These are resource observations, not isolated wall-clock benchmarks.

The stat wrapper reported peak RSS 0.02 GiB, minimum available memory
39,113 MiB and peak load 1.37. This is the profiler's footprint, not a
measurement of the publisher's peak RSS.

Raw profiles, commands, measurements and the destination-readback receipt:
`/mnt/v/output/zensim/shaping-completion-20261107/statistics-cpu-profile-20261108/`,
mirrored under the same relative path below `/mnt/tower/output/zensim/`.

The publisher now reads the replacement evaluation observer's status path;
the previous observer's receipts remain preserved. Scientific fit inputs,
selection and evaluation gates were unchanged.
