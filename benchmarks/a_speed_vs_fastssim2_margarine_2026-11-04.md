# Public A versus fast-ssim2 and margarine — 2026-11-04

Measured on **i270**, Intel Core Ultra 7 270K Plus, runtime v3/AVX2. Affinity is CPUs 0–15 for every arm and both thread counts; this is a mix of P and E cores, with no CCD boundary. These observations do not measure Zen 5.

Each cell uses 20 real images, zenjpeg q40 YCbCr 4:2:0 reconstructions, 100 ms warmup per arm/image, randomized interleaved single-call zenbench rounds. The first 32 clean rounds per image are retained with shared indices across arms (640 calls per arm/cell); all raw rounds remain archived. p50/p95 are type-7 quantiles of those individual calls, pooled equally across images. No native CPU compilation flags. Both builds enable fast-ssim2's rayon feature; RAYON_NUM_THREADS is 1 or 16, and public A uses with_parallel(false/true). All three arms score the same resident RGB8 sRGB bytes in process; decode, resize, JPEG encoding and I/O are excluded. The unchanged private margarine kernel is included directly; no CLI or process-start cost enters latency. Margarine uses simd-row-malta, 128 rows/512 columns, primary max pooling; lower means less distortion.

Small cells mix ten CID22 TRAIN and ten CLIC TRAIN sources; 1024² uses twenty CLIC-1024 TRAIN images. The large cell uses twenty canonical imazen-26 PNG-v3 SDR renders across three folders, native when the long edge is below 4096, otherwise Mitchell-downscaled to long edge 4096, never enlarged. The smaller cells use a central square crop and zenresize Mitchell. This is runtime evidence, not a quality comparison.

| Host | Threads | Size | A p50/p95 ms | fast-ssim2 p50/p95 ms | margarine p50/p95 ms | A/fast | A/margarine |
|---|---:|---|---:|---:|---:|---:|---:|
| i270 | 1 | 64² | 0.7502/1.2536 | 0.4001/0.6974 | 1.4754/3.1743 | 1.875 | 0.509 |
| i270 | 1 | 256² | 3.5328/5.6097 | 4.2090/6.3752 | 6.0524/11.5741 | 0.839 | 0.584 |
| i270 | 1 | 1024² | 33.4841/56.4965 | 73.9088/84.5089 | 56.2368/63.3265 | 0.453 | 0.595 |
| i270 | 1 | large (12 MP class) | 350.9354/376.8514 | 1373.9451/1498.0334 | 666.2212/747.9781 | 0.255 | 0.527 |
| i270 | 16 | 64² | 2.1501/3.4565 | 0.4723/1.7216 | 1.6449/2.6505 | 4.552 | 1.307 |
| i270 | 16 | 256² | 2.6505/5.6641 | 4.7534/7.0600 | 7.3119/10.6318 | 0.558 | 0.362 |
| i270 | 16 | 1024² | 11.2075/16.8119 | 33.8007/42.6956 | 48.4125/55.0390 | 0.332 | 0.232 |
| i270 | 16 | large (12 MP class) | 105.3811/119.9162 | 357.4678/396.6283 | 550.1496/587.3843 | 0.295 | 0.192 |

On i270 at 1 thread, A has lower p50 than fast-ssim2 at 256², 1024², large (12 MP class) and higher p50 at 64². On i270 at 1 thread, A has lower p50 than margarine at 64², 256², 1024², large (12 MP class). On i270 at 16 threads, A has lower p50 than fast-ssim2 at 256², 1024², large (12 MP class) and higher p50 at 64². On i270 at 16 threads, A has lower p50 than margarine at 256², 1024², large (12 MP class) and higher p50 at 64². At 1 thread, A/fast changes from 1.875 at 64² to 0.255 at large, and A/margarine from 0.509 to 0.527. At 16 threads, A/fast changes from 4.552 at 64² to 0.295 at large, and A/margarine from 1.307 to 0.192.

Ratios below one favor A. The large-cell actual pixel count ranges from 12,000,000 to 12,582,912; mean 12,038,784.0.

| Host | Size | A 16-thread speedup | fast-ssim2 speedup | margarine speedup |
|---|---|---:|---:|---:|
| i270 | 64² | 0.349× | 0.847× | 0.897× |
| i270 | 256² | 1.333× | 0.885× | 0.828× |
| i270 | 1024² | 2.988× | 2.187× | 1.162× |
| i270 | large (12 MP class) | 3.330× | 3.844× | 1.211× |

OLS fits use the 80 per-image medians against actual pixels, separately for each arm/thread count: `time_ms = alpha_ms + beta_ms_per_MP * pixels / 1,000,000`. Intercepts are unconstrained regression terms, not measured process-start costs. Content changes at the largest size, so these are descriptive fits rather than causal scaling laws.

| Host | Threads | Arm | α ms | β ms/MP | R² |
|---|---:|---|---:|---:|---:|
| i270 | 1 | zensim_a | 1.59614 | 29.03716 | 0.99969 |
| i270 | 1 | fast_ssim2 | -16.43188 | 116.27186 | 0.99855 |
| i270 | 1 | margarine | 0.97290 | 55.54126 | 0.99969 |
| i270 | 16 | zensim_a | 2.03585 | 8.67350 | 0.99637 |
| i270 | 16 | fast_ssim2 | 2.04224 | 29.72196 | 0.99882 |
| i270 | 16 | margarine | 2.15580 | 45.32139 | 0.99948 |

RSS uses a fresh process for each arm/thread/size and scores all twenty pairs with one pair resident at a time. Values include inputs, the runtime, allocator retention and scoring scratch. GNU time supplies exact KiB; run-heavy's rounded total-process line is retained below.

| Host | Threads | Size | A peak KiB | fast-ssim2 peak KiB | margarine peak KiB |
|---|---:|---|---:|---:|---:|
| i270 | 1 | 64² | 7876 | 4736 | 4752 |
| i270 | 1 | 256² | 14452 | 11308 | 11616 |
| i270 | 1 | 1024² | 32656 | 122688 | 31408 |
| i270 | 1 | large (12 MP class) | 179276 | 1435000 | 156628 |
| i270 | 16 | 64² | 7572 | 5020 | 6032 |
| i270 | 16 | 256² | 27704 | 11532 | 13144 |
| i270 | 16 | 1024² | 159920 | 235020 | 41204 |
| i270 | 16 | large (12 MP class) | 598216 | 2468936 | 167168 |

```text
i270 1t 64 zensim_a: run-heavy: done rc=0 0s | peak-RSS 0.01GiB | min-avail 25719MiB | peak-load 1.57
i270 1t 64 fast_ssim2: run-heavy: done rc=0 0s | peak-RSS 0.01GiB | min-avail 25755MiB | peak-load 1.57
i270 1t 64 margarine: run-heavy: done rc=0 0s | peak-RSS 0.01GiB | min-avail 25774MiB | peak-load 1.57
i270 1t 256 zensim_a: run-heavy: done rc=0 0s | peak-RSS 0.01GiB | min-avail 25820MiB | peak-load 1.19
i270 1t 256 fast_ssim2: run-heavy: done rc=0 0s | peak-RSS 0.01GiB | min-avail 25816MiB | peak-load 1.19
i270 1t 256 margarine: run-heavy: done rc=0 0s | peak-RSS 0.01GiB | min-avail 25811MiB | peak-load 1.10
i270 1t 1024 zensim_a: run-heavy: done rc=0 0s | peak-RSS 0.03GiB | min-avail 25684MiB | peak-load 1.31
i270 1t 1024 fast_ssim2: run-heavy: done rc=0 2s | peak-RSS 0.12GiB | min-avail 25577MiB | peak-load 1.21
i270 1t 1024 margarine: run-heavy: done rc=0 1s | peak-RSS 0.03GiB | min-avail 25662MiB | peak-load 1.21
i270 1t 4096 zensim_a: run-heavy: done rc=0 7s | peak-RSS 0.17GiB | min-avail 25463MiB | peak-load 1.11
i270 1t 4096 fast_ssim2: run-heavy: done rc=0 32s | peak-RSS 1.37GiB | min-avail 24569MiB | peak-load 1.16
i270 1t 4096 margarine: run-heavy: done rc=0 17s | peak-RSS 0.15GiB | min-avail 25593MiB | peak-load 1.11
i270 16t 64 zensim_a: run-heavy: done rc=0 0s | peak-RSS 0.01GiB | min-avail 25870MiB | peak-load 0.91
i270 16t 64 fast_ssim2: run-heavy: done rc=0 0s | peak-RSS 0.01GiB | min-avail 25853MiB | peak-load 0.91
i270 16t 64 margarine: run-heavy: done rc=0 0s | peak-RSS 0.01GiB | min-avail 25853MiB | peak-load 0.92
i270 16t 256 zensim_a: run-heavy: done rc=0 0s | peak-RSS 0.03GiB | min-avail 25869MiB | peak-load 0.83
i270 16t 256 fast_ssim2: run-heavy: done rc=0 0s | peak-RSS 0.01GiB | min-avail 25846MiB | peak-load 0.76
i270 16t 256 margarine: run-heavy: done rc=0 0s | peak-RSS 0.01GiB | min-avail 25824MiB | peak-load 0.76
i270 16t 1024 zensim_a: run-heavy: done rc=0 0s | peak-RSS 0.15GiB | min-avail 25756MiB | peak-load 3.25
i270 16t 1024 fast_ssim2: run-heavy: done rc=0 1s | peak-RSS 0.22GiB | min-avail 25742MiB | peak-load 3.07
i270 16t 1024 margarine: run-heavy: done rc=0 1s | peak-RSS 0.04GiB | min-avail 25588MiB | peak-load 3.07
i270 16t 4096 zensim_a: run-heavy: done rc=0 3s | peak-RSS 0.57GiB | min-avail 24786MiB | peak-load 4.31
i270 16t 4096 fast_ssim2: run-heavy: done rc=0 9s | peak-RSS 2.35GiB | min-avail 23652MiB | peak-load 5.25
i270 16t 4096 margarine: run-heavy: done rc=0 12s | peak-RSS 0.16GiB | min-avail 24978MiB | peak-load 4.91
```

The isolated RSS calls also supply 240 per-pair, per-arm comparisons of printed scores between thread counts; 0 differ. This diagnostic compares printed values, not a separate numerical qualification study.

Source/build identities and exact reproduction commands are recorded in the [metadata](a_speed_vs_fastssim2_margarine_2026-11-04.meta.md). Raw outputs, fixture bytes and hashes are retained outside git in the [artifact pointer](a_speed_vs_fastssim2_margarine_2026-11-04.artifacts.md).

The original margarine CLI is also built unchanged against the frozen dependency graph. Separate zenbench measurements time a fresh `margarine --help` process, including spawn, executable loading, argument handling and exit. This no-input baseline does not initialize image decoding or the scoring Rayon pool, and is not subtracted from resident-pixel timings. It does not measure end-to-end CLI scoring latency. Warmup is 100 ms; the first 32 clean single-call rounds are retained.

| Host | RAYON_NUM_THREADS environment | Command startup p50/p95 ms |
|---|---:|---:|
| i270 | 1 | 3.1596/3.3235 |
| i270 | 16 | 3.2054/3.4282 |

Canonical large-cell source bytes (paths relative to `imazen-26-png-v3/`):

| File | SHA-256 |
|---|---|
| `png-v3/1000-lilith-photos-general/1000_general_red-convertible-car_mission-hills-san-diego-california_note9_iso50-f2p4_20190514-100811_4032x3024.sdr.png` | `df355b76958e2482ed85ca92a354150586041ac3f2e4927509b0746b1d331754` |
| `png-v3/1000-lilith-photos-general/1001_general_oceanfront-resort-view_sheraton-waikiki-east-honolulu_s21u_iso32-f1p8_20210602-055442_4000x3000.sdr.png` | `15cd2f7c2648e3238f41456ceedc8330f6c2bf53a61a1a2f560e61b96d1e7ee6` |
| `png-v3/1000-lilith-photos-general/1002_general_oceanfront-hotel-building_ka-anapali-beach-kaanapali_s21u_iso640-f1p8_20210608-052615_4000x3000.sdr.png` | `17bcef7fefb44b63f32d01ff504ca005f82a891742fce096aa4cac0dc46fc5aa` |
| `png-v3/1000-lilith-photos-general/1003_general_docked-boat-pier_brick-palace-lahaina_s21u_iso50-f2p2_20210609-124821_4000x3000.sdr.png` | `ab58eaceb28a6548a0dbf2bbe5bf35ff2638ad70803b5e00321fa3130d130f86` |
| `png-v3/1000-lilith-photos-general/1004_general_paint-color-samples_s21u_iso200-f1p8_20220304-144007_3000x4000.sdr.png` | `231c517eef3fb98386cdb0bafea30414ac17cbab2626463c80e582242607df0c` |
| `png-v3/1000-lilith-photos-general/1005_general_la-pedrera-rooftop-terrace_l-eixample-barcelona-spain_s21u_iso50-f2p2_20220304-165729_4000x3000.sdr.png` | `4691afe5228af1b4caddf73a87953281dc5df306b4217a294a16c3cca60b9c5e` |
| `png-v3/1000-lilith-photos-general/1006_general_church-steeple-among-trees_s21u_iso50-f1p8_20220304-182325_3000x4000.sdr.png` | `de5c69eb0bab47f20b5a8a99ff0363e992cc68391be56c17a2def0c95a272c21` |
| `png-v3/1200-lilith-interiors/1200_interiors_reflective-egg-sculptures_s23u_iso10-f1p7_20230704-121114_4000x3000.sdr.png` | `7d935f49472cb9f5282b48a2da3dde3eb69030f535260da6b9334a35335ec95a` |
| `png-v3/1200-lilith-interiors/1201_interiors_bar-sign-outside-window_shinjuku-2-shinjuku-japan_s23u_iso12-f1p7_20230706-124619_3000x4000.sdr.png` | `e281b183126aeecdb1ead562a99a815fe7e83b4fcbbaa0743c473d445022f992` |
| `png-v3/1200-lilith-interiors/1202_interiors_abstract-textured-pattern_seattle-center-seattle_s23u_iso1000-f1p7_20230727-164305_3000x4000.sdr.png` | `020cbb1290d5ded0893319e24b9947597b202f438354e362c01bb4bf342b732e` |
| `png-v3/1200-lilith-interiors/1203_interiors_person-in-room_belltown-seattle-washington_s23u_iso800-f1p7_20230727-165257_4000x3000.sdr.png` | `2f7b2a661d2a2965488d01c9b702c5d5e49c143f91fe3ae954cdd2eabeeaa2ea` |
| `png-v3/1200-lilith-interiors/1204_interiors_rustic-outdoor-dining-area_s23u_iso125-f2p2_20230915-124518_3000x4000.sdr.png` | `23579a83cedb6e5009b089e7b1e9c949a6da11ed14a6d7873304b4a170ec5214` |
| `png-v3/1200-lilith-interiors/1205_interiors_modern-living-room_tulum-mexico_s23u_iso160-f1p7_20230917-114545_3000x4000.sdr.png` | `4858b70930bb7f66b94a9642b7cf0237ecb6fb56adffa782ad8d75399362d9ea` |
| `png-v3/1200-lilith-interiors/1206_interiors_tree-like-sculpture-ceiling_azulik-city-arts-francisco-uh-may_s23u_iso80-f2p2_20230917-143302_3000x4000.sdr.png` | `fbb834595b932679385d9cc2be6cb916d7b7b10dcf244b82646da8f47e7f894f` |
| `png-v3/1400-lilith-nature/1400_nature_snowy-mountain-landscape_colorado_s10_iso50-f2p2_20200226-105616_4608x3456.sdr.png` | `7516beec496ae6b44231d37b26d7408e02ddcf4e2dc1b4eb9ea8024c52bda9c1` |
| `png-v3/1400-lilith-nature/1401_nature_dolphin-in-ocean_ala-moana-east-honolulu-hawaii_s21u_iso50-f2p4_20210605-072339_4000x3000.sdr.png` | `9fdb9ae93e4eb74cdbb7c27c4054ad95c8d9e21372b8e34967930c93ced3e16b` |
| `png-v3/1400-lilith-nature/1402_nature_rocky-ocean-coast_spitting-caves-east-honolulu_s21u_iso16-f1p8_20210605-110609_4000x3000.sdr.png` | `56afdc7e89a6b547088843e8d7779116cc95ce76e9bdc1bdb6ec066ce7b6d965` |
| `png-v3/1400-lilith-nature/1403_nature_large-tree-canopy_honolulu-zoo-east-honolulu_s21u_iso50-f2p2_20210605-154933-3_4000x3000.sdr.png` | `d44fbc7aedc39c99c516aeb18eb92ad613f176481cce08d46c9e1a32dc76e626` |
| `png-v3/1400-lilith-nature/1404_nature_palm-trees-sunset_ka-anapali-beach-kaanapali_s21u_iso16-f1p8_20210607-180007_4000x3000.sdr.png` | `9dbdfdf7ad58b38ef0a3f5d962e3e974dc0c4151033c5da11d9bcf056afbc7cf` |
| `png-v3/1400-lilith-nature/1405_nature_sea-cave-waves_keawaiki-bay-hana_s21u_iso16-f1p8_20210608-123424_4000x3000.sdr.png` | `669f4d553f696e032769f977213e18eaed848fe36efeecb9ccd62891b7ae2766` |
