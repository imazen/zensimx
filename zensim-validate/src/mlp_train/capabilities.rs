use super::MlpHyperparams;

/// Normalize structural guarantees and refuse unsupported recipe combinations.
/// Called before CLI data loading and again by the library dispatcher.
/// Panics with the unsupported flag, as the dispatcher historically did.
pub fn validate_training_capabilities(
    hyperparams: &MlpHyperparams,
    has_absolute_group: bool,
    triplet_rows: usize,
) -> MlpHyperparams {
    if hyperparams.smooth_hinge {
        assert!(
            hyperparams.nonneg_distance
                && hyperparams.n_hidden_layers == 1
                && !hyperparams.monotone_cbc
                && hyperparams.steer_pairs.is_none()
                && hyperparams.feature_transforms.is_none()
                && hyperparams.pooled_legs.is_empty()
                && hyperparams.minibatch_size == 1
                && hyperparams.norm_in_norm_weight == 0.0
                && hyperparams.pair_sampling == super::PairSampling::Uniform,
            "smooth hinge requires plain one-layer N, uniform/minibatch1, no transforms, pooled objectives or K penalty"
        );
    }
    if hyperparams.nonneg_thresholds {
        assert!(
            hyperparams.nonneg_distance
                && hyperparams.n_hidden_layers == 1
                && !hyperparams.smooth_hinge
                && hyperparams.steer_pairs.is_none()
                && !hyperparams.pool_head
                && !hyperparams.hybrid_head
                && !hyperparams.per_sample_alpha_head,
            "--nonneg-thresholds requires plain one-layer N without smooth hinge, heads or the K penalty"
        );
    }
    if hyperparams.max_pairs > 0 {
        // A feature mask is allowed: sign-free inputs enter only a pair's v row (see `project_plain`).
        assert!(
            hyperparams.nonneg_thresholds
                && hyperparams.monotone_cbc
                && hyperparams.monotone_product_pairs.is_empty()
                && 2 * (hyperparams.max_pairs
                    + hyperparams.monotone_ramps
                    + hyperparams.free_ramps)
                    <= hyperparams.n_hidden,
            "--max-pairs requires --nonneg-thresholds, --monotone-cbc, no gated products, and room in --hidden"
        );
    }
    if !hyperparams.monotone_product_pairs.is_empty() {
        assert!(
            hyperparams.nonneg_distance
                && hyperparams.monotone_cbc
                && hyperparams.monotone_feature_pin.is_none(),
            "--monotone-gated-products requires --nonneg-distance and --monotone-cbc on every input"
        );
    }
    if hyperparams.free_ramps > 0 {
        assert!(
            hyperparams.nonneg_thresholds
                && hyperparams.monotone_cbc
                && hyperparams.monotone_feature_pin.is_none()
                && hyperparams.free_budget.is_finite()
                && hyperparams.free_budget >= 0.0
                && 2 * (hyperparams.monotone_ramps + hyperparams.free_ramps)
                    <= hyperparams.n_hidden,
            "--free-ramps requires --nonneg-thresholds, --monotone-cbc on every input, a finite \
             --free-budget >= 0 and 2*(ramps + free ramps) <= hidden"
        );
    }
    if hyperparams.monotone_ramps > 0 {
        assert!(
            hyperparams.nonneg_thresholds
                && hyperparams.monotone_cbc
                && hyperparams.monotone_feature_pin.is_none()
                && 2 * hyperparams.monotone_ramps <= hyperparams.n_hidden,
            "--monotone-ramps requires --nonneg-thresholds, --monotone-cbc on every input and 2*ramps <= hidden"
        );
    }
    assert!(
        hyperparams.steer_lambda.is_finite() && hyperparams.steer_lambda >= 0.0,
        "--steer-lambda must be finite and nonnegative"
    );
    assert_eq!(
        hyperparams.steer_pairs.is_some(),
        hyperparams.steer_lambda > 0.0,
        "--steer-pairs and a positive --steer-lambda must be supplied together"
    );
    if hyperparams.steer_pairs.is_some() {
        assert!(
            hyperparams.nonneg_distance
                && !hyperparams.monotone_cbc
                && hyperparams.minibatch_size == 1
                && hyperparams.feature_transforms.is_none()
                && hyperparams.pooled_legs.is_empty()
                && hyperparams.pair_sampling == super::PairSampling::Uniform,
            "K requires plain N, uniform sampling, minibatch 1, no transforms/pooled loss/monotone projection"
        );
    }
    assert!(
        hyperparams.pooled_rank_share.is_finite()
            && (0.0..=1.0).contains(&hyperparams.pooled_rank_share),
        "--pooled-rank-share must be in [0,1]"
    );
    assert!(
        hyperparams.pooled_pearson_weight.is_finite() && hyperparams.pooled_pearson_weight >= 0.0,
        "--pooled-pearson-weight must be nonnegative"
    );
    if !hyperparams.pooled_legs.is_empty() {
        assert!(
            !hyperparams.pool_head
                && !hyperparams.hybrid_head
                && !hyperparams.per_sample_alpha_head
                && hyperparams.minibatch_size == 1
                && hyperparams.norm_in_norm_weight == 0.0
                && hyperparams.pair_sampling == super::PairSampling::Uniform,
            "pooled objective requires plain CPU head, uniform sampling and minibatch-size 1"
        );
    } else {
        assert!(
            hyperparams.pooled_rank_share == 0.0 && hyperparams.pooled_pearson_weight == 0.0,
            "pooled objective requires --pooled-leg"
        );
    }
    // EX-2 std-pool head dispatch (scalar fallback path). Pool-head
    // backprop has not been SIMD-fused yet; we trade ~1.7× per-pair
    // time for the architectural lift (GMSD's std-pooling +
    // Butteraugli's p-norm + IW-style pooling) per
    // `PSYCHOVISUAL_LEARNINGS_FOR_ZENSIM.md §3`. NiN composes with
    // pool-head via the same per-prediction grad-scatter pattern as
    // the standard head (the per-prediction grad is added to the
    // RankNet `dl_dy` before routing through `backprop_step_pool_head`);
    // parallel-batch over chunks is out of scope for the v0 prod
    // wire-in (the sequential mini-batch path covers V_22-mix-LARGE's
    // K=256 recipe at h=128 in ~25 min wall on the 7950X — fast enough
    // that a SIMD/par-chunks port is queued, not load-bearing).
    // `--nonneg-distance` normalization + reachability, ahead of every head
    // branch. ReLU is not a preference here, it is what makes `h(0⃗) = 0⃗` and
    // `h >= 0` true, so the flag OWNS `leaky_alpha` rather than asking the
    // caller to keep two knobs consistent. The refusals below are for
    // combinations whose guarantee this implementation does NOT establish —
    // failing loud is this dispatcher's established response to that class.
    let nonneg_hp;
    let hyperparams = if hyperparams.nonneg_distance {
        assert!(
            !hyperparams.skip_connection,
            "--nonneg-distance is incompatible with --skip-connection: the skip term \
             `x · w_skip` is sign-free (scale-only standardized features are \
             sign-free), so it breaks `raw <= pin` and the C6 guarantee the flag \
             exists to establish. Drop one of the two."
        );
        assert!(
            !(hyperparams.pool_head
                || hyperparams.hybrid_head
                || hyperparams.per_sample_alpha_head),
            "--nonneg-distance is implemented on the plain n_features → n_hidden → 1 \
             path only. The pool/hybrid/alpha heads reduce through additional \
             parameters (reducer_w, rank_w, the alpha gate) whose sign constraints \
             this build does not establish, so the non-negativity guarantee would be \
             claimed and not held. Drop --nonneg-distance or drop the head flag."
        );
        assert!(
            hyperparams.n_hidden_layers <= 1,
            "--nonneg-distance is implemented on the 1-hidden-layer plain path only \
             (--n-hidden-layers {} was requested). Depth >= 2 routes through the \
             alpha-head owner, which this flag refuses.",
            hyperparams.n_hidden_layers
        );
        assert!(
            hyperparams.nonneg_pin.is_finite(),
            "--nonneg-pin must be finite; got {}",
            hyperparams.nonneg_pin
        );
        // The pin is baked through `as f32`. A value that does not round-trip
        // makes `raw(identity) == pin` true in the trainer and FALSE at the f64
        // comparison every gate uses — the "bit-exactly" claim would quietly
        // become "to within an f32 rounding". Refuse, as an unrepresentable
        // `--leaky-alpha` is refused. (Review 2026-09-06.)
        assert!(
            hyperparams.nonneg_pin as f32 as f64 == hyperparams.nonneg_pin,
            "--nonneg-pin {} does not round-trip through f32 (bakes as {}). The pin \
             is stored as an f32 bias, so a value like this makes `raw(identity) == \
             pin` false at f64 and quietly downgrades the guarantee. Use an \
             f32-exact value.",
            hyperparams.nonneg_pin,
            hyperparams.nonneg_pin as f32 as f64
        );
        nonneg_hp = MlpHyperparams {
            leaky_alpha: 0.0,
            ..hyperparams.clone()
        };
        &nonneg_hp
    } else {
        hyperparams
    };

    super::hidden_activation_for(hyperparams.leaky_alpha);
    // `run_parallel_minibatch` / `run_minibatch_with_nin` carry NO absolute term
    // and no polarity reconciliation — they are pure RankNet + PWRC. Before
    // 2026-09-06 a run that set both an absolute term and K>1 (or NiN) silently
    // trained pure rank, throwing the absolute term away without a word, which
    // is the same silent-no-op class the dispatcher already fails loud for.
    // Refuse instead of lying. Rank-only recipes at any K are unaffected.
    if !hyperparams.per_sample_alpha_head
        && !hyperparams.pool_head
        && !hyperparams.hybrid_head
        && (has_absolute_group || hyperparams.nonneg_distance)
        && (hyperparams.parallel_batch && hyperparams.minibatch_size.max(1) > 1
            || hyperparams.norm_in_norm_weight > 0.0)
    {
        panic!(
            "an absolute (mse/both) term is active, but --minibatch-size {} \
             (parallel_batch={}) / --norm-in-norm-weight {} routes pairs through a \
             mini-batch helper that implements RankNet only. The absolute term would \
             be silently discarded and the bake would be byte-identical to a run that \
             never asked for it. Use --minibatch-size 1 with --norm-in-norm-weight 0, \
             or drop the absolute term.",
            hyperparams.minibatch_size, hyperparams.parallel_batch, hyperparams.norm_in_norm_weight,
        );
    }

    // E34 L inherits E33 E5: its linear/identity proof declares no transforms.
    if hyperparams.nonneg_distance && hyperparams.monotone_cbc {
        assert!(
            hyperparams.feature_transforms.is_none(),
            "plain monotone N refuses feature transforms (E34 L identity contract)"
        );
    }

    // The pool and hybrid heads implement NO absolute term — they read
    // `g.loss_mode` only to log it. Before the polarity owner that was merely a
    // silent discard; with it, a `:mse`/`:both` group would flip their RankNet
    // and ladder signs on the strength of a flag they throw away, inverting the
    // bake against its own parent with nothing to anchor it. Refuse, which is
    // this dispatcher's established answer to the class. (Review 2026-09-06.)
    if (hyperparams.pool_head || hyperparams.hybrid_head) && has_absolute_group {
        panic!(
            "--pool-head / --hybrid-head implement no absolute term — they read a \
             group's loss mode only to log it. A `:mse`/`:both` group would flip \
             this run's output polarity (and with it the RankNet target and the TV \
             ladder hinge) on the strength of a term that never fires, inverting \
             the bake with nothing to anchor it. Drop the head flag, or make the \
             groups `:rank`."
        );
    }

    let head_flags = (hyperparams.pool_head as u8)
        + (hyperparams.hybrid_head as u8)
        + (hyperparams.per_sample_alpha_head as u8);
    assert!(
        head_flags <= 1,
        "pool_head / hybrid_head / per_sample_alpha_head are mutually exclusive"
    );

    // Loss-term reachability gates. These MUST live here, in the
    // dispatcher, ahead of every head branch — the versions they replace
    // sat INSIDE `train_mlp_per_sample_alpha_head` and tested
    // `!per_sample_alpha_head`, which is unreachable there by
    // construction. So they were dead code that could never fire, and
    // their doc ("trainer panics if set on other heads") was false:
    // `--mse-weight` on a non-α head silently trained pure rank and threw
    // the flag away. Found 2026-07-15 by a `should_panic` test that did
    // not panic.
    if hyperparams.monotonicity_reg > 0.0 && !hyperparams.per_sample_alpha_head {
        panic!(
            "--monotonicity-reg is only wired on the per_sample_alpha_head \
             path (set --per-sample-alpha-head)."
        );
    }
    if hyperparams.mse_weight > 0.0 && !hyperparams.per_sample_alpha_head && !has_absolute_group {
        // On the plain path the absolute term is opt-in PER GROUP, so a
        // run that sets the weight but flags no group would train pure
        // rank and ignore the flag — exactly the silent failure above.
        panic!(
            "--mse-weight is set but no group opted into an absolute term. \
             On the plain path the regression term is per-group: append \
             `:mse` or `:both` to a --group spec (or use \
             --per-sample-alpha-head)."
        );
    }
    // The rest of the STRATEGY family is in the SAME silent-no-op class, and
    // was unguarded until 2026-09-04: `ema`, `hard_pair`, `dro` and
    // `listwise` are read ONLY inside `train_mlp_per_sample_alpha_head`, so
    // setting one on any other head threw it away without a word.
    //
    // Found while scoping the subset-quality study: `--ema-decay` looked like
    // the natural lever for reducing seed variance — its own CLI doc says
    // "seed-variance reduction" — and it would have silently done nothing on
    // the path every board bake trains through. That failure mode reads as
    // "EMA does not help" when the truth is "EMA never ran".
    //
    // Wiring them through means threading a weight-EMA / DRO / listwise step
    // into three more training loops, a real change to the optimizer and the
    // bake path. Failing loud is this file's established response to the class
    // (see the guards above) and is the honest interim state.
    for (name, on, flag) in [
        ("--ema-decay", hyperparams.ema_decay > 0.0, "ema_decay"),
        (
            "--hard-pair-frac",
            hyperparams.hard_pair_frac > 0.0,
            "hard_pair_frac",
        ),
        ("--dro-eta", hyperparams.dro_eta > 0.0, "dro_eta"),
        (
            "--listwise-weight",
            hyperparams.listwise_weight > 0.0,
            "listwise_weight",
        ),
    ] {
        if on && !hyperparams.per_sample_alpha_head {
            panic!(
                "{name} ({flag}) is only wired on the per_sample_alpha_head path; this run \
                 has per_sample_alpha_head=false, so the flag would be silently ignored and \
                 the bake would be byte-identical to a run that never set it. Add \
                 --per-sample-alpha-head, or drop the flag."
            );
        }
    }
    // Same silent-no-op class as the two guards above: the triplet step lives
    // ONLY inside `train_mlp_per_sample_alpha_head`, so a run that loads a
    // triplet pool but does NOT set --per-sample-alpha-head silently ignores it
    // and produces a bake byte-identical to a no-triplet run. Measured
    // 2026-07-16: depth_v6 (--triplet-weight 0.5, plain 2-layer) == depth_v2
    // byte-for-byte. Fail loud instead of throwing the flag away.
    // Same silent-no-op class as the guards above, and missed until 2026-09-06:
    // `use_2layer` / `use_skip` are read at exactly two lines in this file, both
    // inside `train_mlp_per_sample_alpha_head`. A run that asked for depth 2 or a
    // skip connection on any other head got a 1-layer, skip-less net and a bake
    // byte-identical to one that never asked.
    for (name, on) in [
        ("--n-hidden-layers >= 2", hyperparams.n_hidden_layers >= 2),
        ("--skip-connection", hyperparams.skip_connection),
    ] {
        if on && !hyperparams.per_sample_alpha_head {
            panic!(
                "{name} is only wired on the per_sample_alpha_head path; this run has \
                 per_sample_alpha_head=false, so the architecture flag would be \
                 silently ignored and the bake would be byte-identical to a run that \
                 never set it. Add --per-sample-alpha-head, or drop the flag."
            );
        }
    }
    let triplet_requested = hyperparams.triplet_weight > 0.0;
    if triplet_requested && !hyperparams.per_sample_alpha_head {
        panic!(
            "--triplet-weight/-stimuli/-responses are only wired on the \
             per_sample_alpha_head path; this run has per_sample_alpha_head=false so the \
             loaded triplet pool ({} responses) would be silently ignored (the bake would \
             be byte-identical to a no-triplet run). Set --per-sample-alpha-head, or drop \
             the triplet flags.",
            triplet_rows,
        );
    }

    for (flag, enabled) in [
        ("--anchor-loss-weight", hyperparams.anchor_loss_weight > 0.0),
        (
            "--pjnd-passthrough-weight",
            hyperparams.pjnd_passthrough_weight > 0.0,
        ),
        (
            "--konjnd-aggregation-weight",
            hyperparams.konjnd_aggregation_weight > 0.0,
        ),
        (
            "--cross-codec-eq-weight",
            hyperparams.cross_codec_eq_weight > 0.0,
        ),
        (
            "--cross-codec-rank-preserve-weight",
            hyperparams.cross_codec_rank_preserve_weight > 0.0,
        ),
        (
            "--dynamic-range-floor-weight",
            hyperparams.dynamic_range_floor_weight > 0.0,
        ),
    ] {
        assert!(
            !enabled || hyperparams.per_sample_alpha_head,
            "{flag} is only wired on --per-sample-alpha-head; the requested loss would be ignored"
        );
    }
    for (flag, active) in [
        ("--ranknet-weight", hyperparams.ranknet_weight != 1.0),
        (
            "--tanh-output-head-scale",
            hyperparams.tanh_output_head_scale > 0.0,
        ),
        ("--sigma-weighted-mse", hyperparams.sigma_weighted_mse),
        (
            "--qat-fine-tune-epochs",
            hyperparams.qat_fine_tune_epochs > 0,
        ),
        ("--group-eval-cap", hyperparams.group_eval_cap > 0),
    ] {
        assert!(
            !active || hyperparams.per_sample_alpha_head,
            "{flag} is only wired on --per-sample-alpha-head; the requested behavior would be ignored"
        );
    }
    assert!(
        hyperparams.n_hidden > 0,
        "hidden width must be positive (H1 is supported)"
    );
    assert!(
        !hyperparams.monotone_cbc
            || hyperparams.per_sample_alpha_head
            || hyperparams.nonneg_distance,
        "--monotone-cbc requires --per-sample-alpha-head or plain --nonneg-distance"
    );
    assert!(
        hyperparams.monotone_cbc
            || !(hyperparams.monotone_strict
                || hyperparams.monotone_pin_during_training
                || hyperparams.monotone_feature_pin.is_some()),
        "monotone feature mask/strict/pin-during-training require --monotone-cbc"
    );
    hyperparams.clone()
}
