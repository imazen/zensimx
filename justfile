# Aggregate only completed owner outputs; no new target table is read.
dataset-mix-summary study:
    python3 -B scripts/rev4_featpot/byc_cells.py mix-summary --study {{quote(study)}}

# Caller holds shared heavy.lock and run-heavy; existing bake copies are verified.
dataset-candidate-registry:
    python3 -B scripts/byconstruction/candidate_registry.py

# Copy bytes over the mounted mirror; NAS ownership is managed by the export.
dataset-candidate-mirror destination:
    #!/usr/bin/env bash
    set -euo pipefail
    rsync -r --checksum /mnt/v/output/zensim/candidates-2026-10-25/ {{quote(destination)}}/
    cd {{quote(destination)}}
    sha256sum -c SHA256SUMS

# Metadata-only progress; caller can inspect this while another job holds the lock.
dataset-mix-progress study:
    #!/usr/bin/env bash
    set -euo pipefail
    for arm in Mdata Mw025 Mw100 Mmos Mpair Mlic; do
      folds=kadid,tid2013,konfig,cid22_a25,pipal,pairs,mos7
      case "$arm" in
        Mmos) folds=kadid,tid2013,konfig,cid22_a25,pipal,mos7,full ;;
        Mpair) folds=kadid,tid2013,konfig,cid22_a25,pairs,full ;;
      esac
      just dataset-cell-progress {{quote(study)}}/cells "$arm" "$folds"
    done

# Caller holds shared heavy.lock and run-heavy; refuses unfrozen candidates.
dataset-mix-eval study:
    env ZENSIM_MAX_TIER=v3 ZEN_PANEL_BIN={{quote(study)}}/bin/panel RAYON_NUM_THREADS=2 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 -B scripts/byconstruction/dataset_eval.py mix-eval --study {{quote(study)}} --predict {{quote(study)}}/bin/bake_dial_refit

# Caller holds shared heavy.lock and run-heavy; selection precedes these fits.
dataset-mix-freeze study:
    #!/usr/bin/env bash
    set -euo pipefail
    python3 - {{quote(study)}} "$$" <<'PY'
    from pathlib import Path
    import sys
    runtime = Path(sys.argv[1]) / 'freeze-runtime'
    runtime.mkdir()
    (runtime / 'batch.pid').write_text(sys.argv[2] + '\n')
    (runtime / 'scope.cgroup').write_text(Path('/proc/self/cgroup').read_text())
    PY
    env ZENSIM_MAX_TIER=v3 BYC_COLS={{quote(study)}}/frozen_configuration/cols_c.txt BYC_CELLS={{quote(study)}}/cells REV4_V2_BIN_DIR={{quote(study)}}/bin ZEN_PANEL_BIN={{quote(study)}}/bin/panel RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 -B scripts/rev4_featpot/byc_cells.py mix-freeze --study {{quote(study)}}

# Queue individual existing-owner operations; heavy work takes the shared lock.
dataset-mix-queue study:
    #!/usr/bin/env bash
    set -euo pipefail
    study={{quote(study)}}
    for arm in Mdata Mw025 Mw100 Mmos Mpair Mlic; do
      flock "$HOME/tmp/heavy.lock" "$HOME/work/claudehints/scripts/run-heavy" --mem 36G --jobs 12 -- env TMPDIR="$study/scratch" OPENBLAS_NUM_THREADS=1 just dataset-mix-cells "$study" "$arm" 2>&1 | tee "$study/cells_$arm.log"
    done
    touch "$study/CELLS_COMPLETE"

dataset-mix-cells study arm:
    #!/usr/bin/env bash
    set -euo pipefail
    study={{quote(study)}}
    arm={{quote(arm)}}
    test -f "$study/PREPARE_COMPLETE"
    root="$study/$arm/root"
    folds=kadid,tid2013,konfig,cid22_a25,pipal,pairs,mos7
    case "$arm" in
      Mdata) root=/var/tmp/datalegs-20261025/rev7data/root; folds=pipal,pairs,mos7 ;;
      Mmos) folds=kadid,tid2013,konfig,cid22_a25,pipal,mos7,full ;;
      Mpair) folds=kadid,tid2013,konfig,cid22_a25,pairs,full ;;
      Mw025|Mw100|Mlic) ;;
      *) exit 2 ;;
    esac
    mkdir -p "$study/cells/Mdata" "$study/scratch/$arm"
    for fold in kadid tid2013 konfig cid22_a25; do
      for seed in 0 1 2; do
        dest="$study/cells/Mdata/without_${fold}_s$seed"
        if [ ! -e "$dest" ]; then ln -s "/var/tmp/datalegs-20261025/cells/Mdata/without_${fold}_s$seed" "$dest"; fi
      done
    done
    export BYC_COLS="$study/frozen_configuration/cols_c.txt" BYC_ROOT="$root" REV4_V2_BIN_DIR="$study/bin" BYC_CELLS="$study/cells" BYC_SCRATCH="$study/scratch"
    export ZENSIM_MAX_TIER=v3 RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 ZEN_PANEL_BIN="$study/bin/panel"
    # Twelve single-thread fits under 36 GiB; large new-data predictions serialize.
    python3 -B scripts/rev4_featpot/byc_cells.py run --arm mono:thr:mp32 --label "$arm" --folds "$folds" --seeds 0,1,2 --jobs 12

# Caller holds shared heavy.lock and run-heavy; no new labels outside TRAIN.
dataset-mix-e21 study out arms='Mdata,Mw025,Mw100,Mmos,Mpair,Mlic':
    #!/usr/bin/env bash
    set -euo pipefail
    out={{quote(out)}}
    test ! -e "$out/e21.json"
    mkdir -p "$out/label_reads"
    env BYC_ROOT=/var/tmp/datalegs-20261025/rev7data/root BYC_CELLS={{quote(study)}}/cells BYC_LABEL_LEDGER_DIR="$out/label_reads" OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -B scripts/rev4_featpot/byc_cells.py score --arms {{quote(arms)}} --paired-with Mdata > "$out/e21.json"

# Caller holds shared heavy.lock and run-heavy; no new labels outside TRAIN.
dataset-mix-mixtures study arms='Mdata,Mw025,Mw100,Mmos,Mpair,Mlic':
    #!/usr/bin/env bash
    set -euo pipefail
    study={{quote(study)}}
    arms={{quote(arms)}}
    python3 - "$study" "$arms" <<'PY'
    from pathlib import Path
    import shutil, sys
    study, arms = Path(sys.argv[1]), sys.argv[2].split(',')
    assert arms and len(arms) == len(set(arms))
    assert set(arms) <= {'Mdata','Mw025','Mw100','Mmos','Mpair','Mlic'}
    for name in ('mixture','mixture_global'):
        out, source = study/name, Path('/var/tmp/datalegs-20261025')/name
        images = Path('/mnt/v/output/zensim/byconstruction-2026-10-22')/name/'img'
        if not out.exists():
            out.mkdir()
            shutil.copyfile(source/'manifest.json', out/'manifest.json')
            (out/'img').symlink_to(images)
        assert (out/'manifest.json').read_bytes() == (source/'manifest.json').read_bytes()
        assert (out/'img').is_symlink() and (out/'img').resolve() == images.resolve()
        for fold in ('kadid','tid2013','konfig','cid22_a25'):
            for seed in range(3):
                name = f'scores_Mdata_{fold}_s{seed}.json'
                dest = out/name
                if dest.exists():
                    assert dest.read_bytes() == (source/name).read_bytes(), str(dest)
                else:
                    shutil.copyfile(source/name, dest)
    PY
    IFS=, read -r -a requested <<< "$arms"
    export BYC_CELLS="$study/cells" ZENSIM_MAX_TIER=v3 ZENSIM_PYRAMID=xyb RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
    for set in mixture mixture_global; do
      MIX="$study/$set" BYC_MIX_FOLDS=kadid,tid2013,konfig,cid22_a25,pipal,pairs,mos7,full bash scripts/byconstruction/ring_mixture.sh "$study/bin" raw 12 "${requested[@]}"
    done

# Caller holds shared heavy.lock and run-heavy; no new labels outside TRAIN.
dataset-mix-local-mixtures study:
    #!/usr/bin/env bash
    set -euo pipefail
    study={{quote(study)}}
    just dataset-mix-mixtures "$study" Mdata,Mw025,Mlic
    mkdir "$study/local-mixtures"
    for set in mixture mixture_global; do
      python3 -B scripts/byconstruction/mixture_test.py summary --out "$study/$set" --json-out "$study/local-mixtures/$set.json" Mdata Mw025 Mlic > "$study/local-mixtures/$set.txt"
      python3 -B scripts/byconstruction/mixture_test.py report --out "$study/$set" > "$study/local-mixtures/${set}_all-cells.json"
    done

# Caller holds shared heavy.lock and run-heavy; no new labels outside TRAIN.
dataset-mix-diagnostics study out arms='Mdata,Mw025,Mw100,Mmos,Mpair,Mlic':
    #!/usr/bin/env bash
    set -euo pipefail
    study={{quote(study)}}
    out={{quote(out)}}
    arms={{quote(arms)}}
    mkdir -p "$out"
    test ! -e "$out/type_bias.json"
    test ! -e "$out/knobinv.json"
    test ! -e "$out/panel.json"
    export BYC_ROOT=/var/tmp/datalegs-20261025/rev7data/root BYC_CELLS="$study/cells" REV4_V2_BIN_DIR="$study/bin" AIC_BIN="$study/bin" ZEN_PANEL_BIN="$study/bin/panel"
    export ZENSIM_MAX_TIER=v3 BYC_LABEL_LEDGER_DIR="$out/label_reads" RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
    IFS=, read -r -a requested <<< "$arms"
    for arm in "${requested[@]}"; do
      test ! -e "$out/bytype_$arm.json"
      python3 -B scripts/rev4_featpot/byc_cells.py bytype --arm "$arm" > "$out/bytype_$arm.json"
    done
    python3 -B scripts/byconstruction/type_bias.py "${requested[@]}" --out "$out/type_bias.json" > "$out/type_bias.txt"
    python3 -B scripts/rev4_featpot/byc_cells.py knobinv --arms "$arms" --out "$out/knobinv.json" > "$out/knobinv.txt"
    python3 -B scripts/byconstruction/aic_panel.py --out "$out/panel.json" --arms "base-c,$arms" --geo --no-aic --no-public-test > "$out/panel.txt" 2>&1

# Caller holds shared heavy.lock and run-heavy; no new labels outside TRAIN.
dataset-mix-scorecard study:
    #!/usr/bin/env bash
    set -euo pipefail
    study={{quote(study)}}
    test -f "$study/CELLS_COMPLETE"
    out="$study/scorecard"
    mkdir "$out"
    export BYC_ROOT=/var/tmp/datalegs-20261025/rev7data/root BYC_CELLS="$study/cells" REV4_V2_BIN_DIR="$study/bin" AIC_BIN="$study/bin" ZEN_PANEL_BIN="$study/bin/panel"
    export ZENSIM_MAX_TIER=v3 BYC_LABEL_LEDGER_DIR="$out/label_reads" RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
    python3 -B scripts/rev4_featpot/byc_cells.py mix-audit --study "$study" > "$out/native-audit.txt"
    just dataset-mix-binary-controls "$study" > "$out/binary-controls.txt"
    just dataset-mix-forced-choice-controls "$study" > "$out/forced-choice-controls.txt"
    arms=Mdata,Mw025,Mw100,Mmos,Mpair,Mlic
    just dataset-mix-e21 "$study" "$out" "$arms"
    just dataset-mix-diagnostics "$study" "$out" "$arms"
    just dataset-mix-mixtures "$study" "$arms" > "$study/mixture-score.log" 2>&1
    for set in mixture mixture_global; do
      mix="$study/$set"
      python3 -B scripts/byconstruction/mixture_test.py summary --out "$mix" --json-out "$out/$set.json" Mdata Mw025 Mw100 Mmos Mpair Mlic > "$out/$set.txt" 2>&1
      python3 -B scripts/byconstruction/mixture_test.py report --out "$mix" > "$out/${set}_all-cells.json"
    done
    just dataset-mix-mixture-controls "$study"
    python3 -B scripts/rev4_featpot/byc_cells.py mix-select --study "$study" > "$out/selection.txt"

# Bind completed cell trainer receipts to the frozen study binaries, without labels.
dataset-mix-binary-controls study scope='complete':
    #!/usr/bin/env bash
    set -euo pipefail
    python3 - {{quote(study)}} {{quote(scope)}} <<'PY'
    import hashlib, json, sys
    from pathlib import Path
    study, scope = Path(sys.argv[1]), sys.argv[2]
    assert scope in ('complete', 'partial')
    sha = lambda path: hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()
    pins = study/'BIN_SHA256SUMS'
    assert sha(pins) == 'ef2d70d6e62b4c4bbeb1a553ffabf1e17ceefa1d62d99ac50bcdc21859965202'
    binaries = {}
    for line in pins.read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        path = study/'bin'/Path(name.lstrip('*')).name
        assert sha(path) == digest, str(path)
        binaries[path.name] = digest
    assert len(binaries) == 6
    ownership = json.loads((study/'FLEET_OWNERSHIP.json').read_text())
    assert ownership['source_commit'] == 'b7cb1a224dc7a4835e7259fe41c447ffa4c811e4'
    old = ('kadid', 'tid2013', 'konfig', 'cid22_a25')
    local = {a:[dict(fold=f, seed=s) for f in (*old,'pipal','pairs','mos7') for s in range(3)]
             for a in ownership['local_arms']}
    expected = {**local, **ownership['fleet_cells']}
    assert sum(map(len, expected.values())) == 123
    checked, missing = [], []
    for arm, cells in expected.items():
        for cell in cells:
            fold, seed = cell['fold'], cell['seed']
            path = study/'cells'/arm/f'without_{fold}_s{seed}'/'byc_record.json'
            if not path.is_file():
                missing.append(str(path))
                continue
            record = json.loads(path.read_text())
            reused = arm == 'Mdata' and fold in old
            trainer = ('c966d725b43244ff30c560bff1c9b3ed898b3331ae640afd460239b455b738f2'
                       if reused else binaries['zensim_mlp_train'])
            assert (record['fold'], record['seed'], record['trainer_sha256']) == (fold, seed, trainer), str(path)
            checked.append(dict(arm=arm, fold=fold, seed=seed, trainer_sha256=trainer, record_sha256=sha(path)))
    if scope == 'complete':
        assert not missing, missing
        native = json.loads((study/'NATIVE_CELL_CONTROLS.json').read_text())
        identity = lambda rows: {(r['arm'], r['fold'], r['seed'], r['record_sha256']) for r in rows}
        assert identity(checked) == identity(native['cells'])
    result = dict(schema='dataset-mix-binary-controls-v1', scope=scope, binary_sha256=binaries,
                  verified_cells=len(checked), missing=missing, cells=checked)
    output = study/('BINARY_CELL_CONTROLS.json' if scope == 'complete' else 'BINARY_CELL_CONTROLS.partial.json')
    assert not output.exists(), 'preserve the earlier binary audit'
    output.write_text(json.dumps(result, indent=1)+'\n')
    print(json.dumps(dict(scope=scope, verified_cells=len(checked), pending_cells=len(missing), binaries=len(binaries))))
    PY

# Compare every exact TRAIN hard-agreement reference with the native owner.
dataset-mix-forced-choice-controls study:
    #!/usr/bin/env bash
    set -euo pipefail
    python3 - {{quote(study)}} <<'PY'
    import hashlib, json, sys
    from pathlib import Path
    from collections import defaultdict
    import pandas as pd
    import pyarrow.parquet as pq
    sys.path.insert(0, str(Path.cwd()/'scripts/byconstruction'))
    sys.path.insert(0, str(Path.cwd()/'scripts/rev4_featpot'))
    from dataset_eval import pair_agreement
    from subjective_training_leg import log_dataset_read
    study = Path(sys.argv[1])
    destination = study/'FORCED_CHOICE_CONTROLS.json'
    assert not destination.exists(), 'preserve the completed point controls'
    root = Path('/var/tmp/datalegs-20261025/rev7data/root')
    packet = json.loads((root/'datasets/receipt.json').read_text())
    frozen = json.loads((root/'wide/frozen.json').read_text())
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    assert sha(root/'datasets/receipt.json') == frozen['dataset_legs_sha256']
    groups = {r['collection']: r for r in packet['groups']}
    reads = study/'scorecard/forced-choice-label-reads'
    reads.mkdir()
    cache, checked = {}, []
    for record in json.loads((study/'NATIVE_CELL_CONTROLS.json').read_text())['cells']:
        cell = study/'cells'/record['arm']/f"without_{record['fold']}_s{record['seed']}"
        result = json.loads((cell/'byc_record.json').read_text())
        for collection, value in result.get('dataset_train_statistics', {}).items():
            if collection not in ('bapps', 'pieapp', 'fg-iqa'):
                continue
            if collection not in cache:
                group = groups[collection]
                table = Path(group['path'])
                assert sha(table) == group['table_sha256']
                log_dataset_read(reads, table, group['rows'], ['ref_basename', 'human_score'],
                                 'TRAIN-native-forced-choice-parity', collection=collection,
                                 table_sha256=group['table_sha256'])
                labels = pq.read_table(table, columns=['ref_basename', 'human_score']).to_pylist()
                pairs = defaultdict(list)
                for i, row in enumerate(labels):
                    pairs[row['ref_basename']].append((row['human_score'], i))
                rows = []
                for key, points in pairs.items():
                    assert len(points) == 2 and {v[0] for v in points} == {0., 1.}
                    lose, win = sorted(points)
                    rows.append(dict(group=key, a=win[1], b=lose[1], target=1.))
                cache[collection] = rows
            path = cell/f'pred_dataset_{collection}.tsv'
            assert sha(path) == value['prediction_sha256']
            prediction = pd.read_csv(path, sep='\t')
            assert prediction.row_idx.tolist() == list(range(len(prediction)))
            scores = {i: float(v) for i, v in enumerate(prediction.pred.to_numpy())}
            # Only the point is checked; this helper's bootstrap output is unused.
            acc, _, _, count = pair_agreement(cache[collection], scores, resamples=1)
            error = abs(acc - value['value'])
            assert error <= 1e-8 and count == value['groups'], (cell, collection, error)
            checked.append(dict(arm=record['arm'], fold=record['fold'], seed=record['seed'],
                                collection=collection, pairs=count, reference=value['value'],
                                native=acc, absolute_error=error))
    assert checked
    result = dict(schema='dataset-TRAIN-hard-agreement-native-controls-v1', role='TRAIN',
                  definition_unchanged=True, comparisons=len(checked),
                  point_tolerance=1e-8, reason='native CSV point precision', checks=checked)
    destination.write_text(json.dumps(result, indent=1)+'\n')
    print(json.dumps(dict(comparisons=len(checked), maximum_absolute_error=max(v['absolute_error'] for v in checked))))
    PY

# Check every completed model against both sets' actual scored bake hashes.
dataset-mix-mixture-controls study:
    #!/usr/bin/env bash
    set -euo pipefail
    python3 - {{quote(study)}} <<'PY'
    import hashlib, json, sys
    from pathlib import Path
    study = Path(sys.argv[1])
    cells = json.loads((study/'NATIVE_CELL_CONTROLS.json').read_text())['cells']
    destination = study/'MIXTURE_CELL_CONTROLS.json'
    assert not destination.exists(), 'preserve the completed mixture audit'
    inventories = {}
    for name in ('mixture', 'mixture_global'):
        report = json.loads((study/f'scorecard/{name}_all-cells.json').read_text())
        expected = {f"{r['arm']}_{r['fold']}_s{r['seed']}" for r in cells}
        assert set(report) == expected, (name, 'mixture model coverage differs')
        comparisons = len(json.loads((study/name/'manifest.json').read_text())['rows'])
        for record in cells:
            key = f"{record['arm']}_{record['fold']}_s{record['seed']}"
            cell = study/'cells'/record['arm']/f"without_{record['fold']}_s{record['seed']}"
            dense = list((cell/'dense').glob('*.bin'))
            assert len(dense) == 1, str(cell)
            score = json.loads((study/name/f'scores_{key}.json').read_text())
            assert score['bake_sha256'] == hashlib.sha256(dense[0].read_bytes()).hexdigest(), key
            assert score['ring_form'] == 'raw' and score['pyramid'] == 'xyb', key
            assert report[key]['n'] == comparisons, key
        inventories[name] = dict(models=len(expected), comparisons_per_model=comparisons,
                                 score_bake_hashes_verified=True, all_native_cell_records_covered=True)
    destination.write_text(json.dumps(inventories, indent=1)+'\n')
    print(json.dumps(inventories))
    PY

# Separate queue entry for an addendum while the original five-arm queue runs.
dataset-mix-licence-queue study:
    #!/usr/bin/env bash
    set -euo pipefail
    study={{quote(study)}}
    flock "$HOME/tmp/heavy.lock" "$HOME/work/claudehints/scripts/run-heavy" --mem 36G --jobs 12 -- env TMPDIR="$study/scratch" OPENBLAS_NUM_THREADS=1 just dataset-mix-licence-run "$study" 2>&1 | tee "$study/cells_Mlic.log"

dataset-mix-licence-run study:
    just dataset-leg-tests
    just lint-scripts
    just dataset-mix-licence-prepare {{quote(study)}}
    just dataset-mix-cells {{quote(study)}} Mlic

# Caller holds shared heavy.lock and run-heavy; copies only policy metadata.
dataset-mix-licence-prepare study:
    python3 -B scripts/byconstruction/ring_root.py mix-licence --study {{quote(study)}}

# Caller holds shared heavy.lock and run-heavy; all destinations must be fresh.
dataset-mix-prepare study:
    #!/usr/bin/env bash
    set -euo pipefail
    study={{quote(study)}}
    source=/mnt/v/output/zensim/byconstruction-2026-10-22/rev7raw/root
    admission=/mnt/v/zensim/dataset-admission-20261025
    features=/mnt/v/zensim/dataset-features-rev7-20261025
    library=/mnt/tower/input/iqa/datasets
    mkdir "$study/frozen_configuration"
    cp "$HOME/tmp/byc/cols_c.txt" "$study/frozen_configuration/cols_c.txt"
    cp scripts/rev4_featpot/e33_fx1_declaration.json "$study/frozen_configuration/"
    columns="$study/frozen_configuration/cols_c.txt"
    export OPENBLAS_NUM_THREADS=1 TMPDIR="$study/scratch"
    jj log -r @ --no-graph -T 'commit_id ++ "\n"' > "$study/BUILD_SOURCE_COMMIT.txt"
    jj diff --from 337382e4 --to @ --git > "$study/BUILD_SOURCE_CHANGES.patch"
    sha256sum Cargo.lock Cargo.toml zensim-validate/src/subjective_training_leg.rs scripts/rev4_featpot/subjective_training_leg.py scripts/rev4_featpot/byc_cells.py scripts/rev4_featpot/v2_confirm_fit.py scripts/byconstruction/ring_root.py > "$study/BUILD_SOURCE_FILES_SHA256SUMS"
    just dataset-leg-tests
    just dataset-leg-native-tests /var/tmp/datalegs-target
    CARGO_TARGET_DIR=/var/tmp/datalegs-target just clippy
    just dataset-trainer-build /var/tmp/datalegs-target
    cp /var/tmp/datalegs-target/release/zensim_mlp_train "$study/bin/"
    for name in bake_dial_refit panel byc_fsid byc_predict score_pair_with_bake; do
      cp -L /var/tmp/datalegs-20261025/bin/$name "$study/bin/"
    done
    sha256sum "$study"/bin/* > "$study/BIN_SHA256SUMS"
    mkdir "$study/default-control"
    args=(datasets --root "$source" --out "$study/default-control/root" --admission "$admission" --features "$features" --library "$library" --columns "$columns")
    python3 -B "$study/oracle/scripts/byconstruction/ring_root.py" "${args[@]}"
    mv "$study/default-control" "$study/default-control.bak"
    python3 -B scripts/byconstruction/ring_root.py "${args[@]}"
    diff -qr "$study/default-control.bak/root" "$study/default-control/root" > "$study/default-control.diff"
    sha256sum "$study/default-control.bak/root/wide/frozen.json" "$study/default-control/root/wide/frozen.json" > "$study/DEFAULT_CONTROL_SHA256SUMS"
    for arm in Mw025 Mw100 Mmos Mpair; do
      weight=0.5
      collections=pipal,mdid,live-md,cid-iq,cid-oa,vcl-fer,irccyn-ivc,colourlab-rating-variability-2022,bapps,pieapp,fg-iqa
      case "$arm" in
        Mw025) weight=0.25 ;;
        Mw100) weight=1.0 ;;
        Mmos) collections=pipal,mdid,live-md,cid-iq,cid-oa,vcl-fer,irccyn-ivc,colourlab-rating-variability-2022 ;;
        Mpair) collections=bapps,pieapp,fg-iqa ;;
      esac
      just dataset-root "$source" "$study/$arm/root" "$admission" "$features" "$library" "$columns" "$weight" "$collections"
    done
    just dataset-mix-licence-prepare "$study"
    touch "$study/PREPARE_COMPLETE"

# Synthetic fixture tables; no provided-library response table is opened.
dataset-leg-tests:
    python3 -B -m unittest discover -s scripts/tests -p 'test_dataset_legs.py' -v
    python3 -B -m unittest discover -s scripts/tests -p 'test_subjective_training_leg.py' -v
    python3 -B -m unittest discover -s scripts/tests -p 'test_dataset_votes.py' -v

# Caller holds heavy.lock and run-heavy; preserves pinned sampler and E29 controls.
dataset-order-native-tests target:
    cargo test --locked -p zensim-validate --lib sampling:: --target-dir {{quote(target)}}
    cargo test --locked -p zensim-validate --test matched_codec --test subjective_rank --test subjective_vote --target-dir {{quote(target)}}
    cargo test --locked -p zensim-validate --bin zensim_mlp_train matched_codec:: --target-dir {{quote(target)}}

# Caller holds heavy.lock and run-heavy; this creates fresh matching predictions.
dataset-order-prepare study:
    OPENBLAS_NUM_THREADS=1 python3 -B scripts/rev4_featpot/matched_codec.py {{quote(study)}}

dataset-order-tests:
    python3 -B -m unittest discover -s scripts/tests -p 'test_matched_codec.py' -v

dataset-eval-tests binary:
    ZEN_PANEL_BIN={{quote(binary)}} python3 -B -m unittest discover -s scripts/tests -p 'test_dataset_eval.py' -v

# Includes the native statistics owner used by the evaluation tests.
dataset-tests binary:
    ZEN_PANEL_BIN={{quote(binary)}} python3 -B -m unittest discover -s scripts/tests -p 'test_dataset_*.py' -v

# Caller holds shared heavy.lock and run-heavy. TMPDIR must be disk-backed.
dataset-leg-native-tests target:
    cargo test --locked -p zensim-validate --bin zensim_mlp_train --target-dir {{quote(target)}} subjective_training_leg:: -- --nocapture
    cargo test --locked -p zensim-validate --test subjective_rank --target-dir {{quote(target)}}

# Caller holds heavy.lock and run-heavy; includes the unchanged hard-rank controls.
dataset-vote-native-tests target:
    cargo test --locked -p zensim-validate --lib --target-dir {{quote(target)}} mlp_train::subjective_rank::
    cargo test --locked -p zensim-validate --test subjective_rank --test subjective_vote --target-dir {{quote(target)}}

dataset-trainer-build target:
    cargo build --locked --release -p zensim-validate --bin zensim_mlp_train --target-dir {{quote(target)}}

dataset-root source out admission features library columns weight='0.5' collections='pipal,mdid,live-md,cid-iq,cid-oa,vcl-fer,irccyn-ivc,colourlab-rating-variability-2022,bapps,pieapp,fg-iqa':
    python3 -B scripts/byconstruction/ring_root.py datasets --root {{quote(source)}} --out {{quote(out)}} --admission {{quote(admission)}} --features {{quote(features)}} --library {{quote(library)}} --columns {{quote(columns)}} --nominal-weight {{quote(weight)}} --collections {{quote(collections)}}

# Fresh root only; fixed original Mlic memberships/masses and corrected policy.
# Caller holds heavy.lock and run-heavy, serialized with training/evaluation.
dataset-vote-root source out admission features library columns confidence audit='' audit_sha='':
    #!/usr/bin/env bash
    set -euo pipefail
    extra=()
    if [[ -n {{quote(audit)}} || -n {{quote(audit_sha)}} ]]; then
      extra+=(--bapps-vote-audit {{quote(audit)}} --bapps-vote-audit-sha256 {{quote(audit_sha)}})
    fi
    python3 -B scripts/byconstruction/ring_root.py datasets --root {{quote(source)}} --out {{quote(out)}} --admission {{quote(admission)}} --features {{quote(features)}} --library {{quote(library)}} --columns {{quote(columns)}} --nominal-weight 0.5 --collections bapps --bapps-votes --confidence {{quote(confidence)}} "${extra[@]}"

# Caller holds heavy.lock and run-heavy; selected native TRAIN judgments only.
dataset-vote-source-audit root admission library policy archive output:
    OPENBLAS_NUM_THREADS=1 python3 -B scripts/byconstruction/bapps_vote_audit.py --root {{quote(root)}} --admission {{quote(admission)}} --library {{quote(library)}} --policy {{quote(policy)}} --archive {{quote(archive)}} --output {{quote(output)}}

# Build only. Caller holds heavy.lock and run-heavy; original mix selection stays first.
dataset-subsets-root source out columns:
    python3 -B scripts/byconstruction/ring_root.py datasets --additional-subsets --root {{quote(source)}} --out {{quote(out)}} --admission /mnt/v/zensim/dataset-subsets-20261025/admission --features /mnt/v/zensim/dataset-subsets-20261025/features --library /mnt/tower/input/iqa/datasets --columns {{quote(columns)}} --nominal-weight 0.5

# Label-free linkage, preserving the original prepared root. No fitting.
dataset-subsets-links source out:
    python3 -B scripts/byconstruction/ring_root.py dataset-links --root {{quote(source)}} --out {{quote(out)}}

# Wrapper/receipt admission only: no target payload or optimizer is opened.
dataset-subsets-admission root out columns:
    #!/usr/bin/env bash
    set -euo pipefail
    python3 - {{quote(root)}} {{quote(out)}} {{quote(columns)}} <<'PY'
    from pathlib import Path
    import json, sys
    sys.path.insert(0, 'scripts/rev4_featpot')
    from byc_cells import dataset_holdout_collections
    from subjective_training_leg import dataset_groups, sha
    root, out, columns = map(Path, sys.argv[1:])
    selected = [int(v) for v in columns.read_text().strip().split(',')]
    out.mkdir()
    rows = {}
    for fold in ('kadid','tid2013','konfig','cid22_a25','pipal','mos7','pairs'):
        _, excluded = dataset_holdout_collections(root, fold)
        dest = out/fold
        dest.mkdir()
        groups, authority = dataset_groups(root, [], selected, fold if fold in ('kadid','tid2013','konfig','cid22_a25') else 'full', 0, 1101, 101, dest, exclude=excluded)
        receipt = json.loads((dest/'DATASET_GROUP_SELECTION.json').read_text())
        rows[fold] = dict(excluded=list(excluded), retained_collections=receipt['retained_collections'], groups=len(groups), authority_sha256=authority[1])
    assert not list(out.rglob('*.bin'))
    result = dict(schema='linked-subset-wrapper-admission-control-v1', root_sha256=sha(root/'wide/frozen.json'), folds=rows, target_payloads_opened=False, optimizer_invoked=False)
    (out/'VERIFIED.json').write_text(json.dumps(result, indent=1)+'\n')
    print(json.dumps(result))
    PY

dataset-full root binary dest columns:
    env REV4_V2_BIN_DIR={{quote(binary)}} ZENSIM_MAX_TIER=v3 RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -B scripts/rev4_featpot/v2_confirm_fit.py --root {{quote(root)}} --spec sel:3b7bd5ebe929@h32:H128:cv16:cf98:fx1:mono:thr:mp32 --columns "$(cat {{quote(columns)}})" --head N --seed-index 0 --strict-admission --data-role-decision {{quote(root)}}/human_role_decision.json --dest {{quote(dest)}} --train-only --pack-production --e33-output-stage
    {{quote(binary)}}/bake_dial_refit certify-monotone --bake {{quote(dest)}}/refit/production-f16.bin

dataset-cell root binary cells scratch label fold seed jobs='1':
    env BYC_ROOT={{quote(root)}} REV4_V2_BIN_DIR={{quote(binary)}} BYC_CELLS={{quote(cells)}} BYC_SCRATCH={{quote(scratch)}} ZENSIM_MAX_TIER=v3 RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -B scripts/rev4_featpot/byc_cells.py run --arm mono:thr:mp32 --label {{quote(label)}} --folds {{quote(fold)}} --seeds {{quote(seed)}} --jobs {{quote(jobs)}}

# Caller holds heavy.lock and run-heavy. Payloads and old study remain immutable.
dataset-neff-root source out:
    OPENBLAS_NUM_THREADS=1 python3 -B scripts/byconstruction/ring_root.py dataset-neff --root {{quote(source)}} --out {{quote(out)}}

# The caller owns heavy.lock and a 36G/12-job run-heavy scope; no runtime benchmarks.
dataset-neff-chain study source_commit:
    OPENBLAS_NUM_THREADS=1 python3 -B scripts/byconstruction/mlic_neff_chain.py {{quote(study)}} --source-commit {{quote(source_commit)}}

# Same coordinator, independent registered trial; includes a new-binary Mlic control.
dataset-vote-chain study source_commit trainer_sha256:
    OPENBLAS_NUM_THREADS=1 python3 -B scripts/byconstruction/mlic_neff_chain.py {{quote(study)}} --source-commit {{quote(source_commit)}} --candidate MlicVote --trainer-sha256 {{quote(trainer_sha256)}}

dataset-cell-progress cells arm='Mdata' folds='kadid,tid2013,konfig,cid22_a25':
    env BYC_CELLS={{quote(cells)}} OPENBLAS_NUM_THREADS=1 python3 -B scripts/rev4_featpot/byc_cells.py progress --arm {{quote(arm)}} --folds {{quote(folds)}}

dataset-run-progress study cells arm folds seeds='0,1,2':
    env BYC_CELLS={{quote(cells)}} OPENBLAS_NUM_THREADS=1 python3 -B scripts/rev4_featpot/byc_cells.py progress --arm {{quote(arm)}} --folds {{quote(folds)}} --seeds {{quote(seeds)}} --run-state {{quote(study)}}/RUN_STATE.json --output {{quote(study)}}/CURRENT_RUN.json --brief

dataset-counts root out:
    OPENBLAS_NUM_THREADS=1 python3 -B scripts/byconstruction/ring_root.py dataset-report --root {{quote(root)}} --out {{quote(out)}}

# Fail before candidate cells unless the rebuilt trainer reproduces Mr7.
dataset-cell-control cells:
    cmp {{quote(cells)}}/Mcontrol/without_kadid_s0/pred_kadid.tsv {{quote(cells)}}/Mr7/without_kadid_s0/pred_kadid.tsv
    sha256sum {{quote(cells)}}/Mcontrol/without_kadid_s0/refit/last.bin {{quote(cells)}}/Mr7/without_kadid_s0/refit/last.bin

# Preserve an independent comparison excluding only invocation provenance.
dataset-model-control cells binary out:
    mkdir {{quote(out)}}
    {{quote(binary)}}/bake_dial_refit strip --in {{quote(cells)}}/Mcontrol/without_kadid_s0/refit/last.bin --out {{quote(out)}}/Mcontrol.bin --key zentrain.repro
    {{quote(binary)}}/bake_dial_refit strip --in {{quote(cells)}}/Mr7/without_kadid_s0/refit/last.bin --out {{quote(out)}}/Mr7.bin --key zentrain.repro
    cmp {{quote(out)}}/Mcontrol.bin {{quote(out)}}/Mr7.bin

dataset-eval out model binary:
    python3 -B scripts/byconstruction/dataset_eval.py score --out {{quote(out)}} --predict {{quote(binary)}}/bake_dial_refit --model Mdata={{quote(model)}}
    ZEN_PANEL_BIN={{quote(binary)}}/panel python3 -B scripts/byconstruction/dataset_eval.py report --out {{quote(out)}}

# Paired A − B on each MOS collection through `panel evaluate`; the evaluator is multithreaded, so cap it.
dataset-eval-paired out dest a vs binary threads='2':
    RAYON_NUM_THREADS={{quote(threads)}} ZEN_PANEL_BIN={{quote(binary)}}/panel nice -n19 python3 -B scripts/byconstruction/dataset_eval.py paired --out {{quote(out)}} --dest {{quote(dest)}} --a {{quote(a)}} --vs {{quote(vs)}}

dataset-eval-prepare out collection:
    python3 -B scripts/byconstruction/dataset_eval.py prepare --out {{quote(out)}} --collections {{quote(collection)}}

dataset-eval-bapps out model binary:
    python3 -B scripts/byconstruction/dataset_eval.py score --out {{quote(out)}} --predict {{quote(binary)}}/bake_dial_refit --model Mdata={{quote(model)}} --model M7="$HOME/tmp/byc/full/mono-thr-mp32-r7_s0/refit/production-f16.bin" --model B7="$HOME/tmp/byc/perf/b_rev7.bin"
    ZEN_PANEL_BIN={{quote(binary)}}/panel python3 -B scripts/byconstruction/dataset_eval.py report --out {{quote(out)}}

dataset-scorecard root binary cells out:
    #!/usr/bin/env bash
    set -euo pipefail
    export BYC_ROOT={{quote(root)}} REV4_V2_BIN_DIR={{quote(binary)}} BYC_CELLS={{quote(cells)}} AIC_BIN={{quote(binary)}} ZEN_PANEL_BIN={{quote(binary)}}/panel
    for arm in Mdata Mr7; do
      for fold in kadid tid2013 konfig cid22_a25; do
        for seed in 0 1 2; do test -s "$BYC_CELLS/$arm/without_${fold}_s$seed/byc_record.json"; done
      done
    done
    mkdir {{quote(out)}}
    python3 -B scripts/rev4_featpot/byc_cells.py score --arms Mdata,Mr7 --paired-with Mr7 > {{quote(out)}}/e21.json
    for arm in Mdata Mr7; do python3 -B scripts/rev4_featpot/byc_cells.py bytype --arm "$arm" > {{quote(out)}}/bytype_$arm.json; done
    python3 -B scripts/byconstruction/type_bias.py Mdata Mr7 --out {{quote(out)}}/type_bias.json > {{quote(out)}}/type_bias.txt
    python3 -B scripts/rev4_featpot/byc_cells.py knobinv --arms Mdata,Mr7 --out {{quote(out)}}/knobinv.json > {{quote(out)}}/knobinv.txt
    python3 -B scripts/byconstruction/aic_panel.py --out {{quote(out)}}/panel.json --arms base-c,Mdata,Mr7 --geo --no-aic > {{quote(out)}}/panel.txt 2>&1

dataset-e21 root binary cells out seeds='0,1,2':
    env BYC_ROOT={{quote(root)}} REV4_V2_BIN_DIR={{quote(binary)}} BYC_CELLS={{quote(cells)}} python3 -B scripts/rev4_featpot/byc_cells.py score --arms Mdata,Mr7 --paired-with Mr7 --seeds {{quote(seeds)}} > {{quote(out)}}

dataset-mixture binary cells out:
    env BYC_CELLS={{quote(cells)}} MIX={{quote(out)}} bash scripts/byconstruction/ring_mixture.sh {{quote(binary)}} raw 12 Mdata
    python3 -B scripts/byconstruction/mixture_test.py summary --out {{quote(out)}} Mdata Mr7

dataset-loop stage study:
    bash scripts/byconstruction/dataset_loop.sh {{quote(stage)}} {{quote(study)}}

rev7-inventory out:
    python3 -B scripts/rev7/inventory.py --out {{quote(out)}}

rev7-probe-ssim out:
    python3 -B scripts/rev7/probe_ssim_reference.py {{quote(out)}}

rev7-tool-tests:
    TMPDIR="$HOME/tmp" python3 -B -m unittest discover -s scripts/tests -p 'test_rev7_*.py' -v

# Run builds, scoring and verification under the caller's run-heavy scope.
rev7-baseline-test:
    cargo test --locked --release -p zensim --all-features

rev7-oracle-build:
    cargo build --locked --release -p zensim-validate --bin byc_score_timing --bin byc_stub_bake --bin bake_stamp_revision

rev7-oracle-dumps out:
    bash scripts/rev7/oracle_dumps.sh {{quote(out)}}

rev7-bits-compare base cand:
    python3 -B scripts/byconstruction/perf_bits_compare.py {{quote(base)}} {{quote(cand)}}

rev7-bits-tiers root model:
    python3 -B scripts/byconstruction/perf_bits_compare.py --tiers --model {{quote(model)}} {{quote(root)}}

dataset-assessment-keys collection admission out extractor commit:
    OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 nice -n19 ionice -c3 python3 -B scripts/byconstruction/dataset_keys.py {{collection}} --admission {{admission}} --out {{out}} --extractor {{extractor}} --build-commit {{commit}}

# Caller holds shared heavy.lock and run-heavy; archive payload stays on tower.
dataset-unpack-bapps keys library scratch receipts:
    OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 -B scripts/byconstruction/dataset_unpack_bapps.py --keys {{quote(keys)}} --library {{quote(library)}} --scratch {{quote(scratch)}} --receipt-dir {{quote(receipts)}}

# Caller holds shared heavy.lock and run-heavy; all assembly stays with rev5_bank.
dataset-features root binary:
    OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 -B scripts/byconstruction/dataset_features.py --root {{quote(root)}} --bin {{quote(binary)}}

dataset-subset-keys collection admission out content:
    OPENBLAS_NUM_THREADS=1 python3 -B scripts/byconstruction/dataset_keys.py {{quote(collection)}} --admission {{quote(admission)}} --out {{quote(out)}} --extractor /var/tmp/byconstruction-bin/b77883be18db-rev7/extract_features_372col --build-commit b77883be18db5933a9db4b9e2670a2ba88a9729b --content-root {{quote(content)}}

# Lightweight queue; each collection takes/releases the caller's shared lock.
dataset-features-queue root binary lock wrapper:
    OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 -B scripts/byconstruction/dataset_features.py --root {{quote(root)}} --bin {{quote(binary)}} --queue --heavy-lock {{quote(lock)}} --heavy-wrapper {{quote(wrapper)}}

# Caller holds shared heavy.lock and run-heavy; destination is the physical array.
dataset-features-mirror root host destination cache array:
    python3 -B scripts/byconstruction/dataset_mirror.py --root {{quote(root)}} --host {{quote(host)}} --destination {{quote(destination)}} --capacity-path {{quote(cache)}} {{quote(array)}}

dataset-artifacts-mirror root host destination cache array:
    python3 -B scripts/byconstruction/dataset_mirror.py --root {{quote(root)}} --host {{quote(host)}} --destination {{quote(destination)}} --capacity-path {{quote(cache)}} {{quote(array)}} --artifact-kind training

# Caller holds shared heavy.lock and run-heavy. No remote compute is started.
dataset-artifacts-mirror-nfs root destination kind='features' cold='false':
    #!/usr/bin/env bash
    set -euo pipefail
    flags=()
    case {{quote(cold)}} in
      true) flags+=(--cold-readback) ;;
      false) ;;
      *) exit 2 ;;
    esac
    python3 -B scripts/byconstruction/dataset_mirror.py --root {{quote(root)}} --nfs --destination {{quote(destination)}} --artifact-kind {{quote(kind)}} "${flags[@]}"

# Caller holds shared heavy.lock and run-heavy. Preserve a fresh, closed snapshot.
dataset-snapshot study columns mode='mdata':
    #!/usr/bin/env bash
    set -euo pipefail
    study={{quote(study)}}
    out="$study/archive-ready"
    test ! -e "$out"
    case {{quote(mode)}} in
      mdata) arms="Mdata Mr7" ;;
      mixes) arms="Mdata Mw025 Mw100 Mmos Mpair Mlic"; test -s "$study/FINALISTS.json"; test -s "$study/TRAIN_SELECTION.json"; test -s "$study/eval/report.json"; test -s "$study/eval-bapps/report.json"; test -s "$study/FORCED_CHOICE_CONTROLS.json"; test -s "$study/MIXTURE_CELL_CONTROLS.json"; test -s "$study/BINARY_CELL_CONTROLS.json" ;;
      *) exit 2 ;;
    esac
    for arm in $arms; do
      for fold in kadid tid2013 konfig cid22_a25; do
        for seed in 0 1 2; do
          test -s "$study/cells/$arm/without_${fold}_s$seed/byc_record.json"
          for mix in mixture mixture_global; do test -s "$study/$mix/scores_${arm}_${fold}_s$seed.json"; done
        done
      done
    done
    test -s "$study/scorecard/panel.json"
    if [ {{quote(mode)}} = mdata ]; then test -s "$study/gates-Mdata/COPY_RECEIPT.json"; fi
    test -s "$study/NATIVE_CELL_CONTROLS.json"
    test -s "$study/NUMERIC_SUMMARY.json"
    mkdir "$out"
    rsync -aL --exclude=/archive-ready/ --exclude=/loop-target/ --exclude=img/ --exclude=/snapshot-mirror.log "$study/" "$out/"
    mkdir "$out/input_configuration"
    cp {{quote(columns)}} "$out/input_configuration/cols_c.txt"
    cp scripts/rev4_featpot/e33_fx1_declaration.json "$out/input_configuration/"
    jj log -r @- --no-graph -T 'commit_id ++ "\n"' > "$out/SOURCE_COMMIT.txt"
    test -z "$(find "$out" -type l -print -quit)"
    if [ {{quote(mode)}} = mdata ]; then sha256sum "$out/rev7data/full/refit/production-f16.bin" "$out/rev7data/root/wide/frozen.json"; else sha256sum "$out/FINALISTS.json" "$out/TRAIN_SELECTION.json"; fi

# Caller holds shared heavy.lock and run-heavy; a final receipt requires no gaps.
dataset-features-verify root:
    OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 -B scripts/byconstruction/dataset_verify.py --root {{quote(root)}} --write

# Caller holds shared heavy.lock and run-heavy; records bindings before mirroring.
dataset-bapps-verify root:
    OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 -B scripts/byconstruction/dataset_bapps_cleanup.py --root {{quote(root)}}

# Caller holds shared heavy.lock and run-heavy; mirror must pass before removal.
dataset-bapps-cleanup root receipt:
    OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 -B scripts/byconstruction/dataset_bapps_cleanup.py --root {{quote(root)}} --remove-scratch --removal-receipt {{quote(receipt)}}

addrfix-test target:
    TMPDIR="$HOME/tmp" cargo test --locked -p zensim-validate --bin panel panel_e34:: --target-dir {{quote(target)}}

addrfix-rank-resume:
    python3 scripts/rev4_featpot/addrfix.py rank-score-resume

addrfix-rank-score:
    python3 scripts/rev4_featpot/addrfix.py rank-score

# zensim dev commands

# Fixed-work counter probe includes the actual Adam owner. Caller supplies resource limits.
adam-denormal-build:
    cargo build --release --locked --manifest-path tools/adam-denormal-probe/Cargo.toml

adam-denormal-tests:
    cargo test --release --locked --manifest-path tools/adam-denormal-probe/Cargo.toml

adam-denormal-clippy:
    cargo clippy --release --locked --manifest-path tools/adam-denormal-probe/Cargo.toml --all-targets -- -D warnings

adam-denormal-probe kernel="fused" state="decay" steps="2000" tier="v3" pages="lazy":
    tools/adam-denormal-probe/target/release/adam-denormal-probe {{quote(kernel)}} {{quote(state)}} {{steps}} {{quote(tier)}} {{quote(pages)}}

adam-denormal-cell argv trainer output events:
    python3 scripts/byconstruction/adam_denormal_cell.py --argv-json {{quote(argv)}} --trainer {{quote(trainer)}} --out {{quote(output)}} --events {{quote(events)}}

adam-denormal-trainer-tests:
    cargo test --locked -p zensim-validate --lib adam_simd
    cargo test --locked -p zensim-validate --lib l2_row_form
    cargo test --locked -p zensim-validate --test adam_simd_equivalence --test adam_simd_rsqrt_precision

adam-denormal-counters pid output events seconds="10":
    perf stat -o {{quote(output)}} -e {{quote(events)}} -p {{pid}} -- sleep {{seconds}}

# Caller pins the CPU and supplies run-heavy; output paths must be fresh.
adam-denormal-counter-probe binary output events kernel state steps tier pages="":
    #!/usr/bin/env bash
    set -euo pipefail
    test ! -e {{quote(output)}}
    probe_argv=({{quote(binary)}} {{quote(kernel)}} {{quote(state)}} {{quote(steps)}} {{quote(tier)}})
    if [[ -n {{quote(pages)}} ]]; then probe_argv+=({{quote(pages)}}); fi
    perf stat -o {{quote(output)}} -e {{quote(events)}} -- "${probe_argv[@]}"

# Caller pins the core and supplies run-heavy; preserve both executable versions.
adam-denormal-final-probes before selected prefix:
    #!/usr/bin/env bash
    set -euo pipefail
    before={{quote(before)}}
    selected={{quote(selected)}}
    prefix={{quote(prefix)}}
    for state in normal decay; do
      for phase in before selected; do
        binary="$before"
        if [[ "$phase" == selected ]]; then binary="$selected"; fi
        just adam-denormal-counter-probe "$binary" "$prefix-$phase-$state.stat" cpu_core/cycles/,cpu_core/instructions/,cpu_core/assists.fp/ fused "$state" 2000 v3
      done
    done
    for state in tiny-l2 tiny-moment subnormal-weight; do
      just adam-denormal-counter-probe "$selected" "$prefix-selected-$state.stat" cpu_core/cycles/,cpu_core/instructions/,cpu_core/assists.fp/ fused "$state" 2000 v3
    done

adam-denormal-compare reference candidate tools root output:
    python3 scripts/byconstruction/adam_denormal_cell.py compare --reference {{quote(reference)}} --candidate {{quote(candidate)}} --tools {{quote(tools)}} --root {{quote(root)}} --out {{quote(output)}}

adam-denormal-compare-splice reference candidate tools root output probe:
    python3 scripts/byconstruction/adam_denormal_cell.py compare --reference {{quote(reference)}} --candidate {{quote(candidate)}} --tools {{quote(tools)}} --root {{quote(root)}} --out {{quote(output)}} --probe {{quote(probe)}}

adam-denormal-status output:
    python3 scripts/byconstruction/adam_denormal_cell.py status --out {{quote(output)}}

adam-denormal-docker plan phase output:
    python3 scripts/byconstruction/adam_denormal_cell.py docker --plan {{quote(plan)}} --phase {{quote(phase)}} --output {{quote(output)}}

# Keep the entire matched pair quiet under one shared heavy-lock/resource scope.
adam-denormal-quiet-pair plan tests before after:
    env TMPDIR="$HOME/tmp" flock "$HOME/tmp/heavy.lock" "${ADAM_RUN_HEAVY:-$HOME/work/zen/scripts/run-heavy}" --mem 12G --jobs 1 -- just adam-denormal-pair-held {{quote(plan)}} {{quote(tests)}} {{quote(before)}} {{quote(after)}}

# Freeze auxiliary manifests too: the trainer resolves them again at emission.
adam-denormal-quiet-frozen-pair source root plan tests before after:
    env TMPDIR="$HOME/tmp" flock "$HOME/tmp/heavy.lock" "${ADAM_RUN_HEAVY:-$HOME/work/zen/scripts/run-heavy}" --mem 12G --jobs 1 -- just adam-denormal-freeze-pair-held {{quote(source)}} {{quote(root)}} {{quote(plan)}} {{quote(tests)}} {{quote(before)}} {{quote(after)}}

adam-denormal-freeze-pair-held source root plan tests before after:
    #!/usr/bin/env bash
    set -euo pipefail
    test ! -e {{quote(root)}}
    mkdir -p {{quote(root)}}
    rsync -rL --checksum {{quote(source)}}/ {{quote(root)}}/
    just adam-denormal-pair-held {{quote(plan)}} {{quote(tests)}} {{quote(before)}} {{quote(after)}}

# Reuse a completed quiet baseline when testing a new preserved successor.
adam-denormal-quiet-cell plan phase output:
    env TMPDIR="$HOME/tmp" flock "$HOME/tmp/heavy.lock" "${ADAM_RUN_HEAVY:-$HOME/work/zen/scripts/run-heavy}" --mem 12G --jobs 1 -- just adam-denormal-cell-held {{quote(plan)}} {{quote(phase)}} {{quote(output)}}

adam-denormal-cell-held plan phase output:
    #!/usr/bin/env bash
    set -euo pipefail
    set -o noclobber
    base=$(dirname {{quote(plan)}})
    phase={{quote(phase)}}
    output={{quote(output)}}
    { date -u; lscpu; free -h; mpstat -P ALL 1 2; } > "$base/results/$phase-$output-preflight.log"
    just adam-denormal-docker {{quote(plan)}} {{quote(phase)}} {{quote(output)}}

adam-denormal-pair-held plan tests before after:
    #!/usr/bin/env bash
    set -euo pipefail
    set -o noclobber
    before={{quote(before)}}
    base=$(dirname {{quote(plan)}})
    if pgrep -x zensim_mlp_trai > "$base/results/$before-busy-trainers.txt"; then
      echo 'Existing trainers are active; a quiet timing pair cannot start.' >&2
      exit 1
    fi
    { date -u; lscpu; free -h; mpstat -P ALL 1 2; } > "$base/results/$before-preflight.log"
    env -u ZENSIM_MAX_TIER {{quote(tests)}} --test-threads=1 > "$base/results/$before-native-tests.log" 2>&1
    just adam-denormal-docker {{quote(plan)}} before {{quote(before)}}
    just adam-denormal-docker {{quote(plan)}} after {{quote(after)}}

adam-denormal-repro model output:
    tools/adam-denormal-probe/target/release/adam-denormal-probe --repro {{quote(model)}} {{quote(output)}}

adam-denormal-inspect model output:
    tools/adam-denormal-probe/target/release/adam-denormal-probe --inspect {{quote(model)}} {{quote(output)}}

# Run after all writers finish; caller supplies run-heavy for large archives.
adam-denormal-manifest root:
    #!/usr/bin/env bash
    set -euo pipefail
    python3 - {{quote(root)}} <<'PY'
    import hashlib, sys
    from pathlib import Path
    root = Path(sys.argv[1]).resolve(strict=True)
    files = sorted(p for p in root.rglob('*') if p.is_file())
    if any(p.is_symlink() or '\n' in str(p) or '\\' in str(p) for p in files):
        raise ValueError('Archive requires plain files and portable checksum paths')
    with (root / 'SHA256SUMS').open('x') as out:
        for p in files:
            with p.open('rb') as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            out.write(f'{digest}  {p.relative_to(root).as_posix()}\n')
    PY
    cd {{quote(root)}}
    sha256sum -c SHA256SUMS

adam-denormal-mirror root destination:
    #!/usr/bin/env bash
    set -euo pipefail
    mkdir -p {{quote(destination)}}
    rsync -r --checksum {{quote(root)}}/ {{quote(destination)}}/
    cd {{quote(destination)}}
    sha256sum -c SHA256SUMS

addrfix-cost-report binary:
    python3 scripts/rev4_featpot/addrfix.py cost-report --binary {{quote(binary)}}

addrfix-cost-summary-report:
    python3 scripts/rev4_featpot/addrfix.py cost-summary-report

addrfix-cost-payload:
    python3 scripts/rev4_featpot/addrfix.py cost-payload

addrfix-cost mode root model:
    python3 scripts/rev4_featpot/addrfix_cost.py {{quote(mode)}} --root {{quote(root)}} --model {{quote(model)}}

# Caller supplies the shared heavy.lock and run-heavy scope.
addrfix-owner mode binary trace="":
    python3 scripts/rev4_featpot/addrfix.py {{quote(mode)}} --binary {{quote(binary)}} {{trace}}

addrfix-rank-pairs:
    python3 scripts/rev4_featpot/addrfix.py rank-pairs

addrfix-rank-report:
    python3 scripts/rev4_featpot/addrfix.py rank-report

addrfix-clippy target:
    TMPDIR="$HOME/tmp" cargo clippy --locked -p zensim-validate --bin bake_verdict --bin panel --target-dir {{quote(target)}} -- -D warnings

addrfix-build target:
    TMPDIR="$HOME/tmp" cargo build --release --locked -p zensim-validate --bin bake_verdict --bin panel --target-dir {{quote(target)}}

addrfix-bench-lock:
    TMPDIR="$HOME/tmp" cargo generate-lockfile --manifest-path zensim-bench/addrfix-runtime/Cargo.toml

addrfix-bench-build target:
    TMPDIR="$HOME/tmp" cargo build --release --locked --manifest-path zensim-bench/addrfix-runtime/Cargo.toml --bench ssim2_speed_bar --target-dir {{quote(target)}} --message-format=json

addrfix-bench-clippy target:
    TMPDIR="$HOME/tmp" cargo clippy --release --locked --manifest-path zensim-bench/addrfix-runtime/Cargo.toml --bench ssim2_speed_bar --target-dir {{quote(target)}} -- -D warnings

# Caller supplies shared heavy.lock and run-heavy for builds and scoring.
peergates-build target:
    cargo build --release --locked --manifest-path zensim-bench/peergates/Cargo.toml --target-dir {{quote(target)}}

peergates-clippy target:
    cargo clippy --release --locked --manifest-path zensim-bench/peergates/Cargo.toml --target-dir {{quote(target)}} -- -D warnings

peergates-fmt:
    cargo fmt --manifest-path zensim-bench/peergates/Cargo.toml --check

peergates-inputs:
    python3 scripts/rev4_featpot/peergates.py inputs

peergates-quiet:
    python3 scripts/rev4_featpot/peergates.py quiet

peergates-run binary mode input output:
    RAYON_NUM_THREADS=1 ZENSIM_FORMULA_REV=5 {{quote(binary)}} {{quote(mode)}} {{quote(input)}} {{quote(output)}}

peergates-report:
    python3 scripts/rev4_featpot/peergates.py steering-report

peergates-failures:
    python3 scripts/rev4_featpot/peergates.py failure-report

peergates-archive:
    python3 scripts/rev4_featpot/peergates.py archive

peergates-validate:
    python3 scripts/rev4_featpot/peergates.py validate

peergates-peer-tables grid:
    python3 scripts/rev4_featpot/peergates.py peer-tables --grid {{quote(grid)}}

peergates-landed:
    python3 scripts/rev4_featpot/peergates.py landed

peergates-watch:
    python3 scripts/rev4_featpot/peergates.py watch

peergates-repair-transport:
    python3 scripts/rev4_featpot/peergates.py repair-transport

peergates-cvvdp mode grid="standard":
    python3 scripts/rev4_featpot/peergates_cvvdp.py {{quote(mode)}} --grid {{quote(grid)}}

# Run under the caller's shared heavy.lock and run-heavy wrapper.
peergates-cvvdp-score grid:
    env RAYON_NUM_THREADS=1 /var/tmp/paper-cvvdp/bin/zenmetrics batch --metric cvvdp --display-model standard_fhd --bench-on-start no --use-legacy-scheduler --pairs /mnt/v/output/zensim/peergates-2026-10-16/cvvdp/{{grid}}-pairs.tsv --output /mnt/v/output/zensim/peergates-2026-10-16/cvvdp/{{grid}}-scores.tsv --jobs 4 --group-by-ref

# Existing trainer Parquet loaders on the pinned 699,999-row Float32 table.
# Run the study recipe under run-heavy; its output directory must be new.
trainer-parquet-load-build:
    cargo build --release -p zensim-validate --bin parquet_load_bench

trainer-parquet-load-study output:
    bash benchmarks/trainer_parquet_load_2026-10-13/run.sh {{quote(output)}}

# Dataset-independent evaluation CLI; tests use synthetic fixtures only.
metric-evaluate-check:
    cargo clippy -p zensim-validate --bin panel --bin bake_verdict -- -D warnings

metric-evaluate-tests:
    cargo test -p zensim-validate --bin panel --test metric_evaluate

# Parallel vs sequential training-CSV loader parity. The generated fixtures run in
# every `cargo test`; this recipe adds the real TID CSV (`ZENSIM_TID_IWSSIM_CSV`),
# which fails if the named file is missing.
csv-loader-parity-real path="/mnt/v/zen/zensim-training/2026-05-16/v2/tid_features_iwssim_log_372col.csv":
    ZENSIM_TID_IWSSIM_CSV="{{path}}" cargo test -p zensim-validate --bin zensim_mlp_train csv_load_equivalence_tests -- --nocapture

metric-evaluate-fmt:
    cargo fmt -p zensim-validate

# Explicit manifests only; put large runs under run-heavy.
metric-evaluate manifest outdir:
    target/debug/panel evaluate --manifest {{quote(manifest)}} --output {{quote(outdir)}}

metric-dataset-tests:
    python3 -m unittest discover -s scripts/tests -p test_inspect_metric_dataset.py

metric-aic-suite root manifest:
    python3 scripts/prepare_aic2026_evaluation.py {{quote(root)}} --output {{quote(manifest)}}

# Synthetic D2 read only. Caller supplies a pinned canonical Rust panel.
releasegate-tests:
    TMPDIR=$HOME/tmp python3 -m unittest discover -s scripts/tests -p 'test_kadid_terminal*.py' -v

# Repeat the bound-payload probes against this tree or a source-only old-tip export.
releasegate-bound-payload-tests source=".":
    cd "{{source}}" && TMPDIR=$HOME/tmp python3 -m unittest discover -s scripts/tests -p 'test_kadid_terminal_bound_payloads.py' -v

# Canonical signed-quality CLI and legacy panel mode regressions.
releasegate-panel-tests:
    cargo test -p zensim-validate --bin panel -- --nocapture

releasegate-panel-parity panel_bin:
    python3 scripts/verify_panel_parity.py --bin "{{panel_bin}}"

# The rustdoc-JSON nightly is PINNED (keep in sync with the `api-doc-check`
# job in .github/workflows/ci.yml): an unpinned tracking nightly churns
# cross-crate path rendering with zero repo changes — MEASURED 2026-09-06,
# regenerating zensim-regress.txt against a newer nightly than the one that
# produced the committed snapshot rewrote 11 lines of `std::io::error::Error`
# to `core::io::error::Error` (the core::io re-homing), which is exactly the
# false-diff class this pin exists to prevent. Bump the pin deliberately, in
# the same commit as a `just api-doc` regen.
apidoc_toolchain := "nightly-2026-09-02"

# Caller-selected qualification scope: no evaluation/human-label payloads.
# Keep real-corpus tests intact; these five are outside this lane's scope.
prodqual-workspace-build:
    cargo test --workspace --all-targets --all-features --exclude zensim-wasm-tests --no-run

prodqual-fmt-check:
    cargo fmt -p zensim --check

# The archive mount accepts file contents but not local ownership changes.
prodqual-mirror evidence destination:
    rsync -a --no-owner --no-group '{{evidence}}/' '{{destination}}/'

prodqual-workspace-tests:
    cargo test --workspace --lib --bins --tests --examples --all-features --exclude zensim-wasm-tests --no-fail-fast -- \
        --skip cid22_aggregate_srocc_matches_audit_reference \
        --skip cid22_first_row_matches_bake_verdict_reference \
        --skip parallel_matches_sequential_iwssim_log_target \
        --skip parallel_matches_sequential_default_target_with_scale \
        --skip canonical_dial_grid_is_the_quarantined_v2_grid

prodqual-rev5:
    cargo test -p zensim --release --all-features --test featcanon_rev5_parity -- --nocapture

[positional-arguments]
prodqual-synthetic *models:
    ZENSIM_FORMULA_REV=5 cargo run -p zensim --release --all-features --example serve_custom_bake -- --prodqual "$@"

prodqual-serving-matrix outdir:
    scripts/serving_matrix.sh {{outdir}}

prodqual-feature-matrix:
    python3 scripts/prodqual_feature_matrix.py

# Match the WASI toolchain pin in CI (the stable LLVM workaround).
prodqual-wasm-build:
    RUSTFLAGS='-C target-feature=+simd128' cargo +1.98.1 build -p zensim --target wasm32-wasip1 --release --all-features --example serve_custom_bake

[positional-arguments]
prodqual-wasm-synthetic program *models:
    #!/usr/bin/env bash
    set -euo pipefail
    program="$1"
    shift
    wasmtime run --dir "${PRODQUAL_INPUT_ROOT:?set the pinned artifact directory}::/inputs" \
        --env ZENSIM_FORMULA_REV=5 "$program" --prodqual "$@"

prodqual-training-only:
    cargo clippy -p zensim --no-default-features --features training --lib -- -D warnings
    cargo test -p zensim --no-default-features --features training --lib -- --nocapture

prodqual-workspace-failures:
    cargo test -p zensim -p zensim-validate --all-features --no-fail-fast \
        --test featcanon_rev4_contract --test research_engine_parity \
        --test bake_surface --test feature_set_match

# Format + regenerate the public-API surface snapshots (docs/public-api/).
# The snapshot runner lives in the workspace-excluded apidoc/ package, so it
# is never built or run by plain `cargo test`, nor by any OTHER CI job — only
# the dedicated `api-doc-check` CI job (which sets ZEN_API_DOC=check) runs it.
fmt:
    cargo fmt --all
    ZEN_API_DOC_TOOLCHAIN={{apidoc_toolchain}} cargo test --manifest-path apidoc/Cargo.toml

# Regenerate the public-API surface snapshots only
api-doc:
    ZEN_API_DOC_TOOLCHAIN={{apidoc_toolchain}} cargo test --manifest-path apidoc/Cargo.toml

# Verify the committed snapshots are current (what CI's api-doc-check job runs)
api-doc-check:
    ZEN_API_DOC=check ZEN_API_DOC_TOOLCHAIN={{apidoc_toolchain}} cargo test --manifest-path apidoc/Cargo.toml

# Restored-cut families (COST_CUTS_AUDIT): clean-snapshot extractor build. Sibling repos are
# `git archive`s of their fetched mains under /var/tmp/restore-cuts/src (CODEX_NOTE crates-on-main rule);
# every output stays under /var/tmp/restore-cuts/.
restore-cuts-build:
    /home/lilith/tmp/devin/heavy --mem 16G --jobs 8 -- bash scripts/restore_cuts/build.sh

# The restored-cut parity test at the bank's revision (Rev3/sqrt); a plain `cargo test` runs it at the shipped Rev1.
restore-cuts-parity-rev3:
    ZENSIM_FORMULA_REV=3 ZENSIM_ROOT_FORM=sqrt cargo test -p zensim --features training --test restore_cuts_parity -- --nocapture

# Independent NumPy mirror of mapdev/z1max/gmsnative from XYB plane dumps (needs the instrument build).
#   RESTORE_CUTS_DUMP_DIR=/var/tmp/restore-cuts/mirror_dump just restore-cuts-mirror
restore-cuts-mirror:
    #!/usr/bin/env bash
    set -euo pipefail
    : "${RESTORE_CUTS_DUMP_DIR:?set RESTORE_CUTS_DUMP_DIR to a NEW directory}"
    export CARGO_TARGET_DIR=/var/tmp/restore-cuts/target-instr ZENSIM_FORMULA_REV=3 ZENSIM_ROOT_FORM=sqrt
    RUSTFLAGS='--cfg restore_cuts_instrument' /home/lilith/tmp/devin/heavy --mem 16G --jobs 8 -- \
        cargo test -p zensim --features training --lib -- restore_cuts_plane_dump
    python3 scripts/restore_cuts/numpy_mirror.py "$RESTORE_CUTS_DUMP_DIR"
    python3 scripts/restore_cuts/numpy_mirror.py "$RESTORE_CUTS_DUMP_DIR" --wrong-control

# Explicit caller-owned TRAIN corpus gate. The caller selects the four
# canonical, SHA-checked references; there is no silent skip when absent.
# All generated bytes and cargo targets go under /var/tmp/gmsbank/.
#   GMSBANK_CORPUS_REFS=/var/tmp/gmsbank/corpus_probe/refs.tsv just gmsbank-corpus
[doc("Run GMSBANK's four-reference TRAIN blur/noise/zenjpeg response gate")]
gmsbank-corpus:
    #!/usr/bin/env bash
    set -euo pipefail
    : "${GMSBANK_CORPUS_REFS:?set GMSBANK_CORPUS_REFS to the SHA-checked four-reference TRAIN TSV}"
    export CARGO_TARGET_DIR=/var/tmp/gmsbank/target
    export ZENSIM_FORMULA_REV=3
    export ZENSIM_ROOT_FORM=sqrt
    root=/var/tmp/gmsbank/corpus_probe
    mkdir -p "$root"
    /home/lilith/tmp/devin/heavy --mem 16G --jobs 8 -- cargo build --release --manifest-path zensim-bench/Cargo.toml --example gmsbank_corpus_fixtures --features m3-fixtures,zen-decode
    /home/lilith/tmp/devin/heavy --mem 16G --jobs 8 -- "$CARGO_TARGET_DIR/release/examples/gmsbank_corpus_fixtures" "$GMSBANK_CORPUS_REFS" "$root"
    /home/lilith/tmp/devin/heavy --mem 16G --jobs 8 -- cargo build --release --manifest-path zensim-bench/Cargo.toml --example extract_features_372col --features training,zen-decode
    /home/lilith/tmp/devin/heavy --mem 16G --jobs 8 -- "$CARGO_TARGET_DIR/release/examples/extract_features_372col" --corpus pairs-tsv --path "$root/pairs.tsv" --out "$root/features.csv" --full-gmsbank --input-contract legacy-rgb8
    python3 scripts/gmsbank/corpus_response.py "$root/features.csv"

# Explicit opt-in for mounted Rev4 TRAIN corpora. The two ignored tests fail
# on absent inputs or unexpected unsupported formats. The unignored 16-pair
# synthetic parity gate runs in the ordinary all-features CI test job.
# Run this recipe through the local heavy lock when building on the lab host.
[positional-arguments]
rev4-corpus-tests root kadid_inputs expected_unsupported_safesyn:
    #!/usr/bin/env bash
    set -euo pipefail
    export ZENSIM_REV4_CORPUS_ROOT="$1"
    export ZENSIM_REV4_KADID_INPUTS="$2"
    export ZENSIM_REV4_EXPECT_UNSUPPORTED_SAFESYN="$3"
    cargo test -p zensim --release --all-features --test rev4_featbank_parity rev4_synthetic_16_pair_identity -- --nocapture
    cargo test -p zensim --release --all-features --lib rev4_gridblk_zenjpeg_ladder -- --ignored --nocapture
    cargo test -p zensim --release --all-features --test rev4_featbank_parity rev4_corpus_toggle_identity -- --ignored --nocapture

# REV4SERVE's corpus-gated real-bake check: the v2+basic featpot bake
# (default /var/tmp/rev4-featpot/v2c/cells/set:v2+basic@h32:H128__N/
# without_aic3_s0/refit/last.bin; REV4SERVE_BAKE overrides the path)
# scored from pixels == research::extract features, bit-for-bit, plus
# prepared steering — on >= 20 held-out aic3 original/decoded pairs
# under /mnt/v/dataset/aic3_ctc_epfl. Opt-in only; absent corpus or bake
# fails loudly.
rev4serve-gate:
    cargo test -p zensim --release \
        --features custom-profiles,feature-regime-v2,training \
        --test rev4serve_gate rev4_featpot_bake_served_and_steered \
        -- --ignored --nocapture

# CI-exact clippy
clippy:
    cargo clippy --workspace --all-targets --all-features --exclude zensim-wasm-tests -- -D warnings

# MAINFIX reports stale test contracts without changing their expectations.
mainfix-baseline:
    cargo test -p zensim -p zensim-validate --all-features --no-fail-fast \
        --test featcanon_rev4_contract --test research_engine_parity --test bake_surface

mainfix-contracts:
    cargo test -p zensim --all-features --lib rev5_bake_serves_supported_reads_and_refuses_unsupported -- --nocapture
    cargo test -p zensim --all-features --test palette_research -- --nocapture

[positional-arguments]
mainfix-revision-probe program fixture outdir *models:
    #!/usr/bin/env bash
    set -euo pipefail
    python3 scripts/mainfix_revision_pin_probe.py --program "$1" --fixture "$2" --output-dir "$3" "${@:4}"

mainfix-mirror evidence destination:
    rsync -a --no-owner --no-group '{{evidence}}/' '{{destination}}/'

# Quick offline rank/dial report (not full-eval or product qualification).
# Emits markdown plus a self-contained HTML report. Optional REF
# bake enables the per-zone dial-agreement panel; RAMP grid enables the
# severity-ramp monotonicity section (point it at a regime-matched parquet).
#   just metric-eval zensim/weights/b_sdr_linear_cid80_inclwinsor_dense_dial_2026-07-07.bin
#   just metric-eval <bake> <ref-bake> <ramp-grid.parquet>
[doc("Quick offline rank/dial report; not full-eval or product qualification")]
[positional-arguments]
metric-eval bake ref="" ramp="" out="/mnt/v/output/zensim/reports":
    #!/usr/bin/env bash
    set -euo pipefail
    bake=$1
    ref=$2
    ramp=$3
    out=$4
    cargo build --release -p zensim-validate --bin bake_verdict
    mkdir -p "$out"
    stem=$(basename "$bake" .bin)
    args=(--bake "$bake" --output "$out/$stem.md" --html "$out/$stem.html")
    if [[ -n "$ref" ]]; then args+=(--compare "$ref"); fi
    if [[ -n "$ramp" ]]; then args+=(--ramp-grid "$ramp"); fi
    "${CARGO_TARGET_DIR:-target}/release/bake_verdict" "${args[@]}"
    echo "report: $out/$stem.html"

# Build the viewable spatial-diffmap demo gallery: six imazen-26 TRAIN sources
# x a zenjpeg q20/q50/q80 ladder, each rendered as a zensim diffmap heatmap and
# overlay, laid out as one self-contained page under /mnt/v/output (served at
# localhost:3300). A DEMO, not evidence -- one pass, no replication, no ranking
# claim. Extra args pass through (--max-dim / --profile / --scale-max / --out).
#   just demo-diffmap
#   just demo-diffmap --profile d --max-dim 1200
[doc("Build the spatial-diffmap heatmap demo gallery (demo, not evidence)")]
[positional-arguments]
demo-diffmap *options:
    python3 scripts/demos/diffmap_gallery.py "$@"

# Join the cross-generation speed matrix to the board's full-evaluation rows
# and lay the result out as one self-contained speed-vs-accuracy page under
# /mnt/v/output (served at localhost:3300). Runs NOTHING -- no benchmark, no
# scorer, no cargo -- every number is read out of files that already exist,
# and an arm with no provable accuracy row is drawn on the speed axis alone
# rather than given a lookalike row's value. Also refreshes the committed
# joined table in benchmarks/. `--raster` additionally writes PNGs of the
# charts (needs `resvg` on PATH) so the result can be looked at.
#   just demo-speed-accuracy
#   just demo-speed-accuracy --raster --out ~/tmp/speed-accuracy
[doc("Speed-vs-accuracy page from the speed matrix + board rows (demo, joins existing data)")]
[positional-arguments]
demo-speed-accuracy *options:
    python3 scripts/demos/speed_accuracy_page.py "$@"

# Fail on scripts that cannot run: pinned to a deleted sibling worktree, or
# hardcoding a binary with no source anywhere. On 2026-07-15 an audit found 25
# of 130 scripts in scripts/v_next/ pointing into worktrees that had been
# cleaned up weeks earlier, plus one that had not PARSED since a bulk sed.
# Nobody noticed because nobody ran them. This is the check that notices.
lint-scripts:
    python3 scripts/lint_scripts.py

# Display the canonical detached-compute waiter's progress and terminal record.
[positional-arguments]
await-status heartbeat:
    #!/usr/bin/env bash
    set -euo pipefail
    cat "$1.status"
    if [[ -f "$1.done" ]]; then cat "$1.done"; fi

# Paired arithmetic-revision cost A/B (issue #61). Builds the interleaved
# extraction instrument ONCE and runs it in alternating revision-1/revision-3
# blocks on one pinned core, with `fast_ssim2` as the cross-block drift anchor.
# Do NOT wrap in run-heavy — its cgroup makes zenbench's gating refuse the run.
#   just rev3-cost ~/tmp/rev3-cost 2 1024,2048 8
[positional-arguments]
rev3-cost out blocks="2" sizes="1024,2048" cpu="8":
    cargo bench --no-run --locked --bench extract_paths_bench -p zensim \
        --features custom-profiles,feature-regime-v2,threads,training
    ./scripts/bench/rev3_cost_ab.sh "$1" "$2" "$3" "$4"
    python3 scripts/bench/rev3_cost_report.py "$1"

# Peak-RSS half of the revision cost question: `/usr/bin/time -v` max RSS per
# arm per revision, one arm per process so the reading is attributable.
#   just rev3-rss ~/tmp/rev3-rss
# Two BUILDS at the default revision, alternating blocks on one pinned core:
# "what did the default path pay between commit X and commit Y". Copy both
# bench binaries to equal-length paths first (argv[0] length shifts layout).
#   just perf-ab-binaries ~/tmp/ab-run ~/tmp/ab/bin_A ~/tmp/ab/bin_B
[positional-arguments]
perf-ab-binaries out bin_a bin_b blocks="2" sizes="1024,2048" cpu="8":
    BIN_A="$2" BIN_B="$3" ./scripts/bench/rev3_cost_ab.sh "$1" "$4" "$5" "$6"
    python3 scripts/bench/rev3_cost_report.py "$1"

[positional-arguments]
rev3-rss out sizes="1024 2048" arms="buf_v1_372 fold372_full fold944_full" cpu="8":
    ./scripts/bench/rev3_rss.sh "$1" "$2" "$3" "$4"

# Replay the registered spatial coherence cells at revision 1 and revision 3.
# ARMS THE CROSS-REVISION DIAGNOSTIC BYPASS: the bakes were fit at revision 1,
# so this measures what the extraction change does to a FIXED model. Not a
# qualification, not model quality. See the script header.
[positional-arguments]
rev3-spatial out:
    ./scripts/bench/rev3_spatial_replay.sh "$1"

# The cross-generation speed matrix: every named zensim profile, the two
# frozen Rev3 ensembles, and the peer metrics (fast-ssim2, butteraugli,
# ssimulacra2/rust-av) interleaved across 64/256/1024/2048/4096 squares at 1
# and 16 threads. Builds BOTH fast-ssim2 feature configurations, because its
# rayon parallelism is a cargo feature and an MT row measured against a
# single-threaded opponent is not an MT comparison.
# The BUILDS go through run-heavy; the RUNS deliberately do not (its cgroup
# makes zenbench's own gating refuse the run — see rev3-cost above).
#   just bench-speed-matrix
[doc("Cross-generation + peer-metric speed matrix (writes raw rounds, then the summary)")]
[positional-arguments]
bench-speed-matrix raw="/mnt/v/output/zensim/demos/speed-matrix-2026-09-18/raw" stem="benchmarks/speed_matrix_2026-09-18":
    #!/usr/bin/env bash
    set -euo pipefail
    raw=$1
    stem=$2
    # cargo prints one JSON message per line; run-heavy's own chatter is not
    # JSON, so the `^{` filter keeps the two apart without swallowing errors.
    build() {
        ~/work/zen/scripts/run-heavy --mem 16G --jobs 8 -- \
            cargo bench --no-run --message-format=json-render-diagnostics \
            --manifest-path zensim-bench/Cargo.toml --bench ssim2_speed_bar "$@" \
        | grep '^{' \
        | python3 -c "import json,sys; print(next(e for e in (json.loads(l).get('executable') for l in sys.stdin) if e and 'ssim2_speed_bar' in e))"
    }
    if [[ "${SPEEDQ:-0}" == 1 ]]; then
        export ZENSIM_BENCH_SKIP_CPP_FFI=1
        mkdir -p "$raw/provenance"
        stamp=$(date -u +%Y%m%dT%H%M%SZ)
        build_log="$raw/provenance/build-$stamp.jsonl"
        ~/work/zen/scripts/run-heavy --mem 16G --jobs 8 -- \
            cargo bench --no-run --message-format=json-render-diagnostics \
            --manifest-path zensim-bench/Cargo.toml --bench ssim2_speed_bar \
            --features speedq,ssim2-rayon >"$build_log" 2>&1
        BIN_SPEEDQ=$(python3 scripts/demos/speedq_run.py freeze \
            --build-log "$build_log" --dest "$raw/provenance/instrument-$stamp")
        cp zensim-bench/Cargo.lock "$raw/provenance/Cargo-$stamp.lock"
        export BIN_SPEEDQ
        rc=0
        scripts/demos/speed_matrix_run.sh "$raw" || rc=$?
        python3 scripts/demos/speed_matrix_report.py --speedq --raw-dir "$raw" \
            --out-json "$stem.json" --out-md "$stem.md"
        exit "$rc"
    fi
    BIN_PLAIN=$(build) BIN_RAYON=$(build --features ssim2-rayon) \
        scripts/demos/speed_matrix_run.sh "$raw"
    python3 scripts/demos/speed_matrix_report.py --raw-dir "$raw" \
        --out-json "$stem.json" --out-md "$stem.md" --notes "$stem.notes.md"

# Report only, never fails — for a quick survey.
lint-scripts-list:
    python3 scripts/lint_scripts.py --list

# Target orientation vs raw human labels, every registered eval root.
check-orientation:
    python3 scripts/canonical_corpus/check_target_orientation.py --all-roots

# Which mix legs an external orientation check can reach.
check-provenance:
    python3 scripts/canonical_corpus/check_target_orientation.py --provenance

# Audit THIS recipe's data. Pass owner options explicitly (root remaps, twins,
# leakage roots); no hardcoded August campaign is silently substituted.
# Example: just check-mix path/to/bake.bin.spec.json --leak-eval-root /data/eval
[doc("Audit the supplied recipe's data; pass root/remap/leakage options explicitly")]
[positional-arguments]
check-mix spec *options:
    #!/usr/bin/env bash
    set -euo pipefail
    spec=$1
    shift
    exec python3 scripts/canonical_corpus/check_table_integrity.py --mix-from-spec "$spec" "$@"

# Orientation/provenance plus the supplied recipe's structural data audit.
[positional-arguments]
check-data spec *options:
    #!/usr/bin/env bash
    set -euo pipefail
    spec=$1
    shift
    just check-provenance
    just check-orientation
    just check-mix "$spec" "$@"

# Run the existing offline+coherence owner; this does not run the codec exam.
# Regime is explicit; an optional fourth argument overrides the feature root.
[doc("Offline + coherence evaluation: bake, name, regime, optional feature root")]
[positional-arguments]
full-eval bake name regime root="":
    #!/usr/bin/env bash
    set -euo pipefail
    exec scripts/run_full_eval.sh "$@"

# Render recorded verdicts with the current summer-gauntlet owner.
[positional-arguments]
compare fulleval_dir out *options:
    #!/usr/bin/env bash
    set -euo pipefail
    fulleval_dir=$1
    out=$2
    shift 2
    python3 scripts/v_next/gauntlet.py --fulleval-dir "$fulleval_dir" --out "$out" "$@"
    scripts/v_next/gauntlet_gates.sh "$out" "$fulleval_dir"

# THE CROSS-LIBC GATE (F18 + F19, `zensim::det_math`): build the feature dump
# for glibc AND static musl from THIS commit and compare `to_bits()` over the
# 20-cell parity matrix + 200 ladder cells. Sweeps the 2x2 of the two era knobs
# (`ZENSIM_ROOT_FORM` x `ZENSIM_POW_FORM`) as RUNTIME env vars on the same pair
# of binaries, so only one thing varies per cell, and gates the FEATURES and the
# SCORE separately -- they are two independent defects, and the `sqrt`+`libm`
# cell is what MEASURES that (F18's fix leaves the score exactly as
# libc-dependent as it found it: 1 of 220). Revision 1 MUST differ on BOTH
# columns (the negative controls) and revision 2 MUST be bit-identical on both.
# Needs `rustup target add x86_64-unknown-linux-musl`; no container.
check-cross-libc:
    ./scripts/verify_cross_libc_features.sh

# SAME-CLASS golden check for zensim-validate/tests/legacy_bake_sha.rs: the
# PINNED sha256 digests in that file were measured on THIS box (Zen 4 /
# AVX-512) and are not bit-reproducible on other SIMD tiers/platforms — CI runs
# the in-process A/B in that file's default (env-unset) path instead. Run this
# ONLY on the Zen 4 / AVX-512 dev box that captured PINNED, after touching
# anything in mlp_train's polarity-sensitive sites, to confirm the same-class
# digests still hold.
legacy-bake-zen4-golden:
    ZENSIM_ZEN4_GOLDEN_BAKE_SHA=1 cargo test -p zensim-validate --test legacy_bake_sha -- --nocapture

# E28 registered within-leg pooled objective verification.
e28-rust-check:
    cargo test -p zensim-validate --lib loss_pearson -- --nocapture
    cargo test -p zensim-validate --lib sampling:: -- --nocapture

e28-build:
    cargo build --release -p zensim-validate --bin zensim_mlp_train --bin bake_dial_refit --bin panel --bin subset_sim

e28-python-check:
    python3 -m unittest scripts.tests.test_e28_recipe scripts.tests.test_e28_admission scripts.tests.test_e28_decision scripts.tests.test_e26_hdr_leg

[positional-arguments]
e28-parity baseline new fitbin dest inspector:
    python3 scripts/tests/e28_short_parity.py --baseline "$1" --new "$2" --fit-bin "$3" --dest "$4" --inspector "$5"

# Full caller regression; SHIPPATH_TRAINER and ZEN_PANEL_BIN select built artifacts.
e28-python-full:
    python3 -m unittest discover -s scripts/tests

[positional-arguments]
e28-smokes driver="/var/tmp/e28/v32/run_smokes.sh":
    bash "$1"

[positional-arguments]
e28-nm root dest fold="kadid":
    python3 scripts/rev4_featpot/e28_simplex.py --root "$1" --dest "$2" --heldout "$3"

[positional-arguments]
e28-smoke-receipts root arm tools inspector:
    python3 scripts/tests/e28_smoke_receipts.py --root "$1" --arm "$2" --tools "$3" --inspector "$4"

# Bounded/first-epoch checks through the actual prepared-image executor.
# The evidence directory carries the reviewed image/job/data pins and driver.
e28-executor-image-smoke evidence mode="bounded" arm="s2m":
    bash "{{evidence}}/run_executor_smoke.sh" "{{mode}}" "{{arm}}"

# Research-only PALETTE gates, keep the original Rev5 arithmetic gates intact.
palette-test:
    cargo test -p zensim --all-features --lib palette -- --nocapture
    cargo test -p zensim --all-features --test palette_research -- --nocapture

palette-build:
    cargo build --release --manifest-path zensim-bench/Cargo.toml --example extract_features_372col --features training,zen-decode

# Explicit captures refuse to replace existing base/candidate evidence.
palette-legacy-capture out:
    #!/usr/bin/env bash
    set -eu
    for rev in 1 2 3 4 5; do
        ZENSIM_FORMULA_REV="$rev" PALETTE_LEGACY_CAPTURE="{{out}}/rev$rev.bin" cargo test -p zensim --all-features --test palette_legacy_vectors -- --nocapture
    done

palette-bank bin commit out:
    python3 scripts/rev4_featpot/rev5_bank.py extract palette --palette-instrument /var/tmp/rev4-featpot/v2c5 --bin {{bin}} --build-commit {{commit}} --era palette_v2 --out {{out}} --chunk 512 --threads 8

palette-chromaq bin commit out:
    python3 scripts/rev4_featpot/rev5_bank.py extract palette --palette-chromaq /home/lilith/tmp/chromaq --bin {{bin}} --build-commit {{commit}} --era palette_v2 --out {{out}} --chunk 128 --threads 8

palette-diagnostic-report root:
    python3 scripts/rev4_featpot/palette_diagnostic.py {{root}}

palette-status log:
    rg 'PALETTE |CHROMAQ |Traceback|ValueError|run-heavy: done' {{log}} | tail -8

palette-instrument-views bin commit bank out:
    python3 scripts/rev4_featpot/rev5_bank.py extract palette --palette-views {{bank}} --palette-instrument /var/tmp/rev4-featpot/v2c5 --bin {{bin}} --build-commit {{commit}} --era palette_v2 --out {{out}}

palette-verify bin commit root instrument_sha:
    python3 scripts/rev4_featpot/rev5_bank.py extract palette --palette-verify {{root}} --palette-instrument-manifest-sha256 {{instrument_sha}} --bin {{bin}} --build-commit {{commit}} --era palette_v2

palette-mirror source destination:
    mkdir -p {{destination}}
    rsync -rlt --omit-dir-times --info=progress2 {{source}}/ {{destination}}/

# Research-only semantic admission controls, no image or label access.
palette-admission-test:
    python3 -m unittest discover -s scripts/tests -p test_palette_admission.py -v

# Round-two review gates, serialized with the caller's run-heavy wrapper.
palette-round2-gates:
    cargo test -p zensim --all-features --lib
    just clippy
    just api-doc
    just api-doc-check

palette-round2-build-checks:
    cargo check -p zensim --no-default-features --features feature-regime-v2
    cargo check --manifest-path zensim-bench/Cargo.toml --example extract_features_372col --features training,zen-decode
    just lint-scripts

# D1 role binding, immutable derivation and registered E30/production grids.
shippath10-tests:
    TMPDIR=$HOME/tmp ~/work/zen/scripts/run-heavy --mem 16G --jobs 8 -- python3 -m unittest discover -s scripts/tests -p 'test_shippath*_*.py' -q

# Explicit two-epoch/128-pair local smoke through the real strict owner.
shippath10-smoke root dest bin_dir route:
    TMPDIR=$HOME/tmp OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=1 ZENSIM_MAX_TIER=v3 ~/work/zen/scripts/run-heavy --mem 16G --jobs 8 -- python3 scripts/tests/shippath10_short_smoke.py {{root}} {{dest}} {{bin_dir}} {{route}}

# Canonical model loader verifies preserved admission, revision and sampler metadata.
shippath10-inspect model:
    TMPDIR=$HOME/tmp ~/work/zen/scripts/run-heavy --mem 16G --jobs 8 -- cargo run -p zensim-validate --release --example inspect_qualified_checkpoint -- {{model}}

# The fleet program packer owns source pin and binary inventory validation.
shippath10-program bundle zenmetrics bin_dir:
    TMPDIR=$HOME/tmp ~/work/zen/scripts/run-heavy --mem 16G --jobs 8 -- python3 {{zenmetrics}}/scripts/jobsys/pack_fit_program.py --source . --executor {{zenmetrics}}/scripts/jobsys/fit_cell_exec.py --bin-dir {{bin_dir}} --build-meta {{bundle}}/build-meta.json --profile v2d1 --out {{bundle}}/image-context/program.tar.gz

# Every archive component and declared cell is checked against its owner pins.
shippath10-bundle-check bundle:
    TMPDIR=$HOME/tmp ~/work/zen/scripts/run-heavy --mem 16G --jobs 8 -- python3 scripts/tests/shippath10_bundle_check.py {{bundle}}

# SHIPPATH11 admission/harvest negatives and real executor-entry smoke.
shippath11-tests zenmetrics:
    TMPDIR=$HOME/tmp ~/work/zen/scripts/run-heavy --mem 16G --jobs 8 -- python3 -m unittest discover -s scripts/tests -p 'test_shippath*_*.py' -q
    TMPDIR=$HOME/tmp ~/work/zen/scripts/run-heavy --mem 16G --jobs 8 -- python3 -m unittest discover -s {{zenmetrics}}/scripts/jobsys -p 'test*fit*.py' -q

shippath11-build-fit:
    TMPDIR=$HOME/tmp ~/work/zen/scripts/run-heavy --mem 16G --jobs 8 -- cargo build --release -p zensim-validate --example inspect_qualified_checkpoint --bin zensim_mlp_train --bin bake_dial_refit --bin panel

shippath11-real-check bundle zenmetrics:
    TMPDIR=$HOME/tmp ~/work/zen/scripts/run-heavy --mem 16G --jobs 8 -- python3 scripts/tests/shippath11_real_entry_checks.py --bundle {{bundle}} --zenmetrics {{zenmetrics}}

# UPIQ-380 ingestion is local artifact production; E31 remains a draft.
upiq380-check:
    python3 -m unittest scripts.tests.test_upiq380

[positional-arguments]
upiq380 *options:
    python3 scripts/rev4_featpot/upiq380.py "$@"

upiq380-rust-check:
    cargo test --locked -p zensim-validate --bin upiq_pu_score

upiq380-build:
    cargo build --locked --release -p zensim-validate --bin upiq_pu_score

upiq380-clippy:
    cargo clippy --locked -p zensim-validate --bin upiq_pu_score -- -D warnings

[positional-arguments]
upiq380-binary-refusals binary admission dest prior="":
    python3 scripts/tests/upiq380_binary_refusals.py --binary "$1" --admission "$2" --dest "$3" --prior-binary "$4"

[positional-arguments]
upiq380-split-negative revision out:
    python3 scripts/tests/upiq380_split_negative_control.py --before-revision "$1" --out "$2"

# Scope static Python checks to the UPIQ ingestion/admission owners.
upiq380-python-lint:
    ruff check scripts/rev4_featpot/upiq380.py scripts/tests/test_upiq380.py scripts/tests/upiq380_binary_refusals.py

# Preserve artifact bytes on the NAS without requesting ownership changes.
[positional-arguments]
upiq380-mirror source dest:
    rsync -a --no-owner --no-group "$1/" "$2/"

import 'benchmarks/speedq.just'

# E29 uses the existing trainer, strict four-source admission and actual executor.
e29-tests:
    python3 -m unittest scripts.tests.test_e29_consensus scripts.tests.test_e26_hdr_leg scripts.tests.test_cli_import_guards
    cargo test -p zensim-validate --lib sampling:: -- --nocapture

e29-build:
    cargo build --release -p zensim-validate --bin zensim_mlp_train --example inspect_qualified_checkpoint

e29-scorer-preflight root:
    env -i PATH="$PATH" HOME="$HOME" TMPDIR="$HOME/tmp/e29" python3 scripts/rev4_featpot/e24_rev5.py e29-score --root {{root}} --preflight-only

# Local review evidence only; no fleet queue, image publication or source push.
e29-executor-smoke bundle mode arm:
    TMPDIR=$HOME/tmp/e29 ~/work/zen/scripts/run-heavy --mem 16G --jobs 8 -- bash {{bundle}}/run_executor_smoke.sh {{mode}} {{arm}}

e29-harvest-checks bundle zenmetrics *flags:
    TMPDIR=$HOME/tmp/e29 ~/work/zen/scripts/run-heavy --mem 16G --jobs 8 -- python3 scripts/tests/shippath11_real_entry_checks.py --study e29 --bundle {{bundle}} --zenmetrics {{zenmetrics}} {{flags}}

e29-bundle-check bundle:
    TMPDIR=$HOME/tmp/e29 ~/work/zen/scripts/run-heavy --mem 16G --jobs 8 -- python3 scripts/tests/shippath10_bundle_check.py {{bundle}} --e29

e29-mirror-check bundle mirror:
    TMPDIR=$HOME/tmp/e29 ~/work/zen/scripts/run-heavy --mem 8G --jobs 1 -- python3 scripts/tests/shippath10_bundle_check.py {{bundle}} --e29 --mirror-only {{mirror}}

e29-control-parity bundle e30 dest *flags:
    TMPDIR=$HOME/tmp/e29 ~/work/zen/scripts/run-heavy --mem 8G --jobs 1 -- python3 scripts/tests/e29_control_parity.py --bundle {{bundle}} --e30 {{e30}} --dest {{dest}} {{flags}}

e29-storage-cleanup *flags:
    TMPDIR=$HOME/tmp/e29 ~/work/zen/scripts/run-heavy --mem 8G --jobs 1 -- python3 scripts/tests/e29_storage_cleanup.py {{flags}}

# Native E29 entry: declaration/label-free key refusals before feature targets.
e29-hdr-admission binary prior_binary dest:
    python3 scripts/tests/e29_hdr_binary_refusals.py --binary "{{binary}}" --prior-binary "{{prior_binary}}" --dest "{{dest}}"

# E31's control/admission gate. These commands never fit or enqueue a cell.
e31-control-tests:
    python3 -m unittest scripts.tests.test_e31_control_freeze

e31-control-freeze bundle results out:
    python3 scripts/rev4_featpot/e30_four_source.py completed-control-pins --root {{bundle}}/v2d1 --bundle {{bundle}} --results {{results}} --out {{out}}

e31-pinned-admission bundle upiq dest:
    python3 scripts/tests/e31_pinned_admission.py --bundle {{bundle}} --upiq {{upiq}} --dest {{dest}}

# E31 extension validation; builds binaries only, without creating a fleet pack.
e31-training-tests:
    TMPDIR=$HOME/tmp python3 -m unittest discover -s scripts/tests -p 'test_e31_*.py' -v

e31-python-checks:
    ruff check scripts/rev4_featpot/e31_training.py scripts/rev4_featpot/v2_common.py scripts/rev4_featpot/v2_lodo_mlp.py scripts/tests/test_e31_training.py scripts/tests/e31_control_parity.py scripts/tests/e31_extended_admission.py
    ruff format --check scripts/rev4_featpot/e31_training.py scripts/tests/test_e31_training.py scripts/tests/e31_control_parity.py scripts/tests/e31_extended_admission.py

e31-fit-key-check fit:
    TMPDIR=$HOME/tmp python3 scripts/tests/test_e31_training.py --real-fit {{fit}}

e31-build-trainer:
    TMPDIR=$HOME/tmp ~/work/zen/scripts/run-heavy --mem 16G --jobs 8 -- cargo build --locked --release -p zensim-validate --bin zensim_mlp_train --bin bake_dial_refit --bin panel --example inspect_qualified_checkpoint

e31-crate-tests:
    TMPDIR=$HOME/tmp ~/work/zen/scripts/run-heavy --mem 16G --jobs 8 -- cargo test --locked -p zensim-validate --lib --bin zensim_mlp_train -- --test-threads=1

e31-extended-admission trainer fit dest:
    TMPDIR=$HOME/tmp python3 scripts/tests/e31_extended_admission.py --trainer {{trainer}} --fit {{fit}} --dest {{dest}}

e31-control-parity baseline candidate bin_dir inspector dest:
    python3 scripts/tests/e31_control_parity.py --baseline '{{baseline}}' --candidate {{candidate}} --stripper {{bin_dir}}/bake_dial_refit --inspector {{inspector}} --dest {{dest}}

e31-control-cell root bin_dir dest scratch:
    mkdir -p {{scratch}}
    TMPDIR={{scratch}} OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 RAYON_NUM_THREADS=1 ZENSIM_MAX_TIER=v3 REV4_V2_BIN_DIR={{bin_dir}} ~/work/zen/scripts/run-heavy --mem 16G --jobs 1 -- python3 scripts/rev4_featpot/v2_lodo_mlp.py --spec sel:59f0bbc2f290@h32:H128:cv16:cf98 --head N --heldout kadid --seed-index 0 --root {{root}} --strict-admission --train-only --data-role-decision {{root}}/human_role_decision.json --columns "$(python3 -c 'import sys;sys.path.insert(0,"scripts/rev4_featpot");from e21_cheap_recipe import columns;print(",".join(map(str,columns("by_v2fy"))))')" --dest {{dest}}

# E32 research transport gates. Scratch/targets are caller-owned disk paths.
[positional-arguments]
e32-extension-tests scratch target:
    TMPDIR="$1" CARGO_TARGET_DIR="$2" ~/work/claudehints/scripts/run-heavy --mem 16G --jobs 8 -- cargo test -p zensim-validate --lib palette_training -- --nocapture
    TMPDIR="$1" SHIPPATH_TRAINER="$2/release/zensim_mlp_train" ~/work/claudehints/scripts/run-heavy --mem 4G --jobs 1 -- env PYTHONPATH=scripts/rev4_featpot:scripts/tests python3 -m unittest discover -s scripts/tests -p test_e32_palette_training.py

[positional-arguments]
e32-extension-build scratch target:
    TMPDIR="$1" CARGO_TARGET_DIR="$2" ~/work/claudehints/scripts/run-heavy --mem 16G --jobs 8 -- cargo build --release -p zensim-validate --bin zensim_mlp_train --bin bake_dial_refit --example inspect_qualified_checkpoint

[positional-arguments]
e32-shippath-regression scratch trainer:
    TMPDIR="$1" SHIPPATH_TRAINER="$2" ~/work/claudehints/scripts/run-heavy --mem 4G --jobs 1 -- env PYTHONPATH=scripts/rev4_featpot:scripts/tests python3 -m unittest discover -s scripts/tests -p 'test_shippath*.py'

[positional-arguments]
e32-serving-refusal scratch target:
    TMPDIR="$1" CARGO_TARGET_DIR="$2" ~/work/claudehints/scripts/run-heavy --mem 16G --jobs 8 -- cargo test -p zensim --all-features --lib rev5_for_bake_refuses_reads_outside_the_supported_families -- --nocapture

[positional-arguments]
e32-existing-rust-tests scratch target:
    TMPDIR="$1" CARGO_TARGET_DIR="$2" ~/work/claudehints/scripts/run-heavy --mem 16G --jobs 8 -- cargo test -p zensim-validate --lib feature_set -- --nocapture
    TMPDIR="$1" CARGO_TARGET_DIR="$2" ~/work/claudehints/scripts/run-heavy --mem 16G --jobs 8 -- cargo test -p zensim-validate --lib parquet_loader -- --nocapture

# Full 120 x 50,000 control replay, pinned E30 model comparison, no fleet owner.
[positional-arguments]
e32-control-parity scratch bindir root control freeze dest:
    TMPDIR="$1" ZENSIM_MAX_TIER=v3 OPENBLAS_NUM_THREADS=1 ~/work/claudehints/scripts/run-heavy --mem 16G --jobs 1 -- python3 scripts/tests/e28_short_parity.py --new "$2/zensim_mlp_train" --fit-bin "$2/bake_dial_refit" --inspector "$2/examples/inspect_qualified_checkpoint" --prepared-root "$3" --e30-control "$4" --control-freeze "$5" --dest "$6"


# V40 integrated admission and package checks, local only.
v40-build:
    cargo build --locked --release -p zensim-validate --bin zensim_mlp_train --bin bake_dial_refit --bin panel --bin predict_features_with_bake --example inspect_qualified_checkpoint

v40-native-tests:
    cargo test --locked -p zensim-validate --lib --bin zensim_mlp_train -- --test-threads=1

v40-python-tests trainer:
    TMPDIR=$HOME/tmp/v40 SHIPPATH_TRAINER={{trainer}} PYTHONPATH=scripts/rev4_featpot:scripts/tests python3 -m unittest discover -s scripts/tests -p 'test_e32_palette_training.py'
    TMPDIR=$HOME/tmp/v40 SHIPPATH_TRAINER={{trainer}} PYTHONPATH=scripts/rev4_featpot:scripts/tests python3 -m unittest discover -s scripts/tests -p 'test_shippath*.py'
    TMPDIR=$HOME/tmp/v40 ZEN_PANEL_BIN=$(dirname {{trainer}})/panel PYTHONPATH=scripts/rev4_featpot:scripts/tests python3 -m unittest scripts.tests.test_e29_consensus scripts.tests.test_e26_hdr_leg scripts.tests.test_cli_import_guards scripts.tests.test_e31_training scripts.tests.test_e31_control_freeze

v40-admission binary dest:
    python3 scripts/tests/v40_native_admission.py --binary {{binary}} --dest {{dest}} --upiq-manifest /mnt/v/output/zensim/upiq380-rev5-r2-2026-10-07/upiq380_fit.parquet.manifest.json

v40-projection-tests:
    PYTHONPATH=scripts/rev4_featpot:scripts/tests python3 -m unittest scripts.tests.test_v40_projection

v40-projection source palette out commit:
    python3 scripts/rev4_featpot/e32_palette.py --source {{source}} --palette-root {{palette}} --out {{out}} --fleet-root /var/tmp/rev4-featpot/v2e32 --build-commit {{commit}}

v40-parity bundle e30 dest fold:
    python3 scripts/tests/e29_control_parity.py --bundle {{bundle}} --e30 {{e30}} --dest {{dest}} --fold {{fold}}

# E33 (benchmarks/e33_registration_2026-10-09.md): parity pre-check, identity proof, packet, tests.
e33-parity bundle v40results bindir dest fold="kadid":
    python3 scripts/tests/e33_control_parity.py --bundle {{bundle}} --v40-results {{v40results}} --bin-dir {{bindir}} --dest {{dest}} --fold {{fold}}

e33-predictor-parity bundle v40results fitbin dest:
    python3 scripts/tests/e33_predictor_parity.py --bundle {{bundle}} --v40-results {{v40results}} --fitbin {{fitbin}} --dest {{dest}}

e33-identity sources model out:
    ZENSIM_FORMULA_REV=5 cargo run --release -p zensim --features custom-profiles,candidate-profiles --example serve_custom_bake -- --e33-identity {{sources}} scripts/rev4_featpot/e33_fx1_declaration.json {{model}} {{out}}

e33-package-tests:
    PYTHONPATH=scripts:scripts/rev4_featpot:scripts/tests python3 -m unittest scripts.tests.test_e33_package scripts.tests.test_e33_launch scripts.tests.test_e33_score scripts.tests.test_e33_verdict

# E33 launch owner (coordinator authorization 2026-10-10): verify | caps | launch | placements | harvest.
e33-launch mode packet *flags:
    python3 scripts/rev4_featpot/e33_launch.py {{mode}} --packet {{packet}} {{flags}}

e33-rust-tests:
    cargo test -p zensim --all-features --lib derived_inputs
    cargo test -p zensim-validate --lib -- derived_inputs identity_pinned

v40-executor-smoke bundle image arm mode attempt="1":
    python3 scripts/tests/v40_executor_smoke.py --bundle {{bundle}} --image {{image}} --arm {{arm}} --mode {{mode}} --attempt {{attempt}}

v40-statistics-tests:
    PYTHONPATH=scripts:scripts/rev4_featpot:scripts/tests python3 -m unittest scripts.tests.test_v40_projection scripts.tests.test_v40_statistics scripts.tests.test_v40_launch scripts.tests.test_v40_panels

v40-freeze bundle source_commit metrics_commit:
    python3 scripts/tests/v40_package_freeze.py --bundle {{bundle}} --source {{justfile_directory()}} --source-commit {{source_commit}} --metrics-commit {{metrics_commit}}

v40-cached-projection bundle out attempt:
    python3 scripts/tests/v40_cached_projection_smoke.py --bundle {{bundle}} --out {{out}} --harvest-attempt {{attempt}}

v40-research-tests:
    cargo test --locked -p zensim-validate --bin bake_dial_refit
    ZENSIM_POW_FORM=pure cargo test --locked -p zensim-validate --bin bake_dial_refit research_scalar_tail_is_bit_identical

v40-assessment-build:
    cargo build --locked --release -p zensim-validate --bin bake_dial_refit --bin predict_features_with_bake

v40-bundle-check bundle:
    python3 scripts/tests/v40_bundle_check.py --bundle {{bundle}}

# Reviewer shapes use synthetic payloads and approved label-free TRAIN declarations.
v40-review-admission binary dest:
    python3 scripts/tests/v40_review_admission.py --binary {{binary}} --dest {{dest}}

v40-source-bindings bundle producer:
    python3 scripts/tests/v40_source_bindings.py --source {{justfile_directory()}} --bundle {{bundle}} --producer {{producer}}

v40-authorization-gate bundle dest:
    python3 scripts/tests/v40_authorization_gate.py --bundle {{bundle}} --dest {{dest}}

v40-r2-stage previous bundle quiet_start quiet_release:
    python3 scripts/tests/v40_r2_stage.py --previous {{previous}} --bundle {{bundle}} --quiet-start {{quiet_start}} --quiet-release {{quiet_release}}

# Complete native inventory, including unregistered manifest and auxiliary inputs.
v40-inventory-admission binary dest:
    python3 scripts/tests/v40_inventory_admission.py --binary {{binary}} --dest {{dest}}

v40-postfit-artifacts-tests:
    PYTHONPATH=scripts:scripts/rev4_featpot:scripts/tests python3 -m unittest scripts.tests.test_v40_postfit_artifacts

v40-scoring-rehearsal bundle image dest attempt="77":
    python3 scripts/tests/v40_scoring_rehearsal.py --bundle {{bundle}} --image {{image}} --dest {{dest}} --attempt {{attempt}}

v40-r3-stage previous bundle bindir producer quiet_start quiet_override authority build_log:
    python3 scripts/tests/v40_r3_stage.py --previous {{previous}} --bundle {{bundle}} --bin-dir {{bindir}} --producer {{producer}} --quiet-start {{quiet_start}} --quiet-override {{quiet_override}} --release-authority {{authority}} --build-log {{build_log}}

# Run under the shared heavy lock; the retained trainer is the shipped binary.
v40-r3-checks bundle dest metrics:
    just v40-admission {{bundle}}/bin/zensim_mlp_train {{dest}}/author-admission
    just v40-review-admission {{bundle}}/bin/zensim_mlp_train {{dest}}/review-admission
    just v40-python-tests {{bundle}}/bin/zensim_mlp_train
    just v40-statistics-tests
    just v40-postfit-artifacts-tests
    just clippy
    cargo fmt -p zensim -p zensim-validate --check
    just --justfile {{metrics}}/justfile --working-directory {{metrics}} v40-fit-tools
    just --justfile {{metrics}}/justfile --working-directory {{metrics}} v40-postfit-checks

v40-r3-executors bundle image attempt="1":
    #!/usr/bin/env bash
    set -euo pipefail
    for arm in control hb4 hc4 palette uh4; do
        for mode in bounded first-epoch; do
            just v40-executor-smoke '{{bundle}}' '{{image}}' "$arm" "$mode" '{{attempt}}'
        done
    done

v40-r3-parity bundle e30:
    just v40-parity {{bundle}} {{e30}} {{bundle}}/parity-kadid kadid
    just v40-parity {{bundle}} {{e30}} {{bundle}}/parity-tid2013 tid2013

v40-r3-lint:
    ruff check scripts/rev4_featpot/v40_score.py scripts/tests/v40_inventory_admission.py scripts/tests/v40_postfit_fixture.py scripts/tests/v40_scoring_rehearsal.py scripts/tests/v40_r3_stage.py scripts/tests/v40_evidence_archive.py scripts/tests/test_v40_postfit_artifacts.py
    ruff format --check scripts/rev4_featpot/v40_score.py scripts/tests/v40_inventory_admission.py scripts/tests/v40_postfit_fixture.py scripts/tests/v40_scoring_rehearsal.py scripts/tests/v40_r3_stage.py scripts/tests/v40_evidence_archive.py scripts/tests/test_v40_postfit_artifacts.py

v40-evidence-archive bundle logs mirror:
    python3 scripts/tests/v40_evidence_archive.py archive --bundle {{bundle}} --logs {{logs}} --mirror {{mirror}}

v40-own-cargo-cleanup bundle mirror:
    python3 scripts/tests/v40_evidence_archive.py cleanup --bundle {{bundle}} --mirror {{mirror}} --source {{justfile_directory()}}

# Synthetic gate() authorizations only; fixed approved E30 metadata mounts.
v40-image-authorization bundle image evidence scratch:
    docker run --rm --network=none --cpus=1 --memory=2g --memory-swap=2g --user $(id -u):$(id -g) --entrypoint python3 -e TMPDIR=/scratch/tmp -e PYTHONPYCACHEPREFIX=/scratch/pycache -v {{bundle}}:{{bundle}}:ro -v /mnt/v/output/zensim/v40-2026-10-07/upiq380-fit:/mnt/v/output/zensim/v40-2026-10-07/upiq380-fit:ro -v /var/tmp/rev4-featpot/e30-results/cells:/var/tmp/rev4-featpot/e30-results/cells:ro -v {{evidence}}:/evidence -v {{scratch}}:/scratch -v {{justfile_directory()}}/scripts/tests/v40_authorization_gate.py:/v40-gate.py:ro -v /mnt/v/output/zensim/shippath11-2026-10-07/fit-manifest-fitv2e30-20261007.json:/mnt/v/output/zensim/shippath11-2026-10-07/fit-manifest-fitv2e30-20261007.json:ro -v /mnt/v/output/zensim/shippath11-2026-10-07/PROGRAM_INVENTORY.json:/mnt/v/output/zensim/shippath11-2026-10-07/PROGRAM_INVENTORY.json:ro -v /mnt/v/output/zensim/shippath11-2026-10-07/ARTIFACT_PINS.json:/mnt/v/output/zensim/shippath11-2026-10-07/ARTIFACT_PINS.json:ro -v /mnt/v/output/zensim/shippath11-2026-10-07/image-context/program.tar.gz:/mnt/v/output/zensim/shippath11-2026-10-07/image-context/program.tar.gz:ro -v /mnt/v/output/zensim/shippath11-2026-10-07/d1-fit-data.tar.gz:/mnt/v/output/zensim/shippath11-2026-10-07/d1-fit-data.tar.gz:ro -v /mnt/v/output/zensim/shippath11-2026-10-07/bin/inspect_qualified_checkpoint:/mnt/v/output/zensim/shippath11-2026-10-07/bin/inspect_qualified_checkpoint:ro {{image}} /v40-gate.py --bundle {{bundle}} --dest /evidence/authorization-gate

# Replay compatibility cannot authorize declared protected populations.
v40-historical-admission binary dest:
    python3 scripts/tests/v40_historical_admission.py --binary {{binary}} --dest {{dest}}

v40-binary-metadata-tests:
    PYTHONPATH=scripts/tests python3 -m unittest scripts.tests.test_v40_binary_metadata

v40-r4-prepare bundle producer source_commit metrics_commit image:
    python3 scripts/tests/v40_r4_prepare.py --bundle {{bundle}} --producer {{producer}} --source-commit {{source_commit}} --metrics-commit {{metrics_commit}} --image {{image}}

# Complete inherited checks plus replay and producer regressions; shared heavy wrapper required.
v40-r4-checks bundle dest metrics:
    just v40-historical-admission {{bundle}}/bin/zensim_mlp_train {{dest}}/historical
    just v40-inventory-admission {{bundle}}/bin/zensim_mlp_train {{dest}}/inventory
    just v40-native-tests
    just v40-binary-metadata-tests
    just v40-r3-checks {{bundle}} {{dest}}/inherited {{metrics}}

# Registered E31 reports only; this checks existing outputs and reads no datasets.
v40-e31-report-evidence report exposure out:
    python3 scripts/tests/v40_e31_report_evidence.py --report {{report}} --exposure {{exposure}} --out {{out}}

v40-e31-report-evidence-tests:
    PYTHONPATH=scripts/tests python3 -m unittest scripts.tests.test_v40_e31_report_evidence

# dialviz: static site explaining how zensim is evaluated (docs/DIALVIZ_DESIGN_2026-10-10.md).
# Reads committed sources only; extra args pass through (e.g. --integrity-gates PATH).
dialviz out="/mnt/v/output/zensim/dialviz" *args="":
    python3 scripts/dialviz/build.py --out {{quote(out)}} {{args}}

# Readers against live sources, cross-checks, negative controls and a link-checked build.
dialviz-test:
    TMPDIR=$HOME/tmp python3 -m unittest scripts.tests.test_dialviz -v

# Research maintenance verification.
tidy-check:
    cargo test -p zensim --all-features --lib
    cargo test -p zensim --all-features --test research_engine_parity --test rev4_featbank_parity --test restore_cuts_parity --test palette_research

tidy-python-check:
    TMPDIR=$HOME/tmp PYTHONPATH=scripts:scripts/lib:scripts/tests python3 -m unittest scripts.tests.test_feature_screen_splits

tidy-tool-check:
    cargo test -p zensim-validate --lib feature_set
    cargo check --manifest-path zensim-bench/Cargo.toml --features training,zen-decode --example extract_features_372col

# E34 P1: explicit registered fit root; decodes features and row keys only.
e34-feature-census root contract output:
    python3 scripts/rev4_featpot/e34_prepare.py --root '{{root}}' --contract '{{contract}}' --output '{{output}}'

# Explicit engineering-only builds and P2 packet/fixture preparation.
e34-steering-build:
    RUSTFLAGS='--cfg steerfix_instrument' cargo test -p zensim --lib --features custom-profiles --no-run

e34-linear-probe *args:
    python3 scripts/rev4_featpot/e34_linear_probe.py {{args}}

# Synthetic trainer contracts and the frozen default argv (no corpus labels).
e34-trainer-tests:
    cargo test -p zensim-validate --lib e34_
    python3 scripts/tests/test_e34_prepare.py
    python3 scripts/tests/test_v2c.py TrainRecipe

e34-trainer-regression:
    cargo test -p zensim-validate --lib mlp_train::
    cargo clippy -p zensim-validate --lib --bin zensim_mlp_train -- -D warnings

e34-preparation-build:
    cargo build -p zensim --example e34_pixels --features custom-profiles
    cargo build -p zensim-validate --bin check_holdout_overlap --bin zensim_mlp_train

# Label-free inventory for the explicitly registered E34 sources.
e34-preparation-inventory:
    python3 scripts/rev4_featpot/e34_augmentation.py inventory --out /mnt/v/output/zensim/e34-prep-2026-10-15/p4-inventory --split-owner /mnt/v/output/zensim/e34-prep-2026-10-15/p4-inputs/origin_split.py --family-map /mnt/v/output/zensim/e34-prep-2026-10-15/p4-inputs/split_map_family.tsv --pixels /mnt/v/output/imazen-26-png-v3 --identity /mnt/v/output/zensim/e33-impl-2026-10-09/E3_SOURCES.json --packet /mnt/v/output/zensim/steerfix-2026-10-09/full-packet.json --bank /var/tmp/rev4-featbank/bank --instrument-keys /var/tmp/shippath7/verified/instruments/ladder.keys.parquet --instrument-keys /var/tmp/shippath7/verified/instruments/broad.keys.parquet --instrument-keys /var/tmp/shippath7/verified/instruments/negtail.keys.parquet --instrument-keys /var/tmp/shippath7/verified/instruments/jpeg8.keys.parquet --instrument-keys /var/tmp/shippath7/verified/instruments/identity.keys.parquet --instrument-keys /var/tmp/shippath7/verified/instruments/m3.keys.parquet --instrument-keys /var/tmp/shippath7/verified/instruments/owner.keys.parquet --instrument-keys /var/tmp/shippath7/verified/instruments/standard.keys.parquet

# Part 3 uses the current native pack and serving owners.
e34-serving-build:
    cargo build -p zensim-validate --bin bake_dial_refit --bin bake_verdict
    cargo build -p zensim --example serve_custom_bake --features custom-profiles

# Run each arm separately under run-heavy so it has its own resource receipt.
e34-first-epoch arm:
    python3 scripts/rev4_featpot/e34_smoke.py --arm {{arm}} --root /mnt/v/output/zensim/e33-impl-2026-10-09/packet/container-scratch-1/fit-cell/9c3eff1b740d2b7a77a235a16a3cd3521746af55a72bb92cddd0536674963008/rev4-featpot/v2e29 --dest /mnt/v/output/zensim/e34-prep-2026-10-15/part3/smokes/{{arm}} --bin-dir {{justfile_directory()}}/target/debug

e34-l-gate mode:
    REV4_V2_BIN_DIR={{justfile_directory()}}/target/debug ZENSIM_FORMULA_REV=5 RAYON_NUM_THREADS=1 python3 scripts/rev4_featpot/e34_l_gates.py {{mode}} --smoke /mnt/v/output/zensim/e34-prep-2026-10-15/part3/smokes/l-a --out /mnt/v/output/zensim/e34-prep-2026-10-15/part3/l-gates --root /mnt/v/output/zensim/e33-impl-2026-10-09/packet/container-scratch-1/fit-cell/9c3eff1b740d2b7a77a235a16a3cd3521746af55a72bb92cddd0536674963008/rev4-featpot/v2e29 --bin-dir {{justfile_directory()}}/target/debug

e34-e5-cli-check:
    #!/usr/bin/env bash
    set -euo pipefail
    cargo build -p zensim-validate --bin zensim_mlp_train
    log="$HOME/tmp/e34prep/p3-e5-cli-refusal.log"
    if target/debug/zensim_mlp_train --group "unused:$HOME/tmp/e34prep/e5-must-not-be-read.parquet:1:0:withinref,rank" --nonneg-distance --monotone-cbc --hidden 1 --feature-transform identity:13 --out "$HOME/tmp/e34prep/e5-must-not-exist.bin" > "$log" 2>&1; then
        exit 1
    fi
    rg -q 'plain monotone N refuses feature transforms' "$log"
    test ! -e "$HOME/tmp/e34prep/e5-must-not-exist.bin"

e34-l-model-contract:
    ZENSIM_FORMULA_REV=5 E34_LINEAR_MODEL_CONTRACT=/mnt/v/output/zensim/e34-prep-2026-10-15/part3/l-gates/model-contract-job.json target/debug/deps/zensim-6fc55070ad1b5213 metric::bake::steerfix_packet::e34_linear_model_contract --exact --nocapture

e34-l-p2:
    ZENSIM_FORMULA_REV=5 ZENSIM_NEIGHBOUR_EXACT=1 RAYON_NUM_THREADS=1 STEERFIX_PACKET=/mnt/v/output/zensim/e34-prep-2026-10-15/part3/p2-l-packet.json target/debug/deps/zensim-6fc55070ad1b5213 metric::bake::steerfix_packet::engineering_packet --exact --nocapture

e34-part3-receipt mode root output:
    python3 scripts/rev4_featpot/e34_part3_receipts.py {{mode}} --root {{root}} --output {{output}}

e34-packet-progress path:
    python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); r=d if isinstance(d,list) else d["rows"]; p=[x["linear_probe"] for x in r if x["linear_probe"]]; print(dict(cases=len(r), passed=sum(x["pass"] for x in r), last=r[-1]["key"], max_abs_s_plus_c=max((x["max_abs_s_plus_c"] for x in p),default=None), negative_cases=sum(bool(x["negative_input_rows"]) for x in p)))' {{path}}

e34-part3-backup:
    mkdir -p /mnt/tower/output/zensim/e34-prep-2026-10-15/part3
    rsync -rt --ignore-existing --exclude=bin/ /mnt/v/output/zensim/e34-prep-2026-10-15/part3/ /mnt/tower/output/zensim/e34-prep-2026-10-15/part3/

e34-part3-backup-check:
    #!/usr/bin/env python3
    import hashlib
    from pathlib import Path
    source = Path('/mnt/v/output/zensim/e34-prep-2026-10-15/part3')
    target = Path('/mnt/tower/output/zensim/e34-prep-2026-10-15/part3')
    files = sorted(p for p in source.rglob('*') if p.is_file() and not p.is_symlink())
    def digest(p):
        with p.open('rb') as f:
            return hashlib.file_digest(f, 'sha256').hexdigest()
    for p in files:
        assert digest(p) == digest(target / p.relative_to(source)), p
    print(f'SHA-256 matches for all {len(files)} regular artifact files; executable symlinks excluded')

e34-precision-contracts:
    target/debug/deps/zensim-6fc55070ad1b5213 metric::bake::steerfix_packet::precision::reference_contracts --exact --nocapture

e34-precision-clippy:
    RUSTFLAGS='--cfg steerfix_instrument' cargo clippy -p zensim --lib --tests --features custom-profiles -- -D warnings

e34-precision-run run:
    ZENSIM_FORMULA_REV=5 ZENSIM_NEIGHBOUR_EXACT=1 RAYON_NUM_THREADS=1 E34_PRECISION_PACKET=/mnt/v/output/zensim/e34-prep-2026-10-15/part4/{{run}}-job.json target/debug/deps/zensim-6fc55070ad1b5213 metric::bake::steerfix_packet::precision::gate_precision_packet --exact --nocapture

e34-precision mode:
    python3 scripts/rev4_featpot/e34_precision.py {{mode}} --root /mnt/v/output/zensim/e34-prep-2026-10-15/part4

e34-precision-tail output="a-tail.json":
    E34_PRECISION_TAIL_OUT={{output}} E34_PRECISION_PACKET=/mnt/v/output/zensim/e34-prep-2026-10-15/part4/full-job.json target/debug/deps/zensim-6fc55070ad1b5213 metric::bake::steerfix_packet::precision::cached_tail_probe --exact --nocapture

e34-part4-backup:
    mkdir -p /mnt/tower/output/zensim/e34-prep-2026-10-15/part4
    rsync -rt --ignore-existing /mnt/v/output/zensim/e34-prep-2026-10-15/part4/ /mnt/tower/output/zensim/e34-prep-2026-10-15/part4/

e34-part4-backup-check:
    #!/usr/bin/env python3
    import hashlib
    from pathlib import Path
    source = Path('/mnt/v/output/zensim/e34-prep-2026-10-15/part4')
    target = Path('/mnt/tower/output/zensim/e34-prep-2026-10-15/part4')
    files = sorted(p for p in source.rglob('*') if p.is_file())
    def digest(p):
        with p.open('rb') as f:
            return hashlib.file_digest(f, 'sha256').hexdigest()
    for p in files:
        assert digest(p) == digest(target / p.relative_to(source)), p
    print(f'SHA-256 matches for all {len(files)} artifact files')

# Part 5 begins with summary-only parity; no prediction payload is read here.
e34-stats-reference:
    python3 scripts/rev4_featpot/e34_stats_reference.py --output /mnt/v/output/zensim/e34-prep-2026-10-15/part5/python-seed-parity.json

e34-stats-build:
    cargo build -p zensim-validate --bin panel --bin bake_verdict

e34-stats-parity:
    target/debug/panel e34 seed-parity --input /mnt/v/output/zensim/e33-impl-2026-10-09/assessment-e33/e33_e21.json --output /mnt/v/output/zensim/e34-prep-2026-10-15/part5/rust-seed-parity.json

e34-stats-check:
    python3 scripts/rev4_featpot/e34_stats_parity.py

e34-stats-tests:
    cargo test -p zensim-validate --bin panel panel_e34::
    cargo clippy -p zensim-validate --bin panel --bin bake_verdict -- -D warnings

e34-noise mode:
    python3 scripts/rev4_featpot/e34_noise.py {{mode}} --out /mnt/v/output/zensim/e34-prep-2026-10-15/part5 --panel {{justfile_directory()}}/target/debug/panel

# Approved D11 uses fresh receipts; earlier failed preflight remains intact.
e34-part5-inventory name="p4-inventory":
    python3 scripts/rev4_featpot/e34_augmentation.py inventory --out /mnt/v/output/zensim/e34-prep-2026-10-15/part5/{{name}} --split-owner /mnt/v/output/zensim/e34-prep-2026-10-15/p4-inputs/origin_split.py --family-map /mnt/v/output/zensim/e34-prep-2026-10-15/p4-inputs/split_map_family.tsv --pixels /mnt/v/output/imazen-26-png-v3 --identity /mnt/v/output/zensim/e33-impl-2026-10-09/E3_SOURCES.json --packet /mnt/v/output/zensim/steerfix-2026-10-09/full-packet.json --bank /var/tmp/rev4-featbank/bank --instrument-keys /var/tmp/shippath7/verified/instruments/ladder.keys.parquet --instrument-keys /var/tmp/shippath7/verified/instruments/broad.keys.parquet --instrument-keys /var/tmp/shippath7/verified/instruments/negtail.keys.parquet --instrument-keys /var/tmp/shippath7/verified/instruments/jpeg8.keys.parquet --instrument-keys /var/tmp/shippath7/verified/instruments/identity.keys.parquet --instrument-keys /var/tmp/shippath7/verified/instruments/m3.keys.parquet --instrument-keys /var/tmp/shippath7/verified/instruments/owner.keys.parquet --instrument-keys /var/tmp/shippath7/verified/instruments/standard.keys.parquet

e34-admission-tests:
    python3 scripts/tests/test_e34_admission.py


e34-p4-normalize:
    target/debug/examples/e34_pixels /mnt/v/output/zensim/e34-prep-2026-10-15/part5/p4-inventory-r2/normalize.json

e34-p4-audit:
    target/debug/check_holdout_overlap --native-png --max-decode-pixels 120000000 --cid22-refs /mnt/v/output/zensim/e34-prep-2026-10-15/part5/p4-inventory-r2/excluded-png --training-csv /mnt/v/output/zensim/e34-prep-2026-10-15/part5/p4-inventory-r2/audit.csv --threshold 10 --expected-training 4221 --expected-holdout 1113 --out-tsv /mnt/v/output/zensim/e34-prep-2026-10-15/part5/p4-inventory-r2/audit.tsv

e34-reference-build:
    cargo build --manifest-path tools/e34-reference-features/Cargo.toml

e34-moz-source:
    mkdir -p /mnt/v/output/zensim/e34-prep-2026-10-15/part5/encoder-source
    jj git clone https://github.com/imazen/mozjpeg /mnt/v/output/zensim/e34-prep-2026-10-15/part5/encoder-source/mozjpeg

e34-p4-eligibility:
    python3 scripts/rev4_featpot/e34_augmentation.py audit --root /mnt/v/output/zensim/e34-prep-2026-10-15/part5/p4-inventory-r2 --context-review /mnt/v/output/zensim/e34-prep-2026-10-15/part5/p4-inventory-r2/safesyn-context-review.json

e34-p4-resize:
    target/debug/examples/e34_pixels /mnt/v/output/zensim/e34-prep-2026-10-15/part5/p4-inventory-r2/resize.json

e34-p4-reference-job:
    python3 scripts/rev4_featpot/e34_augmentation.py reference-job --root /mnt/v/output/zensim/e34-prep-2026-10-15/part5/p4-inventory-r2

e34-p4-reference-features:
    ZENSIM_MAX_TIER=v3 tools/e34-reference-features/target/debug/e34-reference-features /mnt/v/output/zensim/e34-prep-2026-10-15/part5/p4-inventory-r2/reference-features-job.json

e34-p4-selection:
    python3 scripts/rev4_featpot/e34_augmentation.py select --root /mnt/v/output/zensim/e34-prep-2026-10-15/part5/p4-inventory-r2

e34-moz-build:
    cmake -S /mnt/v/output/zensim/e34-prep-2026-10-15/part5/encoder-source/mozjpeg -B /mnt/v/output/zensim/e34-prep-2026-10-15/part5/encoder-build/mozjpeg -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DWITH_SIMD=ON -DPNG_SUPPORTED=ON
    cmake --build /mnt/v/output/zensim/e34-prep-2026-10-15/part5/encoder-build/mozjpeg --parallel 8 --target cjpeg-static

e34-p4-encode mode:
    python3 scripts/rev4_featpot/e34_encode.py {{mode}} --root /mnt/v/output/zensim/e34-prep-2026-10-15/part5/p4-inventory-r2

e34-jxl-build:
    python3 scripts/rev4_featpot/e34_jxl_build.py


e34-p4-finish mode:
    python3 scripts/rev4_featpot/e34_p4_finish.py {{mode}}

e34-p4-repairs:
    ZENSIM_FORMULA_REV=5 ZENSIM_MAX_TIER=v3 target/debug/examples/e34_pixels /mnt/v/output/zensim/e34-prep-2026-10-15/part5/p4-inventory-r2/repairs/job.json


e34-k-smoke arm attempt="complete":
    python3 scripts/rev4_featpot/e34_smoke.py --arm {{arm}} --root /mnt/v/output/zensim/e33-impl-2026-10-09/packet/container-scratch-1/fit-cell/9c3eff1b740d2b7a77a235a16a3cd3521746af55a72bb92cddd0536674963008/rev4-featpot/v2e29 --dest /mnt/v/output/zensim/e34-prep-2026-10-15/part5/smokes/{{attempt}}/{{arm}} --bin-dir {{justfile_directory()}}/target/debug --steer-pairs /mnt/v/output/zensim/e34-prep-2026-10-15/part5/p4-inventory-r2/repairs/rank-inputs/augmentation.parquet --steer-lambda 0.1

e34-p4-probe-build:
    cargo build -p zensim --example e34_pixels --features custom-profiles

e34-p4-probe-pixels:
    ZENSIM_FORMULA_REV=5 ZENSIM_MAX_TIER=v3 target/debug/examples/e34_pixels /mnt/v/output/zensim/e34-prep-2026-10-15/part5/p4-inventory-r2/parity/probe-job.json

e34-p4-native-parity:
    ZENSIM_FORMULA_REV=5 ZENSIM_MAX_TIER=v3 /var/tmp/rev5-extract/target/release/examples/extract_features_372col --corpus pairs --path /mnt/v/output/zensim/e34-prep-2026-10-15/part5/p4-inventory-r2/parity/pairs.tsv --out /mnt/v/output/zensim/e34-prep-2026-10-15/part5/p4-inventory-r2/parity/native-paired.csv --restore-cuts basic,peaks,v2 --force-tier v3


e34-p4-parity-check:
    python3 scripts/rev4_featpot/e34_native_parity.py


e34-part5-status:
    wc -l /mnt/v/output/zensim/e34-prep-2026-10-15/part5/p4-inventory-r2/repairs/features.jsonl
    ps -C e34_pixels,zensim_mlp_train,cargo,rustc -o pid,etime,%cpu,rss
    df -h /home

e34-part5-receipts:
    python3 scripts/rev4_featpot/e34_part5_receipts.py

e34-part5-backup:
    mkdir -p /mnt/tower/output/zensim/e34-prep-2026-10-15/part5
    rsync -rt --ignore-existing --exclude='.git/' --exclude='.jj/' /mnt/v/output/zensim/e34-prep-2026-10-15/part5/ /mnt/tower/output/zensim/e34-prep-2026-10-15/part5/

e34-part5-backup-check:
    python3 scripts/rev4_featpot/e34_artifacts.py backup-check

e34-rank-layout:
    python3 scripts/rev4_featpot/e34_rank_layout.py

e34-part5-archive-binaries:
    python3 scripts/rev4_featpot/e34_artifacts.py archive-binaries

e34-part5-checks:
    just lint-scripts
    python3 scripts/tests/test_e34_prepare.py
    python3 scripts/tests/test_e34_admission.py
    cargo test -p zensim-validate --bin panel panel_e34::
    cargo clippy -p zensim-validate --bin panel --bin bake_verdict -- -D warnings
    cargo clippy -p zensim --example e34_pixels --features custom-profiles -- -D warnings
    cargo clippy --manifest-path tools/e34-reference-features/Cargo.toml -- -D warnings
    cargo clippy --manifest-path tools/e34-jxl-scalar/Cargo.toml --config /mnt/v/output/zensim/e34-prep-2026-10-15/part5/jxl-build-patch.toml -- -D warnings
    rustfmt --check --edition 2024 tools/e34-reference-features/src/main.rs tools/e34-jxl-scalar/src/main.rs zensim/examples/e34_pixels.rs zensim-validate/src/bin/panel_e34/mod.rs

e34-final-parity:
    target/debug/panel e34 seed-parity --input /mnt/v/output/zensim/e33-impl-2026-10-09/assessment-e33/e33_e21.json --output /mnt/v/output/zensim/e34-prep-2026-10-15/part5/rust-seed-parity-final.json
    cmp /mnt/v/output/zensim/e34-prep-2026-10-15/part5/rust-seed-parity.json /mnt/v/output/zensim/e34-prep-2026-10-15/part5/rust-seed-parity-final.json


e34-d12-bench-build:
    cargo build -p zensim --example serve_custom_bake --features custom-profiles

e34-d12-timing stage:
    ZENSIM_FORMULA_REV=5 ZENSIM_MAX_TIER=v3 ZENSIM_NEIGHBOUR_EXACT=1 RAYON_NUM_THREADS=1 target/debug/examples/serve_custom_bake --steering-timing /mnt/v/output/zensim/e34-prep-2026-10-15/part4/full-job.json /mnt/v/output/zensim/e34-prep-2026-10-15/part6/{{stage}}-timing.jsonl


e34-d12-owner-tests:
    cargo test -p zensim --lib --features custom-profiles metric::bake::

e34-d12-clippy:
    cargo clippy -p zensim --lib --tests --example serve_custom_bake --features custom-profiles -- -D warnings

e34-d12-gradient-tests:
    cargo test -p zensim --lib --features custom-profiles gradient::tests

e34-d12-reference:
    ZENSIM_FORMULA_REV=5 E34_D12_REFERENCE_JOB=/mnt/v/output/zensim/e34-prep-2026-10-15/part6/reference-job-v2.json target/debug/deps/zensim-6fc55070ad1b5213 metric::bake::steerfix_packet::precision::d12_exact_gradient_reference_packet --exact --nocapture

e34-d12-gate model:
    ZENSIM_FORMULA_REV=5 ZENSIM_NEIGHBOUR_EXACT=1 RAYON_NUM_THREADS=1 STEERFIX_PACKET=/mnt/v/output/zensim/e34-prep-2026-10-15/part6/gate-{{model}}/job.json target/debug/deps/zensim-6fc55070ad1b5213 metric::bake::steerfix_packet::engineering_packet --exact --nocapture

e34-d12-validation-build:
    just e34-d12-gradient-tests
    just e34-steering-build

e34-d12-companion-regression:
    cargo test -p zensim --lib --features custom-profiles,corruption-head metric::bake::gradient::tests::floor_recovery_never_detaches_an_active_corruption_companion -- --exact

e34-d12-report mode:
    python3 scripts/rev4_featpot/e34_precision.py {{mode}} --root /mnt/v/output/zensim/e34-prep-2026-10-15/part6

e34-d12-cached-gate model:
    ZENSIM_FORMULA_REV=5 ZENSIM_NEIGHBOUR_EXACT=1 RAYON_NUM_THREADS=1 STEERFIX_PACKET=/mnt/v/output/zensim/e34-prep-2026-10-15/part6/cached-gate-{{model}}/job.json target/debug/deps/zensim-6fc55070ad1b5213 metric::bake::steerfix_packet::engineering_packet --exact --nocapture

e34-d12-edge-checks:
    just e34-d12-companion-regression
    cargo test -p zensim --lib --features custom-profiles,corruption-head gradient::tests
    just e34-steering-build

# D12 existing score/feature parity gates plus the prepared-stride regression.
e34-d12-parity:
    cargo test -p zensim --lib --features custom-profiles metric::bake::
    cargo test -p zensim --features custom-profiles,training --test dense_layout_round_trip --test per_bake_revision --test featcanon_rev5_parity --test cross_tier --test attribution_cross_tier

e34-d12-final-local:
    just e34-d12-gradient-tests
    just e34-d12-parity
    just e34-d12-clippy
    just e34-d12-bench-build


e34-d12-fresh-controls:
    just e34-steering-build
    ZENSIM_FORMULA_REV=5 E34_D12_REFERENCE_JOB=/mnt/v/output/zensim/e34-prep-2026-10-15/part6/reference-job-final.json target/debug/deps/zensim-6fc55070ad1b5213 metric::bake::steerfix_packet::precision::d12_exact_gradient_reference_packet --exact --nocapture
    just e34-d12-cached-gate c
    just e34-d12-cached-gate seed0
    just e34-d12-cached-gate l

e34-d12-paired-probes:
    just e34-d12-timing after
    just e34-d12-report d12-prodqual-before
    just e34-d12-report d12-prodqual-after

e34-d12-build-probes:
    just e34-d12-clippy
    just e34-d12-bench-build

e34-d12-final-feature-checks:
    cargo test -p zensim --lib --features custom-profiles,corruption-head metric::bake::
    cargo check -p zensim --no-default-features
    cargo clippy -p zensim --lib --tests --features custom-profiles,corruption-head -- -D warnings
    cargo fmt -p zensim --check

e34-d12-no-default-check:
    cargo check -p zensim --no-default-features
    cargo fmt -p zensim --check

# E34 review fixes: public loaded-model shape and transform contracts.
e34-fix1-target filter="":
    cargo test --locked -p zensim --features custom-profiles,corruption-head --test exact_gradient_contract {{filter}} -- --nocapture

e34-fix1-checks:
    just e34-fix1-target
    cargo test --locked -p zensim --lib --features custom-profiles,corruption-head metric::bake::
    just e34-d12-parity
    just api-doc-check
    just clippy
    just lint-scripts
    cargo fmt -p zensim --check
    just e34-d12-bench-build

e34-fix1-score-parity:
    python3 scripts/rev4_featpot/e34_precision.py d12-prodqual-after --root /mnt/v/output/zensim/e34-prep-2026-10-15/fix1
    cmp /mnt/v/output/zensim/e34-prep-2026-10-15/part6/after-prodqual.json /mnt/v/output/zensim/e34-prep-2026-10-15/fix1/after-prodqual.json

# E34 launch binaries use the authorized main snapshot; fleet work stays off this host.
e34-launch-build:
    cargo build --locked --release -p zensim-validate --bin zensim_mlp_train --bin bake_dial_refit --bin panel --bin predict_features_with_bake --example inspect_qualified_checkpoint
    cargo build --locked --release -p zensim --example serve_custom_bake --features custom-profiles

# Existing executor smoke owner, explicit E34 campaign (run on an allowed worker).
e34-launch-smoke bundle image jobset inspector mode="bounded" attempt="1":
    python3 scripts/tests/e33_executor_smoke.py --campaign e34 --bundle {{bundle}} --image {{image}} --jobset {{jobset}} --inspector {{inspector}} --mode {{mode}} --attempt {{attempt}}

e34-launch-backup:
    ~/work/zen/scripts/run-heavy --mem 24G --jobs 1 -- rsync -a --no-owner --no-group /mnt/v/output/zensim/e34-launch-2026-10-15/ /mnt/tower/output/zensim/e34-launch-2026-10-15/

# Registered label-free dev96 only; explicit partial mode never releases K early.
e34-launch-pilot partial="--partial":
    ~/work/zen/scripts/run-heavy --mem 8G --jobs 1 -- python3 scripts/rev4_featpot/e33_launch.py lambda-pilot --packet /mnt/v/output/zensim/e34-launch-2026-10-15/packet --packet-sha f21735ca6402126133eff241464f5f2ab1dfbe0c73c0a910b513d1a941358d31 --instrument /mnt/v/output/zensim/e34-launch-2026-10-15/bin/dev96-instrument-83b13003 {{partial}}

# Run in the existing visible harvest tab; status path is supplied explicitly by the operator.
e34-harvest-supervised status:
    E34_STATUS_FILE={{quote(status)}} ~/work/zen/scripts/run-heavy --mem 8G --jobs 1 -- bash scripts/rev4_featpot/e33_harvest_chain.sh /mnt/v/output/zensim/e34-launch-2026-10-15/packet f21735ca6402126133eff241464f5f2ab1dfbe0c73c0a910b513d1a941358d31 /var/tmp/fitv2/e34_queue

e34-harvest-tests:
    PYTHONPATH=scripts:scripts/rev4_featpot:scripts/tests python3 -m unittest scripts.tests.test_e33_launch

# Registered E34 assessment; the committed exposure ledger precedes capture.
e34-eval-capture:
    ~/work/zen/scripts/run-heavy --mem 16G --jobs 1 -- python3 scripts/rev4_featpot/e34_evaluate.py capture

e34-eval-statistics:
    ~/work/zen/scripts/run-heavy --mem 24G --jobs 24 -- /mnt/v/output/zensim/e34-launch-2026-10-15/evaluation/bin/panel-d13 e34 evaluate --input /mnt/v/output/zensim/e34-launch-2026-10-15/evaluation/panel-input.json --output /mnt/v/output/zensim/e34-launch-2026-10-15/evaluation/statistics.json

e34-eval-progress:
    python3 scripts/rev4_featpot/e34_evaluate.py progress

e34-eval-verdict-tests:
    python3 scripts/tests/test_e34_verdict.py

e34-eval-steering:
    ~/work/zen/scripts/run-heavy --mem 16G --jobs 1 -- python3 scripts/rev4_featpot/e34_evaluate.py steering

e34-eval-development:
    ~/work/zen/scripts/run-heavy --mem 16G --jobs 1 -- python3 scripts/rev4_featpot/e34_evaluate.py development

e34-eval-gates:
    ~/work/zen/scripts/run-heavy --mem 16G --jobs 1 -- python3 scripts/rev4_featpot/e34_evaluate.py gates

e34-eval-worker-inventory:
    python3 scripts/rev4_featpot/e34_fleet.py inventory

e34-eval-worker-tests:
    python3 scripts/tests/test_e34_fleet.py

e34-eval-owners-build:
    ~/work/zen/scripts/run-heavy --mem 16G --jobs 4 -- cargo build --locked --release -p zensim-validate --bin panel --bin bake_verdict
    ~/work/zen/scripts/run-heavy --mem 16G --jobs 4 -- cargo test --locked --release -p zensim-validate --bin panel panel_e34::
    ~/work/zen/scripts/run-heavy --mem 16G --jobs 4 -- cargo build --locked --release -p zensim --example serve_custom_bake --features custom-profiles

e34-eval-p3-build:
    RUSTFLAGS='--cfg steerfix_instrument' ~/work/zen/scripts/run-heavy --mem 16G --jobs 4 -- cargo test --locked --release -p zensim --lib --features custom-profiles --no-run --message-format=json-render-diagnostics

e34-eval-p3:
    E34_P3_JOB=/mnt/v/output/zensim/e34-launch-2026-10-15/evaluation/p3-job.json ZENSIM_FORMULA_REV=5 ZENSIM_MAX_TIER=v3 RAYON_NUM_THREADS=1 ~/work/zen/scripts/run-heavy --mem 8G --jobs 1 -- /mnt/v/output/zensim/e34-launch-2026-10-15/evaluation/bin/p3-instrument metric::bake::steerfix_packet::precision::p3::e34_registered_switch_diagnostic --exact --nocapture

e34-eval-runtime-build:
    CARGO_TARGET_DIR=zensim-bench/target ~/work/zen/scripts/run-heavy --mem 16G --jobs 4 -- cargo bench --locked --no-run --message-format=json-render-diagnostics --manifest-path zensim-bench/e34-runtime/Cargo.toml --bench ssim2_speed_bar --features costcmp

e34-eval-check:
    ~/work/zen/scripts/run-heavy --mem 16G --jobs 4 -- cargo clippy --locked -p zensim-validate --bin panel -- -D warnings
    RUSTFLAGS='--cfg steerfix_instrument' ~/work/zen/scripts/run-heavy --mem 16G --jobs 4 -- cargo clippy --locked -p zensim --lib --tests --features custom-profiles -- -D warnings
    CARGO_TARGET_DIR=zensim-bench/target ~/work/zen/scripts/run-heavy --mem 16G --jobs 4 -- cargo clippy --locked --manifest-path zensim-bench/e34-runtime/Cargo.toml --bench ssim2_speed_bar --features costcmp -- -D warnings
    python3 scripts/lint_scripts.py
    python3 -m py_compile scripts/rev4_featpot/e34_evaluate.py scripts/rev4_featpot/e34_fleet.py scripts/rev4_featpot/e34_runtime.py scripts/rev4_featpot/e34_results.py scripts/rev4_featpot/e33_gates.py
    python3 scripts/tests/test_e34_verdict.py
    python3 scripts/tests/test_e34_fleet.py
    python3 scripts/tests/test_e34_results.py

e34-eval-results-tests:
    python3 scripts/tests/test_e34_results.py

e34-eval-pack-check:
    ~/work/zen/scripts/run-heavy --mem 8G --jobs 1 -- /mnt/v/output/zensim/e34-launch-2026-10-15/evaluation/bin/panel-reports e34 pack-check --input /mnt/v/output/zensim/e34-launch-2026-10-15/evaluation/full-models.json --output /mnt/v/output/zensim/e34-launch-2026-10-15/evaluation/pack-checks.json

e34-eval-prepare-reports:
    ~/work/zen/scripts/run-heavy --mem 16G --jobs 1 -- python3 scripts/rev4_featpot/e34_evaluate.py prepare-reports

e34-eval-reports:
    ~/work/zen/scripts/run-heavy --mem 24G --jobs 24 -- /mnt/v/output/zensim/e34-launch-2026-10-15/evaluation/bin/panel-reports e34 reports --input /mnt/v/output/zensim/e34-launch-2026-10-15/evaluation/report-input.json --output /mnt/v/output/zensim/e34-launch-2026-10-15/evaluation/reports.json

e34-eval-penalties:
    python3 scripts/rev4_featpot/e34_results.py penalties

e34-eval-integrity:
    ~/work/zen/scripts/run-heavy --mem 8G --jobs 1 -- python3 scripts/rev4_featpot/e34_evaluate.py integrity

e34-eval-results:
    python3 scripts/rev4_featpot/e34_results.py build
    just e34-eval-render

e34-eval-render:
    python3 scripts/rev4_featpot/e34_results.py render

e34-eval-publication-check:
    python3 scripts/rev4_featpot/e34_results.py verify-publication

e34-eval-archive:
    ~/work/zen/scripts/run-heavy --mem 8G --jobs 1 -- python3 scripts/demos/rev5perf_archive.py /mnt/v/output/zensim/e34-launch-2026-10-15/evaluation /mnt/tower/output/zensim/e34-launch-2026-10-15/evaluation --verify-all --dereference --exclude-operator-locks

e34-eval-runtime arm mode:
    ~/work/zen/scripts/run-heavy --mem 16G --jobs 1 -- python3 scripts/rev4_featpot/e34_runtime.py {{mode}} --arm {{arm}} --binary /mnt/v/output/zensim/e34-launch-2026-10-15/evaluation/bin/costcmp-instrument

e34-eval-runtime-r7900x arm mode:
    ssh -o BatchMode=yes -o ConnectTimeout=10 r7900x '~/work/claudehints/scripts/run-heavy --mem 16G --jobs 1 -- python3 ~/tmp/e34-runtime-codex-20261015/source/scripts/rev4_featpot/e34_runtime.py {{mode}} --arm {{arm}} --root /home/lilith/tmp/e34-runtime-codex-20261015/evaluation --binary /home/lilith/tmp/e34-runtime-codex-20261015/evaluation/bin/costcmp-instrument --analyzer /home/lilith/tmp/e34-runtime-codex-20261015/evaluation/bin/paired-rounds-analyzer'

e34-eval-reports-r7900x:
    ssh -o BatchMode=yes -o ConnectTimeout=10 r7900x '~/work/claudehints/scripts/run-heavy --mem 16G --jobs 20 -- ~/tmp/e34-runtime-codex-20261015/evaluation/bin/panel-reports e34 reports --input /home/lilith/tmp/e34-runtime-codex-20261015/evaluation/report-input-r7900x.json --output /home/lilith/tmp/e34-runtime-codex-20261015/evaluation/reports.json'

# Reproduce the E34 CI failures on a chosen source revision.
cifix-e34-targets:
    cargo test -p zensim -p zensim-validate --no-fail-fast --test featcanon_rev4_contract --test featcanon_tier_parity --test palette_legacy_vectors --test bake_surface -- --nocapture

# The minimal build must not compile accumulators whose only callers are v2.
cifix-e34-minimal:
    cargo clippy -p zensim --no-default-features --lib -- -D warnings
    cargo clippy -p zensim --no-default-features --features oracle --lib -- -D warnings
    cargo test -p zensim --no-default-features --lib -- --nocapture

# Panic messages in these controls are expected refusals, not failures.
cifix-e34-refusals:
    cargo test -p zensim-validate --lib -- strategy_knobs_fail_loud_when_unreachable e34_k_refuses_incomplete_or_incompatible_options --nocapture
    cargo test -p zensim-validate --bin zensim_mlp_train capability_preflight_tests::gpu_refuses_unimplemented_recipes_before_opening_data -- --nocapture
    cargo test -p zensim-validate --test nonneg_distance unrepresentable_leaky_alpha_fails_loud -- --nocapture

# Isolate the real-bake attribution/session contract without filtering a pass.
cifix-e34-attribution:
    cargo test -p zensim-validate --test bake_surface candidate_attribution_matches_served_features_scores_and_reuses_sessions -- --nocapture

# Match CI's backend split: hosted GPU tests execute the CubeCL CPU backend.
cifix-e34-all-features:
    cargo test --workspace --no-fail-fast --all-features --exclude zensim-wasm-tests --exclude zensim-validate --exclude zensim-train-gpu -- --nocapture
    cargo test --no-fail-fast -p zensim-validate -p zensim-train-gpu --features zensim-validate/gpu-cpu,zensim-train-gpu/gpu-cpu -- --nocapture

# Reusable CI polling without waiting for the slower jobs to hide a failure.
cifix-e34-ci-status run:
    gh run view {{run}} -R imazen/zensim --json status,conclusion,jobs --jq '{status,conclusion,jobs:[.jobs[] | {name,status,conclusion,databaseId}]}'

# Label-free synthetic native repair oracle; includes all fixed grids/edge blocks.
steer-exact-tests:
    cargo test -p zensim --lib --features custom-profiles r_exact_native_parity -- --nocapture

steer-exact-check:
    cargo fmt -p zensim --check
    cargo clippy -p zensim --lib --features custom-profiles -- -D warnings
    cargo test -p zensim --lib --features custom-profiles

# Synthetic palette capture admission; no datasets or binaries opened.
palette-pairs-check:
    python3 -m unittest scripts/tests/test_palette_training_pairs.py scripts/tests/test_palette_pool_authority.py

subjective-admission-check:
    python3 -m unittest scripts/tests/test_subjective_training_leg.py scripts/tests/test_shippath2_admission.py scripts/tests/test_rev5_teacher_admission.py scripts/tests/test_e34_admission.py
    cargo test -p zensim-validate --bin zensim_mlp_train subjective_native
    cargo clippy -p zensim-validate --bin zensim_mlp_train -- -D warnings

subjective-rank-check:
    cargo test -p zensim-validate --test subjective_rank
    just subjective-admission-check

subjective-loso-check:
    python3 -m unittest scripts/tests/test_subjective_loso.py scripts/tests/test_palette_training_pairs.py

# Invented-data qualification only; no job executor or corpus discovery.
registered-implementation-check:
    just palette-pairs-check
    just subjective-rank-check
    just subjective-loso-check
    just registered-statistics-unit
    just registered-loso-report-check
    just registered-verdict-check

registered-loso-report-check:
    cargo test --locked -p zensim-validate --bin registered_loso_report
    cargo clippy --locked -p zensim-validate --bin registered_loso_report -- -D warnings

registered-verdict-check:
    python3 -m unittest scripts/tests/test_registered_verdict.py scripts/tests/test_e34_verdict.py scripts/tests/test_steer_verdict.py

registered-statistics-unit:
    cargo test --locked -p zensim-validate --bin panel registered_ -- --test-threads=1
    cargo clippy --locked -p zensim-validate --bin panel --bin registered_owner_oracle -- -D warnings

registered-statistics-build:
    E34_OWNER_ORACLE=1 cargo build --locked -p zensim-validate --bin registered_owner_oracle --bin panel

registered-statistics-parity receipt:
    python3 scripts/tests/registered_statistics_parity.py --owner "${CARGO_TARGET_DIR:-target}/debug/registered_owner_oracle" --adapter "${CARGO_TARGET_DIR:-target}/debug/panel" --receipt '{{receipt}}'

# Canonical STEER/E34 statistics on invented data only; no corpus discovery.
steer-statistics-check:
    cargo test -p zensim-validate --bin panel panel_e34:: -- --nocapture
    cargo clippy -p zensim-validate --bin panel -- -D warnings

steer-verdict-tests:
    python3 -m unittest scripts.tests.test_steer_verdict scripts.tests.test_e34_verdict

steer-statistics-verdict-integration:
    python3 -c 'from scripts.tests.test_steer_verdict import canonical_receipt_test; canonical_receipt_test()'


# Invented canonical evidence only; no model, corpus or fleet discovery.
steer-fix2-check:
    cargo fmt -p zensim-validate --check
    just steer-statistics-check
    just steer-verdict-tests
    python3 -c 'from scripts.tests.test_steer_verdict import canonical_fix2_receipt_test; canonical_fix2_receipt_test()'

# Explicit fresh invented receipts; never discovers model or corpus payloads.
steer-freeze-check:
    cargo fmt -p zensim-validate --check
    just steer-statistics-check
    just steer-verdict-tests
    just steer-s-recipe-tests
    just steer-statistics-verdict-integration
    python3 -c 'from scripts.tests.test_steer_verdict import canonical_fix2_receipt_test, canonical_a7_receipt_test; canonical_fix2_receipt_test(); canonical_a7_receipt_test()'

# Build the bounded control/R engineering payload; no corpus or label opens.
steer-smoke-build:
    cargo fmt -p zensim-validate --check
    cargo build --locked -p zensim-validate --bin zensim_mlp_train --bin bake_dial_refit --bin predict_features_with_bake --bin panel --bin steer_exact_smoke --example inspect_qualified_checkpoint
    cargo clippy --locked -p zensim-validate --bin steer_exact_smoke -- -D warnings

# Reproduce CI's absent TMPDIR without putting compiler scratch under /tmp.
steer-pending-policy-check:
    cargo test --locked -p zensim-validate --bin panel --no-run --message-format=json > "$HOME/tmp/zensim-steer-pending-build.jsonl"
    python3 -c 'import json,os,pathlib,subprocess; rows=[json.loads(l) for l in pathlib.Path.home().joinpath("tmp/zensim-steer-pending-build.jsonl").read_text().splitlines()]; binary=next(r["executable"] for r in rows if r.get("reason")=="compiler-artifact" and r.get("target",{}).get("name")=="panel" and r.get("executable")); env=dict(os.environ); env.pop("TMPDIR",None); subprocess.run([binary,"panel_e34::steer::tests::steer_pending_policy_does_not_open_input","--exact","--nocapture"],env=env,check=True)'

# Fixed smooth hinge checks against the merged, pinned zenpredict dependency.
steer-s-check:
    cargo test --locked -p zensim-validate --lib steer_hinge -- --nocapture
    cargo test --locked -p zensim-validate --bin bake_dial_refit steer_hinge -- --nocapture
    cargo test --locked -p zensim-validate --bin zensim_mlp_train steer_hinge -- --nocapture
    cargo clippy --locked -p zensim-validate --lib --bin zensim_mlp_train --bin bake_dial_refit -- -D warnings

steer-s-reference-check:
    RUSTFLAGS="--cfg steerfix_instrument" cargo test --locked -p zensim --lib --features custom-profiles steer_smooth_ -- --nocapture

steer-s-recipe-tests:
    python3 -m unittest scripts.tests.test_steer_recipe

steer-s-native-check:
    cargo test --locked -p zensim --lib --features custom-profiles steer_smooth_native_tier_stride_parity -- --nocapture

steer-s-final-check:
    cargo fmt -p zensim -p zensim-validate --check
    just steer-s-recipe-tests
    just steer-s-check
    cargo clippy --locked -p zensim --lib --features custom-profiles -- -D warnings
    cargo test --locked -p zensim --lib --features custom-profiles
    cargo test --locked -p zensim-validate --lib compact_rows_train_identically_to_zero_masked_dense
    just steer-s-reference-check

# Prepare only: no scientific assessment, fit or fleet declaration.
steer-freeze2-build:
    cargo fmt -p zensim-validate -p zensim --check
    cargo build --locked -p zensim-validate --bin steer_exact_smoke --bin bake_verdict --bin panel --example inspect_qualified_checkpoint
    cargo clippy --locked -p zensim-validate --bin steer_exact_smoke -- -D warnings
    cargo build --locked -p zensim --features custom-profiles --example serve_custom_bake
    RUSTFLAGS="--cfg steerfix_instrument" cargo test --locked -p zensim --lib --features custom-profiles --no-run --message-format=json-render-diagnostics > "$HOME/tmp/zensim-paper/rev4/steer-freeze2-instrument-build.jsonl"

steer-freeze2-runtime-build:
    cargo bench --manifest-path zensim-bench/e34-runtime/Cargo.toml --bench ssim2_speed_bar --features costcmp --no-run --message-format=json-render-diagnostics > "$HOME/tmp/zensim-paper/rev4/steer-freeze2-runtime-build.jsonl"

# Label-free checkpoint control precedes this study's first target assessment.
addrfix-peer-evaluate binary input output:
    python3 scripts/rev4_featpot/addrfix.py unpacked-control --binary /mnt/v/output/zensim/addrfix-2026-10-16/bin/bake_verdict-final
    {{quote(binary)}} e34 peer-compare --input {{quote(input)}} --output {{quote(output)}}

addrfix-peer-statistics binary input output:
    {{quote(binary)}} e34 peer-compare --input {{quote(input)}} --output {{quote(output)}}

addrfix-final-epoch-control binary:
    python3 scripts/rev4_featpot/addrfix.py unpacked-control --checkpoint last --binary {{quote(binary)}}

addrfix-map-evaluate binary input output:
    {{quote(binary)}} e34 map-cost --input {{quote(input)}} --output {{quote(output)}}

# Actual pinned-worker transport oracle: invented jobs only, no training/store/network.
steer-worker-scheduling-probe image out:
    flock "$HOME/tmp/zen-heavy.lock" ~/work/zen/scripts/run-heavy --mem 2G --jobs 1 -- python3 scripts/tests/steer_worker_scheduling_probe.py --image {{quote(image)}} --out {{quote(out)}}

steer-worker-entrypoint-probe image original guarded out:
    flock "$HOME/tmp/zen-heavy.lock" ~/work/zen/scripts/run-heavy --mem 2G --jobs 1 -- python3 scripts/tests/steer_worker_scheduling_probe.py --image {{quote(image)}} --original-entrypoint {{quote(original)}} --guarded-entrypoint {{quote(guarded)}} --out {{quote(out)}}

# Native worker owns claims/jobs; caller holds shared heavy lock for enrollment.
steer-worker-enroll packet packet_sha host role jobset:
    python3 scripts/rev4_featpot/steer_worker_lifecycle.py --packet {{quote(packet)}} --packet-sha {{quote(packet_sha)}} --host {{quote(host)}} --role {{quote(role)}} --jobset {{quote(jobset)}}

steer-worker-lifecycle-check:
    PYTHONPATH=scripts/rev4_featpot python3 -m unittest discover -s scripts/tests -p test_steer_worker_lifecycle.py -v

steer-worker-watch packet packet_sha:
    python3 scripts/rev4_featpot/steer_worker_lifecycle.py --packet {{quote(packet)}} --packet-sha {{quote(packet_sha)}} --watch

steer-worker-pause packet packet_sha:
    python3 scripts/rev4_featpot/steer_worker_lifecycle.py --packet {{quote(packet)}} --packet-sha {{quote(packet_sha)}} --pause

steer-worker-preserve-scratch packet packet_sha host:
    ~/work/zen/scripts/run-heavy --mem 2G --jobs 1 -- python3 scripts/rev4_featpot/steer_worker_lifecycle.py --packet {{quote(packet)}} --packet-sha {{quote(packet_sha)}} --host {{quote(host)}} --preserve-interrupted-scratch

steer-worker-scheduler-metadata packet packet_sha:
    flock "$HOME/tmp/zen-heavy.lock" ~/work/zen/scripts/run-heavy --mem 2G --jobs 1 -- python3 scripts/rev4_featpot/steer_worker_lifecycle.py --packet {{quote(packet)}} --packet-sha {{quote(packet_sha)}} --scheduler-metadata

# Read cached operational metadata; keep the last native count labelled during harvest.
steer-launch-status root:
    @python3 -c 'import json, pathlib, shutil, sys; \
        launch = pathlib.Path(sys.argv[1]) / "launch"; \
        harvest = json.loads((launch / "HARVEST_STATUS.json").read_text()); \
        native = json.loads((launch / "SCHEDULER_METADATA.json").read_text()); \
        backup = max((launch / "backups").glob("*.json")); \
        print(json.dumps({"harvest": {key: harvest.get(key) for key in ("state", "installed", "updated_at", "errors")}, \
        "installed_last_native_verified": native["installed"], "counts_verified_at": native["time"], \
        "per_host": native["per_host"], "snapshot": (launch / "loan-snapshots/i270/LATEST").read_text().strip(), \
        "backup": backup.name, "backup_state": json.loads(backup.read_text())["state"], \
        "home_free": shutil.disk_usage(pathlib.Path.home()).free}))' {{quote(root)}}

shipgates-gaddr models binary instruments mentor_standard mentor_ladder output:
    python3 scripts/rev4_featpot/shipgates.py gaddr --models {{quote(models)}} --binary {{quote(binary)}} --instruments {{quote(instruments)}} --mentor-standard {{quote(mentor_standard)}} --mentor-ladder {{quote(mentor_ladder)}} --output {{quote(output)}}

# Explicit owner-authorized light recompute; intentionally no shared paper flock.
shipgates-monotonicity original peers models binary features_root commit output:
    ~/work/zen/scripts/run-heavy --mem 4G --jobs 1 -- python3 scripts/rev4_featpot/shipgates_monotonicity.py --original {{quote(original)}} --peers {{quote(peers)}} --models {{quote(models)}} --binary {{quote(binary)}} --features-root {{quote(features_root)}} --build-commit {{quote(commit)}} --output {{quote(output)}}

shipgates-monotonicity-check:
    python3 -B -m unittest scripts/tests/test_shipgates_monotonicity.py

# Frozen direction tier: canonical serial statistics, explicitly no paper lock.
shipgates-direction spec original models stats_source commit output:
    TMPDIR="$HOME/tmp" RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 ~/work/zen/scripts/run-heavy --mem 4G --jobs 1 -- python3 scripts/rev4_featpot/shipgates_monotonicity.py --direction-spec {{quote(spec)}} --original {{quote(original)}} --models {{quote(models)}} --stats-source {{quote(stats_source)}} --build-commit {{quote(commit)}} --output {{quote(output)}}

shipgates-direction-report result resource_log output:
    python3 scripts/rev4_featpot/shipgates_monotonicity.py --format-direction {{quote(result)}} --resource-log {{quote(resource_log)}} --output {{quote(output)}}

shipgates-monotonicity-report result resource_log output:
    python3 scripts/rev4_featpot/shipgates_monotonicity.py --format-result {{quote(result)}} --resource-log {{quote(resource_log)}} --output {{quote(output)}}

shipgates-corruption models binary head inputs pairs bindings output:
    python3 scripts/rev4_featpot/shipgates.py corruption --models {{quote(models)}} --binary {{quote(binary)}} --head {{quote(head)}} --inputs {{quote(inputs)}} --pairs {{quote(pairs)}} --bindings {{quote(bindings)}} --output {{quote(output)}}

shipgates-palette-ci python="python3":
    {{quote(python)}} -B -m unittest scripts/tests/test_palette_training_pairs.py scripts/tests/test_palette_pool_authority.py
    {{quote(python)}} -B -m unittest discover -s scripts/tests -p test_palette_admission.py

shipgates-extractor-build:
    cargo build --release --manifest-path zensim-bench/Cargo.toml --example extract_features_372col --features training,zen-decode


shipgates-jxl-smoke models binary inputs oracle output:
    python3 scripts/rev4_featpot/shipgates.py jxl-smoke --models {{quote(models)}} --binary {{quote(binary)}} --inputs {{quote(inputs)}} --oracle {{quote(oracle)}} --output {{quote(output)}}


shipgates-panel-build:
    cargo build --release --locked -p zensim-validate --bin panel

shipgates-corruption-results inputs audits bindings output:
    python3 scripts/rev4_featpot/shipgates.py corruption-results --inputs {{quote(inputs)}} --audits {{quote(audits)}} --bindings {{quote(bindings)}} --output {{quote(output)}}

shipgates-panel-check:
    cargo test --release --locked -p zensim-validate --bin panel source_series_uses_registered_clusters_and_refuses_unbound_fields

shipgates-jxl-target-phased models binary inputs resume lock heavy output:
    python3 scripts/rev4_featpot/shipgates.py jxl-target --models {{quote(models)}} --binary {{quote(binary)}} --inputs {{quote(inputs)}} --resume-calibration-root {{quote(resume)}} --heavy-lock {{quote(lock)}} --run-heavy {{quote(heavy)}} --output {{quote(output)}}

shipgates-rd-sources inputs tree output commit:
    python3 scripts/rev4_featpot/shipgates.py rd-sources --inputs {{quote(inputs)}} --tree {{quote(tree)}} --output {{quote(output)}} --build-commit {{quote(commit)}}

shipgates-target-report output:
    python3 scripts/v_next/rd_probe_analyze_2026-07-18.py --target-loop {{quote(output)}} --target-only

shipgates-rd-licenses tree output:
    python3 scripts/rev4_featpot/shipgates.py rd-licenses --tree {{quote(tree)}} --output {{quote(output)}}

shipgates-judge-build:
    cargo build --release --locked --manifest-path tools/shipgates-judge/Cargo.toml --bin shipgates_judge

shipgates-jxl-rd models binary inputs oracle lock heavy png8 output:
    python3 scripts/rev4_featpot/shipgates.py jxl-rd --models {{quote(models)}} --binary {{quote(binary)}} --inputs {{quote(inputs)}} --oracle {{quote(oracle)}} --heavy-lock {{quote(lock)}} --run-heavy {{quote(heavy)}} --png8-binary {{quote(png8)}} --output {{quote(output)}}

shipgates-oracle-help binary:
    {{quote(binary)}} -v --help


shipgates-analysis-check:
    python3 -B -m unittest scripts/tests/test_shipgates_rd_analysis.py

shipgates-jxl-rd-resume models binary inputs oracle lock heavy png8 resume output:
    python3 scripts/rev4_featpot/shipgates.py jxl-rd --models {{quote(models)}} --binary {{quote(binary)}} --inputs {{quote(inputs)}} --oracle {{quote(oracle)}} --heavy-lock {{quote(lock)}} --run-heavy {{quote(heavy)}} --resume-encodes-root {{quote(resume)}} --png8-binary {{quote(png8)}} --output {{quote(output)}}

shipgates-codec-cell codec binary oracle design source output cas extra="":
    python3 scripts/rev4_featpot/shipgates_codec_rd.py --codec {{quote(codec)}} --binary {{quote(binary)}} --oracle {{quote(oracle)}} --design {{quote(design)}} --source-id {{quote(source)}} --output {{quote(output)}} --cas-root {{quote(cas)}} {{extra}}

shipgates-head-bindings binary:
    #!/usr/bin/env python3
    import json,os,subprocess
    from pathlib import Path
    root=Path('/mnt/v/output/zensim/shipgates-2026-10-16')
    binary=Path('{{binary}}').resolve()
    rows={}
    for name,record in json.loads((root/'MODELS.json').read_text()).items():
        result=subprocess.check_output([str(binary),'--inspect-corruption-binding',record['path'],'/var/tmp/shippath6/refit/head.zcth'],env=dict(os.environ,ZENSIM_FORMULA_REV='5'))
        rows[name]=json.loads(result)
    with (root/'CORRUPTION_FEATURE_BINDINGS.json').open('x') as out:json.dump(rows,out,indent=2,allow_nan=False)

shipgates-judge-check:
    cargo clippy --release --locked --manifest-path tools/shipgates-judge/Cargo.toml --bin shipgates_judge -- -D warnings

shipgates-source-build:
    cargo build --release --locked --manifest-path tools/shipgates-judge/Cargo.toml --bin shipgates_prepare_source

shipgates-frozen-judge-build manifest:
    cargo build --release --locked --manifest-path {{quote(manifest)}} --bins

shipgates-frozen-judge-check manifest:
    cargo clippy --release --locked --manifest-path {{quote(manifest)}} --bins -- -D warnings

shipgates-source-prepare binary input output:
    {{quote(binary)}} --crop-opaque {{quote(input)}} {{quote(output)}}

shipgates-codec-matrix codec binary oracle models sources lock heavy png8 output extra="":
    python3 scripts/rev4_featpot/shipgates_codec_rd.py matrix --codec {{quote(codec)}} --binary {{quote(binary)}} --oracle {{quote(oracle)}} --models {{quote(models)}} --sources {{quote(sources)}} --heavy-lock {{quote(lock)}} --run-heavy {{quote(heavy)}} --png8-binary {{quote(png8)}} --output {{quote(output)}} {{extra}}

shipgates-matte-oracle input prepared output:
    python3 scripts/rev4_featpot/shipgates_matte_oracle.py --input {{quote(input)}} --prepared {{quote(prepared)}} --output {{quote(output)}}

shipgates-judge-pairs binary pairs output:
    #!/usr/bin/env python3
    import os,subprocess
    env={k:v for k,v in os.environ.items() if not k.startswith(('ZENSIM_','JXL_','ZENJPEG_ZQ_','ZENWEBP_ZQ_','AVIF_ZENSIM_'))}
    env.update(ZENSIM_FORMULA_REV='1',RAYON_NUM_THREADS='1')
    subprocess.run(['{{binary}}','--pairs','{{pairs}}','--out-dir','{{output}}','--ssim2-map-policy','scalar-only'],env=env,check=True)

shipgates-source-crop-check manifest:
    cargo test --release --locked --manifest-path {{quote(manifest)}} --bin shipgates_prepare_source
    cargo clippy --release --locked --manifest-path {{quote(manifest)}} --bin shipgates_prepare_source -- -D warnings
    cargo build --release --locked --manifest-path {{quote(manifest)}} --bin shipgates_prepare_source

shipgates-source-crop binary input output:
    {{quote(binary)}} --crop-opaque {{quote(input)}} {{quote(output)}}

shipgates-jxl-interventions models binary inputs oracle lock heavy output:
    python3 scripts/rev4_featpot/shipgates.py jxl-interventions --models {{quote(models)}} --binary {{quote(binary)}} --inputs {{quote(inputs)}} --oracle {{quote(oracle)}} --heavy-lock {{quote(lock)}} --run-heavy {{quote(heavy)}} --output {{quote(output)}}

shipgates-crop-oracle receipt output:
    python3 scripts/rev4_featpot/shipgates_crop_oracle.py --receipt {{quote(receipt)}} --output {{quote(output)}}

shipgates-judge-queue encodes binary lock heavy output reuse pid prior:
    python3 scripts/rev4_featpot/shipgates_judge_queue.py --encodes {{quote(encodes)}} --binary {{quote(binary)}} --heavy-lock {{quote(lock)}} --run-heavy {{quote(heavy)}} --output {{quote(output)}} --reuse-first {{quote(reuse)}} --encoder-pid {{pid}} --reuse-source-root {{quote(prior)}} --map-oracle scripts/rev4_featpot/shipgates_map_oracle.py

shipgates-map-oracle judges output:
    python3 scripts/rev4_featpot/shipgates_map_oracle.py --judges {{quote(judges)}} --output {{quote(output)}}

shipgates-intervention-judges root oracle judge:
    python3 scripts/rev4_featpot/shipgates_intervention_judges.py --root {{quote(root)}} --oracle {{quote(oracle)}} --judge {{quote(judge)}} --analyzer scripts/v_next/rd_probe_analyze_2026-07-18.py --map-oracle scripts/rev4_featpot/shipgates_map_oracle.py

shipgates-evidence-mirror source destination:
    rsync -a --no-owner --no-group --partial --info=progress2 --outbuf=L {{quote(source)}}/ {{quote(destination)}}/
    python3 scripts/rev4_featpot/shipgates_mirror_verify.py --source {{quote(source)}} --destination {{quote(destination)}}

shipgates-controls-report evidence output markdown:
    python3 scripts/rev4_featpot/shipgates_controls_report.py --evidence {{quote(evidence)}} --output {{quote(output)}} --markdown {{quote(markdown)}}

shipgates-source-admission sources binary output:
    python3 scripts/rev4_featpot/shipgates_source_admission.py --sources {{quote(sources)}} --binary {{quote(binary)}} --oracle scripts/rev4_featpot/shipgates_crop_oracle.py --output {{quote(output)}}

shipgates-palette-prepare sources binary pin output:
    python3 scripts/rev4_featpot/shipgates_palette_prepare.py --sources {{quote(sources)}} --binary {{quote(binary)}} --tool-pin {{quote(pin)}} --pixel-oracle scripts/rev4_featpot/shipgates_crop_oracle.py --source-id 21 --output {{quote(output)}}

shipgates-map-oracle-lossless judges output:
    python3 scripts/rev4_featpot/shipgates_map_oracle.py --judges {{quote(judges)}} --output {{quote(output)}} --lossless-png-expansion

shipgates-format-unit binary source output:
    python3 scripts/rev4_featpot/shipgates_format_unit.py --binary {{quote(binary)}} --source {{quote(source)}} --oracle scripts/rev4_featpot/shipgates_map_oracle.py --output {{quote(output)}}

shipgates-source-admission-bind sources binary output:
    python3 scripts/rev4_featpot/shipgates_source_admission.py --sources {{quote(sources)}} --binary {{quote(binary)}} --oracle scripts/rev4_featpot/shipgates_crop_oracle.py --output {{quote(output)}} --bind-only

import 'benchmarks/mapperf.just'

shipgates-rd-analysis encodes judges sources panel pin output:
    python3 scripts/v_next/rd_probe_analyze_2026-07-18.py --shipgates-rd {{quote(encodes)}} --judge-dir {{quote(judges)}} --sources {{quote(sources)}} --panel {{quote(panel)}} --judge-pin {{quote(pin)}} --output {{quote(output)}}

shipgates-jxl-group-replay models binary inputs oracle png8 design output:
    python3 scripts/rev4_featpot/shipgates.py jxl-rd-source --models {{quote(models)}} --binary {{quote(binary)}} --inputs {{quote(inputs)}} --oracle {{quote(oracle)}} --png8-binary {{quote(png8)}} --design {{quote(design)}} --source-id 0 --quality-index 0 --cell-workers 8 --output {{quote(output)}}

shipgates-jxl-group-compare replay prior output:
    python3 scripts/rev4_featpot/shipgates_group_replay.py --replay {{quote(replay)}} --prior {{quote(prior)}} --output {{quote(output)}}

shipgates-jxl-rd-bounded models binary inputs oracle lock heavy png8 resume output:
    python3 scripts/rev4_featpot/shipgates.py jxl-rd --models {{quote(models)}} --binary {{quote(binary)}} --inputs {{quote(inputs)}} --oracle {{quote(oracle)}} --heavy-lock {{quote(lock)}} --run-heavy {{quote(heavy)}} --png8-binary {{quote(png8)}} --resume-encodes-root {{quote(resume)}} --cell-workers 8 --output {{quote(output)}}

shipgates-admission-check:
    python3 -B -m unittest scripts/tests/test_shipgates_memory_admission.py

# E36 preparation: no trainer or fit invocation.
e36-pixels-build:
    cargo build --locked --release -p zensim --example e34_pixels --features custom-profiles,feature-regime-v2 --target-dir /home/lilith/tmp/e36-pixels-target

e36-native-lock-check repo snapshot:
    cd {{quote(repo)}} && LOCK_SNAP_DIR={{quote(snapshot)}} scripts/ci/lock.sh --check

e36-source-selection root out standard ladder family="/mnt/v/output/zensim/e36-2026-10-16/corpus-metadata/split_map_family_v2.tsv":
    python3 scripts/rev4_featpot/e34_augmentation.py e36-select --root {{quote(root)}} --out {{quote(out)}} --panel {{quote(standard)}} --panel {{quote(ladder)}} --family-v2 {{quote(family)}}

e36-native-smoke root tool pin references name pixels feature_spec:
    ZENSIM_FORMULA_REV=5 ZENSIM_MAX_TIER=v3 python3 scripts/rev4_featpot/e34_encode.py e36-smoke --root {{quote(root)}} --tool {{quote(tool)}} --tool-pin {{quote(pin)}} --references {{quote(references)}} --smoke-name {{quote(name)}} --pixels {{quote(pixels)}} --feature-spec {{quote(feature_spec)}}

e36-native-plan root tool pin sources:
    python3 scripts/rev4_featpot/e34_encode.py e36-plan --root {{quote(root)}} --tool {{quote(tool)}} --tool-pin {{quote(pin)}} --sources {{quote(sources)}}

e36-consensus-check:
    TMPDIR="$HOME/tmp" python3 -m unittest scripts/tests/test_e36_consensus.py scripts/tests/test_e34_admission.py

e36-color-probe job:
    just e36-pixels-build
    /home/lilith/tmp/e36-pixels-target/release/examples/e34_pixels {{quote(job)}}

e36-normalize job:
    just e36-pixels-build
    /home/lilith/tmp/e36-pixels-target/release/examples/e34_pixels {{quote(job)}}

e36-native-main-effects root tool pin sources:
    python3 scripts/rev4_featpot/e34_encode.py e36-plan --root {{quote(root)}} --tool {{quote(tool)}} --tool-pin {{quote(pin)}} --sources {{quote(sources)}} --main-effects

e36-pixel-ingress-check:
    cargo test --locked --release -p zensim --example e34_pixels --features custom-profiles,feature-regime-v2 --target-dir /home/lilith/tmp/e36-pixels-target tests::

e36-native-artifact-audit root djxl name="artifact-audit":
    python3 scripts/rev4_featpot/e34_encode.py e36-audit --root {{quote(root)}} --djxl {{quote(djxl)}} --audit-name {{quote(name)}}

e36-native-dense-data root tool pin references name pixels feature_spec:
    ZENSIM_FORMULA_REV=5 ZENSIM_MAX_TIER=v3 python3 scripts/rev4_featpot/e34_encode.py e36-smoke --root {{quote(root)}} --tool {{quote(tool)}} --tool-pin {{quote(pin)}} --references {{quote(references)}} --smoke-name {{quote(name)}} --pixels {{quote(pixels)}} --feature-spec {{quote(feature_spec)}} --q 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 22 24 26 28 30 32 34 36 38 40 42 44 46 48 50 52 54 56 58 60 62 64 66 68 70 72 74 76 78 80 82 84 86 88 90 92 94 96 98 100

e36-base-exclusions root selection out:
    python3 scripts/rev4_featpot/e34_augmentation.py e36-base-exclusions --root {{quote(root)}} --selection {{quote(selection)}} --out {{quote(out)}}

e36-ingress-clippy:
    cargo clippy --locked --release -p zensim --example e34_pixels --features custom-profiles,feature-regime-v2 --target-dir /home/lilith/tmp/e36-pixels-target -- -D warnings

e36-feature-parity old out spec pixels:
    python3 scripts/rev4_featpot/e34_native_parity.py e36-prepare {{quote(old)}} {{quote(out)}} {{quote(spec)}}
    ZENSIM_FORMULA_REV=5 ZENSIM_MAX_TIER=v3 {{quote(pixels)}} {{quote(out + "/job.json")}}
    python3 scripts/rev4_featpot/e34_native_parity.py e36-check {{quote(out)}}

e36-native-color-candidates root out standard ladder family:
    python3 scripts/rev4_featpot/e34_augmentation.py e36-select --root {{quote(root)}} --out {{quote(out)}} --panel {{quote(standard)}} --panel {{quote(ladder)}} --family-v2 {{quote(family)}} --prepare-color-only

e36-native-color-selection root out standard ladder family probe:
    python3 scripts/rev4_featpot/e34_augmentation.py e36-select --root {{quote(root)}} --out {{quote(out)}} --panel {{quote(standard)}} --panel {{quote(ladder)}} --family-v2 {{quote(family)}} --color-probe {{quote(probe)}}

e36-pilot-feature-only-data root:
    python3 scripts/canonical_corpus/build_ladder_tv_pairs.py e36-export --root {{quote(root)}}

e36-cohort-feature-only-data root run selection:
    python3 scripts/canonical_corpus/build_ladder_tv_pairs.py e36-export --root {{quote(root)}} --runs {{quote(run)}} --out-name cohort-data --cohort-selection {{quote(selection)}}

e36-native-family-closed-selection root out standard ladder family probe quarantine:
    python3 scripts/rev4_featpot/e34_augmentation.py e36-select --root {{quote(root)}} --out {{quote(out)}} --panel {{quote(standard)}} --panel {{quote(ladder)}} --family-v2 {{quote(family)}} --color-probe {{quote(probe)}} --quarantine-origins {{quote(quarantine)}}

e36-native-ingress-final-check:
    just e36-pixel-ingress-check
    just e36-ingress-clippy

e36-native-cohort-audit root djxl references:
    python3 scripts/rev4_featpot/e34_encode.py e36-audit --root {{quote(root)}} --djxl {{quote(djxl)}} --audit-name artifact-audit-final --normalized-references {{quote(references)}}

e36-close-cohort root out standard ladder family probe quarantine:
    python3 scripts/rev4_featpot/e34_augmentation.py e36-close-cohort --root {{quote(root)}} --out {{quote(out)}} --panel {{quote(standard)}} --panel {{quote(ladder)}} --family-v2 {{quote(family)}} --color-probe {{quote(probe)}} --quarantine-origins {{quote(quarantine)}}

e36-cohort-generation-phase root djxl selection:
    just e36-native-dense-data {{quote(root)}} {{quote(root + "/bin/zenmetrics")}} {{quote(root + "/NATIVE_TOOL_PIN.json")}} {{quote(root + "/smoke-references-cohort-32.json")}} smoke-cohort-32 {{quote(root + "/bin/e36-pixels")}} {{quote(root + "/e34-feature-spec.json")}}
    just e36-cohort-feature-only-data {{quote(root)}} smoke-cohort-32 {{quote(selection)}}
    just e36-native-cohort-audit {{quote(root)}} {{quote(djxl)}} {{quote(root + "/opaque-references.jsonl")}}
    just e36-base-exclusions /mnt/v/output/zensim/e34-prep-2026-10-15/part5/p4-inventory-r2 {{quote(selection)}} {{quote(root + "/OPAQUE_TRAIN_SELECTION_BASE_FILTER.json")}}
    just lint-scripts

e36-evidence-manifest root commit:
    python3 scripts/rev4_featpot/e34_artifacts.py e36-manifest --root {{quote(root)}} --build-commit {{quote(commit)}}

e36-archive root mirror:
    python3 scripts/demos/rev5perf_archive.py {{quote(root)}} {{quote(mirror)}} --dereference --exclude-operator-locks

shipgates-scripts-check:
    just shipgates-monotonicity-check
    just shipgates-cell-scheduling-check
    python3 scripts/lint_scripts.py
    python3 scripts/test_lint_scripts.py
    just shipgates-analysis-check
    just shipgates-admission-check
    just shipgates-source-revision-check
    just shipgates-verified-reuse-check

shipgates-jxl-progress encodes judges:
    python3 scripts/rev4_featpot/shipgates_progress.py --encodes {{quote(encodes)}} --judges {{quote(judges)}}

e36-source-archives root commit producer:
    git -C /home/lilith/work/zen/zensim archive --format=tar --output={{quote(root + "/sources/preparation-code.tar")}} {{quote(commit)}}
    git -C /home/lilith/work/zen/zensim archive --format=tar --output={{quote(root + "/sources/cohort-producer-code.tar")}} {{quote(producer)}}
    just e36-evidence-manifest {{quote(root)}} {{quote(commit)}}
    just e36-archive {{quote(root)}} /mnt/tower/output/zensim/e36-2026-10-16

e36-retention-resume root commit mirror:
    just e36-evidence-manifest {{quote(root)}} {{quote(commit)}}
    just e36-archive {{quote(root)}} {{quote(mirror)}}

shipgates-jxl-qualities-replay models binary inputs oracle png8 design output:
    python3 scripts/rev4_featpot/shipgates.py jxl-rd-source --models {{quote(models)}} --binary {{quote(binary)}} --inputs {{quote(inputs)}} --oracle {{quote(oracle)}} --png8-binary {{quote(png8)}} --design {{quote(design)}} --source-id 0 --quality-start 0 --quality-end 4 --cell-workers 8 --output {{quote(output)}}

shipgates-jxl-qualities-compare replay prior output quality:
    python3 scripts/rev4_featpot/shipgates_group_replay.py --replay {{quote(replay)}} --prior {{quote(prior)}} --output {{quote(output)}} --quality-index {{quality}}

shipgates-jxl-rd-grouped models binary inputs oracle lock heavy png8 resume output:
    python3 scripts/rev4_featpot/shipgates.py jxl-rd --models {{quote(models)}} --binary {{quote(binary)}} --inputs {{quote(inputs)}} --oracle {{quote(oracle)}} --heavy-lock {{quote(lock)}} --run-heavy {{quote(heavy)}} --png8-binary {{quote(png8)}} --resume-encodes-root {{quote(resume)}} --cell-workers 8 --qualities-per-job 4 --output {{quote(output)}}

shipgates-jxl-group-recovery evidence prior replay output:
    python3 scripts/rev4_featpot/shipgates_group_recovery.py --evidence {{quote(evidence)}} --prior {{quote(prior)}} --replay {{quote(replay)}} --output {{quote(output)}}

e36-store-check:
    python3 -m unittest scripts/tests/test_e36_store.py

e36-storage-create host root image container output:
    python3 scripts/rev4_featpot/e36_storage_admin.py --host {{quote(host)}} --root {{quote(root)}} --image {{quote(image)}} --container {{quote(container)}} --output {{quote(output)}}

e36-budget-prepare root features initial:
    python3 scripts/rev4_featpot/e36_budget.py prepare --root {{quote(root)}} --features {{quote(features)}} --initial {{quote(initial)}}

e36-budget-normalize root pixels:
    python3 scripts/rev4_featpot/e36_budget.py normalize --root {{quote(root)}} --pixels {{quote(pixels)}}

e36-data-check:
    python3 -m unittest scripts/tests/test_e36_store.py scripts/tests/test_e36_data.py

e36-data-freeze root initial version commit:
    python3 scripts/rev4_featpot/e36_bulk.py freeze --root {{quote(root)}} --initial {{quote(initial)}} --version {{quote(version)}} --commit {{quote(commit)}}

e36-data-declare root runtime phase name:
    python3 scripts/rev4_featpot/e36_bulk.py declare --root {{quote(root)}} --runtime {{quote(runtime)}} --phase {{quote(phase)}} --name {{quote(name)}}

e36-data-run root name slots:
    python3 scripts/rev4_featpot/e36_bulk.py run --root {{quote(root)}} --name {{quote(name)}} --slots {{quote(slots)}}

e36-data-census root name:
    python3 scripts/rev4_featpot/e36_bulk.py census --root {{quote(root)}} --name {{quote(name)}}

shipgates-jxl-rd-source-batches models binary inputs oracle lock heavy png8 resume output:
    python3 scripts/rev4_featpot/shipgates.py jxl-rd --models {{quote(models)}} --binary {{quote(binary)}} --inputs {{quote(inputs)}} --oracle {{quote(oracle)}} --heavy-lock {{quote(lock)}} --run-heavy {{quote(heavy)}} --png8-binary {{quote(png8)}} --resume-encodes-root {{quote(resume)}} --cell-workers 8 --qualities-per-job 21 --output {{quote(output)}}

shipgates-source-revision evidence output commit:
    python3 scripts/rev4_featpot/shipgates_source_revision.py --evidence {{quote(evidence)}} --output {{quote(output)}} --build-commit {{quote(commit)}}

shipgates-source-admission-count sources binary output count:
    python3 scripts/rev4_featpot/shipgates_source_admission.py --sources {{quote(sources)}} --binary {{quote(binary)}} --oracle scripts/rev4_featpot/shipgates_crop_oracle.py --output {{quote(output)}} --expected-sources {{quote(count)}}

shipgates-source-revision-check:
    python3 -B -m unittest scripts/tests/test_shipgates_source_revision.py

shipgates-verified-reuse-snapshot root sources output:
    python3 scripts/rev4_featpot/shipgates_verified_reuse.py --root {{quote(root)}} --sources {{quote(sources)}} --output {{quote(output)}}

shipgates-verified-reuse-check:
    python3 -B -m unittest scripts/tests/test_shipgates_verified_reuse.py

shipgates-verified-reuse-replay snapshot binary oracle output:
    python3 scripts/rev4_featpot/shipgates_verified_reuse.py --replay-snapshot {{quote(snapshot)}} --binary {{quote(binary)}} --oracle-source {{quote(oracle)}} --output {{quote(output)}}

shipgates-judge-source-phase binary pairs output oracle receipt:
    {{quote(binary)}} --pairs {{quote(pairs)}} --out-dir {{quote(output)}} --ssim2-map-policy scalar-only
    python3 {{quote(oracle)}} --judges {{quote(output)}} --output {{quote(receipt)}} --lossless-png-expansion

shipgates-rd-report result output:
    python3 scripts/rev4_featpot/shipgates_rd_report.py --result {{quote(result)}} --output {{quote(output)}}

shipgates-judge-replay old new oracle output:
    python3 scripts/rev4_featpot/shipgates_judge_replay.py --old {{quote(old)}} --new {{quote(new)}} --oracle {{quote(oracle)}} --output {{quote(output)}}

shipgates-progress encodes judges:
    @python3 -B scripts/rev4_featpot/shipgates_progress.py --encodes {{encodes}} --judges {{judges}}

shipgates-cell-scheduling-check:
    python3 -B -m unittest scripts/tests/test_shipgates_cell_scheduling.py

e36-tv-gradient-check target="/home/lilith/tmp/e36-pixels-target":
    E36_TV_PROTOCOL=1 cargo test --locked --release -p zensim-validate --lib --target-dir {{quote(target)}} e36_tv::tests:: -- --nocapture

e36-tv-cli-check:
    E36_TV_PROTOCOL=1 cargo check --locked --release -p zensim-validate --bin zensim_mlp_train --target-dir /home/lilith/tmp/e36-pixels-target

e36-preflight-all root suffix:
    just e36-data-run {{quote(root)}} first-producer-{{suffix}} 1
    just e36-data-census {{quote(root)}} first-producer-{{suffix}}
    just e36-data-run {{quote(root)}} geometry-{{suffix}} 28
    just e36-data-census {{quote(root)}} geometry-{{suffix}}
    just e36-data-run {{quote(root)}} cost-{{suffix}} 28
    just e36-data-census {{quote(root)}} cost-{{suffix}}
    just e36-data-run {{quote(root)}} square-{{suffix}} 28
    just e36-data-census {{quote(root)}} square-{{suffix}}
    just e36-data-run {{quote(root)}} modes-{{suffix}} 28
    just e36-data-census {{quote(root)}} modes-{{suffix}}

e36-tv-cli-build:
    E36_TV_PROTOCOL=1 cargo build --locked --release -p zensim-validate --bin zensim_mlp_train --target-dir /home/lilith/tmp/e36-pixels-target

e36-tv-ingress-fixtures binary layout out:
    python3 scripts/rev4_featpot/e36_admission_fixtures.py --binary {{quote(binary)}} --layout {{quote(layout)}} --out {{quote(out)}}

e36-preflight-resume root suffix:
    just e36-data-run {{quote(root)}} cost-{{suffix}} 28
    just e36-data-census {{quote(root)}} cost-{{suffix}}
    just e36-data-run {{quote(root)}} square-{{suffix}} 28
    just e36-data-census {{quote(root)}} square-{{suffix}}
    just e36-data-run {{quote(root)}} modes-{{suffix}} 28
    just e36-data-census {{quote(root)}} modes-{{suffix}}

e36-budget-amend root features:
    python3 scripts/rev4_featpot/e36_budget.py amend --root {{quote(root)}} --features {{quote(features)}}

e36-base-filter-plan root exclusions inventory out:
    python3 scripts/rev4_featpot/e36_training.py --root {{quote(root)}} --exclusions {{quote(exclusions)}} --inventory {{quote(inventory)}} --out {{quote(out)}}

e36-reference-prefit-plan root initial rank:
    just e36-budget-amend {{quote(root)}} /mnt/v/output/zensim/e34-prep-2026-10-15/part5/p4-inventory-r2/reference-features.jsonl
    just e36-base-filter-plan {{quote(rank)}} {{quote(initial + "/OPAQUE_TRAIN_SELECTION_BASE_FILTER.json")}} /mnt/v/output/zensim/e34-prep-2026-10-15/part5/p4-inventory-r2/audit.hashes.json {{quote(root + "/matched-base-filter-plan.json")}}

e36-measured-stage-cost root out:
    python3 scripts/rev4_featpot/e36_cost.py --root {{quote(root)}} --out {{quote(out)}}

e36-base-filter-retry root initial rank:
    just e36-base-filter-plan {{quote(rank)}} {{quote(initial + "/OPAQUE_TRAIN_SELECTION_BASE_FILTER.json")}} /mnt/v/output/zensim/e34-prep-2026-10-15/part5/p4-inventory-r2/audit.hashes.json {{quote(root + "/matched-base-filter-plan.json")}}

e36-max-area-preflight root:
    just e36-data-run {{quote(root)}} max-area-v4 28
    just e36-data-census {{quote(root)}} max-area-v4

e36-tv-final-nontraining-check root:
    just e36-data-check
    just e36-tv-gradient-check
    just e36-tv-cli-build
    just e36-tv-ingress-fixtures /home/lilith/tmp/e36-pixels-target/release/zensim_mlp_train {{quote(root + "/FEATURE_LAYOUT_BINDING.json")}} {{quote(root + "/ingress-fixtures-v3")}}
    E36_TV_PROTOCOL=1 cargo test --locked --release -p zensim-validate --bin panel --target-dir /home/lilith/tmp/e36-pixels-target panel_e34::e36::tests:: -- --nocapture

# Versioned correction: normalize canonical member prefixes before family masking.
e36-base-filter-v2 root initial rank:
    just e36-base-filter-plan {{quote(rank)}} {{quote(initial + "/OPAQUE_TRAIN_SELECTION_BASE_FILTER.json")}} /mnt/v/output/zensim/e34-prep-2026-10-15/part5/p4-inventory-r2/audit.hashes.json {{quote(root + "/matched-base-filter-plan-v2.json")}}

e36-bulk-final-preparation root initial rank:
    just e36-base-filter-v2 {{quote(root)}} {{quote(initial)}} {{quote(rank)}}
    just e36-measured-stage-cost {{quote(root)}} {{quote(root + "/MEASURED_STAGE_COSTS-v2.json")}}
    python3 scripts/rev4_featpot/e36_bulk.py resources --root {{quote(root)}}

e36-raw-selection-build:
    E36_TV_PROTOCOL=1 cargo build --locked --release -p zensim-validate --bin e36_raw_score --target-dir /home/lilith/tmp/e36-pixels-target

# Immutable controller/source snapshot: trainer integration cannot alter the data run.
e36-generation-v5 root initial commit:
    python3 {{quote(root + "/control-v5/e36_bulk.py")}} freeze --root {{quote(root)}} --initial {{quote(initial)}} --version runtime-v5 --commit {{quote(commit)}} --references {{quote(root + "/references-v2.json")}} --budget {{quote(root + "/source-budget-v2/plan.json")}}
    python3 {{quote(root + "/control-v5/e36_bulk.py")}} declare --root {{quote(root)}} --runtime {{quote(root + "/runtime-v5/runtime.json")}} --phase first-producer --name first-producer-v5
    python3 {{quote(root + "/control-v5/e36_bulk.py")}} run --root {{quote(root)}} --name first-producer-v5 --slots 1
    python3 {{quote(root + "/control-v5/e36_bulk.py")}} census --root {{quote(root)}} --name first-producer-v5
    python3 {{quote(root + "/control-v5/e36_bulk.py")}} prepare-bulk --root {{quote(root)}} --runtime {{quote(root + "/runtime-v5/runtime.json")}} --version v5
    python3 {{quote(root + "/control-v5/e36_bulk.py")}} bulk --root {{quote(root)}}

e36-matrix-semantic-check:
    python3 -m unittest scripts/tests/test_e36_matrix.py

e36-final-integration-check root:
    just e36-data-check
    just e36-matrix-semantic-check
    just e36-tv-gradient-check
    just e36-tv-cli-build
    just e36-tv-ingress-fixtures /home/lilith/tmp/e36-pixels-target/release/zensim_mlp_train {{quote(root + "/FEATURE_LAYOUT_BINDING.json")}} {{quote(root + "/ingress-fixtures-v4")}}
    E36_TV_PROTOCOL=1 cargo test --locked --release -p zensim-validate --bin panel --target-dir /home/lilith/tmp/e36-pixels-target panel_e34::e36::tests:: -- --nocapture
    just e36-raw-selection-build
    E36_TV_PROTOCOL=1 cargo clippy --locked --release -p zensim-validate --lib --bin zensim_mlp_train --bin panel --bin e36_raw_score --target-dir /home/lilith/tmp/e36-pixels-target

e36-generation-v6 root initial commit:
    just e36-final-mode-admission {{quote(root)}}
    python3 {{quote(root + "/control-v6/e36_transport_probe.py")}} --root {{quote(root)}} --out {{quote(root + "/transport-admission-v6.json")}}
    python3 {{quote(root + "/control-v6/e36_bulk.py")}} freeze --root {{quote(root)}} --initial {{quote(initial)}} --version runtime-v6 --commit {{quote(commit)}} --references {{quote(root + "/references-v2.json")}} --budget {{quote(root + "/source-budget-v2/plan.json")}}
    python3 {{quote(root + "/control-v6/e36_bulk.py")}} declare --root {{quote(root)}} --runtime {{quote(root + "/runtime-v6/runtime.json")}} --phase first-producer --name first-producer-v6
    python3 {{quote(root + "/control-v6/e36_bulk.py")}} run --root {{quote(root)}} --name first-producer-v6 --slots 1
    python3 {{quote(root + "/control-v6/e36_bulk.py")}} census --root {{quote(root)}} --name first-producer-v6
    python3 {{quote(root + "/control-v6/e36_bulk.py")}} prepare-bulk --root {{quote(root)}} --runtime {{quote(root + "/runtime-v6/runtime.json")}} --version v6 --plan {{quote(root + "/bulk-plan-v6.json")}}
    python3 {{quote(root + "/control-v6/e36_bulk.py")}} bulk --root {{quote(root)}} --plan {{quote(root + "/bulk-plan-v6.json")}} --clock {{quote(root + "/bulk-clock-v6.json")}} --inherit-clock {{quote(root + "/bulk-clock.json")}}

e36-final-mode-admission root:
    just e36-raw-selection-build
    cp --reflink=auto /home/lilith/tmp/e36-pixels-target/release/zensim_mlp_train {{quote(root + "/bin/e36-tv-admission-v4")}}
    env -u E36_TV_PROTOCOL cargo build --locked --release -p zensim-validate --bin zensim_mlp_train --target-dir /home/lilith/tmp/e36-pixels-target
    cp --reflink=auto /home/lilith/tmp/e36-pixels-target/release/zensim_mlp_train {{quote(root + "/bin/e36-normal-admission-v1")}}
    python3 scripts/rev4_featpot/e36_nontraining_modes.py --root {{quote(root)}} --private {{quote(root + "/bin/e36-tv-admission-v4")}} --normal {{quote(root + "/bin/e36-normal-admission-v1")}} --raw /home/lilith/tmp/e36-pixels-target/release/e36_raw_score --out {{quote(root + "/mode-admission-v1")}}
    E36_TV_PROTOCOL=1 cargo clippy --locked --release -p zensim-validate --lib --bin zensim_mlp_train --bin panel --bin e36_raw_score --target-dir /home/lilith/tmp/e36-pixels-target

e36-native-catalog root name output:
    #!/usr/bin/env bash
    set -euo pipefail
    e36_args=(catalog --manifest {{quote(root + "/runs/" + name + "/manifest.json")}})
    for e36_ledger_path in {{quote(root + "/runs/" + name)}}/ledger*.parquet; do
        [[ -f "$e36_ledger_path" ]] || continue
        e36_args+=(--ledger "$e36_ledger_path")
    done
    {{quote(root + "/bin/zenfleet-ctl")}} "${e36_args[@]}" > {{quote(output)}}

e36-adapter-unit-check:
    python3 -m unittest scripts/tests/test_e36_data.py scripts/tests/test_e36_store.py scripts/tests/test_e36_matrix.py scripts/tests/test_e36_training.py scripts/tests/test_e36_materialize.py scripts/tests/test_e36_selection.py scripts/tests/test_e36_consensus.py scripts/tests/test_e36_selection_guards.py scripts/tests/test_e36_current_panels.py

# Caller acquires heavy.lock + run-heavy separately for EACH recipe below.
# Do not wrap their sequence in a single shared-lock allocation.
e36-final-normal-build root:
    test ! -e {{quote(root + "/bin/e36-normal-final")}}
    just e36-normal-build-cargo /home/lilith/tmp/e36-pixels-target
    cp --reflink=auto /home/lilith/tmp/e36-pixels-target/release/zensim_mlp_train {{quote(root + "/bin/e36-normal-final")}}

e36-private-build-cargo target:
    E36_TV_PROTOCOL=1 cargo build --locked --release -p zensim-validate --bin zensim_mlp_train --bin panel --bin e36_raw_score --bin bake_dial_refit --example inspect_qualified_checkpoint --bin check_holdout_overlap --target-dir {{quote(target)}}

e36-normal-build-cargo target:
    env -u E36_TV_PROTOCOL cargo build --locked --release -p zensim-validate --bin zensim_mlp_train --target-dir {{quote(target)}}

e36-identity-build-cargo target:
    env -u E36_TV_PROTOCOL cargo build --locked --release -p zensim --example serve_custom_bake --features custom-profiles,candidate-profiles --target-dir {{quote(target)}}

e36-final-private-build root:
    test ! -e {{quote(root + "/bin/e36-fit-final")}}
    test ! -e {{quote(root + "/bin/e36-raw-score-final")}}
    just e36-private-build-cargo /home/lilith/tmp/e36-pixels-target
    mkdir -p {{quote(root + "/bin/e36-fit-final")}}
    cp --reflink=auto /home/lilith/tmp/e36-pixels-target/release/zensim_mlp_train /home/lilith/tmp/e36-pixels-target/release/panel /home/lilith/tmp/e36-pixels-target/release/bake_dial_refit /home/lilith/tmp/e36-pixels-target/release/examples/inspect_qualified_checkpoint /home/lilith/tmp/e36-pixels-target/release/check_holdout_overlap {{quote(root + "/bin/e36-fit-final/")}}
    cp --reflink=auto /home/lilith/tmp/e36-pixels-target/release/e36_raw_score {{quote(root + "/bin/e36-raw-score-final")}}

e36-final-identity-build root:
    test ! -e {{quote(root + "/bin/e36-fit-final/serve_custom_bake")}}
    just e36-identity-build-cargo /home/lilith/tmp/e36-pixels-target
    cp --reflink=auto /home/lilith/tmp/e36-pixels-target/release/examples/serve_custom_bake {{quote(root + "/bin/e36-fit-final/serve_custom_bake")}}

e36-final-pool root out commit:
    #!/usr/bin/env bash
    set -euo pipefail
    mapfile -t e36_selected_runs < <(python3 scripts/rev4_featpot/e36_full_choice.py runs {{quote(root)}})
    (( ${#e36_selected_runs[@]} > 0 ))
    e36_selected_args=()
    for e36_selected_run in "${e36_selected_runs[@]}"; do e36_selected_args+=(--run "$e36_selected_run"); done
    python3 scripts/rev4_featpot/e36_materialize.py --root {{quote(root)}} "${e36_selected_args[@]}" --out {{quote(out)}} --commit {{quote(commit)}}

e36-final-ingress root:
    just e36-tv-ingress-fixtures {{quote(root + "/bin/e36-fit-final/zensim_mlp_train")}} {{quote(root + "/FEATURE_LAYOUT_BINDING.json")}} {{quote(root + "/ingress-fixtures-final")}}
    python3 scripts/rev4_featpot/e36_nontraining_modes.py --root {{quote(root)}} --private {{quote(root + "/bin/e36-fit-final/zensim_mlp_train")}} --normal {{quote(root + "/bin/e36-normal-final")}} --raw {{quote(root + "/bin/e36-raw-score-final")}} --fixtures {{quote(root + "/ingress-fixtures-final")}} --out {{quote(root + "/mode-admission-final")}}

e36-final-actual-pool root pools binaries="bin/e36-fit-final":
    python3 scripts/rev4_featpot/e36_actual_pool_admit.py --binary {{quote(root + "/" + binaries + "/zensim_mlp_train")}} --pool {{quote(pools + "/fit")}} --out {{quote(root + "/actual-pool-admission-final")}}

e36-final-data-packet root pools original out commit binaries="bin/e36-fit-final":
    python3 scripts/rev4_featpot/e36_fit_packet.py data --root {{quote(root)}} --pools {{quote(pools)}} --base {{quote(original + "/e33-fit-data.tar.gz")}} --trainer {{quote(root + "/" + binaries + "/zensim_mlp_train")}} --out {{quote(out)}} --owner {{quote(root + "/fit-native-owner-v2")}} --commit {{quote(commit)}}

e36-final-program-packet root pools original packet commit source_version="program-source-v1" binaries="bin/e36-fit-final" owner_version="fit-native-owner-v2":
    python3 scripts/rev4_featpot/e36_fit_packet.py program --root {{quote(root)}} --pools {{quote(pools)}} --original {{quote(original)}} --packet {{quote(packet)}} --source {{quote(root + "/" + source_version)}} --binaries {{quote(root + "/" + binaries)}} --owner {{quote(root + "/" + owner_version)}} --commit {{quote(commit)}}

e36-final-program-admit packet:
    python3 scripts/rev4_featpot/e36_program_admit.py --program {{quote(packet + "/program.tar.gz")}} --out {{quote(packet + "/program-admission")}}

e36-final-declare root pools original packet owner_version="fit-native-owner-v2":
    python3 scripts/rev4_featpot/e36_fit_packet.py specs --root {{quote(root)}} --pools {{quote(pools)}} --packet {{quote(packet)}} --original {{quote(original)}} --program {{quote(packet + "/program.tar.gz")}} --ctl {{quote(root + "/bin/zenfleet-ctl")}} --owner {{quote(root + "/" + owner_version)}}

# Pure invented fixtures and admission checks; these never run the optimizer.
e36-final-panel-test target="/home/lilith/tmp/e36-pixels-target":
    E36_TV_PROTOCOL=1 cargo test --locked --release -p zensim-validate --bin panel --target-dir {{quote(target)}} panel_e34::e36::tests:: -- --nocapture

e36-final-private-clippy target="/home/lilith/tmp/e36-pixels-target":
    E36_TV_PROTOCOL=1 cargo clippy --locked --release -p zensim-validate --lib --bin zensim_mlp_train --bin panel --bin e36_raw_score --target-dir {{quote(target)}} -- -D warnings

e36-final-assessment-metadata root:
    python3 -c 'from pathlib import Path; import sys; sys.path.insert(0,"scripts/rev4_featpot"); from e36_training import assessment_plan; import json; r=Path(sys.argv[1]); d=json.loads((r/"matched-base-filter-plan-v2.json").read_text()); assessment_plan(Path(d["root"]),r/"assessment-inputs-preflight.json")' {{quote(root)}}

# Read-only admission: a running compute scope excludes the optional host.
e36-idle-probe host:
    ssh -o BatchMode=yes -o ConnectTimeout=5 {{quote(host)}} 'date -u +%FT%TZ; uptime; free -h; systemctl --user list-units --type=scope --state=running --no-pager'

# Native reference-only screen; the existing threshold and hash era stay fixed.
e36-final-current-panel-audit root:
    python3 scripts/rev4_featpot/e36_current_panels.py --root {{quote(root)}} --manifest /mnt/v/output/zensim/shipgates-2026-10-16/rd-inputs-corrected27/SOURCE_MANIFEST.json --prepared /mnt/v/output/zensim/shipgates-2026-10-16/rd-source-admission-corrected27 --binary {{quote(root + "/bin/e36-fit-final/check_holdout_overlap")}} --out {{quote(root + "/current-panel-audit-final")}}

e36-current-panel-input-files root out:
    python3 scripts/rev4_featpot/e36_current_panels.py --root {{quote(root)}} --manifest /mnt/v/output/zensim/shipgates-2026-10-16/rd-inputs-corrected27/SOURCE_MANIFEST.json --prepared /mnt/v/output/zensim/shipgates-2026-10-16/rd-source-admission-corrected27 --out {{quote(out)}} --input-files-only

# Read-only status for a coordinator-requested local allocation window.
e36-allocation-status lock scope:
    @date -u +%FT%TZ
    @lslocks --json --output PID,PATH,MODE | python3 -c 'import json,sys; d=json.load(sys.stdin); print(json.dumps([r for r in d.get("locks",[]) if r.get("path")==sys.argv[1]]))' {{quote(lock)}}
    @systemctl --user show {{quote(scope)}} -p ActiveState -p FreezerState

# Export a committed source tree for a nontraining build, preserving prior versions.
e36-build-source root commit version:
    test ! -e {{quote(root + "/" + version + ".tar")}}
    nice -n19 ionice -c3 git -C /home/lilith/work/zen/zensim archive --format=tar --output={{quote(root + "/" + version + ".tar")}} {{quote(commit)}}
    python3 -c 'import hashlib,json,socket,sys; from pathlib import Path; p=Path(sys.argv[1]); a=Path(str(p)+".tar"); h=hashlib.sha256(); f=a.open("rb"); [h.update(b) for b in iter(lambda:f.read(1048576),b"")]; f.close(); r=dict(source_commit=sys.argv[2],archive_sha256=h.hexdigest(),archive_bytes=a.stat().st_size,source_host=socket.gethostname(),fits=0); o=Path(str(p)+".json").open("x"); json.dump(r,o,indent=2); o.write("\n"); o.close(); print(json.dumps(r))' {{quote(root + "/" + version)}} {{quote(commit)}}

e36-budget-amendment-check:
    python3 -m unittest scripts/tests/test_e36_pause_budget.py scripts/tests/test_e36_matrix.py scripts/tests/test_e36_live_completion.py

e36-native-phase-unit-check:
    python3 -m unittest scripts/tests/test_e36_phase_worker.py scripts/tests/test_e36_phase_handoff.py scripts/tests/test_e36_phase_start.py

e36-budget-live-timer-check out:
    python3 scripts/tests/e36_live_budget_fixture.py --out {{quote(out)}}

e36-budget-resume root scope supervisor controller auto_thaw_shell:
    python3 {{quote(root + "/control-budget-v1/e36_pause_budget.py")}} --root {{quote(root)}} --scope {{quote(scope)}} --supervisor {{supervisor}} --controller {{controller}} --auto-thaw-shell {{auto_thaw_shell}} --registration benchmarks/e36_registration_draft_2026-10-16.md

e36-budget-recover-preapplication root scope supervisor controller auto_thaw_shell:
    python3 {{quote(root + "/control-budget-v2/e36_pause_budget.py")}} --root {{quote(root)}} --scope {{quote(scope)}} --supervisor {{supervisor}} --controller {{controller}} --auto-thaw-shell {{auto_thaw_shell}} --registration benchmarks/e36_registration_draft_2026-10-16.md --recover-preapplication {{quote(root + "/budget-amendment-v1")}}

e36-tower-phase-prepare root out image:
    python3 scripts/rev4_featpot/e36_tower_prepare.py --root {{quote(root)}} --out {{quote(out)}} --image {{quote(image)}}

e36-tower-phase-stage preparation host container name:
    python3 scripts/rev4_featpot/e36_tower_prepare.py --preparation {{quote(preparation)}} --host {{quote(host)}} --container {{quote(container)}} --stage-name {{quote(name)}}

e36-tower-container-create preparation stage name phase out:
    python3 scripts/rev4_featpot/e36_tower_container.py --preparation {{quote(preparation)}} --stage {{quote(stage)}} --name {{quote(name)}} --phase {{quote(phase)}} --out {{quote(out)}}

e36-tower-first-producer-start root binding out:
    python3 scripts/rev4_featpot/e36_phase_start.py first-producer --root {{quote(root)}} --binding {{quote(binding)}} --out {{quote(out)}}

e36-tower-phase-harvest binding out:
    python3 scripts/rev4_featpot/e36_phase_handoff.py harvest --binding {{quote(binding)}} --out {{quote(out)}}

e36-tower-phase-parity root harvest out:
    python3 scripts/rev4_featpot/e36_phase_handoff.py parity --root {{quote(root)}} --harvest {{quote(harvest)}} --out {{quote(out)}}

e36-resume-schedule-prepare root worker out:
    python3 scripts/rev4_featpot/e36_resume_schedule.py prepare --root {{quote(root)}} --worker {{quote(worker)}} --out {{quote(out)}}

e36-resume-schedule root amendment:
    python3 scripts/rev4_featpot/e36_resume_schedule.py run --root {{quote(root)}} --amendment {{quote(amendment)}}

e36-finish-projection root out:
    nice -n19 ionice -c3 python3 scripts/rev4_featpot/e36_finish_projection.py --root {{quote(root)}} --out {{quote(out)}}

e36-stream-census-check:
    python3 -m unittest scripts/tests/test_e36_census_stream.py scripts/tests/test_e36_census_install.py scripts/tests/test_e36_native_ledger_admit.py

# Coordinator resource-only live operation; refuses another scope or repeated application.
e36-budget-extend-live:
    python3 scripts/rev4_featpot/e36_budget_apply.py

e36-resource-census-check:
    python3 -m unittest scripts/tests/test_e36_budget_extend.py scripts/tests/test_e36_pause_budget.py scripts/tests/test_e36_census_stream.py scripts/tests/test_e36_census_install.py scripts/tests/test_e36_native_ledger_admit.py

e36-observed-wide-costs root output:
    nice -n19 ionice -c3 python3 scripts/rev4_featpot/e36_finish_projection.py --root {{quote(root)}} --out {{quote(output)}} --sample-wide-costs

e36-observed-finish-projection root costs output:
    nice -n19 ionice -c3 python3 scripts/rev4_featpot/e36_finish_projection.py --root {{quote(root)}} --costs {{quote(costs)}} --out {{quote(output)}}

e36-svt-finish-projection root costs output:
    nice -n19 ionice -c3 python3 scripts/rev4_featpot/e36_finish_projection.py --root {{quote(root)}} --costs {{quote(costs)}} --out {{quote(output)}} --exclude-rav1e

e36-disk-finish-projection root compute output:
    nice -n19 ionice -c3 python3 scripts/rev4_featpot/e36_disk_projection.py --root {{quote(root)}} --compute {{quote(compute)}} --out {{quote(output)}}

e36-lossless-archive-probe root bundle output:
    ~/work/zen/scripts/run-heavy --mem 2G --jobs 1 -- python3 scripts/rev4_featpot/e36_archive_probe.py --root {{quote(root)}} --bundle {{quote(bundle)}} --out {{quote(output)}}

e36-archive-resource-check:
    TMPDIR="$HOME/tmp" python3 -m unittest scripts/tests/test_e36_archive_store.py scripts/tests/test_e36_resource_handoff.py scripts/tests/test_e36_budget_extend.py scripts/tests/test_e36_generation_resume.py scripts/tests/test_e36_owner_amendment.py scripts/tests/test_e36_disk_policy.py

e36-cache-archive-check:
    TMPDIR="$HOME/tmp" python3 -m unittest scripts/tests/test_e36_cache_archive.py scripts/tests/test_e36_archive_assignments.py scripts/tests/test_e36_archive_store.py scripts/tests/test_e36_resource_handoff.py

e36-native-continuation-close root:
    python3 scripts/rev4_featpot/e36_close_continuation.py --root {{quote(root)}}

e36-svt-finalize-choice root:
    python3 -c 'import sys; from pathlib import Path; sys.path.insert(0,"scripts/rev4_featpot"); from e36_svt_matrix import finalize_choice; finalize_choice(Path(sys.argv[1]))' {{quote(root)}}

e36-owner-amendment-check:
    TMPDIR="$HOME/tmp" python3 -m unittest scripts/tests/test_e36_owner_amendment.py scripts/tests/test_e36_disk_policy.py scripts/tests/test_e36_census_stream.py scripts/tests/test_e36_matrix.py scripts/tests/test_e36_materialize.py

e36-svt-matrix-check root:
    python3 -c 'import sys; from pathlib import Path; sys.path.insert(0,"scripts/rev4_featpot"); from e36_svt_matrix import configuration; configuration(Path(sys.argv[1])); print("exact owner matrix and original budget chain admitted; fits=0")' {{quote(root)}}

e36-artifact-location-freeze root:
    python3 -c 'import sys; from pathlib import Path; sys.path.insert(0,"scripts/rev4_featpot"); from e36_artifact_location import freeze; r=Path(sys.argv[1]); freeze(r,r/"ARTIFACT_LOCATION_ADMISSION.json")' {{quote(root)}}

e36-pack-archive-check:
    TMPDIR="$HOME/tmp" python3 -m unittest scripts/tests/test_e36_closed_archive.py scripts/tests/test_e36_archive_windows.py scripts/tests/test_e36_pack_store.py scripts/tests/test_e36_pack_recovery.py scripts/tests/test_e36_cache_archive.py scripts/tests/test_e36_archive_assignments.py scripts/tests/test_e36_archive_store.py scripts/tests/test_e36_resource_handoff.py

e36-pack-finish-projection root compute output:
    nice -n19 ionice -c3 python3 scripts/rev4_featpot/e36_pack_projection.py --root {{quote(root)}} --compute {{quote(compute)}} --out {{quote(output)}}

e36-pack-resource-admit root:
    python3 scripts/rev4_featpot/e36_pack_binding.py --root {{quote(root)}}

e36-pack-integration-check:
    TMPDIR="$HOME/tmp" python3 -m unittest scripts/tests/test_e36_phase_terminal.py scripts/tests/test_e36_matrix.py scripts/tests/test_e36_generation_resume.py scripts/tests/test_e36_pack_recovery.py scripts/tests/test_e36_pack_store.py scripts/tests/test_e36_packet_proofs.py scripts/tests/test_e36_pool_finalize.py scripts/tests/test_e36_memory_smoke.py scripts/tests/test_e36_fit_resources.py

e36-largest-source-amendment root projection commit:
    nice -n19 ionice -c3 python3 scripts/rev4_featpot/e36_source_cut.py --root {{quote(root)}} --projection {{quote(projection)}} --commit {{quote(commit)}}

e36-fresh-continuation operation root:
    python3 scripts/rev4_featpot/e36_fresh_continuation.py {{quote(operation)}} --root {{quote(root)}}

e36-source-archive-repair root:
    python3 scripts/rev4_featpot/e36_source_cut.py --root {{quote(root)}} --repair-archive

e36-metadata-archive-check:
    TMPDIR="$HOME/tmp" python3 -m unittest scripts/tests/test_e36_metadata_archive.py

e36-metadata-plan root out commit:
    python3 scripts/rev4_featpot/e36_metadata_archive.py plan --root {{quote(root)}} --out {{quote(out)}} --commit {{quote(commit)}}

e36-metadata-archive plan storage out:
    python3 scripts/rev4_featpot/e36_metadata_archive.py archive --plan {{quote(plan)}} --storage {{quote(storage)}} --out {{quote(out)}} --lock "$HOME/tmp/zensim-paper/rev4/heavy.lock"

e36-optional-terminal root out log audit:
    python3 scripts/rev4_featpot/e36_phase_terminal.py --root {{quote(root)}} --out {{quote(out)}} --log {{quote(log)}} --audit {{quote(audit)}}

# Storage windows own the paper lock; no inherited flock around this scope.
e36-raw-drain-run root out resources:
    ~/work/zen/scripts/run-heavy --mem 8G --jobs 2 -- python3 {{quote(resources + "/e36_archive_terminal.py")}} run --root {{quote(root)}} --out {{quote(out)}} --resources {{quote(resources)}}

e36-raw-drain-close root out log:
    python3 scripts/rev4_featpot/e36_archive_terminal.py close --root {{quote(root)}} --out {{quote(out)}} --log {{quote(log)}}

e36-metadata-storage root out commit:
    python3 scripts/rev4_featpot/e36_metadata_storage.py --root {{quote(root)}} --out {{quote(out)}} --commit {{quote(commit)}}

e36-optional-finish-projection root costs out:
    nice -n19 ionice -c3 python3 scripts/rev4_featpot/e36_finish_projection.py --root {{quote(root)}} --costs {{quote(costs)}} --out {{quote(out)}} --phase additions

e36-control-freeze out commit:
    python3 scripts/rev4_featpot/e36_freeze_control.py --out {{quote(out)}} --commit {{quote(commit)}}

e36-largest-support-preview root out:
    nice -n19 ionice -c3 python3 scripts/rev4_featpot/e36_largest_support_preview.py --root {{quote(root)}} --out {{quote(out)}}

# Feature computation may overlap storage; packet creation refuses pending archive proof.
e36-prepare-pool root out commit:
    python3 scripts/rev4_featpot/e36_materialize.py --root {{quote(root)}} --run svt-fallback-v1 --out {{quote(out)}} --commit {{quote(commit)}} --prepare-while-archiving

e36-finalize-prepared-pool root prepared out commit:
    python3 scripts/rev4_featpot/e36_pool_finalize.py --root {{quote(root)}} --prepared {{quote(prepared)}} --out {{quote(out)}} --commit {{quote(commit)}}

e36-memory-smoke-prepare root packet out owner commit:
    python3 scripts/rev4_featpot/e36_memory_smoke.py prepare --root {{quote(root)}} --packet {{quote(packet)}} --out {{quote(out)}} --owner {{quote(owner)}} --commit {{quote(commit)}}

e36-fit-resource-plan root packet resource out verification:
    python3 scripts/rev4_featpot/e36_fit_resources.py --root {{quote(root)}} --packet {{quote(packet)}} --resource {{quote(resource)}} --out {{quote(out)}} --verification {{quote(verification)}}

e36-pool-progress pool:
    nice -n19 ionice -c3 python3 scripts/rev4_featpot/e36_pool_progress.py {{quote(pool)}}

e36-memory-smoke-verify root out owner ledger blobs:
    python3 scripts/rev4_featpot/e36_memory_smoke.py verify --root {{quote(root)}} --out {{quote(out)}} --owner {{quote(owner)}} --ledger {{quote(ledger)}} --blobs {{quote(blobs)}}

e36-zero-support-diagnosis root out:
    nice -n19 ionice -c3 python3 scripts/rev4_featpot/e36_zero_support_diagnosis.py --root {{quote(root)}} --out {{quote(out)}}

e36-full-choice-amend root:
    python3 scripts/rev4_featpot/e36_full_choice.py amend {{quote(root)}}

e36-smoke-launch-prepare out:
    python3 scripts/rev4_featpot/e36_smoke_launch.py prepare {{quote(out)}}

e36-smoke-launch-stage out:
    python3 scripts/rev4_featpot/e36_smoke_launch.py stage {{quote(out)}}

e36-full-choice-check:
    python3 -m unittest scripts.tests.test_e36_full_choice scripts.tests.test_e36_matrix scripts.tests.test_e36_packet_proofs scripts.tests.test_e36_memory_smoke

e36-smoke-launch-run out:
    python3 -u scripts/rev4_featpot/e36_smoke_launch.py run {{quote(out)}}

e36-metadata-frozen-plan root resources out commit:
    python3 {{quote(resources + "/e36_metadata_archive.py")}} plan --root {{quote(root)}} --out {{quote(out)}} --commit {{quote(commit)}}

e36-metadata-frozen-archive resources plan storage out:
    python3 {{quote(resources + "/e36_metadata_archive.py")}} archive --plan {{quote(plan)}} --storage {{quote(storage)}} --out {{quote(out)}} --lock "$HOME/tmp/zensim-paper/rev4/heavy.lock"

e36-ingress-freeze root target:
    python3 scripts/rev4_featpot/e36_ingress_integration.py freeze --root {{quote(root)}} --target {{quote(target)}}

e36-ingress-probe root pool out:
    python3 scripts/rev4_featpot/e36_ingress_integration.py probe --root {{quote(root)}} --pool {{quote(pool)}} --out {{quote(out)}}

# Scalar-only native G-TARGET validation, using the existing instrument and reporter.
gtarget-scalar-check config:
    cargo clippy --manifest-path tools/shipgates-target/Cargo.toml --config {{quote(config)}} -- -D warnings

gtarget-scalar-build config:
    cargo build --release --manifest-path tools/shipgates-target/Cargo.toml --config {{quote(config)}}

gtarget-scalar-tests:
    TMPDIR="$HOME/tmp" python3 -m unittest discover -s scripts/tests -p 'test_reachable_target_analysis.py'


gtarget-scalar-run binary action codec encoder inputs bake calibration output oracle:
    {{quote(binary)}} {{quote(action)}} {{quote(codec)}} {{quote(encoder)}} {{quote(inputs)}} {{quote(bake)}} {{quote(calibration)}} {{quote(output)}} {{quote(oracle)}}

gtarget-scalar-panel models binary encoders inputs lock heavy output proof phase='all':
    python3 scripts/rev4_featpot/shipgates.py scalar-target --models {{quote(models)}} --binary {{quote(binary)}} --encoders {{quote(encoders)}} --inputs {{quote(inputs)}} --heavy-lock {{quote(lock)}} --run-heavy {{quote(heavy)}} --output {{quote(output)}} --svt-selector-proof {{quote(proof)}} --phase {{quote(phase)}}

gtarget-archive-check:
    TMPDIR="$HOME/tmp" python3 -m unittest scripts/tests/test_shipgates_mirror_verify.py

gtarget-archive-verify source destination:
    python3 scripts/rev4_featpot/shipgates_mirror_verify.py --source {{quote(source)}} --destination {{quote(destination)}} --complete-marker scalar-v1/COMPLETE.json

gtarget-scalar-report reports output proof:
    python3 scripts/rev4_featpot/shipgates.py scalar-target-report --reports {{quote(reports)}} --output {{quote(output)}} --svt-selector-proof {{quote(proof)}}

gtarget-svt-selector encoder inputs previous oracle output:
    python3 scripts/rev4_featpot/shipgates.py svt-selector-check --encoder {{quote(encoder)}} --inputs {{quote(inputs)}} --previous {{quote(previous)}} --oracle {{quote(oracle)}} --output {{quote(output)}}

# Matched peer controls reuse the native scalar dispatcher and statistical owner.
gtarget-peer mode root binary output judge plan lock heavy:
    python3 scripts/rev4_featpot/shipgates_peer_target.py {{quote(mode)}} --root {{quote(root)}} --binary {{quote(binary)}} --output {{quote(output)}} --judge {{quote(judge)}} --plan {{quote(plan)}} --lock {{quote(lock)}} --heavy {{quote(heavy)}}

gtarget-peer-clippy config:
    cargo clippy --locked --manifest-path tools/shipgates-target/Cargo.toml --config {{quote(config)}} -- -D warnings

gtarget-peer-build config:
    cargo build --release --locked --manifest-path tools/shipgates-target/Cargo.toml --config {{quote(config)}}

gtarget-peer-tests:
    python3 -m unittest scripts.tests.test_shipgates_peer_target scripts.tests.test_reachable_target_analysis scripts.tests.test_shipgates_mirror_verify

gtarget-controller-tests config:
    cargo test --release --locked --manifest-path tools/shipgates-target/Cargo.toml --config {{quote(config)}} --test controller

gtarget-peer-tests-isolated venv:
    python3 -m venv {{quote(venv)}}
    {{quote(venv + "/bin/python")}} -m pip install --quiet 'pillow>=10'
    TMPDIR="$HOME/tmp" {{quote(venv + "/bin/python")}} -B -m unittest scripts.tests.test_shipgates_peer_target scripts.tests.test_reachable_target_analysis scripts.tests.test_shipgates_mirror_verify

# R8/R10 only reinterpret retained statistics/scores; no encoder/scorer entry.
gtarget-approved-amendments root inputs audits plan output:
    python3 scripts/rev4_featpot/shipgates_peer_target.py amendments --root {{quote(root)}} --inputs {{quote(inputs)}} --audits {{quote(audits)}} --plan {{quote(plan)}} --output {{quote(output)}}

# Frozen human-panel transport; caller supplies run-heavy and the shared lock.
gauntlet-frozen mode output:
    TMPDIR="$HOME/tmp" python3 scripts/v_next/gauntlet_frozen.py {{quote(mode)}} --out {{quote(output)}}

gauntlet-live-images source output:
    TMPDIR="$HOME/tmp" python3 scripts/v_next/gauntlet_frozen.py live-images --source {{quote(source)}} --out {{quote(output)}}

gauntlet-live-trace source output:
    TMPDIR="$HOME/tmp" python3 scripts/v_next/gauntlet_frozen.py live-trace --source {{quote(source)}} --out {{quote(output)}}

gauntlet-live-matrix source output:
    TMPDIR="$HOME/tmp" python3 scripts/v_next/gauntlet_frozen.py live-matrix --source {{quote(source)}} --out {{quote(output)}}

gauntlet-live-report source output:
    TMPDIR="$HOME/tmp" python3 scripts/v_next/gauntlet_frozen.py live-report --source {{quote(source)}} --out {{quote(output)}}

gauntlet-move-intermediates source destination:
    TMPDIR="$HOME/tmp" python3 scripts/v_next/gauntlet_frozen.py move-intermediates --source {{quote(source)}} --out {{quote(destination)}}

# Frozen named profiles use fresh Rev1 extraction, independently of Rev5 caches.
gauntlet-profile-audit binary pairs bake output:
    TMPDIR="$HOME/tmp" ZENSIM_FORMULA_REV=1 RAYON_NUM_THREADS=1 {{quote(binary)}} --corpus pairs --path {{quote(pairs)}} --out {{quote(output + '.csv')}} --audit-jsonl {{quote(output + '.jsonl')}} --audit-bake {{quote(bake)}}

gauntlet-legacy-assess source output:
    TMPDIR="$HOME/tmp" python3 scripts/v_next/gauntlet_frozen.py legacy-assess --source {{quote(source)}} --out {{quote(output)}}

gauntlet-build target:
    cargo build --release --locked -p zensim-validate --bin panel --target-dir {{quote(target)}}

gauntlet-stat-tests target:
    TMPDIR="$HOME/tmp" cargo test --release --locked -p zensim-validate --bin panel panel_rank_interval:: --target-dir {{quote(target)}}

gauntlet-stat-clippy target:
    cargo clippy --release --locked -p zensim-validate --bin panel --target-dir {{quote(target)}} -- -D warnings


gauntlet-adapter-tests panel:
    TMPDIR="$HOME/tmp" ZEN_PANEL_BIN={{quote(panel)}} python3 -B -m unittest scripts.tests.test_gauntlet_frozen scripts.tests.gauntlet_owner_integration

gauntlet-bmp-build target:
    cargo build --release --locked --manifest-path tools/shipgates-judge/Cargo.toml --bin shipgates_prepare_source --target-dir {{quote(target)}}

gauntlet-bmp-prepare binary list output:
    {{quote(binary)}} --bmp-list {{quote(list)}} {{quote(output)}}

gauntlet-comparison manifest output:
    TMPDIR="$HOME/tmp" python3 scripts/v_next/build_peer_fullevals.py --comparison-manifest {{quote(manifest)}} --out-dir {{quote(output)}}

gauntlet-bmp-check target:
    cargo clippy --release --locked --manifest-path tools/shipgates-judge/Cargo.toml --bin shipgates_prepare_source --target-dir {{quote(target)}} -- -D warnings

gauntlet-analyze target output:
    just gauntlet-build {{quote(target)}}
    ZEN_PANEL_BIN={{quote(target + "/release/panel")}} just gauntlet-comparison {{quote(output + "/WITHHELD_INPUTS.json")}} {{quote(output)}}

# Registered severe quantization: exact native tables; base-only scalar and map.
r8-severe-build config:
    python3 -c 'import shutil; from pathlib import Path; margin = 2 if Path("tools/shipgates-target/target/release/deps").is_dir() else 4; assert shutil.disk_usage("/home/lilith").free >= (40+margin)*1024**3, "build disk floor plus headroom unavailable"'
    cargo build --release --locked --manifest-path tools/shipgates-target/Cargo.toml --config {{quote(config)}} --bin r8-severe-quant

r8-severe-clippy config:
    cargo clippy --release --locked --manifest-path tools/shipgates-target/Cargo.toml --config {{quote(config)}} --bin r8-severe-quant -- -D warnings

r8-severe-phase phase inputs audits plan registration binary oracle output:
    python3 scripts/rev4_featpot/shipgates_peer_target.py severe-quant --phase {{quote(phase)}} --inputs {{quote(inputs)}} --audits {{quote(audits)}} --plan {{quote(plan)}} --registration {{quote(registration)}} --binary {{quote(binary)}} --oracle {{quote(oracle)}} --output {{quote(output)}}

r8-severe-tests config:
    TMPDIR="$HOME/tmp" cargo test --release --locked --manifest-path tools/shipgates-target/Cargo.toml --config {{quote(config)}} --bin r8-severe-quant


gauntlet-archive source destination kind="encoded":
    test ! -e {{quote(destination)}}
    rsync -a --no-owner --no-group {{quote(source + "/")}} {{quote(destination + "/")}}
    python3 scripts/rev4_featpot/shipgates_mirror_verify.py --source {{quote(source)}} --destination {{quote(destination)}} --complete-marker _MANIFEST.json --payload-kind {{quote(kind)}}

# Resume only an owned partial mirror with the exact same completion identity.
gauntlet-archive-resume source destination kind="encoded":
    cmp {{quote(source + "/_MANIFEST.json")}} {{quote(destination + "/_MANIFEST.json")}}
    rsync -a --no-owner --no-group {{quote(source + "/")}} {{quote(destination + "/")}}
    python3 scripts/rev4_featpot/shipgates_mirror_verify.py --source {{quote(source)}} --destination {{quote(destination)}} --complete-marker _MANIFEST.json --payload-kind {{quote(kind)}}

gauntlet-reference-view html reference:
    node scripts/v_next/gauntlet_render_check.js {{quote(html)}} --reference {{quote(reference)}}


# TRAIN-only color/noise comparison; local fits serialize through the shared heavy lock.
color-noise-repair mode output:
    TMPDIR="$HOME/tmp" python3 -u scripts/rev4_featpot/color_noise_repair.py {{quote(mode)}} --out {{quote(output)}}

color-noise-assessment mode output:
    TMPDIR="$HOME/tmp" python3 -u scripts/rev4_featpot/color_noise_assessment.py {{quote(mode)}} --out {{quote(output)}}

color-noise-ensemble mode output:
    TMPDIR="$HOME/tmp" python3 -u scripts/rev4_featpot/color_noise_ensemble.py {{quote(mode)}} --out {{quote(output)}}

color-noise-freeze output:
    just color-noise-assessment select {{quote(output)}}
    just color-noise-ensemble score {{quote(output)}}

color-noise-assess output:
    test -f {{quote(output + "/ensembles/TRAIN_SELECTION.json")}}
    just color-noise-assessment external {{quote(output)}}
    just color-noise-ensemble external {{quote(output)}}
    just color-noise-render {{quote(output)}}

color-noise-render output:
    just color-noise-assessment dashboard {{quote(output)}}
    just color-noise-assessment dashboard {{quote(output + "/ensembles")}}
    just color-noise-assessment report {{quote(output)}}

color-noise-verify output:
    scripts/v_next/gauntlet_gates.sh {{quote(output + "/gauntlet.html")}} {{quote(output)}}
    scripts/v_next/gauntlet_gates.sh {{quote(output + "/ensembles/gauntlet.html")}} {{quote(output + "/ensembles")}}
    just gauntlet-reference-view {{quote(output + "/gauntlet.html")}} mos
    just gauntlet-reference-view {{quote(output + "/gauntlet.html")}} jnd
    just gauntlet-reference-view {{quote(output + "/ensembles/gauntlet.html")}} mos
    just gauntlet-reference-view {{quote(output + "/ensembles/gauntlet.html")}} jnd

color-noise-check:
    TMPDIR="$HOME/tmp" PYTHONPATH=scripts:scripts/rev4_featpot python3 -m unittest scripts.tests.test_color_noise_repair scripts.tests.test_e33_package

# Validate stable compatibility while tool builds select the current SIMD beta.
beta-cohort-check:
    cargo fmt --check
    cargo clippy --locked -p zensim --lib -- -D warnings
    cargo test --locked -p zensim --lib

# B / diff+frag-820 promotion: run these through run-heavy on shared hosts.
default-b-tests:
    cargo test -p zensim --lib --tests --examples --no-fail-fast

default-b-minimal-tests:
    cargo test -p zensim --no-default-features --lib --test default_b_diff_frag --no-fail-fast

default-b-all-feature-tests:
    cargo test -p zensim --all-features --lib --tests --no-fail-fast

default-b-clippy:
    cargo clippy -p zensim --all-targets --all-features -- -D warnings

default-b-doc-tests:
    cargo test -p zensim --doc

# Existing tier-parity tests also cover the training-tool constructor aliases.
default-b-final-compat:
    cargo test -p zensim --all-features --test default_b_diff_frag --test medium_hardening
    cargo test -p zensim-train-core --lib
    cargo test -p zensim-validate --lib adam_simd
    just default-b-minimal-head-check
    cargo test -p zensim --doc

# Compile the production library with the optional head and no research API.
default-b-minimal-head-check:
    cargo clippy -p zensim --no-default-features --lib --features corruption-head -- -D warnings

# Check default and threaded minimal production shapes, plus bare wasm compilation.
default-b-portability-checks:
    cargo clippy -p zensim --lib -- -D warnings
    cargo clippy -p zensim --no-default-features --lib --features threads,corruption-head -- -D warnings
    cargo clippy -p zensim --no-default-features --lib --features custom-profiles -- -D warnings
    cargo check -p zensim --no-default-features --target wasm32-unknown-unknown --lib

# The regress crate enables custom profiles without enabling the research API.
default-b-custom-minimal-tests:
    cargo test -p zensim --no-default-features --features custom-profiles --test default_b_diff_frag

# Exercise the in-workspace consumers of the promoted default.
default-b-consumer-tests:
    cargo test -p zensim-regress --lib --tests
    cargo test -p zensim-validate --lib --test bake_surface

# Explicit single-read JND CSV admission; native tools own scoring/statistics.
aic2026-jnd-admission-tests:
    python3 -B -m unittest scripts.tests.test_aic2026_jnd_admission

# Reuse saved B predictions and byte-matched supplied crop metrics; no scoring.
aic2026-jnd-peer-suite scored peer_table peer_images output:
    python3 -B scripts/prepare_aic2026_evaluation.py {{quote(scored)}} --bind-peer-table {{quote(peer_table)}} --peer-images {{quote(peer_images)}} --output {{quote(output)}}

# Bind explicitly selected retained native tables, with per-column identities.
aic2026-jnd-reproduced-suite scored spec peer_images output:
    python3 -B scripts/prepare_aic2026_evaluation.py {{quote(scored)}} --peer-scores-spec {{quote(spec)}} --peer-images {{quote(peer_images)}} --output {{quote(output)}}

# Current B intentionally has no JND map; the report owner must agree.
default-b-verdict-jnd-tests:
    cargo test -p zensim-validate --bin bake_verdict

# By-construction exploration (benchmarks/byconstruction_2026-10-22.md): revision 6 (ringhpm + globaff) and the
# research map painting modes. Heavy: run through ~/work/zen/scripts/run-heavy.
byc-rev6-tests:
    cargo test --release -p zensim --features custom-profiles,feature-regime-v2,training --lib -- globaff map_paint ring_form feature_defs feature_plan research::

# Served (pixel route) against table route for revision-6 research bakes on the mixture images.
byc-serve-parity bin_dir mixture out +bakes:
    python3 scripts/byconstruction/serve_parity.py --bin-dir {{quote(bin_dir)}} --mixture {{quote(mixture)}} --out {{quote(out)}} {{ prepend("--bake ", bakes) }}

# The same against a bake's table-route predictions for a pair list (row order), e.g. AIC2026 public_train.
byc-serve-parity-pairs bin_dir pairs pred out bake:
    python3 scripts/byconstruction/serve_parity.py --bin-dir {{quote(bin_dir)}} --pairs {{quote(pairs)}} --pred {{quote(pred)}} --out {{quote(out)}} --bake {{quote(bake)}} --jobs 12

# Candidate, legacy and judge maps for the map-quality study (zensim-bench `map_dump`).
byc-map-dump pairs out bake tag window="box" paint="nearest" split="repair" *flags="":
    cargo run --release --manifest-path zensim-bench/Cargo.toml --features dvifm-judge --example map_dump -- --pairs {{quote(pairs)}} --out {{quote(out)}} --bake {{quote(bake)}} --tag {{tag}} --window {{window}} --paint {{paint}} --split {{split}} {{flags}}

# By-construction arms on both damage-mixture tests (standard and global-edit): inversions per E21 cell and the largest
# rise, from the score files aic_mixture.sh wrote.
byc-mixture-summary +arms:
    python3 scripts/byconstruction/mixture_test.py summary --out /mnt/v/output/zensim/byconstruction-2026-10-22/mixture {{arms}}
    python3 scripts/byconstruction/mixture_test.py summary --out /mnt/v/output/zensim/byconstruction-2026-10-22/mixture_global {{arms}}

# The built workbench module (PKG = its pkg-zensimx) against the native crate on one synthetic pair: every metric's
# score and every map's sum; same ZENSIM_WORKBENCH_BAKES as the build.
byc-workbench-smoke pkg:
    cargo run -q --release --manifest-path zensim-workbench/Cargo.toml --example smoke > target/byc-smoke-native.jsonl
    node zensim-workbench/smoke.mjs {{quote(pkg)}} > target/byc-smoke-wasm.jsonl
    python3 scripts/byconstruction/workbench_site.py smoke-compare target/byc-smoke-native.jsonl target/byc-smoke-wasm.jsonl

# The assembled workbench SITE in headless Chromium against the native smoke values (needs Playwright; PY is a
# Python with it). Same ZENSIM_WORKBENCH_BAKES as the build.
byc-workbench-check site py="python3":
    cargo run -q --release --manifest-path zensim-workbench/Cargo.toml --example smoke > target/byc-smoke-native.jsonl
    {{py}} zensim-workbench/workbench_check.py {{quote(site)}} target/byc-smoke-native.jsonl

# The workbench module (wasm32 + SIMD128) with the bakes in ZENSIM_WORKBENCH_BAKES; wasm-bindgen-cli 0.2.129. The
# module's menu label names zensim's version and the commit it was built from (`wip on` when the working copy
# differs), unless ZENSIM_WORKBENCH_LABEL is set.
byc-workbench-module out="target/zensim-workbench-web" wasm_bindgen="wasm-bindgen":
    #!/usr/bin/env bash
    set -euo pipefail
    [ "$({{wasm_bindgen}} --version)" = 'wasm-bindgen 0.2.129' ] || { echo 'wasm-bindgen-cli 0.2.129 required' >&2; exit 2; }
    version=$(sed -n 's/^version = "\(.*\)"/\1/p' zensim/Cargo.toml | head -1)
    dirty=$(jj log -r @ --no-graph -T 'if(empty, "", "wip on ")')
    if [ -z "$dirty" ]; then rev='latest(::@ & ~empty())'; else rev='latest(::@- & ~empty())'; fi
    commit=$(jj log -r "$rev" --no-graph -T 'commit_id.short(8)')
    export ZENSIM_WORKBENCH_LABEL="${ZENSIM_WORKBENCH_LABEL:-zensim $version · $dirty$commit (private)}"
    echo "module label: $ZENSIM_WORKBENCH_LABEL"
    RUSTFLAGS="-C target-feature=+simd128" cargo build --release --manifest-path zensim-workbench/Cargo.toml --target wasm32-unknown-unknown --lib
    {{wasm_bindgen}} --target web --out-dir {{out}}/pkg-zensimx zensim-workbench/target/wasm32-unknown-unknown/release/zensim_workbench.wasm
    cargo run -q --release --manifest-path zensim-workbench/Cargo.toml --example workbench_manifest > {{out}}/extra-modules.json

# The trainer-side tools the by-construction cells use (byc_cells.py's REV4_V2_BIN_DIR): zensim_mlp_train,
# bake_dial_refit, panel, score_pair_with_bake, byc_fsid and byc_predict, copied to /var/tmp/byconstruction-bin/<commit>
# with BUILD_COMMIT and SHA256SUMS. The working copy must equal its commit.
byc-train-bins:
    #!/usr/bin/env bash
    set -euo pipefail
    [ -z "$(jj diff --summary -r @)" ] || { echo 'commit first: the binaries name their commit' >&2; exit 2; }
    commit=$(jj log -r 'latest(::@ & ~empty())' --no-graph -T 'commit_id')
    dest=/var/tmp/byconstruction-bin/${commit:0:12}
    [ ! -e "$dest" ] || { echo "$dest exists" >&2; exit 2; }
    bins="zensim_mlp_train bake_dial_refit panel score_pair_with_bake byc_fsid byc_predict"
    cargo build --release -p zensim-validate $(printf -- '--bin %s ' $bins)
    target=$(cargo metadata --format-version 1 --no-deps | python3 -c 'import json,sys; print(json.load(sys.stdin)["target_directory"])')
    mkdir -p "$dest"
    for b in $bins; do cp "$target/release/$b" "$dest/"; done
    printf '%s' "$commit" > "$dest/BUILD_COMMIT"
    (cd "$dest" && sha256sum $bins > SHA256SUMS)
    echo "$dest"

# The native loop tools (loop-converge's binaries and the workbench's loop_calibrate) with the bakes in
# ZENSIM_WORKBENCH_BAKES, copied to /var/tmp/byconstruction-bin/<commit>-loop-TAG with PROVENANCE and SHA256SUMS. NOTE
# goes into PROVENANCE (say "private weights: LAN only" when a bake is private). The working copy must equal its commit.
byc-loop-bins tag note="":
    #!/usr/bin/env bash
    set -euo pipefail
    : "${ZENSIM_WORKBENCH_BAKES:?set ZENSIM_WORKBENCH_BAKES (name=path;...)}"
    [ -z "$(jj diff --summary -r @)" ] || { echo 'commit first: the binaries name their commit' >&2; exit 2; }
    commit=$(jj log -r 'latest(::@ & ~empty())' --no-graph -T 'commit_id.short(12)')
    dest=/var/tmp/byconstruction-bin/$commit-loop-{{tag}}
    [ ! -e "$dest" ] || { echo "$dest exists" >&2; exit 2; }
    cargo build --release --manifest-path zensim-workbench/loop-converge/Cargo.toml --bins
    cargo build --release --manifest-path zensim-workbench/Cargo.toml --example loop_calibrate
    mkdir -p "$dest"
    # CARGO_TARGET_DIR (one directory for both workspaces) keeps the builds off a full disk.
    lc=${CARGO_TARGET_DIR:-zensim-workbench/loop-converge/target}; wb=${CARGO_TARGET_DIR:-zensim-workbench/target}
    cp "$lc"/release/{steer_rd,loop_pairs,map_probe,zensim-workbench-loop-converge} "$wb"/release/examples/loop_calibrate "$dest"/
    bakes=$(echo "$ZENSIM_WORKBENCH_BAKES" | tr ';' '\n' | while IFS='=' read -r name path; do
      [ -n "$name" ] && echo "$name=$(sha256sum "$path" | cut -c1-8)"; done | paste -sd, - | sed 's/,/, /g')
    {
      echo "built from $commit on $(hostname), $(date -u +%F)"
      echo "ZENSIM_WORKBENCH_BAKES: $bakes{{ if note != "" { " (" + note + ")" } else { "" } }}"
      echo "loop_calibrate built from the same commit's zensim-workbench"
    } > "$dest/PROVENANCE"
    (cd "$dest" && sha256sum steer_rd loop_pairs map_probe zensim-workbench-loop-converge loop_calibrate > SHA256SUMS)
    echo "$dest"

# D6 admission controls are metadata/key-only; no public_test or EVAL payloads.
aic-finetune-admission-tests:
    TMPDIR="$HOME/tmp/aic-finetune-20261026" OPENBLAS_NUM_THREADS=1 python3 -B -m unittest scripts.tests.test_aic_finetune_admission scripts.tests.test_shippath2_admission

# Queue behind existing shared-machine timing work; caller supplies frozen source paths.
aic-finetune-folds table pairs assignment output commit:
    TMPDIR="$HOME/tmp/aic-finetune-20261026" OPENBLAS_NUM_THREADS=1 flock "$HOME/tmp/heavy.lock" "$HOME/work/claudehints/scripts/run-heavy" --mem 4G --jobs 1 -- python3 -B scripts/byconstruction/aic_train.py rev7-folds --table {{quote(table)}} --pairs {{quote(pairs)}} --assignment {{quote(assignment)}} --output {{quote(output)}} --build-commit {{quote(commit)}}
