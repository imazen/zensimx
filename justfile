# Chroma Stage A transport: the caller places this in one worker run-heavy scope.
chroma-pilot action root:
    python3 -B scripts/byconstruction/chroma_pilot.py {{quote(action)}} {{quote(root)}}

chroma-pilot-report root report:
    python3 -B scripts/byconstruction/chroma_pilot.py report {{quote(root)}} --report {{quote(report)}}

chroma-pilot-sources root mlic bank imazen candidates commit mirror:
    python3 -B scripts/byconstruction/chroma_sources.py --root {{quote(root)}} --mlic-root {{quote(mlic)}} --bank {{quote(bank)}} --imazen {{quote(imazen)}} --safe-candidates {{quote(candidates)}} --build-commit {{quote(commit)}} --mirror {{quote(mirror)}}

chroma-pilot-check:
    python3 -B -m unittest discover -s scripts/tests -p test_chroma_pilot.py
    just lint-scripts

# Aggregate only completed owner outputs; no new target table is read.
dataset-mix-summary study:
    python3 -B scripts/rev4_featpot/byc_cells.py mix-summary --study {{quote(study)}}

# Historical registry: refuses by default. See scripts/byconstruction/historical_guard.py.
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

dataset-trainer-clippy target:
    cargo clippy --locked -p zensim-validate --lib --bin zensim_mlp_train --target-dir {{quote(target)}} -- -D warnings

# Caller holds heavy.lock and run-heavy; this creates fresh matching predictions.
dataset-order-prepare study:
    OPENBLAS_NUM_THREADS=1 python3 -B scripts/rev4_featpot/matched_codec.py {{quote(study)}}

# Full matching uses each paired full Mlic checkpoint; caller owns heavy.lock/run-heavy.
dataset-order-prepare-full output:
    OPENBLAS_NUM_THREADS=1 python3 -B scripts/rev4_featpot/matched_codec.py {{quote(output)}} --full

dataset-order-tests:
    python3 -B -m unittest discover -s scripts/tests -p 'test_matched_codec.py' -v

dataset-confidence-tests:
    python3 -B -m unittest discover -s scripts/tests -p 'test_confidence_sources.py' -v

# Small metadata/key reads only; no feature, score or uncertainty payloads.
dataset-confidence-populations output:
    OPENBLAS_NUM_THREADS=1 python3 -B scripts/byconstruction/confidence_population.py {{quote(output)}}

# Metadata only: complete omitted assessment references and original dataset families.
dataset-confidence-omissions root bank output:
    OPENBLAS_NUM_THREADS=1 python3 -B scripts/byconstruction/confidence_omissions.py {{quote(root)}} {{quote(bank)}} {{quote(output)}}

dataset-confidence-omission-tests:
    python3 -B -m unittest discover -s scripts/tests -p 'test_confidence_omissions.py' -v

# Native fitting rows only; source audit creates no weights or calibration.
dataset-confidence-audit populations sha256 output:
    OPENBLAS_NUM_THREADS=1 python3 -B scripts/byconstruction/confidence_audit.py {{quote(populations)}} {{quote(sha256)}} {{quote(output)}}

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

# Independent original-Mlic comparison; match preparation follows binary control.
dataset-order-chain study source_commit trainer_sha256:
    OPENBLAS_NUM_THREADS=1 python3 -B scripts/byconstruction/mlic_neff_chain.py {{quote(study)}} --source-commit {{quote(source_commit)}} --candidate MlicOrder --trainer-sha256 {{quote(trainer_sha256)}}

# Attached coordinator records queue identity before acquiring the shared slot.
dataset-order-queue study source_commit trainer_sha256:
    OPENBLAS_NUM_THREADS=1 python3 -B scripts/byconstruction/dataset_queue.py {{quote(study)}} {{quote(source_commit)}} {{quote(trainer_sha256)}}

# Caller holds the compute lock and run-heavy; rejected screens refuse fitting.
dataset-vote-full study output:
    OPENBLAS_NUM_THREADS=1 python3 -B scripts/byconstruction/mlic_vote_full.py {{quote(study)}} {{quote(output)}}

dataset-vote-full-tests:
    python3 -B -m unittest discover -s scripts/tests -p 'test_mlic_vote_full.py' -v

# Native source admission follows the pinned TRAIN queue; this is not a benchmark.
dataset-confidence-admission mode plan:
    OPENBLAS_NUM_THREADS=1 python3 -B scripts/byconstruction/confidence_admission.py {{quote(mode)}} {{quote(plan)}}

dataset-confidence-admission-tests:
    python3 -B -m unittest discover -s scripts/tests -p 'test_confidence_admission.py' -v

dataset-cell-progress cells arm='Mdata' folds='kadid,tid2013,konfig,cid22_a25':
    env BYC_CELLS={{quote(cells)}} OPENBLAS_NUM_THREADS=1 python3 -B scripts/rev4_featpot/byc_cells.py progress --arm {{quote(arm)}} --folds {{quote(folds)}}

dataset-run-progress study cells arm folds seeds='0,1,2':
    env BYC_CELLS={{quote(cells)}} OPENBLAS_NUM_THREADS=1 python3 -B scripts/rev4_featpot/byc_cells.py progress --arm {{quote(arm)}} --folds {{quote(folds)}} --seeds {{quote(seeds)}} --run-state {{quote(study)}}/RUN_STATE.json --output {{quote(study)}}/CURRENT_RUN.json --brief

dataset-study-progress +studies:
    OPENBLAS_NUM_THREADS=1 python3 -B scripts/byconstruction/dataset_study_status.py {{studies}}

# Read live processes and artifacts for the registered October 27 queue; starts no work.
mlic-priority-progress:
    @just dataset-study-progress /mnt/v/output/zensim/mlic-order-20261027-run02 /mnt/v/output/zensim/mlic-confidence-20261027-run02
    @just dataset-confidence-admission status /mnt/tower/output/zensim/mlic-confidence-admission-20261027-run02/PLAN.json
    @just aic-recipe-adaptation status /mnt/v/output/zensim/mlic-aic-adaptation-20261027-run03/PLAN.json
    @just aic-recipe-adaptation-panel status /mnt/v/output/zensim/mlic-aic-adaptation-20261027-run03/PLAN.json /mnt/v/output/zensim/mlic-aic-adaptation-20261027-run03/panels
    @just aic-recipe-adaptation-full status /mnt/v/output/zensim/mlic-aic-full-20261027-run03/PLAN.json
    @just aic-recipe-adaptation-full-panel status /mnt/v/output/zensim/mlic-aic-full-20261027-run03/PLAN.json /mnt/v/output/zensim/mlic-aic-full-20261027-run03/panels
    @just confidence-followthrough-status /mnt/v/output/zensim/mlic-confidence-followthrough-20261027-run01/CONFIG.json
    @if [ -f /mnt/v/output/zensim/mlic-matched-continuation-20261027-run01/PLAN.json ]; then just aic-matched-continuation status /mnt/v/output/zensim/mlic-matched-continuation-20261027-run01/PLAN.json; fi
    @if [ -f /mnt/v/output/zensim/mlic-candidate-gates-20261027-run02/queue.log ]; then tail -n 2 /mnt/v/output/zensim/mlic-candidate-gates-20261027-run02/queue.log; fi
    @if [ -f /mnt/v/output/zensim/mlic-matched-continuation-folds-20261027-run01/PLAN.json ]; then just aic-matched-continuation fold-status /mnt/v/output/zensim/mlic-matched-continuation-folds-20261027-run01/PLAN.json; fi
    @if [ -f /mnt/v/output/zensim/mlic-aic-share-frontier-20261027-run01/PLAN.json ]; then just aic-matched-continuation fold-status /mnt/v/output/zensim/mlic-aic-share-frontier-20261027-run01/PLAN.json; fi
    @if [ -f /mnt/v/output/zensim/mlic-aic-recovery-20261027-run01/PLAN.json ]; then just aic-matched-continuation fold-status /mnt/v/output/zensim/mlic-aic-recovery-20261027-run01/PLAN.json; fi
    @for plan in /mnt/v/output/zensim/mlic-aic-recipe-*-20261027-run*/PLAN.json; do if [ -f "$plan" ] && { [ -f "$(dirname "$plan")/QUEUE.json" ] || [ -f "$(dirname "$plan")/STATE.json" ]; }; then echo "$(basename "$(dirname "$plan")")"; just aic-matched-continuation fold-status "$plan"; fi; done

# Registered matched-continuation diagnostic (byconstruction record, 2026-10-27):
# plan freezes source/tools/inputs; queue waits for the confidence followthrough,
# then takes heavy.lock + run-heavy; status reads live processes and artifacts.
aic-matched-continuation mode plan:
    OPENBLAS_NUM_THREADS=1 python3 -B scripts/byconstruction/aic_matched_continuation.py {{quote(mode)}} {{quote(plan)}}

# Execute a frozen stage-2 plan, with tool paths bound before Python imports.
# The caller resolves current heavy.lock ownership/queues before using this recipe.
fx2-continuation-run mode plan:
    #!/usr/bin/env bash
    set -euo pipefail
    case {{quote(mode)}} in fold-run|full-run|omission-run) ;; *) echo 'expected fold-run, full-run or omission-run' >&2; exit 2 ;; esac
    plan_file=$(realpath {{quote(plan)}})
    run_dir=$(dirname "$plan_file")
    root_dir=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["root"])' "$plan_file")
    log_file="$(dirname "$run_dir")/logs/$(basename "$run_dir").log"
    test ! -e "$run_dir/STATE.json"
    test ! -e "$log_file"
    mkdir -p "$(dirname "$log_file")"
    free -g
    TMPDIR="$HOME/tmp" flock "$HOME/tmp/heavy.lock" "$HOME/work/claudehints/scripts/run-heavy" --mem 30G --jobs 12 -- env \
      OPENBLAS_NUM_THREADS=1 REV4_V2_BIN_DIR="$run_dir/bin" REV4_V2_ROOT="$root_dir" BYC_ROOT="$root_dir" \
      AIC_BIN="$run_dir/bin" ZEN_PANEL_BIN="$run_dir/bin/panel" \
      python3 -B "$run_dir/source/scripts/byconstruction/aic_matched_continuation.py" {{quote(mode)}} "$plan_file" \
      2>&1 | tee "$log_file"

aic-matched-continuation-plan run commit trainer meta_tool mode='plan':
    python3 -B scripts/byconstruction/aic_matched_continuation.py {{quote(mode)}} {{quote(run)}} --source-commit {{quote(commit)}} --trainer {{quote(trainer)}} --meta-tool {{quote(meta_tool)}}

# Label-free gates for one revision-7 packed candidate through gates.py with the
# revision-7 tools, instrument tables and empty human corpus (dial verdicts C1-C6
# on both grids, G-STEER, identity at every available SIMD tier; NEARID on request).
# Caller holds heavy.lock and run-heavy. Refuses an existing output directory.
candidate-gates-rev7 name model out steps='verdict,steer,identity' tools='/var/tmp/byconstruction-bin/688746b78bee-gates' instruments='/mnt/v/output/zensim/byconstruction-2026-10-22/rev7raw/instruments' corpus='/mnt/v/output/zensim/byconstruction-2026-10-22/rev7raw/no-human-corpus':
    #!/usr/bin/env bash
    set -euo pipefail
    (cd {{quote(tools)}} && sha256sum --quiet -c SHA256SUMS)
    test ! -e {{quote(out)}}
    TMPDIR="$HOME/tmp" python3 -B scripts/byconstruction/gates.py --name {{quote(name)}} --model {{quote(model)}} --out {{quote(out)}} --bin-dir {{quote(tools)}} --instruments {{quote(instruments)}} --corpus-root {{quote(corpus)}} --formula-rev 7 --steps {{quote(steps)}}
    python3 -B scripts/byconstruction/gates.py --summary --out {{quote(out)}} --name {{quote(name)}} > {{quote(out)}}.summary.json

# The same gates for every seed-0 model of a completed full AIC comparison, in
# FULL_FITS_COMPLETE order (initial models, then continuations), packed bakes.
candidate-gates-rev7-full fits out:
    #!/usr/bin/env bash
    set -euo pipefail
    mkdir {{quote(out)}}
    python3 -B -c 'import json,sys; c=json.load(open(sys.argv[1])); rows=[dict(r,arm=k) for k,v in c["initial"].items() for r in v]+c["continued"]; [print(r["arm"].lower(), r["directory"]+"/refit/production-f16.bin", r["packed_sha256"]) for r in rows if r["seed"]==0]' {{quote(fits)}} > {{quote(out)}}/MODELS.tsv
    while read -r name model pin; do
      echo "$pin  $model" | sha256sum --quiet -c
      just candidate-gates-rev7 "$name" "$model" {{quote(out)}}/"$name"
    done < {{quote(out)}}/MODELS.tsv

# Native HDR feature precision against the f64 oracle (record: benchmarks/byconstruction_2026-10-22.md,
# "Native HDR precision"): production and ZENSIM_FEATCANON=exact dumps of every 16-bit PQ reference in refs
# under four code-space perturbations (tier_audit_features --hdr-pq, one tier), then per-feature errors and
# each NAME=BAKE's score difference. Build tier_audit_features with --features featcanon-oracle and
# predict_features_with_bake without it.
[positional-arguments]
hdr-precision refs out audit predict rev primaries peak tier +bakes:
    #!/usr/bin/env bash
    set -euo pipefail
    refs=$1 out=$2 audit=$3 predict=$4 rev=$5 prim=$6 peak=$7 tier=$8
    shift 8
    test ! -e "$out"
    mkdir -p "$out/dump-prod" "$out/dump-exact"
    args=()
    for ref in "$refs"/*.png; do
      s=$(basename "$ref" .png)
      for k in dither1 shift4 blend5 shift64; do args+=(--hdr-pq "hdr-$s-$k" "$ref" "perturb:$k" "$prim" "$peak"); done
    done
    export ZENSIM_FORMULA_REV=$rev TIER_AUDIT_ONLY=$tier RAYON_NUM_THREADS=1
    ZENSIM_FEATCANON_DUMP="$out/dump-prod" "$audit" "${args[@]}" > "$out/run-prod.log" 2>&1
    ZENSIM_FEATCANON=exact ZENSIM_FEATCANON_DUMP="$out/dump-exact" "$audit" "${args[@]}" > "$out/run-exact.log" 2>&1
    bakes=()
    for b in "$@"; do bakes+=(--bake "$b"); done
    python3 scripts/byconstruction/rev7_forms.py --current "$out/dump-prod" --exact "$out/dump-exact" --tier "$tier" \
      --out "$out/PRECISION.json" --predict "$predict" "${bakes[@]}" | tee "$out/PRECISION.txt"

aic-matched-continuation-tests:
    python3 -B -m unittest scripts.tests.test_aic_matched_continuation_coordinator scripts.tests.test_aic_adaptation_full_panel
    python3 -B -m unittest discover -s scripts/tests -p 'test_aic_matched_continuation.py'

# Plan bindings and the legacy fx1 continuation contracts used by the fx2 study.
fx2-continuation-tests:
    TMPDIR="$HOME/tmp" OPENBLAS_NUM_THREADS=1 python3 -B -m unittest scripts.tests.test_aic_matched_continuation_coordinator scripts.tests.test_aic_matched_continuation scripts.tests.test_aic_adaptation scripts.tests.test_aic_adaptation_full scripts.tests.test_aic_adaptation_panel scripts.tests.test_aic_adaptation_full_panel scripts.tests.test_subjective_training_leg scripts.tests.test_rev5_assessment scripts.tests.test_v2_production_pack

# Caller holds heavy.lock/run-heavy on an eligible build host; never build on dev.
fx2-identity-tests target:
    TMPDIR="$HOME/tmp" ZENSIM_FORMULA_REV=7 RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 cargo test --locked --release -p zensim --example serve_custom_bake --features custom-profiles --target-dir {{quote(target)}} nearid::tests -- --nocapture

fx2-identity-build target:
    just fx2-identity-tests {{quote(target)}}
    TMPDIR="$HOME/tmp" cargo clippy --locked --release -p zensim --example serve_custom_bake --features custom-profiles --target-dir {{quote(target)}}
    TMPDIR="$HOME/tmp" cargo build --locked --release -p zensim --example serve_custom_bake --features custom-profiles --target-dir {{quote(target)}}

# The verdict embeds the exact producer population registry; caller holds heavy.lock/run-heavy.
fx2-verdict-build target:
    TMPDIR="$HOME/tmp" cargo test --locked --release -p zensim-validate --lib --target-dir {{quote(target)}} assessment_projection -- --nocapture
    TMPDIR="$HOME/tmp" cargo build --locked --release -p zensim-validate --bin bake_verdict --target-dir {{quote(target)}}

dataset-progress-tests:
    python3 -B -m unittest discover -s scripts/tests -p 'test_dataset_study_status.py' -v

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

# `manifest` is a complete `jj file list` snapshot accompanying a source export.
lint-scripts-export manifest:
    LINT_TRACKED_FILES={{quote(manifest)}} python3 scripts/lint_scripts.py

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
    git -C /home/lilith/work/zen/zensimx archive --format=tar --output={{quote(root + "/sources/preparation-code.tar")}} {{quote(commit)}}
    git -C /home/lilith/work/zen/zensimx archive --format=tar --output={{quote(root + "/sources/cohort-producer-code.tar")}} {{quote(producer)}}
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
    nice -n19 ionice -c3 git -C /home/lilith/work/zen/zensimx archive --format=tar --output={{quote(root + "/" + version + ".tar")}} {{quote(commit)}}
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

# Historical Rev6 regression coverage remains enabled; these tests protect old contracts.
# Heavy: use run-heavy on an eligible host, not dev or r7900x.
byc-rev6-tests:
    cargo test --release -p zensim --features custom-profiles,feature-regime-v2,training --lib -- globaff map_paint ring_form feature_defs feature_plan research::

# Historical Rev6 parity launcher: refuses by default before restamping or scoring.
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

# October 27 full-panel comparison; heavy work is serialized by its queue.
aic-recipe-adaptation mode plan:
    OPENBLAS_NUM_THREADS=1 python3 -B scripts/byconstruction/aic_adaptation.py {{quote(mode)}} {{quote(plan)}}

aic-recipe-adaptation-tests:
    python3 -B -m unittest scripts.tests.test_aic_adaptation scripts.tests.test_aic_adaptation_panel scripts.tests.test_aic_adaptation_full scripts.tests.test_aic_adaptation_full_panel

# Caller holds heavy.lock/run-heavy; separate fitted and held-out source panels.
aic-recipe-adaptation-panel mode plan output:
    OPENBLAS_NUM_THREADS=1 python3 -B scripts/byconstruction/aic_adaptation_panel.py {{quote(mode)}} {{quote(plan)}} {{quote(output)}}

# Queue behind existing shared-machine timing work; caller supplies frozen source paths.
aic-finetune-folds table pairs assignment output commit:
    TMPDIR="$HOME/tmp/aic-finetune-20261026" OPENBLAS_NUM_THREADS=1 flock "$HOME/tmp/heavy.lock" "$HOME/work/claudehints/scripts/run-heavy" --mem 4G --jobs 1 -- python3 -B scripts/byconstruction/aic_train.py rev7-folds --table {{quote(table)}} --pairs {{quote(pairs)}} --assignment {{quote(assignment)}} --output {{quote(output)}} --build-commit {{quote(commit)}}

# Full recipe fits and both AIC strengths; the queue waits for complete fold panels.
aic-recipe-adaptation-full mode plan:
    OPENBLAS_NUM_THREADS=1 python3 -B scripts/byconstruction/aic_adaptation_full.py {{quote(mode)}} {{quote(plan)}}

# Full fitted populations; final-f32 and packed TRAIN-calibrated serving representations.
aic-recipe-adaptation-full-panel mode plan output:
    OPENBLAS_NUM_THREADS=1 python3 -B scripts/byconstruction/aic_adaptation_full_panel.py {{quote(mode)}} {{quote(plan)}} {{quote(output)}}

# Standalone native arithmetic owner; run on an eligible non-dev build host.
confidence-contrast-tests output:
    rustc --edition=2024 --test zensim-validate/tests/contrast_precision.rs -o {{quote(output)}}
    {{quote(output)}}

confidence-weight-bindings-tests:
    OPENBLAS_NUM_THREADS=1 python3 -B scripts/tests/test_confidence_weights.py

confidence-weight-bindings base output:
    OPENBLAS_NUM_THREADS=1 python3 -B scripts/byconstruction/confidence_weights.py {{quote(base)}} {{quote(output)}}

confidence-native-tests target:
    cargo test --locked -p zensim-validate --lib confidence_weights:: --target-dir {{quote(target)}}
    cargo test --locked -p zensim-validate --test contrast_precision --target-dir {{quote(target)}}

# Independent confidence arm: native receipts, original-Mlic parity, frozen 21 cells.
dataset-confidence-chain study source_commit trainer_sha256:
    OPENBLAS_NUM_THREADS=1 python3 -B scripts/byconstruction/mlic_neff_chain.py {{quote(study)}} --source-commit {{quote(source_commit)}} --candidate MlicConfidence --trainer-sha256 {{quote(trainer_sha256)}}

dataset-confidence-queue study source_commit trainer_sha256 after_full_panel:
    OPENBLAS_NUM_THREADS=1 python3 -B scripts/byconstruction/dataset_queue.py {{quote(study)}} {{quote(source_commit)}} {{quote(trainer_sha256)}} --candidate MlicConfidence --after-full-panel {{quote(after_full_panel)}}

# Existing native certificate; caller supplies frozen model paths and resource scope.
dataset-certify-bake binary bake:
    {{quote(binary)}} certify-monotone --bake {{quote(bake)}}

# Synthetic serving parity only, using a caller-pinned binary and explicit models.
[positional-arguments]
dataset-serving-parity-rev7 binary +models:
    #!/usr/bin/env bash
    set -euo pipefail
    binary="$1"
    shift
    ZENSIM_FORMULA_REV=7 "$binary" --prodqual-rev7 "$@"

# Existing N1-N3 owners. Caller freezes inputs and supplies the resource scope.
# The native candidate label contract is lowercase ASCII letters/digits.
dataset-nearid-candidate binary packet output label bake bake_sha256:
    #!/usr/bin/env bash
    set -euo pipefail
    label={{quote(label)}}
    [[ "$label" =~ ^[a-z0-9]+$ ]] || { echo 'candidate label must be lowercase ASCII letters/digits' >&2; exit 64; }
    ZENSIM_FORMULA_REV=7 ZENSIM_MAX_TIER=v3 RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 {{quote(binary)}} --nearid {{quote(packet)}} {{quote(output)}} "candidate-$label" {{quote(bake)}} {{quote(bake_sha256)}}
    python3 scripts/prodqual_label_free.py --nearid-candidate {{quote(output)}} "$label"

# Full-data confidence uses its separately pinned original authorities; no omission registry mutation.
confidence-full-weight-bindings base output authorities authorities_sha256:
    OPENBLAS_NUM_THREADS=1 python3 -B scripts/byconstruction/confidence_weights.py {{quote(base)}} {{quote(output)}} --full-authorities {{quote(authorities)}} --full-authorities-sha256 {{quote(authorities_sha256)}}

# Caller must hold heavy.lock/run-heavy. Requires complete independently pinned omission evidence.
confidence-full-fit plan:
    OPENBLAS_NUM_THREADS=1 python3 -B scripts/byconstruction/mlic_confidence_full.py {{quote(plan)}}

confidence-full-fit-tests:
    OPENBLAS_NUM_THREADS=1 python3 -B -m unittest scripts.tests.test_mlic_confidence_full

confidence-followthrough-tests:
    OPENBLAS_NUM_THREADS=1 python3 -B -m unittest scripts.tests.test_confidence_followthrough

# Run the source-frozen copy, attached; each heavy stage acquires the shared lock itself.
confidence-followthrough source config:
    OPENBLAS_NUM_THREADS=1 python3 -B {{quote(source)}}/scripts/byconstruction/confidence_followthrough.py {{quote(config)}}

confidence-followthrough-status config:
    OPENBLAS_NUM_THREADS=1 python3 -B scripts/byconstruction/confidence_followthrough.py {{quote(config)}} --status

# Frozen TRAIN-only parameter-packing diagnostic; caller owns run-heavy/compute lock.
mlic-precision-audit plan output:
    OPENBLAS_NUM_THREADS=1 python3 -B scripts/byconstruction/mlic_precision_audit.py {{quote(plan)}} {{quote(output)}}

# Existing native input/color/cache contracts at the current research arithmetic.
mlic-native-input-tests:
    cargo test -p zensim --test native_hdr_color --features custom-profiles -- --test-threads=1

# Refusal controls launch no workloads and require no model/data dependencies.
byc-historical-guard-tests:
    python3 -B -m unittest discover -s scripts/tests -p 'test_historical_guard.py' -v

# A2 EVIF pilot on one preselected idle worker, inside run-heavy.
chroma-evif root *families:
    python3 scripts/byconstruction/chroma_evif.py {{quote(root)}} --families {{families}}

chroma-cluster root:
    python3 scripts/byconstruction/chroma_cluster.py {{quote(root)}}

chroma-size-cost root:
    python3 -B scripts/byconstruction/chroma_size.py {{quote(root)}}

chroma-hdr-pilot root:
    python3 -B scripts/byconstruction/chroma_hdr.py {{quote(root)}}

chroma-evif-oracle root:
    python3 -B scripts/byconstruction/chroma_evif_oracle.py {{quote(root)}}

chroma-substitute root:
    python3 -B scripts/byconstruction/chroma_substitute.py {{quote(root)}}

chroma-judge-replay root:
    python3 -B scripts/byconstruction/chroma_diagnostics.py {{quote(root)}}

chroma-hdr-oracle root:
    python3 -B scripts/byconstruction/chroma_hdr_oracle.py {{quote(root)}}

chroma-archive action root *args:
    python3 -B scripts/byconstruction/chroma_archive.py {{quote(action)}} {{quote(root)}} {{args}}

chroma-a2-report root report state:
    python3 -B scripts/byconstruction/chroma_a2_report.py {{quote(root)}} {{quote(report)}} --mirror-state {{quote(state)}}

chroma-b0-teacher-replay root binary:
    python3 -B scripts/byconstruction/chroma_b0_replay.py {{quote(root)}} {{quote(binary)}}

chroma-hdr-a3 base:
    python3 scripts/byconstruction/chroma_hdr_a3.py '{{base}}'

chroma-census out:
    python3 scripts/byconstruction/chroma_census.py '{{out}}'

chroma-b0-worker trial image *flags:
    python3 scripts/byconstruction/chroma_b0_worker.py '{{trial}}' '{{image}}' {{flags}}

chroma-stage-sources:
    python3 scripts/byconstruction/chroma_stage_sources.py

chroma-geometry root:
    python3 scripts/byconstruction/chroma_geometry.py '{{root}}'

chroma-host-replay host image:
    python3 scripts/byconstruction/chroma_host_replay.py '{{host}}' '{{image}}'

chroma-hdr-a3-oracle root base:
    python3 scripts/byconstruction/chroma_hdr_oracle.py '{{root}}' --quality-grid --oracle-base '{{base}}'

chroma-cost:
    python3 scripts/byconstruction/chroma_cost.py

chroma-reference-replay:
    python3 scripts/byconstruction/chroma_reference_replay.py

chroma-batch-bench *args:
    python3 scripts/byconstruction/chroma_batch_bench.py {{args}}

chroma-stage-b *args:
    python3 scripts/byconstruction/chroma_b.py {{args}}

# Human HDR source memberships, without target or pixel reads.
hdr-source-membership out build_commit:
    python3 scripts/byconstruction/dataset_keys.py hdr-membership --out {{out}} --build-commit {{build_commit}}

hdr-source-membership-check:
    python3 -m unittest discover -s scripts/tests -p test_hdr_source_membership.py

chroma-hdr-full +args:
    python3 -B scripts/byconstruction/chroma_hdr_full.py {{args}}


# Scientific controls, feature-bank reconciliation and root construction (no fitting).
chroma-stage-c +args:
    python3 -B scripts/byconstruction/chroma_c.py {{args}}

chroma-root +args:
    OPENBLAS_NUM_THREADS=1 python3 -B scripts/byconstruction/ring_root.py chroma {{args}}

chroma-admission-check root receipt:
    cargo run --release --manifest-path tools/chroma-admission/Cargo.toml -- {{quote(root)}} {{quote(receipt)}}

chroma-worker-logs +args:
    python3 -B scripts/byconstruction/chroma_worker_logs.py {{args}}

# Run on an eligible build host, under run-heavy; target storage is explicit.
hdr-native-build target_dir:
    test "$(hostname)" != dev
    ZENSIM_BUILD_COMMIT="$(jj log --no-graph -r @ -T commit_id)" CARGO_TARGET_DIR={{target_dir}} cargo test --manifest-path zensim-bench/Cargo.toml --release --example hdr944_extract --features hdr944-extract
    ZENSIM_BUILD_COMMIT="$(jj log --no-graph -r @ -T commit_id)" CARGO_TARGET_DIR={{target_dir}} cargo build --manifest-path zensim-bench/Cargo.toml --release --example hdr944_extract --features hdr944-extract

# Existing UPIQ owner with separate Rev7 metadata admission; original Rev5 checks stay.
upiq-native-rev7-check target_dir:
    test "$(hostname)" != dev
    ZENSIM_BUILD_COMMIT="$(jj log --no-graph -r @ -T commit_id)" CARGO_TARGET_DIR={{target_dir}} cargo test -p zensim-validate --bin upiq_pu_score
    ZENSIM_BUILD_COMMIT="$(jj log --no-graph -r @ -T commit_id)" CARGO_TARGET_DIR={{target_dir}} cargo build --release -p zensim-validate --bin upiq_pu_score

# Existing teacher-table integration contract, using an explicit verified panel.
hdr-teacher-admission-check panel_bin:
    TMPDIR="$HOME/tmp" ZEN_PANEL_BIN={{panel_bin}} python3 scripts/tests/test_hdr_teacher_admission.py

hdr-native-plan-check:
    python3 -m unittest discover -s scripts/tests -p test_hdr_native_fleet_plan.py

upiq-rev7-library-metadata out build_commit:
    python3 scripts/rev4_featpot/upiq380.py relocate-rev7 --dest {{out}} --images /mnt/tower/input/iqa/datasets/upiq/content/upiq_dataset/images --parent-admission /mnt/v/output/zensim/upiq380-rev5-r2-2026-10-07/extraction-admission.json --library-inventory /mnt/tower/input/iqa/datasets/upiq/SHA256SUMS --build-commit {{build_commit}}

# Revision-7 atlas and MLP inspection instrument (standalone Cargo root).
viz-fmt:
    cargo fmt --manifest-path tools/zensim-viz/Cargo.toml

viz-build:
    mkdir -p "$HOME/tmp/zensim-viz"
    set -o pipefail; TMPDIR="$HOME/tmp/zensim-viz" flock "$HOME/tmp/heavy.lock" "$HOME/work/claudehints/scripts/run-heavy" --mem 16G --jobs 8 -- cargo build --manifest-path tools/zensim-viz/Cargo.toml --release 2>&1 | tee "$HOME/tmp/zensim-viz/build.log"

viz-test:
    set -o pipefail; TMPDIR="$HOME/tmp/zensim-viz" flock "$HOME/tmp/heavy.lock" "$HOME/work/claudehints/scripts/run-heavy" --mem 16G --jobs 8 -- cargo test --manifest-path tools/zensim-viz/Cargo.toml 2>&1 | tee "$HOME/tmp/zensim-viz/test.log"

viz-audit bake bank out:
    set -o pipefail; ZENSIM_FORMULA_REV=7 TMPDIR="$HOME/tmp/zensim-viz" flock "$HOME/tmp/heavy.lock" "$HOME/work/claudehints/scripts/run-heavy" --mem 16G --jobs 8 -- tools/zensim-viz/target/release/zensim-viz-export audit {{quote(bake)}} --bank {{quote(bank)}} --out {{quote(out)}} 2>&1 | tee "$HOME/tmp/zensim-viz/audit.log"

viz-serve directory:
    python3 tools/zensim-viz/serve_pages.py {{quote(directory)}} --port 3310

viz-web-check:
    node --check tools/zensim-viz/pages/app.js
    node --check tools/zensim-viz/pages/archive.js
    node --check tools/zensim-viz/pages/worker.js
    diff -qr /home/lilith/tmp/zensim-viz-brief/design tools/zensim-viz/web/design

viz-lens bake bank out feature:
    set -o pipefail; ZENSIM_FORMULA_REV=7 TMPDIR="$HOME/tmp/zensim-viz" flock "$HOME/tmp/heavy.lock" "$HOME/work/claudehints/scripts/run-heavy" --mem 16G --jobs 8 -- tools/zensim-viz/target/release/zensim-viz-export mlp-lens {{quote(bake)}} --bank {{quote(bank)}} --out {{quote(out)}} --feature {{quote(feature)}} 2>&1 | tee "$HOME/tmp/zensim-viz/lens.log"

viz-browser:
    TMPDIR="$HOME/tmp/zensim-viz" node tools/zensim-viz/tests/browser.mjs

viz-atlas bake bank reference distorted out:
    set -o pipefail; ZENSIM_FORMULA_REV=7 TMPDIR="$HOME/tmp/zensim-viz" flock "$HOME/tmp/heavy.lock" "$HOME/work/claudehints/scripts/run-heavy" --mem 16G --jobs 8 -- tools/zensim-viz/target/release/zensim-viz-export atlas {{quote(bake)}} --bank {{quote(bank)}} --reference {{quote(reference)}} --distorted {{quote(distorted)}} --out {{quote(out)}} 2>&1 | tee "$HOME/tmp/zensim-viz/atlas.log"

viz-browser-empty:
    VIZ_EMPTY_STATE=1 TMPDIR="$HOME/tmp/zensim-viz" node tools/zensim-viz/tests/browser.mjs

viz-compare bake other bank out feature:
    set -o pipefail; ZENSIM_FORMULA_REV=7 TMPDIR="$HOME/tmp/zensim-viz" flock "$HOME/tmp/heavy.lock" "$HOME/work/claudehints/scripts/run-heavy" --mem 16G --jobs 8 -- tools/zensim-viz/target/release/zensim-viz-export compare {{quote(bake)}} --other {{quote(other)}} --bank {{quote(bank)}} --out {{quote(out)}} --feature {{quote(feature)}} 2>&1 | tee "$HOME/tmp/zensim-viz/compare.log"

viz-certify bake:
    "$VIZ_CERTIFIER" certify-monotone --bake {{quote(bake)}}

viz-metric-fmt:
    cargo fmt -p zensim
    rustfmt --edition 2024 zensim/src/feature_v2_viz.rs

viz-http-check:
    curl --noproxy '*' --fail --silent --show-error -o /dev/null -w '%{http_code}\n' http://dev.lan:3310/

viz-format-test:
    node tools/zensim-viz/tests/format.mjs

viz-test-real:
    set -o pipefail; TMPDIR="$HOME/tmp/zensim-viz" ZENSIM_FORMULA_REV=7 flock "$HOME/tmp/heavy.lock" "$HOME/work/claudehints/scripts/run-heavy" --mem 16G --jobs 8 -- cargo test --release --manifest-path tools/zensim-viz/Cargo.toml --features real-bake-tests 2>&1 | tee "$HOME/tmp/zensim-viz/test-real.log"

viz-progress:
    python3 tools/zensim-viz/progress.py

viz-browser-audit:
    VIZ_TEST_SCOPE=audit TMPDIR="$HOME/tmp/zensim-viz" node tools/zensim-viz/tests/browser.mjs

viz-web-fmt:
    node "$VIZ_PRETTIER" --write tools/zensim-viz/pages/app.js tools/zensim-viz/pages/archive.js tools/zensim-viz/pages/worker.js tools/zensim-viz/pages/style.css tools/zensim-viz/pages/index.html tools/zensim-viz/tests/pages-parity.mjs tools/zensim-viz/tests/pages-lens.mjs tools/zensim-viz/tests/pages-smoke.mjs

# Caller supplies private file paths; every native job reacquires the shared lock.
viz-bundle plan:
    python3 -B tools/zensim-viz/bundle.py {{quote(plan)}}

viz-clippy:
    set -o pipefail; TMPDIR="$HOME/tmp/zensim-viz" flock "$HOME/tmp/heavy.lock" "$HOME/work/claudehints/scripts/run-heavy" --mem 16G --jobs 8 -- cargo clippy --release --manifest-path tools/zensim-viz/Cargo.toml --all-targets --features real-bake-tests -- -D warnings 2>&1 | tee "$HOME/tmp/zensim-viz/clippy.log"

viz-assemble config:
    python3 -B tools/zensim-viz/assemble.py {{quote(config)}}

viz-library-check:
    set -o pipefail; TMPDIR="$HOME/tmp/zensim-viz" flock "$HOME/tmp/heavy.lock" "$HOME/work/claudehints/scripts/run-heavy" --mem 16G --jobs 8 -- cargo clippy -p zensim --lib --features viz-export -- -D warnings 2>&1 | tee "$HOME/tmp/zensim-viz/library-clippy.log"

viz-bank bake bank out other:
    set -o pipefail; ZENSIM_FORMULA_REV=7 TMPDIR="$HOME/tmp/zensim-viz" flock "$HOME/tmp/heavy.lock" "$HOME/work/claudehints/scripts/run-heavy" --mem 16G --jobs 8 -- tools/zensim-viz/target/release/zensim-viz-export mlp-bank {{quote(bake)}} --bank {{quote(bank)}} --other {{quote(other)}} --out {{quote(out)}} 2>&1 | tee "$HOME/tmp/zensim-viz/lens-bank.log"

viz-ci-status:
    gh run list -R imazen/zensimx --limit 5 --json databaseId,name,status,conclusion,headSha,event

viz-review-browser:
    TMPDIR="$HOME/tmp/zensim-viz" node tools/zensim-viz/tests/review1.mjs

viz-reference catalog out:
    python3 -B tools/zensim-viz/reference.py --paper "$HOME/work/zen/zenpapers" --catalog {{quote(catalog)}} --out {{quote(out)}}

viz-design2-browser:
    TMPDIR="$HOME/tmp/zensim-viz" node tools/zensim-viz/tests/design2.mjs

# A low-effect family extraction-cost study; all compute runs on the quiet host.
a-family-cost-inputs corpus canonical out remote_root:
    python3 scripts/bench/a_family_cost_inputs.py --corpus {{quote(corpus)}} --canonical {{quote(canonical)}} --out {{quote(out)}} --remote-root {{quote(remote_root)}}

a-family-cost stage study *args:
    bash scripts/bench/a_family_cost.sh {{quote(stage)}} {{quote(study)}} {{args}}

# Public A against the pinned SSIMULACRA2 and selected margarine peers.
# Run on the eligible measurement host, inside its shared run-heavy lock.
a-speed-build:
    cargo build --release --manifest-path zensim-bench/a-speed-runtime/Cargo.toml --bins --benches

# Public A's serial/parallel crossover; same paired instrument and admission.
a-parallel-gate-collect root binary threads sizes:
    env A_SPEED_PARALLEL_GATE=1 BENCH_CPUS=0-15 BENCH_THREADS={{quote(threads)}} BENCH_SIZES={{quote(sizes)}} BENCH_MODES='latency scores' bash zensim-bench/a-speed-runtime/run.sh {{quote(root)}} {{quote(binary)}}

a-parallel-gate-build root:
    #!/usr/bin/env bash
    set -euo pipefail
    cd {{quote(root)}}
    cargo build --release --manifest-path source/a-speed-runtime/Cargo.toml --bins --benches --message-format=json

a-parallel-gate-check root phase:
    python3 -B zensim-bench/a-speed-runtime/check_gate.py {{quote(root)}} {{quote(phase)}}

a-parallel-gate-after root:
    python3 -B zensim-bench/a-speed-runtime/after_gate.py {{quote(root)}}

a-parallel-gate-fixture-oracle root:
    python3 -B zensim-bench/a-speed-runtime/oracle_fixtures.py {{quote(root)}}

# Run under the shared compute lock, run-heavy and the original CPU affinity.
a-parallel-gate-rebuild-checker root:
    python3 -B zensim-bench/a-speed-runtime/rebuild_public_checker.py {{quote(root)}}

a-parallel-gate-report root output phase:
    python3 -B zensim-bench/a-speed-runtime/report.py {{quote(root)}} {{quote(output)}} --parallel-gate {{quote(phase)}}

# Progress view of complete contiguous baseline cells; never admits the matrix.
a-parallel-gate-crossover root output:
    python3 -B zensim-bench/a-speed-runtime/report.py {{quote(root)}} {{quote(output)}} --crossover-prefix

a-parallel-gate-progress host root *args:
    ssh {{quote(host)}} python3 -B {{quote(root + "/source/a-speed-runtime/progress.py")}} {{quote(root)}} {{args}}

a-parallel-gate-host host:
    ssh {{quote(host)}} 'hostname; uptime; free -h; df -h /home; lslocks -o PID,COMMAND,PATH | rg "(heavy|host)[.]lock" || true; ps -eo pid,ppid,etimes,pcpu,comm,args --sort=-pcpu | sed -n "1,8p"'

a-parallel-gate-reference bake revision='7':
    env ZENSIM_FORMULA_REV={{quote(revision)}} ZENSIM_POW_FORM=libm cargo run --release -p zensim --features custom-profiles,training,classification --example split_parity_golden -- --public-a {{quote(bake)}} {{quote(bake)}}

a-parallel-gate-parity root reference7 reference1 public output:
    python3 -B zensim-bench/a-speed-runtime/report.py {{quote(root)}} {{quote(output)}} --split-parity {{quote(reference7)}} {{quote(reference1)}} {{quote(public)}}

a-parallel-gate-test:
    cargo test -p zensim -- --test-threads=1

a-parallel-gate-hdr-test:
    cargo test -p zensim --features custom-profiles --test native_hdr_color -- --test-threads=1

a-parallel-gate-thread-test:
    env ZENSIM_FORMULA_REV=1 cargo test -p zensim --features training,custom-profiles --test research_engine_parity research_output_is_thread_invariant -- --exact --nocapture --test-threads=1
    env ZENSIM_FORMULA_REV=4 cargo test -p zensim --features training,custom-profiles --test research_engine_parity research_output_is_thread_invariant -- --exact --nocapture --test-threads=1

a-parallel-gate-bake-thread-test bake:
    env ZENSIM_FORMULA_REV=7 cargo run --release -p zensim --features training,custom-profiles,classification --example a_parallel_gate_threads -- {{quote(bake)}}

# Requires the outer shared lock and resource wrapper; keeps release controls.
a-parallel-gate-reclaim root label:
    python3 -B zensim-bench/a-speed-runtime/reclaim_cargo.py {{quote(root)}} {{quote(label)}}

a-parallel-gate-admission-test:
    python3 -B -m unittest discover -s zensim-bench/a-speed-runtime -p 'test_*.py'

# Whole-process hardware counters over the same tiny score pairs, before/after.
# The caller supplies the shared compute lock and run-heavy resource scope.
a-parallel-gate-profile root phase:
    test {{quote(phase)}} = before || test {{quote(phase)}} = after
    env A_SPEED_PARALLEL_GATE=1 RAYON_NUM_THREADS=16 perf stat -e cycles,instructions,context-switches,cpu-migrations,page-faults -o {{quote(root + "/" + phase + "/perf-small.txt")}} -- taskset -c 0-15 {{quote(root + "/" + phase + "/bin/a-speed-peers")}} {{quote(root + "/fixtures")}} 16 64 scores

a-speed-prepare sources output:
    zensim-bench/a-speed-runtime/target/release/a-speed-prepare {{quote(sources)}} {{quote(output)}}

a-speed-collect root binary cpus:
    env BENCH_CPUS={{quote(cpus)}} bash zensim-bench/a-speed-runtime/run.sh {{quote(root)}} {{quote(binary)}}

a-speed-report root output:
    python3 -B zensim-bench/a-speed-runtime/report.py {{quote(root)}} {{quote(output)}} --command-startup

# R12 registered linear-head fits; the shared lock serializes resource-heavy work.
r12-linear-fits:
    flock ~/tmp/heavy.lock ~/work/claudehints/scripts/run-heavy --mem 24G -- bash scripts/byconstruction/r12_linear_fits.sh

r12-linear-reads models output:
    flock ~/tmp/heavy.lock ~/work/claudehints/scripts/run-heavy --mem 24G --jobs 16 -- python3 -B scripts/byconstruction/r12_linear_reads.py --models {{quote(models)}} --out {{quote(output)}}

r12-linear-dev output:
    flock ~/tmp/heavy.lock ~/work/claudehints/scripts/run-heavy --mem 24G --jobs 8 -- python3 -B scripts/byconstruction/r12_linear_dev.py {{quote(output)}}

r12-linear-native-check:
    flock ~/tmp/heavy.lock ~/work/claudehints/scripts/run-heavy --mem 24G --jobs 8 -- env TMPDIR="$HOME/tmp/zensim-linear" CARGO_TARGET_DIR=/var/tmp/byc-linear-20261104/target cargo test --release -p zensim-validate --bin zensim_mlp_train subjective_training_leg::tests

r12-linear-head-build:
    flock ~/tmp/heavy.lock ~/work/claudehints/scripts/run-heavy --mem 24G --jobs 8 -- env TMPDIR="$HOME/tmp/zensim-linear" CARGO_TARGET_DIR=/var/tmp/byc-linear-20261104/target cargo build --release -p zensim-validate --example prune_forward_bench

r12-linear-status:
    python3 -B scripts/byconstruction/r12_linear_status.py


r12-linear-continuation:
    flock ~/tmp/heavy.lock ~/work/claudehints/scripts/run-heavy --mem 24G -- bash scripts/byconstruction/r12_linear_continuation.sh

r12-linear-freeze:
    flock ~/tmp/heavy.lock ~/work/claudehints/scripts/run-heavy --mem 24G --jobs 8 -- python3 -B scripts/byconstruction/r12_linear_freeze.py

r12-linear-heads:
    flock ~/tmp/heavy.lock ~/work/claudehints/scripts/run-heavy --mem 24G --jobs 8 -- bash scripts/byconstruction/r12_linear_heads.sh

r12-linear-clippy:
    flock ~/tmp/heavy.lock ~/work/claudehints/scripts/run-heavy --mem 24G --jobs 8 -- env TMPDIR="$HOME/tmp/zensim-linear" CARGO_TARGET_DIR=/var/tmp/byc-linear-20261104/target just clippy


viz-design2-verify root out:
    python3 -B tools/zensim-viz/verify_design2.py {{quote(root)}} {{quote(out)}}

a-speed-report-check:
    python3 -B zensim-bench/a-speed-runtime/test_report.py
    bash -n zensim-bench/a-speed-runtime/run.sh

a-speed-progress host root:
    ssh {{quote(host)}} python3 -B {{quote(root + "/harness/a-speed-runtime/progress.py")}} {{quote(root)}}

# The caller supplies the same resource wrapper, affinity and thread environment.
a-speed-cli-startup binary output:
    zensim-bench/a-speed-runtime/target/release/a-speed-cli-startup {{quote(binary)}} {{quote(output)}}

a-speed-artifact-manifest root:
    python3 -B zensim-bench/a-speed-runtime/artifact_manifest.py {{quote(root)}}

a-speed-command-startup root instrument:
    bash zensim-bench/a-speed-runtime/startup.sh {{quote(root)}} {{quote(instrument)}}

r12-linear-extra-build:
    flock ~/tmp/heavy.lock ~/work/claudehints/scripts/run-heavy --mem 24G --jobs 8 -- bash scripts/byconstruction/r12_linear_extra_build.sh

r12-linear-extra-fits:
    flock ~/tmp/heavy.lock ~/work/claudehints/scripts/run-heavy --mem 24G --jobs 8 -- bash scripts/byconstruction/r12_linear_extra_fits.sh

r12-pruning-proof:
    flock ~/tmp/heavy.lock ~/work/claudehints/scripts/run-heavy --mem 24G --jobs 8 -- python3 -B scripts/byconstruction/r12_pruning_proof.py --root /var/tmp/byc-linear-20261104


r12-linear-extra-heads:
    flock ~/tmp/heavy.lock ~/work/claudehints/scripts/run-heavy --mem 24G --jobs 8 -- bash scripts/byconstruction/r12_linear_heads.sh /var/tmp/byc-linear-20261104/extra

r12-pruning-pixel-proof:
    python3 -B scripts/byconstruction/r12_pruning_pixel_proof.py


r12-linear-head-summary root:
    python3 -B scripts/byconstruction/r12_linear_head_summary.py --root {{quote(root)}}

r12-source-rows-check:
    flock ~/tmp/heavy.lock ~/work/claudehints/scripts/run-heavy --mem 24G --jobs 8 -- env TMPDIR="$HOME/tmp/zensim-linear" CARGO_TARGET_DIR=/var/tmp/byc-linear-20261104/target cargo test --release -p zensim --all-features --lib padded_rows_subset_and_strip_producer

r12-linear-tables output:
    python3 -B scripts/byconstruction/r12_linear_tables.py --root /var/tmp/byc-linear-20261104 --with-extra --out {{quote(output)}}

r12-linear-head-receipt:
    python3 -B scripts/byconstruction/r12_linear_head_summary.py --root /var/tmp/byc-linear-20261104 --receipt benchmarks/r12_head_timing_2026-11-04.json

# All review-2 builds and captures serialize with training/evaluation.
viz-review2-browser:
    set -o pipefail; TMPDIR="$HOME/tmp/zensim-viz" flock "$HOME/tmp/heavy.lock" "$HOME/work/claudehints/scripts/run-heavy" --mem 8G --jobs 4 -- node tools/zensim-viz/tests/review2.mjs 2>&1 | tee "$HOME/tmp/zensim-viz/review2-browser.log"

viz-review2-verify root out prefix="review2-":
    python3 -B tools/zensim-viz/verify_design2.py {{quote(root)}} {{quote(out)}} --prefix {{quote(prefix)}}



r12-linear-final-check:
    flock ~/tmp/heavy.lock ~/work/claudehints/scripts/run-heavy --mem 24G --jobs 8 -- bash scripts/byconstruction/r12_linear_final_check.sh

r12-linear-archive:
    flock ~/tmp/heavy.lock ~/work/claudehints/scripts/run-heavy --mem 24G --jobs 8 -- bash scripts/byconstruction/r12_linear_archive.sh

# R13 takes the resource lock separately inside each base/continuation/pack job.
r13-width-fits:
    python3 -B scripts/byconstruction/r13_width.py fits

r13-width-freeze:
    python3 -B scripts/byconstruction/r13_width_finalize.py

r13-width-native-check:
    flock ~/tmp/heavy.lock ~/work/claudehints/scripts/run-heavy --mem 24G --jobs 8 -- env TMPDIR="$HOME/tmp/zensim-width" CARGO_TARGET_DIR=/var/tmp/byc-width-20261104/target cargo test --release -p zensim-validate --bin zensim_mlp_train subjective_training_leg::tests

r13-width-build:
    flock ~/tmp/heavy.lock ~/work/claudehints/scripts/run-heavy --mem 24G --jobs 8 -- env TMPDIR="$HOME/tmp/zensim-width" CARGO_TARGET_DIR=/var/tmp/byc-width-20261104/target cargo build --release -p zensim-validate --bin zensim_mlp_train

r13-width-w4-build:
    flock ~/tmp/heavy.lock ~/work/claudehints/scripts/run-heavy --mem 24G --jobs 8 -- bash scripts/byconstruction/r13_width_w4_build.sh

r13-width-clippy:
    flock ~/tmp/heavy.lock ~/work/claudehints/scripts/run-heavy --mem 24G --jobs 8 -- env TMPDIR="$HOME/tmp/zensim-width" CARGO_TARGET_DIR=/var/tmp/byc-width-20261104/target just clippy

r13-width-python-check:
    env TMPDIR="$HOME/tmp/zensim-width" python3 -B -m unittest scripts.tests.test_v2c.TrainRecipe.test_recipe_tokens
    env TMPDIR="$HOME/tmp/zensim-width" python3 -B -m unittest discover -s scripts/tests -p test_subjective_training_leg.py
    env TMPDIR="$HOME/tmp/zensim-width" python3 -B -m unittest discover -s scripts/tests -p test_r13_width.py
    env TMPDIR="$HOME/tmp/zensim-width" python3 -B -m unittest discover -s scripts/tests -p test_e36_primary_inventory.py

r13-width-capacity:
    python3 -B scripts/byconstruction/r13_width_analysis.py capacity

r13-width-development:
    python3 -B scripts/byconstruction/r13_width_analysis.py development

r13-width-heads:
    python3 -B scripts/byconstruction/r13_width_analysis.py heads

r13-width-reads:
    python3 -B scripts/byconstruction/r13_width_reads.py

r13-width-tables output:
    python3 -B scripts/byconstruction/r12_linear_tables.py --root /var/tmp/byc-width-20261104 --arms F,W4,W16,W32,W64,W256,W512 --candidate-seeds 0 --title 'R13 measured read tables' --out {{quote(output)}}

r13-width-status:
    python3 -B scripts/byconstruction/r13_width_status.py

r13-width-status-compact:
    python3 -B scripts/byconstruction/r13_width_status.py --compact

r13-width-fit-progress:
    python3 -B scripts/byconstruction/r13_width_fit_progress.py --out benchmarks/r13_fit_progress_2026-11-04.json

r13-width-report output:
    python3 -B scripts/byconstruction/r13_width_report.py --out {{quote(output)}}

r13-width-validate:
    flock ~/tmp/heavy.lock ~/work/claudehints/scripts/run-heavy --mem 24G --jobs 8 -- python3 -B scripts/byconstruction/r13_width_validate.py

r13-width-archive:
    python3 -B -c 'import sys; sys.path.insert(0,"scripts/byconstruction"); import r13_width as r; r.job("archive-final",["bash",r.X/"scripts/byconstruction/r13_width_archive.sh"])'



a-family-cost-report study out host:
    python3 scripts/bench/a_family_cost_report.py {{quote(study)}} {{quote(out)}} --host {{quote(host)}}

a-family-cost-perf study out host:
    python3 scripts/bench/a_family_cost_perf.py {{quote(study)}} {{quote(out)}} --host {{quote(host)}}

a-family-cost-progress study:
    python3 scripts/bench/a_family_cost_progress.py {{quote(study)}}

a-family-cost-select study view *args:
    python3 scripts/bench/a_family_cost_select.py {{quote(study)}} {{quote(view)}} {{args}}

a-family-cost-jpeg-check study out:
    python3 scripts/bench/a_family_cost_jpeg_check.py {{quote(study)}} {{quote(out)}}

a-family-cost-verify view summary out:
    python3 scripts/bench/a_family_cost_verify.py {{quote(view)}} {{quote(summary)}} {{quote(out)}}

a-family-cost-shared study:
    python3 scripts/bench/a_family_cost_shared_run.py {{quote(study)}}

# The caller supplies the measurement host's flock/run-heavy wrapper.
a-family-cost-copy source destination:
    rsync -a --no-owner --no-group --exclude scratch --exclude __pycache__ --exclude .workongoing --exclude control --rsync-path='nice -n19 ionice -c3 rsync' {{quote(source)}}/ {{quote(destination)}}/


# R14 native contract checks; callers select their resource wrapper/host.
r14-transform-checks:
    cargo fmt --check -p zensim-validate -p zensim
    cargo test -p zensim-validate --lib mlp_train::transforms -- --test-threads=1
    cargo test -p zensim-validate --lib mlp_train::warm_start -- --test-threads=1
    cargo check -p zensim-validate --bin zensim_mlp_train

# TRAIN screening is serialized with other research jobs on the fitting host.
r14-transform-screen:
    flock ~/tmp/heavy.lock ~/work/claudehints/scripts/run-heavy --mem 24G --jobs 8 -- python3 -B scripts/byconstruction/r14_transforms_screen.py

r14-certificate-checks:
    cargo test -p zensim-validate --lib mlp_train::transforms -- --test-threads=1
    cargo test -p zensim-validate --lib mlp_train::warm_start -- --test-threads=1
    cargo check -p zensim-validate --bin bake_dial_refit --bin zensim_mlp_train

r14-continuation-checks:
    cargo fmt --check -p zensim-validate
    cargo test -p zensim-validate --bin zensim_mlp_train capability_preflight_tests -- --test-threads=1
    cargo clippy -p zensim-validate --bin zensim_mlp_train

r14-t2-checks:
    cargo fmt --check -p zensim-validate
    cargo test -p zensim-validate --lib mlp_train::r14_knees -- --test-threads=1
    cargo test -p zensim-validate --lib mlp_train::transforms -- --test-threads=1
    cargo test -p zensim-validate --lib mlp_train::warm_start -- --test-threads=1
    cargo test -p zensim-validate --bin zensim_mlp_train r14_authority -- --test-threads=1
    cargo clippy -p zensim-validate --lib --bin zensim_mlp_train

r14-transform-build:
    python3 -B scripts/byconstruction/r14_transforms.py job --label final-build -- bash scripts/byconstruction/r14_transforms_build.sh

r14-transform-fit arm:
    python3 -B scripts/byconstruction/r14_transforms.py fit --arm {{arm}}

r14-trainer-build:
    python3 -B scripts/byconstruction/r14_transforms.py job --label selected-caller-trainer-build -- bash scripts/byconstruction/r14_trainer_build.sh

r14-inspector-build:
    python3 -B scripts/byconstruction/r14_transforms.py job --label transformed-inspector-build -- bash scripts/byconstruction/r14_inspector_build.sh

r14-recover-pack:
    python3 -B scripts/byconstruction/r14_pack_recover.py

r14-transform-retry-continuation arm:
    python3 -B scripts/byconstruction/r14_transforms.py fit --arm {{arm}} --continuation-retry r2

r14-transform-t2:
    python3 -B scripts/byconstruction/r14_t2.py prepare
    python3 -B scripts/byconstruction/r14_t2.py fit
    python3 -B scripts/byconstruction/r14_t2.py freeze

r14-transform-status:
    python3 -B scripts/byconstruction/r14_transforms_status.py --compact

r14-python-checks:
    python3 -B -c 'import ast,pathlib; paths=list(pathlib.Path("scripts/byconstruction").glob("r14_*.py"))+list(pathlib.Path("scripts/rev4_featpot").glob("r14_*.py")); [ast.parse(p.read_text(),filename=str(p)) for p in paths]; print(f"Parsed {len(paths)} R14 Python owners")'
    bash -n scripts/byconstruction/r14_inspector_build.sh scripts/byconstruction/r14_trainer_build.sh scripts/byconstruction/r14_transforms_build.sh
    python3 -B scripts/byconstruction/r14_transforms.py --help
    python3 -B scripts/byconstruction/r14_t2.py --help

r14-transform-heads:
    python3 -B scripts/byconstruction/r14_heads.py

r14-head-status:
    python3 -B scripts/byconstruction/r14_heads.py status

r14-head-summary:
    python3 -B scripts/byconstruction/r14_head_summary.py

r14-native-noop:
    python3 -B scripts/byconstruction/r14_transforms.py job --label native-noop -- python3 -B scripts/byconstruction/r14_transforms_noop.py

r14-transform-capacity:
    python3 -B scripts/byconstruction/r14_transforms.py capacity

r14-transform-reads:
    python3 -B scripts/byconstruction/r14_transforms.py reads

r14-transform-tables:
    python3 -B scripts/byconstruction/r14_tables.py
    python3 -B scripts/byconstruction/r14_summary.py
    python3 -B scripts/byconstruction/r14_width_comparison.py

r14-transform-archive:
    flock ~/tmp/heavy.lock ~/work/claudehints/scripts/run-heavy --mem 24G --jobs 8 -- bash scripts/byconstruction/r13_width_archive.sh /var/tmp/byc-transforms-20261104 /mnt/tower/output/zensim/r14-transforms-20261104

r14-serving-checks:
    cargo test -p zensim --features custom-profiles,training,classification --lib r14_transform_tests -- --test-threads=1
    cargo test -p zensim-validate --lib derived_inputs -- --test-threads=1
    cargo test --manifest-path tools/zensim-viz/Cargo.toml r14_tests -- --test-threads=1

# Run only on i270 inside its shared flock/run-heavy wrapper.
[positional-arguments]
a-fusion-report stage root output:
    #!/usr/bin/env bash
    set -euo pipefail
    test "$(hostname)" = i270
    if [[ $1 = progress ]]; then
        exec python3 -B scripts/bench/a_extraction_fusion.py "$@"
    fi
    exec flock "$HOME/tmp/heavy.lock" "$HOME/work/claudehints/scripts/run-heavy" --mem 16G --jobs 8 -- python3 -B scripts/bench/a_extraction_fusion.py "$@"

[positional-arguments]
a-fusion-compare before after pairs bake size threads output:
    #!/usr/bin/env bash
    set -euo pipefail
    test "$(hostname)" = i270
    case "$6" in 1) cpus=2 ;; 16) cpus=0-15 ;; *) echo 'choose 1 or 16 threads' >&2; exit 2 ;; esac
    exec flock "$HOME/tmp/heavy.lock" "$HOME/work/claudehints/scripts/run-heavy" --mem 16G --jobs 16 -- env ZENSIM_FORMULA_REV=7 ZENBENCH_NO_CALIBRATE=1 RAYON_NUM_THREADS="$6" python3 -B scripts/bench/a_family_cost_admit.py "$7/admission.json" -- taskset -c "$cpus" "$1" fusion compare "$1" "$2" "$3" "$4" "$5" "$6" "$7"

a-fusion-native stage study label:
    bash scripts/bench/a_fusion_native.sh {{quote(stage)}} {{quote(study)}} {{quote(label)}}


# Freeze EVAL RGB ingress once; old/new binaries read the same hash-bound bytes.
a-fusion-freeze-eval binary table output:
    test "$(hostname)" = i270
    flock "$HOME/tmp/heavy.lock" "$HOME/work/claudehints/scripts/run-heavy" --mem 16G -- '{{binary}}' fusion prepare-eval '{{table}}' '{{output}}'

# Optional six-pair detector-map proof, serialized on the measurement host.
a-fusion-maps study label:
    bash scripts/bench/a_fusion_maps.sh {{quote(study)}} {{quote(label)}}

# One complete gate, pixel or timing phase under the shared native heavy lock.
a-fusion-trial phase study label snapshot:
    bash scripts/bench/a_fusion_trial.sh {{quote(phase)}} {{quote(study)}} {{quote(label)}} {{quote(snapshot)}}

# Owner-directed precise miss and deeper Topdown attribution on the Intel host.
a-fusion-stalls study label:
    bash scripts/bench/a_fusion_stalls.sh {{quote(study)}} {{quote(label)}}

# Analyze admitted Intel captures, preserving raw event-count units.
a-fusion-stall-report capture output:
    flock "$HOME/tmp/heavy.lock" "$HOME/work/claudehints/scripts/run-heavy" --mem 16G --jobs 2 -- python3 -B scripts/bench/a_fusion_stall_report.py {{quote(capture)}} {{quote(output)}}

# Resolve a zero-sample 256-square L3 capture without changing its original data.
a-fusion-stall-confirm study label:
    bash scripts/bench/a_fusion_stalls.sh {{quote(study)}} {{quote(label)}} confirm-l3

# Add event-count samples where initial short windows had insufficient events.
a-fusion-stall-sparse study label:
    bash scripts/bench/a_fusion_stalls.sh {{quote(study)}} {{quote(label)}} confirm-sparse

# Short owner-authorized native batches; release the host lock between cells.
a-fusion-host-cell stage study label arg1 arg2="":
    bash scripts/bench/a_fusion_host_cell.sh {{quote(stage)}} {{quote(study)}} {{quote(label)}} {{quote(arg1)}} {{quote(arg2)}}

# One profile cell, so training/evaluation can interleave between AMD windows.
a-fusion-amd-profile study label size kind:
    bash scripts/bench/a_fusion_amd_profile.sh {{quote(study)}} {{quote(label)}} {{quote(size)}} {{quote(kind)}}

# Copy and hash-check the existing packet under dev's shared resource lock.
a-fusion-setup-dev study archive:
    bash scripts/bench/a_fusion_setup_dev.sh {{quote(study)}} {{quote(archive)}}

# Strict AMD attribution from individual, non-multiplexed metric windows.
fusion-amd-report study output:
    flock "$HOME/tmp/heavy.lock" "$HOME/work/claudehints/scripts/run-heavy" --mem 24G --jobs 1 -- python3 -B scripts/bench/a_fusion_amd_report.py {{quote(study)}} {{quote(output)}}

# Check tool-duration scope against a FIFO-controlled hardware counter.
fusion-counter-control study output:
    bash scripts/bench/a_fusion_counter_control.sh {{quote(study)}} {{quote(output)}}

# Retry analysis in a new attempt directory; never rerun the capture.
a-fusion-parse study stage input output:
    bash scripts/bench/a_fusion_parse.sh {{quote(study)}} {{quote(stage)}} {{quote(input)}} {{quote(output)}}

# Manual owner-requested recipe for the nine unmeasured Intel cells; never auto-start.
a-fusion-idle-intel study:
    bash scripts/bench/a_fusion_idle_intel.sh {{quote(study)}}

# Exact native function terms and named feature navigation; retains review-2 gates.
viz-function-browser:
    set -o pipefail; TMPDIR="$HOME/tmp/zensim-viz" flock "$HOME/tmp/heavy.lock" "$HOME/work/claudehints/scripts/run-heavy" --mem 8G --jobs 4 -- node tools/zensim-viz/tests/function.mjs 2>&1 | tee "$HOME/tmp/zensim-viz/function-browser.log"

# Archive data without requiring ownership changes on a shared destination.
viz-function-archive root destination:
    mkdir -p {{quote(destination)}}
    set -o pipefail; TMPDIR="$HOME/tmp/zensim-viz" flock "$HOME/tmp/heavy.lock" "$HOME/work/claudehints/scripts/run-heavy" --mem 2G --jobs 1 -- rsync -rlt --ignore-existing {{quote(root)}}/function-affine-* {{quote(root)}}/function-screenshots {{quote(root)}}/function-preservation.json tools/zensim-viz/function-evidence.json tools/zensim-viz/REPORT.md {{quote(destination)}}/ 2>&1 | tee "$HOME/tmp/zensim-viz/function-archive.log"


# Frozen-A assessment tools; caller supplies the disk-backed study root.
final-read-progress root:
    @python3 -B scripts/byconstruction/final_read_progress.py {{quote(root)}} --brief

final-read-native-build root:
    env TMPDIR={{quote(root + "/scratch")}} CARGO_TARGET_DIR={{quote(root + "/target/decode")}} flock {{env_var('HOME')}}/tmp/heavy.lock {{env_var('HOME')}}/work/claudehints/scripts/run-heavy --mem 24G --jobs 8 -- cargo build --release --locked --manifest-path scripts/byconstruction/trust_decode/Cargo.toml

final-read-panel-build root:
    env TMPDIR={{quote(root + "/scratch")}} CARGO_TARGET_DIR={{quote(root + "/target/assessment")}} flock {{env_var('HOME')}}/tmp/heavy.lock {{env_var('HOME')}}/work/claudehints/scripts/run-heavy --mem 24G --jobs 8 -- cargo build --release --locked -p zensim-validate --bin panel

final-read-native-serving-build root:
    env TMPDIR={{quote(root + "/scratch")}} CARGO_TARGET_DIR={{quote(root + "/target/assessment")}} flock {{env_var('HOME')}}/tmp/heavy.lock {{env_var('HOME')}}/work/claudehints/scripts/run-heavy --mem 24G --jobs 8 -- cargo build --release --locked -p zensim --features training,custom-profiles,classification --example serve_custom_bake

final-read-step action population root:
    env TMPDIR={{quote(root + "/scratch")}} PYTHONDONTWRITEBYTECODE=1 flock {{env_var('HOME')}}/tmp/heavy.lock {{env_var('HOME')}}/work/claudehints/scripts/run-heavy --mem 24G --jobs {{if action == "peers" { "8" } else { "1" }}} -- python3 -B scripts/byconstruction/final_read_a.py {{quote(action)}} {{quote(population)}} {{quote(root)}}

final-read-panel-test root:
    env TMPDIR={{quote(root + "/scratch")}} CARGO_TARGET_DIR={{quote(root + "/target/assessment")}} flock {{env_var('HOME')}}/tmp/heavy.lock {{env_var('HOME')}}/work/claudehints/scripts/run-heavy --mem 24G --jobs 8 -- cargo test --release --locked -p zensim-validate --bin panel tests::signed_reference_resamples_preserve_negative_direction_and_ties -- --exact --test-threads=1

final-read-unit root:
    env TMPDIR={{quote(root + "/scratch")}} PYTHONDONTWRITEBYTECODE=1 flock {{env_var('HOME')}}/tmp/heavy.lock {{env_var('HOME')}}/work/claudehints/scripts/run-heavy --mem 24G --jobs 1 -- python3 -B scripts/tests/test_final_read_a.py

final-read-control root:
    env TMPDIR={{quote(root + "/scratch")}} PYTHONDONTWRITEBYTECODE=1 flock {{env_var('HOME')}}/tmp/heavy.lock {{env_var('HOME')}}/work/claudehints/scripts/run-heavy --mem 24G --jobs 8 -- python3 -B scripts/tests/test_final_read_a.py --panel-control {{quote(root)}}

final-read-stats population root:
    env TMPDIR={{quote(root + "/scratch")}} PYTHONDONTWRITEBYTECODE=1 flock {{env_var('HOME')}}/tmp/heavy.lock {{env_var('HOME')}}/work/claudehints/scripts/run-heavy --mem 24G --jobs 8 -- python3 -B scripts/byconstruction/final_read_a.py stats {{quote(population)}} {{quote(root)}}

final-read-native-check root:
    env TMPDIR={{quote(root + "/scratch")}} CARGO_TARGET_DIR={{quote(root + "/target/decode")}} flock {{env_var('HOME')}}/tmp/heavy.lock {{env_var('HOME')}}/work/claudehints/scripts/run-heavy --mem 24G --jobs 8 -- bash -c 'cargo test --release --locked --manifest-path scripts/byconstruction/trust_decode/Cargo.toml && cargo clippy --release --locked --manifest-path scripts/byconstruction/trust_decode/Cargo.toml -- -D warnings'

final-read-mirror-check root:
    env TMPDIR={{quote(root + "/scratch")}} PYTHONDONTWRITEBYTECODE=1 python3 -B scripts/tests/test_dataset_mirror.py

final-read-mirror root host destination capacity_mount image:
    env TMPDIR={{quote(root + "/scratch")}} PYTHONDONTWRITEBYTECODE=1 flock {{env_var('HOME')}}/tmp/heavy.lock {{env_var('HOME')}}/work/claudehints/scripts/run-heavy --mem 24G --jobs 1 -- python3 -B scripts/byconstruction/dataset_mirror.py --root {{quote(root)}} --host {{quote(host)}} --destination {{quote(destination)}} --capacity-path {{quote(capacity_mount)}} /mnt/cache --artifact-kind evaluation --image {{quote(image)}}

# Preparatory shared pixel graph; frozen runtime and model inputs stay pinned.
sibling-prep-check:
    cargo update -p zenflate
    cargo metadata --format-version 1 > /dev/null
    cargo clippy --workspace --all-targets --all-features --exclude zensim-wasm-tests -- -D warnings

sibling-prep-regress-check:
    cargo test -p zensim-regress --locked --lib --tests
    cargo test -p zensim-regress --locked --doc

# Source identity is stamped separately after the resolved lock is snapshotted.
sibling-prep-build commit:
    ZENSIM_BUILD_COMMIT={{quote(commit)}} RUSTFLAGS='-C force-frame-pointers=yes' cargo build --release --locked -p zensim --example a_family_cost --example split_parity_golden --features training,custom-profiles,classification

# Shared metadata entry points delegate to the same retained EXIF implementation.
sibling-prep-metadata-check:
    cargo test -p zensim --locked --example e34_pixels --features training,custom-profiles,classification

# Complete frozen source replay; input tables and source images are read in place.
sibling-prep-eval binary pairs bake capture:
    ZENSIM_FORMULA_REV=7 RAYON_NUM_THREADS=1 {{quote(binary)}} fusion pairs {{quote(pairs)}} {{quote(bake)}} {{quote(capture)}}

sibling-prep-split binary general codec golden baseline:
    ZENSIM_FORMULA_REV=7 {{quote(binary)}} {{quote(general)}} {{quote(codec)}} > {{quote(golden)}}
    cmp {{quote(baseline)}} {{quote(golden)}}

sibling-prep-frozen-compare root report:
    python3 -B scripts/bench/a_extraction_fusion.py eval-baseline {{quote(root)}} {{quote(report)}}
    python3 -c 'import json,sys; row=json.load(open(sys.argv[1])); assert row["rows"] == 4678 and row["equal"] and not row["differences"], (row["rows"],row["equal"],len(row["differences"]))' {{quote(report)}}

# Existing strict source-capture comparison also checks decoded pixel hashes.
sibling-prep-pixel-compare root report:
    python3 -B scripts/bench/a_extraction_fusion.py eval-bits {{quote(root)}} {{quote(report)}}

# Runtime and optional shared-pixel ingress behavior after the graph update.
sibling-prep-metric-check:
    RUST_TEST_THREADS=1 cargo test -p zensim --locked --lib --tests --features training,custom-profiles,classification,zenpixels
    cargo test -p zensim --locked --doc --features training,custom-profiles,classification,zenpixels

# The existing three-tier oracle must also work with serial libtest formatting.
sibling-prep-steer-check:
    RUST_TEST_THREADS=1 cargo test -p zensim --locked --lib --features training,custom-profiles,classification,zenpixels metric::bake::steer_exact::tests::steer_smooth_native_tier_stride_parity -- --exact --nocapture

# Existing revision-vector oracles also retain their receipts in serial libtest.
sibling-prep-vector-check:
    RUST_TEST_THREADS=1 cargo test -p zensim --locked --test featcanon_rev7_parity --test featcanon_tier_parity --features training,custom-profiles,classification,zenpixels -- --nocapture

viz-pages-check bake target:
    flock "$HOME/tmp/heavy.lock" "$HOME/work/claudehints/scripts/run-heavy" --mem 16G --jobs 6 -- env TMPDIR="$HOME/tmp/zensim-viz" ZENSIM_WORKBENCH_BAKES="A={{bake}}" CARGO_TARGET_DIR={{quote(target)}} cargo check --release --manifest-path zensim-workbench/Cargo.toml --features pages-viz

# Keep browser/native builds behind the shared evaluation lock; never store binaries in git.
viz-pages-module bake target out wasm_bindgen:
    #!/usr/bin/env bash
    set -euo pipefail
    export TMPDIR="$HOME/tmp/zensim-viz" ZENSIM_WORKBENCH_BAKES="A={{bake}}" ZENSIM_WORKBENCH_LABEL="zensim A revision 7 browser" CARGO_TARGET_DIR={{quote(target)}}
    flock "$HOME/tmp/heavy.lock" "$HOME/work/claudehints/scripts/run-heavy" --mem 16G --jobs 6 -- env RUSTFLAGS="-C target-feature=+simd128 -C strip=symbols --remap-path-prefix=$HOME=deps --remap-path-prefix=$PWD=zensimx --remap-path-prefix=$CARGO_TARGET_DIR=target" cargo build --release --locked --manifest-path zensim-workbench/Cargo.toml --features pages-viz --target wasm32-unknown-unknown --lib
    {{quote(wasm_bindgen)}} --target web --out-dir {{quote(out)}}/pkg "$CARGO_TARGET_DIR/wasm32-unknown-unknown/release/zensim_workbench.wasm"

viz-pages-native bake target out:
    #!/usr/bin/env bash
    set -euo pipefail
    unset ZENSIM_VIZ_MAP_DIR
    export TMPDIR="$HOME/tmp/zensim-viz" ZENSIM_FORMULA_REV=7 ZENSIM_WORKBENCH_BAKES="A={{bake}}" CARGO_TARGET_DIR={{quote(target)}}
    flock "$HOME/tmp/heavy.lock" "$HOME/work/claudehints/scripts/run-heavy" --mem 16G --jobs 6 -- cargo run --release --locked --manifest-path zensim-workbench/Cargo.toml --features pages-viz --example viz_parity -- {{quote(bake)}} {{quote(out)}}

viz-pages-export mode bake bank out certifier:
    flock "$HOME/tmp/heavy.lock" "$HOME/work/claudehints/scripts/run-heavy" --mem 16G --jobs 6 -- env TMPDIR="$HOME/tmp/zensim-viz" ZENSIM_FORMULA_REV=7 VIZ_CERTIFIER={{quote(certifier)}} cargo run --release --manifest-path tools/zensim-viz/Cargo.toml -- {{quote(mode)}} {{quote(bake)}} --bank {{quote(bank)}} --out {{quote(out)}}

viz-pages-inventory:
    python3 tools/zensim-viz/r2_pages.py inventory

viz-pages-packer target:
    flock "$HOME/tmp/heavy.lock" "$HOME/work/claudehints/scripts/run-heavy" --mem 16G --jobs 6 -- env TMPDIR="$HOME/tmp/zensim-viz" CARGO_TARGET_DIR={{quote(target)}} cargo build --release --manifest-path tools/zensim-viz/pack/Cargo.toml

viz-pages-parity site native out:
    flock "$HOME/tmp/heavy.lock" "$HOME/work/claudehints/scripts/run-heavy" --mem 8G --jobs 1 -- env TMPDIR="$HOME/tmp/zensim-viz" node tools/zensim-viz/tests/pages-parity.mjs {{quote(site)}} {{quote(native)}} {{quote(out)}}

viz-pages-qualify root bake certifier bindgen:
    flock "$HOME/tmp/heavy.lock" "$HOME/work/claudehints/scripts/run-heavy" --mem 16G --jobs 6 -- bash tools/zensim-viz/qualify_pages.sh {{quote(root)}} {{quote(bake)}} {{quote(certifier)}} {{quote(bindgen)}}

viz-pages-lens-check site native out:
    flock "$HOME/tmp/heavy.lock" "$HOME/work/claudehints/scripts/run-heavy" --mem 8G --jobs 1 -- env TMPDIR="$HOME/tmp/zensim-viz" node tools/zensim-viz/tests/pages-lens.mjs {{quote(site)}} {{quote(native)}} {{quote(out)}}

viz-pages-smoke site out:
    flock "$HOME/tmp/heavy.lock" "$HOME/work/claudehints/scripts/run-heavy" --mem 8G --jobs 1 -- env TMPDIR="$HOME/tmp/zensim-viz" node tools/zensim-viz/tests/pages-smoke.mjs {{quote(site)}} {{quote(out)}}

viz-pages-site source out archive_url:
    python3 -B tools/zensim-viz/pages_build.py site --source {{quote(source)}} --out {{quote(out)}} --archive-url {{quote(archive_url)}}

# Resume after verified native exports; replays the lens oracle, Wasm and browser gates.
viz-pages-finish root bake bindgen:
    flock "$HOME/tmp/heavy.lock" "$HOME/work/claudehints/scripts/run-heavy" --mem 16G --jobs 6 -- bash tools/zensim-viz/finish_pages.sh {{quote(root)}} {{quote(bake)}} {{quote(bindgen)}}

# Caller supplies run-heavy/flock, TMPDIR and a task-owned CARGO_TARGET_DIR.
sibling-final-root-check:
    just clippy
    just lint-scripts

# Compile requested tools; the existing explicit flag excludes C++ benchmark
# oracle arms. No benchmark or historical encoder gate is executed.
sibling-tools-check:
    cargo check --manifest-path zensim-target/Cargo.toml --all-targets --all-features
    cargo check --manifest-path zensim-picker-prep/Cargo.toml --all-targets
    cargo check --manifest-path zensim-ultrahdr-probe/Cargo.toml --all-targets
    ZENSIM_BENCH_SKIP_CPP_FFI=1 cargo check --manifest-path zensim-bench/Cargo.toml --all-targets --features training,zen-decode,zen-decode-tests,dvifm-judge,gmsd-arm,m3-fixtures
    cargo check --manifest-path tools/zensim-viz/Cargo.toml --all-targets --features real-bake-tests
    cargo check --manifest-path tools/shipgates-target/Cargo.toml --all-targets

sibling-source-check out:
    python3 scripts/check_sibling_sources.py --out {{quote(out)}}

# Compile pixel consumers without running extraction, training or evaluation.
pixel-example-check:
    cargo check --locked -p zensim --example e34_pixels --features custom-profiles,feature-regime-v2
    cargo generate-lockfile --manifest-path zensim-bench/Cargo.toml
    cargo check --locked --manifest-path zensim-bench/Cargo.toml --example verify_bitstream_decode --example m3_fixture_gen --example gmsbank_decode_dump --features zen-decode,m3-fixtures,verify-decode

# Constrained shared-depth checks; caller holds the resource wrapper/lock.
shared-depth-check:
    cargo clippy --locked -p zensim-validate --lib --bin zensim_mlp_train -- -D warnings
    cargo test --locked -p zensim-validate --lib shared_depth -- --test-threads=1

shared-depth-regressions:
    cargo test --locked -p zensim-validate --lib mlp_train:: -- --test-threads=1
    cargo test --locked -p zensim-validate --bin zensim_mlp_train -- --test-threads=1

shared-depth-build:
    cargo build --release --locked -p zensim-validate --bin zensim_mlp_train

shared-depth-driver-tests:
    python3 -B -m unittest discover -s scripts/tests -p test_v2c.py

# Independent mathematical probes; no dataset labels or candidate scoring.
activation-curriculum-audit output:
    python3 -B scripts/rev4_featpot/activation_curriculum.py audit --output {{quote(output)}}

# Short Rust fits reuse strict admission, native baking/prediction and panels.
activation-curriculum-pilot baseline_plan output:
    TMPDIR="$HOME/tmp" RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$HOME/work/claudehints/scripts/run-heavy" --mem 8G --jobs 3 -- python3 -B scripts/rev4_featpot/activation_curriculum.py native --baseline-plan {{quote(baseline_plan)}} --output {{quote(output)}} --workers 3

activation-curriculum-tests:
    python3 -B -m unittest discover -s scripts/tests -p test_activation_curriculum.py -v

# Private monotone research heads; dense/ReLU native runtime, no new serving API.
monotone-architecture-check:
    cargo test --locked -p zensim-validate --lib monotone_research -- --test-threads=1 --nocapture
    cargo test --locked -p zensim-validate --lib shared_depth -- --test-threads=1
    cargo clippy --locked -p zensim-validate --lib --bin zensim_mlp_train -- -D warnings

monotone-training-check:
    cargo test --locked -p zensim-validate --lib training_research -- --test-threads=1
    cargo test --locked -p zensim-validate --lib monotone_research -- --test-threads=1
    cargo clippy --locked -p zensim-validate --lib --bin zensim_mlp_train -- -D warnings

monotone-architecture-screen baseline_plan binaries metadata output:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 14G --jobs 6 -- python3 -B scripts/rev4_featpot/activation_curriculum.py architectures --baseline-plan {{quote(baseline_plan)}} --binary-dir {{quote(binaries)}} --metadata {{quote(metadata)}} --output {{quote(output)}} --workers 6

monotone-architecture-long baseline_plan binaries metadata output:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 26G --jobs 12 -- python3 -B scripts/rev4_featpot/activation_curriculum.py architectures --baseline-plan {{quote(baseline_plan)}} --binary-dir {{quote(binaries)}} --metadata {{quote(metadata)}} --output {{quote(output)}} --workers 12 --long

monotone-training-run baseline_plan binaries metadata output:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 14G --jobs 6 -- python3 -B scripts/rev4_featpot/activation_curriculum.py training --baseline-plan {{quote(baseline_plan)}} --binary-dir {{quote(binaries)}} --metadata {{quote(metadata)}} --output {{quote(output)}} --workers 6

monotone-training-replay baseline_plan binaries metadata output:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 8G --jobs 3 -- python3 -B scripts/rev4_featpot/activation_curriculum.py training --baseline-plan {{quote(baseline_plan)}} --binary-dir {{quote(binaries)}} --metadata {{quote(metadata)}} --output {{quote(output)}} --workers 3 --replay-control

monotone-architecture-replay frozen binaries output:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 3G --jobs 1 -- python3 -B scripts/rev4_featpot/activation_curriculum.py replay --frozen {{quote(frozen)}} --binary-dir {{quote(binaries)}} --output {{quote(output)}}

# Content mirror only: network storage need not accept local ownership metadata.
activation-curriculum-mirror source destination:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 2G --jobs 1 -- rsync -rt {{quote(source)}}/ {{quote(destination)}}/

tree-study-tests parity_bin:
    TMPDIR="$HOME/tmp" TREE_PARITY_BIN={{quote(parity_bin)}} python3 -B -m unittest discover -s scripts/tests -p test_tree_study.py -v

tree-study baseline_plan parity_bin output:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 12G --jobs 4 -- python3 -B scripts/rev4_featpot/tree_study.py --baseline-plan {{quote(baseline_plan)}} --parity-bin {{quote(parity_bin)}} --output {{quote(output)}} --workers 4

tree-study-peers baseline_plan run output:
    TMPDIR="$HOME/tmp" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 -B scripts/rev4_featpot/tree_study_report.py --baseline-plan {{quote(baseline_plan)}} --run {{quote(run)}} --output {{quote(output)}}

tree-study-peer-search run metadata output:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 8G --jobs 8 -- python3 -B scripts/rev4_featpot/tree_study_report.py --run {{quote(run)}} --peer-search {{quote(metadata)}} --output {{quote(output)}}

tree-study-summary root output neural_single neural_shared:
    TMPDIR="$HOME/tmp" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$HOME/work/claudehints/scripts/run-heavy" --mem 4G --jobs 1 -- python3 -B scripts/rev4_featpot/tree_study_report.py --summarize --run {{quote(root)}} --output {{quote(output)}} --neural-summary {{quote(neural_single)}} --neural-summary {{quote(neural_shared)}}

tree-study-constraint-controls run output:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 6G --jobs 6 -- python3 -B scripts/rev4_featpot/tree_study_report.py --constraint-controls --run {{quote(run)}} --output {{quote(output)}}

tree-study-closure root baseline_plan:
    TMPDIR="$HOME/tmp" python3 -B scripts/rev4_featpot/tree_study_report.py --closure --run {{quote(root)}} --output {{quote(root)}} --baseline-plan {{quote(baseline_plan)}}

tree-study-mirror root destination:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 2G --jobs 1 -- just dataset-artifacts-mirror-nfs {{quote(root)}} {{quote(destination)}} training true

monotone-trial-progress root:
    python3 -B scripts/rev4_featpot/activation_curriculum.py progress --root {{quote(root)}}

monotone-neutral-compare reference candidate binary output:
    python3 -B scripts/rev4_featpot/activation_curriculum.py compare-neutral --reference {{quote(reference)}} --candidate {{quote(candidate)}} --binary {{quote(binary)}} --output {{quote(output)}}

monotone-receipt-qualification root binary output:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 1G --jobs 1 -- python3 -B scripts/rev4_featpot/activation_curriculum.py qualify-receipts --root {{quote(root)}} --binary {{quote(binary)}} --output {{quote(output)}} --watch

monotone-input-closure root baseline_plan metadata:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 1G --jobs 1 -- python3 -B scripts/rev4_featpot/activation_curriculum.py closure --root {{quote(root)}} --baseline-plan {{quote(baseline_plan)}} --metadata {{quote(metadata)}}

monotone-artifact-mirror root destination:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 2G --jobs 1 -- python3 -B scripts/byconstruction/dataset_mirror.py --root {{quote(root)}} --nfs --cold-readback --destination {{quote(destination)}} --artifact-kind training

monotone-leaky-training baseline_plan binaries metadata output:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 6G --jobs 2 -- python3 -B scripts/rev4_featpot/activation_curriculum.py training --baseline-plan {{quote(baseline_plan)}} --binary-dir {{quote(binaries)}} --metadata {{quote(metadata)}} --output {{quote(output)}} --workers 2 --leaky-switches

monotone-isolated-orders baseline_plan binaries metadata output:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 8G --jobs 3 -- python3 -B scripts/rev4_featpot/activation_curriculum.py training --baseline-plan {{quote(baseline_plan)}} --binary-dir {{quote(binaries)}} --metadata {{quote(metadata)}} --output {{quote(output)}} --workers 3 --isolated-orders

monotone-stage-replay baseline_plan binaries metadata output:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 8G --jobs 3 -- python3 -B scripts/rev4_featpot/activation_curriculum.py training --baseline-plan {{quote(baseline_plan)}} --binary-dir {{quote(binaries)}} --metadata {{quote(metadata)}} --output {{quote(output)}} --workers 3 --stage-replay

monotone-stage-panels original replays baseline_plan binaries audit_binary metadata output:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 2G --jobs 1 -- python3 -B scripts/rev4_featpot/monotone_stages.py --original {{quote(original)}} --replays {{quote(replays)}} --baseline-plan {{quote(baseline_plan)}} --binary-dir {{quote(binaries)}} --audit-binary {{quote(audit_binary)}} --metadata {{quote(metadata)}} --output {{quote(output)}} --watch

monotone-stage-plot stages output:
    TMPDIR="$HOME/tmp" OPENBLAS_NUM_THREADS=1 "$HOME/work/claudehints/scripts/run-heavy" --mem 2G --jobs 1 -- python3 -B scripts/rev4_featpot/monotone_stage_plot.py --stages {{quote(stages)}} --output {{quote(output)}}

monotone-capacity root binary output:
    python3 -B scripts/rev4_featpot/activation_curriculum.py capacity --root {{quote(root)}} --binary {{quote(binary)}} --output {{quote(output)}}

monotone-family-panels root prepared metadata binary output:
    TMPDIR="$HOME/tmp" RAYON_NUM_THREADS=1 "$HOME/work/claudehints/scripts/run-heavy" --mem 2G --jobs 1 -- python3 -B scripts/rev4_featpot/activation_curriculum.py families --root {{quote(root)}} --prepared {{quote(prepared)}} --metadata {{quote(metadata)}} --binary {{quote(binary)}} --output {{quote(output)}}

monotone-study-summary root breakdowns output *extra:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 1G --jobs 1 -- python3 -B scripts/rev4_featpot/monotone_summary.py --root {{quote(root)}} --breakdowns {{quote(breakdowns)}} --output {{quote(output)}} {{extra}}

monotone-source-bootstrap root summary binary output *extra:
    TMPDIR="$HOME/tmp" RAYON_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$HOME/work/claudehints/scripts/run-heavy" --mem 2G --jobs 1 -- python3 -B scripts/rev4_featpot/monotone_summary.py --root {{quote(root)}} --bootstrap-summary {{quote(summary)}} --binary {{quote(binary)}} --output {{quote(output)}} {{extra}}

monotone-study-plot summary bootstrap output title *extra:
    TMPDIR="$HOME/tmp" OPENBLAS_NUM_THREADS=1 "$HOME/work/claudehints/scripts/run-heavy" --mem 2G --jobs 1 -- python3 -B scripts/rev4_featpot/monotone_plot.py --summary {{quote(summary)}} --bootstrap {{quote(bootstrap)}} --output {{quote(output)}} --title {{quote(title)}} {{extra}}

monotone-figure-mirror root destination:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 2G --jobs 1 -- python3 -B scripts/byconstruction/dataset_mirror.py --root {{quote(root)}} --nfs --cold-readback --destination {{quote(destination)}} --artifact-kind evaluation

monotone-spline-export-check:
    cargo test --locked -p zensim-validate --bin bake_dial_refit research_spline_repair -- --test-threads=1
    cargo test --locked -p zensim-validate --lib monotone_research -- --test-threads=1
    cargo clippy --locked -p zensim-validate --lib --bin bake_dial_refit -- -D warnings

monotone-export-audit original baseline_plan binaries repair_binary output:
    TMPDIR="$HOME/tmp" OPENBLAS_NUM_THREADS=1 "$HOME/work/claudehints/scripts/run-heavy" --mem 3G --jobs 1 -- python3 -B scripts/rev4_featpot/monotone_export_audit.py --original {{quote(original)}} --baseline-plan {{quote(baseline_plan)}} --binary-dir {{quote(binaries)}} --repair-binary {{quote(repair_binary)}} --output {{quote(output)}}

monotone-study-regressions:
    cargo test --locked -p zensim-validate --lib mlp_train -- --test-threads=1
    cargo test --locked -p zensim-validate --bin zensim_mlp_train -- --test-threads=1
    cargo test --locked -p zensim-validate --bin bake_dial_refit -- --test-threads=1
    just activation-curriculum-tests
    just lint-scripts

monotone-study-close root figures:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 2G --jobs 1 -- python3 -B scripts/rev4_featpot/monotone_close.py --root {{quote(root)}} --figures {{quote(figures)}}

shaped-linear-native-tests:
    cargo test --locked -p zensim-validate --bin bake_dial_refit linear_study -- --test-threads=1

shaped-linear-native-check:
    cargo test --locked -p zensim-validate --bin bake_dial_refit -- --test-threads=1
    cargo clippy --locked -p zensim-validate --bin bake_dial_refit -- -D warnings

shaped-linear-native-build:
    cargo build --locked --release -p zensim-validate --bin bake_dial_refit --bin corrhead_parity

# Source-separated shaped-linear research; run through run-heavy.
shaped-linear-study stage root:
    python3 scripts/rev4_featpot/linear_probe.py shaped-study {{stage}} --root {{root}}

shaped-linear-controller-tests:
    TMPDIR="$HOME/tmp" python3 -B -m unittest discover -s scripts/tests -p test_shaped_linear_study.py -v

shaped-linear-compare root model="all":
    python3 -B scripts/rev4_featpot/shaped_linear_compare.py --root {{root}} --model {{model}}

shaped-linear-report stage root:
    python3 -B scripts/rev4_featpot/shaped_linear_report.py {{stage}} --root {{root}}

# Verify both existing archive copies without modifying the frozen trees.
study-archive-verify root destination receipt:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 1G --jobs 1 -- python3 -B scripts/byconstruction/dataset_mirror.py --root {{quote(root)}} --nfs --cold-readback --destination {{quote(destination)}} --verify-only --verification-output {{quote(receipt)}}

# Explicitly verifies every durable artifact when the local copy was reclaimed.
study-archive-verify-destination root destination receipt:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 1G --jobs 1 -- python3 -B scripts/byconstruction/dataset_mirror.py --root {{quote(root)}} --nfs --cold-readback --destination {{quote(destination)}} --verify-only --destination-only --verification-output {{quote(receipt)}}

# Exact native analytic research heads, shared-depth replay and trainer regressions.
analytic-head-check:
    cargo test --locked -p zensim-validate --lib -- --test-threads=1
    cargo test --locked -p zensim-validate --bin zensim_mlp_train -- --test-threads=1
    cargo clippy --locked -p zensim-validate --lib --bin zensim_mlp_train -- -D warnings

# Native input-shaping runtime and full trainer regression gates.
input-shaping-check:
    cargo test --locked -p zensim --features training,feature-regime-v2 --lib derived_inputs -- --test-threads=1
    cargo test --locked -p zensim --features training,feature-regime-v2 --lib analytic_activation_sensitivity -- --test-threads=1
    just analytic-head-check
    cargo clippy --locked -p zensim --features training,feature-regime-v2 --lib -- -D warnings

# Freeze source memberships and candidate cells before completion-study fits.
shaping-completion-register original output:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 2G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_completion.py --original {{quote(original)}} --output {{quote(output)}}

shaping-completion-registration-check:
    python3 -B -m unittest discover -s scripts/tests -p test_shaping_completion.py

shaping-completion-resource-pilot study binary_dir:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 6G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_execution.py --study {{quote(study)}} --binary-dir {{quote(binary_dir)}} --resource-pilot

# One measured CPU thread per independent seed; controller allows concurrent lanes.
shaping-completion-inner-relu study binary_dir partition:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 2G --jobs 12 -- python3 -B scripts/rev4_featpot/shaping_execution.py --study {{quote(study)}} --binary-dir {{quote(binary_dir)}} --inner-relu --partition {{quote(partition)}} --workers 12

shaping-completion-status study:
    python3 -B scripts/rev4_featpot/shaping_execution.py --study {{quote(study)}} --status

shaping-completion-inner-analytic study binary_dir partition family:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 2G --jobs 4 -- python3 -B scripts/rev4_featpot/shaping_execution.py --study {{quote(study)}} --binary-dir {{quote(binary_dir)}} --inner-family --partition {{quote(partition)}} --family {{quote(family)}} --workers 4

shaping-completion-inputs study partition:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 2G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_inputs.py --study {{quote(study)}} --partition {{quote(partition)}}

shaping-completion-inputs-check:
    python3 -B -m unittest discover -s scripts/tests -p test_shaping_inputs.py

shaping-completion-inner-shapes study binary_dir partition:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 2G --jobs 2 -- python3 -B scripts/rev4_featpot/shaping_execution.py --study {{quote(study)}} --binary-dir {{quote(binary_dir)}} --inner-family --partition {{quote(partition)}} --family joint_shapes --workers 2

shaping-completion-evaluation-check:
    python3 -B -m unittest discover -s scripts/tests -p test_shaping_evaluation.py

shaping-completion-evaluate-inner study binary_dir run:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 2G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_evaluation.py --study {{quote(study)}} --binary-dir {{quote(binary_dir)}} --run {{quote(run)}}

# Revalidates complete inner admission before opening any outer response view.
shaping-completion-evaluate-outer study binary_dir run:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 2G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_evaluation.py --study {{quote(study)}} --binary-dir {{quote(binary_dir)}} --run {{quote(run)}}

# Existing fit owners, with separate memory budgets for native quadratic solves.
shaping-completion-inner-tree study binary_dir partition:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 2G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_controls.py --study {{quote(study)}} --binary-dir {{quote(binary_dir)}} --partition {{quote(partition)}} --tree

shaping-completion-inner-linear study binary_dir partition:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 4G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_controls.py --study {{quote(study)}} --binary-dir {{quote(binary_dir)}} --partition {{quote(partition)}} --linear

shaping-completion-pretraining-check:
    python3 -B -m unittest discover -s scripts/tests -p test_shaping_pretraining.py

shaping-completion-pretrained-dispatch study binary_dir partition workers='2' memory='4G':
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem {{quote(memory)}} --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_pretrained_dispatch.py --study {{quote(study)}} --binary-dir {{quote(binary_dir)}} --partition {{quote(partition)}} --workers {{quote(workers)}}

shaping-pretrained-dispatch-check:
    TMPDIR="$HOME/tmp" python3 -B -m unittest discover -s scripts/tests -p test_shaping_pretrained_dispatch.py

shaping-host-assignment-check:
    TMPDIR="$HOME/tmp" python3 -B -m unittest discover -s scripts/tests -p test_shaping_host_assignment.py

shaping-completion-convergence-check:
    python3 -B -m unittest discover -s scripts/tests -p test_shaping_convergence.py

shaping-completion-evaluate-checkpoint study binary_dir run epoch:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 2G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_evaluation.py --study {{quote(study)}} --binary-dir {{quote(binary_dir)}} --run {{quote(run)}} --checkpoint {{epoch}}

shaping-completion-convergence run:
    python3 -B scripts/rev4_featpot/shaping_convergence.py --run {{quote(run)}}

shaping-completion-reexport-convergence run root:
    python3 -B scripts/rev4_featpot/shaping_convergence.py --run {{quote(run)}} --reexport-root {{quote(root)}}

shaping-completion-watch-evaluation study binary_dir:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 2G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_supervision.py --study {{quote(study)}} --binary-dir {{quote(binary_dir)}} --watch

shaping-completion-diagnostic-replay study binary_dir run output:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 1G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_diagnostic_replay.py --study {{quote(study)}} --binary-dir {{quote(binary_dir)}} --run {{quote(run)}} --output {{quote(output)}}

shaping-completion-archive root destination:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 1G --jobs 1 -- python3 -B scripts/byconstruction/dataset_mirror.py --root {{quote(root)}} --nfs --cold-readback --destination {{quote(destination)}}

shaping-completion-diagnostic-pilot study binary_dir run output:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 1G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_diagnostic_replay.py --study {{quote(study)}} --binary-dir {{quote(binary_dir)}} --run {{quote(run)}} --output {{quote(output)}} --resource-pilot

shaping-completion-selection-check:
    python3 -B -m unittest discover -s scripts/tests -p test_shaping_selection.py

shaping-completion-select-base study family outer:
    python3 -B scripts/rev4_featpot/shaping_selection.py --study {{quote(study)}} --family {{quote(family)}} --outer {{outer}}

shaping-completion-reclaim-old-grams:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 1G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_reclaim_grams.py

shaping-completion-reclaim-tree-parity:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 1G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_reclaim_tree_parity.py

shaping-completion-inner-fresh study binary_dir partition:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 2G --jobs 2 -- python3 -B scripts/rev4_featpot/shaping_execution.py --study {{quote(study)}} --binary-dir {{quote(binary_dir)}} --fresh-control-inner --partition {{quote(partition)}} --workers 2

shaping-completion-variants-check:
    python3 -B -m unittest discover -s scripts/tests -p test_shaping_variants.py

shaping-completion-register-variants study:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 1G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_variants.py --study {{quote(study)}}

shaping-completion-absent-maps-check:
    python3 -B -m unittest discover -s scripts/tests -p test_shaping_absent_maps.py

shaping-completion-population-admission-check:
    python3 -B -m unittest discover -s scripts/tests -p test_shaping_population_admission.py

shaping-completion-prepare-variant study cell:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 2G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_variant_inputs.py --study {{quote(study)}} --cell {{quote(cell)}}

shaping-completion-prepare-all-variants study:
    TMPDIR="$HOME/tmp" OPENBLAS_NUM_THREADS=1 "$HOME/work/claudehints/scripts/run-heavy" --mem 2G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_variant_inputs.py --study {{quote(study)}} --all

shaping-completion-variant-audit-test:
    TMPDIR="$HOME/tmp" OPENBLAS_NUM_THREADS=1 python3 -B -m unittest scripts/tests/test_shaping_variant_audit.py

shaping-completion-audit-variant-inputs study output:
    TMPDIR="$HOME/tmp" OPENBLAS_NUM_THREADS=1 "$HOME/work/claudehints/scripts/run-heavy" --mem 2G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_variant_audit.py --study {{quote(study)}} --output {{quote(output)}}

shaping-completion-prepare-variant-shapes study:
    TMPDIR="$HOME/tmp" OPENBLAS_NUM_THREADS=1 "$HOME/work/claudehints/scripts/run-heavy" --mem 2G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_variant_shapes.py --study {{quote(study)}}

shaping-completion-reclaim-tree-transport *args:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 1G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_reclaim_tree_transport.py {{args}}

shaping-completion-storage-check:
    python3 -B -m unittest discover -s scripts/tests -p test_shaping_storage.py

shaping-completion-rescue-check:
    python3 -B -m unittest discover -s scripts/tests -p test_shaping_rescue.py

shaping-completion-rescue-fit study binaries qualification cell:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 3G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_rescue_execution.py --study {{quote(study)}} --binary-dir {{quote(binaries)}} --neutral-qualification {{quote(qualification)}} --cell {{quote(cell)}}

shaping-completion-rescue-select study family outer:
    python3 -B scripts/rev4_featpot/shaping_rescue_selection.py --study {{quote(study)}} --family {{quote(family)}} --outer {{outer}}

shaping-completion-rescue-qualify study run binaries output epochs="1":
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 1G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_rescue_qualification.py --study {{quote(study)}} --run {{quote(run)}} --binary-dir {{quote(binaries)}} --output {{quote(output)}} --epochs {{epochs}}

shaping-completion-qualification-worker-check:
    TMPDIR="$HOME/tmp" python3 -B -m unittest discover -s scripts/tests -p test_shaping_qualification_worker.py

shaping-completion-qualify-worker study run binaries output snapshots epochs="240":
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 3G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_rescue_qualification.py --worker --study {{quote(study)}} --run {{quote(run)}} --binary-dir {{quote(binaries)}} --output {{quote(output)}} --snapshot-root {{quote(snapshots)}} --epochs {{epochs}}

shaping-completion-supervision-check:
    python3 -B -m unittest discover -s scripts/tests -p test_shaping_supervision.py

shaping-completion-probe-trace model probes predictions output:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 1G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_probe_trace.py --model {{quote(model)}} --probes {{quote(probes)}} --predictions {{quote(predictions)}} --output {{quote(output)}}

shaping-completion-declare-max-pairs model output:
    python3 -B scripts/rev4_featpot/shaping_probe_trace.py --declare-pairs --model {{quote(model)}} --output {{quote(output)}}

shaping-completion-reexport study run binaries inspector epoch:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 2G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_reexport.py --study {{quote(study)}} --run {{quote(run)}} --binary-dir {{quote(binaries)}} --inspector {{quote(inspector)}} --checkpoint {{quote(epoch)}}

# The caller copies the terminal bundle and sidecar; collection rehashes every file.
shaping-completion-worker-collect action study assignment bundle *args:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 1G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_worker_collect.py {{quote(action)}} --study {{quote(study)}} --assignment {{quote(assignment)}} --bundle {{quote(bundle)}} {{args}}

shaping-completion-worker-collect-test:
    TMPDIR="$HOME/tmp" python3 -B -m unittest scripts/tests/test_shaping_worker_collect.py

shaping-completion-watch-reexport study binary_dir inspector:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 2G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_reexport_supervision.py --study {{quote(study)}} --binary-dir {{quote(binary_dir)}} --inspector {{quote(inspector)}} --watch

shaping-completion-worker-inputs study assignment output:
    TMPDIR="$HOME/tmp" OPENBLAS_NUM_THREADS=1 "$HOME/work/claudehints/scripts/run-heavy" --mem 2G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_worker_inputs.py --study {{quote(study)}} --assignment {{quote(assignment)}} --output {{quote(output)}}

shaping-completion-assign-worker study binary_dir partition host family stage="inner" *args:
    python3 -B scripts/rev4_featpot/shaping_host_assignment.py --study {{quote(study)}} --binary-dir {{quote(binary_dir)}} --partition {{quote(partition)}} --host {{quote(host)}} --family {{quote(family)}} --stage {{quote(stage)}} {{args}}

shaping-completion-parent-transport-check:
    TMPDIR="$HOME/tmp" python3 -B -m unittest discover -s scripts/tests -p test_shaping_parent_transport.py

shaping-completion-worker-install-test:
    TMPDIR="$HOME/tmp" python3 -B -m unittest scripts/tests/test_shaping_worker_input_install.py

shaping-completion-watch-archives study destination:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 1G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_archive_supervision.py --study {{quote(study)}} --destination {{quote(destination)}} --watch

shaping-completion-failed-archives study destination *args:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 1G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_archive_supervision.py --study {{quote(study)}} --destination {{quote(destination)}} --failed-original {{args}}

shaping-completion-failed-archives-check:
    TMPDIR="$HOME/tmp" python3 -B -m unittest discover -s scripts/tests -p test_shaping_failed_archive.py

shaping-completion-extension-check:
    TMPDIR="$HOME/tmp" python3 -B -m unittest scripts/tests/test_shaping_extension.py

shaping-completion-register-extension study outer:
    python3 -B scripts/rev4_featpot/shaping_extension.py --study {{quote(study)}} --outer {{outer}}

shaping-completion-extension-fit study binary_dir neutral cell:
    TMPDIR="$HOME/tmp" python3 -B scripts/rev4_featpot/shaping_extension_execution.py --study {{quote(study)}} --binary-dir {{quote(binary_dir)}} --neutral-qualification {{quote(neutral)}} --cell {{quote(cell)}}

shaping-completion-dispatch-extensions study binary_dir neutral outer *args:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 4G --jobs 2 -- python3 -B scripts/rev4_featpot/shaping_extension_dispatch.py --study {{quote(study)}} --binary-dir {{quote(binary_dir)}} --neutral-qualification {{quote(neutral)}} --outer {{outer}} --workers 2 {{args}}

shaping-completion-extension-dispatch-check:
    python3 -B -m unittest discover -s scripts/tests -p test_shaping_extension_dispatch.py

shaping-completion-progress study *args:
    TMPDIR="$HOME/tmp" python3 -B scripts/rev4_featpot/shaping_progress.py --study {{quote(study)}} {{args}}

# Read saved reports only; --watch refreshes the local statistics page.
shaping-completion-stats study dashboard *args:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 2G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_progress.py --study {{quote(study)}} --dashboard {{quote(dashboard)}} {{args}}

shaping-completion-stats-check:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 1G --jobs 1 -- python3 -B -m unittest scripts.tests.test_shaping_dashboard scripts.tests.test_shaping_progress scripts.tests.test_shaping_corrected_dashboard

shaping-completion-stats-syntax-check:
    python3 -B -c 'import re,subprocess; from pathlib import Path; script=re.search(r"<script>(.*?)</script>",Path("scripts/rev4_featpot/shaping_dashboard.html").read_text(),re.S).group(1); subprocess.run(["node","--check"],input=script,text=True,check=True)'

# Bounded metadata observation only: no fit signals, starts or recovery.
shaping-native-watch study observer output *args:
    TMPDIR="$HOME/tmp" python3 -B scripts/rev4_featpot/shaping_native_watch.py --study {{quote(study)}} --observer {{quote(observer)}} --output {{quote(output)}} {{args}}

shaping-scatter-syntax-check:
    node --check scripts/rev4_featpot/shaping_dashboard_details.js

shaping-completion-progress-check:
    TMPDIR="$HOME/tmp" python3 -B -m unittest discover -s scripts/tests -p test_shaping_progress.py
    TMPDIR="$HOME/tmp" python3 -B -m unittest scripts.tests.test_shaping_approved_progress

# Observe real run-heavy scope survival before a queue-only controller replacement.
shaping-pretrained-handoff-scope-check output:
    TMPDIR="$HOME/tmp" SHAPING_HANDOFF_QUALIFICATION={{quote(output)}} python3 -B -m unittest scripts.tests.test_shaping_pretrained_dispatch.HandoffScopeQualification.verify_scope_child_survival

shaping-completion-pretrained-handoff study binaries partition workers resume old_status receipt qualification:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 2G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_pretrained_dispatch.py --study {{quote(study)}} --binary-dir {{quote(binaries)}} --partition {{quote(partition)}} --workers {{workers}} --resume-name {{quote(resume)}} --handoff-status {{quote(old_status)}} --handoff-receipt {{quote(receipt)}} --handoff-qualification {{quote(qualification)}}

shaping-training-resources run output:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 1G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_training_resources.py --run {{quote(run)}} --output {{quote(output)}}

shaping-training-resources-check:
    TMPDIR="$HOME/tmp" python3 -B -m unittest discover -s scripts/tests -p test_shaping_training_resources.py

shaping-training-resources-qualify study output:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 1G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_training_resources.py --qualify-study {{quote(study)}} --output {{quote(output)}}

shaping-cost-report directory workload output:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 1G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_cost_report.py --directory {{quote(directory)}} --workload {{quote(workload)}} --output {{quote(output)}}

shaping-cost-report-check:
    TMPDIR="$HOME/tmp" python3 -B -m unittest discover -s scripts/tests -p test_shaping_cost_report.py

shaping-cost-memory binary workload output:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 2G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_cost_memory.py --binary {{quote(binary)}} --workload {{quote(workload)}} --output {{quote(output)}}

shaping-cost-memory-check:
    TMPDIR="$HOME/tmp" python3 -B -m unittest discover -s scripts/tests -p test_shaping_cost_memory.py

shaping-model-inventory-build:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 2G --jobs 1 -- cargo build --locked --release -p zensim-validate --example shaping_model_inventory

shaping-model-inventory-clippy:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 2G --jobs 1 -- cargo clippy --locked --release -p zensim-validate --example shaping_model_inventory -- -D warnings

shaping-model-inventory binary workload output:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 1G --jobs 1 -- {{quote(binary)}} {{quote(workload)}} {{quote(output)}}

shaping-model-inventory-check binary inspector workload tree_log output:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 1G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_inventory_check.py --binary {{quote(binary)}} --inspector {{quote(inspector)}} --workload {{quote(workload)}} --tree-log {{quote(tree_log)}} --output {{quote(output)}}

shaping-live-profile pid output:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 1G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_live_profile.py --pid {{pid}} --output {{quote(output)}}

shaping-parameter-count run inspector output:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 1G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_parameter_count.py --run {{quote(run)}} --inspector {{quote(inspector)}} --output {{quote(output)}}

shaping-parameter-count-check:
    TMPDIR="$HOME/tmp" python3 -B -m unittest discover -s scripts/tests -p test_shaping_parameter_count.py

shaping-parameter-count-qualify study inspector output:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 1G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_parameter_count.py --qualify-study {{quote(study)}} --inspector {{quote(inspector)}} --output {{quote(output)}}

shaping-completion-dispatch-rescues study binary_dir neutral outer *args:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 6G --jobs 2 -- python3 -B scripts/rev4_featpot/shaping_rescue_dispatch.py --study {{quote(study)}} --binary-dir {{quote(binary_dir)}} --neutral-qualification {{quote(neutral)}} --outer {{outer}} --workers 2 {{args}}

shaping-completion-dispatch-outer study binary_dir neutral outer *args:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 4G --jobs 2 -- python3 -B scripts/rev4_featpot/shaping_outer_dispatch.py --study {{quote(study)}} --binary-dir {{quote(binary_dir)}} --neutral-qualification {{quote(neutral)}} --outer {{outer}} --workers 2 {{args}}

shaping-completion-outer-dispatch-check:
    python3 -B -m unittest discover -s scripts/tests -p test_shaping_outer_dispatch.py

shaping-completion-optimizer-summary study run output *args:
    python3 -B scripts/rev4_featpot/shaping_optimizer_summary.py --study {{quote(study)}} --run {{quote(run)}} --output {{quote(output)}} {{args}}

shaping-completion-optimizer-summary-check:
    python3 -B -m unittest discover -s scripts/tests -p test_shaping_optimizer_summary.py

shaping-completion-curve-stability study binaries family outer setting output:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 1G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_curve_stability.py --study {{quote(study)}} --binary-dir {{quote(binaries)}} --family {{quote(family)}} --outer {{outer}} --setting {{setting}} --output {{quote(output)}}

shaping-completion-curve-stability-check:
    python3 -B -m unittest discover -s scripts/tests -p test_shaping_curve_stability.py

shaping-completion-curve-stability-partitions study binaries family setting output +partitions:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 1G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_curve_stability.py --study {{quote(study)}} --binary-dir {{quote(binaries)}} --family {{quote(family)}} --setting {{setting}} --output {{quote(output)}} --partitions {{partitions}}

shaping-completion-curve-checkpoint-partitions study binaries family setting epoch output +partitions:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 1G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_curve_stability.py --study {{quote(study)}} --binary-dir {{quote(binaries)}} --family {{quote(family)}} --setting {{setting}} --checkpoint {{epoch}} --output {{quote(output)}} --partitions {{partitions}}

# Matched registered L2 settings across explicit source partitions; descriptive only.
shaping-completion-curve-regularization-partitions study binaries family output +partitions:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 1G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_curve_stability.py --study {{quote(study)}} --binary-dir {{quote(binaries)}} --family {{quote(family)}} --settings 0 1 --output {{quote(output)}} --partitions {{partitions}}

shaping-completion-watch-all-evaluation study binary_dir rescue_binary_dir:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 2G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_supervision.py --study {{quote(study)}} --binary-dir {{quote(binary_dir)}} --rescue-binary-dir {{quote(rescue_binary_dir)}} --watch

shaping-completion-reconcile-prefix-failure study cell:
    python3 -B scripts/rev4_featpot/shaping_reconcile_prefix_failure.py --study {{quote(study)}} --cell {{quote(cell)}}

shaped-linear-feature-curves root output:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 1G --jobs 1 -- python3 -B scripts/rev4_featpot/shaped_linear_report.py feature-curves --root {{quote(root)}} --output {{quote(output)}}

shaped-linear-feature-curves-test:
    TMPDIR="$HOME/tmp" python3 -B -m unittest scripts/tests/test_shaped_feature_curves.py

# A new destination prevents overwriting a prior worker transfer.
shaping-completion-fetch-worker host remote_bundle destination:
    mkdir {{quote(destination)}}
    scp -- {{quote(host + ":" + remote_bundle)}} {{quote(host + ":" + remote_bundle + ".json")}} {{quote(destination + "/")}}

shaping-completion-pack-worker host source study assignment cell bundle attempt='attempt-001':
    ssh {{quote(host)}} 'TMPDIR="$HOME/tmp" PYTHONPATH="{{study}}/../python-runtime" "$HOME/work/claudehints/scripts/run-heavy" --mem 1G --jobs 1 -- python3 -B {{source}}/scripts/rev4_featpot/shaping_worker_collect.py pack --study {{study}} --assignment {{assignment}} --cell {{cell}} --bundle {{bundle}} --attempt {{attempt}}'

shaping-completion-collect-terminal-worker host source study status destination output:
    TMPDIR="$HOME/tmp" python3 -B scripts/rev4_featpot/shaping_worker_collection_wave.py --host {{quote(host)}} --source {{quote(source)}} --study {{quote(study)}} --status {{quote(status)}} --destination {{quote(destination)}} --output {{quote(output)}}

shaping-completion-watch-worker host source status_source study destination output *args:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 1G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_worker_watch.py --host {{quote(host)}} --source {{quote(source)}} --status-source {{quote(status_source)}} --study {{quote(study)}} --destination {{quote(destination)}} --output {{quote(output)}} --watch {{args}}

shaping-completion-worker-watch-check:
    python3 -B -m unittest discover -s scripts/tests -p test_shaping_worker_watch.py

shaping-completion-watch-selection study outer:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 1G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_selection_watch.py --study {{quote(study)}} --outer {{outer}} --watch

shaping-completion-selection-watch-check:
    python3 -B -m unittest discover -s scripts/tests -p test_shaping_selection_watch.py

shaping-completion-reclaim-incremental:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 1G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_reclaim_incremental.py

shaping-completion-reclaim-linear-parity:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 1G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_reclaim_linear_parity.py

shaping-completion-human-qualify study run binaries output cell:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 3G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_rescue_qualification.py --study {{quote(study)}} --run {{quote(run)}} --binary-dir {{quote(binaries)}} --output {{quote(output)}} --human-only-cell {{quote(cell)}}

shaping-completion-register-rescue-trigger study run:
    python3 -B scripts/rev4_featpot/shaping_rescue.py --study {{quote(study)}} --trigger-run {{quote(run)}}

shaping-completion-outer-plan-check:
    python3 -B -m unittest scripts/tests/test_shaping_outer_plan.py

shaping-completion-outer-admission study outer *args:
    python3 -B scripts/rev4_featpot/shaping_outer_admission.py --study {{quote(study)}} --outer {{outer}} {{args}}

shaping-completion-freeze-worker-policies study:
    TMPDIR="$HOME/tmp" python3 -B scripts/rev4_featpot/shaping_policy_snapshot.py --study {{quote(study)}}

shaping-completion-policy-snapshot-check:
    TMPDIR="$HOME/tmp" python3 -B -m unittest discover -s scripts/tests -p test_shaping_policy_snapshot.py

shaping-completion-outer-admission-check:
    python3 -B -m unittest scripts/tests/test_shaping_outer_admission.py

shaping-completion-outer-execution-check:
    python3 -B -m unittest scripts/tests/test_shaping_outer_execution.py

shaping-completion-outer-controls-check:
    python3 -B -m unittest scripts/tests/test_shaping_outer_controls.py

shaping-completion-variant-execution-check:
    TMPDIR="$HOME/tmp" python3 -B -m unittest scripts/tests/test_shaping_variant_execution.py

shaping-completion-downstream-execution-check:
    TMPDIR="$HOME/tmp" python3 -B -m unittest scripts/tests/test_shaping_downstream_execution.py scripts/tests/test_shaping_outer_execution.py scripts/tests/test_shaping_variant_execution.py scripts/tests/test_shaping_corrected_downstream.py scripts/tests/test_shaping_corrected_view.py

shaping-completion-downstream-worker-check:
    TMPDIR="$HOME/tmp" python3 -B -m unittest scripts/tests/test_shaping_downstream_worker.py scripts/tests/test_shaping_rescue_worker.py

shaping-outer-prepare-worker-check:
    TMPDIR="$HOME/tmp" python3 -B -m unittest scripts.tests.test_shaping_outer_prepare_worker

shaping-evaluation-runtime-verify manifest host:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 512M --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_evaluation_runtime.py transfer --manifest {{quote(manifest)}} --host {{quote(host)}}

shaping-outer-pipeline-check:
    TMPDIR="$HOME/tmp" python3 -B -m unittest scripts.tests.test_shaping_outer_pipeline scripts.tests.test_shaping_outer_prepare_worker scripts.tests.test_shaping_rescue_host_launch

shaping-outer-pipeline study output binaries qualification outer host:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 1G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_outer_pipeline.py --study {{quote(study)}} --output {{quote(output)}} --binaries {{quote(binaries)}} --qualification {{quote(qualification)}} --outer {{outer}} --prepare-host {{quote(host)}}

shaping-extension-pipeline study output binaries qualification outer:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 1G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_extension_pipeline.py --study {{quote(study)}} --output {{quote(output)}} --binaries {{quote(binaries)}} --qualification {{quote(qualification)}} --outer {{outer}}

shaping-completion-downstream-collection-check:
    TMPDIR="$HOME/tmp" python3 -B -m unittest scripts/tests/test_shaping_downstream_collection.py scripts/tests/test_shaping_worker_collect.py

shaping-completion-later-evaluation-worker-check:
    TMPDIR="$HOME/tmp" python3 -B -m unittest scripts/tests/test_shaping_later_evaluation_worker.py scripts/tests/test_shaping_evaluation_worker.py scripts/tests/test_shaping_supervision.py

shaping-completion-neutral-trajectory-check:
    TMPDIR="$HOME/tmp" python3 -B -m unittest scripts/tests/test_shaping_neutral_trajectory.py

shaping-completion-control-compatibility-check:
    TMPDIR="$HOME/tmp" python3 -B -m unittest scripts/tests/test_shaping_control_compatibility.py scripts/tests/test_shaping_neutral_trajectory.py

shaping-control-evidence-check:
    TMPDIR="$HOME/tmp" python3 -B -m unittest scripts/tests/test_shaping_control_evidence.py

shaping-lossless-capsule source output:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 1G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_lossless_capsule.py --source {{quote(source)}} --output {{quote(output)}}

shaping-control-evidence-prepare registration family run output:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 1G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_control_evidence.py prepare --registration {{quote(registration)}} --family {{quote(family)}} --run {{quote(run)}} --output {{quote(output)}}

shaping-control-evidence-verify inventory directory:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 1G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_control_evidence.py verify --inventory {{quote(inventory)}} --directory {{quote(directory)}}

shaping-completion-corrected-registration-check:
    TMPDIR="$HOME/tmp" python3 -B -m unittest scripts/tests/test_shaping_corrected_registration.py

shaping-completion-corrected-view-check:
    TMPDIR="$HOME/tmp" python3 -B -m unittest scripts/tests/test_shaping_corrected_view.py

shaping-completion-corrected-worker-check:
    TMPDIR="$HOME/tmp" python3 -B -m unittest scripts/tests/test_shaping_corrected_worker.py scripts/tests/test_shaping_worker_collect.py scripts/tests/test_shaping_rescue_worker.py scripts/tests/test_shaping_downstream_worker.py

shaping-completion-corrected-worker-register study cell host binaries qualification output:
    TMPDIR="$HOME/tmp" python3 -B scripts/rev4_featpot/shaping_rescue_worker.py register --execution-kind corrected_inner --study {{quote(study)}} --cell {{quote(cell)}} --host {{quote(host)}} --binaries {{quote(binaries)}} --qualification {{quote(qualification)}} --output {{quote(output)}}

shaping-completion-corrected-certify study output *proofs:
    TMPDIR="$HOME/tmp" python3 -B scripts/rev4_featpot/shaping_corrected_view.py --study {{quote(study)}} --output {{quote(output)}} {{proofs}}

shaping-completion-corrected-activate study *proofs:
    TMPDIR="$HOME/tmp" python3 -B scripts/rev4_featpot/shaping_corrected_view.py --study {{quote(study)}} --activate {{proofs}}

shaping-rescue-wave-prepare-check:
    TMPDIR="$HOME/tmp" python3 -B -m unittest scripts/tests/test_shaping_rescue_wave_prepare.py

shaping-native-observe-check:
    TMPDIR="$HOME/tmp" python3 -B -m unittest scripts/tests/test_shaping_native_observe.py

shaping-snapshot-retention-check:
    TMPDIR="$HOME/tmp" python3 -B -m unittest scripts/tests/test_shaping_snapshot_retention.py

shaping-snapshot-bind scope run output:
    TMPDIR="$HOME/tmp" python3 -B scripts/rev4_featpot/shaping_snapshot_retention.py bind --scope {{quote(scope)}} --run {{quote(run)}} --output {{quote(output)}}

shaping-snapshot-stage binding output:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 1G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_snapshot_retention.py stage --binding {{quote(binding)}} --output {{quote(output)}}

shaping-snapshot-retire binding stage output *args:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 1G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_snapshot_retention.py retire --binding {{quote(binding)}} --stage-root {{quote(stage)}} --output {{quote(output)}} {{args}}

shaping-rescue-wave-prepare study source binaries qualification outer placements output:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 1G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_rescue_wave_prepare.py --study {{quote(study)}} --source {{quote(source)}} --binaries {{quote(binaries)}} --qualification {{quote(qualification)}} --outer {{outer}} --placements {{quote(placements)}} --output {{quote(output)}}

shaping-completion-corrected-register study approval proposal binaries output *observations:
    TMPDIR="$HOME/tmp" python3 -B scripts/rev4_featpot/shaping_corrected_registration.py --study {{quote(study)}} --approval {{quote(approval)}} --proposal {{quote(proposal)}} --binary-dir {{quote(binaries)}} --output {{quote(output)}} {{observations}}

shaping-completion-control-compatibility-register study approval binaries output:
    TMPDIR="$HOME/tmp" python3 -B scripts/rev4_featpot/shaping_control_compatibility.py register --study {{quote(study)}} --approval {{quote(approval)}} --binary-dir {{quote(binaries)}} --output {{quote(output)}}

shaping-completion-control-compatibility-run study registration binaries family output:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 3G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_control_compatibility.py run --study {{quote(study)}} --registration {{quote(registration)}} --binary-dir {{quote(binaries)}} --family {{quote(family)}} --output {{quote(output)}}

shaping-completion-checkpoint-restore-check:
    TMPDIR="$HOME/tmp" python3 -B -m unittest scripts/tests/test_shaping_checkpoint_restore.py

shaping-completion-checkpoint-restore study cell output:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 1G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_checkpoint_restore.py --study {{quote(study)}} --cell {{quote(cell)}} --output {{quote(output)}}

shaping-completion-neutral-trajectory study run qualification binaries output:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 1G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_neutral_trajectory.py --study {{quote(study)}} --run {{quote(run)}} --qualification {{quote(qualification)}} --binary-dir {{quote(binaries)}} --output {{quote(output)}}

shaping-completion-full-fit-plan-check:
    TMPDIR="$HOME/tmp" python3 -B -m unittest scripts/tests/test_shaping_full_fit_plan.py

shaping-completion-full-fit-admission-check:
    TMPDIR="$HOME/tmp" python3 -B -m unittest scripts/tests/test_shaping_full_fit_admission.py

shaping-completion-full-fit-admission study *args:
    TMPDIR="$HOME/tmp" python3 -B scripts/rev4_featpot/shaping_full_fit_admission.py --study {{quote(study)}} {{args}}

shaping-completion-variant-fit study binary_dir neutral human_qualification cell:
    TMPDIR="$HOME/tmp" python3 -B scripts/rev4_featpot/shaping_variant_execution.py --study {{quote(study)}} --binary-dir {{quote(binary_dir)}} --neutral-qualification {{quote(neutral)}} --human-qualification {{quote(human_qualification)}} --cell {{quote(cell)}}

shaping-completion-variant-pipeline study output binary_dir neutral human_qualification outer outer_status:
    TMPDIR="$HOME/tmp" python3 -B scripts/rev4_featpot/shaping_variant_pipeline.py --study {{quote(study)}} --output {{quote(output)}} --binaries {{quote(binary_dir)}} --qualification {{quote(neutral)}} --human-qualification {{quote(human_qualification)}} --outer {{outer}} --outer-status {{quote(outer_status)}}

shaping-completion-variant-pipeline-check:
    TMPDIR="$HOME/tmp" python3 -B -m unittest scripts.tests.test_shaping_variant_pipeline scripts.tests.test_shaping_outer_pipeline scripts.tests.test_shaping_placement

shaping-completion-variant-control study binary_dir partition cell mode:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 4G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_controls.py --study {{quote(study)}} --binary-dir {{quote(binary_dir)}} --partition {{quote(partition)}} --cell {{quote(cell)}} --{{mode}}

shaping-completion-prepare-outer study outer:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 2G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_outer_execution.py --study {{quote(study)}} --outer {{outer}} --prepare

shaping-completion-fit-outer-neural study outer binary_dir qualification cell:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 6G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_outer_execution.py --study {{quote(study)}} --outer {{outer}} --binary-dir {{quote(binary_dir)}} --neutral-qualification {{quote(qualification)}} --cell {{quote(cell)}}

shaping-completion-checkpoint-retention-check:
    python3 -B -m unittest discover -s scripts/tests -p test_shaping_checkpoint_retention.py

shaping-completion-checkpoint-retention study output *args:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 1G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_checkpoint_retention.py --study {{quote(study)}} --output {{quote(output)}} {{args}}

shaping-completion-worker-checkpoint-retention host study archive output *args:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 1G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_worker_checkpoint_retention.py --host {{quote(host)}} --study {{quote(study)}} --archive {{quote(archive)}} --output {{quote(output)}} {{args}}

shaping-completion-worker-checkpoint-retention-check:
    TMPDIR="$HOME/tmp" python3 -B -m unittest discover -s scripts/tests -p test_shaping_worker_checkpoint_retention.py

shaping-completion-full-fit-native-check:
    python3 -B -m unittest discover -s scripts/tests -p test_shaping_full_fit_execution.py

shaping-completion-full-fit-search-prepare study cell:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 2G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_full_fit_execution.py --study {{quote(study)}} --cell {{quote(cell)}} --prepare

shaping-completion-full-fit-search-native study cell binaries qualification:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 3G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_full_fit_execution.py --study {{quote(study)}} --cell {{quote(cell)}} --binary-dir {{quote(binaries)}} --neutral-qualification {{quote(qualification)}}

shaping-completion-full-fit-control study partition binaries mode:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 6G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_controls.py --study {{quote(study)}} --partition {{quote(partition)}} --binary-dir {{quote(binaries)}} --full-fit-selection --{{mode}}

shaping-completion-training-only-convergence run report:
    python3 -B scripts/rev4_featpot/shaping_convergence.py --run {{quote(run)}} --training-only --final-report {{quote(report)}}

shaping-completion-full-fit-selection-check:
    python3 -B -m unittest discover -s scripts/tests -p test_shaping_full_fit_selection.py

shaping-completion-full-fit-select study role *args:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 1G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_full_fit_selection.py --study {{quote(study)}} --role {{quote(role)}} {{args}}

shaping-completion-full-fit-refit-prepare study cell:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 2G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_full_fit_execution.py --study {{quote(study)}} --cell {{quote(cell)}} --prepare

shaping-completion-full-fit-refit-native study cell binaries qualification:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 3G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_full_fit_execution.py --study {{quote(study)}} --cell {{quote(cell)}} --binary-dir {{quote(binaries)}} --neutral-qualification {{quote(qualification)}}

shaping-completion-full-fit-refit-control study binaries mode:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 6G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_controls.py --study {{quote(study)}} --partition full-fit --binary-dir {{quote(binaries)}} --full-fit-refit --{{mode}}

shaping-completion-worker-bundle-retention-check:
    python3 -B -m unittest discover -s scripts/tests -p test_shaping_worker_bundle_retention.py

shaping-completion-worker-bundle-retention host study archive output *args:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 1G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_worker_bundle_retention.py --host {{quote(host)}} --study {{quote(study)}} --archive {{quote(archive)}} --output {{quote(output)}} {{args}}

shaping-completion-watch-later study primary optimizer human repaired inspector:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 2G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_supervision.py --study {{quote(study)}} --binary-dir {{quote(primary)}} --rescue-binary-dir {{quote(optimizer)}} --candidate-binary-dir {{quote(human)}} --repair-binary-dir {{quote(repaired)}} --repair-inspector {{quote(inspector)}} --later-only --watch

# Reclaim historical Python model duplicates only after complete durable readback.
shaping-completion-reclaim-tree-models:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 1G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_reclaim_tree_models.py

# Metadata-only paired source draws; no predictions or targets are read.
shaping-completion-bootstrap-plan study output:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 1G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_bootstrap_plan.py --study {{quote(study)}} --output {{quote(output)}}

shaping-completion-bootstrap-plan-check:
    TMPDIR="$HOME/tmp" python3 -B -m unittest discover -s scripts/tests -p test_shaping_bootstrap_plan.py

shaping-completion-bootstrap-stats-check panel:
    TMPDIR="$HOME/tmp" ZEN_PANEL_BIN={{quote(panel)}} "$HOME/work/claudehints/scripts/run-heavy" --mem 1G --jobs 1 -- python3 -B -m unittest discover -s scripts/tests -p test_shaping_bootstrap_stats.py

shaping-completion-prediction-evidence-check:
    TMPDIR="$HOME/tmp" python3 -B -m unittest discover -s scripts/tests -p test_shaping_prediction_evidence.py

shaping-completion-outer-evidence-check:
    TMPDIR="$HOME/tmp" python3 -B -m unittest discover -s scripts/tests -p test_shaping_outer_evidence.py

shaping-completion-bootstrap-reduce-check:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 1G --jobs 1 -- python3 -B -m unittest discover -s scripts/tests -p test_shaping_bootstrap_reduce.py

shaping-completion-bootstrap study draws binary_dir output:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 2G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_bootstrap.py --study {{quote(study)}} --draws {{quote(draws)}} --binary-dir {{quote(binary_dir)}} --output {{quote(output)}}

shaping-completion-bootstrap-check panel:
    TMPDIR="$HOME/tmp" ZEN_PANEL_BIN={{quote(panel)}} "$HOME/work/claudehints/scripts/run-heavy" --mem 1G --jobs 1 -- python3 -B -m unittest discover -s scripts/tests -p test_shaping_bootstrap.py

# One nonneural family per queue; existing control trainer owns fitting and gates.
shaping-completion-watch-outer-control study binary_dir partition mode:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 6G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_controls.py --study {{quote(study)}} --binary-dir {{quote(binary_dir)}} --partition {{quote(partition)}} --{{mode}} --watch-outer-admission

# Two independent family queues may run together; each bounds RAM across folds.
shaping-completion-watch-outer-controls study binary_dir mode:
    #!/usr/bin/env bash
    set -euo pipefail
    for outer in 0 1 2; do
        just shaping-completion-watch-outer-control {{quote(study)}} {{quote(binary_dir)}} "outer-$outer" {{quote(mode)}}
    done

# Complete neural variant queue for one admitted outer fold, bounded independently.
shaping-completion-dispatch-variants study binary_dir neutral human outer:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 3G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_outer_dispatch.py --study {{quote(study)}} --binary-dir {{quote(binary_dir)}} --neutral-qualification {{quote(neutral)}} --human-qualification {{quote(human)}} --outer {{outer}} --workers 1 --variants --watch-admission

# One 8 GiB control scope across folds; native variant queues remain independent.
shaping-completion-dispatch-variant-controls study binary_dir:
    #!/usr/bin/env bash
    set -euo pipefail
    for outer in 0 1 2; do
        TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 8G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_outer_dispatch.py --study {{quote(study)}} --binary-dir {{quote(binary_dir)}} --neutral-qualification {{quote(binary_dir)}} --outer "$outer" --workers 1 --variant-controls --watch-admission
    done

shaping-completion-full-fit-dispatch-check:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 1G --jobs 1 -- python3 -B -m unittest discover -s scripts/tests -p 'test_shaping_full_fit*.py'

# Two search jobs; each native phase retains its own cap. Controls stay grouped.
shaping-completion-watch-full-fit-search study binary_dir neutral:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 14G --jobs 2 -- python3 -B scripts/rev4_featpot/shaping_full_fit_execution.py --study {{quote(study)}} --binary-dir {{quote(binary_dir)}} --neutral-qualification {{quote(neutral)}} --watch-search

shaping-completion-watch-full-fit-selection study:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 2G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_full_fit_selection.py --study {{quote(study)}} --watch

shaping-completion-watch-full-fit-refit study binary_dir neutral:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 14G --jobs 2 -- python3 -B scripts/rev4_featpot/shaping_full_fit_execution.py --study {{quote(study)}} --binary-dir {{quote(binary_dir)}} --neutral-qualification {{quote(neutral)}} --watch-refit

# Fixed matched regularization contrast: lr .001, L2 0 versus 1e-5, all three seeds.
shaping-completion-regularization-curves study binaries family outer inner output:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 1G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_curve_stability.py --study {{quote(study)}} --binary-dir {{quote(binaries)}} --family {{quote(family)}} --outer {{outer}} --inner-fold {{inner}} --settings 0 1 --output {{quote(output)}}

# All scheduled checkpoint structures; no new predictions or checkpoint selection.
shaping-completion-structure-trace run binary_dir output:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 1G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_convergence.py --run {{quote(run)}} --binary-dir {{quote(binary_dir)}} --structure-trace {{quote(output)}}

shaping-completion-diagnostic-replay-all study binary_dir run output:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 1G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_diagnostic_replay.py --study {{quote(study)}} --binary-dir {{quote(binary_dir)}} --run {{quote(run)}} --output {{quote(output)}} --all-checkpoints

shaping-completion-diagnostic-replay-check:
    TMPDIR="$HOME/tmp" python3 -B -m unittest discover -s scripts/tests -p test_shaping_diagnostic_replay.py

# Run on an eligible build host under run-heavy with an explicit target directory.
shaping-completion-weight-observer-native-check:
    cargo test --locked --release -p zensim-validate --lib -- --test-threads=1

shaping-completion-probe-trace-check:
    TMPDIR="$HOME/tmp" python3 -B -m unittest discover -s scripts/tests -p test_shaping_probe_trace.py

# Diagnostic probe-only re-export; does not admit or score a study candidate.
shaping-completion-repair-probe original probes binaries inspector output:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 1G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_probe_trace.py --repair-probes --original {{quote(original)}} --probes {{quote(probes)}} --binary-dir {{quote(binaries)}} --inspector {{quote(inspector)}} --output {{quote(output)}}

# Qualify the sampled first-weight observer against every original checkpoint.
shaping-completion-weight-replay study binary_dir run output:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 1G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_diagnostic_replay.py --study {{quote(study)}} --binary-dir {{quote(binary_dir)}} --run {{quote(run)}} --output {{quote(output)}} --all-checkpoints --require-weight-diagnostics

shaping-completion-analytic-reexport study run binaries inspector *args:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 2G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_reexport.py --study {{quote(study)}} --run {{quote(run)}} --binary-dir {{quote(binaries)}} --inspector {{quote(inspector)}} --analytic-diagnostic {{args}}

shaping-completion-reexport-check:
    TMPDIR="$HOME/tmp" python3 -B -m unittest discover -s scripts/tests -p test_shaping_reexport.py

shaping-completion-analytic-convergence run root:
    TMPDIR="$HOME/tmp" python3 -B scripts/rev4_featpot/shaping_convergence.py --run {{quote(run)}} --reexport-root {{quote(root)}} --analytic-diagnostic

# Supplemental Softplus panels only; selection still requires separate qualification.
shaping-completion-watch-softplus-reexports study binaries inspector:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 2G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_reexport_supervision.py --study {{quote(study)}} --binary-dir {{quote(binaries)}} --inspector {{quote(inspector)}} --family softplus --watch

shaping-completion-watch-supplemental-archives study destination:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 1G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_archive_supervision.py --study {{quote(study)}} --destination {{quote(destination)}} --supplemental --watch

shaping-completion-watch-rescue-archives study destination:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 1G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_archive_supervision.py --study {{quote(study)}} --destination {{quote(destination)}} --rescue --watch

shaping-completion-rescue-archive-check:
    TMPDIR="$HOME/tmp" python3 -B -m unittest discover -s scripts/tests -p test_shaping_rescue_archive.py

shaping-completion-supplemental-archive-check:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 1G --jobs 1 -- python3 -B -m unittest discover -s scripts/tests -p test_shaping_supplemental_archive.py

shaping-completion-later-archive-check:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 1G --jobs 1 -- python3 -B -m unittest discover -s scripts/tests -p test_shaping_later_archive.py

shaping-completion-watch-later-archives study destination:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 1G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_archive_supervision.py --study {{quote(study)}} --destination {{quote(destination)}} --later-neural --watch

shaping-completion-control-archive-check:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 1G --jobs 1 -- python3 -B -m unittest discover -s scripts/tests -p test_shaping_control_archive.py

shaping-completion-watch-control-archives study destination:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 1G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_archive_supervision.py --study {{quote(study)}} --destination {{quote(destination)}} --later-controls --watch

shaping-completion-reexport-runtime-compare study run binaries original_binaries inspector output:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 2G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_reexport.py --study {{quote(study)}} --run {{quote(run)}} --binary-dir {{quote(binaries)}} --inspector {{quote(inspector)}} --compare-runtime {{quote(output)}} --original-binary-dir {{quote(original_binaries)}}

shaping-completion-bootstrap-matched-native-check panel:
    TMPDIR="$HOME/tmp" ZEN_PANEL_BIN={{quote(panel)}} "$HOME/work/claudehints/scripts/run-heavy" --mem 1G --jobs 1 -- python3 -B -m unittest discover -s scripts/tests -p test_shaping_bootstrap.py

shaping-completion-bootstrap-matched study draws binaries comparison output:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 2G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_bootstrap.py --study {{quote(study)}} --draws {{quote(draws)}} --binary-dir {{quote(binaries)}} --comparison {{quote(comparison)}} --output {{quote(output)}}

shaping-completion-register-matched-bootstrap study output:
    TMPDIR="$HOME/tmp" python3 -B scripts/rev4_featpot/shaping_bootstrap_plan.py --study {{quote(study)}} --output {{quote(output)}} --matched

shaping-completion-bootstrap-watch-check:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 1G --jobs 1 -- python3 -B -m unittest discover -s scripts/tests -p test_shaping_bootstrap_watch.py

shaping-bootstrap-worker-check:
    TMPDIR="$HOME/tmp" python3 -B -m unittest scripts/tests/test_shaping_bootstrap_worker.py scripts/tests/test_shaping_rescue_worker.py scripts/tests/test_shaping_downstream_worker.py scripts/tests/test_shaping_corrected_worker.py

shaping-control-worker-check:
    TMPDIR="$HOME/tmp" python3 -B -m unittest scripts/tests/test_shaping_control_worker.py scripts/tests/test_shaping_outer_controls.py scripts/tests/test_shaping_rescue_dispatch.py scripts/tests/test_shaping_bootstrap_worker.py

shaping-control-worker-register study cell binaries host output:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 2G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_control_worker.py register --study {{quote(study)}} --cell {{quote(cell)}} --binary-dir {{quote(binaries)}} --host {{quote(host)}} --output {{quote(output)}}

shaping-control-worker-pack request bundle:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 1G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_control_worker.py pack --request {{quote(request)}} --bundle {{quote(bundle)}}

shaping-control-worker-collect request bundle pin:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 1G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_control_worker.py collect --request {{quote(request)}} --bundle {{quote(bundle)}} --bundle-sha256 {{quote(pin)}}

shaping-bootstrap-worker-register study draws binaries result host output *comparison:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 2G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_bootstrap_worker.py register --study {{quote(study)}} --draws {{quote(draws)}} --binary-dir {{quote(binaries)}} --result-output {{quote(result)}} --host {{quote(host)}} --output {{quote(output)}} {{comparison}}

shaping-bootstrap-worker-pack request bundle:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 1G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_bootstrap_worker.py pack --request {{quote(request)}} --bundle {{quote(bundle)}}

shaping-bootstrap-worker-collect request bundle pin:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 1G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_bootstrap_worker.py collect --request {{quote(request)}} --bundle {{quote(bundle)}} --bundle-sha256 {{quote(pin)}}

shaping-completion-watch-bootstraps study draws binaries registry output destination:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 2G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_bootstrap.py --study {{quote(study)}} --draws {{quote(draws)}} --binary-dir {{quote(binaries)}} --matched-registry {{quote(registry)}} --output {{quote(output)}} --archive-destination {{quote(destination)}} --watch

shaping-completion-bootstrap-pipeline study draws binaries registry output destination:
    TMPDIR="$HOME/tmp" python3 -B scripts/rev4_featpot/shaping_bootstrap_pipeline.py --study {{quote(study)}} --draws {{quote(draws)}} --binaries {{quote(binaries)}} --registry {{quote(registry)}} --output {{quote(output)}} --destination {{quote(destination)}}

shaping-completion-bootstrap-pipeline-check:
    TMPDIR="$HOME/tmp" python3 -B -m unittest scripts.tests.test_shaping_bootstrap_pipeline scripts.tests.test_shaping_bootstrap_worker

# Native cost instrument: tests/builds run on an eligible non-dev host.
shaping-cost-native-test:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 2G --jobs 1 -- cargo test --locked --release -p zensim-validate --example shaping_cost

shaping-cost-native-build:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 2G --jobs 1 -- cargo build --locked --release -p zensim-validate --example shaping_cost

shaping-cost-fixture study output binaries:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 2G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_cost_fixture.py --study {{quote(study)}} --output {{quote(output)}} --binary-dir {{quote(binaries)}}

shaping-cost-qualify binary workload output:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 2G --jobs 1 -- {{quote(binary)}} qualify {{quote(workload)}} {{quote(output)}}

shaping-cost-fixture-test:
    TMPDIR="$HOME/tmp" python3 -B -m unittest discover -s scripts/tests -p test_shaping_cost_fixture.py -v

# Qualify reduced uncertainty on an eligible host with its native statistics binary.
shaping-reduced-uncertainty-check panel:
    TMPDIR="$HOME/tmp" ZEN_PANEL_BIN={{quote(panel)}} "$HOME/work/claudehints/scripts/run-heavy" --mem 2G --jobs 1 -- python3 -B -m unittest discover -s scripts/tests -p 'test_shaping_bootstrap*.py'
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 1G --jobs 1 -- python3 -B -m unittest discover -s scripts/tests -p test_shaping_outer_evidence.py
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 1G --jobs 1 -- python3 -B -m unittest discover -s scripts/tests -p test_shaping_cost_finalists.py

shaping-evaluation-watch study binaries repair rescue inspector output source *args:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 1G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_evaluation_worker.py --study {{quote(study)}} --binary-dir {{quote(binaries)}} --repair-binary-dir {{quote(repair)}} --rescue-binary-dir {{quote(rescue)}} --inspector {{quote(inspector)}} --output {{quote(output)}} --remote-source {{quote(source)}} --watch {{args}}

shaping-evaluation-worker-check:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 1G --jobs 1 -- python3 -B -m unittest discover -s scripts/tests -p test_shaping_evaluation_worker.py
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 1G --jobs 1 -- python3 -B -m unittest discover -s scripts/tests -p test_shaping_supervision.py
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 1G --jobs 1 -- python3 -B -m unittest discover -s scripts/tests -p test_shaping_reduced_scope.py

shaping-external-recovery-check:
    TMPDIR="$HOME/tmp" python3 -B -m unittest discover -s scripts/tests -p test_shaping_prefix_retention.py
    TMPDIR="$HOME/tmp" python3 -B -m unittest discover -s scripts/tests -p test_shaping_partial_restart.py
    TMPDIR="$HOME/tmp" python3 -B -m unittest discover -s scripts/tests -p test_shaping_external_recovery.py
    TMPDIR="$HOME/tmp" python3 -B -m unittest discover -s scripts/tests -p test_shaping_worker_collect.py

shaping-rescue-worker-check:
    TMPDIR="$HOME/tmp" python3 -B -m unittest discover -s scripts/tests -p test_shaping_queue_transfer.py
    TMPDIR="$HOME/tmp" python3 -B -m unittest discover -s scripts/tests -p test_shaping_training_budget.py
    TMPDIR="$HOME/tmp" python3 -B -m unittest discover -s scripts/tests -p test_shaping_rescue_worker.py
    TMPDIR="$HOME/tmp" python3 -B -m unittest discover -s scripts/tests -p test_shaping_rescue.py
    TMPDIR="$HOME/tmp" python3 -B -m unittest discover -s scripts/tests -p test_shaping_worker_collect.py
    TMPDIR="$HOME/tmp" python3 -B -m unittest discover -s scripts/tests -p test_shaping_evaluation_worker.py

shaping-rescue-worker-register study cell host binaries qualification output:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 512M --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_rescue_worker.py register --study {{quote(study)}} --cell {{quote(cell)}} --host {{host}} --binaries {{quote(binaries)}} --qualification {{quote(qualification)}} --output {{quote(output)}}

shaping-rescue-worker-transfer request source:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 512M --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_rescue_worker.py transfer --request {{quote(request)}} --remote-source {{quote(source)}}

shaping-rescue-worker-run request:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 2G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_rescue_worker.py work --request {{quote(request)}}

shaping-rescue-worker-wave manifest memory jobs:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem {{memory}} --jobs {{jobs}} -- python3 -B scripts/rev4_featpot/shaping_rescue_worker.py wave --request {{quote(manifest)}}

shaping-prefix-archive-proof study archive proof:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 256M --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_prefix_retention.py prepare --study {{quote(study)}} --archive {{quote(archive)}} --proof {{quote(proof)}}

shaping-prefix-retention study cell proof:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 256M --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_prefix_retention.py watch --study {{quote(study)}} --cell {{quote(cell)}} --proof {{quote(proof)}}

shaping-threshold-probe-check:
    TMPDIR="$HOME/tmp" python3 -B -m unittest discover -s scripts/tests -p test_shaping_threshold_probe.py

shaping-threshold-probe-register study binaries output:
    TMPDIR="$HOME/tmp" python3 -B scripts/rev4_featpot/shaping_threshold_probe.py register --study {{quote(study)}} --binaries {{quote(binaries)}} --output {{quote(output)}}

shaping-threshold-probe-run study binaries output cell:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 1G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_threshold_probe.py run --study {{quote(study)}} --binaries {{quote(binaries)}} --output {{quote(output)}} --cell {{quote(cell)}}

shaping-cost-native-clippy:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 2G --jobs 1 -- cargo clippy --locked --release -p zensim-validate --example shaping_cost -- -D warnings

shaping-cost-check binary workload output:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 2G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_cost_check.py --binary {{quote(binary)}} --workload {{quote(workload)}} --output {{quote(output)}}

shaping-extraction-fixture source output:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 1G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_extract_fixture.py --source {{quote(source)}} --output {{quote(output)}}

shaping-extraction-build:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 2G --jobs 1 -- cargo build --locked --release -p zensim-validate --example shaping_extract_cost

shaping-extraction-clippy:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 2G --jobs 1 -- cargo clippy --locked --release -p zensim-validate --example shaping_extract_cost -- -D warnings

shaping-extraction-qualify binary workload output jobs:
    TMPDIR="$HOME/tmp" ZENSIM_FORMULA_REV=7 "$HOME/work/claudehints/scripts/run-heavy" --mem 4G --jobs {{jobs}} -- {{quote(binary)}} qualify {{quote(workload)}} {{quote(output)}}

shaping-extraction-fixture-check:
    python3 -B -m unittest discover -s scripts/tests -p test_shaping_extract_fixture.py

# Extraction over frozen raw inputs; vector-scoring cost waits for finalists.
shaping-extraction-bench-register study fixtures binary launcher cpu_tool root *args:
    TMPDIR="$HOME/tmp" python3 -B scripts/rev4_featpot/shaping_extract_bench.py register --study {{quote(study)}} --fixtures {{quote(fixtures)}} --binary {{quote(binary)}} --launcher {{quote(launcher)}} --cpu-tool {{quote(cpu_tool)}} --root {{quote(root)}} {{args}}

shaping-extraction-bench study root cpus:
    TMPDIR="$HOME/tmp" python3 -B scripts/rev4_featpot/shaping_extract_bench.py run --study {{quote(study)}} --root {{quote(root)}} --cpus {{quote(cpus)}}

# Profiles qualification calls; sampled runs do not supply benchmark timings.
shaping-extraction-profile binary workload output launcher cpu_tool_dir cpus jobs:
    mkdir -p {{quote(output)}}
    test ! -e {{quote(output + "/perf.data")}}
    TMPDIR="$HOME/tmp" PATH={{quote(cpu_tool_dir)}}:"$PATH" "$HOME/work/claudehints/scripts/run-heavy" --mem 4G --jobs {{jobs}} -- bash {{quote(launcher)}} {{quote(cpus)}} env ZENSIM_FORMULA_REV=7 perf record -F 199 --call-graph dwarf,8192 -o {{quote(output + "/perf.data")}} -- {{quote(binary)}} qualify {{quote(workload)}} {{quote(output + "/native")}}

shaping-cost-finalists study binary_dir output:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 2G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_cost_finalists.py --study {{quote(study)}} --binary-dir {{quote(binary_dir)}} --output {{quote(output)}}

shaping-cost-worker-check:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 1G --jobs 1 -- python3 -B -m unittest scripts.tests.test_shaping_cost_worker scripts.tests.test_shaping_cost_finalists scripts.tests.test_shaping_bootstrap_worker scripts.tests.test_shaping_training_budget

shaping-gradient-pretrained-check:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 512M --jobs 1 -- python3 -B -m unittest scripts.tests.test_shaping_gradient_pretrained

shaping-runtime-equivalence-check:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 1G --jobs 1 -- python3 -B -m unittest scripts.tests.test_shaping_runtime_equivalence scripts.tests.test_shaping_corrected_view scripts.tests.test_shaping_downstream_worker scripts.tests.test_shaping_rescue_worker scripts.tests.test_shaping_downstream_collection

shaping-runtime-equivalence-prepare study baseline candidate identity pretrained:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 512M --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_runtime_equivalence.py prepare --study {{quote(study)}} --baseline {{quote(baseline)}} --candidate {{quote(candidate)}} --identity {{quote(identity)}} --pretrained {{quote(pretrained)}}

shaping-runtime-equivalence-activate study certificate:
    TMPDIR="$HOME/tmp" python3 -B scripts/rev4_featpot/shaping_runtime_equivalence.py activate --study {{quote(study)}} --certificate {{quote(certificate)}}

# Diagnostic attachment only; this does not measure benchmark latency.
shaping-training-profile pid output seconds='15':
    test {{quote(seconds)}} -ge 1
    test {{quote(seconds)}} -le 30
    mkdir -p {{quote(output)}}
    test ! -e {{quote(output + "/perf.data")}}
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 512M --jobs 1 -- prlimit --fsize=67108864:67108864 -- perf record -e cycles:u -F 199 --call-graph dwarf,8192 -o {{quote(output + "/perf.data")}} -p {{quote(pid)}} -- sleep {{quote(seconds)}}
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 512M --jobs 1 -- perf report -i {{quote(output + "/perf.data")}} --stdio --no-children --percent-limit 0 > {{quote(output + "/perf-self.txt")}}

shaping-gradient-pretrained-register study binaries output:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 512M --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_gradient_pretrained.py register --study {{quote(study)}} --binary-dir {{quote(binaries)}} --output {{quote(output)}}

shaping-gradient-pretrained-run study registration family output:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 512M --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_gradient_pretrained.py run --study {{quote(study)}} --registration {{quote(registration)}} --family {{quote(family)}} --output {{quote(output)}}

shaping-vector-cost-run study root cpus:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 512M --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_vector_bench.py run --study {{quote(study)}} --root {{quote(root)}} --cpus {{quote(cpus)}}

shaping-cost-pipeline *args:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 512M --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_cost_pipeline.py {{args}}

shaping-cost-finalists-check:
    python3 -B -m unittest discover -s scripts/tests -p test_shaping_cost_finalists.py
    python3 -B -m unittest discover -s scripts/tests -p test_shaping_full_fit_selection.py

# Read observed training costs from completed native fits without replaying training.
shaping-training-resources-batch study output:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 1G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_training_resources.py --batch-study "{{study}}" --output "{{output}}"

shaping-control-resources study output logs:
    TMPDIR="$HOME/tmp" python3 -B scripts/rev4_featpot/shaping_control_resources.py --study "{{study}}" --output "{{output}}" {{logs}}

shaping-control-resources-check:
    TMPDIR="$HOME/tmp" python3 -B -m unittest discover -s scripts/tests -p test_shaping_control_resources.py

shaping-inner-dispatch-check:
    TMPDIR="$HOME/tmp" python3 -B -m unittest discover -s scripts/tests -p test_shaping_inner_dispatch.py

shaping-inner-dispatch study binary_dir output *args:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 2G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_rescue_dispatch.py --inner-subset --study {{quote(study)}} --binary-dir {{quote(binary_dir)}} --output {{quote(output)}} {{args}}

# Describe every registered setting within one completed paired inner partition.
shaping-completion-curve-all-settings study binaries family outer inner output:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 1G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_curve_stability.py --study {{quote(study)}} --binary-dir {{quote(binaries)}} --family {{quote(family)}} --outer {{outer}} --inner-fold {{inner}} --settings 0 1 2 3 --output {{quote(output)}}

# Compile the private row kernel's boundary/parity tests without the full trainer graph.
shaping-projection-check output:
    #!/usr/bin/env bash
    set -euo pipefail
    mkdir -p {{quote(output)}}
    printf '#[path = "%s/zensim-validate/src/mlp_train/gate_projection.rs"] mod gate_projection;\n' "$PWD" | TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 1G --jobs 1 -- rustc --edition 2024 --test - -o {{quote(output + "/gate-projection-tests")}}
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 1G --jobs 1 -- {{quote(output + "/gate-projection-tests")}}

# Reuse an identified warm dependency build; run the resulting zenbench only on an eligible host.
shaping-projection-bench-binary deps zenbench output:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 2G --jobs 1 -- rustc --edition 2024 -C opt-level=3 zensim-validate/benches/projection_rows.rs -L dependency={{quote(deps)}} --extern zenbench={{quote(zenbench)}} -o {{quote(output)}}

shaping-projection-trainer-check:
    cargo test --locked -p zensim-validate --lib -- --test-threads=1
    cargo clippy --locked -p zensim-validate --lib --bench projection_rows -- -D warnings

# The caller chooses an eligible benchmark host; zenbench owns admission and serialization.
shaping-projection-bench-run binary result:
    #!/usr/bin/env bash
    set -euo pipefail
    case "$(hostname -s)" in dev|r7900x) echo 'choose an eligible benchmark host' >&2; exit 2;; esac
    test ! -e {{quote(result)}}
    TMPDIR="$HOME/tmp" ZENBENCH_NO_CALIBRATE=1 ZENBENCH_RESULT_PATH={{quote(result)}} "$HOME/work/claudehints/scripts/run-heavy" --mem 1G --jobs 1 -- {{quote(binary)}}

# Sample existing process branches; no training is started or restarted.
shaping-cpu-accounting:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 1G --jobs 1 -- python3 -B benchmarks/cpu_branch_accounting_2026-11-07.py

# Read-only storage inventory. Run with root visibility when evaluating active Cargo users.
shaping-temp-storage output:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 1G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_temp_storage.py --output {{quote(output)}}

shaping-cargo-preserve-check:
    python3 -B -m unittest discover -s scripts/tests -p test_shaping_cargo_preserve.py

shaping-cargo-reclaim-check:
    python3 -B -m unittest discover -s scripts/tests -p test_shaping_cargo_reclaim.py

shaping-cargo-reclaim preservation receipt manifest receipt_pin manifest_pin home *args:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 1G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_cargo_reclaim.py --preservation {{quote(preservation)}} --archive-receipt {{quote(receipt)}} --archive-manifest {{quote(manifest)}} --receipt-pin {{quote(receipt_pin)}} --manifest-pin {{quote(manifest_pin)}} --home {{quote(home)}} {{args}}

# Invoke with complete process visibility; this command never deletes Cargo output.
shaping-cargo-preserve inventory output home:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 1G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_cargo_preserve.py --inventory {{quote(inventory)}} --output {{quote(output)}} --home {{quote(home)}}

shaping-host-transition-check:
    python3 -B -m unittest discover -s scripts/tests -p test_shaping_host_transition.py
    python3 -B -m unittest discover -s scripts/tests -p test_shaping_host_assignment.py

shaping-reduced-scope-check:
    python3 -B -m unittest discover -s scripts/tests -p test_shaping_reduced_scope.py

shaping-reduced-downstream-check:
    python3 -B -m unittest scripts.tests.test_shaping_reduced_scope scripts.tests.test_shaping_extension scripts.tests.test_shaping_outer_plan scripts.tests.test_shaping_outer_dispatch scripts.tests.test_shaping_outer_admission scripts.tests.test_shaping_selection_watch scripts.tests.test_shaping_rescue scripts.tests.test_shaping_rescue_dispatch scripts.tests.test_shaping_host_assignment scripts.tests.test_shaping_host_transition scripts.tests.test_shaping_outer_execution scripts.tests.test_shaping_variant_execution
    python3 -B scripts/lint_scripts.py

shaping-reduced-scope-register study approval:
    python3 -B scripts/rev4_featpot/shaping_reduced_scope.py --study {{quote(study)}} --approval {{quote(approval)}}

# Run on an admitted compute worker with an isolated Cargo output directory.
shaping-snapshot-check:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 4G --jobs 2 -- cargo test -p zensim-validate --lib mlp_train::resume -- --nocapture

shaping-snapshot-cli-check study frozen binaries output:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 1G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_snapshot_cli_check.py --study {{quote(study)}} --frozen {{quote(frozen)}} --binaries {{quote(binaries)}} --output {{quote(output)}}

# Archive each compatibility class once after its full trajectory qualification.
shaping-control-archive study source host output destination *args:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 1G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_control_archive.py --study {{quote(study)}} --source {{quote(source)}} --host {{quote(host)}} --output {{quote(output)}} --destination {{quote(destination)}} {{args}}

shaping-corrected-wave-prepare-check:
    TMPDIR="$HOME/tmp" python3 -B -m unittest scripts.tests.test_shaping_corrected_wave_prepare

shaping-corrected-wave-prepare study source binaries placements output:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 512M --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_corrected_wave_prepare.py --study {{quote(study)}} --source {{quote(source)}} --binaries {{quote(binaries)}} --placements {{quote(placements)}} --output {{quote(output)}}

# Metadata only: wait for all six native proof archives, then use the original gate.
shaping-corrected-activation-watch study *args:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 512M --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_corrected_activate_watch.py --study {{quote(study)}} {{args}}

# Scalar-only viewer inputs; existing reports and native statistics own the numbers.
shaping-scatter-prepare study dashboard output:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 1G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_scatter_prepare.py --study {{quote(study)}} --dashboard {{quote(dashboard)}} --output {{quote(output)}}

shaping-scatter-native study panel manifest output workers='4':
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 2G --jobs {{workers}} -- python3 -B scripts/rev4_featpot/shaping_scatter_native.py --study {{quote(study)}} --panel {{quote(panel)}} --manifest {{quote(manifest)}} --output {{quote(output)}} --workers {{workers}}

shaping-gradient-probe-build study source target output:
    TMPDIR="$HOME/tmp" python3 -B scripts/rev4_featpot/shaping_gradient_probe.py --study {{quote(study)}} --source {{quote(source)}} --target {{quote(target)}} --output {{quote(output)}}

shaping-gradient-probe-bench binary output:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 1G --jobs 1 -- {{quote(binary)}} {{quote(output)}}

shaping-gradient-probe-report root binary output:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 512M --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_gradient_report.py --root {{quote(root)}} --binary {{quote(binary)}} --output {{quote(output)}}

shaping-gradient-probe-report-check:
    TMPDIR="$HOME/tmp" python3 -B -m unittest scripts.tests.test_shaping_gradient_report

shaping-paired-gradient-register study original_registration binaries output:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 512M --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_paired_gradient_qualification.py register --study {{quote(study)}} --original-registration {{quote(original_registration)}} --binary-dir {{quote(binaries)}} --output {{quote(output)}}

shaping-paired-gradient-qualify study registration family output:
    TMPDIR="$HOME/tmp" python3 -B scripts/rev4_featpot/shaping_paired_gradient_qualification.py run --study {{quote(study)}} --registration {{quote(registration)}} --family {{quote(family)}} --output {{quote(output)}}

shaping-phase-snapshot-watch study output destination archive_source archive_mount *args:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 512M --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_phase_snapshot_watch.py watch --study {{quote(study)}} --output {{quote(output)}} --destination {{quote(destination)}} --archive-source {{quote(archive_source)}} --archive-mount {{quote(archive_mount)}} {{args}}

shaping-training-budget-check:
    TMPDIR="$HOME/tmp" python3 -B -m unittest scripts.tests.test_shaping_training_budget

# Execute one registered descriptive external bake comparison on an admitted worker.
shaping-external-score root study binaries tree_binary copies cell dataset:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 1G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_external_validation.py --root {{quote(root)}} --study {{quote(study)}} --binaries {{quote(binaries)}} --tree-binary {{quote(tree_binary)}} --copies {{quote(copies)}} --cell {{quote(cell)}} --dataset {{quote(dataset)}}

shaping-external-publish root *args:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 512M --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_external_publish.py --root {{quote(root)}} {{args}}

shaping-external-publish-check:
    TMPDIR="$HOME/tmp" python3 -B -m unittest scripts.tests.test_shaping_external_publish

shaping-external-browser url output modules expected_models:
    TMPDIR="$HOME/tmp" NODE_PATH={{quote(modules)}} node scripts/tests/shaping_external_browser.cjs {{quote(url)}} {{quote(output)}} {{quote(expected_models)}}

shaping-compact-worker-check:
    TMPDIR="$HOME/tmp" python3 -B -m unittest scripts.tests.test_shaping_compact_worker

shaping-snapshot-batch-check:
    TMPDIR="$HOME/tmp" python3 -B -m unittest scripts.tests.test_shaping_snapshot_batch scripts.tests.test_shaping_snapshot_retention scripts.tests.test_shaping_checkpoint_retention scripts.tests.test_shaping_open_file_guard

# Native opaque-vector inference helpers; build away from the coordinator host.
shaping-queue-placement-check:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 512M --jobs 1 -- python3 -B -m unittest scripts.tests.test_shaping_queue_placement scripts.tests.test_shaping_host_transition

shaping-selected-rescue-check:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 1G --jobs 1 -- python3 -B -m unittest scripts.tests.test_shaping_rescue_wave_prepare scripts.tests.test_shaping_rescue_worker scripts.tests.test_shaping_corrected_downstream scripts.tests.test_shaping_rescue_dispatch

shaping-stable-pairs-check:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 1G --jobs 1 -- python3 -B -m unittest scripts.tests.test_shaping_stable_pairs scripts.tests.test_shaping_stable_export_launch scripts.tests.test_shaping_reexport scripts.tests.test_shaping_reexport_cli scripts.tests.test_shaping_selection scripts.tests.test_shaping_selection_watch scripts.tests.test_shaping_rescue_dispatch scripts.tests.test_shaping_rescue scripts.tests.test_shaping_corrected_view scripts.tests.test_shaping_progress scripts.tests.test_shaping_convergence scripts.tests.test_shaping_corrected_dashboard

shaping-stable-pairs-register study qualification source:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 1G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_stable_pairs.py --study {{quote(study)}} --qualification {{quote(qualification)}} --source {{quote(source)}}

shaping-stable-pairs-archive-inputs study root:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 1G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_stable_pairs_archive.py --study {{quote(study)}} --root {{quote(root)}}

shaping-stable-export-prepare study source host output *args:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 1G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_stable_export_prepare.py --study {{quote(study)}} --source {{quote(source)}} --host {{quote(host)}} --output {{quote(output)}} {{args}}

shaping-stable-export-collect wave source output:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 1G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_stable_export_collect.py --wave {{quote(wave)}} --source {{quote(source)}} --output {{quote(output)}}

shaping-stable-export-launch wave:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 1G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_stable_export_launch.py --wave {{quote(wave)}}

shaping-stable-export-watch study source host output *args:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 1G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_stable_export_watch.py --study {{quote(study)}} --source {{quote(source)}} --host {{quote(host)}} --output {{quote(output)}} {{args}}

shaping-stable-export-watch-check:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 1G --jobs 1 -- python3 -B -m unittest scripts.tests.test_shaping_stable_export_watch scripts.tests.test_shaping_stable_export_launch

shaping-curve-metadata-register study collection output:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 1G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_curve_metadata.py register --study {{quote(study)}} --collection {{quote(collection)}} --output {{quote(output)}}

shaping-curve-metadata-analyse study manifest source output:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 1G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_curve_metadata.py analyse --study {{quote(study)}} --manifest {{quote(manifest)}} --source {{quote(source)}} --output {{quote(output)}}

shaping-curve-metadata-check:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 1G --jobs 1 -- python3 -B -m unittest scripts.tests.test_shaping_curve_metadata scripts.tests.test_shaping_curve_stability

shaping-dashboard-browser url output modules family='':
    TMPDIR="$HOME/tmp" NODE_PATH={{quote(modules)}} "$HOME/work/claudehints/scripts/run-heavy" --mem 1G --jobs 1 -- node scripts/tests/shaping_dashboard_browser.cjs {{quote(url)}} {{quote(output)}} {{quote(family)}}

shaping-selected-rescue-prepare study source binaries qualification outer family placements output:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 1G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_rescue_wave_prepare.py --study {{quote(study)}} --source {{quote(source)}} --binaries {{quote(binaries)}} --qualification {{quote(qualification)}} --outer {{outer}} --selected-family {{quote(family)}} --placements {{quote(placements)}} --output {{quote(output)}}

# Native opaque-vector inference helpers; build away from the coordinator host.
shaping-external-native-build target:
    TMPDIR="$HOME/tmp" CARGO_TARGET_DIR={{quote(target)}} CARGO_PROFILE_DEV_DEBUG=0 CARGO_PROFILE_DEV_OPT_LEVEL=1 "$HOME/work/claudehints/scripts/run-heavy" --mem 4G --jobs 1 -- "$HOME/.cargo/bin/cargo" build --locked -p zensim-validate --bin corrhead_parity --bin bake_dial_refit

shaping-external-native-qualify binaries tree_model tree_parity linear_model linear_inputs linear_reference output:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 1G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_external_native_qualify.py --binaries {{quote(binaries)}} --tree-model {{quote(tree_model)}} --tree-parity {{quote(tree_parity)}} --linear-model {{quote(linear_model)}} --linear-inputs {{quote(linear_inputs)}} --linear-reference {{quote(linear_reference)}} --output {{quote(output)}}

# Stage the already registered inner validation closure; this performs no fitting.
shaping-evaluation-inputs mode study manifest *args:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 1G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_evaluation_inputs.py {{quote(mode)}} --study {{quote(study)}} --manifest {{quote(manifest)}} {{args}}

# Research shaping extension; keep build outputs in an isolated worker directory.
shaping-plain-native-check target:
    TMPDIR="$HOME/tmp" CARGO_TARGET_DIR={{quote(target)}} "$HOME/work/claudehints/scripts/run-heavy" --mem 8G --jobs 6 -- cargo test --locked -p zensim-validate --lib mlp_train::input_shaping -- --nocapture

shaping-plain-native-build target:
    TMPDIR="$HOME/tmp" CARGO_TARGET_DIR={{quote(target)}} "$HOME/work/claudehints/scripts/run-heavy" --mem 8G --jobs 6 -- cargo build --locked -p zensim-validate --bin zensim_mlp_train

shaping-plain-native-clippy target:
    TMPDIR="$HOME/tmp" CARGO_TARGET_DIR={{quote(target)}} "$HOME/work/claudehints/scripts/run-heavy" --mem 8G --jobs 6 -- cargo clippy --locked -p zensim-validate --lib -- -D warnings

shaping-rescue-host-launch-test:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 512M --jobs 1 -- python3 -B -m unittest scripts.tests.test_shaping_rescue_host_launch

shaping-rescue-host-launch wave source host output:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 1G --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_rescue_host_launch.py --wave {{quote(wave)}} --source {{quote(source)}} --host {{quote(host)}} --output {{quote(output)}}

shaping-stable-rescue-export-test:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 1G --jobs 1 -- python3 -B -m unittest scripts.tests.test_shaping_stable_rescue_export scripts.tests.test_shaping_stable_export_watch scripts.tests.test_shaping_stable_export_launch

shaping-pending-rescue-placement-test:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 512M --jobs 1 -- python3 -B -m unittest scripts.tests.test_shaping_pending_rescue_placement

shaping-pending-rescue-retire study limit output:
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 512M --jobs 1 -- python3 -B scripts/rev4_featpot/shaping_pending_rescue_placement.py --study {{quote(study)}} --limit {{limit}} --output {{quote(output)}}
