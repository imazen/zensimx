"""Matched-continuation coordinator: registered route, priority wait and exact control. Invented files only."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts.byconstruction import aic_matched_continuation as owner


class RegisteredRoute(unittest.TestCase):
    def test_models_are_initial_plus_fixed_epochs_of_both_arms(self):
        self.assertEqual(owner.labels(), ["Initial", "MixedAIC16@e7", "MixedAIC16@e59", "AICOnly@e7", "AICOnly@e59"])

    def test_fit_command_is_the_diagnostic_route_without_packing(self):
        with tempfile.TemporaryDirectory(dir=Path.home() / "tmp") as tmp:
            run = Path(tmp)
            (run / "cols.txt").write_text("13,14\n")
            plan = dict(root="/invented/root", columns=str(run / "cols.txt"), initial=dict(directory="/invented/Mlic/full_s0"),
                        registered_inputs=dict(aic_table=dict(path="/invented/aic_full_fit.parquet")))
            for arm, population in owner.ARMS.items():
                command = owner.fit_command(plan, run, arm)
                options, rest = {}, command[3:]
                while rest:
                    flag, *rest = rest
                    options[flag] = rest.pop(0) if rest and not rest[0].startswith("--") else True
                self.assertEqual(command[2], str(run / "source/scripts/rev4_featpot/v2_confirm_fit.py"))
                self.assertEqual(options["--diagnostic-population"], population)
                self.assertEqual((float(options["--fine-tune-epochs"]), float(options["--aic-weight"]),
                                  float(options["--fine-tune-lr"])), (60, 16, 0.00001))
                self.assertEqual(options["--init-from"], "/invented/Mlic/full_s0/refit/last.bin")
                self.assertEqual(options["--dest"], str(run / "arms" / arm / "full_s0"))
                self.assertEqual(options["--columns"], "13,14")
                self.assertNotIn("--pack-production", options)
                self.assertIs(options["--strict-admission"], True)
                self.assertIs(options["--train-only"], True)


class PlanConfiguration(unittest.TestCase):
    def test_fx2_plan_runs_both_recipes_all_folds_and_three_initial_seeds(self):
        initial = {str(s): dict(directory=f"/invented/full_s{s}") for s in range(3)}
        config = dict(token="fx2", input_sha256={})
        spec = owner.SPEC.replace("fx1", "fx2")
        with patch.object(owner, "fx2_prerequisites", return_value=(spec, Path("/invented/cols"), initial, config)), \
                patch.object(owner, "seven_fold_plan") as freeze:
            owner.fx2_plan(Path("/invented/run"), "HEAD", Path("t"), Path("m"),
                           Path("/invented/study"), Path("/invented/reproduction"))
        value = freeze.call_args.kwargs
        self.assertEqual(value["arms"], dict(XrefAIC256="mixed-aic256-xref", AICOnly="aic-only"))
        self.assertEqual(value["seeds"], owner.SEEDS)
        self.assertEqual(value["checkpoints"], (7, 15, 29, 59))
        self.assertEqual(len(owner.fold_runs(value)), 42)
        self.assertEqual((value["spec"], value["initial"], value["input_configuration"]), (spec, initial, config))
        self.assertEqual(value["extra"]["aic_weights"], dict(XrefAIC256=256.0, AICOnly=16.0))

    def test_fx2_prerequisites_refuse_failed_controls_before_any_model_read(self):
        with tempfile.TemporaryDirectory(dir=Path.home() / "tmp") as tmp:
            study, repro = Path(tmp) / "study", Path(tmp) / "repro"
            study.mkdir()
            repro.mkdir()
            (study / "CONTROL.json").write_text('{"pass":false}')
            with self.assertRaisesRegex(ValueError, "stage-1 control"):
                owner.fx2_prerequisites(study, repro)
            (study / "CONTROL.json").write_text('{"pass":true}')
            (repro / "CONTROL.json").write_text('{"pass":false}')
            with self.assertRaisesRegex(ValueError, "two-arm fx1 reproduction"):
                owner.fx2_prerequisites(study, repro)

    def test_inherited_prerequisites_cannot_be_silently_repinned(self):
        with tempfile.TemporaryDirectory(dir=Path.home() / "tmp") as tmp:
            path = Path(tmp) / "control.json"
            path.write_text('{"pass":true}')
            configuration = dict(input_sha256={str(path): owner.sha(path)})
            self.assertEqual(owner.configuration_inputs(configuration), [path])
            path.write_text('{"pass":false}')
            with self.assertRaisesRegex(ValueError, "prerequisite changed"):
                owner.configuration_inputs(configuration)

    def test_explicit_initials_cannot_fall_back_to_mlic(self):
        value = dict(schema=owner.SEEDS_SCHEMA, root="/invented/fx2-root",
                     input_configuration=dict(token="fx2"), initial={str(s): dict(seed=s) for s in range(3)})
        self.assertEqual(owner.recipe_point(value), (None, value["initial"], Path(value["root"])))
        with self.assertRaisesRegex(ValueError, "all three initial seeds"):
            owner.recipe_point(dict(value, initial={"0": value["initial"]["0"]}))

    def test_full_plan_preserves_explicit_root_columns_and_initial_models(self):
        with tempfile.TemporaryDirectory(dir=Path.home() / "tmp") as tmp:
            root = Path(tmp)
            source, common, point = root / "fx2-root", root / "old-root", root / "point"
            real = source / "wide/main/real"
            real.mkdir(parents=True)
            anchor = real / "cid22_fit.parquet"
            anchor.write_bytes(b"invented TRAIN anchor")
            (real / "receipt.json").write_text(json.dumps(dict(legs=dict(cid22=dict(fit=dict(
                rel="wide/main/real/cid22_fit.parquet", sha256=owner.sha(anchor)))))))
            initial = {}
            for seed in range(3):
                dest = root / f"initial_s{seed}"
                (dest / "refit").mkdir(parents=True)
                (dest / "refit/production-f16.bin").write_bytes(f"seed {seed}".encode())
                initial[str(seed)] = dict(directory=str(dest), packed_sha256=owner.sha(dest / "refit/production-f16.bin"))
            configuration = dict(token="fx2", input_sha256={})
            spec = owner.SPEC.replace("fx1", "fx2")
            value = dict(schema=owner.SEEDS_SCHEMA, spec=spec, root=str(source), columns="/invented/fx2-cols",
                         input_configuration=configuration, initial=initial, registration=dict(section="invented"),
                         arms={"XrefAIC256": "mixed-aic256-xref", "AICOnly": "aic-only"})
            point.mkdir()
            (point / "PLAN.json").write_text(json.dumps(value))
            fold, fits = root / "fold.json", root / "fits.json"
            fold.write_text(json.dumps(dict(common_root=str(common), columns="/invented/old-cols")))
            fits.write_text(json.dumps(dict(continued=[dict(initial[str(s)], seed=s, arm="MlicFT16x8") for s in range(3)])))
            inputs = dict(owner.INPUTS, fold_plan=(str(fold), owner.sha(fold)), full_fits=(str(fits), owner.sha(fits)))
            with patch.object(owner, "INPUTS", inputs), patch.object(owner, "seven_fold_plan") as freeze, \
                    patch.object(owner, "full_record", side_effect=lambda fits, arm, seed: initial[str(seed)]):
                owner.full_plan(root / "run", "HEAD", root / "t", root / "m", point, "AICOnly", 29, point)
            kwargs = freeze.call_args.kwargs
            self.assertEqual((kwargs["root"], kwargs["spec"], kwargs["columns"]), (source, spec, value["columns"]))
            self.assertEqual(kwargs["initial"], initial)
            self.assertEqual(kwargs["input_configuration"], configuration)
            self.assertEqual(kwargs["extra"]["anchor"]["path"], str(anchor))
            self.assertEqual({k for k in kwargs["extra"]["baselines"] if k.startswith("Initial|")},
                             {f"Initial|s{s}" for s in range(3)})

    def test_explicit_fx2_spec_reaches_fit_and_omission_routes(self):
        with tempfile.TemporaryDirectory(dir=Path.home() / "tmp") as tmp:
            run = Path(tmp)
            cols = run / "columns.txt"
            cols.write_text("0,1,2")
            spec = owner.SPEC.replace("3b7bd5ebe929", "45acffa603e8").replace("fx1", "fx2")
            value = dict(spec=spec, root="/invented/fx2-root", columns=str(cols),
                         arms={"AICOnly": "aic-only"}, epochs=60, lr=0.00001,
                         cells={"kadid_s0": dict(aic_fold=0, directory="/invented/initial", exclusions=[])},
                         aic_tables={"0": dict(path="/invented/aic_k0_fit.parquet")})
            fit = owner.fit_command(value, run, "AICOnly", 0, "/invented/initial", "/invented/table", run / "fit")
            omission = owner.omission_command(value, run, "AICOnly", "kadid", 0, run / "omission")
            for command in (fit, omission):
                self.assertEqual(command[command.index("--spec") + 1], spec)
                self.assertEqual(command[command.index("--root") + 1], value["root"])
                self.assertEqual(command[command.index("--columns") + 1], "0,1,2")
                self.assertEqual(command[command.index("--init-from") + 1], "/invented/initial/refit/last.bin")

    def test_exact_ids_and_declaration_pin_are_required(self):
        from v2_common import derived_declaration
        with tempfile.TemporaryDirectory(dir=Path.home() / "tmp") as tmp:
            columns = Path(tmp) / "columns.txt"
            for token, width in (("fx1", 410), ("fx2", 492)):
                spec = owner.SPEC.replace("fx1", token)
                declaration, pin, content = derived_declaration(token)
                columns.write_text(",".join(map(str, content["direct"])))
                binding = owner.declaration_binding(spec, columns)
                self.assertEqual((binding["token"], binding["sha256"], binding["direct"]), (token, pin, width))
                # Same width with a different raw ID is not the same declaration.
                columns.write_text(",".join(map(str, [99999, *content["direct"][1:]])))
                with self.assertRaisesRegex(ValueError, "columns differ"):
                    owner.declaration_binding(spec, columns)

    def test_panel_root_and_wrapper_follow_only_explicit_configuration(self):
        with tempfile.TemporaryDirectory(dir=Path.home() / "tmp") as tmp:
            run = Path(tmp)
            fold = dict(common_root="/invented/old-root", warmstart_source="/invented/old-source", bin="/invented/old-bin")
            path = run / "fold.json"
            path.write_text(json.dumps(fold))
            value = dict(root="/invented/fx2-root", columns="/invented/fx2-cols",
                         registered_inputs=dict(fold_plan=dict(path=str(path))))
            self.assertEqual(owner.execution_fold(value, run), fold)
            explicit = owner.execution_fold(dict(value, input_configuration=dict(token="fx2")), run)
            self.assertEqual(explicit["common_root"], value["root"])
            self.assertEqual(explicit["warmstart_source"], str(run / "source"))
            self.assertEqual(explicit["bin"], str(run / "bin"))


class PriorityWait(unittest.TestCase):
    def state(self, root, **value):
        (root / "STATE.json").write_text(json.dumps(dict(failures=[], **value)))

    def test_followthrough_terminal_states_and_dead_workers(self):
        with tempfile.TemporaryDirectory(dir=Path.home() / "tmp") as tmp:
            root = Path(tmp)
            inputs = dict(owner.INPUTS, followthrough=(str(root / "CONFIG.json"), "invented"))
            live = owner.process_snapshot(os.getpid())
            child = subprocess.Popen([sys.executable, "-c", "pass"])
            child.wait()
            with patch.object(owner, "INPUTS", inputs):
                self.state(root, stage=owner.FOLLOWTHROUGH_DONE, worker=live)
                self.assertEqual(owner.followthrough_state(), "complete")
                self.state(root, stage="full-panels", worker=live)
                self.assertEqual(owner.followthrough_state(), "running")
                self.state(root, stage="failed", worker=live)
                self.assertEqual(owner.followthrough_state(), "failed")
                self.state(root, stage="full-panels", worker=dict(pid=child.pid, start_ticks=1))
                self.assertEqual(owner.followthrough_state(), "failed")


class ExactControl(unittest.TestCase):
    def test_reproduction_status_refuses_a_stale_running_stage(self):
        with tempfile.TemporaryDirectory(dir=Path.home() / "tmp") as tmp:
            root = Path(tmp)
            (root / "STATE.json").write_text(json.dumps(dict(stage="reproduction-control", failures=[],
                worker=dict(pid=os.getpid(), start_ticks=-1))))
            with patch.object(owner, "compute_lock", return_value=dict(held_by_worker_ancestor=False)):
                owner.reproduction_status(root / "PLAN.json")
            current = json.loads((root / "CURRENT.json").read_text())
            self.assertFalse(current["live"])
            self.assertFalse(current["complete"])
            self.assertIn("exited without completion", current["failures"][0])

    def test_strip_compare_uses_copies_and_reports_differences(self):
        with tempfile.TemporaryDirectory(dir=Path.home() / "tmp") as tmp:
            root = Path(tmp)
            tool = root / "strip-tool"
            # Invented stand-in for the native owner: copies --in to --out.
            tool.write_text('#!/bin/sh\nwhile [ $# -gt 0 ]; do case "$1" in --in) i=$2;; --out) o=$2;; esac; shift; done\ncp "$i" "$o"\n')
            tool.chmod(0o755)
            first, second = root / "a.bin", root / "b.bin"
            first.write_bytes(b"same")
            second.write_bytes(b"same")
            self.assertTrue(owner.nonrepro_identical(tool, first, second, root / "equal")["identical"])
            second.write_bytes(b"different")
            result = owner.nonrepro_identical(tool, first, second, root / "unequal")
            self.assertFalse(result["identical"])
            self.assertEqual((first.read_bytes(), second.read_bytes()), (b"same", b"different"))
            with self.assertRaises(FileExistsError):
                owner.nonrepro_identical(tool, first, second, root / "unequal")


class PlanFreeze(unittest.TestCase):
    def test_existing_run_and_changed_registered_inputs_refuse(self):
        with tempfile.TemporaryDirectory(dir=Path.home() / "tmp") as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, "preserve"):
                owner.plan(root, "HEAD", root / "t", root / "m")
            changed = root / "input.json"
            changed.write_text("{}")
            with patch.object(owner, "INPUTS", dict(fold_plan=(str(changed), "0" * 64))):
                with self.assertRaisesRegex(ValueError, "registered input changed"):
                    owner.plan(root / "fresh", "HEAD", root / "t", root / "m")
            self.assertFalse((root / "fresh").exists())


class SourceFoldStudy(unittest.TestCase):
    def test_registered_inventory_is_two_populations_three_seeds_seven_folds(self):
        runs = owner.fold_runs()
        self.assertEqual(len(runs), 42)
        self.assertEqual(len(set(runs)), 42)
        self.assertEqual({r[0] for r in runs}, set(owner.ARMS))
        self.assertEqual({r[1] for r in runs}, {0, 1, 2})
        self.assertEqual({r[2] for r in runs}, set(range(7)))

    def test_fold_fit_binds_its_seed_initial_model_and_fold_table(self):
        with tempfile.TemporaryDirectory(dir=Path.home() / "tmp") as tmp:
            run = Path(tmp)
            (run / "cols.txt").write_text("13,14\n")
            plan = dict(root="/invented/root", columns=str(run / "cols.txt"))
            command = owner.fit_command(plan, run, "AICOnly", 2, "/invented/Mlic/full_s2",
                                        "/invented/aic_k5_fit.parquet", run / "arms/AICOnly/s2_k5")
            self.assertEqual(command[command.index("--seed-index") + 1], "2")
            self.assertEqual(command[command.index("--init-from") + 1], "/invented/Mlic/full_s2/refit/last.bin")
            self.assertEqual(command[command.index("--aic-public-train") + 1], "/invented/aic_k5_fit.parquet")
            self.assertEqual(command[command.index("--dest") + 1], str(run / "arms/AICOnly/s2_k5"))
            self.assertEqual(command[command.index("--diagnostic-population") + 1], "aic-only")

    def test_out_of_fold_uses_only_each_models_held_out_sources(self):
        import numpy as np
        folds = np.repeat(np.arange(7), 3)
        values = {f"A|s0|k{k}": np.where(folds == k, 100.0 + k, -1.0) for k in range(7)}
        oof = owner.out_of_fold(values, folds, "A|s0")
        self.assertTrue(np.array_equal(oof, 100.0 + folds))
        values["A|s0|k3"] = np.where(folds == 3, np.nan, -1.0)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            owner.out_of_fold(values, folds, "A|s0")

    def test_gates_queue_end_requires_a_terminal_line(self):
        with tempfile.TemporaryDirectory(dir=Path.home() / "tmp") as tmp:
            log = Path(tmp) / "queue.log"
            with patch.object(owner, "GATES_LOG", log):
                self.assertFalse(owner.gates_finished())
                log.write_text("2026-10-27T09:24:58Z waiting for the matched-continuation diagnostic\n")
                self.assertFalse(owner.gates_finished())
                log.write_text(log.read_text() + "2026-10-27T14:00:00Z GATES-DONE rc=0\n")
                self.assertTrue(owner.gates_finished())
                log.write_text("2026-10-27T14:00:00Z GATES-EXIT rc=1\n")
                self.assertTrue(owner.gates_finished())

    def test_frontier_fits_use_each_arms_registered_weight_and_inventory(self):
        with tempfile.TemporaryDirectory(dir=Path.home() / "tmp") as tmp:
            run = Path(tmp)
            (run / "cols.txt").write_text("13,14\n")
            plan = dict(root="/invented/root", columns=str(run / "cols.txt"), arms=owner.FRONTIER_ARMS, seeds=[0])
            for arm, (population, weight) in zip(owner.FRONTIER_ARMS, (("mixed-aic64", 64.0), ("mixed-aic256", 256.0),
                                                                      ("mixed-aic1024", 1024.0))):
                command = owner.fit_command(plan, run, arm, 0, "/invented/Mlic/full_s0", "/invented/aic_k3_fit.parquet",
                                            run / "arms" / arm / "s0_k3")
                self.assertEqual(command[command.index("--diagnostic-population") + 1], population)
                self.assertEqual(float(command[command.index("--aic-weight") + 1]), weight)
                self.assertEqual(int(command[command.index("--fine-tune-epochs") + 1]), 60)
            runs = owner.fold_runs(plan)
            self.assertEqual(len(runs), 21)
            self.assertEqual({r[0] for r in runs}, set(owner.FRONTIER_ARMS))
            self.assertEqual({r[1] for r in runs}, {0})
            # The registered source-fold study keeps its own inventory and weight.
            self.assertEqual(len(owner.fold_runs()), 42)
            matched = owner.fit_command(dict(plan, arms=owner.ARMS), run, "AICOnly", 0, "/invented/i", "/invented/t", run)
            self.assertEqual(float(matched[matched.index("--aic-weight") + 1]), 16.0)

    def test_single_seed_deltas_carry_no_standard_error(self):
        one = owner.paired_delta({0: 0.95}, {0: 0.91}, [0])
        self.assertAlmostEqual(one["mean"], 0.04)
        self.assertIsNone(one["seed_unit_se"])
        three = owner.paired_delta({0: 0.95, 1: 0.94, 2: 0.96}, {0: 0.91, 1: 0.91, 2: 0.92}, [0, 1, 2])
        self.assertIsNotNone(three["seed_unit_se"])

    def test_plan_schema_accepts_only_fold_and_frontier_plans(self):
        with tempfile.TemporaryDirectory(dir=Path.home() / "tmp") as tmp:
            path = Path(tmp) / "PLAN.json"
            for schema in (owner.FOLD_SCHEMA, owner.FRONTIER_SCHEMA, owner.RECOVERY_SCHEMA):
                path.write_text(json.dumps(dict(schema=schema)))
                self.assertEqual(owner.plan_schema(path), schema)
            path.write_text(json.dumps(dict(schema="aic-matched-continuation-plan-v1")))
            with self.assertRaisesRegex(ValueError, "frontier plan required"):
                owner.plan_schema(path)

    def test_recovery_continues_from_each_folds_checkpoint_at_its_weight(self):
        with tempfile.TemporaryDirectory(dir=Path.home() / "tmp") as tmp:
            run = Path(tmp)
            (run / "cols.txt").write_text("13,14\n")
            plan = dict(root="/invented/root", columns=str(run / "cols.txt"), arms=owner.RECOVERY_ARMS, seeds=[0])
            start = "/invented/AICOnly/s0_k2/ckpt/ckpt_epoch059.bin"
            command = owner.fit_command(plan, run, "RecoverAIC64", 0, "/invented/Mlic/full_s0",
                                        "/invented/aic_k2_fit.parquet", run / "arms/RecoverAIC64/s0_k2", start)
            self.assertEqual(command[command.index("--init-from") + 1], start)
            self.assertEqual(float(command[command.index("--aic-weight") + 1]), 64.0)
            self.assertEqual(command[command.index("--aic-public-train") + 1], "/invented/aic_k2_fit.parquet")
            low = owner.fit_command(plan, run, "RecoverAIC16", 0, "/invented/Mlic/full_s0", "/invented/t", run, start)
            self.assertEqual(low[low.index("--diagnostic-population") + 1], "mixed-aic16")
            self.assertEqual(len(owner.fold_runs(plan)), 21)

    def test_predecessor_is_a_completed_or_failed_run(self):
        with tempfile.TemporaryDirectory(dir=Path.home() / "tmp") as tmp:
            run = Path(tmp)
            plan = dict(after_run=str(run))
            self.assertFalse(owner.predecessor_finished(plan))
            (run / "STATE.json").write_text(json.dumps(dict(stage="fit-folds")))
            self.assertFalse(owner.predecessor_finished(plan))
            (run / "STATE.json").write_text(json.dumps(dict(stage="failed")))
            self.assertTrue(owner.predecessor_finished(plan))
            (run / "STATE.json").write_text(json.dumps(dict(stage="fit-folds")))
            (run / "COMPLETE.json").write_text("{}")
            self.assertTrue(owner.predecessor_finished(plan))
            log = run / "queue.log"
            log.write_text("2026-10-27T14:00:00Z GATES-DONE\n")
            self.assertTrue(owner.predecessor_finished(dict(after_log=str(log))))

    def test_studied_epochs_beyond_the_declared_fixed_checkpoints_use_the_dumps(self):
        with tempfile.TemporaryDirectory(dir=Path.home() / "tmp") as tmp:
            dest = Path(tmp)
            (dest / "ckpt").mkdir()
            for e in range(owner.EPOCHS):
                (dest / "ckpt" / f"ckpt_epoch{e:03d}.bin").write_bytes(f"epoch {e}".encode())
            def result(fixed):
                (dest / "result.json").write_text(json.dumps(dict(epochs=owner.EPOCHS, selection=dict(selected_epoch=owner.EPOCHS - 1),
                    diagnostic=dict(population="mixed-aic16", fixed_checkpoints={
                        str(e): dict(sha256=owner.sha(dest / "ckpt" / f"ckpt_epoch{e:03d}.bin")) for e in fixed}))))
            result([7, 59])
            owner.fixed_outputs(dest, "mixed-aic16", "x", (7, 15, 29, 59))
            (dest / "ckpt/ckpt_epoch059.bin").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "incomplete"):
                owner.fixed_outputs(dest, "mixed-aic16", "x", (7, 15, 29, 59))
            result([7, 59])
            with self.assertRaisesRegex(ValueError, "incomplete"):
                owner.fixed_outputs(dest, "mixed-aic16", "x", (15, 29))

    def test_fold_plan_refuses_an_existing_run(self):
        with tempfile.TemporaryDirectory(dir=Path.home() / "tmp") as tmp:
            with self.assertRaisesRegex(ValueError, "preserve"):
                owner.fold_plan(Path(tmp), "HEAD", Path(tmp) / "t", Path(tmp) / "m")


class RecipeStage(unittest.TestCase):
    def test_recipe_inputs_ride_only_on_populations_with_older_legs(self):
        with tempfile.TemporaryDirectory(dir=Path.home() / "tmp") as tmp:
            run = Path(tmp)
            (run / "cols.txt").write_text("13,14\n")
            flags = ["--matched-codec-full", "--matched-codec-manifest", "/invented/m.json", "--matched-codec-sha256", "ab"]
            plan = dict(root="/invented/MlicOrder/root", columns=str(run / "cols.txt"), fit_extra_args=flags,
                        arms={"MlicOrderAIC256": "mixed-aic256", "MlicOrderAICOnly": "aic-only"}, seeds=[0])
            mixed = owner.fit_command(plan, run, "MlicOrderAIC256", 0, "/invented/MlicOrder/full_s0", "/invented/t", run)
            self.assertEqual(mixed[-len(flags):], flags)
            self.assertEqual(float(mixed[mixed.index("--aic-weight") + 1]), 256.0)
            self.assertEqual(mixed[mixed.index("--root") + 1], "/invented/MlicOrder/root")
            alone = owner.fit_command(plan, run, "MlicOrderAICOnly", 0, "/invented/MlicOrder/full_s0", "/invented/t", run)
            self.assertNotIn("--matched-codec-full", alone)
            self.assertEqual(alone[alone.index("--diagnostic-population") + 1], "aic-only")

    def test_recipe_plans_use_their_populations_protocol(self):
        with tempfile.TemporaryDirectory(dir=Path.home() / "tmp") as tmp:
            path = Path(tmp) / "PLAN.json"
            path.write_text(json.dumps(dict(schema=owner.RECIPE_SCHEMA)))
            self.assertEqual(owner.plan_schema(path), owner.RECIPE_SCHEMA)
            for population, protocol in (("mixed-aic16", owner.PROTOCOL), ("mixed-aic256", owner.FRONTIER_PROTOCOL)):
                path.write_text(json.dumps(dict(schema=owner.RECIPE_SCHEMA, protocol=protocol, arms={"x": population},
                                                registered_inputs={}, input_sha256={}, bin_sha256={},
                                                meta_tool_sha256="", source_archive_sha256="")))
                with self.assertRaises(FileNotFoundError):  # passes the protocol check, then hashes its inputs
                    owner.checked(path, owner.RECIPE_SCHEMA)
                other = owner.FRONTIER_PROTOCOL if protocol == owner.PROTOCOL else owner.PROTOCOL
                path.write_text(json.dumps(dict(schema=owner.RECIPE_SCHEMA, protocol=other, arms={"x": population})))
                with self.assertRaisesRegex(ValueError, "schema differs"):
                    owner.checked(path, owner.RECIPE_SCHEMA)

    def test_unknown_recipe_refuses_before_any_read(self):
        with self.assertRaisesRegex(ValueError, "stage B recipe"):
            owner.recipe_inputs("Mlic")

    def test_recipe_runs_take_the_rehearsal_populations_on_their_own_root(self):
        with tempfile.TemporaryDirectory(dir=Path.home() / "tmp") as tmp:
            run = Path(tmp)
            (run / "cols.txt").write_text("13,14\n")
            for population, weight in (("mixed-aic256-xref", 256.0), ("mixed-aic1024-xref", 1024.0)):
                arm = "MlicVote" + owner.RECIPE_ARM[population]
                self.assertEqual(arm, f"MlicVoteXrefAIC{int(weight)}")
                plan = dict(root="/invented/MlicVote/root", columns=str(run / "cols.txt"), arms={arm: population}, seeds=[0])
                command = owner.fit_command(plan, run, arm, 0, "/invented/MlicVote/full_s0", "/invented/t", run)
                self.assertEqual(command[command.index("--diagnostic-population") + 1], population)
                self.assertEqual(float(command[command.index("--aic-weight") + 1]), weight)
                self.assertEqual(command[command.index("--root") + 1], "/invented/MlicVote/root")
                path = run / "PLAN.json"
                path.write_text(json.dumps(dict(schema=owner.RECIPE_SCHEMA, protocol=owner.diagnostic_protocol(population)[0],
                                                arms={arm: population}, registered_inputs={}, input_sha256={}, bin_sha256={},
                                                meta_tool_sha256="", source_archive_sha256="")))
                with self.assertRaises(FileNotFoundError):  # passes the rehearsal protocol check, then hashes its inputs
                    owner.checked(path, owner.RECIPE_SCHEMA)

    def test_multi_population_recipe_plans_refuse_before_any_read(self):
        both = ["mixed-aic256-xref", "mixed-aic1024-xref"]
        two = [(Path("/invented/d"), "XrefAIC256"), (Path("/invented/d"), "XrefAIC1024")]
        for populations, compares, start, message in (
                (["mixed-aic256", "mixed-aic256-xref"], two, None, "one registered protocol"),
                (both, two[:1], None, "one comparison point"),
                (["mixed-aic256-xref", "mixed-aic256-xref"], two, None, "one comparison point"),
                (both, two, ("/invented/d", "XrefAIC256", 7), "one population")):
            with self.assertRaisesRegex(ValueError, message):
                owner.recipe_plan(Path("/invented/run"), "0" * 40, Path("/invented/t"), Path("/invented/m"), "MlicVote",
                                  populations, Path("/invented/after"), compares, start)


class ConfidenceContinuation(unittest.TestCase):
    def test_manifest_wraps_the_unchanged_base_and_receipt_must_match(self):
        sys.path.insert(0, str(Path(owner.__file__).parent))
        import confidence_weights as cw
        with tempfile.TemporaryDirectory(dir=Path.home() / "tmp") as tmp:
            tmp = Path(tmp)
            (tmp / "base.json").write_text("{}")
            (tmp / "start.bin").write_bytes(b"model")
            base = dict(path=str(tmp / "base.json"), sha256=owner.sha(tmp / "base.json"))
            binding = cw.continuation_manifest(base, tmp / "start.bin", 256.0, tmp / "c.json")
            value = json.loads((tmp / "c.json").read_text())
            self.assertEqual(value["schema"], cw.CONTINUATION_SCHEMA)
            self.assertEqual(value["added_group"], dict(name="aic_public_train", train_weight=256.0,
                                                        validation_weight=0.0, mode="rank", binding=None))
            self.assertEqual((value["epochs"], value["pairs_per_epoch"]), (60, 50000))
            self.assertEqual(value["checkpoint"], str((tmp / "start.bin").resolve()))
            with self.assertRaises(FileExistsError):
                cw.continuation_manifest(base, tmp / "start.bin", 256.0, tmp / "c.json")
            receipt = dict(schema="mlic-native-contrast-normalizers-v1", manifest_sha256=binding["sha256"],
                           sampler_changed=False, absolute_loss_changed=False,
                           continuation=dict(base_manifest_sha256=base["sha256"], checkpoint_sha256=owner.sha(tmp / "start.bin"),
                                             epochs=60, added_group="aic_public_train", added_group_weighted=False))
            (tmp / "r.json").write_text(json.dumps(receipt))
            cw.verify_continuation_receipt(tmp / "r.json", binding, base)
            for change in (dict(added_group_weighted=True), dict(epochs=59), dict(base_manifest_sha256="other")):
                (tmp / "r.json").write_text(json.dumps(dict(receipt, continuation=dict(receipt["continuation"], **change))))
                with self.assertRaisesRegex(ValueError, "continuation receipt"):
                    cw.verify_continuation_receipt(tmp / "r.json", binding, base)
            (tmp / "r.json").write_text(json.dumps({k: v for k, v in receipt.items() if k != "continuation"}))
            with self.assertRaisesRegex(ValueError, "continuation receipt"):
                cw.verify_continuation_receipt(tmp / "r.json", binding, base)
            (tmp / "base.json").write_text('{"changed": true}')
            with self.assertRaisesRegex(ValueError, "manifest changed"):
                cw.continuation_manifest(base, tmp / "start.bin", 256.0, tmp / "d.json")


class SeedsStage(unittest.TestCase):
    def test_seeds_plan_accepts_only_frontier_or_recovery_points(self):
        with tempfile.TemporaryDirectory(dir=Path.home() / "tmp") as tmp:
            point = Path(tmp) / "point"
            point.mkdir()
            for schema, arms in ((owner.FOLD_SCHEMA, owner.ARMS), (owner.FRONTIER_SCHEMA, owner.FRONTIER_ARMS)):
                (point / "PLAN.json").write_text(json.dumps(dict(schema=schema, arms=arms)))
                with self.assertRaisesRegex(ValueError, "advanced point"):
                    owner.seeds_plan(Path(tmp) / "run", "HEAD", Path(tmp) / "t", Path(tmp) / "m", point,
                                     "MixedAIC16", Path(tmp))
            path = Path(tmp) / "PLAN.json"
            path.write_text(json.dumps(dict(schema=owner.SEEDS_SCHEMA)))
            self.assertEqual(owner.plan_schema(path), owner.SEEDS_SCHEMA)


class OmissionStage(unittest.TestCase):
    def test_omission_cells_rotate_aic_folds_and_continue_their_cell(self):
        cells = owner.omission_cells()
        self.assertEqual(len(cells), 12)
        for seed in owner.SEEDS:
            self.assertEqual(len({owner.omission_aic_fold(f, seed) for f in owner.FOLDS}), 4)
        with tempfile.TemporaryDirectory(dir=Path.home() / "tmp") as tmp:
            run = Path(tmp)
            (run / "cols.txt").write_text("13,14\n")
            plan = dict(root="/invented/root", columns=str(run / "cols.txt"), arms={"MixedAIC256": "mixed-aic256"},
                        cells={"tid2013_s2": dict(directory="/invented/cells/Mlic/without_tid2013_s2", aic_fold=2,
                                                  exclusions=["tid2013-linked"])},
                        aic_tables={"2": dict(path="/invented/aic_k2_fit.parquet")})
            command = owner.omission_command(plan, run, "MixedAIC256", "tid2013", 2, run / "dest")
            self.assertEqual(command[command.index("--heldout") + 1], "tid2013")
            self.assertEqual(command[command.index("--seed-index") + 1], "2")
            self.assertEqual(command[command.index("--init-from") + 1], "/invented/cells/Mlic/without_tid2013_s2/refit/last.bin")
            self.assertEqual(command[command.index("--research-extra-group") + 1],
                             "aic_public_train:/invented/aic_k2_fit.parquet:256.0:0:rank")
            self.assertEqual(command[command.index("--diagnostic-population") + 1], "mixed-aic256")
            self.assertEqual(command[command.index("--fine-tune-epochs") + 1], "60")
            self.assertEqual(command[-2:], ["--exclude-datasets", "tid2013-linked"])
            self.assertTrue(command[2].endswith("source/scripts/rev4_featpot/v2_lodo_mlp.py"))

    def test_omission_plan_refuses_recipe_points(self):
        with tempfile.TemporaryDirectory(dir=Path.home() / "tmp") as tmp:
            point = Path(tmp) / "point"
            point.mkdir()
            (point / "PLAN.json").write_text(json.dumps(dict(schema=owner.RECIPE_SCHEMA, recipe="MlicOrder",
                                                             arms={"MlicOrderAIC256": "mixed-aic256"})))
            with self.assertRaisesRegex(ValueError, "per-seed factors"):
                owner.omission_plan(Path(tmp) / "run", "HEAD", Path(tmp) / "t", Path(tmp) / "m", point,
                                    "MlicOrderAIC256", Path(tmp))

    def test_recovery_points_continue_aic_only_then_recover_from_its_start(self):
        static = dict(schema=owner.FRONTIER_SCHEMA, arms={"MixedAIC256": "mixed-aic256"})
        self.assertEqual(owner.two_phases(static, "MixedAIC256", "mixed-aic256"), [["MixedAIC256", "mixed-aic256", None]])
        recovery = dict(schema=owner.RECOVERY_SCHEMA, start=dict(arm="AICOnly", epoch=59))
        self.assertEqual(owner.two_phases(recovery, "RecoverAIC64", "mixed-aic64"),
                         [["AICOnly", "aic-only", None], ["RecoverAIC64", "mixed-aic64", 59]])
        with tempfile.TemporaryDirectory(dir=Path.home() / "tmp") as tmp:
            run = Path(tmp)
            (run / "cols.txt").write_text("13,14\n")
            plan = dict(root="/invented/root", columns=str(run / "cols.txt"), arms={"RecoverAIC64": "mixed-aic64"},
                        cells={"kadid_s0": dict(directory="/invented/cells/Mlic/without_kadid_s0", aic_fold=0, exclusions=[])},
                        aic_tables={"0": dict(path="/invented/aic_k0_fit.parquet")})
            first = owner.omission_command(plan, run, "AICOnly", "kadid", 0, run / "a", None, "aic-only")
            self.assertEqual(first[first.index("--diagnostic-population") + 1], "aic-only")
            self.assertEqual(first[first.index("--research-extra-group") + 1], "aic_public_train:/invented/aic_k0_fit.parquet:16.0:0:rank")
            second = owner.omission_command(plan, run, "RecoverAIC64", "kadid", 0, run / "b", run / "a/ckpt/ckpt_epoch059.bin")
            self.assertEqual(second[second.index("--init-from") + 1], str(run / "a/ckpt/ckpt_epoch059.bin"))
            self.assertEqual(second[second.index("--diagnostic-population") + 1], "mixed-aic64")


class RehearsalStage(unittest.TestCase):
    def test_rehearsal_arms_run_their_registered_population_and_weight(self):
        with tempfile.TemporaryDirectory(dir=Path.home() / "tmp") as tmp:
            run = Path(tmp)
            (run / "cols.txt").write_text("13,14\n")
            plan = dict(root="/invented/root", columns=str(run / "cols.txt"), arms=owner.XREF_ARMS, seeds=[0])
            for arm, (population, weight) in zip(owner.XREF_ARMS, (("mixed-aic256-xref", 256.0), ("mixed-aic1024-xref", 1024.0))):
                command = owner.fit_command(plan, run, arm, 0, "/invented/Mlic/full_s0", "/invented/aic_k1_fit.parquet",
                                            run / "arms" / arm / "s0_k1")
                self.assertEqual(command[command.index("--diagnostic-population") + 1], population)
                self.assertEqual(float(command[command.index("--aic-weight") + 1]), weight)
            self.assertEqual(len(owner.fold_runs(plan)), 14)
            path = run / "PLAN.json"
            path.write_text(json.dumps(dict(schema=owner.XREF_SCHEMA)))
            self.assertEqual(owner.plan_schema(path), owner.XREF_SCHEMA)


class InterpolationStage(unittest.TestCase):
    def test_points_cover_every_end_and_weight_once(self):
        arms = {owner.wise_arm(end, alpha) for end in owner.WISE_ENDS for alpha in owner.WISE_ALPHAS}
        self.assertEqual(len(arms), 27)
        self.assertEqual(owner.wise_arm("AICOnly", 0.3), "WiseAICOnlyA30")
        self.assertEqual({epoch for _, _, epoch in owner.WISE_ENDS.values()}, {29, 59})
        plan = dict(arms={arm: "interpolation" for arm in arms}, seeds=[0])
        self.assertEqual(len(owner.fold_runs(plan)), 27 * 7)

    def test_plans_check_their_own_protocol(self):
        with tempfile.TemporaryDirectory(dir=Path.home() / "tmp") as tmp:
            path = Path(tmp) / "PLAN.json"
            path.write_text(json.dumps(dict(schema=owner.WISE_SCHEMA)))
            self.assertEqual(owner.plan_schema(path), owner.WISE_SCHEMA)
            path.write_text(json.dumps(dict(schema=owner.WISE_SCHEMA, protocol=owner.WISE_PROTOCOL,
                                            arms={"WiseAICOnlyA30": "interpolation"}, registered_inputs={},
                                            input_sha256={}, bin_sha256={}, meta_tool_sha256="", source_archive_sha256="")))
            with self.assertRaises(FileNotFoundError):  # passes the protocol check, then hashes its inputs
                owner.checked(path, owner.WISE_SCHEMA)
            path.write_text(json.dumps(dict(schema=owner.WISE_SCHEMA, protocol=owner.FRONTIER_PROTOCOL,
                                            arms={"WiseAICOnlyA30": "interpolation"})))
            with self.assertRaisesRegex(ValueError, "schema differs"):
                owner.checked(path, owner.WISE_SCHEMA)


class CodecEvidence(unittest.TestCase):
    def test_only_the_registered_aic_heavy_arms_take_omission_runs_from_other_studies(self):
        self.assertEqual({arm for _, arm in owner.CODEC_EVIDENCE_POINTS}, {"AICOnly", "MixedAIC1024", "XrefAIC1024"})
        with tempfile.TemporaryDirectory(dir=Path.home() / "tmp") as tmp:
            point = Path(tmp) / "point"
            point.mkdir()
            (point / "PLAN.json").write_text(json.dumps(dict(schema=owner.FOLD_SCHEMA, arms={"AICOnly": "aic-only"})))
            with self.assertRaisesRegex(ValueError, "codec-evidence arm"):
                owner.omission_plan(Path(tmp) / "run", "0" * 40, Path("/invented/t"), Path("/invented/m"), point,
                                    "AICOnly", Path("/invented/after"))


class CodecModel(unittest.TestCase):
    def test_full_data_takes_only_the_picked_codec_point_from_the_fold_study(self):
        self.assertEqual(owner.CODEC_MODEL_POINT, (str(owner.FRONTIER_REUSED_RUN), "AICOnly", 29))
        with tempfile.TemporaryDirectory(dir=Path.home() / "tmp") as tmp:
            point = Path(tmp) / "point"
            point.mkdir()
            (point / "PLAN.json").write_text(json.dumps(dict(schema=owner.FOLD_SCHEMA, arms={"AICOnly": "aic-only"})))
            with self.assertRaisesRegex(ValueError, "or the codec model"):
                owner.full_plan(Path(tmp) / "run", "0" * 40, Path("/invented/t"), Path("/invented/m"), point, "AICOnly",
                                29, Path("/invented/after"))


class RecipeVariant(unittest.TestCase):
    def test_mlic_points_have_no_recipe_and_factor_recipes_refuse_before_any_read(self):
        self.assertEqual(owner.recipe_point(dict(schema=owner.FRONTIER_SCHEMA)), (None, None, None))
        for recipe in ("MlicOrder", "MlicConfidence"):
            with self.assertRaisesRegex(ValueError, "per-seed factors"):
                owner.recipe_point(dict(schema=owner.RECIPE_SCHEMA, recipe=recipe))
        self.assertEqual(set(owner.RECIPE_CELLS), {"MlicVote", "MlicNeff"})


if __name__ == "__main__":
    unittest.main()
