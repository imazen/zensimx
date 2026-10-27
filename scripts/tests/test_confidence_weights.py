"""Original row bindings fail closed before native loss integration."""

import hashlib
import json
import tempfile
from pathlib import Path
import struct
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "byconstruction"))
from confidence_weights import bind, original_cell, logical_width, cell_binding, verify_receipt, sha


class Bindings(unittest.TestCase):
    def test_cell_binding_rejects_wrong_arm_omission_hash_and_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cell = root / "cell.json"
            cell.write_text(json.dumps(dict(candidate="MlicConfidence", fold="kadid", seed_index=0, n_features=2263)))
            binding = dict(path=str(cell), sha256=sha(cell))
            index = root / "index.json"
            cells = {f"{f}-s{s}":binding for f in
                     ("kadid", "tid2013", "konfig", "cid22_a25", "pipal", "pairs", "mos7")
                     for s in range(3)}
            index.write_text(json.dumps(dict(schema="mlic-contrast-manifests-v1", cells=cells)))
            pin = sha(index)
            self.assertEqual(cell_binding(index,pin,"MlicConfidence","kadid",0),binding)
            for arm, fold, seed in [("MlicOrder","kadid",0),("MlicConfidence","tid2013",0),
                                    ("MlicConfidence","kadid",1)]:
                with self.assertRaises(ValueError):
                    cell_binding(index,pin,arm,fold,seed)
            with self.assertRaises(ValueError):
                cell_binding(index,"0"*64,"MlicConfidence","kadid",0)
            receipt = root / "receipt.json"
            value = dict(schema="mlic-native-contrast-normalizers-v1",manifest_sha256=binding["sha256"],
                         sampler_changed=False,absolute_loss_changed=False)
            receipt.write_text(json.dumps(value))
            self.assertEqual(verify_receipt(receipt,binding),sha(receipt))
            receipt.write_text(json.dumps(value | dict(sampler_changed=True)))
            with self.assertRaises(ValueError):
                verify_receipt(receipt,binding)
            cell.write_text(cell.read_text()+" ")
            with self.assertRaises(ValueError):
                cell_binding(index,pin,"MlicConfidence","kadid",0)

    def test_wrapper_sets_child_environment_without_changing_parent(self):
        import os
        from unittest.mock import patch
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rev4_featpot"))
        import byc_cells
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cols = root / "columns.txt"
            cols.write_text("1,2")
            binding = dict(path=str(root / "cell.json"), sha256="a"*64)
            def capture(command, **kwargs):
                env = kwargs["env"]
                self.assertEqual(env["ZENSIM_RESEARCH_CONFIDENCE_MANIFEST"],binding["path"])
                self.assertEqual(env["ZENSIM_RESEARCH_CONFIDENCE_SHA256"],binding["sha256"])
                self.assertEqual(Path(env["ZENSIM_RESEARCH_CONFIDENCE_RECEIPT"]),
                                 root / "cells/MlicConfidence/without_kadid_s0/CONFIDENCE_NORMALIZERS.json")
                self.assertNotIn("ZENSIM_RESEARCH_CONFIDENCE_MANIFEST", os.environ)
                self.assertNotIn("--matched-codec-manifest",command)
                raise RuntimeError("captured native launch")
            with patch.object(byc_cells,"ROOT",root), patch.object(byc_cells,"CELLS",root/"cells"), \
                 patch.object(byc_cells,"COLS",cols), patch.object(byc_cells,"LABEL","MlicConfidence"), \
                 patch.dict(os.environ,{"BYC_CONFIDENCE_MANIFESTS":"index", "BYC_CONFIDENCE_MANIFESTS_SHA256":"pin", "BYC_SCRATCH":str(root/"scratch")}), \
                 patch("confidence_weights.cell_binding",return_value=binding), \
                 patch("subjective_training_leg.linked_subset_exclusions",return_value=[]), \
                 patch.object(byc_cells.subprocess,"run",capture):
                with self.assertRaisesRegex(RuntimeError,"captured native launch"):
                    byc_cells.run_cell("mono:thr:mp32","kadid",0)

    def test_priority_dependency_requires_live_worker_or_bound_completion(self):
        import os
        from dataset_queue import panel_dependency, process_snapshot
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root/"panels").mkdir()
            plan = root/"PLAN.json"; plan.write_text("{}")
            state = root/"FULL_PANEL_QUEUE.json"
            worker = process_snapshot(os.getpid())
            state.write_text(json.dumps(dict(worker=worker,failures=[])))
            self.assertFalse(panel_dependency(plan,sha(plan)))
            state.write_text(json.dumps(dict(worker=worker | dict(start_ticks=0),failures=[])))
            with self.assertRaises(RuntimeError):
                panel_dependency(plan,sha(plan))
            inventory = root/"panels/EVIDENCE_INVENTORY.json";inventory.write_text("{}")
            complete = root/"panels/FULL_PANEL_COMPLETE.json"
            complete.write_text(json.dumps(dict(plan_sha256=sha(plan),selector=None,evidence_inventory_sha256=sha(inventory))))
            self.assertTrue(panel_dependency(plan,sha(plan)))
            inventory.write_text("{ } ")
            with self.assertRaises(ValueError):
                panel_dependency(plan,sha(plan))

    def test_logical_width_preserves_raw_space_and_derived_products(self):
        cell = {"width":1853,"feature_ids":[13,14]}
        derived = {"direct":[13,14],"products":[[13,422],[14,422]]}
        self.assertEqual(logical_width(cell,derived),1855)
        with self.assertRaises(ValueError):
            logical_width(cell,derived | {"direct":[14,13]})

    def test_dataset_omissions_preserve_original_full_authority_namespace(self):
        for fold in ["pipal", "pairs", "mos7", "kadid", "tid2013", "konfig", "cid22_a25"]:
            route = "full" if fold in {"pipal", "pairs", "mos7"} else fold
            authority = {"cells": {f"{route}-s1": {"seed_index": 1}}}
            self.assertEqual(original_cell(authority, fold, 1)["seed_index"], 1)
            with self.assertRaises(ValueError):
                original_cell(authority, fold, 0)

    def rows(self):
        return [
            dict(
                table_row=i,
                collection="fixture",
                reference_group="r",
                native=dict(item=str(i), reference="native-r"),
            )
            for i in range(2)
        ]

    def test_dense_first_seen_refs_and_uncovered_reference_are_preserved(self):
        rows = self.rows()
        result = bind(
            [1.0, 2.0, 3.0, 4.0],
            ["r", "r", "other", "other"],
            rows,
            {("fixture", str(i), "native-r"): [float(i), 0.1] for i in range(2)},
        )
        digest = hashlib.sha256(
            b"".join(
                struct.pack("<dI", s, r)
                for s, r in [(1.0, 0), (2.0, 0), (3.0, 1), (4.0, 1)]
            )
        ).hexdigest()
        self.assertEqual(result["scores_refs_sha256"], digest)
        self.assertEqual(result["covered_rows"], 2)
        self.assertEqual(result["points"][2:], [None, None])
        self.assertEqual(result["rows"], 4)

    def test_partial_missing_duplicated_and_wrong_order_rows_refuse(self):
        rows = self.rows()
        maps = {("fixture", str(i), "native-r"): [float(i), 0.1] for i in range(2)}
        for selected in [rows[:1], rows + [rows[0]]]:
            with self.assertRaises(ValueError):
                bind([1.0, 2.0], ["r", "r"], selected, maps)
        with self.assertRaises(ValueError):
            bind([1.0, 2.0], ["r", "wrong"], rows, maps)
        with self.assertRaises(ValueError):
            bind([1.0, 2.0], ["r", "r"], rows, {})
        with self.assertRaises(ValueError):
            bind([1.0, float("nan")], ["r", "r"], rows, maps)
        with self.assertRaises(ValueError):
            bind(
                [1.0, 2.0],
                ["r", "r"],
                rows,
                maps | {("fixture", "0", "native-r"): [0.0, -1.0]},
            )


if __name__ == "__main__":
    unittest.main()
