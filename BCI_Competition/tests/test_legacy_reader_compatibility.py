"""历史 epoch50 checkpoint 与当前 v1/v2 bundle reader 的窄兼容证明。"""

from __future__ import annotations

import ast
import copy
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = ROOT / "code" / "train"
EVAL_DIR = ROOT / "code" / "eval"
for source_dir in (TRAIN_DIR, EVAL_DIR):
    sys.path.insert(0, str(source_dir))

import oof_training_bundle as current_reader  # noqa: E402
import run_epoch50_online_oof as runner  # noqa: E402


CHECKPOINT_ROOT = ROOT / "results" / "checkpoints" / "eegnet_oof_native250_v1"
SNAPSHOT_ROOT = CHECKPOINT_ROOT / "posthoc_source_snapshot"
HISTORICAL_READER = (
    SNAPSHOT_ROOT / "source" / "code" / "train" / "oof_training_bundle.py"
)
CURRENT_READER = TRAIN_DIR / "oof_training_bundle.py"


def load_historical_reader():
    """以独立模块名加载训练快照，避免覆盖生产 reader。"""
    name = "historical_oof_training_bundle_snapshot"
    specification = importlib.util.spec_from_file_location(name, HISTORICAL_READER)
    if specification is None or specification.loader is None:
        raise RuntimeError("无法加载历史 bundle reader 快照")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def definition_asts(path: Path) -> dict[str, str]:
    """比较 model-facing 定义的语法树，忽略换行和注释差异。"""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.name: ast.dump(node, include_attributes=False)
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    }


class LegacyReaderCompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not HISTORICAL_READER.is_file():
            raise RuntimeError("旧 checkpoint 兼容测试缺少 posthoc reader 快照")
        cls.historical_reader = load_historical_reader()

    def test_source_diff_is_exactly_the_audited_contract_extension(self) -> None:
        historical = definition_asts(HISTORICAL_READER)
        current = definition_asts(CURRENT_READER)
        unchanged = {
            "BundleContext", "BundleSignalStore", "BundleWindowDataset",
            "_safe_relative_file", "file_hash", "fold_entry", "is_sha256",
            "load_bundle", "rows_for", "window_identity_hash",
        }
        self.assertEqual(set(historical) - set(current), set())
        self.assertEqual(set(current) - set(historical), {"artifact_contract"})
        changed = {
            name for name in set(historical) & set(current)
            if historical[name] != current[name]
        }
        self.assertEqual(changed, {"validate_bundle_manifest"})
        self.assertTrue(all(historical[name] == current[name] for name in unchanged))
        self.assertEqual(
            runner.audited_reader_diff_hash(HISTORICAL_READER, CURRENT_READER),
            runner.AUDITED_READER_DIFF_SHA256,
        )

    def test_all_subject_online_materialization_matches_historical_reader(self) -> None:
        window_total = 0
        old = self.historical_reader
        for subject in range(1, 10):
            manifest_path = runner.default_subject_paths(subject).bundle_manifest
            current = current_reader.load_bundle(manifest_path, verify_hashes=True)
            historical = old.load_bundle(manifest_path, verify_hashes=True)
            self.assertEqual(current.manifest, historical.manifest)
            self.assertEqual(current.manifest_sha256, historical.manifest_sha256)
            self.assertEqual(current.rows.dtype, historical.rows.dtype)
            self.assertTrue(np.array_equal(current.rows, historical.rows))

            # 每折的训练/验证行及其身份哈希必须完全一致。
            for fold in range(6):
                for stage in (1, 2):
                    for split in ("train", "validation"):
                        new_rows = current_reader.rows_for(
                            current.manifest, current.rows, fold, stage, split,
                        )
                        old_rows = old.rows_for(
                            historical.manifest, historical.rows, fold, stage, split,
                        )
                        self.assertTrue(np.array_equal(new_rows, old_rows))
                        self.assertEqual(
                            current_reader.window_identity_hash(new_rows),
                            old.window_identity_hash(old_rows),
                        )
            for domain in current.stores:
                self.assertEqual(
                    current.stores[domain]._records,
                    historical.stores[domain]._records,
                )

            # 不只比 manifest：两个 reader 逐 run 物化全部严格在线窗口。
            inventory = runner._build_online_signal_inventory(current)
            for run in range(6):
                mask = np.fromiter(
                    (window.run_id == run for window in inventory.windows), dtype=bool,
                )
                rows = inventory.signal_rows[mask]
                new_features = runner.materialize_rows(current, rows, run)
                old_features = runner.materialize_rows(historical, rows, run)
                self.assertTrue(np.array_equal(new_features, old_features))
                window_total += len(rows)
            for context in (current, historical):
                for store in context.stores.values():
                    store._cache.clear()
        self.assertEqual(window_total, 37020)

    def test_compatibility_gate_accepts_only_exact_v1_bridge(self) -> None:
        config = json.loads((
            CHECKPOINT_ROOT / "stage1_causal_fold0_seed42" / "job_config.json"
        ).read_text(encoding="utf-8"))
        contract = config["contract"]
        self.assertEqual(
            runner.verify_model_sources(CHECKPOINT_ROOT, contract),
            runner.AUDITED_READER_MATCH_MODE,
        )
        snapshot = json.loads(
            (SNAPSHOT_ROOT / "manifest.json").read_text(encoding="utf-8"),
        )
        record = next(
            item for item in snapshot["source_files"]
            if item["role"] == "oof_training_bundle_reader"
        )
        self.assertTrue(runner.is_audited_legacy_reader_extension(
            contract, record, HISTORICAL_READER, CURRENT_READER,
        ))

        # v2 bundle、错旧哈希、改当前源码或改 diff 锁任一情形都必须拒绝。
        v2 = copy.deepcopy(contract)
        subject = v2["job"]["subject"]
        v2["data"]["training_bundle_protocol_id"] = (
            current_reader.BUNDLE_ID.format(subject=subject)
        )
        wrong_record = dict(record, job_sha256="0" * 64)
        self.assertFalse(runner.is_audited_legacy_reader_extension(
            v2, record, HISTORICAL_READER, CURRENT_READER,
        ))
        self.assertFalse(runner.is_audited_legacy_reader_extension(
            contract, wrong_record, HISTORICAL_READER, CURRENT_READER,
        ))
        with tempfile.TemporaryDirectory() as directory:
            changed = Path(directory) / "oof_training_bundle.py"
            changed.write_bytes(CURRENT_READER.read_bytes() + b"\n# changed\n")
            self.assertFalse(runner.is_audited_legacy_reader_extension(
                contract, record, HISTORICAL_READER, changed,
            ))
        with patch.object(runner, "AUDITED_READER_DIFF_SHA256", "0" * 64):
            self.assertFalse(runner.is_audited_legacy_reader_extension(
                contract, record, HISTORICAL_READER, CURRENT_READER,
            ))
        with self.assertRaisesRegex(RuntimeError, "oof_training_bundle_reader"):
            runner.verify_model_sources(CHECKPOINT_ROOT, v2)


if __name__ == "__main__":
    unittest.main()
