"""独立 session0 在线真值库存的构建、迁移和九被试等价回归。"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "code" / "eval"
TRAIN_DIR = ROOT / "code" / "train"
for source_dir in (EVAL_DIR, TRAIN_DIR):
    sys.path.insert(0, str(source_dir))

from online_truth_inventory import (  # noqa: E402
    TRUTH_ID,
    build_truth_inventory,
    file_hash,
    load_truth_inventory,
)
from run_epoch50_online_oof import (  # noqa: E402
    build_online_inventory,
    default_subject_paths,
    verify_inventory_contract,
)
from oof_training_bundle import load_bundle  # noqa: E402


EXPECTED_COUNTS = {
    1: 273, 2: 270, 3: 270, 4: 262, 5: 262,
    6: 219, 7: 271, 8: 264, 9: 237,
}


class RealTruthInventoryTests(unittest.TestCase):
    """真实数据回归证明事件来源改变，但正式评分库存逐项不变。"""

    def setUp(self) -> None:
        self.processed = ROOT / "data" / "processed"
        self.config = ROOT / "config" / "evaluation"

    def test_all_subjects_match_legacy_inventory_exactly(self) -> None:
        total = 0
        for subject, expected_count in EXPECTED_COUNTS.items():
            paths = default_subject_paths(subject)
            context = load_bundle(paths.bundle_manifest, verify_hashes=True)
            truth = load_truth_inventory(paths.truth_manifest, context)
            explicit = build_online_inventory(context, truth)
            legacy = build_online_inventory(
                context, allow_legacy_event_reconstruction=True,
            )
            self.assertEqual(explicit.events, legacy.events)
            self.assertEqual(explicit.segments, legacy.segments)
            self.assertEqual(explicit.windows, legacy.windows)
            self.assertTrue(np.array_equal(explicit.signal_rows, legacy.signal_rows))
            self.assertEqual(len(explicit.events), expected_count)

            v1 = json.loads(paths.legacy_inventory_contract.read_text(encoding="utf-8"))
            v3 = json.loads(paths.inventory_contract.read_text(encoding="utf-8"))
            self.assertEqual(v3["inventory"], v1["inventory"])
            self.assertEqual(v3["per_run_event_count"], v1["per_run_event_count"])
            self.assertEqual(v3["per_run_window_count"], v1["per_run_window_count"])
            verified = verify_inventory_contract(context, explicit, v3)
            self.assertEqual(verified["event_count"], expected_count)
            total += expected_count
        self.assertEqual(total, 2328)

    def test_repeat_build_is_identical_and_store_can_move(self) -> None:
        subject = 1
        paths = default_subject_paths(subject)
        context = load_bundle(paths.bundle_manifest, verify_hashes=True)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, manifest_path = build_truth_inventory(
                self.processed, paths.bundle_manifest, root, subject,
            )
            first_manifest_hash = file_hash(manifest_path)
            first_event_hash = file_hash(manifest_path.parent / "events.npy")
            _, repeated = build_truth_inventory(
                self.processed, paths.bundle_manifest, root, subject,
            )
            self.assertEqual(file_hash(repeated), first_manifest_hash)
            self.assertEqual(file_hash(repeated.parent / "events.npy"), first_event_hash)

            moved = root / "moved"
            shutil.copytree(manifest_path.parent, moved)
            loaded = load_truth_inventory(moved / "manifest.json", context)
            self.assertEqual(len(loaded.events), EXPECTED_COUNTS[subject])

    def test_runtime_loader_does_not_reopen_joint_index_or_test_session(self) -> None:
        paths = default_subject_paths(1)
        real_builtin_open = open
        real_path_open = Path.open
        opened: list[Path] = []

        # pathlib 和 numpy 的打开路径不同：前者拦 Path.open，后者兼容 builtins.open。
        def record_path(file) -> None:
            try:
                opened.append(Path(file).resolve())
            except TypeError:
                pass

        def tracking_builtin_open(file, *args, **kwargs):
            record_path(file)
            return real_builtin_open(file, *args, **kwargs)

        def tracking_path_open(path, *args, **kwargs):
            record_path(path)
            return real_path_open(path, *args, **kwargs)

        with patch("builtins.open", tracking_builtin_open), patch.object(
            Path, "open", tracking_path_open,
        ):
            context = load_bundle(paths.bundle_manifest, verify_hashes=True)
            truth = load_truth_inventory(paths.truth_manifest, context)
            build_online_inventory(context, truth)
        allowed_roots = (paths.bundle_manifest.parent.resolve(), paths.truth_manifest.parent.resolve())
        outside = [
            path for path in opened
            if not any(path.is_relative_to(root) for root in allowed_roots)
        ]
        self.assertGreater(len(opened), 0, "监控未捕获任何文件，隔离测试无效")
        self.assertIn(paths.bundle_manifest.resolve(), opened)
        self.assertIn(paths.truth_manifest.resolve(), opened)
        self.assertEqual(outside, [])
        self.assertFalse(any("A01E.mat" in path.name for path in opened))

    def test_event_file_tampering_is_rejected(self) -> None:
        subject = 1
        paths = default_subject_paths(subject)
        context = load_bundle(paths.bundle_manifest, verify_hashes=True)
        with tempfile.TemporaryDirectory() as directory:
            copied = Path(directory) / TRUTH_ID.format(subject=subject)
            shutil.copytree(paths.truth_manifest.parent, copied)
            event_path = copied / "events.npy"
            rows = np.load(event_path, allow_pickle=False)
            rows[0]["class_id"] = 4 if int(rows[0]["class_id"]) != 4 else 3
            np.save(event_path, rows, allow_pickle=False)
            with self.assertRaisesRegex(RuntimeError, "哈希"):
                load_truth_inventory(copied / "manifest.json", context)


if __name__ == "__main__":
    unittest.main()
