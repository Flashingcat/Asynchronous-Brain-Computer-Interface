"""Subject 1 真实数据的 session0-only OOF bundle 隔离测试。"""

from __future__ import annotations

import builtins
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PREPROCESSING = ROOT / "code" / "preprocessing"
TRAIN = ROOT / "code" / "train"
EVAL = ROOT / "code" / "eval"
sys.path.insert(0, str(PREPROCESSING))
sys.path.insert(0, str(TRAIN))
sys.path.insert(0, str(EVAL))

from build_causal_filter_store import build_causal_filter_store  # noqa: E402
from build_fold_normalization import (  # noqa: E402
    build_normalization_manifest,
    save_normalization_manifest,
)
from build_offline_view import build_offline_view, save_offline_view  # noqa: E402
from build_oof_training_bundle import build_bundle  # noqa: E402
from build_protocol_index import build_subject, save_subject  # noqa: E402
from build_signal_store import build_signal_store  # noqa: E402
from build_validation_folds import build_fold_manifest, save_fold_manifest  # noqa: E402
from build_zero_phase_filter_store import build_zero_phase_filter_store  # noqa: E402
from oof_training_bundle import (  # noqa: E402
    BUNDLE_ID,
    LEGACY_BUNDLE_ID,
    artifact_contract,
    file_hash,
    load_bundle,
)
from train_eegnet_oof import JobSpec, prepare_job_arrays  # noqa: E402
from freeze_online_inventory_contracts import freeze_or_verify  # noqa: E402


class RealOOFTrainingBundleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        value = os.environ.get("BNCI2014001_ROOT")
        if not value:
            raise RuntimeError("必须设置 BNCI2014001_ROOT，真实数据测试不得跳过")
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.index_dir = cls.root / "indices"
        cls.signal_dir = cls.root / "signals"
        cls.bundle_dir = cls.root / "isolated_bundle"

        # 从 MAT 开始重建全部上游，避免隔离测试依赖工作区已有 bundle。
        base = build_subject(Path(value), 1)
        base_manifest = save_subject(cls.index_dir, 1, base)
        offline = build_offline_view(base[0], base[1], base[2], base_manifest)
        base_files = (
            cls.index_dir / f"{base_manifest['protocol_id']}.npz",
            cls.index_dir / f"{base_manifest['protocol_id']}_manifest.json",
        )
        save_offline_view(cls.index_dir, 1, offline, base_files)
        save_fold_manifest(cls.index_dir, build_fold_manifest(cls.index_dir, 1))
        build_signal_store(Path(value), cls.index_dir, cls.signal_dir, 1)
        build_causal_filter_store(cls.signal_dir, cls.signal_dir, 1)
        build_zero_phase_filter_store(cls.signal_dir, cls.signal_dir, 1)
        normalization = build_normalization_manifest(cls.index_dir, cls.signal_dir, 1)
        normalization_path = save_normalization_manifest(cls.index_dir, normalization)
        cls.manifest, cls.manifest_path = build_bundle(
            cls.index_dir, cls.signal_dir, normalization_path, cls.bundle_dir, 1
        )
        cls.bundle_root = cls.manifest_path.parent
        cls.context = load_bundle(cls.manifest_path)

    @classmethod
    def tearDownClass(cls) -> None:
        for store in cls.context.stores.values():
            store._cache.clear()
        cls.temporary.cleanup()

    def test_golden_counts_folds_and_only_session0_segments(self) -> None:
        manifest = self.manifest
        self.assertFalse(manifest["test_session_content_in_bundle"])
        self.assertEqual(
            artifact_contract(manifest),
            {
                "artifact_policy": "official_trial_exclusion",
                "segment_policy": "separate_clean_segments_no_time_compression",
                "artifact_policy_binding": "explicit_bundle_manifest",
            },
        )
        self.assertEqual(manifest["shared_pool"]["stage1_window_count"], 2454)
        self.assertEqual(manifest["shared_pool"]["stage2_window_count"], 1365)
        self.assertEqual(
            [item["train_stage1"]["window_count"] for item in manifest["folds"]],
            [2040, 2064, 2049, 2028, 2039, 2050],
        )
        self.assertEqual(
            [item["train_stage2"]["window_count"] for item in manifest["folds"]],
            [1135, 1145, 1140, 1130, 1135, 1140],
        )
        self.assertTrue(np.all(self.context.rows["session"] == 0))
        for domain in ("causal", "zero_phase"):
            records = manifest["domains"][domain]["segments"]
            self.assertEqual(len(records), 21)
            self.assertTrue(all(item["session"] == 0 for item in records))
            self.assertTrue(all("session1" not in item["file"] for item in records))
            self.assertTrue(all(key[0] == 0 for key in self.context.stores[domain]._records))
        self.assertFalse(any("session1" in path.name
                             for path in self.bundle_root.rglob("*")))

    def test_artifact_contract_rejects_partial_or_wrong_identity(self) -> None:
        partial = dict(self.manifest)
        partial.pop("segment_policy")
        with self.assertRaisesRegex(RuntimeError, "伪迹或连续 segment"):
            artifact_contract(partial)

        wrong = dict(self.manifest)
        wrong["artifact_policy"] = "unknown"
        with self.assertRaisesRegex(RuntimeError, "伪迹或连续 segment"):
            artifact_contract(wrong)

        legacy = dict(self.manifest)
        legacy.pop("artifact_policy")
        legacy.pop("segment_policy")
        legacy["protocol_id"] = LEGACY_BUNDLE_ID.format(subject=1)
        self.assertEqual(
            artifact_contract(legacy)["artifact_policy_binding"],
            "legacy_v1_protocol_contract",
        )

        # 显式字段只能属于 v2，缺字段只能属于精确 v1；禁止一个版本承载两种语义。
        explicit_v1 = dict(self.manifest)
        explicit_v1["protocol_id"] = LEGACY_BUNDLE_ID.format(subject=1)
        with self.assertRaisesRegex(RuntimeError, "伪迹或连续 segment"):
            artifact_contract(explicit_v1)
        missing_v2 = dict(legacy)
        missing_v2["protocol_id"] = BUNDLE_ID.format(subject=1)
        with self.assertRaisesRegex(RuntimeError, "伪迹或连续 segment"):
            artifact_contract(missing_v2)

    def test_bundle_loader_opens_no_file_outside_isolated_root(self) -> None:
        real_open = builtins.open
        opened: list[Path] = []

        def tracking_open(file, *args, **kwargs):
            try:
                opened.append(Path(file).resolve())
            except TypeError:
                pass
            return real_open(file, *args, **kwargs)

        with patch("builtins.open", tracking_open):
            loaded = load_bundle(self.manifest_path, verify_hashes=True)
            loaded.stores["causal"].read_window(loaded.rows[0])
            loaded.stores["zero_phase"].read_window(loaded.rows[0])
        outside = [path for path in opened
                   if not path.is_relative_to(self.bundle_root.resolve())]
        self.assertEqual(outside, [])

    def test_bundle_can_move_without_joint_upstream(self) -> None:
        moved = self.root / "moved_to_another_computer"
        shutil.copytree(self.bundle_root, moved)
        loaded = load_bundle(moved / "manifest.json", verify_hashes=True)
        for domain in ("causal", "zero_phase"):
            signal = loaded.stores[domain].read_window(loaded.rows[0])
            self.assertEqual(signal.shape, (22, 500))
        self.assertTrue(np.array_equal(loaded.rows, self.context.rows))

    def test_explicit_v2_bundle_freezes_independent_v2_inventory_contract(self) -> None:
        """全新 bundle 不得复用仓库内锁定旧 manifest SHA 的 v1 库存文件。"""
        output_dir = self.root / "v2_inventory_contracts"
        created = freeze_or_verify(
            1, True, bundle_root=self.bundle_dir, output_dir=output_dir,
        )
        contract_path = Path(created["contract"])
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        self.assertEqual(created["action"], "created_missing")
        self.assertEqual(
            contract_path.name,
            "bnci2014001_s01_session0_causal_online_v2.json",
        )
        self.assertEqual(contract["artifact_policy"], "official_trial_exclusion")
        self.assertEqual(
            contract["artifact_policy_binding"], "explicit_bundle_manifest",
        )
        verified = freeze_or_verify(
            1, False, bundle_root=self.bundle_dir, output_dir=output_dir,
        )
        self.assertEqual(verified["action"], "verified_existing")

    def test_copied_signal_hashes_and_fold_statistics_equal_sources(self) -> None:
        normalization_path = self.index_dir / (
            "bnci2014001_s01_shared_stage1_window_zscore_native250_v1.json"
        )
        normalization = json.loads(normalization_path.read_text(encoding="utf-8"))
        for fold in range(6):
            self.assertEqual(self.manifest["folds"][fold]["statistics"],
                             normalization["folds"][fold]["statistics"])
        for domain in ("causal", "zero_phase"):
            for record in self.manifest["domains"][domain]["segments"]:
                self.assertEqual(file_hash(self.bundle_root / record["file"]),
                                 record["sha256"])

    def test_trainer_arrays_preserve_run_isolation_for_all_branches(self) -> None:
        for stage in (1, 2):
            for domain in ("causal", "zero_phase"):
                spec = JobSpec(1, 0, stage, domain, 42)
                arrays = prepare_job_arrays(self.context, spec)
                self.assertEqual(arrays.train_x.shape[1:], (22, 500))
                self.assertEqual(arrays.validation_x[domain].shape[1:], (22, 500))
                self.assertTrue(np.all(arrays.validation_rows["session"] == 0))
                self.assertEqual(set(arrays.validation_rows["run"].tolist()), {0})
                expected_train = self.manifest["folds"][0][f"train_stage{stage}"]["window_count"]
                expected_validation = self.manifest["folds"][0][f"validation_stage{stage}"]["window_count"]
                self.assertEqual(len(arrays.train_y), expected_train)
                self.assertEqual(len(arrays.validation_y), expected_validation)
        zero = prepare_job_arrays(self.context, JobSpec(1, 0, 2, "zero_phase", 42))
        self.assertEqual(set(zero.validation_x), {"zero_phase", "causal"})

    def test_tampered_manifest_index_and_segment_are_rejected(self) -> None:
        tampered_manifest = self.root / "tampered_manifest.json"
        value = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        value["included_session"] = 1
        tampered_manifest.write_text(json.dumps(value), encoding="utf-8")
        with self.assertRaises(RuntimeError):
            load_bundle(tampered_manifest)

        index_path = self.bundle_root / self.manifest["index_file"]
        original_index = index_path.read_bytes()
        try:
            index_path.write_bytes(original_index + b"tamper")
            with self.assertRaises(RuntimeError):
                load_bundle(self.manifest_path)
        finally:
            index_path.write_bytes(original_index)

        record = self.manifest["domains"]["causal"]["segments"][0]
        signal_path = self.bundle_root / record["file"]
        original_signal = signal_path.read_bytes()
        try:
            signal_path.write_bytes(original_signal[:-1] + bytes([original_signal[-1] ^ 1]))
            with self.assertRaises(RuntimeError):
                load_bundle(self.manifest_path)
        finally:
            signal_path.write_bytes(original_signal)

    def test_repeat_build_is_byte_identical(self) -> None:
        before = {
            path.relative_to(self.bundle_root): file_hash(path)
            for path in self.bundle_root.rglob("*") if path.is_file()
        }
        normalization_path = self.index_dir / (
            "bnci2014001_s01_shared_stage1_window_zscore_native250_v1.json"
        )
        repeated, repeated_path = build_bundle(
            self.index_dir, self.signal_dir, normalization_path, self.bundle_dir, 1
        )
        after = {
            path.relative_to(self.bundle_root): file_hash(path)
            for path in self.bundle_root.rglob("*") if path.is_file()
        }
        self.assertEqual(repeated, self.manifest)
        self.assertEqual(repeated_path, self.manifest_path)
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main(verbosity=2)
