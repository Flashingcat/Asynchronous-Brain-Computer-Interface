"""共享训练窗口、fold专属标准化和统一数据层的Subject 1真实数据测试。"""

from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


PREPROCESSING_DIR = Path(__file__).resolve().parents[1] / "code" / "preprocessing"
SOURCE_FILE = PREPROCESSING_DIR / "build_fold_normalization.py"
sys.path.insert(0, str(PREPROCESSING_DIR))

from build_causal_filter_store import build_causal_filter_store  # noqa: E402
from build_fold_normalization import (  # noqa: E402
    NORMALIZATION_ID,
    NormalizedWindowDataset,
    build_normalization_manifest,
    evaluation_dataset_for,
    formal_mask,
    save_normalization_manifest,
    select_runs,
    shared_training_pool,
    statistics_for,
    training_dataset_for,
    training_rows_for,
    validate_normalization_manifest,
    window_identity_hash,
)
from build_offline_view import build_offline_view, save_offline_view  # noqa: E402
from build_protocol_index import build_subject, save_subject  # noqa: E402
from build_signal_store import build_signal_store  # noqa: E402
from build_validation_folds import build_fold_manifest, save_fold_manifest  # noqa: E402
from build_zero_phase_filter_store import build_zero_phase_filter_store  # noqa: E402


def nested_strings(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from nested_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from nested_strings(item)
    elif isinstance(value, str):
        yield value


class RealFoldNormalizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        value = os.environ.get("BNCI2014001_ROOT")
        if not value:
            raise RuntimeError("必须设置 BNCI2014001_ROOT，真实数据测试不得跳过")
        cls.data_root = Path(value)
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.index_dir = cls.root / "indices"
        cls.signal_dir = cls.root / "signals"
        cls.output_dir = cls.root / "normalization"

        # 从MAT开始构建全部上游，避免测试偷偷依赖工作区旧产物。
        cls.base = build_subject(cls.data_root, 1)
        cls.base_manifest = save_subject(cls.index_dir, 1, cls.base)
        cls.offline = build_offline_view(
            cls.base[0], cls.base[1], cls.base[2], cls.base_manifest
        )
        base_id = cls.base_manifest["protocol_id"]
        base_files = (cls.index_dir / f"{base_id}.npz",
                      cls.index_dir / f"{base_id}_manifest.json")
        save_offline_view(cls.index_dir, 1, cls.offline, base_files)
        save_fold_manifest(cls.index_dir, build_fold_manifest(cls.index_dir, 1))
        build_signal_store(cls.data_root, cls.index_dir, cls.signal_dir, 1)
        _, cls.causal_path = build_causal_filter_store(
            cls.signal_dir, cls.signal_dir, 1
        )
        _, cls.zero_path = build_zero_phase_filter_store(
            cls.signal_dir, cls.signal_dir, 1
        )
        cls.manifest = build_normalization_manifest(
            cls.index_dir, cls.signal_dir, 1
        )
        cls.manifest_path = save_normalization_manifest(
            cls.output_dir, cls.manifest
        )

        # 通过生产加载路径取得窗口和store，后续测试再独立重算统计量。
        from build_fold_normalization import load_sources  # noqa: PLC0415
        (cls.stage1, cls.stage2, cls.folds, _, _, _,
         cls.causal_store, cls.zero_store) = load_sources(
            cls.index_dir, cls.signal_dir, 1
        )
        cls.pool = shared_training_pool(
            cls.stage1, cls.causal_store, cls.zero_store
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.causal_store._signal_store._cache.clear()
        cls.zero_store._signal_store._cache.clear()
        cls.temporary.cleanup()

    def test_manifest_counts_fold_runs_and_portable_sources(self) -> None:
        manifest = self.manifest
        self.assertEqual(manifest["protocol_id"], NORMALIZATION_ID.format(subject=1))
        self.assertEqual(manifest["artifact_policy"], "official_trial_exclusion")
        self.assertEqual(
            manifest["segment_policy"],
            "separate_clean_segments_no_time_compression",
        )
        self.assertEqual(manifest["shared_pool"]["stage1_window_count"], 2454)
        self.assertEqual(manifest["shared_pool"]["stage2_window_count"], 1365)
        self.assertEqual(
            [item["stage1_window_count"] for item in manifest["folds"]],
            [2040, 2064, 2049, 2028, 2039, 2050],
        )
        self.assertEqual(
            [item["stage2_window_count"] for item in manifest["folds"]],
            [1135, 1145, 1140, 1130, 1135, 1140],
        )
        for item in manifest["folds"]:
            self.assertNotIn(item["fold"], item["train_runs"])
            self.assertEqual(len(item["train_runs"]), 5)
        for value in nested_strings(manifest["sources"]):
            self.assertFalse(Path(value).is_absolute(), value)

    def test_every_fold_uses_exact_shared_rows_and_excludes_validation_run(self) -> None:
        for item in self.manifest["folds"]:
            rows = training_rows_for(self.manifest, self.pool, item["fold"], stage=1)
            task_rows = training_rows_for(self.manifest, self.pool, item["fold"], stage=2)
            self.assertTrue(np.all(np.isin(rows["run"], item["train_runs"])))
            self.assertFalse(np.any(rows["run"] == item["fold"]))
            self.assertEqual(window_identity_hash(rows), item["stage1_window_sha256"])
            self.assertTrue(np.array_equal(task_rows, rows[rows["is_task"]]))
            self.assertEqual(window_identity_hash(task_rows), item["stage2_window_sha256"])
        self.assertEqual(window_identity_hash(self.pool),
                         self.manifest["shared_pool"]["stage1_window_sha256"])
        self.assertTrue(np.array_equal(
            training_rows_for(self.manifest, self.pool, None, stage=1), self.pool
        ))

    def independent_statistics(self, store, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """用分块Welford公式独立复算，避免重复生产代码的sum/square算法。"""
        count = 0
        mean = np.zeros(22, dtype=np.float64)
        m2 = np.zeros(22, dtype=np.float64)
        for row in rows:
            values = store.read_window(row).astype(np.float64)
            batch_count = values.shape[1]
            batch_mean = values.mean(axis=1)
            batch_m2 = np.square(values - batch_mean[:, None]).sum(axis=1)
            delta = batch_mean - mean
            new_count = count + batch_count
            mean += delta * batch_count / new_count
            m2 += batch_m2 + np.square(delta) * count * batch_count / new_count
            count = new_count
        return mean, np.sqrt(m2 / count)

    def test_statistics_match_independent_recalculation_without_clip(self) -> None:
        entry = self.manifest["folds"][0]
        rows = select_runs(self.pool, entry["train_runs"])
        for domain, store in (("causal", self.causal_store),
                              ("zero_phase", self.zero_store)):
            expected_mean, expected_std = self.independent_statistics(store, rows)
            actual = statistics_for(self.manifest, domain, fold=0)
            np.testing.assert_allclose(actual["mean_volts"], expected_mean,
                                       rtol=1e-10, atol=1e-15)
            np.testing.assert_allclose(actual["std_volts"], expected_std,
                                       rtol=1e-10, atol=1e-15)
            self.assertGreater(min(actual["std_volts"]), 4e-6)

    def standardized_moments(self, dataset: NormalizedWindowDataset) -> tuple[np.ndarray, np.ndarray]:
        total = np.zeros(22, dtype=np.float64)
        square = np.zeros(22, dtype=np.float64)
        count = 0
        for index in range(len(dataset)):
            values, _ = dataset[index]
            total += values.sum(axis=1, dtype=np.float64)
            square += np.square(values, dtype=np.float64).sum(axis=1)
            count += values.shape[1]
        mean = total / count
        return mean, np.sqrt(np.maximum(square / count - np.square(mean), 0.0))

    def test_unified_dataset_applies_domain_stats_and_preserves_metadata(self) -> None:
        entry = self.manifest["folds"][0]
        rows = select_runs(self.pool, entry["train_runs"])
        for domain, store in (("causal", self.causal_store),
                              ("zero_phase", self.zero_store)):
            stats = statistics_for(self.manifest, domain, fold=0)
            dataset = training_dataset_for(
                self.manifest, self.pool, store, domain, fold=0, stage=1
            )
            signal, label = dataset[0]
            raw = store.read_window(rows[0])
            expected = ((raw - np.asarray(stats["mean_volts"], np.float32)[:, None]) /
                        np.asarray(stats["std_volts"], np.float32)[:, None])
            self.assertTrue(np.array_equal(signal, expected))
            self.assertEqual(label, int(rows[0]["stage1_label"]))
            self.assertTrue(np.array_equal(dataset.metadata(0), rows[0]))
            mean, std = self.standardized_moments(dataset)
            self.assertLess(float(np.abs(mean).max()), 2e-6)
            self.assertLess(float(np.abs(std - 1.0).max()), 2e-6)

    def test_validation_and_test_apply_causal_stats_without_refitting(self) -> None:
        stats = statistics_for(self.manifest, "causal", fold=0)
        causal_ok = formal_mask(self.causal_store, self.stage1)
        validation = self.stage1[(self.stage1["session"] == 0) &
                                 (self.stage1["run"] == 0) & causal_ok]
        final_test = self.stage1[(self.stage1["session"] == 1) & causal_ok]
        for rows in (validation, final_test):
            dataset = evaluation_dataset_for(
                self.manifest, rows, self.causal_store, "causal", fold=0,
                label_field="stage1_label",
            )
            self.assertGreater(len(dataset), 0)
            self.assertEqual(dataset[0][0].shape, (22, stats["samples_per_window"]))
        self.assertEqual(stats["window_count"], 2040)

    def test_invalid_parameters_or_nonformal_rows_are_rejected(self) -> None:
        stats = copy.deepcopy(statistics_for(self.manifest, "causal", fold=0))
        stats["std_volts"][0] = 0.0
        with self.assertRaises(ValueError):
            NormalizedWindowDataset(
                self.pool[:1], self.causal_store, stats, expected_fold=0
            )
        with self.assertRaises(ValueError):
            NormalizedWindowDataset(
                self.pool[:1], self.causal_store,
                statistics_for(self.manifest, "zero_phase", fold=0),
                expected_fold=0,
            )
        wrong_domain = copy.deepcopy(statistics_for(self.manifest, "causal", fold=0))
        wrong_domain["input_domain"] = "zero_phase"
        with self.assertRaises(ValueError):
            NormalizedWindowDataset(
                self.pool[:1], self.causal_store, wrong_domain, expected_fold=0
            )
        wrong_protocol = copy.deepcopy(statistics_for(self.manifest, "causal", fold=0))
        wrong_protocol["normalization_protocol_id"] = "wrong"
        with self.assertRaises(ValueError):
            NormalizedWindowDataset(
                self.pool[:1], self.causal_store, wrong_protocol, expected_fold=0
            )
        for wrong_stats in (
            statistics_for(self.manifest, "causal", fold=1),
            statistics_for(self.manifest, "causal", fold=None),
        ):
            with self.assertRaises(ValueError):
                NormalizedWindowDataset(
                    self.pool[:1], self.causal_store, wrong_stats, expected_fold=0
                )

        nonformal = self.stage1[
            (self.stage1["session"] == 0) &
            np.logical_not(formal_mask(self.causal_store, self.stage1))
        ][:1]
        self.assertEqual(len(nonformal), 1)
        with self.assertRaises(ValueError):
            NormalizedWindowDataset(
                nonformal, self.causal_store,
                statistics_for(self.manifest, "causal", fold=0),
                expected_fold=0,
            )

        tampered = copy.deepcopy(self.manifest)
        tampered["folds"][0]["train_runs"] = [0, 1, 2, 3, 4]
        with self.assertRaises(RuntimeError):
            validate_normalization_manifest(tampered)
        tampered = copy.deepcopy(self.manifest)
        tampered["sources"]["fold_manifest_sha256"] = "bad"
        with self.assertRaises(RuntimeError):
            validate_normalization_manifest(tampered)

    def test_repeat_build_cli_and_moved_manifest(self) -> None:
        original = self.manifest_path.read_bytes()
        repeated = build_normalization_manifest(self.index_dir, self.signal_dir, 1)
        path = save_normalization_manifest(self.output_dir, repeated)
        self.assertEqual(repeated, self.manifest)
        self.assertEqual(path.read_bytes(), original)

        command = [sys.executable, str(SOURCE_FILE), "--index-dir", str(self.index_dir),
                   "--signal-dir", str(self.signal_dir), "--subjects", "1",
                   "--output-dir", str(self.output_dir)]
        result = subprocess.run(command, check=False, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

        moved = self.root / "moved" / self.manifest_path.name
        moved.parent.mkdir(parents=True)
        shutil.copy2(self.manifest_path, moved)
        loaded = json.loads(moved.read_text(encoding="utf-8"))
        validate_normalization_manifest(loaded)
        loaded_stats = statistics_for(loaded, "causal", fold=None)
        self.assertEqual(loaded_stats["mean_volts"],
                         self.manifest["final_fit"]["statistics"]["causal"]["mean_volts"])
        self.assertEqual((loaded_stats["input_domain"], loaded_stats["subject"]),
                         ("causal", 1))


if __name__ == "__main__":
    unittest.main(verbosity=2)
