"""训练 session 六折留一 run 验证清单的真实数据回归测试。"""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np


PREPROCESSING_DIR = Path(__file__).resolve().parents[1] / "code" / "preprocessing"
SOURCE_FILE = PREPROCESSING_DIR / "build_validation_folds.py"
sys.path.insert(0, str(PREPROCESSING_DIR))

from build_offline_view import build_offline_view, load_base, save_offline_view  # noqa: E402
from build_protocol_index import build_subject, save_subject  # noqa: E402
from build_validation_folds import (  # noqa: E402
    FOLD_ID,
    build_fold_manifest,
    load_offline,
    save_fold_manifest,
)


class RealValidationFoldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        value = os.environ.get("BNCI2014001_ROOT")
        if not value:
            raise RuntimeError("必须设置 BNCI2014001_ROOT，真实数据测试不得跳过")
        cls.temporary = tempfile.TemporaryDirectory()
        cls.index_dir = Path(cls.temporary.name)
        cls.manifests = {}

        # 从原始MAT重新构建全部上游产物，避免测试依赖工作区中的旧缓存。
        for subject in range(1, 10):
            built = build_subject(Path(value), subject)
            base_manifest = save_subject(cls.index_dir, subject, built)
            offline = build_offline_view(built[0], built[1], built[2], base_manifest)
            base_id = base_manifest["protocol_id"]
            base_files = (cls.index_dir / f"{base_id}.npz",
                          cls.index_dir / f"{base_id}_manifest.json")
            save_offline_view(cls.index_dir, subject, offline, base_files)
            cls.manifests[subject] = build_fold_manifest(cls.index_dir, subject)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_fixed_protocol_structure_for_all_subjects(self) -> None:
        for subject, manifest in self.manifests.items():
            self.assertEqual(manifest["protocol_id"], FOLD_ID.format(subject=subject))
            self.assertEqual(manifest["strategy"], "six_fold_leave_one_complete_run_out")
            self.assertEqual(manifest["split_unit"], "complete_run")
            self.assertIsNone(manifest["split_seed"])
            self.assertEqual(manifest["train_session"], 0)
            self.assertEqual(manifest["final_test_session"], 1)
            self.assertEqual(manifest["final_fit_runs"], list(range(6)))
            self.assertEqual(len(manifest["folds"]), 6)

    def test_each_run_is_validation_once_and_never_mixed(self) -> None:
        for manifest in self.manifests.values():
            held_out = []
            for fold in manifest["folds"]:
                train_runs = set(fold["train_runs"])
                validation_runs = set(fold["validation_runs"])
                self.assertTrue(train_runs.isdisjoint(validation_runs))
                self.assertEqual(train_runs | validation_runs, set(range(6)))
                self.assertEqual(validation_runs, {fold["fold"]})
                held_out.extend(validation_runs)
            self.assertEqual(Counter(held_out), Counter({run: 1 for run in range(6)}))

    def test_oof_counts_cover_full_training_session_once(self) -> None:
        for manifest in self.manifests.values():
            full = manifest["full_train_counts"]
            for key, expected in full.items():
                observed = sum(fold["validation_counts"][key] for fold in manifest["folds"])
                self.assertEqual(observed, expected)
            for fold in manifest["folds"]:
                for key, expected in full.items():
                    observed = fold["train_counts"][key] + fold["validation_counts"][key]
                    self.assertEqual(observed, expected)

    def test_source_rows_follow_the_same_run_assignment(self) -> None:
        """离线表和在线母表必须使用完全相同的run边界。"""
        for subject, manifest in self.manifests.items():
            events, segments, online, _, base_index, base_manifest = load_base(self.index_dir, subject)
            stage1, stage2, _, _, _ = load_offline(
                self.index_dir, subject, (base_index, base_manifest)
            )
            for fold in manifest["folds"]:
                train_runs = fold["train_runs"]
                validation_runs = fold["validation_runs"]
                for array in (events, segments, online, stage1, stage2):
                    train = array[(array["session"] == 0) & np.isin(array["run"], train_runs)]
                    validation = array[(array["session"] == 0) &
                                       np.isin(array["run"], validation_runs)]
                    self.assertTrue(set(train["run"]).isdisjoint(set(validation["run"])))
                    self.assertEqual(len(train) + len(validation), int((array["session"] == 0).sum()))

    def test_subject1_golden_counts_per_validation_run(self) -> None:
        folds = self.manifests[1]["folds"]
        self.assertEqual([fold["validation_counts"]["clean_events"] for fold in folds],
                         [46, 44, 45, 47, 46, 45])
        self.assertEqual([fold["validation_counts"]["online_windows"] for fold in folds],
                         [740, 709, 724, 755, 740, 724])
        self.assertEqual([fold["validation_counts"]["offline_stage1_windows"] for fold in folds],
                         [421, 402, 412, 429, 421, 411])
        self.assertEqual([fold["validation_counts"]["offline_task_windows"] for fold in folds],
                         [230, 220, 225, 235, 230, 225])

    def test_test_session_is_counted_but_never_used_for_selection(self) -> None:
        for manifest in self.manifests.values():
            self.assertEqual(manifest["selection_policy"]["validation_predictions"],
                             "out_of_fold_only")
            self.assertEqual(manifest["selection_policy"]["test_usage"],
                             "once_after_all_parameters_are_frozen")
            self.assertEqual(manifest["final_test_counts"]["events"], 288)
            for fold in manifest["folds"]:
                self.assertNotIn("test_runs", fold)

    def test_frozen_save_and_cli(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            manifest = self.manifests[1]
            first = save_fold_manifest(output, manifest)
            original = first.read_bytes()
            self.assertEqual(save_fold_manifest(output, manifest), first)
            changed = copy.deepcopy(manifest)
            changed["folds"][0]["validation_runs"] = [1]
            with self.assertRaises(FileExistsError):
                save_fold_manifest(output, changed)
            self.assertEqual(first.read_bytes(), original)

            command = [sys.executable, str(SOURCE_FILE), "--index-dir", str(self.index_dir),
                       "--subjects", "4", "--output-dir", str(output)]
            result = subprocess.run(command, check=False, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            files = list(output.glob("*_train6fold_leave_one_run_out_v1.json"))
            self.assertEqual(len(files), 2)
            for path in files:
                json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
