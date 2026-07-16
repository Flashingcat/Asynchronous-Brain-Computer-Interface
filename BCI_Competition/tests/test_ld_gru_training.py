"""LD-GRU 训练作业的确定性、产物哈希与恢复入口测试。"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
import json
import shutil

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = PROJECT_ROOT / "code" / "eval"
sys.path.insert(0, str(EVAL_DIR))

from ld_gru_policy import CandidateTensorSet, TokenNormalizer  # noqa: E402
from ld_gru_training import (  # noqa: E402
    TrainingHyperparameters,
    file_hash,
    train_final_job,
    train_inner_pair_job,
    update_early_stopping_endpoint,
)
from run_ld_gru_nested_loso import (  # noqa: E402
    DEFAULT_CONFIG,
    MASK_STAGE1_CONFIG,
    load_config,
)


def small_dataset() -> CandidateTensorSet:
    tokens = np.zeros((4, 9, 12), dtype=np.float32)
    centered = np.zeros((4, 9, 4), dtype=np.float32)
    valid = np.zeros((4, 9), dtype=np.bool_)
    correct = np.zeros((4, 9, 4), dtype=np.bool_)
    valid[:, :2] = True
    tokens[:, 1, 0] = np.asarray([1.0, 0.5, -0.5, -1.0])
    centered[:, 1, 0] = 1.0
    correct[0, 1, 0] = True
    correct[1, 1, 0] = True
    positive = np.any(correct, axis=(1, 2))
    return CandidateTensorSet(
        tokens, centered, valid, correct, positive,
        tuple(f"candidate_{index}" for index in range(4)),
    )


class TrainingArtifactTests(unittest.TestCase):
    def test_closed_validation_endpoint_can_never_reopen_or_change_checkpoint(self) -> None:
        record = {
            "loss": float("inf"), "epoch": 0, "patience": 0, "state": None,
            "closed": False, "closed_epoch": None,
        }
        state1 = {"weight": torch.tensor([1.0])}
        update_early_stopping_endpoint(
            record, current_loss=1.0, epoch=1, model_state=state1,
            min_delta=0.0, patience_limit=1,
        )
        update_early_stopping_endpoint(
            record, current_loss=1.1, epoch=2,
            model_state={"weight": torch.tensor([2.0])},
            min_delta=0.0, patience_limit=1,
        )
        self.assertTrue(record["closed"])
        self.assertEqual(record["closed_epoch"], 2)
        update_early_stopping_endpoint(
            record, current_loss=0.1, epoch=3,
            model_state={"weight": torch.tensor([3.0])},
            min_delta=0.0, patience_limit=1,
        )
        self.assertEqual(record["epoch"], 1)
        self.assertEqual(record["loss"], 1.0)
        self.assertTrue(torch.equal(record["state"]["weight"], state1["weight"]))

    def test_semantically_modified_v1_config_is_rejected(self) -> None:
        original = load_config(DEFAULT_CONFIG)
        modified = json.loads(json.dumps(original))
        modified["candidate_state"]["task_on_probability"] = 0.9
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "modified.json"
            path.write_text(json.dumps(modified), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "冻结协议"):
                load_config(path)

    def test_mask_stage1_config_has_an_independent_frozen_identity(self) -> None:
        """屏蔽组必须有独立 protocol_id，且任何其他字段漂移都会被拒绝。"""
        masked = load_config(MASK_STAGE1_CONFIG)
        self.assertEqual(masked["token_mode"], "mask_stage1")
        self.assertNotEqual(
            masked["protocol_id"], load_config(DEFAULT_CONFIG)["protocol_id"],
        )
        modified = json.loads(json.dumps(masked))
        modified["token"]["mask_value"] = 1.0
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "modified_mask.json"
            path.write_text(json.dumps(modified), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "冻结协议"):
                load_config(path)

    def test_inner_and_final_jobs_are_resumable_and_hash_bound(self) -> None:
        dataset = small_dataset()
        normalizer = TokenNormalizer(
            np.zeros(11, dtype=np.float32), np.ones(11, dtype=np.float32),
        )
        hyper = TrainingHyperparameters(
            batch_size=4,
            max_epochs=2,
            early_stopping_patience=1,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inner = train_inner_pair_job(
                root / "inner",
                dataset,
                {1: dataset, 2: dataset},
                normalizer,
                ablation="stop_only",
                decision_seed=1042,
                hyperparameters=hyper,
                device=torch.device("cpu"),
                contract={"kind": "unit_inner", "subjects": [1, 2]},
                verbose=False,
            )
            repeated = train_inner_pair_job(
                root / "inner",
                dataset,
                {1: dataset, 2: dataset},
                normalizer,
                ablation="stop_only",
                decision_seed=1042,
                hyperparameters=hyper,
                device=torch.device("cpu"),
                contract={"kind": "unit_inner", "subjects": [1, 2]},
                verbose=False,
            )
            self.assertEqual(inner, repeated)
            self.assertEqual(set(inner["best_checkpoints"]), {"1", "2"})
            for artifact in inner["best_checkpoints"].values():
                self.assertEqual(
                    file_hash(root / "inner" / artifact["file"]), artifact["sha256"],
                )
            latest = root / "inner" / inner["artifacts"]["latest"]["file"]
            latest.write_bytes(latest.read_bytes() + b"tamper")
            with self.assertRaisesRegex(RuntimeError, "审计产物哈希"):
                train_inner_pair_job(
                    root / "inner", dataset, {1: dataset, 2: dataset}, normalizer,
                    ablation="stop_only", decision_seed=1042,
                    hyperparameters=hyper, device=torch.device("cpu"),
                    contract={"kind": "unit_inner", "subjects": [1, 2]}, verbose=False,
                )

            final = train_final_job(
                root / "final",
                dataset,
                normalizer,
                ablation="stop_residual",
                decision_seed=1042,
                fixed_epochs=1,
                hyperparameters=hyper,
                device=torch.device("cpu"),
                contract={"kind": "unit_final", "outer": 1},
                verbose=False,
            )
            self.assertEqual(final["fixed_epochs"], 1)
            self.assertEqual(
                file_hash(root / "final" / final["final_checkpoint"]["file"]),
                final["final_checkpoint"]["sha256"],
            )

    def test_mask_stage1_mode_is_bound_into_training_contract(self) -> None:
        """屏蔽方式属于正式协议，恢复时不得把 full checkpoint 当成 mask 使用。"""
        dataset = small_dataset()
        normalizer = TokenNormalizer(
            np.zeros(11, dtype=np.float32), np.ones(11, dtype=np.float32),
        )
        hyper = TrainingHyperparameters(
            batch_size=4,
            max_epochs=1,
            early_stopping_patience=1,
        )
        contract = {
            "kind": "unit_mask_stage1",
            "subjects": [1, 2],
            "token_mode": "mask_stage1",
        }
        with tempfile.TemporaryDirectory() as directory:
            job_dir = Path(directory) / "inner"
            completed = train_inner_pair_job(
                job_dir, dataset, {1: dataset, 2: dataset}, normalizer,
                ablation="stop_only", token_mode="mask_stage1",
                decision_seed=1042, hyperparameters=hyper,
                device=torch.device("cpu"), contract=contract, verbose=False,
            )
            self.assertEqual(completed["status"], "complete")

            # 同一目录切回 full 必须在读 checkpoint 前就失败，防止消融串线。
            with self.assertRaisesRegex(ValueError, "内层作业"):
                train_inner_pair_job(
                    job_dir, dataset, {1: dataset, 2: dataset}, normalizer,
                    ablation="stop_only", token_mode="full",
                    decision_seed=1042, hyperparameters=hyper,
                    device=torch.device("cpu"), contract=contract, verbose=False,
                )

    def test_resume_after_both_endpoints_closed_does_not_add_epoch(self) -> None:
        """模拟 latest 已落盘、completed 尚未落盘时断电的精确恢复边界。"""
        dataset = small_dataset()
        normalizer = TokenNormalizer(
            np.zeros(11, dtype=np.float32), np.ones(11, dtype=np.float32),
        )
        # 巨大的 min_delta 保证 epoch 1 成为 best，epoch 2 两端同时首次关闭。
        hyper = TrainingHyperparameters(
            batch_size=4,
            max_epochs=3,
            early_stopping_patience=1,
            early_stopping_min_delta=1e9,
        )
        contract = {"kind": "unit_inner_closed_resume", "subjects": [1, 2]}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clean_dir = root / "clean"
            clean = train_inner_pair_job(
                clean_dir, dataset, {1: dataset, 2: dataset}, normalizer,
                ablation="stop_only", decision_seed=1042,
                hyperparameters=hyper, device=torch.device("cpu"),
                contract=contract, verbose=False,
            )
            self.assertEqual(clean["completed_epochs"], 2)
            self.assertEqual(
                {value["endpoint_closed_epoch"] for value in clean["best_checkpoints"].values()},
                {2},
            )

            # crash_dir 只保留中断点前已原子落盘的三个文件。
            crash_dir = root / "crash"
            crash_dir.mkdir()
            for name in ("job_config.json", "token_normalizer.npz", "latest.pt"):
                shutil.copy2(clean_dir / name, crash_dir / name)
            resumed = train_inner_pair_job(
                crash_dir, dataset, {1: dataset, 2: dataset}, normalizer,
                ablation="stop_only", decision_seed=1042,
                hyperparameters=hyper, device=torch.device("cpu"),
                contract=contract, verbose=False,
            )
            self.assertEqual(resumed["completed_epochs"], clean["completed_epochs"])
            for subject in ("1", "2"):
                clean_record = clean["best_checkpoints"][subject]
                resumed_record = resumed["best_checkpoints"][subject]
                self.assertEqual(resumed_record["best_epoch"], clean_record["best_epoch"])
                self.assertEqual(
                    resumed_record["endpoint_closed_epoch"],
                    clean_record["endpoint_closed_epoch"],
                )


if __name__ == "__main__":
    unittest.main()
