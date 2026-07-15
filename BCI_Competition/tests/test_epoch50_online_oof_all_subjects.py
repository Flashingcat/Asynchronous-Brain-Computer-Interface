"""跨被试 OOF 汇总必须等权且保持 seed 配对的回归测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


EVAL_DIR = Path(__file__).resolve().parents[1] / "code" / "eval"
sys.path.insert(0, str(EVAL_DIR))

from protocol_metrics import STATEFUL_STRICT, STATELESS_DIAGNOSTIC  # noqa: E402
from run_epoch50_online_oof import KNOWN_SEEDS  # noqa: E402
from run_epoch50_online_oof_all_subjects import aggregate_subject_summaries  # noqa: E402


def fake_manifest(subject: int, seed_values: dict[int, float]) -> dict:
    """构造只含两个指标的最小合法单被试 manifest。"""
    summary = {}
    for mode in (STATELESS_DIAGNOSTIC, STATEFUL_STRICT):
        per_seed = {
            str(seed): {
                "correct_event_rate": value,
                "idle_false_commands_per_minute": value * 10,
            }
            for seed, value in seed_values.items()
        }
        summary[mode] = {
            "per_seed": per_seed,
            "aggregate": {
                field: {"mean": sum(row[field] for row in per_seed.values()) / len(per_seed)}
                for field in next(iter(per_seed.values()))
            },
        }
    seeds = tuple(seed_values)
    protocol_id = (
        f"bnci2014001_s{subject:02d}_epoch50_causal_single_window_oof_v1"
        if seeds == KNOWN_SEEDS
        else f"bnci2014001_s{subject:02d}_epoch50_causal_single_window_"
        f"seed_subset_{'_'.join(map(str, seeds))}_diagnostic_v1"
    )
    return {
        "status": "PASS",
        "protocol_id": protocol_id,
        "subject": subject,
        "seeds": list(seed_values),
        "included_session": 0,
        "test_session_access": "forbidden_and_not_loaded",
        "summary": summary,
    }


class SubjectMacroTests(unittest.TestCase):
    def test_subjects_are_equal_weight_and_seeds_remain_paired(self) -> None:
        manifests = {
            1: fake_manifest(1, {42: 0.2, 43: 0.4}),
            2: fake_manifest(2, {42: 0.8, 43: 0.6}),
        }
        result = aggregate_subject_summaries(manifests, (1, 2), (42, 43))
        stateful = result[STATEFUL_STRICT]

        # 两个被试即使有效事件数不同也只能各占一半；两个 seed 分别配对汇总。
        self.assertEqual(
            stateful["per_seed_subject_macro"]["42"]["correct_event_rate"]["mean"], 0.5,
        )
        self.assertEqual(
            stateful["per_seed_subject_macro"]["43"]["correct_event_rate"]["mean"], 0.5,
        )
        self.assertEqual(
            stateful["aggregate_across_seeds"]["correct_event_rate"]["mean"], 0.5,
        )
        self.assertEqual(
            stateful["across_subjects_from_seed_means"]["correct_event_rate"]
            ["per_subject_seed_mean"],
            {"1": 0.30000000000000004, "2": 0.7},
        )

    def test_subject_identity_mismatch_is_rejected(self) -> None:
        manifests = {1: fake_manifest(2, {42: 0.5})}
        with self.assertRaisesRegex(RuntimeError, "manifest 合同非法"):
            aggregate_subject_summaries(manifests, (1,), (42,))


if __name__ == "__main__":
    unittest.main()
