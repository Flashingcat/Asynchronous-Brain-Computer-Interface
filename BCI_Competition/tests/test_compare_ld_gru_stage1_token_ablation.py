"""Stage 1 token 配对比较器的轴完整性与参考哨兵测试。"""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = PROJECT_ROOT / "code" / "eval"
sys.path.insert(0, str(EVAL_DIR))

from compare_ld_gru_stage1_token_ablation import (  # noqa: E402
    ABLATIONS,
    FULL_PROTOCOL,
    MASK_PROTOCOL,
    REFERENCES,
    SEEDS,
    SUBJECTS,
    run,
)
from run_ld_gru_nested_loso import SUMMARY_FIELDS  # noqa: E402


def write_fixture(root: Path, *, masked: bool, tamper_reference: bool = False) -> None:
    root.mkdir()
    rows = []
    for subject in SUBJECTS:
        for seed in SEEDS:
            for policy in REFERENCES:
                value = 1.0
                if tamper_reference and subject == 1 and seed == 42 and policy == REFERENCES[0]:
                    value = 2.0
                row = {"subject": subject, "seed": seed, "policy": policy}
                row.update({field: value for field in SUMMARY_FIELDS})
                rows.append(row)
            for ablation in ABLATIONS:
                policy = (
                    f"ld_gru_mask_stage1_{ablation}" if masked else f"ld_gru_{ablation}"
                )
                row = {"subject": subject, "seed": seed, "policy": policy}
                row.update({field: (1.1 if masked else 1.0) for field in SUMMARY_FIELDS})
                rows.append(row)
    table = root / "held_out_results.csv"
    with table.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["subject", "seed", "policy", *SUMMARY_FIELDS])
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "status": "PASS",
        "scope": "full",
        "protocol_id": MASK_PROTOCOL if masked else FULL_PROTOCOL,
        "outer_subjects": list(SUBJECTS),
        "base_seeds": list(SEEDS),
        "ablations": list(ABLATIONS),
        "results": {f"r{index}": {} for index in range(54)},
        "input_root_manifest": {"sha256": "input"},
        "anchor_config": {"sha256": "anchor"},
        "artifact_policy": "official_trial_exclusion",
        "artifact_policy_binding": "legacy_v1_protocol_contract",
        "segment_policy": "separate_clean_segments_no_time_compression",
        "source_sha256": {
            "runner": "mask" if masked else "full",
            "policy": "mask" if masked else "full",
            "trainer": "mask" if masked else "full",
            "protocol_metrics": "same_evaluator",
            "shared": "same_shared_source",
        },
        "csv_artifacts": {"held_out_results": {"file": table.name}},
    }
    if masked:
        manifest["token_mode"] = "mask_stage1"
    (root / "run_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8",
    )


class PairedComparisonTests(unittest.TestCase):
    def test_complete_pairing_produces_expected_delta(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            write_fixture(base / "full", masked=False)
            write_fixture(base / "mask", masked=True)
            manifest = run(base / "full", base / "mask", base / "comparison")
            self.assertEqual(manifest["status"], "PASS")
            self.assertEqual(len(manifest["aggregate_rows"]), len(ABLATIONS) * len(SUMMARY_FIELDS))
            for row in manifest["aggregate_rows"]:
                self.assertAlmostEqual(row["paired_delta_mean_over_seeds"], 0.1)

    def test_reference_policy_change_blocks_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            write_fixture(base / "full", masked=False)
            write_fixture(base / "mask", masked=True, tamper_reference=True)
            with self.assertRaisesRegex(RuntimeError, "参考结果发生变化"):
                run(base / "full", base / "mask", base / "comparison")


if __name__ == "__main__":
    unittest.main()
