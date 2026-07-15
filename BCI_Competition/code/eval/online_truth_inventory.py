"""构建并读取独立于离线训练窗口的 BNCI2014001 session0 事件真值库存。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_DIR = PROJECT_ROOT / "code" / "train"
if str(TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(TRAIN_DIR))

from oof_training_bundle import (  # noqa: E402
    BUNDLE_ID,
    LEGACY_BUNDLE_ID,
    ARTIFACT_POLICY,
    SEGMENT_POLICY,
    BundleContext,
    artifact_contract,
    load_bundle,
)
from protocol_metrics import MIEvent  # noqa: E402


TRUTH_ID = "bnci2014001_s{subject:02d}_session0_clean_event_truth_native250_v1"
BASE_INDEX_ID = "bnci2014001_s{subject:02d}_native250_artifact_trial_v1"
EXPLICIT_EVENT_SOURCE = "explicit_session0_clean_event_table"
TRUTH_EVENT_DTYPE = np.dtype([
    ("subject", "u1"), ("session", "u1"), ("run", "u1"),
    ("segment", "u1"), ("trial", "u1"), ("class_id", "u1"),
    ("artifact", "?"), ("onset_native", "<i8"), ("offset_native", "<i8"),
])


@dataclass(frozen=True)
class TruthInventory:
    """已逐字段验证的真值事件及其不可变来源身份。"""

    manifest: dict
    manifest_path: Path
    manifest_sha256: str
    event_file_sha256: str
    events: tuple[MIEvent, ...]


# ---------- 文件身份与冻结写入：小型真值文件也禁止被后续运行静默覆盖 ----------
def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def lf_normalized_hash(path: Path) -> str:
    """源码身份忽略 Windows/Unix 换行差异，保证同一提交可跨电脑重建。"""
    payload = Path(path).read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: object) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _causal_segment_hash(context: BundleContext) -> str:
    """只绑定影响在线评分的因果 segment 语义，不绑定训练 bundle 的无关字段。"""
    fields = (
        "session", "run", "segment", "start_native", "stop_native",
        "formal_start_native", "formal_stop_native",
    )
    records = sorted(
        context.manifest["domains"]["causal"]["segments"],
        key=lambda item: (item["session"], item["run"], item["segment"]),
    )
    if not records or any(record["session"] != 0 for record in records):
        raise RuntimeError("真值库存只允许绑定 session0 因果 segment")
    rows = [{field: int(record[field]) for field in fields} for record in records]
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _install_frozen(temporary: Path, target: Path) -> None:
    if target.exists():
        if file_hash(target) != file_hash(temporary):
            raise FileExistsError(f"冻结真值文件内容不同，请升级协议版本: {target}")
        temporary.unlink()
    else:
        os.replace(temporary, target)


def _write_truth(root: Path, events: np.ndarray, manifest: dict) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    event_path, manifest_path = root / "events.npy", root / "manifest.json"
    temporary_files: list[Path] = []
    try:
        with tempfile.NamedTemporaryFile(dir=root, suffix=".npy", delete=False) as stream:
            temporary_event = Path(stream.name)
        temporary_files.append(temporary_event)
        np.save(temporary_event, events, allow_pickle=False)
        manifest["event_file"] = event_path.name
        manifest["event_file_sha256"] = file_hash(temporary_event)

        payload = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        with tempfile.NamedTemporaryFile(dir=root, suffix=".json", delete=False) as stream:
            temporary_manifest = Path(stream.name)
            stream.write(payload)
        temporary_files.append(temporary_manifest)

        _install_frozen(temporary_event, event_path)
        _install_frozen(temporary_manifest, manifest_path)
        return event_path, manifest_path
    finally:
        for path in temporary_files:
            if path.exists():
                path.unlink()


# ---------- 事件映射：只依据原始 clean event 与因果正式 segment，不读取离线窗口 ----------
def _formal_segment_for_event(context: BundleContext, run: int, onset: int, offset: int) -> int:
    candidates = [
        int(record["segment"])
        for record in context.manifest["domains"]["causal"]["segments"]
        if (record["session"] == 0 and record["run"] == run
            and record["formal_start_native"] <= onset < offset
            and offset <= record["formal_stop_native"])
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            f"run {run} 事件 [{onset}, {offset}) 不能唯一落入因果正式 segment；"
            "必须显式修改排除规则，不能静默删除事件",
        )
    return candidates[0]


def build_truth_inventory(
    index_dir: Path,
    bundle_manifest: Path,
    output_dir: Path,
    subject: int,
) -> tuple[dict, Path]:
    """从联合旧索引中只复制 session0 clean event；来源范围在清单中如实披露。"""
    if subject not in range(1, 10):
        raise ValueError("subject 必须为 1..9")
    context = load_bundle(Path(bundle_manifest).resolve(), verify_hashes=True)
    if context.manifest.get("subject") != subject:
        raise RuntimeError("真值被试与 session0-only bundle 不一致")
    artifact_identity = artifact_contract(context.manifest)

    base_id = BASE_INDEX_ID.format(subject=subject)
    base_index = Path(index_dir).resolve() / f"{base_id}.npz"
    base_manifest_path = Path(index_dir).resolve() / f"{base_id}_manifest.json"
    if not base_index.is_file() or not base_manifest_path.is_file():
        raise FileNotFoundError(f"缺少冻结母索引: {base_index}")
    base_manifest = json.loads(base_manifest_path.read_text(encoding="utf-8"))
    if (
        base_manifest.get("protocol_id") != base_id
        or base_manifest.get("subject") != subject
        or base_manifest.get("artifact_policy") != ARTIFACT_POLICY
        or base_manifest.get("index_file") != base_index.name
        or base_manifest.get("index_sha256") != file_hash(base_index)
    ):
        raise RuntimeError("原始事件母索引身份不匹配")

    with np.load(base_index, allow_pickle=False) as payload:
        if "events" not in payload.files:
            raise RuntimeError("原始事件母索引缺少 events 表")
        source_events = payload["events"].copy()
    required = {"subject", "session", "run", "trial", "class_id", "artifact", "mi_start", "mi_stop"}
    if source_events.dtype.names is None or not required.issubset(source_events.dtype.names):
        raise RuntimeError("原始 events 表字段不完整")
    session_events = source_events[source_events["session"] == 0]
    clean_events = np.sort(session_events[~session_events["artifact"]], order=["run", "trial"])
    expected_clean = base_manifest["summaries"]["0train"]["clean_events"]
    if len(session_events) != 288 or len(clean_events) != expected_clean:
        raise RuntimeError("session0 原始事件或 clean event 数量与母清单不一致")

    events = np.empty(len(clean_events), dtype=TRUTH_EVENT_DTYPE)
    for index, row in enumerate(clean_events):
        onset, offset = int(row["mi_start"]), int(row["mi_stop"])
        run = int(row["run"])
        events[index] = (
            subject, 0, run, _formal_segment_for_event(context, run, onset, offset),
            int(row["trial"]), int(row["class_id"]), False, onset, offset,
        )
    per_run = {str(run): int(np.sum(events["run"] == run)) for run in range(6)}
    if sum(per_run.values()) != len(events) or np.any(events["class_id"] < 1) or np.any(events["class_id"] > 4):
        raise RuntimeError("独立真值事件的 run 或类别身份非法")

    truth_id = TRUTH_ID.format(subject=subject)
    manifest = {
        "protocol_id": truth_id,
        "subject": subject,
        "included_session": 0,
        "purpose": "independent_online_event_denominator",
        "event_source": EXPLICIT_EVENT_SOURCE,
        "test_session_content_in_truth": False,
        "artifact_policy": artifact_identity["artifact_policy"],
        "segment_policy": artifact_identity["segment_policy"],
        "sampling_rate": 250,
        "event_interval": "left_closed_right_open_native_samples",
        "event_count": len(events),
        "per_run_event_count": per_run,
        "event_schema": list(TRUTH_EVENT_DTYPE.names),
        "source_scope_disclosure": (
            "rows are session0-only; source is the existing joint-session v1 index, "
            "so this artifact does not claim historical blind test isolation"
        ),
        "source_index": {
            "protocol_id": base_id,
            "index_file": base_index.name,
            "index_sha256": file_hash(base_index),
            "manifest_file": base_manifest_path.name,
            "manifest_sha256": file_hash(base_manifest_path),
        },
        "source_causal_segment_inventory_sha256": _causal_segment_hash(context),
        "builder_source_lf_sha256": lf_normalized_hash(Path(__file__).resolve()),
    }
    _, manifest_path = _write_truth(Path(output_dir).resolve() / truth_id, events, manifest)
    return manifest, manifest_path


# ---------- 正式读取：事件文件可随 bundle 移动，但必须继续绑定同一被试和因果 segment ----------
def load_truth_inventory(manifest_path: Path, context: BundleContext) -> TruthInventory:
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    subject = context.manifest.get("subject")
    expected = {
        "protocol_id": TRUTH_ID.format(subject=subject),
        "subject": subject,
        "included_session": 0,
        "purpose": "independent_online_event_denominator",
        "event_source": EXPLICIT_EVENT_SOURCE,
        "test_session_content_in_truth": False,
        "artifact_policy": artifact_contract(context.manifest)["artifact_policy"],
        "segment_policy": artifact_contract(context.manifest)["segment_policy"],
        "sampling_rate": 250,
        "event_schema": list(TRUTH_EVENT_DTYPE.names),
    }
    if (
        any(manifest.get(key) != value for key, value in expected.items())
        or manifest.get("source_causal_segment_inventory_sha256")
        != _causal_segment_hash(context)
        or not _is_sha256(manifest.get("event_file_sha256"))
    ):
        raise RuntimeError("独立在线真值清单与 session0-only bundle 不匹配")
    relative = Path(str(manifest.get("event_file", "")))
    if relative.is_absolute() or relative.name != str(relative) or relative.name != "events.npy":
        raise RuntimeError("独立在线真值事件文件路径非法")
    event_path = manifest_path.parent / relative
    if not event_path.is_file() or file_hash(event_path) != manifest["event_file_sha256"]:
        raise RuntimeError("独立在线真值事件文件缺失或哈希不一致")
    rows = np.load(event_path, allow_pickle=False)
    if (
        rows.dtype != TRUTH_EVENT_DTYPE
        or len(rows) != manifest.get("event_count")
        or np.any(rows["subject"] != subject)
        or np.any(rows["session"] != 0)
        or np.any(rows["artifact"])
        or np.any(rows["run"] > 5)
        or np.any(rows["trial"] > 47)
        or np.any(rows["class_id"] < 1)
        or np.any(rows["class_id"] > 4)
        or np.any(rows["offset_native"] - rows["onset_native"] != 1000)
    ):
        raise RuntimeError("独立在线真值事件内容非法")

    events: list[MIEvent] = []
    identities: set[tuple[int, int]] = set()
    for row in rows:
        run, trial = int(row["run"]), int(row["trial"])
        identity = run, trial
        if identity in identities:
            raise RuntimeError("独立在线真值包含重复 run/trial")
        identities.add(identity)
        onset, offset = int(row["onset_native"]), int(row["offset_native"])
        expected_segment = _formal_segment_for_event(context, run, onset, offset)
        if int(row["segment"]) != expected_segment:
            raise RuntimeError("独立在线真值事件的 segment 身份漂移")
        events.append(MIEvent(
            f"s0_r{run}_t{trial}", subject, 0, run, expected_segment,
            onset, offset, int(row["class_id"]),
        ))
    per_run = {str(run): sum(event.run_id == run for event in events) for run in range(6)}
    if per_run != manifest.get("per_run_event_count"):
        raise RuntimeError("独立在线真值逐 run 数量与清单不一致")
    return TruthInventory(
        manifest, manifest_path, file_hash(manifest_path),
        manifest["event_file_sha256"], tuple(events),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 10)))
    parser.add_argument("--index-dir", type=Path, default=PROJECT_ROOT / "data" / "processed")
    parser.add_argument("--bundle-root", type=Path, default=PROJECT_ROOT / "data" / "processed")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "processed")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    records = []
    for subject in tuple(dict.fromkeys(args.subjects)):
        candidates = [
            Path(args.bundle_root) / template.format(subject=subject) / "manifest.json"
            for template in (BUNDLE_ID, LEGACY_BUNDLE_ID)
        ]
        bundle = next((path for path in candidates if path.is_file()), candidates[-1])
        manifest, path = build_truth_inventory(
            args.index_dir, bundle, args.output_dir, subject,
        )
        records.append({"subject": subject, "event_count": manifest["event_count"], "manifest": str(path)})
    print(json.dumps(records, ensure_ascii=False, indent=2))
