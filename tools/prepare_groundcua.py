"""Prepare the official GroundCUA instruction-tuning archive for EasyR1.

The GroundCUA repository publishes a curated instruction-tuning archive. It
contains roughly 700K generated instructions derived from dense screenshot
annotations; this script intentionally does not synthesize prompts from the
3.2M raw element annotations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

PROJECT_ROOT = Path(os.environ.get("QWEN3_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
DATA_ROOT = PROJECT_ROOT / "datasets" / "groundcua"
ANNOTATION_ROOT = DATA_ROOT / "data"
IMAGE_ROOT = DATA_ROOT / "images"
INSTRUCTION_ROOT = DATA_ROOT / "instruction_tuning"
INSTRUCTION_DATA_ROOT = INSTRUCTION_ROOT / "groundcua_data"
OUTPUT_ROOT = DATA_ROOT / "easyr1"
SEED = 42
SPLIT_RATIO = 0.9
VALIDATION_SUBSET_SIZE = 8000
HF_REPO = "ServiceNow/GroundCUA"
HF_REVISION = "5d6845b0116029d46ec762e734701c5b8ce207c3"
ARCHIVE_NAME = "instruction_tuning.tar.gz"
PARQUET_COLUMNS = ["id", "problem", "images", "answer", "data_type", "data_source"]

INSTRUCTION_FILES = {
    "direct_description": "direct_description_instructions.json",
    "direct_general_templates": "direct_general_templates_instructions.json",
    "direct_icon": "direct_icon_instructions.json",
    "direct_miscellaneous": "direct_miscellaneous_instructions.json",
    "direct_text": "direct_text_instructions.json",
    "functional": "functional_instructions.json",
    "functional_extra": "functional_instructions_extra.json",
    "spatial": "spatial_data.json",
}


def _schema() -> pa.Schema:
    return pa.schema([
        ("id", pa.string()), ("problem", pa.string()), ("images", pa.list_(pa.string())),
        ("answer", pa.string()), ("data_type", pa.string()), ("data_source", pa.string()),
    ])


@lru_cache(maxsize=16384)
def image_size(image_path: str) -> tuple[int, int]:
    with Image.open(image_path) as image:
        return image.size


@lru_cache(maxsize=16384)
def screenshot_elements(platform: str, screenshot_id: str) -> tuple[dict[str, Any], ...]:
    path = ANNOTATION_ROOT / platform / f"{screenshot_id}.json"
    if not path.is_file():
        return ()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ()
    return tuple(payload) if isinstance(payload, list) else ()


def find_element_for_coordinate(platform: str, screenshot_id: str, coordinate: list[Any]) -> dict[str, Any] | None:
    """Resolve coordinate-only records using the authors' utility rule."""
    if not isinstance(coordinate, (list, tuple)) or len(coordinate) != 2:
        return None
    try:
        x, y = float(coordinate[0]), float(coordinate[1])
    except (TypeError, ValueError):
        return None
    matches: list[tuple[float, dict[str, Any]]] = []
    for element in screenshot_elements(platform, screenshot_id):
        bbox = element.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        try:
            x1, y1, x2, y2 = map(float, bbox)
        except (TypeError, ValueError):
            continue
        x1, x2 = sorted((x1, x2)); y1, y2 = sorted((y1, y2))
        if x1 <= x <= x2 and y1 <= y <= y2:
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            matches.append((((cx - x) ** 2 + (cy - y) ** 2) ** 0.5, element))
    return min(matches, key=lambda pair: pair[0])[1] if matches else None


def normalize_bbox(bbox: Any, width: int, height: int) -> tuple[list[float], list[float]]:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValueError("bbox_shape")
    try:
        x1, y1, x2, y2 = (float(value) for value in bbox)
    except (TypeError, ValueError):
        raise ValueError("bbox_non_numeric") from None
    if not all(value == value and abs(value) != float("inf") for value in (x1, y1, x2, y2)):
        raise ValueError("bbox_non_finite")
    if not (x1 < x2 and y1 < y2):
        raise ValueError("bbox_non_positive_area")
    tolerance = 1e-3
    if x1 < -tolerance or y1 < -tolerance or x2 > width + tolerance or y2 > height + tolerance:
        raise ValueError("bbox_outside_image_bounds")
    x1, y1 = max(0.0, x1), max(0.0, y1)
    x2, y2 = min(float(width), x2), min(float(height), y2)
    normalized = [x1 / width * 1000, y1 / height * 1000, x2 / width * 1000, y2 / height * 1000]
    center = [(normalized[0] + normalized[2]) / 2, (normalized[1] + normalized[3]) / 2]
    return normalized, center


def is_ambiguous_instruction(instruction: str) -> bool:
    normalized = re.sub(r"\s+", " ", instruction.strip().lower()).rstrip(".!?")
    return normalized in {
        "click the button", "click the button ui element", "click the target",
        "click the target ui element", "click the target ui element in the screenshot",
        "select the button", "select the target", "select the target ui element",
    }


def archive_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not INSTRUCTION_DATA_ROOT.is_dir():
        raise FileNotFoundError(f"Extract {INSTRUCTION_ROOT / ARCHIVE_NAME} into {INSTRUCTION_DATA_ROOT} first")
    rows: list[dict[str, Any]] = []
    excluded: Counter[str] = Counter()
    file_counts: dict[str, int] = {}
    duplicate_keys: Counter[str] = Counter()
    for instruction_type, filename in INSTRUCTION_FILES.items():
        path = INSTRUCTION_DATA_ROOT / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"Expected a JSON list in {path}")
        file_counts[instruction_type] = len(payload)
        for index, item in enumerate(payload):
            if not isinstance(item, dict):
                excluded["non_dict_record"] += 1; continue
            screenshot_id = str(item.get("id") or "").strip()
            platform = str(item.get("platform") or "").strip()
            instruction = str(item.get("instruction") or "").strip()
            if not screenshot_id or not platform:
                excluded["missing_screenshot_reference"] += 1; continue
            if not instruction:
                excluded["empty_instruction"] += 1; continue
            if is_ambiguous_instruction(instruction):
                excluded["ambiguous_generic_instruction"] += 1; continue
            image_path = IMAGE_ROOT / platform / f"{screenshot_id}.png"
            if not image_path.is_file():
                excluded["missing_image"] += 1; continue
            try:
                width, height = image_size(str(image_path))
            except Exception:
                excluded["unreadable_image"] += 1; continue
            bbox = item.get("bbox")
            if bbox is None:
                matched = find_element_for_coordinate(platform, screenshot_id, item.get("coordinate", item.get("cordinate")))
                if matched is None:
                    excluded["coordinate_bbox_not_found"] += 1; continue
                bbox = matched.get("bbox")
            try:
                normalized, center = normalize_bbox(bbox, width, height)
            except ValueError as exc:
                excluded[str(exc)] += 1; continue
            group_key = f"{platform}/{screenshot_id}"
            duplicate_keys[group_key] += 1
            rows.append({"group": group_key, "platform": platform, "screenshot_id": screenshot_id,
                         "instruction_type": instruction_type, "instruction": instruction,
                         "image_path": str(image_path.absolute()), "bbox": normalized, "center": center,
                         "source_index": index})
    return rows, {"file_counts": file_counts, "excluded": dict(sorted(excluded.items())),
                  "duplicate_instruction_groups": sum(count - 1 for count in duplicate_keys.values() if count > 1),
                  "unique_screenshot_groups": len(duplicate_keys)}


def stratified_subset(rows: list[dict[str, Any]], target_size: int, seed: int) -> list[dict[str, Any]]:
    if len(rows) <= target_size:
        return rows
    rng = random.Random(seed)
    strata: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        strata[(row["platform"], row["instruction_type"])].append(row)
    for values in strata.values():
        rng.shuffle(values)
    keys = sorted(strata); selected: list[dict[str, Any]] = []
    allocations = {key: 1 for key in keys}
    remaining = target_size - len(keys)
    total = sum(len(values) for values in strata.values())
    fractional = {key: remaining * len(strata[key]) / total for key in keys}
    for key in keys:
        allocations[key] += min(len(strata[key]) - 1, int(fractional[key]))
    while sum(allocations.values()) < target_size:
        candidates = [key for key in keys if allocations[key] < len(strata[key])]
        if not candidates: break
        key = max(candidates, key=lambda candidate: fractional[candidate] - allocations[candidate])
        allocations[key] += 1
    for key in keys:
        selected.extend(strata[key][:allocations[key]])
    rng.shuffle(selected)
    return selected[:target_size]


def split_rows(rows: list[dict[str, Any]], seed: int, split_ratio: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows: groups[row["group"]].append(row)
    group_platform = {group: values[0]["platform"] for group, values in groups.items()}
    platform_groups: dict[str, list[str]] = defaultdict(list)
    for group, platform in group_platform.items(): platform_groups[platform].append(group)
    rng = random.Random(seed); validation_groups: set[str] = set()
    for platform, values in sorted(platform_groups.items()):
        rng.shuffle(values)
        validation_groups.update(values[:max(1, round(len(values) * (1 - split_ratio)))])
    target_groups = max(1, round(len(groups) * (1 - split_ratio)))
    all_groups = list(groups); rng.shuffle(all_groups)
    for group in all_groups:
        if len(validation_groups) >= target_groups: break
        validation_groups.add(group)
    for instruction_type in INSTRUCTION_FILES:
        if any(row["instruction_type"] == instruction_type and row["group"] in validation_groups for row in rows): continue
        candidates = [group for group, values in groups.items() if group not in validation_groups and any(r["instruction_type"] == instruction_type for r in values)]
        if candidates: validation_groups.add(candidates[0])
    train = [row for row in rows if row["group"] not in validation_groups]
    validation = [row for row in rows if row["group"] in validation_groups]
    return train, validation, {"total_groups": len(groups), "validation_groups": len(validation_groups)}


def write_parquet(rows: list[dict[str, Any]], path: Path) -> int:
    writer = pq.ParquetWriter(path, _schema(), compression="zstd")
    buffers: dict[str, list[Any]] = {column: [] for column in PARQUET_COLUMNS}
    try:
        for index, row in enumerate(rows):
            buffers["id"].append(f"{row['instruction_type']}:{row['group']}#{index}")
            buffers["problem"].append(row["instruction"]); buffers["images"].append([row["image_path"]])
            buffers["answer"].append(json.dumps({"bbox": row["bbox"], "center": row["center"]}, separators=(",", ":")))
            buffers["data_type"].append(row["instruction_type"]); buffers["data_source"].append("groundcua_official_instruction_tuning")
            if len(buffers["id"]) >= 8192:
                writer.write_table(pa.Table.from_pydict(buffers, schema=_schema())); buffers = {column: [] for column in PARQUET_COLUMNS}
        if buffers["id"]: writer.write_table(pa.Table.from_pydict(buffers, schema=_schema()))
    finally:
        writer.close()
    return len(rows)


def split_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {"screenshots": len({row["group"] for row in rows}), "annotations": len(rows),
            "platforms": dict(sorted(Counter(row["platform"] for row in rows).items())),
            "instruction_types": dict(sorted(Counter(row["instruction_type"] for row in rows).items()))}


def prepare(seed: int = SEED, split_ratio: float = SPLIT_RATIO, validation_size: int = VALIDATION_SUBSET_SIZE) -> dict[str, Any]:
    rows, raw_stats = load_rows()
    train_rows, validation_full_rows, split_info = split_rows(rows, seed, split_ratio)
    validation_rows = stratified_subset(validation_full_rows, validation_size, seed)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    write_parquet(train_rows, OUTPUT_ROOT / "train.parquet")
    write_parquet(validation_full_rows, OUTPUT_ROOT / "validation_full.parquet")
    write_parquet(validation_rows, OUTPUT_ROOT / "validation.parquet")
    archive_path = INSTRUCTION_ROOT / ARCHIVE_NAME
    manifest = {
        "dataset": "GroundCUA official instruction-tuning data",
        "source": {"hf_repo": HF_REPO, "revision": HF_REVISION, "archive": ARCHIVE_NAME,
                   "archive_path": str(archive_path.absolute()), "archive_bytes": archive_path.stat().st_size if archive_path.is_file() else None,
                   "archive_sha256": archive_sha256(archive_path) if archive_path.is_file() else None,
                   "repository_instruction_readme": "instruction_tuning/README.md at the same GroundCUA revision"},
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(), "seed": seed, "split_ratio": split_ratio,
        "split_rule": "screenshot-level groups (platform/id) with per-platform 90/10 allocation; all instruction types represented when possible",
        "raw_instruction_files": raw_stats["file_counts"], "raw_instruction_count": sum(raw_stats["file_counts"].values()),
        "remaining_instruction_count_excluded": 449846,
        "usable_instruction_count": len(rows), "excluded_count": sum(raw_stats["excluded"].values()),
        "excluded_reasons": raw_stats["excluded"], "duplicate_instruction_groups": raw_stats["duplicate_instruction_groups"],
        "unique_screenshot_groups": raw_stats["unique_screenshot_groups"],
        "splits": {"train": split_stats(train_rows), "validation_full": split_stats(validation_full_rows),
                   "validation": {**split_stats(validation_rows), "selection": "fixed platform/instruction_type stratified subset", "target_size": validation_size},
                   "group_counts": split_info},
        "raw_schema": {"bbox_files": "id, bbox, text, category, platform, all_instructions, instruction",
                       "coordinate_files": "id, platform, coordinate/cordinate, instruction; bbox recovered from matching raw annotation",
                       "excluded_from_training": "remaining_instructions.json (official README says it was outside the 700K budget)"},
        "coordinate_normalization": "absolute pixel [x1,y1,x2,y2] -> [1000*x1/width, 1000*y1/height, 1000*x2/width, 1000*y2/height]; center derived from normalized bbox",
        "parquet_columns": PARQUET_COLUMNS,
    }
    (OUTPUT_ROOT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUTPUT_ROOT / "SCHEMA_REPORT.md").write_text(
        "# Official GroundCUA instruction-tuning schema\n\n"
        "This preparation uses the eight curated/generated files in the official `instruction_tuning.tar.gz` and excludes `remaining_instructions.json`. "
        "Direct and functional records contain an absolute pixel bbox; spatial and functional-extra records contain a pixel coordinate and are resolved to the closest containing raw element using the authors' utility rule. The original instruction text is copied verbatim into `problem`.\n\n"
        f"Train/validation assignment is screenshot-level (`platform/id`) with seed 42. `validation.parquet` is a fixed {validation_size}-row stratified subset of `validation_full.parquet`; the full held-out split is preserved.\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--split-ratio", type=float, default=SPLIT_RATIO); parser.add_argument("--validation-size", type=int, default=VALIDATION_SUBSET_SIZE)
    args = parser.parse_args(); manifest = prepare(args.seed, args.split_ratio, args.validation_size)
    for split in ("train", "validation_full", "validation"):
        stats = manifest["splits"][split]; print(f"{split}: {stats['screenshots']} screenshots, {stats['annotations']} instructions")
    print(f"Wrote official GroundCUA EasyR1 data to {OUTPUT_ROOT.absolute()}")


if __name__ == "__main__": main()
