from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Mapping


SUPPORTED_SCHEMA = "five-miao-node-annotation-v1"


def bridge_name_from_source(file_name: str) -> str:
    """Recover the human bridge name from the source drawing file name."""
    stem = Path(file_name).stem.strip()
    stem = re.sub(r"^\d+\.", "", stem)
    stem = re.sub(r"(?:-纵剖.*|_页面_\d+.*)$", "", stem)

    location_matches = re.findall(r"(?:县|市)([^.\\/_-]*桥)", stem)
    if location_matches:
        return location_matches[-1]
    bridge_match = re.search(r"([^.\\/_-]*桥)", stem)
    return bridge_match.group(1) if bridge_match else stem


def source_group_from_payload(payload: Mapping[str, Any]) -> str:
    bridge = payload.get("bridge") if isinstance(payload.get("bridge"), Mapping) else {}
    existing = str(bridge.get("source_group_id") or "").strip()
    if existing:
        return existing
    bridge_id = str(bridge.get("bridge_id") or "").strip()
    if bridge_id:
        return f"bridge:{bridge_id}"
    file_name = str(
        payload.get("source_image", {}).get("file_name", "")
        if isinstance(payload.get("source_image"), Mapping)
        else ""
    ).strip()
    source_id = re.match(r"^(\d+)\.", Path(file_name).name)
    if source_id:
        return f"drawing:{source_id.group(1)}"
    return f"drawing:{bridge_name_from_source(file_name)}"


def _first(*values: Any) -> Any:
    """Return the first value that is neither None nor an empty string."""
    for value in values:
        if value is not None and value != "":
            return value
    return None


def repaired_file_name(payload: Mapping[str, Any]) -> str:
    source_image = payload.get("source_image")
    if not isinstance(source_image, Mapping) or not source_image.get("file_name"):
        raise ValueError("source_image.file_name is required")
    source_stem = Path(str(source_image["file_name"])).stem
    span = payload.get("span") if isinstance(payload.get("span"), Mapping) else {}
    bridge = payload.get("bridge") if isinstance(payload.get("bridge"), Mapping) else {}
    annotation = payload.get("annotation") if isinstance(payload.get("annotation"), Mapping) else {}
    span_count = int(_first(span.get("count"), bridge.get("span_count")) or 1)
    span_index = int(_first(span.get("index"), annotation.get("span_index")) or 1)
    if span_count < 1 or not 1 <= span_index <= span_count:
        raise ValueError(f"invalid span index/count: {span_index}/{span_count}")
    if span_count == 1:
        return f"{source_stem}_five_miao_nodes.json"
    width = max(2, len(str(span_count)))
    return f"{source_stem}_span_{span_index:0{width}d}_of_{span_count:0{width}d}_five_miao_nodes.json"


def repair_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("schema_version") != SUPPORTED_SCHEMA:
        raise ValueError(f"unsupported schema_version: {payload.get('schema_version')!r}")
    source_image = payload.get("source_image")
    if not isinstance(source_image, Mapping) or not source_image.get("file_name"):
        raise ValueError("source_image.file_name is required")
    bridge = payload.setdefault("bridge", {})
    if not isinstance(bridge, dict):
        raise ValueError("bridge must be an object")
    bridge["bridge_name"] = bridge_name_from_source(str(source_image["file_name"]))
    bridge["source_group_id"] = source_group_from_payload(payload)
    return payload


def repair_directory(directory: Path, *, apply: bool = False) -> tuple[list[tuple[str, str, str]], Path | None]:
    directory = directory.resolve()
    paths = sorted(directory.glob("*.json"))
    if not paths:
        raise ValueError(f"no JSON files found in {directory}")

    repaired: list[tuple[Path, str, dict[str, Any]]] = []
    targets: dict[str, Path] = {}
    changes: list[tuple[str, str, str]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        old_name = str(payload.get("bridge", {}).get("bridge_name") or "")
        repair_payload(payload)
        target_name = repaired_file_name(payload)
        if target_name in targets:
            raise ValueError(f"multiple records would become {target_name}: {targets[target_name].name}, {path.name}")
        targets[target_name] = path
        repaired.append((path, target_name, payload))
        changes.append((path.name, target_name, f"{old_name} -> {payload['bridge']['bridge_name']}"))

    if not apply:
        return changes, None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = directory.parent / f"{directory.name}_桥名修复备份_{timestamp}"
    backup.mkdir(parents=False, exist_ok=False)
    for path in paths:
        shutil.copy2(path, backup / path.name)

    stage = Path(tempfile.mkdtemp(prefix=f".{directory.name}_repair_", dir=directory.parent))
    try:
        for _, target_name, payload in repaired:
            (stage / target_name).write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        for path in paths:
            path.unlink()
        for staged in stage.glob("*.json"):
            shutil.move(str(staged), directory / staged.name)
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    return changes, backup


def main() -> int:
    parser = argparse.ArgumentParser(description="Repair bridge names and group IDs from annotation source images.")
    parser.add_argument("directory", type=Path, help="Directory containing annotation JSON files")
    parser.add_argument("--apply", action="store_true", help="Back up and rewrite files; default is dry-run")
    args = parser.parse_args()
    changes, backup = repair_directory(args.directory, apply=args.apply)
    for old_file, new_file, name_change in changes:
        print(f"{old_file} -> {new_file} | {name_change}")
    print(f"records={len(changes)} mode={'applied' if args.apply else 'dry-run'}")
    if backup:
        print(f"backup={backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
