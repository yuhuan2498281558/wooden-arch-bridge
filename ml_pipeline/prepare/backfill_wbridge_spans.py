from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterable, Mapping
from urllib.parse import unquote, urlparse
from urllib.request import urlopen


SUPPORTED_SCHEMA = "five-miao-node-annotation-v1"
DEFAULT_API_ROOT = "https://aujrht7gmph6dk84wqh975zv.z7.web.core.windows.net/api"
DETAIL_PAGE_ROOT = "http://w-bridge.wiki/#/detail"

# The structured field is empty for this entry, but the database description
# explicitly states "拱跨约19米". Keep the approximation visible in the audit.
APPROXIMATE_SPAN_OVERRIDES: dict[int, tuple[float, str]] = {
    47: (19.0, "description_approximate: 拱跨约19米"),
}


@dataclass(frozen=True)
class BackfillPlan:
    path: Path
    source_image_file: str
    api_bridge_id: int
    api_bridge_name: str
    span_m: float
    source_field: str

    @property
    def detail_url(self) -> str:
        return f"{DETAIL_PAGE_ROOT}/{self.api_bridge_id}"


def _read_json_url(url: str, *, timeout: float = 30.0) -> Any:
    with urlopen(url, timeout=timeout) as response:
        return json.load(response)


def fetch_bridge_details(
    *,
    api_root: str = DEFAULT_API_ROOT,
    workers: int = 12,
) -> list[dict[str, Any]]:
    listing = _read_json_url(f"{api_root}/bridge/list")
    bridge_ids = [int(item["id"]) for item in listing.get("result", [])]
    if not bridge_ids:
        raise ValueError("the historical bridge database returned no bridge records")

    details: list[dict[str, Any]] = []
    errors: list[str] = []

    def fetch_one(bridge_id: int) -> dict[str, Any]:
        payload = _read_json_url(f"{api_root}/bridge/{bridge_id}/detail")
        result = payload.get("result")
        if not isinstance(result, dict):
            raise ValueError(f"bridge {bridge_id} detail response has no result object")
        return result

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {executor.submit(fetch_one, bridge_id): bridge_id for bridge_id in bridge_ids}
        for future in as_completed(futures):
            bridge_id = futures[future]
            try:
                details.append(future.result())
            except Exception as error:  # pragma: no cover - network-specific path
                errors.append(f"{bridge_id}: {error}")

    if errors:
        preview = "; ".join(errors[:5])
        raise RuntimeError(f"failed to read {len(errors)} bridge detail records: {preview}")
    return details


def drawing_file_name(url: str) -> str:
    return Path(unquote(urlparse(url).path)).name


def build_drawing_index(details: Iterable[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    index: dict[str, list[Mapping[str, Any]]] = {}
    for detail in details:
        for drawing in detail.get("drawings") or []:
            if not isinstance(drawing, Mapping) or not drawing.get("url"):
                continue
            file_name = drawing_file_name(str(drawing["url"]))
            index.setdefault(file_name, []).append(detail)
    return index


def span_from_detail(detail: Mapping[str, Any]) -> tuple[float, str] | None:
    value = detail.get("max_span_length")
    if value is not None and value != "":
        span_m = float(value)
        if span_m > 0:
            return span_m, "max_span_length"

    bridge_id = int(detail.get("id") or 0)
    override = APPROXIMATE_SPAN_OVERRIDES.get(bridge_id)
    if override:
        return override
    return None


def _existing_span(payload: Mapping[str, Any]) -> Any:
    span = payload.get("span") if isinstance(payload.get("span"), Mapping) else {}
    bridge = payload.get("bridge") if isinstance(payload.get("bridge"), Mapping) else {}
    dimensions = payload.get("dimensions") if isinstance(payload.get("dimensions"), Mapping) else {}
    for value in (span.get("clear_span_m"), bridge.get("span_m"), dimensions.get("span_m")):
        if value is not None and value != "":
            return value
    return None


def _span_field_values(payload: Mapping[str, Any]) -> tuple[Any, Any, Any]:
    span = payload.get("span") if isinstance(payload.get("span"), Mapping) else {}
    bridge = payload.get("bridge") if isinstance(payload.get("bridge"), Mapping) else {}
    dimensions = payload.get("dimensions") if isinstance(payload.get("dimensions"), Mapping) else {}
    return span.get("clear_span_m"), bridge.get("span_m"), dimensions.get("span_m")


def plan_directory(
    directory: Path,
    details: Iterable[Mapping[str, Any]],
) -> tuple[list[BackfillPlan], list[str], list[str]]:
    directory = directory.resolve()
    paths = sorted(directory.glob("*.json"))
    if not paths:
        raise ValueError(f"no JSON files found in {directory}")

    drawing_index = build_drawing_index(details)
    plans: list[BackfillPlan] = []
    skipped: list[str] = []
    unresolved: list[str] = []

    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        if payload.get("schema_version") != SUPPORTED_SCHEMA:
            unresolved.append(f"{path.name}: unsupported schema_version")
            continue

        clear_span, bridge_span, dimensions_span = _span_field_values(payload)
        present_values = [value for value in (clear_span, bridge_span, dimensions_span) if value is not None and value != ""]
        numeric_values = [float(value) for value in present_values]
        if numeric_values and max(numeric_values) - min(numeric_values) > 1e-9:
            unresolved.append(f"{path.name}: conflicting existing span fields={present_values}")
            continue
        if clear_span is not None and clear_span != "" and bridge_span is not None and bridge_span != "":
            skipped.append(path.name)
            continue

        source_image = payload.get("source_image")
        source_file = str(source_image.get("file_name") or "") if isinstance(source_image, Mapping) else ""
        if not source_file:
            unresolved.append(f"{path.name}: missing source_image.file_name")
            continue

        matches = drawing_index.get(source_file, [])
        if len(matches) != 1:
            unresolved.append(f"{path.name}: exact drawing matches={len(matches)} for {source_file}")
            continue

        detail = matches[0]
        if present_values:
            span_m = numeric_values[0]
            if clear_span is not None and clear_span != "":
                source_field = "existing_span.clear_span_m"
            elif bridge_span is not None and bridge_span != "":
                source_field = "existing_bridge.span_m"
            else:
                source_field = "existing_dimensions.span_m"
        else:
            span = span_from_detail(detail)
            if span is None:
                unresolved.append(f"{path.name}: bridge {detail.get('id')} has no usable clear span")
                continue
            span_m, source_field = span
        plans.append(BackfillPlan(
            path=path,
            source_image_file=source_file,
            api_bridge_id=int(detail["id"]),
            api_bridge_name=str(detail.get("name") or ""),
            span_m=span_m,
            source_field=source_field,
        ))

    return plans, skipped, unresolved


def _updated_payload(path: Path, span_m: float) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    existing = _existing_span(payload)
    if existing is not None and abs(float(existing) - span_m) > 1e-9:
        raise ValueError(f"refusing to overwrite {path.name} span {existing} with {span_m}")
    bridge = payload.setdefault("bridge", {})
    span = payload.setdefault("span", {})
    if not isinstance(bridge, dict) or not isinstance(span, dict):
        raise ValueError(f"{path.name}: bridge and span must be objects")
    bridge["span_m"] = span_m
    span["clear_span_m"] = span_m
    return payload


def apply_plans(directory: Path, plans: list[BackfillPlan]) -> Path:
    if not plans:
        raise ValueError("there are no span backfills to apply")
    directory = directory.resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = directory.parent / f"{directory.name}_净跨补齐备份_{timestamp}"
    backup.mkdir(parents=False, exist_ok=False)

    all_json_paths = sorted(directory.glob("*.json"))
    for path in all_json_paths:
        shutil.copy2(path, backup / path.name)

    stage = Path(tempfile.mkdtemp(prefix=f".{directory.name}_span_backfill_", dir=directory.parent))
    try:
        for plan in plans:
            payload = _updated_payload(plan.path, plan.span_m)
            (stage / plan.path.name).write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        for staged in stage.glob("*.json"):
            staged.replace(directory / staged.name)
    finally:
        shutil.rmtree(stage, ignore_errors=True)

    with (backup / "span_backfill_audit.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=[
            "json_file", "source_image_file", "api_bridge_id", "api_bridge_name",
            "clear_span_m", "source_field", "detail_url",
        ])
        writer.writeheader()
        for plan in plans:
            writer.writerow({
                "json_file": plan.path.name,
                "source_image_file": plan.source_image_file,
                "api_bridge_id": plan.api_bridge_id,
                "api_bridge_name": plan.api_bridge_name,
                "clear_span_m": plan.span_m,
                "source_field": plan.source_field,
                "detail_url": plan.detail_url,
            })
    return backup


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Backfill and synchronize clear spans by exact source-drawing matches in the historical bridge database."
    )
    parser.add_argument("directory", type=Path, help="Directory containing annotation JSON files")
    parser.add_argument("--apply", action="store_true", help="Back up and rewrite matched files; default is dry-run")
    parser.add_argument("--api-root", default=DEFAULT_API_ROOT, help="Historical bridge database API root")
    parser.add_argument("--workers", type=int, default=12, help="Concurrent bridge-detail requests")
    args = parser.parse_args()

    details = fetch_bridge_details(api_root=args.api_root.rstrip("/"), workers=args.workers)
    plans, skipped, unresolved = plan_directory(args.directory, details)
    for plan in plans:
        print(
            f"SET {plan.path.name} | {plan.span_m:g} m | "
            f"bridge={plan.api_bridge_id}:{plan.api_bridge_name} | source={plan.source_field}"
        )
    for message in unresolved:
        print(f"UNRESOLVED {message}")
    print(
        f"matched={len(plans)} existing={len(skipped)} unresolved={len(unresolved)} "
        f"mode={'apply' if args.apply else 'dry-run'}"
    )
    if unresolved:
        raise SystemExit(2)
    if args.apply and plans:
        backup = apply_plans(args.directory, plans)
        print(f"backup={backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
