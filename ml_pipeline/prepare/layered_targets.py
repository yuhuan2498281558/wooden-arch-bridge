from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

from ml_pipeline.prepare.symmetric_targets import (
    collect_annotation_paths,
    prepare_annotation,
)


TARGET_LAYER_VERSION = "five-miao-target-layers-v1"
HISTORICAL_PROXY_METHOD = "parallel-angle-fixed-left-right-mean-v2"
NORMATIVE_STATUS_VALUES = {"pending", "approved", "rejected"}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("no rows to write")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_historical_reviews(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        records = list(csv.DictReader(handle))
    result: dict[str, dict[str, str]] = {}
    for record in records:
        sample_key = str(record.get("sample_key") or "").strip()
        if not sample_key:
            raise ValueError("historical review row has no sample_key")
        if sample_key in result:
            raise ValueError(f"duplicate historical review sample_key: {sample_key}")
        result[sample_key] = {str(key): str(value or "") for key, value in record.items()}
    return result


def _normative_value(value: Any, target: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"approved {target} must be numeric") from exc
    upper = 1.0 if target == "alpha" else 0.5
    if not math.isfinite(number) or not 0.0 < number < upper:
        raise ValueError(f"approved {target} must be between 0 and {upper}")
    return number


def load_normative_overrides(
    path: Path | None,
    *,
    expected_sample_keys: set[str] | None = None,
) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        records = list(csv.DictReader(handle))
    result: dict[str, dict[str, Any]] = {}
    for record in records:
        sample_key = str(record.get("sample_key") or "").strip()
        if not sample_key:
            raise ValueError("normative override row has no sample_key")
        if sample_key in result:
            raise ValueError(f"duplicate normative override sample_key: {sample_key}")
        if expected_sample_keys is not None and sample_key not in expected_sample_keys:
            raise ValueError(f"unknown normative override sample_key: {sample_key}")
        status = str(record.get("normative_design_status") or "pending").strip().lower()
        if status not in NORMATIVE_STATUS_VALUES:
            raise ValueError(f"invalid normative_design_status for {sample_key}: {status}")
        alpha: float | str = ""
        beta: float | str = ""
        if status == "approved":
            alpha = _normative_value(record.get("normative_design_alpha"), "alpha")
            beta = _normative_value(record.get("normative_design_beta"), "beta")
            if not str(record.get("reviewer") or "").strip():
                raise ValueError(f"approved normative target requires reviewer: {sample_key}")
            if not str(record.get("evidence") or "").strip():
                raise ValueError(f"approved normative target requires evidence: {sample_key}")
        result[sample_key] = {
            "normative_design_status": status,
            "normative_design_alpha": alpha,
            "normative_design_beta": beta,
            "normative_design_reviewer": str(record.get("reviewer") or "").strip(),
            "normative_design_evidence": str(record.get("evidence") or "").strip(),
            "normative_design_notes": str(record.get("notes") or "").strip(),
        }
    return result


def build_layered_rows(
    rows: Iterable[dict[str, Any]],
    *,
    historical_reviews: dict[str, dict[str, str]] | None = None,
    normative_overrides: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    reviews = historical_reviews or {}
    overrides = normative_overrides or {}
    output: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        sample_key = str(row["sample_key"])
        review = reviews.get(sample_key, {})
        review_status = str(review.get("review_status") or "").strip()
        confirmed_historical = review_status == "confirmed_valid_historical_observation"
        override = overrides.get(sample_key, {})
        normative_status = str(override.get("normative_design_status") or "pending")
        normative_approved = normative_status == "approved"
        try:
            span_m = float(row.get("span_m"))
        except (TypeError, ValueError):
            span_m = float("nan")
        proxy_eligible = not any([
            bool(row.get("needs_review")),
            str(row.get("quality") or "") == "low",
            bool(row.get("has_defects")),
            bool(row.get("training_scope_excluded")),
            not math.isfinite(span_m) or span_m <= 0,
        ])
        normative_eligible = normative_approved and proxy_eligible
        row.update({
            "target_layer_version": TARGET_LAYER_VERSION,
            "historical_observation_alpha": float(row["design_target_alpha"]),
            "historical_observation_beta": float(row["design_target_beta"]),
            "historical_observation_method": HISTORICAL_PROXY_METHOD,
            "historical_observation_is_symmetric_proxy": True,
            "historical_review_status": review_status or "not_targeted_for_high_error_review",
            "historical_variation_status": (
                "confirmed_valid_historical_variation"
                if confirmed_historical
                else "not_individually_attributed"
            ),
            "historical_variation_cause": (
                "multiple_possible_causes_not_attributed"
                if confirmed_historical
                else ""
            ),
            "historical_proxy_training_eligible": proxy_eligible,
            "historical_proxy_warning": "historical_proxy_not_expert_normative_design",
            "normative_design_alpha": override.get("normative_design_alpha", ""),
            "normative_design_beta": override.get("normative_design_beta", ""),
            "normative_design_status": normative_status,
            "normative_design_reviewer": override.get("normative_design_reviewer", ""),
            "normative_design_evidence": override.get("normative_design_evidence", ""),
            "normative_design_notes": override.get("normative_design_notes", ""),
            "normative_design_training_eligible": normative_eligible,
            "normative_design_exclusion_reason": (
                ""
                if normative_eligible
                else ("base_geometry_or_quality_ineligible" if normative_approved else "awaiting_expert_normative_target")
            ),
        })
        output.append(row)
    return output


def normative_review_template(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    template: list[dict[str, Any]] = []
    for row in rows:
        template.append({
            "sample_key": row["sample_key"],
            "bridge_name": row["bridge_name"],
            "span_index": row["span_index"],
            "span_count": row["span_count"],
            "span_m": row["span_m"],
            "historical_observation_alpha": row["historical_observation_alpha"],
            "historical_observation_beta": row["historical_observation_beta"],
            "historical_review_status": row["historical_review_status"],
            "normative_design_status": row["normative_design_status"],
            "normative_design_alpha": row["normative_design_alpha"],
            "normative_design_beta": row["normative_design_beta"],
            "reviewer": row["normative_design_reviewer"],
            "evidence": row["normative_design_evidence"],
            "notes": row["normative_design_notes"],
        })
    return template


def run_layered_preprocessing(
    input_dir: Path,
    output_dir: Path,
    *,
    historical_review_file: Path | None = None,
    normative_override_file: Path | None = None,
    pattern: str = "*_five_miao_nodes.json",
) -> dict[str, Any]:
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = collect_annotation_paths([], [input_dir], pattern)
    if not paths:
        raise ValueError(f"no annotation files matched {pattern!r} in {input_dir}")
    base_rows = [prepare_annotation(path) for path in paths]
    reviews = load_historical_reviews(historical_review_file)
    expected_keys = {str(row["sample_key"]) for row in base_rows}
    unknown_reviews = sorted(set(reviews) - expected_keys)
    if unknown_reviews:
        raise ValueError(f"historical review contains unknown sample_key: {unknown_reviews[:5]}")
    overrides = load_normative_overrides(
        normative_override_file,
        expected_sample_keys=expected_keys,
    )
    layered = build_layered_rows(
        base_rows,
        historical_reviews=reviews,
        normative_overrides=overrides,
    )
    _write_csv(output_dir / "layered_targets.csv", layered)
    _write_csv(output_dir / "normative_target_review_template.csv", normative_review_template(layered))
    manifest = {
        "target_layer_version": TARGET_LAYER_VERSION,
        "status": "normative_targets_pending_expert_review",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_directory": str(input_dir),
        "output_directory": str(output_dir),
        "source_rows": len(layered),
        "confirmed_historical_variation_rows": sum(
            row["historical_variation_status"] == "confirmed_valid_historical_variation"
            for row in layered
        ),
        "normative_approved_rows": sum(bool(row["normative_design_training_eligible"]) for row in layered),
        "normative_pending_rows": sum(not bool(row["normative_design_training_eligible"]) for row in layered),
        "source_files": [{"name": path.name, "sha256": _sha256(path)} for path in paths],
        "historical_review_file": str(historical_review_file.resolve()) if historical_review_file else None,
        "historical_review_file_sha256": _sha256(historical_review_file) if historical_review_file else None,
        "normative_override_file": str(normative_override_file.resolve()) if normative_override_file else None,
        "normative_override_file_sha256": _sha256(normative_override_file) if normative_override_file else None,
        "training_contract": {
            "historical_proxy": "allowed_for_research_baseline_only",
            "normative_design": "requires approved target, reviewer and evidence",
            "high_residual_policy": "retain_confirmed_historical_observations",
        },
    }
    (output_dir / "layered_target_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Build historical/proxy/normative target layers.")
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--historical-review-file", type=Path)
    parser.add_argument("--normative-override-file", type=Path)
    parser.add_argument("--pattern", default="*_five_miao_nodes.json")
    args = parser.parse_args()
    result = run_layered_preprocessing(
        args.input_dir,
        args.output_dir,
        historical_review_file=args.historical_review_file,
        normative_override_file=args.normative_override_file,
        pattern=args.pattern,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
