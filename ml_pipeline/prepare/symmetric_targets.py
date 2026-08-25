from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

from bridge_algorithm_service.node_geometry import measure_ratios


SUPPORTED_SCHEMA = "five-miao-node-annotation-v1"
POINT_NAMES = ("A", "B", "C", "D", "Q1", "Q2", "Q3", "Q4")
ASYMMETRY_DIAGNOSTIC_THRESHOLD = 0.03
ZERO_INNER_FLAT_CHORD_RATIO_THRESHOLD = 0.01
TARGET_METHOD = "parallel-angle-fixed-left-right-mean-v2"
RATIO_DEFINITION = "bridge-axis-parallel-chord-v2"
ASYMMETRY_PREPROCESSING = "left-right-mean-for-symmetric-design-v1"
STRUCTURE_SCOPE_POLICY = "exclude-zero-inner-flat-chord-v1"
THREE_MIAO_PROJECTION_POLICY = "three-equal-longitudinal-projections-v1"


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _source_stem(payload: Mapping[str, Any], annotation_path: Path) -> str:
    source_image = payload.get("source_image")
    if isinstance(source_image, Mapping):
        file_name = str(source_image.get("file_name") or "").strip()
        if file_name:
            return Path(file_name).stem
    stem = annotation_path.stem
    stem = re.sub(r"_five_miao_nodes(?:\s*\(\d+\))?$", "", stem, flags=re.IGNORECASE)
    return re.sub(r"_span_\d+_of_\d+$", "", stem, flags=re.IGNORECASE)


def _ratio_delta(stored: Mapping[str, Any], calculated: Mapping[str, float]) -> float | str:
    deltas: list[float] = []
    for key, expected in calculated.items():
        try:
            deltas.append(abs(float(stored[key]) - expected))
        except (KeyError, TypeError, ValueError):
            continue
    return max(deltas) if deltas else ""


def _optional_positive_float(value: Any, name: str) -> float | str:
    if value in (None, ""):
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive number") from exc
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be a positive number")
    return number


def _positive_int(value: Any, name: str, default: int = 1) -> int:
    if value in (None, ""):
        return default
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if not math.isfinite(number) or number < 1 or not number.is_integer():
        raise ValueError(f"{name} must be a positive integer")
    return int(number)


def _first(*values: Any) -> Any:
    """Return the first value that is neither None nor an empty string.

    ``dict.get(key, default)`` only falls back when the key is missing; a key
    present with a ``None`` value (hand-edited or migrated annotations) must
    still fall through the bridge/dimensions chain.
    """
    for value in values:
        if value is not None and value != "":
            return value
    return None


def prepare_annotation(annotation_path: Path) -> dict[str, Any]:
    with annotation_path.open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    payload = _mapping(payload, "annotation")

    schema_version = str(payload.get("schema_version") or "")
    if schema_version != SUPPORTED_SCHEMA:
        raise ValueError(f"unsupported schema_version: {schema_version!r}")

    annotation = _mapping(payload.get("annotation"), "annotation.annotation")
    raw_points = _mapping(annotation.get("points"), "annotation.points")
    missing = [name for name in POINT_NAMES if name not in raw_points]
    if missing:
        raise ValueError(f"missing points: {', '.join(missing)}")

    points = {name: _mapping(raw_points[name], f"annotation.points.{name}") for name in POINT_NAMES}
    measured = measure_ratios(
        points["A"], points["B"], points["C"], points["D"],
        points["Q1"], points["Q2"], points["Q3"], points["Q4"],
    )

    bridge = payload.get("bridge") if isinstance(payload.get("bridge"), Mapping) else {}
    source_image = payload.get("source_image") if isinstance(payload.get("source_image"), Mapping) else {}
    source_stem = _source_stem(payload, annotation_path)
    bridge_id = str(bridge.get("bridge_id") or "").strip()
    bridge_name = str(bridge.get("bridge_name") or "").strip() or source_stem
    dimensions = payload.get("dimensions") if isinstance(payload.get("dimensions"), Mapping) else {}
    span = payload.get("span") if isinstance(payload.get("span"), Mapping) else {}
    span_count = _positive_int(_first(span.get("count"), bridge.get("span_count")), "span.count")
    span_index = _positive_int(_first(span.get("index"), annotation.get("span_index")), "span.index")
    if span_index > span_count:
        raise ValueError("span.index must not exceed span.count")
    span_m = _optional_positive_float(
        _first(span.get("clear_span_m"), bridge.get("span_m"), dimensions.get("span_m")),
        "span.clear_span_m",
    )

    middle_dx = float(points["C"]["x"]) - float(points["B"]["x"])
    middle_dy = float(points["C"]["y"]) - float(points["B"]["y"])
    middle_axis_angle_deg = math.degrees(math.atan2(middle_dy, middle_dx))
    middle_length_px = math.hypot(middle_dx, middle_dy)
    if middle_length_px <= 0:
        raise ValueError("B and C must define a non-zero bridge axis")
    axis_ux = middle_dx / middle_length_px
    axis_uy = middle_dy / middle_length_px
    normal_up_x, normal_up_y = axis_uy, -axis_ux
    span_dx = float(points["D"]["x"]) - float(points["A"]["x"])
    span_dy = float(points["D"]["y"]) - float(points["A"]["y"])
    longitudinal_span_px = abs(span_dx * axis_ux + span_dy * axis_uy)
    if longitudinal_span_px <= 0:
        raise ValueError("A and D must define a non-zero longitudinal span")
    left_rise_px = (
        (float(points["B"]["x"]) - float(points["A"]["x"])) * normal_up_x
        + (float(points["B"]["y"]) - float(points["A"]["y"])) * normal_up_y
    )
    right_rise_px = (
        (float(points["C"]["x"]) - float(points["D"]["x"])) * normal_up_x
        + (float(points["C"]["y"]) - float(points["D"]["y"])) * normal_up_y
    )
    three_miao_rise_span_ratio = ((left_rise_px + right_rise_px) / 2.0) / longitudinal_span_px
    if not math.isfinite(three_miao_rise_span_ratio) or three_miao_rise_span_ratio <= 0:
        raise ValueError("three-miao rise/span ratio must be positive")
    # Under the design constraint, each inclined chord projection and the
    # middle flat chord occupy one third of the clear span.  The chord angle is
    # therefore a deterministic transform of f/L and must not be entered as an
    # additional independent model feature.
    three_miao_design_chord_angle_deg = math.degrees(math.atan(3.0 * three_miao_rise_span_ratio))
    three_miao_rise_m = (
        float(span_m) * three_miao_rise_span_ratio if span_m != "" else ""
    )
    span_center_distance = abs(2.0 * (0.5 if span_count == 1 else (span_index - 1) / (span_count - 1)) - 1.0)

    # Existing bridges may be genuinely asymmetric because of terrain, pier
    # settlement, member dimensions, construction and survey error.  This is
    # an observed historical property, not evidence of a bad annotation.  Keep
    # it for analysis, but symmetrize the design target with the left/right
    # mean and do not exclude the row from training for asymmetry alone.
    asymmetry_flags: list[str] = []
    if measured.alpha_symmetry_error > ASYMMETRY_DIAGNOSTIC_THRESHOLD:
        asymmetry_flags.append("alpha_asymmetry")
    if measured.beta_symmetry_error > ASYMMETRY_DIAGNOSTIC_THRESHOLD:
        asymmetry_flags.append("beta_asymmetry")

    # Q2 and Q3 delimit the horizontal inner chord.  Their projected distance
    # along B--C is independent of drawing rotation and small normal-direction
    # click offsets.  A near-zero value represents a different construction in
    # which the two inclined chords meet directly; retain the observation, but
    # keep it outside the standard five-miao training population.
    inner_flat_chord_ratio = abs(1.0 - measured.beta_left - measured.beta_right)
    has_zero_inner_flat_chord = inner_flat_chord_ratio <= ZERO_INNER_FLAT_CHORD_RATIO_THRESHOLD
    scope_exclusion_reasons = ["zero_inner_flat_chord"] if has_zero_inner_flat_chord else []

    # Structured drawing defects recorded by the annotator.  Any listed defect
    # is a substantive data-quality problem: the sample stays traceable in the
    # prepared data but must not enter standard five-miao training (the pilot
    # loader turns has_defects into a training exclusion reason).
    raw_defects = annotation.get("defects", [])
    if isinstance(raw_defects, str):
        raw_defects = [item.strip() for item in raw_defects.split(";") if item.strip()]
    defects = [str(item).strip() for item in raw_defects if str(item).strip()]
    has_defects = bool(defects)

    review_reasons: list[str] = []
    for warning in measured.warnings:
        if "member-axis offset" not in warning:
            if has_zero_inner_flat_chord and warning == "beta must be between 0 and 0.5":
                continue
            review_reasons.append(warning)

    calculated_ratios = {
        "alpha_left": measured.alpha_left,
        "alpha_right": measured.alpha_right,
        "beta_left": measured.beta_left,
        "beta_right": measured.beta_right,
        "alpha": measured.alpha,
        "beta": measured.beta,
    }
    stored_ratios = annotation.get("ratios") if isinstance(annotation.get("ratios"), Mapping) else {}

    # The source image is a safer fallback than a manually retained bridge
    # name, because sequential annotation sessions may contain a stale name.
    bridge_key = bridge_id or source_stem or bridge_name
    # Blank or whitespace-only source_group_id must fall back to bridge_key,
    # otherwise every such row collapses into a single empty group and the
    # leave-one-group-out split silently loses its meaning.
    source_group_id = str(bridge.get("source_group_id") or "").strip() or bridge_key
    sample_key = f"{bridge_key}:span-{span_index:02d}-of-{span_count:02d}"

    row: dict[str, Any] = {
        "schema_version": schema_version,
        "annotation_file": annotation_path.name,
        "source_image_file": str(source_image.get("file_name") or ""),
        "bridge_key": bridge_key,
        "source_group_id": source_group_id,
        "split_group_key": source_group_id,
        "sample_key": sample_key,
        "bridge_id": bridge_id,
        "bridge_name": bridge_name,
        "span_count": span_count,
        "span_index": span_index,
        "span_position_normalized": 0.5 if span_count == 1 else (span_index - 1) / (span_count - 1),
        "span_center_distance": span_center_distance,
        "is_edge_span": span_index in {1, span_count},
        "span_m": span_m,
        "drawing_url": str(bridge.get("drawing_url") or ""),
        "created_at": str(payload.get("created_at") or ""),
        "annotator": str(annotation.get("annotator") or ""),
        "quality": str(annotation.get("quality") or ""),
        "defects": ";".join(defects),
        "has_defects": has_defects,
        "notes": str(annotation.get("notes") or ""),
        "node_definition": str(annotation.get("node_definition") or ""),
        "ratio_definition": RATIO_DEFINITION,
        "observed_alpha_left": measured.alpha_left,
        "observed_alpha_right": measured.alpha_right,
        "legacy_projected_alpha_left": measured.alpha_legacy_projection_left,
        "legacy_projected_alpha_right": measured.alpha_legacy_projection_right,
        "observed_beta_left": measured.beta_left,
        "observed_beta_right": measured.beta_right,
        "design_target_alpha": measured.alpha,
        "design_target_beta": measured.beta,
        "design_target_method": TARGET_METHOD,
        "design_target_is_symmetric": True,
        "alpha_asymmetry": measured.alpha_symmetry_error,
        "beta_asymmetry": measured.beta_symmetry_error,
        "has_observed_asymmetry": bool(asymmetry_flags),
        "observed_asymmetry_flags": ";".join(asymmetry_flags),
        "asymmetry_preprocessing": ASYMMETRY_PREPROCESSING,
        "inner_flat_chord_ratio": inner_flat_chord_ratio,
        "has_zero_inner_flat_chord": has_zero_inner_flat_chord,
        "structure_scope_policy": STRUCTURE_SCOPE_POLICY,
        "training_scope_excluded": bool(scope_exclusion_reasons),
        "scope_exclusion_reasons": ";".join(scope_exclusion_reasons),
        "middle_axis_angle_deg": middle_axis_angle_deg,
        "three_miao_projection_policy": THREE_MIAO_PROJECTION_POLICY,
        "three_miao_rise_span_ratio": three_miao_rise_span_ratio,
        "three_miao_rise_m": three_miao_rise_m,
        "three_miao_design_chord_angle_deg": three_miao_design_chord_angle_deg,
        "outer_start_offset_left_span": measured.outer_start_offset_left_normalized,
        "outer_start_offset_right_span": measured.outer_start_offset_right_normalized,
        "outer_start_offset_mean_span": (
            measured.outer_start_offset_left_normalized + measured.outer_start_offset_right_normalized
        ) / 2.0,
        # This is retained as an observed normal offset, not used as a v2 target
        # or as an automatic rejection criterion when member thickness is ignored.
        "max_normalized_projection_offset": measured.max_normalized_projection_error,
        "ratio_recalculation_max_delta": _ratio_delta(stored_ratios, calculated_ratios),
        "needs_review": bool(review_reasons),
        "review_reasons": ";".join(review_reasons),
    }
    for name in POINT_NAMES:
        row[f"{name}_x_px"] = float(points[name]["x"])
        row[f"{name}_y_px"] = float(points[name]["y"])
    return row


def collect_annotation_paths(inputs: Iterable[Path], input_dirs: Iterable[Path], pattern: str) -> list[Path]:
    candidates = list(inputs)
    for directory in input_dirs:
        candidates.extend(directory.glob(pattern))
    unique = {path.resolve(): path.resolve() for path in candidates if path.is_file()}
    return sorted(unique.values(), key=lambda path: path.name)


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    if not rows:
        raise ValueError("no annotation rows to write")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert raw left/right node annotations into symmetric design targets."
    )
    parser.add_argument("inputs", nargs="*", type=Path, help="Annotation JSON files")
    parser.add_argument("--input-dir", action="append", default=[], type=Path, help="Directory to scan")
    parser.add_argument("--pattern", default="*_five_miao_nodes.json", help="Glob used with --input-dir")
    parser.add_argument("--output", required=True, type=Path, help="Output CSV path")
    args = parser.parse_args()

    paths = collect_annotation_paths(args.inputs, args.input_dir, args.pattern)
    if not paths:
        parser.error("no annotation JSON files found")
    rows = [prepare_annotation(path) for path in paths]
    write_csv(rows, args.output)

    summary = {
        "rows": len(rows),
        "needs_review": sum(bool(row["needs_review"]) for row in rows),
        "training_scope_excluded": sum(bool(row["training_scope_excluded"]) for row in rows),
        "observed_asymmetry": sum(bool(row["has_observed_asymmetry"]) for row in rows),
        "output": str(args.output.resolve()),
        "target_method": TARGET_METHOD,
    }
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
