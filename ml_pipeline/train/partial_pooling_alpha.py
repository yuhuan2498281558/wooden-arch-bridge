from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut

from ml_pipeline.prepare.symmetric_targets import collect_annotation_paths
from ml_pipeline.train.design_mode_alpha import (
    MODE_NAMES,
    _classes,
    _metrics,
    _passes_integration_gate,
    _screen_specs,
    _target,
    clip_to_design_mode,
    cross_validated_predictions,
    fit_design_mode_model,
    predict_design_mode_model,
)
from ml_pipeline.train.model_improvement import load_frozen_partition
from ml_pipeline.train.pilot import (
    MODEL_SELECTION_TOLERANCE,
    _sha256,
    _write_csv,
    _write_json,
    group_macro_mae,
    load_training_rows,
)


ARTIFACT_VERSION = "five-miao-node-research-v10-partial-pooling-alpha"
ARTIFACT_STATUS = "offline_research_not_for_deployment"
MODEL_NAMES = (
    "mode_median",
    "shared_slope",
    "partial_pooling",
    "v9_independent_experts",
)
SHARED_PENALTIES = (0.0, 0.01, 0.1, 1.0, 10.0)
INTERACTION_PENALTIES = (0.1, 1.0, 10.0, 100.0, 1000.0)
TAIL_ERROR_TOLERANCE = 0.005


def _span(rows: Iterable[dict[str, Any]]) -> np.ndarray:
    return np.asarray([float(row["span_m"]) for row in rows], dtype=float)


def _joint_design(
    rows: list[dict[str, Any]],
    classes: np.ndarray,
    *,
    span_mean: float,
    span_scale: float,
    interaction: bool,
) -> np.ndarray:
    z_span = (_span(rows) - span_mean) / span_scale
    # Symmetric contrast coding makes the third coefficient the common slope
    # and the fourth coefficient the front/back slope difference.
    mode_contrast = np.asarray(classes, dtype=float) - 0.5
    columns = [np.ones(len(rows)), mode_contrast, z_span]
    if interaction:
        columns.append(mode_contrast * z_span)
    return np.column_stack(columns)


def fit_joint_model(
    rows: list[dict[str, Any]],
    spec: dict[str, Any],
) -> dict[str, Any]:
    if len(rows) < 4:
        raise ValueError("joint model requires at least four training rows")
    classes = _classes(rows)
    if set(classes.tolist()) != {0, 1}:
        raise ValueError("joint model requires both design modes")
    span_values = _span(rows)
    span_mean = float(np.mean(span_values))
    span_scale = float(np.std(span_values))
    if not np.isfinite(span_scale) or span_scale <= 1e-12:
        span_scale = 1.0
    interaction = str(spec["model_name"]) == "partial_pooling"
    design = _joint_design(
        rows,
        classes,
        span_mean=span_mean,
        span_scale=span_scale,
        interaction=interaction,
    )
    penalty = np.zeros(design.shape[1], dtype=float)
    penalty[2] = float(spec.get("shared_penalty", 0.0))
    if interaction:
        penalty[3] = float(spec["interaction_penalty"])
    normal = design.T @ design + np.diag(penalty)
    right = design.T @ _target(rows)
    coefficients = np.linalg.pinv(normal) @ right
    return {
        "model_name": str(spec["model_name"]),
        "shared_penalty": float(spec.get("shared_penalty", 0.0)),
        "interaction_penalty": (
            float(spec["interaction_penalty"]) if interaction else None
        ),
        "span_mean": span_mean,
        "span_scale": span_scale,
        "coefficients": coefficients,
    }


def predict_joint_model(
    model: dict[str, Any],
    rows: list[dict[str, Any]],
    modes: Iterable[int],
) -> np.ndarray:
    classes = np.asarray(list(modes), dtype=int)
    if len(classes) != len(rows) or np.any((classes < 0) | (classes > 1)):
        raise ValueError("design modes must align with rows and contain only 0/1")
    interaction = str(model["model_name"]) == "partial_pooling"
    design = _joint_design(
        rows,
        classes,
        span_mean=float(model["span_mean"]),
        span_scale=float(model["span_scale"]),
        interaction=interaction,
    )
    raw = design @ np.asarray(model["coefficients"], dtype=float)
    return clip_to_design_mode(raw, classes)


def _split_indices(
    rows: list[dict[str, Any]],
    splitter: GroupKFold | LeaveOneGroupOut,
) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    groups = np.asarray([row["split_group_key"] for row in rows])
    placeholder = np.zeros((len(rows), 1), dtype=float)
    for train_index, test_index in splitter.split(placeholder, groups=groups):
        if set(groups[train_index]) & set(groups[test_index]):
            raise AssertionError("split_group_key leakage detected")
        yield train_index, test_index


def cross_validated_joint_predictions(
    rows: list[dict[str, Any]],
    spec: dict[str, Any],
    *,
    splitter: GroupKFold | LeaveOneGroupOut,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    classes = _classes(rows)
    groups = np.asarray([row["split_group_key"] for row in rows])
    predictions = np.full(len(rows), np.nan, dtype=float)
    audit: list[dict[str, Any]] = []
    for fold, (train_index, test_index) in enumerate(
        _split_indices(rows, splitter), start=1
    ):
        training = [rows[index] for index in train_index]
        validation = [rows[index] for index in test_index]
        model = fit_joint_model(training, spec)
        predictions[test_index] = predict_joint_model(
            model, validation, classes[test_index]
        )
        audit.append({
            "fold": fold,
            "training_groups": len(set(groups[train_index])),
            "validation_groups": len(set(groups[test_index])),
            "group_overlap": 0,
        })
    if not np.all(np.isfinite(predictions)):
        raise RuntimeError("joint cross-validation produced incomplete predictions")
    return predictions, audit


def _candidate_specs(model_name: str) -> list[dict[str, Any]]:
    if model_name == "shared_slope":
        return [
            {"model_name": model_name, "shared_penalty": value}
            for value in SHARED_PENALTIES
        ]
    if model_name == "partial_pooling":
        return [
            {
                "model_name": model_name,
                "shared_penalty": shared,
                "interaction_penalty": interaction,
            }
            for shared in SHARED_PENALTIES
            for interaction in INTERACTION_PENALTIES
            if interaction >= max(shared, 0.1)
        ]
    raise ValueError(f"unknown joint model: {model_name}")


def _screen_joint_candidate(
    rows: list[dict[str, Any]],
    spec: dict[str, Any],
) -> dict[str, Any]:
    try:
        predictions, _ = cross_validated_joint_predictions(
            rows, spec, splitter=GroupKFold(n_splits=5)
        )
    except (ValueError, FloatingPointError) as exc:
        return {**spec, "valid": False, "failure": str(exc)}
    observed = _target(rows)
    classes = _classes(rows)
    bridge_keys = np.asarray([row["bridge_key"] for row in rows])
    mode_scores = []
    for mode_index in range(2):
        selected = classes == mode_index
        mode_scores.append(
            group_macro_mae(
                observed[selected], predictions[selected], bridge_keys[selected]
            )
        )
    errors = np.abs(observed - predictions)
    return {
        **spec,
        "valid": True,
        "bridge_macro_mae": group_macro_mae(observed, predictions, bridge_keys),
        "mode_macro_bridge_mae": float(np.mean(mode_scores)),
        "q90": float(np.quantile(errors, 0.9)),
    }


def select_joint_spec(records: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [record for record in records if record.get("valid")]
    if not valid:
        raise RuntimeError("no valid joint-model candidates")
    best = min(float(record["mode_macro_bridge_mae"]) for record in valid)
    eligible = [
        record
        for record in valid
        if float(record["mode_macro_bridge_mae"])
        <= best + MODEL_SELECTION_TOLERANCE
    ]
    # Within the pre-declared tolerance, prefer more shrinkage and then the
    # lower tail error. This is the small-sample bias/variance trade-off.
    eligible.sort(
        key=lambda record: (
            -float(record.get("interaction_penalty", 0.0)),
            -float(record.get("shared_penalty", 0.0)),
            float(record["q90"]),
            float(record["mode_macro_bridge_mae"]),
        )
    )
    return {
        key: value
        for key, value in eligible[0].items()
        if key
        not in {
            "valid",
            "failure",
            "bridge_macro_mae",
            "mode_macro_bridge_mae",
            "q90",
        }
    }


def _screen_joint_model(
    rows: list[dict[str, Any]], model_name: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    records = [
        _screen_joint_candidate(rows, spec)
        for spec in _candidate_specs(model_name)
    ]
    return select_joint_spec(records), records


def _median_specs() -> dict[str, dict[str, Any]]:
    spec = {
        "model_name": "median",
        "feature_set": "none",
        "feature_names": [],
        "params": {},
    }
    return {mode_name: dict(spec) for mode_name in MODE_NAMES}


def choose_confirmed_model(metrics: dict[str, dict[str, Any]]) -> str:
    best = min(
        float(metrics[name]["mode_macro_bridge_mae"]) for name in MODEL_NAMES
    )
    eligible = [
        name
        for name in MODEL_NAMES
        if float(metrics[name]["mode_macro_bridge_mae"])
        <= best + MODEL_SELECTION_TOLERANCE
    ]
    return eligible[0]


def _physical_error_metrics(
    rows: list[dict[str, Any]], predictions: np.ndarray
) -> dict[str, float]:
    span = _span(rows)
    rise_ratio = np.asarray(
        [float(row["three_miao_rise_span_ratio"]) for row in rows], dtype=float
    )
    diagonal = np.sqrt((span / 3.0) ** 2 + (span * rise_ratio) ** 2)
    error_m = np.abs(_target(rows) - predictions) * diagonal
    bridge_keys = np.asarray([row["bridge_key"] for row in rows])
    return {
        "mae_m": float(np.mean(error_m)),
        "bridge_macro_mae_m": group_macro_mae(
            np.zeros(len(rows)), error_m, bridge_keys
        ),
        "q90_m": float(np.quantile(error_m, 0.9)),
        "max_m": float(np.max(error_m)),
    }


def grouped_conformal_summary(
    development_rows: list[dict[str, Any]],
    development_predictions: np.ndarray,
    holdout_rows: list[dict[str, Any]],
    holdout_predictions: np.ndarray,
    *,
    coverage: float = 0.9,
) -> dict[str, Any]:
    errors = np.abs(_target(development_rows) - development_predictions)
    groups = np.asarray([row["split_group_key"] for row in development_rows])
    group_scores = np.asarray(
        [float(np.max(errors[groups == group])) for group in np.unique(groups)]
    )
    rank = int(np.ceil((len(group_scores) + 1) * coverage))
    rank = min(max(rank, 1), len(group_scores))
    radius = float(np.sort(group_scores)[rank - 1])
    holdout_classes = _classes(holdout_rows)
    lower = clip_to_design_mode(holdout_predictions - radius, holdout_classes)
    upper = clip_to_design_mode(holdout_predictions + radius, holdout_classes)
    observed = _target(holdout_rows)
    contained = (observed >= lower) & (observed <= upper)
    holdout_groups = np.asarray(
        [row["split_group_key"] for row in holdout_rows]
    )
    group_covered = [
        bool(np.all(contained[holdout_groups == group]))
        for group in np.unique(holdout_groups)
    ]
    return {
        "method": "development_LOGO_group_max_residual",
        "nominal_coverage": coverage,
        "calibration_groups": len(group_scores),
        "radius_alpha": radius,
        "holdout_row_coverage": float(np.mean(contained)),
        "holdout_group_simultaneous_coverage": float(np.mean(group_covered)),
        "holdout_mean_interval_width_alpha": float(np.mean(upper - lower)),
        "holdout_lower": lower,
        "holdout_upper": upper,
    }


def _prediction_rows(
    rows: list[dict[str, Any]],
    predictions: dict[str, np.ndarray],
    conformal: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    classes = _classes(rows)
    observed = _target(rows)
    result: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        record: dict[str, Any] = {
            "sample_key": row["sample_key"],
            "bridge_name": row["bridge_name"],
            "bridge_key": row["bridge_key"],
            "split_group_key": row["split_group_key"],
            "design_mode": MODE_NAMES[int(classes[index])],
            "span_m": float(row["span_m"]),
            "observed_alpha": float(observed[index]),
        }
        for name, values in predictions.items():
            record[f"predicted_{name}"] = float(values[index])
            record[f"absolute_error_{name}"] = float(
                abs(observed[index] - values[index])
            )
            if name in conformal and "holdout_lower" in conformal[name]:
                record[f"lower90_{name}"] = float(
                    conformal[name]["holdout_lower"][index]
                )
                record[f"upper90_{name}"] = float(
                    conformal[name]["holdout_upper"][index]
                )
        result.append(record)
    return result


def _plot_comparison(
    development_metrics: dict[str, dict[str, Any]],
    holdout_metrics: dict[str, dict[str, Any]],
    path: Path,
) -> None:
    labels = ["M0 median", "M1 shared", "M2 partial", "M3 independent"]
    x = np.arange(len(MODEL_NAMES))
    width = 0.36
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.bar(
        x - width / 2,
        [development_metrics[name]["mode_macro_bridge_mae"] for name in MODEL_NAMES],
        width,
        label="Development LOGO",
    )
    axis.bar(
        x + width / 2,
        [holdout_metrics[name]["mode_macro_bridge_mae"] for name in MODEL_NAMES],
        width,
        label="Historical holdout (descriptive)",
    )
    axis.set_ylabel("Mode-macro bridge MAE (alpha)")
    axis.set_xticks(x, labels, rotation=12, ha="right")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _report(
    selected: dict[str, Any],
    development_metrics: dict[str, dict[str, Any]],
    holdout_metrics: dict[str, dict[str, Any]],
    chosen: str,
    partial_gate: dict[str, Any],
    conformal: dict[str, dict[str, Any]],
) -> str:
    lines = [
        "# v10 显式模式部分共享 alpha 离线实验",
        "",
        "本实验只研究斜弦外节点比例 alpha；beta、线上服务和现有模型均未修改。历史样本的模式由 alpha=0.5 阈值回推，仅用于检验‘设计人员已明确给出模式’的条件预测场景。",
        "",
        "## 模型",
        "",
        "- M0：前/后半模式各自训练中位数。",
        "- M1：模式截距不同，但两种模式共用同一跨径斜率。",
        "- M2：模式截距不同，斜率允许不同，但模式斜率差受到额外惩罚（部分共享）。",
        "- M3：现有 v9 两个完全独立专家。",
        "",
        "## 开发集 LOGO 结果（用于结论）",
        "",
        "| 模型 | 桥宏 MAE | 模式宏 MAE | q90 | 外节点物理 MAE/m |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in MODEL_NAMES:
        metric = development_metrics[name]
        lines.append(
            f"| {name} | {metric['bridge_macro_mae']:.5f} | "
            f"{metric['mode_macro_bridge_mae']:.5f} | "
            f"{metric['absolute_error_quantiles']['q90']:.5f} | "
            f"{metric['physical_outer_node_error_m']['bridge_macro_mae_m']:.3f} |"
        )
    lines.extend([
        "",
        f"按 0.002 预设容差并优先简单模型，开发集确认选择：`{chosen}`。",
        f"部分共享相对 M0/M3 的替换门槛：{'通过' if partial_gate['passed'] else '未通过'}。",
    ])
    if partial_gate["reasons"]:
        lines.append("原因：" + "、".join(partial_gate["reasons"]) + "。")
    lines.extend([
        "",
        "## 历史独立留出集（仅描述，不再用于选择）",
        "",
        "| 模型 | 桥宏 MAE | 模式宏 MAE | q90 | 90%区间行覆盖率 |",
        "|---|---:|---:|---:|---:|",
    ])
    for name in MODEL_NAMES:
        metric = holdout_metrics[name]
        lines.append(
            f"| {name} | {metric['bridge_macro_mae']:.5f} | "
            f"{metric['mode_macro_bridge_mae']:.5f} | "
            f"{metric['absolute_error_quantiles']['q90']:.5f} | "
            f"{conformal[name]['holdout_row_coverage']:.3f} |"
        )
    lines.extend([
        "",
        "## 选择参数",
        "",
        "```json",
        json.dumps(selected, ensure_ascii=False, indent=2),
        "```",
        "",
        "结论必须以开发集分组 LOGO 为主。该 21 行历史留出集已被多轮实验反复查看，因此只能做方向性描述；正式替换仍需新增、从未查看过的桥梁/测绘谱系外部组。",
        "",
    ])
    return "\n".join(lines)


def run_partial_pooling_alpha(
    input_dir: Path,
    output_dir: Path,
    *,
    partition_file: Path,
    pattern: str = "*_five_miao_nodes.json",
    bootstrap_iterations: int = 5000,
    random_seed: int = 20260823,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory must be empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = collect_annotation_paths([], [input_dir], pattern)
    all_rows, accepted = load_training_rows(paths)
    development, holdout = load_frozen_partition(accepted, partition_file)

    shared_spec, shared_screen = _screen_joint_model(development, "shared_slope")
    partial_spec, partial_screen = _screen_joint_model(
        development, "partial_pooling"
    )
    independent_specs, independent_screen = _screen_specs(development)
    median_specs = _median_specs()
    selected: dict[str, Any] = {
        "mode_median": median_specs,
        "shared_slope": shared_spec,
        "partial_pooling": partial_spec,
        "v9_independent_experts": independent_specs,
    }

    development_predictions: dict[str, np.ndarray] = {}
    fold_audit: dict[str, Any] = {}
    development_predictions["mode_median"], fold_audit["mode_median"] = (
        cross_validated_predictions(
            development, median_specs, splitter=LeaveOneGroupOut()
        )
    )
    for name, spec in (
        ("shared_slope", shared_spec),
        ("partial_pooling", partial_spec),
    ):
        development_predictions[name], fold_audit[name] = (
            cross_validated_joint_predictions(
                development, spec, splitter=LeaveOneGroupOut()
            )
        )
    (
        development_predictions["v9_independent_experts"],
        fold_audit["v9_independent_experts"],
    ) = cross_validated_predictions(
        development, independent_specs, splitter=LeaveOneGroupOut()
    )

    development_metrics = {
        name: _metrics(
            development,
            values,
            bootstrap_iterations=bootstrap_iterations,
            random_seed=random_seed + index * 10,
        )
        for index, (name, values) in enumerate(development_predictions.items())
    }
    for name in MODEL_NAMES:
        development_metrics[name]["physical_outer_node_error_m"] = (
            _physical_error_metrics(development, development_predictions[name])
        )
    chosen = choose_confirmed_model(development_metrics)
    partial_passed, partial_reasons, comparator = _passes_integration_gate(
        development_metrics["partial_pooling"],
        {
            "mode_median": development_metrics["mode_median"],
            "v9_independent_experts": development_metrics[
                "v9_independent_experts"
            ],
        },
    )
    partial_gate = {
        "passed": partial_passed,
        "reasons": partial_reasons,
        "strongest_comparator": comparator,
    }

    holdout_predictions: dict[str, np.ndarray] = {}
    holdout_classes = _classes(holdout)
    for name, specs in (
        ("mode_median", median_specs),
        ("v9_independent_experts", independent_specs),
    ):
        model = fit_design_mode_model(development, specs)
        holdout_predictions[name] = predict_design_mode_model(
            model, holdout, holdout_classes
        )
    for name, spec in (
        ("shared_slope", shared_spec),
        ("partial_pooling", partial_spec),
    ):
        model = fit_joint_model(development, spec)
        holdout_predictions[name] = predict_joint_model(
            model, holdout, holdout_classes
        )
    holdout_predictions = {
        name: holdout_predictions[name] for name in MODEL_NAMES
    }
    holdout_metrics = {
        name: _metrics(
            holdout,
            values,
            bootstrap_iterations=bootstrap_iterations,
            random_seed=random_seed + 100 + index * 10,
        )
        for index, (name, values) in enumerate(holdout_predictions.items())
    }
    for name in MODEL_NAMES:
        holdout_metrics[name]["physical_outer_node_error_m"] = (
            _physical_error_metrics(holdout, holdout_predictions[name])
        )

    conformal = {
        name: grouped_conformal_summary(
            development,
            development_predictions[name],
            holdout,
            holdout_predictions[name],
        )
        for name in MODEL_NAMES
    }
    trained_at = datetime.now(timezone.utc).isoformat()
    manifest = {
        "artifact_version": ARTIFACT_VERSION,
        "status": ARTIFACT_STATUS,
        "trained_at": trained_at,
        "input_directory": str(input_dir),
        "output_directory": str(output_dir),
        "partition_source": str(partition_file),
        "partition_sha256": _sha256(partition_file),
        "source_files": [
            {"name": path.name, "sha256": _sha256(path)} for path in paths
        ],
        "source_rows": len(all_rows),
        "accepted_rows": len(accepted),
        "development_rows": len(development),
        "independent_holdout_rows": len(holdout),
        "selected_model_by_development": chosen,
        "partial_pooling_integration_gate": partial_gate,
        "validation": {
            "screening": "GroupKFold(5) grouped by split_group_key",
            "confirmation": "LeaveOneGroupOut grouped by split_group_key",
            "score": "equal-weight design-mode bridge-macro MAE",
            "holdout_used_for_selection": False,
            "historical_mode_source": "design_target_alpha threshold at 0.5",
            "deployment_contract": "designer must supply front_half/back_half",
        },
        "bootstrap_iterations": bootstrap_iterations,
        "random_seed": random_seed,
        "no_deployment_artifact_written": True,
    }
    screening = {
        "shared_slope": shared_screen,
        "partial_pooling": partial_screen,
        "v9_independent_experts": independent_screen,
    }
    conformal_json = {
        name: {
            key: value
            for key, value in record.items()
            if key not in {"holdout_lower", "holdout_upper"}
        }
        for name, record in conformal.items()
    }
    _write_json(output_dir / "selected_model.json", selected)
    _write_json(output_dir / "screening_candidates.json", screening)
    _write_json(output_dir / "development_metrics.json", development_metrics)
    _write_json(output_dir / "independent_holdout_metrics.json", holdout_metrics)
    _write_json(output_dir / "partial_pooling_gate.json", partial_gate)
    _write_json(output_dir / "grouped_conformal.json", conformal_json)
    _write_json(output_dir / "fold_audit.json", fold_audit)
    _write_json(output_dir / "training_manifest.json", manifest)
    _write_csv(
        output_dir / "development_predictions.csv",
        _prediction_rows(development, development_predictions, {}),
    )
    _write_csv(
        output_dir / "independent_holdout_predictions.csv",
        _prediction_rows(holdout, holdout_predictions, conformal),
    )
    _plot_comparison(
        development_metrics,
        holdout_metrics,
        output_dir / "partial_pooling_comparison.png",
    )
    (output_dir / "partial_pooling_report.md").write_text(
        _report(
            selected,
            development_metrics,
            holdout_metrics,
            chosen,
            partial_gate,
            conformal,
        ),
        encoding="utf-8",
    )
    return {
        "output_dir": str(output_dir),
        "accepted_rows": len(accepted),
        "development_rows": len(development),
        "independent_holdout_rows": len(holdout),
        "selected_model": chosen,
        "partial_pooling_gate": partial_gate,
        "development_mode_macro_mae": {
            name: development_metrics[name]["mode_macro_bridge_mae"]
            for name in MODEL_NAMES
        },
        "holdout_mode_macro_mae_descriptive": {
            name: holdout_metrics[name]["mode_macro_bridge_mae"]
            for name in MODEL_NAMES
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate explicit-mode partial pooling for alpha."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--partition-file", type=Path, required=True)
    parser.add_argument("--pattern", default="*_five_miao_nodes.json")
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--random-seed", type=int, default=20260823)
    return parser


def main() -> None:
    args = _parser().parse_args()
    summary = run_partial_pooling_alpha(
        args.input_dir,
        args.output_dir,
        partition_file=args.partition_file,
        pattern=args.pattern,
        bootstrap_iterations=args.bootstrap_iterations,
        random_seed=args.random_seed,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
