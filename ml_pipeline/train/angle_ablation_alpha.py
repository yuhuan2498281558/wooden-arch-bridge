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
    cross_validated_predictions,
    fit_design_mode_model,
    predict_design_mode_model,
    clip_to_design_mode,
)
from ml_pipeline.train.model_improvement import load_frozen_partition
from ml_pipeline.train.partial_pooling_alpha import _physical_error_metrics
from ml_pipeline.train.pilot import (
    MODEL_SELECTION_TOLERANCE,
    _sha256,
    _write_csv,
    _write_json,
    group_macro_mae,
    load_training_rows,
)


ARTIFACT_VERSION = "five-miao-node-research-v11-angle-ablation-alpha"
ARTIFACT_STATUS = "offline_research_not_for_deployment"
ANGLE_FIELD = "three_miao_design_chord_angle_deg"
RISE_RATIO_FIELD = "three_miao_rise_span_ratio"
FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "shared_span": ("span_m",),
    "shared_angle": (ANGLE_FIELD,),
    "shared_span_angle": ("span_m", ANGLE_FIELD),
    "shared_span_rise_ratio": ("span_m", RISE_RATIO_FIELD),
}
MODEL_NAMES = (
    "mode_median",
    *FEATURE_SETS.keys(),
    "v9_independent_experts",
)
PENALTIES = (0.0, 0.01, 0.1, 1.0, 10.0)


def _validate_feature_names(feature_names: tuple[str, ...]) -> None:
    if not feature_names:
        raise ValueError("at least one continuous feature is required")
    if ANGLE_FIELD in feature_names and RISE_RATIO_FIELD in feature_names:
        raise ValueError(
            "angle and rise/span ratio are deterministic transforms and must not "
            "be used together"
        )


def _feature_matrix(
    rows: list[dict[str, Any]], feature_names: tuple[str, ...]
) -> np.ndarray:
    _validate_feature_names(feature_names)
    values = np.asarray(
        [[float(row[name]) for name in feature_names] for row in rows],
        dtype=float,
    )
    if not np.all(np.isfinite(values)):
        raise ValueError("angle-ablation features must be finite")
    return values


def _design_matrix(
    rows: list[dict[str, Any]],
    classes: np.ndarray,
    *,
    feature_names: tuple[str, ...],
    feature_means: np.ndarray,
    feature_scales: np.ndarray,
) -> np.ndarray:
    values = _feature_matrix(rows, feature_names)
    standardized = (values - feature_means) / feature_scales
    mode_codes = np.asarray(classes, dtype=float)
    mode_contrast = np.where(mode_codes < 0, 0.0, mode_codes - 0.5)
    return np.column_stack(
        [np.ones(len(rows)), mode_contrast, standardized]
    )


def fit_shared_feature_model(
    rows: list[dict[str, Any]], spec: dict[str, Any]
) -> dict[str, Any]:
    if len(rows) < 4:
        raise ValueError("shared-feature model requires at least four rows")
    feature_names = tuple(spec["feature_names"])
    values = _feature_matrix(rows, feature_names)
    classes = _classes(rows)
    committed = classes[classes >= 0]
    if set(committed.tolist()) != {0, 1}:
        raise ValueError("shared-feature model requires both design modes")
    means = np.mean(values, axis=0)
    scales = np.std(values, axis=0)
    scales = np.where(scales > 1e-12, scales, 1.0)
    design = _design_matrix(
        rows,
        classes,
        feature_names=feature_names,
        feature_means=means,
        feature_scales=scales,
    )
    penalty_value = float(spec["penalty"])
    penalty = np.asarray([0.0, 0.0, *([penalty_value] * len(feature_names))])
    coefficients = np.linalg.pinv(
        design.T @ design + np.diag(penalty)
    ) @ (design.T @ _target(rows))
    return {
        "model_name": str(spec["model_name"]),
        "feature_names": list(feature_names),
        "penalty": penalty_value,
        "feature_means": means,
        "feature_scales": scales,
        "coefficients": coefficients,
    }


def predict_shared_feature_model(
    model: dict[str, Any],
    rows: list[dict[str, Any]],
    modes: Iterable[int],
) -> np.ndarray:
    classes = np.asarray(list(modes), dtype=int)
    if len(classes) != len(rows) or np.any((classes < 0) | (classes > 1)):
        raise ValueError("design modes must align with rows and contain only 0/1")
    design = _design_matrix(
        rows,
        classes,
        feature_names=tuple(model["feature_names"]),
        feature_means=np.asarray(model["feature_means"], dtype=float),
        feature_scales=np.asarray(model["feature_scales"], dtype=float),
    )
    prediction = design @ np.asarray(model["coefficients"], dtype=float)
    return clip_to_design_mode(prediction, classes)


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


def cross_validated_feature_predictions(
    rows: list[dict[str, Any]],
    spec: dict[str, Any],
    *,
    splitter: GroupKFold | LeaveOneGroupOut,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    classes = _classes(rows)
    groups = np.asarray([row["split_group_key"] for row in rows])
    prediction = np.full(len(rows), np.nan, dtype=float)
    audit: list[dict[str, Any]] = []
    for fold, (train_index, test_index) in enumerate(
        _split_indices(rows, splitter), start=1
    ):
        training = [rows[index] for index in train_index]
        validation = [rows[index] for index in test_index]
        model = fit_shared_feature_model(training, spec)
        prediction[test_index] = predict_shared_feature_model(
            model, validation, classes[test_index]
        )
        audit.append({
            "fold": fold,
            "training_groups": len(set(groups[train_index])),
            "validation_groups": len(set(groups[test_index])),
            "group_overlap": 0,
        })
    if not np.all(np.isfinite(prediction)):
        raise RuntimeError("angle ablation produced incomplete predictions")
    return prediction, audit


def _screen_candidate(
    rows: list[dict[str, Any]], spec: dict[str, Any]
) -> dict[str, Any]:
    try:
        prediction, _ = cross_validated_feature_predictions(
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
        if not np.any(selected):
            return {**spec, "valid": False, "failure": "missing_committed_mode"}
        mode_scores.append(
            group_macro_mae(
                observed[selected], prediction[selected], bridge_keys[selected]
            )
        )
    errors = np.abs(observed - prediction)
    return {
        **spec,
        "valid": True,
        "bridge_macro_mae": group_macro_mae(
            observed, prediction, bridge_keys
        ),
        "mode_macro_bridge_mae": float(np.mean(mode_scores)),
        "q90": float(np.quantile(errors, 0.9)),
    }


def select_penalty(records: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [record for record in records if record.get("valid")]
    if not valid:
        raise RuntimeError("no valid angle-ablation candidates")
    best = min(float(record["mode_macro_bridge_mae"]) for record in valid)
    eligible = [
        record
        for record in valid
        if float(record["mode_macro_bridge_mae"])
        <= best + MODEL_SELECTION_TOLERANCE
    ]
    eligible.sort(
        key=lambda record: (
            -float(record["penalty"]),
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


def _screen_feature_set(
    rows: list[dict[str, Any]], model_name: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    records = [
        _screen_candidate(
            rows,
            {
                "model_name": model_name,
                "feature_names": list(FEATURE_SETS[model_name]),
                "penalty": penalty,
            },
        )
        for penalty in PENALTIES
    ]
    return select_penalty(records), records


def choose_angle_candidate(metrics: dict[str, dict[str, Any]]) -> str:
    names = ("shared_angle", "shared_span_angle")
    best = min(float(metrics[name]["mode_macro_bridge_mae"]) for name in names)
    eligible = [
        name
        for name in names
        if float(metrics[name]["mode_macro_bridge_mae"])
        <= best + MODEL_SELECTION_TOLERANCE
    ]
    eligible.sort(
        key=lambda name: (
            len(FEATURE_SETS[name]),
            float(metrics[name]["mode_macro_bridge_mae"]),
            float(metrics[name]["absolute_error_quantiles"]["q90"]),
        )
    )
    return eligible[0]


def _median_specs() -> dict[str, dict[str, Any]]:
    spec = {
        "model_name": "median",
        "feature_set": "none",
        "feature_names": [],
        "params": {},
    }
    return {mode_name: dict(spec) for mode_name in MODE_NAMES}


def _angle_equivalence(rows: list[dict[str, Any]]) -> dict[str, float]:
    angle = np.asarray([float(row[ANGLE_FIELD]) for row in rows])
    ratio = np.asarray([float(row[RISE_RATIO_FIELD]) for row in rows])
    reconstructed = np.degrees(np.arctan(3.0 * ratio))
    return {
        "max_reconstruction_error_deg": float(
            np.max(np.abs(angle - reconstructed))
        ),
        "pearson_angle_rise_ratio": float(np.corrcoef(angle, ratio)[0, 1]),
        "angle_min_deg": float(np.min(angle)),
        "angle_max_deg": float(np.max(angle)),
    }


def paired_group_comparison(
    rows: list[dict[str, Any]],
    candidate: np.ndarray,
    baseline: np.ndarray,
    *,
    bootstrap_iterations: int,
    random_seed: int,
) -> dict[str, Any]:
    observed = _target(rows)
    candidate_error = np.abs(observed - candidate)
    baseline_error = np.abs(observed - baseline)
    bridge_keys = np.asarray([row["bridge_key"] for row in rows])
    unique_bridges = np.unique(bridge_keys)
    differences = np.asarray([
        float(np.mean(candidate_error[bridge_keys == bridge_key]))
        - float(np.mean(baseline_error[bridge_keys == bridge_key]))
        for bridge_key in unique_bridges
    ])
    rng = np.random.default_rng(random_seed)
    bootstrap = rng.choice(
        differences,
        size=(bootstrap_iterations, len(differences)),
        replace=True,
    ).mean(axis=1)
    return {
        "difference_definition": "candidate_bridge_MAE_minus_shared_span_bridge_MAE",
        "bridge_groups": len(unique_bridges),
        "mean_difference": float(np.mean(differences)),
        "median_difference": float(np.median(differences)),
        "candidate_better_groups": int(np.sum(differences < 0.0)),
        "candidate_worse_groups": int(np.sum(differences > 0.0)),
        "equal_groups": int(np.sum(differences == 0.0)),
        "bootstrap_ci95": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
    }


def _prediction_rows(
    rows: list[dict[str, Any]], predictions: dict[str, np.ndarray]
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
            "rise_span_ratio": float(row[RISE_RATIO_FIELD]),
            "chord_angle_deg": float(row[ANGLE_FIELD]),
            "observed_alpha": float(observed[index]),
        }
        for name, values in predictions.items():
            record[f"predicted_{name}"] = float(values[index])
            record[f"absolute_error_{name}"] = float(
                abs(observed[index] - values[index])
            )
        result.append(record)
    return result


def _plot(
    development_metrics: dict[str, dict[str, Any]],
    holdout_metrics: dict[str, dict[str, Any]],
    path: Path,
) -> None:
    labels = [
        "Median",
        "Span",
        "Angle",
        "Span+angle",
        "Span+rise/span",
        "v9 independent",
    ]
    x = np.arange(len(MODEL_NAMES))
    width = 0.36
    figure, axis = plt.subplots(figsize=(11, 5.5))
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
    axis.set_xticks(x, labels, rotation=16, ha="right")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _report(
    selected_specs: dict[str, Any],
    development_metrics: dict[str, dict[str, Any]],
    holdout_metrics: dict[str, dict[str, Any]],
    angle_candidate: str,
    gate: dict[str, Any],
    equivalence: dict[str, float],
    paired_comparisons: dict[str, dict[str, Any]],
) -> str:
    lines = [
        "# v11 显式模式 alpha 倾斜角消融实验",
        "",
        "倾斜角按 `atan(3 × rise/span)` 生成，与矢跨比是一一对应的确定性变换。因此本实验比较两种表示，但禁止将角度与矢跨比放进同一个模型。历史模式仍由 alpha=0.5 回推，只用于设计模式已知条件下的离线研究。",
        "",
        "## 开发集 LOGO（用于结论）",
        "",
        "| 模型 | 特征 | 桥宏 MAE | 模式宏 MAE | q90 | 外节点物理 MAE/m |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for name in MODEL_NAMES:
        metric = development_metrics[name]
        features = selected_specs.get(name, {}).get("feature_names", [])
        feature_text = "+".join(features) if features else "分模式中位数/独立专家"
        lines.append(
            f"| {name} | {feature_text} | {metric['bridge_macro_mae']:.5f} | "
            f"{metric['mode_macro_bridge_mae']:.5f} | "
            f"{metric['absolute_error_quantiles']['q90']:.5f} | "
            f"{metric['physical_outer_node_error_m']['bridge_macro_mae_m']:.3f} |"
        )
    lines.extend([
        "",
        f"角度候选按 0.002 容差选择：`{angle_candidate}`。",
        f"相对共享净跨与 v9 的替换门槛：{'通过' if gate['passed'] else '未通过'}。",
    ])
    if gate["reasons"]:
        lines.append("原因：" + "、".join(gate["reasons"]) + "。")
    lines.extend([
        "",
        "## 历史留出集（仅描述）",
        "",
        "| 模型 | 桥宏 MAE | 模式宏 MAE | q90 |",
        "|---|---:|---:|---:|",
    ])
    for name in MODEL_NAMES:
        metric = holdout_metrics[name]
        lines.append(
            f"| {name} | {metric['bridge_macro_mae']:.5f} | "
            f"{metric['mode_macro_bridge_mae']:.5f} | "
            f"{metric['absolute_error_quantiles']['q90']:.5f} |"
        )
    lines.extend([
        "",
        "## 角度字段核验",
        "",
        f"- 角度范围：`{equivalence['angle_min_deg']:.3f}°—{equivalence['angle_max_deg']:.3f}°`。",
        f"- 由矢跨比重建角度的最大误差：`{equivalence['max_reconstruction_error_deg']:.3e}°`。",
        f"- 角度与矢跨比 Pearson 相关：`{equivalence['pearson_angle_rise_ratio']:.6f}`。",
        "",
        "## 相对共享净跨的桥级配对差异",
        "",
        "正值表示候选误差更大。",
        "",
        "| 候选 | 平均差值 | 95% bootstrap CI | 候选更优桥组/总桥组 |",
        "|---|---:|---:|---:|",
    ])
    for name, record in paired_comparisons.items():
        interval = record["bootstrap_ci95"]
        lines.append(
            f"| {name} | {record['mean_difference']:.5f} | "
            f"[{interval[0]:.5f}, {interval[1]:.5f}] | "
            f"{record['candidate_better_groups']}/{record['bridge_groups']} |"
        )
    lines.extend([
        "",
        "冻结历史留出已经反复查看，不用于选模。只有开发集分组 LOGO 与未来全新外部桥组都稳定通过门槛，才允许讨论替换线上模型。",
        "",
    ])
    return "\n".join(lines)


def run_angle_ablation_alpha(
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

    selected_specs: dict[str, Any] = {}
    screening: dict[str, Any] = {}
    for name in FEATURE_SETS:
        selected_specs[name], screening[name] = _screen_feature_set(
            development, name
        )
    median_specs = _median_specs()
    independent_specs, independent_screen = _screen_specs(development)
    selected_specs["mode_median"] = median_specs
    selected_specs["v9_independent_experts"] = independent_specs
    screening["v9_independent_experts"] = independent_screen

    development_predictions: dict[str, np.ndarray] = {}
    fold_audit: dict[str, Any] = {}
    development_predictions["mode_median"], fold_audit["mode_median"] = (
        cross_validated_predictions(
            development, median_specs, splitter=LeaveOneGroupOut()
        )
    )
    for name in FEATURE_SETS:
        development_predictions[name], fold_audit[name] = (
            cross_validated_feature_predictions(
                development,
                selected_specs[name],
                splitter=LeaveOneGroupOut(),
            )
        )
    (
        development_predictions["v9_independent_experts"],
        fold_audit["v9_independent_experts"],
    ) = cross_validated_predictions(
        development, independent_specs, splitter=LeaveOneGroupOut()
    )
    development_predictions = {
        name: development_predictions[name] for name in MODEL_NAMES
    }
    development_metrics = {
        name: _metrics(
            development,
            prediction,
            bootstrap_iterations=bootstrap_iterations,
            random_seed=random_seed + index * 10,
        )
        for index, (name, prediction) in enumerate(
            development_predictions.items()
        )
    }
    for name in MODEL_NAMES:
        development_metrics[name]["physical_outer_node_error_m"] = (
            _physical_error_metrics(development, development_predictions[name])
        )

    angle_candidate = choose_angle_candidate(development_metrics)
    gate_passed, gate_reasons, comparator = _passes_integration_gate(
        development_metrics[angle_candidate],
        {
            "shared_span": development_metrics["shared_span"],
            "v9_independent_experts": development_metrics[
                "v9_independent_experts"
            ],
        },
    )
    gate = {
        "candidate": angle_candidate,
        "passed": gate_passed,
        "reasons": gate_reasons,
        "strongest_comparator": comparator,
    }

    holdout_classes = _classes(holdout)
    holdout_predictions: dict[str, np.ndarray] = {}
    for name, specs in (
        ("mode_median", median_specs),
        ("v9_independent_experts", independent_specs),
    ):
        model = fit_design_mode_model(development, specs)
        holdout_predictions[name] = predict_design_mode_model(
            model, holdout, holdout_classes
        )
    for name in FEATURE_SETS:
        model = fit_shared_feature_model(development, selected_specs[name])
        holdout_predictions[name] = predict_shared_feature_model(
            model, holdout, holdout_classes
        )
    holdout_predictions = {
        name: holdout_predictions[name] for name in MODEL_NAMES
    }
    holdout_metrics = {
        name: _metrics(
            holdout,
            prediction,
            bootstrap_iterations=bootstrap_iterations,
            random_seed=random_seed + 100 + index * 10,
        )
        for index, (name, prediction) in enumerate(holdout_predictions.items())
    }
    for name in MODEL_NAMES:
        holdout_metrics[name]["physical_outer_node_error_m"] = (
            _physical_error_metrics(holdout, holdout_predictions[name])
        )

    equivalence = _angle_equivalence(accepted)
    paired_comparisons = {
        name: paired_group_comparison(
            development,
            development_predictions[name],
            development_predictions["shared_span"],
            bootstrap_iterations=bootstrap_iterations,
            random_seed=random_seed + 300 + index,
        )
        for index, name in enumerate(
            ("shared_angle", "shared_span_angle", "shared_span_rise_ratio")
        )
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
        "angle_candidate": angle_candidate,
        "integration_gate": gate,
        "feature_policy": {
            "angle_formula": "degrees(atan(3 * rise_span_ratio))",
            "angle_and_rise_ratio_together_forbidden": True,
        },
        "validation": {
            "screening": "GroupKFold(5) grouped by split_group_key",
            "confirmation": "LeaveOneGroupOut grouped by split_group_key",
            "score": "equal-weight design-mode bridge-macro MAE",
            "holdout_used_for_selection": False,
        },
        "bootstrap_iterations": bootstrap_iterations,
        "random_seed": random_seed,
        "no_deployment_artifact_written": True,
    }
    _write_json(output_dir / "selected_models.json", selected_specs)
    _write_json(output_dir / "screening_candidates.json", screening)
    _write_json(output_dir / "development_metrics.json", development_metrics)
    _write_json(output_dir / "independent_holdout_metrics.json", holdout_metrics)
    _write_json(output_dir / "angle_equivalence.json", equivalence)
    _write_json(output_dir / "paired_group_comparisons.json", paired_comparisons)
    _write_json(output_dir / "angle_integration_gate.json", gate)
    _write_json(output_dir / "fold_audit.json", fold_audit)
    _write_json(output_dir / "training_manifest.json", manifest)
    _write_csv(
        output_dir / "development_predictions.csv",
        _prediction_rows(development, development_predictions),
    )
    _write_csv(
        output_dir / "independent_holdout_predictions.csv",
        _prediction_rows(holdout, holdout_predictions),
    )
    _plot(
        development_metrics,
        holdout_metrics,
        output_dir / "angle_ablation_comparison.png",
    )
    (output_dir / "angle_ablation_report.md").write_text(
        _report(
            selected_specs,
            development_metrics,
            holdout_metrics,
            angle_candidate,
            gate,
            equivalence,
            paired_comparisons,
        ),
        encoding="utf-8",
    )
    return {
        "output_dir": str(output_dir),
        "accepted_rows": len(accepted),
        "development_rows": len(development),
        "independent_holdout_rows": len(holdout),
        "angle_candidate": angle_candidate,
        "angle_gate": gate,
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
        description="Run explicit-mode chord-angle ablation for alpha."
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
    summary = run_angle_ablation_alpha(
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
