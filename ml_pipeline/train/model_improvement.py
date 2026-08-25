from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from ml_pipeline.prepare.symmetric_targets import collect_annotation_paths
from ml_pipeline.train.feature_ablation import FEATURE_SETS, split_development_holdout
from ml_pipeline.train.pilot import (
    MODEL_SELECTION_TOLERANCE,
    MODEL_SIMPLICITY_ORDER,
    TARGETS,
    _sha256,
    _write_csv,
    _write_json,
    feature_matrix,
    fit_artifact,
    load_training_rows,
    metric_record,
    nested_group_evaluation,
    predict_artifact,
)


ARTIFACT_VERSION = "five-miao-node-pilot-v6-small-sample-models"
MODEL_CANDIDATES = (
    "ridge",
    "huber",
    "elastic_net",
    "spline_ridge",
    "svr",
    "gpr",
    "extra_trees",
)
BASELINE_CANDIDATES = {
    "alpha": ("span_only", "ridge"),
    "beta": ("span_rise_ratio", "ridge"),
}
TAIL_ERROR_TOLERANCE = 0.005


def load_frozen_partition(
    rows: list[dict[str, Any]],
    partition_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    with partition_path.open(newline="", encoding="utf-8-sig") as handle:
        records = list(csv.DictReader(handle))
    assignments = {record["sample_key"]: record["partition"] for record in records}
    expected = {row["sample_key"] for row in rows}
    missing = sorted(expected - assignments.keys())
    unexpected = sorted(assignments.keys() - expected)
    if missing or unexpected:
        raise ValueError(
            "frozen partition does not match accepted samples: "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )
    invalid = sorted({value for value in assignments.values() if value not in {"development", "independent_holdout"}})
    if invalid:
        raise ValueError(f"invalid frozen partition values: {invalid}")
    development = [row for row in rows if assignments[row["sample_key"]] == "development"]
    holdout = [row for row in rows if assignments[row["sample_key"]] == "independent_holdout"]
    development_groups = {row["split_group_key"] for row in development}
    holdout_groups = {row["split_group_key"] for row in holdout}
    if development_groups & holdout_groups:
        raise ValueError("frozen development and holdout split groups overlap")
    if not development or not holdout:
        raise ValueError("frozen partition must contain development and independent_holdout rows")
    return development, holdout


def choose_screening_candidate(
    metrics: dict[str, dict[str, Any]],
    target_name: str,
) -> tuple[str, str]:
    candidates: list[tuple[str, str, float]] = []
    for feature_set, target_metrics in metrics.items():
        for model_name in MODEL_CANDIDATES:
            record = target_metrics[target_name][model_name]
            candidates.append((feature_set, model_name, float(record["bridge_macro_mae"])))
    best_score = min(score for _, _, score in candidates)
    eligible = [item for item in candidates if item[2] <= best_score + MODEL_SELECTION_TOLERANCE]
    model_rank = {name: index for index, name in enumerate(MODEL_SIMPLICITY_ORDER)}
    eligible.sort(key=lambda item: (model_rank[item[1]], len(FEATURE_SETS[item[0]]), item[2], item[0]))
    feature_set, model_name, _ = eligible[0]
    return feature_set, model_name


def passes_replacement_gate(
    candidate: dict[str, Any],
    baseline: dict[str, Any],
    *,
    mae_margin: float = MODEL_SELECTION_TOLERANCE,
    tail_tolerance: float = TAIL_ERROR_TOLERANCE,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    candidate_mae = float(candidate["bridge_macro_mae"])
    baseline_mae = float(baseline["bridge_macro_mae"])
    if candidate_mae > baseline_mae - mae_margin:
        reasons.append("bridge_macro_mae_not_materially_better")
    candidate_q90 = float(candidate["absolute_error_quantiles"]["q90"])
    baseline_q90 = float(baseline["absolute_error_quantiles"]["q90"])
    if candidate_q90 > baseline_q90 + tail_tolerance:
        reasons.append("q90_tail_error_worse")
    return not reasons, reasons


def _evaluate_pair(
    rows: list[dict[str, Any]],
    feature_set: str,
    model_name: str,
    *,
    bootstrap_iterations: int,
    random_seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    metrics, oof_rows, tuning = nested_group_evaluation(
        rows,
        feature_names=FEATURE_SETS[feature_set],
        model_names=(model_name,),
        outer_splits=None,
        inner_splits=5,
        bootstrap_iterations=bootstrap_iterations,
        random_seed=random_seed,
    )
    return metrics, oof_rows, tuning


def _residual_rows(
    selected_oof: dict[str, list[dict[str, Any]]],
    selection: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    combined: dict[str, dict[str, Any]] = {}
    for target_name, rows in selected_oof.items():
        model_name = selection[target_name]["model_name"]
        for row in rows:
            output = combined.setdefault(
                row["sample_key"],
                {
                    "sample_key": row["sample_key"],
                    "bridge_key": row["bridge_key"],
                    "bridge_name": row["bridge_name"],
                    "split_group_key": row["split_group_key"],
                    "span_m": row["span_m"],
                },
            )
            output[f"observed_{target_name}"] = row[f"observed_{target_name}"]
            output[f"predicted_{target_name}"] = row[f"predicted_{target_name}_{model_name}"]
            output[f"absolute_error_{target_name}"] = row[f"absolute_error_{target_name}_{model_name}"]
    return list(combined.values())


def _report(
    *,
    source_rows: int,
    training_rows: int,
    development_rows: int,
    holdout_rows: int,
    selection: dict[str, dict[str, Any]],
    holdout_metrics: dict[str, dict[str, Any]],
    baseline_holdout_metrics: dict[str, dict[str, Any]],
) -> str:
    lines = [
        "# 五节苗节点 v6 小样本算法改进报告",
        "",
        "> 状态：Pilot，不得直接作为工程定型模型。模型选择未使用独立内部留出集。",
        "",
        "## 数据与验证",
        "",
        f"- 原始标注：{source_rows} 条；有效训练记录：{training_rows} 条。",
        f"- 冻结开发集：{development_rows} 条；冻结独立内部留出：{holdout_rows} 条。",
        "- 第一阶段：开发集 5 折外层、3 折内层分组筛选。",
        "- 第二阶段：候选与 v5 Ridge 基线均执行开发集 Leave-One-Group-Out 确认。",
        "- 替换门槛：桥级宏 MAE 至少改善 0.002，且样本绝对误差 q90 不恶化超过 0.005。",
        "",
        "## 选择结果",
        "",
    ]
    for target_name, record in selection.items():
        holdout = holdout_metrics[target_name]
        baseline = baseline_holdout_metrics[target_name]
        lines.extend([
            f"### {target_name}",
            "",
            f"- 最终模型：`{record['model_name']}` + `{record['feature_set']}`。",
            f"- 开发集 LOGO 桥级宏 MAE：`{record['development_bridge_macro_mae']:.5f}`。",
            f"- 独立内部留出桥级宏 MAE：`{holdout['bridge_macro_mae']:.5f}`；v5 Ridge 对照为 `{baseline['bridge_macro_mae']:.5f}`。",
            f"- 独立内部留出 q90：`{holdout['absolute_error_quantiles']['q90']:.5f}`；最大误差 `{holdout['max_absolute_error']:.5f}`。",
            f"- 选择结论：`{record['decision']}`。",
            "",
        ])
    lines.extend([
        "## 边界",
        "",
        "- 留出数据仍与训练数据来自同一网站资料，不属于真正外部验证。",
        "- 当前没有桥宽、细桥型和构件数量等完整结构化特征，复杂模型无法弥补缺失信息。",
        "- 产物保留规则回退、硬范围检查和局部训练密度提示。",
        "",
    ])
    return "\n".join(lines)


def run_model_improvement(
    input_dir: Path,
    output_dir: Path,
    *,
    partition_file: Path | None = None,
    pattern: str = "*_five_miao_nodes.json",
    bootstrap_iterations: int = 5000,
    random_seed: int = 20260813,
) -> dict[str, Any]:
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = collect_annotation_paths([], [input_dir], pattern)
    if not paths:
        raise ValueError(f"no annotation files matched {pattern!r} in {input_dir}")
    all_rows, accepted = load_training_rows(paths)
    if partition_file is not None:
        partition_file = partition_file.resolve()
        development_rows, holdout_rows = load_frozen_partition(accepted, partition_file)
        partition_source = str(partition_file)
    else:
        development_rows, holdout_rows = split_development_holdout(
            accepted, holdout_fraction=0.2, random_seed=random_seed
        )
        partition_source = "deterministic_group_shuffle_split"

    screening_metrics: dict[str, dict[str, Any]] = {}
    screening_tuning: dict[str, Any] = {}
    for feature_set, feature_names in FEATURE_SETS.items():
        metrics, oof_rows, tuning = nested_group_evaluation(
            development_rows,
            feature_names=feature_names,
            model_names=MODEL_CANDIDATES,
            outer_splits=5,
            inner_splits=3,
            bootstrap_iterations=min(bootstrap_iterations, 1000),
            random_seed=random_seed,
        )
        screening_metrics[feature_set] = metrics
        screening_tuning[feature_set] = tuning
        _write_csv(output_dir / f"screening_oof_{feature_set}.csv", oof_rows)

    screened = {target: choose_screening_candidate(screening_metrics, target) for target in TARGETS}
    pairs = sorted(set(screened.values()) | set(BASELINE_CANDIDATES.values()))
    pair_results: dict[tuple[str, str], tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]] = {}
    for index, (feature_set, model_name) in enumerate(pairs):
        pair_results[(feature_set, model_name)] = _evaluate_pair(
            development_rows,
            feature_set,
            model_name,
            bootstrap_iterations=bootstrap_iterations,
            random_seed=random_seed + index,
        )

    confirmation_metrics: dict[str, Any] = {}
    selection: dict[str, dict[str, Any]] = {}
    selected_oof: dict[str, list[dict[str, Any]]] = {}
    for target_name in TARGETS:
        candidate_pair = screened[target_name]
        baseline_pair = BASELINE_CANDIDATES[target_name]
        candidate_metrics = pair_results[candidate_pair][0][target_name][candidate_pair[1]]
        baseline_metrics = pair_results[baseline_pair][0][target_name][baseline_pair[1]]
        if candidate_pair == baseline_pair:
            passed, reasons = False, ["screening_retained_v5_baseline"]
            selected_pair = baseline_pair
            decision = "retained_v5_baseline"
        else:
            passed, reasons = passes_replacement_gate(candidate_metrics, baseline_metrics)
            selected_pair = candidate_pair if passed else baseline_pair
            decision = "candidate_replaced_baseline" if passed else "candidate_rejected_by_gate"
        selected_metrics = pair_results[selected_pair][0][target_name][selected_pair[1]]
        selected_oof[target_name] = pair_results[selected_pair][1]
        confirmation_metrics[target_name] = {
            "screened_candidate": {
                "feature_set": candidate_pair[0],
                "model_name": candidate_pair[1],
                "metrics": candidate_metrics,
            },
            "v5_baseline": {
                "feature_set": baseline_pair[0],
                "model_name": baseline_pair[1],
                "metrics": baseline_metrics,
            },
            "selected": {
                "feature_set": selected_pair[0],
                "model_name": selected_pair[1],
                "metrics": selected_metrics,
            },
        }
        selection[target_name] = {
            "feature_set": selected_pair[0],
            "feature_names": list(FEATURE_SETS[selected_pair[0]]),
            "model_name": selected_pair[1],
            "decision": decision,
            "replacement_gate_passed": passed,
            "replacement_gate_reasons": reasons,
            "development_bridge_macro_mae": selected_metrics["bridge_macro_mae"],
            "development_q90_absolute_error": selected_metrics["absolute_error_quantiles"]["q90"],
        }

    holdout_predictions: list[dict[str, Any]] = [
        {
            "sample_key": row["sample_key"],
            "bridge_key": row["bridge_key"],
            "bridge_name": row["bridge_name"],
            "split_group_key": row["split_group_key"],
            "span_m": row["span_m"],
        }
        for row in holdout_rows
    ]
    holdout_metrics: dict[str, dict[str, Any]] = {}
    baseline_holdout_metrics: dict[str, dict[str, Any]] = {}
    artifact_dir = output_dir / "model_artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    final_params: dict[str, dict[str, Any]] = {}
    for target_index, target_name in enumerate(TARGETS):
        selected_pair = (selection[target_name]["feature_set"], selection[target_name]["model_name"])
        baseline_pair = BASELINE_CANDIDATES[target_name]
        target_spec = TARGETS[target_name]
        observed = np.asarray([float(row[target_spec["column"]]) for row in holdout_rows])
        split_groups = np.asarray([row["split_group_key"] for row in holdout_rows])
        bridge_keys = np.asarray([row["bridge_key"] for row in holdout_rows])

        development_artifact, _ = fit_artifact(
            target_name,
            selected_pair[1],
            development_rows,
            pair_results[selected_pair][1],
            feature_names=FEATURE_SETS[selected_pair[0]],
        )
        selected_prediction = predict_artifact(
            development_artifact,
            feature_matrix(holdout_rows, FEATURE_SETS[selected_pair[0]]),
        )
        holdout_metrics[target_name] = metric_record(
            observed,
            selected_prediction,
            split_groups,
            bridge_keys=bridge_keys,
            bootstrap_iterations=bootstrap_iterations,
            random_seed=random_seed + target_index,
        )

        baseline_artifact, _ = fit_artifact(
            target_name,
            baseline_pair[1],
            development_rows,
            pair_results[baseline_pair][1],
            feature_names=FEATURE_SETS[baseline_pair[0]],
        )
        baseline_prediction = predict_artifact(
            baseline_artifact,
            feature_matrix(holdout_rows, FEATURE_SETS[baseline_pair[0]]),
        )
        baseline_holdout_metrics[target_name] = metric_record(
            observed,
            baseline_prediction,
            split_groups,
            bridge_keys=bridge_keys,
            bootstrap_iterations=bootstrap_iterations,
            random_seed=random_seed + 10 + target_index,
        )
        for row_index in range(len(holdout_rows)):
            holdout_predictions[row_index][f"observed_{target_name}"] = float(observed[row_index])
            holdout_predictions[row_index][f"predicted_{target_name}"] = float(selected_prediction[row_index])
            holdout_predictions[row_index][f"absolute_error_{target_name}"] = abs(
                float(observed[row_index]) - float(selected_prediction[row_index])
            )
            holdout_predictions[row_index][f"baseline_predicted_{target_name}"] = float(baseline_prediction[row_index])

        final_artifact, params = fit_artifact(
            target_name,
            selected_pair[1],
            accepted,
            selected_oof[target_name],
            feature_names=FEATURE_SETS[selected_pair[0]],
        )
        final_artifact.update({
            "artifact_version": ARTIFACT_VERSION,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "feature_set": selected_pair[0],
            "training_rows": len(accepted),
            "training_groups": len({row["split_group_key"] for row in accepted}),
            "development_confirmation_metrics": confirmation_metrics[target_name]["selected"]["metrics"],
            "independent_holdout_metrics": holdout_metrics[target_name],
            "selection_decision": selection[target_name],
            "validation_design": "frozen-v5-partition-screening-plus-logo-confirmation",
        })
        joblib.dump(final_artifact, artifact_dir / f"{target_name}_model.joblib")
        final_params[target_name] = params
        selection[target_name]["final_params"] = params
        selection[target_name]["independent_holdout_bridge_macro_mae"] = holdout_metrics[target_name]["bridge_macro_mae"]

    residual_rows = _residual_rows(selected_oof, selection)
    _write_csv(output_dir / "development_selected_oof.csv", residual_rows)
    _write_csv(output_dir / "independent_holdout_predictions.csv", holdout_predictions)
    _write_json(output_dir / "screening_metrics.json", screening_metrics)
    _write_json(output_dir / "screening_tuning_history.json", screening_tuning)
    _write_json(output_dir / "confirmation_metrics.json", confirmation_metrics)
    _write_json(output_dir / "independent_holdout_metrics.json", holdout_metrics)
    _write_json(output_dir / "baseline_independent_holdout_metrics.json", baseline_holdout_metrics)
    _write_json(output_dir / "model_selection.json", selection)
    _write_csv(output_dir / "prepared_annotations.csv", all_rows)
    _write_csv(output_dir / "training_rows.csv", accepted)
    _write_csv(output_dir / "excluded_rows.csv", [row for row in all_rows if row not in accepted])
    _write_csv(output_dir / "data_partition.csv", [
        {
            "sample_key": row["sample_key"],
            "bridge_key": row["bridge_key"],
            "split_group_key": row["split_group_key"],
            "partition": "development" if row in development_rows else "independent_holdout",
        }
        for row in accepted
    ])
    manifest = {
        "artifact_version": ARTIFACT_VERSION,
        "status": "pilot_not_for_production",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "input_directory": str(input_dir),
        "output_directory": str(output_dir),
        "partition_source": partition_source,
        "partition_sha256": _sha256(partition_file) if partition_file is not None else None,
        "source_files": [{"name": path.name, "sha256": _sha256(path)} for path in paths],
        "source_rows": len(all_rows),
        "training_rows": len(accepted),
        "development_rows": len(development_rows),
        "independent_holdout_rows": len(holdout_rows),
        "feature_sets": {name: list(values) for name, values in FEATURE_SETS.items()},
        "model_candidates": list(MODEL_CANDIDATES),
        "baseline_candidates": {target: list(pair) for target, pair in BASELINE_CANDIDATES.items()},
        "selection": selection,
        "validation": {
            "screening_outer": "GroupKFold(5)",
            "screening_inner": "GroupKFold(3)",
            "confirmation_outer": "LeaveOneGroupOut",
            "confirmation_inner": "GroupKFold(5)",
            "split_by": "split_group_key",
            "score_by": "bridge_key",
            "holdout_used_for_selection": False,
        },
        "random_seed": random_seed,
        "bootstrap_iterations": bootstrap_iterations,
    }
    _write_json(output_dir / "training_manifest.json", manifest)
    (output_dir / "model_improvement_report.md").write_text(
        _report(
            source_rows=len(all_rows),
            training_rows=len(accepted),
            development_rows=len(development_rows),
            holdout_rows=len(holdout_rows),
            selection=selection,
            holdout_metrics=holdout_metrics,
            baseline_holdout_metrics=baseline_holdout_metrics,
        ),
        encoding="utf-8",
    )
    return {"output_dir": str(output_dir), "selection": selection}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run v6 grouped small-sample model improvement.")
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--partition-file", type=Path)
    parser.add_argument("--pattern", default="*_five_miao_nodes.json")
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--random-seed", type=int, default=20260813)
    args = parser.parse_args()
    result = run_model_improvement(
        args.input_dir,
        args.output_dir,
        partition_file=args.partition_file,
        pattern=args.pattern,
        bootstrap_iterations=args.bootstrap_iterations,
        random_seed=args.random_seed,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
