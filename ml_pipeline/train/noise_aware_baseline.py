from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from ml_pipeline.prepare.layered_targets import (
    TARGET_LAYER_VERSION,
    build_layered_rows,
    load_historical_reviews,
    normative_review_template,
)
from ml_pipeline.prepare.symmetric_targets import collect_annotation_paths
from ml_pipeline.train.model_improvement import (
    load_frozen_partition,
    passes_replacement_gate,
)
from ml_pipeline.train.pilot import (
    MODEL_SELECTION_TOLERANCE,
    MODEL_SIMPLICITY_ORDER,
    TARGETS,
    _sha256,
    _write_csv,
    _write_json,
    choose_winner,
    feature_matrix,
    fit_artifact,
    load_training_rows,
    metric_record,
    nested_group_evaluation,
    predict_artifact,
)


ARTIFACT_VERSION = "five-miao-node-research-v7-historical-proxy"
RESEARCH_STATUS = "research_proxy_not_for_deployment"
ROBUST_MODELS = ("ridge", "huber", "quantile")
TARGET_FEATURES = {
    "alpha": ("span_m",),
    "beta": ("span_m", "three_miao_rise_span_ratio"),
}


def _candidate_from_screening(target_metrics: dict[str, dict[str, Any]]) -> str:
    return choose_winner(target_metrics, tolerance=MODEL_SELECTION_TOLERANCE)


def _report(
    selection: dict[str, dict[str, Any]],
    holdout_metrics: dict[str, dict[str, Any]],
    *,
    source_rows: int,
    training_rows: int,
    confirmed_variation_rows: int,
) -> str:
    lines = [
        "# 五节苗节点 v7 历史代理稳健基线",
        "",
        "> 本报告预测的是历史观测左右均值代理，不是专家批准的规范化设计目标；产物禁止部署。",
        "",
        "## 数据层",
        "",
        f"- 原始标注：{source_rows} 条；标准结构训练记录：{training_rows} 条。",
        f"- 已确认标注正确的高差异历史记录：{confirmed_variation_rows} 条，全部保留。",
        "- 原始左右观测、历史对称代理和规范化设计目标使用独立字段；规范化目标当前为空。",
        "",
        "## 稳健基线结果",
        "",
    ]
    for target, record in selection.items():
        holdout = holdout_metrics[target]
        lines.extend([
            f"### {target}",
            "",
            f"- 开发集候选：`{record['screened_candidate']}`；最终研究基线：`{record['model_name']}`。",
            f"- 开发集 LOGO 桥级宏 MAE：`{record['development_bridge_macro_mae']:.5f}`。",
            f"- 冻结内部留出桥级宏 MAE：`{holdout['bridge_macro_mae']:.5f}`；q90：`{holdout['absolute_error_quantiles']['q90']:.5f}`。",
            f"- 决策：`{record['decision']}`。",
            "",
        ])
    lines.extend([
        "## 结论边界",
        "",
        "- 若稳健回归没有稳定超过 Ridge，说明仅靠损失函数无法分离桥型差异与服役扰动。",
        "- 规范化设计模型必须等待专家目标覆盖达到可训练数量，并继续使用桥梁/谱系分组验证。",
        "- 本产物版本和状态不满足算法服务加载契约，无法误接入线上。",
        "",
    ])
    return "\n".join(lines)


def run_noise_aware_baseline(
    input_dir: Path,
    output_dir: Path,
    *,
    partition_file: Path,
    historical_review_file: Path | None = None,
    pattern: str = "*_five_miao_nodes.json",
    bootstrap_iterations: int = 5000,
    random_seed: int = 20260813,
) -> dict[str, Any]:
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    partition_file = partition_file.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = collect_annotation_paths([], [input_dir], pattern)
    if not paths:
        raise ValueError(f"no annotation files matched {pattern!r} in {input_dir}")
    all_rows, accepted = load_training_rows(paths)
    reviews = load_historical_reviews(historical_review_file)
    expected_keys = {str(row["sample_key"]) for row in all_rows}
    unknown = sorted(set(reviews) - expected_keys)
    if unknown:
        raise ValueError(f"historical review contains unknown sample_key: {unknown[:5]}")
    layered_all = build_layered_rows(all_rows, historical_reviews=reviews)
    layered_by_key = {str(row["sample_key"]): row for row in layered_all}
    layered_accepted = [layered_by_key[str(row["sample_key"])] for row in accepted]
    development, holdout = load_frozen_partition(layered_accepted, partition_file)

    screening: dict[str, Any] = {}
    screening_oof: dict[str, list[dict[str, Any]]] = {}
    for target_index, target in enumerate(TARGETS):
        metrics, oof_rows, tuning = nested_group_evaluation(
            development,
            feature_names=TARGET_FEATURES[target],
            model_names=ROBUST_MODELS,
            outer_splits=5,
            inner_splits=3,
            bootstrap_iterations=min(bootstrap_iterations, 1000),
            random_seed=random_seed + target_index,
        )
        screening[target] = {
            "metrics": metrics[target],
            "tuning": tuning[target],
        }
        screening_oof[target] = oof_rows

    confirmation: dict[str, Any] = {}
    selected_oof: dict[str, list[dict[str, Any]]] = {}
    selection: dict[str, dict[str, Any]] = {}
    for target_index, target in enumerate(TARGETS):
        candidate = _candidate_from_screening(screening[target]["metrics"])
        tuned_models = tuple(dict.fromkeys(
            name for name in ("ridge", candidate) if name not in {"fixed_rule", "train_median"}
        ))
        metrics, oof_rows, tuning = nested_group_evaluation(
            development,
            feature_names=TARGET_FEATURES[target],
            model_names=tuned_models,
            outer_splits=None,
            inner_splits=5,
            bootstrap_iterations=bootstrap_iterations,
            random_seed=random_seed + 10 + target_index,
        )
        baseline_record = metrics[target]["ridge"]
        candidate_record = metrics[target][candidate]
        if candidate == "ridge":
            passed, reasons = False, ["screening_retained_ridge"]
            selected = "ridge"
            decision = "retained_ridge"
        else:
            passed, reasons = passes_replacement_gate(candidate_record, baseline_record)
            selected = candidate if passed else "ridge"
            decision = "robust_candidate_selected" if passed else "robust_candidate_rejected_by_gate"
        selected_record = metrics[target][selected]
        selection[target] = {
            "target_semantics": "historical_symmetric_proxy_not_expert_normative",
            "feature_names": list(TARGET_FEATURES[target]),
            "screened_candidate": candidate,
            "model_name": selected,
            "decision": decision,
            "replacement_gate_passed": passed,
            "replacement_gate_reasons": reasons,
            "development_bridge_macro_mae": selected_record["bridge_macro_mae"],
            "development_q90_absolute_error": selected_record["absolute_error_quantiles"]["q90"],
        }
        confirmation[target] = {
            "metrics": metrics[target],
            "tuning": tuning[target],
        }
        selected_oof[target] = oof_rows

    holdout_metrics: dict[str, dict[str, Any]] = {}
    holdout_predictions: list[dict[str, Any]] = [
        {
            "sample_key": row["sample_key"],
            "bridge_name": row["bridge_name"],
            "bridge_key": row["bridge_key"],
            "split_group_key": row["split_group_key"],
        }
        for row in holdout
    ]
    artifact_dir = output_dir / "research_model_artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    for target_index, target in enumerate(TARGETS):
        target_spec = TARGETS[target]
        model_name = selection[target]["model_name"]
        development_artifact, _ = fit_artifact(
            target,
            model_name,
            development,
            selected_oof[target],
            feature_names=TARGET_FEATURES[target],
        )
        prediction = predict_artifact(
            development_artifact,
            feature_matrix(holdout, TARGET_FEATURES[target]),
        )
        observed = np.asarray([float(row[target_spec["column"]]) for row in holdout])
        holdout_metrics[target] = metric_record(
            observed,
            prediction,
            np.asarray([row["split_group_key"] for row in holdout]),
            bridge_keys=np.asarray([row["bridge_key"] for row in holdout]),
            bootstrap_iterations=bootstrap_iterations,
            random_seed=random_seed + 20 + target_index,
        )
        for index in range(len(holdout)):
            holdout_predictions[index][f"observed_{target}"] = float(observed[index])
            holdout_predictions[index][f"predicted_{target}"] = float(prediction[index])
            holdout_predictions[index][f"absolute_error_{target}"] = abs(float(observed[index]) - float(prediction[index]))

        artifact, params = fit_artifact(
            target,
            model_name,
            layered_accepted,
            selected_oof[target],
            feature_names=TARGET_FEATURES[target],
        )
        artifact.update({
            "artifact_version": ARTIFACT_VERSION,
            "status": RESEARCH_STATUS,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "target_layer_version": TARGET_LAYER_VERSION,
            "target_semantics": "historical_symmetric_proxy_not_expert_normative",
            "deployment_eligible": False,
            "deployment_block_reason": "normative_design_targets_not_expert_approved",
            "training_rows": len(layered_accepted),
            "training_groups": len({row["split_group_key"] for row in layered_accepted}),
            "development_metrics": confirmation[target]["metrics"][model_name],
            "independent_holdout_metrics": holdout_metrics[target],
            "selection_decision": selection[target],
        })
        joblib.dump(artifact, artifact_dir / f"{target}_research_proxy_model.joblib")
        selection[target]["final_params"] = params
        selection[target]["independent_holdout_bridge_macro_mae"] = holdout_metrics[target]["bridge_macro_mae"]

    _write_csv(output_dir / "layered_targets.csv", layered_all)
    _write_csv(output_dir / "normative_target_review_template.csv", normative_review_template(layered_all))
    _write_csv(output_dir / "independent_holdout_predictions.csv", holdout_predictions)
    _write_json(output_dir / "screening.json", screening)
    _write_json(output_dir / "confirmation.json", confirmation)
    _write_json(output_dir / "model_selection.json", selection)
    _write_json(output_dir / "independent_holdout_metrics.json", holdout_metrics)
    _write_csv(output_dir / "data_partition.csv", [
        {
            "sample_key": row["sample_key"],
            "bridge_key": row["bridge_key"],
            "split_group_key": row["split_group_key"],
            "partition": "development" if row in development else "independent_holdout",
        }
        for row in layered_accepted
    ])
    confirmed_variation_rows = sum(
        row["historical_variation_status"] == "confirmed_valid_historical_variation"
        for row in layered_all
    )
    manifest = {
        "artifact_version": ARTIFACT_VERSION,
        "status": RESEARCH_STATUS,
        "deployment_eligible": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_directory": str(input_dir),
        "output_directory": str(output_dir),
        "partition_file": str(partition_file),
        "partition_file_sha256": _sha256(partition_file),
        "historical_review_file": str(historical_review_file.resolve()) if historical_review_file else None,
        "historical_review_file_sha256": _sha256(historical_review_file) if historical_review_file else None,
        "source_rows": len(layered_all),
        "training_rows": len(layered_accepted),
        "development_rows": len(development),
        "holdout_rows": len(holdout),
        "confirmed_historical_variation_rows": confirmed_variation_rows,
        "normative_approved_rows": 0,
        "target_semantics": "historical_symmetric_proxy_not_expert_normative",
        "selection": selection,
        "source_files": [{"name": path.name, "sha256": _sha256(path)} for path in paths],
        "validation": {
            "screening": "GroupKFold(5 outer, 3 inner) on development only",
            "confirmation": "LeaveOneGroupOut outer, GroupKFold(5) inner on development only",
            "holdout_used_for_selection": False,
            "split_by": "split_group_key",
            "score_by": "bridge_key",
        },
    }
    _write_json(output_dir / "training_manifest.json", manifest)
    (output_dir / "noise_aware_baseline_report.md").write_text(
        _report(
            selection,
            holdout_metrics,
            source_rows=len(layered_all),
            training_rows=len(layered_accepted),
            confirmed_variation_rows=confirmed_variation_rows,
        ),
        encoding="utf-8",
    )
    return {"output_dir": str(output_dir), "selection": selection}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run v7 robust baseline on historical proxy targets.")
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--partition-file", required=True, type=Path)
    parser.add_argument("--historical-review-file", type=Path)
    parser.add_argument("--pattern", default="*_five_miao_nodes.json")
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--random-seed", type=int, default=20260813)
    args = parser.parse_args()
    result = run_noise_aware_baseline(
        args.input_dir,
        args.output_dir,
        partition_file=args.partition_file,
        historical_review_file=args.historical_review_file,
        pattern=args.pattern,
        bootstrap_iterations=args.bootstrap_iterations,
        random_seed=args.random_seed,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
