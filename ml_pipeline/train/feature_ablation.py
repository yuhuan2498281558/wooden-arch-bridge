from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

from ml_pipeline.prepare.symmetric_targets import collect_annotation_paths
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


ARTIFACT_VERSION = "five-miao-node-pilot-v5-independent-holdout"
FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "span_only": ("span_m",),
    "rise_ratio_only": ("three_miao_rise_span_ratio",),
    "span_rise_ratio": ("span_m", "three_miao_rise_span_ratio"),
    # rise_m = span_m * rise/span is an explicit interaction term.  It is
    # evaluated separately rather than assumed to be useful.
    "span_rise_with_height_interaction": (
        "span_m",
        "three_miao_rise_span_ratio",
        "three_miao_rise_m",
    ),
    "span_rise_layout": (
        "span_m",
        "three_miao_rise_span_ratio",
        "span_count",
        "is_edge_span",
        "span_center_distance",
    ),
}


def choose_feature_model(
    all_metrics: dict[str, dict[str, Any]],
    target_name: str,
) -> tuple[str, str]:
    candidates: list[tuple[str, str, float]] = []
    for feature_set, target_metrics in all_metrics.items():
        for model_name, record in target_metrics[target_name].items():
            candidates.append((
                feature_set,
                model_name,
                float(record.get("bridge_macro_mae", record["group_macro_mae"])),
            ))
    best_score = min(score for _, _, score in candidates)
    eligible = [item for item in candidates if item[2] <= best_score + MODEL_SELECTION_TOLERANCE]
    model_rank = {name: index for index, name in enumerate(MODEL_SIMPLICITY_ORDER)}
    eligible.sort(
        key=lambda item: (
            model_rank[item[1]],
            len(FEATURE_SETS[item[0]]),
            item[2],
            item[0],
        )
    )
    feature_set, model_name, _ = eligible[0]
    return feature_set, model_name


def report_markdown(
    all_rows: list[dict[str, Any]],
    accepted: list[dict[str, Any]],
    development_rows: list[dict[str, Any]],
    holdout_rows: list[dict[str, Any]],
    metrics: dict[str, dict[str, Any]],
    holdout_metrics: dict[str, dict[str, Any]],
    selection: dict[str, dict[str, Any]],
) -> str:
    lines = [
        "# 五节苗节点 v5 独立留出验证报告",
        "",
        "> 状态：探索性模型，不得直接作为工程定型模型。",
        "",
        "## 数据与约束",
        "",
        f"- 原始标注：{len(all_rows)} 条；参与训练：{len(accepted)} 条。",
        f"- 独立桥梁/谱系分组：{len({row['split_group_key'] for row in accepted})} 组。",
        f"- 开发集：{len(development_rows)} 条；独立内部留出集：{len(holdout_rows)} 条。",
        "- 三节苗设计几何采用左斜弦投影、中间平弦、右斜弦投影各占净跨 1/3。",
        "- `three_miao_design_chord_angle_deg = atan(3 × rise/span)`，与矢跨比确定等价，不同时作为独立特征。",
        "- 开发集内使用 Leave-One-Group-Out + GroupKFold 调参；特征和模型只在开发集选择。",
        "- 独立内部留出集不参与特征、模型或超参数选择，仅用于选择后的最终评估。",
        "",
        "## 特征消融",
        "",
        "| 特征集 | 目标 | Ridge 开发集桥级宏 MAE | Huber 开发集桥级宏 MAE |",
        "|---|---|---:|---:|",
    ]
    for feature_set, target_metrics in metrics.items():
        for target_name in TARGETS:
            lines.append(
                f"| {feature_set} | {target_name} | "
                f"{target_metrics[target_name]['ridge']['bridge_macro_mae']:.5f} | "
                f"{target_metrics[target_name]['huber']['bridge_macro_mae']:.5f} |"
            )
    lines.extend(["", "## 选择结果", ""])
    for target_name, record in selection.items():
        lines.append(
            f"- `{target_name}`：`{record['model_name']}` + `{record['feature_set']}`，"
            f"开发集桥级宏 MAE `{record['development_bridge_macro_mae']:.5f}`；"
            f"独立留出桥级宏 MAE `{holdout_metrics[target_name]['bridge_macro_mae']:.5f}`。"
        )
    lines.extend([
        "",
        "## 解释边界",
        "",
        "- 矢高、矢跨比与斜弦角来自同一组三节苗锚点，不应被误写成三个独立信息源。",
        "- 多跨特征目前只来自少数桥梁/谱系；即使交叉验证改善，也只能作为后续扩样线索。",
        "- 构件数量和更细桥型尚未结构化写入 JSON，本轮不从图像线条猜测。",
        "- 独立留出是同一来源数据内的分组留出，不等同于外部数据验证。",
        "",
    ])
    return "\n".join(lines)


def split_development_holdout(
    rows: list[dict[str, Any]],
    *,
    holdout_fraction: float,
    random_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups = np.asarray([row["split_group_key"] for row in rows])
    splitter = GroupShuffleSplit(n_splits=1, test_size=holdout_fraction, random_state=random_seed)
    development_index, holdout_index = next(splitter.split(np.zeros(len(rows)), groups=groups))
    development = [rows[int(index)] for index in development_index]
    holdout = [rows[int(index)] for index in holdout_index]
    development_groups = {row["split_group_key"] for row in development}
    holdout_groups = {row["split_group_key"] for row in holdout}
    if development_groups & holdout_groups:
        raise RuntimeError("development and holdout split groups overlap")
    if len(development_groups) < 5 or len(holdout_groups) < 2:
        raise ValueError("independent holdout requires at least 5 development groups and 2 holdout groups")
    return development, holdout


def run_feature_ablation(
    input_dir: Path,
    output_dir: Path,
    *,
    pattern: str = "*_five_miao_nodes.json",
    bootstrap_iterations: int = 5000,
    random_seed: int = 20260813,
    holdout_fraction: float = 0.2,
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
    development_rows, holdout_rows = split_development_holdout(
        accepted,
        holdout_fraction=holdout_fraction,
        random_seed=random_seed,
    )

    all_metrics: dict[str, dict[str, Any]] = {}
    all_oof: dict[str, list[dict[str, Any]]] = {}
    all_tuning: dict[str, Any] = {}
    for feature_set, feature_names in FEATURE_SETS.items():
        metrics, oof_rows, tuning = nested_group_evaluation(
            development_rows,
            feature_names=feature_names,
            bootstrap_iterations=bootstrap_iterations,
            random_seed=random_seed,
        )
        all_metrics[feature_set] = metrics
        all_oof[feature_set] = oof_rows
        all_tuning[feature_set] = tuning
        _write_csv(output_dir / f"oof_{feature_set}.csv", oof_rows)

    selection: dict[str, dict[str, Any]] = {}
    holdout_metrics: dict[str, dict[str, Any]] = {}
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
    full_oof_cache: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    artifact_dir = output_dir / "model_artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    for target_name in TARGETS:
        feature_set, model_name = choose_feature_model(all_metrics, target_name)
        feature_names = FEATURE_SETS[feature_set]
        development_artifact, _ = fit_artifact(
            target_name,
            model_name,
            development_rows,
            all_oof[feature_set],
            feature_names=feature_names,
        )
        holdout_prediction = predict_artifact(
            development_artifact,
            feature_matrix(holdout_rows, feature_names),
        )
        target_spec = TARGETS[target_name]
        holdout_observed = np.asarray([
            float(row[target_spec["column"]]) for row in holdout_rows
        ])
        holdout_metrics[target_name] = metric_record(
            holdout_observed,
            holdout_prediction,
            np.asarray([row["split_group_key"] for row in holdout_rows]),
            bridge_keys=np.asarray([row["bridge_key"] for row in holdout_rows]),
            bootstrap_iterations=bootstrap_iterations,
            random_seed=random_seed + len(holdout_metrics),
        )
        for index, prediction in enumerate(holdout_prediction):
            holdout_predictions[index][f"observed_{target_name}"] = float(holdout_observed[index])
            holdout_predictions[index][f"predicted_{target_name}"] = float(prediction)
            holdout_predictions[index][f"absolute_error_{target_name}"] = abs(
                float(holdout_observed[index]) - float(prediction)
            )

        if feature_names not in full_oof_cache:
            _, full_oof, _ = nested_group_evaluation(
                accepted,
                feature_names=feature_names,
                bootstrap_iterations=bootstrap_iterations,
                random_seed=random_seed,
            )
            full_oof_cache[feature_names] = full_oof
        artifact, final_params = fit_artifact(
            target_name,
            model_name,
            accepted,
            full_oof_cache[feature_names],
            feature_names=feature_names,
        )
        artifact.update({
            "artifact_version": ARTIFACT_VERSION,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "feature_set": feature_set,
            "training_rows": len(accepted),
            "training_groups": len({row["split_group_key"] for row in accepted}),
            "development_selection_metrics": all_metrics[feature_set][target_name][model_name],
            "independent_holdout_metrics": holdout_metrics[target_name],
            "validation_design": "grouped-development-selection-plus-independent-internal-holdout",
        })
        joblib.dump(artifact, artifact_dir / f"{target_name}_model.joblib")
        selection[target_name] = {
            "feature_set": feature_set,
            "feature_names": list(feature_names),
            "model_name": model_name,
            "final_params": final_params,
            "development_bridge_macro_mae": all_metrics[feature_set][target_name][model_name]["bridge_macro_mae"],
            "independent_holdout_bridge_macro_mae": holdout_metrics[target_name]["bridge_macro_mae"],
            "group_macro_mae": holdout_metrics[target_name]["bridge_macro_mae"],
        }

    excluded = [row for row in all_rows if row not in accepted]
    _write_csv(output_dir / "prepared_annotations.csv", all_rows)
    _write_csv(output_dir / "training_rows.csv", accepted)
    _write_csv(output_dir / "excluded_rows.csv", excluded)
    _write_json(output_dir / "feature_set_metrics.json", all_metrics)
    _write_json(output_dir / "tuning_history.json", all_tuning)
    _write_json(output_dir / "model_selection.json", selection)
    _write_json(output_dir / "independent_holdout_metrics.json", holdout_metrics)
    _write_csv(output_dir / "independent_holdout_predictions.csv", holdout_predictions)
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
        "source_files": [{"name": path.name, "sha256": _sha256(path)} for path in paths],
        "source_rows": len(all_rows),
        "training_rows": len(accepted),
        "training_groups": len({row["split_group_key"] for row in accepted}),
        "excluded_rows": len(excluded),
        "development_rows": len(development_rows),
        "development_groups": len({row["split_group_key"] for row in development_rows}),
        "independent_holdout_rows": len(holdout_rows),
        "independent_holdout_groups": len({row["split_group_key"] for row in holdout_rows}),
        "holdout_fraction": holdout_fraction,
        "feature_sets": {name: list(values) for name, values in FEATURE_SETS.items()},
        "selection": selection,
        "validation": {
            "development_outer": "LeaveOneGroupOut",
            "development_inner": "GroupKFold(5)",
            "independent_internal_holdout": "GroupShuffleSplit",
            "split_by": "split_group_key",
            "score_by": "bridge_key",
        },
        "selection_metric": "bridge_macro_mae",
        "selection_tolerance": MODEL_SELECTION_TOLERANCE,
        "random_seed": random_seed,
        "bootstrap_iterations": bootstrap_iterations,
    }
    _write_json(output_dir / "training_manifest.json", manifest)
    (output_dir / "feature_ablation_report.md").write_text(
        report_markdown(
            all_rows,
            accepted,
            development_rows,
            holdout_rows,
            all_metrics,
            holdout_metrics,
            selection,
        ),
        encoding="utf-8",
    )
    return {"output_dir": str(output_dir), "selection": selection}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run grouped v4 feature ablation for five-miao nodes.")
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--pattern", default="*_five_miao_nodes.json")
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--random-seed", type=int, default=20260813)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    args = parser.parse_args()
    result = run_feature_ablation(
        args.input_dir,
        args.output_dir,
        pattern=args.pattern,
        bootstrap_iterations=args.bootstrap_iterations,
        random_seed=args.random_seed,
        holdout_fraction=args.holdout_fraction,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
