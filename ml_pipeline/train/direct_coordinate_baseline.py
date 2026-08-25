# -*- coding: utf-8 -*-
"""Compare direct Q1--Q4 coordinate regression with the knowledge parameterization.

The experiment keeps the accepted sample population and frozen v5 partition.
Direct-coordinate candidates learn the four historical Q points without
symmetry, parallel-chord or feasible-domain post-processing.  The knowledge
baseline predicts the current processed alpha/beta targets and reconstructs
Q1--Q4 from the observed A--D anchors.

Two endpoints are reported deliberately:

* historical_observation: reproduction of the annotated, possibly asymmetric
  Q1--Q4 coordinates;
* symmetric_design_proxy: agreement with Q1--Q4 reconstructed from the current
  processed alpha/beta labels.

The second endpoint is still a preprocessing-derived proxy, not expert truth.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from bridge_algorithm_service.node_geometry import Point, measure_ratios, reconstruct_nodes
from ml_pipeline.prepare.symmetric_targets import collect_annotation_paths
from ml_pipeline.train.pilot import load_training_rows


RIDGE_ALPHA = 1e-8
RANDOM_SEED = 20260820
POINT_NAMES = ("Q1", "Q2", "Q3", "Q4")
ANCHOR_NAMES = ("A", "B", "C", "D")
DIRECT_FEATURES = (
    "span_m",
    "three_miao_rise_span_ratio",
    "span_count",
    "span_position_normalized",
)
KNOWLEDGE_FEATURES = {
    "alpha": ("span_m",),
    "beta": ("span_m", "three_miao_rise_span_ratio"),
}
METHODS = (
    "fixed_rule",
    "raw_pixel_direct",
    "image_normalized_direct",
    "local_axis_direct",
    "knowledge_alpha_beta",
)
DEFAULT_DATA_DIR = Path(r"C:\Users\24982\Desktop\数据集")
DEFAULT_PARTITION = Path(
    r"C:\Users\24982\Desktop\机器学习结果\five_miao_pilot_v5_independent_holdout\data_partition.csv"
)
DEFAULT_OUTPUT_DIR = Path("scratch/direct_coordinate_baseline_v1")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _model() -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("ridge", Ridge(alpha=RIDGE_ALPHA)),
    ])


def _point(row: dict[str, Any], name: str) -> Point:
    return Point(float(row[f"{name}_x_px"]), float(row[f"{name}_y_px"]))


def _anchor_payload(row: dict[str, Any]) -> dict[str, Point]:
    return {name: _point(row, name) for name in ANCHOR_NAMES}


def _axis_frame(row: dict[str, Any]) -> tuple[Point, float, float, float, float, float]:
    anchors = _anchor_payload(row)
    a, b, c, d = (anchors[name] for name in ANCHOR_NAMES)
    axis_dx = c.x - b.x
    axis_dy = c.y - b.y
    axis_length = math.hypot(axis_dx, axis_dy)
    if axis_length <= 0:
        raise ValueError(f"invalid B--C axis: {row['sample_key']}")
    ux, uy = axis_dx / axis_length, axis_dy / axis_length
    signed_span = (d.x - a.x) * ux + (d.y - a.y) * uy
    span_scale = abs(signed_span)
    if span_scale <= 0:
        raise ValueError(f"invalid A--D longitudinal span: {row['sample_key']}")
    return a, ux, uy, uy, -ux, signed_span


def _to_local(row: dict[str, Any], point: Point) -> tuple[float, float]:
    origin, ux, uy, nx, ny, signed_span = _axis_frame(row)
    dx, dy = point.x - origin.x, point.y - origin.y
    return (
        (dx * ux + dy * uy) / signed_span,
        (dx * nx + dy * ny) / abs(signed_span),
    )


def _from_local(row: dict[str, Any], u: float, v: float) -> Point:
    origin, ux, uy, nx, ny, signed_span = _axis_frame(row)
    normal_scale = abs(signed_span)
    return Point(
        origin.x + u * signed_span * ux + v * normal_scale * nx,
        origin.y + u * signed_span * uy + v * normal_scale * ny,
    )


def _augment_image_dimensions(
    rows: list[dict[str, Any]], data_dir: Path
) -> list[dict[str, Any]]:
    augmented: list[dict[str, Any]] = []
    for source in rows:
        path = data_dir / str(source["annotation_file"])
        with path.open(encoding="utf-8-sig") as handle:
            payload = json.load(handle)
        image = payload.get("source_image") or {}
        width = float(image.get("width_px") or 0.0)
        height = float(image.get("height_px") or 0.0)
        if not math.isfinite(width) or not math.isfinite(height) or width <= 0 or height <= 0:
            raise ValueError(f"invalid source image dimensions: {path}")
        row = dict(source)
        row["image_width_px"] = width
        row["image_height_px"] = height
        augmented.append(row)
    return augmented


def _feature_matrix(rows: list[dict[str, Any]], names: tuple[str, ...]) -> np.ndarray:
    return np.asarray([[float(row[name]) for name in names] for row in rows], dtype=float)


def _direct_targets(rows: list[dict[str, Any]], method: str) -> np.ndarray:
    values: list[list[float]] = []
    for row in rows:
        item: list[float] = []
        for name in POINT_NAMES:
            point = _point(row, name)
            if method == "raw_pixel_direct":
                x, y = point.x, point.y
            elif method == "image_normalized_direct":
                x = point.x / float(row["image_width_px"])
                y = point.y / float(row["image_height_px"])
            elif method == "local_axis_direct":
                x, y = _to_local(row, point)
            else:
                raise ValueError(f"unknown direct-coordinate method: {method}")
            item.extend([x, y])
        values.append(item)
    return np.asarray(values, dtype=float)


def _decode_direct(
    rows: list[dict[str, Any]], method: str, values: np.ndarray
) -> np.ndarray:
    output = np.empty((len(rows), len(POINT_NAMES), 2), dtype=float)
    for row_index, row in enumerate(rows):
        for point_index, _name in enumerate(POINT_NAMES):
            first = float(values[row_index, 2 * point_index])
            second = float(values[row_index, 2 * point_index + 1])
            if method == "raw_pixel_direct":
                point = Point(first, second)
            elif method == "image_normalized_direct":
                point = Point(
                    first * float(row["image_width_px"]),
                    second * float(row["image_height_px"]),
                )
            elif method == "local_axis_direct":
                point = _from_local(row, first, second)
            else:
                raise ValueError(f"unknown direct-coordinate method: {method}")
            output[row_index, point_index] = (point.x, point.y)
    return output


def _fit_direct(
    train_rows: list[dict[str, Any]], test_rows: list[dict[str, Any]], method: str
) -> np.ndarray:
    estimator = _model().fit(
        _feature_matrix(train_rows, DIRECT_FEATURES),
        _direct_targets(train_rows, method),
    )
    predicted = estimator.predict(_feature_matrix(test_rows, DIRECT_FEATURES))
    return _decode_direct(test_rows, method, predicted)


def _fit_knowledge(
    train_rows: list[dict[str, Any]], test_rows: list[dict[str, Any]]
) -> np.ndarray:
    predicted: dict[str, np.ndarray] = {}
    for target in ("alpha", "beta"):
        features = KNOWLEDGE_FEATURES[target]
        y_train = np.asarray(
            [float(row[f"design_target_{target}"]) for row in train_rows], dtype=float
        )
        predicted[target] = _model().fit(
            _feature_matrix(train_rows, features), y_train
        ).predict(_feature_matrix(test_rows, features))
    return _reconstruct_predictions(test_rows, predicted["alpha"], predicted["beta"])


def _reconstruct_predictions(
    rows: list[dict[str, Any]], alpha: np.ndarray, beta: np.ndarray
) -> np.ndarray:
    output = np.empty((len(rows), len(POINT_NAMES), 2), dtype=float)
    for index, row in enumerate(rows):
        anchors = _anchor_payload(row)
        clipped_alpha = float(np.clip(alpha[index], 1e-9, 1.0 - 1e-9))
        clipped_beta = float(np.clip(beta[index], 1e-9, 0.5 - 1e-9))
        nodes = reconstruct_nodes(**{key.lower(): value for key, value in anchors.items()}, alpha=clipped_alpha, beta=clipped_beta)
        for point_index, name in enumerate(POINT_NAMES):
            output[index, point_index] = (nodes[name].x, nodes[name].y)
    return output


def _fixed_rule(rows: list[dict[str, Any]]) -> np.ndarray:
    return _reconstruct_predictions(
        rows,
        np.full(len(rows), 2.0 / 3.0, dtype=float),
        np.full(len(rows), 1.0 / 4.0, dtype=float),
    )


def _observed_points(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([
        [[float(row[f"{name}_x_px"]), float(row[f"{name}_y_px"])] for name in POINT_NAMES]
        for row in rows
    ], dtype=float)


def _design_proxy_points(rows: list[dict[str, Any]]) -> np.ndarray:
    return _reconstruct_predictions(
        rows,
        np.asarray([float(row["design_target_alpha"]) for row in rows]),
        np.asarray([float(row["design_target_beta"]) for row in rows]),
    )


def _logo_predictions(rows: list[dict[str, Any]], method: str) -> np.ndarray:
    if method == "fixed_rule":
        return _fixed_rule(rows)
    groups = np.asarray([str(row["split_group_key"]) for row in rows], dtype=object)
    output = np.full((len(rows), len(POINT_NAMES), 2), np.nan, dtype=float)
    for group in np.unique(groups):
        train_index = np.flatnonzero(groups != group)
        test_index = np.flatnonzero(groups == group)
        train_rows = [rows[index] for index in train_index]
        test_rows = [rows[index] for index in test_index]
        if method == "knowledge_alpha_beta":
            predicted = _fit_knowledge(train_rows, test_rows)
        else:
            predicted = _fit_direct(train_rows, test_rows, method)
        output[test_index] = predicted
    if not np.all(np.isfinite(output)):
        raise RuntimeError(f"non-finite LOGO predictions for {method}")
    return output


def _holdout_predictions(
    development: list[dict[str, Any]], holdout: list[dict[str, Any]], method: str
) -> np.ndarray:
    if method == "fixed_rule":
        return _fixed_rule(holdout)
    if method == "knowledge_alpha_beta":
        return _fit_knowledge(development, holdout)
    return _fit_direct(development, holdout, method)


def _span_scales(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([abs(_axis_frame(row)[-1]) for row in rows], dtype=float)


def _bridge_macro(values: np.ndarray, bridge_keys: np.ndarray) -> float:
    return float(np.mean([
        float(np.mean(values[bridge_keys == key])) for key in np.unique(bridge_keys)
    ]))


def _error_metrics(
    rows: list[dict[str, Any]], truth: np.ndarray, prediction: np.ndarray
) -> dict[str, Any]:
    errors_px = np.linalg.norm(prediction - truth, axis=2)
    errors_span = errors_px / _span_scales(rows)[:, None]
    spans_m = np.asarray([float(row["span_m"]) for row in rows], dtype=float)
    errors_m = errors_span * spans_m[:, None]
    sample_mean_span = np.mean(errors_span, axis=1)
    sample_mean_m = np.mean(errors_m, axis=1)
    bridge_keys = np.asarray([str(row["bridge_key"]) for row in rows], dtype=object)
    return {
        "bridge_macro_mean_node_error_span": _bridge_macro(sample_mean_span, bridge_keys),
        "bridge_macro_mean_node_error_m": _bridge_macro(sample_mean_m, bridge_keys),
        "mean_node_error_span": float(np.mean(errors_span)),
        "mean_node_error_m": float(np.mean(errors_m)),
        "q90_node_error_span": float(np.quantile(errors_span, 0.9)),
        "q90_node_error_m": float(np.quantile(errors_m, 0.9)),
        "max_node_error_span": float(np.max(errors_span)),
        "max_node_error_m": float(np.max(errors_m)),
        "outer_mean_node_error_m": float(np.mean(errors_m[:, [0, 3]])),
        "inner_mean_node_error_m": float(np.mean(errors_m[:, [1, 2]])),
    }


def _sample_mean_node_error_m(
    rows: list[dict[str, Any]], truth: np.ndarray, prediction: np.ndarray
) -> np.ndarray:
    errors_px = np.linalg.norm(prediction - truth, axis=2)
    errors_span = errors_px / _span_scales(rows)[:, None]
    spans_m = np.asarray([float(row["span_m"]) for row in rows], dtype=float)
    return np.mean(errors_span * spans_m[:, None], axis=1)


def _paired_bridge_bootstrap(
    rows: list[dict[str, Any]], candidate_errors: np.ndarray, knowledge_errors: np.ndarray,
    *, iterations: int = 5000,
) -> dict[str, float]:
    bridge_keys = np.asarray([str(row["bridge_key"]) for row in rows], dtype=object)
    unique = np.unique(bridge_keys)
    paired = np.asarray([
        float(np.mean(candidate_errors[bridge_keys == key]) - np.mean(knowledge_errors[bridge_keys == key]))
        for key in unique
    ])
    rng = np.random.default_rng(RANDOM_SEED)
    draws = np.empty(iterations, dtype=float)
    for index in range(iterations):
        sampled = rng.integers(0, len(paired), size=len(paired))
        draws[index] = float(np.mean(paired[sampled]))
    return {
        "candidate_minus_knowledge_bridge_macro_error_m": float(np.mean(paired)),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "probability_knowledge_better": float(np.mean(draws > 0.0)),
    }


def _geometry_metrics(rows: list[dict[str, Any]], prediction: np.ndarray) -> dict[str, Any]:
    alpha_asymmetry: list[float] = []
    beta_asymmetry: list[float] = []
    projection_offset: list[float] = []
    invalid = 0
    for index, row in enumerate(rows):
        anchors = _anchor_payload(row)
        nodes = {
            name: Point(*prediction[index, point_index])
            for point_index, name in enumerate(POINT_NAMES)
        }
        try:
            measured = measure_ratios(
                anchors["A"], anchors["B"], anchors["C"], anchors["D"],
                nodes["Q1"], nodes["Q2"], nodes["Q3"], nodes["Q4"],
            )
        except ValueError:
            invalid += 1
            continue
        alpha_asymmetry.append(float(measured.alpha_symmetry_error))
        beta_asymmetry.append(float(measured.beta_symmetry_error))
        projection_offset.append(float(measured.max_normalized_projection_error))
        if measured.warnings:
            invalid += 1
    valid = max(len(alpha_asymmetry), 1)
    return {
        "invalid_or_warning_count": invalid,
        "invalid_or_warning_rate": invalid / len(rows),
        "mean_alpha_asymmetry": float(np.mean(alpha_asymmetry)) if alpha_asymmetry else None,
        "mean_beta_asymmetry": float(np.mean(beta_asymmetry)) if beta_asymmetry else None,
        "mean_max_normalized_projection_offset": (
            float(np.mean(projection_offset)) if projection_offset else None
        ),
        "measured_rows": valid,
    }


def _evaluate(
    rows: list[dict[str, Any]], predictions: dict[str, np.ndarray]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    observed = _observed_points(rows)
    design_proxy = _design_proxy_points(rows)
    metrics: dict[str, Any] = {}
    prediction_rows: list[dict[str, Any]] = []
    for method, predicted in predictions.items():
        historical = _error_metrics(rows, observed, predicted)
        design = _error_metrics(rows, design_proxy, predicted)
        metrics[method] = {
            "historical_observation": historical,
            "symmetric_design_proxy": design,
            "geometry": _geometry_metrics(rows, predicted),
        }
        span_scales = _span_scales(rows)
        for row_index, row in enumerate(rows):
            historical_node_errors = np.linalg.norm(
                predicted[row_index] - observed[row_index], axis=1
            ) / span_scales[row_index]
            design_node_errors = np.linalg.norm(
                predicted[row_index] - design_proxy[row_index], axis=1
            ) / span_scales[row_index]
            record: dict[str, Any] = {
                "sample_key": row["sample_key"],
                "bridge_key": row["bridge_key"],
                "split_group_key": row["split_group_key"],
                "method": method,
                "historical_mean_node_error_span": float(np.mean(historical_node_errors)),
                "design_proxy_mean_node_error_span": float(np.mean(design_node_errors)),
            }
            for point_index, name in enumerate(POINT_NAMES):
                record[f"{name}_pred_x_px"] = float(predicted[row_index, point_index, 0])
                record[f"{name}_pred_y_px"] = float(predicted[row_index, point_index, 1])
                record[f"{name}_historical_error_span"] = float(historical_node_errors[point_index])
                record[f"{name}_design_proxy_error_span"] = float(design_node_errors[point_index])
            prediction_rows.append(record)
    knowledge_prediction = predictions["knowledge_alpha_beta"]
    knowledge_historical = _sample_mean_node_error_m(rows, observed, knowledge_prediction)
    knowledge_design = _sample_mean_node_error_m(rows, design_proxy, knowledge_prediction)
    for method, predicted in predictions.items():
        metrics[method]["paired_vs_knowledge"] = {
            "historical_observation": _paired_bridge_bootstrap(
                rows,
                _sample_mean_node_error_m(rows, observed, predicted),
                knowledge_historical,
            ),
            "symmetric_design_proxy": _paired_bridge_bootstrap(
                rows,
                _sample_mean_node_error_m(rows, design_proxy, predicted),
                knowledge_design,
            ),
        }
    return metrics, prediction_rows


def _partition_rows(
    rows: list[dict[str, Any]], partition_path: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    with partition_path.open(encoding="utf-8-sig") as handle:
        partition = {item["sample_key"]: item["partition"] for item in csv.DictReader(handle)}
    missing = [row["sample_key"] for row in rows if row["sample_key"] not in partition]
    if missing:
        raise ValueError(f"partition is missing {len(missing)} accepted rows")
    development = [row for row in rows if partition[row["sample_key"]] == "development"]
    holdout = [row for row in rows if partition[row["sample_key"]] == "independent_holdout"]
    if not development or not holdout:
        raise ValueError("partition must contain development and independent_holdout rows")
    development_groups = {str(row["split_group_key"]) for row in development}
    holdout_groups = {str(row["split_group_key"]) for row in holdout}
    if development_groups & holdout_groups:
        raise ValueError("split_group_key leakage between development and holdout")
    return development, holdout


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(materialized[0]))
        writer.writeheader()
        writer.writerows(materialized)


def _report(result: dict[str, Any]) -> str:
    labels = {
        "fixed_rule": "固定比例 2/3、1/4",
        "raw_pixel_direct": "原始像素直接回归",
        "image_normalized_direct": "图像宽高归一化坐标",
        "local_axis_direct": "桥轴局部坐标直接回归",
        "knowledge_alpha_beta": "知识参数化 alpha/beta",
    }
    lines = [
        "# 原始像素/坐标直接预测与知识参数化对照",
        "",
        "- 样本、冻结 v5 分区和 split_group_key 隔离保持一致。",
        "- 直接坐标模型使用 span、矢跨比、跨数和跨位，给予其可用的布局信息。",
        "- 所有学习方法统一为 StandardScaler + Ridge(alpha=1e-8)。",
        "- 历史观测误差衡量复现现状；对称设计代理误差衡量与当前设计目标的一致性。",
        "",
    ]
    for split_name, title in (
        ("development_logo", "开发集按组留一"),
        ("independent_holdout", "冻结内部留出"),
    ):
        lines.extend([
            f"## {title}",
            "",
            "| 方法 | 历史观测桥级节点误差 m | 历史 q90 m | 对称设计代理桥级节点误差 m | 设计 q90 m | 几何警告率 |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for method in METHODS:
            item = result[split_name][method]
            historical = item["historical_observation"]
            design = item["symmetric_design_proxy"]
            geometry = item["geometry"]
            lines.append(
                f"| {labels[method]} | "
                f"{historical['bridge_macro_mean_node_error_m']:.4f} | "
                f"{historical['q90_node_error_m']:.4f} | "
                f"{design['bridge_macro_mean_node_error_m']:.4f} | "
                f"{design['q90_node_error_m']:.4f} | "
                f"{geometry['invalid_or_warning_rate']:.1%} |"
            )
        lines.append("")
    local_holdout = result["independent_holdout"]["local_axis_direct"]["paired_vs_knowledge"]
    lines.extend([
        "## 冻结留出：最强直接坐标基线与知识参数化的配对差异",
        "",
        "| 共同评估终点 | 局部坐标减知识参数化 m | 95% bootstrap CI | 知识参数化更优概率 |",
        "|---|---:|---:|---:|",
    ])
    for endpoint, title in (
        ("historical_observation", "历史观测复现"),
        ("symmetric_design_proxy", "对称设计代理"),
    ):
        paired = local_holdout[endpoint]
        lines.append(
            f"| {title} | {paired['candidate_minus_knowledge_bridge_macro_error_m']:+.4f} | "
            f"[{paired['ci95_low']:.4f}, {paired['ci95_high']:.4f}] | "
            f"{paired['probability_knowledge_better']:.1%} |"
        )
    lines.append("")
    lines.extend([
        "## 解释边界",
        "",
        "- 原始像素和图像归一化坐标依赖历史图纸的裁切、排版和分辨率，不能直接作为在线设计坐标合同。",
        "- 桥轴局部坐标消除了平移、旋转和尺度，但仍直接学习历史不对称及构件法向偏移。",
        "- 对称设计代理由当前预处理生成，不是独立专家真值；该列不能单独证明更接近原始营造意图。",
        "- 冻结留出仍来自同一网站数据源，只能用于内部比较。",
    ])
    return "\n".join(lines) + "\n"


def run_experiment(
    data_dir: Path, partition_path: Path, output_dir: Path
) -> dict[str, Any]:
    paths = collect_annotation_paths([], [data_dir], "*_five_miao_nodes.json")
    _all_rows, accepted = load_training_rows(paths)
    accepted = _augment_image_dimensions(accepted, data_dir)
    development, holdout = _partition_rows(accepted, partition_path)

    result: dict[str, Any] = {
        "protocol": {
            "comparison": "direct Q1-Q4 coordinates versus processed alpha/beta",
            "model": "StandardScaler + multi-output Ridge",
            "ridge_alpha": RIDGE_ALPHA,
            "direct_features": DIRECT_FEATURES,
            "knowledge_features": KNOWLEDGE_FEATURES,
            "development_validation": "Leave-One-Group-Out by split_group_key",
            "holdout": "frozen v5 independent internal holdout",
            "evaluation_endpoints": ["historical_observation", "symmetric_design_proxy"],
        },
        "data_dir": str(data_dir),
        "partition_path": str(partition_path),
        "accepted_rows": len(accepted),
        "accepted_groups": len({str(row["split_group_key"]) for row in accepted}),
        "development_rows": len(development),
        "development_groups": len({str(row["split_group_key"]) for row in development}),
        "holdout_rows": len(holdout),
        "holdout_groups": len({str(row["split_group_key"]) for row in holdout}),
    }
    prediction_rows: list[dict[str, Any]] = []
    for split_name, rows, prediction_builder in (
        (
            "development_logo",
            development,
            lambda method: _logo_predictions(development, method),
        ),
        (
            "independent_holdout",
            holdout,
            lambda method: _holdout_predictions(development, holdout, method),
        ),
    ):
        predictions = {method: prediction_builder(method) for method in METHODS}
        metrics, rows_out = _evaluate(rows, predictions)
        result[split_name] = metrics
        prediction_rows.extend({"evaluation": split_name, **row} for row in rows_out)

    output_dir.mkdir(parents=True, exist_ok=False)
    _write_json(output_dir / "direct_coordinate_metrics.json", result)
    _write_csv(output_dir / "direct_coordinate_predictions.csv", prediction_rows)
    (output_dir / "direct_coordinate_report.md").write_text(
        _report(result), encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare direct Q1-Q4 coordinate regression with alpha/beta."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--partition", type=Path, default=DEFAULT_PARTITION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error(f"output directory already exists: {args.output_dir}")
    result = run_experiment(args.data_dir, args.partition, args.output_dir)
    print(json.dumps({
        "output_dir": str(args.output_dir),
        "development_rows": result["development_rows"],
        "holdout_rows": result["holdout_rows"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
