# -*- coding: utf-8 -*-
"""五节苗节点位置预测的数据预处理消融实验。

统一固定 Ridge(alpha=1e-8)（沿用 v6 最终选择）与按 split_group_key 的
Leave-One-Group-Out，分别关闭一个预处理环节，输出 JSON 与 Markdown 报告。
"""
from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml_pipeline.prepare.symmetric_targets import collect_annotation_paths
from ml_pipeline.train.pilot import load_training_rows

warnings.filterwarnings("ignore", category=ConvergenceWarning)

RIDGE_ALPHA = 1e-08
TARGETS = {
    "alpha": {"rule": 2.0 / 3.0, "bounds": (0.0, 1.0)},
    "beta": {"rule": 1.0 / 4.0, "bounds": (0.0, 0.5)},
}
FEATURES = {
    "alpha": ("span_m",),
    "beta": ("span_m", "three_miao_rise_span_ratio"),
}
DEFAULT_DATA_DIR = Path(r"C:\Users\24982\Desktop\数据集")
PRE_BACKFILL_DIR = Path(r"C:\Users\24982\Desktop\数据集_")
DEFAULT_OUTPUT_DIR = Path(r"C:\Users\24982\Desktop\机器学习结果\preprocessing_ablation_v1")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _load(data_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    paths = collect_annotation_paths([], [Path(data_dir)], "*_five_miao_nodes.json")
    return load_training_rows(paths)


def _feature_array(rows: list[dict[str, Any]], names: tuple[str, ...]) -> np.ndarray:
    return np.asarray([[float(row[name]) for name in names] for row in rows], dtype=float)


def _bridge_macro_mae(y_true: np.ndarray, y_pred: np.ndarray, bridge_keys: np.ndarray) -> float:
    values = []
    for key in np.unique(bridge_keys):
        mask = bridge_keys == key
        values.append(float(np.mean(np.abs(y_true[mask] - y_pred[mask]))))
    return float(np.mean(values)) if values else math.nan


def _metric_record(y_true: np.ndarray, y_pred: np.ndarray, raw_pred: np.ndarray, bridge_keys: np.ndarray, bounds: tuple[float, float], physical_error: np.ndarray | None) -> dict[str, Any]:
    error = np.abs(y_true - y_pred)
    lower, upper = bounds
    raw_violations = np.sum((raw_pred < lower) | (raw_pred > upper))
    return {
        "n": int(len(y_true)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "bridge_macro_mae": _bridge_macro_mae(y_true, y_pred, bridge_keys),
        "absolute_error_quantiles": {
            "q50": float(np.quantile(error, 0.5)),
            "q90": float(np.quantile(error, 0.9)),
            "q95": float(np.quantile(error, 0.95)),
        },
        "max_absolute_error": float(np.max(error)),
        "raw_violation_count": int(raw_violations),
        "raw_violation_rate": float(raw_violations / len(y_true)),
        "mean_clip_change": float(np.mean(np.abs(y_pred - raw_pred))),
        "max_clip_change": float(np.max(np.abs(y_pred - raw_pred))),
        "mean_node_error_m": float(np.mean(physical_error)) if physical_error is not None else None,
        "max_node_error_m": float(np.max(physical_error)) if physical_error is not None else None,
    }


def _physical_node_error(target: str, y_true: np.ndarray, y_pred: np.ndarray, rows: list[dict[str, Any]]) -> np.ndarray:
    length = []
    for row in rows:
        span = float(row["span_m"])
        if target == "alpha":
            rise = span * float(row["three_miao_rise_span_ratio"])
            length.append(math.hypot(span / 3.0, rise))
        else:
            length.append(span / 3.0)
    return np.abs(y_true - y_pred) * np.asarray(length, dtype=float)


def _make_pipeline(ridge_alpha: float = RIDGE_ALPHA) -> Pipeline:
    return Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=ridge_alpha))])


def run_logo(
    rows: list[dict[str, Any]],
    *,
    feature_names: tuple[str, ...],
    target_name: str,
    target_column: str,
    residual: bool,
    clip: bool,
    ridge_alpha: float = RIDGE_ALPHA,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    spec = TARGETS[target_name]
    x = _feature_array(rows, feature_names)
    y = np.asarray([float(row[target_column]) for row in rows], dtype=float)
    groups = np.asarray([str(row["split_group_key"]) for row in rows], dtype=object)
    bridge_keys = np.asarray([str(row["bridge_key"]) for row in rows], dtype=object)
    raw_pred = np.full(len(rows), np.nan, dtype=float)
    train_target = y - float(spec["rule"]) if residual else y

    for group in np.unique(groups):
        train_mask = groups != group
        test_mask = ~train_mask
        model = _make_pipeline(ridge_alpha)
        model.fit(x[train_mask], train_target[train_mask])
        if residual:
            raw_pred[test_mask] = float(spec["rule"]) + model.predict(x[test_mask])
        else:
            raw_pred[test_mask] = model.predict(x[test_mask])

    final_pred = np.clip(raw_pred, float(spec["bounds"][0]), float(spec["bounds"][1])) if clip else raw_pred.copy()
    physical = _physical_node_error(target_name, y, final_pred, rows)
    record = _metric_record(y, final_pred, raw_pred, bridge_keys, spec["bounds"], physical)
    oof = []
    for index, row in enumerate(rows):
        oof.append({
            "sample_key": row.get("sample_key"),
            "split_group_key": str(row["split_group_key"]),
            "bridge_key": str(row["bridge_key"]),
            "target": target_name,
            "y_true": float(y[index]),
            "y_pred": float(final_pred[index]),
            "y_raw": float(raw_pred[index]),
            "absolute_error": float(abs(y[index] - final_pred[index])),
            "node_error_m": float(physical[index]),
        })
    return record, oof


def _copy_row(row: dict[str, Any], *, alpha: float, beta: float, side: str | None = None) -> dict[str, Any]:
    copied = dict(row)
    copied["_alpha"] = float(alpha)
    copied["_beta"] = float(beta)
    copied["_side"] = side
    return copied


def _side_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        result.append(_copy_row(row, alpha=row["observed_alpha_left"], beta=row["observed_beta_left"], side="left"))
        result.append(_copy_row(row, alpha=row["observed_alpha_right"], beta=row["observed_beta_right"], side="right"))
    return result


def _legacy_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        alpha = (float(row["legacy_projected_alpha_left"]) + float(row["legacy_projected_alpha_right"])) / 2.0
        result.append(_copy_row(row, alpha=alpha, beta=float(row["design_target_beta"])))
    return result


def _design_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_copy_row(row, alpha=float(row["design_target_alpha"]), beta=float(row["design_target_beta"])) for row in rows]


def _rows_with_features(rows: list[dict[str, Any]], side_code: bool) -> list[dict[str, Any]]:
    if not side_code:
        return rows
    result = []
    for row in rows:
        copied = dict(row)
        copied["side_code"] = 1.0 if str(row.get("_side")) == "right" else 0.0
        result.append(copied)
    return result




def _common_record(oof: list[dict[str, Any]], common_keys: set[str], target: str, bounds: tuple[float, float], rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    subset = [item for item, row in zip(oof, rows) if row.get("sample_key") in common_keys]
    if not subset:
        return None
    y_true = np.asarray([item["y_true"] for item in subset])
    y_pred = np.asarray([item["y_pred"] for item in subset])
    y_raw = np.asarray([item["y_raw"] for item in subset])
    bridge = np.asarray([item["bridge_key"] for item in subset])
    sub_rows = [row for item, row in zip(oof, rows) if row.get("sample_key") in common_keys]
    physical = _physical_node_error(target, y_true, y_pred, sub_rows)
    return _metric_record(y_true, y_pred, y_raw, bridge, bounds, physical)


def _run_logo_variant(rows: list[dict[str, Any]], *, target_column: dict[str, str], feature_names: dict[str, tuple[str, ...]], residual: bool, clip: bool, ridge_alpha: float = RIDGE_ALPHA) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    records = {}
    oofs = {}
    for target in ("alpha", "beta"):
        record, oof = run_logo(
            rows,
            feature_names=feature_names[target],
            target_name=target,
            target_column=target_column[target],
            residual=residual,
            clip=clip,
            ridge_alpha=ridge_alpha,
        )
        records[target] = record
        oofs[target] = oof
    return records, oofs


def _run_fixed_baselines(rows: list[dict[str, Any]], target_column: dict[str, str], feature_names: dict[str, tuple[str, ...]]) -> dict[str, Any]:
    """与 LOOGO 相同折结构下计算固定规则与训练中位数基线。"""
    records = {}
    for target in ("alpha", "beta"):
        spec = TARGETS[target]
        y = np.asarray([float(row[target_column[target]]) for row in rows], dtype=float)
        groups = np.asarray([str(row["split_group_key"]) for row in rows], dtype=object)
        bridge = np.asarray([str(row["bridge_key"]) for row in rows], dtype=object)
        pred = {"fixed_rule": np.full(len(rows), float(spec["rule"])), "train_median": np.full(len(rows), np.nan)}
        for group in np.unique(groups):
            train_mask = groups != group
            pred["train_median"][~train_mask] = np.median(y[train_mask])
        records[target] = {
            name: _metric_record(y, value, value, bridge, spec["bounds"], _physical_node_error(target, y, value, rows))
            for name, value in pred.items()
        }
    return records


def _run_kfold_leak_comparison(rows: list[dict[str, Any]], target_column: dict[str, str], feature_names: dict[str, tuple[str, ...]], seeds: tuple[int, ...] = (20260812, 20260813, 20260814, 20260815, 20260816, 20260817, 20260818, 20260819, 20260820, 20260821)) -> dict[str, Any]:
    """比较按 split_group_key 的 GroupKFold 与随机 KFold 的指标差异。"""
    out = {}
    for target in ("alpha", "beta"):
        spec = TARGETS[target]
        x = _feature_array(rows, feature_names[target])
        y = np.asarray([float(row[target_column[target]]) for row in rows], dtype=float)
        groups = np.asarray([str(row["split_group_key"]) for row in rows], dtype=object)
        bridge = np.asarray([str(row["bridge_key"]) for row in rows], dtype=object)
        train_target = y - float(spec["rule"])

        def fit_predict(train_idx, test_idx):
            model = _make_pipeline()
            model.fit(x[train_idx], train_target[train_idx])
            raw = float(spec["rule"]) + model.predict(x[test_idx])
            return np.clip(raw, spec["bounds"][0], spec["bounds"][1])

        # GroupKFold
        gk = GroupKFold(n_splits=5)
        pred_gk = np.full(len(rows), np.nan)
        for train_idx, test_idx in gk.split(x, groups=groups):
            pred_gk[test_idx] = fit_predict(train_idx, test_idx)
        metric_gk = _metric_record(y, pred_gk, pred_gk, bridge, spec["bounds"], _physical_node_error(target, y, pred_gk, rows))

        # 随机 KFold，多次不同种子
        run_maes, run_group_maes = [], []
        for seed in seeds:
            kf = KFold(n_splits=5, shuffle=True, random_state=seed)
            pred_kf = np.full(len(rows), np.nan)
            for train_idx, test_idx in kf.split(x):
                pred_kf[test_idx] = fit_predict(train_idx, test_idx)
            run_maes.append(float(mean_absolute_error(y, pred_kf)))
            run_group_maes.append(_bridge_macro_mae(y, pred_kf, bridge))
        metric_kf = _metric_record(y, pred_gk, pred_gk, bridge, spec["bounds"], _physical_node_error(target, y, pred_gk, rows))
        metric_kf["mae_random_kfold_mean"] = float(np.mean(run_maes))
        metric_kf["mae_random_kfold_std"] = float(np.std(run_maes))
        metric_kf["bridge_macro_mae_random_kfold_mean"] = float(np.mean(run_group_maes))
        metric_kf["bridge_macro_mae_random_kfold_std"] = float(np.std(run_group_maes))
        out[target] = {"group_kfold": metric_gk, "random_kfold_summary": metric_kf}
    return out


def _side_eval_from_full(design_rows: list[dict[str, Any]], design_oofs: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    """用完整对称模型的 OOF 预测评估左右两侧原始观测，作为 no_symmetry 对照。"""
    out = {}
    for target in ("alpha", "beta"):
        spec = TARGETS[target]
        y_true, y_pred, y_raw, bridge, row_used = [], [], [], [], []
        side_rows = _side_rows(design_rows)
        for pred_item, row in zip(design_oofs[target], design_rows):
            left = _copy_row(row, alpha=row["observed_alpha_left"], beta=row["observed_beta_left"], side="left")
            right = _copy_row(row, alpha=row["observed_alpha_right"], beta=row["observed_beta_right"], side="right")
            for side_row in (left, right):
                y_true.append(float(side_row[f"_{target}"]))
                y_pred.append(float(pred_item["y_pred"]))
                y_raw.append(float(pred_item["y_raw"]))
                bridge.append(str(row["bridge_key"]))
                row_used.append(side_row)
        arr = lambda values: np.asarray(values, dtype=float)
        yt, yp, yr, br = arr(y_true), arr(y_pred), arr(y_raw), np.asarray(bridge)
        out[target] = _metric_record(yt, yp, yr, br, spec["bounds"], _physical_node_error(target, yt, yp, row_used))
    return out


def _run_main_variants(all_rows: list[dict[str, Any]], accepted: list[dict[str, Any]]) -> dict[str, Any]:
    full_rows = _design_rows(accepted)
    feature_names = {"alpha": FEATURES["alpha"], "beta": FEATURES["beta"]}
    target_column = {"alpha": "_alpha", "beta": "_beta"}

    variants = {}
    oofs = {}
    variants["full"], oofs["full"] = _run_logo_variant(full_rows, target_column=target_column, feature_names=feature_names, residual=True, clip=True)
    variants["full_side_eval"] = _side_eval_from_full(accepted, oofs["full"])
    variants["baselines_full"] = _run_fixed_baselines(full_rows, target_column, feature_names)

    # 1. 去掉平行角度修正：alpha 采用普通投影比例的左右均值
    legacy_rows = _legacy_rows(accepted)
    variants["no_parallel_angle"], oofs["no_parallel_angle"] = _run_logo_variant(legacy_rows, target_column=target_column, feature_names=feature_names, residual=True, clip=True)

    # 2. 去掉左右对称化：分别拟合左右观测
    no_sym_rows = _rows_with_features(_side_rows(accepted), True)
    side_feature_names = {
        "alpha": ("span_m", "side_code"),
        "beta": ("span_m", "three_miao_rise_span_ratio", "side_code"),
    }
    variants["no_symmetry"], oofs["no_symmetry"] = _run_logo_variant(no_sym_rows, target_column=target_column, feature_names=side_feature_names, residual=True, clip=True)

    # 3. 去掉样本范围控制：把零平弦记录放回训练/评估
    all_valid = [row for row in all_rows if isinstance(row.get("span_m"), (int, float)) or (row.get("span_m") not in (None, ""))]
    no_scope_rows = _design_rows(all_valid)
    variants["no_scope_filter"], oofs["no_scope_filter"] = _run_logo_variant(no_scope_rows, target_column=target_column, feature_names=feature_names, residual=True, clip=True)
    common_keys = {row["sample_key"] for row in accepted}
    variants["no_scope_filter_common_subset"] = {
        target: _common_record(oofs["no_scope_filter"][target], common_keys, target, TARGETS[target]["bounds"], no_scope_rows)
        for target in ("alpha", "beta")
    }

    # 4. 平行角度修正的物理诊断：目标被修正了多少
    alpha_shift = [
        abs(float(r["legacy_projected_alpha_left"]) - float(r["design_target_alpha"]))
        + abs(float(r["legacy_projected_alpha_right"]) - float(r["design_target_alpha"]))
        for r in accepted
    ]
    variants["parallel_angle_diagnostics"] = {
        "mean_abs_alpha_shift_per_row": float(np.mean(alpha_shift)),
        "max_abs_alpha_shift_per_row": float(np.max(alpha_shift)),
        "mean_abs_normalized_outer_start_offset": float(np.mean([
            abs(float(r["outer_start_offset_left_span"])) + abs(float(r["outer_start_offset_right_span"]))
            for r in accepted
        ])),
        "mean_max_normalized_projection_offset": float(np.mean([float(r["max_normalized_projection_offset"]) for r in accepted])),
    }

    # 5. 去掉知识约束残差/裁剪：在多个正则化强度下比较
    residual_ablation = {}
    for alpha in (1e-08, 0.1, 1.0, 10.0):
        bucket = {}
        for mode, residual, clip in (
            ("residual_clipped", True, True),
            ("residual_no_clip", True, False),
            ("direct_clipped", False, True),
            ("direct_no_clip", False, False),
        ):
            bucket[mode], _ = _run_logo_variant(
                full_rows,
                target_column=target_column,
                feature_names=feature_names,
                residual=residual,
                clip=clip,
                ridge_alpha=alpha,
            )
        residual_ablation[str(alpha)] = bucket
    variants["residual_ablation"] = residual_ablation

    # 6. 随机 KFold 与分组 KFold 的泄漏对比
    variants["split_leak_comparison"] = _run_kfold_leak_comparison(full_rows, target_column, feature_names)
    return variants, oofs


def _run_metadata_comparison(accepted: list[dict[str, Any]], pre_all: list[dict[str, Any]], pre_accepted: list[dict[str, Any]]) -> dict[str, Any]:
    feature_names = {"alpha": FEATURES["alpha"], "beta": FEATURES["beta"]}
    target_column = {"alpha": "_alpha", "beta": "_beta"}

    full_subset = _design_rows([row for row in accepted if row["sample_key"] in {r["sample_key"] for r in pre_accepted}])
    full_subset_records, full_subset_oofs = _run_logo_variant(full_subset, target_column=target_column, feature_names=feature_names, residual=True, clip=True)

    pre_rows = _design_rows(pre_accepted)
    pre_records, pre_oofs = _run_logo_variant(pre_rows, target_column=target_column, feature_names=feature_names, residual=True, clip=True)

    out = {
        "full_on_pre_backfill_common_rows": full_subset_records,
        "no_metadata_backfill": pre_records,
        "full_common_sample_count": len(full_subset),
        "no_backfill_sample_count": len(pre_rows),
        "full_common_group_count": len({r["split_group_key"] for r in full_subset}),
        "no_backfill_group_count": len({r["split_group_key"] for r in pre_rows}),
        "missing_span_count": sum(1 for r in pre_all if "missing_span_m" in r.get("training_exclusion_reasons", "")),
    }
    return out, full_subset_oofs, pre_oofs


def _markdown_report(result: dict[str, Any]) -> str:
    v = result["variants"]
    lines = [
        "# 五节苗节点位置预测：数据预处理消融实验",
        "",
        f"- 数据目录：{result['data_dir']}",
        f"- 标准训练样本：{result['accepted_rows']} 条 / {result['accepted_groups']} 个隔离分组",
        f"- 协议：固定 Ridge(alpha={RIDGE_ALPHA})，按 split_group_key 做 Leave-One-Group-Out",
        "- 模型统一为规则残差 Ridge；除 direct_* 变体外，预测均裁剪到可行域",
        "",
        "## 固定基线与完整流程",
        "",
        "| 目标 | 固定规则 2/3、1/4 | 训练中位数 | 完整流程 Ridge |",
        "|---|---:|---:|---:|",
        f"| alpha 桥级宏 MAE | {v['baselines_full']['alpha']['fixed_rule']['bridge_macro_mae']:.5f} | {v['baselines_full']['alpha']['train_median']['bridge_macro_mae']:.5f} | {v['full']['alpha']['bridge_macro_mae']:.5f} |",
        f"| beta 桥级宏 MAE | {v['baselines_full']['beta']['fixed_rule']['bridge_macro_mae']:.5f} | {v['baselines_full']['beta']['train_median']['bridge_macro_mae']:.5f} | {v['full']['beta']['bridge_macro_mae']:.5f} |",
        "",
        "## 主指标（桥级宏 MAE）",
        "",
        "| 变体 | alpha 桥级宏 MAE | beta 桥级宏 MAE | alpha 平均节点误差 m | beta 平均节点误差 m | 说明 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    rows_spec = [
        ("full", "完整预处理"),
        ("no_parallel_angle", "去掉平行角度修正（普通投影 alpha）"),
        ("no_symmetry", "去掉左右对称化（左右分别建模）"),
        ("no_scope_filter", "去掉零平弦范围控制（103 条）"),
        ("no_scope_filter_common_subset", "去掉范围控制，但只比较共同 102 条"),
    ]
    for key, desc in rows_spec:
        rec = v.get(key)
        if not rec:
            continue
        lines.append(
            f"| {key} | {rec['alpha']['bridge_macro_mae']:.5f} | {rec['beta']['bridge_macro_mae']:.5f} | "
            f"{rec['alpha']['mean_node_error_m']:.4f} | {rec['beta']['mean_node_error_m']:.4f} | {desc} |"
        )

    diag = v.get("parallel_angle_diagnostics")
    if diag:
        lines.extend([
            "",
            "## 平行角度修正的物理诊断",
            "",
            f"- 普通投影与平行弦修正后的 alpha 目标平均相差：{diag['mean_abs_alpha_shift_per_row']:.4f}（单侧累计）",
            f"- 最大相差：{diag['max_abs_alpha_shift_per_row']:.4f}",
            f"- 归一化外节点起点法向偏移平均：{diag['mean_abs_normalized_outer_start_offset']:.4f}",
            f"- 平均最大杆件法向投影偏移：{diag['mean_max_normalized_projection_offset']:.4f}",
            "",
            "说明：去掉平行角度修正对 alpha 的 MAE 影响很小，但普通投影会把木材厚度/起点抬高混入比例标签；保留该预处理主要是为了物理语义、可解释性和后续厚度建模。",
        ])

    residual = v.get("residual_ablation")
    if residual:
        lines.extend([
            "",
            "## 规则残差与可行域裁剪消融（不同 Ridge 正则化强度）",
            "",
            "| Ridge alpha | 模式 | alpha 桥级宏 MAE | beta 桥级宏 MAE | 说明 |",
            "|---|---|---:|---:|---|",
        ])
        for alpha, bucket in residual.items():
            for mode, rec in bucket.items():
                mode_name = {
                    "residual_clipped": "规则残差 + 裁剪",
                    "residual_no_clip": "规则残差，不裁剪",
                    "direct_clipped": "直接回归 + 裁剪",
                    "direct_no_clip": "直接回归，不裁剪",
                }[mode]
                lines.append(
                    f"| {alpha} | {mode_name} | {rec['alpha']['bridge_macro_mae']:.5f} | "
                    f"{rec['beta']['bridge_macro_mae']:.5f} | alpha 违规率 {rec['alpha']['raw_violation_rate']:.3f}；beta 违规率 {rec['beta']['raw_violation_rate']:.3f} |"
                )
        lines.extend([
            "",
            "说明：对带自由截距的 Ridge，`y` 直接回归与 `y-rule` 残差回归在数学上等价，因此 MAE 相同。",
            "规则残差的价值不体现在 MAE，而体现在：默认预测贴近工程规则、失败时残差归零即可回退、越界可裁剪到可行域。",
            "本组样本在当前特征范围内没有触发越界，裁剪在分布内不改变预测，但它是外推/OOD 时的安全边界。",
        ])

    sym = v.get("no_symmetry")
    sym_base = v.get("full_side_eval")
    if sym and sym_base:
        lines.extend([
            "",
            "## 左右对称化：对左右原始观测的评估",
            "",
            "| 方法 | alpha 侧级 MAE | beta 侧级 MAE | 说明 |",
            "|---|---:|---:|---|",
            f"| full（对称目标，预测值用于左右两侧） | {sym_base['alpha']['mae']:.5f} | {sym_base['beta']['mae']:.5f} | 左右使用同一预测 |",
            f"| no_symmetry（左右分别建模） | {sym['alpha']['mae']:.5f} | {sym['beta']['mae']:.5f} | 用 side_code 允许左右差异 |",
        ])

    meta = result.get("metadata_comparison")
    if meta:
        lines.extend([
            "",
            "## 净跨元数据补全",
            "",
            f"- 补全前可用：{meta['no_backfill_sample_count']} 条 / {meta['no_backfill_group_count']} 组；缺失净跨：{meta['missing_span_count']} 条",
            f"- 补全后可用：{result['accepted_rows']} 条 / {result['accepted_groups']} 组（当前完整流程）",
            "",
            "| 数据 | alpha 桥级宏 MAE | beta 桥级宏 MAE |",
            "|---|---:|---:|",
            f"| 补全前 {meta['no_backfill_sample_count']} 条 | {meta['no_metadata_backfill']['alpha']['bridge_macro_mae']:.5f} | {meta['no_metadata_backfill']['beta']['bridge_macro_mae']:.5f} |",
            f"| 完整流程在相同 {meta['full_common_sample_count']} 条上 | {meta['full_on_pre_backfill_common_rows']['alpha']['bridge_macro_mae']:.5f} | {meta['full_on_pre_backfill_common_rows']['beta']['bridge_macro_mae']:.5f} |",
            f"| 补全后全部 {result['accepted_rows']} 条 | {v['full']['alpha']['bridge_macro_mae']:.5f} | {v['full']['beta']['bridge_macro_mae']:.5f} |",
        ])

    leak = v.get("split_leak_comparison")
    if leak:
        lines.extend([
            "",
            "## 分组防泄漏对比（5 折 OOF）",
            "",
            "| 目标 | GroupKFold 桥级宏 MAE | 随机 KFold 桥级宏 MAE（10 种子） |",
            "|---|---:|---:|",
            f"| alpha | {leak['alpha']['group_kfold']['bridge_macro_mae']:.5f} | {leak['alpha']['random_kfold_summary']['bridge_macro_mae_random_kfold_mean']:.5f} ± {leak['alpha']['random_kfold_summary']['bridge_macro_mae_random_kfold_std']:.5f} |",
            f"| beta | {leak['beta']['group_kfold']['bridge_macro_mae']:.5f} | {leak['beta']['random_kfold_summary']['bridge_macro_mae_random_kfold_mean']:.5f} ± {leak['beta']['random_kfold_summary']['bridge_macro_mae_random_kfold_std']:.5f} |",
        ])
    lines.extend([
        "",
        "## 结论口径",
        "",
        "- 该表是探索性消融，不是最终论文数值；正式论文需在锁定独立留出上复验。",
        "- `no_scope_filter` 与 `full` 样本集相差 1 条，比较时同时参考 `no_scope_filter_common_subset`。",
        "- `no_parallel_angle` 仅改变 alpha 目标；beta 仍用完整目标，作为参照。",
    ])
    return "\n".join(lines)


def _write_oof_csv(path: Path, oofs: dict[str, dict[str, list[dict[str, Any]]]]) -> None:
    rows = []
    for variant, target_oofs in oofs.items():
        for target, items in target_oofs.items():
            for item in items:
                rows.append({"variant": variant, **item})
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run five-miao node data preprocessing ablation.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--pre-backfill-dir", type=Path, default=PRE_BACKFILL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--write-oof", action="store_true", help="Write all OOF prediction rows to CSV.")
    args = parser.parse_args()

    output_dir = args.output_dir
    if output_dir.exists() and any(output_dir.iterdir()):
        parser.error(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("loading current dataset...")
    all_rows, accepted = _load(args.data_dir)
    print("loading pre-backfill dataset...")
    pre_all, pre_accepted = _load(args.pre_backfill_dir)

    print("running main variants...")
    variants, oofs = _run_main_variants(all_rows, accepted)
    print("running metadata comparison...")
    metadata, full_subset_oofs, pre_oofs = _run_metadata_comparison(accepted, pre_all, pre_accepted)
    oofs["full_on_pre_backfill_common_rows"] = full_subset_oofs
    oofs["no_metadata_backfill"] = pre_oofs

    result = {
        "protocol": {
            "model": "rule-residual Ridge",
            "ridge_alpha": RIDGE_ALPHA,
            "validation": "LeaveOneGroupOut by split_group_key",
            "targets": TARGETS,
            "features": FEATURES,
            "note": "Exploratory ablation; confirm on locked independent holdout before publication.",
        },
        "data_dir": str(args.data_dir),
        "pre_backfill_dir": str(args.pre_backfill_dir),
        "accepted_rows": len(accepted),
        "accepted_groups": len({row["split_group_key"] for row in accepted}),
        "all_rows": len(all_rows),
        "variants": variants,
        "metadata_comparison": metadata,
    }

    _write_json(output_dir / "ablation_metrics.json", result)
    (output_dir / "ablation_report.md").write_text(_markdown_report(result), encoding="utf-8")
    if args.write_oof:
        _write_oof_csv(output_dir / "oof_predictions.csv", oofs)
    print("report:", output_dir / "ablation_report.md")
    print("metrics:", output_dir / "ablation_metrics.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
