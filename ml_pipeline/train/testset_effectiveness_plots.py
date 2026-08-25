# -*- coding: utf-8 -*-
"""五节苗节点预测在独立留出测试集上的效果图。

重点：
- 斜弦节点 alpha：v9 前/后半区条件专家与固定规则、v6 单模型等对比；
- 平弦节点 beta：v6 模型与固定规则、开发集中位数对比；
- MAE / RMSE / R² / q90 / 最大误差 / 物理节点误差。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150

COLORS = {
    "front": "#2f6db3",
    "back": "#d1495b",
    "fixed_rule": "#9aa5b1",
    "v6_single": "#5d8c9e",
    "v6_constrained": "#c9863d",
    "mode_median": "#d8b365",
    "v9": "#27ae60",
    "beta": "#8e6bb1",
}

V6_CSV = Path(r"C:\Users\24982\Desktop\机器学习结果\five_miao_pilot_v6_small_sample\independent_holdout_predictions.csv")
V9_CSV = Path(__file__).resolve().parents[2] / ".review_tmp" / "five_miao_pilot_v9_designer_mode" / "independent_holdout_predictions.csv"
DATA_DIR = Path(r"C:\Users\24982\Desktop\数据集")
PARTITION_CSV = Path(r"C:\Users\24982\Desktop\机器学习结果\five_miao_pilot_v6_small_sample\data_partition.csv")
OUT_DIR = Path(r"C:\Users\24982\Desktop\机器学习结果\testset_effectiveness_plots_v1")


def load_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def load_partition() -> dict[str, str]:
    return {r["sample_key"]: r["partition"] for r in load_csv(PARTITION_CSV)}


def load_geometry_lookup() -> dict[str, dict]:
    from ml_pipeline.prepare.symmetric_targets import collect_annotation_paths
    from ml_pipeline.train.pilot import load_training_rows
    paths = collect_annotation_paths([], [DATA_DIR], "*_five_miao_nodes.json")
    _, accepted = load_training_rows(paths)
    return {
        r["sample_key"]: {
            "rise_ratio": float(r["three_miao_rise_span_ratio"]),
            "alpha": float(r["design_target_alpha"]),
            "beta": float(r["design_target_beta"]),
        }
        for r in accepted
    }


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    error = np.abs(y_true - y_pred)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return {
        "mae": float(np.mean(error)),
        "rmse": float(math.sqrt(np.mean(error ** 2))),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else math.nan,
        "q50": float(np.quantile(error, 0.5)),
        "q90": float(np.quantile(error, 0.9)),
        "q95": float(np.quantile(error, 0.95)),
        "max": float(np.max(error)),
    }


def physical_error(target: str, rows: list[dict], error_key: str, geom: dict[str, dict]) -> np.ndarray:
    values = []
    for r in rows:
        span = float(r["span_m"])
        if target == "alpha":
            length = math.hypot(span / 3.0, span * geom[r["sample_key"]]["rise_ratio"])
        else:
            length = span / 3.0
        values.append(float(r[error_key]) * length)
    return np.asarray(values)


def prepare_data():
    v9_rows = load_csv(V9_CSV)
    v6_rows = load_csv(V6_CSV)
    geom = load_geometry_lookup()
    partition = load_partition()
    v6_by_key = {r["sample_key"]: r for r in v6_rows}
    for r in v9_rows:
        r["_v6_beta_pred"] = v6_by_key[r["sample_key"]]["predicted_beta"]
        r["_v6_beta_abs"] = v6_by_key[r["sample_key"]]["absolute_error_beta"]
        r["observed_beta"] = v6_by_key[r["sample_key"]]["observed_beta"]
    # 开发集 beta 中位数
    dev_beta = np.asarray([
        geom[k]["beta"] for k, part in partition.items()
        if part == "development" and k in geom
    ])
    beta_dev_median = float(np.median(dev_beta))
    alpha_dev_median = {}
    for mode in ("front_half", "back_half"):
        vals = [geom[k]["alpha"] for k, part in partition.items() if part == "development" and k in geom and ((geom[k]["alpha"] < 0.5) == (mode == "front_half"))]
        alpha_dev_median[mode] = float(np.median(vals))
    return v9_rows, geom, beta_dev_median, alpha_dev_median


def _save(fig, out: Path, stem: str) -> None:
    fig.savefig(out / f"{stem}.png", bbox_inches="tight")
    fig.savefig(out / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _annotate_bar(ax, bars, fmt=".3f", fontsize=8):
    for bar in bars:
        ax.annotate(f"{bar.get_height():{fmt}}", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 2), textcoords="offset points", ha="center", fontsize=fontsize)


def _alpha_method_predictions(rows: list[dict]):
    return {
        "fixed_rule": np.asarray([float(r["fixed_rule_predicted_alpha"]) for r in rows]),
        "v6_single": np.asarray([float(r["v6_single_model_predicted_alpha"]) for r in rows]),
        "v6_constrained": np.asarray([float(r["v6_mode_constrained_predicted_alpha"]) for r in rows]),
        "mode_median": np.asarray([float(r["mode_train_median_predicted_alpha"]) for r in rows]),
        "v9": np.asarray([float(r["predicted_alpha"]) for r in rows]),
    }


def fig_alpha_partitioned(rows: list[dict], geom: dict, out: Path) -> None:
    y = np.asarray([float(r["observed_alpha"]) for r in rows])
    preds = _alpha_method_predictions(rows)
    modes = np.asarray([r["derived_design_mode"] for r in rows])
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.0))

    ax = axes[0][0]
    for mode, color, label in (("front_half", COLORS["front"], "前半区"), ("back_half", COLORS["back"], "后半区")):
        mask = modes == mode
        ax.scatter(y[mask], preds["v9"][mask], s=28, color=color, edgecolor="white", linewidth=0.6, label=label)
    lo, hi = min(y.min(), preds["v9"].min()), max(y.max(), preds["v9"].max())
    ax.plot([lo, hi], [lo, hi], ls="--", color="#333333", lw=1.0)
    ax.set_xlabel("观测 alpha（斜弦节点比例）")
    ax.set_ylabel("v9 分区条件专家预测 alpha")
    ax.set_title("(a) 测试集观测 vs 预测（按前/后半区着色）")
    ax.legend(frameon=False)
    ax.grid(ls=":", alpha=0.35)
    ax.set_aspect("equal", adjustable="box")

    ax = axes[0][1]
    order = np.argsort(np.abs(y - preds["v9"]))
    x = np.arange(len(rows))
    style = {"marker": "o", "ms": 3.5, "lw": 0.9}
    ax.plot(x, np.abs(y - preds["fixed_rule"])[order], label="固定规则", color=COLORS["fixed_rule"], **style)
    ax.plot(x, np.abs(y - preds["v6_single"])[order], label="v6 单模型", color=COLORS["v6_single"], **style)
    ax.plot(x, np.abs(y - preds["v9"])[order], label="v9 分区专家", color=COLORS["v9"], **style)
    ax.set_xlabel("按 v9 误差排序的测试样本")
    ax.set_ylabel("绝对误差 |Δalpha|")
    ax.set_title("(b) 测试样本绝对误差排序")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(ls=":", alpha=0.35)

    ax = axes[1][0]
    methods = ["fixed_rule", "v6_single", "v6_constrained", "mode_median", "v9"]
    labels = ["固定规则", "v6 单模型", "v6 区间约束", "分区中位数", "v9 分区专家"]
    xpos = np.arange(2)
    width = 0.15
    for offset, (method, label, color) in enumerate(zip(methods, labels, [COLORS["fixed_rule"], COLORS["v6_single"], COLORS["v6_constrained"], COLORS["mode_median"], COLORS["v9"]])):
        values = [metrics(y[modes == mode], preds[method][modes == mode])["mae"] for mode in ("front_half", "back_half")]
        bars = ax.bar(xpos + (offset - 2) * width, values, width, label=label, color=color)
        _annotate_bar(ax, bars)
    ax.set_xticks(xpos, ["前半区", "后半区"])
    ax.set_ylabel("测试集 MAE")
    ax.set_title("(c) 分前/后半区的 alpha 测试集 MAE")
    ax.grid(axis="y", ls=":", alpha=0.35)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1][1]
    metric_names = ["MAE", "RMSE", "q90"]
    xpos = np.arange(3)
    width = 0.15
    for offset, (method, label, color) in enumerate(zip(methods, labels, [COLORS["fixed_rule"], COLORS["v6_single"], COLORS["v6_constrained"], COLORS["mode_median"], COLORS["v9"]])):
        rec = metrics(y, preds[method])
        values = [rec["mae"], rec["rmse"], rec["q90"]]
        bars = ax.bar(xpos + (offset - 2) * width, values, width, label=label, color=color)
        _annotate_bar(ax, bars)
    ax.set_xticks(xpos, metric_names)
    ax.set_ylabel("误差值")
    ax.set_title("(d) 测试集总体误差指标")
    ax.grid(axis="y", ls=":", alpha=0.35)
    ax.legend(frameon=False, fontsize=8)

    fig.suptitle("斜弦节点 alpha：测试集分区预测效果", y=0.995)
    _save(fig, out, "testset_alpha_partitioned_prediction")


def _beta_predictions(rows: list[dict], dev_median: float):
    y = np.asarray([float(r["observed_beta"]) for r in rows])
    return y, {
        "fixed_rule": np.full(len(rows), 0.25),
        "dev_median": np.full(len(rows), dev_median),
        "v6": np.asarray([float(r["_v6_beta_pred"]) for r in rows]),
    }


def fig_beta_flat_chord(rows: list[dict], geom: dict, dev_median: float, out: Path) -> None:
    y, preds = _beta_predictions(rows, dev_median)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.6))

    ax = axes[0][0]
    ax.scatter(y, preds["v6"], s=28, color=COLORS["beta"], edgecolor="white", linewidth=0.6)
    lo, hi = min(y.min(), preds["v6"].min()), max(y.max(), preds["v6"].max())
    ax.plot([lo, hi], [lo, hi], ls="--", color="#333333", lw=1.0)
    rec = metrics(y, preds["v6"])
    ax.text(0.04, 0.94, f"MAE = {rec['mae']:.4f}\nR² = {rec['r2']:.2f}", transform=ax.transAxes, va="top", fontsize=10)
    ax.set_xlabel("观测 beta（平弦节点比例）")
    ax.set_ylabel("v6 模型预测 beta")
    ax.set_title("(a) 测试集平弦节点观测 vs 预测")
    ax.grid(ls=":", alpha=0.35)
    ax.set_aspect("equal", adjustable="box")

    ax = axes[0][1]
    order = np.argsort(np.abs(y - preds["v6"]))
    x = np.arange(len(rows))
    style = {"marker": "o", "ms": 3.5, "lw": 0.9}
    ax.plot(x, np.abs(y - preds["fixed_rule"])[order], label="固定规则 0.25", color=COLORS["fixed_rule"], **style)
    ax.plot(x, np.abs(y - preds["dev_median"])[order], label="开发集中位数", color=COLORS["mode_median"], **style)
    ax.plot(x, np.abs(y - preds["v6"])[order], label="v6 模型", color=COLORS["beta"], **style)
    ax.set_xlabel("按 v6 误差排序的测试样本")
    ax.set_ylabel("绝对误差 |Δbeta|")
    ax.set_title("(b) 测试样本绝对误差排序")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(ls=":", alpha=0.35)

    ax = axes[1][0]
    methods = ["fixed_rule", "dev_median", "v6"]
    labels = ["固定规则", "开发集中位数", "v6 模型"]
    colors = [COLORS["fixed_rule"], COLORS["mode_median"], COLORS["beta"]]
    metric_names = ["MAE", "RMSE", "q90", "Max"]
    xpos = np.arange(4)
    width = 0.24
    for offset, (method, label, color) in enumerate(zip(methods, labels, colors)):
        rec = metrics(y, preds[method])
        values = [rec["mae"], rec["rmse"], rec["q90"], rec["max"]]
        bars = ax.bar(xpos + (offset - 1) * width, values, width, label=label, color=color)
        _annotate_bar(ax, bars)
    ax.set_xticks(xpos, metric_names)
    ax.set_ylabel("误差值")
    ax.set_title("(c) 测试集误差指标")
    ax.grid(axis="y", ls=":", alpha=0.35)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1][1]
    phys = {}
    for method in methods:
        err = np.abs(y - preds[method])
        length = np.asarray([float(r["span_m"]) / 3.0 for r in rows])
        phys[method] = err * length
    labels = ["固定规则", "开发集中位数", "v6 模型"]
    values = [float(np.mean(phys[m])) for m in methods]
    bars = ax.bar(labels, values, color=colors)
    _annotate_bar(ax, bars, fmt=".3f")
    ax.set_ylabel("平均平弦节点位置误差 (m)")
    ax.set_title("(d) 测试集折算到平弦节点的物理误差")
    ax.grid(axis="y", ls=":", alpha=0.35)

    fig.suptitle("平弦节点 beta：测试集预测效果", y=0.995)
    _save(fig, out, "testset_beta_flat_chord_prediction")


def fig_summary_heatmap(rows: list[dict], geom: dict, dev_median: float, out: Path) -> None:
    y_alpha = np.asarray([float(r["observed_alpha"]) for r in rows])
    preds_alpha = _alpha_method_predictions(rows)
    y_beta = np.asarray([float(r["observed_beta"]) for r in rows])
    preds_beta = {
        "fixed_rule": np.full(len(rows), 0.25),
        "dev_median": np.full(len(rows), dev_median),
        "v6": np.asarray([float(r["_v6_beta_pred"]) for r in rows]),
    }

    alpha_methods = [("fixed_rule", "固定规则"), ("v6_single", "v6 单模型"), ("v6_constrained", "v6 区间约束"), ("mode_median", "分区中位数"), ("v9", "v9 分区专家")]
    beta_methods = [("fixed_rule", "固定规则"), ("dev_median", "开发集中位数"), ("v6", "v6 模型")]

    def records(y, preds, methods):
        return [metrics(y, preds[key]) for key, _ in methods]

    alpha_recs = records(y_alpha, preds_alpha, alpha_methods)
    beta_recs = records(y_beta, preds_beta, beta_methods)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    for ax, recs, methods, title in (
        (axes[0], alpha_recs, alpha_methods, "alpha 斜弦节点"),
        (axes[1], beta_recs, beta_methods, "beta 平弦节点"),
    ):
        metric_names = ["MAE", "RMSE", "q90", "Max"]
        values = np.asarray([[rec["mae"], rec["rmse"], rec["q90"], rec["max"]] for rec in recs])
        im = ax.imshow(values, cmap="YlGnBu", aspect="auto")
        ax.set_xticks(np.arange(len(metric_names)), metric_names)
        ax.set_yticks(np.arange(len(methods)), [label for _, label in methods])
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                ax.text(j, i, f"{values[i, j]:.4f}", ha="center", va="center", color="#111111", fontsize=8)
        ax.set_title(f"{title} 测试集指标")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("独立留出测试集能力总览", y=1.00)
    _save(fig, out, "testset_metric_heatmap")


def fig_residual_distributions(rows: list[dict], geom: dict, dev_median: float, out: Path) -> None:
    y_alpha = np.asarray([float(r["observed_alpha"]) for r in rows])
    preds_alpha = _alpha_method_predictions(rows)
    modes = np.asarray([r["derived_design_mode"] for r in rows])
    y_beta = np.asarray([float(r["observed_beta"]) for r in rows])
    preds_beta = {
        "fixed_rule": np.full(len(rows), 0.25),
        "dev_median": np.full(len(rows), dev_median),
        "v6": np.asarray([float(r["_v6_beta_pred"]) for r in rows]),
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.2))
    ax = axes[0][0]
    positions = []
    labels = []
    for index, (method, label) in enumerate([("fixed_rule", "固定规则"), ("v6_single", "v6 单模型"), ("v6_constrained", "v6 区间约束"), ("mode_median", "分区中位数"), ("v9", "v9 分区专家")]):
        err = np.abs(y_alpha - preds_alpha[method])
        parts = ax.violinplot([err], positions=[index], showmeans=False, showmedians=True, widths=0.75)
        for body in parts["bodies"]:
            body.set_facecolor([COLORS["fixed_rule"], COLORS["v6_single"], COLORS["v6_constrained"], COLORS["mode_median"], COLORS["v9"]][index])
            body.set_alpha(0.55)
        parts["cmedians"].set_color("#333333")
        positions.append(index)
        labels.append(label)
    ax.set_xticks(positions, labels, fontsize=8)
    ax.set_ylabel("测试集 |Δalpha| 分布")
    ax.set_title("(a) alpha 斜弦节点误差分布")
    ax.grid(axis="y", ls=":", alpha=0.35)

    ax = axes[0][1]
    positions = []
    labels = []
    for index, (method, label) in enumerate([("fixed_rule", "固定规则"), ("dev_median", "开发集中位数"), ("v6", "v6 模型")]):
        err = np.abs(y_beta - preds_beta[method])
        parts = ax.violinplot([err], positions=[index], showmeans=False, showmedians=True, widths=0.75)
        for body in parts["bodies"]:
            body.set_facecolor([COLORS["fixed_rule"], COLORS["mode_median"], COLORS["beta"]][index])
            body.set_alpha(0.55)
        parts["cmedians"].set_color("#333333")
        positions.append(index)
        labels.append(label)
    ax.set_xticks(positions, labels, fontsize=8)
    ax.set_ylabel("测试集 |Δbeta| 分布")
    ax.set_title("(b) beta 平弦节点误差分布")
    ax.grid(axis="y", ls=":", alpha=0.35)

    ax = axes[1][0]
    for mode, color, label in (("front_half", COLORS["front"], "前半区"), ("back_half", COLORS["back"], "后半区")):
        mask = modes == mode
        span = np.asarray([float(r["span_m"]) for r in rows])[mask]
        rise_ratio = np.asarray([geom[r["sample_key"]]["rise_ratio"] for r in rows])[mask]
        length = np.sqrt((span / 3.0) ** 2 + (span * rise_ratio) ** 2)
        err = np.abs(y_alpha[mask] - preds_alpha["v9"][mask]) * length
        ax.scatter(span, err, s=26, color=color, edgecolor="white", linewidth=0.5, label=label)
    ax.set_xlabel("净跨 (m)")
    ax.set_ylabel("v9 alpha 节点位置误差 (m)")
    ax.set_title("(c) 测试集斜弦节点物理误差随跨径分布")
    ax.grid(ls=":", alpha=0.35)
    ax.legend(frameon=False)

    ax = axes[1][1]
    span = np.asarray([float(r["span_m"]) for r in rows])
    length = span / 3.0
    err = np.abs(y_beta - preds_beta["v6"]) * length
    ax.scatter(span, err, s=26, color=COLORS["beta"], edgecolor="white", linewidth=0.5)
    ax.set_xlabel("净跨 (m)")
    ax.set_ylabel("v6 beta 节点位置误差 (m)")
    ax.set_title("(d) 测试集平弦节点物理误差随跨径分布")
    ax.grid(ls=":", alpha=0.35)

    fig.suptitle("测试集残差与物理误差分布", y=0.995)
    _save(fig, out, "testset_residual_and_physical_distributions")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate test-set effectiveness figures for five-miao node prediction.")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows, geom, beta_dev_median, _alpha_dev_median = prepare_data()
    fig_alpha_partitioned(rows, geom, args.out_dir)
    fig_beta_flat_chord(rows, geom, beta_dev_median, args.out_dir)
    fig_summary_heatmap(rows, geom, beta_dev_median, args.out_dir)
    fig_residual_distributions(rows, geom, beta_dev_median, args.out_dir)
    report = build_metrics_report(rows, geom, beta_dev_median)
    write_metrics_report(report, args.out_dir)
    print("saved testset figures to", args.out_dir)
    return 0




def build_metrics_report(rows: list[dict], geom: dict, beta_dev_median: float) -> dict:
    y_alpha = np.asarray([float(r["observed_alpha"]) for r in rows])
    preds_alpha = _alpha_method_predictions(rows)
    modes = np.asarray([r["derived_design_mode"] for r in rows])
    y_beta = np.asarray([float(r["observed_beta"]) for r in rows])
    preds_beta = {
        "fixed_rule": np.full(len(rows), 0.25),
        "dev_median": np.full(len(rows), beta_dev_median),
        "v6": np.asarray([float(r["_v6_beta_pred"]) for r in rows]),
    }
    alpha_len = np.asarray([math.hypot(float(r["span_m"]) / 3.0, float(r["span_m"]) * geom[r["sample_key"]]["rise_ratio"]) for r in rows])
    beta_len = np.asarray([float(r["span_m"]) / 3.0 for r in rows])

    def full_metrics(y, pred, length):
        rec = metrics(y, pred)
        rec["mean_node_error_m"] = float(np.mean(np.abs(y - pred) * length))
        rec["max_node_error_m"] = float(np.max(np.abs(y - pred) * length))
        return rec

    alpha_names = {
        "fixed_rule": "固定规则",
        "v6_single": "v6 单模型",
        "v6_constrained": "v6 区间约束",
        "mode_median": "分区中位数",
        "v9": "v9 分区专家",
    }
    beta_names = {"fixed_rule": "固定规则", "dev_median": "开发集中位数", "v6": "v6 模型"}

    report = {"n_test_rows": len(rows), "alpha": {"overall": {}, "by_mode": {}}, "beta": {"overall": {}}}
    for method, label in alpha_names.items():
        rec = full_metrics(y_alpha, preds_alpha[method], alpha_len)
        report["alpha"]["overall"][method] = {"label": label, **rec}
    for mode in ("front_half", "back_half"):
        report["alpha"]["by_mode"][mode] = {}
        mask = modes == mode
        for method, label in alpha_names.items():
            rec = full_metrics(y_alpha[mask], preds_alpha[method][mask], alpha_len[mask])
            report["alpha"]["by_mode"][mode][method] = {"label": label, "n": int(mask.sum()), **rec}
    for method, label in beta_names.items():
        rec = full_metrics(y_beta, preds_beta[method], beta_len)
        report["beta"]["overall"][method] = {"label": label, **rec}
    return report


def write_metrics_report(report: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "testset_metrics_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# 五节苗节点预测独立留出测试集指标",
        "",
        f"- 测试样本：{report['n_test_rows']} 条",
        "",
        "## alpha 斜弦节点（整体）",
        "",
        "| 方法 | MAE | RMSE | R² | q50 | q90 | q95 | Max | 平均节点误差 m | 最大节点误差 m |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method, rec in report["alpha"]["overall"].items():
        lines.append(f"| {rec['label']} | {rec['mae']:.5f} | {rec['rmse']:.5f} | {rec['r2']:.3f} | {rec['q50']:.5f} | {rec['q90']:.5f} | {rec['q95']:.5f} | {rec['max']:.5f} | {rec['mean_node_error_m']:.3f} | {rec['max_node_error_m']:.3f} |")
    lines.extend(["", "## alpha 斜弦节点（按前/后半区）", ""])
    for mode, mode_name in (("front_half", "前半区"), ("back_half", "后半区")):
        lines.append(f"### {mode_name}")
        lines.append("")
        lines.append("| 方法 | n | MAE | RMSE | R² | q90 | 平均节点误差 m | 最大节点误差 m |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for method, rec in report["alpha"]["by_mode"][mode].items():
            lines.append(f"| {rec['label']} | {rec['n']} | {rec['mae']:.5f} | {rec['rmse']:.5f} | {rec['r2']:.3f} | {rec['q90']:.5f} | {rec['mean_node_error_m']:.3f} | {rec['max_node_error_m']:.3f} |")
        lines.append("")
    lines.extend(["## beta 平弦节点（整体）", "", "| 方法 | MAE | RMSE | R² | q50 | q90 | q95 | Max | 平均节点误差 m | 最大节点误差 m |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for method, rec in report["beta"]["overall"].items():
        lines.append(f"| {rec['label']} | {rec['mae']:.5f} | {rec['rmse']:.5f} | {rec['r2']:.3f} | {rec['q50']:.5f} | {rec['q90']:.5f} | {rec['q95']:.5f} | {rec['max']:.5f} | {rec['mean_node_error_m']:.3f} | {rec['max_node_error_m']:.3f} |")
    (out / "testset_metrics_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

if __name__ == "__main__":
    raise SystemExit(main())
