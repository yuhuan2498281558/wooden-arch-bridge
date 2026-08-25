# -*- coding: utf-8 -*-
"""为 preprocessing_ablation.py 的 JSON/CSV 输出生成论文用图表。"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150

COLORS = {
    "alpha": "#2f6db3",
    "beta": "#c44e52",
    "rule": "#9aa5b1",
    "median": "#d8b365",
    "full": "#27ae60",
    "group": "#5d8c9e",
    "random": "#c9863d",
}


def _bar(ax, x, values, width, label, color, fmt=".4f", fontsize=8):
    bars = ax.bar(x, values, width, label=label, color=color)
    for bar, value in zip(bars, values):
        ax.annotate(f"{value:{fmt}}", xy=(bar.get_x() + bar.get_width() / 2, value),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=fontsize)
    return bars


def load_metrics(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def fig_baseline_ablation(d: dict, out: Path) -> None:
    v = d["variants"]
    labels = ["alpha", "beta"]
    rule = [v["baselines_full"][t]["fixed_rule"]["bridge_macro_mae"] for t in labels]
    median = [v["baselines_full"][t]["train_median"]["bridge_macro_mae"] for t in labels]
    full = [v["full"][t]["bridge_macro_mae"] for t in labels]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    ax = axes[0]
    x = np.arange(2)
    width = 0.24
    _bar(ax, x - width, rule, width, "固定规则 2/3、1/4", COLORS["rule"])
    _bar(ax, x, median, width, "训练中位数", COLORS["median"])
    _bar(ax, x + width, full, width, "完整预处理流程", COLORS["full"])
    ax.set_xticks(x, labels)
    ax.set_ylabel("桥级宏 MAE")
    ax.set_title("(a) 完整流程与不建模基线")
    ax.legend(frameon=False)
    ax.grid(axis="y", ls=":", alpha=0.35)
    ax.set_ylim(0, max(rule) * 1.18)

    ax = axes[1]
    variants = ["full", "no_parallel_angle", "no_symmetry", "no_scope_filter", "no_scope_filter_common_subset"]
    names = ["完整\n流程", "去掉平行\n角度修正", "去掉左右\n对称化", "去掉零平弦\n过滤(103)", "共同样本\n比较"]
    alpha_vals = [v[k]["alpha"]["bridge_macro_mae"] for k in variants]
    beta_vals = [v[k]["beta"]["bridge_macro_mae"] for k in variants]
    x = np.arange(len(variants))
    width = 0.35
    _bar(ax, x - width / 2, alpha_vals, width, "alpha", COLORS["alpha"])
    _bar(ax, x + width / 2, beta_vals, width, "beta", COLORS["beta"])
    ax.set_xticks(x, names, fontsize=8.5)
    ax.set_ylabel("桥级宏 MAE")
    ax.set_title("(b) 逐项关闭预处理")
    ax.legend(frameon=False)
    ax.grid(axis="y", ls=":", alpha=0.35)
    ax.set_ylim(0, max(alpha_vals + beta_vals) * 1.18)
    fig.tight_layout()
    fig.savefig(out / "fig1_baseline_and_ablation.png", bbox_inches="tight")
    fig.savefig(out / "fig1_baseline_and_ablation.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_detail_panels(d: dict, out: Path) -> None:
    v = d["variants"]
    meta = d["metadata_comparison"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.2))

    ax = axes[0][0]
    x = np.arange(2)
    full = [v["full_side_eval"][t]["mae"] for t in ("alpha", "beta")]
    nosym = [v["no_symmetry"][t]["mae"] for t in ("alpha", "beta")]
    width = 0.34
    _bar(ax, x - width / 2, full, width, "完整对称流程", COLORS["full"])
    _bar(ax, x + width / 2, nosym, width, "左右分别建模", COLORS["alpha"])
    ax.set_xticks(x, ["alpha", "beta"])
    ax.set_ylabel("左右观测侧级 MAE")
    ax.set_title("(a) 左右对称化")
    ax.grid(axis="y", ls=":", alpha=0.35)
    ax.legend(frameon=False)

    ax = axes[0][1]
    x = np.arange(3)
    groups = ["补全前\n59条", "完整流程\n相同59条", "补全后\n102条"]
    a = [meta["no_metadata_backfill"]["alpha"]["bridge_macro_mae"],
         meta["full_on_pre_backfill_common_rows"]["alpha"]["bridge_macro_mae"],
         v["full"]["alpha"]["bridge_macro_mae"]]
    b = [meta["no_metadata_backfill"]["beta"]["bridge_macro_mae"],
         meta["full_on_pre_backfill_common_rows"]["beta"]["bridge_macro_mae"],
         v["full"]["beta"]["bridge_macro_mae"]]
    width = 0.34
    _bar(ax, x - width / 2, a, width, "alpha", COLORS["alpha"])
    _bar(ax, x + width / 2, b, width, "beta", COLORS["beta"])
    ax.set_xticks(x, groups, fontsize=9)
    ax.set_ylabel("桥级宏 MAE")
    ax.set_title("(b) 净跨元数据补全")
    ax.grid(axis="y", ls=":", alpha=0.35)
    ax.legend(frameon=False)

    ax = axes[1][0]
    x = np.arange(2)
    group = [v["split_leak_comparison"][t]["group_kfold"]["bridge_macro_mae"] for t in ("alpha", "beta")]
    random_mean = [v["split_leak_comparison"][t]["random_kfold_summary"]["bridge_macro_mae_random_kfold_mean"] for t in ("alpha", "beta")]
    random_std = [v["split_leak_comparison"][t]["random_kfold_summary"]["bridge_macro_mae_random_kfold_std"] for t in ("alpha", "beta")]
    width = 0.34
    _bar(ax, x - width / 2, group, width, "GroupKFold", COLORS["group"])
    ax.bar(x + width / 2, random_mean, width, yerr=random_std, capsize=4, label="随机 KFold（10 种子）", color=COLORS["random"], error_kw={"elinewidth": 1})
    ax.set_xticks(x, ["alpha", "beta"])
    ax.set_ylabel("5 折 OOF 桥级宏 MAE")
    ax.set_title("(c) 分组防泄漏对比")
    ax.grid(axis="y", ls=":", alpha=0.35)
    ax.legend(frameon=False)

    ax = axes[1][1]
    x = np.arange(2)
    node = [v["full"][t]["mean_node_error_m"] for t in ("alpha", "beta")]
    node_rule = [v["baselines_full"][t]["fixed_rule"]["mean_node_error_m"] for t in ("alpha", "beta")]
    width = 0.34
    _bar(ax, x - width / 2, node_rule, width, "固定规则", COLORS["rule"])
    _bar(ax, x + width / 2, node, width, "完整流程", COLORS["full"], fmt=".3f")
    ax.set_xticks(x, ["alpha 节点", "beta 节点"])
    ax.set_ylabel("平均节点位置误差 (m)")
    ax.set_title("(d) 折算到节点位置的物理误差")
    ax.grid(axis="y", ls=":", alpha=0.35)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(out / "fig2_preprocessing_contributions.png", bbox_inches="tight")
    fig.savefig(out / "fig2_preprocessing_contributions.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_parallel_angle(d: dict, rows: list[dict], out: Path) -> None:
    diag = d["variants"]["parallel_angle_diagnostics"]
    legacy = np.array([(float(r["legacy_projected_alpha_left"]) + float(r["legacy_projected_alpha_right"])) / 2.0 for r in rows])
    design = np.array([float(r["design_target_alpha"]) for r in rows])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    ax = axes[0]
    ax.scatter(design, legacy, s=18, alpha=0.72, color=COLORS["alpha"], edgecolor="white", linewidth=0.5)
    lo, hi = min(np.min(design), np.min(legacy)), max(np.max(design), np.max(legacy))
    ax.plot([lo, hi], [lo, hi], ls="--", color="#555555", lw=1.0, label="y=x")
    ax.set_xlabel("平行角度修正后的设计 alpha")
    ax.set_ylabel("普通投影 alpha（左右均值）")
    ax.set_title("(a) 两种 alpha 标签的差异")
    ax.legend(frameon=False)
    ax.grid(ls=":", alpha=0.35)
    ax.set_aspect("equal", adjustable="box")

    ax = axes[1]
    labels = ["平均 alpha\n标签移动", "最大 alpha\n标签移动", "平均法向\n起点偏移", "平均最大\n法向投影偏移"]
    values = [
        diag["mean_abs_alpha_shift_per_row"],
        diag["max_abs_alpha_shift_per_row"],
        diag["mean_abs_normalized_outer_start_offset"],
        diag["mean_max_normalized_projection_offset"],
    ]
    bars = ax.bar(labels, values, color=[COLORS["alpha"], COLORS["beta"], COLORS["group"], COLORS["random"]])
    for bar, value in zip(bars, values):
        ax.annotate(f"{value:.4f}", xy=(bar.get_x() + bar.get_width() / 2, value), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)
    ax.set_ylabel("归一化差异")
    ax.set_title("(b) 平行角度修正的物理诊断")
    ax.grid(axis="y", ls=":", alpha=0.35)
    fig.tight_layout()
    fig.savefig(out / "fig3_parallel_angle_diagnostics.png", bbox_inches="tight")
    fig.savefig(out / "fig3_parallel_angle_diagnostics.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_residual_clip(d: dict, out: Path) -> None:
    residual = d["variants"]["residual_ablation"]
    alphas = [float(k) for k in residual]
    modes = ["residual_clipped", "residual_no_clip", "direct_clipped", "direct_no_clip"]
    labels = ["规则残差+裁剪", "规则残差", "直接回归+裁剪", "直接回归"]
    colors = [COLORS["full"], COLORS["group"], COLORS["alpha"], COLORS["beta"]]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for target, ax in zip(("alpha", "beta"), axes):
        for mode, label, color in zip(modes, labels, colors):
            values = [residual[str(a)][mode][target]["bridge_macro_mae"] for a in alphas]
            ax.plot(alphas, values, marker="o", ms=4, label=label, color=color)
        ax.set_xscale("log")
        ax.set_xlabel("Ridge 正则化 alpha")
        ax.set_ylabel(f"{target} 桥级宏 MAE")
        ax.set_title(f"({'ab'[target == 'beta']}) {target}：残差与裁剪消融")
        ax.grid(ls=":", alpha=0.35)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig4_residual_clip_ablation.png", bbox_inches="tight")
    fig.savefig(out / "fig4_residual_clip_ablation.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_oof_scatter(d: dict, csv_path: Path, out: Path) -> None:
    rows_by_variant = {}
    with csv_path.open(encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            rows_by_variant.setdefault(row["variant"], []).append(row)

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.6))
    for ax, target in zip(axes[0], ("alpha", "beta")):
        items = [r for r in rows_by_variant.get("full", []) if r["target"] == target]
        y_true = np.asarray([float(r["y_true"]) for r in items])
        y_pred = np.asarray([float(r["y_pred"]) for r in items])
        ax.scatter(y_true, y_pred, s=20, alpha=0.72, color=COLORS[target], edgecolor="white", linewidth=0.5)
        lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
        ax.plot([lo, hi], [lo, hi], ls="--", color="#555555", lw=1.0)
        ax.set_xlabel(f"{target} 观测值")
        ax.set_ylabel(f"{target} OOF 预测值")
        ax.set_title(f"({'ab'[target == 'beta']}) 完整流程 {target} 预测")
        ax.grid(ls=":", alpha=0.35)
        ax.set_aspect("equal", adjustable="box")

    ax = axes[1][0]
    full_items = [r for r in rows_by_variant.get("full", []) if r["target"] == "alpha"]
    legacy_items = [r for r in rows_by_variant.get("no_parallel_angle", []) if r["target"] == "alpha"]
    full_pred = np.asarray([float(r["y_pred"]) for r in full_items])
    legacy_pred = np.asarray([float(r["y_pred"]) for r in legacy_items])
    ax.scatter(full_pred, legacy_pred, s=20, alpha=0.72, color=COLORS["group"], edgecolor="white", linewidth=0.5)
    lo, hi = min(full_pred.min(), legacy_pred.min()), max(full_pred.max(), legacy_pred.max())
    ax.plot([lo, hi], [lo, hi], ls="--", color="#555555", lw=1.0)
    ax.set_xlabel("完整流程 alpha 预测")
    ax.set_ylabel("去掉平行角度修正的 alpha 预测")
    ax.set_title("(c) 两种 alpha 标签下的 OOF 预测")
    ax.grid(ls=":", alpha=0.35)
    ax.set_aspect("equal", adjustable="box")

    ax = axes[1][1]
    for target, color in (("alpha", COLORS["alpha"]), ("beta", COLORS["beta"])):
        items = [r for r in rows_by_variant.get("full", []) if r["target"] == target]
        error = np.abs(np.asarray([float(r["y_true"]) for r in items]) - np.asarray([float(r["y_pred"]) for r in items]))
        ax.hist(error, bins=18, alpha=0.55, color=color, label=target)
    ax.set_xlabel("绝对误差")
    ax.set_ylabel("样本数")
    ax.set_title("(d) 完整流程 OOF 绝对误差分布")
    ax.grid(axis="y", ls=":", alpha=0.35)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(out / "fig5_oof_predictions.png", bbox_inches="tight")
    fig.savefig(out / "fig5_oof_predictions.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate figures from preprocessing ablation outputs.")
    parser.add_argument("--result-dir", type=Path, default=Path(r"C:\Users\24982\Desktop\机器学习结果\preprocessing_ablation_v3"))
    args = parser.parse_args()
    result_dir = args.result_dir
    metrics_path = result_dir / "ablation_metrics.json"
    oof_path = result_dir / "oof_predictions.csv"
    if not metrics_path.exists():
        parser.error(f"missing metrics json: {metrics_path}")

    from ml_pipeline.prepare.symmetric_targets import collect_annotation_paths
    from ml_pipeline.train.pilot import load_training_rows
    data_dir = Path(load_metrics(metrics_path)["data_dir"])
    paths = collect_annotation_paths([], [data_dir], "*_five_miao_nodes.json")
    _, accepted = load_training_rows(paths)

    fig_dir = result_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    d = load_metrics(metrics_path)
    fig_baseline_ablation(d, fig_dir)
    fig_detail_panels(d, fig_dir)
    fig_parallel_angle(d, accepted, fig_dir)
    fig_residual_clip(d, fig_dir)
    if oof_path.exists():
        fig_oof_scatter(d, oof_path, fig_dir)
    print("figures saved to", fig_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
