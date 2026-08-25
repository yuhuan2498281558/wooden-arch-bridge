# -*- coding: utf-8 -*-
"""Create publication-ready plots for preprocessing_strict_control.py."""
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
plt.rcParams["figure.dpi"] = 180

COLORS = {
    "raw": "#a7afb8",
    "processed": "#2674a6",
    "positive": "#2a8c62",
    "negative": "#c66a4b",
    "reference": "#626b73",
}


def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _annotate_bars(ax, bars, digits: int = 5) -> None:
    for bar in bars:
        value = bar.get_height()
        ax.annotate(
            f"{value:.{digits}f}",
            (bar.get_x() + bar.get_width() / 2.0, value),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )


def figure_mae(metrics: dict, output_dir: Path) -> None:
    splits = ["development_logo", "independent_holdout"]
    split_names = ["开发集 LOOGO", "独立内部留出"]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5))
    for ax, target, panel in zip(axes, ("alpha", "beta"), ("a", "b")):
        raw = [metrics[split][target]["raw_labels"]["bridge_macro_mae"] for split in splits]
        processed = [metrics[split][target]["processed"]["bridge_macro_mae"] for split in splits]
        x = np.arange(len(splits))
        width = 0.34
        raw_bars = ax.bar(x - width / 2, raw, width, label="原始标签基线", color=COLORS["raw"])
        full_bars = ax.bar(x + width / 2, processed, width, label="完整预处理", color=COLORS["processed"])
        _annotate_bars(ax, raw_bars)
        _annotate_bars(ax, full_bars)
        for index, split in enumerate(splits):
            rate = metrics[split][target]["processed_bridge_macro_mae_improvement_rate"]
            ymax = max(raw[index], processed[index])
            label = f"降低 {rate:.1%}" if rate > 1e-8 else "数值等价"
            ax.annotate(label, (index, ymax), xytext=(0, 22), textcoords="offset points", ha="center", fontsize=9)
        ax.set_xticks(x, split_names)
        ax.set_ylabel("桥级宏 MAE")
        ax.set_title(f"({panel}) {target}：严格同条件对照")
        ax.grid(axis="y", ls=":", alpha=0.35)
        ax.legend(frameon=False)
        ax.set_ylim(0, max(raw + processed) * 1.33)
    fig.tight_layout()
    fig.savefig(output_dir / "fig6_strict_raw_vs_preprocessed_mae.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "fig6_strict_raw_vs_preprocessed_mae.pdf", bbox_inches="tight")
    plt.close(fig)


def _load_predictions(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def figure_diagnostics(rows: list[dict], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5))
    ax = axes[0]
    for evaluation, marker, label in (
        ("development_logo", "o", "开发集 LOOGO"),
        ("independent_holdout", "^", "独立内部留出"),
    ):
        items = [row for row in rows if row["target"] == "alpha" and row["evaluation"] == evaluation]
        raw = np.asarray([float(row["raw_averaged_prediction"]) for row in items])
        processed = np.asarray([float(row["processed_prediction"]) for row in items])
        ax.scatter(raw, processed, s=26, alpha=0.72, marker=marker, label=label, edgecolor="white", linewidth=0.5)
    all_alpha = [row for row in rows if row["target"] == "alpha"]
    raw_all = np.asarray([float(row["raw_averaged_prediction"]) for row in all_alpha])
    full_all = np.asarray([float(row["processed_prediction"]) for row in all_alpha])
    lo, hi = min(raw_all.min(), full_all.min()), max(raw_all.max(), full_all.max())
    ax.plot([lo, hi], [lo, hi], ls="--", lw=1.0, color=COLORS["reference"], label="y=x")
    ax.set_xlabel("原始标签模型 alpha 预测")
    ax.set_ylabel("完整预处理 alpha 预测")
    ax.set_title("(a) 平行角度修正对预测的影响")
    ax.grid(ls=":", alpha=0.35)
    ax.legend(frameon=False)
    ax.set_aspect("equal", adjustable="box")

    ax = axes[1]
    items = [row for row in rows if row["target"] == "alpha" and row["evaluation"] == "independent_holdout"]
    delta = np.asarray([
        float(row["raw_absolute_error"]) - float(row["processed_absolute_error"])
        for row in items
    ])
    colors = [COLORS["positive"] if value >= 0 else COLORS["negative"] for value in delta]
    order = np.argsort(delta)
    ax.bar(np.arange(len(delta)), delta[order], color=np.asarray(colors, dtype=object)[order])
    ax.axhline(0, color=COLORS["reference"], lw=1.0)
    ax.set_xlabel("独立内部留出样本（按误差差值排序）")
    ax.set_ylabel("原始标签误差 − 完整预处理误差")
    ax.set_title("(b) alpha 逐样本配对误差变化")
    ax.grid(axis="y", ls=":", alpha=0.35)
    ax.text(0.02, 0.97, "正值：完整预处理误差更小", transform=ax.transAxes, va="top", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "fig7_strict_control_diagnostics.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "fig7_strict_control_diagnostics.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot strict preprocessing control results.")
    parser.add_argument("--result-dir", required=True, type=Path)
    args = parser.parse_args()
    metrics_path = args.result_dir / "strict_control_metrics.json"
    predictions_path = args.result_dir / "strict_control_predictions.csv"
    if not metrics_path.exists() or not predictions_path.exists():
        parser.error("strict control metrics or predictions are missing")
    output_dir = args.result_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_mae(_load_json(metrics_path), output_dir)
    figure_diagnostics(_load_predictions(predictions_path), output_dir)
    print("figures saved to", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
