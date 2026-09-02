"""Lightweight visualization for DyTwin DP+DM simulation outputs."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
matplotlib.rcParams["axes.unicode_minus"] = False
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")


METRIC_NAMES = {
    "rouge_l_f1": "ROUGE-L",
    "bert_score_f1": "BERTScore",
    "embedding_similarity": "Embedding Similarity",
    "llm_semantic": "Semantic Similarity",
    "llm_emotion": "Emotion Tendency",
    "llm_stance": "Stance/Opinion",
    "llm_style": "Expression Style",
    "llm_focus": "Focus Point",
    "llm_average": "Overall Score",
}


def visualize_simulation_results(
    csv_path: Path,
    output_dir: Optional[Path] = None,
    show_plots: bool = False,
    train_size: int = 0,
) -> Path:
    """Generate compact diagnostic figures for one DP+DM simulation file."""
    df = pd.read_csv(csv_path)
    forward_df = df[df["type"] == "转发"].copy()
    if len(forward_df) == 0:
        print("警告：没有转发数据可供可视化")
        return output_dir or csv_path.parent / "visualization"

    if output_dir is None:
        output_dir = csv_path.parent / "visualization"
    output_dir.mkdir(parents=True, exist_ok=True)

    forward_df["datetime"] = pd.to_datetime(forward_df["datetime"])
    forward_df = forward_df.sort_values("datetime").reset_index(drop=True)
    forward_df["seq_num"] = range(1, len(forward_df) + 1)

    basic_metrics = ["rouge_l_f1", "bert_score_f1", "embedding_similarity"]
    llm_metrics = [
        "llm_semantic",
        "llm_emotion",
        "llm_stance",
        "llm_style",
        "llm_focus",
        "llm_average",
    ]
    all_metrics = basic_metrics + llm_metrics

    for col in all_metrics:
        if col in forward_df.columns:
            forward_df[col] = pd.to_numeric(forward_df[col], errors="coerce")

    print(f"\n{'=' * 60}")
    print("开始生成可视化图表...")
    print(f"转发数据条数: {len(forward_df)}")
    print(f"{'=' * 60}\n")

    _plot_metrics(
        forward_df,
        basic_metrics,
        "basic",
        "Basic Text Similarity",
        output_dir,
        show_plots,
    )
    _plot_metrics(
        forward_df,
        llm_metrics,
        "llm_score",
        "LLM Multi-dimensional Scoring",
        output_dir,
        show_plots,
        ylim=(0, 10.5),
    )
    _plot_correlation_heatmap(forward_df, all_metrics, output_dir, show_plots)
    _generate_stats_report(forward_df, basic_metrics, llm_metrics, output_dir)

    print(f"\n{'=' * 60}")
    print(f"可视化完成！图表已保存至: {output_dir}")
    print(f"{'=' * 60}\n")
    return output_dir


def _plot_metrics(
    df: pd.DataFrame,
    metrics: list[str],
    category: str,
    title: str,
    output_dir: Path,
    show: bool,
    ylim: tuple[float, float] | None = None,
) -> None:
    valid_metrics = [metric for metric in metrics if metric in df.columns]
    if not valid_metrics:
        return

    n_metrics = len(valid_metrics)
    nrows, ncols = (1, n_metrics) if n_metrics <= 3 else (2, 3)
    figsize = (6 * ncols, 4.8 * nrows)
    colors = [
        "#4C78A8",
        "#F58518",
        "#54A24B",
        "#E45756",
        "#72B7B2",
        "#B279A2",
    ]

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.array(axes).reshape(-1)

    for i, metric in enumerate(valid_metrics):
        ax = axes[i]
        valid = df[["seq_num", metric]].dropna()
        if len(valid) > 0:
            ax.plot(
                valid["seq_num"],
                valid[metric],
                "-o",
                markersize=3,
                linewidth=1.5,
                alpha=0.85,
                color=colors[i % len(colors)],
            )
            ax.axhline(valid[metric].mean(), color="gray", linestyle="--", linewidth=1, alpha=0.7)
        _style_axis(ax, metric, ylim)

    for j in range(n_metrics, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(f"{title} - Trend", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / f"{category}_trend.png", dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
    print(f"  ✓ 生成: {category}_trend.png")

    if len(df) < 5:
        print(f"  ⚠ 跳过: {category}_rolling.png（数据不足）")
        return

    window_size = min(5, max(3, len(df) // 2))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.array(axes).reshape(-1)

    for i, metric in enumerate(valid_metrics):
        ax = axes[i]
        valid = df[["seq_num", metric]].dropna()
        if len(valid) > 0:
            rolling = valid[metric].rolling(window=window_size, center=True, min_periods=1).mean()
            ax.plot(
                valid["seq_num"],
                rolling,
                "-o",
                markersize=3,
                linewidth=1.5,
                alpha=0.9,
                color=colors[i % len(colors)],
            )
        _style_axis(ax, metric, ylim)

    for j in range(n_metrics, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(f"{title} - Rolling Average (Window={window_size})", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / f"{category}_rolling.png", dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
    print(f"  ✓ 生成: {category}_rolling.png")


def _style_axis(ax: plt.Axes, metric: str, ylim: tuple[float, float] | None) -> None:
    ax.set_xlabel("Forward Sequence")
    ax.set_ylabel(METRIC_NAMES.get(metric, metric))
    ax.set_title(METRIC_NAMES.get(metric, metric), fontsize=11, fontweight="bold")
    if ylim:
        ax.set_ylim(ylim)
    ax.grid(True, alpha=0.3)


def _plot_correlation_heatmap(df: pd.DataFrame, metrics: list[str], output_dir: Path, show: bool) -> None:
    valid_metrics = [metric for metric in metrics if metric in df.columns and df[metric].dropna().shape[0] > 2]
    if len(valid_metrics) < 2:
        print("  ⚠ 跳过: correlation_heatmap.png（数据不足）")
        return

    corr_df = df[valid_metrics].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr_df.values, cmap="RdYlBu_r", aspect="auto", vmin=-1, vmax=1)

    labels = [METRIC_NAMES.get(metric, metric) for metric in valid_metrics]
    ax.set_xticks(range(len(valid_metrics)))
    ax.set_yticks(range(len(valid_metrics)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)

    for i in range(len(valid_metrics)):
        for j in range(len(valid_metrics)):
            val = corr_df.values[i, j]
            color = "white" if abs(val) > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=8)

    plt.colorbar(im, ax=ax, label="Correlation Coefficient")
    ax.set_title("Evaluation Metrics Correlation Heatmap", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_heatmap.png", dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
    print("  ✓ 生成: correlation_heatmap.png")


def _generate_stats_report(
    df: pd.DataFrame,
    basic_metrics: list[str],
    llm_metrics: list[str],
    output_dir: Path,
) -> None:
    lines = [
        "=" * 60,
        "DyTwin DP+DM Evaluation Report",
        "=" * 60,
        f"",
        f"Repost records: {len(df)}",
        f"Time range: {df['datetime'].min()} ~ {df['datetime'].max()}",
        "",
        "-" * 60,
        "Basic Text Similarity",
        "-" * 60,
    ]

    for metric in basic_metrics:
        _append_metric_stats(lines, df, metric)

    lines.extend(["", "-" * 60, "LLM Multi-dimensional Scores", "-" * 60])
    for metric in llm_metrics:
        _append_metric_stats(lines, df, metric)

    lines.append("=" * 60)
    (output_dir / "stats_report.txt").write_text("\n".join(lines), encoding="utf-8")
    print("  ✓ 生成: stats_report.txt")


def _append_metric_stats(lines: list[str], df: pd.DataFrame, metric: str) -> None:
    if metric not in df.columns:
        return
    valid = df[metric].dropna()
    if len(valid) == 0:
        return

    lines.extend(
        [
            "",
            f"{METRIC_NAMES.get(metric, metric)}:",
            f"  mean: {valid.mean():.4f}",
            f"  std: {valid.std():.4f}",
            f"  min: {valid.min():.4f}",
            f"  max: {valid.max():.4f}",
            f"  median: {valid.median():.4f}",
        ]
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m dytwin.visualization <simulation.csv>")
        sys.exit(1)

    csv_file = Path(sys.argv[1])
    if not csv_file.exists():
        print(f"File not found: {csv_file}")
        sys.exit(1)

    visualize_simulation_results(csv_file, show_plots=False)
