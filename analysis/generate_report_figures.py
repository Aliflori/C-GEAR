#!/usr/bin/env python3
"""Generate publication-quality C-GEAR report figures from canonical CSV data."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "cgear-matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd


BLUE = "#3569A8"
RED = "#C94F45"
GOLD = "#D99A2B"
GREEN = "#2E8B72"
GRAY = "#707070"
LIGHT_GRAY = "#D9DDE3"
METHOD_COLORS = {"Greedy IncreLoRA": BLUE, "C-GEAR": RED}
METHOD_MARKERS = {"Greedy IncreLoRA": "o", "C-GEAR": "D"}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.7,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": "#E1E4E8",
            "grid.linewidth": 0.55,
            "grid.alpha": 0.9,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def save(fig, output_dir: Path, stem: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{stem}.png"
    fig.savefig(path, dpi=600, bbox_inches="tight", metadata={"Software": "C-GEAR final report pipeline"})
    plt.close(fig)
    return path


def method_workflow(output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(3.45, 2.05))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    labels = [
        ("Greedy\nanchor", BLUE),
        ("Genetic\nexploration", GOLD),
        ("Training-only\ncalibration", GREEN),
        ("LCB score\n$\\bar g-\\lambda\\sigma_g$", GREEN),
        ("Budget/quality\ngates", RED),
        ("Choose $k$\n$\\in\\{0,\\ldots,K\\}$", RED),
        ("Grow rank\nor consolidate", BLUE),
    ]
    top_x = [0.02, 0.265, 0.51, 0.755]
    bottom_x = [0.68, 0.36, 0.04]
    positions = [(x, 0.57) for x in top_x] + [(x, 0.19) for x in bottom_x]
    widths = [0.22] * 4 + [0.28] * 3
    height = 0.22
    for index, ((label, color), (x, y), width) in enumerate(zip(labels, positions, widths)):
        box = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.008,rounding_size=0.016",
            linewidth=1.0,
            edgecolor=color,
            facecolor=colors.to_rgba(color, 0.09),
        )
        ax.add_patch(box)
        ax.text(x + width / 2, y + height / 2, label, ha="center", va="center", fontsize=6.4)
        if index < 3:
            next_x = positions[index + 1][0]
            ax.add_patch(
                FancyArrowPatch(
                    (x + width + 0.002, y + height / 2),
                    (next_x - 0.006, y + height / 2),
                    arrowstyle="-|>",
                    mutation_scale=8,
                    linewidth=0.8,
                    color="#555555",
                )
            )
        elif index == 3:
            ax.add_patch(FancyArrowPatch((0.865, 0.56), (0.82, 0.43), connectionstyle="arc3,rad=-0.25", arrowstyle="-|>", mutation_scale=8, linewidth=0.8, color="#555555"))
        elif index in (4, 5):
            next_x, next_y = positions[index + 1]
            next_w = widths[index + 1]
            ax.add_patch(FancyArrowPatch((x - 0.004, y + height / 2), (next_x + next_w + 0.004, next_y + height / 2), arrowstyle="-|>", mutation_scale=8, linewidth=0.8, color="#555555"))
    ax.text(
        0.5,
        0.91,
        "C-GEAR allocation event",
        ha="center",
        va="center",
        fontsize=9.2,
        fontweight="bold",
    )
    ax.text(
        0.50,
        0.06,
        "$k=0$: no growth; repeated zeros stop allocation",
        ha="center",
        va="center",
        fontsize=6.6,
        color=RED,
    )
    return save(fig, output_dir, "method_workflow")


def accuracy_efficiency(results: pd.DataFrame, output_dir: Path) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(3.45, 4.25), constrained_layout=True)
    seeds = sorted(results.seed.unique())
    lookup = results.set_index(["method", "seed"])

    ax = axes[0]
    for seed in seeds:
        g = lookup.loc[("Greedy IncreLoRA", seed)]
        c = lookup.loc[("C-GEAR", seed)]
        ax.plot([seed - 0.10, seed + 0.10], [g.accuracy_percent, c.accuracy_percent], color="#B8BDC5", lw=0.8)
        ax.scatter(seed - 0.10, g.accuracy_percent, color=BLUE, marker="o", s=24, zorder=3)
        ax.scatter(seed + 0.10, c.accuracy_percent, color=RED, marker="D", s=22, zorder=3)
        ax.text(seed, max(g.accuracy_percent, c.accuracy_percent) + 0.18, f"{c.accuracy_percent-g.accuracy_percent:+.2f}", ha="center", fontsize=6.2, color="#444444")
    for method in ("Greedy IncreLoRA", "C-GEAR"):
        ax.scatter([], [], color=METHOD_COLORS[method], marker=METHOD_MARKERS[method], label=method)
    ax.axhline(results[results.method == "Greedy IncreLoRA"].accuracy_percent.mean(), color=BLUE, lw=0.7, ls=":")
    ax.axhline(results[results.method == "C-GEAR"].accuracy_percent.mean(), color=RED, lw=0.7, ls=":")
    ax.set_title("(a) Matched-seed RTE accuracy", loc="left", fontweight="bold")
    ax.set_xlabel("Seed")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(seeds)
    ax.set_ylim(85.2, 90.2)
    ax.legend(loc="lower right", ncol=2, fontsize=6.3)

    ax = axes[1]
    for seed in seeds:
        g = lookup.loc[("Greedy IncreLoRA", seed)]
        c = lookup.loc[("C-GEAR", seed)]
        x = [g.selected_active_parameters / 1000.0, c.selected_active_parameters / 1000.0]
        y = [g.accuracy_percent, c.accuracy_percent]
        ax.plot(x, y, color="#B8BDC5", lw=0.8, zorder=1)
    for method in ("Greedy IncreLoRA", "C-GEAR"):
        subset = results[results.method == method]
        ax.scatter(
            subset.selected_active_parameters / 1000.0,
            subset.accuracy_percent,
            color=METHOD_COLORS[method],
            marker=METHOD_MARKERS[method],
            s=27,
            label=method,
            zorder=3,
        )
        for row in subset.itertuples():
            ax.annotate(str(row.seed), (row.selected_active_parameters / 1000.0, row.accuracy_percent), xytext=(3, 3), textcoords="offset points", fontsize=5.8, color=METHOD_COLORS[method])
    ax.set_title("(b) Selected-checkpoint trade-off", loc="left", fontweight="bold")
    ax.set_xlabel("Active parameters (thousands)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(85.2, 90.2)
    ax.legend(loc="lower right", ncol=2, fontsize=6.3)
    return save(fig, output_dir, "accuracy_efficiency")


def _trajectory_states(rank_df: pd.DataFrame) -> pd.DataFrame:
    keep_roles = {"initial_trajectory", "trajectory", "final_trajectory"}
    states = rank_df[rank_df.state_role.isin(keep_roles)].copy()
    states = states[states.event_type.isin(["run_start", "allocation_event", "run_end"])]
    states = states.sort_values(["method", "seed", "run_segment", "global_step", "event_type"])
    return states.drop_duplicates(["method", "seed", "run_segment", "global_step"], keep="last")


def rank_behavior(rank_df: pd.DataFrame, allocation_df: pd.DataFrame, output_dir: Path) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(3.45, 4.00), constrained_layout=True)
    states = _trajectory_states(rank_df)
    method_map = {"greedy": "Greedy IncreLoRA", "genetic_budgeted_calibrated": "C-GEAR"}
    ax = axes[0]
    for method, label in method_map.items():
        subset = states[states.method == method]
        for seed, seed_rows in subset.groupby("seed"):
            seed_rows = seed_rows.sort_values("global_step")
            ax.step(
                seed_rows.global_step,
                seed_rows.total_active_rank,
                where="post",
                color=METHOD_COLORS[label],
                alpha=0.28,
                linewidth=0.9,
            )
        mean_curve = []
        grid = sorted(subset.global_step.unique())
        for step in grid:
            values = []
            for _, seed_rows in subset.groupby("seed"):
                before = seed_rows[seed_rows.global_step <= step]
                if not before.empty:
                    values.append(float(before.sort_values("global_step").iloc[-1].total_active_rank))
            mean_curve.append(np.mean(values))
        ax.step(grid, mean_curve, where="post", color=METHOD_COLORS[label], linewidth=1.8, label=f"{label} mean")
    ax.set_title("(a) Logged active-rank growth", loc="left", fontweight="bold")
    ax.set_xlabel("Optimization step")
    ax.set_ylabel("Total active rank")
    ax.set_xlim(0, 1950)
    ax.set_ylim(68, 150)
    ax.legend(loc="lower right")

    ax = axes[1]
    alloc = allocation_df[
        (allocation_df.method == "genetic_budgeted_calibrated")
        & (allocation_df.selected_source != "allocation_stopped_consolidation")
    ].copy()
    event_steps = sorted(alloc.global_step.unique())
    seeds = sorted(alloc.seed.unique())
    matrix = np.full((len(seeds), len(event_steps)), np.nan)
    for i, seed in enumerate(seeds):
        for row in alloc[alloc.seed == seed].itertuples():
            matrix[i, event_steps.index(row.global_step)] = row.selected_k
    cmap = colors.ListedColormap(["#F7F7F7", "#F8D7A2", "#F1B466", "#E98A51", "#D95D4D", "#A9323A"])
    cmap.set_bad("#ECEFF2")
    image = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap=cmap, vmin=-0.5, vmax=5.5)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if np.isfinite(value):
                ax.text(j, i, str(int(value)), ha="center", va="center", fontsize=5.6, color="#222222")
    ax.set_title("(b) C-GEAR event size $k$", loc="left", fontweight="bold")
    ax.set_xlabel("Allocation step")
    ax.set_ylabel("Seed")
    tick_indices = list(range(0, len(event_steps), 2))
    ax.set_xticks(tick_indices, [str(event_steps[i]) for i in tick_indices], rotation=45, ha="right")
    ax.set_yticks(range(len(seeds)), seeds)
    cbar = fig.colorbar(image, ax=ax, fraction=0.047, pad=0.02, ticks=range(6))
    cbar.set_label("selected $k$")
    return save(fig, output_dir, "rank_behavior")


def accuracy_by_seed(results: pd.DataFrame, output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.15, 3.7), constrained_layout=True)
    for method in ("Greedy IncreLoRA", "C-GEAR"):
        subset = results[results.method == method].sort_values("seed")
        ax.plot(subset.seed, subset.accuracy_percent, color=METHOD_COLORS[method], marker=METHOD_MARKERS[method], linewidth=1.6, label=method)
        for row in subset.itertuples():
            ax.annotate(f"{row.accuracy_percent:.2f}", (row.seed, row.accuracy_percent), xytext=(0, 7 if method == "C-GEAR" else -12), textcoords="offset points", ha="center", fontsize=6.6, color=METHOD_COLORS[method])
    ax.set_title("RTE selected-checkpoint accuracy across six matched seeds", loc="left", fontweight="bold")
    ax.set_xlabel("Seed")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(sorted(results.seed.unique()))
    ax.legend()
    return save(fig, output_dir, "accuracy_by_seed")


def parameters_by_seed(results: pd.DataFrame, output_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.25), sharey=True, constrained_layout=True)
    seeds = sorted(results.seed.unique())
    x = np.arange(len(seeds))
    width = 0.36
    for ax, prefix, title in (
        (axes[0], "selected", "Selected checkpoint"),
        (axes[1], "final", "Final trajectory (architecture only)"),
    ):
        for index, method in enumerate(("Greedy IncreLoRA", "C-GEAR")):
            subset = results[results.method == method].sort_values("seed")
            values = subset[f"{prefix}_active_parameters"] / 1000.0
            ax.bar(x + (index - 0.5) * width, values, width, color=METHOD_COLORS[method], label=method)
        ax.set_title(title, fontweight="bold")
        ax.set_xticks(x, seeds)
        ax.set_xlabel("Seed")
        ax.set_ylim(0, 1010)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Active parameters (thousands)")
    axes[0].legend(loc="upper left")
    return save(fig, output_dir, "active_parameters_by_seed")


def final_rank_by_layer(module_df: pd.DataFrame, output_dir: Path) -> Path:
    final = module_df[module_df.state_role == "final_trajectory"].copy()
    layer = final.groupby(["method", "seed", "transformer_layer"], as_index=False).active_rank.sum()
    mean = layer.groupby(["method", "transformer_layer"], as_index=False).active_rank.mean()
    fig, ax = plt.subplots(figsize=(7.15, 3.7), constrained_layout=True)
    x = np.arange(12)
    width = 0.38
    for index, (method, label) in enumerate((("greedy", "Greedy IncreLoRA"), ("genetic_budgeted_calibrated", "C-GEAR"))):
        values = mean[mean.method == method].sort_values("transformer_layer").active_rank
        ax.bar(x + (index - 0.5) * width, values, width, color=METHOD_COLORS[label], label=label)
    ax.set_title("Mean final active rank by DeBERTa transformer layer", loc="left", fontweight="bold")
    ax.set_xlabel("Transformer layer")
    ax.set_ylabel("Summed active rank across six LoRA modules")
    ax.set_xticks(x, range(12))
    ax.legend()
    return save(fig, output_dir, "final_rank_by_layer")


def family_distribution(module_df: pd.DataFrame, output_dir: Path) -> Path:
    final = module_df[module_df.state_role == "final_trajectory"].copy()
    totals = final.groupby(["method", "seed", "module_family"], as_index=False).active_rank.sum()
    mean = totals.groupby(["method", "module_family"], as_index=False).active_rank.mean()
    families = ["attention_query", "attention_key", "attention_value", "attention_output", "ffn_intermediate", "ffn_output"]
    short = ["Query", "Key", "Value", "Attn out", "FFN in", "FFN out"]
    fig, ax = plt.subplots(figsize=(3.45, 2.30), constrained_layout=True)
    x = np.arange(len(families))
    width = 0.38
    for index, (method, label) in enumerate((("greedy", "Greedy IncreLoRA"), ("genetic_budgeted_calibrated", "C-GEAR"))):
        lookup = dict(zip(mean[mean.method == method].module_family, mean[mean.method == method].active_rank))
        ax.bar(x + (index - 0.5) * width, [lookup[f] for f in families], width, color=METHOD_COLORS[label], label=label)
    ax.set_title("Final rank by module family", loc="left", fontweight="bold")
    ax.set_ylabel("Total active rank across 12 layers")
    ax.set_xticks(x, short, rotation=22, ha="right")
    ax.legend(fontsize=6.2, ncol=2)
    return save(fig, output_dir, "module_family_rank_distribution")


def compact_rank_heatmap(module_df: pd.DataFrame, output_dir: Path, seed: int = 41) -> Path:
    final = module_df[(module_df.state_role == "final_trajectory") & (module_df.seed == seed)].copy()
    families = ["attention_query", "attention_key", "attention_value", "attention_output", "ffn_intermediate", "ffn_output"]
    short = ["Q", "K", "V", "Attn out", "FFN in", "FFN out"]
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.65), constrained_layout=True)
    vmax = int(final.active_rank.max())
    for ax, (method, label) in zip(axes, (("greedy", "Greedy IncreLoRA"), ("genetic_budgeted_calibrated", "C-GEAR"))):
        subset = final[final.method == method]
        pivot = subset.pivot_table(index="transformer_layer", columns="module_family", values="active_rank", aggfunc="sum").reindex(index=range(12), columns=families)
        image = ax.imshow(pivot.values, aspect="auto", cmap="Blues" if method == "greedy" else "Reds", vmin=1, vmax=vmax)
        for i in range(12):
            for j in range(6):
                ax.text(j, i, str(int(pivot.iloc[i, j])), ha="center", va="center", fontsize=6, color="#222222")
        ax.set_title(label, fontweight="bold")
        ax.set_xlabel("Module family")
        ax.set_xticks(range(6), short, rotation=35, ha="right")
        ax.set_yticks(range(12), range(12))
        ax.set_ylabel("Transformer layer")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02, label="active rank")
    fig.suptitle(f"Final non-uniform rank maps (representative seed {seed})", x=0.02, ha="left", fontweight="bold")
    return save(fig, output_dir, f"compact_rank_heatmap_seed{seed}")


def mechanistic_rank_allocation(module_df: pd.DataFrame, output_dir: Path, seed: int = 41) -> Path:
    """Combine aggregate and representative final-rank structure in one figure."""
    final = module_df[module_df.state_role == "final_trajectory"].copy()
    families = ["attention_query", "attention_key", "attention_value", "attention_output", "ffn_intermediate", "ffn_output"]
    family_labels = ["Query", "Key", "Value", "Attn out", "FFN in", "FFN out"]
    heatmap_labels = ["Q", "K", "V", "Attn\nout", "FFN\nin", "FFN\nout"]
    methods = (("greedy", "Greedy IncreLoRA"), ("genetic_budgeted_calibrated", "C-GEAR"))

    # Show growth above the shared rank-one initialization rather than total
    # active rank: this isolates what each allocator actually added. Aggregate
    # only after per-seed family totals are computed so every run contributes
    # equally even though C-GEAR stops at different active ranks.
    initial = module_df[module_df.state_role == "initial_trajectory"]
    totals = final.groupby(["method", "seed", "module_family"], as_index=False).active_rank.sum()
    baselines = initial.groupby(["method", "seed", "module_family"], as_index=False).active_rank.sum()
    growth = totals.merge(
        baselines,
        on=["method", "seed", "module_family"],
        how="inner",
        suffixes=("_final", "_initial"),
        validate="one_to_one",
    )
    growth["added_rank"] = growth.active_rank_final - growth.active_rank_initial
    if (growth.added_rank < 0).any():
        raise ValueError("Final module-family rank cannot be lower than its initialization.")
    stats = growth.groupby(["method", "module_family"], as_index=False).added_rank.agg(["mean", "std"]).reset_index()

    representative = final[final.seed == seed]
    expected_rows = 12 * len(families)
    for method, _ in methods:
        method_rows = representative[representative.method == method]
        if len(method_rows) != expected_rows:
            raise ValueError(
                f"Expected {expected_rows} final module ranks for {method} seed {seed}, "
                f"found {len(method_rows)}."
            )

    fig = plt.figure(figsize=(7.15, 3.25), constrained_layout=True)
    outer = fig.add_gridspec(1, 2, width_ratios=(1.05, 1.72), wspace=0.10)

    # Panel (a): six-seed family growth with variability across matched runs.
    ax = fig.add_subplot(outer[0, 0])
    y = np.arange(len(families))
    height = 0.36
    for index, (method, label) in enumerate(methods):
        method_stats = stats[stats.method == method].set_index("module_family")
        means = method_stats.reindex(families)["mean"].to_numpy()
        deviations = method_stats.reindex(families)["std"].to_numpy()
        offset = (index - 0.5) * height
        ax.barh(
            y + offset,
            means,
            height,
            xerr=deviations,
            color=METHOD_COLORS[label],
            edgecolor="white",
            linewidth=0.35,
            error_kw={"ecolor": "#454545", "elinewidth": 0.65, "capsize": 1.5, "capthick": 0.65},
            label=label,
        )
        for family_y, mean_value in zip(y + offset, means):
            if np.isclose(mean_value, 0.0):
                ax.text(0.25, family_y, "0", va="center", ha="left", fontsize=5.6, fontweight="bold", color=METHOD_COLORS[label])
    ax.set_title("(a) Six-seed mean rank growth by family", loc="left", fontweight="bold", fontsize=7.9, pad=4)
    ax.set_xlabel(r"Added rank above rank-one initialization (mean $\pm$ SD)")
    ax.set_yticks(y, family_labels)
    ax.invert_yaxis()
    upper_rank = int(np.ceil(max((stats["mean"] + stats["std"]).max(), 5) / 5.0) * 5)
    ax.set_xlim(0, upper_rank)
    ax.set_xticks(np.arange(0, upper_rank, 10))
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", fontsize=6.1)

    # Panel (b): the two methods share one color normalization, making their
    # representative per-layer ranks directly comparable.
    heat_grid = outer[0, 1].subgridspec(2, 1, hspace=0.12)
    heat_axes = [fig.add_subplot(heat_grid[row, 0]) for row in range(2)]
    vmax = int(representative.active_rank.max())
    norm = colors.Normalize(vmin=1, vmax=vmax)
    image = None
    for row, (axis, (method, label)) in enumerate(zip(heat_axes, methods)):
        subset = representative[representative.method == method]
        pivot = (
            subset.pivot_table(
                index="transformer_layer",
                columns="module_family",
                values="active_rank",
                aggfunc="sum",
            )
            .reindex(index=range(12), columns=families)
        )
        if pivot.isna().any().any():
            raise ValueError(f"Incomplete final rank map for {method} seed {seed}.")
        image = axis.imshow(pivot.to_numpy(), aspect="auto", cmap="YlGnBu", norm=norm, interpolation="nearest")
        for layer in range(12):
            for family_index in range(len(families)):
                value = int(pivot.iloc[layer, family_index])
                axis.text(
                    family_index,
                    layer,
                    str(value),
                    ha="center",
                    va="center",
                    fontsize=5.1,
                    color="white" if norm(value) > 0.58 else "#1E252B",
                )
        if row == 0:
            axis.set_title(
                f"(b) Representative rank map (seed {seed})",
                loc="left",
                fontweight="bold",
                fontsize=7.9,
                pad=8,
            )
            axis.tick_params(axis="x", bottom=False, labelbottom=False)
        else:
            axis.set_xticks(range(len(families)), heatmap_labels)
            axis.set_xlabel("Module family", labelpad=1)
        axis.set_yticks(range(12), range(12))
        axis.set_ylabel("Layer", labelpad=2)
        axis.grid(False)
        axis.text(
            1.0,
            1.015,
            label,
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=6.4,
            fontweight="bold",
            color=METHOD_COLORS[label],
        )

    assert image is not None
    cbar = fig.colorbar(image, ax=heat_axes, fraction=0.030, pad=0.018, ticks=range(1, vmax + 1))
    cbar.set_label("Active rank", labelpad=3)
    cbar.ax.tick_params(labelsize=6)
    return save(fig, output_dir, "mechanistic_rank_allocation")


def calibration_history(calibration_df: pd.DataFrame, output_dir: Path, seed: int = 41) -> Path:
    data = calibration_df[(calibration_df.seed == seed) & (calibration_df.method == "genetic_budgeted_calibrated")].copy()
    data = data[pd.to_numeric(data.calibration_gain_lcb, errors="coerce").notna()]
    data["lcb"] = pd.to_numeric(data.calibration_gain_lcb)
    fig, ax = plt.subplots(figsize=(7.15, 3.65), constrained_layout=True)
    other = data[~data.is_selected]
    selected = data[data.is_selected]
    ax.scatter(other.global_step, other.lcb, s=16, color="#AEB4BC", alpha=0.65, label="Calibrated candidate")
    ax.scatter(selected.global_step, selected.lcb, s=38, color=RED, marker="D", edgecolor="white", linewidth=0.5, label="Selected candidate", zorder=3)
    ax.axhline(0, color="#555555", linewidth=0.7)
    ax.set_title(f"C-GEAR conservative calibration scores (seed {seed})", loc="left", fontweight="bold")
    ax.set_xlabel("Allocation step")
    ax.set_ylabel("Calibration LCB ($\\bar g-0.5\\sigma_g$)")
    ax.legend()
    return save(fig, output_dir, f"calibration_history_seed{seed}")


def parse_args() -> argparse.Namespace:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=repository / "final_report" / "data")
    parser.add_argument("--output-dir", type=Path, default=repository / "final_report" / "figures")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_style()
    data_dir, output_dir = args.data_dir.resolve(), args.output_dir.resolve()
    results = pd.read_csv(data_dir / "canonical_results_seeds_41_46.csv")
    rank_df = pd.read_csv(data_dir / "telemetry" / "rank_trajectory.csv")
    module_df = pd.read_csv(data_dir / "telemetry" / "module_rank_trajectory.csv")
    allocation_df = pd.read_csv(data_dir / "telemetry" / "allocation_events.csv")
    calibration_df = pd.read_csv(data_dir / "telemetry" / "calibration_events.csv")
    generated = [
        method_workflow(output_dir),
        accuracy_efficiency(results, output_dir),
        rank_behavior(rank_df, allocation_df, output_dir),
        accuracy_by_seed(results, output_dir),
        parameters_by_seed(results, output_dir),
        final_rank_by_layer(module_df, output_dir),
        family_distribution(module_df, output_dir),
        compact_rank_heatmap(module_df, output_dir, seed=41),
        mechanistic_rank_allocation(module_df, output_dir, seed=41),
        calibration_history(calibration_df, output_dir, seed=41),
    ]
    manifest = {
        "schema_version": "cgear_report_figures.v1",
        "source_data": str(data_dir),
        "paper_figures": [
            "method_workflow.png",
            "accuracy_efficiency.png",
            "rank_behavior.png",
            "mechanistic_rank_allocation.png",
        ],
        "generated_files": [path.name for path in generated],
    }
    (output_dir / "figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Generated {len(generated)} report figures in {output_dir}")


if __name__ == "__main__":
    main()
