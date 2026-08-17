#!/usr/bin/env python3
"""Summarize and plot the canonical Greedy IncreLoRA vs C-GEAR RTE evidence.

The script treats accuracy and selected-checkpoint parameter counts as one
deployable-model observation. Final-trajectory counts are summarized and
plotted separately because the final architecture was not independently
evaluated after Trainer restored the selected best checkpoint.
"""

import argparse
import csv
import json
import math
import os
import statistics
import tempfile
from pathlib import Path


METHODS = ("Greedy IncreLoRA", "C-GEAR")
COLORS = {
    "Greedy IncreLoRA": "#3B6FB6",
    "C-GEAR": "#D05A47",
}
MARKERS = {
    "Greedy IncreLoRA": "o",
    "C-GEAR": "D",
}
REQUIRED_COLUMNS = {
    "method",
    "seed",
    "selected_step",
    "accuracy",
    "correct_predictions",
    "total_examples",
    "selected_active_rank",
    "selected_active_parameters",
    "final_active_rank",
    "final_active_parameters",
    "stop_step",
    "stop_reason",
    "runtime_seconds",
    "source_artifact",
    "notes",
}


def _optional_int(value):
    return None if value.strip() == "" else int(value)


def _mean(values):
    return statistics.mean(values)


def _sample_stdev(values):
    return statistics.stdev(values) if len(values) > 1 else 0.0


def load_results(path):
    """Load and validate the canonical one-row-per-method-and-seed table."""

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS.difference(reader.fieldnames or ())
        if missing:
            raise ValueError("Canonical CSV is missing columns: %s" % sorted(missing))
        rows = []
        for raw in reader:
            row = {
                "method": raw["method"],
                "seed": int(raw["seed"]),
                "selected_step": int(raw["selected_step"]),
                "accuracy": float(raw["accuracy"]),
                "correct_predictions": int(raw["correct_predictions"]),
                "total_examples": int(raw["total_examples"]),
                "selected_active_rank": int(raw["selected_active_rank"]),
                "selected_active_parameters": int(raw["selected_active_parameters"]),
                "final_active_rank": int(raw["final_active_rank"]),
                "final_active_parameters": int(raw["final_active_parameters"]),
                "stop_step": _optional_int(raw["stop_step"]),
                "stop_reason": raw["stop_reason"].strip() or None,
                "runtime_seconds": float(raw["runtime_seconds"]),
                "source_artifact": raw["source_artifact"],
                "notes": raw["notes"],
            }
            if row["method"] not in METHODS:
                raise ValueError("Unexpected method: %s" % row["method"])
            if not 0.0 <= row["accuracy"] <= 1.0:
                raise ValueError("Accuracy is outside [0, 1] for %s" % row)
            if row["total_examples"] <= 0:
                raise ValueError("total_examples must be positive.")
            derived_correct = row["accuracy"] * row["total_examples"]
            if not math.isclose(
                derived_correct,
                row["correct_predictions"],
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise ValueError(
                    "correct_predictions does not match accuracy * total_examples "
                    "for %s seed %s." % (row["method"], row["seed"])
                )
            for field in (
                "selected_step",
                "selected_active_rank",
                "selected_active_parameters",
                "final_active_rank",
                "final_active_parameters",
            ):
                if row[field] <= 0:
                    raise ValueError("%s must be positive for %s" % (field, row))
            rows.append(row)

    identities = [(row["method"], row["seed"]) for row in rows]
    if len(set(identities)) != len(identities):
        raise ValueError("Canonical CSV contains duplicate method/seed rows.")
    seeds_by_method = {
        method: {row["seed"] for row in rows if row["method"] == method}
        for method in METHODS
    }
    if any(not seeds for seeds in seeds_by_method.values()):
        raise ValueError("Both methods must be present in the canonical CSV.")
    if len({tuple(sorted(seeds)) for seeds in seeds_by_method.values()}) != 1:
        raise ValueError("Methods do not have matched seed sets.")
    return sorted(rows, key=lambda row: (row["seed"], METHODS.index(row["method"])))


def load_trajectory(path):
    """Load exact logged C-GEAR active-rank states; no interpolation is added."""

    with path.open(newline="", encoding="utf-8") as handle:
        rows = []
        for raw in csv.DictReader(handle):
            rows.append(
                {
                    "seed": int(raw["seed"]),
                    "step": int(raw["step"]),
                    "active_rank": int(raw["active_rank"]),
                    "event_rank": int(raw["event_rank"]),
                    "event_source": raw["event_source"],
                    "is_consolidation_boundary": raw[
                        "is_consolidation_boundary"
                    ].lower()
                    == "true",
                }
            )
    for seed in sorted({row["seed"] for row in rows}):
        seed_rows = sorted(
            (row for row in rows if row["seed"] == seed),
            key=lambda row: (row["step"], row["is_consolidation_boundary"]),
        )
        states = [row for row in seed_rows if not row["is_consolidation_boundary"]]
        if not states or states[0]["step"] != 0 or states[0]["event_source"] != "initial_rank":
            raise ValueError("Seed %s trajectory has no step-zero initial rank." % seed)
        previous_rank = states[0]["active_rank"]
        previous_step = states[0]["step"]
        for row in states[1:]:
            if row["step"] < previous_step:
                raise ValueError("Seed %s trajectory steps are not monotonic." % seed)
            if row["event_source"] == "final_trajectory":
                if row["active_rank"] != previous_rank:
                    raise ValueError("Seed %s final trajectory rank is inconsistent." % seed)
            else:
                expected_rank = previous_rank + row["event_rank"]
                if row["active_rank"] != expected_rank:
                    raise ValueError(
                        "Seed %s event at step %s does not match its rank increment."
                        % (seed, row["step"])
                    )
                previous_rank = row["active_rank"]
            previous_step = row["step"]
        for boundary in (
            row for row in seed_rows if row["is_consolidation_boundary"]
        ):
            preceding = [
                row
                for row in states
                if row["step"] <= boundary["step"]
                and row["event_source"] != "final_trajectory"
            ]
            if not preceding or boundary["active_rank"] != preceding[-1]["active_rank"]:
                raise ValueError(
                    "Seed %s consolidation boundary rank is inconsistent." % seed
                )
    return rows


def build_summary(rows):
    by_method = {
        method: sorted(
            (row for row in rows if row["method"] == method),
            key=lambda row: row["seed"],
        )
        for method in METHODS
    }
    method_summary = {}
    for method, method_rows in by_method.items():
        accuracies = [row["accuracy"] for row in method_rows]
        method_summary[method] = {
            "seed_count": len(method_rows),
            "mean_accuracy": _mean(accuracies),
            "sample_stdev_accuracy": _sample_stdev(accuracies),
            "mean_selected_active_parameters": _mean(
                [row["selected_active_parameters"] for row in method_rows]
            ),
            "mean_selected_active_rank": _mean(
                [row["selected_active_rank"] for row in method_rows]
            ),
            "mean_final_active_parameters": _mean(
                [row["final_active_parameters"] for row in method_rows]
            ),
            "mean_final_active_rank": _mean(
                [row["final_active_rank"] for row in method_rows]
            ),
            "mean_training_runtime_seconds": _mean(
                [row["runtime_seconds"] for row in method_rows]
            ),
        }

    greedy = {row["seed"]: row for row in by_method["Greedy IncreLoRA"]}
    cgear = {row["seed"]: row for row in by_method["C-GEAR"]}
    per_seed = []
    wins = ties = losses = 0
    for seed in sorted(greedy):
        accuracy_delta = cgear[seed]["accuracy"] - greedy[seed]["accuracy"]
        if math.isclose(accuracy_delta, 0.0, rel_tol=0.0, abs_tol=1e-12):
            ties += 1
        elif accuracy_delta > 0.0:
            wins += 1
        else:
            losses += 1
        selected_reduction = (
            greedy[seed]["selected_active_parameters"]
            - cgear[seed]["selected_active_parameters"]
        )
        final_reduction = (
            greedy[seed]["final_active_parameters"]
            - cgear[seed]["final_active_parameters"]
        )
        per_seed.append(
            {
                "seed": seed,
                "cgear_minus_greedy_accuracy": accuracy_delta,
                "selected_parameter_reduction": selected_reduction,
                "selected_parameter_reduction_percent": 100.0
                * selected_reduction
                / greedy[seed]["selected_active_parameters"],
                "final_parameter_reduction": final_reduction,
                "final_parameter_reduction_percent": 100.0
                * final_reduction
                / greedy[seed]["final_active_parameters"],
            }
        )

    return {
        "schema_version": 1,
        "dataset": "GLUE RTE",
        "comparison_basis": {
            "accuracy_parameters": "selected checkpoint",
            "architecture_only": "final allocation trajectory",
            "runtime": "training runtime from train_results.json",
        },
        "methods": method_summary,
        "paired": {
            "mean_cgear_minus_greedy_accuracy": _mean(
                [item["cgear_minus_greedy_accuracy"] for item in per_seed]
            ),
            "wins_ties_losses": {
                "cgear_wins": wins,
                "ties": ties,
                "cgear_losses": losses,
            },
            "mean_selected_parameter_reduction": _mean(
                [item["selected_parameter_reduction"] for item in per_seed]
            ),
            "mean_selected_parameter_reduction_percent": _mean(
                [item["selected_parameter_reduction_percent"] for item in per_seed]
            ),
            "mean_final_parameter_reduction": _mean(
                [item["final_parameter_reduction"] for item in per_seed]
            ),
            "mean_final_parameter_reduction_percent": _mean(
                [item["final_parameter_reduction_percent"] for item in per_seed]
            ),
            "per_seed": per_seed,
        },
    }


def _configure_plotting():
    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "increlora-matplotlib")
    )
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise SystemExit(
            "Plotting requires analysis/requirements.txt. Install it with "
            "`python -m pip install -r analysis/requirements.txt`."
        ) from error

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.8,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    return plt


def _save_figure(fig, output_dir, stem, title):
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "Title": title,
        "Creator": "analysis/summarize_rte_results.py",
        "CreationDate": None,
        "ModDate": None,
    }
    fig.savefig(output_dir / (stem + ".pdf"), bbox_inches="tight", metadata=metadata)
    fig.savefig(
        output_dir / (stem + ".png"),
        bbox_inches="tight",
        dpi=300,
        metadata={"Title": title, "Software": "summarize_rte_results.py"},
    )


def plot_accuracy_vs_parameters(plt, rows, output_dir):
    title = "RTE Accuracy vs. Selected-Checkpoint Active Parameters"
    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    seeds = sorted({row["seed"] for row in rows})
    lookup = {(row["method"], row["seed"]): row for row in rows}
    for seed in seeds:
        pair = [lookup[(method, seed)] for method in METHODS]
        ax.plot(
            [row["selected_active_parameters"] / 1000.0 for row in pair],
            [100.0 * row["accuracy"] for row in pair],
            color="#B8B8B8",
            linewidth=0.9,
            zorder=1,
        )
    for method in METHODS:
        method_rows = [row for row in rows if row["method"] == method]
        ax.scatter(
            [row["selected_active_parameters"] / 1000.0 for row in method_rows],
            [100.0 * row["accuracy"] for row in method_rows],
            color=COLORS[method],
            edgecolor="white",
            linewidth=0.8,
            marker=MARKERS[method],
            s=58,
            label=method,
            zorder=3,
        )
        for row in method_rows:
            offset = (5, 5) if method == "C-GEAR" else (5, -12)
            ax.annotate(
                "s%s" % row["seed"],
                (
                    row["selected_active_parameters"] / 1000.0,
                    100.0 * row["accuracy"],
                ),
                xytext=offset,
                textcoords="offset points",
                fontsize=8,
                color=COLORS[method],
            )
    ax.set_title(title, loc="left", fontweight="bold", pad=28)
    ax.text(
        0.0,
        1.012,
        "Lines join matched seeds; parameter counts belong to the evaluated checkpoint.",
        transform=ax.transAxes,
        fontsize=8.5,
        color="#555555",
    )
    ax.set_xlabel("Selected active adaptation/model parameters (thousands)")
    ax.set_ylabel("Accuracy (%)")
    ax.legend(loc="lower left")
    ax.margins(x=0.08, y=0.14)
    _save_figure(fig, output_dir, "accuracy_vs_active_parameters", title)
    plt.close(fig)


def plot_accuracy_by_seed(plt, rows, output_dir):
    title = "RTE Accuracy by Seed"
    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    for method in METHODS:
        method_rows = sorted(
            (row for row in rows if row["method"] == method),
            key=lambda row: row["seed"],
        )
        ax.plot(
            [row["seed"] for row in method_rows],
            [100.0 * row["accuracy"] for row in method_rows],
            color=COLORS[method],
            marker=MARKERS[method],
            markersize=6.5,
            linewidth=1.6,
            label=method,
        )
    lookup = {(row["method"], row["seed"]): row for row in rows}
    for seed in sorted({row["seed"] for row in rows}):
        greedy_accuracy = lookup[("Greedy IncreLoRA", seed)]["accuracy"]
        cgear_accuracy = lookup[("C-GEAR", seed)]["accuracy"]
        for method in METHODS:
            row = lookup[(method, seed)]
            is_higher = row["accuracy"] >= (
                cgear_accuracy if method == "Greedy IncreLoRA" else greedy_accuracy
            )
            ax.annotate(
                "%.2f" % (100.0 * row["accuracy"]),
                (row["seed"], 100.0 * row["accuracy"]),
                xytext=(0, 8 if is_higher else -14),
                textcoords="offset points",
                ha="center",
                fontsize=7.5,
                color=COLORS[method],
            )
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlabel("Seed")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(sorted({row["seed"] for row in rows}))
    ax.legend(loc="upper center", ncol=2)
    ax.margins(y=0.20)
    _save_figure(fig, output_dir, "accuracy_by_seed", title)
    plt.close(fig)


def plot_parameters_by_seed(plt, rows, output_dir):
    title = "Active Adaptation/Model Parameters by Seed"
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.5), sharey=True, constrained_layout=True)
    seeds = sorted({row["seed"] for row in rows})
    lookup = {(row["method"], row["seed"]): row for row in rows}
    width = 0.34
    for ax, stage, subtitle in (
        (axes[0], "selected", "Selected checkpoint (paired with accuracy)"),
        (axes[1], "final", "Final allocation trajectory (architecture only)"),
    ):
        x_values = list(range(len(seeds)))
        for method_index, method in enumerate(METHODS):
            offset = (method_index - 0.5) * width
            values = [
                lookup[(method, seed)][stage + "_active_parameters"] / 1000.0
                for seed in seeds
            ]
            bars = ax.bar(
                [x + offset for x in x_values],
                values,
                width=width,
                color=COLORS[method],
                label=method,
            )
            for bar, value in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    value + 13,
                    "%.0f" % value,
                    ha="center",
                    va="bottom",
                    fontsize=7.2,
                    color="#333333",
                )
        ax.set_title(subtitle, fontsize=10.2)
        ax.set_xlabel("Seed")
        ax.set_xticks(x_values, seeds)
        ax.set_ylim(0, 1030)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Active adaptation/model parameters (thousands)")
    axes[0].legend(loc="upper left")
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold")
    _save_figure(fig, output_dir, "active_parameters_by_seed", title)
    plt.close(fig)


def plot_rank_trajectory(plt, trajectory_rows, output_dir, seed):
    rows = sorted(
        (row for row in trajectory_rows if row["seed"] == seed),
        key=lambda row: (row["step"], row["is_consolidation_boundary"]),
    )
    if not rows:
        raise ValueError("No trajectory rows found for seed %s." % seed)
    states = [row for row in rows if not row["is_consolidation_boundary"]]
    boundaries = [row for row in rows if row["is_consolidation_boundary"]]
    title = "C-GEAR Active-Rank Trajectory (RTE Seed %s)" % seed
    fig, ax = plt.subplots(figsize=(7.4, 4.6), constrained_layout=True)
    ax.step(
        [row["step"] for row in states],
        [row["active_rank"] for row in states],
        where="post",
        color=COLORS["C-GEAR"],
        linewidth=1.8,
        label="Active rank",
    )
    positive = [row for row in states if row["event_rank"] > 0]
    zero = [row for row in states if row["event_source"] == "calibrated_zero_rank"]
    ax.scatter(
        [row["step"] for row in positive],
        [row["active_rank"] for row in positive],
        color=COLORS["C-GEAR"],
        edgecolor="white",
        linewidth=0.7,
        s=35,
        zorder=3,
        label="Positive allocation event",
    )
    if zero:
        ax.scatter(
            [row["step"] for row in zero],
            [row["active_rank"] for row in zero],
            facecolor="white",
            edgecolor=COLORS["C-GEAR"],
            linewidth=1.1,
            s=38,
            zorder=3,
            label="No-growth event (k=0)",
        )
    if boundaries:
        boundary = boundaries[0]
        ax.axvspan(
            boundary["step"],
            max(row["step"] for row in states),
            color="#E8E8E8",
            alpha=0.7,
            linewidth=0,
        )
        ax.axvline(
            boundary["step"],
            color="#666666",
            linestyle="--",
            linewidth=1.0,
            label="Consolidation begins",
        )
    ax.set_title(title, loc="left", fontweight="bold", pad=28)
    ax.text(
        0.0,
        1.012,
        "Exact logged states; the step line changes only at recorded events.",
        transform=ax.transAxes,
        fontsize=8.5,
        color="#555555",
    )
    ax.set_xlabel("Optimization step")
    ax.set_ylabel("Total active rank")
    ax.set_xlim(0, max(row["step"] for row in states))
    ax.margins(y=0.12)
    ax.legend(loc="upper left", ncol=2, fontsize=8)
    _save_figure(fig, output_dir, "active_rank_vs_training_step_seed%s" % seed, title)
    plt.close(fig)


def print_summary(rows, summary):
    print("Selected-checkpoint RTE evidence")
    print("seed  Greedy accuracy  C-GEAR accuracy  delta (percentage points)")
    lookup = {(row["method"], row["seed"]): row for row in rows}
    for seed in sorted({row["seed"] for row in rows}):
        greedy = lookup[("Greedy IncreLoRA", seed)]["accuracy"]
        cgear = lookup[("C-GEAR", seed)]["accuracy"]
        print("%4d  %15.6f  %15.6f  %+25.6f" % (seed, greedy, cgear, 100 * (cgear - greedy)))
    for method in METHODS:
        values = summary["methods"][method]
        print(
            "%s: mean accuracy %.6f; sample SD %.6f; mean selected params %.1f; "
            "mean final params %.1f"
            % (
                method,
                values["mean_accuracy"],
                values["sample_stdev_accuracy"],
                values["mean_selected_active_parameters"],
                values["mean_final_active_parameters"],
            )
        )
    paired = summary["paired"]
    record = paired["wins_ties_losses"]
    print(
        "Paired C-GEAR - Greedy mean accuracy: %+.6f (%+.4f percentage points); "
        "wins/ties/losses: %d/%d/%d"
        % (
            paired["mean_cgear_minus_greedy_accuracy"],
            100.0 * paired["mean_cgear_minus_greedy_accuracy"],
            record["cgear_wins"],
            record["ties"],
            record["cgear_losses"],
        )
    )
    print(
        "Mean matched parameter reduction: selected %.1f (%.4f%%); final %.1f (%.4f%%)"
        % (
            paired["mean_selected_parameter_reduction"],
            paired["mean_selected_parameter_reduction_percent"],
            paired["mean_final_parameter_reduction"],
            paired["mean_final_parameter_reduction_percent"],
        )
    )


def parse_args():
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=repository / "results/rte/cgear_vs_greedy_seeds_41_45.csv",
        help="Canonical method-by-seed CSV.",
    )
    parser.add_argument(
        "--trajectory",
        type=Path,
        default=repository / "results/rte/cgear_active_rank_trajectories.csv",
        help="Exact logged C-GEAR trajectory CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repository / "results/rte/figures",
        help="Directory for generated PDF and PNG figures.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=repository / "results/rte/summary.json",
        help="Path for generated descriptive statistics.",
    )
    parser.add_argument(
        "--trajectory-seed",
        type=int,
        default=41,
        help="Representative seed for the exact active-rank trajectory plot.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Validate and summarize the CSV without importing matplotlib.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rows = load_results(args.input)
    summary = build_summary(rows)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print_summary(rows, summary)
    if not args.no_plots:
        plt = _configure_plotting()
        plot_accuracy_vs_parameters(plt, rows, args.output_dir)
        plot_accuracy_by_seed(plt, rows, args.output_dir)
        plot_parameters_by_seed(plt, rows, args.output_dir)
        trajectory_rows = load_trajectory(args.trajectory)
        plot_rank_trajectory(
            plt, trajectory_rows, args.output_dir, args.trajectory_seed
        )


if __name__ == "__main__":
    main()
