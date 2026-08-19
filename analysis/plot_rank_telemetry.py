#!/usr/bin/env python3
"""Create report-quality figures from parse_rank_telemetry.py CSV outputs."""

import argparse
import csv
import hashlib
import os
import re
import tempfile
from collections import defaultdict
from pathlib import Path


TABLES = {
    "rank": "rank_trajectory.csv",
    "module": "module_rank_trajectory.csv",
    "allocation": "allocation_events.csv",
    "calibration": "calibration_events.csv",
    "evaluation": "evaluation_trajectory.csv",
}
COLORS = ("#3B6FB6", "#D05A47", "#4C956C", "#8E6CBB", "#D18F2F", "#6B7280")


def _load(path):
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    except OSError as error:
        raise SystemExit("Cannot read %s: %s" % (path, error)) from error


def load_tables(input_dir):
    input_dir = Path(input_dir)
    return {name: _load(input_dir / filename) for name, filename in TABLES.items()}


def _optional_float(value):
    return None if value is None or value == "" else float(value)


def _optional_int(value):
    return None if value is None or value == "" else int(value)


def _run_key(row):
    return (
        row["source_artifact"],
        row["method"],
        int(row["seed"]),
        int(row.get("run_segment", 0) or 0),
    )


def _run_label(key):
    source, method, seed, segment = key
    source_label = Path(source).parent.name or Path(source).stem
    return "%s (seed %s, segment %s, %s)" % (
        method,
        seed,
        segment,
        source_label,
    )


def _slug(value):
    return re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()


def _run_slug(key):
    source, method, seed, segment = key
    source_label = Path(source).parent.name or Path(source).stem
    fingerprint = hashlib.sha1(source.encode("utf-8")).hexdigest()[:8]
    return "%s_seed%s_segment%s_%s_%s" % (
        _slug(method),
        seed,
        segment,
        _slug(source_label) or "run",
        fingerprint,
    )


def _is_trajectory_state(row):
    return row.get("state_role", "trajectory") != "selected_best_checkpoint"


def _is_training_evaluation(row):
    return (
        row.get("state_role", "training_trajectory_evaluation")
        == "training_trajectory_evaluation"
    )


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
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9.5,
            "axes.edgecolor": "#333333",
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.6,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    return plt


def _save(fig, output_dir, stem, title, formats, dpi):
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for extension in formats:
        path = output_dir / (stem + "." + extension)
        metadata = {
            "Title": title,
            "Creator": "analysis/plot_rank_telemetry.py",
        }
        if extension == "pdf":
            metadata.update({"CreationDate": None, "ModDate": None})
        fig.savefig(path, bbox_inches="tight", dpi=dpi, metadata=metadata)
        paths.append(path)
    return paths


def _stop_steps(rank_rows):
    return {
        _run_key(row): int(row["global_step"])
        for row in rank_rows
        if row["event_type"] == "allocator_stop"
    }


def _annotate_stops(ax, stop_steps, keys, axis="x"):
    for key in keys:
        if key not in stop_steps:
            continue
        step = stop_steps[key]
        if axis == "x":
            ax.axvline(step, color="#777777", linestyle="--", linewidth=0.9)


def _trajectory_plot(plt, rows, field, ylabel, title, stem, output_dir, formats, dpi, stops):
    usable = [
        row
        for row in rows
        if _is_trajectory_state(row)
        and _optional_float(row.get(field)) is not None
    ]
    if not usable:
        return []
    fig, ax = plt.subplots(figsize=(7.4, 4.5), constrained_layout=True)
    groups = defaultdict(list)
    for row in usable:
        groups[_run_key(row)].append(row)
    for index, (key, group) in enumerate(sorted(groups.items(), key=lambda item: item[0][1:])):
        group.sort(key=lambda row: (int(row["global_step"]), float(row["wall_time_seconds"])))
        ax.step(
            [int(row["global_step"]) for row in group],
            [float(row[field]) for row in group],
            where="post",
            color=COLORS[index % len(COLORS)],
            linewidth=1.6,
            label=_run_label(key),
        )
    _annotate_stops(ax, stops, groups)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlabel("Optimization step")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=7.2)
    paths = _save(fig, output_dir, stem, title, formats, dpi)
    plt.close(fig)
    return paths


def plot_accuracy(plt, rows, output_dir, formats, dpi, stops):
    usable = [
        row
        for row in rows
        if _is_training_evaluation(row)
        and _optional_float(row.get("accuracy")) is not None
    ]
    if not usable:
        return []
    fig, ax = plt.subplots(figsize=(7.4, 4.5), constrained_layout=True)
    groups = defaultdict(list)
    for row in usable:
        groups[_run_key(row)].append(row)
    for index, (key, group) in enumerate(sorted(groups.items(), key=lambda item: item[0][1:])):
        group.sort(key=lambda row: int(row["global_step"]))
        ax.plot(
            [int(row["global_step"]) for row in group],
            [100.0 * float(row["accuracy"]) for row in group],
            marker="o",
            markersize=3.5,
            color=COLORS[index % len(COLORS)],
            linewidth=1.3,
            label=_run_label(key),
        )
    _annotate_stops(ax, stops, groups)
    title = "Accuracy vs. Optimization Step"
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlabel("Optimization step")
    ax.set_ylabel("Accuracy (%)")
    ax.legend(fontsize=7.2)
    paths = _save(fig, output_dir, "accuracy_vs_step", title, formats, dpi)
    plt.close(fig)
    return paths


def plot_accuracy_vs_parameters(plt, rows, output_dir, formats, dpi):
    usable = [
        row
        for row in rows
        if _is_training_evaluation(row)
        and _optional_float(row.get("accuracy")) is not None
        and _optional_int(row.get("active_model_parameter_count")) is not None
    ]
    if not usable:
        return []
    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    groups = defaultdict(list)
    for row in usable:
        groups[_run_key(row)].append(row)
    for index, (key, group) in enumerate(sorted(groups.items(), key=lambda item: item[0][1:])):
        group.sort(key=lambda row: int(row["global_step"]))
        ax.plot(
            [int(row["active_model_parameter_count"]) / 1000.0 for row in group],
            [100.0 * float(row["accuracy"]) for row in group],
            marker="o",
            markersize=4,
            color=COLORS[index % len(COLORS)],
            linewidth=1.1,
            label=_run_label(key),
        )
    title = "Accuracy vs. Active Adaptation/Model Parameters"
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlabel("Active adaptation/model parameters (thousands)")
    ax.set_ylabel("Accuracy (%)")
    ax.legend(fontsize=7.2)
    paths = _save(fig, output_dir, "accuracy_vs_active_parameters", title, formats, dpi)
    plt.close(fig)
    return paths


def _module_snapshots(rows):
    groups = defaultdict(lambda: defaultdict(dict))
    role_priority = {
        "initial_trajectory": 0,
        "trajectory": 1,
        "final_trajectory": 2,
    }
    for row in rows:
        if row["event_type"] not in ("run_start", "allocation_event", "allocator_stop", "run_end"):
            continue
        if not _is_trajectory_state(row):
            continue
        by_module = groups[_run_key(row)][int(row["global_step"])]
        previous = by_module.get(row["module_name"])
        if previous is None or role_priority.get(
            row.get("state_role", "trajectory"), 1
        ) >= role_priority.get(previous.get("state_role", "trajectory"), 1):
            by_module[row["module_name"]] = row
    return groups


def _completed_trajectory_runs(module_rows):
    return {
        _run_key(row)
        for row in module_rows
        if row["event_type"] == "run_end"
        and row.get("state_role", "final_trajectory") == "final_trajectory"
    }


def plot_module_heatmaps(plt, module_rows, output_dir, formats, dpi):
    paths = []
    completed = _completed_trajectory_runs(module_rows)
    for key, by_step in sorted(_module_snapshots(module_rows).items(), key=lambda item: item[0][1:]):
        steps = sorted(by_step)
        modules = sorted({name for values in by_step.values() for name in values})
        complete_steps = [step for step in steps if set(by_step[step]) == set(modules)]
        if not modules or not complete_steps:
            continue
        matrix = [[int(by_step[step][name]["active_rank"]) for step in complete_steps] for name in modules]
        fig_height = min(15.0, max(4.5, 0.18 * len(modules)))
        fig, ax = plt.subplots(figsize=(8.2, fig_height), constrained_layout=True)
        image = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="viridis")
        tick_positions = list(range(len(complete_steps)))
        if len(tick_positions) > 15:
            stride = max(1, len(tick_positions) // 12)
            tick_positions = tick_positions[::stride]
        ax.set_xticks(tick_positions, [complete_steps[index] for index in tick_positions])
        ax.set_yticks(range(len(modules)), modules, fontsize=5.5)
        ax.set_xlabel("Optimization step")
        ax.set_ylabel("LoRA module")
        completion_suffix = "" if key in completed else " (latest observed; run incomplete)"
        title = "Module-wise Active-Rank Heatmap — %s%s" % (
            _run_label(key),
            completion_suffix,
        )
        ax.set_title(title, loc="left", fontweight="bold")
        fig.colorbar(image, ax=ax, label="Active rank", shrink=0.7)
        stem = "module_rank_heatmap_%s" % _run_slug(key)
        paths.extend(_save(fig, output_dir, stem, title, formats, dpi))
        plt.close(fig)
    return paths


def _final_module_rows(module_rows):
    final = {}
    for key, by_step in _module_snapshots(module_rows).items():
        step = max(by_step)
        final[key] = dict(by_step[step])
    return final, _completed_trajectory_runs(module_rows)


def plot_final_layer_ranks(plt, module_rows, output_dir, formats, dpi):
    final, completed = _final_module_rows(module_rows)
    if not final:
        return []
    paths = []
    for key, modules in sorted(final.items(), key=lambda item: item[0][1:]):
        values = defaultdict(lambda: defaultdict(int))
        for row in modules.values():
            layer = _optional_int(row["transformer_layer"])
            if layer is not None:
                values[layer][row["module_group"]] += int(row["active_rank"])
        if not values:
            continue
        layers = sorted(values)
        fig, ax = plt.subplots(figsize=(8.0, 4.5), constrained_layout=True)
        bottoms = [0] * len(layers)
        for group, color in (("attention", COLORS[0]), ("ffn", COLORS[1]), ("other", COLORS[5])):
            heights = [values[layer][group] for layer in layers]
            ax.bar(layers, heights, bottom=bottoms, label=group, color=color)
            bottoms = [bottom + height for bottom, height in zip(bottoms, heights)]
        if key in completed:
            title = "Final-Trajectory Active Rank by Transformer Layer — %s" % _run_label(key)
            stem_prefix = "final_trajectory_rank_by_layer"
        else:
            title = "Latest Observed Active Rank by Transformer Layer — %s (run incomplete)" % _run_label(key)
            stem_prefix = "latest_observed_rank_by_layer"
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlabel("Transformer layer")
        ax.set_ylabel("Total active rank")
        ax.set_xticks(layers)
        ax.legend()
        stem = "%s_%s" % (stem_prefix, _run_slug(key))
        paths.extend(_save(fig, output_dir, stem, title, formats, dpi))
        plt.close(fig)
    return paths


def plot_cumulative_layer_allocations(plt, module_rows, output_dir, formats, dpi):
    paths = []
    completed = _completed_trajectory_runs(module_rows)
    for key, by_step in sorted(_module_snapshots(module_rows).items(), key=lambda item: item[0][1:]):
        steps = sorted(by_step)
        if len(steps) < 2:
            continue
        first = by_step[steps[0]]
        if not first:
            continue
        layers = sorted(
            {
                _optional_int(row["transformer_layer"])
                for values in by_step.values()
                for row in values.values()
                if _optional_int(row["transformer_layer"]) is not None
            }
        )
        initial = defaultdict(int)
        for row in first.values():
            layer = _optional_int(row["transformer_layer"])
            if layer is not None:
                initial[layer] += int(row["active_rank"])
        series = {layer: [] for layer in layers}
        valid_steps = []
        for step in steps:
            current = defaultdict(int)
            for row in by_step[step].values():
                layer = _optional_int(row["transformer_layer"])
                if layer is not None:
                    current[layer] += int(row["active_rank"])
            valid_steps.append(step)
            for layer in layers:
                series[layer].append(current[layer] - initial[layer])
        fig, ax = plt.subplots(figsize=(8.2, 4.7), constrained_layout=True)
        for index, layer in enumerate(layers):
            ax.step(
                valid_steps,
                series[layer],
                where="post",
                linewidth=1.1,
                color=plt.cm.tab20(index % 20),
                label="layer %s" % layer,
            )
        completion_suffix = "" if key in completed else " (latest observed; run incomplete)"
        title = "Cumulative Rank Allocation by Layer — %s%s" % (
            _run_label(key),
            completion_suffix,
        )
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlabel("Optimization step")
        ax.set_ylabel("Cumulative rank added")
        ax.legend(ncol=3, fontsize=7)
        stem = "cumulative_rank_by_layer_%s" % _run_slug(key)
        paths.extend(_save(fig, output_dir, stem, title, formats, dpi))
        plt.close(fig)
    return paths


def plot_attention_vs_ffn(plt, module_rows, output_dir, formats, dpi):
    final, completed = _final_module_rows(module_rows)
    if not final:
        return []
    keys = sorted(final, key=lambda key: key[1:])
    values = []
    for key in keys:
        totals = defaultdict(int)
        for row in final[key].values():
            totals[row["module_group"]] += int(row["active_rank"])
        values.append(totals)
    fig, ax = plt.subplots(figsize=(max(7.0, 1.2 * len(keys)), 4.5), constrained_layout=True)
    positions = list(range(len(keys)))
    width = 0.36
    ax.bar([x - width / 2 for x in positions], [v["attention"] for v in values], width, label="Attention", color=COLORS[0])
    ax.bar([x + width / 2 for x in positions], [v["ffn"] for v in values], width, label="FFN", color=COLORS[1])
    labels = [
        "%s\ns%s seg%s%s"
        % (key[1], key[2], key[3], "" if key in completed else "*")
        for key in keys
    ]
    ax.set_xticks(positions, labels, fontsize=7.5)
    if all(key in completed for key in keys):
        title = "Final-Trajectory Attention-family vs. FFN Active Rank"
    else:
        title = "Final / Latest-Observed Attention-family vs. FFN Active Rank (* incomplete)"
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_ylabel("Total active rank")
    ax.legend()
    paths = _save(fig, output_dir, "attention_vs_ffn_final_rank", title, formats, dpi)
    plt.close(fig)
    return paths


def plot_selected_k(plt, rows, output_dir, formats, dpi, stops):
    usable = [
        row
        for row in rows
        if _optional_int(row.get("selected_k")) is not None
        or _optional_int(row.get("selected_event_rank")) is not None
    ]
    if not usable:
        return []
    fig, ax = plt.subplots(figsize=(7.4, 4.5), constrained_layout=True)
    groups = defaultdict(list)
    for row in usable:
        groups[_run_key(row)].append(row)
    for index, (key, group) in enumerate(sorted(groups.items(), key=lambda item: item[0][1:])):
        group.sort(key=lambda row: int(row["global_step"]))
        steps = [int(row["global_step"]) for row in group]
        values = [
            _optional_int(row.get("selected_k"))
            if _optional_int(row.get("selected_k")) is not None
            else int(row["selected_event_rank"])
            for row in group
        ]
        color = COLORS[index % len(COLORS)]
        ax.plot(steps, values, marker="o", linewidth=1.1, color=color, label=_run_label(key))
        zero_steps = [step for step, value in zip(steps, values) if value == 0]
        if zero_steps:
            ax.scatter(zero_steps, [0] * len(zero_steps), facecolor="white", edgecolor=color, linewidth=1.2, zorder=4)
    _annotate_stops(ax, stops, groups)
    title = "Selected Allocation Size (k)"
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlabel("Optimization step")
    ax.set_ylabel("Selected modules (k)")
    ax.set_ylim(bottom=-0.25)
    ax.legend(fontsize=7.2)
    paths = _save(fig, output_dir, "selected_k_vs_step", title, formats, dpi)
    plt.close(fig)
    return paths


def _selected_calibration(rows):
    selected = [row for row in rows if row.get("is_selected", "").lower() == "true"]
    if selected:
        return selected
    best = {}
    for row in rows:
        lcb = _optional_float(row.get("calibration_gain_lcb"))
        if lcb is None:
            continue
        key = _run_key(row) + (int(row["global_step"]),)
        if key not in best or lcb > float(best[key]["calibration_gain_lcb"]):
            best[key] = row
    return list(best.values())


def plot_calibration_scores(plt, rows, output_dir, formats, dpi):
    chosen = [
        row for row in _selected_calibration(rows)
        if _optional_float(row.get("calibration_gain_lcb")) is not None
    ]
    if not chosen:
        return []
    fig, ax = plt.subplots(figsize=(7.4, 4.5), constrained_layout=True)
    groups = defaultdict(list)
    for row in chosen:
        groups[_run_key(row)].append(row)
    for index, (key, group) in enumerate(sorted(groups.items(), key=lambda item: item[0][1:])):
        group.sort(key=lambda row: int(row["global_step"]))
        steps = [int(row["global_step"]) for row in group]
        lcb = [float(row["calibration_gain_lcb"]) for row in group]
        color = COLORS[index % len(COLORS)]
        ax.plot(steps, lcb, marker="o", linewidth=1.2, color=color, label="LCB: " + _run_label(key))
        means = [_optional_float(row.get("calibration_gain_mean")) for row in group]
        if all(value is not None for value in means):
            ax.plot(steps, means, linestyle="--", linewidth=0.9, color=color, alpha=0.7, label="mean: " + _run_label(key))
    ax.axhline(0.0, color="#777777", linewidth=0.8)
    title = "Selected Calibration-Score Trajectory"
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlabel("Optimization step")
    ax.set_ylabel("Calibrated loss-reduction score")
    ax.legend(fontsize=6.8, ncol=2)
    paths = _save(fig, output_dir, "calibration_score_vs_step", title, formats, dpi)
    plt.close(fig)
    return paths


def plot_candidate_uncertainty(plt, rows, output_dir, formats, dpi):
    usable = [
        row for row in rows
        if _optional_float(row.get("calibration_gain_mean")) is not None
        and _optional_float(row.get("calibration_gain_std")) is not None
        and _optional_float(row.get("calibration_gain_lcb")) is not None
    ]
    if not usable:
        return []
    fig, ax = plt.subplots(figsize=(7.0, 4.8), constrained_layout=True)
    scatter = ax.scatter(
        [float(row["calibration_gain_mean"]) for row in usable],
        [float(row["calibration_gain_std"]) for row in usable],
        c=[float(row["calibration_gain_lcb"]) for row in usable],
        cmap="coolwarm",
        edgecolor="white",
        linewidth=0.4,
        s=32,
        alpha=0.85,
    )
    selected = [row for row in usable if row.get("is_selected", "").lower() == "true"]
    if selected:
        ax.scatter(
            [float(row["calibration_gain_mean"]) for row in selected],
            [float(row["calibration_gain_std"]) for row in selected],
            facecolor="none",
            edgecolor="black",
            linewidth=1.0,
            s=70,
            label="Selected candidate",
        )
        ax.legend()
    title = "Calibration Candidate Mean vs. Uncertainty"
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlabel("Mean calibrated gain")
    ax.set_ylabel("Gain standard deviation")
    fig.colorbar(scatter, ax=ax, label="Lower-confidence bound")
    paths = _save(fig, output_dir, "candidate_mean_vs_uncertainty", title, formats, dpi)
    plt.close(fig)
    return paths


def generate_plots(tables, output_dir, formats=("pdf", "png"), dpi=300):
    plt = _configure_plotting()
    stops = _stop_steps(tables["rank"])
    generated = []
    generated += _trajectory_plot(
        plt,
        tables["rank"],
        "total_active_rank",
        "Total active rank",
        "Total Active Rank vs. Optimization Step",
        "total_active_rank_vs_step",
        output_dir,
        formats,
        dpi,
        stops,
    )
    generated += _trajectory_plot(
        plt,
        tables["rank"],
        "active_model_parameter_count",
        "Active adaptation/model parameters",
        "Active Parameters vs. Optimization Step",
        "active_parameters_vs_step",
        output_dir,
        formats,
        dpi,
        stops,
    )
    generated += plot_accuracy(plt, tables["evaluation"], output_dir, formats, dpi, stops)
    generated += plot_accuracy_vs_parameters(plt, tables["evaluation"], output_dir, formats, dpi)
    generated += plot_module_heatmaps(plt, tables["module"], output_dir, formats, dpi)
    generated += plot_final_layer_ranks(plt, tables["module"], output_dir, formats, dpi)
    generated += plot_cumulative_layer_allocations(plt, tables["module"], output_dir, formats, dpi)
    generated += plot_attention_vs_ffn(plt, tables["module"], output_dir, formats, dpi)
    generated += plot_selected_k(plt, tables["allocation"], output_dir, formats, dpi, stops)
    generated += plot_calibration_scores(plt, tables["calibration"], output_dir, formats, dpi)
    generated += plot_candidate_uncertainty(plt, tables["calibration"], output_dir, formats, dpi)
    return generated


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing the five CSVs from parse_rank_telemetry.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for generated figures.",
    )
    parser.add_argument(
        "--formats",
        default="pdf,png",
        help="Comma-separated output formats (default: pdf,png).",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Raster output DPI.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    formats = tuple(value.strip().lower() for value in args.formats.split(",") if value.strip())
    if not formats or any(value not in ("pdf", "png") for value in formats):
        raise SystemExit("--formats must contain only pdf and/or png")
    if args.dpi <= 0:
        raise SystemExit("--dpi must be positive")
    generated = generate_plots(load_tables(args.input_dir), args.output_dir, formats, args.dpi)
    if not generated:
        print("No applicable plots: the parsed CSVs contain insufficient telemetry.")
        return
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
