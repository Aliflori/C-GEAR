#!/usr/bin/env python3
"""Regenerate the official six-seed C-GEAR RTE report data.

This script reads completed experiment artifacts only. It never imports the
training stack, loads model weights, edits experiment directories, or starts
training. Selected-checkpoint accuracy is paired only with selected-checkpoint
active parameters; final-trajectory architecture statistics remain separate.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path


SEEDS = tuple(range(41, 47))
METHODS = (
    ("Greedy IncreLoRA", "greedy", "greedy/ali_last_seed{seed}"),
    (
        "C-GEAR",
        "genetic_budgeted_calibrated",
        "genetic_budgeted_calibrated/ali_last_seed{seed}_budget0.94",
    ),
)
EXPECTED_STEPS = 1950
EXPECTED_SANITY = {
    "greedy_mean_accuracy_percent": 87.30445,
    "cgear_mean_accuracy_percent": 88.08664,
    "mean_accuracy_improvement_pp": 0.78219,
    "greedy_mean_selected_active_parameters": 828005.0,
    "cgear_mean_selected_active_parameters": 795224.6666666666,
    "greedy_mean_final_active_parameters": 933650.0,
    "cgear_mean_final_active_parameters": 804059.6666666666,
}


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Malformed JSONL at {path}:{line_number}") from error
    return rows


def sample_sd(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def checkpoint_step(path: str) -> int:
    name = Path(path).name
    if not name.startswith("checkpoint-"):
        raise ValueError(f"Unexpected checkpoint path: {path}")
    return int(name.split("-", 1)[1])


def collect_run(
    repository: Path,
    allocator_root: Path,
    label: str,
    telemetry_method: str,
    relative_template: str,
    seed: int,
) -> tuple[dict, dict, dict, Path]:
    relative_run = relative_template.format(seed=seed)
    run_dir = allocator_root / relative_run
    telemetry_path = run_dir / "telemetry.jsonl"
    results_path = run_dir / "model" / "all_results.json"
    trainer_state_path = run_dir / "model" / "trainer_state.json"
    rank_pattern_path = run_dir / "rank_pattern.json"
    required = (telemetry_path, results_path, trainer_state_path, rank_pattern_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing official run artifacts: " + ", ".join(missing))

    telemetry = read_jsonl(telemetry_path)
    starts = [row for row in telemetry if row.get("event_type") == "run_start"]
    ends = [row for row in telemetry if row.get("event_type") == "run_end"]
    if len(starts) != 1 or len(ends) != 1:
        raise ValueError(
            f"Expected one run_start and one run_end in {telemetry_path}; "
            f"found {len(starts)} and {len(ends)}."
        )
    start, end = starts[0], ends[0]
    if start.get("schema_version") != "rank_telemetry.v1":
        raise ValueError(f"Unsupported telemetry schema in {telemetry_path}")
    if int(start.get("seed")) != seed or start.get("method") != telemetry_method:
        raise ValueError(f"Run identity mismatch in {telemetry_path}")
    if end.get("status") != "completed" or int(end.get("global_step")) != EXPECTED_STEPS:
        raise ValueError(f"Run did not complete {EXPECTED_STEPS} steps: {telemetry_path}")

    results = read_json(results_path)
    trainer_state = read_json(trainer_state_path)
    best_checkpoint = str(end["best_checkpoint"])
    best_step = checkpoint_step(best_checkpoint)
    expected_best_dir = run_dir / "model" / f"checkpoint-{best_step}"
    if not expected_best_dir.is_dir():
        raise FileNotFoundError(f"Best checkpoint directory is absent: {expected_best_dir}")
    if checkpoint_step(str(trainer_state["best_model_checkpoint"])) != best_step:
        raise ValueError(f"Trainer-state best checkpoint disagrees with telemetry: {run_dir}")

    accuracy = float(end["best_accuracy"])
    if not math.isclose(accuracy, float(results["eval_accuracy"]), abs_tol=1e-15):
        raise ValueError(f"Final selected-checkpoint accuracy mismatch: {run_dir}")
    eval_samples = int(results["eval_samples"])
    correct_predictions = round(accuracy * eval_samples)
    if not math.isclose(accuracy, correct_predictions / eval_samples, abs_tol=1e-15):
        raise ValueError(f"Accuracy is not an exact count over evaluation examples: {run_dir}")

    selected_rank = int(end["selected_active_rank"])
    final_rank = int(end["final_active_rank"])
    selected_map = end["selected_module_active_ranks"]
    final_map = end["final_module_active_ranks"]
    if sum(int(value) for value in selected_map.values()) != selected_rank:
        raise ValueError(f"Selected rank map is inconsistent: {run_dir}")
    if sum(int(value) for value in final_map.values()) != final_rank:
        raise ValueError(f"Final rank map is inconsistent: {run_dir}")

    row = {
        "method": label,
        "allocator": telemetry_method,
        "seed": seed,
        "selected_step": best_step,
        "accuracy": accuracy,
        "accuracy_percent": 100.0 * accuracy,
        "correct_predictions": correct_predictions,
        "evaluation_examples": eval_samples,
        "selected_active_parameters": int(end["selected_active_parameter_count"]),
        "selected_active_rank": selected_rank,
        "final_active_parameters": int(end["final_active_parameter_count"]),
        "final_active_rank": final_rank,
        "allocator_stop_step": end.get("allocator_stop_step"),
        "allocator_stop_reason": end.get("allocator_stop_reason"),
        "runtime_seconds": float(end["runtime_seconds"]),
        "completed_steps": int(end["global_step"]),
        "run_directory": str(run_dir.relative_to(repository)),
        "telemetry_artifact": str(telemetry_path.relative_to(repository)),
        "results_artifact": str(results_path.relative_to(repository)),
        "trainer_state_artifact": str(trainer_state_path.relative_to(repository)),
        "rank_pattern_artifact": str(rank_pattern_path.relative_to(repository)),
        "best_checkpoint": str(expected_best_dir.relative_to(repository)),
    }
    return row, start, end, telemetry_path


def method_summary(rows: list[dict]) -> dict:
    fields = (
        "accuracy_percent",
        "selected_active_parameters",
        "selected_active_rank",
        "final_active_parameters",
        "final_active_rank",
        "runtime_seconds",
    )
    summary = {"seed_count": len(rows)}
    for field in fields:
        values = [float(row[field]) for row in rows]
        summary[f"mean_{field}"] = statistics.mean(values)
        summary[f"sample_sd_{field}"] = sample_sd(values)
    return summary


def build_summary(rows: list[dict]) -> tuple[dict, list[dict]]:
    by_method = {
        label: sorted((row for row in rows if row["method"] == label), key=lambda r: r["seed"])
        for label, _, _ in METHODS
    }
    greedy = {row["seed"]: row for row in by_method["Greedy IncreLoRA"]}
    cgear = {row["seed"]: row for row in by_method["C-GEAR"]}
    paired_rows = []
    wins = ties = losses = 0
    for seed in SEEDS:
        g, c = greedy[seed], cgear[seed]
        delta = c["accuracy_percent"] - g["accuracy_percent"]
        if math.isclose(delta, 0.0, abs_tol=1e-12):
            ties += 1
            outcome = "tie"
        elif delta > 0:
            wins += 1
            outcome = "C-GEAR win"
        else:
            losses += 1
            outcome = "C-GEAR loss"
        selected_reduction = g["selected_active_parameters"] - c["selected_active_parameters"]
        final_reduction = g["final_active_parameters"] - c["final_active_parameters"]
        paired_rows.append(
            {
                "seed": seed,
                "greedy_accuracy_percent": g["accuracy_percent"],
                "cgear_accuracy_percent": c["accuracy_percent"],
                "cgear_minus_greedy_accuracy_pp": delta,
                "outcome": outcome,
                "greedy_selected_active_parameters": g["selected_active_parameters"],
                "cgear_selected_active_parameters": c["selected_active_parameters"],
                "selected_parameter_reduction": selected_reduction,
                "selected_parameter_reduction_percent": 100.0 * selected_reduction / g["selected_active_parameters"],
                "greedy_final_active_parameters": g["final_active_parameters"],
                "cgear_final_active_parameters": c["final_active_parameters"],
                "final_parameter_reduction": final_reduction,
                "final_parameter_reduction_percent": 100.0 * final_reduction / g["final_active_parameters"],
            }
        )

    method_summaries = {label: method_summary(method_rows) for label, method_rows in by_method.items()}
    g_summary = method_summaries["Greedy IncreLoRA"]
    c_summary = method_summaries["C-GEAR"]
    summary = {
        "schema_version": "cgear_rte_final.v1",
        "dataset": "GLUE RTE validation set",
        "official_seeds": list(SEEDS),
        "comparison_rule": {
            "primary": "selected-checkpoint accuracy paired with selected-checkpoint active parameters",
            "architecture_only": "final-trajectory active parameters and ranks are reported separately",
        },
        "methods": method_summaries,
        "paired": {
            "mean_accuracy_difference_percentage_points": statistics.mean(
                [row["cgear_minus_greedy_accuracy_pp"] for row in paired_rows]
            ),
            "sample_sd_accuracy_difference_percentage_points": sample_sd(
                [row["cgear_minus_greedy_accuracy_pp"] for row in paired_rows]
            ),
            "wins_ties_losses": {"wins": wins, "ties": ties, "losses": losses},
            "mean_selected_parameter_reduction": statistics.mean(
                [row["selected_parameter_reduction"] for row in paired_rows]
            ),
            "mean_paired_selected_parameter_reduction_percent": statistics.mean(
                [row["selected_parameter_reduction_percent"] for row in paired_rows]
            ),
            "selected_reduction_from_aggregate_means_percent": 100.0
            * (
                g_summary["mean_selected_active_parameters"]
                - c_summary["mean_selected_active_parameters"]
            )
            / g_summary["mean_selected_active_parameters"],
            "mean_final_parameter_reduction": statistics.mean(
                [row["final_parameter_reduction"] for row in paired_rows]
            ),
            "mean_paired_final_parameter_reduction_percent": statistics.mean(
                [row["final_parameter_reduction_percent"] for row in paired_rows]
            ),
            "final_reduction_from_aggregate_means_percent": 100.0
            * (g_summary["mean_final_active_parameters"] - c_summary["mean_final_active_parameters"])
            / g_summary["mean_final_active_parameters"],
        },
    }
    return summary, paired_rows


def sanity_check(summary: dict) -> None:
    observed = {
        "greedy_mean_accuracy_percent": summary["methods"]["Greedy IncreLoRA"]["mean_accuracy_percent"],
        "cgear_mean_accuracy_percent": summary["methods"]["C-GEAR"]["mean_accuracy_percent"],
        "mean_accuracy_improvement_pp": summary["paired"]["mean_accuracy_difference_percentage_points"],
        "greedy_mean_selected_active_parameters": summary["methods"]["Greedy IncreLoRA"]["mean_selected_active_parameters"],
        "cgear_mean_selected_active_parameters": summary["methods"]["C-GEAR"]["mean_selected_active_parameters"],
        "greedy_mean_final_active_parameters": summary["methods"]["Greedy IncreLoRA"]["mean_final_active_parameters"],
        "cgear_mean_final_active_parameters": summary["methods"]["C-GEAR"]["mean_final_active_parameters"],
    }
    for key, expected in EXPECTED_SANITY.items():
        if not math.isclose(observed[key], expected, rel_tol=0.0, abs_tol=5e-5):
            raise ValueError(
                f"Verified artifacts disagree with the supplied sanity check for {key}: "
                f"observed {observed[key]}, expected approximately {expected}."
            )


def build_configuration(starts: list[dict]) -> dict:
    common_keys = (
        "task",
        "model_name_or_path",
        "num_train_epochs",
        "total_optimization_steps",
        "max_seq_length",
        "per_device_train_batch_size",
        "per_device_eval_batch_size",
        "learning_rate",
        "optimizer",
        "scheduler_type",
        "warmup_steps",
        "weight_decay",
        "max_grad_norm",
        "fp16",
        "evaluation_strategy",
        "eval_steps",
        "save_strategy",
        "save_steps",
        "metric_for_best_model",
    )
    common = {}
    for key in common_keys:
        values = [start.get(key) for start in starts]
        if any(value != values[0] for value in values[1:]):
            raise ValueError(f"Matched-run training configuration differs for {key}: {values}")
        common[key] = values[0]
    lora_values = [start["lora_configuration"] for start in starts]
    if any(value != lora_values[0] for value in lora_values[1:]):
        raise ValueError("Matched-run LoRA configuration differs across runs.")
    hardware_values = [start["software_hardware"] for start in starts]
    if any(value != hardware_values[0] for value in hardware_values[1:]):
        raise ValueError("Matched-run software/hardware metadata differs across runs.")
    representative = {}
    for label, method, _ in METHODS:
        start = next(item for item in starts if item["method"] == method)
        representative[label] = start["allocator_configuration"]
    return {
        "common_training_configuration": common,
        "lora_configuration": lora_values[0],
        "software_hardware": hardware_values[0],
        "allocator_configuration_by_method": representative,
    }


def write_latex_outputs(paper_dir: Path, summary: dict) -> None:
    g = summary["methods"]["Greedy IncreLoRA"]
    c = summary["methods"]["C-GEAR"]
    p = summary["paired"]
    macros = f"""% Generated by regenerate_six_seed_analysis.py; do not edit by hand.
\\newcommand{{\\GreedyAccuracy}}{{{g['mean_accuracy_percent']:.2f}}}
\\newcommand{{\\CgearAccuracy}}{{{c['mean_accuracy_percent']:.2f}}}
\\newcommand{{\\AccuracyGain}}{{{p['mean_accuracy_difference_percentage_points']:.2f}}}
\\newcommand{{\\SelectedPairedReduction}}{{{p['mean_paired_selected_parameter_reduction_percent']:.2f}}}
\\newcommand{{\\FinalPairedReduction}}{{{p['mean_paired_final_parameter_reduction_percent']:.2f}}}
\\newcommand{{\\CgearWins}}{{{p['wins_ties_losses']['wins']}}}
\\newcommand{{\\CgearTies}}{{{p['wins_ties_losses']['ties']}}}
\\newcommand{{\\CgearLosses}}{{{p['wins_ties_losses']['losses']}}}
"""
    table = f"""% Generated by regenerate_six_seed_analysis.py; do not edit by hand.
\\begin{{table}}[!t]
\\centering
\\caption{{Six-seed RTE results (mean $\\pm$ sample SD). Accuracy and selected counts describe the same selected checkpoint; final counts are architectural.}}
\\label{{tab:main-results}}
\\setlength{{\\tabcolsep}}{{3.6pt}}
\\begin{{tabular}}{{lcc}}
\\toprule
Metric & Greedy IncreLoRA & C-GEAR \\\\
\\midrule
Accuracy $\\uparrow$ & {g['mean_accuracy_percent']:.2f} $\\pm$ {g['sample_sd_accuracy_percent']:.2f} & \\textbf{{{c['mean_accuracy_percent']:.2f}}} $\\pm$ {c['sample_sd_accuracy_percent']:.2f} \\\\
Selected params $\\downarrow$ & {g['mean_selected_active_parameters']/1000:.1f}k $\\pm$ {g['sample_sd_selected_active_parameters']/1000:.1f}k & \\textbf{{{c['mean_selected_active_parameters']/1000:.1f}k}} $\\pm$ {c['sample_sd_selected_active_parameters']/1000:.1f}k \\\\
Selected rank $\\downarrow$ & {g['mean_selected_active_rank']:.1f} $\\pm$ {g['sample_sd_selected_active_rank']:.1f} & \\textbf{{{c['mean_selected_active_rank']:.1f}}} $\\pm$ {c['sample_sd_selected_active_rank']:.1f} \\\\
Final params $\\downarrow$ & {g['mean_final_active_parameters']/1000:.1f}k $\\pm$ {g['sample_sd_final_active_parameters']/1000:.1f}k & \\textbf{{{c['mean_final_active_parameters']/1000:.1f}k}} $\\pm$ {c['sample_sd_final_active_parameters']/1000:.1f}k \\\\
Final rank $\\downarrow$ & {g['mean_final_active_rank']:.1f} $\\pm$ {g['sample_sd_final_active_rank']:.1f} & \\textbf{{{c['mean_final_active_rank']:.1f}}} $\\pm$ {c['sample_sd_final_active_rank']:.1f} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
    paper_dir.mkdir(parents=True, exist_ok=True)
    (paper_dir / "results_macros.tex").write_text(macros, encoding="utf-8")
    (paper_dir / "results_table.tex").write_text(table, encoding="utf-8")


def write_markdown(path: Path, rows: list[dict], summary: dict) -> None:
    lookup = {(row["method"], row["seed"]): row for row in rows}
    lines = [
        "# Official C-GEAR RTE results: seeds 41--46",
        "",
        "Accuracy and selected active parameters refer to the same selected checkpoint. "
        "Final counts are a separate terminal-architecture observation.",
        "",
        "| Seed | Greedy accuracy (%) | C-GEAR accuracy (%) | Difference (pp) | Greedy selected params | C-GEAR selected params | Greedy final params | C-GEAR final params |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for seed in SEEDS:
        g = lookup[("Greedy IncreLoRA", seed)]
        c = lookup[("C-GEAR", seed)]
        lines.append(
            f"| {seed} | {g['accuracy_percent']:.3f} | {c['accuracy_percent']:.3f} | "
            f"{c['accuracy_percent'] - g['accuracy_percent']:+.3f} | "
            f"{g['selected_active_parameters']:,} | {c['selected_active_parameters']:,} | "
            f"{g['final_active_parameters']:,} | {c['final_active_parameters']:,} |"
        )
    p = summary["paired"]
    lines.extend(
        [
            "",
            f"- Accuracy W/T/L: {p['wins_ties_losses']['wins']}/{p['wins_ties_losses']['ties']}/{p['wins_ties_losses']['losses']}.",
            f"- Mean C-GEAR accuracy improvement: {p['mean_accuracy_difference_percentage_points']:+.5f} percentage points.",
            f"- Mean paired selected-parameter reduction: {p['mean_paired_selected_parameter_reduction_percent']:.5f}%.",
            f"- Mean paired final-parameter reduction: {p['mean_paired_final_parameter_reduction_percent']:.5f}%.",
            "- Scope: controlled local GLUE RTE diagnostics; no claim of statistical significance or universal superiority.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=repository)
    parser.add_argument("--output-dir", type=Path, default=repository / "final_report" / "data")
    parser.add_argument("--paper-dir", type=Path, default=repository / "final_report" / "paper")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repository = args.repository.resolve()
    output_dir = args.output_dir.resolve()
    allocator_root = repository / "NLU" / "output" / "glue" / "rte_allocator"
    rows, starts, telemetry_paths = [], [], []
    for seed in SEEDS:
        for label, telemetry_method, relative_template in METHODS:
            row, start, _end, telemetry_path = collect_run(
                repository,
                allocator_root,
                label,
                telemetry_method,
                relative_template,
                seed,
            )
            rows.append(row)
            starts.append(start)
            telemetry_paths.append(telemetry_path)
    rows.sort(key=lambda row: (row["seed"], 0 if row["method"] == "Greedy IncreLoRA" else 1))
    summary, paired_rows = build_summary(rows)
    sanity_check(summary)
    configuration = build_configuration(starts)

    canonical_fields = list(rows[0].keys())
    write_csv(output_dir / "canonical_results_seeds_41_46.csv", canonical_fields, rows)
    write_csv(output_dir / "run_manifest.csv", canonical_fields, rows)
    write_csv(output_dir / "paired_differences.csv", list(paired_rows[0].keys()), paired_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "aggregate_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "experiment_configuration.json").write_text(
        json.dumps(configuration, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_markdown(output_dir / "canonical_results.md", rows, summary)
    write_latex_outputs(args.paper_dir.resolve(), summary)

    telemetry_output = output_dir / "telemetry"
    subprocess.run(
        [
            sys.executable,
            str(repository / "analysis" / "parse_rank_telemetry.py"),
            *[str(path) for path in telemetry_paths],
            "--output-dir",
            str(telemetry_output),
        ],
        check=True,
    )
    print(f"Verified and regenerated official data for {len(rows)} completed runs.")
    print(json.dumps(summary["paired"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
