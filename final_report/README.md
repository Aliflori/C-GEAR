# C-GEAR final report and reproducibility bundle

This directory contains the canonical, lightweight final-project evidence for **C-GEAR: Calibrated Genetic Efficiency-Aware Rank Allocation for Incremental LoRA Fine-Tuning**. No training output or checkpoint weights are duplicated here.

## Contents

- `paper/cgear_final_report.pdf`: final English, two-column scientific report (three pages).
- `paper/cgear_final_report.tex` and `paper/references.bib`: report source and verified bibliography.
- `paper/results_macros.tex` and `paper/results_table.tex`: generated LaTeX quantities; do not edit manually.
- `data/canonical_results_seeds_41_46.csv`: one validated row per method and seed.
- `data/paired_differences.csv`: matched accuracy and parameter differences.
- `data/aggregate_summary.json`: official means, sample SDs, paired reductions, and W/T/L.
- `data/experiment_configuration.json`: training, LoRA, allocator, software, and hardware metadata recovered from telemetry.
- `data/run_manifest.csv`: exact selected run directory and best-checkpoint provenance.
- `data/telemetry/`: schema-validated trajectory, module-rank, allocation, calibration, and evaluation tables for all 12 runs.
- `figures/`: four paper figures plus six presentation/diagnostic figures and `figure_manifest.json`.

The canonical regeneration and plotting tools are maintained in the repository's top-level `analysis/` directory.

## Official evidence provenance

All statistics use matched seeds **41, 42, 43, 44, 45, and 46**. The canonical local source directories are:

| Method | Seed | Run directory relative to `NLU/` |
|---|---:|---|
| Greedy IncreLoRA | 41--46 | `output/glue/rte_allocator/greedy/ali_last_seed{seed}` |
| C-GEAR | 41--46 | `output/glue/rte_allocator/genetic_budgeted_calibrated/ali_last_seed{seed}_budget0.94` |

Each source run completed 1,950 optimization steps and contains a complete `telemetry.jsonl`, `trainer_state.json`, result JSON, dynamic rank patterns, and terminal log. Those large/local outputs remain ignored by Git and are read in place. The generated `run_manifest.csv` records the exact repository-relative source path, selected checkpoint, and terminal stop reason for each run.

## Measurement contract

The primary comparison pairs:

> selected-checkpoint validation accuracy ↔ selected-checkpoint active model parameters.

`active_model_parameter_count` is the fixed trainable task head plus LoRA components represented by the active rank map. It is not the raw count of tensors with `requires_grad=True`. Terminal allocation trajectories are a separate architecture-only observation and are never assigned the selected checkpoint's accuracy.

Percentage reductions in the paper are the mean of six seed-wise paired reductions. Aggregate-mean reductions remain available in `aggregate_summary.json` and are labeled separately.

## Regeneration

From the repository root:

```bash
source /home/ali/LoRa_Project/.venv/bin/activate
python analysis/regenerate_six_seed_analysis.py
python analysis/generate_report_figures.py
```

The first command fails loudly if a run is missing, incomplete, mismatched, or disagrees across telemetry, rank-pattern, checkpoint, trainer-state, and result artifacts. It also invokes the canonical `analysis/parse_rank_telemetry.py` parser to regenerate the five telemetry tables.

Compile the report:

```bash
cd final_report/paper
pdflatex -interaction=nonstopmode -halt-on-error cgear_final_report.tex
bibtex cgear_final_report
pdflatex -interaction=nonstopmode -halt-on-error cgear_final_report.tex
pdflatex -interaction=nonstopmode -halt-on-error cgear_final_report.tex
```

Alternatively, after activating the environment, `bash analysis/build_final_report.sh` performs data validation, figure generation, compilation, and page-count/log checks in one command. LaTeX intermediates are ignored; the PDF, source, canonical data, and figures are versioned.

## Figures

The PDF includes:

1. C-GEAR allocation-event schematic;
2. matched-seed accuracy and selected-checkpoint accuracy--efficiency trade-off;
3. active-rank growth and C-GEAR event-size history;
4. aggregate module-family allocation and a representative seed-41 layer--family rank map.

Additional generated figures cover per-seed accuracy, selected/final active parameters, layer-wise final rank, the standalone module-family distribution, a representative compact rank heatmap, and calibration history. They are intended for review or later presentation material, not as extra evidence silently omitted from the paper.
