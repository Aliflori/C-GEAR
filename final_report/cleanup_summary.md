# Final repository cleanup summary

Cleanup was deliberately conservative because the repository contains scientifically useful completed experiments and regression coverage.

## Actions taken

- Added targeted ignore rules for the local `HEY ALI.zip` transfer package, rendered PDF QA pages, and LaTeX intermediates.
- Kept the final PDF, LaTeX/BibTeX source, canonical six-seed CSV/JSON data, publication figures, regeneration scripts, and audit documentation under `final_report/`.
- Reused the canonical telemetry parser rather than creating a duplicate parser for the report.
- Kept local `NLU/output/` experiment directories intact and ignored. No checkpoint, model weight, optimizer state, scheduler state, TensorBoard file, or terminal log was moved or deleted.

## Audited and retained

- Original IncreLoRA NLU/NLG task scripts and non-RTE task support, including SST-2, MRPC, CoLA, and other GLUE configurations.
- Greedy, genetic, budgeted, calibrated, dynamic-checkpoint, reserve-initialization, and telemetry implementation files.
- Allocator, calibration, lazy-capacity, reserve-initialization, checkpoint restoration, parameter-accounting, telemetry-invariance, parser, and plotter regression tests.
- Existing Git commits and the `calibrated-v2-prefixed-results` tag.
- All completed Greedy and C-GEAR result directories for seeds 41--46.

## Deletions

No tracked source, test, script, or scientific output was deleted. The audit found no clearly dead project-specific tracked file whose removal was safer than preservation. The local review ZIP was not deleted; it is simply excluded from version control because its canonical contents are represented by source, summaries, and report artifacts.
