# Historical C-GEAR development artifacts

This directory preserves project-specific development artifacts that predate
the final C-GEAR six-seed pipeline. They are retained for provenance and are
not inputs to the official seeds 41--46 analysis or final report.

The active experimental evidence remains under
`NLU/output/glue/rte_allocator/` in these exact run families:

- `greedy/ali_last_seed{41..46}`
- `genetic_budgeted_calibrated/ali_last_seed{41..46}_budget0.94`

The canonical analysis and report pipeline remains under `final_report/`.
Original upstream IncreLoRA files remain in their original locations.

Contents:

- `old_runs/`: local, ignored training outputs from superseded experiments;
- `old_logs/`: loose logs from superseded experiments;
- `old_scripts/`: superseded project-added run/debug wrappers;
- `old_analysis/`: the earlier five-seed analysis and result snapshot;
- `old_packages/`: the reproducible temporary `HEY ALI.zip` review package.

No checkpoint, model, optimizer, telemetry, or log file was deleted as part of
this archival cleanup. See `cleanup_inventory.md` for the pre-move
classification and provenance decisions.

## Executed archival move

The archival move relocated 53,820,032,208 bytes (approximately 51 GiB) into
`history/`. The large local run, log, and package trees remain intentionally
ignored by Git; tracked scripts, analysis code, summaries, and this
documentation are versioned as renames/additions.

Archived run families include the early `NLU/output/glue/rte/` and
`sst2_fast/` outputs, pre-calibration `genetic_budgeted` and
`genetic_budgeted_eval`, calibrated v2/v3/reserve-fix development runs, and
noncanonical duplicate Greedy generations. The twelve official `ali_last`
seeds remain at their original active paths.
