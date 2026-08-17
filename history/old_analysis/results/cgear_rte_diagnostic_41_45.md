# C-GEAR RTE diagnostic snapshot (seeds 41–45)

This snapshot records the completed local GLUE RTE diagnostic runs produced by the reserve-initialization-fixed `genetic_budgeted_calibrated` implementation. Each run used a 0.94 active-parameter budget relative to its matched Greedy final rank pattern and completed 1,950 optimization steps.

`Best accuracy` is the selected best checkpoint's RTE accuracy. `Selected` reports that checkpoint's active-model parameter count and total active rank; `final` reports the completed allocation trajectory. Active-model counts include the fixed trainable task head and active LoRA components, not inactive reserves or the frozen backbone.

| Seed | C-GEAR best accuracy | Greedy best accuracy | C-GEAR selected active params / rank | C-GEAR final active params / rank | Greedy final active params / rank | Stop reason (step) |
| ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| 41 | 0.884477 | 0.859206 | 836,456 / 102 | 853,357 / 107 | 924,050 / 144 | minimum consolidation window (1,650) |
| 42 | 0.873646 | 0.877256 | 758,090 / 72 | 758,090 / 72 | 940,178 / 144 | zero-rank patience exhausted (300) |
| 43 | 0.862816 | 0.884477 | 771,920 / 78 | 806,492 / 90 | 935,570 / 144 | zero-rank patience exhausted (1,300) |
| 44 | 0.888087 | 0.873646 | 811,103 / 93 | 812,640 / 94 | 935,570 / 144 | zero-rank patience exhausted (1,000) |
| 45 | 0.880866 | 0.870036 | 835,689 / 103 | 835,689 / 103 | 933,266 / 144 | minimum consolidation window (1,650) |

Descriptively, C-GEAR's mean best accuracy is 0.877978 versus 0.872924 for the matched Greedy runs (mean paired difference +0.005054, with C-GEAR higher on 3 of 5 seeds). Mean C-GEAR final active-model count is 813,253.6 versus 933,726.8 for Greedy, a mean per-seed reduction of 12.8819%. These five local diagnostic runs are development evidence only; they do not establish statistical significance or universal superiority.

## Provenance

- C-GEAR run roots: `NLU/output/glue/rte_allocator/genetic_budgeted_calibrated/rte_calibrated_v3_reservefix_retry1_s{seed}_seed{seed}_budget0.94`
- Greedy run roots: `NLU/output/glue/rte_allocator/greedy/rte_greedy_fresh_s{seed}_seed{seed}`
- Best accuracy and checkpoint selection: each run's `model/trainer_state.json`
- Active parameter and rank accounting: C-GEAR final-verification records, saved dynamic rank patterns, and `NLU/scripts/rank=2/report_rte_allocator_parameters.py`
- Stop reason: each C-GEAR run's `terminal.log`
- Machine-readable values: `results/cgear_rte_diagnostic_41_45.json`

The source run directories and checkpoint artifacts remain local and are intentionally not versioned.
