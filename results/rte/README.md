# C-GEAR vs. Greedy IncreLoRA on GLUE RTE

This directory organizes five matched local diagnostic runs (seeds 41–45) for the GLUE Recognizing Textual Entailment task. The backbone is `microsoft/deberta-v3-base`; the baseline is **Greedy IncreLoRA**, and `rank_allocator=genetic_budgeted_calibrated` implements the proposed **C-GEAR** method.

These are controlled local development experiments, not a reproduction of paper-reported hardware-dependent numbers. Five seeds help inspect robustness, but this small diagnostic set does not establish statistical significance or universal superiority.

## Canonical evidence

- [`cgear_vs_greedy_seeds_41_45.csv`](cgear_vs_greedy_seeds_41_45.csv): one canonical row per method and seed.
- [`summary.json`](summary.json): descriptive statistics regenerated from the canonical CSV.
- [`cgear_active_rank_trajectories.csv`](cgear_active_rank_trajectories.csv): exact C-GEAR rank states parsed from allocation logs; no interpolated measurements.
- [`parameter_accounting.md`](parameter_accounting.md): active-parameter definitions and validation.
- [`cgear_method_evidence.md`](cgear_method_evidence.md): implementation evidence map.
- [`reproducibility.md`](reproducibility.md): verified method and execution settings.
- [`../../analysis/summarize_rte_results.py`](../../analysis/summarize_rte_results.py): validation, descriptive statistics, and figure generation.

The earlier preservation snapshot in `results/cgear_rte_diagnostic_41_45.{md,json}` remains unchanged for history. This directory is the canonical organized reporting layer and does not rewrite that prior commit.

“Active adaptation/model parameters” means the fixed trainable non-LoRA component plus the LoRA A/E/B components represented by the active rank pattern. It does **not** mean that the frozen DeBERTa backbone was pruned or made smaller. Physical reserve capacity, raw runtime trainability, and full-model size are separate quantities.

## Selected-checkpoint evidence

Accuracy and active parameters in this table belong to the same selected best checkpoint. `Correct` is exactly derived as `accuracy × 277`; per-example prediction files were not saved.

| Seed | Method | Selected step | Accuracy | Correct | Selected active rank | Selected active parameters |
| ---: | :--- | ---: | ---: | ---: | ---: | ---: |
| 41 | Greedy IncreLoRA | 1,200 | 0.859206 | 238 / 277 | 122 | 876,412 |
| 41 | C-GEAR | 1,600 | 0.884477 | 245 / 277 | 102 | 836,456 |
| 42 | Greedy IncreLoRA | 300 | 0.877256 | 243 / 277 | 77 | 774,991 |
| 42 | C-GEAR | 700 | 0.873646 | 242 / 277 | 72 | 758,090 |
| 43 | Greedy IncreLoRA | 1,800 | 0.884477 | 245 / 277 | 144 | 935,570 |
| 43 | C-GEAR | 600 | 0.862816 | 239 / 277 | 78 | 771,920 |
| 44 | Greedy IncreLoRA | 400 | 0.873646 | 242 / 277 | 82 | 787,284 |
| 44 | C-GEAR | 700 | 0.888087 | 246 / 277 | 93 | 811,103 |
| 45 | Greedy IncreLoRA | 600 | 0.870036 | 241 / 277 | 92 | 818,782 |
| 45 | C-GEAR | 1,600 | 0.880866 | 244 / 277 | 103 | 835,689 |

## Final-architecture evidence

The final trajectory records the architecture after allocation and consolidation. It was not separately evaluated after Trainer restored the best checkpoint, so these final counts are not paired with the accuracy above as a deployed-model result.

| Seed | Greedy final rank / active parameters | C-GEAR final rank / active parameters | Matched parameter reduction | C-GEAR stop reason (step) |
| ---: | ---: | ---: | ---: | :--- |
| 41 | 144 / 924,050 | 107 / 853,357 | 7.6503% | minimum consolidation window (1,650) |
| 42 | 144 / 940,178 | 72 / 758,090 | 19.3674% | zero-rank patience exhausted (300) |
| 43 | 144 / 935,570 | 90 / 806,492 | 13.7967% | zero-rank patience exhausted (1,300) |
| 44 | 144 / 935,570 | 94 / 812,640 | 13.1396% | zero-rank patience exhausted (1,000) |
| 45 | 144 / 933,266 | 103 / 835,689 | 10.4554% | minimum consolidation window (1,650) |

Greedy IncreLoRA has no adaptive stopping event, so its stop fields are not applicable.

## Descriptive aggregates

| Method | Mean accuracy | Accuracy sample SD | Mean selected active parameters / rank | Mean final active parameters / rank |
| :--- | ---: | ---: | ---: | ---: |
| Greedy IncreLoRA | 0.872924 | 0.009345 | 838,607.8 / 103.4 | 933,726.8 / 144.0 |
| C-GEAR | 0.877978 | 0.010018 | 802,651.6 / 89.6 | 813,253.6 / 93.2 |

The mean paired C-GEAR minus Greedy accuracy difference is **+0.005054** (+0.5054 percentage points), with **3 wins, 0 ties, and 2 losses**. At selected checkpoints, the mean matched active-parameter reduction is 35,956.2 parameters (3.8283% as the mean of per-seed percentages); C-GEAR instead uses more selected active parameters on seeds 44 and 45. For the separate final architectures, the mean matched reduction is 120,473.2 parameters (12.8819%). These are descriptive results only.

## Artifact provenance

For seed `S`, the canonical comparison reads:

- Greedy root: `NLU/output/glue/rte_allocator/greedy/rte_greedy_fresh_sS_seedS`
- C-GEAR root: `NLU/output/glue/rte_allocator/genetic_budgeted_calibrated/rte_calibrated_v3_reservefix_retry1_sS_seedS_budget0.94`
- Best step and accuracy: `model/trainer_state.json`
- Independently matching selected-model accuracy and 277-example count: `model/Eval_rte_results.json`
- Training runtime: `model/train_results.json`
- Selected checkpoint rank metadata: `model/checkpoint-STEP/config.json`
- Final rank metadata: `rank_pattern.json`
- C-GEAR stop and event evidence: `terminal.log`

The selected Greedy checkpoint steps for seeds 41–45 are 1,200, 300, 1,800, 400, and 600. The selected C-GEAR steps are 1,600, 700, 600, 700, and 1,600.

Each C-GEAR run originally used `NLU/output/glue/rte_allocator/greedy/ali_rte_greedy_sS_seedS/rank_pattern.json` as its hard-budget reference. For every seed, that pattern is exactly equal to the canonical `rte_greedy_fresh` final pattern; their selected checkpoint patterns, selected steps, and accuracies also match. Their measured training runtimes differ, so the canonical table consistently uses only the `rte_greedy_fresh` runtime rather than mixing sources.
