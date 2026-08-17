# Conservative cleanup inventory

This inventory was recorded before any artifact was moved. The authoritative
upstream comparison was a temporary read-only clone of
<https://github.com/FeiyuZhang98/IncreLoRA> at commit
`51a2d734af7883ef90425206057c1ef25997b359`. Classification also used the
current Git history, final-report scripts, canonical data, README, telemetry,
and exact run paths.

## KEEP_ACTIVE

| Current path or family | Origin | Purpose | Reason |
|---|---|---|---|
| `NLU/output/glue/rte_allocator/greedy/ali_last_seed{41..46}` | project output | Official matched Greedy evidence | Exact paths consumed by the final six-seed pipeline. |
| `NLU/output/glue/rte_allocator/genetic_budgeted_calibrated/ali_last_seed{41..46}_budget0.94` | project output | Official matched C-GEAR evidence | Exact paths consumed by the final six-seed pipeline. |
| `final_report/` | project-added | Canonical six-seed data, report, figures, and regeneration scripts | Final scientific deliverable and source of truth. |
| `analysis/parse_rank_telemetry.py`, `analysis/plot_rank_telemetry.py`, `analysis/requirements.txt`, `analysis/tests/` | project-added | Canonical telemetry parsing, plotting, and tests | Used by the final report and reproducibility workflow. |
| `NLU/scripts/rank=2/run_rte_allocator.sh` | project-added | Final Greedy/C-GEAR RTE launcher | Exact launcher used for final runs. |
| `NLU/scripts/rank=2/run_sst2_allocator.sh`, `run_sst2_fast.sh` | project-added | Maintained cross-task allocator/smoke wrappers | Meaningful non-RTE functionality, not scratch clutter. |
| C-GEAR, calibration, budget, checkpoint-accounting, and telemetry modules under `NLU/src/transformers/` and `loralib/` | project-added/modified | Final method plus retained ablation and compatibility paths | Required by current implementation, checkpoints, or regression coverage. |
| `NLU/tests/test_evo_allocator.py`, `test_budgeted_evo_allocator.py`, `test_calibrated_budgeted_evo_allocator.py`, `test_rank_telemetry.py` | project-added | Allocator, calibration, reserve, checkpoint, accounting, RNG, and telemetry regressions | Scientifically meaningful coverage; conservatively retained. |
| `docs/`, root `README.md`, and `NLU/scripts/rank=2/report_rte_allocator_parameters.py` | project-added/modified | Documentation and parameter-audit tooling | Still useful for review and reproducibility. |

## KEEP_UPSTREAM

| Current path or family | Origin | Purpose | Reason |
|---|---|---|---|
| `NLU/examples/`, original `NLU/scripts/`, original `NLU/src/`, and upstream configuration/tests | upstream | Full NLU and GLUE support | Preserves SST-2, MNLI, MRPC, CoLA, QNLI, QQP, STS-B, RTE, and other upstream functionality. |
| `NLG_QA/` | upstream | Original generation/question-answering support | Explicitly protected upstream scope. |
| `loralib/` | upstream plus C-GEAR modifications | Core LoRA/IncreLoRA implementation | Required upstream and final method code. |
| Upstream README, LICENSE, setup, environment, and download infrastructure | upstream | Installation, licensing, and reproduction | Authoritative project infrastructure. |

## MOVE_TO_HISTORY

| Current path or family | Origin | Purpose | Reason |
|---|---|---|---|
| `NLU/output/glue/rte/` | project output | Early RTE/fast runs | Superseded and not referenced by the official analysis. |
| `NLU/output/glue/sst2_fast/` | project output | Earlier SST-2 smoke output | Useful history, but not an official final RTE input. |
| `NLU/output/glue/rte_allocator/ali_last` | project output | Stray/incomplete root-level log directory | Not one of the twelve canonical run paths. |
| `NLU/output/glue/rte_allocator/genetic_budgeted/` and `genetic_budgeted_eval/` | project output | Pre-calibration budgeted GA and one-off checkpoint evaluation | Superseded by final calibrated runs and not referenced by final reporting. |
| Non-`ali_last` children of `genetic_budgeted_calibrated/` | project output | v2, v3, reserve-fix, and retry development generations | Superseded; exact canonical final children remain active. |
| Non-official children of `greedy/` (`ali_last_seed4`, `ali_rte_greedy_*`, `rte_greedy_fresh_*`, `rte_greedy_s42`, `rte_greedy_s43`) | project output | Earlier/duplicate Greedy generations | Superseded; exact canonical final children remain active. |
| Loose `NLU/*rte*gaonly*.log` and `NLU/ali_rte_greedy_*.log` files | project output | Early standalone terminal logs | Not referenced by the final pipeline; official logs remain inside canonical runs. |
| `NLU/scripts/rank=2/run_rte_fast.sh` | project-added | Old seed-41-only RTE wrapper | Superseded one-off wrapper, not the current allocator launcher. |
| `NLU/scripts/rank=2/run_rte_budgeted_checkpoint_eval.sh` | project-added | Old budgeted checkpoint re-evaluation wrapper | One-off pre-C-GEAR diagnostic tied to superseded output. |
| `analysis/summarize_rte_results.py` and root `results/` | project-added | Earlier five-seed analysis/snapshot | Superseded by canonical six-seed `final_report/` data and scripts. |
| `HEY ALI.zip` | project package | Temporary review transfer | Fully reproducible package; archive rather than delete. |

## UNCERTAIN

| Current path or family | Origin | Purpose | Decision |
|---|---|---|---|
| `NLU/scripts/rank=2/report_rte_allocator_parameters.py` | project-added | Standalone selected/final parameter comparison | Useful audit CLI despite not being the canonical six-seed generator; retained untouched. |
| Legacy allocator-mode source and its regression tests | project-added | Ablation/backward-compatibility behavior | Retained because checkpoints and scientific comparisons may depend on it. |
| Any unlisted output, script, source, test, or documentation file | mixed | Not proven obsolete | Retained untouched under the conservative rule. |
