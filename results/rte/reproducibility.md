# RTE reproducibility record

## Scientific method settings

The canonical wrapper is [`NLU/scripts/rank=2/run_rte_allocator.sh`](../../NLU/scripts/rank=2/run_rte_allocator.sh). The values below were checked against both that script and serialized run metadata.

| Setting | Verified value |
| :--- | :--- |
| Task / metric | GLUE RTE / accuracy |
| Backbone | `microsoft/deberta-v3-base` |
| Seeds | 41, 42, 43, 44, 45 |
| Training / validation examples | 2,490 / 277 |
| Sequence length | 192 |
| Epochs / optimization steps | 25 / 1,950 |
| Train / evaluation batch size | 32 / 8 per device |
| Gradient accumulation | 1 |
| Optimizer | repository Transformers AdamW |
| Learning rate / scheduler | `1.2e-3` / linear |
| Adam betas / epsilon | 0.9, 0.999 / `1e-8` |
| Weight decay / max gradient norm | 0.01 / 1.0 |
| Scheduler warmup | 100 steps |
| Precision | FP16 training; `fp16_full_eval=False` |
| Classification dropout | 0.20 |
| Orthogonal regularization coefficient | 0.3 |
| LoRA type / alpha | SVD / 32 |
| Adapted modules | query, key, value, intermediate, layer output, attention output (72 modules) |
| Initial / maximum total active rank | 72 / 144 (initial rank 1, target average rank 2) |
| Allocation warmup / interval | 100 / 100 steps; first eligible event is step 200 |
| Maximum event size | `top_h=5` |
| Importance EMA betas | 0.85 / 0.85 (not Adam optimizer betas) |
| Evaluation / checkpoint interval | 100 / 100 steps |
| Checkpoint retention | 1 retained checkpoint; best model selected by accuracy |

Greedy IncreLoRA uses the original importance threshold/tie rule. C-GEAR uses population 12, 4 generations, mutation 0.10, crossover 0.80, interaction weight 0.20, redundancy weight 0.20, cost weight 0.30, diversity weight 0.10, and no local search.

C-GEAR-specific settings are: active-parameter budget ratio 0.94; legal event sizes 0–5; maximum two Greedy-anchor replacements; three calibration folds with paired batches of eight training examples; calibration seed offset 1000; shortlist size six; LCB beta 0.5; Greedy quality-floor ratio 0.99; relative quality tolerance 0.01; two-event no-growth stopping patience; 300-step minimum consolidation window; and 25-step new-rank warmup.

## Resource-driven execution settings

- Virtual environment: `/home/ali/LoRa_Project/.venv`
- Recorded device: `cuda:0`, one GPU, single-process execution
- External GPU selection used for the local workflow: `CUDA_VISIBLE_DEVICES=0`
- The GPU make/model, VRAM capacity, driver version, and CUDA toolkit version were not serialized and are therefore unavailable.
- `runtime_seconds` in the canonical CSV is Trainer's `train_runtime`, not end-to-end shell wall time and not evaluation runtime.

Resource choices such as GPU identity do not alter the allocator definition, but they can affect runtime and floating-point execution.

## Safe command templates

From `/home/ali/LoRa_Project/IncreLoRA/NLU`, after activating the virtual environment, a new Greedy run is launched with a unique experiment name:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/rank=2/run_rte_allocator.sh \
  <new_greedy_experiment_name> greedy <seed>
```

After that matched Greedy run produces `rank_pattern.json`, C-GEAR is launched with another unique name:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/rank=2/run_rte_allocator.sh \
  <new_cgear_experiment_name> \
  genetic_budgeted_calibrated \
  <seed> \
  <absolute_path_to_matched_greedy_rank_pattern.json> \
  0.94
```

Existing completed experiment names must not be reused because the calibrated wrapper intentionally refuses to overwrite them.

## Local artifact provenance nuance

The completed C-GEAR logs name `greedy/ali_rte_greedy_s{seed}_seed{seed}/rank_pattern.json` as their original budget reference. The canonical comparison uses `greedy/rte_greedy_fresh_s{seed}_seed{seed}`. For all five seeds, their final and selected rank patterns, selected steps, and accuracies match exactly. Only runtime differs; the canonical CSV uses the fresh-run runtime consistently.

## Regenerating summaries and figures

```bash
cd /home/ali/LoRa_Project/IncreLoRA
source /home/ali/LoRa_Project/.venv/bin/activate
/home/ali/.local/bin/uv pip install \
  --python /home/ali/LoRa_Project/.venv/bin/python \
  -r analysis/requirements.txt
python analysis/summarize_rte_results.py
```

The script validates the canonical CSV, rewrites `results/rte/summary.json`, and regenerates all PDF/PNG figures from tracked tabular data. It does not load checkpoints or run training.
