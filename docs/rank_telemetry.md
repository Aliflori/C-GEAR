# Dynamic-rank telemetry and offline analysis

Rank telemetry is a lightweight, append-only JSON Lines record of training and
allocator decisions. It is evidence, not a control input: the offline tools in
`analysis/` never import the training stack, load checkpoints, or change a run.
Historical terminal logs and the existing `results/rte` reporting pipeline are
not inputs to this format and remain unchanged.

## Enabling and location

Pass one master switch to a dynamic-rank GLUE run:

```text
--rank_telemetry true
```

The maintained RTE and SST-2 allocator wrappers already pass this switch. The
argument defaults to `false` so historical/custom workflows are not changed
implicitly. On the world process only, records are written to:

```text
<root_output_dir>/telemetry.jsonl
```

A fresh training run truncates a stale telemetry file in its newly selected
output directory; a normal checkpoint resume appends a new `run_start`
segment. Evaluation-only reuse of an existing output root also appends instead
of destroying prior training evidence. The
same schema covers Greedy IncreLoRA and C-GEAR
(`genetic_budgeted_calibrated`); fields that do not apply to Greedy, such as
calibration folds, are simply absent.

Telemetry reads only ranks, parameter counts, already-computed scores/losses,
metrics, paths, and timers at existing execution boundaries. It performs no
model forward/backward pass, evaluation, candidate generation, calibration,
or RNG-based operation. Each complete JSON object is newline terminated and
flushed immediately. Therefore, completed records are visible on disk after a
crash; at worst, a process-level write failure can leave one final unterminated
partial line. The parser warns about and ignores only that final fragment while
preserving all completed lines, including when the fragment ends inside a
multibyte UTF-8 character. A valid final JSON object does not require a newline;
malformed newline-terminated records remain errors.
An explicitly requested sink that cannot be opened fails clearly during
initialization. A later optional serialization/write failure is logged and
disables telemetry rather than changing training control flow.

For the 72-module RTE configuration, expected size is approximately 0.2–0.3 MB
for Greedy and 0.4–0.7 MB for C-GEAR, depending on the number of calibration
candidates and checkpoints. This remains tiny relative to model checkpoints.

## JSONL contract

Every nonblank line is one JSON object and must contain:

| Field | Type | Meaning |
|---|---|---|
| `schema_version` | string | Exactly `rank_telemetry.v1` |
| `event_type` | string | One of the canonical types below |
| `global_step` | nonnegative integer | Completed optimizer steps at emission |
| `seed` | nonnegative integer | Training seed |
| `method` | nonempty string | Stable user-facing method name |
| `wall_time_seconds` | nonnegative finite number | Elapsed run time from one monotonic clock |

Canonical event types are:

- `run_start`
- `allocation_event`
- `calibration_event`
- `candidate_selection`
- `evaluation`
- `checkpoint_save`
- `allocator_stop`
- `run_end`
- `warning`

Additional fields are permitted and ignored by older postprocessors. Changing
the meaning or type of an established field requires a new schema version.
Within one file, `method` and `seed` are constant and `global_step` is
nondecreasing. `wall_time_seconds` is nondecreasing within an uninterrupted
process segment. Blank lines, raw JSON `NaN`/`Infinity` constants, unknown event
types, and malformed records are rejected.

An appended checkpoint resume may add another `run_start` record at its resume
step. The parser therefore permits repeated lifecycle markers while continuing
to enforce one constant run identity. A resumed `run_start` is the only event
allowed to reset its global step or process-local wall clock. This matters when
the original process wrote evidence beyond checkpoint step N before crashing:
the abandoned tail remains segment 0, while a resume from N begins segment 1.
The parser never connects the abandoned and resumed lineages.

The writer's canonical string sentinels `NaN`, `Infinity`, and `-Infinity` are
accepted only in calibration score/fold fields for an explicitly invalid
candidate. They are retained as provenance but emitted as unavailable numeric
CSV cells, so plotting cannot silently treat them as real scores.

### Rank and allocation fields

An `allocation_event` must include `module_active_ranks`, a nonempty mapping of
full module names to nonnegative integer active ranks. Recommended fields are:

```json
{
  "schema_version": "rank_telemetry.v1",
  "event_type": "allocation_event",
  "global_step": 300,
  "seed": 41,
  "method": "C-GEAR",
  "wall_time_seconds": 245.7,
  "module_active_ranks": {
    "deberta.encoder.layer.0.attention.self.query_proj": 2
  },
  "total_active_rank": 73,
  "pre_total_active_rank": 72,
  "post_total_active_rank": 73,
  "selected_k": 1,
  "selected_event_rank": 1,
  "selected_modules": [
    "deberta.encoder.layer.0.attention.self.query_proj"
  ],
  "selected_source": "calibrated_global_ga",
  "active_model_parameter_count": 700123,
  "pre_active_parameter_count": 698586,
  "post_active_parameter_count": 700123,
  "runtime_trainable_parameter_count": 812345,
  "full_model_parameter_count": 184000000,
  "budget_limit": 868607,
  "budget_used": 700123,
  "budget_remaining": 168484,
  "rank_increments": {
    "deberta.encoder.layer.0.attention.self.query_proj": 1
  },
  "allocation_stopped": false
}
```

When consecutive complete rank maps exist, the parser verifies that ranks do
not decrease and that their summed delta equals `selected_event_rank`. A
zero-growth event therefore has `selected_event_rank: 0`, an empty
`selected_modules` list, and an unchanged rank map. `total_active_rank`, when
present, must equal the sum of the module map. Parameter-count fields use the
project's established meanings:

- `active_model_parameter_count`: fixed trainable task components plus active
  LoRA A/E/B components represented by the rank map;
- `runtime_trainable_parameter_count`: raw `requires_grad=True` parameters,
  including preparatory reserves;
- `full_model_parameter_count`: all model parameters.

A `run_start` should include the initial complete rank map and counts. If an
`allocator_stop` omits the unchanged map, the parser carries forward the last
verified snapshot for plotting and records `stop_reason` without inventing an
intermediate state.

### Selected checkpoint versus final trajectory

With `load_best_model_at_end`, the model resident at normal completion can be
an earlier selected checkpoint, while the allocation schedule's final
architecture was reached later in the training trajectory. A canonical
`run_end` therefore carries both snapshots with deliberately different names:

| State | Rank/map fields | Parameter-count fields |
|---|---|---|
| Selected best checkpoint | `selected_active_rank`, `selected_module_active_ranks` | `selected_active_parameter_count`, `selected_runtime_trainable_parameter_count`, `selected_full_model_parameter_count`, `selected_physical_rank_component_count` |
| Final allocation trajectory | `final_active_rank`, `final_module_active_ranks` | `final_active_parameter_count`, `final_runtime_trainable_parameter_count`, `final_full_model_parameter_count`, `final_physical_rank_component_count` |

The ordinary `module_active_ranks`, `total_active_rank`, and parameter-count
snapshot on `run_end` describe the currently loaded selected checkpoint and
must agree with the explicit `selected_*` aliases. The parser preserves the two
terminal states as separate rows in `rank_trajectory.csv` and separate module
snapshots in `module_rank_trajectory.csv`, labeled by `state_role`:

- `selected_best_checkpoint`
- `final_trajectory`

Earlier rows are labeled `initial_trajectory` or `trajectory`. Selected-best
accuracy may be paired only with the selected checkpoint's parameter/rank
state. It must never be paired with final-trajectory parameters and presented
as one deployed model. Conversely, final-architecture plots use only the final
trajectory and do not silently substitute the selected checkpoint.

### Calibration fields

Both supported encodings preserve the same candidate-level evidence.

One event containing a candidate array:

```json
{
  "schema_version": "rank_telemetry.v1",
  "event_type": "calibration_event",
  "global_step": 300,
  "seed": 41,
  "method": "C-GEAR",
  "wall_time_seconds": 244.1,
  "selected_candidate_id": "candidate-0",
  "candidates": [
    {
      "candidate_id": "candidate-0",
      "candidate_modules": ["deberta.encoder.layer.0.attention.self.query_proj"],
      "candidate_size": 1,
      "candidate_cost": 1537,
      "candidate_family": "global_ga",
      "fold_gains": [0.013, 0.009, 0.011],
      "calibration_gain_mean": 0.011,
      "calibration_gain_std": 0.001633,
      "calibration_gain_lcb": 0.010184,
      "calibration_gain_per_parameter": 0.00000716,
      "calibration_valid": true,
      "invalid_reason": null
    }
  ]
}
```

Alternatively, emit one `calibration_event` per candidate with those fields at
the top level (or under a `candidate` object). The parser flattens all variants
into one row per candidate. A `candidate_selection` at the same step identifies
the winner with `selected_candidate_id`, `selected_modules`, or a
`selected_candidate` object. Candidate-level `is_selected` is also accepted.

### Evaluation and lifecycle fields

- `evaluation` records accuracy, when available, as top-level `accuracy`,
  `metrics.accuracy`, or `metrics.eval_accuracy`; if more than one is present,
  the values must agree. Tasks with another metric still produce an evaluation
  row with a blank `accuracy` cell and preserve the complete `metrics` object.
  Optional fields include `loss`, `split`, active counts, rank, and `checkpoint`
  (with `checkpoint_path` accepted as an alias).
  `state_role=training_trajectory_evaluation` identifies evaluations performed
  during the training lineage. A post-training evaluation after loading the
  selected checkpoint uses `state_role=selected_best_checkpoint_evaluation`
  and `evaluated_checkpoint`. The parser retains both roles, but trajectory
  plots do not connect the selected-checkpoint evaluation to the training
  curve merely because it was emitted at the final global step.
  An evaluation-only invocation with no training lineage uses
  `state_role=standalone_evaluation`; it is retained in the CSV but likewise
  excluded from training-trajectory curves.
- `checkpoint_save` records a checkpoint path or identifier in `checkpoint`
  (or the `checkpoint_path` alias).
- `allocator_stop` records `stop_reason`; `allocation_stopped` is treated as
  true.
- `warning` should carry a concise `message` and does not become a scientific
  measurement.
- `run_end` is emitted only after normal completion and carries the distinct
  selected-best and final-trajectory snapshots described above. Its
  `best_accuracy` and `best_checkpoint` identify the selected result; absence
  of `run_end` means the telemetry is incomplete.

## Parsing

From the repository root and existing environment:

```bash
source /home/ali/LoRa_Project/.venv/bin/activate
python analysis/parse_rank_telemetry.py \
  /path/to/root_output_dir/telemetry.jsonl \
  --output-dir /path/to/derived/rank_telemetry
```

Multiple JSONL files may be supplied. Each source remains distinguishable via
the absolute `source_artifact` column. Inputs are validated completely before
their rows are written, and all five files are written atomically:

| Output | Grain |
|---|---|
| `rank_trajectory.csv` | Rank-bearing lifecycle/allocation state; terminal states distinguished by `state_role` |
| `module_rank_trajectory.csv` | Module × rank-bearing state; terminal states distinguished by `state_role` |
| `allocation_events.csv` | Allocation event |
| `calibration_events.csv` | Calibrated candidate |
| `evaluation_trajectory.csv` | Evaluation event |

Every output is created with a header even when that telemetry family is not
available. The parser does not interpolate ranks, infer missing evaluation
accuracy, or reconstruct events from prose logs.

Every derived table includes `run_segment`, a zero-based integer assigned in
file order. Segment 0 is the initial process lineage and each appended
`run_start` increments it. Rank validation, carried count state, and
candidate-selection matching are isolated within each segment. This preserves an
abandoned pre-crash tail as evidence without letting it supply ranks, counts,
or a same-step candidate selection to the resumed lineage.

Derived CSV vocabulary is intentionally stable: `budget_limit` becomes
`target_active_parameter_count`, `budget_remaining` becomes
`remaining_hard_budget`, and both `selected_k` (number of selected modules) and
`selected_event_rank` (summed rank increment) are retained separately.

## Plotting

```bash
python analysis/plot_rank_telemetry.py \
  --input-dir /path/to/derived/rank_telemetry \
  --output-dir /path/to/derived/rank_telemetry/figures
```

The plotter reads only the derived CSVs. It produces PDF and PNG by default and
generates each family only when its required evidence exists. Trajectory plots
exclude `selected_best_checkpoint` rows, preventing an artificial jump at the
normal-completion step. Final-architecture plots explicitly select the
`final_trajectory` snapshot. If a crash leaves no `run_end`, they are labeled
"latest observed" and "run incomplete" instead of "final":

Each `run_segment` is a separate plotted lineage and a separate per-run figure.
Training accuracy curves include only
`training_trajectory_evaluation`; selected-best-checkpoint evaluations remain
available in `evaluation_trajectory.csv` for matched reporting but are not
joined to the optimization-step trajectory.

1. total active rank versus optimization step;
2. active adaptation/model parameters versus step;
3. accuracy versus step;
4. accuracy versus active parameters;
5. module-wise rank heatmap over recorded rank events;
6. final layer-wise rank bar chart;
7. cumulative rank allocated by transformer layer;
8. final attention-family versus FFN rank allocation;
9. selected allocation size `k` (number of modules) versus step;
10. selected calibration mean/LCB trajectory;
11. candidate mean versus uncertainty, colored by LCB.

Zero-growth decisions are shown as hollow points in the selected-`k` plot, and
allocator-stop steps are marked on step-based plots. These annotations are
drawn only from explicit `allocation_event` and `allocator_stop` records.
Module layer/family labels are derived from full module names; unrecognized
names remain in the module heatmap and are classified as `other` for aggregate
plots.

Per-run figure names include the method, seed, run segment, source-directory
label, and a short source-identity digest. Thus two inputs or resume segments
with the same method and seed do not overwrite one another. The selected-`k`
title is method-neutral so Greedy
and C-GEAR telemetry can be plotted alone or together without relabeling a
baseline as C-GEAR.

## Lightweight validation

No training is needed:

```bash
python -m unittest analysis/tests/test_parse_rank_telemetry.py
python analysis/parse_rank_telemetry.py --help
python analysis/plot_rank_telemetry.py --help
```

The tests create temporary synthetic telemetry only. They cover nested and
candidate-level calibration, selection matching, output schemas, module-family
classification, selected-best/final-trajectory separation, crash-tail
recovery, rollback resume segmentation, evaluation roles, strict JSON
constants, and rejection of schema/rank inconsistencies.
