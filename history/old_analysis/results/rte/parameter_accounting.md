# RTE active-parameter accounting

## Count used for scientific comparisons

For an adapted `SVDLinear` module (m), one active rank component contains:

- one row of LoRA A: `in_features_m` parameters;
- one LoRA E scalar: 1 parameter;
- one column of LoRA B: `out_features_m` parameters.

Therefore:

```text
rank_one_cost_m = in_features_m + 1 + out_features_m

active_model_parameter_count
  = fixed_non_dynamic_trainable_parameters
  + sum_m(active_rank_m * rank_one_cost_m)
```

The implementation is [`get_module_rank_one_cost`](../../loralib/loralib/increlora.py#L27) and [`get_active_model_parameter_count`](../../loralib/loralib/increlora.py#L65). Versioned rank patterns are independently recomputed by [`get_rank_pattern_active_model_parameter_count`](../../loralib/loralib/increlora.py#L146), and the cross-method report is validated by [`build_rank_budget_report`](../../NLU/src/transformers/rank_budget_reporting.py#L253).

For these RTE runs, the verified fixed non-dynamic trainable component is **592,130 parameters**. It contains the trainable non-LoRA parameters selected by this project configuration (including the task classifier/pooler and other configured non-LoRA biases or normalization parameters); it is not the frozen DeBERTa weight matrices.

The 72 adapted modules have three verified cost classes:

| Module shape | Module count | Per-rank A/E/B cost |
| :--- | ---: | ---: |
| 768 → 768 | 48 | 1,537 |
| 768 → 3,072 | 12 | 3,841 |
| 3,072 → 768 | 12 | 3,841 |

This is why two allocation events with the same added total rank can have different parameter costs.

## Counts that must remain distinct

- **Active adaptation/model parameters**: the fixed non-dynamic trainable component plus A/E/B components represented by the active module-rank pattern. This is the hard-budget and Greedy-reference metric.
- **Runtime trainable parameters**: the raw sum of parameters with `requires_grad=True`. During growth it may include preparatory reserve components, so it can exceed the active count.
- **Physical reserve capacity**: materialized future LoRA components. Inactive reserves are excluded from the active count even when they temporarily require gradients.
- **Full model parameters**: every parameter, independently of trainability. No result here claims that the full DeBERTa backbone was reduced.

These distinctions are implemented by [`get_active_model_parameter_count`](../../loralib/loralib/increlora.py#L65), [`get_runtime_trainable_parameter_count`](../../loralib/loralib/increlora.py#L53), and [`get_full_model_parameter_count`](../../loralib/loralib/increlora.py#L59).

## Selected checkpoint versus final trajectory

The selected checkpoint and the final allocation trajectory can have different rank patterns. Accuracy is reported with the selected checkpoint's active count. Final active count is architecture-only evidence unless that final architecture is evaluated separately. The canonical CSV keeps both fields, and [`report_rte_allocator_parameters.py`](../../NLU/scripts/rank=2/report_rte_allocator_parameters.py) verifies both without loading model weights.

All selected and final counts for seeds 41–45 were verified exactly. No seed has an estimated or unresolved active-parameter count.
