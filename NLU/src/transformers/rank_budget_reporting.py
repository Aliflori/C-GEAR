# coding=utf-8
"""Model-free parameter reporting for dynamic IncreLoRA checkpoints.

This module deliberately reads checkpoint metadata rather than model weights.  It
keeps the final allocation trajectory comparison separate from the comparison of
the checkpoints selected by ``load_best_model_at_end``.
"""

import argparse
import json
import math
from pathlib import Path


REPORT_FORMAT_VERSION = 1


def _load_json(path, description):
    path = Path(path).expanduser()
    if not path.is_file():
        raise ValueError("{} does not exist or is not a file: {}".format(description, path))
    try:
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, ValueError) as error:
        raise ValueError("Could not read {} '{}': {}".format(description, path, error))
    if not isinstance(payload, dict):
        raise ValueError("{} must contain a JSON object: {}".format(description, path))
    return path.resolve(), payload


def _integer(value, description, minimum=0):
    if isinstance(value, bool):
        raise ValueError("{} must be an integer.".format(description))
    try:
        result = int(value)
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("{} must be an integer.".format(description))
    if not math.isfinite(numeric) or numeric != result or result < minimum:
        raise ValueError("{} must be an integer greater than or equal to {}.".format(description, minimum))
    return result


def _rank_map(pattern, description):
    if not isinstance(pattern, dict):
        raise ValueError("{} must be a JSON object.".format(description))
    modules = pattern.get("modules", pattern)
    if not isinstance(modules, dict) or not modules:
        raise ValueError("{} must contain a nonempty module map.".format(description))

    ranks = {}
    for name, value in modules.items():
        if not isinstance(name, str) or not name:
            raise ValueError("{} contains an invalid module name.".format(description))
        if isinstance(value, dict):
            if "active_rank" not in value:
                raise ValueError("{} module '{}' has no active_rank.".format(description, name))
            value = value["active_rank"]
        ranks[name] = _integer(
            value,
            "{} active rank for '{}'".format(description, name),
            minimum=1,
        )
    return ranks


def _accounting_basis(budgeted_pattern):
    modules = budgeted_pattern.get("modules")
    if not isinstance(modules, dict) or not modules:
        raise ValueError(
            "The budgeted rank pattern must be versioned and contain rich per-module metadata."
        )
    non_dynamic = _integer(
        budgeted_pattern.get("non_dynamic_trainable_params"),
        "budgeted non_dynamic_trainable_params",
        minimum=0,
    )
    costs = {}
    for name, metadata in modules.items():
        if not isinstance(metadata, dict) or "rank_one_cost" not in metadata:
            raise ValueError("Budgeted module '{}' has no rank_one_cost metadata.".format(name))
        costs[name] = _integer(
            metadata["rank_one_cost"],
            "rank_one_cost for '{}'".format(name),
            minimum=1,
        )
    return non_dynamic, costs


def _validate_module_set(ranks, costs, description):
    rank_names = set(ranks)
    cost_names = set(costs)
    if rank_names != cost_names:
        missing = sorted(cost_names.difference(rank_names))
        unexpected = sorted(rank_names.difference(cost_names))
        raise ValueError(
            "{} modules do not match the budgeted accounting basis: missing={} unexpected={}.".format(
                description, missing[:3], unexpected[:3]
            )
        )


def _validate_rich_reference_metadata(reference_pattern, budgeted_pattern, description):
    """Validate matched accounting metadata whenever the reference supplies it."""

    if "non_dynamic_trainable_params" in reference_pattern:
        reference_non_dynamic = _integer(
            reference_pattern["non_dynamic_trainable_params"],
            "{} non_dynamic_trainable_params".format(description),
            minimum=0,
        )
        budgeted_non_dynamic = _integer(
            budgeted_pattern.get("non_dynamic_trainable_params"),
            "budgeted non_dynamic_trainable_params",
            minimum=0,
        )
        if reference_non_dynamic != budgeted_non_dynamic:
            raise ValueError(
                "{} non-dynamic accounting does not match the budgeted model.".format(
                    description
                )
            )

    reference_modules = reference_pattern.get("modules")
    budgeted_modules = budgeted_pattern.get("modules")
    if not isinstance(reference_modules, dict) or not isinstance(budgeted_modules, dict):
        return
    for name, reference_metadata in reference_modules.items():
        budgeted_metadata = budgeted_modules.get(name)
        if not isinstance(reference_metadata, dict) or not isinstance(
            budgeted_metadata, dict
        ):
            continue
        for field in ("rank_one_cost", "in_features", "out_features"):
            if field not in reference_metadata or field not in budgeted_metadata:
                continue
            reference_value = _integer(
                reference_metadata[field],
                "{} {} for '{}'".format(description, field, name),
                minimum=1,
            )
            budgeted_value = _integer(
                budgeted_metadata[field],
                "budgeted {} for '{}'".format(field, name),
                minimum=1,
            )
            if reference_value != budgeted_value:
                raise ValueError(
                    "{} {} for '{}' does not match the budgeted model.".format(
                        description, field, name
                    )
                )


def _validate_checkpoint_identity(budgeted_config, greedy_config):
    compared = 0
    for field in (
        "model_type",
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "intermediate_size",
        "num_labels",
    ):
        if field not in budgeted_config or field not in greedy_config:
            continue
        compared += 1
        if budgeted_config[field] != greedy_config[field]:
            raise ValueError(
                "Greedy best checkpoint {} does not match the budgeted model: {} != {}."
                .format(field, greedy_config[field], budgeted_config[field])
            )
    return compared


def _active_count(ranks, non_dynamic, costs, description):
    _validate_module_set(ranks, costs, description)
    return int(non_dynamic + sum(ranks[name] * costs[name] for name in costs))


def _checkpoint_config(checkpoint, description):
    checkpoint = Path(checkpoint).expanduser()
    if not checkpoint.is_absolute():
        checkpoint = Path.cwd() / checkpoint
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_dir():
        raise ValueError("{} is not a checkpoint directory: {}".format(description, checkpoint))
    config_path, config = _load_json(checkpoint / "config.json", "{} config".format(description))
    dynamic = config.get("dynamic_lora_rank_pattern")
    if not isinstance(dynamic, dict) or not dynamic:
        raise ValueError(
            "{} config has no dynamic_lora_rank_pattern: {}".format(description, config_path)
        )
    return checkpoint, config, _rank_map(dynamic, "{} dynamic rank pattern".format(description))


def _resolve_best_checkpoint(trainer_state_path, trainer_state):
    supplied = trainer_state.get("best_model_checkpoint")
    if not isinstance(supplied, str) or not supplied.strip():
        raise ValueError("Budgeted trainer_state.json has no best_model_checkpoint.")

    raw = Path(supplied).expanduser()
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.extend(
            (
                Path.cwd() / raw,
                trainer_state_path.parent / raw,
                trainer_state_path.parent / raw.name,
            )
        )

    unique_candidates = []
    seen = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique_candidates.append(resolved)
        if resolved.is_dir() and (resolved / "config.json").is_file():
            return resolved

    raise ValueError(
        "Could not resolve best_model_checkpoint '{}' from {}. Checked: {}".format(
            supplied,
            trainer_state_path,
            [str(candidate) for candidate in unique_candidates],
        )
    )


def _required_mapping(mapping, key, description):
    value = mapping.get(key) if isinstance(mapping, dict) else None
    if not isinstance(value, dict):
        raise ValueError("{} has no '{}' object.".format(description, key))
    return value


def _stored_metric(metrics, key, description):
    return _integer(metrics.get(key), "{} {}".format(description, key), minimum=0)


def _reduction(candidate_count, reference_count):
    if reference_count <= 0:
        raise ValueError("A reference active-parameter count must be positive.")
    absolute = int(reference_count - candidate_count)
    return absolute, 100.0 * absolute / float(reference_count)


def build_rank_budget_report(
    budgeted_trainer_state,
    budgeted_rank_pattern,
    greedy_reference_checkpoint,
    greedy_reference_rank_pattern,
):
    """Build two independently validated active-parameter comparisons."""

    trainer_state_path, trainer_state = _load_json(
        budgeted_trainer_state, "budgeted trainer state"
    )
    budgeted_pattern_path, budgeted_pattern = _load_json(
        budgeted_rank_pattern, "budgeted rank pattern"
    )
    greedy_pattern_path, greedy_pattern = _load_json(
        greedy_reference_rank_pattern, "Greedy reference rank pattern"
    )

    non_dynamic, costs = _accounting_basis(budgeted_pattern)
    budgeted_root_ranks = _rank_map(budgeted_pattern, "budgeted selected-best rank pattern")
    greedy_final_ranks = _rank_map(greedy_pattern, "Greedy final rank pattern")

    budgeted_best_checkpoint = _resolve_best_checkpoint(trainer_state_path, trainer_state)
    budgeted_best_checkpoint, budgeted_best_config, budgeted_best_ranks = _checkpoint_config(
        budgeted_best_checkpoint, "budgeted best checkpoint"
    )
    greedy_best_checkpoint, greedy_best_config, greedy_best_ranks = _checkpoint_config(
        greedy_reference_checkpoint, "Greedy reference checkpoint"
    )

    _validate_module_set(budgeted_root_ranks, costs, "budgeted selected-best rank pattern")
    _validate_module_set(budgeted_best_ranks, costs, "budgeted best checkpoint")
    _validate_module_set(greedy_final_ranks, costs, "Greedy final rank pattern")
    _validate_module_set(greedy_best_ranks, costs, "Greedy best checkpoint")
    _validate_rich_reference_metadata(
        greedy_pattern, budgeted_pattern, "Greedy final rank pattern"
    )
    compared_identity_fields = _validate_checkpoint_identity(
        budgeted_best_config, greedy_best_config
    )
    if budgeted_root_ranks != budgeted_best_ranks:
        raise ValueError(
            "The budgeted root rank pattern does not describe trainer_state.best_model_checkpoint."
        )

    budgeted_best_active = _active_count(
        budgeted_best_ranks,
        non_dynamic,
        costs,
        "budgeted best checkpoint",
    )
    greedy_final_active = _active_count(
        greedy_final_ranks,
        non_dynamic,
        costs,
        "Greedy final rank pattern",
    )
    greedy_best_active = _active_count(
        greedy_best_ranks,
        non_dynamic,
        costs,
        "Greedy best checkpoint",
    )

    top_level_best = budgeted_pattern.get("active_model_parameter_count")
    if top_level_best is not None:
        top_level_best = _integer(
            top_level_best,
            "budgeted rank-pattern active_model_parameter_count",
            minimum=0,
        )
        if top_level_best != budgeted_best_active:
            raise ValueError(
                "Budgeted selected-best active count mismatch: serialized={} computed={}.".format(
                    top_level_best, budgeted_best_active
                )
            )

    budget = _required_mapping(budgeted_pattern, "budget", "budgeted rank pattern")
    final_trajectory = _required_mapping(budget, "final_trajectory", "budget metadata")
    selected_best = _required_mapping(
        budget, "selected_best_checkpoint", "budget metadata"
    )
    final_active = _stored_metric(
        final_trajectory,
        "active_model_parameter_count",
        "final trajectory",
    )
    final_rank = _stored_metric(final_trajectory, "total_active_rank", "final trajectory")
    final_rank_pattern = final_trajectory.get("rank_pattern")
    allocator_mode = budgeted_pattern.get(
        "allocator_mode", budget.get("allocator_mode")
    )
    if (
        allocator_mode == "genetic_budgeted_calibrated"
        and final_rank_pattern is None
    ):
        raise ValueError(
            "Calibrated final-trajectory reporting requires its saved rank_pattern."
        )
    if final_rank_pattern is not None:
        final_ranks = _rank_map(
            final_rank_pattern, "budgeted final-trajectory rank pattern"
        )
        computed_final_active = _active_count(
            final_ranks,
            non_dynamic,
            costs,
            "budgeted final-trajectory rank pattern",
        )
        if computed_final_active != final_active or sum(final_ranks.values()) != final_rank:
            raise ValueError(
                "Serialized final-trajectory metrics do not match its rank pattern: "
                "serialized_active={} computed_active={} serialized_rank={} computed_rank={}."
                .format(
                    final_active,
                    computed_final_active,
                    final_rank,
                    sum(final_ranks.values()),
                )
            )
    selected_best_active = _stored_metric(
        selected_best,
        "active_model_parameter_count",
        "selected best checkpoint",
    )
    selected_best_rank = _stored_metric(
        selected_best,
        "total_active_rank",
        "selected best checkpoint",
    )
    computed_best_rank = sum(budgeted_best_ranks.values())
    if selected_best_active != budgeted_best_active or selected_best_rank != computed_best_rank:
        raise ValueError(
            "Serialized selected-best metrics do not match its checkpoint: "
            "serialized_active={} computed_active={} serialized_rank={} computed_rank={}.".format(
                selected_best_active,
                budgeted_best_active,
                selected_best_rank,
                computed_best_rank,
            )
        )

    reference_cost = budget.get("reference_cost")
    if reference_cost is not None:
        reference_cost = _integer(reference_cost, "budget reference_cost", minimum=1)
        if reference_cost != greedy_final_active:
            raise ValueError(
                "Greedy final reference mismatch: budget reference_cost={} computed={}.".format(
                    reference_cost, greedy_final_active
                )
            )

    target_cost = budget.get("target_cost")
    if target_cost is not None:
        target_cost = _integer(target_cost, "budget target_cost", minimum=1)
        if final_active > target_cost:
            raise ValueError(
                "Final trajectory violates its serialized hard budget: active={} target={}.".format(
                    final_active, target_cost
                )
            )
        if budgeted_best_active > target_cost:
            raise ValueError(
                "Selected best checkpoint violates its serialized hard budget: active={} target={}.".format(
                    budgeted_best_active, target_cost
                )
            )

    final_absolute, final_percentage = _reduction(final_active, greedy_final_active)
    best_absolute, best_percentage = _reduction(budgeted_best_active, greedy_best_active)

    return {
        "format_version": REPORT_FORMAT_VERSION,
        "allocator_mode": allocator_mode,
        "accounting_basis": {
            "non_dynamic_trainable_params": non_dynamic,
            "module_count": len(costs),
            "matched_checkpoint_identity_fields": compared_identity_fields,
        },
        "budget": {
            "reference_active_parameter_count": reference_cost,
            "target_active_parameter_count": target_cost,
            "budget_ratio": budget.get("budget_ratio"),
            "final_trajectory_budget_satisfied": (
                final_active <= target_cost if target_cost is not None else None
            ),
            "selected_best_checkpoint_budget_satisfied": (
                budgeted_best_active <= target_cost if target_cost is not None else None
            ),
        },
        "final_trajectory_comparison": {
            "budgeted_final_trajectory": {
                "total_active_rank": final_rank,
                "active_model_parameter_count": final_active,
            },
            "greedy_final_rank_pattern": {
                "total_active_rank": sum(greedy_final_ranks.values()),
                "active_model_parameter_count": greedy_final_active,
            },
            "absolute_parameter_reduction": final_absolute,
            "percentage_parameter_reduction": final_percentage,
        },
        "best_checkpoint_comparison": {
            "budgeted_best_checkpoint": {
                "checkpoint": str(budgeted_best_checkpoint),
                "total_active_rank": computed_best_rank,
                "active_model_parameter_count": budgeted_best_active,
            },
            "greedy_best_checkpoint": {
                "checkpoint": str(greedy_best_checkpoint),
                "total_active_rank": sum(greedy_best_ranks.values()),
                "active_model_parameter_count": greedy_best_active,
            },
            "absolute_parameter_reduction": best_absolute,
            "percentage_parameter_reduction": best_percentage,
        },
        "sources": {
            "budgeted_trainer_state": str(trainer_state_path),
            "budgeted_rank_pattern": str(budgeted_pattern_path),
            "greedy_reference_checkpoint": str(greedy_best_checkpoint),
            "greedy_reference_rank_pattern": str(greedy_pattern_path),
        },
    }


def build_argument_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Report matched final-trajectory and best-checkpoint active-parameter "
            "comparisons without loading model weights."
        )
    )
    parser.add_argument("--budgeted_trainer_state", required=True)
    parser.add_argument("--budgeted_rank_pattern", required=True)
    parser.add_argument("--greedy_reference_checkpoint", required=True)
    parser.add_argument("--greedy_reference_rank_pattern", required=True)
    return parser


def main(argv=None):
    parser = build_argument_parser()
    arguments = parser.parse_args(argv)
    try:
        report = build_rank_budget_report(
            budgeted_trainer_state=arguments.budgeted_trainer_state,
            budgeted_rank_pattern=arguments.budgeted_rank_pattern,
            greedy_reference_checkpoint=arguments.greedy_reference_checkpoint,
            greedy_reference_rank_pattern=arguments.greedy_reference_rank_pattern,
        )
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
