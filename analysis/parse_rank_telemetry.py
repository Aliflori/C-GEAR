#!/usr/bin/env python3
"""Validate rank-allocation telemetry JSONL and build analysis-ready CSVs.

This is an offline postprocessor.  It never imports the training stack, loads a
model, or modifies an experiment directory.
"""

import argparse
import csv
import json
import math
import re
import tempfile
import warnings
from pathlib import Path


SCHEMA_VERSION = "rank_telemetry.v1"
NONFINITE_SENTINELS = {"NaN", "Infinity", "-Infinity"}
EVENT_TYPES = {
    "run_start",
    "allocation_event",
    "calibration_event",
    "candidate_selection",
    "evaluation",
    "checkpoint_save",
    "allocator_stop",
    "run_end",
    "warning",
}
COMMON_FIELDS = (
    "schema_version",
    "event_type",
    "global_step",
    "seed",
    "method",
    "wall_time_seconds",
)

RANK_COLUMNS = (
    "source_artifact",
    "schema_version",
    "method",
    "seed",
    "run_segment",
    "global_step",
    "wall_time_seconds",
    "event_type",
    "state_role",
    "total_active_rank",
    "active_model_parameter_count",
    "runtime_trainable_parameter_count",
    "full_model_parameter_count",
    "physical_rank_component_count",
    "target_active_parameter_count",
    "selected_event_rank",
    "allocation_stopped",
    "stop_reason",
)
MODULE_COLUMNS = (
    "source_artifact",
    "schema_version",
    "method",
    "seed",
    "run_segment",
    "global_step",
    "wall_time_seconds",
    "event_type",
    "state_role",
    "module_name",
    "transformer_layer",
    "module_family",
    "module_group",
    "active_rank",
)
ALLOCATION_COLUMNS = (
    "source_artifact",
    "schema_version",
    "method",
    "seed",
    "run_segment",
    "global_step",
    "wall_time_seconds",
    "selected_candidate_id",
    "selected_k",
    "selected_event_rank",
    "selected_source",
    "selected_modules",
    "selected_module_count",
    "rank_increments",
    "pre_total_active_rank",
    "post_total_active_rank",
    "total_active_rank",
    "pre_active_parameter_count",
    "post_active_parameter_count",
    "active_model_parameter_count",
    "runtime_trainable_parameter_count",
    "target_active_parameter_count",
    "budget_used",
    "remaining_hard_budget",
    "zero_rank_patience",
    "zero_growth",
    "allocation_stopped",
    "stop_reason",
)
CALIBRATION_COLUMNS = (
    "source_artifact",
    "schema_version",
    "method",
    "seed",
    "run_segment",
    "global_step",
    "wall_time_seconds",
    "candidate_index",
    "candidate_id",
    "candidate_modules",
    "candidate_size",
    "candidate_cost",
    "candidate_family",
    "is_selected",
    "calibration_gain_mean",
    "calibration_gain_std",
    "calibration_gain_lcb",
    "calibration_gain_per_parameter",
    "calibration_valid",
    "invalid_reason",
    "fold_gains",
)
EVALUATION_COLUMNS = (
    "source_artifact",
    "schema_version",
    "method",
    "seed",
    "run_segment",
    "global_step",
    "wall_time_seconds",
    "state_role",
    "split",
    "accuracy",
    "loss",
    "total_active_rank",
    "active_model_parameter_count",
    "runtime_trainable_parameter_count",
    "full_model_parameter_count",
    "checkpoint",
    "metrics",
)

OUTPUT_SCHEMAS = {
    "rank_trajectory.csv": RANK_COLUMNS,
    "module_rank_trajectory.csv": MODULE_COLUMNS,
    "allocation_events.csv": ALLOCATION_COLUMNS,
    "calibration_events.csv": CALIBRATION_COLUMNS,
    "evaluation_trajectory.csv": EVALUATION_COLUMNS,
}


class TelemetryValidationError(ValueError):
    """Raised when a JSONL record violates the telemetry contract."""


def _context(path, line_number, message):
    raise TelemetryValidationError("%s:%s: %s" % (path, line_number, message))


def _is_integer(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _is_nonfinite_sentinel(value):
    return isinstance(value, str) and value in NONFINITE_SENTINELS


def _integer_value(value, key, path, line_number, minimum=0, required=False):
    if value is None and not required:
        return None
    if not _is_integer(value) or value < minimum:
        _context(path, line_number, "%s must be an integer >= %s" % (key, minimum))
    return value


def _integer(record, key, path, line_number, minimum=0, required=False):
    return _integer_value(
        record.get(key), key, path, line_number, minimum=minimum, required=required
    )


def _number(record, key, path, line_number, minimum=None, required=False):
    value = record.get(key)
    if value is None and not required:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _context(path, line_number, "%s must be numeric" % key)
    value = float(value)
    if not math.isfinite(value) or (minimum is not None and value < minimum):
        _context(path, line_number, "%s must be finite%s" % (
            key,
            " and >= %s" % minimum if minimum is not None else "",
        ))
    return value


def _boolean(value, path, line_number, field):
    if value is None:
        return None
    if not isinstance(value, bool):
        _context(path, line_number, "%s must be boolean" % field)
    return value


def _string(value, path, line_number, field, required=False):
    if value is None and not required:
        return None
    if not isinstance(value, str) or not value.strip():
        _context(path, line_number, "%s must be a nonempty string" % field)
    return value.strip()


def _string_list(value, path, line_number, field):
    if value is None:
        return []
    if not isinstance(value, list):
        _context(path, line_number, "%s must be an array" % field)
    result = []
    for item in value:
        result.append(_string(item, path, line_number, field, required=True))
    if len(result) != len(set(result)):
        _context(path, line_number, "%s must not contain duplicates" % field)
    return result


def _rank_map(value, path, line_number, required=False):
    if value is None and not required:
        return None
    if not isinstance(value, dict) or not value:
        _context(path, line_number, "module_active_ranks must be a nonempty object")
    result = {}
    for name, rank in value.items():
        module = _string(name, path, line_number, "module name", required=True)
        if not _is_integer(rank) or rank < 0:
            _context(path, line_number, "active rank for %s must be an integer >= 0" % module)
        result[module] = rank
    return result


def _json_cell(value):
    if value is None:
        return ""
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _module_coordinates(name):
    matches = re.findall(r"(?:^|\.)(?:layer|layers)\.(\d+)(?:\.|$)", name)
    layer = int(matches[-1]) if matches else ""
    lowered = name.lower()
    if "attention" in lowered and ("query" in lowered or "q_proj" in lowered):
        family = "attention_query"
    elif "attention" in lowered and ("key" in lowered or "k_proj" in lowered):
        family = "attention_key"
    elif "attention" in lowered and ("value" in lowered or "v_proj" in lowered):
        family = "attention_value"
    elif "attention" in lowered and ("output" in lowered or "o_proj" in lowered):
        family = "attention_output"
    elif "intermediate" in lowered or "up_proj" in lowered or "gate_proj" in lowered:
        family = "ffn_intermediate"
    elif "output" in lowered or "down_proj" in lowered:
        family = "ffn_output"
    else:
        family = "other"
    group = "attention" if family.startswith("attention_") else (
        "ffn" if family.startswith("ffn_") else "other"
    )
    return layer, family, group


def _common_record(raw, path, line_number):
    missing = [field for field in COMMON_FIELDS if field not in raw]
    if missing:
        _context(path, line_number, "missing common fields: %s" % ", ".join(missing))
    version = raw["schema_version"]
    if not isinstance(version, str):
        _context(path, line_number, "schema_version must be a string")
    if version != SCHEMA_VERSION:
        _context(
            path,
            line_number,
            "unsupported schema_version %s (expected %s)" % (version, SCHEMA_VERSION),
        )
    event_type = _string(raw["event_type"], path, line_number, "event_type", required=True)
    if event_type not in EVENT_TYPES:
        _context(path, line_number, "unknown event_type %r" % event_type)
    global_step = _integer(raw, "global_step", path, line_number, minimum=0, required=True)
    seed = _integer(raw, "seed", path, line_number, minimum=0, required=True)
    method = _string(raw["method"], path, line_number, "method", required=True)
    wall_time = _number(
        raw, "wall_time_seconds", path, line_number, minimum=0.0, required=True
    )
    normalized = dict(raw)
    normalized.update(
        schema_version=version,
        event_type=event_type,
        global_step=global_step,
        seed=seed,
        method=method,
        wall_time_seconds=wall_time,
        _line_number=line_number,
        _source=str(path.resolve()),
    )
    return normalized


def _reject_json_constant(value):
    raise ValueError("non-finite JSON constant %s is not permitted" % value)


def load_jsonl(path):
    """Read one JSONL stream and enforce its common ordering invariants."""

    path = Path(path)
    records = []
    identity = None
    previous_step = -1
    previous_wall_time = -1.0
    run_segment = 0
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise TelemetryValidationError("Cannot read %s: %s" % (path, error)) from error
    lines = payload.splitlines(keepends=True)
    trailing_partial = bool(lines and not lines[-1].endswith(b"\n"))
    for line_number, encoded_line in enumerate(lines, 1):
        try:
            line = encoded_line.decode("utf-8")
        except UnicodeDecodeError as error:
            if trailing_partial and line_number == len(lines):
                warnings.warn(
                    "%s:%s: ignored one incomplete trailing crash record"
                    % (path, line_number),
                    RuntimeWarning,
                )
                break
            _context(path, line_number, "invalid UTF-8: %s" % error)
        if not line.strip():
            _context(path, line_number, "blank lines are not valid JSONL records")
        try:
            raw = json.loads(line, parse_constant=_reject_json_constant)
        except json.JSONDecodeError as error:
            if trailing_partial and line_number == len(lines):
                warnings.warn(
                    "%s:%s: ignored one incomplete trailing crash record"
                    % (path, line_number),
                    RuntimeWarning,
                )
                break
            _context(path, line_number, "invalid JSON: %s" % error.msg)
        except ValueError as error:
            _context(path, line_number, "invalid JSON: %s" % error)
        if not isinstance(raw, dict):
            _context(path, line_number, "each JSONL record must be an object")
        record = _common_record(raw, path, line_number)
        current_identity = (record["method"], record["seed"])
        if identity is None:
            identity = current_identity
        elif current_identity != identity:
            _context(path, line_number, "method and seed must remain constant within a file")
        if record["event_type"] == "run_start":
            if records:
                run_segment += 1
        else:
            if record["global_step"] < previous_step:
                _context(
                    path,
                    line_number,
                    "global_step may reset only at an appended run_start",
                )
            if record["wall_time_seconds"] < previous_wall_time:
                _context(
                    path,
                    line_number,
                    "wall_time_seconds may reset only at an appended run_start",
                )
        record["_run_segment"] = run_segment
        previous_step = record["global_step"]
        previous_wall_time = record["wall_time_seconds"]
        records.append(record)
    if not records:
        raise TelemetryValidationError("%s: telemetry file is empty" % path)
    return records


def _optional_counts(record, path, line_number):
    active = record.get(
        "active_model_parameter_count", record.get("post_active_parameter_count")
    )
    target = record.get("target_active_parameter_count", record.get("budget_limit"))
    return {
        "active_model_parameter_count": _integer_value(
            active, "active_model_parameter_count", path, line_number, minimum=0
        ),
        "runtime_trainable_parameter_count": _integer(
            record, "runtime_trainable_parameter_count", path, line_number, minimum=0
        ),
        "full_model_parameter_count": _integer(
            record, "full_model_parameter_count", path, line_number, minimum=0
        ),
        "target_active_parameter_count": _integer_value(
            target, "target_active_parameter_count", path, line_number, minimum=0
        ),
    }


def _candidate_payloads(record, path, line_number):
    if "candidates" in record:
        candidates = record["candidates"]
        if not isinstance(candidates, list) or not candidates:
            _context(path, line_number, "calibration_event.candidates must be a nonempty array")
        if not all(isinstance(candidate, dict) for candidate in candidates):
            _context(path, line_number, "every calibration candidate must be an object")
        return candidates
    if "candidate" in record:
        if not isinstance(record["candidate"], dict):
            _context(path, line_number, "calibration_event.candidate must be an object")
        return [record["candidate"]]
    candidate_keys = {
        "candidate_id",
        "candidate_modules",
        "modules",
        "calibration_gain_mean",
        "calibration_gain_lcb",
        "fold_gains",
    }
    if candidate_keys.intersection(record):
        return [record]
    _context(
        path,
        line_number,
        "calibration_event must contain candidates, candidate, or candidate-level fields",
    )


def _selection_identity(record, path, line_number):
    candidate = record.get("selected_candidate")
    if candidate is not None and not isinstance(candidate, dict):
        _context(path, line_number, "selected_candidate must be an object")
    candidate = candidate or {}
    candidate_id = record.get("selected_candidate_id", candidate.get("candidate_id"))
    if candidate_id is not None:
        candidate_id = _string(candidate_id, path, line_number, "selected_candidate_id", True)
    modules_value = record.get(
        "selected_modules",
        candidate.get("candidate_modules", candidate.get("modules")),
    )
    modules = tuple(sorted(_string_list(modules_value, path, line_number, "selected_modules")))
    if candidate_id is None and not modules:
        _context(
            path,
            line_number,
            "candidate_selection requires selected_candidate_id or selected_modules",
        )
    return candidate_id, modules


def _selected_event_sizes(record, path, line_number):
    selected_event_rank = _integer(
        record, "selected_event_rank", path, line_number, minimum=0
    )
    selected_k = _integer(record, "selected_k", path, line_number, minimum=0)
    return selected_k, selected_event_rank


def _candidate_row(record, candidate, index, selected_ids, selected_modules):
    path = Path(record["_source"])
    line_number = record["_line_number"]
    modules_value = candidate.get("candidate_modules", candidate.get("modules"))
    modules = _string_list(modules_value, path, line_number, "candidate_modules")
    candidate_id = candidate.get("candidate_id")
    if candidate_id is not None:
        candidate_id = _string(candidate_id, path, line_number, "candidate_id", True)
    else:
        candidate_id = "modules:" + _json_cell(sorted(modules))
    size_value = candidate.get(
        "candidate_size", candidate.get("chromosome_size", candidate.get("k"))
    )
    if size_value is None:
        size = len(modules)
    else:
        if not _is_integer(size_value) or size_value < 0:
            _context(path, line_number, "candidate_size must be an integer >= 0")
        size = size_value
    if modules and size != len(modules):
        _context(path, line_number, "candidate_size does not match candidate_modules")
    cost = candidate.get(
        "candidate_cost", candidate.get("actual_cost", candidate.get("parameter_cost"))
    )
    if cost is not None and (not _is_integer(cost) or cost < 0):
        _context(path, line_number, "candidate_cost must be an integer >= 0")
    family = candidate.get(
        "candidate_family",
        candidate.get(
            "candidate_families",
            candidate.get("candidate_source", candidate.get("candidate_sources")),
        ),
    )
    if family is not None and not isinstance(family, (str, list)):
        _context(path, line_number, "candidate_family must be a string or array")
    numeric = {}
    unavailable_numeric = False
    score_aliases = {
        "calibration_gain_mean": "mean_score",
        "calibration_gain_std": "std_score",
        "calibration_gain_lcb": "lcb_score",
        "calibration_gain_per_parameter": "gain_per_parameter",
    }
    for field, alias in score_aliases.items():
        value = candidate.get(field, candidate.get(alias))
        if _is_nonfinite_sentinel(value):
            value = None
            unavailable_numeric = True
        if value is not None:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                _context(path, line_number, "%s must be finite numeric" % field)
            value = float(value)
            if field == "calibration_gain_std" and value < 0.0:
                _context(path, line_number, "calibration_gain_std must be nonnegative")
        numeric[field] = value
    folds = candidate.get(
        "fold_gains", candidate.get("calibration_gains", candidate.get("fold_scores"))
    )
    if folds is not None:
        if not isinstance(folds, list) or any(
            not _is_nonfinite_sentinel(value)
            and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            )
            for value in folds
        ):
            _context(
                path,
                line_number,
                "fold_gains must contain finite numbers or canonical non-finite sentinels",
            )
        unavailable_numeric = unavailable_numeric or any(
            _is_nonfinite_sentinel(value) for value in folds
        )
        folds = [
            value if _is_nonfinite_sentinel(value) else float(value) for value in folds
        ]
    valid_value = candidate.get("calibration_valid", candidate.get("calibration_signal_valid"))
    valid = _boolean(valid_value, path, line_number, "calibration_valid")
    invalid_reason = candidate.get(
        "invalid_reason", candidate.get("calibration_invalid_reason")
    )
    if invalid_reason is not None:
        invalid_reason = _string(
            invalid_reason, path, line_number, "invalid_reason", required=True
        )
    if unavailable_numeric and valid is not False:
        _context(
            path,
            line_number,
            "non-finite calibration scores require calibration_valid=false",
        )
    explicit_selected = _boolean(candidate.get("is_selected"), path, line_number, "is_selected")
    selected = bool(
        explicit_selected
        or candidate_id in selected_ids
        or (modules and tuple(sorted(modules)) in selected_modules)
    )
    return {
        "source_artifact": record["_source"],
        "schema_version": record["schema_version"],
        "method": record["method"],
        "seed": record["seed"],
        "run_segment": record["_run_segment"],
        "global_step": record["global_step"],
        "wall_time_seconds": record["wall_time_seconds"],
        "candidate_index": index,
        "candidate_id": candidate_id,
        "candidate_modules": _json_cell(modules),
        "candidate_size": size,
        "candidate_cost": "" if cost is None else cost,
        "candidate_family": _json_cell(family) if isinstance(family, list) else (family or ""),
        "is_selected": str(selected).lower(),
        "calibration_gain_mean": "" if numeric["calibration_gain_mean"] is None else numeric["calibration_gain_mean"],
        "calibration_gain_std": "" if numeric["calibration_gain_std"] is None else numeric["calibration_gain_std"],
        "calibration_gain_lcb": "" if numeric["calibration_gain_lcb"] is None else numeric["calibration_gain_lcb"],
        "calibration_gain_per_parameter": "" if numeric["calibration_gain_per_parameter"] is None else numeric["calibration_gain_per_parameter"],
        "calibration_valid": "" if valid is None else str(valid).lower(),
        "invalid_reason": invalid_reason or "",
        "fold_gains": _json_cell(folds),
    }


def _aliased_terminal_integer(record, selected_field, snapshot_field, path, line_number):
    selected = _integer(record, selected_field, path, line_number, minimum=0)
    snapshot = _integer(record, snapshot_field, path, line_number, minimum=0)
    if selected is not None and snapshot is not None and selected != snapshot:
        _context(
            path,
            line_number,
            "%s does not match terminal snapshot field %s"
            % (selected_field, snapshot_field),
        )
    return selected if selected is not None else snapshot


def _selected_terminal_state(record, path, line_number):
    selected_fields = (
        "selected_active_rank",
        "selected_active_parameter_count",
        "selected_runtime_trainable_parameter_count",
        "selected_full_model_parameter_count",
        "selected_physical_rank_component_count",
        "selected_module_active_ranks",
    )
    if not any(record.get(field) is not None for field in selected_fields):
        return None

    selected_ranks = _rank_map(
        record.get("selected_module_active_ranks"), path, line_number
    )
    snapshot_ranks = _rank_map(record.get("module_active_ranks"), path, line_number)
    if (
        selected_ranks is not None
        and snapshot_ranks is not None
        and selected_ranks != snapshot_ranks
    ):
        _context(
            path,
            line_number,
            "selected_module_active_ranks do not match the terminal snapshot",
        )
    ranks = selected_ranks if selected_ranks is not None else snapshot_ranks
    total = _aliased_terminal_integer(
        record, "selected_active_rank", "total_active_rank", path, line_number
    )
    if ranks is not None:
        computed_total = sum(ranks.values())
        if total is not None and total != computed_total:
            _context(
                path,
                line_number,
                "selected_active_rank does not equal selected_module_active_ranks sum",
            )
        total = computed_total

    return {
        "ranks": ranks,
        "total_active_rank": total,
        "active_model_parameter_count": _aliased_terminal_integer(
            record,
            "selected_active_parameter_count",
            "active_model_parameter_count",
            path,
            line_number,
        ),
        "runtime_trainable_parameter_count": _aliased_terminal_integer(
            record,
            "selected_runtime_trainable_parameter_count",
            "runtime_trainable_parameter_count",
            path,
            line_number,
        ),
        "full_model_parameter_count": _aliased_terminal_integer(
            record,
            "selected_full_model_parameter_count",
            "full_model_parameter_count",
            path,
            line_number,
        ),
        "physical_rank_component_count": _aliased_terminal_integer(
            record,
            "selected_physical_rank_component_count",
            "physical_rank_component_count",
            path,
            line_number,
        ),
    }


def transform_records(records):
    """Transform validated records from one source into the five table schemas."""

    rank_rows = []
    module_rows = []
    allocation_rows = []
    calibration_rows = []
    evaluation_rows = []
    last_ranks = None
    last_counts = {
        "active_model_parameter_count": None,
        "runtime_trainable_parameter_count": None,
        "full_model_parameter_count": None,
        "target_active_parameter_count": None,
    }
    last_total_rank = None
    last_physical_capacity = None

    selections = {}
    for record in records:
        has_selection = any(
            field in record
            for field in (
                "selected_candidate_id",
                "selected_modules",
                "selected_candidate",
            )
        )
        if record["event_type"] != "candidate_selection" and not (
            record["event_type"] == "calibration_event" and has_selection
        ):
            continue
        path = Path(record["_source"])
        candidate_id, modules = _selection_identity(record, path, record["_line_number"])
        selection_key = (record["_run_segment"], record["global_step"])
        entry = selections.setdefault(selection_key, {"ids": set(), "modules": set()})
        if candidate_id is not None:
            entry["ids"].add(candidate_id)
        if modules:
            entry["modules"].add(modules)

    for record in records:
        path = Path(record["_source"])
        line_number = record["_line_number"]
        event_type = record["event_type"]
        if event_type == "run_start" and record["_run_segment"] > 0:
            last_ranks = None
            last_counts = {
                "active_model_parameter_count": None,
                "runtime_trainable_parameter_count": None,
                "full_model_parameter_count": None,
                "target_active_parameter_count": None,
            }
            last_total_rank = None
            last_physical_capacity = None
        snapshot_counts = _optional_counts(record, path, line_number)
        counts = dict(snapshot_counts)
        state_role = "trajectory"
        selected_terminal = None
        has_final_state = False
        if event_type == "run_start":
            state_role = "initial_trajectory"
        if event_type == "run_end":
            selected_terminal = _selected_terminal_state(record, path, line_number)
            has_final_state = any(
                record.get(field) is not None
                for field in (
                    "final_active_rank",
                    "final_active_parameter_count",
                    "final_runtime_trainable_parameter_count",
                    "final_full_model_parameter_count",
                    "final_physical_rank_component_count",
                    "final_module_active_ranks",
                )
            )
            state_role = (
                "final_trajectory"
                if has_final_state or selected_terminal is None
                else "selected_best_checkpoint"
            )
        if has_final_state:
            counts = {
                "active_model_parameter_count": None,
                "runtime_trainable_parameter_count": None,
                "full_model_parameter_count": None,
                "target_active_parameter_count": snapshot_counts[
                    "target_active_parameter_count"
                ],
            }
            for field in (
                "active_model_parameter_count",
                "runtime_trainable_parameter_count",
                "full_model_parameter_count",
            ):
                last_counts[field] = None
            for target_field, source_field in (
                ("active_model_parameter_count", "final_active_parameter_count"),
                (
                    "runtime_trainable_parameter_count",
                    "final_runtime_trainable_parameter_count",
                ),
                ("full_model_parameter_count", "final_full_model_parameter_count"),
            ):
                value = record.get(source_field)
                if value is not None:
                    counts[target_field] = _integer_value(
                        value, source_field, path, line_number, minimum=0
                    )
        for key, value in counts.items():
            if value is not None:
                last_counts[key] = value
        physical_capacity = (
            _integer(
                record,
                "final_physical_rank_component_count",
                path,
                line_number,
                minimum=0,
            )
            if has_final_state
            else _integer(
                record, "physical_rank_component_count", path, line_number, minimum=0
            )
        )
        if has_final_state:
            # An omitted final-trajectory capacity is unknown; never inherit
            # the currently loaded selected checkpoint's physical capacity.
            last_physical_capacity = physical_capacity
        elif physical_capacity is not None:
            last_physical_capacity = physical_capacity

        if event_type == "calibration_event":
            selected = selections.get(
                (record["_run_segment"], record["global_step"]),
                {"ids": set(), "modules": set()},
            )
            for index, candidate in enumerate(_candidate_payloads(record, path, line_number)):
                calibration_rows.append(
                    _candidate_row(
                        record,
                        candidate,
                        index,
                        selected["ids"],
                        selected["modules"],
                    )
                )
            continue

        rank_payload = (
            record.get("final_module_active_ranks")
            if has_final_state
            else record.get("module_active_ranks")
        )
        ranks = _rank_map(
            rank_payload,
            path,
            line_number,
            required=event_type == "allocation_event",
        )
        supplied_total = _integer_value(
            (
                record.get("final_active_rank")
                if has_final_state
                else record.get("total_active_rank")
            ),
            "final_active_rank" if has_final_state else "total_active_rank",
            path,
            line_number,
            minimum=0,
        )
        if ranks is not None:
            computed_total = sum(ranks.values())
            if supplied_total is not None and supplied_total != computed_total:
                _context(path, line_number, "total_active_rank does not equal module_active_ranks sum")
        else:
            computed_total = supplied_total

        if event_type == "allocation_event":
            selected_modules = _string_list(
                record.get("selected_modules"), path, line_number, "selected_modules"
            )
            selected_k, selected_rank = _selected_event_sizes(
                record, path, line_number
            )
            if selected_k is None and selected_modules:
                selected_k = len(selected_modules)
            if selected_k is not None and selected_modules and selected_k != len(selected_modules):
                _context(path, line_number, "selected_k does not match selected_modules")
            pre_total = _integer(record, "pre_total_active_rank", path, line_number, minimum=0)
            post_total = _integer(record, "post_total_active_rank", path, line_number, minimum=0)
            if post_total is not None and post_total != computed_total:
                _context(path, line_number, "post_total_active_rank does not match module_active_ranks")
            if last_ranks is not None:
                if set(ranks) != set(last_ranks):
                    _context(path, line_number, "module_active_ranks module set changed within a run")
                differences = {name: ranks[name] - last_ranks[name] for name in ranks}
                if any(value < 0 for value in differences.values()):
                    _context(path, line_number, "active module ranks must not decrease")
                rank_delta = sum(differences.values())
                changed = {name for name, value in differences.items() if value > 0}
                if selected_rank is None:
                    selected_rank = rank_delta
                elif selected_rank != rank_delta:
                    _context(path, line_number, "selected_event_rank does not match rank-map delta")
                if selected_modules and changed != set(selected_modules):
                    _context(path, line_number, "selected_modules do not match changed module ranks")
            elif selected_rank is None:
                selected_rank = len(selected_modules) if selected_modules else None
            if pre_total is not None and post_total is not None:
                if post_total < pre_total:
                    _context(path, line_number, "post_total_active_rank must not be below pre_total_active_rank")
                if selected_rank is not None and post_total - pre_total != selected_rank:
                    _context(path, line_number, "pre/post total-rank delta does not match selected_event_rank")
            if selected_rank is not None and selected_modules and selected_rank < len(selected_modules):
                _context(path, line_number, "selected_event_rank is smaller than selected module count")
            allocation_stopped = _boolean(
                record.get("allocation_stopped"), path, line_number, "allocation_stopped"
            )
            stop_reason = _string(record.get("stop_reason"), path, line_number, "stop_reason")
            target = counts["target_active_parameter_count"]
            active = counts["active_model_parameter_count"]
            remaining = _integer_value(
                record.get("remaining_hard_budget", record.get("budget_remaining")),
                "remaining_hard_budget",
                path,
                line_number,
                minimum=0,
            )
            budget_used = _integer(record, "budget_used", path, line_number, minimum=0)
            pre_active = _integer(record, "pre_active_parameter_count", path, line_number, minimum=0)
            post_active = _integer(record, "post_active_parameter_count", path, line_number, minimum=0)
            if post_active is not None and active is not None and post_active != active:
                _context(path, line_number, "post_active_parameter_count does not match active_model_parameter_count")
            if pre_active is not None and post_active is not None and post_active < pre_active:
                _context(path, line_number, "post_active_parameter_count must not be below pre_active_parameter_count")
            if budget_used is not None and active is not None and budget_used != active:
                _context(path, line_number, "budget_used does not match active_model_parameter_count")
            if remaining is None and target is not None and active is not None:
                if active > target:
                    _context(path, line_number, "active model parameters exceed target budget")
                remaining = target - active
            if remaining is not None and target is not None and budget_used is not None:
                if target - budget_used != remaining:
                    _context(path, line_number, "budget_limit - budget_used does not equal budget_remaining")
            rank_increments = record.get("rank_increments")
            if rank_increments is not None:
                if not isinstance(rank_increments, dict):
                    _context(path, line_number, "rank_increments must be an object")
                parsed_increments = {}
                for name, increment in rank_increments.items():
                    module = _string(name, path, line_number, "rank_increments module", True)
                    if not _is_integer(increment) or increment < 0:
                        _context(path, line_number, "rank increments must be nonnegative integers")
                    parsed_increments[module] = increment
                positive_increment_modules = {
                    name for name, increment in parsed_increments.items() if increment > 0
                }
                if selected_modules and positive_increment_modules != set(selected_modules):
                    _context(path, line_number, "rank_increments do not match selected_modules")
                if selected_rank is not None and sum(parsed_increments.values()) != selected_rank:
                    _context(path, line_number, "rank_increments do not sum to selected_event_rank")
            allocation_rows.append(
                {
                    "source_artifact": record["_source"],
                    "schema_version": record["schema_version"],
                    "method": record["method"],
                    "seed": record["seed"],
                    "run_segment": record["_run_segment"],
                    "global_step": record["global_step"],
                    "wall_time_seconds": record["wall_time_seconds"],
                    "selected_candidate_id": record.get("selected_candidate_id", ""),
                    "selected_k": "" if selected_k is None else selected_k,
                    "selected_event_rank": "" if selected_rank is None else selected_rank,
                    "selected_source": record.get("selected_source", ""),
                    "selected_modules": _json_cell(selected_modules),
                    "selected_module_count": len(selected_modules),
                    "rank_increments": _json_cell(rank_increments),
                    "pre_total_active_rank": "" if pre_total is None else pre_total,
                    "post_total_active_rank": "" if post_total is None else post_total,
                    "total_active_rank": computed_total,
                    "pre_active_parameter_count": "" if pre_active is None else pre_active,
                    "post_active_parameter_count": "" if post_active is None else post_active,
                    "active_model_parameter_count": "" if active is None else active,
                    "runtime_trainable_parameter_count": "" if counts["runtime_trainable_parameter_count"] is None else counts["runtime_trainable_parameter_count"],
                    "target_active_parameter_count": "" if target is None else target,
                    "budget_used": "" if budget_used is None else budget_used,
                    "remaining_hard_budget": "" if remaining is None else remaining,
                    "zero_rank_patience": "" if record.get("zero_rank_patience") is None else _integer(record, "zero_rank_patience", path, line_number, minimum=0),
                    "zero_growth": "" if selected_rank is None else str(selected_rank == 0).lower(),
                    "allocation_stopped": "" if allocation_stopped is None else str(allocation_stopped).lower(),
                    "stop_reason": stop_reason or "",
                }
            )

        if event_type == "evaluation":
            evaluation_role = record.get(
                "state_role", "training_trajectory_evaluation"
            )
            if evaluation_role not in (
                "training_trajectory_evaluation",
                "selected_best_checkpoint_evaluation",
                "standalone_evaluation",
            ):
                _context(
                    path,
                    line_number,
                    "evaluation.state_role must identify training trajectory, selected best checkpoint, or standalone evaluation",
                )
            metrics = record.get("metrics", {})
            if not isinstance(metrics, dict):
                _context(path, line_number, "evaluation.metrics must be an object")
            for name, value in metrics.items():
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                    _context(path, line_number, "evaluation metric %s must be finite numeric" % name)
            accuracy_values = [
                value
                for value in (
                    record.get("accuracy"),
                    metrics.get("accuracy"),
                    metrics.get("eval_accuracy"),
                )
                if value is not None
            ]
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
                for value in accuracy_values
            ):
                _context(path, line_number, "accuracy must be finite and within [0, 1]")
            accuracy = float(accuracy_values[0]) if accuracy_values else ""
            if accuracy_values and any(
                not math.isclose(accuracy, float(value), abs_tol=1e-12)
                for value in accuracy_values[1:]
            ):
                _context(path, line_number, "conflicting accuracy values in evaluation")
            loss_values = [
                value
                for value in (record.get("loss"), metrics.get("loss"), metrics.get("eval_loss"))
                if value is not None
            ]
            loss = ""
            if loss_values:
                if any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    for value in loss_values
                ):
                    _context(path, line_number, "evaluation loss must be finite numeric")
                loss = float(loss_values[0])
            evaluation_rows.append(
                {
                    "source_artifact": record["_source"],
                    "schema_version": record["schema_version"],
                    "method": record["method"],
                    "seed": record["seed"],
                    "run_segment": record["_run_segment"],
                    "global_step": record["global_step"],
                    "wall_time_seconds": record["wall_time_seconds"],
                    "state_role": evaluation_role,
                    "split": record.get("split", "validation"),
                    "accuracy": accuracy,
                    "loss": loss,
                    "total_active_rank": computed_total if computed_total is not None else (last_total_rank if last_total_rank is not None else ""),
                    "active_model_parameter_count": counts["active_model_parameter_count"] if counts["active_model_parameter_count"] is not None else (last_counts["active_model_parameter_count"] if last_counts["active_model_parameter_count"] is not None else ""),
                    "runtime_trainable_parameter_count": counts["runtime_trainable_parameter_count"] if counts["runtime_trainable_parameter_count"] is not None else (last_counts["runtime_trainable_parameter_count"] if last_counts["runtime_trainable_parameter_count"] is not None else ""),
                    "full_model_parameter_count": counts["full_model_parameter_count"] if counts["full_model_parameter_count"] is not None else (last_counts["full_model_parameter_count"] if last_counts["full_model_parameter_count"] is not None else ""),
                    "checkpoint": record.get(
                        "evaluated_checkpoint",
                        record.get("checkpoint", record.get("checkpoint_path", "")),
                    ),
                    "metrics": _json_cell(metrics),
                }
            )

        if ranks is not None:
            last_ranks = dict(ranks)
            last_total_rank = computed_total
        elif supplied_total is not None:
            last_total_rank = supplied_total

        rank_event = event_type in {
            "run_start",
            "allocation_event",
            "allocator_stop",
            "run_end",
        }
        if rank_event and (ranks is not None or last_ranks is not None or last_total_rank is not None):
            if event_type == "run_end" and has_final_state:
                # A separately labeled terminal trajectory must stand on its
                # explicit evidence; never borrow the loaded best checkpoint's
                # rank map or total when a final field is missing.
                effective_ranks = ranks
                effective_total = computed_total
            else:
                effective_ranks = ranks if ranks is not None else last_ranks
                effective_total = (
                    computed_total
                    if computed_total is not None
                    else last_total_rank
                )
            stop_reason = _string(record.get("stop_reason"), path, line_number, "stop_reason")
            stopped = _boolean(record.get("allocation_stopped"), path, line_number, "allocation_stopped")
            if event_type == "allocator_stop":
                stopped = True
            selected_rank = _integer(
                record, "selected_event_rank", path, line_number, minimum=0
            )
            rank_rows.append(
                {
                    "source_artifact": record["_source"],
                    "schema_version": record["schema_version"],
                    "method": record["method"],
                    "seed": record["seed"],
                    "run_segment": record["_run_segment"],
                    "global_step": record["global_step"],
                    "wall_time_seconds": record["wall_time_seconds"],
                    "event_type": event_type,
                    "state_role": state_role,
                    "total_active_rank": "" if effective_total is None else effective_total,
                    "active_model_parameter_count": "" if last_counts["active_model_parameter_count"] is None else last_counts["active_model_parameter_count"],
                    "runtime_trainable_parameter_count": "" if last_counts["runtime_trainable_parameter_count"] is None else last_counts["runtime_trainable_parameter_count"],
                    "full_model_parameter_count": "" if last_counts["full_model_parameter_count"] is None else last_counts["full_model_parameter_count"],
                    "physical_rank_component_count": "" if last_physical_capacity is None else last_physical_capacity,
                    "target_active_parameter_count": "" if last_counts["target_active_parameter_count"] is None else last_counts["target_active_parameter_count"],
                    "selected_event_rank": "" if selected_rank is None else selected_rank,
                    "allocation_stopped": "" if stopped is None else str(stopped).lower(),
                    "stop_reason": stop_reason or "",
                }
            )
            if effective_ranks is not None:
                for name in sorted(effective_ranks):
                    layer, family, group = _module_coordinates(name)
                    module_rows.append(
                        {
                            "source_artifact": record["_source"],
                            "schema_version": record["schema_version"],
                            "method": record["method"],
                            "seed": record["seed"],
                            "run_segment": record["_run_segment"],
                            "global_step": record["global_step"],
                            "wall_time_seconds": record["wall_time_seconds"],
                            "event_type": event_type,
                            "state_role": state_role,
                            "module_name": name,
                            "transformer_layer": layer,
                            "module_family": family,
                            "module_group": group,
                            "active_rank": effective_ranks[name],
                        }
                    )

            if (
                event_type == "run_end"
                and has_final_state
                and selected_terminal is not None
            ):
                selected_ranks = selected_terminal["ranks"]
                selected_total = selected_terminal["total_active_rank"]
                rank_rows.append(
                    {
                        "source_artifact": record["_source"],
                        "schema_version": record["schema_version"],
                        "method": record["method"],
                        "seed": record["seed"],
                        "run_segment": record["_run_segment"],
                        "global_step": record["global_step"],
                        "wall_time_seconds": record["wall_time_seconds"],
                        "event_type": event_type,
                        "state_role": "selected_best_checkpoint",
                        "total_active_rank": (
                            "" if selected_total is None else selected_total
                        ),
                        "active_model_parameter_count": (
                            ""
                            if selected_terminal["active_model_parameter_count"] is None
                            else selected_terminal["active_model_parameter_count"]
                        ),
                        "runtime_trainable_parameter_count": (
                            ""
                            if selected_terminal[
                                "runtime_trainable_parameter_count"
                            ]
                            is None
                            else selected_terminal[
                                "runtime_trainable_parameter_count"
                            ]
                        ),
                        "full_model_parameter_count": (
                            ""
                            if selected_terminal["full_model_parameter_count"] is None
                            else selected_terminal["full_model_parameter_count"]
                        ),
                        "physical_rank_component_count": (
                            ""
                            if selected_terminal["physical_rank_component_count"] is None
                            else selected_terminal["physical_rank_component_count"]
                        ),
                        "target_active_parameter_count": (
                            ""
                            if last_counts["target_active_parameter_count"] is None
                            else last_counts["target_active_parameter_count"]
                        ),
                        "selected_event_rank": "",
                        "allocation_stopped": "",
                        "stop_reason": "",
                    }
                )
                if selected_ranks is not None:
                    for name in sorted(selected_ranks):
                        layer, family, group = _module_coordinates(name)
                        module_rows.append(
                            {
                                "source_artifact": record["_source"],
                                "schema_version": record["schema_version"],
                                "method": record["method"],
                                "seed": record["seed"],
                                "run_segment": record["_run_segment"],
                                "global_step": record["global_step"],
                                "wall_time_seconds": record["wall_time_seconds"],
                                "event_type": event_type,
                                "state_role": "selected_best_checkpoint",
                                "module_name": name,
                                "transformer_layer": layer,
                                "module_family": family,
                                "module_group": group,
                                "active_rank": selected_ranks[name],
                            }
                        )

    return {
        "rank_trajectory.csv": rank_rows,
        "module_rank_trajectory.csv": module_rows,
        "allocation_events.csv": allocation_rows,
        "calibration_events.csv": calibration_rows,
        "evaluation_trajectory.csv": evaluation_rows,
    }


def _sort_rows(filename, rows):
    common = lambda row: (
        row["method"],
        int(row["seed"]),
        row["source_artifact"],
        int(row["run_segment"]),
        int(row["global_step"]),
    )
    state_order = {
        "initial_trajectory": 0,
        "trajectory": 1,
        "selected_best_checkpoint": 2,
        "final_trajectory": 3,
    }
    if filename == "module_rank_trajectory.csv":
        return sorted(
            rows,
            key=lambda row: common(row)
            + (state_order.get(row.get("state_role"), 9), row["module_name"]),
        )
    if filename == "rank_trajectory.csv":
        return sorted(
            rows,
            key=lambda row: common(row)
            + (state_order.get(row.get("state_role"), 9),),
        )
    if filename == "calibration_events.csv":
        return sorted(rows, key=lambda row: common(row) + (int(row["candidate_index"]),))
    return sorted(rows, key=common)


def write_tables(tables, output_dir):
    """Atomically write all table headers, including tables with zero rows."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for filename, columns in OUTPUT_SCHEMAS.items():
        destination = output_dir / filename
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=str(output_dir),
            prefix=filename + ".",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
            writer.writeheader()
            writer.writerows(_sort_rows(filename, tables.get(filename, [])))
        temporary.replace(destination)
        written.append(destination)
    return written


def parse_files(inputs, output_dir):
    combined = {filename: [] for filename in OUTPUT_SCHEMAS}
    resolved = [Path(path).resolve() for path in inputs]
    if len(resolved) != len(set(resolved)):
        raise TelemetryValidationError("Input telemetry paths must be unique.")
    for path in resolved:
        transformed = transform_records(load_jsonl(path))
        for filename in combined:
            combined[filename].extend(transformed[filename])
    written = write_tables(combined, output_dir)
    return written, {filename: len(rows) for filename, rows in combined.items()}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="One or more version-1 telemetry JSONL files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New or existing directory for the five derived CSV files.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        written, counts = parse_files(args.inputs, args.output_dir)
    except TelemetryValidationError as error:
        raise SystemExit("Telemetry validation failed: %s" % error) from error
    for path in written:
        print("%s rows=%s" % (path, counts[path.name]))


if __name__ == "__main__":
    main()
