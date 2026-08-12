# coding=utf-8
"""Observational JSONL telemetry for dynamic-rank experiments.

This module deliberately has no training integration.  Its public helpers only
read model state, and :class:`JsonlTelemetryWriter` isolates optional telemetry
failures from the training process unless the caller explicitly requests a
raising failure policy.
"""

from __future__ import absolute_import, division, print_function

import json
import logging
import math
import os
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

from loralib import (
    get_active_model_parameter_count,
    get_full_model_parameter_count,
    get_runtime_trainable_parameter_count,
)


SCHEMA_VERSION = "rank_telemetry.v1"
_NONFINITE_POLICIES = ("string", "null", "raise")
_FAILURE_POLICIES = ("disable", "raise")
logger = logging.getLogger(__name__)


def _reject_nonfinite_json_constant(value):
    raise ValueError("Non-finite JSON constant %s is not valid JSON." % value)


def _json_safe_float(value, nonfinite_policy):
    value = float(value)
    if math.isfinite(value):
        return value
    if nonfinite_policy == "null":
        return None
    if nonfinite_policy == "raise":
        raise ValueError("Non-finite floating-point values are not valid telemetry.")
    if math.isnan(value):
        return "NaN"
    return "Infinity" if value > 0 else "-Infinity"


def to_json_safe(value, nonfinite_policy="string"):
    """Recursively convert common scientific Python values to strict JSON data.

    Non-finite floats are encoded as ``"NaN"``, ``"Infinity"``, and
    ``"-Infinity"`` by default.  ``nonfinite_policy="null"`` maps them to JSON
    null, while ``"raise"`` rejects them.  The returned value contains no live
    NumPy arrays or Torch tensors.
    """

    if nonfinite_policy not in _NONFINITE_POLICIES:
        raise ValueError(
            "nonfinite_policy must be one of %s." % (list(_NONFINITE_POLICIES),)
        )
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, np.floating)):
        return _json_safe_float(value, nonfinite_policy)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, torch.Tensor):
        detached = value.detach()
        if detached.numel() == 1:
            return to_json_safe(detached.item(), nonfinite_policy)
        return to_json_safe(detached.cpu().tolist(), nonfinite_policy)
    if isinstance(value, np.ndarray):
        return to_json_safe(value.tolist(), nonfinite_policy)
    if isinstance(value, dict):
        return {
            str(key): to_json_safe(item, nonfinite_policy)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [to_json_safe(item, nonfinite_policy) for item in value]
    if isinstance(value, (set, frozenset)):
        return [
            to_json_safe(item, nonfinite_policy)
            for item in sorted(value, key=lambda item: repr(item))
        ]
    if isinstance(value, Path):
        return str(value)
    raise TypeError(
        "Telemetry value of type %s is not JSON serializable." % type(value).__name__
    )


def _snapshot_dynamic_rank_metadata(model):
    ranks = OrderedDict()
    capacities = OrderedDict()
    for name, module in sorted(model.named_modules(), key=lambda item: item[0]):
        metadata_getter = getattr(module, "get_dynamic_lora_metadata", None)
        if not callable(metadata_getter):
            continue
        metadata = metadata_getter()
        if not isinstance(metadata, dict) or "active_rank" not in metadata:
            raise ValueError(
                "Dynamic LoRA module '%s' did not report an active rank." % name
            )
        rank = metadata["active_rank"]
        capacity = metadata.get("rank_component_count")
        if isinstance(rank, bool) or isinstance(capacity, bool):
            raise ValueError("Dynamic LoRA module '%s' reported an invalid rank." % name)
        try:
            numeric_rank = float(rank)
            rank = int(rank)
            numeric_capacity = float(capacity)
            capacity = int(capacity)
        except (TypeError, ValueError, OverflowError):
            raise ValueError("Dynamic LoRA module '%s' reported an invalid rank." % name)
        if (
            not math.isfinite(numeric_rank)
            or numeric_rank != rank
            or rank < 1
            or not math.isfinite(numeric_capacity)
            or numeric_capacity != capacity
            or capacity < rank
        ):
            raise ValueError("Dynamic LoRA module '%s' reported an invalid rank." % name)
        ranks[name] = rank
        capacities[name] = capacity
    return dict(ranks), dict(capacities)


def snapshot_module_ranks(model):
    """Return a stable module-name-to-active-rank map without mutating ``model``."""

    ranks, _ = _snapshot_dynamic_rank_metadata(model)
    return ranks


def snapshot_parameter_counts(model):
    """Read the three established model parameter-count definitions."""

    return {
        "active_model_parameter_count": int(get_active_model_parameter_count(model)),
        "runtime_trainable_parameter_count": int(
            get_runtime_trainable_parameter_count(model)
        ),
        "full_model_parameter_count": int(get_full_model_parameter_count(model)),
    }


def snapshot_rank_state(model):
    """Return canonical ranks and counts in one detached observational snapshot."""

    module_ranks, module_capacities = _snapshot_dynamic_rank_metadata(model)
    snapshot = {
        "module_active_ranks": module_ranks,
        "total_active_rank": int(sum(module_ranks.values())),
        "physical_rank_component_count": int(sum(module_capacities.values())),
    }
    snapshot.update(snapshot_parameter_counts(model))
    return snapshot


class JsonlTelemetryWriter(object):
    """Append-and-flush writer for optional, crash-resilient rank telemetry.

    ``append=True`` preserves existing JSONL for resume; ``append=False``
    truncates stale telemetry for a fresh run.  With
    ``failure_policy="disable"`` an I/O or serialization failure is remembered
    in :attr:`last_error`, disables subsequent writes, and returns ``False``.
    ``failure_policy="raise"`` lets explicitly enabled callers make telemetry
    failure fatal instead.
    """

    def __init__(
        self,
        path=None,
        enabled=False,
        base_fields=None,
        start_time=None,
        failure_policy="disable",
        nonfinite_policy="string",
        fsync=False,
        append=True,
    ):
        if failure_policy not in _FAILURE_POLICIES:
            raise ValueError("failure_policy must be 'disable' or 'raise'.")
        if nonfinite_policy not in _NONFINITE_POLICIES:
            raise ValueError(
                "nonfinite_policy must be one of %s." % (list(_NONFINITE_POLICIES),)
            )
        if enabled and path is None:
            raise ValueError("An enabled telemetry writer requires a path.")
        if base_fields is not None and not isinstance(base_fields, dict):
            raise TypeError("base_fields must be a dictionary or None.")

        self.path = Path(path).expanduser() if path is not None else None
        self.enabled = bool(enabled)
        self.base_fields = dict(base_fields or {})
        self.start_time = time.monotonic() if start_time is None else float(start_time)
        self.failure_policy = failure_policy
        self.nonfinite_policy = nonfinite_policy
        self.fsync = bool(fsync)
        self.append = bool(append)
        self.last_error = None
        self._stream = None

        # Opening an explicitly enabled sink is configuration, not an event.
        # Fail early and clearly instead of silently running without telemetry.
        if self.enabled:
            self._open()

    def _disable_or_raise(self, error):
        self.enabled = False
        try:
            self.close()
        except Exception as close_error:
            logger.warning("Rank telemetry also failed while closing: %s", close_error)
        # Preserve the event/serialization failure as the primary cause even
        # if best-effort close encountered a second error.
        self.last_error = error
        if self.failure_policy == "raise":
            raise error
        logger.warning(
            "Disabling rank telemetry after an observational failure: %s", error
        )
        return False

    def _open(self):
        if self._stream is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self.append:
                self._repair_append_tail()
            self._stream = self.path.open("a" if self.append else "w", encoding="utf-8")
            # After the first open, never truncate this writer's stream again.
            self.append = True
        return self._stream

    def _repair_append_tail(self):
        """Make a crash-interrupted JSONL tail safe before a resumed append.

        A complete final JSON value without its newline is preserved by adding
        the missing delimiter.  Only an invalid/incomplete final fragment is
        removed; every preceding newline-terminated byte remains untouched.
        """

        try:
            stream = self.path.open("r+b")
        except FileNotFoundError:
            return
        with stream:
            stream.seek(0, os.SEEK_END)
            end = stream.tell()
            if end == 0:
                return
            stream.seek(end - 1)
            if stream.read(1) == b"\n":
                return

            # Find the start of only the unterminated final fragment without
            # reading an otherwise potentially large telemetry file.
            position = end
            fragment_start = 0
            block_size = 8192
            while position > 0:
                read_start = max(0, position - block_size)
                stream.seek(read_start)
                block = stream.read(position - read_start)
                newline_index = block.rfind(b"\n")
                if newline_index >= 0:
                    fragment_start = read_start + newline_index + 1
                    break
                position = read_start

            stream.seek(fragment_start)
            fragment = stream.read(end - fragment_start)
            try:
                decoded = fragment.decode("utf-8")
                json.loads(
                    decoded,
                    parse_constant=_reject_nonfinite_json_constant,
                )
            except (UnicodeDecodeError, ValueError):
                stream.truncate(fragment_start)
                logger.warning(
                    "Removed %s-byte incomplete telemetry tail before append: %s",
                    end - fragment_start,
                    self.path,
                )
            else:
                stream.seek(end)
                stream.write(b"\n")

    def _record(self, event_type, global_step, fields):
        record = dict(self.base_fields)
        record.update(fields)
        record["schema_version"] = SCHEMA_VERSION
        record["event_type"] = event_type
        record["global_step"] = global_step
        record["wall_time_seconds"] = time.monotonic() - self.start_time
        # Keep these common fields present even if a caller has no value yet.
        record.setdefault("seed", None)
        record.setdefault("method", None)
        return record

    def emit(self, event_or_record, global_step=None, **fields):
        """Append one event, accepting either a mapping or event-style arguments.

        ``emit(record)`` expects a mapping and fills missing common fields.
        ``emit(event_type, global_step, **fields)`` supplies them positionally.
        Returns ``True`` on success and ``False`` when disabled or after an
        isolated failure.
        """

        if not self.enabled:
            return False
        try:
            if isinstance(event_or_record, dict):
                if global_step is not None or fields:
                    raise TypeError(
                        "emit(record) cannot be combined with global_step or keyword fields."
                    )
                supplied = dict(event_or_record)
                event_type = supplied.pop("event_type", None)
                step = supplied.pop("global_step", None)
            else:
                event_type = event_or_record
                step = global_step
                supplied = fields
            if not isinstance(event_type, str) or not event_type:
                raise ValueError("event_type must be a nonempty string.")

            record = self._record(event_type, step, supplied)
            safe = to_json_safe(record, nonfinite_policy=self.nonfinite_policy)
            line = json.dumps(
                safe,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            stream = self._open()
            stream.write(line + "\n")
            stream.flush()
            if self.fsync:
                os.fsync(stream.fileno())
            return True
        except Exception as error:
            return self._disable_or_raise(error)

    write = emit

    def flush(self):
        if self._stream is not None:
            self._stream.flush()
            if self.fsync:
                os.fsync(self._stream.fileno())

    def close(self):
        if self._stream is not None:
            try:
                self._stream.close()
            except Exception as error:
                self.last_error = error
                self.enabled = False
                logger.warning("Rank telemetry failed while closing: %s", error)
                if self.failure_policy == "raise":
                    raise
            finally:
                self._stream = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False


__all__ = [
    "JsonlTelemetryWriter",
    "SCHEMA_VERSION",
    "snapshot_module_ranks",
    "snapshot_parameter_counts",
    "snapshot_rank_state",
    "to_json_safe",
]
