import importlib.util
import json
import random
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch
from torch import nn

from loralib import RankAllocator, SVDLinear
from transformers import Trainer, TrainingArguments
from transformers.rank_telemetry import (
    JsonlTelemetryWriter,
    SCHEMA_VERSION,
    snapshot_module_ranks,
    snapshot_parameter_counts,
    snapshot_rank_state,
    to_json_safe,
)
from transformers.optimization import AdamW

from NLU.tests.test_budgeted_evo_allocator import TinyDynamicModel


def load_rank_telemetry_parser():
    path = Path(__file__).resolve().parents[2] / "analysis" / "parse_rank_telemetry.py"
    spec = importlib.util.spec_from_file_location("rank_telemetry_test_parser", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load the rank telemetry parser from %s" % path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rng_snapshot():
    return (
        random.getstate(),
        np.random.get_state(),
        torch.random.get_rng_state().clone(),
        (
            [state.clone() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available() and torch.cuda.is_initialized()
            else None
        ),
    )


def assert_rng_unchanged(testcase, before, after):
    testcase.assertEqual(before[0], after[0])
    testcase.assertEqual(before[1][0], after[1][0])
    testcase.assertTrue(np.array_equal(before[1][1], after[1][1]))
    testcase.assertEqual(before[1][2:], after[1][2:])
    testcase.assertTrue(torch.equal(before[2], after[2]))
    testcase.assertEqual(before[3] is None, after[3] is None)
    if before[3] is not None:
        testcase.assertEqual(len(before[3]), len(after[3]))
        for before_state, after_state in zip(before[3], after[3]):
            testcase.assertTrue(torch.equal(before_state, after_state))


class JsonSafeSerializationTest(unittest.TestCase):
    def test_recursive_numpy_and_torch_conversion_is_strict_json(self):
        value = {
            "numpy_scalar": np.int64(7),
            "numpy_array": np.array([np.float32(1.5), np.float32(2.5)]),
            "torch_scalar": torch.tensor(3.25),
            "torch_array": torch.tensor([[1, 2], [3, 4]]),
            "nested": (np.bool_(True), Path("rank.json"), {np.int32(4)}),
        }
        safe = to_json_safe(value)
        self.assertEqual(safe["numpy_scalar"], 7)
        self.assertEqual(safe["numpy_array"], [1.5, 2.5])
        self.assertEqual(safe["torch_scalar"], 3.25)
        self.assertEqual(safe["torch_array"], [[1, 2], [3, 4]])
        self.assertEqual(safe["nested"], [True, "rank.json", [4]])
        json.dumps(safe, allow_nan=False)

    def test_nonfinite_values_have_deliberate_policy(self):
        value = {
            "nan": np.float64(np.nan),
            "positive": torch.tensor(float("inf")),
            "negative": float("-inf"),
        }
        self.assertEqual(
            to_json_safe(value),
            {"nan": "NaN", "positive": "Infinity", "negative": "-Infinity"},
        )
        self.assertEqual(
            to_json_safe(value, nonfinite_policy="null"),
            {"nan": None, "positive": None, "negative": None},
        )
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            to_json_safe(value, nonfinite_policy="raise")


class ObservationalSnapshotTest(unittest.TestCase):
    def test_rank_and_count_snapshot_does_not_mutate_model_or_rng(self):
        model = TinyDynamicModel()
        model.small.add_reserve_param(2, advance_learn=True)
        with torch.no_grad():
            model.small.ranknum.fill_(2.0)
        model.large.lora_E[0].requires_grad_(False)

        ranks_before = {
            name: int(round(float(module.ranknum.item())))
            for name, module in model.named_modules()
            if isinstance(module, SVDLinear)
        }
        flags_before = {
            name: bool(parameter.requires_grad)
            for name, parameter in model.named_parameters()
        }
        tensors_before = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
        }
        before_rng = rng_snapshot()

        ranks = snapshot_module_ranks(model)
        counts = snapshot_parameter_counts(model)
        state = snapshot_rank_state(model)

        self.assertEqual(ranks, {"large": 1, "small": 2})
        self.assertEqual(state["module_active_ranks"], ranks)
        self.assertEqual(state["total_active_rank"], 3)
        self.assertEqual(state["physical_rank_component_count"], 4)
        self.assertEqual(
            counts,
            {
                key: state[key]
                for key in (
                    "active_model_parameter_count",
                    "runtime_trainable_parameter_count",
                    "full_model_parameter_count",
                )
            },
        )
        # Runtime requires_grad accounting is deliberately independent from
        # active-rank accounting; either may be larger for an unusual flag mask.
        self.assertGreater(counts["runtime_trainable_parameter_count"], 0)
        self.assertGreater(counts["active_model_parameter_count"], 0)

        ranks_after = {
            name: int(round(float(module.ranknum.item())))
            for name, module in model.named_modules()
            if isinstance(module, SVDLinear)
        }
        flags_after = {
            name: bool(parameter.requires_grad)
            for name, parameter in model.named_parameters()
        }
        self.assertEqual(ranks_after, ranks_before)
        self.assertEqual(flags_after, flags_before)
        for name, parameter in model.named_parameters():
            self.assertTrue(torch.equal(parameter, tensors_before[name]), name)
        assert_rng_unchanged(self, before_rng, rng_snapshot())

    def test_invalid_physical_rank_capacity_is_rejected(self):
        class InvalidDynamicModule(nn.Module):
            def get_dynamic_lora_metadata(self):
                return {"active_rank": 2, "rank_component_count": 1}

        model = nn.Module()
        model.dynamic = InvalidDynamicModule()
        with self.assertRaisesRegex(ValueError, "invalid rank"):
            snapshot_rank_state(model)


class JsonlTelemetryWriterTest(unittest.TestCase):
    def test_append_flush_schema_and_resume_semantics(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            with mock.patch(
                "transformers.rank_telemetry.time.monotonic",
                side_effect=[10.0, 12.5],
            ):
                writer = JsonlTelemetryWriter(
                    path,
                    enabled=True,
                    append=False,
                    base_fields={"seed": np.int64(41), "method": "C-GEAR"},
                )
                self.assertTrue(
                    writer.emit(
                        "allocation",
                        200,
                        gain=torch.tensor(0.5),
                        score=np.float32(np.nan),
                    )
                )
                # Flush makes the complete newline-terminated event immediately visible.
                self.assertTrue(path.read_bytes().endswith(b"\n"))
                writer.close()

            resumed = JsonlTelemetryWriter(
                path,
                enabled=True,
                append=True,
                base_fields={"seed": 41, "method": "C-GEAR"},
                start_time=0.0,
            )
            self.assertTrue(
                resumed.emit(
                    {
                        "event_type": "resume",
                        "global_step": 200,
                        "checkpoint": Path("checkpoint-200"),
                    }
                )
            )
            resumed.close()

            records = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(len(records), 2)
            for record in records:
                self.assertEqual(record["schema_version"], SCHEMA_VERSION)
                for field in (
                    "event_type",
                    "global_step",
                    "seed",
                    "method",
                    "wall_time_seconds",
                ):
                    self.assertIn(field, record)
            self.assertEqual(records[0]["wall_time_seconds"], 2.5)
            self.assertEqual(records[0]["score"], "NaN")
            self.assertEqual(records[1]["checkpoint"], "checkpoint-200")

    def test_fresh_writer_truncates_stale_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            path.write_text("stale\n", encoding="utf-8")
            writer = JsonlTelemetryWriter(path, enabled=True, append=False)
            writer.emit("start", 0)
            writer.close()
            self.assertNotIn("stale", path.read_text(encoding="utf-8"))

    def test_resume_removes_only_incomplete_trailing_crash_fragment(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            completed = {
                "schema_version": SCHEMA_VERSION,
                "event_type": "allocation_event",
                "global_step": 200,
                "seed": 41,
                "method": "C-GEAR",
                "wall_time_seconds": 12.0,
            }
            completed_bytes = json.dumps(completed, sort_keys=True).encode("utf-8")
            path.write_bytes(completed_bytes + b'\n{"schema_version":"rank_telemetry.v1"')
            before_rng = rng_snapshot()

            writer = JsonlTelemetryWriter(
                path,
                enabled=True,
                append=True,
                base_fields={"seed": 41, "method": "C-GEAR"},
            )
            self.assertTrue(
                writer.emit(
                    "run_start",
                    200,
                    resumed=True,
                    resume_from_checkpoint="checkpoint-200",
                )
            )
            writer.close()

            payload = path.read_bytes()
            self.assertTrue(payload.startswith(completed_bytes + b"\n"))
            self.assertTrue(payload.endswith(b"\n"))
            records = [json.loads(line) for line in payload.splitlines()]
            self.assertEqual(len(records), 2)
            self.assertEqual(records[0], completed)
            self.assertEqual(records[1]["event_type"], "run_start")
            self.assertTrue(records[1]["resumed"])
            assert_rng_unchanged(self, before_rng, rng_snapshot())

    def test_resume_preserves_complete_final_json_missing_only_newline(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            completed = {
                "schema_version": SCHEMA_VERSION,
                "event_type": "checkpoint_save",
                "global_step": 200,
                "seed": 41,
                "method": "C-GEAR",
                "wall_time_seconds": 13.0,
            }
            completed_bytes = json.dumps(completed, sort_keys=True).encode("utf-8")
            path.write_bytes(completed_bytes)

            writer = JsonlTelemetryWriter(
                path,
                enabled=True,
                append=True,
                base_fields={"seed": 41, "method": "C-GEAR"},
            )
            self.assertTrue(writer.emit("run_start", 200, resumed=True))
            writer.close()

            payload = path.read_bytes()
            self.assertTrue(payload.startswith(completed_bytes + b"\n"))
            records = [json.loads(line) for line in payload.splitlines()]
            self.assertEqual(records[0], completed)
            self.assertEqual(records[1]["event_type"], "run_start")

    def test_emit_preserves_initialized_cuda_rng(self):
        if not (torch.cuda.is_available() and torch.cuda.is_initialized()):
            self.skipTest("CUDA RNG is not already initialized")
        with tempfile.TemporaryDirectory() as directory:
            before = [state.clone() for state in torch.cuda.get_rng_state_all()]
            writer = JsonlTelemetryWriter(
                Path(directory) / "events.jsonl",
                enabled=True,
                append=False,
            )
            self.assertTrue(writer.emit("allocation_event", 100))
            writer.close()
            after = torch.cuda.get_rng_state_all()
            self.assertEqual(len(before), len(after))
            for before_state, after_state in zip(before, after):
                self.assertTrue(torch.equal(before_state, after_state))

    def test_disabled_writer_is_true_noop(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            before_rng = rng_snapshot()
            writer = JsonlTelemetryWriter(path, enabled=False)
            self.assertFalse(writer.emit(object()))
            self.assertFalse(path.exists())
            self.assertIsNone(writer.last_error)
            assert_rng_unchanged(self, before_rng, rng_snapshot())

    def test_emit_does_not_consume_rng(self):
        with tempfile.TemporaryDirectory() as directory:
            before_rng = rng_snapshot()
            writer = JsonlTelemetryWriter(
                Path(directory) / "events.jsonl", enabled=True, base_fields={"seed": 41}
            )
            self.assertTrue(writer.emit("allocation", 100, value=torch.tensor(1.0)))
            writer.close()
            assert_rng_unchanged(self, before_rng, rng_snapshot())

    def test_failure_isolated_or_raised_according_to_policy(self):
        class Unsupported(object):
            pass

        with tempfile.TemporaryDirectory() as directory:
            isolated = JsonlTelemetryWriter(
                Path(directory) / "isolated.jsonl",
                enabled=True,
                failure_policy="disable",
            )
            self.assertFalse(isolated.emit("bad", 1, value=Unsupported()))
            self.assertFalse(isolated.enabled)
            self.assertIsInstance(isolated.last_error, TypeError)
            self.assertFalse(isolated.emit("ignored", 2))

            strict = JsonlTelemetryWriter(
                Path(directory) / "strict.jsonl",
                enabled=True,
                failure_policy="raise",
            )
            with self.assertRaises(TypeError):
                strict.emit("bad", 1, value=Unsupported())
            strict.close()

    def test_enabled_constructor_open_failure_is_not_silenced(self):
        with tempfile.TemporaryDirectory() as directory:
            invalid_path = Path(directory) / "directory"
            invalid_path.mkdir()
            with self.assertRaises(OSError):
                JsonlTelemetryWriter(
                    invalid_path,
                    enabled=True,
                    failure_policy="disable",
                )

    def test_close_failure_follows_failure_policy(self):
        class FailingStream(object):
            def close(self):
                raise OSError("synthetic stream close failure")

        isolated = JsonlTelemetryWriter(enabled=False, failure_policy="disable")
        isolated.enabled = True
        isolated._stream = FailingStream()
        isolated.close()
        self.assertFalse(isolated.enabled)
        self.assertIsInstance(isolated.last_error, OSError)
        self.assertIsNone(isolated._stream)

        strict = JsonlTelemetryWriter(enabled=False, failure_policy="raise")
        strict.enabled = True
        strict._stream = FailingStream()
        with self.assertRaisesRegex(OSError, "synthetic stream close failure"):
            strict.close()
        self.assertFalse(strict.enabled)
        self.assertIsInstance(strict.last_error, OSError)
        self.assertIsNone(strict._stream)


class AllocatorDecisionInvarianceTest(unittest.TestCase):
    @staticmethod
    def _run_greedy(writer):
        class GreedyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.first = SVDLinear(2, 2, r=1)
                self.second = SVDLinear(2, 2, r=1)
                self.third = SVDLinear(2, 2, r=1)

        torch.manual_seed(17)
        model = GreedyModel()
        allocator = RankAllocator(
            model,
            lora_r=1,
            target_rank=1,
            target_total_rank=5,
            init_warmup=0,
            incre_interval=1,
            top_h=1,
            advance_learn=True,
            beta1=0.85,
            beta2=0.85,
            incre_rank_num=1,
            rank_allocator="greedy",
        )
        allocator.set_rank_telemetry_writer(writer)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        for module in (model.first, model.second, model.third):
            module.score = torch.tensor(0.0)
        allocator.update_and_increase(model, 0, optimizer)
        scores = {"first": 1.0, "second": 2.0, "third": 2.0}
        allocator.calculate_score = lambda name, layer, metric="ipt": scores[name]
        allocator.global_step = 1
        allocator.increase_to_target_rank(model, optimizer)
        return {
            "ranks": snapshot_module_ranks(model),
            "total_rank": allocator.total_rank,
            "optimizer_group_sizes": [
                sum(parameter.numel() for parameter in group["params"])
                for group in optimizer.param_groups
            ],
        }

    def test_allocator_decision_is_exactly_equal_with_telemetry_on_or_off(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "telemetry.jsonl"
            disabled = JsonlTelemetryWriter(enabled=False)
            without_telemetry = self._run_greedy(disabled)
            enabled = JsonlTelemetryWriter(
                path,
                enabled=True,
                append=False,
                base_fields={"seed": 41, "method": "greedy"},
            )
            with_telemetry = self._run_greedy(enabled)
            enabled.close()

            self.assertEqual(with_telemetry, without_telemetry)
            records = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual([record["event_type"] for record in records], ["allocation_event"])
            self.assertEqual(records[0]["selected_modules"], ["second", "third"])
            self.assertEqual(records[0]["module_active_ranks"], with_telemetry["ranks"])
            for field in (
                "active_model_parameter_count",
                "runtime_trainable_parameter_count",
                "full_model_parameter_count",
                "physical_rank_component_count",
            ):
                self.assertIn(field, records[0])

    @staticmethod
    def _run_calibrated(writer, target_total_rank=4, followup_step=None):
        torch.manual_seed(29)
        model = TinyDynamicModel()
        for module in (model.small, model.large):
            module.score = torch.tensor(0.0)
        allocator = RankAllocator(
            model,
            lora_r=1,
            target_rank=1,
            target_total_rank=target_total_rank,
            init_warmup=0,
            incre_interval=1,
            top_h=1,
            advance_learn=True,
            beta1=0.85,
            beta2=0.85,
            incre_rank_num=1,
            weight_decay=0.0,
            rank_allocator="genetic_budgeted_calibrated",
            ga_population=6,
            ga_generations=1,
            ga_local_search=False,
            ga_max_final_trainable_params=1000,
            ga_allow_variable_event_rank=True,
            ga_max_greedy_replacements=1,
            ga_calibration_batches=1,
            ga_calibration_batch_size=1,
            ga_calibration_topk=6,
            ga_allocation_stop_patience=2,
            ga_min_event_rank=1,
            ga_max_event_rank=1,
            ga_min_consolidation_steps=0,
            ga_new_rank_lr_warmup_steps=2,
            training_seed=41,
        )
        allocator.set_rank_telemetry_writer(writer)
        allocator.set_total_step(100)
        optimizer = AdamW(model.parameters(), lr=1e-3)
        calibration_pairs = [({"unused": 1}, {"unused": 2})]

        def finite_calibration(**kwargs):
            names = list(kwargs["candidate_module_names"])
            gain = 1.0 if names else 0.0
            return {
                "candidate_modules": names,
                "candidate_size": len(names),
                "candidate_cost": len(names),
                "fold_gains": [gain],
                "fold_details": [],
                "calibration_gain_mean": gain,
                "calibration_gain_std": 0.0,
                "calibration_gain_lcb": gain,
                "calibration_gain_per_parameter": gain,
                "calibration_valid": True,
                "calibration_invalid_reason": None,
                "calibration_runtime_seconds": 0.0,
            }

        allocator.update_and_increase(
            model,
            0,
            optimizer,
            calibration_batch_pairs=calibration_pairs,
            calibration_loss_fn=lambda model, batch: torch.tensor(0.0),
        )
        with mock.patch(
            "transformers.calibrated_rank_calibration.score_virtual_candidate",
            side_effect=finite_calibration,
        ):
            decision = allocator.update_and_increase(
                model,
                1,
                optimizer,
                calibration_batch_pairs=calibration_pairs,
                calibration_loss_fn=lambda model, batch: torch.tensor(0.0),
            )
            followup_decision = None
            if followup_step is not None:
                followup_decision = allocator.update_and_increase(
                    model,
                    followup_step,
                    optimizer,
                    calibration_batch_pairs=calibration_pairs,
                    calibration_loss_fn=lambda model, batch: torch.tensor(0.0),
                )
        return {
            "decision": decision,
            "followup_decision": followup_decision,
            "ranks": snapshot_module_ranks(model),
            "total_rank": allocator.total_rank,
            "allocation_stopped": allocator.allocation_stopped,
            "allocation_stop_reason": allocator.allocation_stop_reason,
            "final_trajectory_metrics": (
                None
                if allocator.final_trajectory_metrics is None
                else dict(allocator.final_trajectory_metrics)
            ),
            "optimizer_group_sizes": [
                sum(parameter.numel() for parameter in group["params"])
                for group in optimizer.param_groups
            ],
        }

    def test_cgear_decision_is_exactly_equal_with_telemetry_on_or_off(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "telemetry.jsonl"
            without_telemetry = self._run_calibrated(
                JsonlTelemetryWriter(enabled=False)
            )
            enabled = JsonlTelemetryWriter(
                path,
                enabled=True,
                append=False,
                base_fields={
                    "seed": 41,
                    "method": "genetic_budgeted_calibrated",
                },
            )
            with_telemetry = self._run_calibrated(enabled)
            enabled.close()

            self.assertEqual(with_telemetry, without_telemetry)
            records = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(
                [record["event_type"] for record in records],
                ["calibration_event", "candidate_selection", "allocation_event"],
            )
            self.assertEqual(
                records[-1]["module_active_ranks"], with_telemetry["ranks"]
            )
            self.assertIn("active_model_parameter_count", records[-1])
            self.assertIn("budget_limit", records[-1])

    def test_target_reaching_allocation_precedes_stop_and_parses(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "telemetry.jsonl"
            writer = JsonlTelemetryWriter(
                path,
                enabled=True,
                append=False,
                base_fields={
                    "seed": 41,
                    "method": "genetic_budgeted_calibrated",
                },
            )
            result = self._run_calibrated(writer, target_total_rank=3)
            writer.close()

            records = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(
                [record["event_type"] for record in records],
                [
                    "calibration_event",
                    "candidate_selection",
                    "allocation_event",
                    "allocator_stop",
                ],
            )
            allocation, stop = records[-2:]
            self.assertEqual(allocation["selected_event_rank"], 1)
            self.assertTrue(allocation["allocation_stopped"])
            self.assertEqual(allocation["stop_reason"], "maximum_rank_reached")
            self.assertEqual(
                allocation["module_active_ranks"], stop["module_active_ranks"]
            )
            self.assertEqual(
                allocation["total_active_rank"], stop["total_active_rank"]
            )
            self.assertTrue(result["allocation_stopped"])

            parser = load_rank_telemetry_parser()
            tables = parser.transform_records(parser.load_jsonl(path))
            self.assertEqual(len(tables["allocation_events.csv"]), 1)
            self.assertEqual(
                tables["allocation_events.csv"][0]["selected_event_rank"], 1
            )

    def test_stopped_allocator_emits_explicit_unchanged_consolidation_event(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "telemetry.jsonl"
            writer = JsonlTelemetryWriter(
                path,
                enabled=True,
                append=False,
                base_fields={
                    "seed": 41,
                    "method": "genetic_budgeted_calibrated",
                },
            )
            result = self._run_calibrated(
                writer,
                target_total_rank=3,
                followup_step=2,
            )
            writer.close()

            records = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(records[-2]["event_type"], "allocator_stop")
            consolidation = records[-1]
            self.assertEqual(consolidation["event_type"], "allocation_event")
            self.assertEqual(consolidation["selected_k"], 0)
            self.assertEqual(consolidation["selected_event_rank"], 0)
            self.assertEqual(consolidation["selected_modules"], [])
            self.assertEqual(consolidation["rank_increments"], {})
            self.assertEqual(
                consolidation["selected_source"],
                "allocation_stopped_consolidation",
            )
            self.assertTrue(consolidation["allocation_stopped"])
            self.assertEqual(
                consolidation["pre_total_active_rank"],
                consolidation["post_total_active_rank"],
            )
            self.assertEqual(
                consolidation["module_active_ranks"],
                records[-2]["module_active_ranks"],
            )
            self.assertEqual(result["followup_decision"], (0, None))

            parser = load_rank_telemetry_parser()
            tables = parser.transform_records(parser.load_jsonl(path))
            self.assertEqual(len(tables["allocation_events.csv"]), 2)
            self.assertEqual(
                tables["allocation_events.csv"][-1]["zero_growth"], "true"
            )

    def test_final_trajectory_metrics_include_physical_capacity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "telemetry.jsonl"
            writer = JsonlTelemetryWriter(
                path,
                enabled=True,
                append=False,
                base_fields={
                    "seed": 41,
                    "method": "genetic_budgeted_calibrated",
                },
            )
            result = self._run_calibrated(writer, target_total_rank=3)
            writer.close()

            metrics = result["final_trajectory_metrics"]
            self.assertIsNotNone(metrics)
            self.assertIn("physical_rank_component_count", metrics)
            self.assertGreaterEqual(
                metrics["physical_rank_component_count"],
                metrics["total_active_rank"],
            )
            stop = json.loads(path.read_text().splitlines()[-1])
            self.assertEqual(
                metrics["physical_rank_component_count"],
                stop["physical_rank_component_count"],
            )


class TrainerEventIntegrationTest(unittest.TestCase):
    class EvaluationModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.adapter = SVDLinear(2, 2, r=1)
            self.classifier = nn.Linear(2, 2)
            self.config = SimpleNamespace()

        def forward(self, input_ids=None, labels=None):
            logits = self.classifier(self.adapter(input_ids.float()))
            loss = nn.functional.cross_entropy(logits, labels)
            return loss, logits

    class EvaluationDataset(torch.utils.data.Dataset):
        examples = (
            (torch.tensor([1.0, 0.0]), torch.tensor(0)),
            (torch.tensor([0.0, 1.0]), torch.tensor(1)),
        )

        def __len__(self):
            return len(self.examples)

        def __getitem__(self, index):
            features, label = self.examples[index]
            return {"input_ids": features, "labels": label}

    def _build_evaluation_trainer(self, directory):
        model = self.EvaluationModel()
        allocator = RankAllocator(
            model,
            lora_r=1,
            target_rank=1,
            target_total_rank=2,
            init_warmup=0,
            incre_interval=1,
            top_h=1,
            advance_learn=True,
            beta1=0.85,
            beta2=0.85,
            incre_rank_num=1,
            rank_allocator="greedy",
        )
        arguments = TrainingArguments(
            output_dir=str(Path(directory) / "model"),
            per_device_eval_batch_size=2,
            disable_tqdm=True,
            report_to=[],
            metric_for_best_model="accuracy",
            greater_is_better=True,
        )
        arguments.rank_telemetry = True
        arguments.multi_lr = True
        arguments.root_output_dir = directory
        arguments._rank_telemetry_resume = False
        arguments._rank_telemetry_metadata = {
            "task": "synthetic",
            "model_name_or_path": "synthetic",
            "max_seq_length": 2,
        }
        metric_calls = []

        def compute_metrics(prediction):
            metric_calls.append(1)
            predicted = np.argmax(prediction.predictions, axis=1)
            return {"accuracy": float(np.mean(predicted == prediction.label_ids))}

        trainer = Trainer(
            model=model,
            args=arguments,
            eval_dataset=self.EvaluationDataset(),
            compute_metrics=compute_metrics,
            rankallocator=allocator,
        )
        return model, allocator, trainer, metric_calls

    def test_existing_evaluation_and_checkpoint_emit_without_extra_evaluation(self):

        with tempfile.TemporaryDirectory() as directory:
            model, allocator, trainer, metric_calls = self._build_evaluation_trainer(
                directory
            )
            with mock.patch.object(
                trainer,
                "prediction_loop",
                wraps=trainer.prediction_loop,
            ) as prediction_loop:
                metrics = trainer.evaluate()
            self.assertEqual(prediction_loop.call_count, 1)
            self.assertEqual(len(metric_calls), 1)

            trainer.create_optimizer_and_scheduler(num_training_steps=1)
            trainer.state.global_step = 1
            trainer._save_checkpoint(model, trial=None, metrics=metrics)
            model.adapter.add_reserve_param(1, advance_learn=True)
            final_trajectory = snapshot_rank_state(model)
            final_trajectory["module_active_ranks"] = {"adapter": 2}
            final_trajectory["total_active_rank"] = 2
            final_trajectory["active_model_parameter_count"] += 5
            allocator.rank_telemetry_final_trajectory_snapshot = final_trajectory
            trainer.emit_rank_telemetry_run_end({"train_runtime": 1.25})

            records = [
                json.loads(line)
                for line in (Path(directory) / "telemetry.jsonl").read_text().splitlines()
            ]
            self.assertEqual(
                [record["event_type"] for record in records],
                ["run_start", "evaluation", "checkpoint_save", "run_end"],
            )
            run_start = records[0]
            self.assertEqual(run_start["task"], "synthetic")
            self.assertEqual(run_start["allocator_type"], "greedy")
            self.assertEqual(run_start["optimizer"], "adamw")
            self.assertIn("ga_quality_relative_tolerance", run_start["allocator_configuration"])
            self.assertIn("lora_configuration", run_start)
            evaluation = records[1]
            checkpoint = records[2]
            self.assertEqual(evaluation["accuracy"], metrics["eval_accuracy"])
            self.assertEqual(evaluation["state_role"], "standalone_evaluation")
            self.assertEqual(
                evaluation["module_active_ranks"], snapshot_module_ranks(model)
            )
            self.assertTrue(checkpoint["is_best_checkpoint"])
            self.assertEqual(checkpoint["best_metric_so_far"], metrics["eval_accuracy"])
            self.assertTrue(Path(checkpoint["checkpoint_path"]).name.startswith("checkpoint-"))
            self.assertEqual(records[-1]["selected_active_rank"], 1)
            self.assertEqual(records[-1]["final_active_rank"], 2)
            self.assertEqual(records[-1]["selected_module_active_ranks"], {"adapter": 1})
            self.assertEqual(records[-1]["final_module_active_ranks"], {"adapter": 2})

    def test_run_end_isolates_writer_close_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, trainer, _ = self._build_evaluation_trainer(directory)
            writer = trainer._rank_telemetry_writer
            close_error = OSError("synthetic close failure")

            with mock.patch.object(
                writer,
                "close",
                side_effect=close_error,
            ) as close:
                trainer.emit_rank_telemetry_run_end({"train_runtime": 0.25})

            self.assertTrue(close.called)
            self.assertTrue(trainer._rank_telemetry_run_ended)
            self.assertFalse(writer.enabled)
            self.assertIs(writer.last_error, close_error)
            records = [
                json.loads(line)
                for line in (Path(directory) / "telemetry.jsonl").read_text().splitlines()
            ]
            self.assertEqual(
                [record["event_type"] for record in records],
                ["run_start", "run_end"],
            )
            writer.close()

    def test_evaluation_labels_trajectory_and_selected_best_without_extra_eval(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, trainer, metric_calls = self._build_evaluation_trainer(directory)

            trainer.is_in_train = True
            with mock.patch.object(
                trainer,
                "prediction_loop",
                wraps=trainer.prediction_loop,
            ) as prediction_loop:
                trainer.evaluate()
            self.assertEqual(prediction_loop.call_count, 1)
            trainer.is_in_train = False

            selected_checkpoint = str(Path(directory) / "model" / "checkpoint-100")
            trainer.state.global_step = 200
            trainer.state.best_model_checkpoint = selected_checkpoint
            trainer._best_model_was_loaded = True
            with mock.patch.object(
                trainer,
                "prediction_loop",
                wraps=trainer.prediction_loop,
            ) as prediction_loop:
                trainer.evaluate()
            self.assertEqual(prediction_loop.call_count, 1)
            self.assertEqual(len(metric_calls), 2)
            trainer._rank_telemetry_writer.close()

            path = Path(directory) / "telemetry.jsonl"
            records = [json.loads(line) for line in path.read_text().splitlines()]
            evaluations = [
                record for record in records if record["event_type"] == "evaluation"
            ]
            self.assertEqual(len(evaluations), 2)
            self.assertEqual(
                evaluations[0]["state_role"],
                "training_trajectory_evaluation",
            )
            self.assertEqual(
                evaluations[0]["best_state_phase"],
                "before_current_checkpoint_selection",
            )
            self.assertIsNone(evaluations[0]["evaluated_checkpoint"])
            self.assertEqual(
                evaluations[1]["state_role"],
                "selected_best_checkpoint_evaluation",
            )
            self.assertEqual(
                evaluations[1]["best_state_phase"],
                "selected_best_checkpoint_loaded",
            )
            self.assertEqual(
                evaluations[1]["evaluated_checkpoint"], selected_checkpoint
            )

            parser = load_rank_telemetry_parser()
            tables = parser.transform_records(parser.load_jsonl(path))
            evaluation_rows = tables["evaluation_trajectory.csv"]
            self.assertEqual(
                [row["state_role"] for row in evaluation_rows],
                [
                    "training_trajectory_evaluation",
                    "selected_best_checkpoint_evaluation",
                ],
            )
            self.assertEqual(
                evaluation_rows[1]["checkpoint"], selected_checkpoint
            )


if __name__ == "__main__":
    unittest.main()
