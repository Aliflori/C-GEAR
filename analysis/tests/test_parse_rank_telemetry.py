#!/usr/bin/env python3
"""Lightweight synthetic regression tests for rank telemetry parsing."""

import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "parse_rank_telemetry.py"
SPEC = importlib.util.spec_from_file_location("parse_rank_telemetry", SCRIPT)
TELEMETRY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TELEMETRY)


MODULES = (
    "deberta.encoder.layer.0.attention.self.query_proj",
    "deberta.encoder.layer.0.attention.output.dense",
    "deberta.encoder.layer.0.intermediate.dense",
    "deberta.encoder.layer.0.output.dense",
)


def common(event_type, step, wall_time):
    return {
        "schema_version": "rank_telemetry.v1",
        "event_type": event_type,
        "global_step": step,
        "seed": 41,
        "method": "C-GEAR",
        "wall_time_seconds": wall_time,
    }


def valid_records():
    initial = {name: 1 for name in MODULES}
    grown = dict(initial)
    grown[MODULES[0]] = 2
    grown[MODULES[2]] = 2
    records = [
        dict(
            common("run_start", 0, 0.0),
            module_active_ranks=initial,
            total_active_rank=4,
            active_model_parameter_count=100,
            runtime_trainable_parameter_count=110,
            full_model_parameter_count=1000,
            target_active_parameter_count=150,
        ),
        dict(
            common("calibration_event", 100, 10.0),
            selected_candidate_id="c1",
            candidates=[
                {
                    "candidate_id": "c1",
                    "modules": [MODULES[0], MODULES[2]],
                    "k": 2,
                    "parameter_cost": 20,
                    "candidate_source": "global_ga",
                    "fold_scores": [0.05, 0.04, 0.03],
                    "mean_score": 0.04,
                    "std_score": 0.0081649658,
                    "lcb_score": 0.0359175171,
                    "gain_per_parameter": 0.002,
                    "calibration_valid": True,
                },
                {
                    "candidate_id": "zero",
                    "candidate_modules": [],
                    "candidate_size": 0,
                    "candidate_cost": 0,
                    "candidate_family": "zero_rank",
                    "fold_gains": [0.0, 0.0, 0.0],
                    "calibration_gain_mean": 0.0,
                    "calibration_gain_std": 0.0,
                    "calibration_gain_lcb": 0.0,
                    "calibration_valid": True,
                },
            ],
        ),
        dict(common("candidate_selection", 100, 11.0), selected_candidate_id="c1"),
        dict(
            common("allocation_event", 100, 12.0),
            module_active_ranks=grown,
            total_active_rank=6,
            pre_total_active_rank=4,
            post_total_active_rank=6,
            selected_k=2,
            selected_event_rank=2,
            selected_modules=[MODULES[0], MODULES[2]],
            selected_source="calibrated_global_ga",
            active_model_parameter_count=120,
            pre_active_parameter_count=100,
            post_active_parameter_count=120,
            runtime_trainable_parameter_count=130,
            budget_limit=150,
            budget_used=120,
            budget_remaining=30,
            rank_increments={MODULES[0]: 1, MODULES[2]: 1},
            allocation_stopped=False,
        ),
        dict(
            common("evaluation", 100, 13.0),
            split="validation",
            metrics={"eval_accuracy": 0.75, "eval_loss": 0.5},
            active_model_parameter_count=120,
            total_active_rank=6,
        ),
        dict(
            common("calibration_event", 200, 20.0),
            candidate_id="zero",
            candidate_modules=[],
            candidate_size=0,
            candidate_cost=0,
            candidate_family="zero_rank",
            fold_gains=[0.0, 0.0, 0.0],
            calibration_gain_mean=0.0,
            calibration_gain_std=0.0,
            calibration_gain_lcb=0.0,
            calibration_valid=True,
            selected_candidate_id="zero",
        ),
        dict(common("candidate_selection", 200, 21.0), selected_candidate_id="zero"),
        dict(
            common("allocation_event", 200, 22.0),
            module_active_ranks=grown,
            total_active_rank=6,
            pre_total_active_rank=6,
            post_total_active_rank=6,
            selected_k=0,
            selected_event_rank=0,
            selected_modules=[],
            selected_source="calibrated_zero_rank",
            active_model_parameter_count=120,
            pre_active_parameter_count=120,
            post_active_parameter_count=120,
            runtime_trainable_parameter_count=130,
            budget_limit=150,
            budget_used=120,
            budget_remaining=30,
            rank_increments={},
            allocation_stopped=False,
        ),
        dict(
            common("evaluation", 200, 23.0),
            accuracy=0.80,
            loss=0.4,
            active_model_parameter_count=120,
            total_active_rank=6,
        ),
        dict(
            common("allocator_stop", 200, 24.0),
            stop_reason="zero_rank_patience_exhausted",
            allocation_stopped=True,
        ),
        dict(common("checkpoint_save", 200, 25.0), checkpoint="checkpoint-200"),
        dict(common("warning", 200, 26.0), message="synthetic warning"),
        dict(common("run_end", 300, 30.0), status="completed"),
    ]
    return records


def write_jsonl(path, records):
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


class RankTelemetryParserTest(unittest.TestCase):
    def test_valid_nested_and_candidate_level_calibration(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "telemetry.jsonl"
            output = root / "parsed"
            write_jsonl(source, valid_records())
            paths, counts = TELEMETRY.parse_files([source], output)

            self.assertEqual({path.name for path in paths}, set(TELEMETRY.OUTPUT_SCHEMAS))
            self.assertEqual(counts["rank_trajectory.csv"], 5)
            self.assertEqual(counts["module_rank_trajectory.csv"], 20)
            self.assertEqual(counts["allocation_events.csv"], 2)
            self.assertEqual(counts["calibration_events.csv"], 3)
            self.assertEqual(counts["evaluation_trajectory.csv"], 2)

            with (output / "calibration_events.csv").open(newline="", encoding="utf-8") as handle:
                candidates = list(csv.DictReader(handle))
            selected = [row["candidate_id"] for row in candidates if row["is_selected"] == "true"]
            self.assertEqual(selected, ["c1", "zero"])

            with (output / "module_rank_trajectory.csv").open(newline="", encoding="utf-8") as handle:
                modules = list(csv.DictReader(handle))
            query = next(row for row in modules if row["module_name"] == MODULES[0])
            intermediate = next(row for row in modules if row["module_name"] == MODULES[2])
            self.assertEqual(query["module_group"], "attention")
            self.assertEqual(intermediate["module_group"], "ffn")
            self.assertEqual(query["transformer_layer"], "0")

    def test_rejects_noncanonical_schema_version(self):
        records = valid_records()
        records[0]["schema_version"] = 1
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "bad.jsonl"
            write_jsonl(source, records)
            with self.assertRaisesRegex(
                TELEMETRY.TelemetryValidationError, "schema_version must be a string"
            ):
                TELEMETRY.load_jsonl(source)

    def test_rejects_missing_allocation_rank_map(self):
        records = valid_records()
        del records[3]["module_active_ranks"]
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "bad.jsonl"
            write_jsonl(source, records)
            loaded = TELEMETRY.load_jsonl(source)
            with self.assertRaisesRegex(
                TELEMETRY.TelemetryValidationError, "module_active_ranks"
            ):
                TELEMETRY.transform_records(loaded)

    def test_rejects_rank_delta_inconsistent_with_selected_k(self):
        records = valid_records()
        records[3]["selected_event_rank"] = 1
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "bad.jsonl"
            write_jsonl(source, records)
            loaded = TELEMETRY.load_jsonl(source)
            with self.assertRaisesRegex(
                TELEMETRY.TelemetryValidationError, "selected_event_rank"
            ):
                TELEMETRY.transform_records(loaded)

    def test_writes_headers_for_inapplicable_tables(self):
        record = dict(
            common("warning", 0, 0.0),
            message="no trajectory data in this partial synthetic stream",
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "warning.jsonl"
            output = root / "parsed"
            write_jsonl(source, [record])
            TELEMETRY.parse_files([source], output)
            for filename, columns in TELEMETRY.OUTPUT_SCHEMAS.items():
                with (output / filename).open(newline="", encoding="utf-8") as handle:
                    reader = csv.reader(handle)
                    self.assertEqual(tuple(next(reader)), columns)
                    self.assertEqual(list(reader), [])

    def test_nonfinite_calibration_sentinels_are_unavailable_not_numeric(self):
        records = valid_records()
        candidate = records[1]["candidates"][0]
        candidate["fold_gains"] = ["NaN", "Infinity", "-Infinity"]
        candidate["calibration_gain_mean"] = "NaN"
        candidate["calibration_gain_std"] = "NaN"
        candidate["calibration_gain_lcb"] = "NaN"
        candidate["calibration_valid"] = False
        candidate["invalid_reason"] = "non_finite_calibration_gradient"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "invalid_candidate.jsonl"
            output = root / "parsed"
            write_jsonl(source, records)
            TELEMETRY.parse_files([source], output)
            with (output / "calibration_events.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["calibration_gain_mean"], "")
            self.assertEqual(row["calibration_gain_lcb"], "")
            self.assertEqual(row["calibration_valid"], "false")
            self.assertEqual(
                row["invalid_reason"], "non_finite_calibration_gradient"
            )
            self.assertEqual(
                json.loads(row["fold_gains"]),
                ["NaN", "Infinity", "-Infinity"],
            )

    def test_evaluation_without_accuracy_and_checkpoint_path_alias(self):
        records = valid_records()
        evaluation = records[4]
        evaluation["metrics"] = {"eval_matthews_correlation": 0.61}
        evaluation["checkpoint_path"] = "checkpoint-100"
        evaluation["state_role"] = "selected_best_checkpoint_evaluation"
        evaluation["evaluated_checkpoint"] = "checkpoint-best"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "non_accuracy_task.jsonl"
            output = root / "parsed"
            write_jsonl(source, records)
            TELEMETRY.parse_files([source], output)
            with (output / "evaluation_trajectory.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["accuracy"], "")
            self.assertEqual(row["state_role"], "selected_best_checkpoint_evaluation")
            self.assertEqual(row["checkpoint"], "checkpoint-best")

    def test_standalone_evaluation_role_is_retained(self):
        records = valid_records()
        evaluation = records[4]
        evaluation["state_role"] = "standalone_evaluation"
        evaluation["evaluated_checkpoint"] = "external-checkpoint"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "standalone.jsonl"
            output = root / "parsed"
            write_jsonl(source, records)
            TELEMETRY.parse_files([source], output)
            with (output / "evaluation_trajectory.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["state_role"], "standalone_evaluation")
            self.assertEqual(row["checkpoint"], "external-checkpoint")

    def test_appended_resume_may_reset_wall_clock_only_at_run_start(self):
        records = valid_records()
        records += [
            dict(common("run_start", 300, 0.0), resume=True),
            dict(common("warning", 300, 1.0), message="resumed segment"),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "resumed.jsonl"
            write_jsonl(source, records)
            self.assertEqual(len(TELEMETRY.load_jsonl(source)), len(records))

        records[-2] = dict(common("warning", 300, 0.0), message="illegal reset")
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "bad_reset.jsonl"
            write_jsonl(source, records)
            with self.assertRaisesRegex(
                TELEMETRY.TelemetryValidationError, "may reset only"
            ):
                TELEMETRY.load_jsonl(source)

        illegal_step_records = valid_records() + [
            dict(common("warning", 299, 31.0), message="illegal step rollback")
        ]
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "bad_step_reset.jsonl"
            write_jsonl(source, illegal_step_records)
            with self.assertRaisesRegex(
                TELEMETRY.TelemetryValidationError, "global_step may reset only"
            ):
                TELEMETRY.load_jsonl(source)

    def test_resume_segments_preserve_abandoned_tail_and_isolate_selections(self):
        initial = {name: 1 for name in MODULES}
        checkpoint = dict(initial)
        checkpoint[MODULES[0]] = 2
        abandoned_tail = dict(checkpoint)
        abandoned_tail[MODULES[1]] = 2

        def candidate(candidate_id, module):
            return {
                "candidate_id": candidate_id,
                "modules": [module],
                "k": 1,
                "parameter_cost": 10,
                "fold_scores": [0.1],
                "mean_score": 0.1,
                "std_score": 0.0,
                "lcb_score": 0.1,
                "calibration_valid": True,
            }

        candidates = [
            candidate("old-choice", MODULES[0]),
            candidate("new-choice", MODULES[2]),
        ]
        records = [
            dict(
                common("run_start", 0, 0.0),
                module_active_ranks=initial,
                total_active_rank=4,
                active_model_parameter_count=100,
            ),
            dict(
                common("calibration_event", 200, 10.0),
                selected_candidate_id="old-choice",
                candidates=candidates,
            ),
            dict(
                common("candidate_selection", 200, 11.0),
                selected_candidate_id="old-choice",
            ),
            dict(
                common("allocation_event", 200, 12.0),
                module_active_ranks=checkpoint,
                total_active_rank=5,
                pre_total_active_rank=4,
                post_total_active_rank=5,
                selected_k=1,
                selected_event_rank=1,
                selected_modules=[MODULES[0]],
                rank_increments={MODULES[0]: 1},
                active_model_parameter_count=110,
                pre_active_parameter_count=100,
                post_active_parameter_count=110,
            ),
            dict(
                common("allocation_event", 240, 20.0),
                module_active_ranks=abandoned_tail,
                total_active_rank=6,
                pre_total_active_rank=5,
                post_total_active_rank=6,
                selected_k=1,
                selected_event_rank=1,
                selected_modules=[MODULES[1]],
                rank_increments={MODULES[1]: 1},
                active_model_parameter_count=120,
                pre_active_parameter_count=110,
                post_active_parameter_count=120,
            ),
            dict(
                common("run_start", 200, 0.0),
                resumed=True,
                module_active_ranks=checkpoint,
                total_active_rank=5,
                active_model_parameter_count=110,
            ),
            dict(
                common("calibration_event", 200, 1.0),
                selected_candidate_id="new-choice",
                candidates=candidates,
            ),
            dict(
                common("candidate_selection", 200, 2.0),
                selected_candidate_id="new-choice",
            ),
            dict(common("warning", 201, 3.0), message="resumed lineage"),
        ]

        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "rollback_resume.jsonl"
            write_jsonl(source, records)
            loaded = TELEMETRY.load_jsonl(source)
            self.assertEqual(
                [record["_run_segment"] for record in loaded],
                [0, 0, 0, 0, 0, 1, 1, 1, 1],
            )
            transformed = TELEMETRY.transform_records(loaded)

        rank_rows = transformed["rank_trajectory.csv"]
        self.assertTrue(
            any(
                row["run_segment"] == 0
                and row["global_step"] == 240
                and row["total_active_rank"] == 6
                for row in rank_rows
            )
        )
        self.assertTrue(
            any(
                row["run_segment"] == 1
                and row["global_step"] == 200
                and row["state_role"] == "initial_trajectory"
                and row["total_active_rank"] == 5
                for row in rank_rows
            )
        )
        selected_by_segment = {
            segment: {
                row["candidate_id"]
                for row in transformed["calibration_events.csv"]
                if row["run_segment"] == segment and row["is_selected"] == "true"
            }
            for segment in (0, 1)
        }
        self.assertEqual(selected_by_segment[0], {"old-choice"})
        self.assertEqual(selected_by_segment[1], {"new-choice"})

    def test_incomplete_trailing_crash_record_preserves_completed_lines(self):
        records = valid_records()
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "interrupted.jsonl"
            write_jsonl(source, records)
            with source.open("a", encoding="utf-8") as stream:
                stream.write('{"schema_version":"rank_telemetry.v1"')
            with self.assertWarnsRegex(RuntimeWarning, "incomplete trailing"):
                loaded = TELEMETRY.load_jsonl(source)
            self.assertEqual(len(loaded), len(records))

    def test_incomplete_unicode_crash_tail_preserves_completed_lines(self):
        completed = json.dumps(common("warning", 0, 0.0)).encode("utf-8") + b"\n"
        partial = (
            b'{"schema_version":"rank_telemetry.v1","event_type":"warning",'
            b'"global_step":1,"seed":41,"method":"C-GEAR",'
            b'"wall_time_seconds":1.0,"message":"' + b"\xe2\x82"
        )
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "unicode_interrupted.jsonl"
            source.write_bytes(completed + partial)
            with self.assertWarnsRegex(RuntimeWarning, "incomplete trailing"):
                loaded = TELEMETRY.load_jsonl(source)
            self.assertEqual(len(loaded), 1)

    def test_valid_final_json_without_newline_is_accepted(self):
        record = dict(common("warning", 0, 0.0), message="complete")
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "no_final_newline.jsonl"
            source.write_text(json.dumps(record), encoding="utf-8")
            self.assertEqual(len(TELEMETRY.load_jsonl(source)), 1)

    def test_malformed_newline_terminated_final_record_is_rejected(self):
        record = dict(common("warning", 0, 0.0), message="complete")
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "malformed_final.jsonl"
            source.write_text(json.dumps(record) + "\n{bad json\n", encoding="utf-8")
            with self.assertRaisesRegex(
                TELEMETRY.TelemetryValidationError, "invalid JSON"
            ):
                TELEMETRY.load_jsonl(source)

    def test_raw_nonfinite_json_constant_is_rejected(self):
        raw = (
            '{"schema_version":"rank_telemetry.v1","event_type":"warning",'
            '"global_step":0,"seed":41,"method":"C-GEAR",'
            '"wall_time_seconds":0.0,"diagnostic":NaN}\n'
        )
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "raw_nan.jsonl"
            source.write_text(raw, encoding="utf-8")
            with self.assertRaisesRegex(
                TELEMETRY.TelemetryValidationError, "non-finite JSON constant"
            ):
                TELEMETRY.load_jsonl(source)

    def test_run_end_keeps_selected_checkpoint_and_final_trajectory_distinct(self):
        records = valid_records()
        initial = {name: 1 for name in MODULES}
        final = dict(initial)
        final[MODULES[0]] = 2
        final[MODULES[2]] = 2
        records[-1].update(
            module_active_ranks=initial,
            total_active_rank=4,
            active_model_parameter_count=100,
            physical_rank_component_count=9,
            selected_active_rank=4,
            selected_active_parameter_count=100,
            selected_physical_rank_component_count=9,
            selected_module_active_ranks=initial,
            final_active_rank=6,
            final_active_parameter_count=120,
            final_module_active_ranks=final,
        )
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "best_vs_final.jsonl"
            write_jsonl(source, records)
            transformed = TELEMETRY.transform_records(TELEMETRY.load_jsonl(source))
            terminal_rank_rows = {
                row["state_role"]: row
                for row in transformed["rank_trajectory.csv"]
                if row["event_type"] == "run_end"
            }
            self.assertEqual(
                set(terminal_rank_rows),
                {"selected_best_checkpoint", "final_trajectory"},
            )
            selected_row = terminal_rank_rows["selected_best_checkpoint"]
            final_row = terminal_rank_rows["final_trajectory"]
            self.assertEqual(selected_row["total_active_rank"], 4)
            self.assertEqual(selected_row["active_model_parameter_count"], 100)
            self.assertEqual(final_row["total_active_rank"], 6)
            self.assertEqual(final_row["active_model_parameter_count"], 120)
            self.assertEqual(final_row["physical_rank_component_count"], "")
            self.assertEqual(
                selected_row["physical_rank_component_count"], 9
            )
            terminal_modules = transformed["module_rank_trajectory.csv"]
            selected_modules = {
                row["module_name"]: row["active_rank"]
                for row in terminal_modules
                if row["global_step"] == 300
                and row["state_role"] == "selected_best_checkpoint"
            }
            final_modules = {
                row["module_name"]: row["active_rank"]
                for row in terminal_modules
                if row["global_step"] == 300
                and row["state_role"] == "final_trajectory"
            }
            self.assertEqual(selected_modules, initial)
            self.assertEqual(final_modules, final)

    def test_partial_final_state_never_inherits_selected_checkpoint_fields(self):
        records = valid_records()
        selected = {name: 1 for name in MODULES}
        records[-1].update(
            module_active_ranks=selected,
            total_active_rank=4,
            active_model_parameter_count=100,
            runtime_trainable_parameter_count=110,
            full_model_parameter_count=1000,
            physical_rank_component_count=9,
            selected_active_rank=4,
            selected_active_parameter_count=100,
            selected_runtime_trainable_parameter_count=110,
            selected_full_model_parameter_count=1000,
            selected_physical_rank_component_count=9,
            selected_module_active_ranks=selected,
            final_active_rank=6,
        )
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "partial_final.jsonl"
            write_jsonl(source, records)
            transformed = TELEMETRY.transform_records(TELEMETRY.load_jsonl(source))
            terminal = {
                row["state_role"]: row
                for row in transformed["rank_trajectory.csv"]
                if row["event_type"] == "run_end"
            }
            self.assertEqual(terminal["final_trajectory"]["total_active_rank"], 6)
            self.assertEqual(
                terminal["final_trajectory"]["active_model_parameter_count"], ""
            )
            self.assertEqual(
                terminal["final_trajectory"]["runtime_trainable_parameter_count"],
                "",
            )
            self.assertEqual(
                terminal["final_trajectory"]["physical_rank_component_count"], ""
            )
            final_modules = [
                row
                for row in transformed["module_rank_trajectory.csv"]
                if row["event_type"] == "run_end"
                and row["state_role"] == "final_trajectory"
            ]
            self.assertEqual(final_modules, [])


if __name__ == "__main__":
    unittest.main()
