#!/usr/bin/env python3
"""Lightweight semantic tests for rank-telemetry plot inputs."""

import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "plot_rank_telemetry.py"
SPEC = importlib.util.spec_from_file_location("plot_rank_telemetry", SCRIPT)
PLOTS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PLOTS)


def module_row(step, rank, role, event_type, source="/runs/a/telemetry.jsonl"):
    return {
        "source_artifact": source,
        "method": "genetic_budgeted_calibrated",
        "seed": "41",
        "run_segment": "0",
        "global_step": str(step),
        "event_type": event_type,
        "state_role": role,
        "module_name": "deberta.encoder.layer.0.attention.self.query_proj",
        "transformer_layer": "0",
        "module_group": "attention",
        "active_rank": str(rank),
    }


class RankTelemetryPlotInputTest(unittest.TestCase):
    def test_final_trajectory_wins_over_selected_checkpoint_at_same_step(self):
        rows = [
            module_row(0, 1, "initial_trajectory", "run_start"),
            module_row(300, 4, "selected_best_checkpoint", "run_end"),
            module_row(300, 6, "final_trajectory", "run_end"),
        ]
        final, completed = PLOTS._final_module_rows(rows)
        key = next(iter(final))
        module = next(iter(final[key].values()))
        self.assertEqual(module["active_rank"], "6")
        self.assertIn(key, completed)
        self.assertFalse(PLOTS._is_trajectory_state(rows[1]))
        self.assertTrue(PLOTS._is_trajectory_state(rows[2]))

    def test_interrupted_run_uses_latest_observed_snapshot(self):
        rows = [
            module_row(0, 1, "initial_trajectory", "run_start"),
            module_row(100, 2, "trajectory", "allocation_event"),
        ]
        final, completed = PLOTS._final_module_rows(rows)
        key = next(iter(final))
        module = next(iter(final[key].values()))
        self.assertEqual(module["active_rank"], "2")
        self.assertNotIn(key, completed)

    def test_same_method_and_seed_have_unique_per_run_stems(self):
        first = ("/runs/a/telemetry.jsonl", "greedy", 41, 0)
        second = ("/runs/b/telemetry.jsonl", "greedy", 41, 0)
        self.assertNotEqual(PLOTS._run_slug(first), PLOTS._run_slug(second))

    def test_segments_and_selected_evaluation_are_separate_lineages(self):
        first = module_row(240, 3, "trajectory", "allocation_event")
        resumed = module_row(200, 2, "initial_trajectory", "run_start")
        resumed["run_segment"] = "1"
        self.assertNotEqual(PLOTS._run_key(first), PLOTS._run_key(resumed))
        self.assertNotEqual(
            PLOTS._run_slug(PLOTS._run_key(first)),
            PLOTS._run_slug(PLOTS._run_key(resumed)),
        )
        self.assertTrue(
            PLOTS._is_training_evaluation(
                {"state_role": "training_trajectory_evaluation"}
            )
        )
        self.assertFalse(
            PLOTS._is_training_evaluation(
                {"state_role": "selected_best_checkpoint_evaluation"}
            )
        )


if __name__ == "__main__":
    unittest.main()
