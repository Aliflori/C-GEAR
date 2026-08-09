import inspect
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
from torch import nn

from loralib import (
    RankAllocator,
    SVDLinear,
    build_trainable_rank_pattern,
    get_active_model_parameter_count,
    get_current_active_trainable_cost,
    get_current_dynamic_trainable_cost,
    get_full_model_parameter_count,
    get_module_rank_one_cost,
    get_rank_pattern_active_model_parameter_count,
    get_rank_pattern_trainable_cost,
    get_runtime_trainable_parameter_count,
)
from transformers import (
    PreTrainedModel,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    budgeted_evo_allocator as budgeted,
)
from transformers import evo_allocator
from transformers.modeling_utils import (
    DYNAMIC_LORA_NON_DYNAMIC_TRAINABILITY,
    is_dynamic_lora_parameter_name,
)


class TinyDynamicModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.small = SVDLinear(3, 2, r=1)
        self.large = SVDLinear(4, 5, r=1)
        self.config = SimpleNamespace()


class TinyDynamicConfig(PretrainedConfig):
    model_type = "tiny-dynamic-lora-test"

    def __init__(self, hidden_size=4, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size


class TinyDynamicPretrainedModel(PreTrainedModel):
    config_class = TinyDynamicConfig
    base_model_prefix = "backbone"

    def __init__(self, config):
        super().__init__(config)
        self.backbone = nn.Module()
        self.backbone.frozen_dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.backbone.adapter = SVDLinear(config.hidden_size, config.hidden_size, r=1)
        self.classifier = nn.Linear(config.hidden_size, 2)

    def forward(self, input_ids=None, **kwargs):
        hidden = torch.zeros(1, self.config.hidden_size)
        return self.classifier(self.backbone.adapter(self.backbone.frozen_dense(hidden)))


def freeze_tiny_backbone(model):
    for name, parameter in model.named_parameters():
        if name.startswith("backbone") and not is_dynamic_lora_parameter_name(name):
            parameter.requires_grad_(False)


def make_reference(model, ranks):
    return {
        "format_version": 2,
        "non_dynamic_trainable_params": sum(
            parameter.numel()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and "lora_" not in name
        ),
        "modules": {
            name: {
                "active_rank": ranks[name],
                "in_features": module.in_features,
                "out_features": module.out_features,
                "rank_one_cost": get_module_rank_one_cost(module),
            }
            for name, module in model.named_modules()
            if isinstance(module, SVDLinear)
        },
    }


class ExactCostAndReferenceTest(unittest.TestCase):
    def test_mixed_dimensions_exact_cost_and_pattern_count(self):
        model = TinyDynamicModel()
        self.assertEqual(get_module_rank_one_cost(model.small), 3 + 2 + 1)
        self.assertEqual(get_module_rank_one_cost(model.large), 4 + 5 + 1)

        pattern = build_trainable_rank_pattern(model)
        direct = get_runtime_trainable_parameter_count(model)
        derived = get_rank_pattern_trainable_cost(model, pattern)
        self.assertEqual(direct, derived)
        self.assertEqual(direct, get_active_model_parameter_count(model))
        self.assertGreater(get_full_model_parameter_count(model), direct)

    def test_reference_pattern_ratio_and_strict_reduction(self):
        model = TinyDynamicModel()
        reference = {"small": 1, "large": 2}
        reference_cost = get_rank_pattern_trainable_cost(model, reference)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rank_pattern.json"
            path.write_text(json.dumps(reference), encoding="utf-8")
            allocator = RankAllocator(
                model,
                lora_r=1,
                target_rank=1,
                target_total_rank=3,
                init_warmup=0,
                incre_interval=1,
                top_h=1,
                advance_learn=True,
                beta1=0.85,
                beta2=0.85,
                incre_rank_num=1,
                rank_allocator="genetic_budgeted",
                ga_budget_reference_pattern=str(path),
                ga_budget_ratio=0.95,
            )
        self.assertEqual(allocator.reference_greedy_cost, reference_cost)
        self.assertEqual(allocator.target_final_trainable_params, math.floor(reference_cost * 0.95))
        self.assertLess(allocator.target_final_trainable_params, reference_cost)

    def test_incompatible_dimensions_fail_clearly(self):
        model = TinyDynamicModel()
        reference = make_reference(model, {"small": 1, "large": 2})
        reference["modules"]["large"]["out_features"] = 99
        with self.assertRaisesRegex(ValueError, "dimensions"):
            get_rank_pattern_trainable_cost(model, reference)

    def test_mathematically_infeasible_budget_fails_before_training(self):
        model = TinyDynamicModel()
        with self.assertRaisesRegex(ValueError, "theoretical_minimum_achievable_cost"):
            RankAllocator(
                model,
                lora_r=1,
                target_rank=1,
                target_total_rank=4,
                init_warmup=0,
                incre_interval=1,
                top_h=1,
                advance_learn=True,
                beta1=0.85,
                beta2=0.85,
                incre_rank_num=1,
                rank_allocator="genetic_budgeted",
                ga_max_final_trainable_params=get_current_active_trainable_cost(model),
            )


class FrozenCompatibilityTest(unittest.TestCase):
    def test_greedy_keeps_threshold_ties_and_never_enters_budgeted_path(self):
        class GreedyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.first = SVDLinear(2, 2, r=1)
                self.second = SVDLinear(2, 2, r=1)
                self.third = SVDLinear(2, 2, r=1)

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
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad], lr=1e-3
        )
        for module in (model.first, model.second, model.third):
            module.score = torch.tensor(0.0)
        allocator.update_and_increase(model, 0, optimizer)
        fixed_scores = {"first": 1.0, "second": 2.0, "third": 2.0}
        allocator.calculate_score = lambda name, layer, metric="ipt": fixed_scores[name]

        with mock.patch.object(
            budgeted, "select_modules_budgeted", side_effect=AssertionError("budget path entered")
        ):
            allocator.increase_to_target_rank(model, optimizer)

        self.assertEqual(int(model.first.ranknum.item()), 1)
        self.assertEqual(int(model.second.ranknum.item()), 2)
        self.assertEqual(int(model.third.ranknum.item()), 2)

    def test_old_ga_only_and_memetic_modes_need_no_budget_reference(self):
        for local_search in (False, True):
            allocator = RankAllocator(
                TinyDynamicModel(),
                lora_r=1,
                target_rank=1,
                target_total_rank=3,
                init_warmup=0,
                incre_interval=1,
                top_h=1,
                advance_learn=True,
                beta1=0.85,
                beta2=0.85,
                incre_rank_num=1,
                rank_allocator="genetic",
                ga_local_search=local_search,
            )
            self.assertEqual(allocator.rank_allocator, "genetic")
            self.assertEqual(allocator.ga_local_search, local_search)

    def test_budgeted_mode_rejects_local_search(self):
        with self.assertRaisesRegex(ValueError, "forbids local search"):
            RankAllocator(
                TinyDynamicModel(),
                lora_r=1,
                target_rank=1,
                target_total_rank=3,
                init_warmup=0,
                incre_interval=1,
                top_h=1,
                advance_learn=True,
                beta1=0.85,
                beta2=0.85,
                incre_rank_num=1,
                rank_allocator="genetic_budgeted",
                ga_max_final_trainable_params=100,
                ga_local_search=True,
            )


class TrainingAwareSignalTest(unittest.TestCase):
    def test_adam_second_moment_and_gradient_fallback(self):
        module = SVDLinear(3, 2, r=1)
        module.add_reserve_param(1, True)
        a = module.lora_A[-1]
        b = module.lora_B[-1]
        a.grad = torch.ones_like(a)
        b.grad = torch.full_like(b, 2.0)
        optimizer = SimpleNamespace(defaults={"eps": 1e-8}, state={})
        optimizer.state[a] = {"exp_avg_sq": torch.full_like(a, 4.0)}

        gain = budgeted.expected_optimizer_gain(module, optimizer)
        self.assertAlmostEqual(gain["raw_gain"], 3 * 0.5 + 2 * 4.0, places=6)
        self.assertEqual(gain["optimizer_moment_elements"], 3)
        self.assertEqual(gain["fallback_gradient_elements"], 2)

    def test_selector_api_has_no_validation_or_evaluation_inputs(self):
        parameter_names = set(inspect.signature(budgeted.select_modules_budgeted).parameters)
        self.assertFalse(any("validation" in name or "evaluation" in name for name in parameter_names))


class BudgetedEvolutionTest(unittest.TestCase):
    def setUp(self):
        self.names = ["encoder.layer.%d.proj" % index for index in range(8)]
        self.scores = list(zip(self.names, [8, 7, 6, 5, 4, 3, 2, 1]))
        self.costs = dict(zip(self.names, [12, 11, 10, 9, 8, 7, 6, 5]))
        self.gains = dict(zip(self.names, [10, 9, 8, 7, 7, 6, 5, 4]))

    def allocate(self, **overrides):
        arguments = dict(
            scores=self.scores,
            costs=self.costs,
            training_gains=self.gains,
            top_h=2,
            current_active_cost=40,
            target_final_cost=75,
            future_rank_increments=2,
            population_size=10,
            generations=3,
            seed=41,
        )
        arguments.update(overrides)
        return budgeted.select_modules_budgeted(**arguments)

    def test_determinism_uniqueness_and_no_local_search(self):
        first, first_diagnostics = self.allocate()
        second, second_diagnostics = self.allocate()
        self.assertEqual(first, second)
        self.assertEqual(first_diagnostics["shortlist"], second_diagnostics["shortlist"])
        chromosomes = [tuple(item["chromosome"]) for item in first_diagnostics["shortlist"]]
        self.assertEqual(len(chromosomes), len(set(chromosomes)))
        self.assertFalse(first_diagnostics["local_search_enabled"])
        self.assertTrue(first_diagnostics["budget_constraint_satisfied"])

    def test_every_selection_and_simulated_final_schedule_obeys_budget(self):
        current = 40
        remaining = 4
        for event_size in (2, 2):
            selected, diagnostics = self.allocate(
                current_active_cost=current,
                future_rank_increments=remaining - event_size,
                top_h=event_size,
                seed=41 + current,
            )
            chosen_cost = sum(self.costs[name] for name in selected)
            current += chosen_cost
            remaining -= event_size
            self.assertLessEqual(
                diagnostics["selected_candidate"]["projected_minimum_final_cost"], 75
            )
        self.assertEqual(remaining, 0)
        self.assertLessEqual(current, 75)

    def test_infeasible_event_reports_projected_minimum(self):
        with self.assertRaisesRegex(ValueError, "projected_minimum_final_cost"):
            self.allocate(target_final_cost=55)

    def test_budgeted_path_does_not_call_old_ga_or_local_refinement(self):
        with mock.patch.object(
            evo_allocator, "select_modules_genetic", side_effect=AssertionError("old/local path called")
        ):
            _, diagnostics = self.allocate()
        self.assertFalse(diagnostics["local_search_enabled"])

    def test_quality_guard_allows_material_gain_to_beat_cheapest(self):
        candidates = [
            {
                "actual_cost": 10,
                "raw_training_gain": 5.0,
                "gain_per_parameter": 0.5,
                "structural_fitness": 0.9,
                "canonical_modules": ("cheap",),
            },
            {
                "actual_cost": 12,
                "raw_training_gain": 12.0,
                "gain_per_parameter": 1.0,
                "structural_fitness": 0.8,
                "canonical_modules": ("quality",),
            },
        ]
        selected, _ = budgeted.select_quality_preserving_candidate(candidates, 0.05)
        self.assertEqual(selected["canonical_modules"], ("quality",))

    def test_cost_preference_within_normalized_gain_tolerance(self):
        candidates = [
            {
                "actual_cost": 10,
                "raw_training_gain": 9.5,
                "gain_per_parameter": 0.95,
                "structural_fitness": 0.7,
                "canonical_modules": ("cheap",),
            },
            {
                "actual_cost": 12,
                "raw_training_gain": 12.0,
                "gain_per_parameter": 1.0,
                "structural_fitness": 0.9,
                "canonical_modules": ("expensive",),
            },
            {
                "actual_cost": 11,
                "raw_training_gain": 5.5,
                "gain_per_parameter": 0.5,
                "structural_fitness": 1.0,
                "canonical_modules": ("anchor",),
            },
        ]
        selected, quality_set = budgeted.select_quality_preserving_candidate(candidates, 0.11)
        self.assertEqual(len(quality_set), 2)
        self.assertEqual(selected["canonical_modules"], ("cheap",))


class CheckpointAndFinalVerificationTest(unittest.TestCase):
    def test_load_best_model_restores_frozen_backbone_and_all_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            model = TinyDynamicPretrainedModel(TinyDynamicConfig())
            freeze_tiny_backbone(model)
            model.backbone.adapter.add_reserve_param(1, True)
            checkpoint_active = get_active_model_parameter_count(model)
            checkpoint_runtime = get_runtime_trainable_parameter_count(model)
            checkpoint_full = get_full_model_parameter_count(model)

            arguments = TrainingArguments(
                output_dir=str(Path(directory) / "trainer-output"),
                no_cuda=True,
                report_to=[],
            )
            trainer = Trainer(model=model, args=arguments)
            checkpoint = Path(directory) / "checkpoint-best"
            trainer.save_model(str(checkpoint))
            saved_config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
            self.assertIn(DYNAMIC_LORA_NON_DYNAMIC_TRAINABILITY, saved_config)

            # Represent a later trajectory state before load_best_model_at_end.
            model.backbone.adapter.lora_E[-1].requires_grad_(True)
            model.backbone.adapter.ranknum.data.add_(1)
            model.backbone.adapter.add_reserve_param(1, True)
            trainer.state.best_model_checkpoint = str(checkpoint)
            trainer.state.best_metric = 1.0
            trainer._load_best_model()

            loaded = trainer.model
            self.assertFalse(loaded.backbone.frozen_dense.weight.requires_grad)
            self.assertFalse(loaded.backbone.frozen_dense.bias.requires_grad)
            self.assertFalse(loaded.backbone.adapter.weight.requires_grad)
            self.assertFalse(loaded.backbone.adapter.bias.requires_grad)
            self.assertTrue(loaded.classifier.weight.requires_grad)
            self.assertEqual(get_active_model_parameter_count(loaded), checkpoint_active)
            self.assertEqual(get_runtime_trainable_parameter_count(loaded), checkpoint_runtime)
            self.assertEqual(get_full_model_parameter_count(loaded), checkpoint_full)
            self.assertLess(
                get_runtime_trainable_parameter_count(loaded),
                get_full_model_parameter_count(loaded),
            )

    def test_budget_and_dynamic_metadata_round_trip(self):
        model = TinyDynamicModel()
        model.small.add_reserve_param(1, True)
        model.small.lora_E[-1].requires_grad_(True)
        model.small.ranknum.data.add_(1)
        dynamic = model.small.get_dynamic_lora_metadata()
        budget = {
            "allocator_mode": "genetic_budgeted",
            "reference_cost": 100,
            "target_cost": 98,
            "budget_ratio": 0.98,
            "current_accumulated_cost": 80,
        }
        payload = json.loads(json.dumps({"dynamic": dynamic, "budget": budget}))

        restored = SVDLinear(3, 2, r=1)
        restored.set_dynamic_lora_metadata(payload["dynamic"])
        self.assertEqual(restored.get_dynamic_lora_metadata(), dynamic)
        self.assertTrue(budgeted.validate_budget_metadata(budget, payload["budget"]))
        incompatible = dict(payload["budget"], target_cost=97)
        with self.assertRaisesRegex(ValueError, "Incompatible"):
            budgeted.validate_budget_metadata(budget, incompatible)

    def test_active_pattern_count_is_independent_of_runtime_reserves(self):
        model = TinyDynamicModel()
        for module in (model.small, model.large):
            module.add_reserve_param(1, True)
        model.large.lora_E[-1].requires_grad_(True)
        model.large.ranknum.data.add_(1)
        model.large.add_reserve_param(1, True)
        self.assertGreater(
            get_current_dynamic_trainable_cost(model), get_current_active_trainable_cost(model)
        )

        pattern = build_trainable_rank_pattern(model)
        self.assertEqual(
            get_active_model_parameter_count(model),
            get_rank_pattern_active_model_parameter_count(model, pattern),
        )
        self.assertGreater(
            get_runtime_trainable_parameter_count(model),
            pattern["active_model_parameter_count"],
        )
        self.assertEqual(
            pattern["runtime_trainable_parameter_count"],
            get_runtime_trainable_parameter_count(model),
        )

    def test_allocator_finalization_enforces_target_and_writes_versioned_pattern(self):
        model = TinyDynamicModel()
        allocator = RankAllocator(
            model,
            lora_r=1,
            target_rank=1,
            target_total_rank=3,
            init_warmup=0,
            incre_interval=1,
            top_h=1,
            advance_learn=True,
            beta1=0.85,
            beta2=0.85,
            incre_rank_num=1,
            rank_allocator="genetic_budgeted",
            ga_max_final_trainable_params=40,
        )
        model.large.add_reserve_param(1, True)
        model.large.lora_E[-1].requires_grad_(True)
        model.large.ranknum.data.add_(1)
        model.large.add_reserve_param(1, True)
        allocator.final_trajectory_metrics = {
            "total_active_rank": 4,
            "active_model_parameter_count": 39,
            "runtime_trainable_parameter_count": 55,
            "full_model_parameter_count": 100,
        }

        pattern = allocator.finalize_budget(model)
        self.assertEqual(pattern["format_version"], 2)
        self.assertEqual(pattern["allocator_mode"], "genetic_budgeted")
        self.assertLessEqual(pattern["active_model_parameter_count"], 40)
        self.assertEqual(
            pattern["active_model_parameter_count"],
            get_active_model_parameter_count(model),
        )
        self.assertGreater(
            pattern["runtime_trainable_parameter_count"],
            pattern["active_model_parameter_count"],
        )
        self.assertEqual(pattern["budget"]["final_trajectory"]["total_active_rank"], 4)
        self.assertEqual(
            pattern["budget"]["selected_best_checkpoint"]["total_active_rank"], 3
        )


if __name__ == "__main__":
    unittest.main()
