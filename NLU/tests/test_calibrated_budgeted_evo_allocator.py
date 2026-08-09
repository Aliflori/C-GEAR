import copy
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

from loralib import (
    RankAllocator,
    SVDLinear,
    get_active_model_parameter_count,
)
from transformers import (
    Trainer,
    TrainingArguments,
    budgeted_evo_allocator,
    calibrated_budgeted_evo_allocator as calibrated,
    calibrated_rank_calibration as virtual,
    evo_allocator,
)
from transformers.rank_budget_reporting import build_rank_budget_report
from transformers.optimization import AdamW

from NLU.tests.test_budgeted_evo_allocator import TinyDynamicModel


def assert_nested_equal(testcase, left, right, path="root"):
    testcase.assertEqual(type(left), type(right), path)
    if isinstance(left, torch.Tensor):
        testcase.assertTrue(torch.equal(left, right), path)
    elif isinstance(left, np.ndarray):
        testcase.assertTrue(np.array_equal(left, right), path)
    elif isinstance(left, dict):
        testcase.assertEqual(set(left), set(right), path)
        for key in left:
            assert_nested_equal(testcase, left[key], right[key], "%s.%s" % (path, key))
    elif isinstance(left, (list, tuple)):
        testcase.assertEqual(len(left), len(right), path)
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            assert_nested_equal(
                testcase, left_item, right_item, "%s[%s]" % (path, index)
            )
    else:
        testcase.assertEqual(left, right, path)


def assert_rng_equal(testcase, left, right):
    testcase.assertEqual(left.python_state, right.python_state)
    assert_nested_equal(testcase, left.numpy_state, right.numpy_state, "numpy_rng")
    testcase.assertTrue(torch.equal(left.torch_cpu_state, right.torch_cpu_state))
    if left.torch_cuda_states is None or right.torch_cuda_states is None:
        testcase.assertIsNone(left.torch_cuda_states)
        testcase.assertIsNone(right.torch_cuda_states)
    else:
        testcase.assertEqual(len(left.torch_cuda_states), len(right.torch_cuda_states))
        for before, after in zip(left.torch_cuda_states, right.torch_cuda_states):
            testcase.assertTrue(torch.equal(before, after))


def calibrated_allocator(model, **overrides):
    arguments = dict(
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
    arguments.update(overrides)
    return RankAllocator(model, **arguments)


def search_fixture():
    names = ["encoder.layer.%s.query_proj" % index for index in range(8)]
    return {
        "names": names,
        "scores": list(zip(names, [8, 7, 6, 5, 4, 3, 2, 1])),
        "costs": dict(zip(names, [12, 11, 10, 9, 8, 7, 6, 5])),
        "gains": dict(zip(names, [10, 9, 8, 7, 7, 6, 5, 4])),
    }


def generate_candidates(**overrides):
    fixture = search_fixture()
    arguments = dict(
        scores=fixture["scores"],
        costs=fixture["costs"],
        training_gains=fixture["gains"],
        top_h=3,
        current_active_cost=40,
        target_final_cost=100,
        min_event_rank=1,
        max_event_rank=3,
        max_greedy_replacements=2,
        population_size=10,
        generations=2,
        seed=41,
    )
    arguments.update(overrides)
    return calibrated.generate_calibrated_candidates(**arguments)


def quality_candidate(
    name,
    lcb,
    cost,
    projected,
    family,
    reference=False,
    chromosome_size=1,
    chromosome_index=0,
    target=300,
):
    return {
        "chromosome": () if chromosome_size == 0 else (chromosome_index,),
        "modules": [] if chromosome_size == 0 else [name],
        "canonical_modules": () if chromosome_size == 0 else (name,),
        "chromosome_size": chromosome_size,
        "actual_cost": cost,
        "projected_final_active_parameter_count": projected,
        "target_final_active_parameter_count": target,
        "budget_feasible": projected <= target,
        "candidate_families": [family],
        "greedy_quality_reference": reference,
        "hamming_distance_from_greedy": 0 if reference else 1,
        "structural_fitness": 1.0,
        "calibration_gain_mean": lcb,
        "calibration_gain_std": 0.0,
        "calibration_gain_lcb": lcb,
    }


def zero_candidate(target=300):
    return quality_candidate(
        "zero",
        0.0,
        0,
        100,
        "zero_rank",
        chromosome_size=0,
        target=target,
    )


class TinyVirtualModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(0.2)
        self.adapter = SVDLinear(2, 2, r=1)
        self.adapter.add_reserve_param(1, True)

    def forward(self, features):
        return self.adapter(self.dropout(features))


def virtual_loss(model, batch):
    prediction = model(batch["features"])
    return (prediction - batch["targets"]).pow(2).mean()


def virtual_fixture():
    torch.manual_seed(7)
    model = TinyVirtualModel()
    optimizer = AdamW(model.parameters(), lr=0.03, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: 1.0 / float(step + 1)
    )
    initialization_batch = {
        "features": torch.tensor([[1.0, -0.5], [0.25, 0.75]]),
        "targets": torch.tensor([[0.2, -0.1], [0.0, 0.3]]),
    }
    optimizer.zero_grad()
    virtual_loss(model, initialization_batch).backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
    virtual_loss(model, initialization_batch).backward()
    pairs = [
        (
            {
                "features": torch.tensor([[0.4, -0.2], [0.1, 0.6]]),
                "targets": torch.tensor([[0.0, 0.1], [0.2, -0.2]]),
            },
            {
                "features": torch.tensor([[0.3, 0.8], [-0.7, 0.2]]),
                "targets": torch.tensor([[0.2, 0.0], [-0.1, 0.4]]),
            },
        ),
        (
            {
                "features": torch.tensor([[0.9, -0.4], [0.2, 0.3]]),
                "targets": torch.tensor([[0.1, -0.3], [0.0, 0.2]]),
            },
            {
                "features": torch.tensor([[-0.1, 0.5], [0.7, -0.6]]),
                "targets": torch.tensor([[0.3, 0.1], [-0.2, 0.0]]),
            },
        ),
    ]
    return model, optimizer, scheduler, pairs


class TrackingDataset(torch.utils.data.Dataset):
    _fingerprint = "training-only-fixture"

    def __init__(self, size):
        self.size = size
        self.accessed = []

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        self.accessed.append(index)
        return {"value": float(index)}


class EvaluationSentinel(torch.utils.data.Dataset):
    def __init__(self):
        self.accessed = 0

    def __len__(self):
        return 5

    def __getitem__(self, index):
        self.accessed += 1
        raise AssertionError("calibration accessed evaluation data")


class LegacyAllocatorRegressionTest(unittest.TestCase):
    def test_a_greedy_tie_behavior_and_dispatch_are_unchanged(self):
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
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        for module in (model.first, model.second, model.third):
            module.score = torch.tensor(0.0)
        allocator.update_and_increase(model, 0, optimizer)
        fixed_scores = {"first": 1.0, "second": 2.0, "third": 2.0}
        allocator.calculate_score = lambda name, layer, metric="ipt": fixed_scores[name]

        with mock.patch.object(
            RankAllocator,
            "_increase_calibrated_rank",
            side_effect=AssertionError("calibrated path entered Greedy"),
        ):
            allocator.increase_to_target_rank(model, optimizer)

        self.assertEqual(int(model.first.ranknum.item()), 1)
        self.assertEqual(int(model.second.ranknum.item()), 2)
        self.assertEqual(int(model.third.ranknum.item()), 2)

    def test_b_existing_genetic_and_budgeted_outputs_are_unchanged(self):
        identifiers = ["encoder.layer.%s.query_proj" % index for index in range(12)]
        scores = [1.00, 0.98, 0.96, 0.94, 0.92, 0.90, 0.88, 0.86, 0.60, 0.55, 0.50, 0.45]
        phase_a = [1, 3, 1, 3, 1, 3, 1, 3]
        phase_b = [3, 1, 3, 1, 3, 1, 3, 1]
        zigzag = [1, 2, 3, 2, 1, 2, 3, 2]
        reverse = [3, 2, 1, 2, 3, 2, 1, 2]
        features = {}
        for index, identifier in enumerate(identifiers):
            history = phase_a if index < 4 else phase_b if index < 8 else zigzag if index % 2 == 0 else reverse
            features[identifier] = [value + index * 0.01 for value in history]
        selected_genetic, genetic_diagnostics = evo_allocator.select_modules_genetic(
            scores=list(zip(identifiers, scores)),
            costs={identifier: 100 for identifier in identifiers},
            top_h=4,
            population_size=12,
            generations=4,
            mutation_rate=0.10,
            crossover_rate=0.80,
            interaction_weight=0.20,
            redundancy_weight=0.20,
            cost_weight=0.30,
            diversity_weight=0.10,
            seed=4,
            module_features=features,
            local_search=False,
        )
        self.assertEqual(
            selected_genetic,
            [identifiers[0], identifiers[1], identifiers[4], identifiers[5]],
        )
        self.assertEqual(genetic_diagnostics["selected_source"], "ga_only")

        fixture = search_fixture()
        selected_budgeted, budgeted_diagnostics = budgeted_evo_allocator.select_modules_budgeted(
            scores=fixture["scores"],
            costs=fixture["costs"],
            training_gains=fixture["gains"],
            top_h=2,
            current_active_cost=40,
            target_final_cost=75,
            future_rank_increments=2,
            population_size=10,
            generations=3,
            seed=41,
        )
        self.assertEqual(selected_budgeted, [fixture["names"][4], fixture["names"][5]])
        self.assertEqual(
            budgeted_diagnostics["selected_source"], "budgeted_ga_low_cost_winner"
        )


class CalibrationIsolationAndRestorationTest(unittest.TestCase):
    def test_c_calibration_batches_use_training_dataset_only(self):
        with tempfile.TemporaryDirectory() as directory:
            model = TinyDynamicModel()
            allocator = calibrated_allocator(
                model,
                ga_calibration_batches=2,
                ga_calibration_batch_size=2,
            )
            training = TrackingDataset(20)
            evaluation = EvaluationSentinel()
            arguments = TrainingArguments(
                output_dir=directory,
                no_cuda=True,
                report_to=[],
                per_device_train_batch_size=2,
            )
            trainer = Trainer(
                model=model,
                args=arguments,
                train_dataset=training,
                eval_dataset=evaluation,
                data_collator=lambda examples: {
                    "value": torch.tensor([example["value"] for example in examples])
                },
                rankallocator=allocator,
            )
            pairs = trainer._prepare_rank_allocator_calibration_batches()

            expected_indices = [
                index
                for pair in allocator.calibration_training_indices
                for side in ("batch_a", "batch_b")
                for index in pair[side]
            ]
            self.assertEqual(training.accessed, expected_indices)
            self.assertEqual(evaluation.accessed, 0)
            self.assertEqual(len(pairs), 2)
            self.assertEqual(len(set(expected_indices)), len(expected_indices))
            first_indices = copy.deepcopy(allocator.calibration_training_indices)
            allocator.get_or_create_calibration_indices(20, training._fingerprint)
            self.assertEqual(allocator.calibration_training_indices, first_indices)

    def test_d_virtual_update_restores_every_mutable_state_bitwise(self):
        model, optimizer, scheduler, pairs = virtual_fixture()
        model_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
        gradients = {
            name: None if parameter.grad is None else parameter.grad.detach().clone()
            for name, parameter in model.named_parameters()
        }
        flags = {name: parameter.requires_grad for name, parameter in model.named_parameters()}
        ranks = model.adapter.get_dynamic_lora_metadata()
        optimizer_state = copy.deepcopy(optimizer.state_dict())
        scheduler_state = copy.deepcopy(scheduler.state_dict())
        modes = [module.training for module in model.modules()]
        rng_state = virtual.capture_rng_state()

        result = virtual.score_virtual_candidate(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            candidate_module_names=["adapter"],
            module_map={"adapter": model.adapter},
            rank_increment=1,
            calibration_batch_pairs=pairs,
            loss_fn=virtual_loss,
            beta=0.5,
            max_grad_norm=1.0,
        )

        self.assertTrue(result["calibration_valid"])
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, model_state[name]), name)
        for name, parameter in model.named_parameters():
            before = gradients[name]
            if before is None:
                self.assertIsNone(parameter.grad, name)
            else:
                self.assertTrue(torch.equal(parameter.grad, before), name)
            self.assertEqual(parameter.requires_grad, flags[name], name)
        self.assertEqual(model.adapter.get_dynamic_lora_metadata(), ranks)
        assert_nested_equal(self, optimizer.state_dict(), optimizer_state, "optimizer")
        assert_nested_equal(self, scheduler.state_dict(), scheduler_state, "scheduler")
        self.assertEqual([module.training for module in model.modules()], modes)
        assert_rng_equal(self, rng_state, virtual.capture_rng_state())

    def test_e_virtual_calibration_is_deterministic(self):
        model, optimizer, scheduler, pairs = virtual_fixture()
        arguments = dict(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            candidate_module_names=["adapter"],
            module_map={"adapter": model.adapter},
            rank_increment=1,
            calibration_batch_pairs=pairs,
            loss_fn=virtual_loss,
            beta=0.5,
            max_grad_norm=1.0,
        )
        first = virtual.score_virtual_candidate(**arguments)
        second = virtual.score_virtual_candidate(**arguments)
        self.assertEqual(first["fold_gains"], second["fold_gains"])
        self.assertEqual(first["calibration_gain_mean"], second["calibration_gain_mean"])
        self.assertEqual(first["calibration_gain_std"], second["calibration_gain_std"])
        self.assertEqual(first["calibration_gain_lcb"], second["calibration_gain_lcb"])

        reference = quality_candidate(
            "adapter", 0.0, first["candidate_cost"], 150,
            "greedy_anchor", True, chromosome_index=0,
        )
        reference["fold_gains"] = list(first["fold_gains"])
        alternative = quality_candidate(
            "alternative", 0.0, first["candidate_cost"] + 1, 151,
            "greedy_neighborhood", chromosome_index=1,
        )
        alternative["fold_gains"] = [
            gain - 0.01 for gain in first["fold_gains"]
        ]
        selection_arguments = dict(
            candidates=[zero_candidate(), reference, alternative],
            min_calibrated_marginal_gain=-1e6,
            require_global_quality_improvement=False,
            target_final_cost=300,
        )
        first_winner, _ = calibrated.select_calibrated_candidate(
            **selection_arguments
        )
        second_winner, _ = calibrated.select_calibrated_candidate(
            **copy.deepcopy(selection_arguments)
        )
        self.assertEqual(first_winner["modules"], second_winner["modules"])


class CandidateGenerationAndSelectionTest(unittest.TestCase):
    def test_f_variable_sizes_include_zero_through_top_h_and_old_modes_remain_fixed(self):
        candidates, diagnostics = generate_candidates()
        self.assertEqual(diagnostics["allowed_event_rank_sizes"], [0, 1, 2, 3])
        self.assertEqual(
            {candidate["chromosome_size"] for candidate in candidates},
            {0, 1, 2, 3},
        )
        self.assertTrue(all(diagnostics["candidate_size_counts"][size] > 0 for size in range(4)))

        fixture = search_fixture()
        old_selected, _ = budgeted_evo_allocator.select_modules_budgeted(
            scores=fixture["scores"],
            costs=fixture["costs"],
            training_gains=fixture["gains"],
            top_h=2,
            current_active_cost=40,
            target_final_cost=75,
            future_rank_increments=2,
            population_size=10,
            generations=1,
            seed=41,
        )
        self.assertEqual(len(old_selected), 2)

    def test_g_h_greedy_anchor_and_bounded_neighborhood_are_present(self):
        candidates, diagnostics = generate_candidates()
        anchor = set(diagnostics["greedy_anchor_modules"])
        anchors = [
            candidate
            for candidate in candidates
            if "greedy_anchor" in candidate["candidate_families"]
        ]
        self.assertTrue(diagnostics["greedy_anchor_feasible"])
        self.assertTrue(any(set(candidate["modules"]) == anchor for candidate in anchors))
        neighborhoods = [
            candidate
            for candidate in candidates
            if "greedy_neighborhood" in candidate["candidate_families"]
        ]
        self.assertTrue(any((candidate["replacement_count"] or 0) > 0 for candidate in neighborhoods))
        self.assertTrue(
            all((candidate["replacement_count"] or 0) <= 2 for candidate in neighborhoods)
        )

        fixture = search_fixture()
        tied_scores = list(fixture["scores"])
        tied_scores[1] = (tied_scores[1][0], tied_scores[0][1])
        _, tied_diagnostics = generate_candidates(
            scores=tied_scores,
            top_h=1,
            max_event_rank=1,
            greedy_anchor_modules=fixture["names"][:2],
        )
        self.assertEqual(
            tied_diagnostics["greedy_anchor_modules"], fixture["names"][:2]
        )
        self.assertFalse(tied_diagnostics["greedy_anchor_feasible"])
        self.assertEqual(len(tied_diagnostics["repaired_greedy_anchor_modules"]), 1)

    def test_i_quality_floor_blocks_materially_worse_cheap_candidate(self):
        reference = quality_candidate("greedy", 1.0, 100, 200, "greedy_anchor", True, chromosome_index=0)
        cheap = quality_candidate("cheap", 0.8, 20, 120, "greedy_neighborhood", chromosome_index=1)
        selected, diagnostics = calibrated.select_calibrated_candidate(
            [zero_candidate(), reference, cheap],
            greedy_quality_floor_ratio=0.99,
            require_global_quality_improvement=False,
            target_final_cost=300,
        )
        self.assertEqual(selected["modules"], ["greedy"])
        rejected = next(item for item in diagnostics["candidates"] if item["modules"] == ["cheap"])
        self.assertFalse(rejected["greedy_quality_floor_satisfied"])

    def test_j_quality_equivalent_candidate_is_selected_by_lower_cost(self):
        reference = quality_candidate("greedy", 1.0, 100, 200, "greedy_anchor", True, chromosome_index=0)
        cheap = quality_candidate("cheap", 0.995, 50, 150, "greedy_neighborhood", chromosome_index=1)
        selected, diagnostics = calibrated.select_calibrated_candidate(
            [zero_candidate(), reference, cheap],
            quality_relative_tolerance=0.01,
            greedy_quality_floor_ratio=0.99,
            require_global_quality_improvement=False,
            target_final_cost=300,
        )
        self.assertEqual(selected["modules"], ["cheap"])
        self.assertTrue(diagnostics["quality_band_satisfied"])

    def test_k_expensive_candidate_wins_when_quality_is_materially_better(self):
        reference = quality_candidate("greedy", 1.0, 60, 160, "greedy_anchor", True, chromosome_index=0)
        quality = quality_candidate("quality", 1.2, 100, 200, "global_ga", chromosome_index=1)
        selected, diagnostics = calibrated.select_calibrated_candidate(
            [zero_candidate(), reference, quality],
            quality_relative_tolerance=0.01,
            greedy_quality_floor_ratio=0.99,
            target_final_cost=300,
        )
        self.assertEqual(selected["modules"], ["quality"])
        self.assertEqual(diagnostics["selected_source"], "calibrated_global_ga")

    def test_n_hard_budget_is_never_exceeded_and_infeasible_initial_state_fails(self):
        current = 40
        target = 75
        for event in range(3):
            candidates, _ = generate_candidates(
                top_h=2,
                min_event_rank=1,
                max_event_rank=2,
                current_active_cost=current,
                target_final_cost=target,
                seed=41 + event,
            )
            scored = []
            for candidate in candidates:
                candidate = dict(candidate)
                candidate["fold_gains"] = (
                    [0.0, 0.0]
                    if candidate["chromosome_size"] == 0
                    else [1.0 + candidate["structural_fitness"]] * 2
                )
                scored.append(candidate)
            selected, _ = calibrated.select_calibrated_candidate(
                scored,
                require_global_quality_improvement=False,
                target_final_cost=target,
            )
            self.assertLessEqual(selected["projected_final_active_parameter_count"], target)
            current = selected["projected_final_active_parameter_count"]
        self.assertLessEqual(current, target)
        with self.assertRaisesRegex(ValueError, "infeasible before allocation"):
            generate_candidates(current_active_cost=76, target_final_cost=75)

    def test_q_calibrated_search_never_invokes_old_or_memetic_search(self):
        model = TinyDynamicModel()
        for module in (model.small, model.large):
            module.score = torch.tensor(0.0)
        allocator = calibrated_allocator(model)
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
                "calibration_runtime_seconds": 0.0,
            }

        allocator.update_and_increase(
            model,
            0,
            optimizer,
            calibration_batch_pairs=calibration_pairs,
            calibration_loss_fn=lambda model, batch: torch.tensor(0.0),
        )
        with mock.patch.object(
            evo_allocator,
            "select_modules_genetic",
            side_effect=AssertionError("old genetic/local-search path entered"),
        ), mock.patch.object(
            budgeted_evo_allocator,
            "select_modules_budgeted",
            side_effect=AssertionError("old budgeted path entered"),
        ), mock.patch(
            "transformers.calibrated_rank_calibration.score_virtual_candidate",
            side_effect=finite_calibration,
        ):
            selected_rank, _ = allocator.update_and_increase(
                model,
                1,
                optimizer,
                calibration_batch_pairs=calibration_pairs,
                calibration_loss_fn=lambda model, batch: torch.tensor(0.0),
            )
        self.assertGreater(selected_rank, 0)
        self.assertLessEqual(
            get_active_model_parameter_count(model),
            allocator.target_final_trainable_params,
        )
        allocator.record_final_trajectory(model, global_step=1)
        final_pattern = allocator.finalize_budget(model)
        self.assertEqual(
            final_pattern["active_model_parameter_count"],
            get_active_model_parameter_count(model),
        )
        self.assertEqual(
            final_pattern["budget"]["export_metadata_role"],
            "selected_best_checkpoint_with_separate_final_trajectory",
        )

class AdaptiveStoppingConsolidationWarmupTest(unittest.TestCase):
    @staticmethod
    def _zero_calibration(**kwargs):
        names = list(kwargs["candidate_module_names"])
        folds = len(kwargs["calibration_batch_pairs"])
        return {
            "candidate_modules": names,
            "candidate_size": len(names),
            "candidate_cost": len(names),
            "fold_gains": [0.0] * folds,
            "fold_details": [],
            "calibration_gain_mean": 0.0,
            "calibration_gain_std": 0.0,
            "calibration_gain_lcb": 0.0,
            "calibration_gain_per_parameter": 0.0,
            "calibration_valid": True,
            "calibration_runtime_seconds": 0.0,
        }

    def test_l_m_low_gain_stops_growth_and_consolidation_cannot_restart(self):
        model = TinyDynamicModel()
        for module in (model.small, model.large):
            module.score = torch.tensor(0.0)
        allocator = calibrated_allocator(model)
        allocator.set_total_step(100)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        calibration_pairs = [({"unused": 1}, {"unused": 2})]

        with mock.patch(
            "transformers.calibrated_rank_calibration.score_virtual_candidate",
            side_effect=self._zero_calibration,
        ):
            allocator.update_and_increase(
                model,
                0,
                optimizer,
                calibration_batch_pairs=calibration_pairs,
                calibration_loss_fn=lambda model, batch: torch.tensor(0.0),
            )
            allocator.update_and_increase(
                model,
                1,
                optimizer,
                calibration_batch_pairs=calibration_pairs,
                calibration_loss_fn=lambda model, batch: torch.tensor(0.0),
            )
            self.assertEqual(allocator.zero_rank_event_counter, 1)
            allocator.update_and_increase(
                model,
                2,
                optimizer,
                calibration_batch_pairs=calibration_pairs,
                calibration_loss_fn=lambda model, batch: torch.tensor(0.0),
            )

        self.assertTrue(allocator.allocation_stopped)
        self.assertEqual(allocator.allocation_stopped_step, 2)
        self.assertEqual(allocator.consolidation_remaining_steps, 98)
        ranks = [int(module.ranknum.item()) for module in (model.small, model.large)]
        active = get_active_model_parameter_count(model)
        lengths = [len(module.lora_A) for module in (model.small, model.large)]

        with mock.patch(
            "transformers.calibrated_budgeted_evo_allocator.generate_calibrated_candidates",
            side_effect=AssertionError("GA restarted during consolidation"),
        ):
            selected_rank, threshold = allocator.update_and_increase(
                model,
                3,
                optimizer,
                calibration_batch_pairs=calibration_pairs,
                calibration_loss_fn=lambda model, batch: torch.tensor(0.0),
            )
        self.assertEqual((selected_rank, threshold), (0, None))
        self.assertEqual(
            [int(module.ranknum.item()) for module in (model.small, model.large)], ranks
        )
        self.assertEqual(get_active_model_parameter_count(model), active)
        self.assertEqual([len(module.lora_A) for module in (model.small, model.large)], lengths)

    def test_minimum_consolidation_window_forces_fixed_architecture(self):
        model = TinyDynamicModel()
        allocator = calibrated_allocator(model, ga_min_consolidation_steps=3)
        allocator.set_total_step(5)
        optimizer = AdamW(model.parameters(), lr=1e-3)
        calibration_pairs = [({"unused": 1}, {"unused": 2})]
        allocator.update_and_increase(
            model,
            0,
            optimizer,
            calibration_batch_pairs=calibration_pairs,
            calibration_loss_fn=lambda model, batch: torch.tensor(0.0),
        )
        ranks_before = [
            int(module.ranknum.item()) for module in (model.small, model.large)
        ]
        with mock.patch(
            "transformers.calibrated_budgeted_evo_allocator.generate_calibrated_candidates",
            side_effect=AssertionError("GA ran inside the forced consolidation window"),
        ):
            decision = allocator.update_and_increase(
                model,
                2,
                optimizer,
                calibration_batch_pairs=calibration_pairs,
                calibration_loss_fn=lambda model, batch: torch.tensor(0.0),
            )
        self.assertEqual(decision, (0, None))
        self.assertTrue(allocator.allocation_stopped)
        self.assertEqual(allocator.allocation_stopped_step, 2)
        self.assertEqual(allocator.consolidation_remaining_steps, 3)
        self.assertEqual(
            [int(module.ranknum.item()) for module in (model.small, model.large)],
            ranks_before,
        )

    def test_new_rank_warmup_scales_only_registered_parameter_deltas(self):
        model = TinyDynamicModel()
        allocator = calibrated_allocator(model, ga_new_rank_lr_warmup_steps=2)
        warmed = model.small.lora_E[0]
        control = model.large.lora_E[0]
        allocator.register_new_rank_warmup(model, [id(warmed)], global_step=10)

        allocator.snapshot_new_rank_warmup_parameters(model)
        allocator.apply_new_rank_lr_warmup(model)
        warmed_name = next(
            name for name, parameter in model.named_parameters() if parameter is warmed
        )
        self.assertEqual(
            allocator.new_rank_warmup_state[warmed_name]["steps_completed"], 0
        )

        warmed_before = warmed.detach().clone()
        control_before = control.detach().clone()
        warmed.grad = torch.zeros_like(warmed)
        allocator.snapshot_new_rank_warmup_parameters(model)
        with torch.no_grad():
            warmed.add_(2.0)
            control.add_(2.0)
        allocator.apply_new_rank_lr_warmup(model)
        self.assertTrue(torch.equal(warmed, warmed_before + 1.0))
        self.assertTrue(torch.equal(control, control_before + 2.0))

        warmed_before = warmed.detach().clone()
        control_before = control.detach().clone()
        warmed.grad = torch.zeros_like(warmed)
        allocator.snapshot_new_rank_warmup_parameters(model)
        with torch.no_grad():
            warmed.add_(2.0)
            control.add_(2.0)
        allocator.apply_new_rank_lr_warmup(model)
        self.assertTrue(torch.equal(warmed, warmed_before + 2.0))
        self.assertTrue(torch.equal(control, control_before + 2.0))
        self.assertEqual(allocator.new_rank_warmup_state, {})


class ResumeAndReportingTest(unittest.TestCase):
    def test_calibrated_sampler_and_rotation_preserve_resume_state(self):
        with tempfile.TemporaryDirectory() as directory:
            model = TinyDynamicModel()
            allocator = calibrated_allocator(model)
            training = TrackingDataset(20)
            arguments = TrainingArguments(
                output_dir=directory,
                no_cuda=True,
                report_to=[],
                per_device_train_batch_size=2,
                save_total_limit=3,
                seed=41,
            )
            trainer = Trainer(
                model=model,
                args=arguments,
                train_dataset=training,
                data_collator=lambda examples: examples,
                rankallocator=allocator,
            )
            first_sampler = trainer._get_train_sampler()
            second_sampler = trainer._get_train_sampler()
            first_sampler.set_epoch(4)
            second_sampler.set_epoch(4)
            self.assertEqual(list(first_sampler), list(second_sampler))

            root = Path(directory)
            for step in (100, 200, 300, 400):
                (root / ("checkpoint-%s" % step)).mkdir()
            trainer.state.best_model_checkpoint = str(root / "checkpoint-100")
            trainer._rotate_checkpoints(output_dir=str(root))
            self.assertEqual(
                sorted(path.name for path in root.glob("checkpoint-*")),
                ["checkpoint-100", "checkpoint-300", "checkpoint-400"],
            )

    def test_o_allocator_metadata_and_optimizer_order_resume_equivalence(self):
        model = TinyDynamicModel()
        allocator = calibrated_allocator(model)
        allocator.set_total_step(20)
        training_configuration = {"fixture": "stable"}
        allocator.set_training_configuration(training_configuration)
        allocator.get_or_create_calibration_indices(8, "resume-fixture")
        for index, module in enumerate((model.small, model.large)):
            module.score = torch.tensor(float(2 - index))
        named = list(model.named_parameters())
        first = [parameter for index, (_, parameter) in enumerate(named) if index % 2 == 0]
        second = [parameter for index, (_, parameter) in enumerate(named) if index % 2 == 1]
        optimizer = AdamW([{"params": first}, {"params": second}], lr=1e-3)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda step: 1.0
        )
        calibration_pairs = [({"unused": 1}, {"unused": 2})]
        allocator.update_and_increase(
            model,
            0,
            optimizer,
            lr_scheduler=scheduler,
            calibration_batch_pairs=calibration_pairs,
            calibration_loss_fn=lambda model, batch: torch.tensor(0.0),
        )
        allocator.zero_rank_event_counter = 1
        allocator.register_new_rank_warmup(model, [id(model.small.lora_E[0])], 7)
        allocator.capture_checkpoint_state(model, optimizer, global_step=7)
        metadata = copy.deepcopy(model.config.budgeted_rank_allocator)
        model_state = copy.deepcopy(model.state_dict())
        optimizer_state = copy.deepcopy(optimizer.state_dict())
        scheduler_state = copy.deepcopy(scheduler.state_dict())

        resumed_model = TinyDynamicModel()
        for name in ("small", "large"):
            getattr(resumed_model, name).set_dynamic_lora_metadata(
                getattr(model, name).get_dynamic_lora_metadata()
            )
        resumed_model.load_state_dict(model_state)
        resumed_model.config.budgeted_rank_allocator = metadata
        resumed = calibrated_allocator(resumed_model)
        resumed.set_total_step(20)
        resumed.set_training_configuration(training_configuration)
        with self.assertRaisesRegex(ValueError, "total optimization steps"):
            resumed.set_total_step(21)
        resumed_named = list(resumed_model.named_parameters())
        resumed_first = [
            parameter for index, (_, parameter) in enumerate(resumed_named) if index % 2 == 1
        ]
        resumed_second = [
            parameter for index, (_, parameter) in enumerate(resumed_named) if index % 2 == 0
        ]
        resumed_optimizer = AdamW(
            [{"params": resumed_first}, {"params": resumed_second}], lr=1e-3
        )
        resumed.prepare_optimizer_for_resume(resumed_model, resumed_optimizer)
        resumed_optimizer.load_state_dict(optimizer_state)
        resumed_scheduler = torch.optim.lr_scheduler.LambdaLR(
            resumed_optimizer, lr_lambda=lambda step: 1.0
        )
        resumed_scheduler.load_state_dict(scheduler_state)

        def group_names(current_model, current_optimizer):
            names = {id(parameter): name for name, parameter in current_model.named_parameters()}
            return [
                [names[id(parameter)] for parameter in group["params"]]
                for group in current_optimizer.param_groups
            ]

        self.assertEqual(group_names(resumed_model, resumed_optimizer), allocator.optimizer_parameter_groups)
        self.assertEqual(resumed.calibration_training_indices, allocator.calibration_training_indices)
        self.assertEqual(resumed.zero_rank_event_counter, allocator.zero_rank_event_counter)
        self.assertEqual(resumed.new_rank_warmup_state, allocator.new_rank_warmup_state)
        self.assertEqual(resumed.rank_pattern, allocator.rank_pattern)

        def deterministic_calibration(**kwargs):
            names = list(kwargs["candidate_module_names"])
            gain = 0.0 if not names else 1.0 + 0.01 * names.count("small")
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
                "calibration_runtime_seconds": 0.0,
            }

        rng_state = virtual.capture_rng_state()
        for index, (uninterrupted_module, resumed_module) in enumerate(
            zip(
                (model.small, model.large),
                (resumed_model.small, resumed_model.large),
            )
        ):
            score = torch.tensor(float(2 - index))
            uninterrupted_module.score = score.clone()
            resumed_module.score = score.clone()
        with mock.patch(
            "transformers.calibrated_rank_calibration.score_virtual_candidate",
            side_effect=deterministic_calibration,
        ):
            uninterrupted_decision = allocator.update_and_increase(
                model,
                8,
                optimizer,
                lr_scheduler=scheduler,
                calibration_batch_pairs=calibration_pairs,
                calibration_loss_fn=lambda model, batch: torch.tensor(0.0),
            )
            virtual.restore_rng_state(rng_state)
            resumed_decision = resumed.update_and_increase(
                resumed_model,
                8,
                resumed_optimizer,
                lr_scheduler=resumed_scheduler,
                calibration_batch_pairs=calibration_pairs,
                calibration_loss_fn=lambda model, batch: torch.tensor(0.0),
            )

        self.assertEqual(uninterrupted_decision, resumed_decision)
        self.assertEqual(allocator.rank_pattern, resumed.rank_pattern)
        self.assertEqual(allocator.total_rank, resumed.total_rank)
        self.assertEqual(
            get_active_model_parameter_count(model),
            get_active_model_parameter_count(resumed_model),
        )
        for name in allocator.name_set:
            self.assertEqual(
                allocator._scalar_value(allocator.exp_avg_ipt[name]),
                resumed._scalar_value(resumed.exp_avg_ipt[name]),
            )
            self.assertEqual(
                allocator._scalar_value(allocator.exp_avg_unc[name]),
                resumed._scalar_value(resumed.exp_avg_unc[name]),
            )

    def test_p_reporter_keeps_final_and_best_checkpoint_comparisons_distinct(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            budgeted = root / "budgeted"
            greedy = root / "greedy"
            budgeted_best = budgeted / "model" / "checkpoint-10"
            greedy_best = greedy / "model" / "checkpoint-4"
            budgeted_best.mkdir(parents=True)
            greedy_best.mkdir(parents=True)
            (budgeted_best / "config.json").write_text(
                json.dumps(
                    {
                        "dynamic_lora_rank_pattern": {
                            "a": {"active_rank": 1},
                            "b": {"active_rank": 1},
                        }
                    }
                ),
                encoding="utf-8",
            )
            (greedy_best / "config.json").write_text(
                json.dumps(
                    {
                        "dynamic_lora_rank_pattern": {
                            "a": {"active_rank": 1},
                            "b": {"active_rank": 2},
                        }
                    }
                ),
                encoding="utf-8",
            )
            trainer_state = budgeted / "model" / "trainer_state.json"
            trainer_state.write_text(
                json.dumps({"best_model_checkpoint": str(budgeted_best)}),
                encoding="utf-8",
            )
            budgeted_pattern = budgeted / "rank_pattern.json"
            budgeted_pattern.write_text(
                json.dumps(
                    {
                        "format_version": 2,
                        "allocator_mode": "genetic_budgeted_calibrated",
                        "non_dynamic_trainable_params": 10,
                        "modules": {
                            "a": {"active_rank": 1, "rank_one_cost": 3},
                            "b": {"active_rank": 1, "rank_one_cost": 5},
                        },
                        "active_model_parameter_count": 18,
                        "budget": {
                            "allocator_mode": "genetic_budgeted_calibrated",
                            "reference_cost": 29,
                            "target_cost": 25,
                            "budget_ratio": 0.86,
                            "final_trajectory": {
                                "total_active_rank": 3,
                                "active_model_parameter_count": 21,
                                "rank_pattern": {"a": 2, "b": 1},
                            },
                            "selected_best_checkpoint": {
                                "total_active_rank": 2,
                                "active_model_parameter_count": 18,
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            greedy_pattern = greedy / "rank_pattern.json"
            greedy_pattern.write_text(json.dumps({"a": 3, "b": 2}), encoding="utf-8")

            report = build_rank_budget_report(
                trainer_state,
                budgeted_pattern,
                greedy_best,
                greedy_pattern,
            )
            final = report["final_trajectory_comparison"]
            best = report["best_checkpoint_comparison"]
            self.assertEqual(final["budgeted_final_trajectory"]["active_model_parameter_count"], 21)
            self.assertEqual(final["greedy_final_rank_pattern"]["active_model_parameter_count"], 29)
            self.assertEqual(best["budgeted_best_checkpoint"]["active_model_parameter_count"], 18)
            self.assertEqual(best["greedy_best_checkpoint"]["active_model_parameter_count"], 23)
            self.assertNotEqual(final["absolute_parameter_reduction"], best["absolute_parameter_reduction"])

            payload = json.loads(budgeted_pattern.read_text(encoding="utf-8"))
            payload["budget"]["final_trajectory"]["active_model_parameter_count"] = 20
            budgeted_pattern.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "final-trajectory metrics"):
                build_rank_budget_report(
                    trainer_state,
                    budgeted_pattern,
                    greedy_best,
                    greedy_pattern,
                )


if __name__ == "__main__":
    unittest.main()
