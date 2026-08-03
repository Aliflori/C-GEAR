import importlib.util
import math
from pathlib import Path
import unittest


ALLOCATOR_PATH = (
    Path(__file__).resolve().parents[1] / "src" / "transformers" / "evo_allocator.py"
)
SPEC = importlib.util.spec_from_file_location("evo_allocator_under_test", ALLOCATOR_PATH)
EVO_ALLOCATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVO_ALLOCATOR)


class InteractionAwareEvolutionTest(unittest.TestCase):
    def setUp(self):
        self.identifiers = [
            "encoder.layer.%s.query_proj" % index for index in range(12)
        ]
        self.scores = [
            1.00,
            0.98,
            0.96,
            0.94,
            0.92,
            0.90,
            0.88,
            0.86,
            0.60,
            0.55,
            0.50,
            0.45,
        ]
        positive_phase = [1, 3, 1, 3, 1, 3, 1, 3]
        complementary_phase = [3, 1, 3, 1, 3, 1, 3, 1]
        zigzag = [1, 2, 3, 2, 1, 2, 3, 2]
        reverse_zigzag = [3, 2, 1, 2, 3, 2, 1, 2]
        self.features = {}
        for index, identifier in enumerate(self.identifiers):
            if index < 4:
                self.features[identifier] = [
                    value + index * 0.01 for value in positive_phase
                ]
            elif index < 8:
                self.features[identifier] = [
                    value + index * 0.01 for value in complementary_phase
                ]
            elif index % 2 == 0:
                self.features[identifier] = zigzag
            else:
                self.features[identifier] = reverse_zigzag
        self.costs = {identifier: 100 for identifier in self.identifiers}

    def allocate(self, **overrides):
        arguments = {
            "scores": list(zip(self.identifiers, self.scores)),
            "costs": self.costs,
            "top_h": 4,
            "population_size": 12,
            "generations": 4,
            "mutation_rate": 0.10,
            "crossover_rate": 0.80,
            "interaction_weight": 0.20,
            "redundancy_weight": 0.20,
            "cost_weight": 0.30,
            "diversity_weight": 0.10,
            "seed": 4,
            "module_features": self.features,
            "local_search": False,
        }
        arguments.update(overrides)
        return EVO_ALLOCATOR.select_modules_genetic(**arguments)

    def test_ga_only_beats_greedy_on_complementary_modules(self):
        selected, diagnostics = self.allocate()

        self.assertTrue(diagnostics["ga_beats_greedy"])
        self.assertGreater(
            diagnostics["ga_best_fitness"], diagnostics["greedy_fitness"]
        )
        self.assertTrue(diagnostics["ga_improved_over_initial"])
        self.assertFalse(diagnostics["local_search_enabled"])
        self.assertFalse(diagnostics["local_search_improved"])
        self.assertEqual(diagnostics["selected_source"], "ga_only")
        self.assertEqual(diagnostics["best_single_swap_modules"], [])
        self.assertNotEqual(set(selected), set(self.identifiers[:4]))
        self.assertGreater(diagnostics["interaction_gain"], 0.0)

    def test_population_remains_unique_and_diverse(self):
        _, diagnostics = self.allocate()

        self.assertEqual(diagnostics["unique_population_count_initial"], 12)
        self.assertEqual(diagnostics["unique_population_count_final"], 12)
        self.assertGreater(diagnostics["population_diversity_final"], 0.25)
        self.assertGreater(diagnostics["evaluated_unique_chromosome_count"], 12)
        self.assertEqual(len(diagnostics["population_diversity_history"]), 5)

    def test_deberta_scale_search_improves_beyond_greedy(self):
        projections = [
            "query",
            "key",
            "value",
            "intermediate",
            "layer.output",
            "attention.output",
        ]
        identifiers = [
            "encoder.layer.%s.%s" % (layer, projection)
            for layer in range(12)
            for projection in projections
        ]
        scores = [1.0 - index * 0.008 for index in range(len(identifiers))]
        phase_a = [1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4]
        phase_b = [4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1]
        wave_a = [1, 2, 4, 3, 1, 2, 4, 3, 1, 2, 4, 3]
        wave_b = [4, 3, 1, 2, 4, 3, 1, 2, 4, 3, 1, 2]
        features = {}
        for index, identifier in enumerate(identifiers):
            if index < 5:
                history = phase_a
            elif index < 15:
                history = phase_b
            elif index % 4 == 0:
                history = phase_a
            elif index % 4 == 1:
                history = phase_b
            elif index % 4 == 2:
                history = wave_a
            else:
                history = wave_b
            features[identifier] = [
                value + index * 0.0001 for value in history
            ]
        costs = {
            identifier: 100 + (index % len(projections)) * 10
            for index, identifier in enumerate(identifiers)
        }

        selected, diagnostics = EVO_ALLOCATOR.select_modules_genetic(
            scores=list(zip(identifiers, scores)),
            costs=costs,
            top_h=5,
            population_size=12,
            generations=4,
            mutation_rate=0.10,
            crossover_rate=0.80,
            interaction_weight=0.20,
            redundancy_weight=0.20,
            cost_weight=0.30,
            diversity_weight=0.10,
            seed=3,
            module_features=features,
            local_search=False,
        )

        self.assertEqual(len(identifiers), 72)
        self.assertEqual(len(selected), 5)
        self.assertTrue(diagnostics["ga_beats_greedy"])
        self.assertTrue(diagnostics["ga_improved_over_initial"])
        self.assertFalse(diagnostics["local_search_enabled"])
        self.assertEqual(diagnostics["unique_population_count_final"], 12)

    def test_allocator_is_deterministic_and_fixed_cardinality(self):
        first_selected, first_diagnostics = self.allocate()
        second_selected, second_diagnostics = self.allocate()

        self.assertEqual(first_selected, second_selected)
        self.assertEqual(first_diagnostics["ga_best_fitness"], second_diagnostics["ga_best_fitness"])
        self.assertEqual(len(first_selected), 4)
        self.assertEqual(len(set(first_selected)), 4)

    def test_local_search_is_an_explicit_ablation(self):
        _, diagnostics = self.allocate(local_search=True, generations=0)

        self.assertTrue(diagnostics["local_search_enabled"])
        self.assertTrue(diagnostics["best_single_swap_modules"])

    def test_invalid_or_nonfinite_scores_do_not_break_allocation(self):
        noisy_scores = list(zip(self.identifiers, self.scores))
        noisy_scores[0] = (self.identifiers[0], float("nan"))
        noisy_scores[1] = (self.identifiers[1], float("inf"))
        selected, diagnostics = self.allocate(
            scores=noisy_scores,
            module_features=None,
            generations=1,
        )

        self.assertEqual(len(selected), 4)
        self.assertTrue(math.isfinite(diagnostics["ga_best_fitness"]))
        self.assertEqual(
            diagnostics["interaction_feature_mode"], "structural_redundancy_only"
        )


if __name__ == "__main__":
    unittest.main()
