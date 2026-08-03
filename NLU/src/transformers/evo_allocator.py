# coding=utf-8
"""Interaction-aware evolutionary rank allocation for EvoIncreLoRA."""

import math
import random


_FITNESS_TOLERANCE = 1e-12
_LOCAL_SEARCH_IMPROVEMENT_TOLERANCE = 1e-12


def _finite_float(value, default=0.0):
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return value if math.isfinite(value) else default


def _ordered_items(values):
    if values is None:
        return []
    if hasattr(values, "items"):
        return list(values.items())

    items = []
    for position, value in enumerate(list(values)):
        if isinstance(value, (tuple, list)) and len(value) == 2:
            items.append((value[0], value[1]))
        else:
            items.append((position, value))
    return items


def _deduplicated_scores(scores):
    items = []
    seen = set()
    for identifier, score in _ordered_items(scores):
        try:
            is_duplicate = identifier in seen
        except TypeError:
            raise TypeError("Candidate module identifiers must be hashable.")
        if is_duplicate:
            continue
        seen.add(identifier)
        items.append((identifier, _finite_float(score)))
    return items


def _value_map(values, converter):
    result = {}
    for identifier, value in _ordered_items(values):
        try:
            if identifier not in result:
                result[identifier] = converter(value)
        except TypeError:
            raise TypeError("Candidate module identifiers must be hashable.")
    return result


def _normalized_scores(scores):
    if not scores:
        return []
    low = min(scores)
    high = max(scores)
    width = high - low
    if width <= 0.0 or not math.isfinite(width):
        return [0.0 for _ in scores]
    return [(score - low) / width for score in scores]


def _history_vector(value):
    if value is None or isinstance(value, (str, bytes)):
        return ()
    try:
        entries = list(value)
    except TypeError:
        entries = [value]

    history = []
    for entry in entries:
        try:
            entry = float(entry)
        except (TypeError, ValueError, OverflowError):
            entry = None
        if entry is not None and not math.isfinite(entry):
            entry = None
        history.append(entry)
    return tuple(history)


def _centered_temporal_features(
    identifiers, module_features, max_window=20, epsilon=1e-12
):
    """Create scale-invariant temporal trajectories for pair interactions."""

    history_map = _value_map(module_features, _history_vector)
    histories = [history_map.get(identifier, ()) for identifier in identifiers]
    if len(histories) < 2 or any(len(history) < 2 for history in histories):
        return {}

    window = min(max_window, min(len(history) for history in histories))
    histories = [history[-window:] for history in histories]
    trajectories = [[] for _ in identifiers]

    for time_index in range(window):
        finite_values = [
            history[time_index]
            for history in histories
            if history[time_index] is not None
        ]
        if len(finite_values) < 2:
            continue
        time_mean = sum(finite_values) / len(finite_values)
        time_variance = sum(
            (value - time_mean) ** 2 for value in finite_values
        ) / len(finite_values)
        time_scale = max(abs(value) for value in finite_values)
        variance_floor = (epsilon * time_scale) ** 2
        if (
            not math.isfinite(time_variance)
            or time_scale == 0.0
            or time_variance <= variance_floor
        ):
            continue
        time_std = math.sqrt(time_variance)
        for module_index, history in enumerate(histories):
            value = history[time_index]
            trajectories[module_index].append(
                0.0 if value is None else (value - time_mean) / time_std
            )

    if not trajectories or len(trajectories[0]) < 2:
        return {}

    features = {}
    for module_index, trajectory in enumerate(trajectories):
        trajectory_mean = sum(trajectory) / len(trajectory)
        centered = [value - trajectory_mean for value in trajectory]
        feature = list(centered)
        if len(centered) >= 3:
            feature.extend(
                centered[index] - centered[index - 1]
                for index in range(1, len(centered))
            )
        norm = math.sqrt(sum(value * value for value in feature))
        if not math.isfinite(norm) or norm <= epsilon:
            return {}
        features[module_index] = tuple(value / norm for value in feature)
    return features


def _structural_signature(identifier):
    parts = str(identifier).lower().split(".")
    layer = None
    for index, part in enumerate(parts[:-1]):
        if part in ("layer", "layers", "block", "blocks", "h"):
            layer = ".".join(parts[: index + 2])
            break

    joined = ".".join(parts)
    if "query" in parts[-1]:
        projection = "query"
    elif "key" in parts[-1]:
        projection = "key"
    elif "value" in parts[-1]:
        projection = "value"
    elif ".intermediate." in "." + joined + ".":
        projection = "intermediate"
    elif ".attention.output." in "." + joined + ".":
        projection = "attention_output"
    elif ".output." in "." + joined + ".":
        projection = "layer_output"
    else:
        projection = parts[-1] if parts else None
    return layer, projection


def _structural_similarity(left, right):
    left_layer, left_projection = left
    right_layer, right_projection = right
    similarity = 0.0
    if left_layer is not None and left_layer == right_layer:
        similarity += 0.5
    if left_projection is not None and left_projection == right_projection:
        similarity += 0.5
    return similarity


def _cosine_similarity(left, right):
    """Return signed cosine similarity so anti-correlation remains observable."""

    if not left or not right:
        return 0.0
    pairs = list(zip(left, right))
    if not pairs:
        return 0.0
    dot = sum(a * b for a, b in pairs)
    left_norm = math.sqrt(sum(a * a for a, _ in pairs))
    right_norm = math.sqrt(sum(b * b for _, b in pairs))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return min(1.0, max(-1.0, dot / (left_norm * right_norm)))


def _validate_configuration(
    population_size,
    generations,
    mutation_rate,
    crossover_rate,
    interaction_weight,
    redundancy_weight,
    cost_weight,
    diversity_weight,
):
    if population_size <= 0:
        raise ValueError("ga_population must be positive.")
    if generations < 0:
        raise ValueError("ga_generations must be nonnegative.")
    if not math.isfinite(mutation_rate) or not 0.0 <= mutation_rate <= 1.0:
        raise ValueError("ga_mutation_rate must be between 0 and 1.")
    if not math.isfinite(crossover_rate) or not 0.0 <= crossover_rate <= 1.0:
        raise ValueError("ga_crossover_rate must be between 0 and 1.")
    nonnegative_weights = (
        ("ga_interaction_weight", interaction_weight),
        ("ga_redundancy_weight", redundancy_weight),
        ("ga_cost_weight", cost_weight),
        ("ga_diversity_weight", diversity_weight),
    )
    for name, value in nonnegative_weights:
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("%s must be nonnegative." % name)


def select_modules_genetic(
    scores,
    costs,
    top_h,
    population_size=12,
    generations=4,
    mutation_rate=0.10,
    crossover_rate=0.80,
    interaction_weight=0.20,
    redundancy_weight=0.20,
    cost_weight=0.30,
    diversity_weight=0.10,
    seed=0,
    module_features=None,
    local_search=False,
):
    """Select a fixed-cardinality module set with interaction-aware evolution.

    The default result is the best chromosome discovered by the evolutionary
    search itself. One-swap local refinement is disabled by default and exists
    only as an explicit ablation through ``local_search=True``.
    """

    population_size = int(population_size)
    generations = int(generations)
    mutation_rate = float(mutation_rate)
    crossover_rate = float(crossover_rate)
    interaction_weight = float(interaction_weight)
    redundancy_weight = float(redundancy_weight)
    cost_weight = float(cost_weight)
    diversity_weight = float(diversity_weight)
    local_search = bool(local_search)
    _validate_configuration(
        population_size,
        generations,
        mutation_rate,
        crossover_rate,
        interaction_weight,
        redundancy_weight,
        cost_weight,
        diversity_weight,
    )

    score_items = _deduplicated_scores(scores)
    identifiers = [identifier for identifier, _ in score_items]
    raw_scores = [score for _, score in score_items]
    candidate_count = len(identifiers)
    selection_size = min(max(int(top_h), 0), candidate_count)
    normalized_scores = _normalized_scores(raw_scores)

    cost_map = _value_map(costs, lambda value: max(0.0, _finite_float(value)))
    candidate_costs = [cost_map.get(identifier, 0.0) for identifier in identifiers]
    sorted_costs = sorted(candidate_costs)
    minimum_feasible_cost = sum(sorted_costs[:selection_size])
    maximum_feasible_cost = (
        sum(sorted_costs[-selection_size:]) if selection_size else 0.0
    )
    feasible_cost_width = maximum_feasible_cost - minimum_feasible_cost

    temporal_features = _centered_temporal_features(identifiers, module_features)
    features_available = len(temporal_features) == candidate_count and candidate_count >= 2
    structural_signatures = [_structural_signature(identifier) for identifier in identifiers]

    if features_available:
        interaction_feature_mode = "temporal_complementarity"

        def pair_relation(left_index, right_index):
            return _cosine_similarity(
                temporal_features[left_index], temporal_features[right_index]
            )

    else:
        structural_similarities = [
            _structural_similarity(structural_signatures[left], structural_signatures[right])
            for left in range(candidate_count)
            for right in range(left + 1, candidate_count)
        ]
        if any(similarity > 0.0 for similarity in structural_similarities):
            interaction_feature_mode = "structural_redundancy_only"

            def pair_relation(left_index, right_index):
                return _structural_similarity(
                    structural_signatures[left_index], structural_signatures[right_index]
                )

        else:
            interaction_feature_mode = "unavailable"

            def pair_relation(left_index, right_index):
                return 0.0

    def pair_redundancy(left_index, right_index):
        return max(0.0, pair_relation(left_index, right_index))

    def pair_interaction_gain(left_index, right_index):
        if not features_available:
            return 0.0
        complementarity = max(0.0, -pair_relation(left_index, right_index))
        importance_gate = math.sqrt(
            max(0.0, normalized_scores[left_index])
            * max(0.0, normalized_scores[right_index])
        )
        return complementarity * importance_gate

    candidate_pair_relations = [
        pair_relation(left, right)
        for left in range(candidate_count)
        for right in range(left + 1, candidate_count)
    ]
    candidate_pair_interactions = [
        pair_interaction_gain(left, right)
        for left in range(candidate_count)
        for right in range(left + 1, candidate_count)
    ]

    greedy_indices = sorted(
        range(candidate_count), key=lambda index: (-raw_scores[index], index)
    )[:selection_size]
    greedy_chromosome = tuple(sorted(greedy_indices))
    greedy_index_set = frozenset(greedy_chromosome)
    greedy_cost = sum(candidate_costs[index] for index in greedy_chromosome)

    def identifier_index_key(index):
        return str(identifiers[index]), index

    def chromosome_identifier_key(chromosome):
        return tuple(sorted(identifier_index_key(index) for index in chromosome))

    def is_valid_chromosome(chromosome):
        return (
            len(chromosome) == selection_size
            and len(set(chromosome)) == selection_size
            and all(
                isinstance(index, int) and 0 <= index < candidate_count
                for index in chromosome
            )
        )

    def evaluate_subset(chromosome):
        importance_reward = (
            sum(normalized_scores[index] for index in chromosome) / len(chromosome)
            if chromosome
            else 0.0
        )
        selected_cost = sum(candidate_costs[index] for index in chromosome)
        normalized_cost_penalty = (
            (selected_cost - minimum_feasible_cost) / feasible_cost_width
            if feasible_cost_width > 0.0
            else 0.0
        )
        normalized_cost_penalty = min(1.0, max(0.0, normalized_cost_penalty))

        redundancies = []
        interactions = []
        for left_position, left_index in enumerate(chromosome):
            for right_index in chromosome[left_position + 1 :]:
                redundancies.append(pair_redundancy(left_index, right_index))
                interactions.append(pair_interaction_gain(left_index, right_index))
        redundancy_score = (
            sum(redundancies) / len(redundancies) if redundancies else 0.0
        )
        interaction_gain = (
            sum(interactions) / len(interactions) if interactions else 0.0
        )
        weighted_interaction_gain = interaction_weight * interaction_gain
        weighted_redundancy_penalty = redundancy_weight * redundancy_score
        weighted_cost_penalty = cost_weight * normalized_cost_penalty
        total_fitness = (
            importance_reward
            + weighted_interaction_gain
            - weighted_redundancy_penalty
            - weighted_cost_penalty
        )
        return {
            "total_fitness": total_fitness,
            "importance_reward": importance_reward,
            "interaction_gain": interaction_gain,
            "weighted_interaction_gain": weighted_interaction_gain,
            "redundancy_score": redundancy_score,
            "weighted_redundancy_penalty": weighted_redundancy_penalty,
            "cost_difference": abs(selected_cost - greedy_cost),
            "normalized_cost_penalty": normalized_cost_penalty,
            "weighted_cost_penalty": weighted_cost_penalty,
            "selected_parameter_cost": selected_cost,
            "greedy_parameter_cost": greedy_cost,
            "minimum_feasible_parameter_cost": minimum_feasible_cost,
            "maximum_feasible_parameter_cost": maximum_feasible_cost,
            "redundancy_features_available": features_available,
            "redundancy_feature_mode": interaction_feature_mode,
            "interaction_features_available": features_available,
            "interaction_feature_mode": interaction_feature_mode,
            "candidate_pair_similarity_min": (
                min(candidate_pair_relations) if candidate_pair_relations else 0.0
            ),
            "candidate_pair_similarity_mean": (
                sum(candidate_pair_relations) / len(candidate_pair_relations)
                if candidate_pair_relations
                else 0.0
            ),
            "candidate_pair_similarity_max": (
                max(candidate_pair_relations) if candidate_pair_relations else 0.0
            ),
            "candidate_pair_interaction_mean": (
                sum(candidate_pair_interactions) / len(candidate_pair_interactions)
                if candidate_pair_interactions
                else 0.0
            ),
            "candidate_pair_interaction_max": (
                max(candidate_pair_interactions) if candidate_pair_interactions else 0.0
            ),
        }

    greedy_components = evaluate_subset(greedy_chromosome)
    fitness_cache = {}
    evaluated_ga_chromosome_sets = set()
    best_non_greedy_chromosome = None
    best_non_greedy_components = None

    def evaluate_ga_chromosome(chromosome):
        nonlocal best_non_greedy_chromosome, best_non_greedy_components
        chromosome = tuple(sorted(chromosome))
        if not is_valid_chromosome(chromosome):
            raise ValueError("GA produced an invalid chromosome.")
        if chromosome not in fitness_cache:
            fitness_cache[chromosome] = evaluate_subset(chromosome)
        components = fitness_cache[chromosome]
        chromosome_set = frozenset(chromosome)
        evaluated_ga_chromosome_sets.add(chromosome_set)
        if chromosome_set != greedy_index_set and (
            best_non_greedy_chromosome is None
            or components["total_fitness"]
            > best_non_greedy_components["total_fitness"] + _FITNESS_TOLERANCE
            or (
                abs(
                    components["total_fitness"]
                    - best_non_greedy_components["total_fitness"]
                )
                <= _FITNESS_TOLERANCE
                and chromosome_identifier_key(chromosome)
                < chromosome_identifier_key(best_non_greedy_chromosome)
            )
        ):
            best_non_greedy_chromosome = chromosome
            best_non_greedy_components = components
        return components

    def fitness(chromosome):
        return evaluate_ga_chromosome(chromosome)["total_fitness"]

    def chromosome_distance(left, right):
        if selection_size == 0:
            return 0.0
        return 1.0 - len(set(left).intersection(right)) / selection_size

    def population_diversity(population):
        if len(population) < 2:
            return 0.0
        distances = [
            chromosome_distance(left, right)
            for left_position, left in enumerate(population)
            for right in population[left_position + 1 :]
        ]
        return sum(distances) / len(distances) if distances else 0.0

    def is_better(left, right):
        left_fitness = fitness(left)
        right_fitness = fitness(right)
        if left_fitness > right_fitness + _FITNESS_TOLERANCE:
            return True
        if right_fitness > left_fitness + _FITNESS_TOLERANCE:
            return False
        return chromosome_identifier_key(left) < chromosome_identifier_key(right)

    def best_chromosome(population):
        best = population[0]
        for chromosome in population[1:]:
            if is_better(chromosome, best):
                best = chromosome
        return best

    one_swap_cache = {}

    def best_single_swap_neighbor(start_chromosome):
        if selection_size <= 0 or selection_size >= candidate_count:
            return None, None, None, None
        start_chromosome = tuple(sorted(start_chromosome))
        if start_chromosome in one_swap_cache:
            return one_swap_cache[start_chromosome]

        selected_set = set(start_chromosome)
        best_neighbor = None
        best_components = None
        best_removed_index = None
        best_added_index = None
        for removed_index in sorted(start_chromosome, key=identifier_index_key):
            for added_index in sorted(
                (index for index in range(candidate_count) if index not in selected_set),
                key=identifier_index_key,
            ):
                neighbor = tuple(
                    sorted(
                        selected_set.difference((removed_index,)).union((added_index,))
                    )
                )
                neighbor_components = evaluate_subset(neighbor)
                if (
                    best_neighbor is None
                    or neighbor_components["total_fitness"]
                    > best_components["total_fitness"] + _FITNESS_TOLERANCE
                    or (
                        abs(
                            neighbor_components["total_fitness"]
                            - best_components["total_fitness"]
                        )
                        <= _FITNESS_TOLERANCE
                        and chromosome_identifier_key(neighbor)
                        < chromosome_identifier_key(best_neighbor)
                    )
                ):
                    best_neighbor = neighbor
                    best_components = neighbor_components
                    best_removed_index = removed_index
                    best_added_index = added_index
        result = (
            best_neighbor,
            best_components,
            best_removed_index,
            best_added_index,
        )
        one_swap_cache[start_chromosome] = result
        return result

    def refine_once(start_chromosome, start_source):
        before_components = evaluate_subset(start_chromosome)
        neighbor, neighbor_components, removed_index, added_index = (
            best_single_swap_neighbor(start_chromosome)
        )
        improved = (
            neighbor_components is not None
            and neighbor_components["total_fitness"]
            > before_components["total_fitness"]
            + _LOCAL_SEARCH_IMPROVEMENT_TOLERANCE
        )
        return {
            "chromosome": neighbor if improved else start_chromosome,
            "start_source": start_source,
            "improved": improved,
            "removed_index": removed_index if improved else None,
            "added_index": added_index if improved else None,
            "fitness_before": before_components["total_fitness"],
            "fitness_after": (
                neighbor_components["total_fitness"]
                if improved
                else before_components["total_fitness"]
            ),
        }

    def diagnostic_components(prefix, components):
        if components is None:
            return {
                prefix + "fitness": None,
                prefix + "importance_reward": None,
                prefix + "interaction_gain": None,
                prefix + "redundancy_score": None,
                prefix + "normalized_cost_penalty": None,
            }
        return {
            prefix + "fitness": components["total_fitness"],
            prefix + "importance_reward": components["importance_reward"],
            prefix + "interaction_gain": components["interaction_gain"],
            prefix + "redundancy_score": components["redundancy_score"],
            prefix + "normalized_cost_penalty": components[
                "normalized_cost_penalty"
            ],
        }

    def finalize_selection(
        ga_best_chromosome,
        initial_population,
        final_population,
        initial_best_fitness,
        diversity_history,
        unique_history,
        generation_best_history,
        initialization_source_counts,
    ):
        ga_best_chromosome = tuple(sorted(ga_best_chromosome))
        ga_best_components = evaluate_subset(ga_best_chromosome)
        selected_chromosome = ga_best_chromosome
        selection_metadata = {
            "start_source": "ga",
            "improved": False,
            "removed_index": None,
            "added_index": None,
            "fitness_before": ga_best_components["total_fitness"],
            "fitness_after": ga_best_components["total_fitness"],
            "selected_source": "ga_only",
        }
        best_single_swap_chromosome = None
        best_single_swap_components = None
        best_single_swap_removed_index = None
        best_single_swap_added_index = None

        if local_search:
            (
                best_single_swap_chromosome,
                best_single_swap_components,
                best_single_swap_removed_index,
                best_single_swap_added_index,
            ) = best_single_swap_neighbor(greedy_chromosome)
            greedy_refinement = refine_once(greedy_chromosome, "greedy")
            ga_refinement = refine_once(ga_best_chromosome, "ga")
            candidates = [(ga_best_chromosome, "ga", None)]
            if greedy_refinement["improved"]:
                candidates.append(
                    (
                        greedy_refinement["chromosome"],
                        "greedy_local_refinement",
                        greedy_refinement,
                    )
                )
            if ga_refinement["improved"]:
                candidates.append(
                    (
                        ga_refinement["chromosome"],
                        "ga_local_refinement",
                        ga_refinement,
                    )
                )
            selected_chromosome, selected_source, selected_refinement = candidates[0]
            for candidate_chromosome, candidate_source, candidate_refinement in candidates[1:]:
                if is_better(candidate_chromosome, selected_chromosome):
                    selected_chromosome = candidate_chromosome
                    selected_source = candidate_source
                    selected_refinement = candidate_refinement
            if selected_refinement is None:
                selected_components = evaluate_subset(selected_chromosome)
                selection_metadata = {
                    "start_source": selected_source,
                    "improved": False,
                    "removed_index": None,
                    "added_index": None,
                    "fitness_before": selected_components["total_fitness"],
                    "fitness_after": selected_components["total_fitness"],
                    "selected_source": selected_source,
                }
            else:
                selection_metadata = dict(selected_refinement)
                selection_metadata["selected_source"] = selected_source

        selected_components = evaluate_subset(selected_chromosome)
        selected_set = set(selected_chromosome)
        greedy_set = set(greedy_chromosome)
        union = selected_set.union(greedy_set)
        greedy_fitness = greedy_components["total_fitness"]
        ga_best_fitness = ga_best_components["total_fitness"]
        diagnostics = dict(selected_components)
        diagnostics.update(
            {
                "candidate_count": candidate_count,
                "selection_size": selection_size,
                "selected_modules": [identifiers[index] for index in selected_chromosome],
                "selected_set_equals_greedy": selected_set == greedy_set,
                "selected_greedy_jaccard": (
                    len(selected_set.intersection(greedy_set)) / len(union)
                    if union
                    else 1.0
                ),
                "selected_non_greedy_count": len(selected_set.difference(greedy_set)),
                "ga_best_modules": [
                    identifiers[index] for index in ga_best_chromosome
                ],
                "ga_best_fitness": ga_best_fitness,
                "ga_best_fitness_minus_greedy": ga_best_fitness - greedy_fitness,
                "ga_beats_greedy": (
                    ga_best_fitness > greedy_fitness + _FITNESS_TOLERANCE
                ),
                "ga_best_equals_greedy": (
                    frozenset(ga_best_chromosome) == greedy_index_set
                ),
                "ga_initial_best_fitness": initial_best_fitness,
                "ga_improved_over_initial": (
                    ga_best_fitness
                    > initial_best_fitness + _FITNESS_TOLERANCE
                ),
                "ga_pre_local_fitness": ga_best_fitness,
                "population_diversity": population_diversity(final_population),
                "population_diversity_initial": population_diversity(
                    initial_population
                ),
                "population_diversity_final": population_diversity(
                    final_population
                ),
                "population_diversity_history": list(diversity_history),
                "unique_population_count_initial": len(set(initial_population)),
                "unique_population_count_final": len(set(final_population)),
                "unique_population_count_history": list(unique_history),
                "evaluated_unique_chromosome_count": len(
                    evaluated_ga_chromosome_sets
                ),
                "generation_best_fitness_history": list(
                    generation_best_history
                ),
                "initialization_source_counts": dict(
                    initialization_source_counts
                ),
                "best_non_greedy_found": best_non_greedy_chromosome is not None,
                "best_non_greedy_modules": (
                    [identifiers[index] for index in best_non_greedy_chromosome]
                    if best_non_greedy_chromosome is not None
                    else []
                ),
                "best_non_greedy_fitness_minus_greedy": (
                    best_non_greedy_components["total_fitness"] - greedy_fitness
                    if best_non_greedy_components is not None
                    else None
                ),
                "local_search_enabled": local_search,
                "local_search_start_source": selection_metadata["start_source"],
                "local_search_improved": selection_metadata["improved"],
                "local_search_removed_module": (
                    identifiers[selection_metadata["removed_index"]]
                    if selection_metadata["removed_index"] is not None
                    else None
                ),
                "local_search_added_module": (
                    identifiers[selection_metadata["added_index"]]
                    if selection_metadata["added_index"] is not None
                    else None
                ),
                "local_search_fitness_before": selection_metadata[
                    "fitness_before"
                ],
                "local_search_fitness_after": selection_metadata["fitness_after"],
                "local_search_fitness_gain": (
                    selection_metadata["fitness_after"]
                    - selection_metadata["fitness_before"]
                ),
                "selected_source": selection_metadata["selected_source"],
                "best_single_swap_modules": (
                    [identifiers[index] for index in best_single_swap_chromosome]
                    if best_single_swap_chromosome is not None
                    else []
                ),
                "best_single_swap_fitness_minus_greedy": (
                    best_single_swap_components["total_fitness"] - greedy_fitness
                    if best_single_swap_components is not None
                    else None
                ),
                "best_single_swap_removed_module": (
                    identifiers[best_single_swap_removed_index]
                    if best_single_swap_removed_index is not None
                    else None
                ),
                "best_single_swap_added_module": (
                    identifiers[best_single_swap_added_index]
                    if best_single_swap_added_index is not None
                    else None
                ),
                "ga_found_best_single_swap": (
                    frozenset(best_single_swap_chromosome)
                    in evaluated_ga_chromosome_sets
                    if best_single_swap_chromosome is not None
                    else None
                ),
                "selected_matches_diagnostic_best_single_swap": (
                    frozenset(selected_chromosome)
                    == frozenset(best_single_swap_chromosome)
                    if best_single_swap_chromosome is not None
                    else None
                ),
            }
        )
        diagnostics.update(diagnostic_components("greedy_", greedy_components))
        diagnostics.update(
            diagnostic_components("best_non_greedy_", best_non_greedy_components)
        )
        diagnostics.update(
            diagnostic_components(
                "best_single_swap_", best_single_swap_components
            )
        )
        return diagnostics["selected_modules"], diagnostics

    evaluate_ga_chromosome(greedy_chromosome)
    if selection_size == 0 or selection_size == candidate_count:
        singleton_population = [greedy_chromosome]
        return finalize_selection(
            greedy_chromosome,
            singleton_population,
            singleton_population,
            greedy_components["total_fitness"],
            [0.0],
            [1],
            [greedy_components["total_fitness"]],
            {"greedy": 1, "greedy_neighbor": 0, "diversity_aware": 0, "random": 0},
        )

    rng = random.Random(seed)

    def repair(chromosome):
        repaired = []
        used = set()
        for index in chromosome:
            if isinstance(index, int) and 0 <= index < candidate_count and index not in used:
                repaired.append(index)
                used.add(index)
                if len(repaired) == selection_size:
                    break
        missing = [index for index in range(candidate_count) if index not in used]
        rng.shuffle(missing)
        repaired.extend(missing[: selection_size - len(repaired)])
        return tuple(sorted(repaired))

    minimum_candidate_cost = min(candidate_costs) if candidate_costs else 0.0
    maximum_candidate_cost = max(candidate_costs) if candidate_costs else 0.0
    candidate_cost_width = maximum_candidate_cost - minimum_candidate_cost

    def marginal_utility(candidate, selected):
        if candidate in selected:
            return float("-inf")
        interaction = (
            sum(pair_interaction_gain(candidate, other) for other in selected)
            / len(selected)
            if selected
            else 0.0
        )
        redundancy = (
            sum(pair_redundancy(candidate, other) for other in selected)
            / len(selected)
            if selected
            else 0.0
        )
        normalized_candidate_cost = (
            (candidate_costs[candidate] - minimum_candidate_cost)
            / candidate_cost_width
            if candidate_cost_width > 0.0
            else 0.0
        )
        return (
            normalized_scores[candidate]
            + interaction_weight * interaction
            - redundancy_weight * redundancy
            - cost_weight * normalized_candidate_cost / max(selection_size, 1)
        )

    def rank_candidates(candidates, selected, frequency=None):
        frequency = frequency or {}
        return sorted(
            candidates,
            key=lambda candidate: (
                -(
                    marginal_utility(candidate, selected)
                    + diversity_weight
                    * (1.0 - frequency.get(candidate, 0.0))
                ),
                identifier_index_key(candidate),
            ),
        )

    def restricted_choice(candidates, selected, frequency=None, exploration=0.15):
        ordered = rank_candidates(candidates, selected, frequency=frequency)
        if len(ordered) == 1 or rng.random() >= exploration:
            return ordered[0]
        restricted_size = min(3, len(ordered))
        return ordered[rng.randrange(restricted_size)]

    def random_neighbor(chromosome, swap_count):
        selected = set(chromosome)
        swap_count = min(swap_count, selection_size, candidate_count - selection_size)
        removed = rng.sample(sorted(selected), swap_count)
        available = [index for index in range(candidate_count) if index not in selected]
        added = rng.sample(available, swap_count)
        return tuple(sorted(selected.difference(removed).union(added)))

    def construct_diversity_aware(existing_population):
        frequency = {}
        denominator = max(len(existing_population), 1)
        for chromosome in existing_population:
            for index in chromosome:
                frequency[index] = frequency.get(index, 0) + 1
        frequency = {
            index: count / denominator for index, count in frequency.items()
        }
        selected = []
        while len(selected) < selection_size:
            candidates = [
                index for index in range(candidate_count) if index not in selected
            ]
            selected.append(
                restricted_choice(
                    candidates,
                    selected,
                    frequency=frequency,
                    exploration=0.35,
                )
            )
        return tuple(sorted(selected))

    population = []
    population_seen = set()
    initialization_source_counts = {
        "greedy": 0,
        "greedy_neighbor": 0,
        "diversity_aware": 0,
        "random": 0,
    }

    def add_initial(chromosome, source):
        chromosome = repair(chromosome)
        if chromosome in population_seen or len(population) >= population_size:
            return False
        population_seen.add(chromosome)
        population.append(chromosome)
        initialization_source_counts[source] += 1
        evaluate_ga_chromosome(chromosome)
        return True

    add_initial(greedy_chromosome, "greedy")
    non_greedy_slots = max(population_size - 1, 0)
    neighbor_target = max(1, non_greedy_slots // 3) if non_greedy_slots else 0
    diverse_target = max(1, non_greedy_slots // 3) if non_greedy_slots else 0

    attempts = 0
    while initialization_source_counts["greedy_neighbor"] < neighbor_target and attempts < population_size * 20:
        attempts += 1
        swap_count = 1 + ((attempts - 1) % min(2, selection_size))
        add_initial(
            random_neighbor(greedy_chromosome, swap_count), "greedy_neighbor"
        )

    attempts = 0
    while initialization_source_counts["diversity_aware"] < diverse_target and attempts < population_size * 20:
        attempts += 1
        add_initial(construct_diversity_aware(population), "diversity_aware")

    attempts = 0
    max_attempts = max(50, population_size * 30)
    while len(population) < population_size and attempts < max_attempts:
        attempts += 1
        add_initial(
            tuple(sorted(rng.sample(range(candidate_count), selection_size))),
            "random",
        )
    while len(population) < population_size:
        chromosome = tuple(sorted(rng.sample(range(candidate_count), selection_size)))
        population.append(chromosome)
        initialization_source_counts["random"] += 1
        evaluate_ga_chromosome(chromosome)

    initial_population = list(population)
    initial_best = best_chromosome(population)
    initial_best_fitness = fitness(initial_best)
    archive_best = initial_best
    diversity_history = [population_diversity(population)]
    unique_history = [len(set(population))]
    generation_best_history = [initial_best_fitness]

    def population_novelty(chromosome, current_population):
        others = [other for other in current_population if other != chromosome]
        if not others:
            return 0.0
        return sum(chromosome_distance(chromosome, other) for other in others) / len(others)

    def select_parent(current_population):
        tournament_size = min(3, len(current_population))
        contestants = rng.sample(current_population, tournament_size)
        return min(
            contestants,
            key=lambda chromosome: (
                -(
                    fitness(chromosome)
                    + diversity_weight
                    * population_novelty(chromosome, current_population)
                ),
                chromosome_identifier_key(chromosome),
            ),
        )

    def crossover(left, right):
        if rng.random() >= crossover_rate:
            return left if rng.random() < 0.5 else right
        shared = set(left).intersection(right)
        parental_union = sorted(set(left).union(right))
        selected = []
        while len(selected) < selection_size:
            candidates = [index for index in parental_union if index not in selected]
            if not candidates:
                candidates = [
                    index for index in range(candidate_count) if index not in selected
                ]
            ordered = sorted(
                candidates,
                key=lambda candidate: (
                    -(
                        marginal_utility(candidate, selected)
                        + (diversity_weight * 0.5 if candidate in shared else 0.0)
                    ),
                    identifier_index_key(candidate),
                ),
            )
            if len(ordered) > 1 and rng.random() < 0.15:
                chosen = ordered[rng.randrange(min(3, len(ordered)))]
            else:
                chosen = ordered[0]
            selected.append(chosen)
        child = tuple(sorted(selected))
        evaluate_ga_chromosome(child)
        return child

    def mutate(chromosome, effective_mutation_rate):
        child = set(chromosome)
        mutation_count = sum(
            1 for _ in chromosome if rng.random() < effective_mutation_rate
        )
        if mutation_count == 0:
            evaluate_ga_chromosome(chromosome)
            return chromosome
        mutation_count = min(
            mutation_count, selection_size, candidate_count - selection_size
        )
        for _ in range(mutation_count):
            selected = sorted(child)
            removal_order = sorted(
                selected,
                key=lambda candidate: (
                    marginal_utility(
                        candidate, [other for other in selected if other != candidate]
                    ),
                    identifier_index_key(candidate),
                ),
            )
            removed = (
                removal_order[rng.randrange(min(2, len(removal_order)))]
                if len(removal_order) > 1 and rng.random() < 0.20
                else removal_order[0]
            )
            child.remove(removed)
            candidates = [index for index in range(candidate_count) if index not in child]
            added = restricted_choice(
                candidates,
                sorted(child),
                exploration=0.20,
            )
            child.add(added)
        mutated = tuple(sorted(child))
        evaluate_ga_chromosome(mutated)
        return mutated

    def environmental_selection(candidate_population):
        unique_candidates = list(dict.fromkeys(candidate_population))
        if len(unique_candidates) <= population_size:
            selected = list(unique_candidates)
            while len(selected) < population_size:
                selected.append(unique_candidates[len(selected) % len(unique_candidates)])
            return selected

        candidate_fitnesses = [fitness(chromosome) for chromosome in unique_candidates]
        lowest_fitness = min(candidate_fitnesses)
        highest_fitness = max(candidate_fitnesses)
        fitness_width = highest_fitness - lowest_fitness

        selected = [best_chromosome(unique_candidates)]
        remaining = [
            chromosome for chromosome in unique_candidates if chromosome not in selected
        ]
        while len(selected) < population_size and remaining:
            chosen = min(
                remaining,
                key=lambda chromosome: (
                    -(
                        (
                            (fitness(chromosome) - lowest_fitness) / fitness_width
                            if fitness_width > 0.0
                            else 0.0
                        )
                        + diversity_weight
                        * min(
                            chromosome_distance(chromosome, survivor)
                            for survivor in selected
                        )
                    ),
                    -fitness(chromosome),
                    chromosome_identifier_key(chromosome),
                ),
            )
            selected.append(chosen)
            remaining.remove(chosen)
        return selected

    for _ in range(generations):
        current_diversity = population_diversity(population)
        diversity_deficit = max(0.0, 0.60 - current_diversity)
        effective_mutation_rate = min(
            0.50, mutation_rate * (1.0 + 2.0 * diversity_deficit)
        )
        offspring = []
        offspring_seen = set()
        attempts = 0
        max_attempts = max(50, population_size * 20)
        while len(offspring) < population_size and attempts < max_attempts:
            attempts += 1
            child = mutate(
                crossover(select_parent(population), select_parent(population)),
                effective_mutation_rate,
            )
            if child not in offspring_seen:
                offspring_seen.add(child)
                offspring.append(child)
        while len(offspring) < population_size:
            offspring.append(
                tuple(sorted(rng.sample(range(candidate_count), selection_size)))
            )
            evaluate_ga_chromosome(offspring[-1])

        population = environmental_selection(population + offspring)
        generation_best = best_chromosome(population)
        if is_better(generation_best, archive_best):
            archive_best = generation_best
        diversity_history.append(population_diversity(population))
        unique_history.append(len(set(population)))
        generation_best_history.append(fitness(generation_best))

    return finalize_selection(
        archive_best,
        initial_population,
        population,
        initial_best_fitness,
        diversity_history,
        unique_history,
        generation_best_history,
        initialization_source_counts,
    )
