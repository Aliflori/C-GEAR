# coding=utf-8
"""Lightweight fixed-cardinality genetic allocation for EvoIncreLoRA."""

import math
import random


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
            # Keep the first occurrence so malformed duplicate input is deterministic.
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


def _centered_temporal_features(identifiers, module_features, max_window=20, epsilon=1e-12):
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
    similarity = dot / (left_norm * right_norm)
    # Redundancy penalizes positive similarity, not opposite feature directions.
    return min(1.0, max(0.0, similarity))


def _validate_configuration(
    population_size,
    generations,
    mutation_rate,
    crossover_rate,
    redundancy_weight,
    cost_weight,
):
    if population_size <= 0:
        raise ValueError("ga_population must be positive.")
    if generations < 0:
        raise ValueError("ga_generations must be nonnegative.")
    if not math.isfinite(mutation_rate) or mutation_rate < 0.0 or mutation_rate > 1.0:
        raise ValueError("ga_mutation_rate must be between 0 and 1.")
    if not math.isfinite(crossover_rate) or crossover_rate < 0.0 or crossover_rate > 1.0:
        raise ValueError("ga_crossover_rate must be between 0 and 1.")
    if not math.isfinite(redundancy_weight) or redundancy_weight < 0.0:
        raise ValueError("ga_redundancy_weight must be nonnegative.")
    if not math.isfinite(cost_weight) or cost_weight < 0.0:
        raise ValueError("ga_cost_weight must be nonnegative.")


def select_modules_genetic(
    scores,
    costs,
    top_h,
    population_size=12,
    generations=4,
    mutation_rate=0.10,
    crossover_rate=0.80,
    redundancy_weight=0.20,
    cost_weight=0.30,
    seed=0,
    module_features=None,
):
    """Select a deterministic, fixed-size module set with a lightweight GA.

    ``scores`` may be a mapping or an ordered iterable of ``(identifier, score)``
    pairs. Invalid and non-finite numerical inputs are replaced with zero. The
    first occurrence of a duplicate identifier is retained.
    """

    population_size = int(population_size)
    generations = int(generations)
    mutation_rate = float(mutation_rate)
    crossover_rate = float(crossover_rate)
    redundancy_weight = float(redundancy_weight)
    cost_weight = float(cost_weight)
    _validate_configuration(
        population_size,
        generations,
        mutation_rate,
        crossover_rate,
        redundancy_weight,
        cost_weight,
    )

    score_items = _deduplicated_scores(scores)
    identifiers = [identifier for identifier, _ in score_items]
    raw_scores = [score for _, score in score_items]
    candidate_count = len(identifiers)
    selection_size = min(max(int(top_h), 0), candidate_count)

    cost_map = _value_map(costs, lambda value: max(0.0, _finite_float(value)))
    candidate_costs = [cost_map.get(identifier, 0.0) for identifier in identifiers]
    normalized_scores = _normalized_scores(raw_scores)

    temporal_features = _centered_temporal_features(identifiers, module_features)
    features_available = len(temporal_features) == candidate_count and candidate_count >= 2
    structural_signatures = [_structural_signature(identifier) for identifier in identifiers]

    if features_available:
        redundancy_feature_mode = "temporal_centered"

        def pair_similarity(left_index, right_index):
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
            redundancy_feature_mode = "structural_fallback"

            def pair_similarity(left_index, right_index):
                return _structural_similarity(
                    structural_signatures[left_index], structural_signatures[right_index]
                )

        else:
            redundancy_feature_mode = "unavailable"

            def pair_similarity(left_index, right_index):
                return 0.0

    candidate_pair_similarities = [
        pair_similarity(left, right)
        for left in range(candidate_count)
        for right in range(left + 1, candidate_count)
    ]
    candidate_pair_similarity_min = (
        min(candidate_pair_similarities) if candidate_pair_similarities else 0.0
    )
    candidate_pair_similarity_mean = (
        sum(candidate_pair_similarities) / len(candidate_pair_similarities)
        if candidate_pair_similarities
        else 0.0
    )
    candidate_pair_similarity_max = (
        max(candidate_pair_similarities) if candidate_pair_similarities else 0.0
    )

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
        cost_difference = abs(selected_cost - greedy_cost)
        normalized_cost_penalty = cost_difference / max(greedy_cost, 1.0)

        similarities = []
        for left_position, left_index in enumerate(chromosome):
            for right_index in chromosome[left_position + 1 :]:
                similarities.append(pair_similarity(left_index, right_index))
        redundancy_score = (
            sum(similarities) / len(similarities) if similarities else 0.0
        )
        weighted_redundancy_penalty = redundancy_weight * redundancy_score
        weighted_cost_penalty = cost_weight * normalized_cost_penalty
        total_fitness = (
            importance_reward
            - weighted_redundancy_penalty
            - weighted_cost_penalty
        )
        return {
            "total_fitness": total_fitness,
            "importance_reward": importance_reward,
            "redundancy_score": redundancy_score,
            "weighted_redundancy_penalty": weighted_redundancy_penalty,
            "cost_difference": cost_difference,
            "normalized_cost_penalty": normalized_cost_penalty,
            "weighted_cost_penalty": weighted_cost_penalty,
            "selected_parameter_cost": selected_cost,
            "greedy_parameter_cost": greedy_cost,
            "redundancy_features_available": features_available,
            "redundancy_feature_mode": redundancy_feature_mode,
            "candidate_pair_similarity_min": candidate_pair_similarity_min,
            "candidate_pair_similarity_mean": candidate_pair_similarity_mean,
            "candidate_pair_similarity_max": candidate_pair_similarity_max,
        }

    greedy_components = evaluate_subset(greedy_chromosome)
    fitness_cache = {}
    evaluated_ga_chromosome_sets = set()
    best_non_greedy_chromosome = None
    best_non_greedy_components = None

    def evaluate_ga_chromosome(chromosome):
        nonlocal best_non_greedy_chromosome, best_non_greedy_components
        if not is_valid_chromosome(chromosome):
            raise ValueError("GA produced an invalid chromosome.")

        if chromosome not in fitness_cache:
            fitness_cache[chromosome] = evaluate_subset(chromosome)
        subset_components = fitness_cache[chromosome]
        chromosome_set = frozenset(chromosome)
        evaluated_ga_chromosome_sets.add(chromosome_set)

        if chromosome_set != greedy_index_set and (
            best_non_greedy_chromosome is None
            or subset_components["total_fitness"]
            > best_non_greedy_components["total_fitness"]
            or (
                subset_components["total_fitness"]
                == best_non_greedy_components["total_fitness"]
                and chromosome_identifier_key(chromosome)
                < chromosome_identifier_key(best_non_greedy_chromosome)
            )
        ):
            best_non_greedy_chromosome = chromosome
            best_non_greedy_components = subset_components
        return subset_components

    one_swap_cache = {}

    def best_single_swap_neighbor(start_chromosome):
        if selection_size <= 0 or selection_size >= candidate_count:
            return None, None, None, None

        start_chromosome = tuple(sorted(start_chromosome))
        if not is_valid_chromosome(start_chromosome):
            raise ValueError("Local search received an invalid chromosome.")
        if start_chromosome in one_swap_cache:
            return one_swap_cache[start_chromosome]

        selected_set = set(start_chromosome)
        best_chromosome = None
        best_components = None
        best_removed_index = None
        best_added_index = None
        removed_indices = sorted(start_chromosome, key=identifier_index_key)
        added_indices = sorted(
            (index for index in range(candidate_count) if index not in selected_set),
            key=identifier_index_key,
        )
        for removed_index in removed_indices:
            for added_index in added_indices:
                neighbor = tuple(
                    sorted(
                        selected_set.difference((removed_index,)).union((added_index,))
                    )
                )
                # Fixed cardinality is the allocator's feasibility/budget contract.
                if not is_valid_chromosome(neighbor):
                    continue
                neighbor_components = evaluate_subset(neighbor)
                if (
                    best_chromosome is None
                    or neighbor_components["total_fitness"]
                    > best_components["total_fitness"]
                    or (
                        neighbor_components["total_fitness"]
                        == best_components["total_fitness"]
                        and chromosome_identifier_key(neighbor)
                        < chromosome_identifier_key(best_chromosome)
                    )
                ):
                    best_chromosome = neighbor
                    best_components = neighbor_components
                    best_removed_index = removed_index
                    best_added_index = added_index
        result = (
            best_chromosome,
            best_components,
            best_removed_index,
            best_added_index,
        )
        one_swap_cache[start_chromosome] = result
        return result

    def refine_once(start_chromosome, start_source):
        before_components = evaluate_subset(start_chromosome)
        (
            neighbor,
            neighbor_components,
            removed_index,
            added_index,
        ) = best_single_swap_neighbor(start_chromosome)
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

    def diagnostic_components(prefix, subset_components):
        if subset_components is None:
            return {
                prefix + "fitness": None,
                prefix + "importance_reward": None,
                prefix + "redundancy_score": None,
                prefix + "normalized_cost_penalty": None,
            }
        return {
            prefix + "fitness": subset_components["total_fitness"],
            prefix + "importance_reward": subset_components["importance_reward"],
            prefix + "redundancy_score": subset_components["redundancy_score"],
            prefix + "normalized_cost_penalty": subset_components[
                "normalized_cost_penalty"
            ],
        }

    def diagnostics(
        chromosome,
        initial_unique_count,
        final_unique_count,
        ga_pre_local_chromosome,
        selection_metadata,
    ):
        result = evaluate_subset(chromosome)
        selected_set = set(chromosome)
        greedy_set = set(greedy_chromosome)
        union = selected_set.union(greedy_set)
        (
            best_single_swap_chromosome,
            best_single_swap_components,
            best_single_swap_removed_index,
            best_single_swap_added_index,
        ) = best_single_swap_neighbor(greedy_chromosome)
        greedy_fitness = greedy_components["total_fitness"]
        ga_pre_local_fitness = evaluate_subset(ga_pre_local_chromosome)[
            "total_fitness"
        ]
        result.update(
            {
                "candidate_count": candidate_count,
                "selection_size": selection_size,
                "selected_set_equals_greedy": selected_set == greedy_set,
                "selected_greedy_jaccard": (
                    len(selected_set.intersection(greedy_set)) / len(union)
                    if union
                    else 1.0
                ),
                "selected_non_greedy_count": len(selected_set.difference(greedy_set)),
                "unique_population_count_initial": initial_unique_count,
                "unique_population_count_final": final_unique_count,
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
                    else False
                ),
                "ga_pre_local_fitness": ga_pre_local_fitness,
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
                "selected_matches_diagnostic_best_single_swap": (
                    frozenset(chromosome) == frozenset(best_single_swap_chromosome)
                    if best_single_swap_chromosome is not None
                    else None
                ),
            }
        )
        result.update(diagnostic_components("greedy_", greedy_components))
        result.update(
            diagnostic_components("best_non_greedy_", best_non_greedy_components)
        )
        result.update(
            diagnostic_components("best_single_swap_", best_single_swap_components)
        )
        return result

    def finalize_selection(ga_pre_local_chromosome, initial_unique_count, final_unique_count):
        ga_pre_local_chromosome = tuple(sorted(ga_pre_local_chromosome))
        ga_pre_local_is_greedy = (
            frozenset(ga_pre_local_chromosome) == greedy_index_set
        )
        greedy_refinement = refine_once(greedy_chromosome, "greedy")
        ga_refinement = (
            greedy_refinement
            if ga_pre_local_is_greedy
            else refine_once(ga_pre_local_chromosome, "ga")
        )

        candidates = [(greedy_chromosome, "greedy", None)]
        if not ga_pre_local_is_greedy:
            candidates.append((ga_pre_local_chromosome, "ga", None))
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

        source_priority = {
            "greedy": 0,
            "ga": 1,
            "greedy_local_refinement": 2,
            "ga_local_refinement": 3,
        }
        selected_chromosome, selected_source, selected_refinement = min(
            candidates,
            key=lambda candidate: (
                -evaluate_subset(candidate[0])["total_fitness"],
                chromosome_identifier_key(candidate[0]),
                source_priority[candidate[1]],
            ),
        )

        if selected_refinement is None:
            selected_fitness = evaluate_subset(selected_chromosome)["total_fitness"]
            selection_metadata = {
                "start_source": selected_source,
                "improved": False,
                "removed_index": None,
                "added_index": None,
                "fitness_before": selected_fitness,
                "fitness_after": selected_fitness,
                "selected_source": selected_source,
            }
        else:
            selection_metadata = dict(selected_refinement)
            selection_metadata["selected_source"] = selected_source

        selected_modules = [identifiers[index] for index in selected_chromosome]
        final_diagnostics = diagnostics(
            selected_chromosome,
            initial_unique_count,
            final_unique_count,
            ga_pre_local_chromosome,
            selection_metadata,
        )
        return selected_modules, final_diagnostics

    if selection_size == 0:
        evaluate_ga_chromosome(greedy_chromosome)
        return finalize_selection(greedy_chromosome, 1, 1)

    if selection_size == candidate_count:
        evaluate_ga_chromosome(greedy_chromosome)
        return finalize_selection(greedy_chromosome, 1, 1)

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

    def crossover(left, right):
        if rng.random() >= crossover_rate:
            child = left if rng.random() < 0.5 else right
        else:
            shared = sorted(set(left).intersection(right))
            remaining = sorted(set(left).union(right).difference(shared))
            rng.shuffle(remaining)
            child = repair(shared + remaining)
        evaluate_ga_chromosome(child)
        return child

    def mutate(chromosome):
        if rng.random() >= mutation_rate or selection_size >= candidate_count:
            evaluate_ga_chromosome(chromosome)
            return chromosome
        child = list(chromosome)
        replace_position = rng.randrange(selection_size)
        unselected = [
            index for index in range(candidate_count) if index not in chromosome
        ]
        child[replace_position] = rng.choice(unselected)
        mutated = repair(child)
        evaluate_ga_chromosome(mutated)
        return mutated

    def fitness(chromosome):
        return evaluate_ga_chromosome(chromosome)["total_fitness"]

    def ranked(population):
        # Python's stable sort retains population order for equal fitness.
        return sorted(population, key=lambda chromosome: -fitness(chromosome))

    def select_parent(population):
        left = population[rng.randrange(len(population))]
        right = population[rng.randrange(len(population))]
        if fitness(left) == fitness(right):
            return left
        return left if fitness(left) > fitness(right) else right

    population = [greedy_chromosome]
    evaluate_ga_chromosome(greedy_chromosome)
    seen = {greedy_chromosome}
    attempts = 0
    max_attempts = max(20, population_size * 10)
    while len(population) < population_size and attempts < max_attempts:
        attempts += 1
        chromosome = tuple(sorted(rng.sample(range(candidate_count), selection_size)))
        if chromosome not in seen:
            seen.add(chromosome)
            population.append(chromosome)
            evaluate_ga_chromosome(chromosome)
    while len(population) < population_size:
        chromosome = tuple(sorted(rng.sample(range(candidate_count), selection_size)))
        population.append(chromosome)
        evaluate_ga_chromosome(chromosome)

    initial_unique_count = len(set(population))
    best = ranked(population)[0]
    for _ in range(generations):
        ordered_population = ranked(population)
        elite_count = min(2, max(1, len(ordered_population) // 4))
        next_population = list(ordered_population[:elite_count])
        next_seen = set(next_population)
        duplicate_attempts = 0
        max_duplicate_attempts = max(20, population_size * 10)
        while len(next_population) < population_size:
            left = select_parent(ordered_population)
            right = select_parent(ordered_population)
            child = mutate(crossover(left, right))
            if child in next_seen and duplicate_attempts < max_duplicate_attempts:
                duplicate_attempts += 1
                continue
            next_population.append(child)
            next_seen.add(child)
        population = next_population
        generation_best = ranked(population)[0]
        if fitness(generation_best) > fitness(best):
            best = generation_best

    final_ordered_population = ranked(population)
    if fitness(final_ordered_population[0]) > fitness(best):
        best = final_ordered_population[0]
    return finalize_selection(best, initial_unique_count, len(set(population)))
