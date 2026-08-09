# coding=utf-8
"""Hard-budget, training-aware evolutionary selection for IncreLoRA.

This module is intentionally separate from ``evo_allocator`` so the published
Greedy and earlier GA/memetic ablations retain their existing behavior.
"""

import math
import random

from .evo_allocator import (
    _centered_temporal_features,
    _cosine_similarity,
    _deduplicated_scores,
    _finite_float,
    _normalized_scores,
    _structural_signature,
    _structural_similarity,
    _value_map,
)


_TOLERANCE = 1e-12


def expected_optimizer_gain(module, optimizer, epsilon=None):
    """Estimate next-rank utility from training gradients and Adam moments.

    The prepared final A/E/B entries are the next reserve component. For every
    available gradient element this computes ``g^2 / (sqrt(v) + eps)`` when an
    Adam ``exp_avg_sq`` tensor is present, and deterministically falls back to
    ``g^2`` otherwise. It is a proxy, not a measured loss decrease.
    """

    if epsilon is None:
        epsilon = float(getattr(optimizer, "defaults", {}).get("eps", 1e-8))
    epsilon = max(_finite_float(epsilon, 1e-8), 1e-30)
    raw_gain = 0.0
    optimizer_elements = 0
    fallback_elements = 0
    gradient_elements = 0

    for parameter in (module.lora_A[-1], module.lora_E[-1], module.lora_B[-1]):
        gradient = parameter.grad
        if gradient is None:
            continue
        gradient = gradient.detach().float()
        finite = gradient.isfinite()
        if not bool(finite.any()):
            continue
        squared = gradient[finite].pow(2)
        gradient_elements += int(squared.numel())
        state = optimizer.state.get(parameter, {}) if optimizer is not None else {}
        second_moment = state.get("exp_avg_sq")
        if second_moment is not None and tuple(second_moment.shape) == tuple(parameter.shape):
            denominator = second_moment.detach().float()[finite].clamp_min(0.0).sqrt().add(epsilon)
            raw_gain += float((squared / denominator).sum().item())
            optimizer_elements += int(squared.numel())
        else:
            raw_gain += float(squared.sum().item())
            fallback_elements += int(squared.numel())

    if not math.isfinite(raw_gain):
        raw_gain = 0.0
    return {
        "raw_gain": raw_gain,
        "gradient_elements": gradient_elements,
        "optimizer_moment_elements": optimizer_elements,
        "fallback_gradient_elements": fallback_elements,
        "signal_available": gradient_elements > 0,
    }


def validate_budget_metadata(expected, loaded):
    """Reject a resumed/checkpoint configuration with a different hard budget."""

    keys = ("allocator_mode", "reference_cost", "target_cost", "budget_ratio")
    if not isinstance(loaded, dict):
        raise ValueError("Checkpoint does not contain budgeted allocator metadata.")
    mismatches = {
        key: (expected.get(key), loaded.get(key))
        for key in keys
        if expected.get(key) != loaded.get(key)
    }
    if mismatches:
        raise ValueError("Incompatible budgeted allocator checkpoint metadata: %s" % mismatches)
    return True


def _canonical(candidate, identifiers):
    return tuple(sorted(str(identifiers[index]) for index in candidate))


def _population_diversity(population, selection_size):
    if len(population) < 2 or selection_size <= 0:
        return 0.0
    distances = [
        1.0 - len(set(left).intersection(right)) / float(selection_size)
        for position, left in enumerate(population)
        for right in population[position + 1 :]
    ]
    return sum(distances) / len(distances) if distances else 0.0


def _dominates(left, right):
    no_worse = (
        left["gain_per_parameter"] >= right["gain_per_parameter"] - _TOLERANCE
        and left["structural_fitness"] >= right["structural_fitness"] - _TOLERANCE
        and left["actual_cost"] <= right["actual_cost"] + _TOLERANCE
    )
    strictly_better = (
        left["gain_per_parameter"] > right["gain_per_parameter"] + _TOLERANCE
        or left["structural_fitness"] > right["structural_fitness"] + _TOLERANCE
        or left["actual_cost"] < right["actual_cost"] - _TOLERANCE
    )
    return no_worse and strictly_better


def _pareto_fronts(chromosomes, evaluations):
    remaining = list(dict.fromkeys(chromosomes))
    fronts = []
    while remaining:
        front = [
            candidate
            for candidate in remaining
            if not any(
                other != candidate
                and _dominates(evaluations[other], evaluations[candidate])
                for other in remaining
            )
        ]
        fronts.append(front)
        front_set = set(front)
        remaining = [candidate for candidate in remaining if candidate not in front_set]
    return fronts


def select_quality_preserving_candidate(candidates, gain_tolerance):
    """Apply the documented quality guard and deterministic cost-first rule."""

    if not candidates:
        raise ValueError("No budget-feasible evolutionary candidates are available.")
    tolerance = float(gain_tolerance)
    if not math.isfinite(tolerance) or tolerance < 0.0 or tolerance > 1.0:
        raise ValueError("ga_gain_tolerance must be between 0 and 1.")

    efficiencies = [max(0.0, _finite_float(item["gain_per_parameter"])) for item in candidates]
    low = min(efficiencies)
    high = max(efficiencies)
    width = high - low
    for item, efficiency in zip(candidates, efficiencies):
        item["normalized_training_gain"] = (
            (efficiency - low) / width if width > _TOLERANCE else 1.0
        )
    best_gain = max(item["normalized_training_gain"] for item in candidates)
    quality_set = [
        item
        for item in candidates
        if item["normalized_training_gain"] >= best_gain - tolerance - _TOLERANCE
    ]
    selected = min(
        quality_set,
        key=lambda item: (
            item["actual_cost"],
            -item["raw_training_gain"],
            -item["structural_fitness"],
            item["canonical_modules"],
        ),
    )
    return selected, quality_set


def select_modules_budgeted(
    scores,
    costs,
    training_gains,
    top_h,
    current_active_cost,
    target_final_cost,
    future_rank_increments,
    rank_increment=1,
    population_size=12,
    generations=4,
    mutation_rate=0.10,
    crossover_rate=0.80,
    interaction_weight=0.20,
    redundancy_weight=0.20,
    diversity_weight=0.10,
    gain_tolerance=0.05,
    seed=0,
    module_features=None,
):
    """Run fixed-cardinality multiobjective evolution under a hard final budget."""

    population_size = int(population_size)
    generations = int(generations)
    selection_size = int(top_h)
    rank_increment = int(rank_increment)
    future_rank_increments = int(future_rank_increments)
    if population_size <= 0 or generations < 0:
        raise ValueError("GA population must be positive and generations nonnegative.")
    if not 0.0 <= float(mutation_rate) <= 1.0:
        raise ValueError("ga_mutation_rate must be between 0 and 1.")
    if not 0.0 <= float(crossover_rate) <= 1.0:
        raise ValueError("ga_crossover_rate must be between 0 and 1.")
    if selection_size <= 0 or rank_increment <= 0 or future_rank_increments < 0:
        raise ValueError("Budgeted allocation requires positive fixed event rank.")

    score_items = _deduplicated_scores(scores)
    identifiers = [identifier for identifier, _ in score_items]
    raw_scores = [score for _, score in score_items]
    candidate_count = len(identifiers)
    if selection_size > candidate_count:
        raise ValueError("top_h exceeds the number of eligible LoRA modules.")
    score_values = _normalized_scores(raw_scores)
    cost_map = _value_map(costs, lambda value: int(max(0, _finite_float(value))))
    gain_map = _value_map(training_gains, lambda value: max(0.0, _finite_float(value)))
    unit_costs = [cost_map.get(identifier, 0) for identifier in identifiers]
    raw_gains = [gain_map.get(identifier, 0.0) for identifier in identifiers]
    if any(cost <= 0 for cost in unit_costs):
        raise ValueError("Every eligible module must have a positive exact rank-one cost.")

    minimum_future_cost = future_rank_increments * min(unit_costs)
    maximum_event_cost = int(target_final_cost) - int(current_active_cost) - minimum_future_cost
    cheapest_event_cost = sum(sorted(unit_costs)[:selection_size]) * rank_increment
    if cheapest_event_cost > maximum_event_cost:
        raise ValueError(
            "No fixed-cardinality allocation is feasible: current_cost=%s target_cost=%s "
            "event_size=%s cheapest_event_cost=%s future_rank_increments=%s "
            "minimum_future_cost=%s projected_minimum_final_cost=%s"
            % (
                current_active_cost,
                target_final_cost,
                selection_size,
                cheapest_event_cost,
                future_rank_increments,
                minimum_future_cost,
                int(current_active_cost) + cheapest_event_cost + minimum_future_cost,
            )
        )

    temporal = _centered_temporal_features(identifiers, module_features)
    temporal_available = len(temporal) == candidate_count and candidate_count >= 2
    signatures = [_structural_signature(identifier) for identifier in identifiers]

    def relation(left, right):
        if temporal_available:
            return _cosine_similarity(temporal[left], temporal[right])
        return _structural_similarity(signatures[left], signatures[right])

    def event_cost(chromosome):
        return int(sum(unit_costs[index] for index in chromosome) * rank_increment)

    def projected_final(chromosome):
        return int(current_active_cost) + event_cost(chromosome) + minimum_future_cost

    def feasible(chromosome):
        return projected_final(chromosome) <= int(target_final_cost)

    evaluations = {}
    evaluated = set()

    def evaluate(chromosome):
        chromosome = tuple(sorted(chromosome))
        if chromosome not in evaluations:
            importance = sum(score_values[index] for index in chromosome) / selection_size
            redundancies = []
            interactions = []
            for position, left in enumerate(chromosome):
                for right in chromosome[position + 1 :]:
                    pair = relation(left, right)
                    redundancies.append(max(0.0, pair))
                    if temporal_available:
                        gate = math.sqrt(max(0.0, score_values[left] * score_values[right]))
                        interactions.append(max(0.0, -pair) * gate)
            redundancy = sum(redundancies) / len(redundancies) if redundancies else 0.0
            interaction = sum(interactions) / len(interactions) if interactions else 0.0
            structural = importance + float(interaction_weight) * interaction - float(redundancy_weight) * redundancy
            cost = event_cost(chromosome)
            gain = sum(raw_gains[index] for index in chromosome)
            evaluations[chromosome] = {
                "chromosome": chromosome,
                "modules": [identifiers[index] for index in chromosome],
                "canonical_modules": _canonical(chromosome, identifiers),
                "actual_cost": cost,
                "raw_training_gain": gain,
                "gain_per_parameter": gain / cost if cost else 0.0,
                "structural_fitness": structural,
                "importance_reward": importance,
                "interaction_gain": interaction,
                "redundancy_score": redundancy,
                "projected_minimum_final_cost": projected_final(chromosome),
                "budget_feasible": feasible(chromosome),
            }
        evaluated.add(chromosome)
        return evaluations[chromosome]

    repair_count = 0
    infeasible_count = 0

    def repair(chromosome):
        nonlocal repair_count, infeasible_count
        chromosome = tuple(sorted(set(chromosome)))
        selected = set(chromosome)
        if len(selected) > selection_size:
            selected = set(
                sorted(
                    selected,
                    key=lambda index: (
                        -raw_gains[index] / unit_costs[index],
                        -score_values[index],
                        unit_costs[index],
                        str(identifiers[index]),
                    ),
                )[:selection_size]
            )
        while len(selected) < selection_size:
            candidates = [index for index in range(candidate_count) if index not in selected]
            selected.add(
                min(
                    candidates,
                    key=lambda index: (
                        unit_costs[index],
                        -raw_gains[index],
                        -score_values[index],
                        str(identifiers[index]),
                    ),
                )
            )
        repaired = tuple(sorted(selected))
        if feasible(repaired):
            return repaired

        infeasible_count += 1
        repair_count += 1
        while not feasible(repaired):
            selected = set(repaired)
            best_swap = None
            for removed in selected:
                for added in range(candidate_count):
                    if added in selected or unit_costs[added] >= unit_costs[removed]:
                        continue
                    trial = tuple(sorted(selected.difference((removed,)).union((added,))))
                    key = (
                        event_cost(trial),
                        -raw_gains[added] / unit_costs[added],
                        -score_values[added],
                        str(identifiers[removed]),
                        str(identifiers[added]),
                    )
                    if best_swap is None or key < best_swap[0]:
                        best_swap = (key, trial)
            if best_swap is None:
                repaired = tuple(
                    sorted(
                        sorted(
                            range(candidate_count),
                            key=lambda index: (unit_costs[index], str(identifiers[index])),
                        )[:selection_size]
                    )
                )
                break
            repaired = best_swap[1]
        if not feasible(repaired):
            raise ValueError("Deterministic budget repair could not construct a feasible chromosome.")
        return repaired

    greedy = tuple(
        sorted(
            sorted(range(candidate_count), key=lambda index: (-raw_scores[index], index))[:selection_size]
        )
    )
    greedy_feasible = feasible(greedy)
    greedy_reference = greedy if greedy_feasible else repair(greedy)
    cheapest = tuple(
        sorted(
            sorted(range(candidate_count), key=lambda index: (unit_costs[index], str(identifiers[index])))[:selection_size]
        )
    )

    rng = random.Random(int(seed))
    population = []
    population_seen = set()

    def add_candidate(chromosome):
        chromosome = repair(chromosome)
        evaluate(chromosome)
        if chromosome in population_seen:
            return False
        population_seen.add(chromosome)
        population.append(chromosome)
        return True

    add_candidate(greedy_reference)
    add_candidate(cheapest)
    attempts = 0
    max_attempts = max(100, population_size * 40)
    while len(population) < population_size and attempts < max_attempts:
        attempts += 1
        if attempts <= max(1, population_size // 3):
            base = set(greedy_reference)
            removed = rng.choice(sorted(base))
            base.remove(removed)
            base.add(rng.choice([index for index in range(candidate_count) if index not in base]))
            candidate = tuple(sorted(base))
        else:
            candidate = tuple(sorted(rng.sample(range(candidate_count), selection_size)))
        add_candidate(candidate)
    if not population:
        raise ValueError("No feasible initial budgeted GA population could be constructed.")

    initial_population = list(population)
    diversity_history = [_population_diversity(population, selection_size)]
    unique_history = [len(set(population))]

    def ranked_population(candidates):
        candidates = list(dict.fromkeys(candidates))
        fronts = _pareto_fronts(candidates, evaluations)
        result = []
        for front in fronts:
            ordered = sorted(
                front,
                key=lambda chromosome: (
                    -evaluations[chromosome]["structural_fitness"],
                    -evaluations[chromosome]["gain_per_parameter"],
                    evaluations[chromosome]["actual_cost"],
                    evaluations[chromosome]["canonical_modules"],
                ),
            )
            while ordered and len(result) < population_size:
                if not result:
                    chosen = ordered[0]
                else:
                    chosen = max(
                        ordered,
                        key=lambda chromosome: (
                            min(
                                1.0 - len(set(chromosome).intersection(other)) / selection_size
                                for other in result
                            )
                            * float(diversity_weight),
                            evaluations[chromosome]["structural_fitness"],
                            evaluations[chromosome]["gain_per_parameter"],
                            -evaluations[chromosome]["actual_cost"],
                        ),
                    )
                result.append(chosen)
                ordered.remove(chosen)
            if len(result) >= population_size:
                break
        return result

    def choose_parent():
        if len(population) == 1:
            return population[0]
        contestants = rng.sample(population, min(3, len(population)))
        return ranked_population(contestants)[0]

    def crossover(left, right):
        if rng.random() >= float(crossover_rate):
            return left if rng.random() < 0.5 else right
        union = sorted(set(left).union(right))
        selected = set(index for index in left if index in right)
        while len(selected) < selection_size:
            choices = [index for index in union if index not in selected]
            if not choices:
                choices = [index for index in range(candidate_count) if index not in selected]
            choices = sorted(
                choices,
                key=lambda index: (
                    -raw_gains[index] / unit_costs[index],
                    -score_values[index],
                    unit_costs[index],
                    str(identifiers[index]),
                ),
            )
            selected.add(choices[rng.randrange(min(3, len(choices)))])
        return repair(tuple(sorted(selected)))

    def mutate(chromosome):
        selected = set(chromosome)
        for original in list(chromosome):
            if rng.random() >= float(mutation_rate):
                continue
            selected.remove(original)
            choices = [index for index in range(candidate_count) if index not in selected]
            feasible_choices = []
            for added in choices:
                trial = tuple(sorted(selected.union((added,))))
                if feasible(trial):
                    feasible_choices.append(added)
            choices = feasible_choices or choices
            choices = sorted(
                choices,
                key=lambda index: (
                    -raw_gains[index] / unit_costs[index],
                    -score_values[index],
                    unit_costs[index],
                    str(identifiers[index]),
                ),
            )
            selected.add(choices[rng.randrange(min(3, len(choices)))])
        return repair(tuple(sorted(selected)))

    for _ in range(generations):
        offspring = []
        offspring_seen = set()
        attempts = 0
        while len(offspring) < population_size and attempts < max_attempts:
            attempts += 1
            child = mutate(crossover(choose_parent(), choose_parent()))
            evaluate(child)
            if child not in offspring_seen:
                offspring_seen.add(child)
                offspring.append(child)
        population = ranked_population(population + offspring)
        diversity_history.append(_population_diversity(population, selection_size))
        unique_history.append(len(set(population)))

    all_feasible = [chromosome for chromosome in evaluated if evaluations[chromosome]["budget_feasible"]]
    fronts = _pareto_fronts(all_feasible, evaluations)
    shortlist_chromosomes = list(fronts[0]) if fronts else []
    structural_best = max(
        all_feasible,
        key=lambda chromosome: (
            evaluations[chromosome]["structural_fitness"],
            evaluations[chromosome]["gain_per_parameter"],
            -evaluations[chromosome]["actual_cost"],
            tuple(reversed(evaluations[chromosome]["canonical_modules"])),
        ),
    )
    for reference in (greedy_reference, structural_best, cheapest):
        if reference not in shortlist_chromosomes:
            shortlist_chromosomes.append(reference)
    shortlist_chromosomes = sorted(
        set(shortlist_chromosomes), key=lambda chromosome: evaluations[chromosome]["canonical_modules"]
    )
    shortlist = [dict(evaluations[chromosome]) for chromosome in shortlist_chromosomes]
    selected, quality_set = select_quality_preserving_candidate(shortlist, gain_tolerance)
    selected_chromosome = selected["chromosome"]

    if not greedy_feasible and selected_chromosome == greedy_reference:
        selected_source = "repaired_greedy_reference"
    elif selected_chromosome == cheapest and len(shortlist) == 1:
        selected_source = "cheapest_feasible_emergency"
    elif selected["actual_cost"] < evaluations[structural_best]["actual_cost"]:
        selected_source = "budgeted_ga_low_cost_winner"
    else:
        selected_source = "budgeted_ga_quality_winner"

    selected_modules = [identifiers[index] for index in selected_chromosome]
    greedy_modules = [identifiers[index] for index in greedy]
    repaired_greedy_modules = None if greedy_feasible else [identifiers[index] for index in greedy_reference]
    diagnostics = {
        "selected_modules": selected_modules,
        "selected_source": selected_source,
        "greedy_modules": greedy_modules,
        "greedy_feasible": greedy_feasible,
        "greedy_candidate": dict(evaluations.get(greedy, evaluate(greedy))),
        "repaired_greedy_modules": repaired_greedy_modules,
        "repaired_greedy_candidate": dict(evaluations[greedy_reference]),
        "ga_structural_best_modules": evaluations[structural_best]["modules"],
        "shortlist": shortlist,
        "quality_set_modules": [item["modules"] for item in quality_set],
        "selected_candidate": selected,
        "minimum_future_cost": minimum_future_cost,
        "maximum_event_cost": maximum_event_cost,
        "population_unique_count_initial": len(set(initial_population)),
        "population_unique_count_final": len(set(population)),
        "population_diversity_initial": diversity_history[0],
        "population_diversity_final": diversity_history[-1],
        "population_diversity_history": diversity_history,
        "population_unique_history": unique_history,
        "evaluated_unique_chromosome_count": len(evaluated),
        "repair_count": repair_count,
        "infeasible_candidate_count": infeasible_count,
        "local_search_enabled": False,
        "selected_set_equals_greedy": set(selected_chromosome) == set(greedy),
        "selected_non_greedy_count": len(set(selected_chromosome).difference(greedy)),
        "budget_constraint_satisfied": selected["projected_minimum_final_cost"] <= int(target_final_cost),
        "quality_guard_satisfied": selected in quality_set,
        "interaction_feature_mode": (
            "temporal_complementarity" if temporal_available else "structural_redundancy_only"
        ),
    }
    return selected_modules, diagnostics
