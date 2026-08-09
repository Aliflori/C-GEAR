# coding=utf-8
"""Pure candidate search and quality selection for calibrated IncreLoRA.

This module is deliberately isolated from :mod:`evo_allocator` and
:mod:`budgeted_evo_allocator`.  It contains no model, dataset, validation, or
checkpoint integration.  Callers provide training-only importance and
optimizer-gain signals, then attach virtual-calibration fold gains to the
shortlisted candidate dictionaries before calling
``select_calibrated_candidate``.

Chromosomes are canonical tuples of integer module indices.  Unlike the older
allocators, cardinality is variable and the empty tuple is a valid zero-rank
decision.  There is no local-search or one-swap refinement stage.
"""

import itertools
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
_FAMILY_ORDER = {
    "zero_rank": 0,
    "greedy_anchor": 1,
    "repaired_greedy_anchor": 2,
    "greedy_neighborhood": 3,
    "global_ga": 4,
}


def allowed_event_rank_sizes(
    top_h,
    min_event_rank=1,
    max_event_rank=None,
    candidate_count=None,
):
    """Return the deterministic legal chromosome sizes, always including zero.

    ``min_event_rank=0`` means that positive chromosomes begin at one; zero is
    represented separately and is never lost when the positive lower bound is
    greater than zero.
    """

    top_h = int(top_h)
    min_event_rank = int(min_event_rank)
    max_event_rank = top_h if max_event_rank is None else int(max_event_rank)
    if top_h < 0:
        raise ValueError("top_h must be nonnegative.")
    if min_event_rank < 0:
        raise ValueError("ga_min_event_rank must be nonnegative.")
    if max_event_rank < 0:
        raise ValueError("ga_max_event_rank must be nonnegative.")
    maximum = min(top_h, max_event_rank)
    if candidate_count is not None:
        candidate_count = int(candidate_count)
        if candidate_count < 0:
            raise ValueError("candidate_count must be nonnegative.")
        maximum = min(maximum, candidate_count)
    if maximum == 0:
        return (0,)
    minimum_positive = max(1, min_event_rank)
    if minimum_positive > maximum:
        raise ValueError(
            "ga_min_event_rank exceeds the effective ga_max_event_rank."
        )
    return (0,) + tuple(range(minimum_positive, maximum + 1))


def chromosome_hamming_distance(left, right):
    """Return set Hamming distance (symmetric-difference cardinality)."""

    return len(set(left).symmetric_difference(right))


def calibration_statistics(fold_gains, lcb_beta=0.5):
    """Compute mean, population standard deviation, and conservative LCB."""

    lcb_beta = float(lcb_beta)
    if not math.isfinite(lcb_beta) or lcb_beta < 0.0:
        raise ValueError("ga_calibration_lcb_beta must be nonnegative.")
    try:
        gains = [float(value) for value in fold_gains]
    except (TypeError, ValueError, OverflowError):
        gains = []
    valid = bool(gains) and all(math.isfinite(value) for value in gains)
    if not valid:
        return {
            "calibration_gains": gains,
            "calibration_gain_mean": None,
            "calibration_gain_std": None,
            "calibration_gain_lcb": None,
            "calibration_signal_valid": False,
        }
    mean = sum(gains) / float(len(gains))
    variance = sum((value - mean) ** 2 for value in gains) / float(len(gains))
    standard_deviation = math.sqrt(max(0.0, variance))
    return {
        "calibration_gains": gains,
        "calibration_gain_mean": mean,
        "calibration_gain_std": standard_deviation,
        "calibration_gain_lcb": mean - lcb_beta * standard_deviation,
        "calibration_signal_valid": True,
    }


def _canonical(chromosome, identifiers):
    return tuple(sorted(str(identifiers[index]) for index in chromosome))


def _family_key(family):
    return (_FAMILY_ORDER.get(family, len(_FAMILY_ORDER)), str(family))


def _candidate_primary_family(families):
    return min(families, key=_family_key)


def _is_reference_family(candidate):
    families = set(candidate.get("candidate_families", ()))
    return bool(
        families.intersection(("greedy_anchor", "repaired_greedy_anchor"))
    )


def _is_neighborhood_family(candidate):
    families = set(candidate.get("candidate_families", ()))
    return "greedy_neighborhood" in families


def _is_global_only(candidate):
    families = set(candidate.get("candidate_families", ()))
    return "global_ga" in families and not families.intersection(
        ("greedy_anchor", "repaired_greedy_anchor", "greedy_neighborhood")
    )


def _population_diversity(population):
    if len(population) < 2:
        return 0.0
    distances = []
    for position, left in enumerate(population):
        for right in population[position + 1 :]:
            denominator = float(max(len(left), len(right), 1))
            distances.append(chromosome_hamming_distance(left, right) / denominator)
    return sum(distances) / len(distances) if distances else 0.0


def generate_calibrated_candidates(
    scores,
    costs,
    training_gains,
    top_h,
    current_active_cost,
    target_final_cost,
    rank_increment=1,
    min_event_rank=1,
    max_event_rank=None,
    max_greedy_replacements=2,
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
    greedy_anchor_modules=None,
):
    """Generate Greedy-anchored and global variable-cardinality candidates.

    The hard budget is an upper bound.  Consequently projected final cost is
    exactly ``current_active_cost + event_cost``; no hypothetical future rank
    growth is reserved.  The returned candidate dictionaries contain only
    inexpensive structural/training-proxy metrics and are ready for a caller to
    attach virtual-calibration fold gains.
    """

    population_size = int(population_size)
    generations = int(generations)
    rank_increment = int(rank_increment)
    max_greedy_replacements = int(max_greedy_replacements)
    current_active_cost = int(current_active_cost)
    target_final_cost = int(target_final_cost)
    mutation_rate = float(mutation_rate)
    crossover_rate = float(crossover_rate)
    interaction_weight = float(interaction_weight)
    redundancy_weight = float(redundancy_weight)
    cost_weight = float(cost_weight)
    diversity_weight = float(diversity_weight)

    if population_size <= 0:
        raise ValueError("ga_population must be positive.")
    if generations < 0:
        raise ValueError("ga_generations must be nonnegative.")
    if rank_increment <= 0:
        raise ValueError("rank_increment must be positive.")
    if max_greedy_replacements < 0:
        raise ValueError("ga_max_greedy_replacements must be nonnegative.")
    if not math.isfinite(mutation_rate) or not 0.0 <= mutation_rate <= 1.0:
        raise ValueError("ga_mutation_rate must be between 0 and 1.")
    if not math.isfinite(crossover_rate) or not 0.0 <= crossover_rate <= 1.0:
        raise ValueError("ga_crossover_rate must be between 0 and 1.")
    for name, value in (
        ("ga_interaction_weight", interaction_weight),
        ("ga_redundancy_weight", redundancy_weight),
        ("ga_cost_weight", cost_weight),
        ("ga_diversity_weight", diversity_weight),
    ):
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("%s must be nonnegative." % name)
    if current_active_cost > target_final_cost:
        raise ValueError(
            "The hard budget is infeasible before allocation: current_active_cost=%s "
            "target_final_cost=%s."
            % (current_active_cost, target_final_cost)
        )

    score_items = _deduplicated_scores(scores)
    identifiers = [identifier for identifier, _ in score_items]
    raw_scores = [score for _, score in score_items]
    candidate_count = len(identifiers)
    sizes = allowed_event_rank_sizes(
        top_h,
        min_event_rank=min_event_rank,
        max_event_rank=max_event_rank,
        candidate_count=candidate_count,
    )
    positive_sizes = tuple(size for size in sizes if size > 0)
    normalized_scores = _normalized_scores(raw_scores)
    cost_map = _value_map(costs, lambda value: int(max(0, _finite_float(value))))
    gain_map = _value_map(
        training_gains, lambda value: max(0.0, _finite_float(value))
    )
    unit_costs = [cost_map.get(identifier, 0) for identifier in identifiers]
    raw_gains = [gain_map.get(identifier, 0.0) for identifier in identifiers]
    if any(cost <= 0 for cost in unit_costs):
        raise ValueError("Every eligible module must have a positive exact rank-one cost.")

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
        return current_active_cost + event_cost(chromosome)

    def feasible(chromosome):
        return projected_final(chromosome) <= target_final_cost

    maximum_size = max(sizes) if sizes else 0
    minimum_positive = min(positive_sizes) if positive_sizes else None
    maximum_event_cost = int(
        sum(sorted(unit_costs, reverse=True)[:maximum_size]) * rank_increment
    )

    def utility_key(index):
        return (
            raw_gains[index] / float(unit_costs[index]),
            normalized_scores[index],
            raw_gains[index],
            -unit_costs[index],
            str(identifiers[index]),
        )

    def normalize_cardinality(chromosome):
        selected = set(
            index
            for index in chromosome
            if isinstance(index, int) and 0 <= index < candidate_count
        )
        if len(selected) > maximum_size:
            selected = set(
                sorted(selected, key=utility_key, reverse=True)[:maximum_size]
            )
        if selected and len(selected) not in positive_sizes:
            if minimum_positive is None or len(selected) < minimum_positive:
                selected = set()
            else:
                legal_size = max(size for size in positive_sizes if size <= len(selected))
                selected = set(
                    sorted(selected, key=utility_key, reverse=True)[:legal_size]
                )
        return tuple(sorted(selected))

    repair_count = 0

    def repair_to_budget(chromosome):
        """Deterministically enforce cardinality and budget, without refinement."""

        nonlocal repair_count
        original = tuple(sorted(set(chromosome)))
        repaired = normalize_cardinality(original)
        changed = repaired != original
        while repaired and not feasible(repaired):
            selected = set(repaired)
            best_swap = None
            for removed in sorted(selected):
                for added in range(candidate_count):
                    if added in selected or unit_costs[added] >= unit_costs[removed]:
                        continue
                    trial = tuple(
                        sorted(selected.difference((removed,)).union((added,)))
                    )
                    if event_cost(trial) >= event_cost(repaired):
                        continue
                    key = (
                        event_cost(trial),
                        -raw_gains[added],
                        -normalized_scores[added],
                        _canonical(trial, identifiers),
                    )
                    if best_swap is None or key < best_swap[0]:
                        best_swap = (key, trial)
            if best_swap is not None:
                repaired = best_swap[1]
                changed = True
                continue

            current_size = len(repaired)
            smaller_sizes = [size for size in sizes if size < current_size]
            next_size = max(smaller_sizes) if smaller_sizes else 0
            if next_size == 0:
                repaired = ()
            else:
                repaired = tuple(
                    sorted(
                        sorted(repaired, key=utility_key, reverse=True)[:next_size]
                    )
                )
            changed = True
        if changed:
            repair_count += 1
        return repaired

    greedy_order = sorted(
        range(candidate_count), key=lambda index: (-raw_scores[index], index)
    )
    if greedy_anchor_modules is None:
        greedy_anchor = (
            tuple(sorted(greedy_order[:maximum_size])) if maximum_size > 0 else ()
        )
    else:
        identifier_to_index = {
            identifier: index for index, identifier in enumerate(identifiers)
        }
        supplied_anchor = list(greedy_anchor_modules)
        if len(supplied_anchor) != len(set(supplied_anchor)):
            raise ValueError("greedy_anchor_modules must not contain duplicates.")
        unknown = [
            identifier
            for identifier in supplied_anchor
            if identifier not in identifier_to_index
        ]
        if unknown:
            raise ValueError(
                "greedy_anchor_modules contains ineligible modules: %s" % unknown
            )
        greedy_anchor = tuple(
            sorted(identifier_to_index[identifier] for identifier in supplied_anchor)
        )
    greedy_anchor_feasible = (
        len(greedy_anchor) in sizes and feasible(greedy_anchor)
    )
    repaired_greedy_anchor = (
        greedy_anchor if greedy_anchor_feasible else repair_to_budget(greedy_anchor)
    )

    evaluations = {}

    def evaluate(chromosome):
        chromosome = tuple(sorted(chromosome))
        if chromosome not in evaluations:
            size = len(chromosome)
            importance = (
                sum(normalized_scores[index] for index in chromosome) / float(size)
                if size
                else 0.0
            )
            redundancies = []
            interactions = []
            for position, left in enumerate(chromosome):
                for right in chromosome[position + 1 :]:
                    pair = relation(left, right)
                    redundancies.append(max(0.0, pair))
                    if temporal_available:
                        gate = math.sqrt(
                            max(0.0, normalized_scores[left] * normalized_scores[right])
                        )
                        interactions.append(max(0.0, -pair) * gate)
            redundancy = (
                sum(redundancies) / len(redundancies) if redundancies else 0.0
            )
            interaction = (
                sum(interactions) / len(interactions) if interactions else 0.0
            )
            structural = (
                importance
                + interaction_weight * interaction
                - redundancy_weight * redundancy
            )
            cost = event_cost(chromosome)
            normalized_cost = (
                cost / float(maximum_event_cost) if maximum_event_cost else 0.0
            )
            structural -= cost_weight * normalized_cost
            raw_gain = sum(raw_gains[index] for index in chromosome)
            evaluations[chromosome] = {
                "chromosome": chromosome,
                "modules": [identifiers[index] for index in chromosome],
                "canonical_modules": _canonical(chromosome, identifiers),
                "chromosome_size": size,
                "selected_event_rank": size,
                "actual_cost": cost,
                "candidate_cost": cost,
                "raw_training_gain": raw_gain,
                "optimizer_raw_gain": raw_gain,
                "gain_per_parameter": raw_gain / float(cost) if cost else 0.0,
                "structural_fitness": structural,
                "importance_reward": importance,
                "interaction_gain": interaction,
                "redundancy_score": redundancy,
                "normalized_cost_penalty": normalized_cost,
                "weighted_cost_penalty": cost_weight * normalized_cost,
                "projected_final_active_parameter_count": projected_final(chromosome),
                "target_final_active_parameter_count": target_final_cost,
                "budget_feasible": feasible(chromosome),
                "hamming_distance_from_greedy": chromosome_hamming_distance(
                    chromosome, greedy_anchor
                ),
                "candidate_families": [],
                "replacement_count": None,
                "local_search_enabled": False,
            }
        return evaluations[chromosome]

    def register(chromosome, family, replacement_count=None):
        candidate = evaluate(tuple(sorted(chromosome)))
        families = set(candidate["candidate_families"])
        families.add(family)
        candidate["candidate_families"] = sorted(families, key=_family_key)
        candidate["candidate_family"] = _candidate_primary_family(
            candidate["candidate_families"]
        )
        if replacement_count is not None:
            current = candidate.get("replacement_count")
            candidate["replacement_count"] = (
                int(replacement_count)
                if current is None
                else min(current, int(replacement_count))
            )
        candidate["is_global_candidate"] = _is_global_only(candidate)
        return candidate

    register((), "zero_rank", replacement_count=0)
    reference_family = (
        "greedy_anchor" if greedy_anchor_feasible else "repaired_greedy_anchor"
    )
    reference_candidate = register(
        repaired_greedy_anchor,
        reference_family,
        replacement_count=(
            0
            if greedy_anchor_feasible
            else len(set(repaired_greedy_anchor).difference(greedy_anchor))
        ),
    )
    reference_candidate["greedy_quality_reference"] = True

    # Greedy prefixes guarantee coverage of every legal positive cardinality.
    greedy_prefixes = {}
    for size in positive_sizes:
        prefix = tuple(sorted(greedy_order[:size]))
        greedy_prefixes[size] = prefix
        if prefix == greedy_anchor and greedy_anchor_feasible:
            register(prefix, "greedy_anchor", replacement_count=0)
        else:
            register(prefix, "greedy_neighborhood", replacement_count=0)

    # Enumerating at most two replacements is tractable for the 72-module
    # DeBERTa search space and avoids an arbitrary hidden neighborhood cap.
    for size, base in sorted(greedy_prefixes.items()):
        base_set = set(base)
        outside = [index for index in range(candidate_count) if index not in base_set]
        replacement_limit = min(max_greedy_replacements, size, len(outside))
        for replacement_count in range(1, replacement_limit + 1):
            for removed in itertools.combinations(sorted(base), replacement_count):
                retained = base_set.difference(removed)
                for added in itertools.combinations(outside, replacement_count):
                    chromosome = tuple(sorted(retained.union(added)))
                    register(
                        chromosome,
                        "greedy_neighborhood",
                        replacement_count=replacement_count,
                    )

    rng = random.Random(int(seed))

    def global_rank_key(chromosome):
        candidate = evaluate(chromosome)
        return (
            -candidate["structural_fitness"],
            -candidate["raw_training_gain"],
            -candidate["gain_per_parameter"],
            candidate["actual_cost"],
            candidate["hamming_distance_from_greedy"],
            candidate["canonical_modules"],
        )

    population = []
    population_seen = set()

    def add_population(chromosome):
        chromosome = repair_to_budget(chromosome)
        register(chromosome, "global_ga")
        if chromosome in population_seen:
            return False
        population_seen.add(chromosome)
        population.append(chromosome)
        return True

    add_population(repaired_greedy_anchor)
    for size in positive_sizes:
        cheapest = tuple(
            sorted(
                sorted(
                    range(candidate_count),
                    key=lambda index: (
                        unit_costs[index],
                        -raw_gains[index],
                        -normalized_scores[index],
                        str(identifiers[index]),
                    ),
                )[:size]
            )
        )
        raw_best = tuple(
            sorted(
                sorted(
                    range(candidate_count),
                    key=lambda index: (
                        -raw_gains[index],
                        -normalized_scores[index],
                        unit_costs[index],
                        str(identifiers[index]),
                    ),
                )[:size]
            )
        )
        add_population(cheapest)
        add_population(raw_best)

    maximum_attempts = max(100, population_size * 50)
    attempts = 0
    while len(population) < population_size and attempts < maximum_attempts:
        attempts += 1
        size = rng.choice(sizes)
        chromosome = (
            tuple(sorted(rng.sample(range(candidate_count), size))) if size else ()
        )
        add_population(chromosome)
    if not population:
        add_population(())

    initial_population = list(population)
    diversity_history = [_population_diversity(population)]
    unique_history = [len(set(population))]

    def choose_parent():
        if len(population) == 1:
            return population[0]
        contestants = rng.sample(population, min(3, len(population)))
        return min(contestants, key=global_rank_key)

    def set_crossover(left, right):
        if rng.random() >= crossover_rate:
            return left if rng.random() < 0.5 else right
        target_size = rng.choice(sizes)
        if target_size == 0:
            return ()
        shared = list(sorted(set(left).intersection(right)))
        parental = [
            index
            for index in sorted(set(left).union(right))
            if index not in shared
        ]
        rng.shuffle(parental)
        selected = shared[:target_size]
        for index in parental:
            if len(selected) >= target_size:
                break
            if index not in selected:
                selected.append(index)
        remaining = [
            index for index in range(candidate_count) if index not in selected
        ]
        rng.shuffle(remaining)
        selected.extend(remaining[: max(0, target_size - len(selected))])
        return repair_to_budget(tuple(sorted(selected)))

    def mutate(chromosome):
        if rng.random() >= mutation_rate or candidate_count == 0:
            return chromosome
        selected = set(chromosome)
        operations = []
        if len(selected) < maximum_size:
            operations.append("add")
        if selected:
            operations.append("remove")
        if selected and len(selected) < candidate_count:
            operations.append("replace")
        if not operations:
            return chromosome
        operation = rng.choice(operations)
        if operation == "add":
            target_size = (
                minimum_positive if not selected else len(selected) + 1
            )
            if target_size not in sizes:
                larger = [size for size in sizes if size > len(selected)]
                target_size = min(larger) if larger else len(selected)
            available = [
                index for index in range(candidate_count) if index not in selected
            ]
            rng.shuffle(available)
            selected.update(available[: max(0, target_size - len(selected))])
        elif operation == "remove":
            if len(selected) == minimum_positive:
                selected.clear()
            else:
                selected.remove(rng.choice(sorted(selected)))
        else:
            removed = rng.choice(sorted(selected))
            selected.remove(removed)
            available = [
                index for index in range(candidate_count) if index not in selected
            ]
            selected.add(rng.choice(available))
        return repair_to_budget(tuple(sorted(selected)))

    def environmental_selection(chromosomes):
        unique = list(dict.fromkeys(chromosomes))
        if len(unique) <= population_size:
            return sorted(unique, key=global_rank_key)
        selected = [min(unique, key=global_rank_key)]
        remaining = [item for item in unique if item not in selected]
        while remaining and len(selected) < population_size:
            chosen = max(
                remaining,
                key=lambda chromosome: (
                    min(
                        chromosome_hamming_distance(chromosome, other)
                        / float(max(len(chromosome), len(other), 1))
                        for other in selected
                    )
                    * diversity_weight,
                    evaluate(chromosome)["structural_fitness"],
                    evaluate(chromosome)["raw_training_gain"],
                    evaluate(chromosome)["gain_per_parameter"],
                    -evaluate(chromosome)["actual_cost"],
                    tuple(reversed(evaluate(chromosome)["canonical_modules"])),
                ),
            )
            selected.append(chosen)
            remaining.remove(chosen)
        return selected

    for _ in range(generations):
        offspring = []
        offspring_seen = set()
        attempts = 0
        while len(offspring) < population_size and attempts < maximum_attempts:
            attempts += 1
            child = mutate(set_crossover(choose_parent(), choose_parent()))
            register(child, "global_ga")
            if child not in offspring_seen:
                offspring_seen.add(child)
                offspring.append(child)
        population = environmental_selection(population + offspring)
        diversity_history.append(_population_diversity(population))
        unique_history.append(len(set(population)))

    for candidate in evaluations.values():
        candidate["is_global_candidate"] = _is_global_only(candidate)
        candidate.setdefault("greedy_quality_reference", False)

    ordered_candidates = sorted(
        (dict(candidate) for candidate in evaluations.values()),
        key=lambda candidate: (
            candidate["chromosome_size"],
            candidate["canonical_modules"],
        ),
    )
    family_counts = {
        family: sum(
            family in candidate["candidate_families"]
            for candidate in ordered_candidates
        )
        for family in _FAMILY_ORDER
    }
    size_counts = {
        size: sum(candidate["chromosome_size"] == size for candidate in ordered_candidates)
        for size in sizes
    }
    global_candidates = [
        candidate
        for candidate in ordered_candidates
        if "global_ga" in candidate["candidate_families"]
        and candidate["budget_feasible"]
    ]
    global_only_candidates = [
        candidate for candidate in global_candidates if _is_global_only(candidate)
    ]
    global_best_pool = global_only_candidates or global_candidates
    global_best = (
        min(
            global_best_pool,
            key=lambda item: global_rank_key(item["chromosome"]),
        )
        if global_best_pool
        else None
    )

    diagnostics = {
        "allowed_event_rank_sizes": list(sizes),
        "candidate_count": candidate_count,
        "evaluated_unique_chromosome_count": len(ordered_candidates),
        "candidate_size_counts": size_counts,
        "candidate_family_counts": family_counts,
        "greedy_anchor_chromosome": greedy_anchor,
        "greedy_anchor_modules": [identifiers[index] for index in greedy_anchor],
        "greedy_anchor_feasible": greedy_anchor_feasible,
        "repaired_greedy_anchor_chromosome": (
            None if greedy_anchor_feasible else repaired_greedy_anchor
        ),
        "repaired_greedy_anchor_modules": (
            None
            if greedy_anchor_feasible
            else [identifiers[index] for index in repaired_greedy_anchor]
        ),
        "quality_reference_chromosome": repaired_greedy_anchor,
        "quality_reference_modules": [
            identifiers[index] for index in repaired_greedy_anchor
        ],
        "global_best_modules": global_best["modules"] if global_best else [],
        "population_unique_count_initial": len(set(initial_population)),
        "population_unique_count_final": len(set(population)),
        "population_diversity_initial": diversity_history[0],
        "population_diversity_final": diversity_history[-1],
        "population_diversity_history": diversity_history,
        "population_unique_history": unique_history,
        "budget_repair_count": repair_count,
        "hard_budget_target": target_final_cost,
        "current_active_parameter_count": current_active_cost,
        "local_search_enabled": False,
    }
    return ordered_candidates, diagnostics


def build_calibration_shortlist(candidates, calibration_topk=6, include_zero=True):
    """Build a deterministic mandatory-representative calibration shortlist.

    ``calibration_topk`` counts positive-rank candidates.  The zero-rank control
    is appended separately so all six required positive representatives can be
    present when ``calibration_topk=6``.
    """

    calibration_topk = int(calibration_topk)
    if calibration_topk <= 0:
        raise ValueError("ga_calibration_topk must be positive.")
    candidates = [dict(candidate) for candidate in candidates]
    feasible = [candidate for candidate in candidates if candidate.get("budget_feasible")]
    positive = [candidate for candidate in feasible if int(candidate.get("chromosome_size", 0)) > 0]
    zero_candidates = [candidate for candidate in feasible if int(candidate.get("chromosome_size", 0)) == 0]

    def canonical_key(candidate):
        return tuple(candidate.get("canonical_modules", ()))

    def best_by(items, key):
        return min(items, key=key) if items else None

    reference = best_by(
        [candidate for candidate in positive if _is_reference_family(candidate)],
        lambda candidate: (
            not bool(candidate.get("greedy_quality_reference")),
            candidate.get("hamming_distance_from_greedy", 0),
            -candidate.get("chromosome_size", 0),
            canonical_key(candidate),
        ),
    )
    neighborhood_candidates = [
        candidate
        for candidate in positive
        if _is_neighborhood_family(candidate)
        and int(candidate.get("replacement_count") or 0) > 0
    ]
    if not neighborhood_candidates:
        neighborhood_candidates = [
            candidate
            for candidate in positive
            if _is_neighborhood_family(candidate)
        ]
    neighborhood = best_by(
        neighborhood_candidates,
        lambda candidate: (
            -candidate.get("structural_fitness", 0.0),
            -candidate.get("raw_training_gain", 0.0),
            -candidate.get("gain_per_parameter", 0.0),
            candidate.get("actual_cost", 0),
            canonical_key(candidate),
        ),
    )
    raw_best = best_by(
        positive,
        lambda candidate: (
            -candidate.get("raw_training_gain", 0.0),
            -candidate.get("structural_fitness", 0.0),
            candidate.get("actual_cost", 0),
            canonical_key(candidate),
        ),
    )
    efficiency_best = best_by(
        positive,
        lambda candidate: (
            -candidate.get("gain_per_parameter", 0.0),
            -candidate.get("raw_training_gain", 0.0),
            candidate.get("actual_cost", 0),
            canonical_key(candidate),
        ),
    )
    lowest_cost = best_by(
        positive,
        lambda candidate: (
            candidate.get("actual_cost", 0),
            -candidate.get("raw_training_gain", 0.0),
            -candidate.get("structural_fitness", 0.0),
            canonical_key(candidate),
        ),
    )
    global_candidates = [
        candidate
        for candidate in positive
        if "global_ga" in candidate.get("candidate_families", ())
    ]
    global_only_candidates = [
        candidate for candidate in global_candidates if _is_global_only(candidate)
    ]
    global_best = best_by(
        global_only_candidates or global_candidates,
        lambda candidate: (
            -candidate.get("structural_fitness", 0.0),
            -candidate.get("raw_training_gain", 0.0),
            -candidate.get("gain_per_parameter", 0.0),
            candidate.get("actual_cost", 0),
            canonical_key(candidate),
        ),
    )

    mandatory = (
        ("greedy_quality_reference", reference),
        ("best_greedy_neighborhood", neighborhood),
        ("best_raw_optimizer_gain", raw_best),
        ("best_gain_per_parameter", efficiency_best),
        ("lowest_nonzero_cost", lowest_cost),
        ("best_global_ga", global_best),
    )
    selected = []
    selected_by_chromosome = {}
    reasons = {}

    def add(candidate, reason):
        if candidate is None:
            return
        chromosome = tuple(candidate["chromosome"])
        if chromosome not in selected_by_chromosome:
            copied = dict(candidate)
            selected_by_chromosome[chromosome] = copied
            selected.append(copied)
        reasons.setdefault(chromosome, []).append(reason)

    for reason, candidate in mandatory:
        add(candidate, reason)

    fill_order = sorted(
        positive,
        key=lambda candidate: (
            -candidate.get("structural_fitness", 0.0),
            -candidate.get("raw_training_gain", 0.0),
            -candidate.get("gain_per_parameter", 0.0),
            candidate.get("actual_cost", 0),
            candidate.get("hamming_distance_from_greedy", 0),
            canonical_key(candidate),
        ),
    )
    effective_positive_target = max(calibration_topk, len(selected))
    for candidate in fill_order:
        if len(selected) >= effective_positive_target:
            break
        add(candidate, "deterministic_fill")

    zero = best_by(zero_candidates, canonical_key)
    if include_zero and zero is not None:
        add(zero, "zero_rank_control")

    for candidate in selected:
        candidate["shortlist_reasons"] = list(
            reasons.get(tuple(candidate["chromosome"]), ())
        )

    diagnostics = {
        "requested_positive_shortlist_size": calibration_topk,
        "effective_positive_shortlist_size": sum(
            int(candidate.get("chromosome_size", 0)) > 0 for candidate in selected
        ),
        "total_shortlist_size": len(selected),
        "zero_rank_control_included": any(
            int(candidate.get("chromosome_size", 0)) == 0 for candidate in selected
        ),
        "mandatory_representatives": {
            reason: (candidate.get("modules") if candidate is not None else None)
            for reason, candidate in mandatory
        },
        "shortlist_modules": [candidate.get("modules", []) for candidate in selected],
        "shortlist_reasons": [candidate["shortlist_reasons"] for candidate in selected],
        "duplicate_mandatory_count": sum(
            candidate is not None for _, candidate in mandatory
        )
        - len(
            {
                tuple(candidate["chromosome"])
                for _, candidate in mandatory
                if candidate is not None
            }
        ),
        "local_search_enabled": False,
    }
    return selected, diagnostics


def _annotate_calibration(candidate, lcb_beta):
    candidate = dict(candidate)
    if "calibration_gains" in candidate:
        statistics = calibration_statistics(candidate["calibration_gains"], lcb_beta)
    elif "calibration_fold_gains" in candidate:
        statistics = calibration_statistics(
            candidate["calibration_fold_gains"], lcb_beta
        )
    elif "fold_gains" in candidate:
        statistics = calibration_statistics(candidate["fold_gains"], lcb_beta)
    else:
        mean = candidate.get("calibration_gain_mean")
        standard_deviation = candidate.get("calibration_gain_std")
        lcb = candidate.get("calibration_gain_lcb")
        valid = all(
            value is not None and math.isfinite(_finite_float(value, float("nan")))
            for value in (mean, standard_deviation, lcb)
        )
        statistics = {
            "calibration_gains": list(candidate.get("calibration_gains", ())),
            "calibration_gain_mean": float(mean) if valid else None,
            "calibration_gain_std": float(standard_deviation) if valid else None,
            "calibration_gain_lcb": float(lcb) if valid else None,
            "calibration_signal_valid": valid,
        }
    candidate.update(statistics)
    cost = int(candidate.get("actual_cost", candidate.get("candidate_cost", 0)))
    candidate["calibration_gain_per_parameter"] = (
        statistics["calibration_gain_mean"] / float(max(cost, 1))
        if statistics["calibration_signal_valid"]
        else None
    )
    return candidate


def select_calibrated_candidate(
    candidates,
    lcb_beta=0.5,
    quality_absolute_tolerance=0.0,
    quality_relative_tolerance=0.01,
    greedy_quality_floor_ratio=0.99,
    greedy_quality_floor_absolute=0.0,
    min_calibrated_marginal_gain=0.0,
    target_final_cost=None,
    require_global_quality_improvement=True,
):
    """Select by calibrated quality first, then projected active cost.

    Global-only chromosomes must exceed the Greedy/repaired-Greedy reference
    by the configured absolute/relative quality band.  This implements the
    wider global search as an evidence-gated escape from the Greedy trust
    region rather than a cheap replacement for it.
    """

    quality_absolute_tolerance = float(quality_absolute_tolerance)
    quality_relative_tolerance = float(quality_relative_tolerance)
    greedy_quality_floor_ratio = float(greedy_quality_floor_ratio)
    greedy_quality_floor_absolute = float(greedy_quality_floor_absolute)
    min_calibrated_marginal_gain = float(min_calibrated_marginal_gain)
    for name, value in (
        ("ga_quality_absolute_tolerance", quality_absolute_tolerance),
        ("ga_quality_relative_tolerance", quality_relative_tolerance),
        ("ga_greedy_quality_floor_absolute", greedy_quality_floor_absolute),
    ):
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("%s must be nonnegative." % name)
    if not math.isfinite(greedy_quality_floor_ratio) or not 0.0 <= greedy_quality_floor_ratio <= 1.0:
        raise ValueError("ga_greedy_quality_floor_ratio must be between 0 and 1.")
    if not math.isfinite(min_calibrated_marginal_gain):
        raise ValueError("ga_min_calibrated_marginal_gain must be finite.")

    annotated = [_annotate_calibration(candidate, lcb_beta) for candidate in candidates]
    if not annotated:
        raise ValueError("No calibrated candidates were supplied.")
    if target_final_cost is not None:
        target_final_cost = int(target_final_cost)
    for candidate in annotated:
        projected = int(candidate.get("projected_final_active_parameter_count", 0))
        candidate_target = candidate.get("target_final_active_parameter_count")
        effective_target = target_final_cost if target_final_cost is not None else candidate_target
        candidate["budget_feasible"] = bool(
            candidate.get("budget_feasible", True)
            and effective_target is not None
            and projected <= int(effective_target)
        )
        candidate["quality_band_satisfied"] = False
        candidate["greedy_quality_floor_satisfied"] = False
        candidate["global_quality_requirement_satisfied"] = False
        candidate["reliable_positive_gain"] = False

    feasible_valid = [
        candidate
        for candidate in annotated
        if candidate["budget_feasible"] and candidate["calibration_signal_valid"]
    ]
    zero_candidates = [
        candidate
        for candidate in feasible_valid
        if int(candidate.get("chromosome_size", 0)) == 0
    ]
    reference_candidates = [
        candidate
        for candidate in feasible_valid
        if bool(candidate.get("greedy_quality_reference"))
        or _is_reference_family(candidate)
    ]
    reference = min(
        reference_candidates,
        key=lambda candidate: (
            not bool(candidate.get("greedy_quality_reference")),
            candidate.get("hamming_distance_from_greedy", 0),
            -int(candidate.get("chromosome_size", 0)),
            tuple(candidate.get("canonical_modules", ())),
        ),
    ) if reference_candidates else None

    if reference is None or not reference["calibration_signal_valid"]:
        if not zero_candidates:
            raise ValueError(
                "A finite Greedy or repaired-Greedy calibration reference is required."
            )
        selected = min(
            zero_candidates, key=lambda candidate: tuple(candidate.get("canonical_modules", ()))
        )
        selected_source = "calibrated_zero_rank"
        diagnostics = {
            "selected_candidate": selected,
            "selected_modules": selected.get("modules", []),
            "selected_source": selected_source,
            "selected_event_rank": 0,
            "zero_rank_selected": True,
            "zero_rank_reason": "invalid_or_missing_greedy_quality_reference",
            "greedy_quality_reference": reference,
            "quality_set_modules": [],
            "finite_budget_feasible_candidate_count": len(feasible_valid),
            "local_search_enabled": False,
            "candidates": annotated,
        }
        return selected, diagnostics

    reference_lcb = reference["calibration_gain_lcb"]
    floor_width = max(
        greedy_quality_floor_absolute,
        abs(reference_lcb) * (1.0 - greedy_quality_floor_ratio),
    )
    floor_threshold = reference_lcb - floor_width
    global_width = max(
        quality_absolute_tolerance,
        abs(reference_lcb) * quality_relative_tolerance,
    )
    global_threshold = reference_lcb + global_width

    eligible_positive = []
    for candidate in feasible_valid:
        size = int(candidate.get("chromosome_size", 0))
        if size == 0:
            candidate["greedy_quality_floor_satisfied"] = True
            candidate["global_quality_requirement_satisfied"] = True
            continue
        lcb = candidate["calibration_gain_lcb"]
        is_reference = candidate is reference or _is_reference_family(candidate)
        floor_satisfied = is_reference or lcb >= floor_threshold - _TOLERANCE
        global_satisfied = (
            not require_global_quality_improvement
            or not _is_global_only(candidate)
            or lcb > global_threshold + _TOLERANCE
        )
        reliable = lcb > min_calibrated_marginal_gain + _TOLERANCE
        candidate["greedy_quality_floor_satisfied"] = floor_satisfied
        candidate["global_quality_requirement_satisfied"] = global_satisfied
        candidate["reliable_positive_gain"] = reliable
        if floor_satisfied and global_satisfied and reliable:
            eligible_positive.append(candidate)

    if not eligible_positive:
        if not zero_candidates:
            raise ValueError(
                "No reliable positive-rank candidate and no finite zero-rank control are available."
            )
        selected = min(
            zero_candidates, key=lambda candidate: tuple(candidate.get("canonical_modules", ()))
        )
        selected["quality_band_satisfied"] = True
        diagnostics = {
            "selected_candidate": selected,
            "selected_modules": selected.get("modules", []),
            "selected_source": "calibrated_zero_rank",
            "selected_event_rank": 0,
            "selected_hamming_distance": selected.get("hamming_distance_from_greedy", 0),
            "zero_rank_selected": True,
            "zero_rank_reason": "no_reliable_quality_floor_eligible_positive_candidate",
            "greedy_quality_reference": reference,
            "greedy_quality_floor_threshold": floor_threshold,
            "global_quality_improvement_threshold": global_threshold,
            "quality_set_modules": [],
            "finite_budget_feasible_candidate_count": len(feasible_valid),
            "eligible_positive_candidate_count": 0,
            "local_search_enabled": False,
            "candidates": annotated,
        }
        return selected, diagnostics

    best_lcb = max(candidate["calibration_gain_lcb"] for candidate in eligible_positive)
    quality_width = max(
        quality_absolute_tolerance,
        abs(best_lcb) * quality_relative_tolerance,
    )
    quality_threshold = best_lcb - quality_width
    quality_set = [
        candidate
        for candidate in eligible_positive
        if candidate["calibration_gain_lcb"] >= quality_threshold - _TOLERANCE
    ]
    for candidate in quality_set:
        candidate["quality_band_satisfied"] = True

    selected = min(
        quality_set,
        key=lambda candidate: (
            int(candidate["projected_final_active_parameter_count"]),
            -candidate["calibration_gain_lcb"],
            -candidate["calibration_gain_mean"],
            int(candidate.get("hamming_distance_from_greedy", 0)),
            -float(candidate.get("structural_fitness", 0.0)),
            tuple(candidate.get("canonical_modules", ())),
        ),
    )

    if _is_reference_family(selected):
        selected_source = (
            "calibrated_repaired_greedy_anchor"
            if "repaired_greedy_anchor" in selected.get("candidate_families", ())
            else "calibrated_greedy_anchor"
        )
    elif _is_global_only(selected):
        selected_source = "calibrated_global_ga"
    elif _is_neighborhood_family(selected):
        best_quality_candidate = max(
            quality_set,
            key=lambda candidate: (
                candidate["calibration_gain_lcb"],
                candidate["calibration_gain_mean"],
                -int(candidate["projected_final_active_parameter_count"]),
            ),
        )
        selected_source = (
            "calibrated_low_cost_quality_equivalent"
            if selected is not best_quality_candidate
            and selected["projected_final_active_parameter_count"]
            < best_quality_candidate["projected_final_active_parameter_count"]
            else "calibrated_greedy_neighborhood"
        )
    else:
        selected_source = "calibrated_low_cost_quality_equivalent"

    diagnostics = {
        "selected_candidate": selected,
        "selected_modules": selected.get("modules", []),
        "selected_source": selected_source,
        "selected_event_rank": int(selected.get("chromosome_size", 0)),
        "selected_hamming_distance": selected.get("hamming_distance_from_greedy", 0),
        "zero_rank_selected": False,
        "zero_rank_reason": None,
        "greedy_quality_reference": reference,
        "greedy_quality_floor_threshold": floor_threshold,
        "global_quality_improvement_threshold": global_threshold,
        "best_calibration_gain_lcb": best_lcb,
        "quality_band_width": quality_width,
        "quality_band_threshold": quality_threshold,
        "quality_set_modules": [candidate.get("modules", []) for candidate in quality_set],
        "finite_budget_feasible_candidate_count": len(feasible_valid),
        "eligible_positive_candidate_count": len(eligible_positive),
        "quality_set_candidate_count": len(quality_set),
        "budget_constraint_satisfied": selected["budget_feasible"],
        "greedy_quality_floor_satisfied": selected[
            "greedy_quality_floor_satisfied"
        ],
        "quality_band_satisfied": selected["quality_band_satisfied"],
        "global_quality_requirement_satisfied": selected[
            "global_quality_requirement_satisfied"
        ],
        "local_search_enabled": False,
        "candidates": annotated,
    }
    return selected, diagnostics


__all__ = [
    "allowed_event_rank_sizes",
    "build_calibration_shortlist",
    "calibration_statistics",
    "chromosome_hamming_distance",
    "generate_calibrated_candidates",
    "select_calibrated_candidate",
]
