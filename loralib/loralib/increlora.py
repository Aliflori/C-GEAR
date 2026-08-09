#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import logging
import json
import math
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

import ipdb
import re
import numpy as np

from .layers import LoRALayer 
from typing import Optional, List 

logger = logging.getLogger(__name__)


BUDGETED_ALLOCATOR_METADATA = "budgeted_rank_allocator"


def get_module_rank_one_cost(module):
    """Return the exact active A/E/B parameter cost of one SVDLinear rank."""

    if not isinstance(module, SVDLinear):
        raise TypeError("Rank-one cost is only defined for SVDLinear modules.")
    a = module.lora_A[0]
    e = module.lora_E[0]
    b = module.lora_B[0]
    a_rank = int(a.size(0))
    e_rank = int(e.numel())
    b_rank = int(b.size(1))
    if a_rank <= 0 or a_rank != e_rank or a_rank != b_rank:
        raise ValueError("SVDLinear A/E/B tensors have inconsistent rank dimensions.")
    return int(a.numel() // a_rank + e.numel() // e_rank + b.numel() // b_rank)


def _dynamic_parameter_ids(model):
    parameter_ids = set()
    for module in model.modules():
        if isinstance(module, SVDLinear):
            for parameters in (module.lora_A, module.lora_E, module.lora_B):
                parameter_ids.update(id(parameter) for parameter in parameters)
            parameter_ids.add(id(module.ranknum))
    return parameter_ids


def get_runtime_trainable_parameter_count(model):
    """Return raw ``requires_grad`` parameters, including preparatory reserves."""

    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def get_full_model_parameter_count(model):
    """Return all model parameters, independently of trainability."""

    return int(sum(parameter.numel() for parameter in model.parameters()))


def get_active_model_parameter_count(model):
    """Return non-LoRA trainables plus active (not preparatory reserve) A/E/B."""

    dynamic_ids = _dynamic_parameter_ids(model)
    non_dynamic = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in dynamic_ids
    )
    active_dynamic = 0
    for module in model.modules():
        if isinstance(module, SVDLinear):
            active_dynamic += int(round(float(module.ranknum.item()))) * get_module_rank_one_cost(module)
    return int(non_dynamic + active_dynamic)


# Backward-compatible helper aliases used by the first budgeted implementation.
def get_current_dynamic_trainable_cost(model):
    return get_runtime_trainable_parameter_count(model)


def get_current_active_trainable_cost(model):
    return get_active_model_parameter_count(model)


def get_dynamic_lora_trainable_breakdown(model):
    """Separate active LoRA, trainable reserve, and non-dynamic parameters."""

    dynamic_ids = _dynamic_parameter_ids(model)
    direct_dynamic = int(
        sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad and id(parameter) in dynamic_ids
        )
    )
    active_dynamic = int(
        sum(
            int(round(float(module.ranknum.item()))) * get_module_rank_one_cost(module)
            for module in model.modules()
            if isinstance(module, SVDLinear)
        )
    )
    non_dynamic = int(
        sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad and id(parameter) not in dynamic_ids
        )
    )
    return {
        "active_dynamic_lora_trainable_params": active_dynamic,
        "reserve_preparatory_trainable_params": max(0, direct_dynamic - active_dynamic),
        "direct_dynamic_lora_trainable_params": direct_dynamic,
        "non_dynamic_trainable_params": non_dynamic,
        "direct_total_model_trainable_params": direct_dynamic + non_dynamic,
    }


def _plain_rank_map(pattern):
    if not isinstance(pattern, dict):
        raise ValueError("Rank pattern must be a JSON object.")
    modules = pattern.get("modules") if "modules" in pattern else pattern
    if not isinstance(modules, dict):
        raise ValueError("Rank pattern modules must be a JSON object.")
    ranks = {}
    for name, value in modules.items():
        if isinstance(value, dict):
            value = value.get("active_rank")
        if isinstance(value, bool):
            raise ValueError("Rank values must be positive integers.")
        try:
            rank = int(value)
        except (TypeError, ValueError, OverflowError):
            raise ValueError("Invalid active rank for module '%s'." % name)
        if rank <= 0 or float(value) != rank:
            raise ValueError("Invalid active rank for module '%s'." % name)
        ranks[name] = rank
    return ranks


def get_rank_pattern_active_model_parameter_count(model, pattern):
    """Calculate exact total trainables represented by an active-rank pattern.

    Legacy patterns map module names directly to integer ranks. Versioned patterns
    additionally carry dimensions and the non-dynamic trainable count.
    """

    ranks = _plain_rank_map(pattern)
    modules = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, SVDLinear)
    }
    if set(ranks) != set(modules):
        missing = sorted(set(modules).difference(ranks))
        unexpected = sorted(set(ranks).difference(modules))
        raise ValueError(
            "Rank-pattern modules do not match the current model; missing=%s unexpected=%s"
            % (missing, unexpected)
        )

    rich_modules = pattern.get("modules") if isinstance(pattern, dict) else None
    for name, module in modules.items():
        metadata = rich_modules.get(name) if isinstance(rich_modules, dict) else None
        if isinstance(metadata, dict):
            expected = (int(module.in_features), int(module.out_features))
            supplied = (metadata.get("in_features"), metadata.get("out_features"))
            if supplied != expected:
                raise ValueError(
                    "Rank-pattern dimensions for '%s' are %s, expected %s."
                    % (name, supplied, expected)
                )
            supplied_cost = metadata.get("rank_one_cost")
            if supplied_cost is not None and int(supplied_cost) != get_module_rank_one_cost(module):
                raise ValueError("Rank-pattern cost for '%s' is incompatible." % name)

    dynamic_ids = _dynamic_parameter_ids(model)
    current_non_dynamic = int(
        sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad and id(parameter) not in dynamic_ids
        )
    )
    if isinstance(pattern, dict) and "non_dynamic_trainable_params" in pattern:
        supplied_non_dynamic = int(pattern["non_dynamic_trainable_params"])
        if supplied_non_dynamic != current_non_dynamic:
            raise ValueError(
                "Rank-pattern non-dynamic trainable count %s does not match current model %s."
                % (supplied_non_dynamic, current_non_dynamic)
            )
    return int(
        current_non_dynamic
        + sum(ranks[name] * get_module_rank_one_cost(module) for name, module in modules.items())
    )


def get_rank_pattern_trainable_cost(model, pattern):
    """Backward-compatible alias for active rank-pattern accounting."""

    return get_rank_pattern_active_model_parameter_count(model, pattern)


def build_trainable_rank_pattern(model, allocator_metadata=None):
    """Build a versioned active-rank pattern with independently checkable costs."""

    dynamic_ids = _dynamic_parameter_ids(model)
    non_dynamic = int(
        sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad and id(parameter) not in dynamic_ids
        )
    )
    modules = {}
    for name, module in model.named_modules():
        if isinstance(module, SVDLinear):
            rank = int(round(float(module.ranknum.item())))
            rank_one_cost = get_module_rank_one_cost(module)
            metadata = module.get_dynamic_lora_metadata()
            modules[name] = {
                "active_rank": rank,
                "in_features": int(module.in_features),
                "out_features": int(module.out_features),
                "rank_one_cost": rank_one_cost,
                "active_trainable_params": rank * rank_one_cost,
                "rank_component_count": int(metadata["rank_component_count"]),
            }
    result = {
        "format_version": 2,
        "allocator_mode": (
            allocator_metadata.get("allocator_mode", "genetic_budgeted")
            if isinstance(allocator_metadata, dict)
            else "genetic_budgeted"
        ),
        "non_dynamic_trainable_params": non_dynamic,
        "modules": modules,
    }
    breakdown = get_dynamic_lora_trainable_breakdown(model)
    result["total_dynamic_lora_trainable_params"] = breakdown[
        "active_dynamic_lora_trainable_params"
    ]
    result["reserve_preparatory_trainable_params"] = breakdown[
        "reserve_preparatory_trainable_params"
    ]
    result["active_model_parameter_count"] = get_rank_pattern_active_model_parameter_count(
        model, result
    )
    result["runtime_trainable_parameter_count"] = get_runtime_trainable_parameter_count(model)
    result["full_model_parameter_count"] = get_full_model_parameter_count(model)
    # Retain the original field as an explicitly active-count compatibility alias.
    result["total_model_trainable_params"] = result["active_model_parameter_count"]
    if allocator_metadata is not None:
        result["budget"] = dict(allocator_metadata)
    return result


def deactivate_inactive_reserve_parameters(model):
    """Make only active A/E/B rank components trainable, leaving tensors intact."""

    affected = []
    for name, module in model.named_modules():
        if not isinstance(module, SVDLinear):
            continue
        active_rank = int(round(float(module.ranknum.item())))
        for label, parameters, rank_dimension in (
            ("lora_A", module.lora_A, 0),
            ("lora_E", module.lora_E, None),
            ("lora_B", module.lora_B, 1),
        ):
            cursor = 0
            for parameter in parameters:
                width = parameter.numel() if rank_dimension is None else parameter.size(rank_dimension)
                component_end = cursor + int(width)
                if cursor < active_rank < component_end:
                    raise ValueError(
                        "Active rank cuts through parameter chunk %s.%s; cannot set exact trainability."
                        % (name, label)
                    )
                should_train = component_end <= active_rank
                if parameter.requires_grad != should_train:
                    parameter.requires_grad_(should_train)
                    affected.append("%s.%s" % (name, label))
                cursor = component_end
    return affected


def load_rank_pattern(path):
    if not path:
        raise ValueError("A Greedy rank-pattern path is required.")
    if not os.path.isfile(path):
        raise ValueError("Greedy rank-pattern file does not exist: %s" % path)
    try:
        with open(path, "r", encoding="utf-8") as stream:
            return json.load(stream)
    except (OSError, ValueError) as error:
        raise ValueError("Could not load Greedy rank pattern '%s': %s" % (path, error))

class loraW(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, A, E, B, scaling, ranknum):
        return torch.cat([b for b in B], 1) @                                 \
                (torch.cat([a for a in A], 0) * torch.cat([e for e in E], 0)) \
                    * scaling / (ranknum+1e-5)
    
class SVDLinear(nn.Linear, LoRALayer):
    # SVD-based adaptation implemented in a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False, 
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)

        self.module_name = ""
        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.ParameterList([nn.Parameter(
                self.weight.new_zeros((r, in_features))
            )])
            self.lora_E = nn.ParameterList([nn.Parameter(
                self.weight.new_zeros(r, 1)
            )])
            self.lora_B = nn.ParameterList([nn.Parameter(
                self.weight.new_zeros((out_features, r))
            )])
            self.W = loraW()
            self.hook_handle = self.W.register_full_backward_hook(self.backward_hook)
            
            self.score = 0
            self.gradMatrix_trace = 0
            self.ranknum = nn.Parameter(
                self.weight.new_zeros(1), requires_grad=False
            )
            self.ranknum.data.fill_(float(self.r))
            self.scaling = self.lora_alpha if self.lora_alpha>0 else float(self.r)   
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
            self.ranknum.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def backward_hook(self, module, grad_input, grad_output):
        # print("Output_Grad:", grad_output)
        grad_Matrix = grad_output[0]
        try:
            W = (
                
                 self.W(self.lora_A, self.lora_E, self.lora_B, self.scaling, self.ranknum)
                 ).abs()
            # scale_W = torch.mean(W)
            scale_W=1
            self.score = torch.sum(((W / scale_W) * grad_Matrix).abs().detach()) / math.sqrt(W.numel())
            # self.score = torch.mean((grad_Matrix ** 2).detach())
        except:
            ipdb.set_trace()
        
    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A,B the same way as the default for nn.Linear 
            # and E (singular values) for zero 
            nn.init.zeros_(self.lora_E[0])
            nn.init.normal_(self.lora_A[0], mean=0.0, std=0.02)
            nn.init.normal_(self.lora_B[0], mean=0.0, std=0.02)

    def add_reserve_param(self, add_r, advance_learn=True):
        for _ in range(add_r):
            e = nn.Parameter(self.weight.new_zeros(1, 1), requires_grad=False)
            a = nn.Parameter(self.weight.new_zeros((1, self.in_features)), requires_grad=advance_learn)
            b = nn.Parameter(self.weight.new_zeros((self.out_features, 1)), requires_grad=advance_learn)
            e[0][0] = 1e-5 if advance_learn else 0.
            nn.init.normal_(a, mean=0.0, std=0.02)
            nn.init.normal_(b, mean=0.0, std=0.02)
            self.lora_E.append(e)
            self.lora_A.append(a)
            self.lora_B.append(b)

    def get_dynamic_lora_metadata(self):
        list_lengths = (len(self.lora_A), len(self.lora_E), len(self.lora_B))
        if len(set(list_lengths)) != 1:
            raise ValueError("Dynamic LoRA parameter lists have inconsistent lengths.")

        a_capacity = sum(parameter.size(0) for parameter in self.lora_A)
        e_capacity = sum(parameter.numel() for parameter in self.lora_E)
        b_capacity = sum(parameter.size(1) for parameter in self.lora_B)
        if a_capacity != e_capacity or a_capacity != b_capacity:
            raise ValueError("Dynamic LoRA parameter lists have inconsistent rank capacity.")

        active_rank_value = float(self.ranknum.item())
        active_rank = int(round(active_rank_value))
        if (
            not math.isfinite(active_rank_value)
            or abs(active_rank_value - active_rank) > 1e-6
            or active_rank < 1
            or active_rank > a_capacity
        ):
            raise ValueError("Dynamic LoRA active rank is invalid.")

        return {
            "parameter_list_length": list_lengths[0],
            "rank_component_count": a_capacity,
            "active_rank": active_rank,
            "lora_A_requires_grad": [parameter.requires_grad for parameter in self.lora_A],
            "lora_E_requires_grad": [parameter.requires_grad for parameter in self.lora_E],
            "lora_B_requires_grad": [parameter.requires_grad for parameter in self.lora_B],
            "ranknum_requires_grad": self.ranknum.requires_grad,
        }

    def set_dynamic_lora_metadata(self, metadata):
        if not isinstance(metadata, dict):
            raise ValueError("Dynamic LoRA module metadata must be a dictionary.")
        if self.merged:
            raise ValueError("Cannot resize a merged dynamic LoRA module.")

        target_length = int(metadata.get("parameter_list_length", 0))
        target_capacity = int(metadata.get("rank_component_count", 0))
        active_rank = int(metadata.get("active_rank", 0))
        if target_length < 1 or target_capacity < 1 or active_rank < 1:
            raise ValueError("Dynamic LoRA ranks and parameter-list length must be positive.")

        current_lengths = (len(self.lora_A), len(self.lora_E), len(self.lora_B))
        if len(set(current_lengths)) != 1:
            raise ValueError("Dynamic LoRA parameter lists have inconsistent lengths.")

        if target_length < current_lengths[0]:
            self.lora_A = nn.ParameterList(list(self.lora_A)[:target_length])
            self.lora_E = nn.ParameterList(list(self.lora_E)[:target_length])
            self.lora_B = nn.ParameterList(list(self.lora_B)[:target_length])
        elif target_length > current_lengths[0]:
            self.add_reserve_param(target_length - current_lengths[0], advance_learn=False)

        restored_a_capacity = sum(parameter.size(0) for parameter in self.lora_A)
        restored_e_capacity = sum(parameter.numel() for parameter in self.lora_E)
        restored_b_capacity = sum(parameter.size(1) for parameter in self.lora_B)
        if (
            restored_a_capacity != target_capacity
            or restored_e_capacity != target_capacity
            or restored_b_capacity != target_capacity
            or active_rank > target_capacity
        ):
            raise ValueError(
                "Dynamic LoRA metadata is incompatible with the configured initial rank."
            )

        flag_fields = (
            ("lora_A_requires_grad", self.lora_A),
            ("lora_E_requires_grad", self.lora_E),
            ("lora_B_requires_grad", self.lora_B),
        )
        for field_name, parameters in flag_fields:
            flags = metadata.get(field_name)
            if not isinstance(flags, list) or len(flags) != target_length:
                raise ValueError("Invalid dynamic LoRA trainability metadata for %s." % field_name)
            for parameter, requires_grad in zip(parameters, flags):
                parameter.requires_grad_(bool(requires_grad))

        self.ranknum.requires_grad_(bool(metadata.get("ranknum_requires_grad", False)))
        with torch.no_grad():
            self.ranknum.fill_(float(active_rank))
    
    def train(self, mode: bool = True):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if mode == True:
            self.lora_A.requires_grad = True
            self.lora_E.requires_grad = True
            self.lora_B.requires_grad = True
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    self.weight.data -= T(
                        self.W(self.lora_A, self.lora_E, self.lora_B, self.scaling, self.ranknum)
                    )
                self.merged = False
        else:
            self.lora_A.requires_grad = False
            self.lora_E.requires_grad = False
            self.lora_B.requires_grad = False
    
    def eval(self):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        nn.Linear.eval(self)
        if self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.r > 0:
                self.weight.data += T(
                    self.W(self.lora_A, self.lora_E, self.lora_B, self.scaling, self.ranknum)
                )
            self.merged = True

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:
                try:
                    result += (
                        self.lora_dropout(x) @ self.W(self.lora_A, self.lora_E, self.lora_B, self.scaling, self.ranknum).T
                    )
                except:
                    ipdb.set_trace()
                    print(self.W)
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)


class RankAllocator(object):
    """
    The RankAllocator for IncreLoRA Model that will be called every training step. 

    Args:
        model: the model that we apply IncreLoRA to.
        lora_r (`int`): The initial rank for each incremental matrix.
        target_rank (`int`): The target average rank of incremental matrix.
        init_warmup (`int`): The steps of initial fine-tuning warmup.
        incre_interval (`int`): The time internval between two budget allocations.
        top_h (`int`): The number of modules selected.
        advance_learn (`bool`): With or without advance learning.
        beta1 (`float`): The hyperparameter of EMA for sensitivity smoothing.
        beta2 (`float`): The hyperparameter of EMA for undertainty quantification.
        total_step (`int`): The total training steps, correctly configured before training.
        target_total_rank (`Optinal[int]`): The speficified final total rank. 
        tb_writter (`SummaryWriter`): Tensorboard SummaryWriter. 
        tb_writter_loginterval (`int`): The logging interval of SummaryWriter. 
    """
    def __init__(
        self, model, 
        lora_r:int,
        target_rank:int, 
        init_warmup:int, 
        incre_interval:int,
        top_h:int,
        advance_learn:bool,
        beta1:float, 
        beta2:float, 
        total_step:Optional[int]=None, 
        target_total_rank:Optional[int]=None,
        weight_decay=None,
        incre_rank_num=None,
        tb_writter=None,
        tb_writter_loginterval:int=500, 
        rank_allocator:str="greedy",
        ga_population:int=12,
        ga_generations:int=4,
        ga_mutation_rate:float=0.10,
        ga_crossover_rate:float=0.80,
        ga_interaction_weight:float=0.20,
        ga_redundancy_weight:float=0.20,
        ga_cost_weight:float=0.30,
        ga_diversity_weight:float=0.10,
        ga_local_search:bool=False,
        ga_budget_reference_pattern:Optional[str]=None,
        ga_max_final_trainable_params:Optional[int]=None,
        ga_budget_ratio:float=0.98,
        ga_gain_tolerance:float=0.05,
        ga_allow_variable_event_rank:bool=False,
        ga_max_greedy_replacements:int=2,
        ga_calibration_batches:int=3,
        ga_calibration_batch_size:int=8,
        ga_calibration_seed_offset:int=1000,
        ga_calibration_topk:int=6,
        ga_calibration_lcb_beta:float=0.5,
        ga_quality_absolute_tolerance:float=0.0,
        ga_quality_relative_tolerance:float=0.01,
        ga_greedy_quality_floor_ratio:float=0.99,
        ga_greedy_quality_floor_absolute:float=0.0,
        ga_min_calibrated_marginal_gain:float=0.0,
        ga_allocation_stop_patience:int=2,
        ga_min_event_rank:int=1,
        ga_max_event_rank:Optional[int]=None,
        ga_min_consolidation_steps:int=300,
        ga_new_rank_lr_warmup_steps:int=25,
        greedy_reference_checkpoint:Optional[str]=None,
        training_seed:int=42,
    ):
        self.ave_target_rank = target_rank
        self.target_rank = target_total_rank
        self.lora_init_rank = lora_r 

        self.init_warmup = init_warmup
        self.incre_interval = incre_interval
        self.advance_learn = advance_learn
        self.top_h = top_h
        if incre_rank_num:
            self.incre_rank_num = incre_rank_num
        else:
            rank_dic = {2:1, 4:2, 6:3, 8:4}
            self.incre_rank_num = rank_dic[self.ave_target_rank]
        
        self.beta1 = beta1
        self.beta2 = beta2
        self.total_step = total_step

        if rank_allocator not in (
            "greedy",
            "genetic",
            "genetic_budgeted",
            "genetic_budgeted_calibrated",
        ):
            raise ValueError(
                "rank_allocator must be 'greedy', 'genetic', 'genetic_budgeted', or "
                "'genetic_budgeted_calibrated'."
            )
        if ga_population <= 0:
            raise ValueError("ga_population must be positive.")
        if ga_generations < 0:
            raise ValueError("ga_generations must be nonnegative.")
        if not math.isfinite(ga_mutation_rate) or not 0.0 <= ga_mutation_rate <= 1.0:
            raise ValueError("ga_mutation_rate must be between 0 and 1.")
        if not math.isfinite(ga_crossover_rate) or not 0.0 <= ga_crossover_rate <= 1.0:
            raise ValueError("ga_crossover_rate must be between 0 and 1.")
        if not math.isfinite(ga_interaction_weight) or ga_interaction_weight < 0.0:
            raise ValueError("ga_interaction_weight must be nonnegative.")
        if not math.isfinite(ga_redundancy_weight) or ga_redundancy_weight < 0.0:
            raise ValueError("ga_redundancy_weight must be nonnegative.")
        if not math.isfinite(ga_cost_weight) or ga_cost_weight < 0.0:
            raise ValueError("ga_cost_weight must be nonnegative.")
        if not math.isfinite(ga_diversity_weight) or ga_diversity_weight < 0.0:
            raise ValueError("ga_diversity_weight must be nonnegative.")
        if not math.isfinite(ga_gain_tolerance) or not 0.0 <= ga_gain_tolerance <= 1.0:
            raise ValueError("ga_gain_tolerance must be between 0 and 1.")
        for name, value in (
            ("ga_calibration_lcb_beta", ga_calibration_lcb_beta),
            ("ga_quality_absolute_tolerance", ga_quality_absolute_tolerance),
            ("ga_quality_relative_tolerance", ga_quality_relative_tolerance),
            ("ga_greedy_quality_floor_absolute", ga_greedy_quality_floor_absolute),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError("%s must be finite and nonnegative." % name)
        if (
            not math.isfinite(ga_greedy_quality_floor_ratio)
            or not 0.0 <= ga_greedy_quality_floor_ratio <= 1.0
        ):
            raise ValueError("ga_greedy_quality_floor_ratio must be between 0 and 1.")
        if not math.isfinite(ga_min_calibrated_marginal_gain):
            raise ValueError("ga_min_calibrated_marginal_gain must be finite.")
        if rank_allocator == "genetic_budgeted" and ga_local_search:
            raise ValueError("genetic_budgeted forbids local search; set ga_local_search=false.")
        if rank_allocator == "genetic_budgeted" and ga_allow_variable_event_rank:
            raise ValueError(
                "ga_allow_variable_event_rank is not implemented; fixed top_h is required."
            )
        if rank_allocator == "genetic_budgeted_calibrated" and ga_local_search:
            raise ValueError(
                "genetic_budgeted_calibrated forbids local search; set ga_local_search=false."
            )
        if rank_allocator == "genetic_budgeted_calibrated" and not ga_allow_variable_event_rank:
            raise ValueError(
                "genetic_budgeted_calibrated requires ga_allow_variable_event_rank=true."
            )
        self.rank_allocator = rank_allocator
        self.ga_population = ga_population
        self.ga_generations = ga_generations
        self.ga_mutation_rate = ga_mutation_rate
        self.ga_crossover_rate = ga_crossover_rate
        self.ga_interaction_weight = ga_interaction_weight
        self.ga_redundancy_weight = ga_redundancy_weight
        self.ga_cost_weight = ga_cost_weight
        self.ga_diversity_weight = ga_diversity_weight
        self.ga_local_search = ga_local_search
        self.ga_budget_reference_pattern = ga_budget_reference_pattern
        self.ga_max_final_trainable_params = ga_max_final_trainable_params
        self.ga_budget_ratio = float(ga_budget_ratio)
        self.ga_gain_tolerance = float(ga_gain_tolerance)
        self.ga_allow_variable_event_rank = bool(ga_allow_variable_event_rank)
        self.ga_max_greedy_replacements = int(ga_max_greedy_replacements)
        self.ga_calibration_batches = int(ga_calibration_batches)
        self.ga_calibration_batch_size = int(ga_calibration_batch_size)
        self.ga_calibration_seed_offset = int(ga_calibration_seed_offset)
        self.ga_calibration_topk = int(ga_calibration_topk)
        self.ga_calibration_lcb_beta = float(ga_calibration_lcb_beta)
        self.ga_quality_absolute_tolerance = float(ga_quality_absolute_tolerance)
        self.ga_quality_relative_tolerance = float(ga_quality_relative_tolerance)
        self.ga_greedy_quality_floor_ratio = float(ga_greedy_quality_floor_ratio)
        self.ga_greedy_quality_floor_absolute = float(ga_greedy_quality_floor_absolute)
        self.ga_min_calibrated_marginal_gain = float(ga_min_calibrated_marginal_gain)
        self.ga_allocation_stop_patience = int(ga_allocation_stop_patience)
        self.ga_min_event_rank = int(ga_min_event_rank)
        self.ga_max_event_rank = (
            self.top_h if ga_max_event_rank is None else int(ga_max_event_rank)
        )
        self.ga_min_consolidation_steps = int(ga_min_consolidation_steps)
        self.ga_new_rank_lr_warmup_steps = int(ga_new_rank_lr_warmup_steps)
        self.greedy_reference_checkpoint = greedy_reference_checkpoint
        self.training_seed = training_seed

        if self.ga_max_greedy_replacements < 0:
            raise ValueError("ga_max_greedy_replacements must be nonnegative.")
        if self.ga_calibration_batches <= 0 or self.ga_calibration_batch_size <= 0:
            raise ValueError("Calibration batch count and batch size must be positive.")
        if self.ga_calibration_topk < 6:
            raise ValueError("ga_calibration_topk must be at least 6.")
        if self.ga_allocation_stop_patience <= 0:
            raise ValueError("ga_allocation_stop_patience must be positive.")
        if self.ga_min_event_rank <= 0 or self.ga_max_event_rank < self.ga_min_event_rank:
            raise ValueError("Calibrated event rank bounds are invalid.")
        if self.ga_max_event_rank > self.top_h:
            raise ValueError("ga_max_event_rank cannot exceed top_h.")
        if self.ga_min_consolidation_steps < 0 or self.ga_new_rank_lr_warmup_steps < 0:
            raise ValueError("Consolidation and new-rank warmup steps must be nonnegative.")

        self.model = model
        self.weight_decay = weight_decay
        self.ipt = {} 
        self.exp_avg_ipt = {}
        self.exp_avg_unc = {}
        self.cat_ipt = {}
        self.rank_pattern = {} 
        self.get_lora_param_name()
        self.total_rank = self.initial_total_rank 
        self.history_window = 20
        self.importance_history = (
            {name: [] for name in self.name_set}
            if self.rank_allocator in (
                "genetic",
                "genetic_budgeted",
                "genetic_budgeted_calibrated",
            )
            else None
        )

        self.reference_greedy_cost = None
        self.target_final_trainable_params = None
        self.theoretical_minimum_final_cost = None
        self.budget_source = None
        self.budget_metadata = None
        self.final_trajectory_metrics = None
        self.calibration_training_indices = None
        self.calibration_dataset_size = None
        self.calibration_dataset_fingerprint = None
        self.zero_rank_event_counter = 0
        self.allocation_stopped = False
        self.allocation_stopped_step = None
        self.consolidation_started_step = None
        self.consolidation_remaining_steps = None
        self.new_rank_warmup_state = {}
        self._warmup_parameter_snapshots = {}
        self.optimizer_parameter_groups = None
        self.training_configuration = None
        self._checkpoint_total_step = None
        self._loaded_calibrated_metadata = None
        if self.rank_allocator in ("genetic_budgeted", "genetic_budgeted_calibrated"):
            self._initialize_hard_budget(model)

        self.tb_writter = tb_writter
        self.log_interval = tb_writter_loginterval 
        
        assert (self.beta1<1 and self.beta1>0)
        assert (self.beta2<1 and self.beta2>0)

    def _initialize_hard_budget(self, model):
        if not math.isfinite(self.ga_budget_ratio) or not 0.0 < self.ga_budget_ratio <= 1.0:
            raise ValueError("ga_budget_ratio must be greater than 0 and at most 1.")

        targets = []
        reference_pattern = None
        if self.ga_budget_reference_pattern:
            reference_pattern = load_rank_pattern(self.ga_budget_reference_pattern)
            self.reference_greedy_cost = get_rank_pattern_active_model_parameter_count(
                model, reference_pattern
            )
            reference_total_rank = sum(_plain_rank_map(reference_pattern).values())
            if reference_total_rank != self.target_rank:
                raise ValueError(
                    "Greedy reference total rank %s does not match configured target total rank %s."
                    % (reference_total_rank, self.target_rank)
                )
            ratio_target = int(math.floor(self.reference_greedy_cost * self.ga_budget_ratio))
            if self.ga_budget_ratio < 1.0 and ratio_target >= self.reference_greedy_cost:
                raise ValueError("Budget ratio does not produce a strict reduction from Greedy.")
            targets.append(("greedy_rank_pattern", ratio_target))

        if self.ga_max_final_trainable_params is not None:
            maximum = int(self.ga_max_final_trainable_params)
            if maximum <= 0:
                raise ValueError("ga_max_final_trainable_params must be positive.")
            targets.append(("explicit_maximum", maximum))
        if not targets:
            raise ValueError(
                "genetic_budgeted requires --ga_budget_reference_pattern or "
                "--ga_max_final_trainable_params."
            )

        self.target_final_trainable_params = min(target for _, target in targets)
        self.budget_source = "+".join(source for source, _ in targets)
        initial_cost = get_active_model_parameter_count(model)
        remaining_increments = self.target_rank - self.initial_total_rank
        if remaining_increments < 0:
            raise ValueError("Configured target rank is below the model's initial LoRA rank.")
        module_costs = [
            get_module_rank_one_cost(module)
            for module in model.modules()
            if isinstance(module, SVDLinear)
        ]
        if not module_costs:
            raise ValueError("Budgeted allocation requires at least one SVDLinear module.")
        if self.rank_allocator == "genetic_budgeted_calibrated":
            # Variable-rank growth makes the hard budget an upper bound. No future
            # increment is compulsory, so the current active architecture is the
            # true feasibility floor.
            self.theoretical_minimum_final_cost = initial_cost
        else:
            self.theoretical_minimum_final_cost = (
                initial_cost + remaining_increments * min(module_costs)
            )
        feasible = self.theoretical_minimum_final_cost <= self.target_final_trainable_params
        if not feasible:
            raise ValueError(
                "Infeasible final parameter budget: initial_cost=%s target_cost=%s "
                "total_remaining_rank_increments=%s theoretical_minimum_achievable_cost=%s "
                "reason=%s"
                % (
                    initial_cost,
                    self.target_final_trainable_params,
                    remaining_increments,
                    self.theoretical_minimum_final_cost,
                    (
                        "current active architecture already exceeds the supplied budget"
                        if self.rank_allocator == "genetic_budgeted_calibrated"
                        else "fixed total-rank schedule cannot fit the supplied budget"
                    ),
                )
            )

        self.budget_metadata = {
            "allocator_mode": self.rank_allocator,
            "reference_cost": self.reference_greedy_cost,
            "target_cost": self.target_final_trainable_params,
            "budget_ratio": self.ga_budget_ratio,
            "current_accumulated_cost": initial_cost,
            "current_total_rank": self.initial_total_rank,
        }
        if self.rank_allocator == "genetic_budgeted_calibrated":
            self.budget_metadata.update(self._calibrated_configuration_metadata())
            self.budget_metadata["greedy_reference_checkpoint"] = (
                self.greedy_reference_checkpoint
            )
        loaded_metadata = getattr(getattr(model, "config", None), BUDGETED_ALLOCATOR_METADATA, None)
        if loaded_metadata is not None:
            if self.rank_allocator == "genetic_budgeted_calibrated":
                self._restore_calibrated_metadata(model, loaded_metadata)
            else:
                from transformers.budgeted_evo_allocator import validate_budget_metadata

                validate_budget_metadata(self.budget_metadata, loaded_metadata)
            loaded_trajectory = loaded_metadata.get("final_trajectory")
            if isinstance(loaded_trajectory, dict):
                self.final_trajectory_metrics = dict(loaded_trajectory)
                self.budget_metadata["final_trajectory"] = dict(loaded_trajectory)
        self._save_budget_metadata(model)
        logger.info(
            "Budgeted IncreLoRA initialization allocator_mode=%s "
            "budget_source=%s greedy_reference_pattern=%s reference_greedy_parameter_count=%s "
            "budget_ratio=%.6f target_final_parameter_count=%s "
            "initial_active_model_parameter_count=%s "
            "initial_runtime_trainable_parameter_count=%s full_model_parameter_count=%s "
            "theoretical_minimum_final_parameter_count=%s budget_feasible=%s "
            "calibration_training_indices=%s calibration_batch_count=%s "
            "calibration_shortlist_size=%s variable_rank_enabled=%s "
            "maximum_greedy_replacements=%s minimum_consolidation_steps=%s",
            self.rank_allocator,
            self.budget_source,
            self.ga_budget_reference_pattern,
            self.reference_greedy_cost,
            self.ga_budget_ratio,
            self.target_final_trainable_params,
            initial_cost,
            get_runtime_trainable_parameter_count(model),
            get_full_model_parameter_count(model),
            self.theoretical_minimum_final_cost,
            feasible,
            self.calibration_training_indices,
            self.ga_calibration_batches if self.rank_allocator == "genetic_budgeted_calibrated" else None,
            self.ga_calibration_topk if self.rank_allocator == "genetic_budgeted_calibrated" else None,
            self.ga_allow_variable_event_rank,
            self.ga_max_greedy_replacements if self.rank_allocator == "genetic_budgeted_calibrated" else None,
            self.ga_min_consolidation_steps if self.rank_allocator == "genetic_budgeted_calibrated" else None,
        )

    def _calibrated_configuration_metadata(self):
        return {
            "evolution_configuration": {
                "population": self.ga_population,
                "generations": self.ga_generations,
                "mutation_rate": self.ga_mutation_rate,
                "crossover_rate": self.ga_crossover_rate,
                "interaction_weight": self.ga_interaction_weight,
                "redundancy_weight": self.ga_redundancy_weight,
                "cost_weight": self.ga_cost_weight,
                "diversity_weight": self.ga_diversity_weight,
                "local_search": False,
                "training_seed": self.training_seed,
            },
            "allocation_schedule_configuration": {
                "top_h": self.top_h,
                "rank_increment": self.incre_rank_num,
                "maximum_total_rank": self.target_rank,
                "initial_warmup": self.init_warmup,
                "allocation_interval": self.incre_interval,
                "importance_beta1": self.beta1,
                "importance_beta2": self.beta2,
                "advance_learn": self.advance_learn,
                "weight_decay": self.weight_decay,
            },
            "calibration_configuration": {
                "batches": self.ga_calibration_batches,
                "batch_size": self.ga_calibration_batch_size,
                "seed_offset": self.ga_calibration_seed_offset,
                "shortlist_topk": self.ga_calibration_topk,
                "lcb_beta": self.ga_calibration_lcb_beta,
            },
            "quality_tolerances": {
                "absolute": self.ga_quality_absolute_tolerance,
                "relative": self.ga_quality_relative_tolerance,
                "greedy_floor_ratio": self.ga_greedy_quality_floor_ratio,
                "greedy_floor_absolute": self.ga_greedy_quality_floor_absolute,
                "minimum_marginal_gain": self.ga_min_calibrated_marginal_gain,
            },
            "greedy_trust_region": {
                "maximum_replacements": self.ga_max_greedy_replacements,
            },
            "adaptive_growth_configuration": {
                "variable_event_rank": self.ga_allow_variable_event_rank,
                "minimum_event_rank": self.ga_min_event_rank,
                "maximum_event_rank": self.ga_max_event_rank,
                "stop_patience": self.ga_allocation_stop_patience,
                "minimum_consolidation_steps": self.ga_min_consolidation_steps,
                "new_rank_lr_warmup_steps": self.ga_new_rank_lr_warmup_steps,
            },
        }

    def _restore_calibrated_metadata(self, model, loaded_metadata):
        if not isinstance(loaded_metadata, dict):
            raise ValueError("Checkpoint does not contain calibrated allocator metadata.")
        expected_static = {
            "allocator_mode": self.rank_allocator,
            "reference_cost": self.reference_greedy_cost,
            "target_cost": self.target_final_trainable_params,
            "budget_ratio": self.ga_budget_ratio,
        }
        expected_static.update(self._calibrated_configuration_metadata())
        mismatches = {
            key: (expected_static.get(key), loaded_metadata.get(key))
            for key in expected_static
            if expected_static.get(key) != loaded_metadata.get(key)
        }
        if mismatches:
            raise ValueError(
                "Incompatible calibrated allocator checkpoint metadata: %s" % mismatches
            )

        current_total_rank = int(
            sum(
                round(float(module.ranknum.item()))
                for module in model.modules()
                if isinstance(module, SVDLinear)
            )
        )
        loaded_total_rank = int(loaded_metadata.get("current_total_rank", current_total_rank))
        if current_total_rank != loaded_total_rank:
            raise ValueError(
                "Calibrated checkpoint active-rank mismatch: model=%s metadata=%s."
                % (current_total_rank, loaded_total_rank)
            )
        current_active = get_active_model_parameter_count(model)
        loaded_active = int(loaded_metadata.get("current_accumulated_cost", current_active))
        if current_active != loaded_active:
            raise ValueError(
                "Calibrated checkpoint active-parameter mismatch: model=%s metadata=%s."
                % (current_active, loaded_active)
            )

        self.total_rank = current_total_rank
        self.calibration_training_indices = loaded_metadata.get("calibration_training_indices")
        self.calibration_dataset_size = loaded_metadata.get("calibration_dataset_size")
        self.calibration_dataset_fingerprint = loaded_metadata.get(
            "calibration_dataset_fingerprint"
        )
        self.zero_rank_event_counter = int(loaded_metadata.get("zero_rank_event_counter", 0))
        self.allocation_stopped = bool(loaded_metadata.get("allocation_stopped", False))
        self.allocation_stopped_step = loaded_metadata.get("allocation_stopped_step")
        self.consolidation_started_step = loaded_metadata.get("consolidation_started_step")
        self.consolidation_remaining_steps = loaded_metadata.get("consolidation_remaining_steps")
        self._checkpoint_total_step = loaded_metadata.get("total_optimization_steps")
        self.training_configuration = loaded_metadata.get("training_configuration")
        self.new_rank_warmup_state = {
            str(name): dict(state)
            for name, state in loaded_metadata.get("new_rank_warmup_state", {}).items()
        }
        optimizer_groups = loaded_metadata.get("optimizer_parameter_groups")
        self.optimizer_parameter_groups = (
            [list(group) for group in optimizer_groups]
            if isinstance(optimizer_groups, list)
            else None
        )
        self.rank_pattern = {
            str(name): int(rank)
            for name, rank in loaded_metadata.get("active_rank_pattern", {}).items()
        }
        saved_ipt = loaded_metadata.get("importance_state", {})
        for attribute_name, target in (
            ("ipt", self.ipt),
            ("exp_avg_ipt", self.exp_avg_ipt),
            ("exp_avg_unc", self.exp_avg_unc),
        ):
            values = saved_ipt.get(attribute_name, {})
            for name, value in values.items():
                if name not in self.name_set:
                    raise ValueError(
                        "Calibrated checkpoint importance state contains an unknown module: %s"
                        % name
                    )
                # RankAllocator is constructed before Trainer moves the model
                # to CUDA. Keep serialized scalars device-neutral here; the
                # first update coerces them to layer.score's live dtype/device.
                target[name] = float(value)
        saved_history = saved_ipt.get("importance_history", {})
        if isinstance(saved_history, dict):
            self.importance_history = {
                name: [float(value) for value in saved_history.get(name, [])]
                for name in self.name_set
            }
        if self.allocation_stopped:
            self._freeze_consolidation_reserves(model)
        self._loaded_calibrated_metadata = dict(loaded_metadata)
        self.budget_metadata.update(dict(loaded_metadata))

    def _synchronize_calibrated_metadata(self, model):
        if self.rank_allocator != "genetic_budgeted_calibrated":
            return
        active_rank_pattern = {
            name: int(round(float(module.ranknum.item())))
            for name, module in model.named_modules()
            if isinstance(module, SVDLinear)
        }
        self.rank_pattern = dict(active_rank_pattern)
        self.budget_metadata.update(
            {
                "current_accumulated_cost": get_active_model_parameter_count(model),
                "current_total_rank": sum(active_rank_pattern.values()),
                "calibration_training_indices": self.calibration_training_indices,
                "calibration_dataset_size": self.calibration_dataset_size,
                "calibration_dataset_fingerprint": self.calibration_dataset_fingerprint,
                "zero_rank_event_counter": self.zero_rank_event_counter,
                "allocation_stopped": self.allocation_stopped,
                "allocation_stopped_step": self.allocation_stopped_step,
                "consolidation_started_step": self.consolidation_started_step,
                "consolidation_remaining_steps": self.consolidation_remaining_steps,
                "active_rank_pattern": active_rank_pattern,
                "new_rank_warmup_state": {
                    name: dict(state) for name, state in self.new_rank_warmup_state.items()
                },
                "optimizer_parameter_groups": self.optimizer_parameter_groups,
                "total_optimization_steps": (
                    self.total_step
                    if self.total_step is not None
                    else self._checkpoint_total_step
                ),
                "training_configuration": self.training_configuration,
                "importance_state": {
                    "ipt": {name: self._scalar_value(value) for name, value in self.ipt.items()},
                    "exp_avg_ipt": {
                        name: self._scalar_value(value)
                        for name, value in self.exp_avg_ipt.items()
                    },
                    "exp_avg_unc": {
                        name: self._scalar_value(value)
                        for name, value in self.exp_avg_unc.items()
                    },
                    "importance_history": {
                        name: [float(value) for value in history]
                        for name, history in (self.importance_history or {}).items()
                    },
                },
            }
        )
        if self.final_trajectory_metrics is not None:
            self.budget_metadata["final_trajectory"] = dict(self.final_trajectory_metrics)

    def _save_budget_metadata(self, model):
        self._synchronize_calibrated_metadata(model)
        if self.budget_metadata is not None and getattr(model, "config", None) is not None:
            setattr(model.config, BUDGETED_ALLOCATOR_METADATA, dict(self.budget_metadata))
        
            
    def set_total_step(self, total_step:int): 
        # Set total step number 
        total_step = int(total_step)
        if (
            self.rank_allocator == "genetic_budgeted_calibrated"
            and self._checkpoint_total_step is not None
            and int(self._checkpoint_total_step) != total_step
        ):
            raise ValueError(
                "Incompatible calibrated resume total optimization steps: "
                "checkpoint=%s current=%s."
                % (self._checkpoint_total_step, total_step)
            )
        self.total_step = total_step
        if self.rank_allocator == "genetic_budgeted_calibrated":
            self.budget_metadata["total_optimization_steps"] = total_step
            self._save_budget_metadata(self.model)
            maximum_events = max(
                0,
                math.ceil(
                    max(0, self.target_rank - self.total_rank)
                    / float(max(1, self.ga_max_event_rank * self.incre_rank_num))
                ),
            )
            logger.info(
                "Calibrated allocation schedule total_steps=%s current_total_rank=%s "
                "maximum_target_rank=%s maximum_growth_events=%s "
                "minimum_consolidation_steps=%s",
                total_step,
                self.total_rank,
                self.target_rank,
                maximum_events,
                self.ga_min_consolidation_steps,
            )
            return
        rank_per_round = self.top_h * self.incre_rank_num
        total_round = math.ceil((self.target_rank - self.initial_total_rank) / rank_per_round)
        total_incre_step = self.incre_interval * total_round
                            
        print("Total incremental step: total_incre_step: {}, of total steps: {:.0%}"
              .format(total_incre_step, total_incre_step / total_step))

    def set_training_configuration(self, configuration):
        """Persist decision-relevant Trainer settings and reject resume drift."""

        if self.rank_allocator != "genetic_budgeted_calibrated":
            return
        normalized = json.loads(json.dumps(configuration, sort_keys=True))
        loaded = (
            self._loaded_calibrated_metadata.get("training_configuration")
            if isinstance(self._loaded_calibrated_metadata, dict)
            else None
        )
        if loaded is not None and loaded != normalized:
            raise ValueError(
                "Incompatible calibrated resume training configuration: "
                "checkpoint=%s current=%s." % (loaded, normalized)
            )
        if isinstance(self._loaded_calibrated_metadata, dict) and loaded is None:
            raise ValueError(
                "Calibrated resume checkpoint is missing training_configuration metadata."
            )
        self.training_configuration = normalized
        self.budget_metadata["training_configuration"] = normalized
        self._save_budget_metadata(self.model)

    def get_or_create_calibration_indices(self, dataset_size, dataset_fingerprint=None):
        if self.rank_allocator != "genetic_budgeted_calibrated":
            return None
        dataset_size = int(dataset_size)
        required = 2 * self.ga_calibration_batches * self.ga_calibration_batch_size
        if dataset_size < required:
            raise ValueError(
                "Training split is too small for calibrated allocation: size=%s required=%s."
                % (dataset_size, required)
            )
        rng = __import__("random").Random(
            int(self.training_seed) + int(self.ga_calibration_seed_offset)
        )
        sampled = rng.sample(range(dataset_size), required)
        expected = []
        cursor = 0
        for _ in range(self.ga_calibration_batches):
            batch_a = sampled[cursor : cursor + self.ga_calibration_batch_size]
            cursor += self.ga_calibration_batch_size
            batch_b = sampled[cursor : cursor + self.ga_calibration_batch_size]
            cursor += self.ga_calibration_batch_size
            expected.append({"batch_a": batch_a, "batch_b": batch_b})

        normalized_fingerprint = (
            None if dataset_fingerprint is None else str(dataset_fingerprint)
        )
        if self.calibration_training_indices is not None:
            if self.calibration_training_indices != expected:
                raise ValueError(
                    "Calibrated resume training indices do not match the deterministic training split."
                )
            if self.calibration_dataset_size not in (None, dataset_size):
                raise ValueError("Calibrated resume training dataset size is incompatible.")
            if (
                self.calibration_dataset_fingerprint is not None
                and normalized_fingerprint is not None
                and self.calibration_dataset_fingerprint != normalized_fingerprint
            ):
                raise ValueError("Calibrated resume training dataset fingerprint is incompatible.")
        else:
            self.calibration_training_indices = expected
        self.calibration_dataset_size = dataset_size
        self.calibration_dataset_fingerprint = normalized_fingerprint
        self._save_budget_metadata(self.model)
        logger.info(
            "Calibrated training-only indices=%s dataset_size=%s dataset_fingerprint=%s "
            "calibration_folds=%s calibration_batch_size=%s",
            self.calibration_training_indices,
            self.calibration_dataset_size,
            self.calibration_dataset_fingerprint,
            self.ga_calibration_batches,
            self.ga_calibration_batch_size,
        )
        return [dict(pair) for pair in self.calibration_training_indices]

    def _add_calibrated_optimizer_parameters(self, optimizer, parameters):
        parameters = list(parameters)
        if not parameters:
            return
        existing = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        new_parameters = [parameter for parameter in parameters if id(parameter) not in existing]
        if not new_parameters:
            return
        candidate_groups = [
            group
            for group in optimizer.param_groups
            if float(group.get("weight_decay", 0.0)) == float(self.weight_decay or 0.0)
        ]
        target_group = candidate_groups[0] if candidate_groups else optimizer.param_groups[0]
        target_group["params"].extend(new_parameters)

    def capture_checkpoint_state(self, model, optimizer, global_step):
        if self.rank_allocator != "genetic_budgeted_calibrated":
            return
        name_by_id = {id(parameter): name for name, parameter in model.named_parameters()}
        groups = []
        seen = set()
        for group_index, group in enumerate(optimizer.param_groups):
            names = []
            for parameter in group["params"]:
                name = name_by_id.get(id(parameter))
                if name is None:
                    raise ValueError(
                        "Optimizer group %s contains a parameter absent from the calibrated model."
                        % group_index
                    )
                if name in seen:
                    raise ValueError("Optimizer parameter '%s' appears in multiple groups." % name)
                seen.add(name)
                names.append(name)
            groups.append(names)
        model_names = set(name_by_id.values())
        if seen != model_names:
            missing = sorted(model_names.difference(seen))
            raise ValueError(
                "Calibrated optimizer layout omits model parameters: %s" % missing[:3]
            )
        self.optimizer_parameter_groups = groups
        self.global_step = int(global_step)
        self._save_budget_metadata(model)

    def prepare_optimizer_for_resume(self, model, optimizer):
        if (
            self.rank_allocator != "genetic_budgeted_calibrated"
            or self.optimizer_parameter_groups is None
        ):
            return
        if len(optimizer.param_groups) != len(self.optimizer_parameter_groups):
            raise ValueError(
                "Calibrated optimizer group count is incompatible with the checkpoint: "
                "current=%s checkpoint=%s."
                % (len(optimizer.param_groups), len(self.optimizer_parameter_groups))
            )
        parameters = dict(model.named_parameters())
        expected_names = [
            name for group in self.optimizer_parameter_groups for name in group
        ]
        if len(expected_names) != len(set(expected_names)):
            raise ValueError("Calibrated optimizer checkpoint contains duplicate parameter names.")
        if set(expected_names) != set(parameters):
            raise ValueError("Calibrated optimizer parameter manifest is incompatible with the model.")
        for group, names in zip(optimizer.param_groups, self.optimizer_parameter_groups):
            group["params"] = [parameters[name] for name in names]
        logger.info(
            "Restored calibrated optimizer parameter ordering groups=%s parameters=%s",
            len(optimizer.param_groups),
            len(expected_names),
        )

    def register_new_rank_warmup(self, model, parameter_ids, global_step):
        if self.ga_new_rank_lr_warmup_steps <= 0:
            return
        names = {
            id(parameter): name for name, parameter in model.named_parameters()
        }
        for parameter_id in parameter_ids:
            name = names.get(parameter_id)
            if name is None:
                raise ValueError("Newly activated LoRA parameter is absent from the model.")
            self.new_rank_warmup_state[name] = {
                "activation_step": int(global_step),
                "steps_completed": 0,
                "warmup_steps": self.ga_new_rank_lr_warmup_steps,
            }

    def snapshot_new_rank_warmup_parameters(self, model):
        if self.rank_allocator != "genetic_budgeted_calibrated":
            return
        parameters = dict(model.named_parameters())
        self._warmup_parameter_snapshots = {
            name: {
                "value": parameters[name].detach().clone(),
                "had_gradient": parameters[name].grad is not None,
            }
            for name in self.new_rank_warmup_state
            if name in parameters
        }

    def apply_new_rank_lr_warmup(self, model, optimizer_step_was_run=True):
        if self.rank_allocator != "genetic_budgeted_calibrated":
            return
        if not optimizer_step_was_run:
            self._warmup_parameter_snapshots = {}
            return
        parameters = dict(model.named_parameters())
        completed = []
        with torch.no_grad():
            for name, snapshot in self._warmup_parameter_snapshots.items():
                state = self.new_rank_warmup_state.get(name)
                parameter = parameters.get(name)
                if state is None or parameter is None:
                    continue
                # E is activated after the allocation step's backward pass and
                # therefore has no gradient on that first real optimizer step.
                # Do not consume a per-parameter warmup tick on a no-op.
                if not snapshot["had_gradient"]:
                    continue
                before = snapshot["value"]
                next_step = int(state["steps_completed"]) + 1
                warmup_steps = max(1, int(state["warmup_steps"]))
                scale = min(1.0, next_step / float(warmup_steps))
                parameter.copy_(before + (parameter.detach() - before) * scale)
                state["steps_completed"] = next_step
                if next_step >= warmup_steps:
                    completed.append(name)
        for name in completed:
            self.new_rank_warmup_state.pop(name, None)
        self._warmup_parameter_snapshots = {}

    def _freeze_consolidation_reserves(self, model):
        deactivate_inactive_reserve_parameters(model)
        with torch.no_grad():
            for module in model.modules():
                if not isinstance(module, SVDLinear):
                    continue
                active_rank = int(round(float(module.ranknum.item())))
                for parameters, rank_dimension, zero_inactive in (
                    (module.lora_A, 0, False),
                    (module.lora_E, None, True),
                    (module.lora_B, 1, False),
                ):
                    cursor = 0
                    for parameter in parameters:
                        width = int(
                            parameter.numel()
                            if rank_dimension is None
                            else parameter.size(rank_dimension)
                        )
                        if cursor >= active_rank:
                            # Consolidation can begin after backward and before
                            # optimizer.step(). Clear stale reserve gradients so
                            # frozen capacity cannot receive one last update.
                            parameter.grad = None
                            parameter.requires_grad_(False)
                            if zero_inactive:
                                parameter.zero_()
                        cursor += width
                hook = getattr(module, "hook_handle", None)
                if hook is not None:
                    hook.remove()
                    module.hook_handle = None

    def _start_consolidation(self, model, global_step, reason):
        if self.rank_allocator != "genetic_budgeted_calibrated" or self.allocation_stopped:
            return
        self.allocation_stopped = True
        self.allocation_stopped_step = int(global_step)
        self.consolidation_started_step = int(global_step)
        self.consolidation_remaining_steps = max(0, int(self.total_step) - int(global_step))
        self._freeze_consolidation_reserves(model)
        self.record_final_trajectory(model, global_step=global_step)
        self._save_budget_metadata(model)
        logger.info(
            "Calibrated consolidation started reason=%s consolidation_started_step=%s "
            "consolidation_remaining_steps=%s fixed_total_active_rank=%s "
            "fixed_active_parameter_count=%s",
            reason,
            self.consolidation_started_step,
            self.consolidation_remaining_steps,
            self.final_trajectory_metrics["total_active_rank"],
            self.final_trajectory_metrics["active_model_parameter_count"],
        )

    def record_final_trajectory(self, model, global_step=None):
        if self.rank_allocator != "genetic_budgeted_calibrated":
            return self.final_trajectory_metrics
        trajectory_rank_pattern = {
            name: int(round(float(module.ranknum.item())))
            for name, module in model.named_modules()
            if isinstance(module, SVDLinear)
        }
        total_active_rank = int(sum(trajectory_rank_pattern.values()))
        self.final_trajectory_metrics = {
            "total_active_rank": total_active_rank,
            "active_model_parameter_count": get_active_model_parameter_count(model),
            "runtime_trainable_parameter_count": get_runtime_trainable_parameter_count(model),
            "full_model_parameter_count": get_full_model_parameter_count(model),
            "recorded_global_step": (
                int(global_step) if global_step is not None else getattr(self, "global_step", None)
            ),
            "allocation_stopped": self.allocation_stopped,
            "allocation_stopped_step": self.allocation_stopped_step,
            "consolidation_started_step": self.consolidation_started_step,
            "consolidation_remaining_steps": self.consolidation_remaining_steps,
            "rank_pattern": trajectory_rank_pattern,
        }
        if self.budget_metadata is not None:
            self.budget_metadata["final_trajectory"] = dict(self.final_trajectory_metrics)
        return self.final_trajectory_metrics

    def get_rank_pattern(self):
        # Return rank pattern 
        return self.rank_pattern

    def get_lora_param_name(self):
        # Prepare the budget scheduler 
        self.name_set = set() 
        self.initial_total_rank = 0 
        self.shape_dict = {}
        for n, layer in self.model.named_modules():
            if isinstance(layer, SVDLinear):
                self.name_set.add(n)
                self.initial_total_rank += layer.lora_A[0].size(0) 
                self.shape_dict[n+'.lora_A'] = layer.lora_A[0].shape
                self.shape_dict[n+'.lora_B'] = layer.lora_B[0].shape
                
        self.name_set = list(sorted(self.name_set))
        if self.target_rank is None:
            self.target_rank = self.ave_target_rank * len(self.name_set) 


    def update_ipt(self, model):
        lora_layers = []
        for n, layer in model.named_modules():
            if isinstance(layer, SVDLinear):
                lora_layers.append((n, layer))
                if n not in self.ipt:
                    self.ipt[n] = 0
                    self.exp_avg_ipt[n] = 0
                    self.exp_avg_unc[n] = 0
                if self.rank_allocator == "genetic_budgeted_calibrated":
                    layer_score = torch.as_tensor(
                        layer.score,
                        dtype=layer.weight.dtype,
                        device=layer.weight.device,
                    )
                    for state in (self.exp_avg_ipt, self.exp_avg_unc):
                        state[n] = torch.as_tensor(
                            state[n],
                            dtype=layer_score.dtype,
                            device=layer_score.device,
                        )
                else:
                    layer_score = layer.score
                
                # self.tb_writter.add_scalar("GradMatrix_Rank/%s"%(n[:-7],), layer.gradMatrix_rank, global_step)
                try:
                    self.ipt[n] = layer_score
                
                    # Update sensitivity 
                    self.exp_avg_ipt[n] = self.beta1 * self.exp_avg_ipt[n] + \
                                        (1-self.beta1)*self.ipt[n]
                    # Update uncertainty 
                    self.exp_avg_unc[n] = self.beta2 * self.exp_avg_unc[n] + \
                                        (1-self.beta2)*(self.ipt[n]-self.exp_avg_ipt[n]).abs()
                except:
                    ipdb.set_trace()
                    print(layer)

        if self.rank_allocator in (
            "genetic",
            "genetic_budgeted",
            "genetic_budgeted_calibrated",
        ) and len(lora_layers) == len(self.name_set):
            snapshot = {
                n: self._scalar_value(self.calculate_score(n, layer, metric="ipt"))
                for n, layer in lora_layers
            }
            for n in self.name_set:
                history = self.importance_history[n]
                history.append(snapshot[n])
                if len(history) > self.history_window:
                    del history[:-self.history_window]

    def calculate_score(self, n, layer, metric="ipt"):
        if metric == "ipt":
            # Combine the senstivity and uncertainty 
            ipt_score = self.exp_avg_ipt[n] * self.exp_avg_unc[n]
        elif metric == "mag":
            ipt_score = 0.
            for n,p in layer.named_parameters():
                ipt_score += p.abs().detach().clone() 
        else:
            raise ValueError("Unexcptected Metric: %s"%metric)
        return ipt_score 

    @staticmethod
    def _scalar_value(value):
        if hasattr(value, "item"):
            value = value.item()
        return float(value)

    @staticmethod
    def _format_diagnostic_float(value):
        return "none" if value is None else "%.6f" % value

    @staticmethod
    def _rank_one_parameter_cost(layer):
        # add_reserve_param creates A(1, in), E(1, 1), and B(out, 1).
        return (
            layer.lora_A[-1].numel()
            + layer.lora_E[-1].numel()
            + layer.lora_B[-1].numel()
        )

    def _increase_calibrated_rank(
        self,
        model,
        optimizer,
        is_dict,
        module_layers,
        maximum_event_size,
        increase_threshold,
        lr_scheduler=None,
        calibration_batch_pairs=None,
        calibration_loss_fn=None,
        max_grad_norm=None,
        amp_loss_scale=1.0,
    ):
        from transformers.budgeted_evo_allocator import expected_optimizer_gain
        from transformers.calibrated_budgeted_evo_allocator import (
            build_calibration_shortlist,
            generate_calibrated_candidates,
            select_calibrated_candidate,
        )
        from transformers.calibrated_rank_calibration import (
            get_candidate_rank_parameters,
            score_virtual_candidate,
        )

        if calibration_batch_pairs is None or calibration_loss_fn is None:
            raise ValueError(
                "Calibrated allocation requires fixed training-only calibration batches "
                "and a Trainer-owned loss callback."
            )
        if maximum_event_size <= 0:
            self._start_consolidation(model, self.global_step, "maximum_rank_reached")
            return increase_threshold

        ga_started = time.perf_counter()
        candidate_names = list(is_dict)
        module_costs = {
            name: get_module_rank_one_cost(module_layers[name])
            for name in candidate_names
        }
        gain_details = {
            name: expected_optimizer_gain(module_layers[name], optimizer)
            for name in candidate_names
        }
        training_gains = {
            name: details["raw_gain"] for name, details in gain_details.items()
        }
        current_active_cost = get_active_model_parameter_count(model)
        candidates, generation_diagnostics = generate_calibrated_candidates(
            scores=[
                (name, self._scalar_value(is_dict[name]))
                for name in candidate_names
            ],
            costs=module_costs,
            training_gains=training_gains,
            top_h=maximum_event_size,
            current_active_cost=current_active_cost,
            target_final_cost=self.target_final_trainable_params,
            rank_increment=self.incre_rank_num,
            min_event_rank=min(self.ga_min_event_rank, maximum_event_size),
            max_event_rank=maximum_event_size,
            max_greedy_replacements=self.ga_max_greedy_replacements,
            population_size=self.ga_population,
            generations=self.ga_generations,
            mutation_rate=self.ga_mutation_rate,
            crossover_rate=self.ga_crossover_rate,
            interaction_weight=self.ga_interaction_weight,
            redundancy_weight=self.ga_redundancy_weight,
            cost_weight=self.ga_cost_weight,
            diversity_weight=self.ga_diversity_weight,
            seed=self.training_seed + self.global_step,
            module_features={
                name: list(self.importance_history.get(name, ()))
                for name in candidate_names
            },
            # Reuse the existing threshold/tie-preserving Greedy selection
            # computed by increase_to_target_rank as the trust-region anchor.
            greedy_anchor_modules=[
                name for name in candidate_names if is_dict[name] >= increase_threshold
            ],
        )
        shortlist, shortlist_diagnostics = build_calibration_shortlist(
            candidates,
            calibration_topk=self.ga_calibration_topk,
            include_zero=True,
        )
        ga_runtime = time.perf_counter() - ga_started

        calibrated_candidates = []
        total_calibration_runtime = 0.0
        for candidate in shortlist:
            calibration = score_virtual_candidate(
                model=model,
                optimizer=optimizer,
                scheduler=lr_scheduler,
                candidate_module_names=candidate["modules"],
                module_map=module_layers,
                rank_increment=self.incre_rank_num,
                calibration_batch_pairs=calibration_batch_pairs,
                loss_fn=calibration_loss_fn,
                beta=self.ga_calibration_lcb_beta,
                max_grad_norm=max_grad_norm,
                loss_scale=amp_loss_scale,
                virtual_update_scale=(
                    1.0 / float(self.ga_new_rank_lr_warmup_steps)
                    if self.ga_new_rank_lr_warmup_steps > 0
                    else 1.0
                ),
            )
            total_calibration_runtime += calibration["calibration_runtime_seconds"]
            calibrated = dict(candidate)
            calibrated.update(calibration)
            calibrated_candidates.append(calibrated)
            logger.info(
                "Calibrated candidate global_step=%s modules=%s candidate_size=%s "
                "hamming_distance_from_greedy=%s optimizer_raw_gain=%.12g "
                "gain_per_parameter=%.12g structural_fitness=%.12g "
                "calibration_fold_gains=%s calibration_gain_mean=%s "
                "calibration_gain_std=%s calibration_gain_lcb=%s "
                "calibrated_gain_per_parameter=%s candidate_cost=%s "
                "projected_final_active_parameter_count=%s budget_feasible=%s "
                "candidate_families=%s calibration_valid=%s",
                self.global_step,
                calibrated["modules"],
                calibrated["chromosome_size"],
                calibrated["hamming_distance_from_greedy"],
                calibrated["optimizer_raw_gain"],
                calibrated["gain_per_parameter"],
                calibrated["structural_fitness"],
                calibrated["fold_gains"],
                calibrated["calibration_gain_mean"],
                calibrated["calibration_gain_std"],
                calibrated["calibration_gain_lcb"],
                calibrated["calibration_gain_per_parameter"],
                calibrated["actual_cost"],
                calibrated["projected_final_active_parameter_count"],
                calibrated["budget_feasible"],
                calibrated["candidate_families"],
                calibrated["calibration_valid"],
            )

        selected, selection_diagnostics = select_calibrated_candidate(
            calibrated_candidates,
            lcb_beta=self.ga_calibration_lcb_beta,
            quality_absolute_tolerance=self.ga_quality_absolute_tolerance,
            quality_relative_tolerance=self.ga_quality_relative_tolerance,
            greedy_quality_floor_ratio=self.ga_greedy_quality_floor_ratio,
            greedy_quality_floor_absolute=self.ga_greedy_quality_floor_absolute,
            min_calibrated_marginal_gain=self.ga_min_calibrated_marginal_gain,
            target_final_cost=self.target_final_trainable_params,
            require_global_quality_improvement=True,
        )
        for candidate in selection_diagnostics["candidates"]:
            logger.info(
                "Calibrated candidate decision global_step=%s modules=%s "
                "budget_feasible=%s greedy_quality_floor_satisfied=%s "
                "global_quality_requirement_satisfied=%s quality_band_satisfied=%s "
                "calibration_signal_valid=%s",
                self.global_step,
                candidate.get("modules"),
                candidate.get("budget_feasible"),
                candidate.get("greedy_quality_floor_satisfied"),
                candidate.get("global_quality_requirement_satisfied"),
                candidate.get("quality_band_satisfied"),
                candidate.get("calibration_signal_valid"),
            )
        selected_modules = list(selected["modules"])
        selected_event_rank = len(selected_modules)
        self.last_selected_event_rank = selected_event_rank

        if selected_event_rank == 0:
            self.zero_rank_event_counter += 1
            if self.zero_rank_event_counter >= self.ga_allocation_stop_patience:
                self._start_consolidation(
                    model, self.global_step, "zero_rank_patience_exhausted"
                )
        else:
            self.zero_rank_event_counter = 0
            newly_activated_parameter_ids = []
            for name in selected_modules:
                module = module_layers[name]
                spec = get_candidate_rank_parameters(module, self.incre_rank_num)
                with torch.no_grad():
                    module.ranknum.fill_(float(spec["new_active_rank"]))
                for parameter in spec["parameters"]:
                    parameter.requires_grad_(True)
                    newly_activated_parameter_ids.append(id(parameter))

            self.total_rank = int(
                sum(
                    round(float(module.ranknum.item()))
                    for module in model.modules()
                    if isinstance(module, SVDLinear)
                )
            )
            self.register_new_rank_warmup(
                model, newly_activated_parameter_ids, self.global_step
            )

            if self.total_rank < self.target_rank:
                new_reserve_parameters = []
                for name in selected_modules:
                    module = module_layers[name]
                    module.add_reserve_param(self.incre_rank_num, self.advance_learn)
                    new_reserve_parameters.extend(module.lora_A[-self.incre_rank_num :])
                    new_reserve_parameters.extend(module.lora_E[-self.incre_rank_num :])
                    new_reserve_parameters.extend(module.lora_B[-self.incre_rank_num :])
                self._add_calibrated_optimizer_parameters(
                    optimizer, new_reserve_parameters
                )

            current_active_cost = get_active_model_parameter_count(model)
            if current_active_cost > self.target_final_trainable_params:
                raise RuntimeError(
                    "Calibrated hard parameter budget violated after allocation: "
                    "current=%s target=%s"
                    % (current_active_cost, self.target_final_trainable_params)
                )
            if self.total_rank >= self.target_rank:
                self._start_consolidation(model, self.global_step, "maximum_rank_reached")

        for name, module in model.named_modules():
            if isinstance(module, SVDLinear):
                self.rank_pattern[name] = int(round(float(module.ranknum.item())))
        self._save_budget_metadata(model)

        selected_after_annotation = selection_diagnostics["selected_candidate"]
        logger.info(
            "Calibrated IncreLoRA allocation global_step=%s current_active_rank=%s "
            "current_active_parameters=%s remaining_hard_budget=%s "
            "greedy_anchor_modules=%s repaired_greedy_anchor_modules=%s "
            "selected_modules=%s selected_event_rank=%s selected_source=%s "
            "selected_hamming_distance=%s selected_optimizer_raw_gain=%.12g "
            "selected_structural_fitness=%.12g selected_calibration_gain_lcb=%s "
            "selected_calibration_gain_mean=%s selected_candidate_cost=%s "
            "selected_projected_final_active_parameters=%s "
            "greedy_quality_floor_satisfied=%s quality_band_satisfied=%s "
            "rank_growth_stop_counter=%s allocation_stopped=%s "
            "candidate_size_counts=%s shortlist=%s calibration_runtime_seconds=%.6f "
            "ga_runtime_seconds=%.6f local_search_enabled=%s",
            self.global_step,
            self.total_rank,
            get_active_model_parameter_count(model),
            self.target_final_trainable_params - get_active_model_parameter_count(model),
            generation_diagnostics["greedy_anchor_modules"],
            generation_diagnostics["repaired_greedy_anchor_modules"],
            selected_modules,
            selected_event_rank,
            selection_diagnostics["selected_source"],
            selected_after_annotation.get("hamming_distance_from_greedy"),
            selected_after_annotation.get("optimizer_raw_gain", 0.0),
            selected_after_annotation.get("structural_fitness", 0.0),
            selected_after_annotation.get("calibration_gain_lcb"),
            selected_after_annotation.get("calibration_gain_mean"),
            selected_after_annotation.get("actual_cost"),
            selected_after_annotation.get("projected_final_active_parameter_count"),
            selected_after_annotation.get("greedy_quality_floor_satisfied"),
            selected_after_annotation.get("quality_band_satisfied"),
            self.zero_rank_event_counter,
            self.allocation_stopped,
            generation_diagnostics["candidate_size_counts"],
            shortlist_diagnostics["shortlist_modules"],
            total_calibration_runtime,
            ga_runtime,
            selection_diagnostics["local_search_enabled"],
        )
        return increase_threshold

    def increase_to_target_rank(self, model, optimizer): 
        is_dict = {}
        all_is = []
        module_layers = {}
        # Calculate the importance score for each sub matrix 
        for n, layer in model.named_modules():
            if isinstance(layer, SVDLinear):
                ipt_score = self.calculate_score(n, layer, metric="ipt")                
                is_dict[n] = ipt_score
                all_is.append(ipt_score)
                module_layers[n] = layer

        # Calculate the increasing threshold 
        if self.rank_allocator == "genetic_budgeted":
            remaining_rank = self.target_rank - self.total_rank
            if remaining_rank % self.incre_rank_num:
                raise ValueError(
                    "Remaining rank increments are not divisible by incre_rank_num under fixed scheduling."
                )
            k = min(self.top_h, remaining_rank // self.incre_rank_num)
        elif self.rank_allocator == "genetic_budgeted_calibrated":
            remaining_rank = self.target_rank - self.total_rank
            k = min(
                self.top_h,
                self.ga_max_event_rank,
                remaining_rank // self.incre_rank_num,
            )
        else:
            k = min(self.top_h, self.target_rank - self.total_rank)
        increase_threshold = torch.topk(torch.tensor(all_is), k)[0][-1].item() 
        # Preserve the original threshold-based selection, including its tie behavior.
        greedy_selected_modules = [
            n for n in is_dict if is_dict[n] >= increase_threshold
        ]
        selected_modules = greedy_selected_modules

        if self.rank_allocator == "genetic_budgeted_calibrated":
            return self._increase_calibrated_rank(
                model,
                optimizer,
                is_dict,
                module_layers,
                k,
                increase_threshold,
                lr_scheduler=getattr(self, "_current_lr_scheduler", None),
                calibration_batch_pairs=getattr(
                    self, "_current_calibration_batch_pairs", None
                ),
                calibration_loss_fn=getattr(
                    self, "_current_calibration_loss_fn", None
                ),
                max_grad_norm=getattr(self, "_current_max_grad_norm", None),
                amp_loss_scale=getattr(self, "_current_amp_loss_scale", 1.0),
            )

        if self.rank_allocator == "genetic":
            from transformers.evo_allocator import select_modules_genetic

            candidate_names = list(is_dict)
            module_costs = {
                n: self._rank_one_parameter_cost(module_layers[n]) * self.incre_rank_num
                for n in candidate_names
            }
            module_features = {
                n: list(self.importance_history.get(n, ()))
                for n in candidate_names
            }
            greedy_top_h_modules = sorted(
                candidate_names, key=lambda n: -self._scalar_value(is_dict[n])
            )[:k]
            selected_modules, diagnostics = select_modules_genetic(
                scores=[(n, self._scalar_value(is_dict[n])) for n in candidate_names],
                costs=module_costs,
                top_h=k,
                population_size=self.ga_population,
                generations=self.ga_generations,
                mutation_rate=self.ga_mutation_rate,
                crossover_rate=self.ga_crossover_rate,
                interaction_weight=self.ga_interaction_weight,
                redundancy_weight=self.ga_redundancy_weight,
                cost_weight=self.ga_cost_weight,
                diversity_weight=self.ga_diversity_weight,
                seed=self.training_seed + self.global_step,
                module_features=module_features,
                local_search=self.ga_local_search,
            )
            logger.info(
                "EvoIncreLoRA allocation step=%s allocator=genetic "
                "selected_modules=%s greedy_modules=%s ga_best_modules=%s "
                "greedy_fitness=%s ga_best_fitness=%s ga_beats_greedy=%s "
                "ga_best_fitness_minus_greedy=%s fitness=%.6f "
                "importance_reward=%.6f interaction_gain=%.6f "
                "weighted_interaction_gain=%.6f redundancy_score=%.6f "
                "weighted_redundancy_penalty=%.6f normalized_cost_penalty=%.6f "
                "selected_parameter_cost=%.0f population_diversity_initial=%.6f "
                "population_diversity_final=%.6f unique_population_count_initial=%s "
                "unique_population_count_final=%s evaluated_unique_chromosome_count=%s "
                "selected_set_equals_greedy=%s selected_non_greedy_count=%s "
                "interaction_feature_mode=%s local_search_enabled=%s "
                "local_search_improved=%s selected_source=%s",
                self.global_step,
                selected_modules,
                greedy_top_h_modules,
                diagnostics["ga_best_modules"],
                self._format_diagnostic_float(diagnostics["greedy_fitness"]),
                self._format_diagnostic_float(diagnostics["ga_best_fitness"]),
                diagnostics["ga_beats_greedy"],
                self._format_diagnostic_float(
                    diagnostics["ga_best_fitness_minus_greedy"]
                ),
                diagnostics["total_fitness"],
                diagnostics["importance_reward"],
                diagnostics["interaction_gain"],
                diagnostics["weighted_interaction_gain"],
                diagnostics["redundancy_score"],
                diagnostics["weighted_redundancy_penalty"],
                diagnostics["normalized_cost_penalty"],
                diagnostics["selected_parameter_cost"],
                diagnostics["population_diversity_initial"],
                diagnostics["population_diversity_final"],
                diagnostics["unique_population_count_initial"],
                diagnostics["unique_population_count_final"],
                diagnostics["evaluated_unique_chromosome_count"],
                diagnostics["selected_set_equals_greedy"],
                diagnostics["selected_non_greedy_count"],
                diagnostics["interaction_feature_mode"],
                diagnostics["local_search_enabled"],
                diagnostics["local_search_improved"],
                diagnostics["selected_source"],
            )
        elif self.rank_allocator == "genetic_budgeted":
            from transformers.budgeted_evo_allocator import (
                expected_optimizer_gain,
                select_modules_budgeted,
            )

            allocation_start = time.time()
            candidate_names = list(is_dict)
            module_costs = {
                name: get_module_rank_one_cost(module_layers[name])
                for name in candidate_names
            }
            gain_details = {
                name: expected_optimizer_gain(module_layers[name], optimizer)
                for name in candidate_names
            }
            training_gains = {
                name: details["raw_gain"] for name, details in gain_details.items()
            }
            current_active_cost = get_active_model_parameter_count(model)
            remaining_rank_increments = self.target_rank - self.total_rank
            future_rank_increments = remaining_rank_increments - k * self.incre_rank_num
            greedy_top_h_modules = sorted(
                candidate_names, key=lambda name: -self._scalar_value(is_dict[name])
            )[:k]
            selected_modules, diagnostics = select_modules_budgeted(
                scores=[
                    (name, self._scalar_value(is_dict[name]))
                    for name in candidate_names
                ],
                costs=module_costs,
                training_gains=training_gains,
                top_h=k,
                current_active_cost=current_active_cost,
                target_final_cost=self.target_final_trainable_params,
                future_rank_increments=future_rank_increments,
                rank_increment=self.incre_rank_num,
                population_size=self.ga_population,
                generations=self.ga_generations,
                mutation_rate=self.ga_mutation_rate,
                crossover_rate=self.ga_crossover_rate,
                interaction_weight=self.ga_interaction_weight,
                redundancy_weight=self.ga_redundancy_weight,
                diversity_weight=self.ga_diversity_weight,
                gain_tolerance=self.ga_gain_tolerance,
                seed=self.training_seed + self.global_step,
                module_features={
                    name: list(self.importance_history.get(name, ()))
                    for name in candidate_names
                },
            )
            selected = diagnostics["selected_candidate"]
            logger.info(
                "Budgeted IncreLoRA allocation global_step=%s current_total_rank=%s "
                "current_active_model_parameter_count=%s "
                "current_runtime_trainable_parameter_count=%s full_model_parameter_count=%s "
                "remaining_parameter_budget=%s remaining_rank_increments=%s "
                "greedy_modules=%s greedy_candidate_cost=%s greedy_training_aware_gain=%.12g "
                "repaired_greedy_modules=%s ga_structural_best_modules=%s shortlist=%s "
                "selected_modules=%s selected_source=%s selected_actual_cost=%s "
                "selected_training_aware_gain=%.12g selected_gain_per_parameter=%.12g "
                "selected_structural_fitness=%.12g selected_projected_minimum_final_cost=%s "
                "budget_constraint_satisfied=%s quality_guard_satisfied=%s "
                "selected_set_equals_greedy=%s selected_non_greedy_count=%s "
                "population_unique_count_initial=%s population_unique_count_final=%s "
                "population_diversity_initial=%.6f population_diversity_final=%.6f "
                "evaluated_unique_chromosome_count=%s local_search_enabled=%s "
                "repair_count=%s infeasible_candidate_count=%s optimizer_gain_details=%s "
                "allocation_runtime_seconds=%.6f",
                self.global_step,
                self.total_rank,
                current_active_cost,
                get_runtime_trainable_parameter_count(model),
                get_full_model_parameter_count(model),
                self.target_final_trainable_params - current_active_cost,
                remaining_rank_increments,
                greedy_top_h_modules,
                diagnostics["greedy_candidate"]["actual_cost"],
                diagnostics["greedy_candidate"]["raw_training_gain"],
                diagnostics["repaired_greedy_modules"],
                diagnostics["ga_structural_best_modules"],
                diagnostics["shortlist"],
                selected_modules,
                diagnostics["selected_source"],
                selected["actual_cost"],
                selected["raw_training_gain"],
                selected["gain_per_parameter"],
                selected["structural_fitness"],
                selected["projected_minimum_final_cost"],
                diagnostics["budget_constraint_satisfied"],
                diagnostics["quality_guard_satisfied"],
                diagnostics["selected_set_equals_greedy"],
                diagnostics["selected_non_greedy_count"],
                diagnostics["population_unique_count_initial"],
                diagnostics["population_unique_count_final"],
                diagnostics["population_diversity_initial"],
                diagnostics["population_diversity_final"],
                diagnostics["evaluated_unique_chromosome_count"],
                diagnostics["local_search_enabled"],
                diagnostics["repair_count"],
                diagnostics["infeasible_candidate_count"],
                gain_details,
                time.time() - allocation_start,
            )

        selected_module_set = set(selected_modules)
        with torch.no_grad():
            curr_sum_rank = 0
            sum_param = 0
            new_param_list = []
            add_r = self.incre_rank_num
            for n, layer in model.named_modules():
                if isinstance(layer, SVDLinear):
                    if n in selected_module_set:
                        # rank increase 1
                        layer.ranknum += add_r
                        self.total_rank += add_r
                        
                        # add lora_E
                        for param in layer.lora_E[ -add_r: ]:
                            param.requires_grad = True
                            new_param_list.append(param)
                        
                        if self.advance_learn:
                            layer.add_reserve_param(add_r, True)
                            new_param_list.extend(layer.lora_A[ -add_r: ])
                            new_param_list.extend(layer.lora_B[ -add_r: ])
                        else:
                            for param in layer.lora_A[ -add_r: ]:
                                param.requires_grad = True
                                new_param_list.append(param)
                            for param in layer.lora_B[ -add_r: ]:
                                param.requires_grad = True
                                new_param_list.append(param)
                            layer.add_reserve_param(add_r, False)
                            
                        print("The lora parameters rank of {} increased by {}".format(n, add_r))
                    
                    ranknum = layer.ranknum
                    if self.tb_writter is not None or self.rank_allocator == "genetic_budgeted":
                        self.rank_pattern[n] = int(ranknum.item())
                    if self.tb_writter is not None:
                        self.tb_writter.add_scalar("Ranknum/%s"%(n,), ranknum, self.global_step) 
                        curr_sum_rank += ranknum
                        sum_param += ranknum*self.shape_dict[n+".lora_A"][1]  
                        sum_param += ranknum*self.shape_dict[n+".lora_B"][0]  

            optimizer.add_param_group({'params': new_param_list, "weight_decay": self.weight_decay,})
            
            if self.total_rank == self.target_rank:
                for name, module in model.named_modules():
                    if isinstance(module, SVDLinear):
                        module.hook_handle.remove()
                        for param in module.lora_E[ -add_r: ]:
                            param.fill_(0.)

            if self.rank_allocator == "genetic_budgeted":
                current_active_cost = get_active_model_parameter_count(model)
                if current_active_cost > self.target_final_trainable_params:
                    raise RuntimeError(
                        "Hard parameter budget violated after allocation: current=%s target=%s"
                        % (current_active_cost, self.target_final_trainable_params)
                    )
                self.budget_metadata.update(
                    {
                        "current_accumulated_cost": current_active_cost,
                        "current_total_rank": self.total_rank,
                    }
                )
                if self.total_rank == self.target_rank:
                    self.final_trajectory_metrics = {
                        "total_active_rank": self.total_rank,
                        "active_model_parameter_count": current_active_cost,
                        "runtime_trainable_parameter_count": get_runtime_trainable_parameter_count(
                            model
                        ),
                        "full_model_parameter_count": get_full_model_parameter_count(model),
                    }
                    self.budget_metadata["final_trajectory"] = dict(
                        self.final_trajectory_metrics
                    )
                    logger.info(
                        "Budgeted IncreLoRA final trajectory total_active_rank=%s "
                        "active_model_parameter_count=%s runtime_trainable_parameter_count=%s "
                        "full_model_parameter_count=%s target_final_parameter_count=%s "
                        "budget_satisfied=%s",
                        self.final_trajectory_metrics["total_active_rank"],
                        self.final_trajectory_metrics["active_model_parameter_count"],
                        self.final_trajectory_metrics["runtime_trainable_parameter_count"],
                        self.final_trajectory_metrics["full_model_parameter_count"],
                        self.target_final_trainable_params,
                        current_active_cost <= self.target_final_trainable_params,
                    )
                self._save_budget_metadata(model)
                            
            if self.tb_writter is not None:
                self.tb_writter.add_scalar("Budget/total_rank", curr_sum_rank, self.global_step)
                self.tb_writter.add_scalar("Budget/increase_threshold", increase_threshold, self.global_step)
                self.tb_writter.add_scalar("Budget/sum_param", sum_param, self.global_step)

        return increase_threshold


    def update_and_increase(
        self,
        model,
        global_step,
        optimizer,
        lr_scheduler=None,
        calibration_batch_pairs=None,
        calibration_loss_fn=None,
        max_grad_norm=None,
        amp_loss_scale=1.0,
    ):
        self.global_step = global_step
        increase_threshold=None
        add_r = self.incre_rank_num    
        # 为模型添加初始的储备参数
        if global_step == 0:
            new_param_list = []
            for name, module in model.named_modules():
                    if isinstance(module, SVDLinear):
                        module.add_reserve_param(add_r, self.advance_learn)
                        new_param_list.extend(module.lora_A[ -add_r: ])
                        new_param_list.extend(module.lora_B[ -add_r: ])
                        if self.rank_allocator == "genetic_budgeted_calibrated":
                            # Register inactive E reserves up front so stateless
                            # AdamW calibration can use the exact optimizer group.
                            new_param_list.extend(module.lora_E[ -add_r: ])
            if self.rank_allocator == "genetic_budgeted_calibrated":
                self._add_calibrated_optimizer_parameters(optimizer, new_param_list)
                self._save_budget_metadata(model)
            elif self.advance_learn:
                optimizer.add_param_group({'params': new_param_list, "weight_decay": self.weight_decay,})

        if self.rank_allocator == "genetic_budgeted_calibrated":
            self._current_lr_scheduler = lr_scheduler
            self._current_calibration_batch_pairs = calibration_batch_pairs
            self._current_calibration_loss_fn = calibration_loss_fn
            self._current_max_grad_norm = max_grad_norm
            self._current_amp_loss_scale = amp_loss_scale
            if self.allocation_stopped:
                if global_step % self.incre_interval == 0:
                    logger.info(
                        "Calibrated IncreLoRA allocation global_step=%s "
                        "selected_source=allocation_stopped_consolidation "
                        "selected_event_rank=0 fixed_total_active_rank=%s "
                        "fixed_active_parameter_count=%s allocation_stopped=True",
                        global_step,
                        self.total_rank,
                        get_active_model_parameter_count(model),
                    )
                self._maybe_tb_writter_log(model)
                return 0, None
            consolidation_boundary = max(
                0, int(self.total_step) - self.ga_min_consolidation_steps
            )
            if global_step >= consolidation_boundary:
                self._start_consolidation(
                    model, global_step, "minimum_consolidation_window"
                )
                self._maybe_tb_writter_log(model)
                return 0, None
            remaining_rank = self.target_rank - self.total_rank
            if remaining_rank < self.ga_min_event_rank * self.incre_rank_num:
                self._start_consolidation(
                    model, global_step, "insufficient_remaining_event_rank"
                )
                self._maybe_tb_writter_log(model)
                return 0, None

        if self.total_rank < self.target_rank:
            self.update_ipt(model)
            if global_step > self.init_warmup and global_step % self.incre_interval == 0:
                increase_threshold = self.increase_to_target_rank(model, optimizer) 
        
        self._maybe_tb_writter_log(model)
        if self.rank_allocator == "genetic_budgeted_calibrated":
            return (
                0 if increase_threshold is None else getattr(self, "last_selected_event_rank", 0),
                increase_threshold,
            )
        return self.top_h, increase_threshold

    def finalize_budget(self, model):
        """Finalize and independently verify the new mode's hard parameter budget."""

        if self.rank_allocator not in (
            "genetic_budgeted",
            "genetic_budgeted_calibrated",
        ):
            return self.rank_pattern
        if (
            self.rank_allocator == "genetic_budgeted_calibrated"
            and self.final_trajectory_metrics is None
        ):
            self.record_final_trajectory(model, global_step=getattr(self, "global_step", None))
        selected_best_total_rank = sum(
            int(round(float(module.ranknum.item())))
            for module in model.modules()
            if isinstance(module, SVDLinear)
        )
        selected_best_metrics = {
            "total_active_rank": selected_best_total_rank,
            "active_model_parameter_count": get_active_model_parameter_count(model),
            "runtime_trainable_parameter_count": get_runtime_trainable_parameter_count(model),
            "full_model_parameter_count": get_full_model_parameter_count(model),
        }
        selected_update = {
            "current_accumulated_cost": selected_best_metrics[
                "active_model_parameter_count"
            ],
            "current_total_rank": selected_best_total_rank,
            "selected_best_checkpoint": dict(selected_best_metrics),
        }
        if self.rank_allocator == "genetic_budgeted_calibrated":
            # load_best_model_at_end replaces Trainer.model with the selected
            # checkpoint. Start from that checkpoint's own allocator metadata
            # so capacity, optimizer manifest, EMA and warmup state remain
            # internally consistent with the exported weights. Attach the
            # separately captured final trajectory only as reporting data.
            checkpoint_metadata = getattr(
                getattr(model, "config", None), BUDGETED_ALLOCATOR_METADATA, None
            )
            if (
                not isinstance(checkpoint_metadata, dict)
                or checkpoint_metadata.get("allocator_mode")
                != "genetic_budgeted_calibrated"
            ):
                raise RuntimeError(
                    "Selected calibrated checkpoint is missing compatible allocator metadata."
                )
            export_metadata = dict(checkpoint_metadata)
            export_metadata.update(selected_update)
            export_metadata["active_rank_pattern"] = {
                name: int(round(float(module.ranknum.item())))
                for name, module in model.named_modules()
                if isinstance(module, SVDLinear)
            }
            export_metadata["final_trajectory"] = dict(
                self.final_trajectory_metrics
            )
            export_metadata["export_metadata_role"] = (
                "selected_best_checkpoint_with_separate_final_trajectory"
            )
            self.budget_metadata = export_metadata
            setattr(model.config, BUDGETED_ALLOCATOR_METADATA, export_metadata)
        else:
            self.budget_metadata.update(selected_update)
            if self.final_trajectory_metrics is not None:
                self.budget_metadata["final_trajectory"] = dict(
                    self.final_trajectory_metrics
                )
            self._save_budget_metadata(model)
        pattern = build_trainable_rank_pattern(model, self.budget_metadata)
        pattern_active_count = get_rank_pattern_active_model_parameter_count(
            model, pattern
        )
        selected_active_count = selected_best_metrics["active_model_parameter_count"]
        if selected_active_count != pattern_active_count:
            raise RuntimeError(
                "Selected best-checkpoint active parameter accounting mismatch: "
                "model_active=%s rank_pattern_active=%s difference=%s"
                % (
                    selected_active_count,
                    pattern_active_count,
                    selected_active_count - pattern_active_count,
                )
            )
        if selected_active_count > self.target_final_trainable_params:
            raise RuntimeError(
                "Selected best-checkpoint hard parameter budget violated: active=%s target=%s"
                % (selected_active_count, self.target_final_trainable_params)
            )
        trajectory_active_count = None
        if self.final_trajectory_metrics is not None:
            trajectory_active_count = self.final_trajectory_metrics[
                "active_model_parameter_count"
            ]
            if trajectory_active_count > self.target_final_trainable_params:
                raise RuntimeError(
                    "Final trajectory hard parameter budget violated: active=%s target=%s"
                    % (trajectory_active_count, self.target_final_trainable_params)
                )
        if (
            self.reference_greedy_cost is not None
            and self.ga_budget_ratio < 1.0
            and selected_active_count >= self.reference_greedy_cost
        ):
            raise RuntimeError(
                "Strict Greedy parameter reduction was not achieved by selected checkpoint: "
                "active=%s reference=%s"
                % (selected_active_count, self.reference_greedy_cost)
            )
        if (
            self.reference_greedy_cost is not None
            and self.ga_budget_ratio < 1.0
            and trajectory_active_count is not None
            and trajectory_active_count >= self.reference_greedy_cost
        ):
            raise RuntimeError(
                "Strict Greedy parameter reduction was not achieved by final trajectory: "
                "active=%s reference=%s"
                % (trajectory_active_count, self.reference_greedy_cost)
            )
        selected_reduction = (
            self.reference_greedy_cost - selected_active_count
            if self.reference_greedy_cost is not None
            else None
        )
        selected_reduction_percent = (
            100.0 * selected_reduction / self.reference_greedy_cost
            if self.reference_greedy_cost
            else None
        )
        trajectory_reduction = (
            self.reference_greedy_cost - trajectory_active_count
            if self.reference_greedy_cost is not None
            and trajectory_active_count is not None
            else None
        )
        trajectory_reduction_percent = (
            100.0 * trajectory_reduction / self.reference_greedy_cost
            if self.reference_greedy_cost and trajectory_reduction is not None
            else None
        )
        logger.info(
            "Budgeted IncreLoRA final verification "
            "allocator_mode=%s "
            "final_trajectory_metrics=%s selected_best_checkpoint_metrics=%s "
            "selected_best_rank_pattern_active_model_parameter_count=%s "
            "reference_greedy_active_model_parameter_count=%s "
            "target_final_active_model_parameter_count=%s "
            "final_trajectory_absolute_reduction=%s final_trajectory_percentage_reduction=%s "
            "selected_best_absolute_reduction=%s selected_best_percentage_reduction=%s "
            "final_trajectory_budget_satisfied=%s selected_best_budget_satisfied=%s "
            "allocation_stopped_step=%s total_consolidation_steps=%s "
            "final_budget_satisfied=%s",
            self.rank_allocator,
            self.final_trajectory_metrics,
            selected_best_metrics,
            pattern_active_count,
            self.reference_greedy_cost,
            self.target_final_trainable_params,
            trajectory_reduction,
            trajectory_reduction_percent,
            selected_reduction,
            selected_reduction_percent,
            (
                trajectory_active_count <= self.target_final_trainable_params
                if trajectory_active_count is not None
                else None
            ),
            selected_active_count <= self.target_final_trainable_params,
            self.allocation_stopped_step,
            (
                max(0, int(self.total_step) - int(self.allocation_stopped_step))
                if self.allocation_stopped_step is not None and self.total_step is not None
                else None
            ),
            (
                selected_active_count <= self.target_final_trainable_params
                and (
                    trajectory_active_count <= self.target_final_trainable_params
                    if trajectory_active_count is not None
                    else True
                )
            ),
        )
        self.rank_pattern = pattern
        return pattern

    def _maybe_tb_writter_log(self, model):
        def compute_and_log(mat_cov, name):
            I = torch.eye(*mat_cov.size(), out=torch.empty_like(mat_cov))
            I.requires_grad = False
            orth_regu = torch.norm(mat_cov-I, p="fro")
            regu_loss.append(orth_regu.item())
            self.tb_writter.add_scalar(
                "Orth_regu_loss/%s"%name, orth_regu.item(), self.global_step
            )
            
        if self.tb_writter is not None and self.global_step%self.log_interval==0:
            with torch.no_grad():
                regu_loss = []
                for n, layer in model.named_modules():
                    if isinstance(layer, SVDLinear):
                        wA = torch.cat([a for a in layer.lora_A], 0) 
                        wB = torch.cat([b for b in layer.lora_B], 1)
                        mat_cov_A = wA @ wA.T
                        mat_cov_B = wB.T @ wB 
                        compute_and_log(mat_cov_A, n+'.lora_A')
                        compute_and_log(mat_cov_B, n+'.lora_B')

                self.tb_writter.add_scalar(
                    "train/orth_regu_loss", sum(regu_loss)/len(regu_loss), self.global_step
                )


def compute_orth_regu(model, regu_weight=0.1):
    # The function to compute orthongonal regularization for SVDLinear in `model`. 
    regu_loss, num_param = 0., 0
    for n,p in model.named_parameters():
        if "lora_A" in n or "lora_B" in n:
            para_cov = p @ p.T if "lora_A" in n else p.T @ p 
            I = torch.eye(*para_cov.size(), out=torch.empty_like(para_cov))
            I.requires_grad = False
            regu_loss += torch.norm(para_cov-I, p="fro")
            num_param += 1
    return regu_weight*regu_loss/num_param
