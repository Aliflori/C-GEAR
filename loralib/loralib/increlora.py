#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import ipdb
import re
import numpy as np

from .layers import LoRALayer 
from typing import Optional, List 

logger = logging.getLogger(__name__)

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

        if rank_allocator not in ("greedy", "genetic"):
            raise ValueError("rank_allocator must be either 'greedy' or 'genetic'.")
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
        self.training_seed = training_seed

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
            if self.rank_allocator == "genetic"
            else None
        )

        self.tb_writter = tb_writter
        self.log_interval = tb_writter_loginterval 
        
        assert (self.beta1<1 and self.beta1>0)
        assert (self.beta2<1 and self.beta2>0)
        
            
    def set_total_step(self, total_step:int): 
        # Set total step number 
        self.total_step = total_step
        rank_per_round = self.top_h * self.incre_rank_num
        total_round = math.ceil((self.target_rank - self.initial_total_rank) / rank_per_round)
        total_incre_step = self.incre_interval * total_round
                            
        print("Total incremental step: total_incre_step: {}, of total steps: {:.0%}"
              .format(total_incre_step, total_incre_step / total_step))

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
                
                # self.tb_writter.add_scalar("GradMatrix_Rank/%s"%(n[:-7],), layer.gradMatrix_rank, global_step)
                try:
                    self.ipt[n] = layer.score
                
                    # Update sensitivity 
                    self.exp_avg_ipt[n] = self.beta1 * self.exp_avg_ipt[n] + \
                                        (1-self.beta1)*self.ipt[n]
                    # Update uncertainty 
                    self.exp_avg_unc[n] = self.beta2 * self.exp_avg_unc[n] + \
                                        (1-self.beta2)*(self.ipt[n]-self.exp_avg_ipt[n]).abs()
                except:
                    ipdb.set_trace()
                    print(layer)

        if self.rank_allocator == "genetic" and len(lora_layers) == len(self.name_set):
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
        k = min(self.top_h, self.target_rank - self.total_rank)
        increase_threshold = torch.topk(torch.tensor(all_is), k)[0][-1].item() 
        # Preserve the original threshold-based selection, including its tie behavior.
        greedy_selected_modules = [
            n for n in is_dict if is_dict[n] >= increase_threshold
        ]
        selected_modules = greedy_selected_modules

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
                    if self.tb_writter is not None:
                        self.tb_writter.add_scalar("Ranknum/%s"%(n,), ranknum, self.global_step) 
                        self.rank_pattern[n] = int(ranknum.item())
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
                            
            if self.tb_writter is not None:
                self.tb_writter.add_scalar("Budget/total_rank", curr_sum_rank, self.global_step)
                self.tb_writter.add_scalar("Budget/increase_threshold", increase_threshold, self.global_step)
                self.tb_writter.add_scalar("Budget/sum_param", sum_param, self.global_step)

        return increase_threshold


    def update_and_increase(self, model, global_step, optimizer):
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
            if self.advance_learn:
                optimizer.add_param_group({'params': new_param_list, "weight_decay": self.weight_decay,})
        
        if self.total_rank < self.target_rank:
            self.update_ipt(model)
            if global_step > self.init_warmup and global_step % self.incre_interval == 0:
                increase_threshold = self.increase_to_target_rank(model, optimizer) 
        
        self._maybe_tb_writter_log(model)
        return self.top_h, increase_threshold

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
