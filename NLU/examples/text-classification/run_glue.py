#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.

import logging
import ipdb
import math
import os
import random
import sys
import json 
import torch
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from datasets import load_dataset, load_metric

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.modeling_utils import (
    DYNAMIC_LORA_RANK_PATTERN,
    is_dynamic_lora_parameter_name,
)
from transformers.utils import check_min_version
from loralib import RankAllocator 

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.4.0")

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """
    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_val_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of validation examples to this "
            "value if set."
        },
    )
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of test examples to this "
            "value if set."
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})

    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys.keys():
                raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))
        elif self.train_file is None or self.validation_file is None:
            raise ValueError("Need either a GLUE task or a training/validation file.")
        else:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                validation_extension == train_extension
            ), "`validation_file` should have the same extension (csv or json) as `train_file`."


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    apply_lora: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to apply LoRA or not."},
    )
    lora_type: Optional[str] = field(
        default="frd",
        metadata={"help": "The lora type: frd or svd."},
    )
    lora_module: Optional[str] = field(
        default="query,value",
        metadata={"help": "The modules applying lora: query,key,value,intermediate,layer.output,attention.output"},
    )
    lora_alpha: Optional[int] = field(
        default=None,
        metadata={"help": "LoRA alpha"},
    )
    lora_dropout: Optional[float] = field(
        default=0.,
        metadata={"help": "LoRA dropout"},
    )
    lora_r: Optional[int] = field(
        default=None,
        metadata={"help": "LoRA r"},
    )
    lora_path: Optional[str] = field(
        default=None,
        metadata={"help": "The file path of LoRA parameters."},
    )
    apply_adapter: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to apply adapter or not."},
    )
    adapter_path: Optional[str] = field(
        default=None,
        metadata={"help": "The file path of adapter parameters."},
    )
    adapter_type: Optional[str] = field(
        default='houlsby',
        metadata={"help": "houlsby or pfeiffer"},
    )
    adapter_size: Optional[int] = field(
        default=64,
        metadata={"help": "8, 16, 32, 64"},
    )
    apply_bitfit: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to apply bitfit or not."},
    )
    reg_loss_wgt: Optional[float] = field(
        default=0.0,
        metadata={"help": "Regularization Loss Weight"},
    )
    reg_orth_coef: Optional[float] = field(
        default=0.0,
        metadata={"help": "Orthogonal regularization coefficient"},
    )
    masking_prob: Optional[float] = field(
        default=0.0,
        metadata={"help": "Token Masking Probability"},
    )
    apply_increlora: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to apply rank selector or not."},
    )
    target_rank: Optional[int] = field(
        default=16,
        metadata={"help": "Average target rank."},
    )
    target_total_rank: Optional[int] = field(
        default=None,
        metadata={"help": "Specifying target number of total singular values"},
    )
    init_warmup: Optional[int] = field(
        default=4500,
        metadata={"help": "Total steps of inital warmup"},
    )
    incre_interval: Optional[int] = field(
        default=10,
        metadata={"help": "Increasing interval"},
    )
    beta1: Optional[float] = field(
        default=0.85,
        metadata={"help": "The coefficient of EMA"},
    )
    beta2: Optional[float] = field(
        default=0.85,
        metadata={"help": "The coefficient of EMA"},
    )
    tb_writter_loginterval: Optional[int] = field(
        default=500,
        metadata={"help": "The logging interval for tb_writter."},
    )
    
@dataclass
class TrainingArguments(TrainingArguments):
    experiment_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the experiment"},
    )
    top_h: Optional[int] = field(
        default=10,
        metadata={"help": "The number of selected modules per allocation."},
    )
    incre_rank_num: Optional[int] = field(
        default=None,
        metadata={"help": "Incre ranks for single matrix"},
    )
    advance_learn: Optional[bool] = field(
        default=True,
        metadata={"help": "Advance learning"},
    )
    multi_lr: Optional[bool] = field(
        default=True,
        metadata={"help": "New lr scheduler for new param group"},
    )
    rank_allocator: str = field(
        default="greedy",
        metadata={
            "help": (
                "Rank allocator: greedy, genetic, genetic_budgeted, or "
                "genetic_budgeted_calibrated."
            )
        },
    )
    ga_population: int = field(
        default=12,
        metadata={"help": "Population size for the genetic rank allocator."},
    )
    ga_generations: int = field(
        default=4,
        metadata={"help": "Number of generations for the genetic rank allocator."},
    )
    ga_mutation_rate: float = field(
        default=0.10,
        metadata={"help": "Replacement-mutation probability for the genetic allocator."},
    )
    ga_crossover_rate: float = field(
        default=0.80,
        metadata={"help": "Set-aware crossover probability for the genetic allocator."},
    )
    ga_interaction_weight: float = field(
        default=0.20,
        metadata={"help": "Weight of temporal module-complementarity gain."},
    )
    ga_redundancy_weight: float = field(
        default=0.20,
        metadata={"help": "Weight of the pairwise redundancy penalty."},
    )
    ga_cost_weight: float = field(
        default=0.30,
        metadata={"help": "Weight of the added-parameter cost penalty."},
    )
    ga_diversity_weight: float = field(
        default=0.10,
        metadata={"help": "Weight of population novelty during evolution."},
    )
    ga_local_search: bool = field(
        default=False,
        metadata={"help": "Enable one-swap local refinement for ablation only."},
    )
    ga_budget_reference_pattern: Optional[str] = field(
        default=None,
        metadata={"help": "Matched Greedy rank_pattern.json used only to derive a hard budget."},
    )
    ga_max_final_trainable_params: Optional[int] = field(
        default=None,
        metadata={"help": "Explicit hard maximum for final total trainable parameters."},
    )
    ga_budget_ratio: float = field(
        default=0.98,
        metadata={"help": "Fraction of the matched Greedy parameter cost allowed in budgeted modes."},
    )
    ga_gain_tolerance: float = field(
        default=0.05,
        metadata={"help": "Normalized training-gain tolerance for the budgeted quality guard."},
    )
    ga_allow_variable_event_rank: bool = field(
        default=False,
        metadata={"help": "Allow calibrated allocation events to select k=0 through top_h modules."},
    )
    ga_max_greedy_replacements: int = field(
        default=2,
        metadata={"help": "Maximum replacements in a calibrated Greedy-neighborhood chromosome."},
    )
    ga_calibration_batches: int = field(
        default=3,
        metadata={"help": "Number of deterministic paired training-only calibration folds."},
    )
    ga_calibration_batch_size: int = field(
        default=8,
        metadata={"help": "Training examples per calibration batch."},
    )
    ga_calibration_seed_offset: int = field(
        default=1000,
        metadata={"help": "Deterministic offset added to the training seed for calibration sampling."},
    )
    ga_calibration_topk: int = field(
        default=6,
        metadata={"help": "Maximum positive-rank candidates reranked by virtual calibration."},
    )
    ga_calibration_lcb_beta: float = field(
        default=0.5,
        metadata={"help": "Standard-deviation penalty in the calibrated lower-confidence score."},
    )
    ga_quality_absolute_tolerance: float = field(
        default=0.0,
        metadata={"help": "Absolute calibrated-quality equivalence tolerance."},
    )
    ga_quality_relative_tolerance: float = field(
        default=0.01,
        metadata={"help": "Relative calibrated-quality equivalence tolerance."},
    )
    ga_greedy_quality_floor_ratio: float = field(
        default=0.99,
        metadata={"help": "Minimum calibrated-quality ratio relative to the Greedy anchor."},
    )
    ga_greedy_quality_floor_absolute: float = field(
        default=0.0,
        metadata={"help": "Absolute calibrated-quality allowance below the Greedy anchor."},
    )
    ga_min_calibrated_marginal_gain: float = field(
        default=0.0,
        metadata={"help": "Minimum reliable calibrated LCB required for positive rank growth."},
    )
    ga_allocation_stop_patience: int = field(
        default=2,
        metadata={"help": "Consecutive zero-rank events before allocation stops permanently."},
    )
    ga_min_event_rank: int = field(
        default=1,
        metadata={"help": "Minimum positive calibrated event cardinality."},
    )
    ga_max_event_rank: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum calibrated event cardinality; defaults to top_h."},
    )
    ga_min_consolidation_steps: int = field(
        default=300,
        metadata={"help": "Minimum fixed-architecture training window at the end of training."},
    )
    ga_new_rank_lr_warmup_steps: int = field(
        default=25,
        metadata={"help": "Effective learning-rate warmup steps for newly activated LoRA components."},
    )
    greedy_reference_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "Matched Greedy best checkpoint used only for post-training parameter reporting."},
    )

    def __post_init__(self):
        super().__post_init__()
        if self.rank_allocator not in (
            "greedy",
            "genetic",
            "genetic_budgeted",
            "genetic_budgeted_calibrated",
        ):
            raise ValueError(
                "rank_allocator must be 'greedy', 'genetic', 'genetic_budgeted', or "
                "'genetic_budgeted_calibrated'."
            )
        if self.ga_population <= 0:
            raise ValueError("ga_population must be positive.")
        if self.ga_generations < 0:
            raise ValueError("ga_generations must be nonnegative.")
        if not math.isfinite(self.ga_mutation_rate) or not 0.0 <= self.ga_mutation_rate <= 1.0:
            raise ValueError("ga_mutation_rate must be between 0 and 1.")
        if not math.isfinite(self.ga_crossover_rate) or not 0.0 <= self.ga_crossover_rate <= 1.0:
            raise ValueError("ga_crossover_rate must be between 0 and 1.")
        if not math.isfinite(self.ga_interaction_weight) or self.ga_interaction_weight < 0.0:
            raise ValueError("ga_interaction_weight must be nonnegative.")
        if not math.isfinite(self.ga_redundancy_weight) or self.ga_redundancy_weight < 0.0:
            raise ValueError("ga_redundancy_weight must be nonnegative.")
        if not math.isfinite(self.ga_cost_weight) or self.ga_cost_weight < 0.0:
            raise ValueError("ga_cost_weight must be nonnegative.")
        if not math.isfinite(self.ga_diversity_weight) or self.ga_diversity_weight < 0.0:
            raise ValueError("ga_diversity_weight must be nonnegative.")
        if not math.isfinite(self.ga_budget_ratio) or not 0.0 < self.ga_budget_ratio <= 1.0:
            raise ValueError("ga_budget_ratio must be greater than 0 and at most 1.")
        if not math.isfinite(self.ga_gain_tolerance) or not 0.0 <= self.ga_gain_tolerance <= 1.0:
            raise ValueError("ga_gain_tolerance must be between 0 and 1.")
        if self.rank_allocator == "genetic_budgeted" and self.ga_local_search:
            raise ValueError("genetic_budgeted requires ga_local_search=false.")
        if self.rank_allocator == "genetic_budgeted" and self.ga_allow_variable_event_rank:
            raise ValueError("Variable event rank is not implemented; preserve the fixed schedule.")
        if self.rank_allocator == "genetic_budgeted_calibrated" and self.ga_local_search:
            raise ValueError("genetic_budgeted_calibrated requires ga_local_search=false.")
        if self.rank_allocator == "genetic_budgeted_calibrated" and not self.ga_allow_variable_event_rank:
            raise ValueError("genetic_budgeted_calibrated requires ga_allow_variable_event_rank=true.")
        if self.rank_allocator in ("greedy", "genetic") and self.ga_allow_variable_event_rank:
            raise ValueError("Variable event rank is disabled for greedy and genetic allocators.")
        if (
            self.rank_allocator in ("genetic_budgeted", "genetic_budgeted_calibrated")
            and self.ga_budget_reference_pattern is None
            and self.ga_max_final_trainable_params is None
        ):
            raise ValueError(
                "%s requires ga_budget_reference_pattern or "
                "ga_max_final_trainable_params."
                % self.rank_allocator
            )
        if self.ga_max_greedy_replacements < 0:
            raise ValueError("ga_max_greedy_replacements must be nonnegative.")
        if self.ga_calibration_batches <= 0 or self.ga_calibration_batch_size <= 0:
            raise ValueError("Calibration batch count and batch size must be positive.")
        if self.ga_calibration_topk < 6:
            raise ValueError("ga_calibration_topk must be at least 6.")
        for name, value in (
            ("ga_calibration_lcb_beta", self.ga_calibration_lcb_beta),
            ("ga_quality_absolute_tolerance", self.ga_quality_absolute_tolerance),
            ("ga_quality_relative_tolerance", self.ga_quality_relative_tolerance),
            ("ga_greedy_quality_floor_absolute", self.ga_greedy_quality_floor_absolute),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError("%s must be finite and nonnegative." % name)
        if (
            not math.isfinite(self.ga_greedy_quality_floor_ratio)
            or not 0.0 <= self.ga_greedy_quality_floor_ratio <= 1.0
        ):
            raise ValueError("ga_greedy_quality_floor_ratio must be between 0 and 1.")
        if not math.isfinite(self.ga_min_calibrated_marginal_gain):
            raise ValueError("ga_min_calibrated_marginal_gain must be finite.")
        if self.ga_allocation_stop_patience <= 0:
            raise ValueError("ga_allocation_stop_patience must be positive.")
        if self.ga_min_event_rank <= 0:
            raise ValueError("ga_min_event_rank must be positive.")
        if self.ga_max_event_rank is not None and self.ga_max_event_rank < self.ga_min_event_rank:
            raise ValueError("ga_max_event_rank must be at least ga_min_event_rank.")
        if self.ga_min_consolidation_steps < 0 or self.ga_new_rank_lr_warmup_steps < 0:
            raise ValueError("Consolidation and new-rank warmup steps must be nonnegative.")


def to_serializable(val):
    """Changes non-serializable values to serializable ones."""
    if isinstance(val, torch.Tensor) and val.dim() == 0:
        return val.item()
    elif isinstance(val, torch.Tensor):
        return val.tolist()
    elif hasattr(val, 'name'):  # This is for enums like IntervalStrategy
        return val.name
    else:
        return str(val)
    
def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # torch.use_deterministic_algorithms(training_args.use_deterministic_algorithms)
    # logger.info("use_deterministic_algorithms: " + str(torch.are_deterministic_algorithms_enabled()))

    # Setup output dir 
    os.makedirs(training_args.root_output_dir, exist_ok=True)
    training_args.output_dir = os.path.join(training_args.root_output_dir, "model")
    import datetime
    now = datetime.datetime.now()
    training_args.logging_dir = os.path.join(training_args.root_output_dir, now.strftime('%Y-%m-%d %H:%M:%S') +" "+ training_args.experiment_name +"_seed"+ str(training_args.seed))
    training_args.run_name = training_args.output_dir 

    if "debug" in training_args.output_dir:
        ipdb.set_trace()

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
    

    # Setup logging
    logging.basicConfig(
        filename= os.path.join(training_args.root_output_dir, 'log.txt'), filemode='a',
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN, 
        # handlers=[logging.StreamHandler(sys.stdout)],
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)
    logger.info(training_args.root_output_dir)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set tb_writter 
    if is_main_process(training_args.local_rank):
        tb_writter = SummaryWriter(log_dir=training_args.logging_dir)
    else:
        tb_writter = None

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.
    #
    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        datasets = load_dataset("glue", data_args.task_name)
    else:
        # Loading a dataset from your local files.
        # CSV/JSON training and evaluation files are needed.
        data_files = {"train": data_args.train_file, "validation": data_args.validation_file}

        # Get the test dataset: you can provide your own CSV/JSON test file (see below)
        # when you use `do_predict` without specifying a GLUE benchmark task.
        if training_args.do_predict:
            if data_args.test_file is not None:
                train_extension = data_args.train_file.split(".")[-1]
                test_extension = data_args.test_file.split(".")[-1]
                assert (
                    test_extension == train_extension
                ), "`test_file` should have the same extension (csv or json) as `train_file`."
                data_files["test"] = data_args.test_file
            else:
                raise ValueError("Need either a GLUE task or a test file for `do_predict`.")

        for key in data_files.keys():
            logger.info(f"load a local file for {key}: {data_files[key]}")

        if data_args.train_file.endswith(".csv"):
            # Loading a dataset from local csv files
            datasets = load_dataset("csv", data_files=data_files)
        else:
            # Loading a dataset from local json files
            datasets = load_dataset("json", data_files=data_files)
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Labels
    if data_args.task_name is not None:
        is_regression = data_args.task_name == "stsb"
        if not is_regression:
            label_list = datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    calibrated_resume_checkpoint = (
        last_checkpoint
        if training_args.rank_allocator == "genetic_budgeted_calibrated"
        else None
    )
    model_load_path = calibrated_resume_checkpoint or model_args.model_name_or_path
    config_load_path = (
        calibrated_resume_checkpoint
        or model_args.config_name
        or model_args.model_name_or_path
    )
    config = AutoConfig.from_pretrained(
        config_load_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        cls_dropout=training_args.cls_dropout,
        apply_lora=model_args.apply_lora,
        lora_type=model_args.lora_type, 
        lora_module=model_args.lora_module, 
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        lora_r=model_args.lora_r,
        apply_adapter=model_args.apply_adapter,
        adapter_type=model_args.adapter_type,
        adapter_size=model_args.adapter_size,
        reg_loss_wgt=model_args.reg_loss_wgt,
        masking_prob=model_args.masking_prob,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_load_path,
        from_tf=bool(".ckpt" in model_load_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    for name, module in model.named_modules():
        if hasattr(module, 'module_name'):
            module.module_name = name
    
    trainable_params = []
    if model_args.apply_lora:
        if model_args.lora_path is not None:
            lora_state_dict = torch.load(model_args.lora_path)
            logger.info(f"Apply LoRA state dict from {model_args.lora_path}.")
            logger.info(lora_state_dict.keys())
            model.load_state_dict(lora_state_dict, strict=False)
        trainable_params.append('lora')

    if model_args.apply_adapter:
        if model_args.adapter_path is not None:
            adapter_state_dict = torch.load(os.path.join(model_args.adapter_path, 'pytorch_adapter.bin'))
            head_state_dict = torch.load(os.path.join(model_args.adapter_path, 'pytorch_model_head.bin'))
            added_state_dict = {}
            for k, v in adapter_state_dict.items():
                new_k = k.replace(data_args.task_name + '.', '').replace('adapter_down.0.', 'adapter_A.').replace('adapter_up.', 'adapter_B.').replace('.adapters.', '.adapter.')
                added_state_dict[new_k] = v
            for k, v in head_state_dict.items():
                new_k = k.replace('heads.' + data_args.task_name + '.1', 'classifier.dense').replace('heads.' + data_args.task_name + '.4', 'classifier.out_proj')
                added_state_dict[new_k] = v
            logger.info(f"Apply adapter state dict from {model_args.adapter_path}.")
            logger.info(added_state_dict.keys())
            missing_keys, unexpected_keys = model.load_state_dict(added_state_dict, strict=False)
            for missing_key in missing_keys:
                assert 'adapter' not in missing_key, missing_key + ' is missed in the model'
            assert len(unexpected_keys) == 0, 'Unexpected keys ' + str(unexpected_keys)
        trainable_params.append('adapter')

    if model_args.apply_bitfit:
        trainable_params.append('bias')

    num_param = 0 
    restored_dynamic_rank_pattern = getattr(model.config, DYNAMIC_LORA_RANK_PATTERN, None)
    if len(trainable_params) > 0:
        for name, param in model.named_parameters():
            if name.startswith('deberta') or name.startswith('roberta'):
                if restored_dynamic_rank_pattern is not None and is_dynamic_lora_parameter_name(name):
                    if param.requires_grad:
                        sub_num_param = 1
                        for dim in param.shape:
                            sub_num_param *= dim
                        num_param += sub_num_param
                    continue
                param.requires_grad = False
                for trainable_param in trainable_params:
                    if trainable_param in name:
                        param.requires_grad = True
                        sub_num_param = 1 
                        for dim in param.shape:
                            sub_num_param *= dim  
                        num_param += sub_num_param 
                        break
            else:
                param.requires_grad = True
    else:
        for name, param in model.named_parameters():
            sub_num_param = 1 
            for dim in param.shape:
                sub_num_param *= dim  
            num_param += sub_num_param
    logger.info("Number of Trainable Parameters: %d"%(int(num_param))) 
    if tb_writter is not None: 
        tb_writter.add_scalar("train/num_train_param", num_param, 0)   


    # Preprocessing the datasets
    if data_args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and data_args.task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            label_to_id = {i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}
        else:
            logger.warn(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    elif data_args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warn(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)

        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in examples:
            result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
        return result

    datasets = datasets.map(preprocess_function, batched=True, load_from_cache_file=not data_args.overwrite_cache)
    if training_args.do_train:
        if "train" not in datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))

    if training_args.do_eval:
        if "validation" not in datasets and "validation_matched" not in datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = datasets["validation_matched" if data_args.task_name == "mnli" else "validation"]
        if data_args.max_val_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_val_samples))

    if training_args.do_predict or data_args.task_name is not None or data_args.test_file is not None:
        if "test" not in datasets and "test_matched" not in datasets:
            raise ValueError("--do_predict requires a test dataset")
        test_dataset = datasets["test_matched" if data_args.task_name == "mnli" else "test"]
        if data_args.max_test_samples is not None:
            test_dataset = test_dataset.select(range(data_args.max_test_samples))

    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Get the metric function
    if data_args.task_name is not None:
        metric = load_metric("glue", data_args.task_name)
    # TODO: When datasets metrics include regular accuracy, make an else here and remove special branch from
    # compute_metrics

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        if data_args.task_name is not None:
            result = metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result
        elif is_regression:
            return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        else:
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

    # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None


    # Initialize the rankallocator
    if model_args.lora_type == "svd" and model_args.apply_increlora:
        rankallocator = RankAllocator(
            model, 
            lora_r=model_args.lora_r,
            target_rank=model_args.target_rank,
            init_warmup=model_args.init_warmup, 
            incre_interval=model_args.incre_interval, 
            top_h=training_args.top_h,
            advance_learn=training_args.advance_learn,
            beta1=model_args.beta1, 
            beta2=model_args.beta2, 
            target_total_rank=model_args.target_total_rank,
            weight_decay=training_args.weight_decay,
            incre_rank_num=training_args.incre_rank_num,
            tb_writter=tb_writter, 
            tb_writter_loginterval=model_args.tb_writter_loginterval,
            rank_allocator=training_args.rank_allocator,
            ga_population=training_args.ga_population,
            ga_generations=training_args.ga_generations,
            ga_mutation_rate=training_args.ga_mutation_rate,
            ga_crossover_rate=training_args.ga_crossover_rate,
            ga_interaction_weight=training_args.ga_interaction_weight,
            ga_redundancy_weight=training_args.ga_redundancy_weight,
            ga_cost_weight=training_args.ga_cost_weight,
            ga_diversity_weight=training_args.ga_diversity_weight,
            ga_local_search=training_args.ga_local_search,
            ga_budget_reference_pattern=training_args.ga_budget_reference_pattern,
            ga_max_final_trainable_params=training_args.ga_max_final_trainable_params,
            ga_budget_ratio=training_args.ga_budget_ratio,
            ga_gain_tolerance=training_args.ga_gain_tolerance,
            ga_allow_variable_event_rank=training_args.ga_allow_variable_event_rank,
            ga_max_greedy_replacements=training_args.ga_max_greedy_replacements,
            ga_calibration_batches=training_args.ga_calibration_batches,
            ga_calibration_batch_size=training_args.ga_calibration_batch_size,
            ga_calibration_seed_offset=training_args.ga_calibration_seed_offset,
            ga_calibration_topk=training_args.ga_calibration_topk,
            ga_calibration_lcb_beta=training_args.ga_calibration_lcb_beta,
            ga_quality_absolute_tolerance=training_args.ga_quality_absolute_tolerance,
            ga_quality_relative_tolerance=training_args.ga_quality_relative_tolerance,
            ga_greedy_quality_floor_ratio=training_args.ga_greedy_quality_floor_ratio,
            ga_greedy_quality_floor_absolute=training_args.ga_greedy_quality_floor_absolute,
            ga_min_calibrated_marginal_gain=training_args.ga_min_calibrated_marginal_gain,
            ga_allocation_stop_patience=training_args.ga_allocation_stop_patience,
            ga_min_event_rank=training_args.ga_min_event_rank,
            ga_max_event_rank=training_args.ga_max_event_rank,
            ga_min_consolidation_steps=training_args.ga_min_consolidation_steps,
            ga_new_rank_lr_warmup_steps=training_args.ga_new_rank_lr_warmup_steps,
            greedy_reference_checkpoint=training_args.greedy_reference_checkpoint,
            training_seed=training_args.seed,
        )
    else:
        rankallocator = None

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
        rankallocator=rankallocator,
        model_args=model_args, 
        tb_writter=tb_writter, 
    )


    budget_finalized = False

    # Training
    if training_args.do_train:
        checkpoint = None
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        elif os.path.isdir(model_args.model_name_or_path):
            # Check the config from that potential checkpoint has the right number of labels before using it as a
            # checkpoint.
            if AutoConfig.from_pretrained(model_args.model_name_or_path).num_labels == num_labels:
                checkpoint = model_args.model_name_or_path

        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        if rankallocator is not None and training_args.rank_allocator in (
            "genetic_budgeted",
            "genetic_budgeted_calibrated",
        ):
            rankallocator.finalize_budget(trainer.model)
            budget_finalized = True

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        eval_datasets = [eval_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            eval_datasets.append(datasets["validation_mismatched"])

        for eval_dataset, task in zip(eval_datasets, tasks):
            metrics = trainer.evaluate(eval_dataset=eval_dataset)

            max_val_samples = data_args.max_val_samples if data_args.max_val_samples is not None else len(eval_dataset)
            metrics["eval_samples"] = min(max_val_samples, len(eval_dataset))
            for key in metrics:
                if tb_writter:
                    tb_writter.add_scalar("Eval_%s/%s"%(task, key), metrics[key], training_args.num_train_epochs)
                logger.info("{task} {key}: {value}:".format(task=task, key=key, value=metrics[key]))

            trainer.log_metrics("Eval_%s"%task, metrics)
            trainer.save_metrics("Eval_%s"%task, metrics)

    if training_args.do_predict:
        logger.info("*** Test ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        test_datasets = [test_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            test_datasets.append(datasets["test_mismatched"])

        for test_dataset, task in zip(test_datasets, tasks):
            # Removing the `label` columns because it contains -1 and Trainer won't like that.
            test_dataset.remove_columns_("label")
            predictions = trainer.predict(test_dataset=test_dataset).predictions
            predictions = np.squeeze(predictions) if is_regression else np.argmax(predictions, axis=1)

            output_test_file = os.path.join(training_args.output_dir, f"test_results_{task}.txt")
            if trainer.is_world_process_zero():
                with open(output_test_file, "w") as writer:
                    logger.info(f"***** Test results {task} *****")
                    writer.write("index\tprediction\n")
                    for index, item in enumerate(predictions):
                        if is_regression:
                            writer.write(f"{index}\t{item:3.3f}\n")
                        else:
                            item = label_list[item]
                            writer.write(f"{index}\t{item}\n")

    if tb_writter is not None:
        tb_writter.close() 

    if rankallocator is not None and is_main_process(training_args.local_rank):
        if training_args.rank_allocator in (
            "genetic_budgeted",
            "genetic_budgeted_calibrated",
        ) and not budget_finalized:
            rankallocator.finalize_budget(trainer.model)
        rank_pattern = rankallocator.get_rank_pattern()
        rank_pattern_path = os.path.join(
            training_args.root_output_dir, "rank_pattern.json"
        )
        with open(rank_pattern_path, "w") as f:
            json.dump(rank_pattern, f) 
        if (
            training_args.rank_allocator == "genetic_budgeted_calibrated"
            and training_args.greedy_reference_checkpoint is not None
        ):
            from transformers.rank_budget_reporting import build_rank_budget_report

            parameter_report = build_rank_budget_report(
                budgeted_trainer_state=os.path.join(
                    training_args.output_dir, "trainer_state.json"
                ),
                budgeted_rank_pattern=rank_pattern_path,
                greedy_reference_checkpoint=training_args.greedy_reference_checkpoint,
                greedy_reference_rank_pattern=training_args.ga_budget_reference_pattern,
            )
            report_path = os.path.join(
                training_args.root_output_dir, "parameter_comparison.json"
            )
            with open(report_path, "w", encoding="utf-8") as report_stream:
                json.dump(parameter_report, report_stream, indent=2, sort_keys=True)
            logger.info(
                "Calibrated matched parameter report=%s",
                parameter_report,
            )


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
