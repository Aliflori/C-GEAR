#!/usr/bin/env bash

set -euo pipefail

checkpoint=${1:-}
greedy_rank_pattern=${2:-}
experiment_name=${3:-rte_budgeted_checkpoint_eval}
seed=${4:-41}
budget_ratio=${5:-0.98}

if [ -z "${checkpoint}" ] || [ ! -f "${checkpoint}/pytorch_model.bin" ]; then
    echo "Argument 1 must be a readable dynamic LoRA checkpoint directory." >&2
    exit 1
fi
if [ -z "${greedy_rank_pattern}" ] || [ ! -f "${greedy_rank_pattern}" ]; then
    echo "Argument 2 must be the readable matched Greedy rank_pattern.json." >&2
    exit 1
fi

target_rank=2
top_h=5
output_root="./output/glue/rte_allocator/genetic_budgeted_eval/${experiment_name}_seed${seed}_budget${budget_ratio}"
mkdir -p "${output_root}"

python \
examples/text-classification/run_glue.py \
--advance_learn True \
--multi_lr True \
--experiment_name "${experiment_name}_rank=${target_rank}" \
--model_name_or_path "${checkpoint}" \
--task_name rte \
--apply_increlora --apply_lora \
--lora_type svd --target_rank "${target_rank}" --lora_r 1 \
--reg_orth_coef 0.3 \
--init_warmup 100 --incre_interval 100 \
--top_h "${top_h}" \
--beta1 0.85 --beta2 0.85 \
--lora_module query,key,value,intermediate,layer.output,attention.output \
--lora_alpha 32 \
--do_eval --max_seq_length 192 \
--fp16 \
--cls_dropout 0.20 --weight_decay 0.01 \
--seed "${seed}" \
--rank_allocator genetic_budgeted \
--ga_population 12 \
--ga_generations 4 \
--ga_mutation_rate 0.10 \
--ga_crossover_rate 0.80 \
--ga_redundancy_weight 0.20 \
--ga_cost_weight 0.30 \
--ga_budget_reference_pattern "${greedy_rank_pattern}" \
--ga_budget_ratio "${budget_ratio}" \
--ga_gain_tolerance 0.05 \
--root_output_dir "${output_root}" \
--overwrite_output_dir \
2>&1 | tee "${output_root}/terminal.log"
