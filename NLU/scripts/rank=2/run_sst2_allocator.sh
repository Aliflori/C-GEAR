#!/usr/bin/env bash

set -euo pipefail

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
    echo "Usage: $0 <experiment_name> <greedy|genetic> [seed]"
    exit 0
fi

experiment_name=${1:-}
allocator=${2:-}
seed=${3:-41}

if [ -z "${experiment_name}" ] || [ -z "${allocator}" ]; then
    echo "Usage: $0 <experiment_name> <greedy|genetic> [seed]" >&2
    exit 1
fi

case "${allocator}" in
    greedy|genetic)
        ;;
    *)
        echo "Unsupported allocator '${allocator}'; expected 'greedy' or 'genetic'." >&2
        exit 1
        ;;
esac

export CUDA_VISIBLE_DEVICES=0

target_rank=2
top_h=5
output_root="./output/glue/sst2_allocator/${allocator}/${experiment_name}_seed${seed}"
mkdir -p "${output_root}"

python \
examples/text-classification/run_glue.py \
--experiment_name "${experiment_name}" \
--model_name_or_path microsoft/deberta-v3-base \
--task_name sst2 \
--apply_increlora --apply_lora --lora_type svd \
--target_rank "${target_rank}" --lora_r 1 \
--reg_orth_coef 0.1 \
--init_warmup 1000 --incre_interval 1000 \
--top_h "${top_h}" \
--beta1 0.85 --beta2 0.85 \
--lora_module query,key,value,intermediate,layer.output,attention.output \
--lora_alpha 16 \
--lora_dropout 0.1 \
--do_train --do_eval \
--max_seq_length 128 \
--per_device_train_batch_size 32 \
--learning_rate 8e-4 \
--num_train_epochs 24 \
--warmup_steps 1000 --cls_dropout 0. --weight_decay 0.01 \
--evaluation_strategy steps --eval_steps 1000 \
--save_strategy steps --save_steps 10000 \
--save_total_limit 1 \
--load_best_model_at_end True \
--metric_for_best_model accuracy \
--greater_is_better True \
--logging_steps 500 \
--tb_writter_loginterval 500 \
--report_to tensorboard \
--seed "${seed}" \
--rank_allocator "${allocator}" \
--ga_population 12 \
--ga_generations 4 \
--ga_mutation_rate 0.10 \
--ga_crossover_rate 0.80 \
--ga_redundancy_weight 0.20 \
--ga_cost_weight 0.30 \
--root_output_dir "${output_root}" \
--overwrite_output_dir \
2>&1 | tee "${output_root}/terminal.log"
