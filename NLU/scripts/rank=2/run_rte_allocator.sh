#!/usr/bin/env bash

set -euo pipefail

experiment_name=${1:-}
allocator=${2:-}
seed=${3:-41}

if [ -z "${experiment_name}" ] || [ -z "${allocator}" ]; then
    echo "Usage: $0 <experiment_name> <allocator: greedy|genetic> [seed]" >&2
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

target_rank=2
top_h=5
output_root="./output/glue/rte_allocator/${allocator}/${experiment_name}_seed${seed}"

python \
examples/text-classification/run_glue.py \
--advance_learn True \
--multi_lr True \
--experiment_name "${experiment_name}_rank=${target_rank}" \
--model_name_or_path microsoft/deberta-v3-base \
--task_name rte \
--apply_increlora --apply_lora \
--lora_type svd --target_rank "${target_rank}" --lora_r 1 \
--reg_orth_coef 0.3 \
--init_warmup 100 --incre_interval 100 \
--top_h "${top_h}" \
--beta1 0.85 --beta2 0.85 \
--lora_module query,key,value,intermediate,layer.output,attention.output \
--lora_alpha 32 \
--do_train --do_eval --max_seq_length 192 \
--per_device_train_batch_size 32 --learning_rate 1.2e-3 \
--num_train_epochs 25 --warmup_steps 100 \
--fp16 \
--cls_dropout 0.20 --weight_decay 0.01 \
--evaluation_strategy steps --eval_steps 100 \
--save_strategy steps --save_steps 100 \
--save_total_limit 1 \
--load_best_model_at_end True \
--metric_for_best_model accuracy \
--greater_is_better True \
--logging_steps 10 --report_to tensorboard \
--tb_writter_loginterval 50 \
--seed "${seed}" \
--rank_allocator "${allocator}" \
--ga_population 12 \
--ga_generations 4 \
--ga_mutation_rate 0.10 \
--ga_crossover_rate 0.80 \
--ga_redundancy_weight 0.20 \
--ga_cost_weight 0.30 \
--root_output_dir "${output_root}" \
--overwrite_output_dir
