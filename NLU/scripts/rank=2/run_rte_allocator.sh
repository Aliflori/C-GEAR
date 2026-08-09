#!/usr/bin/env bash

set -euo pipefail

experiment_name=${1:-}
allocator=${2:-}
seed=${3:-41}

if [ -z "${experiment_name}" ] || [ -z "${allocator}" ]; then
    echo "Usage: $0 <experiment_name> <allocator: greedy|genetic|genetic_budgeted|genetic_budgeted_calibrated> [seed] [greedy_rank_pattern] [budget_ratio]" >&2
    exit 1
fi

case "${allocator}" in
    greedy|genetic)
        ;;
    genetic_budgeted)
        ;;
    genetic_budgeted_calibrated)
        ;;
    *)
        echo "Unsupported allocator '${allocator}'; expected 'greedy', 'genetic', 'genetic_budgeted', or 'genetic_budgeted_calibrated'." >&2
        exit 1
        ;;
esac

target_rank=2
top_h=5
budget_args=()
overwrite_args=(--overwrite_output_dir)
tee_mode=truncate
if [ "${allocator}" = "genetic_budgeted" ] || [ "${allocator}" = "genetic_budgeted_calibrated" ]; then
    greedy_rank_pattern=${4:-}
    if [ "${allocator}" = "genetic_budgeted_calibrated" ]; then
        budget_ratio=${5:-0.94}
    else
        budget_ratio=${5:-0.98}
    fi
    if [ -z "${greedy_rank_pattern}" ] || [ ! -f "${greedy_rank_pattern}" ]; then
        echo "${allocator} requires a readable matched Greedy rank_pattern.json as argument 4." >&2
        exit 1
    fi
    if [[ ! "${budget_ratio}" =~ ^(0(\.[0-9]+)?|1(\.0+)?)$ ]] || [ "${budget_ratio}" = "0" ]; then
        echo "budget_ratio must be greater than 0 and at most 1." >&2
        exit 1
    fi
    output_root="./output/glue/rte_allocator/${allocator}/${experiment_name}_seed${seed}_budget${budget_ratio}"
    budget_args=(
        --ga_budget_reference_pattern "${greedy_rank_pattern}"
        --ga_budget_ratio "${budget_ratio}"
    )
    if [ "${allocator}" = "genetic_budgeted" ]; then
        budget_args+=(--ga_gain_tolerance 0.05)
    else
        overwrite_args=()
        budget_args+=(
            --ga_local_search false
            --ga_allow_variable_event_rank true
            --ga_max_greedy_replacements 2
            --ga_calibration_batches 3
            --ga_calibration_batch_size 8
            --ga_calibration_seed_offset 1000
            --ga_calibration_topk 6
            --ga_calibration_lcb_beta 0.5
            --ga_quality_absolute_tolerance 0.0
            --ga_quality_relative_tolerance 0.01
            --ga_greedy_quality_floor_ratio 0.99
            --ga_greedy_quality_floor_absolute 0.0
            --ga_min_calibrated_marginal_gain 0.0
            --ga_allocation_stop_patience 2
            --ga_min_event_rank 1
            --ga_max_event_rank "${top_h}"
            --ga_min_consolidation_steps 300
            --ga_new_rank_lr_warmup_steps 25
        )
    fi
else
    output_root="./output/glue/rte_allocator/${allocator}/${experiment_name}_seed${seed}"
fi

run_training() {
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
"${budget_args[@]}" \
--root_output_dir "${output_root}" \
"${overwrite_args[@]}"
}

if [ "${allocator}" = "genetic_budgeted" ] || [ "${allocator}" = "genetic_budgeted_calibrated" ]; then
    if [ "${allocator}" = "genetic_budgeted_calibrated" ] && [ -d "${output_root}" ] && [ -n "$(find "${output_root}" -mindepth 1 -print -quit)" ]; then
        if [ -f "${output_root}/model/trainer_state.json" ] && [ -f "${output_root}/model/pytorch_model.bin" ]; then
            echo "Refusing to resume or overwrite a completed calibrated run: ${output_root}" >&2
            echo "Use a new experiment name for a new run." >&2
            exit 1
        elif [ "${RESUME_CALIBRATED:-0}" = "1" ] && [ -n "$(find "${output_root}/model" -maxdepth 1 -type d -name 'checkpoint-*' -print -quit 2>/dev/null)" ]; then
            tee_mode=append
        else
            echo "Refusing to overwrite existing calibrated output directory: ${output_root}" >&2
            echo "Set RESUME_CALIBRATED=1 only to resume a checkpoint in this same directory." >&2
            exit 1
        fi
    fi
    mkdir -p "${output_root}"
    if [ "${tee_mode}" = "append" ]; then
        run_training 2>&1 | tee -a "${output_root}/terminal.log"
    else
        run_training 2>&1 | tee "${output_root}/terminal.log"
    fi
else
    run_training
fi
