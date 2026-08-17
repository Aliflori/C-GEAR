# IncreLoRA: Incremental Parameter Allocation Method for Parameter-Efficient Fine-tuning

## C-GEAR final project

This fork adds **C-GEAR (Calibrated Genetic Efficiency-Aware Rank Allocation)** to the original IncreLoRA training stack. The implementation is exposed as `rank_allocator=genetic_budgeted_calibrated`: it combines Greedy-anchored evolutionary search, training-only candidate calibration, adaptive event size (including no growth), hard active-parameter budgeting, and allocation stopping. The original Greedy allocator and broader IncreLoRA task support remain available.

The official local comparison uses GLUE RTE, `microsoft/deberta-v3-base`, and matched seeds 41--46. At selected checkpoints, C-GEAR obtains **88.09% ± 1.14** accuracy versus **87.30% ± 0.84** for Greedy, with mean active parameters of **795,225** versus **828,005**. C-GEAR wins/ties/loses 4/0/2 seeds. The mean paired selected-parameter reduction is 3.55%. Terminal architectures are reported separately: C-GEAR reduces active parameters by 13.86% on average per matched pair and finishes at mean rank 89.7 versus 144.0.

The final three-page, two-column paper is at [`final_report/paper/cgear_final_report.pdf`](final_report/paper/cgear_final_report.pdf). Its source data, figures, scripts, and reproducibility documentation are under [`final_report/`](final_report/).

### Reproduce the RTE training commands

Activate the existing environment and enter `NLU`:

```bash
cd /home/ali/LoRa_Project/IncreLoRA/NLU
source /home/ali/LoRa_Project/.venv/bin/activate
```

For a new Greedy run (example seed 41), use a unique experiment name and tee the complete shell output:

```bash
set -o pipefail
mkdir -p ./output/glue/rte_allocator/greedy/repro_greedy_s41_seed41
CUDA_VISIBLE_DEVICES=0 bash scripts/rank=2/run_rte_allocator.sh \
  repro_greedy_s41 greedy 41 2>&1 \
  | tee ./output/glue/rte_allocator/greedy/repro_greedy_s41_seed41/terminal.log
```

Then run matched C-GEAR using that Greedy final rank pattern as the budget reference. The wrapper itself writes `terminal.log`:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/rank=2/run_rte_allocator.sh \
  repro_cgear_s41 genetic_budgeted_calibrated 41 \
  /home/ali/LoRa_Project/IncreLoRA/NLU/output/glue/rte_allocator/greedy/repro_greedy_s41_seed41/rank_pattern.json \
  0.94
```

Use new names for every run; the calibrated wrapper refuses to overwrite a populated result directory.

### Regenerate final results and figures (no training)

```bash
cd /home/ali/LoRa_Project/IncreLoRA
source /home/ali/LoRa_Project/.venv/bin/activate
python final_report/scripts/regenerate_six_seed_analysis.py
python final_report/scripts/generate_report_figures.py
cd final_report/paper
pdflatex -interaction=nonstopmode -halt-on-error cgear_final_report.tex
bibtex cgear_final_report
pdflatex -interaction=nonstopmode -halt-on-error cgear_final_report.tex
pdflatex -interaction=nonstopmode -halt-on-error cgear_final_report.tex
```

The regeneration script validates all 12 completed run artifacts before writing any paper statistics. Environment and exact allocator settings are preserved in [`final_report/data/experiment_configuration.json`](final_report/data/experiment_configuration.json); the complete workflow is documented in [`final_report/README.md`](final_report/README.md).

## Repository Overview

There are several directories in this repo:

* [loralib/](loralib) contains the source code of the updated package `loralib`, which include our implementation of IncreLoRA ([loralib/increlora.py](loralib/loralib/increlora.py)) and needs to be installed to run the examples;
* [NLU/](NLU) contains an example implementation of IncreLoRA in DeBERTaV3-base, which produces the results on the GLUE benchmark;
* [NLG_QA/](NLG_QA) contains an example implementation of IncreLoRA in BART-large and DeBERTaV3-base, which can be used to reproduce the results of summarization and question-answering tasks. 


## Quickstart of IncreLoRA

1. Install the updated `loralib`:

  ```bash 
  pip install -e loralib/ 
  ```


2. Then we apply SVD-based adaptation of IncreLoRA. Here is an example (For more examples, please see [modeling_debertav2.py](NLU/src/transformers/models/deberta_v2/modeling_deberta_v2.py) for how we adapte DeBERTa): 

  ```python
  # ===== Before =====
  # layer = nn.Linear(in_features, out_features)
  
  # ===== After ======
  import loralib 
  # Add a SVD-based adaptation matrices with rank r=12
  layer = loralib.SVDLinear(in_features, out_features, r=12)
  ```

   Also, before the training loop begins, mark only LoRA parameters as trainable.
  ```python
  model = BigModel()
  # This sets requires_grad to False for all parameters without the string "lora_" in their names
  loralib.mark_only_lora_as_trainable(model)
  ```

3. During the training loop, we apply RankAllocator of IncreLoRA to update importance scores of incremental matrices and allocate budget accordingly. 
  ```python
  from loralib import RankAllocator
  from loralib import compute_orth_regu 
  # Initialize the RankAllocator 
  rankallocator = RankAllocator(
      model, lora_r=1, target_rank=2,
      init_warmup=1000, incre_interval=1000, 
      top_h=2, beta1=0.85, beta2=0.85, 
  )
  ```
+ `lora_r`: The initial rank of each incremental matrix. 
+ `target_rank`: The average target rank of final incremental matrices, i.e. the average number of singular values per matrix. 
+ `init_warmup`: The steps of initial warmup for budget scheduler.
+ `incre_interval`: The time internval between two budget allocations.
+ `top_h`: The number of selected modules per allocation.
+ `beta1` and `beta2`: The coefficient of exponentional moving average when updating importance scores. 

  At each step of back-propagation, we apply an additional regularization to enforce the orthongonality of `SVDLinear` modules by `compute_orth_regu(model)`. Before each step of `optimizer.step()`, we call `RankAllocator` to update importance estimation and allocate the budget accordingly: 
  ```python
  # ===== Before =====
  # loss.backward() 
  # optimizer.step() 
  # global_step += 1 
  
  # ===== After ======
  (loss+compute_orth_regu(model, regu_weight=0.1)).backward
  rankallocator.update_and_increase(model, global_step)
  optimizer.step()
  global_step += 1
  ```


## GLUE benchmark

Check the folder `NLU` for more details about reproducing the GLUE results. 
An example of adapting DeBERTaV3-base on MNLI: 

```bash
python \
examples/text-classification/run_glue.py \
--model_name_or_path microsoft/deberta-v3-base \
--task_name mnli \
--apply_increlora --apply_lora --lora_type svd \
--target_rank 2  --lora_r 1  \
--reg_orth_coef 0.1 \
--init_warmup 1000 --incre_interval 1000 \
--top_h 2 \
--beta1 0.85 --beta2 0.85 \
--lora_module query,key,value,intermediate,layer.output,attention.output \
--lora_alpha 16 \
--lora_dropout 0.2 \
--do_train --do_eval \
--max_seq_length 256 \
--per_device_train_batch_size 32 --learning_rate 3.5e-4 --num_train_epochs 9 \
--warmup_steps 1000 \
--cls_dropout 0.15 --weight_decay 0 \
--evaluation_strategy steps --eval_steps 1000 \
--save_strategy steps --save_steps 10000 \
--logging_steps 500 \
--seed 41 \
--root_output_dir ./output/glue/mnli \
--overwrite_output_dir
```

Please see [`NLU/scripts`](NLU/scripts/) for more examples of GLUE. 


## Summarization and Question Answering Task

Check the folder [`NLG_QA`](NLG_QA/) for more details about reproducing the results of summarization and question-answering tasks.  
An example of adapting DeBERTaV3-base on SQuADv2: 

```bash
python \
examples/question-answering/run_qa.py \
--advance_learn True \
--multi_lr True \
--model_name_or_path microsoft/deberta-v3-base \
--dataset_name squad_v2 \
--apply_lora --apply_increlora \
--lora_type svd --target_rank 2 --lora_r 1 \
--reg_orth_coef 0.1 \
--init_warmup 1000 --incre_interval 1000 \
--top_h 1 \
--incre_rank_num 1 \
--beta1 0.85 --beta2 0.85 \
--lora_module query,key,value,intermediate,layer.output,attention.output \
--lora_alpha 16 \
--lora_dropout 0. \
--do_train --do_eval --version_2_with_negative \
--max_seq_length 384 --doc_stride 128 \
--per_device_train_batch_size 16 \
--learning_rate 1e-3 \
--num_train_epochs 14 \
--warmup_steps 1000 --per_device_eval_batch_size 128 \
--evaluation_strategy steps --eval_steps 1000 \
--save_strategy steps --save_steps 100000 \
--logging_steps 300 \
--tb_writter_loginterval 300 \
--report_to tensorboard \
--seed 9 \
--root_output_dir ./output/debertav3-base/squadv2 \
--overwrite_output_dir 
```
