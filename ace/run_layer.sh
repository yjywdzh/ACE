#!/bin/bash
export TOKENIZERS_PARALLELISM=false

python ./edit/evaluate_filterqa_layer.py \
  --model_path path/to/model \
  --model_name Qwen/Qwen3-8B \
  --alg_name ace \
  --hparams_fname qwen3_8b.json \
  --ds_name filteredQA \
  --skip_generation_tests \
  --num_edits 1 \
  --use_cache \