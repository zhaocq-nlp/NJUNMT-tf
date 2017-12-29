#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=

python -m bin.train \
  --config_paths "
        ./njunmt/example_configs/toy_transformer.yml,
        ./default_configs/adam_noam_decay.yml,
        ./njunmt/example_configs/toy_training_options.yml" \
  --model_dir test_models2 \
  --train "
      train_steps: 200
  "
