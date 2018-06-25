export CUDA_VISIBLE_DEVICES=


python -m bin.train \
  --problem_name "heheda" \
  --model_dir "models_target_r2l" \
  --train "
        labels_r2l: true
        update_cycle: 1" \
  --config_paths "
        ./njunmt/example_configs/toy_seq2seq.yml,
        ./default_configs/default_optimizer.yml,
        ./njunmt/example_configs/toy_training_options.yml"
