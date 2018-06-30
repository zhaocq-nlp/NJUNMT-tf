export CUDA_VISIBLE_DEVICES=


python -m bin.train \
  --problem_name "heheda" \
  --model_dir "test_models" \
  --train "
        shuffle_every_epoch: hehe
        update_cycle: 1" \
  --config_paths "
        ./njunmt/example_configs/toy_seq2seq.yml,
        ./default_configs/default_optimizer.yml,
        ./njunmt/example_configs/toy_training_options.yml"
