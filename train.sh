DATA_PATH=./testdata
# vocabulary
VOCAB_SOURCE=${DATA_PATH}/vocab.zh
VOCAB_TARGET=${DATA_PATH}/vocab.en
# VOCAB_TARGET=${DATA_PATH}/vocab.zh

# datesets
TRAIN_SOURCE=${DATA_PATH}/mt02.src
TRAIN_TARGET=${DATA_PATH}/mt02.ref0
# TRAIN_TARGET=${DATA_PATH}/mt02.src
DEV_SOURCE=${DATA_PATH}/mt02.src
DEV_TARGET=${DATA_PATH}/mt02.ref
# DEV_TARGET=${DATA_PATH}/mt02.src

# training options
BATCH_SIZE=13
TRAIN_STEPS=200
TRAIN_EPOCHS=100
SOURCE_MAX_SEQ_LEN=50
TARGET_MAX_SEQ_LEN=50
DISPLAY_STEPS=10
SAVE_CHECKPOINTS_STEPS=10

MODEL_DIR=./test_models

export CUDA_VISIBLE_DEVICES=0


python -m train \
  --config_paths "
        ./seq2seq/example_configs/nmt_local_debug.json,
        ./seq2seq/example_configs/debug_hooks.json" \
  --train_source $TRAIN_SOURCE \
  --train_target $TRAIN_TARGET \
  --dev_source $DEV_SOURCE \
  --dev_target $DEV_TARGET \
  --training_params "
      vocab_source: $VOCAB_SOURCE,
      vocab_target: $VOCAB_TARGET,
      batch_size: $BATCH_SIZE,
      source_max_seq_len: $SOURCE_MAX_SEQ_LEN,
      target_max_seq_len: $TARGET_MAX_SEQ_LEN" \
  --training_options "
      train_steps: $TRAIN_STEPS,
      train_epochs: $TRAIN_EPOCHS,
      display_steps: $DISPLAY_STEPS,
      save_checkpoints_every_n_steps: $SAVE_CHECKPOINTS_STEPS" \
  --output_dir $MODEL_DIR
