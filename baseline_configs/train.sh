DATA_PATH=./testdata
# vocabulary
VOCAB_SOURCE=${DATA_PATH}/vocab.zh
VOCAB_TARGET=${DATA_PATH}/vocab.en

# datesets
TRAIN_SOURCE=${DATA_PATH}/zh.shuf
TRAIN_TARGET=${DATA_PATH}/en.shuf
DEV_SOURCE=${DATA_PATH}/testsets/mt03.src
DEV_TARGET=${DATA_PATH}/testsets/mt03.ref

# training options
BATCH_SIZE=80
TRAIN_STEPS=20000000
TRAIN_EPOCHS=100
SOURCE_MAX_SEQ_LEN=50
TARGET_MAX_SEQ_LEN=50
DISPLAY_STEPS=100
SAVE_CHECKPOINTS_STEPS=1000

MODEL_DIR=./models

export CUDA_VISIBLE_DEVICES=0


python -m train \
  --config_paths "
        ./baseline_configs/xxx.json,
        ./baseline_configs/baseline_hooks.json" \
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
