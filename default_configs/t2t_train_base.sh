#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=

MODEL_DIR=./models
BATCH_TOKENS_SIZE=2048
MAXIMUM_FEATURES_LENGTH=50
MAXIMUM_LABELS_LENGTH=50
SAVE_CHECKPOINT_STEPS=1000

# to fill datasets
TRAIN_SOURCE=
TRAIN_TARGET=
EVAL_SOURCE=
EVAL_TARGETS=
SOURCE_VOCABULARY=
TARGET_VOCABULARY=
# optional
SOURCE_BPECODES=""
TARGET_BPECODES=""


python -m bin.train \
  --model_dir ${MODEL_DIR} \
  --config_paths "
        ./default_configs/transformer_base.yml" \
  --train "
        batch_size: 32
        batch_tokens_size: ${BATCH_TOKENS_SIZE}
        save_checkpoint_steps: ${SAVE_CHECKPOINT_STEPS}
        train_steps:
        eval_steps: 100
        maximum_features_length: ${MAXIMUM_FEATURES_LENGTH}
        maximum_labels_length: ${MAXIMUM_LABELS_LENGTH}
        shuffle_every_epoch:  " \
  --data "
        train_features_file: ${TRAIN_SOURCE}
        train_labels_file: ${TRAIN_TARGET}
        eval_features_file: ${EVAL_SOURCE}
        eval_labels_file: ${EVAL_TARGETS}
        source_words_vocabulary: ${SOURCE_VOCABULARY}
        target_words_vocabulary: ${TARGET_VOCABULARY}
        source_bpecodes: ${SOURCE_BPECODES}
        target_bpecodes: ${TARGET_BPECODES}"

