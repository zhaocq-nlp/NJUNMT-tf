#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=

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
  --config_paths "
        ./default_configs/adam_loss_decay.yml,
        ./default_configs/default_metrics.yml,
        ./default_configs/default_training_options.yml,
        ./default_configs/seq2seq_cgru.yml" \
  --data "
        train_features_file: ${TRAIN_SOURCE}
        train_labels_file: ${TRAIN_TARGET}
        eval_features_file: ${EVAL_SOURCE}
        eval_labels_file: ${EVAL_TARGETS}
        source_words_vocabulary: ${SOURCE_VOCABULARY}
        target_words_vocabulary: ${TARGET_VOCABULARY}
        source_bpecodes: ${SOURCE_BPECODES}
        target_bpecodes: ${TARGET_BPECODES}"