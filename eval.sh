#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=

DATA_DIR=./testdata

MODEL_DIR=./test_models
VOCAB_SOURCE=${DATA_DIR}/vocab.zh
VOCAB_TARGET=${DATA_DIR}/vocab.en

BATCH_SIZE=13


python -m bin.eval \
  --model_dir ${MODEL_DIR} \
  --eval "
    batch_size: ${BATCH_SIZE}
    source_words_vocabulary: ${VOCAB_SOURCE}
    target_words_vocabulary: ${VOCAB_TARGET}
    source_bpecodes:
    target_bpecodes: " \
  --eval_data "
    - features_file: ${DATA_DIR}/toy.zh
      labels_file: ${DATA_DIR}/toy.en
      output_attention: false"
