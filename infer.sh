#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=

DATA_DIR=./testdata

MODEL_DIR=./test_models
VOCAB_SOURCE=${DATA_DIR}/vocab.zh
VOCAB_TARGET=${DATA_DIR}/vocab.en

BATCH_SIZE=13
BEAM_SIZE=5
DELIMITER=" "
MAXIMUM_LABELS_LENGTH=30
CHAR_LEVEL=false

python -m bin.infer \
  --model_dir ${MODEL_DIR} \
  --infer "
    batch_size: ${BATCH_SIZE}
    beam_size: ${BEAM_SIZE}
    maximum_labels_length: ${MAXIMUM_LABELS_LENGTH}
    delimiter: ${DELIMITER}
    source_words_vocabulary: ${VOCAB_SOURCE}
    target_words_vocabulary: ${VOCAB_TARGET}
    source_bpecodes:
    target_bpecodes:
    char_level: ${CHAR_LEVEL}" \
  --infer_data "
    - features_file: ${DATA_DIR}/toy.zh
      output_file: ./toy.trans
      output_attention: true
    - features_file: ${DATA_DIR}/toy.zh
      labels_file: ${DATA_DIR}/toy.en
      output_file: ./heheda
      output_attention: false"
