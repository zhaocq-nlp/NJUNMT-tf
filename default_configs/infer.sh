export CUDA_VISIBLE_DEVICES=


VOCAB_SOURCE=
VOCAB_TARGET=

MODEL_DIR=./models
BEAM_SIZE=4
BATCH_SIZE=32
LENGTH_PENALTY=-1.0
MAX_LABELS_LENGTH=150

python -m bin.infer \
  --model_dir ${MODEL_DIR} \
  --infer "
    char_level: false
    batch_size: ${BATCH_SIZE}
    beam_size: ${BEAM_SIZE}
    length_penalty: ${LENGTH_PENALTY}
    maximum_labels_length: ${MAX_LABELS_LENGTH}
    source_words_vocabulary: ${VOCAB_SOURCE}
    target_words_vocabulary: ${VOCAB_TARGET}
    source_bpecodes:
    target_bpecodes: " \
  --infer_data "
    - features_file:
      labels_file:
      output_file:
      output_attention: false
    - features_file:
      labels_file:
      output_file:
      output_attention: false"