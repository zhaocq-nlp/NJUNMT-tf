export DATA_PATH=./testdata

export VOCAB_SOURCE=${DATA_PATH}/vocab.zh
export VOCAB_TARGET=${DATA_PATH}/vocab.en
export INFER_SOURCE=${DATA_PATH}/mt02.src

export MODEL_DIR=./test_models
export BATCH_SIZE=32
export BEAM_SIZE=10
export MAX_SEQ_LEN=200

export CUDA_VISIBLE_DEVICES=1


python -m infer \
  --model_dir $MODEL_DIR \
  --vocab_source $VOCAB_SOURCE \
  --vocab_target $VOCAB_TARGET \
  --batch_size $BATCH_SIZE \
  --infer_source $INFER_SOURCE \
  --max_seq_len $MAX_SEQ_LEN \
  --beam_size $BEAM_SIZE \
  --batch_size $BATCH_SIZE \
  --delimiter " " \
  --output heheda
