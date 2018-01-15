# Neural Machine Translation
-------
**This** is an in-house implemented attention-based encoder-decoder model for neural machine translation system based on Tensorflow. 
 It has several features:
 - **Deep Model**: multi-layer bidirectional GRU/LSTM, multi-layer conditional GRU/LSTM
 - **All Layer Dropout**
 - **Batch Beam Search**: fast and accurate
 - **Learning Rate Annealing**: automatic learning rate annealing according to the loss on devset
 - **BLEU validation**: using BLEU score on devset to choose the best checkpoint
 

## Requirements
-------
1. Tensorflow >= 1.2.1
2. compatible for both python2 & python3

## New zh2en baseline for NJUNLP
Train: 1.34M LDC data
Dev: NIST03
Test: NIST04, NIST05, NIST06

|cell|dp|len|MT03(dev)|MT04|MT05|MT06|avg45|avg456|
|---|---|---|---|---|---|---|---|---|
|GRU|-|50|36.65|37.98|34.61|32.42|36.30|35.00|
|GRU|5.E-01|50|37.11|38.61|35.4|33.09|37.01|35.70|
|LSTM|-|50|37.62|38.66|34.4|32.71|36.53|35.26|
|LSTM|5.E-01|50|37.83|39.08|35|33.64|37.04|35.91|
|GRU|-|80|38.56|40.54|36.27|35.12|38.41|37.31|
|GRU|5.E-01|80|38.62|40.77|37.19|36.4|38.98|38.12|
|LSTM|-|80|39.64|41.46|37.13|36.4|39.30|38.33|
|LSTM|5.E-01|80|39.9|41.97|38.59|36.86|40.28|39.14|


## Simple Results (on GeForce GTX 1080)
------
I train a baseline NMT model on our in-house zh2en dataset (1.6M). The parameters are listed in `seq2seq/example_configs/baseline_hooks.json` and `seq2seq/example_configs/nmt_baseline.json`.

Training takes around 50k ~ 70k steps (less than 12 hours even with BLEU validation) with batch size=80 (there exists the best BLEU score on dev set). I use beam size=5 in inference procedure.

|cell|MT02dev|MT03|MT04|MT05|MT06|avg345|avg3456|
| --- | ---|---|---|---|---|---|---|
|GRU|36.49|35.08|37.24|33.51|31.59|35.28|34.36|
|LSTM|38.05|36.66|38.39|34.70|32.24|36.58|35.50|

I also test the inference time of this code on MT02 (878 sentences) with beam size=5.

|batch size|GPU mem occupied| time |
|---| ---|---|
|1|1.2G|89.96s|
|2|1.2G|87.36s|
|4|1.2G|67.17s|
|8|1.2G|55.76s|
|16|1.2G|56.66s|
|**32**|**1.2G**|**54.34s**|
|64|2.2G|53.47s|
|128|4.3G|50.19s|
|256|4.3G|50.88s|
|512|8G|48.23s|


## Usage
-------

> Data Preparation =>Generating Vocabulary => Hyperparameters => Training => Testing

### Data Preparation

- **Data Cleaning**: filter out bad characters, unaligned sentence pairs
- **Tokenization**: [tokenizer.pl from MosesDecoder](https://github.com/moses-smt/mosesdecoder/tree/master/scripts/tokenizer)
- **Lowercase**: if needed
- **Subword**: use [Byte-Pair Encoding](https://github.com/rsennrich/subword-nmt)

### Generating Vocabulary

execute seq2seq/tools/generate_vocab.py to generate the vocabulary

| parameter|description |
| --- | ---|
|[infile]| Input tokenized text file to be processed|
|--min_frequency|Minimum frequency of a word to be included in the vocabulary|
|--max_vocab_size | Maximum number of tokens in the vocabulary|
|--downcase | If set to true, down case all text before processing|
|--delimiter| Delimiter character for tokenizing. Use " " and "" for word and char level respectively|

e.g. generate the vocabulary of testdata/mt02.src with max_vocab_size=1000

``` bash
python seq2seq/tools/generate_vocab.py testdata/mt02.src \
      --max_vocab_size 1000  > vocab.zh
```

### Hyperparameters

See seq2seq/example_configs/nmt_local_debug.json and seq2seq/example_configs/default_hook.json.

**nmt_local_debug.json** is the model configuration file via JSON format (key-value pairs).
Parameter levels are as follows:

| key-level1|key-level2|default|description (value)|
| --- | ---| ---|---|
|model_params|-|-|(dict) of model parameters|
|-|embedding.dim.source|512|(int) dimension of source word embeddings|
|-|embedding.dim.target|512|(int) dimension of target word embeddings|
|-|encoder.params|-|(dict) of encoder parameters|
|-|decoder.params|-|(dict) of decoder parameters|
|-|embedding.share|false|(bool) whether to share the source and target embedding table|
|-|loss|CrossEntropy|(str) serve as an interface for many losses, not used now|
|-|inference.max_seq_len|100|(int) max sequence length, used when mode="INFER"|
|-|inference.beam_size|5|(int) beam size, used when mode="INFER"|
|optimizer_params|-|-|(dict) of optimizer parameters|
|-|optimizer.name|"Adam"|(str) name of optimizer, e.g. "Adam"|
|-|optimizer.params|-|(dict) parameters for the specific optimizer (see tensorflow document)|
|-|optimizer.learning_rate|5e-4|(float) initial learning rate|

Now, the "encoder.class" supports `encoders.rnn_encoder.UnidirectionalRNNEncoder` (for unidirectional RNN encoder) and `encoders.rnn_encoder.StackBidirectionalRNNEncoder` (for bidirectional RNN encoder). The detailed parameters are as follows:

| key-level2|key-level3|key-level4|default|description (value)|
| --- | ---| ---|---|---|
|encoder.params|-|-|-|(dict) of encoder parameters|
|-|rnn_cell|-|-|(dict) of rnn cell parameters|
|-|-|cell_class| "LSTMCell"|(str) encoder RNN cell class, "LSTMCell" or "GRUCell"|
|-|-|cell_params|{"num_units":1024}| (dict) RNN cell parameters, see details of the constructor of tf.contrib.rnn.LSTMCell or tf.contrib.rnn.GRUCell|
|-|-|dropout_input_keep_prob|1.0|(float) (take a good look at its name)|
|-|-|dropout_state_keep_prob|1.0|(float) (take a good look at its name)|
|-|-|num_layers|1|(int) number of the layers|

The "decoder.class" supports `decoders.attention_decoder.CondAttentionDecode` (for cGRU/cLSTM decoder) and `decoders.attention_decoder.SimpleDecoder` (for simple RNN decoder without attention). The detailed parameters are as follows:

| key-level2|key-level3|key-level4|default|description (value)|
| --- | ---| ---|---|---|
|decoder.params|-|-|-|(dict) of decoder parameters|
|-|attention_units|-|2048|(int) number of attention units, for CondAttentionDecoder only|
|-|rnn_cell|-|-|(dict) of rnn cell parameters|
|-|-|cell_class| "LSTMCell"|(str) decoder RNN cell class, "LSTMCell" or "GRUCell"|
|-|-|cell_params|{"num_units":1024}| (dict) RNN cell parameters, see details of the constructor of tf.contrib.rnn.LSTMCell or tf.contrib.rnn.GRUCell|
|-|-|dropout_input_keep_prob|1.0|(float) (take a good look at its name)|
|-|-|dropout_state_keep_prob|1.0|(float) (take a good look at its name)|
|-|-|num_layers|1|(int) number of the layers|
|-|dropout_context_keep_prob|-|1.0|(float) (take a good look at its name)|
|-|dropout_hidden_keep_prob|-|1.0|(float) (take a good look at its name)|
|-|dropout_embedding_keep_prob|-|1.0|(float) (take a good look at its name)|
|-|dropout_logit_keep_prob|-|1.0|(float) (take a good look at its name)|

**default_hook.json** is the configuration file for training hook. Hooks are additional code that you want to run either on every worker or just on the chief.

Take `SessionRunHook` as an example, when you implement a subclass of `SessionRunHook`, you are supposed to implement at least `before_run()` and `after_run()` function. Then, when you call `sess.run(train_op)` to train the model, the `MonitoredSession` will automatically do the following things:

1. call `before_run()` of all training hooks
2. run `train_op` to train the model
3. call `after_run()` of all training hooks

See seq2seq/training.training_hooks.py and train.py for more details.

Let's turn to two training hooks implemented here, `LearningRateAnnealHook` for learning rate annealing according to dev loss and `BLEUValidationHook` for evaluating the model via BLEU score on dev set tranlsations. The detailed parameters are as follows:

| key-level1|key-level2|key-level3|default|description (value)|
| --- | ---| ---|---|---|
|training_hooks|-|-|-|(dict) of training hooks parameters|
|-|LossMetricHook|-|-|(dict) of parameters of this hook|
|-|-|start_at|0|(int) start training step of evaluating dev loss|
|-|-|eval_every_n_steps|100|(int) (take a good look at its name)|
|-|-|batch_size|128|(int) batch size for evaluating model|
|-|-|half_lr|true|(bool) whether to automatically half the learning rate|
|-|-|max_patience|20|(int) maximum patience for annealing|
|-|-|min_learning_rate|5e-5| (float) minimum learning rate|
|-|-|dev_target_bpe|""|(str) if provided, use this file as evaluating target file|
|-|BLEUValidationHook|-|-|(dict) of parameters of this hook|
|-|-|start_at|10,000|(int) start training step of doing inference|
|-|-|eval_every_n_steps|1,000|(int) (take a good look at its name)|
|-|-|batch_size|32|(int) batch size for inference model|
|-|-|beam_size|5|(int) beam size for inference model|
|-|-|delimiter|" "|(str) delimiter for output tokens|
|-|-|tokenize_output|false|(bool) whether to tokenize output|
|-|-|maximum_keep_models|10|(int) store model file with top K BLEU scores|
|-|-|bleu_script|"./seq2seq/tools/multi-bleu.perl"|(str) multi-bleu script|

### Training
See train.sh for more details.


### Testing
See infer.sh for more details.


