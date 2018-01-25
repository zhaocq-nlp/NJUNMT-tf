# NJUNMT-tf

NJUNMT-tf is a general purpose sequence modeling tool in TensorFlow while neural machine translation is the main target task.


## Key features

**NJUNMT-tf builds NMT models almost from scratch without any high-level TensorFlow APIs which often hide details of many network components and lead to obscure code structure that is difficult to understand and manipulate. NJUNMT-tf only depends on basic TensorFlow modules, like array_ops, math_ops and nn_ops. Each operation in the code is under control.** </br>

NJUNMT-tf focuses on modularity and extensibility using standard TensorFlow modules and practices to support advanced modeling capability:

- arbitrarily complex encoder architectures, e.g. Bidirectional RNN encoder, Unidirectional RNN encoder and self-attention.
- arbitrarily complex decoder architectures, e.g. Conditional GRU/LSTM decoder, attention decoder and self-attention.
- hybrid encoder-decoder models, e.g. self-attention encoder and RNN decoder or vice versa.

and all of the above can be used simultaneously to train novel and complex architectures.

The code also supports:

- model ensemble.
- learning rate decaying according to loss on evaluation data.
- model validation on evaluation data with BLEU score and early stop strategy.
- monitoring with [TensorBoard](https://www.tensorflow.org/get_started/summaries_and_tensorboard).
- capability for [BPE](https://github.com/rsennrich/subword-nmt)


## Requirements

- `tensorflow` (`>=1.4`)
- `pyyaml`


## Quickstart

Here is a minimal workflow to get you started in using NJUNMT-tf. This example uses a toy Chinese-English dataset for machine translation with a toy setting.

1\. Build the word vocabularies:

``` bash
python -m bin.generate_vocab testdata/toy.zh --max_vocab_size 100  > testdata/vocab.zh
python -m bin.generate_vocab testdata/toy.en0 --max_vocab_size 100  > testdata/vocab.en
```

2\. Train with preset sequence-to-sequence parameters:
``` bash
export CUDA_VISIBLE_DEVICES=
python -m bin.train --model_dir test_model \
    --config_paths "
        ./njunmt/example_configs/toy_seq2seq.yml,
        ./njunmt/example_configs/toy_training_options.yml,
        ./default_configs/default_optimizer.yml"
```

3\. Translate a test file with the latest checkpoint:
``` bash
export CUDA_VISIBLE_DEVICES=
python -m bin.infer --model_dir test_models \
  --infer "
    beam_size: 4
    source_words_vocabulary: testdata/vocab.zh
    target_words_vocabulary: testdata/vocab.en" \
  --infer_data "
    - features_file: testdata/toy.zh
      labels_file: testdata/toy.en
      output_file: toy.trans
      output_attention: false"
```

**Note:** do not expect any good translation results with this toy example. Consider training on [larger parallel datasets](http://www.statmt.org/wmt16/translation-task.html) instead.

## Configuration

As you can see, there are two ways to manipulate hyperparameters of the process:

- tf FLAGS
- yaml-style config file

For example, there is a config file specifying the datasets for training procedure.
```
# datasets.yml
data:
  train_features_file: testdata/toy.zh
  train_labels_file: testdata/toy.en0
  eval_features_file: testdata/toy.zh
  eval_labels_file: testdata/toy.en
  source_words_vocabulary: testdata/vocab.zh
  target_words_vocabulary: testdata/vocab.en
```

You can either use the command:
``` bash
python -m bin.train --config_paths "datasets.yml" ...
```
or
``` bash
python -m bin.train --data "
    train_features_file: testdata/toy.zh
    train_labels_file: testdata/toy.en0
    eval_features_file: testdata/toy.zh
    eval_labels_file: testdata/toy.en
    source_words_vocabulary: testdata/vocab.zh
    target_words_vocabulary: testdata/vocab.en" ...
```
They are of the same effect.

The available FLAGS (or the top levels of yaml configs) for bin.train are as follows:
- **config_paths**: the paths for config files
- **model_dir**: the directory for saving checkpoints
- **train**: training options, e.g. batch size, maximum length
- **data**: training data, evaluation data, vocabulary and (optional) BPE codes
- **hooks**: a list of training hooks (not provided, in the current version)
- **metrics**: a list of validation metrics on evaluation data
- **model**: the class name of the model
- **model_params**: parameters for the model
- **optimizer_params**: parameters for optimizer

The available FLAGS (or the top levels of yaml configs) for bin.infer are as follows:
- **config_paths**: the paths for config files
- **model_dir**: the checkpoint directory or directories separated by commas for model ensemble
- **infer**: inference options, e.g. beam size, length penalty rate
- **infer_data**: a list of data file to be translated
- **weight_scheme**: the weight scheme for model ensemble (only "average" available now)

**Note that:**
- each FLAG should be a string of yaml-style
- the hyperparameters provided by FLAGS will overwrite those presented in config files
- illegal parameters will interrupt the program, so see [sample.yml](https://github.com/zhaocq-nlp/NJUNMT-tf/blob/master/njunmt/example_configs/sample.yml) of more detailed discription for each parameter.


## TODO

The following features remain unimplemented:

- multi-gpu training
- schedule sampling
- minimum risk training

and trustable results on open datasets (WMT) are supposed to be reported.


## Acknowledgments

The implementation is inspired by the following:
- *[Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)*
- [dl4mt-tutorial](https://github.com/nyu-dl/dl4mt-tutorial)
- [OpenNMT-tf](https://github.com/OpenNMT/OpenNMT-tf)
- [Google's seq2seq](https://github.com/google/seq2seq) </br>
*[Massive Exploration of Neural Machine Translation Architectures](https://arxiv.org/abs/1703.03906)*
- [Google's tensor2tensor](https://github.com/tensorflow/tensor2tensor) </br>
*[Attention is All You Need](https://arxiv.org/abs/1706.03762)*
- *[Stronger Baselines for Trustable Results in Neural Machine Translation](http://www.aclweb.org/anthology/W17-3203.pdf)*

## Contact

Any comments or suggestions are welcome.

Please email [zhaocq.nlp@gmail.com](mailto:zhaocq.nlp@gmail.com).

