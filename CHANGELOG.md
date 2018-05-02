# Changelog
All notable changes to this project will be documented in this file.


## [Unreleased]

### Added
- WMT2017 EN<->DE benchmark.
- Load pretrained model.
- Layer normalization for fflayer and LayerNormLSTMCell
- Multi-GPU inference and evaluation (something wrong with training)

### Changed
- Default loss function.
- Process beam results on GPU.
- Replace multi-bleu.perl with an equivalent python version.
- Replace model_analysis APIs with tf.profiler.
- Move `input_fields` from class Dataset to class SequenceToSequence.

### Removed
- Configuration: ``multi_bleu_script`` and ``tokenize_scropt``.

### Fixed
- Bug when mode=EVAL and output attention with BPE.
- All str.split(" ") => str.split().
- Python3 compatibility of njunmt/tools/tokenizeChinese.py.

## [0.6.0] - 2018-01-26
### Added
- Evaluation entrance (bin/eval.py).
- Cache for decoders (especially for TransformerDecoder).
- Change Log.
- Attention output of Transformer.


### Changed
- Tensorflow 1.4 requirement.
- Attention file format when output_attention flag is enabled under mode=INFER/EVAL.
- BPE in ``Vocab`` accepts an extra vocab file (also see --vocabulary in njunmt/tools/apply_bpe.py).
- ``maximum_features_length`` and ``maximum_labels_length`` now indicate the length of encoded symbols (e.g. after BPE).
- Attention interfaces.
- ``BaseSeq2Seq`` -> ``SequenceToSequence``.
- More flexible and concise code structure of decoders and ``EnsembleModel``.

### Removed
- ``input_prepose_processing_fn`` interface in decoders.
- Redundant computation in TransformerDecoder (cache).

## 0.5.0 - 2018-01-09
### Added
- Transformer model.
- Model ensemble.
- Learning decaying functions.
- Evaluation metrics with loss or BLEU score (multi-bleu.perl).
- Beam search strategy with batch.
- Tensorboard visualization.
- Capability for BPE.
- Shell script to fetch WMT2014 en-de data.

### Changed
- More flexible and concise code structure.


## 0.1.0 - 2017-11
### Added
- Sequence-to-sequence model with attention.


[Unreleased]: https://github.com/zhaocq-nlp/NJUNMT-tf/compare/v0.6...master
[0.6.0]: https://github.com/zhaocq-nlp/NJUNMT-tf/compare/v0.5...v0.6

