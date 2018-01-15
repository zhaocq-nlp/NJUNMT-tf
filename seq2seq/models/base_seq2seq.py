# Copyright 2017 ZhaoChengqi, zhaocq@nlp.nju.edu.cn, Natural Language Processing Group, Nanjing University.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple
import tensorflow as tf

import seq2seq
from seq2seq.models import bridges
from seq2seq.decoders.feedback import TrainingFeedback
from seq2seq.decoders.feedback import BeamFeedback
from seq2seq.utils.global_names import GlobalNames
from seq2seq.utils.configurable import Configurable
from seq2seq.training.loss import crossentropy_loss


class EmbeddingTable(object):
    def __init__(self, vocab_size, dim_emb, init_scale=0.04, name=None):
        """ initialize embedding table

        :param vocab_size: vocabulary size
        :param dim_emb: dimension of embedding

        :param init_scale: init scale
        :param name: scope name of embedding table
        """
        self.embedding_table = tf.get_variable(name=(name or "embedding_table"),
                                               shape=[vocab_size, dim_emb],
                                               initializer=tf.random_uniform_initializer(
                                                   -init_scale, init_scale))

    def get_shape(self):
        return self.embedding_table.get_shape()

    def embed_words(self, words, time=0):
        """ embed the word

        :param word_ids: 1/2-dim tensor, the first dimension indicates batch_size
        :param time: indicating the position
        :return: embeddings: [batch_size, length, dim_emb]
        """
        emb = tf.nn.embedding_lookup(self.embedding_table, words)
        return emb


class BaseSeq2Seq(Configurable):
    def __init__(self, params, mode,
                 vocab_source, vocab_target,
                 scope="AttentionSeq2Seq"):
        """

        :param params:
        :param mode: tf.contrib.learn.Modekeys.TRAIN / INFER / EVAL
        :param vocab_source: vocab object
        :param vocab_target: vocab object
        """
        super(BaseSeq2Seq, self).__init__(params, mode, verbose=False)

        self.vocab_source = vocab_source
        self.vocab_target = vocab_target
        self.scope = scope
        self.seq2seq_info_tuple_type = namedtuple(
            "Seq2SeqInfo", "predictions attention_scores loss")

    @staticmethod
    def default_params():
        return {
            "encoder.class": "seq2seq.encoders.rnn_encoder.StackBidirectionalRNNEncoder",
            "encoder.params": {},  # Arbitrary parameters for the encoder
            "decoder.class": "seq2seq.decoders.rnn_decoder.CondAttentionDecoder",
            "decoder.params": {},  # Arbitrary parameters for the decoder
            "bridge.class": "ZeroBridge",
            "bridge.params": {},
            "embedding.dim.source": 512,
            "embedding.dim.target": 512,
            "embedding.init_scale": 0.04,
            "embedding.share": False,
            "inference.beam_size": 12,
            "inference.max_seq_len": 200,
            "loss": "CrossEntropy"
        }

    def build(self, input_fields):
        """
        build model
        :param input_fields: only source ids & seq_len for inference, both source and target for evaluate and training
        :return:
        """
        default_enc_scope = self.params["encoder.class"].split(".")[-1]
        default_dec_scope = self.params["decoder.class"].split(".")[-1]
        with tf.variable_scope(self.scope):
            encoder = self._create_encoder()
            encoder_output = self._encode(encoder=encoder,
                                          input_fields=input_fields,
                                          scope=default_enc_scope)

            encdec_bridge = self._create_bridge(encoder_output)
            decoder = self._create_decoder()
            decoder_output = self._decode(decoder=decoder,
                                          encdec_bridge=encdec_bridge,
                                          encoder_output=encoder_output,
                                          input_fileds=input_fields,
                                          scope=default_dec_scope)
        return decoder_output

    def _build_target_embedding_table(self):
        """ build target embedding table

        :return:
        """
        if self.params["embedding.share"]:
            vs = tf.get_variable_scope()
            with tf.variable_scope(vs, reuse=True):
                target_embedding_table = EmbeddingTable(vocab_size=self.vocab_target.vocab_size,
                                                        init_scale=self.params["embedding.init_scale"],
                                                        dim_emb=self.params["embedding.dim.target"])
        else:
            target_embedding_table = EmbeddingTable(vocab_size=self.vocab_target.vocab_size,
                                                    init_scale=self.params["embedding.init_scale"],
                                                    dim_emb=self.params["embedding.dim.target"],
                                                    name="target_embedding_table")
        return target_embedding_table

    def _decode(self, decoder, encdec_bridge, encoder_output, input_fileds, scope=None):
        """ dynamic decode

        :param decoder: decoder object
        :param encdec_bridge: `Bridge` object
        :param encoder_output: encoder output namedtuple
        :param input_fileds: input fields
        :param scope: scope
        :return:
        """
        target_embedding_table = self._build_target_embedding_table()
        if self.mode == tf.contrib.learn.ModeKeys.TRAIN \
                or self.mode == tf.contrib.learn.ModeKeys.EVAL:
            trg_seq = input_fileds[GlobalNames.PH_TARGET_SEQIDS_NAME]
            trg_seq_len = input_fileds[GlobalNames.PH_TARGET_SEQLENGTH_NAME]
            helper = TrainingFeedback(target_embedding_table=target_embedding_table,
                                      vocab=self.vocab_target,
                                      sequence=trg_seq,
                                      sequence_length=trg_seq_len)
            decoder_output, _ = decoder.decode(encoder_output, encdec_bridge, helper, scope=scope)
            loss = self._loss_func(decoder, decoder_output, trg_seq, trg_seq_len, scope)
            return self._finalize(encoder_output, decoder_output, loss, **input_fileds)

        else:  # self.mode == tf.contrib.learn.tf.contrib.learn.ModeKeys.INFER
            helper = BeamFeedback(target_embedding_table=target_embedding_table,
                                  vocab=self.vocab_target,
                                  batch_size=tf.shape(input_fileds[GlobalNames.PH_SOURCE_SEQIDS_NAME])[
                                      0],
                                  max_seq_len=self.params["inference.max_seq_len"],
                                  beam_size=self.params["inference.beam_size"])
            decoder_output, _ = decoder.decode(encoder_output, encdec_bridge, helper, scope=scope)

        return self._finalize(encoder_output, decoder_output, None, **input_fileds)

    def _loss_func(self, decoder, decoder_output, target_seq, target_seq_len, scope):
        """ compute loss

        :param decoder:
        :param decoder_output:
        :param target_seq: [batch_size, n_timesteps_trg, n_words_trg], with eos, without sos
        :param target_seq_len: [batch_size,]
        :param scope: decoder scope
        :return:
        """
        with tf.variable_scope(scope, "crossentropy"):
            logits = decoder.compute_logit(decoder_output)
            loss = crossentropy_loss(logits=logits,
                                     # transpose: [n_timesteps_trg, batch_size, n_words_trg]
                                     targets=tf.transpose(target_seq, [1, 0]),
                                     sequence_length=target_seq_len)
        return loss

    def _build_source_embedding_table(self):
        """ build source embedding table

        :return:
        """
        if self.params["embedding.share"]:
            assert self.vocab_source.equals_to(self.vocab_target), \
                "source vocabulary should be equal to target vocabulary when embedding.share==True"
            assert self.params["embedding.dim.source"] == self.params["embedding.dim.target"], \
                "embedding dimension should be the same when embedding.share==True"
            source_embedding_table = EmbeddingTable(vocab_size=self.vocab_source.vocab_size,
                                                    init_scale=self.params["embedding.init_scale"],
                                                    dim_emb=self.params["embedding.dim.source"])
        else:
            source_embedding_table = EmbeddingTable(vocab_size=self.vocab_source.vocab_size,
                                                    init_scale=self.params["embedding.init_scale"],
                                                    dim_emb=self.params["embedding.dim.source"],
                                                    name="source_embedding_table")
        return source_embedding_table

    def _encode(self, encoder, input_fields, scope=None):
        """
        encode source ids: 1. to embedding; 2. encode
        :param encoder:
        :param input_fields:
        :param scope:
        :return:
        """
        source_ids = input_fields[GlobalNames.PH_SOURCE_SEQIDS_NAME]
        source_seq_length = input_fields[GlobalNames.PH_SOURCE_SEQLENGTH_NAME]
        source_embedding_table = self._build_source_embedding_table()
        source_embeddings = source_embedding_table.embed_words(source_ids)
        encoder_output = encoder.encode(source_embeddings, source_seq_length, scope=scope)
        return encoder_output

    def _create_encoder(self):
        """
        create encoder
        :return:
        """
        tf.logging.info("Creating ENCODER: %s for %s", self.params["encoder.class"], self.mode)
        encoder = eval(self.params["encoder.class"])(params=self.params['encoder.params'], mode=self.mode)
        return encoder

    def _create_decoder(self):
        """
        building decoder
        :return:
        """
        tf.logging.info("Creating DECODER: %s for %s", self.params["decoder.class"], self.mode)
        decoder = eval(self.params["decoder.class"])(
            self.params['decoder.params'], self.mode,
            self.vocab_target.vocab_size)
        return decoder

    def _create_bridge(self, encoder_output):
        """ create bridge between encoder & decoder

        :param encoder_output: encoder output namedtuple
        :return:
        """
        if hasattr(bridges, self.params["bridge.class"]):
            encdec_bridge = getattr(bridges, self.params["bridge.class"])(
                params={},
                encoder_output=encoder_output,
                mode=self.mode)
        else:
            raise ValueError("bridge.class: %s not exists" % self.params["bridge.class"])
        return encdec_bridge

    def _finalize(self, encoder_output, decoder_output, loss, **kwargs):
        predict_out = None
        att = None
        if hasattr(decoder_output, "attention_scores"):
            att = decoder_output.attention_scores
        if self.mode == tf.contrib.learn.ModeKeys.INFER:
            predict_out = dict()
            predict_out["predicted_ids"] = decoder_output.predicted_ids
            predict_out["sequence_lengths"] = decoder_output.lengths
            predict_out["beam_ids"] = decoder_output.beam_ids
            predict_out["log_probs"] = decoder_output.log_probs
            if att is not None:
                predict_out["attention_scores"] = att

        return self.seq2seq_info_tuple_type(
            predictions=predict_out,
            attention_scores=att,
            loss=loss)
