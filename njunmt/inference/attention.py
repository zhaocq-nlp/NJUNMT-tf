# Copyright 2017 Natural Language Processing Group, Nanjing University, zhaocq.nlp@gmail.com.
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
""" Common functions to process attention and pack for saving. """
import numpy
import json
import tensorflow as tf
from tensorflow import gfile


def postprocess_attention(beam_ids, attention_dict, gather_idx):
    """ Processes attention information.

    Args:
        beam_ids: beam ids returned by predictions, a numpy.ndarray.
        attention_dict: A dict of attention results (with numpy.ndarray).
        gather_idx: The gathered index(es) to return.

    Returns: Attention information.
    """
    new_attention_dict = dict()
    for k_att, v_att in attention_dict.items():
        if "encoder_self_attention" in k_att:
            # with shape [batch_size, num_heads, n_timesteps_trg, n_timesteps_src]
            # encoder self attention is not stacked by beam size
            new_attention_dict[k_att] = v_att
        else:
            # for encdec_attention or decoder self attention
            # [n_timesteps_trg, batch_size * beam_size, n_timesteps_src] if ndims=3
            # [n_timesteps_trg, batch_size * beam_size, num_heads, n_timesteps_src] if ndims=4
            gathered_att = numpy.zeros_like(v_att)
            num_shapes = len(gathered_att.shape)
            for idx in range(beam_ids.shape[0]):
                if num_shapes == 3:
                    gathered_att = gathered_att[:, beam_ids[idx], :]
                    gathered_att[idx, :, :] = v_att[idx]
                elif num_shapes == 4:
                    gathered_att = gathered_att[:, beam_ids[idx], :, :]
                    gathered_att[idx, :, :, :] = v_att[idx]
                else:
                    raise ValueError
            # if num_shapes == 3:
            # [n_timesteps_trg, batch_size, n_timesteps_src]
            # else:
            # [n_timesteps_trg, batch_size, num_heads, n_timesteps_src]
            new_attention_dict[k_att] = gathered_att[:, gather_idx]
    return select_attention_sample_by_sample(new_attention_dict)


def select_attention_sample_by_sample(attention_dict):
    """ The format of `attention_dict` is :
        { "attention_name": attention_matrix,
        ... }
        The shapes are described as follows:
        Attention Name(type)         Shape
        encoder_self_attention      [batch_size, num_heads, n_timesteps_src, n_timesteps_src]
        decoder_self_attention      [n_timesteps_trg, batch_size, num_heads, n_timesteps_trg]
        encoder_decoder_attention   [n_timesteps_trg, batch_size, num_heads, n_timesteps_src]
                                    or [n_timesteps_trg, batch_size, n_timesteps_src]

        The return values is a list:
            [sample0's attention, sample1's attention, ...].
            For each attention, it is a dict like:
            { "attention_name": attention_matrix,
            ... }
            The shapes are described as follows:
            Attention Name(type)         Shape
            encoder_self_attention      [num_heads, n_timesteps_src, n_timesteps_src]
            decoder_self_attention      [n_timesteps_trg, num_heads, n_timesteps_trg]
            encoder_decoder_attention   [n_timesteps_trg, num_heads, n_timesteps_src]
                                        or [n_timesteps_trg, n_timesteps_src]
    Args:
        attention_dict: A dict.

    Returns: A list of dicts.
    """
    num_samples = None
    all_attentions = []

    def emptys_fill_dict():
        assert num_samples
        for _ in range(num_samples):
            all_attentions.append(dict())

    for k_att, v_att in attention_dict.items():
        if "encoder_self_attention" in k_att:
            # with shape [batch_size, num_heads, n_timesteps_trg, n_timesteps_src]
            if not num_samples:
                num_samples = v_att.shape[0]
                emptys_fill_dict()
            for i in range(num_samples):
                all_attentions[i][k_att] = v_att[i]
        else:
            # for encdec_attention or decoder self attention
            # [n_timesteps_trg, batch_size, n_timesteps_src] if ndims=3
            # [n_timesteps_trg, batch_size, num_heads, n_timesteps_src] if ndims=4
            if not num_samples:
                num_samples = v_att.shape[1]
                emptys_fill_dict()
            for i in range(num_samples):
                all_attentions[i][k_att] = v_att[:, i]
    return all_attentions


def pack_batch_attention_dict(
        base_index,
        source_tokens,
        candidate_tokens,
        attentions):
    """ Packs the attention information into a dictionary for visualization.

    Args:
        base_index: An integer.
        source_tokens: A list of samples. Each sample is a list of string tokens.
        candidate_tokens: A list of sample candidate. Each sample candidate is a list of string tokens.
        attentions: A list of attentions.

    Returns: A packed dictionary of attention information for visualization.
    """
    ret_attentions = dict()
    for idx, (src, hypo, attention) in enumerate(
            zip(source_tokens, candidate_tokens, attentions)):
        att = {"source": " ".join(src),
               "translation": " ".join(hypo),
               "attentions": []}
        for key, val in attention.items():
            if "encoder_self_attention" in key:
                len_src = len(src) + 1
                len_trg = len(src) + 1
            elif "encoder_decoder_attention" in key:
                len_src = len(src) + 1
                len_trg = len(hypo) + 1
            elif "decoder_self_attention" in key:
                len_src = len(hypo) + 1
                len_trg = len(hypo) + 1
            else:
                raise NotImplementedError
            num_shapes = len(val.shape)
            if num_shapes == 2:
                # [n_timesteps_trg, n_timesteps_src]
                att["attentions"].append({
                    "name": key,
                    "value": val[:len_trg, :len_src].tolist(),
                    "type": "simple"})
            elif num_shapes == 3:
                if "decoder" in key:
                    # with shape [n_timesteps_trg, num_heads, n_timesteps_src]
                    #    transpose to [num_heads, n_timesteps_trg, n_timesteps_src]
                    att["attentions"].append({
                        "name": key,
                        "value": (val[:len_trg, :, :len_src]).transpose([1, 0, 2]).tolist(),
                        "type": "multihead"})
                else:
                    # with shape [num_heads, n_timesteps_trg, n_timesteps_src]
                    att["attentions"].append({
                        "name": key,
                        "value": val[:, :len_trg, :len_src].tolist(),
                        "type": "multihead"})
            else:
                raise NotImplementedError
        ret_attentions[base_index + idx] = att
    return ret_attentions


def dump_attentions(output_filename_prefix, attentions):
    """ Dumps attention as json format.

    Args:
        output_filename_prefix: A string.
        attentions: A dict of attention arrays.
    """
    tf.logging.info("Saving attention information into {}.attention.".format(output_filename_prefix))
    with gfile.GFile(output_filename_prefix + ".attention", "wb") as f:
        f.write(json.dumps(attentions).encode("utf-8"))
