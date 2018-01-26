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


def process_attention_output(predict_out, gather_idx):
    """ Processes attention information.

    Args:
        predict_out: A dict of prediction output (with numpy.ndarray).
        gather_idx: The gathered index(es) to return.

    Returns: Attention information.
    """
    beam_ids = predict_out["beam_ids"]
    all_attentions = dict()
    for k_att, v_att in predict_out["attentions"].items():
        if "encoder_self_attention" in k_att:
            all_attentions[k_att] = v_att
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
            if num_shapes == 3:
                # [n_timesteps_trg, batch_size, n_timesteps_src]
                all_attentions[k_att] = gathered_att[:, gather_idx, :]
            else:  # [n_timesteps_trg, batch_size, num_heads, n_timesteps_src]
                # transpose to [batch_size, num_heads, n_timesteps_trg, n_timesteps_src]
                all_attentions[k_att] = gathered_att[:, gather_idx, :, :].transpose([1, 2, 0, 3])
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
        attentions: A dict of attentions.

    Returns: A packed dictionary of attention information for visualization.
    """
    ret_attentions = dict()
    for idx in range(len(source_tokens)):
        att = {"source": " ".join(source_tokens[idx]),
               "translation": " ".join(candidate_tokens[idx]),
               "attentions": []}
        for key, val in attentions.items():
            if "encoder_self_attention" in key:
                len_src = len(source_tokens[idx]) + 1
                len_trg = len(source_tokens[idx]) + 1
            elif "encoder_decoder_attention" in key:
                len_src = len(source_tokens[idx]) + 1
                len_trg = len(candidate_tokens[idx]) + 1
            elif "decoder_self_attention" in key:
                len_src = len(candidate_tokens[idx]) + 1
                len_trg = len(candidate_tokens[idx]) + 1
            else:
                raise NotImplementedError
            num_shapes = len(val.shape)
            if num_shapes == 3:
                # [n_timesteps_trg, batch_size, n_timesteps_src]
                att["attentions"].append({
                    "name": key,
                    "value": val[:len_trg, idx, :len_src].tolist(),
                    "type": "simple"})
            elif num_shapes == 4:
                # [batch_size, num_heads, length_q, length_k]
                att["attentions"].append({
                    "name": key,
                    "value": val[idx, :, :len_trg, :len_src].tolist(),
                    "type": "multihead"})
            else:
                raise NotImplementedError
        ret_attentions[base_index + idx] = att
    return ret_attentions
