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

import abc
import copy
import os
import six
import json
from pydoc import locate

import tensorflow as tf
from tensorflow import gfile

from seq2seq.utils.global_names import GlobalNames


class abstractstaticmethod(staticmethod):  # pylint: disable=C0111,C0103
    #  """Decorates a method as abstract and static"""
    __slots__ = ()

    def __init__(self, function):
        super(abstractstaticmethod, self).__init__(function)
        function.__isabstractmethod__ = True

    __isabstractmethod__ = True


def _toggle_dropout(params, mode):
    """
    disable dropout during evaluate/infer mode
    toggle each parameter in "params" whose name has string "dropout"
    :param cell_params:
    :param mode:
    :return:
    """
    params = copy.deepcopy(params)
    if mode != tf.contrib.learn.ModeKeys.TRAIN:
        for key, val in params.items():
            if type(val) is dict:
                params[key] = _toggle_dropout(params[key], mode)
            elif "dropout" in key:
                params[key] = 1.0 if "keep" in key else 0.
    return params


def _params_to_stringlist(params, prefix=""):
    param_list = []
    for key, val in params.items():
        param_list.append(prefix + key + ": ")
        if type(val) is dict:
            param_list.extend(_params_to_stringlist(val, prefix + "\t"))
        else:
            param_list[-1] += str(val)
    return param_list


def _create_from_dict(dict_, default_module, *args, **kwargs):
    """Creates a configurable class from a dictionary. The dictionary must have
    "class" and "params" properties. The class can be either fully qualified, or
    it is looked up in the modules passed via `default_module`.
    """
    class_ = locate(dict_["class"]) or getattr(default_module, dict_["class"])
    params = {}
    if "params" in dict_:
        params = dict_["params"]
    instance = class_(params, *args, **kwargs)
    return instance


def _maybe_load_json(item):
    """Parses `item` only if it is a string. If `item` is a dictionary
    it is returned as-is.
    load as json from yaml style
    """
    if isinstance(item, six.string_types):
        item = ''.join(item.strip().split())
        new_item = '{ '
        for i in item.split(','):
            if i == "":
                break
            tokens = i.split(':')
            new_item += '"%s":' % tokens[0]
            try:
                hehe = int(tokens[1])
            except:
                new_item += '"%s",' % tokens[1]
                continue
            new_item += '%s,' % tokens[1]
        new_item = new_item[:-1] + '}'
        # print(new_item)
        return json.loads(new_item)
    elif isinstance(item, dict):
        return item
    else:
        raise ValueError("Got {}, expected json string or dict", type(item))


def _deep_merge_dict(dict_x, dict_y, path=None):
    """Recursively merges dict_y into dict_x.
    """
    if path is None: path = []
    for key in dict_y:
        if key in dict_x:
            if isinstance(dict_x[key], dict) and isinstance(dict_y[key], dict):
                _deep_merge_dict(dict_x[key], dict_y[key], path + [str(key)])
            elif dict_x[key] == dict_y[key]:
                pass  # same leaf value
            else:
                dict_x[key] = dict_y[key]
        else:
            dict_x[key] = dict_y[key]
    return dict_x


def _parse_params(params, default_params):
    """Parses parameter values to the types defined by the default parameters.
    Default parameters are used for missing values.
    """
    # Cast parameters to correct types
    if params is None:
        params = {}
    result = copy.deepcopy(default_params)
    for key, value in params.items():
        # If param is unknown, drop it to stay compatible with past versions
        if key not in default_params:
            raise ValueError("%s is not a valid model parameter" % key)
        # Param is a dictionary
        if isinstance(value, dict):
            default_dict = default_params[key]
            if not isinstance(default_dict, dict):
                raise ValueError("%s should not be a dictionary", key)
            if default_dict:
                value = _parse_params(value, default_dict)
            else:
                # If the default is an empty dict we do not typecheck it
                # and assume it's done downstream
                pass
        if value is None:
            continue
        if default_params[key] is None:
            result[key] = value
        else:
            result[key] = type(default_params[key])(value)
    return result


@six.add_metaclass(abc.ABCMeta)
class Configurable(object):
    """Interface for all classes that are configurable
    via a parameters dictionary.

    Args:
        params: A dictionary of parameters.
        mode: A value in tf.contrib.learn.ModeKeys
    """

    def __init__(self, params, mode, verbose=True):
        self._params = _deep_merge_dict(self.default_params(), params)
        self._params = _toggle_dropout(self.params, mode)
        self._mode = mode
        if verbose:
            self._print_params()

    def _print_params(self):
        """Logs parameter values"""
        classname = self.__class__.__name__

        tf.logging.info("=============== Creating %s in mode=%s ==============", classname, self.mode)
        for s in _params_to_stringlist(self.params):
            tf.logging.info("| " + s)
        tf.logging.info("============================================================")

    @property
    def mode(self):
        """Returns a value in tf.contrib.learn.ModeKeys.
        """
        return self._mode

    @property
    def params(self):
        """Returns a dictionary of parsed parameters.
        """
        return self._params

    @abstractstaticmethod
    def default_params():
        """Returns a dictionary of default parameters. The default parameters
        are used to define the expected type of passed parameters. Missing
        parameter values are replaced with the defaults returned by this method.
        """
        raise NotImplementedError


class ModelConfigs:
    """ for dumping and load model configs

    """
    @staticmethod
    def dump(model_config, output_dir):
        model_config_filename = os.path.join(output_dir, GlobalNames.MODEL_CONFIG_JSON_FILENAME)
        if not gfile.Exists(output_dir):
            gfile.MakeDirs(output_dir)
        with gfile.GFile(model_config_filename, "wb") as file:
            file.write(json.dumps(model_config).encode("utf-8"))

    @staticmethod
    def load(output_dir):
        model_config_filename = os.path.join(output_dir, GlobalNames.MODEL_CONFIG_JSON_FILENAME)
        if not gfile.Exists(model_config_filename):
            raise OSError("Fail to find model config file: %s" % model_config_filename)
        with gfile.GFile(model_config_filename, "rb") as file:
            model_configs = json.load(file, encoding="utf-8")
        return model_configs
