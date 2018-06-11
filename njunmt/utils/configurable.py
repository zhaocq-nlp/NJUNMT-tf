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
""" Abstract base class for objects that are configurable using
a parameters dictionary.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import copy
import os
import six
import yaml
import tensorflow as tf
from tensorflow import gfile

from njunmt.utils.constants import Constants, ModeKeys


class abstractstaticmethod(staticmethod):  # pylint: disable=C0111,C0103
    #  """Decorates a method as abstract and static"""
    __slots__ = ()

    def __init__(self, function):
        super(abstractstaticmethod, self).__init__(function)
        function.__isabstractmethod__ = True

    __isabstractmethod__ = True


def _toggle_dropout(params, mode):
    """ Disable dropout probability during EVAL/INFER mode.

    Args:
        params: A dictionary of parameters.
        mode: A mode.

    Returns: A result dictionary.
    """
    params = copy.deepcopy(params)
    if mode != ModeKeys.TRAIN:
        for key, val in params.items():
            if type(val) is dict:
                params[key] = _toggle_dropout(params[key], mode)
            elif "dropout" in key:
                params[key] = 1.0 if "keep" in key else 0.
    return params


def _params_to_stringlist(params, prefix="   "):
    """ Convert a dictionary/list of parameters to a format string.

    Args:
        params: A dictionary/list of parameters.
        prefix: A string.

    Returns: A format string.

    Raises:
        ValueError: if unknown type of `params`.
    """
    param_list = []
    if isinstance(params, dict):
        for key, val in params.items():
            param_list.append(prefix + key + ": ")
            if isinstance(val, dict):
                param_list.extend(_params_to_stringlist(val, prefix + "   "))
            else:
                param_list[-1] += str(val)
    elif isinstance(params, list):
        prefix += "  "
        for item in params:
            for idx, (key, val) in enumerate(item.items()):
                if idx == 0:
                    newprefix = copy.deepcopy(prefix[:-2])
                    newprefix += "- "
                    param_list.append(newprefix + key + ": ")
                else:
                    param_list.append(prefix + key + ": ")
                if isinstance(val, dict):
                    param_list.extend(_params_to_stringlist(val, prefix + "   "))
                else:
                    param_list[-1] += str(val)
    else:
        raise ValueError("Unrecognized type of params: {}".format(str(params)))
    return param_list


def define_tf_flags(args):
    """ Defines tf FLAGS.

    Args:
        args: A dict, with format: {arg_name: [type, default_val, helper]}

    Returns: tf FLAGS.
    """
    for key, val in args.items():
        eval("tf.flags.DEFINE_{}".format(val[0]))(key, val[1], val[2])
    return tf.flags.FLAGS


def update_configs_from_flags(model_configs, tf_flags, flag_keys):
    """ Replaces `model_configs` with options defined in `tf_flags`.

    Args:
        model_configs: A dict.
        tf_flags: tf FLAGS.
        flag_keys: A set of keys.

    Returns: The updated dict.
    """

    def _update(mc, param_name):
        param_str = getattr(tf_flags, param_name)
        if param_str is None:
            return mc
        params = yaml.load(param_str)
        if params is None:
            return mc
        return deep_merge_dict(model_configs, {param_name: params})

    for key in flag_keys:
        model_configs = _update(model_configs, key)
    return model_configs


def load_from_config_path(config_paths):
    """ Loads configurations from files of yaml format.

    Args:
        config_paths: A string (each file name is seperated by ",") or
          a list of strings (file names).

    Returns: A dictionary of model configurations, parsed from config files.
    """
    if isinstance(config_paths, six.string_types):
        config_paths = config_paths.strip().split(",")
    assert isinstance(config_paths, list) or isinstance(config_paths, tuple)
    model_configs = dict()
    for config_path in config_paths:
        config_path = config_path.strip()
        if not config_path:
            continue
        if not gfile.Exists(config_path):
            raise OSError("config file does not exist: {}".format(config_path))
        config_path = os.path.abspath(config_path)
        tf.logging.info("loading configurations from {}".format(config_path))
        with gfile.GFile(config_path, "r") as config_file:
            config_flags = yaml.load(config_file)
            model_configs = deep_merge_dict(model_configs, config_flags)
    return model_configs


def maybe_load_yaml(item):
    """Parses `item` only if it is a string. If `item` is a dictionary
    it is returned as-is.
    Args:
        item:

    Returns: A dictionary.

    Raises:
        ValueError: if unknown type of `item`.
    """
    if isinstance(item, six.string_types):
        return yaml.load(item)
    elif isinstance(item, dict):
        return item
    else:
        raise ValueError("Got {}, expected string or dict", type(item))


def deep_merge_dict(dict_x, dict_y, path=None):
    """ Recursively merges dict_y into dict_x.

    Args:
        dict_x: A dict.
        dict_y: A dict.
        path:

    Returns: An updated dict of dict_x
    """
    if path is None: path = []
    for key in dict_y:
        if key in dict_x:
            if isinstance(dict_x[key], dict) and isinstance(dict_y[key], dict):
                deep_merge_dict(dict_x[key], dict_y[key], path + [str(key)])
            elif dict_x[key] == dict_y[key]:
                pass  # same leaf value
            else:
                dict_x[key] = dict_y[key]
        else:
            dict_x[key] = dict_y[key]
    return dict_x


def parse_params(params, default_params):
    """Parses parameter values to the types defined by the default parameters.
    Default parameters are used for missing values.

    Args:
        params: A dict.
        default_params: A dict to provide parameter structure and missing values.

    Returns: A updated dict.
    """
    # Cast parameters to correct types
    if params is None:
        params = {}
    result = copy.deepcopy(default_params)
    for key, value in params.items():
        # If param is unknown, drop it to stay compatible with past versions
        if key not in default_params:
            raise ValueError("{} is not a valid model parameter".format(key))
        # Param is a dictionary
        if isinstance(value, dict):
            default_dict = default_params[key]
            if not isinstance(default_dict, dict):
                raise ValueError("{} should not be a dictionary".format(key))
            if default_dict:
                value = parse_params(value, default_dict)
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


def print_params(title, params):
    """ Prints parameters.

    Args:
        title: A string.
        params: A dict.
    """
    tf.logging.info(title)
    for info in _params_to_stringlist(params):
        tf.logging.info(info)


def update_infer_params(
        model_configs,
        beam_size=None,
        maximum_labels_length=None,
        length_penalty=None):
    """ Resets inference-specific parameters.

    Args:
        model_configs: A dictionary of all model configurations.
        beam_size: The beam width, if provided, pass it to `model_configs`'s
           "model_params".
        maximum_labels_length: The maximum length of sequence that model generates,
          if provided, pass it to `model_configs`'s "model_params".
        length_penalty: The length penalty, if provided, pass it to
          `model_configs`'s "model_params".

    Returns: An updated dict.
    """
    if beam_size is not None:
        model_configs["model_params"]["inference.beam_size"] = beam_size
    if maximum_labels_length is not None:
        model_configs["model_params"]["inference.maximum_labels_length"] = maximum_labels_length
    if length_penalty is not None:
        model_configs["model_params"]["inference.length_penalty"] = length_penalty
    return model_configs


def update_eval_metric(
        model_configs,
        metric):
    """ Resets evaluation-specific parameters.

    Args:
        model_configs: A dictionary of all model configurations.
        metric: A string.

    Returns: A tuple `(updated_dict, metric_str)`.
    """
    if "modality.target.params" in model_configs["model_params"]:
        metric_str = model_configs["model_params"]["modality.target.params"]["loss"]
    else:
        metric_str = model_configs["model_params"]["modality.params"]["loss"]
    if metric is not None:
        metric_str = metric
    if "modality.target.params" in model_configs["model_params"]:
        model_configs["model_params"]["modality.target.params"]["loss"] = metric_str
    else:
        model_configs["model_params"]["modality.params"]["loss"] = metric_str
    return model_configs, metric_str


@six.add_metaclass(abc.ABCMeta)
class Configurable(object):
    """ Interface for all classes that are configurable
    via a parameters dictionary.
    """

    def __init__(self, params, mode, name=None, verbose=True):
        """ Initializes a class.

        Args:
            params: A dict of parameters.
            mode: A mode.
            name: The name of the object.
            verbose: Print object parameters if set True.
        """
        self._params = parse_params(params, self.default_params())
        self._params = _toggle_dropout(self.params, mode)
        self._verbose = verbose
        self._name = name
        self._mode = mode
        if verbose:
            print_params("Parameters for {} under mode={}:"
                         .format(self.__class__.__name__, self.mode),
                         self.params)
        self._check_parameters()

    def _check_parameters(self):
        """ Checks availability of parameters. """
        pass

    @property
    def name(self):
        """ Returns the name of the object. """
        return self._name

    @name.setter
    def name(self, val):
        """ Set the name. """
        self._name = val

    @property
    def verbose(self):
        """ Returns the verbose property. """
        return self._verbose

    @verbose.setter
    def verbose(self, val):
        """ Set the verbose property. """
        self._verbose = val

    @property
    def mode(self):
        """Returns the mode. """
        return self._mode

    @property
    def params(self):
        """Returns a dictionary of parsed parameters. """
        return self._params

    @abstractstaticmethod
    def default_params():
        """Returns a dictionary of default parameters. The default parameters
        are used to define the expected type of passed parameters. Missing
        parameter values are replaced with the defaults returned by this method.
        """
        raise NotImplementedError


class ModelConfigs:
    """ A class for dumping and loading model configurations. """

    @staticmethod
    def dump(model_config, output_dir):
        """ Dumps model configurations.

        Args:
            model_config: A dict.
            output_dir: A string, the output directory.
        """
        model_config_filename = os.path.join(output_dir, Constants.MODEL_CONFIG_YAML_FILENAME)
        if not gfile.Exists(output_dir):
            gfile.MakeDirs(output_dir)
        with gfile.GFile(model_config_filename, "w") as file:
            yaml.dump(model_config, file)

    @staticmethod
    def load(model_dir):
        """ Loads model configurations.

        Args:
            model_dir: A string, the directory.

        Returns: A dict.
        """
        model_config_filename = os.path.join(model_dir, Constants.MODEL_CONFIG_YAML_FILENAME)
        if not gfile.Exists(model_config_filename):
            raise OSError("Fail to find model config file: %s" % model_config_filename)
        with gfile.GFile(model_config_filename, "r") as file:
            model_configs = yaml.load(file)
        return model_configs
