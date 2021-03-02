# Written by Seonwoo Min, Seoul National University (mswzeus@gmail.com)

import os
import sys
import json

import torch

from src.utils import Print


class DataConfig():
    def __init__(self, file=None, idx="data_config"):
        """ data configurations """
        self.idx = idx
        self.path = {}
        self.data_idx = None

        # load config from json file
        if file is not None:
            if not os.path.exists(file): sys.exit("data-config [%s] does not exists" % file)
            else: cfg = json.load(open(file, "r"))

            for key, value in cfg.items():
                if "path" in key:               self.path[key.split("_")[0]] = value
                elif "idx" in key:              self.data_idx = value
                else: sys.exit("# ERROR: invalid key [%s] in data-config file" % key)

        self.split = json.load(open(self.path["data"] + "FASTA/data_split.json", "r"))

    def get_config(self):
        configs = []
        configs.append(["path", self.path])
        configs.append(["idx",  self.data_idx])
        return configs


class ModelConfig():
    def __init__(self, file=None, idx="model_config"):
        """ model configurations """
        self.idx = idx
        self.embedder = None
        self.embed_dim = None
        self.num_channels = None
        self.kernel_size = None

        # load config from json file
        if file is not None:
            if not os.path.exists(file): sys.exit("model-config [%s] does not exists" % file)
            else: cfg = json.load(open(file, "r"))

            for key, value in cfg.items():
                if key == "embedder":                       self.embedder = value
                elif key == "embed_dim":                    self.embed_dim = value
                elif key == "num_channels":                 self.num_channels = value
                elif key == "kernel_size":                  self.kernel_size = value
                else: sys.exit("# ERROR: invalid key [%s] in model-config file" % key)

    def get_config(self):
        configs = []
        configs.append(["embedder", self.embedder])
        configs.append(["embed_dim", self.embed_dim])
        configs.append(["num_channels", self.num_channels])
        configs.append(["kernel_size", self.kernel_size])

        return configs


class RunConfig():
    def __init__(self, file=None, idx="run_config", eval=False, sanity_check=False):
        """ run configurations """
        self.idx = idx
        self.eval = eval
        self.batch_size = None
        self.num_epochs = None
        self.learning_rate = None
        self.dropout_rate = None
        self.class_weight = None

        # load config from json file
        if file is not None:
            if not os.path.exists(file): sys.exit("run-config [%s] does not exists" % file)
            else: cfg = json.load(open(file, "r"))

            for key, value in cfg.items():
                if   key == "batch_size":                   self.batch_size = value
                elif key == "num_epochs":                   self.num_epochs = value
                elif key == "learning_rate":                self.learning_rate = value
                elif key == "dropout_rate":                 self.dropout_rate = value
                elif key == "class_weight":                 self.class_weight = value
                else: sys.exit("# ERROR: invalid key [%s] in run-config file" % key)

        if sanity_check:
            self.batch_size = 32
            self.num_epochs = 4

    def get_config(self):
        configs = []
        configs.append(["batch_size", self.batch_size])
        if not self.eval:
            configs.append(["num_epochs", self.num_epochs])
            configs.append(["learning_rate", self.learning_rate])
            configs.append(["dropout_rate", self.dropout_rate])
            configs.append(["class_weight", self.class_weight])

        return configs


def print_configs(args, cfgs, device, output):
    if args["sanity_check"]: Print(" ".join(['##### SANITY_CHECK #####']), output)
    Print(" ".join(['##### arguments #####']), output)
    for cfg in cfgs:
        Print(" ".join(['%s:' % cfg.idx, str(args[cfg.idx])]), output)
        for c, v in cfg.get_config():
            Print(" ".join(['-- %s: %s' % (c, v)]), output)
    if args["checkpoint"] is not None:
        Print(" ".join(['checkpoint: %s' % (args["checkpoint"])]), output)
    Print(" ".join(['device: %s (%d GPUs)' % (device, torch.cuda.device_count())]), output)
    Print(" ".join(['output_path:', str(args["output_path"])]), output)
    Print(" ".join(['log_file:', str(output.name)]), output, newline=True)