# Written by Seonwoo Min, Seoul National University (mswzeus@gmail.com)

from thop import profile

import torch.nn as nn

from src.model.model import CNN
from src.embed import get_embedding_dim
from src.utils import Print


def get_model(model_cfg, run_cfg):
    """ get a DeepHSP/DeeperHSP model """
    in_channels = get_embedding_dim(model_cfg.embedder)
    model = CNN(model_cfg, in_channels, run_cfg.dropout_rate)
    params = get_params_and_initialize(model)

    return model, params


def get_params_and_initialize(model):
    """ parameter initialization """
    params = []
    for name, param in model.named_parameters():
        if "weight" in name: nn.init.kaiming_normal_(param, nonlinearity='relu')
        else:                nn.init.zeros_(param)
        params.append(param)

    return params


def get_profile(model, dataset, output):
    """ Params, """
    data = dataset.X[0].unsqueeze(0)
    macs, params = profile(model, (data, ), verbose=False)
    Print("Params(M): %.3f" % (params / 10**6), output)
