# -*- coding: utf-8 -*-

import torch

def get_optimizer(model, optim_type, lr):
    if optim_type in ['sgd', 'SGD']:
        opt = torch.optim.SGD(lr=lr, params=model.parameters())
    else:
        raise ModuleNotFoundError
    return opt