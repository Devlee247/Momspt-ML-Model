# -*- coding: utf-8 -*-

import os
import torch
from lib.core import parse_args
from lib.core import Trainer
from lib.dataset.loader import get_data_loader
from lib.models import GIFV
from lib.utils.utils import get_optimizer
import torch.nn.functional as F

def main(cfg):
    # ====== CUDA 혹은 CUDNN 설정, GPU 관련 ====== #
    
    # ====== 데이터 로드 ====== #
    print(cfg)
    data_loader = get_data_loader(cfg)

    # ====== 네트워크, 옵티마이저 초기화 ====== #
    model = GIFV()

    if cfg.TRAIN.PRETRAINED != '' and os.path.isfile(cfg.TRAIN.PRETRAINED):
        checkpoint = torch.load(cfg.TRAIN.PRETRAINED)
        best_performance = checkpoint['loss']
        model.load_state_dict(checkpoint['gen_state_dict'])
        print(f'==> Loaded pretrained model from {cfg.TRAIN.PRETRAINED}...')
        print(f'Performance test set {best_performance}')
    else:
        print(f'{cfg.TRAIN.PRETRAINED} is not a pretrained model!!!!')

    optimizer = get_optimizer(
        model=model,
        optim_type=cfg.TRAIN.GEN_OPTIM,
        lr=cfg.TRAIN.GEN_LR
        )

    # ====== Start Training ====== #
    Trainer(
        model=model,
        data_loader=data_loader,
        optimizer=optimizer,
        epochs=cfg.TRAIN.EPOCHS
    ).fit()


if __name__ == '__main__':
    cfg, cfg_file = parse_args()
    
    main(cfg)