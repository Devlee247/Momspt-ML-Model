#-*- coding: utf-8 -*-

from torch.utils.data import DataLoader

from lib.dataset import *


def get_data_loader(cfg):
    csv_path = cfg.DATASET_DIR
    gesture_db = GestureDataset(csv_path,)
    gesture_data_loader = DataLoader(
        dataset = gesture_db,
        batch_size= cfg.TRAIN.BATCH_SIZE,
        shuffle=True
    )
    return gesture_data_loader