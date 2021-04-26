# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import random
import numpy as np
from torch.utils.data import Dataset, ConcatDataset, DataLoader

from lib.dataset import *


class MultipleDatasets(Dataset):
    def __init__(self, dbs, make_same_len=True):
        self.dbs = dbs
        self.db_num = len(self.dbs)
        self.max_db_data_num = max([len(db) for db in dbs])
        self.db_len_cumsum = np.cumsum([len(db) for db in dbs])
        self.make_same_len = make_same_len

    def __len__(self):
        # all dbs have the same length
        if self.make_same_len:
            return self.max_db_data_num * self.db_num
        # each db has different length
        else:
            return sum([len(db) for db in self.dbs])

    def __getitem__(self, index):
        if self.make_same_len:
            db_idx = random.randint(0,self.db_num-1) # uniform sampling
            data_idx = index % self.max_db_data_num
            if data_idx >= len(self.dbs[db_idx]) * (self.max_db_data_num // len(self.dbs[db_idx])): # last batch: random sampling
                data_idx = random.randint(0,len(self.dbs[db_idx])-1)
            else: # before last batch: use modular
                data_idx = data_idx % len(self.dbs[db_idx])
        else:
            for i in range(self.db_num):
                if index < self.db_len_cumsum[i]:
                    db_idx = i
                    break
            if db_idx == 0:
                data_idx = index
            else:
                data_idx = index - self.db_len_cumsum[db_idx-1]

        return self.dbs[db_idx][data_idx]


def get_data_loaders(cfg):
    if cfg.TRAIN.OVERLAP:
        overlap = ((cfg.DATASET.SEQLEN-1)/float(cfg.DATASET.SEQLEN))
    else:
        overlap = 0

    def get_2d_datasets(dataset_names):
        datasets = []
        for dataset_name in dataset_names:
            db = eval(dataset_name)(load_opt=cfg.TITLE,  seqlen=cfg.DATASET.SEQLEN, overlap=overlap, debug=cfg.DEBUG)
            datasets.append(db)
        return ConcatDataset(datasets)

    def get_3d_datasets(dataset_names):
        datasets = []
        for dataset_name in dataset_names:
            db = eval(dataset_name)(load_opt=cfg.TITLE, set='train', seqlen=cfg.DATASET.SEQLEN, overlap=overlap, debug=cfg.DEBUG)
            datasets.append(db)
        return ConcatDataset(datasets)

    # ===== 2D keypoint datasets =====
    train_2d_dataset_names = cfg.TRAIN.DATASETS_2D
    train_2d_db = get_2d_datasets(train_2d_dataset_names)

    data_2d_batch_size = int(cfg.TRAIN.BATCH_SIZE * cfg.TRAIN.DATA_2D_RATIO)
    data_3d_batch_size = cfg.TRAIN.BATCH_SIZE - data_2d_batch_size

    train_2d_loader = DataLoader(
        dataset=train_2d_db,
        batch_size=data_2d_batch_size,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
    )

    # ===== 3D keypoint datasets =====
    train_3d_dataset_names = cfg.TRAIN.DATASETS_3D
    train_3d_db = get_3d_datasets(train_3d_dataset_names)

    train_3d_loader = DataLoader(
        dataset=train_3d_db,
        batch_size=data_3d_batch_size,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
    )

    # exclude motion discriminator
    # ===== Motion Discriminator dataset =====
    # motion_disc_db = AMASS(seqlen=cfg.DATASET.SEQLEN)
    #
    # motion_disc_loader = DataLoader(
    #     dataset=motion_disc_db,
    #     batch_size=cfg.TRAIN.BATCH_SIZE,
    #     shuffle=True,
    #     num_workers=cfg.NUM_WORKERS,
    # )

    # ===== Evaluation dataset =====
    overlap = ((cfg.DATASET.SEQLEN-1)/float(cfg.DATASET.SEQLEN))
    valid_db = eval(cfg.TRAIN.DATASET_EVAL)(load_opt=cfg.TITLE, set='val', seqlen=cfg.DATASET.SEQLEN, overlap=overlap, debug=cfg.DEBUG)
    # valid_db.vid_indices = valid_db.vid_indices[::2]

    valid_loader = DataLoader(
        dataset=valid_db,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
    )

    return train_2d_loader, train_3d_loader, valid_loader
    # exclude motion discriminator
    # return train_2d_loader, train_3d_loader, motion_disc_loader, valid_loader