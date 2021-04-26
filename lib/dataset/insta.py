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

import h5py
import torch
import logging
import numpy as np
import os.path as osp

from torch.utils.data import Dataset
from lib.core.config import TCMR_DB_DIR
from lib.data_utils._kp_utils import convert_kps
from lib.data_utils._img_utils import normalize_2d_kp, split_into_chunks

logger = logging.getLogger(__name__)

class Insta(Dataset):
    def __init__(self, load_opt, seqlen, overlap=0., debug=False):
        self.seqlen = seqlen
        self.mid_frame = int(seqlen/2)
        self.stride = int(seqlen * (1-overlap) + 0.5)
        self.h5_file = osp.join(TCMR_DB_DIR, 'insta_train_db.h5')

        with h5py.File(self.h5_file, 'r') as db:
            self.db = db
            self.vid_indices = split_into_chunks(self.db['vid_name'], seqlen, self.stride)

        print(f'InstaVariety number of dataset objects {self.__len__()}')

    def __len__(self):
        return len(self.vid_indices)

    def __getitem__(self, index):
        return self.get_single_item(index)

    def get_sequence(self, start_index, end_index, data):
        if start_index != end_index:
            return data[start_index:end_index+1]
        else:
            return data[start_index:start_index+1].repeat(self.seqlen, axis=0)

    def get_single_item(self, index):
        start_index, end_index = self.vid_indices[index]

        with h5py.File(self.h5_file, 'r') as db:
            self.db = db

            kp_2d = self.get_sequence(start_index, end_index, self.db['joints2D'])
            kp_2d = convert_kps(kp_2d, src='insta', dst='spin')
            kp_2d_tensor = np.ones((self.seqlen, 49, 3), dtype=np.float16)

            input = torch.from_numpy(self.get_sequence(start_index, end_index, self.db['features'])).float()

            vid_name = self.get_sequence(start_index, end_index, self.db['vid_name'])
            frame_id = self.get_sequence(start_index, end_index, self.db['frame_id']).astype(str)
            instance_id = np.array([v.decode('ascii') + f for v, f in zip(vid_name, frame_id)])

        for idx in range(self.seqlen):
            kp_2d[idx,:,:2] = normalize_2d_kp(kp_2d[idx,:,:2], 224)
            kp_2d_tensor[idx] = kp_2d[idx]

        repeat_num = 3
        target = {
            'features': input,
            'kp_2d': torch.from_numpy(kp_2d_tensor).float()[self.mid_frame].repeat(repeat_num, 1, 1), # 2D keypoints transformed according to bbox cropping
            # 'instance_id': instance_id
        }

        return target