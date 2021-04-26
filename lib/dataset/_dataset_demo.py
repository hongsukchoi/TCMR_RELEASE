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

import os
import cv2
import numpy as np
import os.path as osp
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor

from lib.utils.smooth_bbox import get_all_bbox_params
from lib.data_utils._img_utils import get_single_image_crop_demo


class CropDataset(Dataset):
    def __init__(self, image_folder, frames, bboxes=None, joints2d=None, scale=1.0, crop_size=224):
        self.image_file_names = [
            osp.join(image_folder, x)
            for x in os.listdir(image_folder)
            if x.endswith('.png') or x.endswith('.jpg')
        ]
        self.image_file_names = sorted(self.image_file_names)
        self.image_file_names = np.array(self.image_file_names)[frames]
        self.bboxes = bboxes
        self.joints2d = joints2d
        self.scale = scale
        self.crop_size = crop_size
        self.frames = frames
        self.has_keypoints = True if joints2d is not None else False

        self.norm_joints2d = np.zeros_like(self.joints2d)

        if self.has_keypoints:
            bboxes, time_pt1, time_pt2 = get_all_bbox_params(joints2d, vis_thresh=0.3)
            bboxes[:, 2:] = 150. / bboxes[:, 2:]
            self.bboxes = np.stack([bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 2]]).T

            self.image_file_names = self.image_file_names[time_pt1:time_pt2]
            self.joints2d = joints2d[time_pt1:time_pt2]
            self.frames = frames[time_pt1:time_pt2]

    def __len__(self):
        return len(self.image_file_names)

    def __getitem__(self, idx):
        img = cv2.cvtColor(cv2.imread(self.image_file_names[idx]), cv2.COLOR_BGR2RGB)

        bbox = self.bboxes[idx]

        j2d = self.joints2d[idx] if self.has_keypoints else None

        norm_img, raw_img, kp_2d = get_single_image_crop_demo(
            img,
            bbox,
            kp_2d=j2d,
            scale=self.scale,
            crop_size=self.crop_size)
        if self.has_keypoints:
            return norm_img, kp_2d
        else:
            return norm_img


class FeatureDataset(Dataset):
    def __init__(self, image_folder, frames, seq_len=16):
        self.image_file_names = [
            osp.join(image_folder, x)
            for x in os.listdir(image_folder)
            if x.endswith('.png') or x.endswith('.jpg')
        ]
        self.image_file_names = sorted(self.image_file_names)
        self.image_file_names = np.array(self.image_file_names)[frames]
        self.feature_list = None

        self.seq_len = seq_len
        self.seq_list = [[i, i+seq_len-1] for i in range(len(self.image_file_names) - seq_len + 1)]  # [i, i+seq_len-1] is inclusive
        for i in range(1, int(seq_len/2)+1):
            self.seq_list.insert(0, [int(seq_len/2)-i, int(seq_len/2)-i])
        for i in range(1, int(seq_len/2)):
            self.seq_list.append([-int(seq_len/2)+i, -int(seq_len/2)+i])

    def __len__(self):
        return len(self.seq_list)

    def get_sequence(self, start_index, end_index):
        if start_index != end_index:
            return self.feature_list[start_index:end_index+1]
        else:
            return self.feature_list[start_index][None,:].expand(self.seq_len,-1)

    def __getitem__(self, idx):
        start_idx, end_idx = self.seq_list[idx]

        return self.get_sequence(start_idx, end_idx)

