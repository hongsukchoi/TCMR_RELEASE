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
import torch
import random
import logging
import numpy as np
import os.path as osp
import joblib

from torch.utils.data import Dataset
from lib.core.config import TCMR_DB_DIR
from lib.data_utils._kp_utils import convert_kps
from lib.data_utils._img_utils import normalize_2d_kp, transfrom_keypoints, split_into_chunks, get_single_image_crop
from lib.utils.eval_utils import compute_accel


logger = logging.getLogger(__name__)


class Dataset3D(Dataset):
    def __init__(self, load_opt, set, seqlen, overlap=0., folder=None, dataset_name=None, debug=False, target_vid=''):

        self.load_opt = load_opt
        self.folder = folder
        self.set = set
        self.dataset_name = dataset_name
        self.seqlen = seqlen
        self.mid_frame = int(seqlen/2)

        self.stride = int(seqlen * (1-overlap) + 0.5)
        self.debug = debug
        self.db = self.load_db()

        if (set!='train') and (dataset_name=='3dpw') and (target_vid!=''):
            self.select_vid(target_vid)

        print("is_train: ", (set=='train'))
        self.vid_indices = split_into_chunks(self.db['vid_name'], seqlen, self.stride, is_train=(set=='train'))

    def select_vid(self, target_vid=''):
        valid_names = self.db['vid_name']
        unique_names = np.unique(valid_names)
        for u_n in unique_names:
            if not target_vid in u_n:
                continue

            indexes = valid_names == u_n
            if "valid" in self.db:
                valids = self.db['valid'][indexes].astype(bool)
            else:
                valids = np.ones(self.db['features'][indexes].shape[0]).astype(bool)

            new_db = {
                'vid_name': self.db['vid_name'][indexes][valids],
                'frame_id': self.db['frame_id'][indexes][valids],
                'img_name': self.db['img_name'][indexes][valids],
                'joints3D': self.db['joints3D'][indexes][valids],
                'joints2D': self.db['joints2D'][indexes][valids],
                'shape': self.db['shape'][indexes][valids],
                'pose': self.db['pose'][indexes][valids],
                'bbox': self.db['bbox'][indexes][valids],
                'valid': self.db['valid'][indexes][valids],
                'features': self.db['features'][indexes][valids]
            }
        self.db = new_db

    def __len__(self):
        return len(self.vid_indices)

    def __getitem__(self, index):
        return self.get_single_item(index)

    def load_db(self):

        db_file = osp.join(TCMR_DB_DIR, f'{self.dataset_name}_{self.set}_db.pt')

        if self.set == 'train':
            if self.load_opt == 'repr_table4_3dpw_model':
                if self.dataset_name == '3dpw':
                    db_file = osp.join(TCMR_DB_DIR, f'{self.dataset_name}_{self.set}_occ_db.pt')
                elif self.dataset_name == 'mpii3d':
                    db_file = osp.join(TCMR_DB_DIR, f'{self.dataset_name}_{self.set}_scale12_occ_db.pt')
                elif self.dataset_name == 'h36m':
                    db_file = osp.join(TCMR_DB_DIR, f'{self.dataset_name}_{self.set}_25fps_occ_db.pt')

            elif self.load_opt == 'repr_table4_h36m_mpii3d_model':
                if self.dataset_name == '3dpw':
                    db_file = osp.join(TCMR_DB_DIR, f'{self.dataset_name}_{self.set}_db.pt')
                elif self.dataset_name == 'mpii3d':
                    db_file = osp.join(TCMR_DB_DIR, f'{self.dataset_name}_{self.set}_scale12_db.pt')
                elif self.dataset_name == 'h36m':
                    db_file = osp.join(TCMR_DB_DIR, f'{self.dataset_name}_{self.set}_25fps_db.pt')

            elif self.load_opt == 'repr_table6_3dpw_model':
                if self.dataset_name == 'mpii3d':
                    db_file = osp.join(TCMR_DB_DIR, f'{self.dataset_name}_{self.set}_scale12_new_occ_db.pt')
                elif self.dataset_name == 'h36m':
                    db_file = osp.join(TCMR_DB_DIR, f'{self.dataset_name}_{self.set}_25fps_occ_db.pt')

            elif self.load_opt == 'repr_table6_h36m_model':
                if self.dataset_name == 'mpii3d':
                    db_file = osp.join(TCMR_DB_DIR, f'{self.dataset_name}_{self.set}_scale1_db.pt')
                elif self.dataset_name == 'h36m':
                    db_file = osp.join(TCMR_DB_DIR, f'{self.dataset_name}_{self.set}_25fps_tight_db.pt')

            elif self.load_opt == 'repr_table6_mpii3d_model':
                if self.dataset_name == 'mpii3d':
                    db_file = osp.join(TCMR_DB_DIR, f'{self.dataset_name}_{self.set}_scale12_db.pt')
                elif self.dataset_name == 'h36m':
                    db_file = osp.join(TCMR_DB_DIR, f'{self.dataset_name}_{self.set}_25fps_db.pt')

        elif self.set == 'val' and self.dataset_name == 'h36m':
            # if self.load_opt == 'repr_table4_h36m_mpii3d_model':
            #     db_file = osp.join(TCMR_DB_DIR, f'{self.dataset_name}_{self.set}_25fps_tight_db.pt')
            if self.load_opt == 'repr_table6_h36m_model':
                db_file = osp.join(TCMR_DB_DIR, f'{self.dataset_name}_test_front_25fps_tight_db.pt')

        elif self.set == 'val' and self.dataset_name == 'mpii3d':
            db_file = osp.join(TCMR_DB_DIR, f'{self.dataset_name}_{self.set}_scale12_db.pt')

        if osp.isfile(db_file):
            db = joblib.load(db_file)
        else:
            raise ValueError(f'{db_file} do not exists')

        print(f'Loaded {self.dataset_name} dataset from {db_file}')
        return db

    def get_sequence(self, start_index, end_index, data):
        if start_index != end_index:
            return data[start_index:end_index+1]
        else:
            return data[start_index:start_index+1].repeat(self.seqlen, axis=0)

    def get_single_item(self, index):
        start_index, end_index = self.vid_indices[index]

        is_train = self.set == 'train'

        if self.dataset_name == '3dpw':
            kp_2d = convert_kps(self.get_sequence(start_index, end_index, self.db['joints2D']), src='common', dst='spin')
            kp_3d = self.get_sequence(start_index, end_index, self.db['joints3D'])

        elif self.dataset_name == 'mpii3d':
            kp_2d = self.get_sequence(start_index, end_index, self.db['joints2D'])
            if is_train:
                kp_3d = self.get_sequence(start_index, end_index, self.db['joints3D'])

            else:
                kp_3d = convert_kps(self.get_sequence(start_index, end_index, self.db['joints3D']), src='spin', dst='mpii3d_test')
        elif self.dataset_name == 'h36m':
            kp_2d = self.get_sequence(start_index, end_index, self.db['joints2D'])
            if is_train:
                kp_3d = self.get_sequence(start_index, end_index, self.db['joints3D'])
            else:
                kp_3d = convert_kps(self.get_sequence(start_index, end_index, self.db['joints3D']), src='spin', dst='common')

        kp_2d_tensor = np.ones((self.seqlen, 49, 3), dtype=np.float16)
        if is_train:
            nj = 49
        else:
            if self.dataset_name == 'mpii3d':
                nj = 17
            else:
                nj =14

        kp_3d_tensor = np.zeros((self.seqlen, nj, 3), dtype=np.float16)

        if self.dataset_name == '3dpw':
            pose = self.get_sequence(start_index, end_index, self.db['pose'])
            shape = self.get_sequence(start_index, end_index, self.db['shape'])

            w_smpl = torch.ones(self.seqlen).float()
            w_3d = torch.ones(self.seqlen).float()
        elif self.dataset_name == 'h36m':
            if not is_train:
                pose = np.zeros((kp_2d.shape[0], 72))
                shape = np.zeros((kp_2d.shape[0], 10))
                w_smpl = torch.zeros(self.seqlen).float()
                w_3d = torch.ones(self.seqlen).float()
            else:
                pose = self.get_sequence(start_index, end_index, self.db['pose'])
                shape = self.get_sequence(start_index, end_index, self.db['shape'])
                # SMPL parameters obtained by NeuralAnnot is released now! - 06/17/2022
                w_smpl = torch.ones(self.seqlen).float()
                if self.load_opt == 'repr_table4_3dpw_model':
                    w_smpl = torch.zeros(self.seqlen).float()
                w_3d = torch.ones(self.seqlen).float()
        elif self.dataset_name == 'mpii3d':
            pose = np.zeros((kp_2d.shape[0], 72))
            shape = np.zeros((kp_2d.shape[0], 10))
            w_smpl = torch.zeros(self.seqlen).float()
            w_3d = torch.ones(self.seqlen).float()

        bbox = self.get_sequence(start_index, end_index, self.db['bbox'])
        # img_names = self.get_sequence(start_index, end_index, self.db['img_name'])
        # video = torch.cat(
        #     [get_single_image_crop(image, None, bbox, scale=1.2).unsqueeze(0) for idx, (image, bbox) in
        #      enumerate(zip(img_names, bbox))], dim=0
        # )
        input = torch.from_numpy(self.get_sequence(start_index, end_index, self.db['features'])).float()

        theta_tensor = np.zeros((self.seqlen, 85), dtype=np.float16)

        for idx in range(self.seqlen):
            # crop image and transform 2d keypoints
            kp_2d[idx,:,:2], trans = transfrom_keypoints(
                kp_2d=kp_2d[idx,:,:2],
                center_x=bbox[idx,0],
                center_y=bbox[idx,1],
                width=bbox[idx,2],
                height=bbox[idx,3],
                patch_width=224,
                patch_height=224,
                do_augment=False,
            )

            kp_2d[idx,:,:2] = normalize_2d_kp(kp_2d[idx,:,:2], 224)

            # theta shape (85,)
            theta = np.concatenate((np.array([1., 0., 0.]), pose[idx], shape[idx]), axis=0)

            kp_2d_tensor[idx] = kp_2d[idx]
            theta_tensor[idx] = theta
            kp_3d_tensor[idx] = kp_3d[idx]

        # (N-2)xnjx3
        # accel_gt = kp_3d_tensor[:-2] - 2 * kp_3d_tensor[1:-1] + kp_3d_tensor[2:]
        # accel_gt = np.linalg.norm(accel_gt, axis=2) # (N-2)xnj

        repeat_num = 3
        target = {
            'features': input,
            'theta': torch.from_numpy(theta_tensor).float()[self.mid_frame].repeat(repeat_num, 1), # camera, pose and shape
            'kp_2d': torch.from_numpy(kp_2d_tensor).float()[self.mid_frame].repeat(repeat_num, 1, 1), # 2D keypoints transformed according to bbox cropping
            'kp_3d': torch.from_numpy(kp_3d_tensor).float()[self.mid_frame].repeat(repeat_num, 1, 1), # 3D keypoints
            'w_smpl': w_smpl[self.mid_frame].repeat(repeat_num),
            'w_3d': w_3d[self.mid_frame].repeat(repeat_num),
        }

        if self.dataset_name == 'mpii3d' and not is_train:
            target['valid'] = self.get_sequence(start_index, end_index, self.db['valid_i'])[self.mid_frame]
            target['theta'] = target['theta'][0]
            target['kp_2d'] = target['kp_2d'][0]
            target['kp_3d'] = target['kp_3d'][0]
            target['w_smpl'] = target['w_smpl'][0]
            target['w_3d'] = target['w_3d'][0]

        if self.dataset_name == 'h36m' and not is_train:
            target['valid'] = np.ones(1, dtype=np.float32)
            target['theta'] = target['theta'][0]
            target['kp_2d'] = target['kp_2d'][0]
            target['kp_3d'] = target['kp_3d'][0]
            target['w_smpl'] = target['w_smpl'][0]
            target['w_3d'] = target['w_3d'][0]

            vn = self.get_sequence(start_index, end_index, self.db['vid_name'])
            fi = self.get_sequence(start_index, end_index, self.db['frame_id'])

            target['instance_id'] = [f'{v}_{f:06d}'.split('/')[-1] for v, f in zip(vn, fi)]
            target['bbox'] = bbox[self.mid_frame]
            target['imgname'] = self.get_sequence(start_index, end_index, self.db['img_name']).tolist()

        if self.dataset_name == '3dpw' and not is_train:
            target['valid'] = np.ones(1, dtype=np.float32)
            target['theta'] = target['theta'][1]
            target['kp_2d'] = target['kp_2d'][1]
            target['kp_3d'] = target['kp_3d'][1]
            target['w_smpl'] = target['w_smpl'][1]
            target['w_3d'] = target['w_3d'][1]

            vn = self.get_sequence(start_index, end_index, self.db['vid_name'])
            fi = self.get_sequence(start_index, end_index, self.db['frame_id'])

            target['instance_id'] = [f'{v}_{f:06d}' for v,f in zip(vn,fi)]
            target['bbox'] = bbox[self.mid_frame]
            target['imgname'] = self.get_sequence(start_index, end_index, self.db['img_name']).tolist()

        if self.debug:
            if self.dataset_name == 'mpii3d':
                video = self.get_sequence(start_index, end_index, self.db['img_name'])
                # print(video)
            elif self.dataset_name == 'h36m':
                video = self.get_sequence(start_index, end_index, self.db['img_name'])
            else:
                vid_name = self.db['vid_name'][start_index]
                vid_name = '_'.join(vid_name.split('_')[:-1])
                f = osp.join(self.folder, 'imageFiles', vid_name)
                video_file_list = [osp.join(f, x) for x in sorted(os.listdir(f)) if x.endswith('.jpg')]
                frame_idxs = self.get_sequence(start_index, end_index, self.db['frame_id'])
                video = [video_file_list[i] for i in frame_idxs]

            video = torch.cat(
                [get_single_image_crop(image, bbox).unsqueeze(0) for image, bbox in zip(video, bbox)], dim=0
            )

            target['video'] = video

        return target





