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

import sys
sys.path.append('.')

import glob
import torch
import joblib
import argparse
from tqdm import tqdm
import os.path as osp
from skimage import io
from scipy.io import loadmat
import numpy as np
np.seterr('raise')

from lib.models import spin
from lib.data_utils._kp_utils import *
from lib.core.config import TCMR_DB_DIR
from lib.data_utils._img_utils import get_bbox_from_kp2d
from lib.data_utils._feature_extractor import extract_features


def calc_kpt_bound(kp_2d):
    MAX_COORD = 10000
    x = kp_2d[:, 0]
    y = kp_2d[:, 1]
    z = kp_2d[:, 2]
    u = MAX_COORD
    d = -1
    l = MAX_COORD
    r = -1
    for idx, (vis1, vis2) in enumerate(zip(x,y)):
        if vis1 == 0 or vis2 == 0:  # skip invisible joint
            continue
        u = min(u, y[idx])
        d = max(d, y[idx])
        l = min(l, x[idx])
        r = max(r, x[idx])
    return u, d, l, r


def load_mat(path):
    mat = loadmat(path)
    del mat['pose'], mat['__header__'], mat['__globals__'], mat['__version__'], mat['train'], mat['action']
    mat['nframes'] = mat['nframes'][0][0]

    return mat


def read_data(folder):
    dataset = {
        'img_name' : [],
        'joints2D': [],
        'bbox': [],
        'vid_name': [],
        'features': [],
    }

    model = spin.get_pretrained_hmr()

    file_names = sorted(glob.glob(folder + '/labels/'+'*.mat'))

    for ii, fname in enumerate(tqdm(file_names)):
        vid_dict=load_mat(fname)
        imgs = sorted(glob.glob(folder + '/frames/'+ fname.strip().split('/')[-1].split('.')[0]+'/*.jpg'))
        kp_2d = np.zeros((vid_dict['nframes'], 13, 3))
        perm_idxs = get_perm_idxs('pennaction', 'common')

        kp_2d[:, :, 0] = vid_dict['x']
        kp_2d[:, :, 1] = vid_dict['y']
        kp_2d[:, :, 2] = vid_dict['visibility'] # TEMP  1
        kp_2d = kp_2d[:, perm_idxs, :]

        # fix inconsistency
        n_kp_2d = np.zeros((kp_2d.shape[0], 14, 3))
        n_kp_2d[:, :12, :] = kp_2d[:, :-1, :]
        n_kp_2d[:, 13, :] = kp_2d[:, 12, :]
        kp_2d = n_kp_2d

        bbox = np.zeros((vid_dict['nframes'], 4))

        for fr_id, fr in enumerate(kp_2d):
            u, d, l, r = calc_kpt_bound(fr)
            center = np.array([(l + r) * 0.5, (u + d) * 0.5], dtype=np.float32)
            c_x, c_y = center[0], center[1]
            w, h = r - l, d - u

            # if ii > 36:
            #     print("check w, h: ", w, h)
            # try:
            #     w = h = np.where(w / h > 1, w, h)
            # except:
            #     import pdb; pdb.set_trace()

            w = h = np.where(w / h > 1, w, h)
            # w = h = h * 1.1
            bbox[fr_id,:] = np.array([c_x, c_y, w, h])

        dataset['vid_name'].append(np.array([f'{fname}']* vid_dict['nframes']))
        dataset['img_name'].append(np.array(imgs))
        dataset['joints2D'].append(kp_2d)
        dataset['bbox'].append(bbox)

        features = extract_features(model, None, np.array(imgs) , bbox, dataset='pennaction', debug=False, scale=1.2)
        dataset['features'].append(features)

    for k in dataset.keys():
        dataset[k] = np.array(dataset[k])
    for k in dataset.keys():
        dataset[k] = np.concatenate(dataset[k])

    return dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, help='dataset directory', default='data/penn_action')
    args = parser.parse_args()


    dataset = read_data(args.dir)
    joblib.dump(dataset, osp.join(TCMR_DB_DIR, 'pennaction_train_scale12_db.pt'))

