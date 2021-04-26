# The code is largely borrowd from https://github.com/mkocabas/VIBE
# Adhere to their licence to use this script
import os
import os.path as osp
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import torch
import pprint
import random
import numpy as np
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from lib.core.loss import TCMRLoss
from lib.core.trainer import Trainer
from lib.core.config import parse_args, BASE_DATA_DIR
from lib.utils.utils import prepare_output_dir
from lib.models import TCMR, MotionDiscriminator
from lib.dataset._loaders import get_data_loaders
from lib.utils.utils import create_logger, get_optimizer


def main(cfg):
    if cfg.SEED_VALUE >= 0:
        print(f'Seed value for the experiment {cfg.SEED_VALUE}')
        os.environ['PYTHONHASHSEED'] = str(cfg.SEED_VALUE)
        random.seed(cfg.SEED_VALUE)
        torch.manual_seed(cfg.SEED_VALUE)
        np.random.seed(cfg.SEED_VALUE)

    logger = create_logger(cfg.LOGDIR, phase='train')

    logger.info(f'GPU name -> {torch.cuda.get_device_name()}')
    logger.info(f'GPU feat -> {torch.cuda.get_device_properties("cuda")}')

    logger.info(pprint.pformat(cfg))

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    writer = SummaryWriter(log_dir=cfg.LOGDIR)
    writer.add_text('config', pprint.pformat(cfg), 0)

    # ========= Dataloaders ========= #
    data_loaders = get_data_loaders(cfg)

    # ========= Compile Loss ========= #
    loss = TCMRLoss(
        e_loss_weight=cfg.LOSS.KP_2D_W,
        e_3d_loss_weight=cfg.LOSS.KP_3D_W,
        e_pose_loss_weight=cfg.LOSS.POSE_W,
        e_shape_loss_weight=cfg.LOSS.SHAPE_W,
        d_motion_loss_weight=cfg.LOSS.D_MOTION_LOSS_W,
    )

    # ========= Initialize networks, optimizers and lr_schedulers ========= #
    generator = TCMR(
        n_layers=cfg.MODEL.TGRU.NUM_LAYERS,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        seqlen=cfg.DATASET.SEQLEN,
        hidden_size=cfg.MODEL.TGRU.HIDDEN_SIZE,
        pretrained=cfg.TRAIN.PRETRAINED_REGRESSOR
    ).to(cfg.DEVICE)

    gen_optimizer = get_optimizer(
        model=generator,
        optim_type=cfg.TRAIN.GEN_OPTIM,
        lr=cfg.TRAIN.GEN_LR,
        weight_decay=cfg.TRAIN.GEN_WD,
        momentum=cfg.TRAIN.GEN_MOMENTUM,
    )

    motion_discriminator = MotionDiscriminator(
        rnn_size=cfg.TRAIN.MOT_DISCR.HIDDEN_SIZE,
        input_size=69,
        num_layers=cfg.TRAIN.MOT_DISCR.NUM_LAYERS,
        output_size=1,
        feature_pool=cfg.TRAIN.MOT_DISCR.FEATURE_POOL,
        attention_size=None if cfg.TRAIN.MOT_DISCR.FEATURE_POOL !='attention' else cfg.TRAIN.MOT_DISCR.ATT.SIZE,
        attention_layers=None if cfg.TRAIN.MOT_DISCR.FEATURE_POOL !='attention' else cfg.TRAIN.MOT_DISCR.ATT.LAYERS,
        attention_dropout=None if cfg.TRAIN.MOT_DISCR.FEATURE_POOL !='attention' else cfg.TRAIN.MOT_DISCR.ATT.DROPOUT
    ).to(cfg.DEVICE)
    
    dis_motion_optimizer = get_optimizer(
        model=motion_discriminator,
        optim_type=cfg.TRAIN.MOT_DISCR.OPTIM,
        lr=cfg.TRAIN.MOT_DISCR.LR,
        weight_decay=cfg.TRAIN.MOT_DISCR.WD,
        momentum=cfg.TRAIN.MOT_DISCR.MOMENTUM
    )
    
    motion_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        dis_motion_optimizer,
        mode='min',
        factor=0.1,
        patience=cfg.TRAIN.LR_PATIENCE,
        verbose=True,
    )

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        gen_optimizer,
        mode='min',
        factor=0.1,
        patience=cfg.TRAIN.LR_PATIENCE,
        verbose=True,
    )

    # ========= Start Training ========= #
    Trainer(
        data_loaders=data_loaders,
        generator=generator,
        motion_discriminator=motion_discriminator,
        criterion=loss,
        dis_motion_optimizer=dis_motion_optimizer,
        dis_motion_update_steps=cfg.TRAIN.MOT_DISCR.UPDATE_STEPS,
        gen_optimizer=gen_optimizer,
        start_epoch=cfg.TRAIN.START_EPOCH,
        end_epoch=cfg.TRAIN.END_EPOCH,
        device=cfg.DEVICE,
        writer=writer,
        debug=cfg.DEBUG,
        logdir=cfg.LOGDIR,
        lr_scheduler=lr_scheduler,
        motion_lr_scheduler=motion_lr_scheduler,
        resume=cfg.TRAIN.RESUME,
        num_iters_per_epoch=cfg.TRAIN.NUM_ITERS_PER_EPOCH,
        debug_freq=cfg.DEBUG_FREQ,
    ).fit()


if __name__ == '__main__':
    cfg, cfg_file, _ = parse_args()
    cfg = prepare_output_dir(cfg, cfg_file)

    main(cfg)
