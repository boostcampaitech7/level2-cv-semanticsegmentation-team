# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import logging
import os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.runner import Runner
from mmengine.hooks import Hook
from mmseg.registry import RUNNERS

import mmsegext
from mmseg.models import LOSSES
import torch
import torch.nn as nn
import torch.nn.functional as F
#from mmengine.utils import build_from_cfg
#LOSSES.register_module()(MulticlassDiceLoss)
'''
def parse_args():

    parser = argparse.ArgumentParser(description='Train a segmentor')
    #parser.add_argument('config', help='train config file path',default='/data/ephemeral/home/wind/mmseg-extension/configs/internimage/upernet_internimage_b_512_160k_ade20k.py')
    parser.add_argument('--work-dir', help='the dir to save logs and models',default='wind')
    parser.add_argument(
        '--resume',
        action='store_true',
        default=False,
        help='resume from the latest checkpoint in the work_dir automatically')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file. If the value to '
             'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
             'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
             'Note that the quotation marks are necessary and that no white space '
             'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['config','none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args

@HOOKS.register_module()
class CustomDiceEvaluationHook(Hook):
    def __init__(self, data_loader, interval=1):
        self.data_loader = data_loader
        self.interval = interval

    def after_train_epoch(self, runner):
        if self.every_n_epochs(runner, self.interval):
            model = runner.model
            device = next(model.parameters()).device
            criterion = MulticlassDiceLoss().to(device)
            
            avg_dice = validation(runner.epoch, model, self.data_loader, criterion)
            runner.logger.info(f"Epoch {runner.epoch}, Custom Dice Evaluation:")
            runner.logger.info(f"Average Dice Score: {avg_dice:.4f}")
'''
def main():
    #args = parse_args()

    # load config
    #cfg = Config.fromfile(args.config)
    #cfg =  Config.fromfile('/data/ephemeral/home/wind/mmseg-extension/configs/vit_comer/upernet_vit_comer_small_512_160k_ade20k.py')
    cfg =  Config.fromfile('/data/ephemeral/home/wind/mmseg-extension/configs/internimage/upernet_internimage_b_512_160k_ade20k.py')
    #cfg.launcher = args.launcher
    #if args.cfg_options is not None:
    #    cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    #if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
    #    cfg.work_dir = args.work_dir
    #elif cfg.get('work_dir', None) is None:
    #    # use config filename as default work_dir if cfg.work_dir is None
    #cfg.work_dir = osp.join('./work_dirs',osp.splitext(osp.basename(cfg))[0])
    cfg.work_dir='/data/ephemeral/home/nulcear_launch_detected'
    # enable automatic-mixed-precision training

    CLASSES = [
    'background','finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna' ]
    if 'train_dataloader' in cfg:
        if 'dataset' in cfg.train_dataloader:
            cfg.train_dataloader.dataset.metainfo = {'classes': CLASSES}

    if 'val_dataloader' in cfg:
        if 'dataset' in cfg.val_dataloader:
           cfg.val_dataloader.dataset.metainfo = {'classes': CLASSES}

    if 'test_dataloader' in cfg:
        if 'dataset' in cfg.test_dataloader:
            cfg.test_dataloader.dataset.metainfo = {'classes': CLASSES}

    '''
    if args.amp is True:
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == 'AmpOptimWrapper':
            print_log(
                'AMP training is already enabled in your config.',
                logger='current',
                level=logging.WARNING)
        else:
            assert optim_wrapper == 'OptimWrapper', (
                '`--amp` is only supported when the optimizer wrapper type is '
                f'`OptimWrapper` but got {optim_wrapper}.')
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'

    # resume training
    cfg.resume = args.resume
    '''
    #실험용으로 val100에서 빨리
    #cfg.train_cfg=dict(_scope_='mmseg',
    #max_iters=160000,
    #type='IterBasedTrainLoop',
    #val_interval=100)


    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)
    runner.train()


if __name__ == '__main__':
    main()
