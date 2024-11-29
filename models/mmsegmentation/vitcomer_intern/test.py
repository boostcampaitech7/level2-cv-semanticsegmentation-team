# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.runner import Runner

import mmsegext

#from mmseg.apis import inference_segmentor

from glob import glob
import pandas as pd
from tqdm import tqdm
import mmcv
import os
from tqdm import tqdm
import numpy as np
from glob import glob
import cv2

CLASSES = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
]
CLASS2IND = {v: i for i, v in enumerate(CLASSES)}
IND2CLASS = {v: k for k, v in CLASS2IND.items()}
def rle_encode(mask):
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def trigger_visualization_hook(cfg, show_dir):
    default_hooks = cfg.default_hooks
    if 'visualization' in default_hooks:
        visualization_hook = default_hooks['visualization']
        # Turn on visualization
        visualization_hook['draw'] = True
        '''
        if args.show:
            visualization_hook['show'] = True
            visualization_hook['wait_time'] = args.wait_time
        '''
        if show_dir:
            visualizer = cfg.visualizer
            visualizer['save_dir'] = show_dir
    else:
        raise RuntimeError(
            'VisualizationHook must be included in default_hooks.'
            'refer to usage '
            '"visualization=dict(type=\'VisualizationHook\')"')

    return cfg

def inference(rles, filename_and_class, thr=0.5):
    classes,classid, filename = zip(*[x.split("_") for x in filename_and_class])
    image_name = [os.path.basename(f) for f in filename]
    df = pd.DataFrame(
        {
            "image_name": image_name,
            "class": classes,
            "rle": rles,
        }
    )
    df.to_csv("outputwhereisbackground.csv",index=False)
    df_filtered = df[df['class'] != 'background']
    df_filtered.to_csv("output_backgroundbyebye.csv", index=False)
def main():
    cfg = Config.fromfile('/data/ephemeral/home/wind/mmseg-extension/configs/internimage/upernet_internimage_b_512_160k_ade20k.py')

    cfg.work_dir='/data/ephemeral/home/nulcear_launch_detected'

    cfg.load_from = '/data/ephemeral/home/nulcear_launch_detected/iter_32000.pth'

    runner = Runner.from_cfg(cfg)
    
    # start testing
    runner.test()


    rles, filename_and_class = runner.test_evaluator.metrics[0].get_results()
    inference(rles, filename_and_class)

if __name__ == '__main__':
    main()
