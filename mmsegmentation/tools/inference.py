import argparse
import os.path as osp
from mmengine.config import Config
from mmseg.apis import init_model
from mmengine.runner import Runner
import logging
import pandas as pd
import os
import numpy as np

def get_sorted_files_by_type(path,file_type='json'):
    current_dir = os.getcwd()  # 현재 작업 디렉토리 기준으로 상대 경로 생성
    files = {
        os.path.relpath(os.path.join(root, fname), start=current_dir)
        for root, _dirs, files in os.walk(path)
        for fname in files
        if os.path.splitext(fname)[1].lower() == '.' + file_type
    }
    files = sorted(files)

    return files

def set_xraydataset(config):
    TEST_DATA_DIR = '/data/ephemeral/home/data/test'

    image_root = os.path.join(TEST_DATA_DIR, 'DCM')
    pngs = get_sorted_files_by_type(image_root, 'png')

    config.test_dataloader.dataset.image_files = np.array(pngs)
    config.test_dataloader.dataset.label_files = None

    return config

# Argument parser 설정
parser = argparse.ArgumentParser(description="MMsegmentation Runner")
parser.add_argument('--config', type=str, required=True, help='Path to the config file.')
parser.add_argument('--checkpoint', type=str, required=True, help='Path to the checkpoint file.')
parser.add_argument('--work-dir', type=str, default=None, help='Path to the working directory.')

args = parser.parse_args()

# Logger 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Config 파일 로드
cfg = Config.fromfile(args.config)
cfg = set_xraydataset(cfg)

# Work directory 설정
if args.work_dir is not None:
    cfg.work_dir = args.work_dir  # CLI로 전달된 work_dir 사용
elif cfg.get('work_dir', None) is None:
    # config 파일 이름을 기본 work_dir로 사용
    cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])
else:
    # Config 파일에 정의된 work_dir 사용
    cfg.work_dir = cfg.work_dir

logger.info(f"Work directory: {cfg.work_dir}")

# 모델 초기화
model = init_model(cfg, args.checkpoint)
logger.info("Model loaded successfully.")

# Runner 생성 및 테스트 실행
runner = Runner.from_cfg(cfg)
logger.info("Runner initialized. Starting test...")
runner.test()
