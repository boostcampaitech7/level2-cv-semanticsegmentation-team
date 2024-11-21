import json
import os
import os.path as osp
import argparse
import logging
from mmengine.config import Config
from mmengine.runner import Runner
import wandb

# 고정값 설정
DEFAULT_CONFIG_PATH = "configs/config.py"
DEFAULT_DATA_ROOT = "/data/ephemeral/home/data"

def load_fold_data(fold_path: str):
    """Load train and validation data from fold JSON file."""
    with open(fold_path, 'r') as f:
        fold_data = json.load(f)
    return fold_data['train'], fold_data['validation']

def set_xraydataset_from_fold(config, fold_path, data_root):
    """Set train and validation datasets from JSON fold file."""
    train_data, val_data = load_fold_data(fold_path)

    # Extract image and label paths for train and validation
    train_images = [osp.join(data_root, data['image_path']) for data in train_data]
    train_labels = [osp.join(data_root, data['json_path']) for data in train_data]
    val_images = [osp.join(data_root, data['image_path']) for data in val_data]
    val_labels = [osp.join(data_root, data['json_path']) for data in val_data]

    # Set the dataset configuration
    config.train_dataloader.dataset.image_files = train_images
    config.train_dataloader.dataset.label_files = train_labels

    config.val_dataloader.dataset.image_files = val_images
    config.val_dataloader.dataset.label_files = val_labels

    return config

def parse_args():
    """CLI arguments for train."""
    parser = argparse.ArgumentParser(description="Train for MMsegmentation")
    parser.add_argument(
        "--fold-path",
        required=True,
        help="Path to the JSON file defining train/validation splits"
    )
    parser.add_argument(
        "--work-dir",
        required=True,
        help="Directory to save logs and models"
    )
    return parser.parse_args()

def train(args):
    """
    Smoke Test: 기본 동작 확인을 위한 빠른 실행.
    
    Args:
        args: CLI arguments containing config path, fold path, and work_dir.
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("Train")

    # Config 파일 로드
    cfg = Config.fromfile(DEFAULT_CONFIG_PATH)
    cfg = set_xraydataset_from_fold(cfg, args.fold_path, DEFAULT_DATA_ROOT)

    cfg.launcher = "none"
    
    # 작업 디렉토리 설정
    cfg.work_dir = args.work_dir

    runner = Runner.from_cfg(cfg)

    # 학습 시작
    logger.info("Starting training...")
    runner.train()

if __name__ == "__main__":
    wandb.login(key="97ae30a3f47da7ce905379af5f6bfc8d2d4dd531")

    # CLI 인자 파싱
    args = parse_args()

    # Smoke Test 실행
    train(args)
