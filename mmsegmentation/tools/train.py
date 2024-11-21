import json
import os
import os.path as osp
import argparse
import logging
from mmengine.config import Config
from mmengine.runner import Runner
import wandb

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
    """CLI arguments for smoke test."""
    parser = argparse.ArgumentParser(description="Run Smoke Test for MMsegmentation")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the config file"
    )
    parser.add_argument(
        "--fold-path",
        required=True,
        help="Path to the JSON file defining train/validation splits"
    )
    parser.add_argument(
        "--data-root",
        required=True,
        help="Root directory containing the dataset"
    )
    parser.add_argument(
        "--work-dir",
        default=None,
        help="Directory to save logs and models"
    )

    return parser.parse_args()

def run_smoke_test(args):
    """
    Smoke Test: 기본 동작 확인을 위한 빠른 실행.
    
    Args:
        args: CLI arguments containing config path, fold path, and work_dir.
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("SmokeTest")

    # Config 파일 로드
    cfg = Config.fromfile(args.config)
    cfg = set_xraydataset_from_fold(cfg, args.fold_path, args.data_root)
    logger.info(f"Loaded config from {args.config} and fold data from {args.fold_path}")

    cfg.launcher = "none"

    # Set up working dir to save files and logs.
    if args.work_dir is not None:
        # CLI로 전달된 work_dir 사용
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # config 파일 이름을 기본 work_dir로 사용
        cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])
    else:
        # Config 파일에 정의된 work_dir 사용
        cfg.work_dir = cfg.work_dir

    logger.info(f"Work directory: {cfg.work_dir}")

    runner = Runner.from_cfg(cfg)

    # 학습 시작
    logger.info("Starting smoke test training...")
    runner.train()
    logger.info("Smoke test completed successfully.")

if __name__ == "__main__":
    wandb.login(key="97ae30a3f47da7ce905379af5f6bfc8d2d4dd531")

    # CLI 인자 파싱
    args = parse_args()

    # Smoke Test 실행
    run_smoke_test(args)
