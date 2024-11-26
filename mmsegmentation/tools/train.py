import json
import os
import os.path as osp
import argparse
import logging
import glob
from mmengine.config import Config
from mmengine.runner import Runner
import wandb
from mmengine.runner import load_checkpoint
from mmseg.registry import MODELS

# 고정값 설정
DEFAULT_CONFIG_PATH = "configs/config.py"
DEFAULT_DATA_ROOT = "/data/ephemeral/home/data"
FOLD_JSONS = [
    # "/data/ephemeral/home/data/kfold_splits/fold_1.json",
    "/data/ephemeral/home/data/kfold_splits/fold_2.json",
    "/data/ephemeral/home/data/kfold_splits/fold_3.json"
    "/data/ephemeral/home/data/kfold_splits/fold_4.json",
]
INITIAL_CHECKPOINT = "/data/ephemeral/home/github/mmsegmentation/work_dir/segformer/fold1/best_mDice_epoch_1.pth"

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

def find_best_checkpoint(work_dir, metric="mDice"):
    """
    Find the best checkpoint file based on the metric (e.g., 'mDice').

    Args:
        work_dir (str): Directory where checkpoints are saved.
        metric (str): Metric used to identify the best checkpoint.

    Returns:
        str: Path to the best checkpoint file.
    """
    pattern = osp.join(work_dir, f"best_{metric}_epoch_*.pth")
    checkpoint_files = glob.glob(pattern)
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint matching '{pattern}' found in {work_dir}.")
    
    # Since max_keep_ckpts=1, there should be only one file.
    return checkpoint_files[0]

def parse_args():
    """CLI arguments for train."""
    parser = argparse.ArgumentParser(description="Train for MMsegmentation")
    """ 
    parser.add_argument(
        "--fold-path",
        required=True,
        help="Path to the JSON file defining train/validation splits"
    )
    """
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
    # Config 파일 로드
    cfg = Config.fromfile(DEFAULT_CONFIG_PATH)
    cfg = set_xraydataset_from_fold(cfg, args.fold_path, DEFAULT_DATA_ROOT)

    cfg.launcher = "none"
    
    # 작업 디렉토리 설정
    cfg.work_dir = args.work_dir

    runner = Runner.from_cfg(cfg)

    # 학습 시작
    runner.train()

def train_fold(fold_path, prev_checkpoint, work_dir):
    """
    Train a specific Fold using a previous checkpoint as initialization.
    
    Args:
        fold_path: Path to the Fold JSON file.
        prev_checkpoint: Path to the checkpoint file for initialization.
        work_dir: Directory to save logs and models.
    """
    # Config 파일 로드
    cfg = Config.fromfile(DEFAULT_CONFIG_PATH)
    cfg = set_xraydataset_from_fold(cfg, fold_path, DEFAULT_DATA_ROOT)

    # 이전 체크포인트를 초기 가중치로 사용
    cfg.load_from = prev_checkpoint
    cfg.resume = False  # 초기화 후 학습 시작

    cfg.launcher = "none"
    cfg.work_dir = work_dir

    runner = Runner.from_cfg(cfg)

    # 학습 시작
    runner.train()

     # Fold 학습 완료 후 최적의 모델 저장 경로 반환
    latest_checkpoint = osp.join(work_dir, "last_ckpt.pth")
    return latest_checkpoint

if __name__ == "__main__":
    wandb.login(key="97ae30a3f47da7ce905379af5f6bfc8d2d4dd531")

    # CLI 인자 파싱
    args = parse_args()

    # train 실행
    # train(args)

    # Fold 순차 학습 루프 (Fold 1부터 시작)
    prev_checkpoint = INITIAL_CHECKPOINT  # Fold 0 최적 모델로 시작
    for i, fold_path in enumerate(FOLD_JSONS, start=2):  # Fold 1부터 번호 시작
        print(f"Training Fold {i}...")
        work_dir = osp.join(args.work_dir, f"fold{i}")
        os.makedirs(work_dir, exist_ok=True)  # 작업 디렉토리 생성

        # 이전 Fold에서 최적 checkpoint를 가져와 현재 Fold 학습 실행
        prev_checkpoint = train_fold(fold_path, prev_checkpoint, work_dir)

        """         
        # 이전 Fold에서 최적 checkpoint를 가져와 현재 Fold 학습 실행
        train_fold(fold_path, prev_checkpoint, work_dir)

        # Fold 학습 완료 후 최적 checkpoint 탐색
        try:
            prev_checkpoint = find_best_checkpoint(work_dir, metric="mDice")
        except FileNotFoundError as e:
            print(e)
            break 
        """

        print(f"Completed Fold {i}. Best checkpoint saved at: {prev_checkpoint}")