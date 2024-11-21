import json
import os
import os.path as osp
import argparse
import logging
import yaml
import wandb
from mmengine.config import Config
from mmengine.runner import Runner
from mmseg.apis import init_model

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
    """CLI arguments for smoke test."""
    parser = argparse.ArgumentParser(description="Run Smoke Test for MMsegmentation")
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

def train_with_wandb_sweep():
    """Train function to be used with WandB sweep."""
    wandb.init()
    
    # CLI 인자 로드
    args = parse_args() 

    # Config 파일 로드
    cfg = Config.fromfile(DEFAULT_CONFIG_PATH)

    # Fold 데이터셋 설정
    cfg = set_xraydataset_from_fold(cfg, args.fold_path, DEFAULT_DATA_ROOT)

    # 작업 디렉토리 설정
    cfg.work_dir = args.work_dir

    # WandB Sweep의 하이퍼파라미터 가져오기
    cfg.optim_wrapper.optimizer.lr = wandb.config.lr
    cfg.optim_wrapper.optimizer.weight_decay = wandb.config.weight_decay

    # Runner 생성
    runner = Runner.from_cfg(cfg)

    # 학습 시작
    runner.train()

if __name__ == "__main__":
    wandb.login(key="97ae30a3f47da7ce905379af5f6bfc8d2d4dd531")

    sweep_configuration = {
        "method": "bayes",
        "metric": {"goal": "maximize", "name": "mDice"},
        "parameters": {
            "lr": {"max": 1e-2, "min": 1e-4, "distribution": "uniform"},
            "weight_decay": {"max": 1e-2, "min": 1e-4, "distribution": "uniform"}
        },
        "early_terminate":{
            "type": "hyperband",
            "s": 3,
            "eta": 2, # half halving or one-third halving. 2 or 3 recommended
            "min_iter": 10,
        }
    }

    sweep_id = wandb.sweep(sweep=sweep_configuration, project="semantic_segmentation_mmseg_segformer")  

    wandb.agent(sweep_id, function=train_with_wandb_sweep, count=5)
