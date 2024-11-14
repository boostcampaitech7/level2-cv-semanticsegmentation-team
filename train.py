import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import datetime
import json
import numpy as np
from tqdm import tqdm
import wandb
import argparse
import albumentations as A
from torch.utils.data import DataLoader
import yaml
from pathlib import Path

from config.config import Config
from models.model import get_model 
from data.dataset import XRayDataset
from utils.utils import set_seed, save_model

def parse_args():
    parser = argparse.ArgumentParser(description='Training script for segmentation')
    
    # 데이터 관련 인자들
    parser.add_argument('--base_dir', type=str, default='/data/ephemeral/home/data',
                      help='학습 데이터가 저장된 기본 디렉토리 경로')
    parser.add_argument('--fold_dir', type=str, default='/data/ephemeral/home/data/kfold_splits',
                      help='K-fold 분할 정보가 저장된 디렉토리 경로')
    parser.add_argument('--fold', type=int, default=None,
                      help='학습할 특정 fold 번호 (None일 경우 모든 fold 학습)')
    
    # 모델 관련 인자들
    parser.add_argument('--model_name', type=str, default='UnetPlusPlus',
                      help='사용할 모델 아키텍처 이름 (예: Unet, UnetPlusPlus, DeepLabV3 등)')
    parser.add_argument('--encoder', type=str, default='efficientnet-b0',
                      help='모델의 인코더 백본 (예: efficientnet-b0, resnet50 등)')
    parser.add_argument('--encoder_weights', type=str, default='imagenet',
                      help='인코더의 사전학습 가중치 (예: imagenet, None 등)')
    
    # 학습 관련 인자들
    parser.add_argument('--batch_size', type=int, default=8,
                      help='배치 크기')
    parser.add_argument('--num_epochs', type=int, default=50,
                      help='총 학습 에폭 수')
    parser.add_argument('--lr', type=float, default=1e-4,
                      help='학습률 (learning rate)')
    parser.add_argument('--seed', type=int, default=42,
                      help='재현성을 위한 랜덤 시드')
    parser.add_argument('--num_workers', type=int, default=4,
                      help='데이터 로딩에 사용할 워커 수')
    parser.add_argument('--patience', type=int, default=7,
                      help='Early Stopping을 위한 인내심 값 (성능 개선이 없을 때 기다릴 에폭 수)')
    parser.add_argument('--min_delta', type=float, default=1e-4,
                      help='성능 개선으로 간주할 최소 변화량')
    
    # 데이터 증강 관련 인자들
    parser.add_argument('--use_augmentation', action='store_true',
                      help='데이터 증강 사용 여부 (True/False)')
    parser.add_argument('--aug_hflip_prob', type=float, default=0.5,
                      help='수평 뒤집기 확률')
    parser.add_argument('--aug_vflip_prob', type=float, default=0.5,
                      help='수직 뒤집기 확률')
    parser.add_argument('--aug_rotate_prob', type=float, default=0.5,
                      help='회전 변환 확률')
    parser.add_argument('--aug_brightness_prob', type=float, default=0.2,
                      help='밝기 조정 확률')
    parser.add_argument('--aug_contrast_prob', type=float, default=0.2,
                      help='대비 조정 확률')
    parser.add_argument('--aug_noise_prob', type=float, default=0.2,
                      help='가우시안 노이즈 추가 확률')
    parser.add_argument('--aug_dropout_prob', type=float, default=0.2,
                      help='Coarse Dropout 적용 확률')
    
    # Wandb 관련 인자들
    parser.add_argument('--wandb_project', type=str, default='xray-segmentation',
                      help='Weights & Biases 프로젝트 이름')
    parser.add_argument('--wandb_entity', type=str, default=None,
                      help='Weights & Biases 엔티티(팀/개인) 이름')
    parser.add_argument('--no_wandb', action='store_true',
                      help='Weights & Biases 로깅 비활성화 여부')
    
    return parser.parse_args()

class EarlyStopping:
    def __init__(self, patience=7, min_delta=1e-4, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_score_min = float('inf')
    
    def __call__(self, val_score):
        if self.best_score is None:
            self.best_score = val_score
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_score
            self.counter = 0

class Trainer:
    def __init__(self, 
                 config,
                 args,
                 num_epochs: int = 50,
                 device: str = 'cuda'):
        self.config = config
        self.args = args
        self.num_epochs = num_epochs
        self.device = device
        
        # 결과 저장 디렉토리 생성
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = Path("experiments") / f"{current_time}_{args.model_name}_{args.encoder}"
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 설정 저장
        self.save_config()
        
    def save_config(self):
        """학습 설정 저장"""
        config = {
            'model': {
                'name': self.args.model_name,
                'encoder': self.args.encoder,
                'encoder_weights': self.args.encoder_weights
            },
            'training': {
                'batch_size': self.args.batch_size,
                'num_epochs': self.args.num_epochs,
                'learning_rate': self.args.lr,
                'seed': self.args.seed,
                'num_workers': self.args.num_workers,
                'early_stopping_patience': self.args.patience,
                'early_stopping_min_delta': self.args.min_delta
            },
            'augmentation': {
                'enabled': self.args.use_augmentation,
                'horizontal_flip_prob': self.args.aug_hflip_prob,
                'vertical_flip_prob': self.args.aug_vflip_prob,
                'rotate_prob': self.args.aug_rotate_prob,
                'brightness_prob': self.args.aug_brightness_prob,
                'contrast_prob': self.args.aug_contrast_prob,
                'noise_prob': self.args.aug_noise_prob,
                'dropout_prob': self.args.aug_dropout_prob
            },
            'data': {
                'base_dir': self.args.base_dir,
                'fold_dir': self.args.fold_dir
            }
        }
        
        with open(self.save_dir / 'config.yaml', 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
    def _init_model(self):
        """모델 초기화"""
        model = get_model(self.config)
        return model.to(self.device)
    
    def _get_transforms(self, is_train=True):
        """Get augmentation transforms"""
        if is_train and self.args.use_augmentation:
            return A.Compose([
                A.Resize(512, 512),
                A.HorizontalFlip(p=self.args.aug_hflip_prob),
                A.VerticalFlip(p=self.args.aug_vflip_prob),
                A.RandomRotate90(p=self.args.aug_rotate_prob),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=self.args.aug_brightness_prob
                ),
                A.GaussNoise(p=self.args.aug_noise_prob),
                A.GridDistortion(p=0.2),
                A.CoarseDropout(
                    max_holes=8,
                    max_height=32,
                    max_width=32,
                    p=self.args.aug_dropout_prob
                ),
            ])
        else:
            return A.Compose([
                A.Resize(512, 512),
            ])
    
    def _dice_coef(self, y_true, y_pred):
        """Dice coefficient 계산"""
        y_true_f = y_true.flatten(2)
        y_pred_f = y_pred.flatten(2)
        intersection = torch.sum(y_true_f * y_pred_f, -1)
        
        eps = 0.0001
        return (2. * intersection + eps) / (torch.sum(y_true_f, -1) + torch.sum(y_pred_f, -1) + eps)
    
    def _train_epoch(self, model, train_loader, criterion, optimizer):
        """한 에폭 학습"""
        model.train()
        total_loss = 0
        
        with tqdm(train_loader, desc='Training') as pbar:
            for images, masks in pbar:
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(images)['out']
                
                # 출력 크기가 다른 경우 보간
                if outputs.shape != masks.shape:
                    outputs = F.interpolate(outputs, size=masks.shape[2:], mode='bilinear')
                
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / len(train_loader)
    
    def _validate(self, model, val_loader, threshold=0.5):
        """검증 수행"""
        model.eval()
        dice_scores = []
        val_loss = 0
        criterion = nn.BCEWithLogitsLoss()
        
        with torch.no_grad():
            with tqdm(val_loader, desc='Validation') as pbar:
                for images, masks in pbar:
                    images = images.to(self.device)
                    masks = masks.to(self.device)
                    
                    outputs = model(images)['out']
                    
                    if outputs.shape != masks.shape:
                        outputs = F.interpolate(outputs, size=masks.shape[2:], mode='bilinear')
                    
                    # Calculate validation loss
                    loss = criterion(outputs, masks)
                    val_loss += loss.item()
                    
                    # Apply sigmoid and threshold
                    outputs = torch.sigmoid(outputs)
                    outputs = (outputs > threshold).float()
                    
                    dice = self._dice_coef(outputs, masks)
                    dice_scores.append(dice.cpu())
                    
                    pbar.set_postfix({'val_loss': f'{loss.item():.4f}'})
        
        dice_scores = torch.cat(dice_scores, 0)
        dice_per_class = torch.mean(dice_scores, 0)
        
        # Print dice scores per class
        print('\nDice scores for each class:')
        for class_name, dice_score in zip(self.config.CLASSES, dice_per_class):
            print(f'{class_name:<12}: {dice_score.item():.4f}')
        
        avg_val_loss = val_loss / len(val_loader)
        print(f'Average validation loss: {avg_val_loss:.4f}')
            
        return torch.mean(dice_per_class).item(), avg_val_loss
    def _init_optimizer_and_scheduler(self, model):
        """모델별 최적화된 optimizer와 scheduler 설정"""
        if self.config.MODEL_NAME in ['Unet', 'UnetPlusPlus', 'MAnet']:
            optimizer = optim.AdamW(
                model.parameters(),
                lr=self.config.LEARNING_RATE,
                weight_decay=1e-2,
                betas=(0.9, 0.999)
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.num_epochs,
                eta_min=self.config.LEARNING_RATE * 1e-2
            )
        
        elif self.config.MODEL_NAME in ['DeepLabV3', 'DeepLabV3Plus']:
            optimizer = optim.AdamW(
                model.parameters(),
                lr=self.config.LEARNING_RATE,
                weight_decay=2e-2,
                betas=(0.9, 0.999)
            )
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.config.LEARNING_RATE,
                epochs=self.num_epochs,
                steps_per_epoch=len(self.train_loader),
                pct_start=0.1
            )
        
        elif self.config.MODEL_NAME in ['FPN', 'PSPNet']:
            optimizer = optim.AdamW(
                model.parameters(),
                lr=self.config.LEARNING_RATE,
                weight_decay=1e-2
            )
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='max',
                factor=0.5,
                patience=5,
                verbose=True
            )
        
        else:  # 기본 설정
            optimizer = optim.AdamW(
                model.parameters(),
                lr=self.config.LEARNING_RATE,
                weight_decay=1e-2
            )
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='max',
                factor=0.5,
                patience=3,
                verbose=True
            )
        
        return optimizer, scheduler
    def train_fold(self, fold_num: int, fold_data: dict):
        """특정 fold에 대한 학습 수행"""
        print(f"\nTraining Fold {fold_num}")
        
        # Config 업데이트
        self.config.FOLD = fold_num
        self.config.EXP_NAME = f"{self.config.MODEL_NAME}_{self.config.ENCODER}_fold{fold_num}"
        fold_dir = self.save_dir / f"fold_{fold_num}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        
        # Transform 준비
        train_transform = self._get_transforms(is_train=True)
        val_transform = self._get_transforms(is_train=False)
        
        # 데이터로더 생성
        train_dataset = XRayDataset(self.config, fold_data, is_train=True, transforms=train_transform)
        valid_dataset = XRayDataset(self.config, fold_data, is_train=False, transforms=val_transform)
        
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            num_workers=self.config.NUM_WORKERS,
            drop_last=True
        )
        
        val_loader = DataLoader(
            dataset=valid_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            drop_last=False
        )
        
        # 모델 초기화
        model = self._init_model()
        criterion = nn.BCEWithLogitsLoss()
        optimizer, scheduler = self._init_optimizer_and_scheduler(model)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
        
        # Early Stopping 초기화
        early_stopping = EarlyStopping(patience=self.args.patience, min_delta=self.args.min_delta)
        
        # wandb 초기화
        if not self.args.no_wandb:
            wandb.init(
                project=self.args.wandb_project,
                name=self.config.EXP_NAME,
                config=vars(self.args),
                group="kfold_training",
                reinit=True
            )
        
        best_dice = 0
        best_loss = float('inf')
        
        # 학습 기록을 위한 log 파일 생성
        log_file = fold_dir / 'training_log.txt'
        
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch+1}/{self.num_epochs}")
            
            # 학습
            train_loss = self._train_epoch(model, train_loader, criterion, optimizer)
            
            # 검증
            val_dice, val_loss = self._validate(model, val_loader)
            
            # Learning rate 조정
            scheduler.step(val_dice)
            current_lr = optimizer.param_groups[0]['lr']
            
            # 로그 저장
            log_entry = f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
            log_entry += f"Val Dice={val_dice:.4f}, LR={current_lr:.6f}\n"
            with open(log_file, 'a') as f:
                f.write(log_entry)
            
            # Logging to wandb
            if not self.args.no_wandb:
                wandb.log({
                    f'fold{fold_num}/train_loss': train_loss,
                    f'fold{fold_num}/val_loss': val_loss,
                    f'fold{fold_num}/val_dice': val_dice,   
                    f'fold{fold_num}/lr': current_lr,
                    'epoch': epoch
                })
            
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}, LR: {current_lr:.6f}")
            
            # 최고 성능 모델 저장 (Dice Score 기준)
            if val_dice > best_dice:
                best_dice = val_dice
                save_path = fold_dir / f'best_dice_model_fold{fold_num}.pth'
                save_model(model, save_path)
                print(f"New best dice score: {best_dice:.4f}, saved to {save_path}")
            
            # 최고 성능 모델 저장 (Loss 기준)
            if val_loss < best_loss:
                best_loss = val_loss
                save_path = fold_dir / f'best_loss_model_fold{fold_num}.pth'
                save_model(model, save_path)
                print(f"New best validation loss: {best_loss:.4f}, saved to {save_path}")
            
            # Early stopping 조건 확인
            early_stopping(val_loss)  # loss 기준으로 early stopping
            if early_stopping.early_stop:
                print("Early stopping triggered")
                break
        
        if not self.args.no_wandb:
            wandb.finish()
            
        # 최종 결과 저장
        results = {
            'best_dice': best_dice,
            'best_loss': best_loss,
            'final_lr': current_lr,
            'epochs_trained': epoch + 1,
            'training_params': {
                'batch_size': self.args.batch_size,
                'initial_lr': self.args.lr,
                'model': self.args.model_name,
                'encoder': self.args.encoder,
                'augmentation_used': self.args.use_augmentation
            }
        }
        
        with open(fold_dir / 'results.json', 'w') as f:
            json.dump(results, f, indent=4)
            
        return best_dice

def main():
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Initialize config
    config = Config(args)
    
    # GPU 사용 가능 여부 확인
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create trainer
    trainer = Trainer(config, args, num_epochs=args.num_epochs, device=device)
    
    if args.fold is not None:
        # Train single fold
        with open(os.path.join(args.fold_dir, f"fold_{args.fold}.json"), 'r') as f:
            fold_data = json.load(f)
        best_dice = trainer.train_fold(args.fold, fold_data)
        print(f"\nFold {args.fold} Best Dice: {best_dice:.4f}")
    else:
        # Train all folds
        fold_scores = []
        for fold_num in range(5):
            with open(os.path.join(args.fold_dir, f"fold_{fold_num}.json"), 'r') as f:
                fold_data = json.load(f)
            best_dice = trainer.train_fold(fold_num, fold_data)
            fold_scores.append(best_dice)
            
        print("\nTraining Completed!")
        print("=" * 50)
        print("\nResults Summary:")
        print("-" * 30)
        for fold_num, score in enumerate(fold_scores):
            print(f"Fold {fold_num} Best Dice: {score:.4f}")
        mean_dice = sum(fold_scores)/len(fold_scores)
        std_dice = np.std(fold_scores)
        print("-" * 30)
        print(f"Average Dice: {mean_dice:.4f}")
        print(f"Std Dice: {std_dice:.4f}")
        
        # Save overall results
        overall_results = {
            'fold_scores': {f'fold_{i}': score for i, score in enumerate(fold_scores)},
            'mean_dice': mean_dice,
            'std_dice': std_dice,
            'training_params': {
                'batch_size': args.batch_size,
                'initial_lr': args.lr,
                'model': args.model_name,
                'encoder': args.encoder,
                'augmentation_used': args.use_augmentation,
                'num_epochs': args.num_epochs
            }
        }
        
        with open(trainer.save_dir / 'overall_results.json', 'w') as f:
            json.dump(overall_results, f, indent=4)

if __name__ == "__main__":
    main()