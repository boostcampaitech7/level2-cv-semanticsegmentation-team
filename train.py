# train.py
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

from config.config import Config
from models.model import get_model
from data.dataset import XRayDataset
from utils.utils import set_seed, save_model

def parse_args():
    parser = argparse.ArgumentParser(description='Training script for segmentation')
    
    # Data args
    parser.add_argument('--base_dir', type=str, default='/data/ephemeral/home/data',
                      help='base directory containing data')
    parser.add_argument('--fold_dir', type=str, default='kfold_splits',
                      help='directory containing fold splits')
    parser.add_argument('--fold', type=int, default=None,
                      help='specific fold to train (None for all folds)')
    
    # Model args
    parser.add_argument('--model_name', type=str, default='UnetPlusPlus',
                      help='model architecture')
    parser.add_argument('--encoder', type=str, default='efficientnet-b0',
                      help='encoder backbone')
    parser.add_argument('--encoder_weights', type=str, default='imagenet',
                      help='pretrained weights for encoder')
    
    # Training args
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4)
    
    # Augmentation args
    parser.add_argument('--use_augmentation', action='store_true',
                      help='whether to use data augmentation')
    
    # Wandb args
    parser.add_argument('--wandb_project', type=str, default='xray-segmentation')
    parser.add_argument('--wandb_entity', type=str, default=None)
    parser.add_argument('--no_wandb', action='store_true')
    
    return parser.parse_args()

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
        self.save_dir = "experiments"
        os.makedirs(self.save_dir, exist_ok=True)
        
    def _init_model(self):
        """모델 초기화"""
        model = get_model(self.config)
        return model.to(self.device)
    
    def _get_transforms(self, is_train=True):
        """Get augmentation transforms"""
        if is_train and self.args.use_augmentation:
            return A.Compose([
                A.Resize(512, 512),
                # A.HorizontalFlip(p=0.5),
                # A.VerticalFlip(p=0.5),
                # A.RandomRotate90(p=0.5),
                # A.ShiftScaleRotate(p=0.5),
                # A.RandomBrightnessContrast(p=0.2),
                # A.GaussNoise(p=0.2),
                # A.GridDistortion(p=0.2),
                # A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.2),
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
        
        with torch.no_grad():
            with tqdm(val_loader, desc='Validation') as pbar:
                for images, masks in pbar:
                    images = images.to(self.device)
                    masks = masks.to(self.device)
                    
                    outputs = model(images)['out']
                    
                    if outputs.shape != masks.shape:
                        outputs = F.interpolate(outputs, size=masks.shape[2:], mode='bilinear')
                    
                    outputs = torch.sigmoid(outputs)
                    outputs = (outputs > threshold).float()
                    
                    dice = self._dice_coef(outputs, masks)
                    dice_scores.append(dice.cpu())
        
        dice_scores = torch.cat(dice_scores, 0)
        dice_per_class = torch.mean(dice_scores, 0)
        
        # Print dice scores per class
        print('\nDice scores for each class:')
        for class_name, dice_score in zip(self.config.CLASSES, dice_per_class):
            print(f'{class_name:<12}: {dice_score.item():.4f}')
            
        return torch.mean(dice_per_class).item()
    
    def train_fold(self, fold_num: int, fold_data: dict):
        """특정 fold에 대한 학습 수행"""
        print(f"\nTraining Fold {fold_num}")
        
        # Config 업데이트
        self.config.FOLD = fold_num
        self.config.EXP_NAME = f"{self.config.MODEL_NAME}_{self.config.ENCODER}_fold{fold_num}"
        self.config.SAVED_DIR = os.path.join(self.save_dir, self.config.EXP_NAME)
        os.makedirs(self.config.SAVED_DIR, exist_ok=True)
        
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
        optimizer = optim.AdamW(model.parameters(), lr=self.config.LEARNING_RATE, weight_decay=1e-2)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
        
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
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch+1}/{self.num_epochs}")
            
            # 학습
            train_loss = self._train_epoch(model, train_loader, criterion, optimizer)
            
            # 검증
            val_dice = self._validate(model, val_loader)
            
            # Learning rate 조정
            scheduler.step(val_dice)
            
            # Logging
            if not self.args.no_wandb:
                wandb.log({
                    f'fold{fold_num}/train_loss': train_loss,
                    f'fold{fold_num}/val_dice': val_dice,
                    f'fold{fold_num}/lr': optimizer.param_groups[0]['lr'],
                    'epoch': epoch
                })
            
            print(f"Train Loss: {train_loss:.4f}, Val Dice: {val_dice:.4f}")
            
            # 최고 성능 모델 저장
            if val_dice > best_dice:
                best_dice = val_dice
                save_model(model, self.config.SAVED_DIR, f'best_model_fold{fold_num}.pth')
                print(f"New best dice score: {best_dice:.4f}")
        
        if not self.args.no_wandb:
            wandb.finish()
            
        return best_dice

def main():
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Initialize config
    config = Config(args)
    
    # Create trainer
    trainer = Trainer(config, args, num_epochs=args.num_epochs)
    
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
        for fold_num, score in enumerate(fold_scores):
            print(f"Fold {fold_num} Best Dice: {score:.4f}")
        print(f"Average Dice: {sum(fold_scores)/len(fold_scores):.4f}")
        print(f"Std Dice: {np.std(fold_scores):.4f}")

if __name__ == "__main__":
    main()