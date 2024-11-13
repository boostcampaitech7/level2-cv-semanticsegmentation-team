import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models
import numpy as np
import json
from tqdm import tqdm
from dataset import create_dataloader
from model import build_model
from loss import get_loss_function
from config import config
import wandb

class Trainer:
    def __init__(self):
        """Trainer 클래스 초기화"""
        self.data_dir = config.data_root
        self.fold_dir = config.fold_root
        self.num_epochs = config.num_epochs
        self.batch_size = config.batch_size
        self.learning_rate = config.lr
        self.device = config.device
        self.save_dir = config.saved_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def _init_model(self):
        """모델 초기화 및 설정"""
        model = build_model(num_classes=len(config.classes))
        return model.to(self.device)

    def _train_epoch(self, model, train_loader, criterion, optimizer):
        """한 에폭 동안 학습 수행"""
        model.train()
        total_loss = 0

        with tqdm(train_loader, desc='Training') as pbar:
            for images, masks in pbar:
                images, masks = images.to(self.device), masks.to(self.device)
                optimizer.zero_grad()
                outputs = model(images)['out']

                if outputs.shape != masks.shape:
                    outputs = F.interpolate(outputs, size=masks.shape[2:], mode='bilinear')

                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        return total_loss / len(train_loader)

    def _validate(self, model, val_loader):
        """검증 수행"""
        model.eval()
        dice_scores = []
        with torch.no_grad():
            with tqdm(val_loader, desc='Validation') as pbar:
                for images, masks in pbar:
                    images, masks = images.to(self.device), masks.to(self.device)
                    outputs = model(images)['out']
                    if outputs.shape != masks.shape:
                        outputs = F.interpolate(outputs, size=masks.shape[2:], mode='bilinear')
                    outputs = torch.sigmoid(outputs)
                    outputs = (outputs > 0.5).float()
                    dice_scores.append(self._dice_coef(outputs, masks).cpu())
        dice_scores = torch.cat(dice_scores, 0)
        return torch.mean(dice_scores).item()

    def _dice_coef(self, y_pred, y_true, eps=1e-6):
        """Dice coefficient 계산"""
        y_true_f = y_true.flatten(2)
        y_pred_f = y_pred.flatten(2)
        intersection = torch.sum(y_true_f * y_pred_f, dim=-1)
        return (2. * intersection + eps) / (torch.sum(y_true_f, dim=-1) + torch.sum(y_pred_f, dim=-1) + eps)

    def train_fold(self, fold):
        """특정 fold에 대한 학습 수행"""
        print(f"\nTraining Fold {fold}")
        fold_file = os.path.join(self.fold_dir, f"fold_{fold}.json")
  
        # 데이터 로더 생성
        train_loader = create_dataloader(self.data_dir, fold_file, is_train=True, batch_size=self.batch_size)
        val_loader = create_dataloader(self.data_dir, fold_file, is_train=False, batch_size=config.batch_size_valid)

        # 모델 초기화
        model = self._init_model()
        criterion = get_loss_function()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)

        # WandB 설정
        run = wandb.init(
            project="xray-segmentation",
            name=f"fold_{fold}",
            config={
                "learning_rate": self.learning_rate,
                "epochs": self.num_epochs,
                "batch_size": self.batch_size,
                "fold": fold
            }
        )

        best_dice = 0
        fold_results = {'train_losses': [], 'val_dices': [], 'best_epoch': 0, 'best_dice': 0}

        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")

            # 학습
            train_loss = self._train_epoch(model, train_loader, criterion, optimizer)

            # 검증
            val_dice = self._validate(model, val_loader)

            # 로그 기록
            wandb.log({"train_loss": train_loss, "val_dice": val_dice, "epoch": epoch})
            print(f"Train Loss: {train_loss:.4f}, Val Dice: {val_dice:.4f}")

            # 최고 성능 모델 저장
            if val_dice > best_dice:
                best_dice = val_dice
                fold_results['best_epoch'] = epoch
                fold_results['best_dice'] = val_dice
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_dice': best_dice
                }, os.path.join(self.save_dir, f'best_model_fold_{fold}.pth'))

            fold_results['train_losses'].append(train_loss)
            fold_results['val_dices'].append(val_dice)

        # 결과 저장
        with open(os.path.join(self.save_dir, f'results_fold_{fold}.json'), 'w') as f:
            json.dump(fold_results, f, indent=2)
        wandb.finish()
        return fold_results

    def train_all_folds(self):
        """모든 fold에 대해 학습 수행"""
        all_results = []
        for fold in range(5):  # 5-Fold Cross Validation
            fold_results = self.train_fold(fold)
            all_results.append(fold_results)

        # 전체 성능 요약
        mean_best_dice = np.mean([res['best_dice'] for res in all_results])
        std_best_dice = np.std([res['best_dice'] for res in all_results])
        summary = {
            'mean_best_dice': mean_best_dice,
            'std_best_dice': std_best_dice,
            'fold_results': all_results
        }

        # 전체 결과 저장
        with open(os.path.join(self.save_dir, 'all_results.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nTraining Completed! Mean Best Dice: {mean_best_dice:.4f} ± {std_best_dice:.4f}")

if __name__ == "__main__":
    trainer = Trainer()
    trainer.train_all_folds()
