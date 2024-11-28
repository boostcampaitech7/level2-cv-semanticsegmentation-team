from ultralytics import YOLO
import argparse
from ultralytics.models.yolo.segment.train import SegmentationTrainer
from ultralytics.utils.metrics import SegmentMetrics
import torch
import torch.nn as nn
import numpy as np
import time

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
        
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred = pred.view(-1)
        target = target.view(-1)
        
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        
        return 1 - dice

class CustomValidator:
    def __init__(self, dataloader, save_dir, args):
        self.dataloader = dataloader
        self.save_dir = save_dir
        self.args = args
        self.metrics = SegmentMetrics()
        
    def __call__(self, trainer):
        model = trainer.model
        device = trainer.device
        
        model.eval()
        dice_scores = []
        
        for batch_i, batch in enumerate(self.dataloader):
            images = batch['img'].to(device)
            masks = batch['masks'].to(device)
            
            with torch.no_grad():
                preds = model(images)
                pred_masks = torch.sigmoid(preds[0])
                
                # Calculate Dice score
                for pred, target in zip(pred_masks, masks):
                    intersection = (pred * target).sum((1, 2))
                    total = pred.sum((1, 2)) + target.sum((1, 2))
                    dice = (2. * intersection + 1e-6) / (total + 1e-6)
                    dice_scores.append(dice.mean().item())
        
        # Calculate and print mean Dice score
        mean_dice = np.mean(dice_scores)
        print(f"\nValidation Dice Score: {mean_dice:.4f}")
        return mean_dice

class CustomSegmentationTrainer(SegmentationTrainer):
    def __init__(self, cfg=None, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)
        self.dice_loss = DiceLoss()
    
    def get_validator(self):
        """Validator 초기화"""
        self.loss_names = ['dice']
        return CustomValidator(self.test_loader, self.save_dir, self.args)

    def validate(self):
        """매 에폭마다 Validation 실행"""
        # Initialize validator
        self.validator = self.get_validator()
        
        # Run validation
        with torch.no_grad():
            self.metrics = self.validator(self)
        
        return self.metrics

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='yolov8x-seg.pt')
    parser.add_argument('--data', type=str, default='config/custom.yaml')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--save-period', type=int, default=1)
    parser.add_argument('--device', default='0')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--project', default='runs/segment')
    parser.add_argument('--name', default='exp')
    parser.add_argument('--exist-ok', action='store_true')
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--optimizer', type=str, default='AdamW')
    parser.add_argument('--lr0', type=float, default=0.001)
    parser.add_argument('--weight-decay', type=float, default=0.05)
    return parser.parse_args()

def main(opt):
    model = YOLO(opt.model)
    model.task = 'segment'
    model.trainer = CustomSegmentationTrainer
    
    train_args = {
        'data': opt.data, 
        'epochs': opt.epochs,
        'batch': opt.batch,
        'imgsz': opt.imgsz,
        'save_period': 1,
        'device': opt.device,
        'workers': opt.workers,
        'project': opt.project,
        'name': opt.name,
        'exist_ok': opt.exist_ok,
        'pretrained': opt.pretrained,
        'optimizer': opt.optimizer,
        'lr0': opt.lr0,
        'weight_decay': opt.weight_decay,
        'val': True,
        'plots': True
    }
    
    model.train(**train_args)

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)