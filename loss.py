import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def forward(self, y_pred, y_true):
        y_pred = torch.sigmoid(y_pred)
        intersection = (y_true * y_pred).sum()
        return 1 - (2. * intersection) / (y_true.sum() + y_pred.sum())

class BCEWithDiceLoss(nn.Module):
    def forward(self, y_pred, y_true):
        bce_loss = F.binary_cross_entropy_with_logits(y_pred, y_true)
        dice_loss = DiceLoss()(y_pred, y_true)
        return bce_loss + dice_loss

def get_loss_function():
    return BCEWithDiceLoss()
