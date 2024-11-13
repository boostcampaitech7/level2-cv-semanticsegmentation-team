import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
from config import config

def visualize_sample(image, mask, prediction=None):
    """이미지, 실제 마스크, 예측 결과 시각화"""
    image = image.permute(1, 2, 0).cpu().numpy() if isinstance(image, torch.Tensor) else image
    mask = mask.permute(1, 2, 0).cpu().numpy() if isinstance(mask, torch.Tensor) else mask

    # 색상 팔레트를 활용해 마스크 시각화
    def apply_palette(mask):
        colored_mask = np.zeros((*mask.shape[:2], 3), dtype=np.uint8)
        for i, color in enumerate(config.palette):
            colored_mask[mask[..., i] == 1] = color
        return colored_mask

    mask_colored = apply_palette(mask.argmax(axis=-1))

    if prediction is not None:
        prediction = prediction.permute(1, 2, 0).cpu().numpy() if isinstance(prediction, torch.Tensor) else prediction
        prediction_colored = apply_palette(prediction.argmax(axis=-1))
    
    # 시각화
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("Input Image")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(mask_colored)
    plt.title("Ground Truth Mask")
    plt.axis('off')
    
    if prediction is not None:
        plt.subplot(1, 3, 3)
        plt.imshow(prediction_colored)
        plt.title("Predicted Mask")
        plt.axis('off')
    
    plt.show()

def save_visualization(image, mask, prediction, save_path):
    """시각화 결과를 파일로 저장"""
    image = image.permute(1, 2, 0).cpu().numpy() if isinstance(image, torch.Tensor) else image
    mask = mask.permute(1, 2, 0).cpu().numpy() if isinstance(mask, torch.Tensor) else mask
    prediction = prediction.permute(1, 2, 0).cpu().numpy() if isinstance(prediction, torch.Tensor) else prediction

    def apply_palette(mask):
        colored_mask = np.zeros((*mask.shape[:2], 3), dtype=np.uint8)
        for i, color in enumerate(config.palette):
            colored_mask[mask == i] = color
        return colored_mask

    mask_colored = apply_palette(mask.argmax(axis=-1))
    prediction_colored = apply_palette(prediction.argmax(axis=-1))

    # 합쳐서 저장
    result = np.hstack((image, mask_colored, prediction_colored))
    cv2.imwrite(save_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    print(f"Saved visualization to {save_path}")
