# data/dataset.py

import os
import cv2
import json
import numpy as np
import torch
from torch.utils.data import Dataset

class XRayDataset(Dataset):
    def __init__(self, 
                 config,
                 fold_data,
                 is_train: bool = True,
                 transforms = None):
        self.config = config
        self.is_train = is_train
        self.transforms = transforms
        
        # Load fold data
        if isinstance(fold_data, dict):
            if 'train' in fold_data and 'validation' in fold_data:
                self.data = fold_data['train'] if is_train else fold_data['validation']
            else:
                print("Warning: Unexpected fold data format")
                self.data = []
        else:
            print("Warning: fold_data is not a dictionary")
            self.data = []
        
        if not self.data:
            raise ValueError("No data loaded for dataset")
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load image
        image_path = os.path.join(self.config.BASE_DIR, item['image_path'])
        image = cv2.imread(image_path)
        image = image / 255.
        
        if self.is_train or not self.is_train:  # validation에도 레이블 필요
            # Load label
            json_path = os.path.join(self.config.BASE_DIR, item['json_path'])
            label = self._load_label(json_path, image.shape)
            
            if self.transforms:
                transformed = self.transforms(image=image, mask=label)
                image = transformed['image']
                label = transformed['mask']
            
            # Prepare tensors
            image = image.transpose(2, 0, 1)
            label = label.transpose(2, 0, 1)
            
            return torch.from_numpy(image).float(), torch.from_numpy(label).float()
        
        else:  # test
            if self.transforms:
                transformed = self.transforms(image=image)
                image = transformed['image']
            
            image = image.transpose(2, 0, 1)
            return torch.from_numpy(image).float()
    
    def _load_label(self, json_path: str, image_shape: tuple) -> np.ndarray:
        label = np.zeros(image_shape[:2] + (len(self.config.CLASSES),), dtype=np.uint8)
        
        with open(json_path, 'r') as f:
            data = json.load(f)
            annotations = data['annotations']
            
        for ann in annotations:
            class_idx = self.config.CLASS2IND[ann['label']]
            points = np.array(ann['points'])
            
            class_label = np.zeros(image_shape[:2], dtype=np.uint8)
            cv2.fillPoly(class_label, [points], 1)
            label[..., class_idx] = class_label
            
        return label