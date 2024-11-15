# data/dataset.py

import os
import cv2
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import struct

class XRayDataset(Dataset):
    def __init__(self, 
                 config,
                 fold_data,
                 is_train: bool = True,
                 transforms = None):
        self.config = config
        self.is_train = is_train
        self.transforms = transforms
        
        self.num_classes = len(config.CLASSES)
        
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
        # image = cv2.imread(image_path)
        image =np.load(image_path)
        image = image / 255.
        
        if len(image.shape) == 2:  # (H, W) 형태일 경우
            image = np.stack([image] * 3, axis=-1)  # (H, W, 3) 형태로 확장
        
        if self.is_train or not self.is_train:  # validation에도 레이블 필요
            # Load label
            bin_path = os.path.join(self.config.BASE_DIR, item['json_path'])  # item['json_path'] 'json_path'의 value는 바꿔뒀는데, key는 혹시 몰라서 안바꿈
            # label = self._load_label(json_path, image.shape)
            
            # .bin 파일에서 헤더 읽기
            with open(bin_path, "rb") as f:
                height, width = struct.unpack('ii', f.read(8))
                packed_mask = np.frombuffer(f.read(), dtype=np.uint8)
            
            # 비트 언팩 후 (height, width, num_classes) 형태로 변환
            flat_mask = np.unpackbits(packed_mask)[:height * width * self.num_classes]
            label = flat_mask.reshape((height, width, self.num_classes))
            
            # 이미지와 라벨 크기 일치 여부 확인
            assert image.shape[:2] == (height, width), f"Image shape {image.shape[:2]} and label shape {(height, width)} do not match"
            
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