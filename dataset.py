import os
import cv2
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import albumentations as A

class XRayDataset(Dataset):
    def __init__(self, data_dir, fold_file, is_train=True, transforms=None):
        self.data_dir = data_dir
        with open(fold_file, 'r') as f:
            fold_data = json.load(f)
        
        data = fold_data['train'] if is_train else fold_data['validation']
        self.image_paths = [os.path.join(data_dir, item['image_path']) for item in data]
        self.json_paths = [os.path.join(data_dir, item['json_path']) for item in data]
        self.is_train = is_train
        self.transforms = transforms

        # 클래스 정보 (필요 시 수정)
        self.CLASSES = [
            'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
            'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
            'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
            'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
            'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
            'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
        ]
        self.CLASS2IND = {v: i for i, v in enumerate(self.CLASSES)}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 이미지 로드 및 정규화
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
        
        # JSON 파일에서 주석 로드
        with open(self.json_paths[idx], 'r') as f:
            annotations = json.load(f)['annotations']

        # 레이블 배열 생성 (이미지 크기에 맞게 동적으로 생성)
        label_shape = (image.shape[0], image.shape[1], len(self.CLASSES))
        label = np.zeros(label_shape, dtype=np.uint8)

        # 주석을 순회하며 레이블 생성
        for ann in annotations:
            c = ann['label']
            
            # 클래스 인덱스 가져오기 (유효하지 않으면 건너뜀)
            if c not in self.CLASS2IND:
                print(f"Warning: Unknown class label '{c}'")
                continue
            
            class_idx = self.CLASS2IND[c]
            points = np.array(ann['points'], dtype=np.int32)
            
            # 마스크 생성 및 레이블 배열에 추가
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            cv2.fillPoly(mask, [points], 1)
            
            try:
                label[..., class_idx] = mask
            except IndexError:
                print(f"IndexError: class_idx {class_idx} is out of bounds for label shape {label.shape}")
        
        # 데이터 증강 적용
        if self.transforms:
            transformed = self.transforms(image=image, mask=label)
            image, label = transformed['image'], transformed['mask']
        
        # 채널 순서 변경 및 텐서 변환
        image = image.transpose(2, 0, 1)
        label = label.transpose(2, 0, 1)

        return torch.from_numpy(image).float(), torch.from_numpy(label).float()

def create_dataloader(data_dir, fold_file, is_train=True, batch_size=16, num_workers=4):
    transforms = A.Compose([
        A.Resize(512, 512),
        A.HorizontalFlip(p=0.5) if is_train else A.NoOp(),
        A.RandomBrightnessContrast(p=0.2) if is_train else A.NoOp(),
    ])
    
    dataset = XRayDataset(data_dir, fold_file, is_train, transforms)
    return DataLoader(dataset, batch_size=batch_size, shuffle=is_train, num_workers=num_workers)
