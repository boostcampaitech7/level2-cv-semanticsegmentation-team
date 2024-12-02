import os
import cv2
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import albumentations as A

class XRayDataset(Dataset):
    def __init__(self, data_dir: str, fold_file: str, is_train: bool = True, transforms = None):
        """
        데이터셋 클래스 초기화
        Args:
            data_dir: 데이터 루트 디렉토리
            fold_file: fold 정보가 저장된 JSON 파일 경로
            is_train: 학습용 데이터셋 여부
            transforms: 적용할 데이터 변환
        """
        self.data_dir = data_dir
        
        with open(fold_file, 'r') as f:
            fold_data = json.load(f)
            
        data = fold_data['train'] if is_train else fold_data['validation']
        
        self.image_paths = [os.path.join(data_dir, item['image_path']) for item in data]
        self.json_paths = [os.path.join(data_dir, item['json_path']) for item in data]
        
        self.is_train = is_train
        self.transforms = transforms
        
        # 클래스 정보 (실제 코드에 맞게 수정 필요)
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
        image = image / 255.
        
        # 레이블 생성
        label_shape = tuple(image.shape[:2]) + (len(self.CLASSES),)
        label = np.zeros(label_shape, dtype=np.uint8)
        
        # JSON 파일에서 레이블 정보 로드
        with open(self.json_paths[idx], 'r') as f:
            annotations = json.load(f)['annotations']
            
        # 각 클래스별 마스크 생성
        for ann in annotations:
            c = ann['label']
            class_ind = self.CLASS2IND[c]
            points = np.array(ann['points'])
            
            class_label = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(class_label, [points], 1)
            label[..., class_ind] = class_label
        
        # 데이터 증강 적용
        if self.transforms:
            inputs = {'image': image, 'mask': label} if self.is_train else {'image': image}
            result = self.transforms(**inputs)
            image = result['image']
            label = result['mask'] if self.is_train else label
        
        # 채널 순서 변경 및 텐서 변환
        image = image.transpose(2, 0, 1)
        label = label.transpose(2, 0, 1)
        
        return torch.from_numpy(image).float(), torch.from_numpy(label).float()

def create_dataloader(data_dir: str, 
                     fold_file: str, 
                     is_train: bool = True, 
                     batch_size: int = 8, 
                     num_workers: int = 4):
    """데이터로더 생성"""
    # 기본 transforms 정의
    transforms = A.Compose([
        A.Resize(512, 512),
        A.HorizontalFlip(p=0.5) if is_train else A.NoOp(),
        A.RandomBrightnessContrast(p=0.2) if is_train else A.NoOp(),
    ])
    
    dataset = XRayDataset(
        data_dir=data_dir,
        fold_file=fold_file,
        is_train=is_train,
        transforms=transforms
    )
    
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        drop_last=is_train,
        pin_memory=True
    )
    
    return dataloader