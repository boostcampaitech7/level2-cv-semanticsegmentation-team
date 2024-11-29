import os
import json
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.model_selection import KFold
from typing import List, Dict, Tuple

class XRayKFoldSplitter:
    def __init__(self, 
                 data_dir: str = "/data/ephemeral/home/data",
                 n_splits: int = 10,
                 seed: int = 42):
        self.data_dir = data_dir
        self.image_dir = os.path.join(data_dir, "train/DCM")
        self.label_dir = os.path.join(data_dir, "train/outputs_json")
        self.n_splits = n_splits
        self.seed = seed
        
    def get_image_label_info(self) -> List[Dict]:
        """이미지와 레이블 정보 수집"""
        image_info = []
        for root, _, files in os.walk(self.image_dir):
            for file in files:
                if not file.endswith('.png'):
                    continue
                    
                image_path = os.path.join(root, file)
                rel_path = os.path.relpath(image_path, self.image_dir)
                patient_id = rel_path.split(os.sep)[0]
                
                json_path = os.path.join(self.label_dir, patient_id, 
                                       file.replace('.png', '.json'))
                
                if not os.path.exists(json_path):
                    continue
                
                with open(json_path, 'r') as f:
                    data = json.load(f)
                
                labels = [ann['label'] for ann in data['annotations']]
                
                image_info.append({
                    'image_path': image_path,
                    'json_path': json_path,
                    'patient_id': patient_id,
                    'labels': labels
                })
        
        return image_info
    
    def create_kfold_splits(self, image_info: List[Dict]) -> List[Dict]:
        """K-fold 분할 생성"""
        # 환자 ID 기준으로 분할
        unique_patients = sorted(list(set(info['patient_id'] for info in image_info)))
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
        
        folds = []
        for fold, (train_idx, val_idx) in enumerate(kf.split(unique_patients)):
            train_patients = [unique_patients[i] for i in train_idx]
            val_patients = [unique_patients[i] for i in val_idx]
            
            train_data = [info for info in image_info if info['patient_id'] in train_patients]
            val_data = [info for info in image_info if info['patient_id'] in val_patients]
            
            # 각 fold의 레이블 분포 계산
            train_label_dist = self._count_labels([label for info in train_data for label in info['labels']])
            val_label_dist = self._count_labels([label for info in val_data for label in info['labels']])
            
            folds.append({
                'fold': fold,
                'train': train_data,
                'validation': val_data,
                'train_distribution': train_label_dist,
                'val_distribution': val_label_dist
            })
            
            # Fold 정보 출력
            print(f"\nFold {fold}:")
            print(f"Train samples: {len(train_data)}")
            print(f"Validation samples: {len(val_data)}")
            print("Train label distribution:", train_label_dist)
            print("Validation label distribution:", val_label_dist)
        
        return folds
    
    def _count_labels(self, labels: List[str]) -> Dict[str, int]:
        """레이블 카운트 집계"""
        counts = defaultdict(int)
        for label in labels:
            counts[label] += 1
        return dict(counts)
    
    def save_folds(self, folds: List[Dict], output_dir: str = "kfold_splits"):
        """K-fold 분할 정보 저장"""
        os.makedirs(output_dir, exist_ok=True)
        
        for fold_info in folds:
            fold_num = fold_info['fold']
            
            # 상대 경로로 변환
            train_data = [{
                'image_path': os.path.relpath(info['image_path'], self.data_dir),
                'json_path': os.path.relpath(info['json_path'], self.data_dir),
                'patient_id': info['patient_id'],
                'labels': info['labels']
            } for info in fold_info['train']]
            
            val_data = [{
                'image_path': os.path.relpath(info['image_path'], self.data_dir),
                'json_path': os.path.relpath(info['json_path'], self.data_dir),
                'patient_id': info['patient_id'],
                'labels': info['labels']
            } for info in fold_info['validation']]
            
            fold_data = {
                'fold': fold_num,
                'train': train_data,
                'validation': val_data,
                'train_distribution': fold_info['train_distribution'],
                'val_distribution': fold_info['val_distribution']
            }
            
            output_path = os.path.join(output_dir, f"fold_{fold_num}.json")
            with open(output_path, 'w') as f:
                json.dump(fold_data, f, indent=2)
        
        print(f"\nK-fold 분할 정보가 {output_dir}에 저장되었습니다.")

if __name__ == "__main__":
    # K-fold 분할 실행
    splitter = XRayKFoldSplitter()
    image_info = splitter.get_image_label_info()
    folds = splitter.create_kfold_splits(image_info)
    splitter.save_folds(folds)