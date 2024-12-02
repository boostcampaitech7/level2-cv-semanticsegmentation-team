import os
import cv2
import json
import numpy as np
from PIL import Image
from tqdm import tqdm

def convert_to_coco(data_dir, fold_file, output_file):
    """X-Ray 데이터를 COCO 포맷으로 변환"""
    
    # COCO 포맷 기본 구조
    coco_format = {
        "info": {
            "description": "X-Ray Hand Dataset",
            "version": "1.0",
            "year": 2024,
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    # 카테고리 정보 추가
    CLASSES = [
        'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
        'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
        'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
        'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
        'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
        'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
    ]
    
    for i, category in enumerate(CLASSES):
        coco_format["categories"].append({
            "id": i,
            "name": category,
            "supercategory": "hand"
        })
    
    # 데이터 로드
    with open(fold_file, 'r') as f:
        fold_data = json.load(f)
    
    # 이미지와 어노테이션 변환
    ann_id = 0
    for img_id, item in enumerate(tqdm(fold_data['train'])):
        # 이미지 정보
        img_path = os.path.join(data_dir, item['image_path'])
        img = cv2.imread(img_path)
        height, width = img.shape[:2]
        
        coco_format["images"].append({
            "id": img_id,
            "file_name": item['image_path'],
            "height": height,
            "width": width
        })
        
        # 어노테이션 정보
        json_path = os.path.join(data_dir, item['json_path'])
        with open(json_path, 'r') as f:
            ann_data = json.load(f)
            
        for ann in ann_data['annotations']:
            points = np.array(ann['points']).flatten().tolist()
            x, y, max_x, max_y = np.min(points[::2]), np.min(points[1::2]), np.max(points[::2]), np.max(points[1::2])
            width = max_x - x
            height = max_y - y
            area = width * height
            
            coco_format["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": CLASSES.index(ann['label']),
                "segmentation": [points],
                "area": float(area),
                "bbox": [float(x), float(y), float(width), float(height)],
                "iscrowd": 0
            })
            ann_id += 1
    
    # COCO 포맷 저장
    with open(output_file, 'w') as f:
        json.dump(coco_format, f)

def convert_to_ade20k(data_dir, fold_file, output_dir):
    """X-Ray 데이터를 ADE20K 포맷으로 변환"""
    
    # 출력 디렉토리 생성
    os.makedirs(os.path.join(output_dir, 'images/training'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'annotations/training'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images/validation'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'annotations/validation'), exist_ok=True)
    
    CLASSES = [
        'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
        'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
        'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
        'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
        'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
        'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
    ]
    CLASS2IND = {v: i+1 for i, v in enumerate(CLASSES)}  # ADE20K은 1부터 시작
    
    # 클래스 정보 저장
    with open(os.path.join(output_dir, 'objectInfo150.txt'), 'w') as f:
        for i, cls in enumerate(CLASSES, 1):
            f.write(f"{i}\t{cls}\n")
    
    def process_split(split_data, split_name):
        for idx, item in enumerate(tqdm(split_data)):
            # 이미지 복사
            img_path = os.path.join(data_dir, item['image_path'])
            new_img_path = os.path.join(output_dir, f'images/{split_name}/{idx:08d}.jpg')
            img = cv2.imread(img_path)
            cv2.imwrite(new_img_path, img)
            
            # 마스크 생성
            height, width = img.shape[:2]
            mask = np.zeros((height, width), dtype=np.uint8)
            
            json_path = os.path.join(data_dir, item['json_path'])
            with open(json_path, 'r') as f:
                ann_data = json.load(f)
            
            for ann in ann_data['annotations']:
                points = np.array(ann['points'])
                class_id = CLASS2IND[ann['label']]
                cv2.fillPoly(mask, [points], class_id)
            
            # 마스크 저장
            mask_path = os.path.join(output_dir, f'annotations/{split_name}/{idx:08d}.png')
            cv2.imwrite(mask_path, mask)
    
    # 데이터 로드 및 변환
    with open(fold_file, 'r') as f:
        fold_data = json.load(f)
    
    process_split(fold_data['train'], 'training')
    process_split(fold_data['validation'], 'validation')

if __name__ == "__main__":
    data_dir = "/data/ephemeral/home/data"
    # fold_file = "kfold_splits/fold_0.json"
    fold_file = "/data/ephemeral/home/data/kfold_splits/fold_0.json"
    
    # COCO 포맷으로 변환
    # convert_to_coco(data_dir, fold_file, "annotations_coco.json")
    
    # ADE20K 포맷으로 변환
    convert_to_ade20k(data_dir, fold_file, "ade20k_format")