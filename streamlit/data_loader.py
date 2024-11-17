# data_loader.py
import pandas as pd
import numpy as np
import json
import cv2


def rle_to_mask(rle, height, width):
    """RLE (Run-Length Encoding)를 마스크로 변환하는 함수."""
    # 빈 마스크 생성
    mask = np.zeros(height * width, dtype=np.uint8)
    
    # RLE 문자열을 숫자로 변환
    rle_numbers = [int(num) for num in rle.split()]
    rle_pairs = np.array(rle_numbers).reshape(-1, 2)
    
    # RLE 쌍을 사용하여 마스크 생성
    for start, length in rle_pairs:
        start -= 1  # RLE 인덱스는 1부터 시작하므로 조정
        mask[start:start + length] = 1
    
    # 올바르게 reshape (세로가 height, 가로가 width)
    mask = mask.reshape((height, width))
    return mask

def extract_annotations_from_csv(csv_path, image_name, height, width):
    """CSV 파일에서 어노테이션을 추출하고 폴리곤 좌표로 변환."""
    df = pd.read_csv(csv_path)
    df = df[df['image_name'] == image_name]
    
    annotations = []
    for _, row in df.iterrows():
        label = row['class']
        rle = row['rle']
        mask = rle_to_mask(rle, height, width)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            points = contour.reshape(-1, 2).tolist()
            annotations.append({'points': points, 'label': label})
    
    return annotations

def load_json_annotations(json_path):
    """JSON 파일에서 어노테이션 로드."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data.get("annotations", [])
