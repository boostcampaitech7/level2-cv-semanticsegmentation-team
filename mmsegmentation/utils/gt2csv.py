import os
import cv2
import numpy as np
import json
import pandas as pd

CLASSES = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
]
CLASS2IND = {v: i for i, v in enumerate(CLASSES)}
IMAGE_ROOT = "/data/ephemeral/home/data/train/DCM"
LABEL_ROOT = "/data/ephemeral/home/data/train/outputs_json"

def encode_mask_to_rle(mask):
    '''
    mask: numpy array binary mask 
    1 - mask 
    0 - background
    Returns encoded run length 
    '''
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

results = {"image_name": [], "class": [], "rle": []}

# ID 폴더 탐색 및 정렬
all_id_folders = sorted([folder for folder in os.listdir(IMAGE_ROOT) if folder.startswith("ID")])

# 288개 ID 폴더만 선택
selected_id_folders = all_id_folders[:144]
print(len(selected_id_folders))

# ID 폴더 반복
for id_folder in selected_id_folders:
    id_path = os.path.join(IMAGE_ROOT, id_folder)
    if not os.path.isdir(id_path):  # 디렉토리가 아니면 건너뜀
        continue

    # 이미지 파일 탐색
    for image_file in sorted(os.listdir(id_path)):
        if not image_file.endswith(".png"):
            continue  # PNG 파일이 아닌 경우 건너뜀

        image_path = os.path.join(id_path, image_file)
        image_name = f"{image_file}"

        # 레이블 경로 생성
        label_path = os.path.join(LABEL_ROOT, id_folder, image_file.replace(".png", ".json"))

        # 이미지 로드
        image = cv2.imread(image_path)
        if image is None:
            print(f"Image not found: {image_path}")
            continue
        image = image / 255.0

        # 레이블 파일 로드
        if not os.path.exists(label_path):
            print(f"Label file not found: {label_path}")
            continue

        with open(label_path, "r") as f:
            annotations = json.load(f)["annotations"]

        # 각 클래스별로 RLE 생성
        for ann in annotations:
            c = ann["label"]
            class_ind = CLASS2IND[c]
            points = np.array(ann["points"])

            # polygon 포맷을 dense한 mask 포맷으로 변환
            class_label = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(class_label, [points], 1)

            # RLE 인코딩
            rle = encode_mask_to_rle(class_label)
            results["image_name"].append(image_name)
            results["class"].append(c)
            results["rle"].append(rle)

df = pd.DataFrame(results)
df.to_csv("gt_mask.csv", index=False)