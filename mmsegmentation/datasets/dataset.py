import os
import cv2
import json
import numpy as np
from mmseg.datasets import CustomDataset
from mmseg.registry import DATASETS

@DATASETS.register_module()
class XRayDataset(CustomDataset):
    METAINFO = {
        "classes": [
            'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
            'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
            'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
            'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
            'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
            'Triquetrum', 'Pisiform', 'Radius', 'Ulna'
        ],
        "palette": [
            (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
            (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
            (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30), (165, 42, 42),
            (255, 77, 255), (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157),
            (110, 76, 0), (174, 57, 255), (199, 100, 0), (72, 0, 118), (255, 179, 240),
            (0, 125, 92), (209, 0, 151), (188, 208, 182), (0, 220, 176)
        ]
    }

    def __init__(self, pipeline, img_dir, ann_dir=None, **kwargs):
        """
        img_dir: 이미지를 포함하는 디렉토리 경로.
        ann_dir: JSON 주석 파일이 포함된 디렉토리 경로 (테스트 시 선택 사항).
        """
        super().__init__(
            img_suffix='.png',
            seg_map_suffix='.json',
            reduce_zero_label=False,
            pipeline=pipeline,
            img_dir=img_dir,
            ann_dir=ann_dir,
            **kwargs
        )

    def load_annotations(self, img_dir, ann_dir, img_suffix, seg_map_suffix):
        """데이터셋의 주석을 로드합니다."""
        img_infos = []
        for root, dirs, files in os.walk(img_dir):
            for file in files:
                if not file.endswith(img_suffix):
                    continue
                
                # 이미지 파일 정보
                img_path = os.path.join(root, file)
                img_info = dict(filename=os.path.relpath(img_path, img_dir))
                
                # 레이블 파일 정보 (학습 데이터만 필요)
                if ann_dir:
                    rel_dir = os.path.relpath(root, img_dir)  # ID별 디렉토리
                    label_file = os.path.join(ann_dir, rel_dir, file.replace(img_suffix, seg_map_suffix))
                    if os.path.exists(label_file):
                        img_info['ann'] = dict(seg_map=os.path.relpath(label_file, ann_dir))
                    else:
                        raise FileNotFoundError(f"Annotation file {label_file} not found.")
                
                img_infos.append(img_info)
        return img_infos

    def get_gt_seg_map_by_idx(self, idx):
        """주어진 인덱스에 대한 정답 세그멘테이션 맵을 로드합니다."""
        img_info = self.data_infos[idx]
        if 'ann' not in img_info:
            return None  # 테스트 데이터에는 레이블 없음

        label_path = os.path.join(self.ann_dir, img_info['ann']['seg_map'])
        img_path = os.path.join(self.img_dir, img_info['filename'])

        # 이미지 크기에 맞는 빈 레이블 생성
        img = cv2.imread(img_path)
        img_h, img_w = img.shape[:2]
        label_shape = (img_h, img_w)
        label = np.zeros(label_shape, dtype=np.uint8)

        # JSON 파일 로드
        with open(label_path, "r") as f:
            annotations = json.load(f)["annotations"]

        # 각 클래스에 대한 폴리곤 처리
        for ann in annotations:
            class_name = ann["label"]
            class_ind = self.METAINFO['classes'].index(class_name)
            points = np.array(ann["points"], dtype=np.int32)

            # 폴리곤을 마스크로 변환
            class_label = np.zeros(label_shape, dtype=np.uint8)
            cv2.fillPoly(class_label, [points], 1)
            label[class_label == 1] = class_ind

        return label
