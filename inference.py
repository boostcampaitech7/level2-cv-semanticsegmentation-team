import os
import cv2
import torch
import numpy as np
import pandas as pd
import torch.nn as nn  
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import albumentations as A
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

class XRayInferenceDataset(Dataset):
    def __init__(self, image_root: str, transforms=None):
        """
        Args:
            image_root: 테스트 이미지가 있는 루트 디렉토리
            transforms: 적용할 전처리 변환
        """
        self.image_root = image_root
        self.transforms = transforms
        
        # 이미지 파일 목록 수집
        self.filenames = []
        for root, _, files in os.walk(image_root):
            for fname in files:
                if fname.endswith('.png'):
                    rel_path = os.path.relpath(os.path.join(root, fname), start=image_root)
                    self.filenames.append(rel_path)
        self.filenames.sort()
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        image_name = self.filenames[idx]
        image_path = os.path.join(self.image_root, image_name)
        
        image = cv2.imread(image_path)
        image = image / 255.
        
        if self.transforms is not None:
            result = self.transforms(image=image)
            image = result["image"]

        # channel first format
        image = image.transpose(2, 0, 1)
        image = torch.from_numpy(image).float()
            
        return image, image_name

class XRayInference:
    def __init__(self, 
                 model_path: str,
                 test_dir: str = "test/DCM",
                 device: str = 'cuda',
                 batch_size: int = 2,
                 output_dir: str = "inference_results"):
        """
        Args:
            model_path: 학습된 모델의 체크포인트 경로
            test_dir: 테스트 이미지 디렉토리
            device: 추론에 사용할 디바이스
            batch_size: 배치 크기
            output_dir: 결과물 저장 디렉토리
        """
        self.test_dir = test_dir
        self.device = device
        self.batch_size = batch_size
        self.output_dir = output_dir
        
        # 결과물 저장 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "visualizations"), exist_ok=True)
        
        # 클래스 정보
        self.CLASSES = [
            'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
            'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
            'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
            'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
            'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
            'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
        ]
        self.CLASS2IND = {v: i for i, v in enumerate(self.CLASSES)}
        self.IND2CLASS = {v: k for k, v in self.CLASS2IND.items()}
        
        # 시각화를 위한 색상 팔레트
        self.PALETTE = [
            (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
            (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
            (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30), (165, 42, 42),
            (255, 77, 255), (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157),
            (110, 76, 0), (174, 57, 255), (199, 100, 0), (72, 0, 118), (255, 179, 240),
            (0, 125, 92), (209, 0, 151), (188, 208, 182), (0, 220, 176),
        ]
        
        # 모델 로드
        self.model = self._load_model(model_path)
        
    def _load_model(self, model_path):
        """체크포인트에서 모델 로드"""
        # baseline_code.ipynb와 동일한 방식으로 모델 로드
        checkpoint = torch.load(model_path)
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                # train.py에서 저장한 형식
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
        else:
            # 전체 모델이 저장된 경우
            return checkpoint.to(self.device)
            
        # baseline과 동일한 방식으로 모델 생성
        model = models.segmentation.fcn_resnet50(pretrained=True)
        model.classifier[4] = nn.Conv2d(512, len(self.CLASSES), kernel_size=1)
        
        try:
            # 저장된 가중치 로드
            model.load_state_dict(state_dict)
        except Exception as e:
            print(f"Warning: Failed to load state_dict exactly. Error: {e}")
            print("Trying to load only the classifier weights...")
            
            # classifier의 가중치만 로드
            classifier_state_dict = {
                k: v for k, v in state_dict.items() 
                if k.startswith('classifier')
            }
            model.load_state_dict(classifier_state_dict, strict=False)
            
        return model.to(self.device)
    
    def _encode_mask_to_rle(self, mask):
        """마스크를 RLE 포맷으로 인코딩"""
        pixels = mask.flatten()
        pixels = np.concatenate([[0], pixels, [0]])
        runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
        runs[1::2] -= runs[::2]
        return ' '.join(str(x) for x in runs)
    
    def _decode_rle_to_mask(self, rle, height, width):
        """RLE를 마스크로 디코딩"""
        s = rle.split()
        starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
        starts -= 1
        ends = starts + lengths
        img = np.zeros(height * width, dtype=np.uint8)
        
        for lo, hi in zip(starts, ends):
            img[lo:hi] = 1
        
        return img.reshape(height, width)
    
    def _label2rgb(self, label):
        """레이블을 RGB 이미지로 변환"""
        image_size = label.shape[1:] + (3, )
        image = np.zeros(image_size, dtype=np.uint8)
        
        for i, class_label in enumerate(label):
            image[class_label == 1] = self.PALETTE[i]
            
        return image
    
    def predict(self, threshold=0.5):
        """테스트 데이터에 대한 추론 수행"""
        transforms = A.Compose([A.Resize(512, 512)])
        
        # 데이터셋과 로더 설정
        dataset = XRayInferenceDataset(self.test_dir, transforms=transforms)
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            drop_last=False
        )
        
        self.model.eval()
        rles = []
        filename_and_class = []
        
        with torch.no_grad():
            for images, image_names in tqdm(data_loader):
                images = images.to(self.device)
                outputs = self.model(images)['out']  # aux_classifier 무시
                
                # 원본 크기로 리사이즈
                outputs = F.interpolate(outputs, size=(2048, 2048), mode="bilinear")
                outputs = torch.sigmoid(outputs)
                outputs = (outputs > threshold).detach().cpu().numpy()
                
                for output, image_name in zip(outputs, image_names):
                    for c, segm in enumerate(output):
                        rle = self._encode_mask_to_rle(segm)
                        rles.append(rle)
                        filename_and_class.append(f"{self.IND2CLASS[c]}_{image_name}")
        
        return rles, filename_and_class
    
    def save_predictions(self, rles, filename_and_class, output_name=None):
        """예측 결과를 CSV 파일로 저장
        
        Args:
            rles: RLE 인코딩된 마스크 리스트
            filename_and_class: 파일명과 클래스 정보 리스트
            output_name: 저장할 CSV 파일명 (없으면 타임스탬프로 생성)
        """
        if output_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_name = f"predictions_{timestamp}.csv"
            
        output_path = os.path.join(self.output_dir, output_name)
        
        classes, filename = zip(*[x.split("_") for x in filename_and_class])
        image_name = [os.path.basename(f) for f in filename]
        
        df = pd.DataFrame({
            "image_name": image_name,
            "class": classes,
            "rle": rles,
        })
        
        df.to_csv(output_path, index=False)
        print(f"Predictions saved to {output_path}")
        
        return output_path
    
    def visualize_predictions(self, rles, filename_and_class, indices=None, output_prefix=None, show_plot=True):
        """예측 결과 시각화 및 저장
        
        Args:
            rles: RLE 인코딩된 마스크 리스트
            filename_and_class: 파일명과 클래스 정보 리스트
            indices: 시각화할 이미지의 인덱스 리스트 (없으면 전체 시각화)
            output_prefix: 저장될 이미지 파일명의 접두어
            show_plot: matplotlib로 결과를 보여줄지 여부
        """
        if indices is None:
            indices = range(len(filename_and_class) // len(self.CLASSES))
            
        if output_prefix is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_prefix = f"visualization_{timestamp}"
        
        for idx in indices:
            # 원본 이미지 로드
            image_name = filename_and_class[idx * len(self.CLASSES)].split("_", 1)[1]
            image_path = os.path.join(self.test_dir, image_name)
            image = cv2.imread(image_path)
            
            # 마스크 복원
            preds = []
            for rle in rles[idx * len(self.CLASSES):(idx + 1) * len(self.CLASSES)]:
                pred = self._decode_rle_to_mask(rle, height=2048, width=2048)
                preds.append(pred)
            preds = np.stack(preds, 0)
            
            # 시각화
            plt.figure(figsize=(24, 12))
            
            plt.subplot(121)
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.title('Original Image')
            plt.axis('off')
            
            plt.subplot(122)
            plt.imshow(self._label2rgb(preds))
            plt.title('Segmentation Result')
            plt.axis('off')
            
            # 결과 저장
            output_filename = f"{output_prefix}_sample_{idx}.png"
            output_path = os.path.join(self.output_dir, "visualizations", output_filename)
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
            print(f"Visualization saved to {output_path}")
            
            if show_plot:
                plt.show()
            plt.close()

if __name__ == "__main__":
    # 예시 사용법
    inferencer = XRayInference(
        model_path="/data/ephemeral/home/data/model_results/best_model_fold_2.pth",
        test_dir="/data/ephemeral/home/data/test/DCM",
        output_dir="/data/ephemeral/home/data/inference_results_fold_2"
    )
    
    # 추론 실행
    rles, filename_and_class = inferencer.predict()
    
    # 결과 저장
    csv_path = inferencer.save_predictions(
        rles, 
        filename_and_class,
        output_name="/data/ephemeral/home/data/fold_2_predictions.csv"
    )
    
    # 결과 시각화 (첫 5개 샘플)
    inferencer.visualize_predictions(
        rles,
        filename_and_class,
        indices=range(5),
        output_prefix="sample_visualization",
        show_plot=True
    )