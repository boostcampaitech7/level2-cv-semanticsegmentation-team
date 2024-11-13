import torch
import cv2
import numpy as np
from model import build_model
from utils import visualize_sample, save_visualization
from config import config

def load_model(checkpoint_path):
    """저장된 모델 체크포인트를 로드하고 초기화"""
    model = build_model(num_classes=len(config.classes))
    checkpoint = torch.load(checkpoint_path, map_location=config.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(config.device)
    model.eval()
    return model

def preprocess_image(image_path):
    """이미지를 전처리하여 모델 입력 형식으로 변환"""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
    image = cv2.resize(image, (512, 512)).transpose(2, 0, 1)
    image_tensor = torch.from_numpy(image).unsqueeze(0).float()
    return image_tensor.to(config.device)

def predict(image_path, model):
    """단일 이미지에 대한 예측 수행"""
    image_tensor = preprocess_image(image_path)
    with torch.no_grad():
        output = model(image_tensor)['out']
        prediction = torch.sigmoid(output).cpu().numpy()[0]
    return prediction

def main():
    # 모델 로드
    checkpoint_path = config.pretrained_dir if config.pretrained else f"{config.saved_dir}/best_model.pth"
    model = load_model(checkpoint_path)

    # 테스트 이미지 경로 설정
    test_image_path = f"{config.test_image_root}/test_image.jpg"
    
    # 예측 수행
    prediction = predict(test_image_path, model)
    
    # 원본 이미지 로드
    original_image = cv2.imread(test_image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB) / 255.0
    original_image = original_image.transpose(2, 0, 1)

    # 시각화 및 저장
    visualize_sample(original_image, mask=None, prediction=prediction)
    save_visualization(
        torch.from_numpy(original_image), None, torch.from_numpy(prediction), 
        f"{config.saved_dir}/inference_result.png"
    )

if __name__ == "__main__":
    main()
