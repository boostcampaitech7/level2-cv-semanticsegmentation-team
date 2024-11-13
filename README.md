## Train (학습) 설정

### 1. WandB 설정
WandB를 통해 학습 로그를 추적하려면 `config.yaml`에서 다음 항목을 수정하세요:

### 2. 모델 설정
사전 학습된 모델을 사용하려면 다음을 설정합니다:
```yaml
pretrained: false # 사전 학습된 모델 사용 여부 (True/False) 
pretrained_dir: "/path/to/best_model.pt" # 사전 학습된 모델 체크포인트 경로 (필요 시 수정) 
device: "cuda" # 사용할 디바이스 (cuda 또는 cpu)
```
- **`pretrained`**: `true`로 설정하면 사전 학습된 모델을 사용합니다.
- **`pretrained_dir`**: 사전 학습된 모델 경로를 정확히 입력하세요.
- **`device`**: GPU를 사용할 경우 `cuda`, 아니면 `cpu`로 설정합니다.

### 3. 학습 하이퍼파라미터 설정
자신의 시스템에 맞게 배치 크기, 학습률, 에포크 수 등을 조정합니다:
```yaml
batch_size: 16 
learning_rate: 0.0001 
num_epochs: 100 
val_every: 5 
random_seed: 42
```

### 5. 데이터 및 경로 설정
자신의 데이터 및 저장 경로를 지정합니다:
```yaml
data_root: "/data/your_path/data" 
image_root: "/data/your_path/data/train/DCM" 
label_root: "/data/your_path/kfold_splits/" 
saved_dir: "/data/your_path/saved_model"
```
- **`data_root`**: 데이터셋의 루트 디렉토리 경로.
- **`image_root`**: 학습 이미지들이 저장된 경로.
- **`label_root`**: KFold JSON 파일들이 저장된 경로.
- **`saved_dir`**: 모델 체크포인트와 결과를 저장할 디렉토리.

---

## Inference (추론) 설정


## **Inference (추론) 설정**

추론을 시작하기 전에 **`config.yaml`**에서 다음 항목들을 수정하세요. 이 섹션에서는 추론 시 필요한 설정을 중점적으로 설명합니다.

---

### **1. 모델 체크포인트 설정**
추론에 사용할 사전 학습된 모델 체크포인트를 지정합니다.
```yaml
pretrained: true                      # 추론 시 사전 학습된 모델 사용 (True로 설정)
pretrained_dir: "/path/to/best_model.pth"  # 사전 학습된 모델 체크포인트 경로
device: "cuda"                        # 사용할 디바이스 (cuda 또는 cpu)
```


### **2. 테스트 데이터 경로 설정**
테스트 이미지가 저장된 경로와 추론 결과를 저장할 경로를 지정합니다.

```yaml
test_image_root: "/data/your_path/data/test/DCM"
saved_dir: "/data/your_path/saved_model"
```
test_image_root: 테스트할 이미지들이 저장된 디렉토리 경로를 입력합니다.
saved_dir: 추론 결과를 저장할 디렉토리 경로를 지정합니다. 추론된 마스크 이미지 및 CSV 파일이 이 경로에 저장됩니다.


### **3,추론 하이퍼파라미터**
추론 시, 예측된 마스크에 적용할 임계값(threshold)을 조정할 수 있습니다. inference.py 내부에서 설정합니다.

```python
코드 복사
threshold = 0.5  # 마스크 생성 시 사용할 임계값
threshold: 모델의 출력값을 0.5 이상으로 설정된 픽셀을 마스크로 인식합니다. 필요에 따라 조정 가능합니다.
```

### **4. 추론 실행 방법**
추론을 위해 inference.py 스크립트를 실행하세요:

```bash
python inference.py
```

추론 결과는 config.yaml에 지정한 saved_dir 경로에 CSV 파일과 마스크 이미지로 저장됩니다.


### **5. Inference 설정 요약**
**pretrained**를 true로 설정하여 체크포인트를 사용합니다.
**pretrained_dir**에서 사전 학습된 모델을 불러옵니다.
**test_image_root**에 테스트 이미지 경로를 지정합니다.
**saved_dir**에 추론 결과를 저장합니다.
필요한 경우, threshold 값을 inference.py에서 조정합니다.
이제 사용자는 config.yaml 파일을 수정하여 손쉽게 추론을 수행할 수 있습니다. 🚀😊