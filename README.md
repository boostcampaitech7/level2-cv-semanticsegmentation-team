# 🦴 **Hand Bone X-Ray Segmentation**
![image](https://github.com/user-attachments/assets/dede01a4-3a6c-4392-9596-762b8cb085d0)

> **정확한 뼈 분할을 통해 의료 진단, 수술 계획 및 다양한 분야에서 활용 가능!**

---

## 🎯 **프로젝트 개요**

### 📋 **프로젝트 목표**
- 손 뼈 X-Ray 이미지에서 **29개 클래스**를 분할.
- **멀티채널 확률 맵**을 생성하여 의료 데이터 분석의 정확성 향상.

### 📂 **데이터셋**
- **입력 데이터**: 손 뼈 X-Ray 이미지.
- **출력 데이터**: 픽셀별 클래스를 예측한 멀티채널 확률 맵 (RLE 형식으로 제출).

### 🏆 **팀 구성**
- 김건우, 김범조, 김석현, 임홍철, 정수현, 조소윤

### 🔑 **주요 기여**
- SMP UNet++ 및 다양한 segmentation 모델 학습.
- 데이터 전처리 최적화 (JSON to Binary, PNG to NPY).
- 앙상블 전략(Soft Voting, Hard Voting)으로 성능 향상.

---

## 🔧 주요 기능

### 1️⃣ **모델 성능 개선**
- **Loss Function Experiments**:
  - Focal Loss: 작은 클래스나 불균형한 데이터에서 성능 개선.
  - Combo Loss: Dice와 BCE를 결합하여 분할 경계 및 클래스 예측 성능 향상.

- **Ensemble Strategies**:
  - Soft Voting: 다양한 해상도와 손실 함수를 사용한 모델 결과 결합.
  - Hard Voting: 다수결 방식을 사용하여 안정적인 결과 도출.

### 2️⃣ **데이터 전처리**
- **속도 개선**:
  - JSON → Binary 변환으로 데이터 로드 속도 향상.
  - PNG → NPY 변환으로 데이터 처리 시간 **54.4% 감소**.

- **EDA 및 이상치 처리**:
  - Streamlit 대시보드를 활용한 데이터 및 결과 시각화.
  - 마스킹 오류 이미지와 특이사항 분석 (예: 반지, 네일, 손목 꺾임).

### 3️⃣ **시각화 및 대시보드**
- **Streamlit 앱**:
  - 모델 결과와 데이터 탐색을 직관적으로 확인 가능.
  - 각 모델 예측 결과 비교 시각화. 

---

## 🖼️ **모델 결과물**

| 입력 이미지 | 정답 라벨 | 예측 결과 |
|-------------|-----------|-----------|
| ![Input](https://via.placeholder.com/200x200?text=Input) | ![Ground Truth](https://via.placeholder.com/200x200?text=Ground+Truth) | ![Prediction](https://via.placeholder.com/200x200?text=Prediction) |

---

## 🗂️ **프로젝트 구조**

```plaintext
.
├── models                # 모델 관련 디렉토리
│   ├── mmsegmentation    # MMSegmentation 기반 모델 코드
│   ├── sam2unet          # SAM2UNet 모델 코드
│   └── smp               # SMP 기반 모델 코드
├── notebooks             # Jupyter Notebook 분석 및 실험
│   ├── EDA.ipynb         # 탐색적 데이터 분석 (EDA)
│   └── yolov11.ipynb     # YOLOv11 실험 결과 분석
└── tools                 # 유틸리티 및 시각화 도구
    ├── create_kfolds.py  # K-Folds 생성 스크립트
    ├── streamlit_ClassChecker.py  # 클래스별 성능 확인용 Streamlit 앱
    ├── streamlit_visual.py        # 시각화용 Streamlit 앱
    └── visualization      # 기타 시각화 도구
```

## 📋 구조 설명

### 1️⃣ **models/**
모델 아키텍처 및 학습 코드가 포함된 디렉토리입니다.
- **`mmsegmentation/`**: MMSegmentation을 활용한 모델 구현.
- **`sam2unet/`**: SAM2 백본과 U-Net 디코더를 결합한 모델.
- **`smp/`**: Semantic Segmentation 모델(SMP 기반) 관련 코드.

### 2️⃣ **notebooks/**
Jupyter Notebook을 활용한 데이터 분석 및 실험 결과 기록.
- **`EDA.ipynb`**: 데이터셋의 특징과 문제점 분석.
- **`yolov11.ipynb`**: YOLOv11 실험 결과와 성능 분석.

### 3️⃣ **tools/**
프로젝트 유틸리티 및 시각화 스크립트 모음.
- **`create_kfolds.py`**: 데이터셋을 K-Folds로 분할하는 스크립트.
- **`streamlit_ClassChecker.py`**: Streamlit 앱으로 클래스별 분할 결과를 확인.
- **`streamlit_visual.py`**: Streamlit을 활용한 시각화 및 데이터 탐색.
- **`visualization/`**: 추가적인 시각화 도구.



## 📈 실험 결과 요약

| 실험명                  | 주요 방법론                     | 성능 (mDice)  |
|-------------------------|--------------------------------|--------------|
| **손실 함수 실험**       | Combo Loss (Dice + BCE)        | **0.9611**   |
| **데이터 전처리**        | JSON → Binary, PNG → NPY       | **속도 ↑ 54.4%** |
| **앙상블 학습**          | Soft Voting, Hard Voting       | **0.9734**   |

---

## 🛠️ 사용 방법

이 리포지토리는 **여러 모델 구현**과 **시각화 도구**를 포함하고 있으며, 각 모델 및 도구 사용 방법은 아래와 같습니다.  
모델 별로 필요한 패키지와 실행 방법이 다르므로, 적합한 설정에 따라 실행하세요.

---

### 1️⃣ **모델 실행**

#### 지원 모델
- **MMSegmentation**: `models/mmsegmentation`
- **SAM2UNet**: `models/sam2unet`
- **SMP (Semantic Segmentation Models)**: `models/smp`

#### 공통 사항
1. **Python 버전**: Python 3.8 이상 권장.
2. **CUDA 지원**: GPU가 필요할 경우 모델별 CUDA 요구사항 확인.
3. **환경 격리**: 각 모델별 독립적인 가상 환경(venv 또는 conda) 사용 권장.

#### 📦 필수 패키지 설치
모델 별 요구사항을 설치합니다.
```bash
# MMSegmentation
pip install -r models/mmsegmentation/requirements.txt

# SAM2UNet
pip install -r models/sam2unet/requirements.txt

# SMP
pip install -r models/smp/requirements.txt
```

## ▶️ 학습 실행
모델 학습 실행은 다음과 같습니다:

```bash
# MMSegmentation
python models/mmsegmentation/tools/train.py --config models/mmsegmentation/configs/train_config.yaml

# SAM2UNet
python models/sam2unet/train.py --config models/sam2unet/configs/train_config.yaml

# SMP
python models/smp/train.py --config models/smp/configs/train_config.yaml
```
## 시각화 도구
---
### 1️⃣ Streamlit Visual
아래 명령어를 사용해 Streamlit 앱을 실행합니다.
```bash
streamlit run streamlit_visual.py
```
실행 후 비교하고자 하는 두 개의 CSV 파일을 업로드합니다.

결과 해석:
White: 두 CSV 파일이 동일하게 예측한 픽셀.
Red/Green: 두 CSV 파일이 서로 다르게 예측한 픽셀.
!!! 실행 전 코드에서 test/DCM의 경로를 올바르게 설정해야 합니다.

![image](https://github.com/user-attachments/assets/20fb9bef-2f0e-4ff1-aaa8-fd86ab2dd870)
---

### 2️⃣ Class Checker
아래 명령어를 사용해 Streamlit 앱을 실행합니다.
```bash
streamlit run streamlit_ClassChecker.py
```
실행 후 비교하고자 하는 두 개의 CSV 파일을 업로드합니다.
결과 해석:
막대그래프를 통해 클래스별 겹침 정도를 확인할 수 있습니다.
최대 값은 33.3%로, 두 CSV가 거의 동일하게 예측했음을 의미합니다.
특정 클래스를 선택하면 어느 CSV가 더 많이 예측했는지 확인 가능합니다.
![image](https://github.com/user-attachments/assets/8d8240eb-4335-4a45-8e52-a224c718baf6)

---

### 3️⃣ Train/Test Data Visualization
데이터 시각화를 위한 Streamlit 앱 실행:
```bash
streamlit run app.py
```
웹 브라우저가 열리며 다음 기능을 사용할 수 있습니다.

Train Data Visualization: 학습 데이터와 Ground Truth 확인 (EDA 용도).

Test Data Visualization: 테스트 데이터와 예측 결과를 비교 (정확도 확인).
markdown
코드 복사

![image](https://github.com/user-attachments/assets/a5a46b46-e892-4a53-ae92-b68fadf6fa48)
---

### K-Fold 생성
데이터를 K-Folds로 나누는 스크립트를 실행합니다.
```
python tools/create_kfolds.py
```
---

## 📢 프로젝트 정보

이 프로젝트는 **네이버 Boostcamp AI Tech 7기**에서 **Upstage.ai 플랫폼**을 통해 진행되었습니다.


