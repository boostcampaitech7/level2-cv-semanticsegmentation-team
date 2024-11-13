import streamlit as st
import os
from PIL import Image
import numpy as np
from data_loader import extract_annotations_from_csv, load_json_annotations
from utils import draw_annotations
from constants import classes

# Streamlit 앱 타이틀 설정
st.title("Hand Bone Segmentation Viewer")

# 데이터 경로 설정
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# 탭 구성
tab1, tab2, tab3 = st.tabs(["Train Data Visualization", "Test Data Visualization", "Validation Data Visualization"])

# 탭 1: 훈련 데이터 시각화
with tab1:
    st.header("Train Data Visualization")
    
    DCM_DIR = os.path.join(BASE_DIR, "data", "train", "DCM")
    JSON_DIR = os.path.join(BASE_DIR, "data", "train", "outputs_json")
    
    ids = sorted(os.listdir(DCM_DIR))
    selected_id = st.selectbox("Select ID", ids)

    if selected_id:
        image_files = sorted([f for f in os.listdir(os.path.join(DCM_DIR, selected_id)) if f.endswith(".png")])
        selected_image = st.selectbox("Select Image", image_files)
        
        if selected_image:
            image_path = os.path.join(DCM_DIR, selected_id, selected_image)
            json_path = os.path.join(JSON_DIR, selected_id, selected_image.replace(".png", ".json"))
            
            # 이미지 및 JSON 로드
            image = np.array(Image.open(image_path))
            annotations = load_json_annotations(json_path)
            
            # 어노테이션 시각화
            annotated_image = draw_annotations(image, annotations)
            st.image(annotated_image, caption="Ground Truth", use_column_width=True)

# 탭 2: 테스트 데이터 시각화
with tab2:
    st.header("Test Data Visualization")
    
    TEST_IMG_DIR = os.path.join(BASE_DIR, "data", "test", "DCM")
    TEST_CSV_DIR = os.path.join(BASE_DIR, "data", "test", "outputs_csv")
    
    ids = sorted(os.listdir(TEST_IMG_DIR))
    selected_id = st.selectbox("Select Test ID", ids)

    if selected_id:
        image_files = sorted([f for f in os.listdir(os.path.join(TEST_IMG_DIR, selected_id)) if f.endswith(".png")])
        selected_image = st.selectbox("Select Image", image_files)
        csv_path = os.path.join(TEST_CSV_DIR, f"{selected_id}.csv")
        
        if selected_image and os.path.exists(csv_path):
            image_path = os.path.join(TEST_IMG_DIR, selected_id, selected_image)
            image = np.array(Image.open(image_path))
            height, width = image.shape[:2]
            
            # 예측 CSV에서 어노테이션 로드 및 시각화
            predictions = extract_annotations_from_csv(csv_path, selected_image, height, width)
            annotated_image = draw_annotations(image, predictions)
            st.image(annotated_image, caption="Predictions", use_column_width=True)

# 탭 3: 검증 데이터 시각화
with tab3:
    st.header("Validation Data Visualization")
    
    FOLD_IMG_DIR = os.path.join(BASE_DIR, "data", "fold", "DCM")
    FOLD_JSON_DIR = os.path.join(BASE_DIR, "data", "fold", "outputs_json")
    FOLD_CSV_PATH = os.path.join(BASE_DIR, "data", "fold", "submission.csv")
    
    ids = sorted(os.listdir(FOLD_IMG_DIR))
    selected_id = st.selectbox("Select Validation ID", ids)

    if selected_id:
        img_folder = os.path.join(FOLD_IMG_DIR, selected_id)
        json_folder = os.path.join(FOLD_JSON_DIR, selected_id)
        
        image_files = sorted([f for f in os.listdir(img_folder) if f.endswith(".png")])
        selected_image = st.selectbox("Select Image", image_files)

        if selected_image and os.path.exists(FOLD_CSV_PATH):
            image_path = os.path.join(img_folder, selected_image)
            image = np.array(Image.open(image_path))
            height, width = image.shape[:2]
            
            # 정답 JSON 로드
            json_path = os.path.join(json_folder, selected_image.replace(".png", ".json"))
            if os.path.exists(json_path):
                gt_annotations = load_json_annotations(json_path)
                gt_image = draw_annotations(image.copy(), gt_annotations)
            
            # 예측 CSV 로드
            predictions = extract_annotations_from_csv(FOLD_CSV_PATH, selected_image, height, width)
            pred_image = draw_annotations(image.copy(), predictions)
            
            # 정답 및 예측 시각화
            st.subheader("Ground Truth vs Predictions")
            st.image(gt_image, caption="Ground Truth", use_column_width=True)
            st.image(pred_image, caption="Predictions", use_column_width=True)
