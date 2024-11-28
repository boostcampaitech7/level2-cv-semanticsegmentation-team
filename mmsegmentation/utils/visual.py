import os
import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt

# RLE 디코딩 함수
def rle_decode(mask_rle, shape):
    """
    RLE 데이터를 2D 바이너리 마스크로 변환
    """
    if pd.isna(mask_rle):  # RLE가 비어있는 경우
        return np.zeros(shape, dtype=np.uint8)
    s = list(map(int, mask_rle.split()))
    starts, lengths = s[0::2], s[1::2]
    starts = np.array(starts) - 1
    ends = starts + lengths
    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for start, end in zip(starts, ends):
        mask[start:end] = 1
    return mask.reshape(shape)

# 색상이 있는 부분을 찾아서 자르는 함수
def get_colored_area_mask(mask):
    """
    마스크에서 색상이 있는 부분(비어있지 않은 부분)만 잘라냄
    """
    non_zero_indices = np.where(mask > 0)
    
    if len(non_zero_indices[0]) == 0 or len(non_zero_indices[1]) == 0:
        # 색상이 있는 부분이 없는 경우
        return None

    min_x, max_x = min(non_zero_indices[0]), max(non_zero_indices[0])
    min_y, max_y = min(non_zero_indices[1]), max(non_zero_indices[1])
    
    return min_x, max_x, min_y, max_y

# 이미지 크롭 함수
def crop_image(image, min_x, max_x, min_y, max_y):
    """
    주어진 좌표에 맞게 이미지를 크롭
    """
    return image[min_x:max_x+1, min_y:max_y+1]

# 두 이미지의 마스크 생성 및 시각화
def visualize_masks_and_images(csv1, csv2, test_image_name, gt_image_name, image_shape, test_dir, train_dir, selected_class=None):
    """
    두 CSV의 특정 테스트 이미지와 GT 이미지 및 클래스에 대해 마스크와 원본 이미지를 생성하고 시각화
    """
    # 선택한 클래스에 따라 필터링
    if selected_class is not None:
        csv1 = csv1[csv1['class'] == selected_class]
        csv2 = csv2[csv2['class'] == selected_class]

    # 공통 클래스 추출
    common_classes = sorted(set(csv1['class']).union(csv2['class']))

    for cls in common_classes:
        rle1 = csv1[(csv1['image_name'] == gt_image_name) & (csv1['class'] == cls)]['rle']
        rle2 = csv2[(csv2['image_name'] == test_image_name) & (csv2['class'] == cls)]['rle']

        # 마스크 생성
        mask1 = rle_decode(rle1.iloc[0] if not rle1.empty else np.nan, image_shape)
        mask2 = rle_decode(rle2.iloc[0] if not rle2.empty else np.nan, image_shape)

        # 두 마스크를 겹치는 방식으로 색칠
        combined_mask = np.zeros((*image_shape, 3), dtype=np.uint8)
        combined_mask[..., 0] = mask2 * 255  # 빨간색 채널 (csv2 - pred)
        combined_mask[..., 1] = mask1 * 255  # 초록색 채널 (csv1 - gt)

        overlap = np.logical_and(mask1, mask2)
        combined_mask[overlap, :] = [255, 255, 255]  # 겹치는 부분은 흰색

        # 색상 있는 부분만 잘라서 확대
        colored_area = get_colored_area_mask(combined_mask)
        if colored_area is None:
            st.warning(f"Class '{cls}'에서 색상이 있는 부분이 없습니다.")
            continue  # 다음 클래스 처리

        # 색상 있는 부분만 잘라서 확대
        min_x, max_x, min_y, max_y = colored_area
        cropped_mask_image = combined_mask[min_x:max_x+1, min_y:max_y+1]

        # DICOM 이미지 로드
        test_image_path = None
        gt_image_path = None

        # test_dir에서 테스트 이미지 찾기
        for root, dirs, files in os.walk(test_dir):
            if test_image_name in files:
                test_image_path = os.path.join(root, test_image_name)
                break

        # train_dir에서 GT 이미지 찾기
        for root, dirs, files in os.walk(train_dir):
            if gt_image_name in files:
                gt_image_path = os.path.join(root, gt_image_name)
                break

        if test_image_path and gt_image_path:
            test_image = Image.open(test_image_path)
            gt_image = Image.open(gt_image_path)

            test_image = np.array(test_image)
            gt_image = np.array(gt_image)

            cropped_test_image = crop_image(test_image, min_x, max_x, min_y, max_y)
            cropped_gt_image = crop_image(gt_image, min_x, max_x, min_y, max_y)

            # 시각화
            st.subheader(f"Class: {cls}")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.image(cropped_mask_image, caption=f"Mask of {test_image_name} & {gt_image_name}, Class: {cls}")
                st.caption("🟢 Green: GT (csv1), 🔴 Red: Pred (csv2), ⚪ White: Overlap")

            with col2:
                st.image(cropped_test_image, caption=f"Test Image: {test_image_name}, Class: {cls}")

            with col3:
                st.image(cropped_gt_image, caption=f"GT Image: {gt_image_name}, Class: {cls}")
        else:
            # 원본 이미지를 찾지 못한 경우 경고 메시지 표시
            if not test_image_path:
                st.warning(f"테스트 이미지 '{test_image_name}'을 {test_dir} 폴더에서 찾을 수 없습니다.")
            if not gt_image_path:
                st.warning(f"GT 이미지 '{gt_image_name}'을 {train_dir} 폴더에서 찾을 수 없습니다.")

            # 마스크 이미지만 표시
            st.subheader(f"Class: {cls} - Mask Only")
            st.image(cropped_mask_image, caption=f"Mask of {test_image_name} & {gt_image_name}, Class: {cls}")
            st.caption("🟢 Green: GT (csv1), 🔴 Red: Pred (csv2), ⚪ White: Overlap")

# Streamlit 앱 시작
st.title("CSV 파일 간 RLE 마스크 및 원본 이미지 비교 도구")

# CSV 파일 업로드
uploaded_files = st.file_uploader("CSV 파일을 2개 업로드하세요", accept_multiple_files=True, type=["csv"])

if len(uploaded_files) == 2:
    # 업로드된 파일 로드
    csv1 = pd.read_csv(uploaded_files[0])  # GT CSV
    csv2 = pd.read_csv(uploaded_files[1])  # Pred CSV

    # 이미지 선택
    gt_image_names = sorted(set(csv1['image_name']))  # GT 이미지 이름 리스트
    test_image_names = sorted(set(csv2['image_name']))  # Pred 이미지 이름 리스트

    col1, col2 = st.columns(2)

    with col1:
        selected_gt_image = st.selectbox("GT 이미지 선택 (image_name)", gt_image_names)

    with col2:
        selected_test_image = st.selectbox("테스트 이미지 선택 (image_name)", test_image_names)

    # 이미지 크기 입력
    image_shape = st.text_input("이미지 크기 입력 (예: 256,256)", value="2028,2048")
    image_shape = tuple(map(int, image_shape.split(',')))

    # 특정 클래스 선택
    class_options = sorted(set(csv1['class']).union(csv2['class']))
    selected_class = st.selectbox("클래스 선택 (모두 보기: None)", [None] + class_options)

    # 디렉터리 경로 입력
    train_dir = st.text_input("GT 이미지 디렉토리 경로", value="/data/ephemeral/home/data/train/DCM")
    test_dir = st.text_input("테스트 이미지 디렉토리 경로", value="/data/ephemeral/home/data/test/DCM")

    # 마스크 및 원본 이미지 시각화
    st.subheader("마스크와 원본 이미지 시각화")
    visualize_masks_and_images(
        csv1, csv2,  # GT와 Pred 순서로 전달
        test_image_name=selected_test_image,
        gt_image_name=selected_gt_image,
        image_shape=image_shape,
        test_dir=test_dir,
        train_dir=train_dir,
        selected_class=selected_class
    )
