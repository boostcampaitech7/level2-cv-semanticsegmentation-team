import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def rle_decode(mask_rle, shape=(2048, 2048)):
    """RLE을 디코딩하여 마스크 이미지를 생성합니다."""
    if not isinstance(mask_rle, str) or mask_rle.strip() == "":
        # rle 값이 비어 있거나 문자열이 아닌 경우 0으로 채워진 마스크 반환
        return np.zeros(shape[0] * shape[1], dtype=np.uint8).reshape(shape)
    
    try:
        s = mask_rle.split()
        starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
        starts -= 1  # 0-based index
        ends = starts + lengths
        img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
        for lo, hi in zip(starts, ends):
            img[lo:hi] = 1
        return img.reshape(shape)
    except Exception as e:
        raise ValueError(f"RLE 디코딩 오류: {mask_rle}. 에러: {e}")


def calculate_overlap(df1, df2, image_name):
    """주어진 이미지에 대한 두 데이터프레임의 클래스별 겹침 비율을 계산합니다."""
    df1_image = df1[df1['image_name'] == image_name]
    df2_image = df2[df2['image_name'] == image_name]
    
    classes = sorted(set(df1_image['class']) | set(df2_image['class']))
    overlap_results = []
    
    for cls in classes:
        mask1 = np.zeros((2048, 2048), dtype=np.uint8)
        mask2 = np.zeros((2048, 2048), dtype=np.uint8)
        
        for _, row in df1_image[df1_image['class'] == cls].iterrows():
            mask1 |= rle_decode(row['rle'])
        
        for _, row in df2_image[df2_image['class'] == cls].iterrows():
            mask2 |= rle_decode(row['rle'])
        
        overlap = np.logical_and(mask1, mask2)
        only_df1 = np.logical_and(mask1, ~mask2)
        only_df2 = np.logical_and(mask2, ~mask1)
        
        total_area = np.sum(overlap) + np.sum(only_df1) + np.sum(only_df2)
        
        overlap_percentage = (np.sum(overlap) / total_area * 100) if total_area > 0 else 0
        only_df1_percentage = (np.sum(only_df1) / total_area * 100) if total_area > 0 else 0
        only_df2_percentage = (np.sum(only_df2) / total_area * 100) if total_area > 0 else 0
        
        overlap_results.append((cls, overlap_percentage, only_df1_percentage, only_df2_percentage))
    
    return overlap_results

def visualize_overlap(df1, df2, image_name, selected_class=None, threshold=0):
    """겹침 비율을 시각화합니다."""
    overlap_results = calculate_overlap(df1, df2, image_name)

    # 선택된 클래스가 없을 경우 겹침 임계값 기반으로 필터링
    if not selected_class:
        overlap_results = [result for result in overlap_results if result[1] < threshold]

    # 선택된 클래스가 있을 경우 해당 클래스만 남김
    if selected_class:
        overlap_results = [result for result in overlap_results if result[0] == selected_class]

    # 데이터 준비
    classes = [result[0] for result in overlap_results]
    
    # 클래스가 없으면 경고 메시지 표시
    if not classes:
        st.warning("조건에 맞는 클래스가 없습니다.")
        return

    overlap_percentages = [result[1] for result in overlap_results]
    only_csv1_percentages = [result[2] for result in overlap_results]
    only_csv2_percentages = [result[3] for result in overlap_results]

    # 시각화
    x_labels = range(len(classes))

    fig, ax = plt.subplots(figsize=(12, 6))  # 그래프 크기 조정

    bar_width = 0.25

    # 막대 그래프 그리기
    if not selected_class:
        bars = ax.bar(x_labels, overlap_percentages, width=bar_width, label='Overlap', align='center')
        lowest_indices = np.argsort(overlap_percentages)[:2]
        
        for i in range(len(bars)):
            if i in lowest_indices:
                bars[i].set_color('red')  # 빨간색으로 설정
    else:
        ax.bar(x_labels, overlap_percentages, width=bar_width, label='Overlap', align='center')
        ax.bar(np.array(x_labels) + bar_width, only_csv1_percentages, width=bar_width, label='Only CSV 1', align='center')
        ax.bar(np.array(x_labels) + bar_width * 2, only_csv2_percentages, width=bar_width, label='Only CSV 2', align='center')

        for i in range(len(classes)):
            ax.text(i - bar_width / 3, overlap_percentages[i] + 0.5,
                    f'{overlap_percentages[i]:.1f}%', ha='center')

    ax.set_xlabel('Class')  
    ax.set_ylabel('Percentage')
    ax.set_title(f'Mask Overlap for {image_name}')

    ax.set_xticks(np.array(x_labels) + bar_width)
    ax.set_xticklabels(classes)

    plt.xticks(rotation=45)  
    plt.legend()
    plt.tight_layout()

    st.pyplot(fig)

def main():
    st.title("CSV Mask Overlap Visualization")

    uploaded_files = st.file_uploader("CSV 파일을 선택하세요 (2개)", type="csv", accept_multiple_files=True)

    if len(uploaded_files) < 2:
        st.warning("최소 2개의 CSV 파일을 선택해주세요.")
        return

    dataframes = [pd.read_csv(file) for file in uploaded_files]

    # 이미지 이름을 가져오고 'All' 옵션 추가
    image_names = list(set(dataframes[0]['image_name']) & set(dataframes[1]['image_name']))
    image_names.insert(0, 'All')  
    selected_image = st.selectbox("이미지 선택", image_names)

    class_names = list(set().union(*[df['class'].unique() for df in dataframes]))
    selected_class = st.selectbox("클래스 선택 (선택사항)", [''] + class_names)

    threshold = st.slider("겹침 임계값 (%)", 0, 100, 50)

    if st.button("시각화"):
        if selected_image == 'All':
            for img in image_names[1:]:  
                visualize_overlap(dataframes[0], dataframes[1], img, None, threshold)
        else:
            visualize_overlap(dataframes[0], dataframes[1], selected_image, selected_class if selected_class else None, threshold)

if __name__ == "__main__":
    main()
