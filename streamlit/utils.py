# utils.py
import cv2
import numpy as np
from constants import class_colors

# def draw_annotations(image, annotations, alpha=0.5):
#     if len(image.shape) == 2:
#         image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
#     overlay = image.copy()
    
#     for annotation in annotations:
#         points = annotation.get('points', [])
#         label = annotation.get('label', 'Unknown')
#         points_array = np.array(points, dtype=np.int32)
#         color = class_colors.get(label, (255, 255, 255))
        
#         cv2.fillPoly(overlay, [points_array], color=color)
#         cv2.polylines(overlay, [points_array], isClosed=True, color=color, thickness=2)
        
#         if points:
#             x, y = points[0]
#             cv2.putText(overlay, label, (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX,
#                         1.0, (0, 0, 0), 5, lineType=cv2.LINE_AA)
#             cv2.putText(overlay, label, (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX,
#                         1.0, (255, 255, 255), 2, lineType=cv2.LINE_AA)

#     return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
def draw_annotations(image, annotations, alpha=0.5):
    """이미지 위에 예측된 폴리곤과 라벨을 시각화하는 함수"""
    # 이미지가 흑백인 경우, BGR로 변환
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    overlay = image.copy()
    
    # 어노테이션을 순회하며 시각화
    for annotation in annotations:
        points = annotation.get('points', [])
        label = annotation.get('label', 'Unknown')
        points_array = np.array(points, dtype=np.int32)
        
        # 클래스별 색상 가져오기 (기본 색상은 흰색)
        color = class_colors.get(label, (255, 255, 255))
        
        # 폴리곤 채우기
        if points_array.size > 0:
            cv2.fillPoly(overlay, [points_array], color=color)
            cv2.polylines(overlay, [points_array], isClosed=True, color=color, thickness=2)
        
        # 라벨 표시
        if points:
            x, y = points[0]
            cv2.putText(overlay, label, (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 0, 0), 5, lineType=cv2.LINE_AA)  # 검정색 테두리
            cv2.putText(overlay, label, (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (255, 255, 255), 2, lineType=cv2.LINE_AA)  # 흰색 텍스트

    # 반투명 오버레이 적용
    return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

