# utils.py
import cv2
import numpy as np
from constants import class_colors

def draw_annotations(image, annotations, alpha=0.5):
    """폴리곤 어노테이션을 이미지 위에 그립니다."""
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    overlay = image.copy()
    
    for annotation in annotations:
        points = annotation.get('points', [])
        label = annotation.get('label', 'Unknown')
        points_array = np.array(points, dtype=np.int32)
        color = class_colors.get(label, (255, 255, 255))
        
        cv2.fillPoly(overlay, [points_array], color=color)
        cv2.polylines(overlay, [points_array], isClosed=True, color=color, thickness=2)
        
        if points:
            x, y = points[0]
            cv2.putText(overlay, label, (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 0, 0), 5, lineType=cv2.LINE_AA)
            cv2.putText(overlay, label, (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (255, 255, 255), 2, lineType=cv2.LINE_AA)

    return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
