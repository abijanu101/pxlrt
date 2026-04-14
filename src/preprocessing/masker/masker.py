import numpy as np
import cv2

def generate_priority_mask(img: np.ndarray) -> np.ndarray:
    'Returns a heatmap for areas of relative importance based on trends in color and texture variance'
    
    
    return img


# playground

if __name__ == '__main__':
    img = cv2.imread('resources/ashlord00/images/ffc6459a-b287-4707-bda3-31e8e8168d20.png')
    edges = cv2.Canny(img, 250, 150)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    # img = cv2.GaussianBlur(img, (25, 25), 0)
    img = np.hstack([img, edges])
    # img = cv2.Canny(img, 10, 10)
    cv2.imshow('hi', img)
    cv2.waitKey(0)
