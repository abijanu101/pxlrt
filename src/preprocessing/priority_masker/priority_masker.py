import numpy as np
import cv2

'''
Global (Context) Layer
    
    Divide image into grids
    Calculate the mean variance for each cell
    Increment proportional to deviance from global variance

    Inputs:
    - Blurred Image (per channel contrast)
        lightness, color pallette shifts, etc.
    - Canny Edges
        for identifying regions of high detail levels
    - Canny Variance Map
        for capturing deviation in variance trends (smoother region vs high texture region)
    - Raw Image Variance Map
        for capturing deviation in variance trends (high contrast abundance v. low contrast minority)

Local (Detail) Layer

    Pixel-by-pixel instead of grid-based
    Calculate a normal weighted average for the region
    Difference between local average and actual pixel as priority

    Inputs:
    - Raw Image
        color contrast details
    - Local Variance Map
        for capturing deviations in local variance trends
    - Canny
        for identifying local edge density deviations

Semantic Layer (for future extension)
    YOLO object detection based maps
'''

def generate_priority_map(img: np.ndarray) -> np.ndarray:
    'Returns a heatmap for areas of relative importance based on trends in color and texture variance'



    return img # placeholder

if __name__ == '__main__':
    img = cv2.imread('resources/ashlord00/images/ffc6459a-b287-4707-bda3-31e8e8168d20.png')
    edges = cv2.Canny(img, 250, 150)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    img = np.hstack([img, edges])
    # img = cv2.Canny(img, 10, 10)
    cv2.imshow('hi', img)
    cv2.waitKey(0)
