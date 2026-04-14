'''
1. Global (Context) Layer
        
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

2. Local (Detail) Layer

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

3. Semantic Layer (for future extension) 
    YOLO object detection based maps
'''

__all__ = ['generate_priority_mask']

from .masker import generate_priority_mask