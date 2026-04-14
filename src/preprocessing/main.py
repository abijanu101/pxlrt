import os
import pathlib

from src.preprocessing.scaler import get_scaled
from src.preprocessing.masker import generate_priority_mask
from src.preprocessing.cropper import get_cropped


if __name__ == '__main__':
    '''
        pipline orchestration for all files
            downscaler -> priority_masks -> cropping_agent 
    '''
