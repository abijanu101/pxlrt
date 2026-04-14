'''
    Takes in an image and a priority mask -> returns a list of candidate crop-outs
'''

__all__=['get_cropped']

from .cropper import get_cropped