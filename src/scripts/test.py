import pathlib
import cv2
import math
import sympy
import numpy as np
import matplotlib.pyplot as plt
import threading

DIR = './resources/ashlord00/images'
# dir = pathlib.Path(DIR, '01119409-894e-4be0-a0a7-a804acbed38e.png')
dir = pathlib.Path(DIR, '2f9a5860-78bf-4f7f-83f1-5b40eabe64e7.png')
my_img = cv2.imread(dir)
# my_img = cv2.cvtColor(my_img, cv2.COLOR_BGR2RGB)
# plt.imshow(my_img)
# plt.show()

def largest_flat_region(img, y, x, k):
    '''Takes in the top-left corner (x,y) of a kxk region within an input image.
        Returns the width for the largest flat square within the region that contains (x,y)
    '''
    base_cell = img[y][x]
    for c in range(1, k):
        for i in range(0, c + 1):
            if not np.array_equal(base_cell, img[y + i][x + c]):
                return c
            if not np.array_equal(base_cell, img[y + c][x + i]):
                return c
    return k

def get_pixel_size(img: np.ndarray) -> np.ndarray:
    '''Find true pixel size for an upscaled image assuming no anti-aliasing'''
    
    h,w,_channels = img.shape
    d1 = set(sympy.divisors(h))
    d2 = set(sympy.divisors(w))
    hypotheses = sorted(d1.intersection(d2))

    i = 0
    while (i != h):
        j = 0
        while (j != w):
            if len(hypotheses) == 1:
                print(i,j)
                return 1

            c = largest_flat_region(img, i, j, hypotheses[-1])
            hypotheses = [hyp for hyp in hypotheses if hyp <= c]

            j -= j % hypotheses[-1]
            j += hypotheses[-1]
        i += hypotheses[-1]

    return hypotheses[-1]

def dumb_scale(img, ratio, interpolation):
    h = int(img.shape[0] * ratio)
    w = int(img.shape[1] * ratio)
    return cv2.resize(img, (w,h), interpolation=interpolation)

cv2.imshow('images', np.hstack([
    dumb_scale(my_img[128:256,128:256], 5, cv2.INTER_LINEAR),
    dumb_scale(my_img[128:256, 128:256], 5, cv2.INTER_NEAREST),
]))
cv2.waitKey(0)
# print(get_pixel_size(my_img))