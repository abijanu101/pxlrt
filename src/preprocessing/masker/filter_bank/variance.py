import numpy as np
import cv2


def block_variance(img, block=8):
    'x <- std(neighborhood(x))'
    h, w = img.shape

    out = np.zeros_like(img, dtype=np.float32)

    for y in range(0, h, block):
        for x in range(0, w, block):
            patch = img[y:y+block, x:x+block]
            out[y:y+block, x:x+block] = patch.std()

    return out / (out.max() + 1e-6)


def multiscale_block_variance(
    img,
    scale_divisors=(64, 32, 16, 8),
    min_block=4
):
    """
    scales are generated relative to image size:
        small   = min_dim / 64
        medium  = min_dim / 32
        large   = min_dim / 16
        xlarge  = min_dim / 8

    min_block just lets you add guard rails for smaller min_dims
    """

    h, w = img.shape[:2]
    min_dim = min(h, w)

    # dynamically generate scales from image size
    scales = sorted(set([
        max(min_block, min_dim // d)
        for d in scale_divisors
    ]))

    accum = np.zeros((h, w), dtype=np.float32)

    for block in scales:
        accum += block_variance(img, block)

    return accum / (accum.max() + 1e-6)
