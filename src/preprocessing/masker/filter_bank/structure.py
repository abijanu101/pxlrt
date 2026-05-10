import numpy as np
import cv2

def edge_distance_transform(img):
    'distance to nearest edge'
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 100, 200)
    inv = 255 - edges

    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 5)

    return dist / (dist.max() + 1e-6)


def connected_components(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)

    out = np.zeros_like(gray, dtype=np.float32)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        out[labels == i] = area

    cv2.imshow(
        "connected component area",
        out / (out.max() + 1e-6)
    )
