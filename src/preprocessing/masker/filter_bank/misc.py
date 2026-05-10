import numpy as np
import cv2

def channel_intensity(img) -> tuple[np.ndarray]:
    'B, G, R intensity'

    b, g, r = cv2.split(img)
    return b, g, r


def fft_lowpass(img:np.ndarray) -> np.ndarray:
    'i dont know how this works but it gave W results fr'

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)

    h, w = gray.shape
    mask = np.zeros((h, w), np.uint8)

    r = min(h, w) // 12
    cy, cx = h // 2, w // 2
    mask[cy-r:cy+r, cx-r:cx+r] = 1

    filtered = fshift * mask
    recon = np.fft.ifft2(np.fft.ifftshift(filtered))
    recon = np.abs(recon)

    return recon / (recon.max() + 1e-6)

def regional_isolation(img, scale:np.float64=0.25):
    'down and re-scaled norm of img - blur'

    h, w = img.shape[:2]
    new_w = max(8, int(w * scale))
    new_h = max(8, int(h * scale))

    small = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    blur = cv2.GaussianBlur(small, (0, 0), 5)

    diff = np.linalg.norm(small.astype(np.float32) - blur.astype(np.float32), axis=2)
    diff = cv2.resize(diff, (w, h), interpolation=cv2.INTER_NEAREST)

    return diff / (diff.max() + 1e-6)


def pseudo_mutual_information(img):
    'apparently the thing img * blur approximates is something called mutual information, idk man it just gave good results'

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (0, 0), 2)

    mi = gray.astype(np.float32) * blur.astype(np.float32)

    return mi / (mi.max() + 1e-6)