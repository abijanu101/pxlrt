import numpy as np
import cv2

DIR = './resources/ashlord00/images'


# =========================
# INTENSITY OPERATORS
# =========================

def intensity_thing(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("intensity", gray)


def inverted_intensity_thing(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("inverted intensity", 255 - gray)


# =========================
# COLOR OPERATORS
# =========================

def channel_thing(img):
    b, g, r = cv2.split(img)
    cv2.imshow("blue", b)
    cv2.imshow("green", g)
    cv2.imshow("red", r)


def color_range_thing(img):
    maxc = np.max(img, axis=2)
    minc = np.min(img, axis=2)
    diff = maxc - minc
    cv2.imshow("color range", diff)


def color_novelty_thing(img):
    mean_color = img.reshape(-1, 3).mean(axis=0)
    diff = img.astype(np.float32) - mean_color
    score = np.linalg.norm(diff, axis=2)
    cv2.imshow("color novelty", score / (score.max() + 1e-6))


# =========================
# LOCAL STATISTICS
# =========================

def variance_thing(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    mean = cv2.blur(gray, (7, 7))
    mean2 = cv2.blur(gray ** 2, (7, 7))

    var = mean2 - mean ** 2
    cv2.imshow("local variance", var / (var.max() + 1e-6))


def std_thing(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    mean = cv2.blur(gray, (7, 7))
    mean2 = cv2.blur(gray ** 2, (7, 7))

    std = np.sqrt(mean2 - mean ** 2)
    cv2.imshow("local std", std / (std.max() + 1e-6))


# =========================
# FREQUENCY / RESIDUAL
# =========================

def residual_thing(img):
    base = cv2.GaussianBlur(img, (0, 0), 3)
    detail = np.abs(img.astype(np.float32) - base.astype(np.float32))

    cv2.imshow("gaussian residual", detail / (detail.max() + 1e-6))


def dog_thing(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    g1 = cv2.GaussianBlur(gray, (0, 0), 1)
    g2 = cv2.GaussianBlur(gray, (0, 0), 5)

    dog = g1 - g2
    cv2.imshow("DoG", dog / (np.max(np.abs(dog)) + 1e-6))


# =========================
# FFT OPERATORS
# =========================

def fft_thing(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)

    mag = np.log1p(np.abs(fshift))
    mag = mag / mag.max()

    cv2.imshow("FFT magnitude", mag)


def fft_spectrum_thing(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)

    mag = np.log1p(np.abs(fshift))
    cv2.imshow("FFT spectrum", mag / mag.max())


def fft_lowpass_thing(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)

    h, w = gray.shape
    mask = np.zeros((h, w), np.uint8)

    r = 30
    cy, cx = h // 2, w // 2
    mask[cy-r:cy+r, cx-r:cx+r] = 1

    filtered = fshift * mask
    recon = np.fft.ifft2(np.fft.ifftshift(filtered))
    recon = np.abs(recon)

    cv2.imshow("FFT low-pass", recon / (recon.max() + 1e-6))


def fft_highpass_thing(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)

    h, w = gray.shape
    mask = np.ones((h, w), np.uint8)

    r = 30
    cy, cx = h // 2, w // 2
    mask[cy-r:cy+r, cx-r:cx+r] = 0

    filtered = fshift * mask
    recon = np.fft.ifft2(np.fft.ifftshift(filtered))
    recon = np.abs(recon)

    cv2.imshow("FFT high-pass", recon / (recon.max() + 1e-6))


# =========================
# PALETTE OPERATORS
# =========================

def palette_thing(img):
    data = img.reshape(-1, 3).astype(np.float32)

    _, labels, centers = cv2.kmeans(
        data, 3, None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0),
        5, cv2.KMEANS_RANDOM_CENTERS
    )

    out = centers[labels.flatten()].reshape(img.shape)
    cv2.imshow("palette reconstruction", out.astype(np.uint8))


def kmeans_error_thing(img):
    data = img.reshape(-1, 3).astype(np.float32)

    _, labels, centers = cv2.kmeans(
        data, 4, None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0),
        5, cv2.KMEANS_RANDOM_CENTERS
    )

    recon = centers[labels.flatten()].reshape(img.shape)
    error = np.linalg.norm(img.astype(np.float32) - recon.astype(np.float32), axis=2)

    cv2.imshow("palette error", error / (error.max() + 1e-6))


# =========================
# STRUCTURE OPERATORS
# =========================

def canny_thing(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 200, 250)
    cv2.imshow("canny edges", edges)


def edge_density_thing(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 200, 250)

    density = cv2.GaussianBlur(edges.astype(np.float32), (5, 5), 1)
    cv2.imshow("edge density", density / (density.max() + 1e-6))


# =========================
# PIXEL-ART OPERATORS
# =========================

def block_variance_thing(img, block=8):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape
    out = np.zeros_like(gray, dtype=np.float32)

    for y in range(0, h, block):
        for x in range(0, w, block):
            patch = gray[y:y+block, x:x+block]
            out[y:y+block, x:x+block] = patch.std()

    cv2.imshow("block variance", out / (out.max() + 1e-6))


def neighbor_contrast_thing(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    r = np.roll(gray, 1, axis=0)
    l = np.roll(gray, -1, axis=0)
    u = np.roll(gray, 1, axis=1)
    d = np.roll(gray, -1, axis=1)

    contrast = np.abs(gray - r) + np.abs(gray - l) + np.abs(gray - u) + np.abs(gray - d)

    cv2.imshow("neighbor contrast", contrast / (contrast.max() + 1e-6))


# =========================
# OPERATOR BANK
# =========================

OPERATORS = [
    # canny_thing,
    # edge_density_thing,

    # intensity_thing,
    # inverted_intensity_thing,

    # channel_thing,
    color_range_thing,
    color_novelty_thing,

    variance_thing,
    # std_thing,

    # residual_thing,
    dog_thing,

    # fft_thing,
    # fft_spectrum_thing,
    fft_lowpass_thing,
    fft_highpass_thing,

    palette_thing,
    kmeans_error_thing,

    block_variance_thing,
    neighbor_contrast_thing
]


# =========================
# PLAYGROUND RUNNER
# =========================

def playingAround(path):
    img = cv2.imread(path)

    for op in OPERATORS:
        op(img)

    cv2.imshow("original", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# =========================
# DATASET LOOP
# =========================

imgs = [
    DIR + '/464a4159-95fa-492e-9227-517ec3a425b9.png',
    DIR + '/3ed3c7ce-e0ca-4625-b792-2cc86a6632fc.png',
    DIR + "/648a448f-5584-4ec1-8cf3-237d3cbf5953.png",
    DIR + '/2b569f4c-ec87-4cd1-8481-2e5db5903369.png',
    DIR + '/7dc0dab6-7d41-41a7-a79b-4b7576378032.png',
    DIR + '/0b0611b4-fd91-48f3-a2b7-bb191aeb4e3d.png',
    DIR + '/1775215d-5a39-4cc7-a4e3-9ff9a032653d.png',
    DIR + '/6cbd9c21-4976-442e-b25f-67cb080e9004.png',
    DIR + '/30245fb3-15e3-4e74-a92f-d1c9dbca8ac6.png',
    DIR + '/89991a49-4986-4848-b02b-701f031504e2.png',
    DIR + '/1763714b-cf5f-405b-9eae-00fb22114227.png',
    DIR + '/849b63ae-b47b-4b65-a892-87a32e3ffcd8.png',
    DIR + '/228661ff-79ae-4c45-bff0-84d0e19207b3.png',
    DIR + '/288112c3-64c7-4fef-89f6-6e8e7da48e13.png',
    DIR + '/01119409-894e-4be0-a0a7-a804acbed38e.png'
]


for p in imgs:
    playingAround(p)