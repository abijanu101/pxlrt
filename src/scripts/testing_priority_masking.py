import numpy as np
import cv2

DIR = './resources/ashlord00/images'


def intensity_thing(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("intensity", gray)


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


def dog_thing(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    g1 = cv2.GaussianBlur(gray, (0, 0), 1)
    g2 = cv2.GaussianBlur(gray, (0, 0), 5)

    dog = g1 - g2
    cv2.imshow("DoG", dog / (np.max(np.abs(dog)) + 1e-6))

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

def distance_transform_thing(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 100, 200)
    inv = 255 - edges

    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 5)

    cv2.imshow(
        "distance transform",
        dist / (dist.max() + 1e-6)
    )

def connected_components_thing(img):
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

def multiscale_block_variance_thing(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    scales = [4, 8, 16, 32]

    accum = np.zeros_like(gray, dtype=np.float32)

    for block in scales:
        h, w = gray.shape
        out = np.zeros_like(gray, dtype=np.float32)

        for y in range(0, h, block):
            for x in range(0, w, block):
                patch = gray[y:y+block, x:x+block]
                out[y:y+block, x:x+block] = patch.std()

        accum += out

    cv2.imshow(
        "multiscale block variance",
        accum / (accum.max() + 1e-6)
    )

def regional_isolation_thing(img):
    h, w = img.shape[:2]

    # scale factor instead of fixed size
    scale = 0.25  # 25% resolution (tunable)

    new_w = max(8, int(w * scale))
    new_h = max(8, int(h * scale))

    small = cv2.resize(
        img,
        (new_w, new_h),
        interpolation=cv2.INTER_NEAREST
    )

    blur = cv2.GaussianBlur(
        small,
        (0, 0),
        5
    )

    diff = np.linalg.norm(
        small.astype(np.float32) - blur.astype(np.float32),
        axis=2
    )

    diff = cv2.resize(
        diff,
        (w, h),
        interpolation=cv2.INTER_NEAREST
    )

    cv2.imshow(
        "regional isolation (scale-aware)",
        diff / (diff.max() + 1e-6)
    )

def local_mi_thing(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (0, 0), 2)

    mi = gray.astype(np.float32) * blur.astype(np.float32)

    cv2.imshow("local MI proxy", mi / (mi.max() + 1e-6))


def fft_lowpass_multiscale_variance_thing(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # -------------------------
    # FFT low-pass reconstruction
    # -------------------------
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)

    h, w = gray.shape
    cy, cx = h // 2, w // 2

    mask = np.zeros((h, w), np.uint8)

    # scale cutoff relative to image size (important fix)
    r = min(h, w) // 12
    mask[cy - r:cy + r, cx - r:cx + r] = 1

    low_freq = fshift * mask
    recon = np.fft.ifft2(np.fft.ifftshift(low_freq))
    recon = np.abs(recon).astype(np.float32)

    recon_norm = recon / (recon.max() + 1e-6)

    cv2.imshow("fft lowpass", recon_norm)

    # -------------------------
    # residual (optional but useful)
    # -------------------------
    residual = np.abs(gray - recon)
    residual = residual / (residual.max() + 1e-6)
    cv2.imshow("fft residual", residual)

    # -------------------------
    # multiscale variance on lowpass
    # -------------------------
    h, w = recon.shape

    scales = [
        max(4, min(h, w) // 64),
        max(8, min(h, w) // 32),
        max(16, min(h, w) // 16),
        max(32, min(h, w) // 8),
    ]

    out = np.zeros_like(recon, dtype=np.float32)

    for s in scales:
        for y in range(0, h, s):
            for x in range(0, w, s):
                patch = recon[y:y+s, x:x+s]
                out[y:y+s, x:x+s] = patch.std()

    out = out / (out.max() + 1e-6)

    cv2.imshow("fft lowpass + multiscale variance", out)


# =========================
# OPERATOR BANK
# =========================


OPERATORS = [
    fft_lowpass_multiscale_variance_thing
    # # Color Based Activation Maps
    # color_range_thing,
    # color_novelty_thing,
    # kmeans_error_thing,

    # # Edge and Texture
    # block_variance_thing,
    # multiscale_block_variance_thing,
    # neighbor_contrast_thing,

    # # Unique
    # distance_transform_thing,
    # regional_isolation_thing,
    # local_mi_thing,

    # # Experimental (useless alone, probably need an abs(z_scores(img)).)
    # intensity_thing,
    # dog_thing,
    # fft_lowpass_thing,
    # connected_components_thing,
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
    DIR + "/0b4f83ac-fc76-434a-95d8-853a472f386a.png",
    
    DIR + "/02ff04e4-f65c-44bd-8c7f-9d9a41ecdb79.png",
    DIR + "/e0d0c74b-d6ba-42d5-a6c1-bbaedc14ff7e.png",
    DIR + "/ffc6459a-b287-4707-bda3-31e8e8168d20.png",
    DIR + "/40675b7e-5c87-45b2-b3ef-dfd63dc5ea19.png",
    # DIR + "/01119409-894e-4be0-a0a7-a804acbed38e.png"
]


for p in imgs:
    playingAround(p)