import os
import cv2
import numpy as np
from PIL import Image

def analyze_images(folder_path):
    image_list = []

    total_w, total_h = 0, 0
    max_h, max_w = 0, 0
    min_w, min_h = float('inf'), float('inf')

    maxWFile = maxHFile = minWFile = minHFile = None

    for fname in os.listdir(folder_path):
        image_path = os.path.join(folder_path, fname)

        try:
            img = Image.open(image_path)
            img.verify()
            img = Image.open(image_path)
        except:
            continue

        width, height = img.size

        image_list.append({
            "filename": fname,
            "width": width,
            "height": height
        })

        total_w += width
        total_h += height

        if width > max_w:
            max_w = width
            maxWFile = fname

        if height > max_h:
            max_h = height
            maxHFile = fname

        if width < min_w:
            min_w = width
            minWFile = fname

        if height < min_h:
            min_h = height
            minHFile = fname

    if len(image_list) == 0:
        return None

    avg_w = total_w / len(image_list)
    avg_h = total_h / len(image_list)

    return {
        "avg_width": avg_w,
        "avg_height": avg_h,
        "max_width": (max_w, maxWFile),
        "max_height": (max_h, maxHFile),
        "min_width": (min_w, minWFile),
        "min_height": (min_h, minHFile),
    }

def compute_mse(img1, img2):
    return np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)


def block_variance_score(image, k):
    h, w = image.shape[:2]
    total_var = 0
    count = 0

    for y in range(0, h - k, k):
        for x in range(0, w - k, k):
            block = image[y:y+k, x:x+k]
            total_var += np.var(block)
            count += 1

    return total_var / (count + 1e-5)


def find_best_k(image, k_values=range(2, 10)):
    h, w = image.shape[:2]
    results = []

    for k in k_values:
        if h // k == 0 or w // k == 0:
            continue

        down = cv2.resize(image, (w // k, h // k), interpolation=cv2.INTER_NEAREST)
        up = cv2.resize(down, (w, h), interpolation=cv2.INTER_NEAREST)

        mse = compute_mse(image, up)
        results.append((k, mse))

    results.sort(key=lambda x: x[1])
    return results


def process_images(folder_path, num_images=5):
    image_files = os.listdir(folder_path)[:num_images]

    for img_name in image_files:
        path = os.path.join(folder_path, img_name)
        img = cv2.imread(path)

        if img is None:
            print(f"Error loading {img_name}")
            continue

        print(f"\nProcessing: {img_name}")

        results = find_best_k(img)

        print("Top K values:")
        for k, mse in results[:3]:
            print(f"k={k}, mse={mse:.4f}")

        best_k = results[0][0]
        print(f"Best k: {best_k}")

        h, w = img.shape[:2]

        down = cv2.resize(img, (w // best_k, h // best_k), interpolation=cv2.INTER_NEAREST)
        up = cv2.resize(down, (w, h), interpolation=cv2.INTER_NEAREST)

        print("Original shape:", img.shape)
        print("Downscaled shape:", down.shape)

        unique_colors = len(np.unique(img.reshape(-1, 3), axis=0))
        print("Unique colors:", unique_colors)

        variance_score = block_variance_score(img, best_k)
        print("Block variance score:", variance_score)

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# go from src/gan/scaler → project root (pxlrt)
    project_root = os.path.abspath(os.path.join(BASE_DIR, "../../../.."))

# now point to your actual folder
    image_folder = os.path.join(project_root, "resources", "ashlord00")

    print("Using image folder:", image_folder)

    stats = analyze_images(image_folder)
    print("\nDataset stats:", stats)

    process_images(image_folder, num_images=3)