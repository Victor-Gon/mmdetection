import os
import cv2
import json
import numpy as np
from tqdm import tqdm

# ——— CONFIG ———————————————————————————————————————————————
TRAIN_IMG_DIR          = "/mnt/hd/waymococo_f0/train2020"
OUTPUT_STATS_JSON      = "utilities_mmdet/annotations/dataset_stats_gray_clahe.json"
CLIP_LIMIT             = 2.0
TILE_GRID_SIZE         = (8, 8)

# supported image extensions
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

def compute_mean_std_gray_clahe(img_dir, clip_limit, tile_grid_size):
    total_pixels = 0
    sum_pixels = 0.0
    sum_pixels_sq = 0.0

    # prepare CLAHE object once
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    # gather all image file paths
    img_paths = []
    for root, _, files in os.walk(img_dir):
        for fname in files:
            if os.path.splitext(fname)[1].lower() in IMG_EXTS:
                img_paths.append(os.path.join(root, fname))

    # process each image
    for img_path in tqdm(img_paths, desc="Processing images"):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            # skip files that cannot be read
            continue

        # convert to gray
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # apply CLAHE
        img_eq = clahe.apply(gray).astype(np.uint8)

        # accumulate statistics
        flat = img_eq.ravel().astype(np.float64)
        sum_pixels += flat.sum()
        sum_pixels_sq += np.square(flat).sum()
        total_pixels += flat.size

    # compute mean and std
    mean = sum_pixels / total_pixels
    var = (sum_pixels_sq / total_pixels) - (mean ** 2)
    std = np.sqrt(var)

    return mean, std

if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUTPUT_STATS_JSON), exist_ok=True)
    print(f"Scanning '{TRAIN_IMG_DIR}' for images...")
    mean, std = compute_mean_std_gray_clahe(
        TRAIN_IMG_DIR, 
        clip_limit=CLIP_LIMIT, 
        tile_grid_size=TILE_GRID_SIZE
    )
    stats = {"mean": mean, "std": std}
    with open(OUTPUT_STATS_JSON, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\n→ Mean: {mean:.4f}")
    print(f"→ Std:  {std:.4f}")
    print(f"\nSaved stats to '{OUTPUT_STATS_JSON}'.")