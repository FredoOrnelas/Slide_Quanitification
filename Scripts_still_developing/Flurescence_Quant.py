#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 17 13:20:31 2025

@author: fredoornelas
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch fluorescence processing:
- Walk a folder of images (non-recursive by default; see RECURSIVE flag)
- For each image, compute mask (Otsu or manual), basic stats
- Save: <name>_mask.png and <name>_overlap.png into OUTPUT_DIR
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Config
# ----------------------------
# Folder with images to process:
INPUT_DIR = r"/path/to/your/input/folder"

# If None, will create "<INPUT_DIR>/outputs"
OUTPUT_DIR = None

# Process subfolders too?
RECURSIVE = False

# Which extensions to include (case-insensitive)
INCLUDE_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}


# Threshold config:
USE_OTSU = False          # Set True to auto-threshold with Otsu on grayscale
MANUAL_THRESHOLD = 45     # Ignored if USE_OTSU=True. Interpreted in native image scale (8- or 16-bit, etc.)

# Morphology (cleanup)
MIN_OBJECT_AREA = 0       # Remove tiny specks (in pixels). Set 0 to skip.

# Red overlap settings (for the saved overlay PNG)
OVERLAY_ALPHA = 0.15      # Transparency for the red mask (0..1)

# ----------------------------
# Helpers
# ----------------------------
def imread_anydepth(path: str) -> np.ndarray:
    """Read image preserving bit depth and channels."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img

def to_gray(img: np.ndarray) -> np.ndarray:
    """Convert to grayscale for thresholding; keeps native dtype/scale."""
    if img.ndim == 2:
        return img
    if img.ndim == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.max(axis=-1) if img.ndim == 3 else img

def normalize_for_display(img: np.ndarray) -> np.ndarray:
    """Normalize any-depth image to [0,1] for display (robust contrast)."""
    img = img.astype(np.float32)
    vmin, vmax = np.percentile(img, (0.5, 99.5))
    if vmax <= vmin:
        vmax = img.max() if img.max() > 0 else 1.0
        vmin = 0.0
    img = np.clip((img - vmin) / (vmax - vmin), 0, 1)
    return img

def binary_mask(gray: np.ndarray, use_otsu: bool, manual_thresh: float) -> np.ndarray:
    """Compute boolean mask; threshold interpreted in native scale."""
    if use_otsu:
        if gray.dtype != np.uint8:
            denom = gray.max() if gray.max() > 0 else 1.0
            g8 = np.clip((gray.astype(np.float64) / denom) * 255, 0, 255).astype(np.uint8)
        else:
            g8 = gray
        _, th = cv2.threshold(g8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if gray.dtype != np.uint8 and gray.max() > 0:
            scaled_thresh = (th / 255.0) * gray.max()
            return gray > scaled_thresh
        return gray > th
    else:
        return gray > manual_thresh

def clean_mask(mask: np.ndarray, min_area: int) -> np.ndarray:
    """Remove tiny components with morphology + area filtering."""
    if min_area <= 0:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_u8 = mask.astype(np.uint8) * 255
    opened = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel, iterations=1)

    cnts, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    keep = np.zeros_like(opened)
    for c in cnts:
        if cv2.contourArea(c) >= min_area:
            cv2.drawContours(keep, [c], -1, 255, thickness=cv2.FILLED)
    return keep.astype(bool)

def masked_mean(img: np.ndarray, mask: np.ndarray) -> float:
    vals = img[mask]
    return float(vals.mean()) if vals.size > 0 else float("nan")

def compose_red_overlap(gray_norm01: np.ndarray, mask: np.ndarray, alpha: float) -> np.ndarray:
    """
    Create an RGB image with gray background and red overlay blended in.
    gray_norm01: float32 [0..1], shape (H,W)
    mask: bool array (H,W)
    returns uint8 RGB image shape (H,W,3)
    """
    g = (gray_norm01 * 255.0).astype(np.uint8)
    rgb = np.stack([g, g, g], axis=-1).astype(np.float32)

    # red layer (255,0,0)
    red = np.zeros_like(rgb)
    red[..., 2] = 0    # B
    red[..., 1] = 0    # G
    red[..., 0] = 255  # R (OpenCV uses BGR, but we'll save with cv2 which expects BGR.
                       # To keep things clear, we'll convert to BGR before imwrite.)
    # Blend only where mask==True
    m = mask.astype(np.float32)[..., None]
    rgb = rgb * (1 - m * alpha) + red * (m * alpha)

    # Convert from RGB to BGR for cv2.imwrite consistency
    bgr = rgb[..., ::-1].clip(0, 255).astype(np.uint8)
    return bgr

# ----------------------------
# Core per-file processing
# ----------------------------
def process_one(file_path: str, out_dir: str) -> None:
    try:
        img = imread_anydepth(file_path)
    except FileNotFoundError as e:
        print(e)
        return

    gray = to_gray(img)
    mask = binary_mask(gray, USE_OTSU, MANUAL_THRESHOLD)
    mask = clean_mask(mask, MIN_OBJECT_AREA)

    # Stats (optional print)
    whole_mean = float(gray.mean())
    mask_mean = masked_mean(gray, mask)
    bg_mean = masked_mean(gray, ~mask)
    area_frac = float(mask.mean())

    base = os.path.splitext(os.path.basename(file_path))[0]

    # Save mask (white=mask, black=bg)
    out_mask = os.path.join(out_dir, f"{base}_mask.png")
    cv2.imwrite(out_mask, (mask.astype(np.uint8) * 255))

    # Save overlap (grayscale + semi-transparent red)
    gray_disp = normalize_for_display(gray)
    overlap_bgr = compose_red_overlap(gray_disp, mask, OVERLAY_ALPHA)
    out_overlap = os.path.join(out_dir, f"{base}_overlap.png")
    cv2.imwrite(out_overlap, overlap_bgr)

    print(f"[OK] {base} | px>th={int(mask.sum())} ({area_frac*100:.2f}%) "
          f"| mean(whole)={whole_mean:.2f} mean(mask)={mask_mean:.2f} mean(bg)={bg_mean:.2f}")
    print(f"     Saved: {out_mask}")
    print(f"            {out_overlap}")

# ----------------------------
# Main
# ----------------------------
def iter_paths(root: str, recursive: bool):
    root = os.path.abspath(root)
    if recursive:
        for dirpath, _, files in os.walk(root):
            for fn in files:
                ext = os.path.splitext(fn)[1].lower()
                if ext in INCLUDE_EXTS:
                    yield os.path.join(dirpath, fn)
    else:
        for fn in os.listdir(root):
            fp = os.path.join(root, fn)
            if os.path.isfile(fp) and os.path.splitext(fn)[1].lower() in INCLUDE_EXTS:
                yield fp

def main():
    if not os.path.isdir(INPUT_DIR):
        raise NotADirectoryError(f"INPUT_DIR does not exist or is not a directory: {INPUT_DIR}")

    out_dir = OUTPUT_DIR or os.path.join(INPUT_DIR, "outputs")
    os.makedirs(out_dir, exist_ok=True)

    any_found = False
    for path in iter_paths(INPUT_DIR, RECURSIVE):
        any_found = True
        process_one(path, out_dir)

    if not any_found:
        print(f"No images with extensions {sorted(INCLUDE_EXTS)} found in {INPUT_DIR} "
              f"(recursive={RECURSIVE}).")
    else:
        print(f"\nDone. Outputs saved in: {out_dir}")

if __name__ == "__main__":
    main()
