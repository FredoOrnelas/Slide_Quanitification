#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  7 19:46:43 2025

@author: fredoornelas
"""

# ij_watershed_best_only.py
# ImageJ watershed + particle counting (headless-safe) from Python/Spyder via PyImageJ.
# Saves mask/outline ONLY for the threshold(s) that yield the max particle count per image.

from pathlib import Path
import numpy as np
import pandas as pd

# ---- Configure JVM BEFORE importing imagej ----
from scyjava import config as sj_config
sj_config.add_option("-Djava.awt.headless=true")  # enforce headless JVM

import imagej
from scyjava import jimport

# =========================
# ========= CONFIG ========
# =========================
INPUT_DIR  = r"PATH/TO/DAPI/IM"   # folder with images
OUTPUT_DIR = r"CUSTOMIZE/PATH/TO/YOUR/LIKING"

# Threshold sweep: lower threshold from MIN_THR..MAX_THR (inclusive) with STEP
MIN_THR, MAX_THR, STEP = 15, 200, 5

# Particle size filter in pixels (ImageJ "size=MIN-Infinity")
MIN_SIZE_PIXELS = 20.0
MAX_SIZE_PIXELS = 1e20

# Polarity for Convert to Mask (matches setOption("BlackBackground", ...))
BLACK_BACKGROUND = False   # False → objects white on black after Convert to Mask

# Save controls
SAVE_MASK = True
SAVE_BINARY_OUTLINE = False       # thin outline of saved mask (approx "Outlines")
SAVE_ALL_TIES = False             # if True, save all thresholds tied for max; else save lowest threshold only

# File types to consider
EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}

# =========================
# ====== INITIALIZE =======
# =========================
# Use Fiji to ensure all IJ1 commands exist; stay headless.
ij = imagej.init('sc.fiji:fiji', mode='headless')

# Java classes
IJ               = jimport('ij.IJ')
Duplicator       = jimport('ij.plugin.Duplicator')
Prefs            = jimport('ij.Prefs')
ResultsTable     = jimport('ij.measure.ResultsTable')
ParticleAnalyzer = jimport('ij.plugin.filter.ParticleAnalyzer')
Measurements     = jimport('ij.measure.Measurements')
ImageProcessor   = jimport('ij.process.ImageProcessor')

# =========================
# ========= HELPERS =======
# =========================
def _p(pathlike):
    return str(Path(pathlike).resolve()).replace("\\", "/")

def ensure_dirs(*paths):
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)

def save_png(imp, path):
    IJ.saveAs(imp, "PNG", _p(path))

def make_binary_outline(imp_bin):
    # ImageJ's Binary ▸ Outline (works headlessly)
    IJ.run(imp_bin, "Outline", "")

# =========================
# ===== CORE ROUTINES =====
# =========================
def make_mask_and_watershed(imp, thr_low):
    """Duplicate, 8-bit, threshold(thr_low..255), Convert to Mask, Watershed → return binary ImagePlus."""
    dup = Duplicator().run(imp)
    IJ.run(dup, "8-bit", "")
    ip = dup.getProcessor()
    ip.setThreshold(float(thr_low), 255.0, ImageProcessor.NO_LUT_UPDATE)
    Prefs.blackBackground = bool(BLACK_BACKGROUND)
    IJ.run(dup, "Convert to Mask", "")
    IJ.run(dup, "Watershed", "")
    return dup

def count_particles(imp_binary):
    """ParticleAnalyzer without GUI; returns integer count."""
    rt = ResultsTable()
    options = ParticleAnalyzer.SHOW_NONE
    measurements = Measurements.AREA
    pa = ParticleAnalyzer(options, measurements, rt, float(MIN_SIZE_PIXELS), float(MAX_SIZE_PIXELS))
    ok = pa.analyze(imp_binary)  # returns boolean
    return int(rt.getCounter())

def analyze_one_threshold(img_path, thr_low, save=False, out_dir=None):
    """
    Process one image at a single threshold; optionally save outputs.
    Returns particle count.
    """
    imp = IJ.openImage(_p(img_path))
    if imp is None:
        raise RuntimeError(f"Could not open image: {img_path}")

    try:
        mask_imp = make_mask_and_watershed(imp, thr_low)
        count = count_particles(mask_imp)

        if save and out_dir is not None:
            out_dir = Path(out_dir)
            ensure_dirs(out_dir)
            stem = Path(img_path).stem
            if SAVE_MASK:
                mask_path = out_dir / f"{stem}__thr_{thr_low:03d}__mask.png"
                save_png(mask_imp, mask_path)
            if SAVE_BINARY_OUTLINE:
                outline_imp = Duplicator().run(mask_imp)
                make_binary_outline(outline_imp)
                outlines_path = out_dir / f"{stem}__thr_{thr_low:03d}__outline.png"
                save_png(outline_imp, outlines_path)
                outline_imp.close()

        mask_imp.close()
        return count
    finally:
        imp.close()

def process_image(img_path, out_root):
    """
    Sweep thresholds, pick max, save only best (or ties), write per-image CSV.
    Returns (df, df_best_row or df_best_rows).
    """
    per_image_out = Path(out_root) / Path(img_path).stem
    ensure_dirs(per_image_out)

    # 1) Sweep without saving to find max
    rows = []
    for thr in range(MIN_THR, MAX_THR + 1, STEP):
        try:
            cnt = analyze_one_threshold(img_path, thr, save=False)
        except Exception as e:
            cnt = np.nan
            print(f"[WARN] {Path(img_path).name} thr={thr}: {e}")
        rows.append({"image": Path(img_path).name, "threshold_low": thr, "count": cnt})

    df = pd.DataFrame(rows)
    df.to_csv(per_image_out / f"{Path(img_path).stem}__threshold_sweep.csv", index=False)

    if not df["count"].notna().any():
        return df, None  # nothing to save

    # 2) Choose best threshold(s)
    max_count = df["count"].max()
    best_rows = df[df["count"] == max_count].copy()

    if SAVE_ALL_TIES:
        chosen_thresholds = sorted(best_rows["threshold_low"].tolist())
    else:
        # pick the lowest threshold among ties
        chosen_thresholds = [int(best_rows["threshold_low"].min())]
        best_rows = best_rows.loc[best_rows["threshold_low"].idxmin()].to_frame().T

    # 3) Re-run only best threshold(s) with saving enabled
    for thr in chosen_thresholds:
        _ = analyze_one_threshold(img_path, thr, save=True, out_dir=per_image_out)

    return df, best_rows

def main():
    in_dir = Path(INPUT_DIR)
    out_dir = Path(OUTPUT_DIR)
    ensure_dirs(out_dir)

    files = sorted([p for p in in_dir.iterdir() if p.is_file() and p.suffix.lower() in EXTS])
    if not files:
        print(f"No images found in {in_dir}")
        return

    print(f"Found {len(files)} images. Processing...")
    all_rows, best_rows_accum = [], []

    for i, fp in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {fp.name}")
        df, best = process_image(fp, out_dir)
        all_rows.append(df)
        if best is not None:
            best_rows_accum.append(best)

    # Master CSVs
    if all_rows:
        df_all = pd.concat(all_rows, ignore_index=True)
        df_all.to_csv(out_dir / "all_threshold_sweeps.csv", index=False)

    if best_rows_accum:
        df_best = pd.concat(best_rows_accum, ignore_index=True)
        # standardize column names
        df_best = df_best.rename(columns={"threshold_low": "best_threshold", "count": "max_particles"})
        df_best.to_csv(out_dir / "best_thresholds_per_image.csv", index=False)
        print("\n=== Best thresholds per image ===")
        print(df_best.to_string(index=False))
    else:
        print("No results to summarize.")

if __name__ == "__main__":
    main()

