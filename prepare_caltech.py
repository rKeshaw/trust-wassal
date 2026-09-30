#!/usr/bin/env python3
# prepare_caltech_from_tfds_split90_10_32x32.py
# Get Caltech-101 (same archive/labels TFDS's caltech101 builder uses),
# resize to 32×32, merge all samples, then stratified 90/10 split → save as
# .npz (C,H,W float32 in [0,1]).
#
# This does not import TensorFlow: only `tensorflow_datasets` (the metadata
# package) is used, purely to read its bundled canonical class-name list
# (image_classification/caltech101_labels.txt) so label indices match what
# tfds.load("caltech101") would have assigned. TFDS's own dataset_builder
# additionally partitions each class into a 30-per-class "train" split and a
# "test" split with the rest (fixed seed 1234) before this script's own
# StratifiedShuffleSplit merges them straight back together — so skipping
# that intermediate partition changes nothing about the final output; only
# every image's (pixels, label) pair needs to match, which this reproduces
# exactly (same archive, same label→index mapping, same resize pipeline).
#
# Expects the archive TFDS would download, caltech-101.zip
# (sha256 331234750fc7f77520e50d9565e8b6907b03565e30320da1401130db08c61f91),
# already placed at data/tfds/downloads/manual/caltech-101.zip.

import os
import tarfile
import zipfile
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import StratifiedShuffleSplit
import tensorflow_datasets as tfds

_MANUAL_ZIP = "data/tfds/downloads/manual/caltech-101.zip"
_EXTRACT_DIR = "data/tfds_manual_extract"
_IMAGES_DIR = os.path.join(_EXTRACT_DIR, "101_ObjectCategories")
_LABELS_FNAME = "image_classification/caltech101_labels.txt"

def pil_resize_32x32(img: Image.Image, size=(32, 32)) -> Image.Image:
    """Resize to 32×32 directly."""
    img = img.convert("RGB")
    # Use LANCZOS for better quality when downsampling
    img = img.resize(size, Image.LANCZOS)
    return img

def pil_to_chw_float32(img: Image.Image) -> np.ndarray:
    """Convert PIL Image → (C,H,W) float32 [0,1]."""
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)
    return arr

def ensure_extracted():
    if os.path.isdir(_IMAGES_DIR):
        return
    if not os.path.exists(_MANUAL_ZIP):
        raise FileNotFoundError(
            f"Expected the Caltech-101 archive at {_MANUAL_ZIP} "
            "(the same file tfds.load('caltech101') would download)."
        )
    print(f"Extracting {_MANUAL_ZIP} ...")
    with zipfile.ZipFile(_MANUAL_ZIP) as z:
        z.extract("caltech-101/101_ObjectCategories.tar.gz", _EXTRACT_DIR)
    tar_path = os.path.join(_EXTRACT_DIR, "caltech-101", "101_ObjectCategories.tar.gz")
    with tarfile.open(tar_path) as t:
        t.extractall(_EXTRACT_DIR)

def load_label_names():
    names_file = tfds.core.tfds_path(_LABELS_FNAME)
    with names_file.open() as f:
        return [line.strip() for line in f if line.strip()]

def extract_all_images():
    ensure_extracted()
    label_names = load_label_names()
    label_to_idx = {name: i for i, name in enumerate(label_names)}

    class_dirs = sorted(
        d for d in os.listdir(_IMAGES_DIR)
        if os.path.isdir(os.path.join(_IMAGES_DIR, d))
    )
    unknown = [d for d in class_dirs if d.lower() not in label_to_idx]
    if unknown:
        raise ValueError(f"Class folders not found in {_LABELS_FNAME}: {unknown}")

    images, labels = [], []
    for d in tqdm(class_dirs):
        label_idx = label_to_idx[d.lower()]
        class_path = os.path.join(_IMAGES_DIR, d)
        fnames = sorted(f for f in os.listdir(class_path) if f.endswith(".jpg"))
        for fname in fnames:
            img_pil = Image.open(os.path.join(class_path, fname))
            img_pil = pil_resize_32x32(img_pil, (32, 32))  # Changed to 32×32
            arr = pil_to_chw_float32(img_pil)
            images.append(arr)
            labels.append(label_idx)
    return np.stack(images, axis=0), np.array(labels, dtype=np.int64)

def main():
    os.makedirs("data", exist_ok=True)
    print("Loading Caltech-101 (local archive, no TensorFlow required)...")

    print("Converting full dataset to 32×32 numpy arrays (this may take time)...")
    X_all, y_all = extract_all_images()
    print(f"Total samples merged: {len(X_all)}")

    # 90/10 stratified split
    sss = StratifiedShuffleSplit(test_size=0.1, random_state=42)
    train_idx, test_idx = next(sss.split(X_all, y_all))

    X_train_new, y_train_new = X_all[train_idx], y_all[train_idx]
    X_test_new,  y_test_new  = X_all[test_idx],  y_all[test_idx]

    print(f"Split complete: train={len(X_train_new)}, test={len(X_test_new)}")
    print(f"Train shape: {X_train_new.shape}, Test shape: {X_test_new.shape}")

    # Save both as compressed .npz with 32x32 in filename
    np.savez_compressed("data/caltech101_train_90_32.npz", X=X_train_new, y=y_train_new)
    np.savez_compressed("data/caltech101_test_10_32.npz",  X=X_test_new,  y=y_test_new)

    print("✅ Done — saved:")
    print("  data/caltech101_train_90_32.npz")
    print("  data/caltech101_test_10_32.npz")

if __name__ == "__main__":
    main()