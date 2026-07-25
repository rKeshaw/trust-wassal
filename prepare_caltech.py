#!/usr/bin/env python3
# prepare_caltech_from_tfds_split90_10_32x32.py
# Download Caltech-101 (TFDS), resize to 32×32, merge all samples,
# then stratified 90/10 split → save as .npz (C,H,W float32 in [0,1]).

import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import StratifiedShuffleSplit
import tensorflow_datasets as tfds

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

def extract_all_images(tfds_split):
    images, labels = [], []
    for img, lbl in tqdm(tfds_split):
        img_pil = Image.fromarray(img.numpy())
        img_pil = pil_resize_32x32(img_pil, (32, 32))  # Changed to 32×32
        arr = pil_to_chw_float32(img_pil)
        images.append(arr)
        labels.append(int(lbl.numpy()))
    return np.stack(images, axis=0), np.array(labels, dtype=np.int64)

def main():
    os.makedirs("data", exist_ok=True)
    print("Downloading / loading Caltech-101 from TensorFlow Datasets...")
    splits = tfds.load("caltech101", split=["train", "test"],
                       as_supervised=True, data_dir="./data/tfds", download=True)
    train_split, test_split = splits

    print("Converting full dataset to 32×32 numpy arrays (this may take time)...")
    X_train, y_train = extract_all_images(train_split)
    X_test, y_test = extract_all_images(test_split)

    # Merge all
    X_all = np.concatenate([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])
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