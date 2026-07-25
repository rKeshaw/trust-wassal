#!/usr/bin/env python3
# prepare_stl10_channel_first.py
# Save STL-10 as channel-first float arrays (N, C, H, W) scaled to [0,1]

import os
import numpy as np
from torchvision.datasets import STL10
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

def pil_to_chw_float32(img):
    """PIL Image -> (C, H, W) float32 in [0,1]."""
    arr = np.array(img)  # H x W x C (uint8)
    if arr.ndim == 2:  # grayscale -> convert to 3 channels
        arr = np.stack([arr, arr, arr], axis=-1)
    # convert to float in [0,1], transpose to C,H,W
    arr = arr.astype(np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)  # C, H, W
    return arr

def prepare_and_save(datadir="data", out_name="stl10_raw_channel_first.npz", download=True):
    os.makedirs(datadir, exist_ok=True)

    print("Loading STL-10 via torchvision (may download if missing)...")
    train = STL10(root=datadir, split="train", download=download, transform=None)
    test = STL10(root=datadir, split="test", download=download, transform=None)
    unlabeled = STL10(root=datadir, split="unlabeled", download=download, transform=None)

    print("Converting train split to (N, C, H, W) float32 ...")
    X_train = np.zeros((len(train), 3, 96, 96), dtype=np.float32)
    y_train = np.zeros((len(train),), dtype=np.int64)
    for i, (img, lbl) in enumerate(tqdm(train)):
        X_train[i] = pil_to_chw_float32(img)
        y_train[i] = int(lbl)

    print("Converting test split to (N, C, H, W) float32 ...")
    X_test = np.zeros((len(test), 3, 96, 96), dtype=np.float32)
    y_test = np.zeros((len(test),), dtype=np.int64)
    for i, (img, lbl) in enumerate(tqdm(test)):
        X_test[i] = pil_to_chw_float32(img)
        y_test[i] = int(lbl)

    print("Converting unlabeled split to (N, C, H, W) float32 ...")
    X_unlabeled = np.zeros((len(unlabeled), 3, 96, 96), dtype=np.float32)
    # labels for unlabeled are set to -1 placeholder
    y_unlabeled = np.full((len(unlabeled),), -1, dtype=np.int64)
    for i, (img, lbl) in enumerate(tqdm(unlabeled)):
        X_unlabeled[i] = pil_to_chw_float32(img)
        # unlabeled dataset may contain noisy labels; we intentionally ignore them and set -1

    out_path = os.path.join(datadir, out_name)
    print(f"Saving compressed npz to: {out_path} (this may take a moment)...")
    np.savez_compressed(
        out_path,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        X_unlabeled=X_unlabeled,
        y_unlabeled=y_unlabeled,
    )

    # Sanity checks
    print("Saved. Sanity check:")
    print("  X_train:", X_train.shape, X_train.dtype, "min/max:", X_train.min(), X_train.max())
    print("  y_train:", y_train.shape, "unique labels:", np.unique(y_train))
    print("  X_test:", X_test.shape, X_test.dtype, "min/max:", X_test.min(), X_test.max())
    print("  y_test:", y_test.shape, "unique labels:", np.unique(y_test))
    print("  X_unlabeled:", X_unlabeled.shape, X_unlabeled.dtype, "min/max:", X_unlabeled.min(), X_unlabeled.max())
    print("  y_unlabeled placeholder shape:", y_unlabeled.shape)

if __name__ == "__main__":
    prepare_and_save(datadir="data", out_name="stl10_raw_channel_first.npz", download=True)
