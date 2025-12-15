# src/features/preprocessing.py
from pathlib import Path
from typing import Tuple, List

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_image_paths_and_labels(
    data_raw_dir: Path,
) -> Tuple[List[Path], np.ndarray, List[str]]:
    """
    Asumsi struktur:
      data/raw/
        tulip/
          xxx.jpg
        rose/
          yyy.jpg
        ...
    Label = nama folder.
    """
    image_paths: List[Path] = []
    labels: List[str] = []

    for class_dir in sorted(data_raw_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        label = class_dir.name
        for img_path in class_dir.glob("*.jpg"):
            image_paths.append(img_path)
            labels.append(label)

    labels_arr = np.array(labels)
    classes = sorted(list(set(labels)))
    return image_paths, labels_arr, classes


def image_to_array(
    img_path: Path,
    image_size: Tuple[int, int] = (128, 128),
) -> np.ndarray:
    """
    Baca gambar, resize, convert ke array (flatten).
    """
    img = Image.open(img_path).convert("RGB")
    img = img.resize(image_size)
    arr = np.asarray(img, dtype=np.float32) / 255.0  # normalisasi 0-1
    return arr.reshape(-1)  # flatten (H*W*3,)


def build_dataset(
    data_raw_dir: Path,
    image_size: Tuple[int, int] = (128, 128),
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Load semua gambar, ekstrak fitur (flatten), split train/test, dan scaling.
    Return:
      X_train_scaled, X_test_scaled, y_train, y_test, scaler, classes
    """
    image_paths, labels, classes = load_image_paths_and_labels(data_raw_dir)

    X = np.stack([image_to_array(p, image_size=image_size) for p in image_paths], axis=0)
    y = labels

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, classes
