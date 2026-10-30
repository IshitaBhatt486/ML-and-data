"""Dataset input pipelines and training augmentation."""

from pathlib import Path
from typing import Tuple

import tensorflow as tf

AUTOTUNE = tf.data.AUTOTUNE


def load_datasets(data_dir: str, image_size: int = 64, batch_size: int = 32,
                  validation_split: float = 0.2, seed: int = 42) -> Tuple[tf.data.Dataset, tf.data.Dataset, list]:
    """Load a folder-per-class dataset into reproducible train/validation sets."""
    path = Path(data_dir)
    if not path.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {path}. Expected one subdirectory per breed.")
    options = dict(directory=str(path), validation_split=validation_split, seed=seed,
                   image_size=(image_size, image_size), batch_size=batch_size, label_mode="int")
    train_ds = tf.keras.utils.image_dataset_from_directory(subset="training", shuffle=True, **options)
    validation_ds = tf.keras.utils.image_dataset_from_directory(subset="validation", shuffle=False, **options)
    return train_ds.prefetch(AUTOTUNE), validation_ds.prefetch(AUTOTUNE), train_ds.class_names


def augmentation() -> tf.keras.Sequential:
    """Return conservative augmentations suitable for animal photographs."""
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.04),
        tf.keras.layers.RandomTranslation(0.1, 0.1),
        tf.keras.layers.RandomZoom(0.1),
    ], name="augmentation")
