"""Train the bovine-breed classifier from a folder-per-class dataset."""

import argparse
import json
from pathlib import Path

import tensorflow as tf

from src.data import augmentation, load_datasets
from src.model import build_resnet50


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    return parser.parse_args()


def main():
    args = parse_args()
    tf.keras.utils.set_random_seed(42)
    train_ds, validation_ds, class_names = load_datasets(args.data_dir, args.image_size, args.batch_size)
    if len(class_names) < 2:
        raise ValueError("At least two class folders are required for classification.")
    model = tf.keras.Sequential([augmentation(), build_resnet50(args.image_size, len(class_names))], name="bovine_classifier")
    model.compile(optimizer=tf.keras.optimizers.SGD(args.learning_rate, momentum=0.9),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=["accuracy"])
    models_dir, logs_dir = Path("models"), Path("logs")
    models_dir.mkdir(exist_ok=True)
    logs_dir.mkdir(exist_ok=True)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(models_dir / "best_model.keras", monitor="val_accuracy", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=3, factor=0.2),
        tf.keras.callbacks.TensorBoard(log_dir=str(logs_dir)),
    ]
    history = model.fit(train_ds, validation_data=validation_ds, epochs=args.epochs, callbacks=callbacks)
    model.save(models_dir / "final_model.keras")
    (models_dir / "class_names.json").write_text(json.dumps(class_names, indent=2))
    (models_dir / "history.json").write_text(json.dumps(history.history, indent=2))


if __name__ == "__main__":
    main()
