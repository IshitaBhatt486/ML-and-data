"""Evaluate a saved model on the deterministic validation split."""

import argparse

import tensorflow as tf

from src.data import load_datasets


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def main():
    args = parse_args()
    _, validation_ds, class_names = load_datasets(args.data_dir, args.image_size, args.batch_size)
    model = tf.keras.models.load_model(args.model)
    loss, accuracy = model.evaluate(validation_ds, verbose=0)
    print(f"Classes: {len(class_names)}")
    print(f"Validation loss: {loss:.4f}")
    print(f"Validation accuracy: {accuracy:.2%}")


if __name__ == "__main__":
    main()
