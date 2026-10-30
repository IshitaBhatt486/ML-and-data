"""Custom ResNet-50 model definition."""

import tensorflow as tf

layers = tf.keras.layers


def identity_block(inputs, filters, kernel_size: int = 3):
    """Residual block that preserves input dimensions."""
    first, second, third = filters
    x = layers.Conv2D(first, 1, use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(second, kernel_size, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(third, 1, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    return layers.Activation("relu")(layers.Add()([x, inputs]))


def convolutional_block(inputs, filters, strides: int = 2, kernel_size: int = 3):
    """Residual block with a projection shortcut for shape changes."""
    first, second, third = filters
    x = layers.Conv2D(first, 1, strides=strides, use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(second, kernel_size, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(third, 1, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    shortcut = layers.Conv2D(third, 1, strides=strides, use_bias=False)(inputs)
    shortcut = layers.BatchNormalization()(shortcut)
    return layers.Activation("relu")(layers.Add()([x, shortcut]))


def build_resnet50(image_size: int, num_classes: int) -> tf.keras.Model:
    """Build an uncompiled ResNet-50 classifier with in-model rescaling."""
    inputs = layers.Input(shape=(image_size, image_size, 3))
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(64, 7, strides=2, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPool2D(3, strides=2, padding="same")(x)
    for filters, blocks, first_stride in (([64, 64, 256], 3, 1), ([128, 128, 512], 4, 2), ([256, 256, 1024], 6, 2), ([512, 512, 2048], 3, 2)):
        x = convolutional_block(x, filters, strides=first_stride)
        for _ in range(blocks - 1):
            x = identity_block(x, filters)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    return tf.keras.Model(inputs, outputs, name="bovine_resnet50")
