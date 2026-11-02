"""
GeoGuard Landslide Detection System
Convolutional Neural Network Architecture for Multispectral Patch Classification.

Design:
- Input: (128, 128, 4) [Sentinel-2 B2, B3, B4, B8]
- 3 Convolutional blocks with BatchNormalization, ReLU, and MaxPooling
- Spatial Regularization: Dropout (0.2 - 0.3)
- GlobalAveragePooling2D: Preserves spatial information while drastically reducing
  parameters compared to Flatten, preventing overfitting on small/medium datasets.
- Classification Head: Dense(64) -> Dropout(0.4) -> Dense(1, activation='sigmoid')
- Output: Calibrated scalar probability P(Landslide) in [0.0, 1.0]
"""

import logging
from typing import Tuple, Optional
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

logger = logging.getLogger(__name__)


def build_landslide_cnn(
    input_shape: Tuple[int, int, int] = (128, 128, 4),
    dropout_rate: float = 0.25,
    l2_reg: float = 1e-4,
    learning_rate: float = 1e-3,
) -> tf.keras.Model:
    """
    Constructs and compiles the lightweight custom 4-channel CNN.

    Args:
        input_shape: Dimensions of patch tensor (height, width, channels).
        dropout_rate: Dropout rate for regularization.
        l2_reg: L2 kernel regularization coefficient.
        learning_rate: Learning rate for Adam optimizer.

    Returns:
        Compiled tf.keras.Model.
    """
    reg = regularizers.l2(l2_reg) if l2_reg > 0 else None

    inputs = layers.Input(shape=input_shape, name="satellite_patch_input")

    # --- Block 1: Low-level spatial features (edges, soil boundaries) ---
    x = layers.Conv2D(32, (3, 3), padding="same", kernel_regularizer=reg, name="conv1_1")(inputs)
    x = layers.BatchNormalization(name="bn1_1")(x)
    x = layers.Activation("relu", name="relu1_1")(x)
    x = layers.Conv2D(32, (3, 3), padding="same", kernel_regularizer=reg, name="conv1_2")(x)
    x = layers.BatchNormalization(name="bn1_2")(x)
    x = layers.Activation("relu", name="relu1_2")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool1")(x)  # -> 64x64
    x = layers.Dropout(dropout_rate, name="drop1")(x)

    # --- Block 2: Mid-level spectral-spatial textures (vegetation canopy vs bare earth) ---
    x = layers.Conv2D(64, (3, 3), padding="same", kernel_regularizer=reg, name="conv2_1")(x)
    x = layers.BatchNormalization(name="bn2_1")(x)
    x = layers.Activation("relu", name="relu2_1")(x)
    x = layers.Conv2D(64, (3, 3), padding="same", kernel_regularizer=reg, name="conv2_2")(x)
    x = layers.BatchNormalization(name="bn2_2")(x)
    x = layers.Activation("relu", name="relu2_2")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool2")(x)  # -> 32x32
    x = layers.Dropout(dropout_rate + 0.05, name="drop2")(x)

    # --- Block 3: High-level landslide scar patterns & context ---
    x = layers.Conv2D(128, (3, 3), padding="same", kernel_regularizer=reg, name="conv3_1")(x)
    x = layers.BatchNormalization(name="bn3_1")(x)
    x = layers.Activation("relu", name="relu3_1")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool3")(x)  # -> 16x16
    x = layers.Dropout(dropout_rate + 0.1, name="drop3")(x)

    # --- Head: Global spatial aggregation and binary classification ---
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(64, kernel_regularizer=reg, name="dense1")(x)
    x = layers.BatchNormalization(name="bn_dense")(x)
    x = layers.Activation("relu", name="relu_dense")(x)
    x = layers.Dropout(0.4, name="drop_head")(x)

    outputs = layers.Dense(1, activation="sigmoid", name="landslide_probability")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="GeoGuard_Landslide_CNN_v1")

    # Compile with Adam and Binary Crossentropy
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )

    logger.info(f"Built {model.name}: {model.count_params():,} total parameters")
    return model
