#!/usr/bin/env python3
"""
src/fer/train_fer.py

Facial Expression Recognition (FER) training script.
Uses EfficientNetB0 backbone, corrected preprocessing, mild augmentation,
class-balanced weights, staged fine-tuning, and early stopping.

Expected dataset layout (relative to repo root):
data/FER-2013/
    train/
        angry/
        disgust/
        fear/
        happy/
        neutral/
        sad/
        surprise/
    test/
        angry/
        disgust/
        fear/
        happy/
        neutral/
        sad/
        surprise/
"""

import os
import json
import numpy as np
from pathlib import Path
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, TensorBoard
from tensorflow.keras.applications import EfficientNetB0
import matplotlib.pyplot as plt

# -----------------------
# Configuration & Paths
# -----------------------
BASE_DIR = Path(__file__).resolve().parents[2]
DATASET_PATH = BASE_DIR / 'data' / 'FER-2013'
TRAIN_DIR = DATASET_PATH / 'train'
TEST_DIR = DATASET_PATH / 'test'

MODEL_DIR = BASE_DIR / 'models' / 'Fer_model'
REPORT_DIR = MODEL_DIR / 'report_and_log'
LOGS_DIR = REPORT_DIR / 'logs'

IMG_SIZE = (96, 96)
BATCH_SIZE = 32
INITIAL_EPOCHS = 20
FINE_TUNE_EPOCHS = 25
INITIAL_LR = 1e-4          # Transfer learning initial LR
FINE_TUNE_LR = 1e-5         # Lower LR for fine-tuning
UNFREEZE_LAST_N = 200      # Layers from end of base model to unfreeze
RANDOM_SEED = 42

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

SAVE_MODEL_PATH = MODEL_DIR / 'fer_model.keras'
CSV_LOG = REPORT_DIR / 'fer_training.csv'
HISTORY_JSON = REPORT_DIR / 'fer_history.json'
HISTORY_PLOT = REPORT_DIR / 'fer_history.png'
REPORT_TXT = REPORT_DIR / 'fer_report.txt'

tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def train():
    if not TRAIN_DIR.exists() or not TEST_DIR.exists():
        raise FileNotFoundError(
            f"FER dataset not found at {DATASET_PATH}. Expected 'train' and 'test' subdirectories.\n"
            "Place FER images organized by class in data/FER-2013/train and data/FER-2013/test."
        )

    # -----------------------
    # Data Generators (Mild augmentation preserving facial cues)
    # -----------------------
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=8,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.05,
        zoom_range=0.08,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    valid_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_gen = train_datagen.flow_from_directory(
        str(TRAIN_DIR),
        target_size=IMG_SIZE,
        color_mode='rgb',
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True,
        seed=RANDOM_SEED
    )

    valid_gen = valid_datagen.flow_from_directory(
        str(TEST_DIR),
        target_size=IMG_SIZE,
        color_mode='rgb',
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    print("\nClass indices (folder name -> class index):")
    print(train_gen.class_indices)
    num_classes = train_gen.num_classes
    print(f"Detected {train_gen.samples} training images, {valid_gen.samples} validation images, {num_classes} classes.")

    # -----------------------
    # Compute class weights to correct severe class imbalance
    # -----------------------
    y_train = train_gen.classes
    cw = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights = {i: float(w) for i, w in enumerate(cw)}
    print("\nComputed class weights:")
    print(class_weights)

    # -----------------------
    # Build Transfer Learning Model
    # -----------------------
    base_model = EfficientNetB0(
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

    x = base_model.output
    x = layers.GlobalAveragePooling2D(name='gap')(x)
    x = layers.Dense(512, activation='relu', name='fc1')(x)
    x = layers.BatchNormalization(name='bn1')(x)
    x = layers.Dropout(0.4, name='drop1')(x)
    x = layers.Dense(256, activation='relu', name='fc2')(x)
    x = layers.Dropout(0.3, name='drop2')(x)
    output = layers.Dense(num_classes, activation='softmax', name='predictions')(x)

    model = models.Model(inputs=base_model.input, outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=INITIAL_LR),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    callbacks = [
        ModelCheckpoint(str(SAVE_MODEL_PATH), monitor='val_accuracy', verbose=1, save_best_only=True),
        CSVLogger(str(CSV_LOG)),
        TensorBoard(log_dir=str(LOGS_DIR)),
        EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1)
    ]

    steps_per_epoch = max(1, train_gen.samples // BATCH_SIZE)
    validation_steps = max(1, valid_gen.samples // BATCH_SIZE)

    # Stage 1: Train classification head
    print("\n--- STAGE 1: Training Classification Head ---")
    history1 = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=INITIAL_EPOCHS,
        validation_data=valid_gen,
        validation_steps=validation_steps,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=2
    )

    # Stage 2: Fine-tune backbone
    print("\n--- STAGE 2: Fine-Tuning Backbone ---")
    total_layers = len(base_model.layers)
    n = min(UNFREEZE_LAST_N, total_layers)
    for layer in base_model.layers[-n:]:
        layer.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=FINE_TUNE_LR),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    initial_epoch = history1.epoch[-1] + 1 if hasattr(history1, 'epoch') and history1.epoch else INITIAL_EPOCHS
    total_epochs = INITIAL_EPOCHS + FINE_TUNE_EPOCHS

    history2 = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=total_epochs,
        initial_epoch=initial_epoch,
        validation_data=valid_gen,
        validation_steps=validation_steps,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=2
    )

    # Combine histories and save reports
    history = history1
    for k, v in history2.history.items():
        history.history.setdefault(k, []).extend(v)

    try:
        with open(HISTORY_JSON, 'w') as f:
            json.dump(history.history, f)
    except Exception as e:
        print("Warning: failed to write history JSON:", e)

    try:
        plt.figure(figsize=(9, 5))
        if 'accuracy' in history.history:
            plt.plot(history.history['accuracy'], label='train_acc')
        if 'val_accuracy' in history.history:
            plt.plot(history.history['val_accuracy'], label='val_acc')
        if 'loss' in history.history:
            plt.plot(history.history['loss'], label='train_loss')
        if 'val_loss' in history.history:
            plt.plot(history.history['val_loss'], label='val_loss')
        plt.legend()
        plt.title('FER Training History')
        plt.xlabel('Epoch')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(HISTORY_PLOT)
        plt.close()
    except Exception as e:
        print("Warning: failed to save training plot:", e)

    try:
        final_val_acc = history.history.get('val_accuracy', [None])[-1]
        final_val_loss = history.history.get('val_loss', [None])[-1]
        with open(REPORT_TXT, 'w') as f:
            f.write("FER CNN Training Report\n")
            f.write(f"Model saved at: {SAVE_MODEL_PATH}\n")
            f.write(f"Final Validation Accuracy: {final_val_acc}\n")
            f.write(f"Final Validation Loss: {final_val_loss}\n")
            f.write(f"Class indices: {train_gen.class_indices}\n")
    except Exception as e:
        print("Warning: failed to write report:", e)

    print("\nTraining complete. Model saved to:", SAVE_MODEL_PATH)


if __name__ == '__main__':
    train()
