#!/usr/bin/env python3
"""
train_fer_fixed.py

Single-file FER training script (EfficientNetB0 backbone, corrected preprocessing,
mild augmentation, class weights, staged fine-tuning).

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
        ...
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
# Configuration
# -----------------------
BASE_DIR = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd()
DATASET_PATH = BASE_DIR / 'data' / 'FER-2013'   # <-- adjust if needed
TRAIN_DIR = DATASET_PATH / 'train'
TEST_DIR = DATASET_PATH / 'test'

MODEL_DIR = BASE_DIR / 'models' / 'fer_model'
REPORT_DIR = MODEL_DIR / 'report_and_log'
LOGS_DIR = REPORT_DIR / 'logs'

IMG_SIZE = (96, 96)
BATCH_SIZE = 32
INITIAL_EPOCHS = 20
FINE_TUNE_EPOCHS = 25
INITIAL_LR = 1e-4          # lower initial LR for transfer learning
FINE_TUNE_LR = 1e-5
UNFREEZE_LAST_N = 200      # how many layers from the end to unfreeze for fine-tuning
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

# -----------------------
# Sanity checks
# -----------------------
if not TRAIN_DIR.exists() or not TEST_DIR.exists():
    raise FileNotFoundError(
        f"FER dataset not found at {DATASET_PATH}. Expected 'train' and 'test' directories."
        "\nPut FER images organized by class in data/FER-2013/train and data/FER-2013/test."
    )

# -----------------------
# Data generators
# -----------------------
# Use mild augmentation only — don't destroy facial structure.
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=8,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=0.05,
    zoom_range=0.08,
    horizontal_flip=True,
    fill_mode='nearest'
)

valid_datagen = ImageDataGenerator(rescale=1.0/255.0)

# flow_from_directory: request rgb. PIL will convert grayscale -> RGB automatically.
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

NUM_CLASSES = train_gen.num_classes
print(f"Detected {train_gen.samples} training images, {valid_gen.samples} validation images, {NUM_CLASSES} classes.")

# -----------------------
# Compute class weights to handle imbalance
# -----------------------
y_train = train_gen.classes  # numeric labels per sample in the order of the generator
cw = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = {i: float(w) for i, w in enumerate(cw)}
print("\nComputed class weights (applied to training):")
print(class_weights)

# -----------------------
# Build model
# -----------------------
base_model = EfficientNetB0(
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # freeze backbone for initial stage

x = base_model.output
x = layers.GlobalAveragePooling2D(name='gap')(x)
x = layers.Dense(512, activation='relu', name='fc1')(x)
x = layers.BatchNormalization(name='bn1')(x)
x = layers.Dropout(0.4, name='drop1')(x)
x = layers.Dense(256, activation='relu', name='fc2')(x)
x = layers.Dropout(0.3, name='drop2')(x)
output = layers.Dense(NUM_CLASSES, activation='softmax', name='predictions')(x)

model = models.Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=INITIAL_LR),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# -----------------------
# Callbacks
# -----------------------
callbacks = [
    ModelCheckpoint(str(SAVE_MODEL_PATH), monitor='val_accuracy', verbose=1, save_best_only=True),
    CSVLogger(str(CSV_LOG)),
    TensorBoard(log_dir=str(LOGS_DIR)),
    EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1)
]

# Steps
steps_per_epoch = max(1, train_gen.samples // BATCH_SIZE)
validation_steps = max(1, valid_gen.samples // BATCH_SIZE)

# -----------------------
# Initial training (top layers only)
# -----------------------
print("\n--- STARTING INITIAL TRAINING (top layers only) ---")
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

# -----------------------
# Fine-tuning: unfreeze last N layers of base_model
# -----------------------
print("\n--- PREPARING FINE-TUNING ---")
# Unfreeze the last UNFREEZE_LAST_N layers (if UNFREEZE_LAST_N > number of layers, unfreeze all)
total_layers = len(base_model.layers)
n = min(UNFREEZE_LAST_N, total_layers)
print(f"Total layers in base model: {total_layers}. Unfreezing last {n} layers for fine-tuning.")

for layer in base_model.layers[-n:]:
    layer.trainable = True

# Recompile with lower LR
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=FINE_TUNE_LR),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Continue training from where we left off
initial_epoch = history1.epoch[-1] + 1 if hasattr(history1, 'epoch') and history1.epoch else INITIAL_EPOCHS
total_epochs = INITIAL_EPOCHS + FINE_TUNE_EPOCHS

print(f"Resuming training: initial_epoch={initial_epoch}, total_epochs={total_epochs}")

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

# -----------------------
# Combine histories
# -----------------------
history = history1
for k, v in history2.history.items():
    history.history.setdefault(k, []).extend(v)

# -----------------------
# Save history & plot
# -----------------------
try:
    with open(HISTORY_JSON, 'w') as f:
        json.dump(history.history, f)
    print(f"Wrote history JSON to {HISTORY_JSON}")
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
    plt.title('Training history')
    plt.xlabel('Epoch')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(HISTORY_PLOT)
    plt.close()
    print(f"Wrote training plot to {HISTORY_PLOT}")
except Exception as e:
    print("Warning: failed to save training plot:", e)

# -----------------------
# Save model and short report
# -----------------------
try:
    model.save(str(SAVE_MODEL_PATH))
    print(f"Saved model to {SAVE_MODEL_PATH}")
except Exception as e:
    print("Warning: failed to save model:", e)

try:
    final_val_acc = history.history.get('val_accuracy', [None])[-1]
    final_val_loss = history.history.get('val_loss', [None])[-1]
    with open(REPORT_TXT, 'w') as f:
        f.write("FER training report\n")
        f.write(f"Model path: {SAVE_MODEL_PATH}\n")
        f.write(f"Final validation loss: {final_val_loss}\n")
        f.write(f"Final validation accuracy: {final_val_acc}\n")
        f.write("\nSaved artifacts:\n")
        f.write(f" - history JSON: {HISTORY_JSON}\n")
        f.write(f" - CSV log: {CSV_LOG}\n")
        f.write(f" - plot: {HISTORY_PLOT}\n")
        f.write(f" - tensorboard logs: {LOGS_DIR}\n")
    print(f"Wrote report to {REPORT_TXT}")
except Exception as e:
    print("Warning: failed to write report:", e)

print("\nDone. All artifacts saved under:", MODEL_DIR)
