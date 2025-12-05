import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import os
import json
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, TensorBoard
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input

# --------------------
# Paths (use repo-relative defaults)
# --------------------
BASE_DIR = os.path.join(os.path.dirname(__file__), '..', '..')
DATASET_PATH = os.path.join(BASE_DIR, 'data', 'FER-2013')  # repo-relative default

train_dir = os.path.join(DATASET_PATH, 'train')
test_dir = os.path.join(DATASET_PATH, 'test')
FER_MODEL_DIR = os.path.join(BASE_DIR, 'models', 'fer_model')
os.makedirs(FER_MODEL_DIR, exist_ok=True)
SAVE_MODEL = os.path.join(FER_MODEL_DIR, 'fer_model.keras')

# fail fast with a helpful message if data not found
if not os.path.exists(train_dir) or not os.path.exists(test_dir):
    raise FileNotFoundError(
        f"FER dataset not found at {DATASET_PATH}. Expected subfolders 'train' and 'test'.\n"
        "Place FER-2013 under 'data/FER-2013' relative to repo root, or update DATASET_PATH in this script."
    )


# --------------------
# Image Generators (transfer learning + stronger augmentation)
# --------------------
# Use a larger input for pretrained backbone and convert grayscale -> RGB in preprocessing
img_size = (96, 96)
batch_size = 32

def to_rgb_and_preprocess(img):
    # ImageDataGenerator passes images as float32 arrays in [0,255] or after rescale.
    # If generator returns a 1-channel image, repeat channels to make 3 channels.
    if img.ndim == 2:
        img = np.expand_dims(img, axis=-1)
    if img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=2)
    return preprocess_input(img)

train_gen = ImageDataGenerator(
    preprocessing_function=to_rgb_and_preprocess,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    zoom_range=0.3,
    horizontal_flip=True,
    brightness_range=(0.7, 1.3),
    fill_mode='reflect'
)

test_gen = ImageDataGenerator(preprocessing_function=to_rgb_and_preprocess)

train_data = train_gen.flow_from_directory(
    train_dir,
    target_size=img_size,
    color_mode='rgb',
    batch_size=batch_size,
    class_mode='categorical'
)

test_data = test_gen.flow_from_directory(
    test_dir,
    target_size=img_size,
    color_mode='rgb',
    batch_size=batch_size,
    class_mode='categorical'
)

# --------------------
# Build model using pretrained backbone (EfficientNetB0 - stronger than MobileNetV2)
# --------------------
base_model = EfficientNetB0(
    input_shape=(img_size[0], img_size[1], 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # freeze backbone for initial training

x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(512, activation='relu')(x)  # larger top layer
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.3)(x)
output = layers.Dense(7, activation='softmax')(x)

model = models.Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# --------------------
# Train
# --------------------
# --------------------
# Callbacks: save best model, CSV log, tensorboard, early stopping
# --------------------
REPORT_DIR = os.path.join(FER_MODEL_DIR, 'report_and_log')
LOGS_DIR = os.path.join(REPORT_DIR, 'logs')
CSV_LOG = os.path.join(REPORT_DIR, 'fer_training.csv')
HISTORY_JSON = os.path.join(REPORT_DIR, 'fer_history.json')
HISTORY_PLOT = os.path.join(REPORT_DIR, 'fer_history.png')
REPORT_TXT = os.path.join(REPORT_DIR, 'fer_report.txt')

os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

callbacks = [
    ModelCheckpoint(SAVE_MODEL, monitor='val_accuracy', save_best_only=True, verbose=1),
    CSVLogger(CSV_LOG),
    TensorBoard(log_dir=LOGS_DIR),
    EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1),
]

# Two-stage training: initial training of top layers, then extensive fine-tuning
initial_epochs = 20
history1 = model.fit(
    train_data,
    validation_data=test_data,
    epochs=initial_epochs,
    callbacks=callbacks
)

# Fine-tune: unfreeze more layers in the base_model and continue training longer
fine_tune_at = -50  # unfreeze last ~50 layers (more aggressive)
for layer in base_model.layers[fine_tune_at:]:
    layer.trainable = True

# recompile with lower LR
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

fine_tune_epochs = 25  # more epochs for fine-tuning
total_epochs = initial_epochs + fine_tune_epochs
history2 = model.fit(
    train_data,
    validation_data=test_data,
    epochs=total_epochs,
    initial_epoch=history1.epoch[-1] + 1 if hasattr(history1, 'epoch') and history1.epoch else initial_epochs,
    callbacks=callbacks
)

# Combine histories for reporting
history = history1
for k, v in history2.history.items():
    history.history.setdefault(k, []).extend(v)

# --------------------
# Save human-readable training outputs: JSON, plot, and summary report
# --------------------
# Save history to JSON
try:
    with open(HISTORY_JSON, 'w') as f:
        json.dump(history.history, f)
except Exception as e:
    print('Warning: failed to write history JSON:', e)

# Plot training curves
try:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 4))
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
except Exception as e:
    print('Warning: failed to save training plot:', e)

# Write a short report
try:
    final_acc = history.history.get('val_accuracy', [None])[-1]
    final_loss = history.history.get('val_loss', [None])[-1]
    with open(REPORT_TXT, 'w') as f:
        f.write(f'FER training report\n')
        f.write(f'Model path: {SAVE_MODEL}\n')
        f.write(f'Final validation loss: {final_loss}\n')
        f.write(f'Final validation accuracy: {final_acc}\n')
        f.write('\nSaved artifacts:\n')
        f.write(f' - history JSON: {HISTORY_JSON}\n')
        f.write(f' - CSV log: {CSV_LOG}\n')
        f.write(f' - plot: {HISTORY_PLOT}\n')
        f.write(f' - tensorboard logs: {LOGS_DIR}\n')
except Exception as e:
    print('Warning: failed to write report:', e)

# --------------------
# Save model
# --------------------
model.save(SAVE_MODEL)

print(f"Model saved as {SAVE_MODEL}")
print(f"All artifacts saved to: {FER_MODEL_DIR}")
