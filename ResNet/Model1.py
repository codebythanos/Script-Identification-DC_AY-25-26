# Generated from: dc_resnet(data_aug).ipynb
# Converted at: 2026-04-28T14:56:44.541Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, classification_report

train_dir = "12-way script classification dataset/train_1800"
test_dir = "12-way script classification dataset/test_478"
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 12
EPOCHS = 25



def count_images(folder):
    total = 0
    for root, dirs, files in os.walk(folder):
        total += len([f for f in files if f.lower().endswith(('png','jpg','jpeg'))])
    return total

total_train_images = count_images(train_dir)
total_test_images = count_images(test_dir)

print("Original Train Images:", total_train_images)
print("Test Images:", total_test_images)

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=1,   # IMPORTANT: use 1 for splitting
    shuffle=True
)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = train_ds.class_names

def vertical_split(image, label):
    image = tf.squeeze(image, axis=0)   # remove batch dim

    width = tf.shape(image)[1]
    mid = width // 2

    left = image[:, :mid, :]
    right = image[:, mid:, :]

    left = tf.image.resize(left, (IMG_SIZE, IMG_SIZE))
    right = tf.image.resize(right, (IMG_SIZE, IMG_SIZE))

    images = tf.stack([left, right])
    labels = tf.stack([label[0], label[0]])

    return images, labels


train_ds = train_ds.map(vertical_split)
train_ds = train_ds.unbatch()
train_ds = train_ds.batch(BATCH_SIZE)

new_train_size = total_train_images * 2
steps_per_epoch = new_train_size // BATCH_SIZE

validation_steps = total_test_images // BATCH_SIZE

print("Steps per epoch:", steps_per_epoch)

train_ds = train_ds.repeat()
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.25),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2),
    layers.RandomBrightness(0.2),
])

base_model = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

base_model.trainable = False

inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

x = data_augmentation(inputs)
x = preprocess_input(x)

x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)

outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

model = models.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ReduceLROnPlateau(patience=3, factor=0.3)
]

history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    callbacks=callbacks,
    verbose=1
)

base_model.trainable = True

for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history_fine = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=15,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    callbacks=callbacks,
    verbose=1
)

y_true = []
y_pred = []

for images, labels in test_ds.take(validation_steps):
    preds = model.predict(images)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=class_names,
            yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

print(classification_report(y_true, y_pred, target_names=class_names))

model.save("bharat_script_resnet50_model.keras")
