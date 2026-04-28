# Generated from: 82.97.ipynb
# Converted at: 2026-04-28T14:52:19.720Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

from google.colab import drive
drive.mount('/content/drive')

import zipfile
import os

zip_path     = '/content/drive/MyDrive/12-way script classification dataset-20260320T081258Z-3-001.zip'
extract_path = '/content/dataset'

os.makedirs(extract_path, exist_ok=True)

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

print("Extraction done")
print(os.listdir(extract_path))

!pip install tensorflow keras-hub scikit-learn seaborn matplotlib pillow -q

import os

extract_path = '/content/dataset'

for root, dirs, files in os.walk(extract_path):
    level = root.replace(extract_path, '').count(os.sep)
    if level < 3:
        print(' ' * 2 * level + os.path.basename(root) + '/')

train_dir = '/content/dataset/12-way script classification dataset/train_1800'
test_dir  = '/content/dataset/12-way script classification dataset/test_478'

print("Train classes:", sorted(os.listdir(train_dir)))
print("Test classes :", sorted(os.listdir(test_dir)))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras_hub
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print("GPU enabled:", gpus)
else:
    print("No GPU - go to Runtime > Change runtime type > T4 GPU")

print("TensorFlow:", tf.__version__)
print("Keras Hub :", keras_hub.__version__)

from tensorflow.keras import mixed_precision

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
print("Mixed precision:", policy.name)

train_dir   = '/content/dataset/12-way script classification dataset/train_1800'
test_dir    = '/content/dataset/12-way script classification dataset/test_478'

IMG_SIZE    = 224
BATCH_SIZE  = 32
NUM_CLASSES = 12
SEED        = 42
LR          = 1e-3

np.random.seed(SEED)
tf.random.set_seed(SEED)

AUTOTUNE = tf.data.AUTOTUNE

def load_and_preprocess(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32) / 255.0
    return img, label

def build_dataset(data_dir, training=True):
    data_dir   = Path(data_dir)
    class_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    cls_names  = [d.name for d in class_dirs]
    cls_to_idx = {name: idx for idx, name in enumerate(cls_names)}

    all_paths, all_labels = [], []

    for class_dir in class_dirs:
        label = cls_to_idx[class_dir.name]
        for img_path in class_dir.glob('*'):
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                all_paths.append(str(img_path))
                all_labels.append(label)

    all_paths  = np.array(all_paths)
    all_labels = np.array(all_labels)

    idx        = np.random.permutation(len(all_paths))
    all_paths  = all_paths[idx]
    all_labels = all_labels[idx]

    if training:
        split      = int(0.9 * len(all_paths))
        tr_paths   = all_paths[:split]
        val_paths  = all_paths[split:]
        tr_labels  = all_labels[:split]
        val_labels = all_labels[split:]

        tr_labels_oh  = tf.keras.utils.to_categorical(tr_labels,  NUM_CLASSES)
        val_labels_oh = tf.keras.utils.to_categorical(val_labels, NUM_CLASSES)

        train_ds = (tf.data.Dataset.from_tensor_slices((tr_paths, tr_labels_oh))
                    .shuffle(len(tr_paths), seed=SEED)
                    .map(load_and_preprocess, num_parallel_calls=AUTOTUNE)
                    .batch(BATCH_SIZE)
                    .prefetch(AUTOTUNE))

        val_ds   = (tf.data.Dataset.from_tensor_slices((val_paths, val_labels_oh))
                    .map(load_and_preprocess, num_parallel_calls=AUTOTUNE)
                    .batch(BATCH_SIZE)
                    .prefetch(AUTOTUNE))

        return train_ds, val_ds, cls_names, tr_labels, val_labels

    else:
        labels_oh = tf.keras.utils.to_categorical(all_labels, NUM_CLASSES)

        test_ds  = (tf.data.Dataset.from_tensor_slices((all_paths, labels_oh))
                    .map(load_and_preprocess, num_parallel_calls=AUTOTUNE)
                    .batch(BATCH_SIZE)
                    .prefetch(AUTOTUNE))

        return test_ds, cls_names, all_labels


train_ds, val_ds, class_names, train_labels, val_labels = build_dataset(train_dir, training=True)
test_ds,  _,      true_labels                           = build_dataset(test_dir,  training=False)

print("Classes    :", class_names)
print("Train size :", len(train_labels))
print("Val size   :", len(val_labels))
print("Test size  :", len(true_labels))
print("Train steps:", len(train_labels) // BATCH_SIZE)

backbone = keras_hub.models.ViTBackbone.from_preset(
    "vit_base_patch16_224_imagenet"
)

backbone.trainable = False
print("Pretrained ViT backbone loaded and frozen")

def build_model(learning_rate=LR, dropout=0.3, trainable_backbone=False):

    backbone.trainable = trainable_backbone

    inputs    = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x         = backbone(inputs, training=False)

    # CLS token at index 0
    cls_token = x[:, 0, :]

    x = layers.LayerNormalization()(cls_token)
    x = layers.Dense(512, activation='gelu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(256, activation='gelu')(x)
    x = layers.Dropout(dropout * 0.5)(x)

    # float32 for mixed precision stability
    outputs = layers.Dense(NUM_CLASSES, activation='softmax', dtype='float32')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


model = build_model()
model.summary()

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

def get_callbacks(name):
    return [
        EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            name,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-8,
            verbose=1
        )
    ]

print("Callbacks ready")

configs = [
    {'lr': 3e-4, 'dropout': 0.2, 'trainable': False},
    {'lr': 1e-4, 'dropout': 0.3, 'trainable': False},
    {'lr': 5e-5, 'dropout': 0.2, 'trainable': True},
]

results = []

for i, cfg in enumerate(configs):
    print(f"\nConfig {i+1}: {cfg}")
    print("-" * 45)

    m = build_model(
        learning_rate=cfg['lr'],
        dropout=cfg['dropout'],
        trainable_backbone=cfg['trainable']
    )
    m.compile(
    optimizer=keras.optimizers.Adam(learning_rate=cfg['lr']),
    loss='categorical_crossentropy',
    metrics=['accuracy']
        )

    hist = m.fit(
        train_ds,
        epochs=8,
        validation_data=val_ds,
        callbacks=get_callbacks(f'tune_{i+1}.keras'),
        verbose=1
    )

    best_val = max(hist.history['val_accuracy'])
    results.append({'config': cfg, 'val_acc': best_val, 'model': m})
    print(f"Config {i+1} best val acc: {best_val*100:.2f}%")

best       = max(results, key=lambda x: x['val_acc'])
best_model = best['model']
print("\nBest config :", best['config'])
print("Best val acc:", f"{best['val_acc']*100:.2f}%")

test_loss, test_acc = best_model.evaluate(test_ds, verbose=1)
print(f"\nFinal Test Accuracy: {test_acc*100:.2f}%")

y_pred_final = np.argmax(best_model.predict(test_ds, verbose=1), axis=1)
y_true_final = true_labels

cm_final    = confusion_matrix(y_true_final, y_pred_final)
class_acc_f = cm_final.diagonal() / cm_final.sum(axis=1)

print("\nFinal Class-wise Accuracy:")
print("-" * 35)
for name, acc in zip(class_names, class_acc_f):
    print(f"  {name:20s}: {acc*100:.2f}%")
print("-" * 35)
print(f"  Mean Accuracy: {class_acc_f.mean()*100:.2f}%")

print("\nFull Classification Report:")
print(classification_report(y_true_final, y_pred_final, target_names=class_names, digits=4))

plt.figure(figsize=(14, 12))
sns.heatmap(
    cm_final,
    annot=True,
    fmt='d',
    cmap='Greens',
    xticklabels=class_names,
    yticklabels=class_names
)
plt.title('Final Confusion Matrix - Best ViT', fontsize=14)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('final_confusion_matrix.png', dpi=150)
plt.show()