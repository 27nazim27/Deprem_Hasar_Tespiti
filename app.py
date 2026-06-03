# ==================================================
# KÜTÜPHANELER
# ==================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    classification_report
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Dense,
    Dropout,
    GlobalAveragePooling2D
)

from tensorflow.keras.preprocessing import image_dataset_from_directory

# ==================================================
# TensorFlow Sürümü
# ==================================================

print("TensorFlow Sürümü:", tf.__version__)

# ==================================================
# AYARLAR
# ==================================================

VERI_YOLU = "Veri_Seti"

BATCH_SIZE = 32
IMG_SIZE = (256, 256)
EPOCHS = 8

# ==================================================
# VERİ SETİ
# ==================================================

train_dataset = image_dataset_from_directory(
    VERI_YOLU,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

validation_dataset = image_dataset_from_directory(
    VERI_YOLU,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# ==================================================
# SINIF İSİMLERİ
# ==================================================

class_names = train_dataset.class_names
print("Bulunan Sınıflar:", class_names)

# ==================================================
# PERFORMANS İYİLEŞTİRME
# ==================================================

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)

validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)

# ==================================================
# DATA AUGMENTATION
# ==================================================

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomContrast(0.1),
])

# ==================================================
# 1. MODEL -> CNN MODELİ
# ==================================================

model = Sequential([

    # Data Augmentation
    data_augmentation,

    # Normalize
    tf.keras.layers.Rescaling(1./255, input_shape=(256, 256, 3)),

    # 1. Conv Bloğu
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    # 2. Conv Bloğu
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    # 3. Conv Bloğu
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    # Global Average Pooling
    GlobalAveragePooling2D(),

    # Dense Layer
    Dense(128, activation='relu'),

    # Dropout
    Dropout(0.6),

    # Output
    Dense(len(class_names), activation='softmax')
])

# ==================================================
# MODEL ÖZETİ
# ==================================================

print("\n1. MODEL -> CNN MODELİ\n")

model.summary()

# ==================================================
# MODEL DERLEME
# ==================================================

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ==================================================
# EARLY STOPPING
# ==================================================

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# ==================================================
# MODEL EĞİTİMİ
# ==================================================

print("\nCNN Model eğitimi başlıyor...\n")

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS,
    callbacks=[early_stop]
)

# ==================================================
# GRAFİKLER
# ==================================================

plt.style.use('seaborn-v0_8-darkgrid')

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(acc))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# ---------------- Accuracy ----------------

ax1.plot(epochs_range, acc, label='Eğitim (Train)', lw=2)
ax1.plot(epochs_range, val_acc, label='Doğrulama (Validation)', lw=2)

ax1.set_title('CNN Model Doğruluğu', fontsize=14, pad=15)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Doğruluk')
ax1.legend(loc='lower right')

# ---------------- Loss ----------------

ax2.plot(epochs_range, loss, label='Eğitim (Train)', lw=2)
ax2.plot(epochs_range, val_loss, label='Doğrulama (Validation)', lw=2)

ax2.set_title('CNN Model Kaybı (Loss)', fontsize=14, pad=15)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Kayıp')
ax2.legend(loc='upper right')

plt.tight_layout()
plt.show()

# ==================================================
# CNN SONUÇLARI
# ==================================================

print("\nCNN Final Eğitim Accuracy:",
      round(acc[-1] * 100, 2), "%")

print("CNN Final Validation Accuracy:",
      round(val_acc[-1] * 100, 2), "%")

print("CNN Final Eğitim Loss:",
      round(loss[-1], 4))

print("CNN Final Validation Loss:",
      round(val_loss[-1], 4))

# ==================================================
# CONFUSION MATRIX -> CNN MODELİ
# ==================================================

y_true = []
y_pred = []

for images, labels in validation_dataset:

    predictions = model.predict(images)

    predicted_labels = np.argmax(predictions, axis=1)

    y_true.extend(labels.numpy())
    y_pred.extend(predicted_labels)

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel("Tahmin")
plt.ylabel("Gerçek")
plt.title("CNN Confusion Matrix")

plt.show()

print("\nCNN Classification Report\n")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names
    )
)

# ==================================================
# ==================================================
# 2. MODEL -> MOBILENETV2
# ==================================================
# ==================================================

print("\n\n2. MODEL -> MobileNetV2\n")

# ==================================================
# PRETRAINED MODEL
# ==================================================

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(256, 256, 3),
    include_top=False,
    weights='imagenet'
)

# Base model dondur
base_model.trainable = False

# ==================================================
# MOBILENET MODELİ
# ==================================================

mobilenet_model = Sequential([

    data_augmentation,

    tf.keras.layers.Rescaling(1./255),

    base_model,

    GlobalAveragePooling2D(),

    Dropout(0.4),

    Dense(128, activation='relu'),

    Dense(len(class_names), activation='softmax')

])

# ==================================================
# MODEL ÖZETİ
# ==================================================

mobilenet_model.summary()

# ==================================================
# MODEL DERLEME
# ==================================================

mobilenet_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ==================================================
# EARLY STOPPING
# ==================================================

early_stop_2 = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# ==================================================
# MODEL EĞİTİMİ
# ==================================================

print("\nMobileNetV2 eğitimi başlıyor...\n")

history_mobilenet = mobilenet_model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS,
    callbacks=[early_stop_2]
)

# ==================================================
# GRAFİKLER
# ==================================================

acc2 = history_mobilenet.history['accuracy']
val_acc2 = history_mobilenet.history['val_accuracy']

loss2 = history_mobilenet.history['loss']
val_loss2 = history_mobilenet.history['val_loss']

epochs_range2 = range(len(acc2))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy

ax1.plot(epochs_range2, acc2,
         label='Eğitim (Train)', lw=2)

ax1.plot(epochs_range2, val_acc2,
         label='Doğrulama (Validation)', lw=2)

ax1.set_title('MobileNetV2 Doğruluğu')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()

# Loss

ax2.plot(epochs_range2, loss2,
         label='Eğitim (Train)', lw=2)

ax2.plot(epochs_range2, val_loss2,
         label='Doğrulama (Validation)', lw=2)

ax2.set_title('MobileNetV2 Loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()

plt.tight_layout()
plt.show()

# ==================================================
# MOBILENET SONUÇLARI
# ==================================================

print("\nMobileNetV2 Eğitim Accuracy:",
      round(acc2[-1] * 100, 2), "%")

print("MobileNetV2 Validation Accuracy:",
      round(val_acc2[-1] * 100, 2), "%")

print("MobileNetV2 Eğitim Loss:",
      round(loss2[-1], 4))

print("MobileNetV2 Validation Loss:",
      round(val_loss2[-1], 4))

# ==================================================
# CONFUSION MATRIX -> MOBILENETV2
# ==================================================

y_true2 = []
y_pred2 = []

for images, labels in validation_dataset:

    predictions2 = mobilenet_model.predict(images)

    predicted_labels2 = np.argmax(predictions2, axis=1)

    y_true2.extend(labels.numpy())
    y_pred2.extend(predicted_labels2)

cm2 = confusion_matrix(y_true2, y_pred2)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm2,
    annot=True,
    fmt='d',
    cmap='Greens',
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel("Tahmin")
plt.ylabel("Gerçek")
plt.title("MobileNetV2 Confusion Matrix")

plt.show()

print("\nMobileNetV2 Classification Report\n")

print(
    classification_report(
        y_true2,
        y_pred2,
        target_names=class_names
    )
)

# ==================================================
# MODEL KARŞILAŞTIRMA TABLOSU
# ==================================================

print("\n==============================")
print("MODEL KARŞILAŞTIRMASI")
print("==============================\n")

print(f"CNN Validation Accuracy: %{round(val_acc[-1] * 100, 2)}")
print(f"MobileNetV2 Validation Accuracy: %{round(val_acc2[-1] * 100, 2)}")

print(f"\nCNN Validation Loss: {round(val_loss[-1], 4)}")
print(f"MobileNetV2 Validation Loss: {round(val_loss2[-1], 4)}")