# ==========================================
# AI-ML Assignment 9
# Image Classification using CNN
# Cats vs Dogs
# ==========================================

import os
import random
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    precision_score,
    recall_score,
    f1_score
)

import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense
)

# ----------------------------------------
# Dataset Path
# ----------------------------------------

dataset_path = "PetImages"

# ----------------------------------------
# Task 1 : Data Understanding
# ----------------------------------------

print("Folder Structure\n")

for folder in os.listdir(dataset_path):
    print(folder)

classes = os.listdir(dataset_path)

print("\nNumber of Classes:")
print(len(classes))

total_images = 0

print("\nImages per Class")

for cls in classes:

    path = os.path.join(dataset_path, cls)

    count = len(os.listdir(path))

    total_images += count

    print(f"{cls} : {count}")

print("\nTotal Images:")
print(total_images)

# ----------------------------------------
# Display Sample Images
# ----------------------------------------

plt.figure(figsize=(12,8))

index = 1

for cls in classes:

    folder = os.path.join(dataset_path, cls)

    images = random.sample(os.listdir(folder), 3)

    for image in images[:]:

        img = plt.imread(os.path.join(folder, image))

        plt.subplot(2,3,index)

        plt.imshow(img)

        plt.title(cls)

        plt.axis("off")

        index += 1

plt.show()

print("\nImage Size after preprocessing : 128 x 128")

# ----------------------------------------
# Task 2 : Data Preprocessing
# ----------------------------------------

train_generator = ImageDataGenerator(
    rescale=1/255,
    validation_split=0.20
)

train_data = train_generator.flow_from_directory(
    dataset_path,
    target_size=(128,128),
    batch_size=32,
    class_mode="binary",
    subset="training"
)

test_data = train_generator.flow_from_directory(
    dataset_path,
    target_size=(128,128),
    batch_size=32,
    class_mode="binary",
    subset="validation",
    shuffle=False
)

print("\nTraining Images:", train_data.samples)

print("Testing Images:", test_data.samples)

# ----------------------------------------
# Task 3 : CNN Model
# ----------------------------------------

model = Sequential()

model.add(
    Conv2D(
        32,
        (3,3),
        activation="relu",
        input_shape=(128,128,3)
    )
)

model.add(
    MaxPooling2D((2,2))
)

model.add(
    Conv2D(
        64,
        (3,3),
        activation="relu"
    )
)

model.add(
    MaxPooling2D((2,2))
)

model.add(
    Conv2D(
        128,
        (3,3),
        activation="relu"
    )
)

model.add(
    MaxPooling2D((2,2))
)

model.add(
    Flatten()
)

model.add(
    Dense(
        128,
        activation="relu"
    )
)

model.add(
    Dense(
        1,
        activation="sigmoid"
    )
)

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ----------------------------------------
# Training
# ----------------------------------------

history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=10
)

# ----------------------------------------
# Evaluation
# ----------------------------------------

loss, accuracy = model.evaluate(test_data)

print("\nTest Accuracy :", accuracy)

predictions = model.predict(test_data)

predicted_classes = (predictions > 0.5).astype(int).flatten()

true_classes = test_data.classes

precision = precision_score(
    true_classes,
    predicted_classes
)

recall = recall_score(
    true_classes,
    predicted_classes
)

f1 = f1_score(
    true_classes,
    predicted_classes
)

print("\nPrecision :", precision)

print("Recall :", recall)

print("F1 Score :", f1)

print("\nClassification Report\n")

print(
    classification_report(
        true_classes,
        predicted_classes,
        target_names=list(test_data.class_indices.keys())
    )
)

# ----------------------------------------
# Confusion Matrix
# ----------------------------------------

cm = confusion_matrix(
    true_classes,
    predicted_classes
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=list(test_data.class_indices.keys())
)

disp.plot(cmap="Blues")

plt.title("Confusion Matrix")

plt.show()

# ----------------------------------------
# Accuracy Graph
# ----------------------------------------

plt.figure(figsize=(8,5))

plt.plot(history.history["accuracy"])

plt.plot(history.history["val_accuracy"])

plt.title("Accuracy vs Epoch")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend([
    "Training Accuracy",
    "Validation Accuracy"
])

plt.grid(True)

plt.show()

# ----------------------------------------
# Loss Graph
# ----------------------------------------

plt.figure(figsize=(8,5))

plt.plot(history.history["loss"])

plt.plot(history.history["val_loss"])

plt.title("Loss vs Epoch")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend([
    "Training Loss",
    "Validation Loss"
])

plt.grid(True)

plt.show()