# Image Classification using Convolutional Neural Networks (CNN)

## Objective

The objective of this project is to develop a Convolutional Neural Network (CNN) that classifies pet images as either Cats or Dogs. The project demonstrates image preprocessing, CNN model development, training, evaluation, and visualization of the model's performance.

---

## Dataset Link

Cats vs Dogs Dataset

https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset

---

## Libraries Used

- TensorFlow / Keras
- NumPy
- Matplotlib
- Scikit-learn
- Pillow
- OS
- Random

---

## Methodology

1. Load the Cats vs Dogs dataset from the folder structure.
2. Display the dataset folder structure and sample images.
3. Resize all images to **128 × 128** pixels.
4. Normalize pixel values to the range **0–1**.
5. Split the dataset into **80% training** and **20% validation** using `ImageDataGenerator`.
6. Build the Convolutional Neural Network.
7. Train the model for **10 epochs**.
8. Evaluate the model using Test Accuracy, Precision, Recall, F1-Score, and a Confusion Matrix.
9. Visualize the training process using Accuracy vs Epoch and Loss vs Epoch graphs.

---

## CNN Architecture

- Input Image Size: **128 × 128 × 3**
- Conv2D (32 Filters, 3×3, ReLU)
- MaxPooling2D (2×2)
- Conv2D (64 Filters, 3×3, ReLU)
- MaxPooling2D (2×2)
- Conv2D (128 Filters, 3×3, ReLU)
- MaxPooling2D (2×2)
- Flatten Layer
- Dense Layer (128 Neurons, ReLU)
- Output Layer (1 Neuron, Sigmoid)

Optimizer:
- Adam

Loss Function:
- Binary Crossentropy

Metric:
- Accuracy

Epochs:
- 10

---

## Results

The CNN model successfully classified cat and dog images with good accuracy. The evaluation metrics, confusion matrix, and training graphs demonstrate that the network effectively learned visual features from the dataset. The convolution and pooling layers contributed to improved feature extraction and reduced computational complexity.

---

## Conclusion

This project demonstrates the effectiveness of Convolutional Neural Networks for binary image classification. CNNs automatically learn important visual features from images using convolution and pooling operations, making them significantly more suitable than traditional Artificial Neural Networks for image-based tasks. Although CNNs generally achieve high classification accuracy, they require larger datasets, longer training times, and more computational resources than simpler machine learning models.
