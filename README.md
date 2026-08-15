# Persian-Digits-Hand-Writing-Image-Classification-Fully-Connected-Neural-Network
## 1. Import Libraries

The required libraries are imported to handle data loading, visualization, and the development of the Fully Connected Neural Network.

```python
import cv2
import numpy as np
from scipy import io
from tensorflow import keras
import matplotlib.pyplot as plt
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.losses import SparseCategoricalCrossentropy
from keras.layers import Dense, Dropout, Softmax, Flatten
```

### Libraries Overview
* **sklearn.model_selection** provides tools for splitting data and evaluating machine learning models.
* **cv2** — Used for image processing, such as resizing images to a consistent shape before training the model.
* **NumPy** — Used for numerical operations and array manipulation.
* **SciPy** — Used to load and work with the dataset files.
* **TensorFlow / Keras** — Used to build, compile, and train the neural network.
* **Matplotlib** — Used for visualizing images and model results.
* **Sequential** — Used to create the neural network architecture layer by layer.
* **Dense** — Fully connected neural network layers.
* **Dropout** — Used as a regularization technique to help reduce overfitting.
* **Flatten** — Converts image data into a one-dimensional vector before passing it to fully connected layers.
* **SparseCategoricalCrossentropy** — Used as the loss function for multi-class classification with integer labels.
* **Softmax** — Converts the final layer's logits into class probabilities.
## 3. Load Dataset & Data Inspection

The dataset is loaded using `scipy.io.loadmat()` and its structure is inspected by checking the available data and label shapes.

```python
digits = io.loadmat('Data_hoda_full.mat')

print(f"Data Shape: {digits['Data'].shape}\nLabels Shape: {digits['labels'].shape}")
```

**Output:**

```text
Data Shape: (60000, 1)
Labels Shape: (60000, 1)
```
## 4. Data Reshaping

Remove unnecessary singleton dimensions from the data and labels using `np.squeeze()`.

```python
digits['Data'] = np.squeeze(digits['Data'])
digits['labels'] = np.squeeze(digits['labels'])

print(f"Data Shape: {digits['Data'].shape}\nLabels Shape: {digits['labels'].shape}")
```
**Output:**

```text
Data Shape: (60000,)
Labels Shape: (60000,)
```
## 5. Visualize Sample Image

Display a sample handwritten digit from the dataset along with its corresponding label.

```python
i = int(input('Enter the number of image sample: '))

plt.figure(figsize=(5, 5))
plt.imshow(digits['Data'][i], cmap='gray')
plt.title(digits['labels'][i])
```
## 6.Image Preprocessing

Since the images in the dataset have different dimensions, preprocessing is required before training the model.

For example, the original dimensions of the samples are different:

```text
Sample 1: (18, 17)
Sample 2: (54, 11)
Sample 3: (20, 32)
```

To ensure that all images have a consistent input shape, each image is resized to **8 × 8 pixels** using OpenCV.

The pixel values are then normalized from the original `[0, 255]` range to `[0, 1]`:

```python
digits['Data'] = np.array([
    cv2.resize(img, (8, 8)) / 255
    for img in digits['Data']
])
```

After preprocessing, the dataset shape becomes:

```text
(60000, 8, 8)
```

This means the dataset contains **60,000 images**, with each image represented as an **8 × 8 grayscale image**.

Standardizing the image dimensions and normalizing pixel values makes the data more suitable for training a neural network.


## 7.Train-Test Split

The dataset is split into **80% training** and **20% testing** sets using `train_test_split`.

* **Training:** 48,000 samples
* **Testing:** 12,000 samples
* **Image shape:** 8 × 8

### Experiment 1 — Baseline Model

The initial neural network was trained for **30 epochs** and evaluated on both the training and test sets. This model was used as the **baseline** for comparing the results of subsequent experiments.

```python
model1 = Sequential([
    Flatten(),
    Dense(64, activation='relu'),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(256, activation='relu'),
    Dropout(0.35),
    Dense(10)
])

model1.compile(
    loss=SparseCategoricalCrossentropy(from_logits=True),
    optimizer='adam',
    metrics=['accuracy']
)

his1 = model1.fit(
    X_train,
    y_train,
    batch_size=32,
    epochs=30,
    verbose=2,
    validation_split=0.2
)
```

The model was evaluated on both training and test data:

```python
test_loss, test_acc = model1.evaluate(X_test, y_test)
train_loss, train_acc = model1.evaluate(X_train, y_train)

print(
    f'Test Accuracy: {test_acc * 100:.2f}\n'
    f'Test Loss: {test_loss:.2f}\n'
    f'Train Accuracy: {train_acc * 100:.2f}\n'
    f'Train Loss: {train_loss:.2f}'
)
```

### Results

| Metric   |  Train |       Test |
| -------- | -----: | ---------: |
| Accuracy | 99.38% | **98.19%** |
| Loss     |   0.02 |   **0.09** |

The model achieved **98.19% test accuracy** with a relatively small gap between training and test performance, indicating that the model generalizes well without significant overfitting.

This result will be used as the **baseline** for further experiments and model improvements.

### Experiment 2 — Model Architecture Improvement

The model architecture was simplified and balanced by using fewer layers with **128 → 128 → 64** neurons and adjusted Dropout rates.

```python
model2 = Sequential([
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.15),
    Dense(128, activation='relu'),
    Dropout(0.20),
    Dense(64, activation='relu'),
    Dropout(0.15),
    Dense(10)
])

model2.compile(
    loss=SparseCategoricalCrossentropy(from_logits=True),
    optimizer='adam',
    metrics=['accuracy']
)

his2 = model2.fit(
    X_train,
    y_train,
    batch_size=32,
    epochs=30,
    verbose=2,
    validation_split=0.2
)
```

The model was evaluated on both training and test sets:

```python
test_loss, test_acc = model.evaluate(X_test, y_test)
train_loss, train_acc = model.evaluate(X_train, y_train)

print(
    f'Test Accuracy: {test_acc * 100:.2f}\n'
    f'Test Loss: {test_loss:.2f}\n'
    f'Train Accuracy: {train_acc * 100:.2f}\n'
    f'Train Loss: {train_loss:.2f}'
)
```

#### Results

| Metric   |  Train |       Test |
| -------- | -----: | ---------: |
| Accuracy | 99.56% | **98.56%** |
| Loss     |   0.02 |   **0.06** |

Compared with the baseline model, this architecture improved the **Test Accuracy from 98.19% to 98.43%** and reduced the **Test Loss from 0.09 to 0.06**.

This is currently the **best-performing model**.
