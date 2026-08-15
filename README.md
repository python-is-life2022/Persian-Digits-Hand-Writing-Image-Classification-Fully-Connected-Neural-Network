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
from keras.losses import SparseCategoricalCrossentropy
from keras.layers import Dense, Dropout, Softmax, Flatten
```

### Libraries Overview
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


