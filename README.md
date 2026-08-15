# Persian-Digits-Hand-Writing-Image-Classification-Fully-Connected-Neural-Network
## 1. Import Libraries

The required libraries are imported to handle data loading, visualization, and the development of the Fully Connected Neural Network.

```python
import numpy as np
from scipy import io
from tensorflow import keras
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.losses import SparseCategoricalCrossentropy
from keras.layers import Dense, Dropout, Softmax, Flatten
```

### Libraries Overview
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

