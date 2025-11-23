🩺 Breast Cancer Detection — Deep Learning Backend (TensorFlow)

This project uses a Neural Network classifier built using TensorFlow/Keras to predict whether a breast tumor is Malignant or Benign.

The backend uses:

TensorFlow/Keras model (.h5)

Scikit-learn StandardScaler (.pkl)

FastAPI for prediction inference (optional deployment)

This repository contains only backend & model training, no frontend.

🚀 Features

Uses the Breast Cancer Wisconsin Dataset from sklearn

StandardScaler preprocessing

2-class neural network classifier

Accuracy & loss plots during training

API-ready prediction code

📊 Dataset

Imported via:

from sklearn.datasets import load_breast_cancer


Samples: 569

Features: 30

Labels:

0 → Malignant

1 → Benign

🧠 Model Architecture
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(30,)),
    keras.layers.Dense(20, activation='relu'),
    keras.layers.Dense(2, activation='sigmoid')
])


Loss: sparse_categorical_crossentropy
Optimizer: adam
Epochs: 10

📈 Training Steps (Already Implemented)

Your code performs:

Load dataset

Convert to DataFrame

Standardize using StandardScaler()

Train Neural Network

Plot accuracy / loss curves

Evaluate on test dataset

Predict for a single input sample

This generates:

breast_cancer_model.h5
scaler.pkl

🔍 Prediction Example
input_data = (11.76, 21.6, 74.72, 427.9, ...)

input_data_as_numpy_array = np.asarray(input_data)
input_reshaped = input_data_as_numpy_array.reshape(1, -1)
input_std = scaler.transform(input_reshaped)

prediction = model.predict(input_std)
label = np.argmax(prediction)


Output:

0 → Malignant
1 → Benign


📦 Installation
pip install numpy pandas scikit-learn tensorflow matplotlib

▶ Run Training Script
python train.py
