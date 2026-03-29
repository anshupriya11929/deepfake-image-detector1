import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import sys

MODEL_PATH = "models/deepfake_detector.keras"
IMG_SIZE = (224, 224)

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Check if image path is passed
if len(sys.argv) < 2:
    print("Usage: python predict.py <image_path>")
    sys.exit()

img_path = sys.argv[1]

# Load and preprocess image
img = image.load_img(img_path, target_size=IMG_SIZE)
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)

# EfficientNet preprocessing
img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)

# Predict
prediction = model.predict(img_array)[0][0]

# Class mapping:
# fake = 0, real = 1 (check class_names output from train.py)
if prediction > 0.5:
    print(f"Prediction: REAL ({prediction*100:.2f}%)")
else:
    print(f"Prediction: FAKE ({(1-prediction)*100:.2f}%)")