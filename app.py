import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image

MODEL_PATH = "models/deepfake_detector.keras"
IMG_SIZE = (224, 224)

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

st.set_page_config(page_title="DeepFake Detector", layout="centered")
st.title("🧬 DeepFake Image Detector")
st.write("Upload a face image to check whether it is Real or Fake.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Preprocess image
    img_resized = img.resize(IMG_SIZE)
    img_array = image.img_to_array(img_resized)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)[0][0]

    if prediction > 0.5:
        st.success(f"Prediction: REAL ({prediction*100:.2f}%)")
    else:
        st.error(f"Prediction: FAKE ({(1-prediction)*100:.2f}%)")