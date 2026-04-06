#  DeepFake Image Detector

##  Problem Statement
With the rapid advancement of Artificial Intelligence and image generation technologies, fake images (DeepFakes) are becoming increasingly realistic and difficult to identify with the human eye. This creates serious concerns in areas such as social media, journalism, cybersecurity, privacy, and digital trust.

The goal of this project is to build a **Deep Learning-based image classification system** that can detect whether a face image is **Real** or **Fake**.

---

##  Project Explanation
This project uses **Deep Learning** to classify images into two categories:

- **Real**
- **Fake**

The model is trained on a dataset containing facial images labeled as either real or fake. A **Convolutional Neural Network (CNN)**-based transfer learning approach is used to automatically learn visual patterns and distinguish between authentic and manipulated images.

###  Workflow of the Project
1. **Dataset Collection**
   - A dataset containing real and fake face images is used.

2. **Data Preprocessing**
   - Images are resized to a fixed input size.
   - Pixel values are normalized/preprocessed.
   - Data is split into training and validation sets.

3. **Model Building**
   - A pretrained deep learning model (**EfficientNetB0**) is used as the base model.
   - Additional classification layers are added for binary classification.

4. **Model Training**
   - The model is trained to identify whether an image is real or fake.
   - Performance is evaluated using accuracy and validation loss.

5. **Prediction**
   - The trained model predicts whether a new uploaded image is:
     - **REAL**
     - **FAKE**

6. **Deployment**
   - A simple and interactive **Streamlit web app** is created where users can upload an image and check the prediction instantly.

---

##  Technologies Used
- **Python**
- **TensorFlow / Keras**
- **EfficientNetB0**
- **NumPy**
- **Pillow**
- **Streamlit**

---

##  Dataset Used
This project uses the **140K Real and Fake Faces Dataset** from Kaggle.

###  Dataset Link:
:contentReference[oaicite:0]{index=0} — [140K Real and Fake Faces Dataset](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces)

---

##  Features
- Detects whether an image is **Real** or **Fake**
- Uses **Deep Learning Transfer Learning**
- User-friendly **web interface**
- Fast prediction system
- Useful for **AI ethics and digital media verification**

---

##  Input
The user uploads a face image in one of the following formats:

- `.jpg`
- `.jpeg`
- `.png`

---

## 📤 Output
The system predicts:

- **REAL**
or
- **FAKE**

along with the confidence score.

---

##  Future Scope
This project can be extended further by adding:

- **Video DeepFake Detection**
- **Face region heatmap visualization**
- **Explainable AI (Grad-CAM)**
- **Confidence threshold tuning**
- **API deployment**
- **Mobile app integration**

---
