import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import threading
import pyttsx3
from config import MODEL_PATH

# ---------------- BACKGROUND IMAGE ----------------
def set_bg():
    bg_url = "https://www.tycosecurityproducts.in/blogs/img/face-mask-detection-technology-big.jpg"
    
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{bg_url}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}

        /* REMOVE DARK OVERLAY */
        .block-container {{
            background-color: transparent;
        }}

        /* CENTER TITLE WITH WHITE BOX */
        h1 {{
            text-align: center;
            background-color: white;
            padding: 12px 20px;
            border-radius: 12px;
            display: inline-block;
            margin: auto;
        }}

        /* WRAP TITLE CENTER */
        div[data-testid="stTitle"] {{
            text-align: center;
        }}

        /* WHITE BOX FOR ALL TEXT ELEMENTS */
        .stMarkdown, .stText, .stMetric, .stAlert {{
            background-color: white;
            padding: 10px;
            border-radius: 10px;
            margin-bottom: 10px;
        }}

        </style>
        """,
        unsafe_allow_html=True
    )

set_bg()

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_my_model():
    return load_model(MODEL_PATH)

model = load_my_model()

# ---------------- TOP TITLE ----------------
st.title("😷 Face Mask Detection System")

# ---------------- SIDEBAR ----------------
st.sidebar.title("⚙️ Options")
option = st.sidebar.radio("Choose Mode", ["Upload Image", "Capture from Camera"])

class_names = [
    "Mask Proper 😷",
    "No Mask ❌",
    "Mask Incorrect 😕"
]

# ---------------- VOICE ----------------
def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# ---------------- PREDICTION FUNCTION ----------------
def predict_image(image):
    img = image.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)
    label_index = np.argmax(pred)
    confidence = np.max(pred) * 100

    return label_index, confidence

# ---------------- UPLOAD MODE ----------------
if option == "Upload Image":
    st.markdown("### Upload Image")

    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image")

        label_index, confidence = predict_image(image)

        st.success(class_names[label_index])
        st.metric("Confidence", f"{confidence:.2f}%")

# ---------------- CAMERA CAPTURE MODE ----------------
elif option == "Capture from Camera":

    st.markdown("### 📸 Capture your face")

    img_file = st.camera_input("Take a picture")

    if img_file:
        image = Image.open(img_file).convert("RGB")
        st.image(image, caption="Captured Image")

        label_index, confidence = predict_image(image)

        if label_index == 0:
            st.success("✅ MASK WORN PROPERLY")

        elif label_index == 1:
            st.error("❌ NOT WEARING MASK")
            threading.Thread(target=speak, args=("Please wear a mask",)).start()

        else:
            st.warning("⚠️ MASK WORN INCORRECTLY")
            threading.Thread(target=speak, args=("Please wear your mask properly",)).start()

        st.metric("Confidence", f"{confidence:.2f}%")