import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Set Streamlit page configuration
st.set_page_config(
    page_title="Skin Disease Detector",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Load the trained model
model = tf.keras.models.load_model("model.h5")

# Disease labels and explanations
disease_info = {
    "Actinic keratosis": "A rough, scaly patch on the skin caused by years of sun exposure. It can sometimes develop into skin cancer.",
    "Basal cell carcinoma": "A common form of skin cancer that grows slowly. It's often caused by UV damage and rarely spreads.",
    "Benign keratosis": "Non-cancerous skin growths that are usually brown, black, or light tan. Generally harmless.",
    "Dermatofibroma": "A small, firm bump on the skin, often caused by minor injury. Typically harmless and doesn’t require treatment.",
    "Melanoma": "A serious type of skin cancer that can spread quickly. Early detection is critical for treatment.",
    "Nevus": "Commonly known as a mole. Most are harmless, but changes in shape or color should be checked.",
    "Vascular lesion": "An abnormal growth of blood vessels, often red or purple. Most are benign and cosmetic."
}

# --- University Header ---
st.markdown("<h1 style='text-align: center; color: navy;'>Beaconhouse National University</h1>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar with project info and credits
with st.sidebar:
    st.title("📘 About the App")
    st.markdown("""
        This app uses a trained AI model to **predict skin diseases** from uploaded images.

        **Steps to Use:**
        1. Upload a clear skin image.
        2. The AI will analyze it.
        3. View prediction and disease information.

        ---
        **Group Members:**
        - Muhammad Zakriya Ahmed (F2023-726)
        - Abdul Raffay Qasim (F2023-737)
        - Mohammad Abdullah Abbas (F2023-871)

        ---
        **Model:** CNN (Convolutional Neural Network)  
        **Built with:** TensorFlow & Streamlit
    """)

# Main app title and instructions
st.title("🩺 Skin Disease Detection")
st.markdown("Upload an image of a skin lesion and let the AI identify the possible condition.")

# Image uploader
uploaded_file = st.file_uploader("📤 Upload a skin image (JPG or PNG)", type=["jpg", "jpeg", "png"])

# Prediction block
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📷 Uploaded Image", use_container_width=True)

    # Preprocess image for model (64x64, normalized, batch dimension added)
    image = image.resize((64, 64))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)  # Shape: (1, 64, 64, 3)

    with st.spinner("🧠 Analyzing the image..."):
        prediction = model.predict(image_array)
        predicted_index = np.argmax(prediction)
        predicted_class = list(disease_info.keys())[predicted_index]
        confidence = prediction[0][predicted_index] * 100

    # Display prediction
    st.success(f"✅ Predicted Disease: **{predicted_class}**")
    st.info(f"📊 Confidence: **{confidence:.2f}%**")

    # Show disease info
    st.markdown("### 🩻 About the Disease:")
    st.write(disease_info[predicted_class])
else:
    st.warning("👈 Please upload a skin image to begin diagnosis.")
