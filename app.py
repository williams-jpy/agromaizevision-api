import streamlit as st
import numpy as np
from PIL import Image
import keras
from huggingface_hub import hf_hub_download
import io

# Page config
st.set_page_config(
    page_title="AgromaizeVision",
    page_icon="🌽",
    layout="centered"
)

# Custom CSS
st.markdown("""
    <style>
    .main { background-color: #f0fdf4; }
    .title { color: #1b4332; font-size: 40px; font-weight: bold; text-align: center; }
    .subtitle { color: #2d6a4f; text-align: center; font-size: 16px; }
    .result-box { background-color: #d1fae5; padding: 20px; border-radius: 12px; }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="title">🌽 AgromaizeVision</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-Powered Maize Disease Detection | agromaize.com</p>', unsafe_allow_html=True)
st.markdown("---")

# Disease info
disease_info = {
    "Common Rust": {
        "cause": "Caused by the fungus Puccinia sorghi, spread by wind-borne spores.",
        "symptoms": "Small, circular to elongated brown pustules on both leaf surfaces.",
        "treatment": "Apply fungicides like Mancozeb or Propiconazole. Plant resistant varieties.",
        "color": "🟠"
    },
    "Gray Leaf Spot": {
        "cause": "Caused by the fungus Cercospora zeae-maydis, favored by humid conditions.",
        "symptoms": "Rectangular gray to brown lesions running parallel to leaf veins.",
        "treatment": "Use crop rotation, remove infected debris, apply strobilurin fungicides.",
        "color": "🔵"
    },
    "Healthy": {
        "cause": "No disease detected.",
        "symptoms": "The plant appears healthy with no visible disease symptoms.",
        "treatment": "Continue regular crop management and monitoring.",
        "color": "🟢"
    },
    "Northern Leaf Blight": {
        "cause": "Caused by the fungus Exserohilum turcicum, spread by wind and rain.",
        "symptoms": "Long, cigar-shaped gray-green to tan lesions on leaves.",
        "treatment": "Apply fungicides early, use resistant hybrids, practice crop rotation.",
        "color": "🔴"
    },
}

# Load model
@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="williams-jpy/agromaizevision",
        filename="3.keras"
    )
    return keras.models.load_model(model_path)

class_names = ["Common Rust", "Gray Leaf Spot", "Healthy", "Northern Leaf Blight"]

# Upload image
st.subheader("📸 Upload a Maize Leaf Image")
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Predict button
    if st.button("🔍 Predict Disease", use_container_width=True):
        with st.spinner("Analyzing image..."):
            # Load model
            model = load_model()

            # Preprocess
            img = image.resize((256, 256))
            img_array = np.array(img) / 255.0
            img_batch = np.expand_dims(img_array, 0)

            # Predict
            predictions = model.predict(img_batch)
            predicted_class = class_names[np.argmax(predictions[0])]
            confidence = float(np.max(predictions[0])) * 100

        # Show results
        st.markdown("---")
        st.subheader("🧪 Diagnosis Result")

        info = disease_info[predicted_class]
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Disease Detected", f"{info['color']} {predicted_class}")
        with col2:
            st.metric("Confidence", f"{confidence:.2f}%")

        st.markdown("### 📋 Disease Information")
        st.info(f"**🦠 Cause:** {info['cause']}")
        st.warning(f"**🔍 Symptoms:** {info['symptoms']}")
        st.success(f"**💊 Treatment:** {info['treatment']}")

st.markdown("---")
st.markdown('<p style="text-align:center; color:gray;">© 2026 AgromaizeVision · Powered by AI</p>', unsafe_allow_html=True)