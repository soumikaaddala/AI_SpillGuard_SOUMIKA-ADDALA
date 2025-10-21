import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

# ------------------------------
# Load the trained model
# ------------------------------
@st.cache_resource
def load_unet_model():
    model = load_model("oilspill_unet.h5", compile=False)
    return model

model = load_unet_model()
st.title("🌊 Oil Spill Segmentation using U-Net")

st.markdown("""
Upload a **satellite image** to detect and visualize oil spill regions.  
Model trained on the [Zenodo Oil Spill Dataset](https://zenodo.org/records/10555314).
""")

# ------------------------------
# Image Upload Section
# ------------------------------
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    st.subheader("🖼️ Original Image")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # ------------------------------
    # Preprocess the image for prediction
    # ------------------------------
    img_size = (256, 256)
    img_resized = cv2.resize(img_array, img_size)
    img_norm = img_resized / 255.0
    img_input = np.expand_dims(img_norm, axis=0)  # shape (1,256,256,3)

    # ------------------------------
    # Predict mask
    # ------------------------------
    with st.spinner("Detecting oil spill..."):
        pred_mask = model.predict(img_input)[0]  # shape (256,256,1)
        pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255

    # Resize mask back to original size
    pred_mask_resized = cv2.resize(pred_mask, (img_array.shape[1], img_array.shape[0]))

    # Create overlay (red color mask on image)
    overlay = img_array.copy()
    overlay[pred_mask_resized > 0] = [255, 0, 0]  # red overlay
    blended = cv2.addWeighted(img_array, 0.7, overlay, 0.3, 0)

    # ------------------------------
    # Display results
    # ------------------------------
    st.subheader("🧠 Segmentation Result")
    st.image(blended, caption="Oil Spill Region Highlighted", use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.image(img_array, caption="Original Image", use_container_width=True)
    with col2:
        st.image(pred_mask_resized, caption="Predicted Mask", use_container_width=True)

    st.success("✅ Oil spill detection complete!")

else:
    st.info("Please upload a satellite image to start segmentation.")
