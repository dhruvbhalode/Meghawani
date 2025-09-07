import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

st.set_page_config(page_title="Flood Detection App", layout="centered")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(r"C:/Users/dhruv/mera_Unet_flood_model.keras")

model = load_model()

def preprocess_image(image):
    image = image.resize((256, 256))
    rgb_array = np.array(image) / 255.0

    if rgb_array.shape[-1] != 3:
        raise ValueError("Uploaded image must be in RGB format")

    padded_input = np.concatenate([rgb_array, np.zeros((256, 256, 6))], axis=-1)
    padded_input = np.expand_dims(padded_input, axis=0)
    return tf.convert_to_tensor(padded_input, dtype=tf.float32)

st.title("Flood Detection Using ML")
st.markdown("Upload an image (RGB format only). Model predicts whether flooding is present.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

show_debug = st.checkbox("Show Debug Info")

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)
        probability = float(prediction.flatten()[0])

        is_flooded = probability > 0.4

        st.subheader("Prediction Result")
        if is_flooded:
            st.error(f"🚨 Flood Detected!\n\n**Probability:** {round(probability * 100, 2)}%")
        else:
            st.success(f"No Flood Detected.\n\n**Probability:** {round(probability * 100, 2)}%")

        if show_debug:
            st.code(f"Raw Prediction Tensor: {prediction}", language='python')

    except Exception as e:
        st.warning(f"Error: {e}")
