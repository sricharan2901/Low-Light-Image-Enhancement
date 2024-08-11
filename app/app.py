import streamlit as st
from PIL import Image
from zerodce import ZeroDCE
import tensorflow as tf
import numpy as np
import keras
import h5py

st.set_page_config(page_title="Low Light Image Detection", page_icon="📷", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
    .centered-title {
        text-align: center;
    }
    .larger-heading {
        font-size: 24px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

def main():
    show_home()

def infer(original_image, model):
    image = keras.utils.img_to_array(original_image)
    image = image.astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)
    output_image = model(image)
    output_image = tf.cast((output_image[0, :, :, :] * 255), dtype=np.uint8)
    output_image = Image.fromarray(output_image.numpy())
    return output_image

def zeroimage(original_image):
    model = ZeroDCE()
    model.load_weights('./zerodce1.h5', False, False, None)
    output_image = infer(original_image, model)
    return output_image

def show_home():
    st.markdown("<h1 class='centered-title'>Low Light Image Detection 📷</h1>", unsafe_allow_html=True)
    st.markdown("<h3 class='larger-heading'>Upload Image</h3>", unsafe_allow_html=True)
    uploaded_image = st.file_uploader("", type=["png", "jpg", "jpeg"])

    if uploaded_image:
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        original_image = Image.open(uploaded_image)

        output_image = zeroimage(original_image)
        st.image(output_image, caption="ZeroDCE", use_column_width=True)

if __name__ == "__main__":
    main()
