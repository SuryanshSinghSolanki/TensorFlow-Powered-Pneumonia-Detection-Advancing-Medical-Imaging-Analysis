import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from keras.models import load_model
from keras.applications.vgg16 import preprocess_input

model = tf.keras.models.load_model('our_model.h5')


def preprocess_image(image):
    image = image.convert('RGB')
    image = image.resize((224, 224))
    image = np.asarray(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    return image
st.set_page_config(page_title='Pneumonia Detection App', page_icon=':microscope:')

st.markdown("""
    <style>
        body {
            background-color: #f5f5f5;
        }

        .header {
            font-size: 36px;
            font-weight: bold;
            color: #0066ec;
            text-align: center;
            margin-bottom: 50px;
        }

        .desc {
            font-size: 20px;
            font-weight: bold;
            color: #0066a0;
            text-align: center;
            margin-bottom: 50px;
        }
        .result {
            font-size: 24px;
            font-weight: bold;
            color: #333;
            text-align: center;
            margin-top: 50px;
        }
    </style>
""", unsafe_allow_html=True)

def set_background_color_inf():
    """
    Sets the background color of the app.
    """
    # Add CSS style
    style = """
    <style>
    [data-testid="stAppViewContainer"] {
    background-color: #3f0d12;
    background-image: linear-gradient(180deg, #3f0d12 0%, #a71d31 74%);
    }
    </style>
    """
    st.markdown(style, unsafe_allow_html=True)

def set_background_color_uninf():
    """
    Sets the background color of the app.
    """
    # Add CSS style
    style = """
    <style>
    [data-testid="stAppViewContainer"] {
    background-color: #63a91f;
    background-image: linear-gradient(360deg, #63a91f 0%, #1a512e 74%);
    
}
    </style>
    """
    st.markdown(style, unsafe_allow_html=True)


# Create the Streamlit app
def app():
    
    st.markdown('<div class="header">Pneumonia Detection App</div>', unsafe_allow_html=True)
    st.markdown('<div class="desc">This app uses a pre-trained deep learning model to detect pneumonia from chest X-ray images. To use the app, upload a chest X-ray image using the file uploader below and click the "Detect Pneumonia" button to get a prediction.</div>', unsafe_allow_html=True)

    # Add a file uploader widget
    file = st.file_uploader('Choose a chest X-ray image', type=['jpg', 'jpeg', 'png'])

    # Make a prediction on the uploaded image
    if file is not None:
        image = Image.open(file)
        st.image(image, width=700 , caption='Uploaded Image')
        image = preprocess_image(image)
        col1, col2, col3 = st.columns([1.26,1,1])
        if col2.button('Detect Pneumonia'):
            prediction = model.predict(image)
            if prediction[0][0] > prediction[0][1]:
                set_background_color_uninf()
                st.markdown('<div class="result">Result: Normal</div>', unsafe_allow_html=True)
            else:
                set_background_color_inf()
                st.markdown('<div class="result">Result: Pneumonia</div>', unsafe_allow_html=True)
app()
