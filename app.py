import streamlit as st
from streamlit_image_select import image_select
import tensorflow as tf
import numpy as np
from keras.models import load_model
import os
from PIL import Image
from matplotlib import image

@st.cache_resource
def get_model():
    return load_model('bird_classifier.h5')

@st.cache_data
def get_images():
    images = []
    for i in range(1, 9):
        filename = f'{i}.jpg'
        img = image.imread(os.path.join('res', filename))
        images.append(img)
    return images

@st.cache_data
def get_labels():
    with open('labels.txt', 'r') as fp:
        labels = fp.read().split('\n')
    return labels

def top_5_pred(input):
    preds = model.predict(tf.convert_to_tensor([input]))[0]
    pred_indices = np.argpartition(preds, -5)[-5:]
    pred_species = pred_indices[np.argsort(preds[pred_indices])][::-1]
    pred_text = '\n'.join([f'{preds[i]:.2%} {labs[i]}' for i in pred_species])
    return pred_text


st.set_page_config(layout='wide')
st.title('Bird species classifier')

st.write("""This model classifies images of birds into one of 525 species.
                The dataset used to train the model can be found 
                [here](https://www.kaggle.com/datasets/gpiosenka/100-bird-species). 
                The model is a convolutional neural network (CNN) built on top of
                EfficientNet-B0.
                You can either select one of the bird pictures, which were 
                not used in training or validating the model, or 
                upload a bird picture to 
                get the 5 most likely species for that bird according to the model. 
                For the best results, use a photo that only contains one bird 
                that takes up most of the photo. This model has a 89.73% test 
                accuracy and a 97.45% top 5 test accuracy.""")

col1, col2 = st.columns(2)

model = get_model()
imgs = get_images()
labs = get_labels()

with col1:
    img = image_select(label='Select a bird',
                   images=imgs,
                   use_container_width=False)
    st.text_area('Predictions:', top_5_pred(img), height=150)

with col2:
    uploaded_file = st.file_uploader('Upload a bird picture', type=['jpg', 'png'])
    if uploaded_file is not None:
        try:
            im = Image.open(uploaded_file)
            st.image(im)
            im = np.array(im)
            im = tf.image.resize(im, size=(224, 224))

            st.text_area('Predictions:', top_5_pred(im), height=150)

        except:
            st.text_area('Predictions:',
                         'Unable to perform predictions on this image',
                         height=150)

