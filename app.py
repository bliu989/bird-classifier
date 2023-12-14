import streamlit as st
from streamlit_image_select import image_select
import tensorflow as tf
import numpy as np
from keras.models import load_model
import os

model = load_model('bird_classifier.h5')

from matplotlib import image
# load images as pixel array
images = []

for i in range(1, 9):
    filename = f'{i}.jpg'
    img = image.imread(os.path.join('res', filename))
    images.append(img)
 
with open('labels.txt', 'r') as fp:
    labels = fp.read().split('\n')

def top_5_pred(input):
    preds = model.predict(np.array([input]))[0]
    pred_indices = np.argpartition(preds, -5)[-5:]
    pred_species = pred_indices[np.argsort(preds[pred_indices])][::-1]
    pred_text = '\n'.join([f'{preds[i]:.2%} {labels[i]}' for i in pred_species])
    
    return pred_text

st.title('Bird species classifier')

with st.sidebar:
    st.write("""This model classifies images of birds into one of 525 species.
                The dataset used to train the model can be found 
                [here](https://www.kaggle.com/datasets/gpiosenka/100-bird-species). 
                The model is a convolutional neural network (CNN) without any 
                transfer learning.
                You can either select one of the bird pictures, which were 
                not used in training or validating the model, or 
                [upload](#upload-picture-to-predict) a bird picture to 
                get the 5 most likely species for that bird according to the model. 
                For the best results, use a photo that only contains one bird 
                that takes up most of the photo. \n \n This model has a 89.73% test 
                accuracy and a 97.45% top 5 test accuracy.""")

st.header('Select picture to predict')

img = image_select(label = 'Select a bird',
                   images = images,
                   use_container_width = False)
                   

st.text_area('Predictions:', top_5_pred(img), height=150)

from PIL import Image
st.divider()
st.header('Upload picture to predict')

uploaded_file = st.file_uploader('Upload your own bird picture', type = ['jpg', 'png'])
if uploaded_file is not None:
    try:
        im = Image.open(uploaded_file)
        st.image(im)
        im = np.array(im)
        im = tf.image.resize(im, size=(224,224))
        
        st.text_area('Predictions:', top_5_pred(im), height=150)
        
    except:
        st.text_area('Predictions:', 
                     'Unable to perform predictions on this image',
                     height = 150)
