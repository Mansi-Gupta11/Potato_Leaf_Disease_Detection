import numpy as np
import streamlit as st
import cv2
from keras.models import load_model
import tensorflow as tf

model=load_model('C:/Users/asus/Potato_Leaf_Disease_Detection/Streamlit/model/disease.h5')

CLASS_NAMES=['Potato__Early_blight','Potato__healthy','Potato__Late_blight']
st.title("Plant Leaf Disease Detection")
st.markdown("Upload")

plant_image=st.file_uploader("Choose an image..",type="jpg")
submit=st.button("Predict Disease")
if submit:
    if plant_image is not None:
        file_bytes = np.asarray(bytearray(plant_image.read()), dtype=np.uint8)
        opencv_image=cv2.imdecode(file_bytes,1)
        st.image(opencv_image,channels="BGR")
        st.write(opencv_image.shape)
        opencv_image=cv2.resize(opencv_image,(256,256))
        opencv_image.shape=(1,256,256,3)
        Y_pred = model.predict(opencv_image)
        result_index = np.argmax(Y_pred)
        result = CLASS_NAMES[result_index]
        
        # st.write("Model Predictions (Probabilities):", Y_pred)
        st.write("Predicted Class Index:", result_index)
        st.write("Predicted Class Name:", result)