from fastai.vision.widgets import *
from fastai.vision.all import *
import os
from pathlib import Path
import streamlit as st

# def load_learner(file):
#     print(file)

# class PILImage:
#     @staticmethod
#     def create(file):
#         print(file)

class Predict:
    def __init__(self, filename):
        self.learn_inference = load_learner(filename)
        self.img = self.get_image_from_upload()
        if self.img is not None:
            self.display_output()
            self.get_prediction()
    
    @staticmethod
    def get_image_from_upload():
        uploaded_file = st.file_uploader("Upload Files",type=['png','jpeg', 'jpg'])
        if uploaded_file is not None:
            return PILImage.create((uploaded_file))
        return None

    def display_output(self):
        st.image(self.img.to_thumb(500,500), caption='Uploaded Image')

    def get_prediction(self):

        if st.button('Classify'):
            pred, pred_idx, probs = self.learn_inference.predict(self.img)
            st.write(f'Prediction: {pred}; Probability: {probs[pred_idx]:.04f}')
        else: 
            st.write(f'Click the button to classify') 

if __name__=='__main__':
    
    path = Path('export.pkl')
    if path.is_file():
        file_name='export.pkl'
        predictor = Predict(file_name)

