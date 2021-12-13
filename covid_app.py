from fastai.vision.widgets import *
from fastai.vision.all import *
import os
from pathlib import Path
import streamlit as st


class Predictor:
    def __init__(self, filename):
        self.learner = load_learner(filename)
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
            pred, pred_idx, probs = self.learner.predict(self.img)
            st.write(f'Prediction: {pred}; Probability: {probs[pred_idx]:.04f}')
        else: 
            st.write(f'Click the button to classify') 

if __name__=='__main__':
    st.write("# Covid detection from CT Scans")
    st.text("**Disclaimer!** This is just a proof of concept. It is not, by any means, a medical diagnosis and should not be relied upon without consulting a medic or health specialist.")
    st.write("### How precise is it?")
    st.write("Here is the confusion matrix of the test set results:")
    st.image('cm.png', caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
    st.write("Here is the F1 score from training:")
    st.image('results.png', caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
    st.write("### How does it work?")
    st.write("Upload your CT scan image below")
    st.write("Sample image:")
    st.image('ct_sample.png', caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
    path = Path('export.pkl')
    if path.is_file():
        file_name='export.pkl'
        predictor = Predictor(file_name)

