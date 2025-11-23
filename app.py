import streamlit as st
import cv2
import numpy as np
from deployment.inference import InferenceEngine
import tempfile
import os

st.set_page_config(page_title="Advanced FER Demo", layout="wide")

st.title("Advanced Facial Expression Recognition")
st.sidebar.header("Settings")

use_webcam = st.sidebar.checkbox("Use Webcam", value=True)
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5)

@st.cache_resource
def load_engine():
    model_path = "models/fer_model_best.pth"
    if not os.path.exists(model_path):
        model_path = "models/fer_model.pth"
    
    if os.path.exists(model_path):
        return InferenceEngine(model_path=model_path, device='cpu')
    else:
        st.warning("Model not found. Using random weights.")
        return InferenceEngine(device='cpu')

engine = load_engine()

st.write("## Real-time Inference")

if use_webcam:
    run = st.checkbox('Run')
    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(0)
    
    while run:
        _, frame = camera.read()
        if frame is None:
            break
            
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run inference
        output_frame = engine.predict(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        output_frame = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)
        
        FRAME_WINDOW.image(output_frame)
    
    camera.release()
else:
    st.write("Upload a video or image to test.")
    uploaded_file = st.file_uploader("Choose a file", type=['jpg', 'png', 'mp4'])
    
    if uploaded_file is not None:
        # Handle image
        if uploaded_file.type.startswith('image'):
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, 1)
            output_frame = engine.predict(frame)
            st.image(cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB))
