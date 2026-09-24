import os
import cv2
import time
import numpy as np
import streamlit as st
from PIL import Image

from deployment.inference import InferenceEngine

st.set_page_config(
    page_title="ADVANCE-FER: Production Emotion AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        background: -webkit-linear-gradient(45deg, #FF4B4B, #FF8F00);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .badge-bar {
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 10px;
        border-left: 4px solid #FF4B4B;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">ADVANCE-FER: Production Emotion Recognition</div>', unsafe_allow_html=True)
st.caption("Zero-training multi-face emotion recognition powered by MediaPipe FaceMesh & Quantized Vision Transformer (ONNX Runtime).")


@st.cache_resource
def get_inference_engine():
    return InferenceEngine()


engine = get_inference_engine()

st.sidebar.header("Configuration")
mode = st.sidebar.radio("Input Source", ["Image Upload", "Live Webcam", "Test Generator"])
conf_threshold = st.sidebar.slider("Confidence Threshold", min_value=0.1, max_value=1.0, value=0.35, step=0.05)
draw_landmarks = st.sidebar.checkbox("Draw Facial Landmark Mesh", value=False)
show_metrics = st.sidebar.checkbox("Display Latency & FPS HUD", value=True)

st.sidebar.divider()
st.sidebar.markdown("### Engine Status")
st.sidebar.success("Engine: **SOTA ViT-FER (ONNX)**")
st.sidebar.info("Runtime: **CPU / CUDA (Auto)**")
st.sidebar.info("Labels: 7 Discrete Emotions + Russell VA")

if mode == "Image Upload":
    st.subheader("Upload an Image")
    uploaded_file = st.file_uploader("Choose a face image (JPG, PNG, WEBP)", type=["jpg", "jpeg", "png", "webp"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        col1, col2 = st.columns([3, 2])

        with col1:
            st.markdown("#### Annotated Detections")
            annotated = engine.predict(image, draw_landmarks=draw_landmarks, show_metrics=show_metrics)
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

        with col2:
            st.markdown("#### Emotion Analytics")
            faces, latency = engine.process_frame(image)

            st.metric("Inference Latency", f"{latency:.1f} ms")
            st.metric("Detected Faces", len(faces))

            if faces:
                for idx, face in enumerate(faces):
                    with st.expander(f"Face #{idx+1}: {face['dominant_emotion'].capitalize()} ({face['confidence']*100:.1f}%)", expanded=(idx==0)):
                        col_m1, col_m2 = st.columns(2)
                        col_m1.metric("Valence (Pleasantness)", f"{face['valence']:+.2f}")
                        col_m2.metric("Arousal (Activation)", f"{face['arousal']:+.2f}")

                        st.write("**Probability Distribution:**")
                        prob_chart = {k.capitalize(): v for k, v in face['probabilities'].items()}
                        st.bar_chart(prob_chart)
            else:
                st.warning("No human faces detected in the uploaded image. Try another angle or lighting.")

elif mode == "Live Webcam":
    st.subheader("Real-Time Webcam Stream")
    run_camera = st.checkbox("Start Camera", value=False)
    frame_window = st.image([])

    if run_camera:
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            st.error("Could not access webcam (Device index 0). Please ensure webcam permissions are granted.")
        else:
            prev_time = time.time()
            while run_camera:
                ret, frame = camera.read()
                if not ret:
                    st.warning("Webcam feed interrupted.")
                    break

                annotated = engine.predict(frame, draw_landmarks=draw_landmarks, show_metrics=show_metrics)
                frame_window.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))

            camera.release()

else:
    st.subheader("Synthetic Test Pattern")
    st.info("Generates a synthetic frame to verify rendering pipeline without an external webcam.")

    test_img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.circle(test_img, (320, 240), 120, (210, 180, 140), -1)
    cv2.circle(test_img, (280, 200), 15, (255, 255, 255), -1)
    cv2.circle(test_img, (360, 200), 15, (255, 255, 255), -1)
    cv2.circle(test_img, (280, 200), 6, (50, 50, 50), -1)
    cv2.circle(test_img, (360, 200), 6, (50, 50, 50), -1)
    cv2.ellipse(test_img, (320, 280), (40, 20), 0, 0, 180, (50, 50, 200), 4)

    annotated = engine.predict(test_img, draw_landmarks=draw_landmarks, show_metrics=show_metrics)
    st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Synthetic Test Output", use_container_width=True)
